"""Domain models for measurement data and specifications.

Design notes
------------
* **Full precision.** Measured values are stored as ``float`` without rounding.
* **Regime awareness.** The :class:`DataRegime` enum is the guardrail that
  drives the whole sequence-of-analysis: within-subgroup capability (Cp/Cpk) is
  only offered when a genuine subgroup structure exists. This module records
  the regime; enforcement lives in the services/capability layers (later
  milestones), but the vocabulary is fixed here.
* **One-sided specs.** :class:`SpecLimits` allows either bound to be ``None`` so
  the capability layer can correctly report only Cpu/Ppu or only Cpl/Ppl.
* **Units as metadata.** No unit is hard-coded; the ``unit`` string travels with
  the specification and is displayed consistently in UI/reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DataRegime",
    "SpecLimits",
    "Measurement",
    "Dataset",
]


class DataRegime(str, Enum):
    """How the measurement data is structured in time.

    The regime determines which analyses are statistically valid:

    * ``SUBGROUPED`` — rational subgroups of size >= 2. Supports X-bar/R or
      X-bar/S charts and *both* within-subgroup capability (Cp/Cpk) and overall
      performance (Pp/Ppk).
    * ``INDIVIDUALS`` — one measurement per time point (n = 1). Supports the
      I-MR chart; within-subgroup sigma is estimated from the moving range.
    * ``UNORDERED`` — a bare sample with no time order or subgroup structure.
      *Only* overall performance (Pp/Ppk) is defensible; Cp/Cpk must be
      suppressed because there is no valid within-subgroup variation to
      estimate.
    """

    SUBGROUPED = "subgrouped"
    INDIVIDUALS = "individuals"
    UNORDERED = "unordered"


@dataclass(frozen=True, slots=True)
class SpecLimits:
    """Engineering specification limits for one characteristic.

    Parameters
    ----------
    usl:
        Upper specification limit, or ``None`` if the characteristic has no
        upper bound (one-sided lower spec).
    lsl:
        Lower specification limit, or ``None`` if the characteristic has no
        lower bound (one-sided upper spec).
    nominal:
        Target/nominal value, if defined. Used for centering interpretation.
    unit:
        Measurement unit as free-text metadata (e.g. ``"mm"``, ``"inch"``,
        ``"°C"``). Never interpreted numerically; displayed as-is.

    Raises
    ------
    ValueError
        If both bounds are ``None`` (no specification at all), if
        ``usl <= lsl`` when both are given, or if a supplied ``nominal`` lies
        outside a given bound.
    """

    usl: Optional[float] = None
    lsl: Optional[float] = None
    nominal: Optional[float] = None
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.usl is None and self.lsl is None:
            raise ValueError(
                "SpecLimits requires at least one of USL or LSL; both are None."
            )
        for name, val in (("usl", self.usl), ("lsl", self.lsl), ("nominal", self.nominal)):
            if val is not None and not np.isfinite(val):
                raise ValueError(f"SpecLimits.{name} must be finite, got {val!r}.")
        if self.usl is not None and self.lsl is not None and self.usl <= self.lsl:
            raise ValueError(
                f"USL ({self.usl}) must be strictly greater than LSL ({self.lsl})."
            )
        if self.nominal is not None:
            if self.usl is not None and self.nominal > self.usl:
                raise ValueError(
                    f"Nominal ({self.nominal}) lies above USL ({self.usl})."
                )
            if self.lsl is not None and self.nominal < self.lsl:
                raise ValueError(
                    f"Nominal ({self.nominal}) lies below LSL ({self.lsl})."
                )

    @property
    def is_two_sided(self) -> bool:
        """True if both USL and LSL are defined."""
        return self.usl is not None and self.lsl is not None

    @property
    def tolerance_width(self) -> Optional[float]:
        """USL - LSL when two-sided, else ``None``."""
        if self.is_two_sided:
            return float(self.usl - self.lsl)  # type: ignore[operator]
        return None

    def implied_nominal(self) -> Optional[float]:
        """Return ``nominal`` if given, else the midpoint of a two-sided spec.

        For a one-sided spec with no explicit nominal, returns ``None``.
        """
        if self.nominal is not None:
            return self.nominal
        if self.is_two_sided:
            return float((self.usl + self.lsl) / 2.0)  # type: ignore[operator]
        return None


@dataclass(frozen=True, slots=True)
class Measurement:
    """A single measured observation with optional traceability metadata.

    Only ``value`` is required. The optional metadata fields support the
    traceability an engineer expects (part, machine, operator, time,
    characteristic) without forcing them into the statistical core, which only
    ever consumes ``value``.
    """

    value: float
    sample_id: Optional[str] = None
    timestamp: Optional[str] = None
    machine: Optional[str] = None
    operator: Optional[str] = None
    characteristic: Optional[str] = None
    subgroup_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or not np.isfinite(self.value):
            raise ValueError(
                f"Measurement.value must be a finite number, got {self.value!r}."
            )


@dataclass(frozen=True, slots=True)
class Dataset:
    """A collection of measurements for a single characteristic.

    The dataset stores values plus an optional parallel subgroup-label array.
    It does not itself decide the regime — that is the job of the services
    layer, which inspects subgroup structure — but it exposes the helpers that
    layer needs.

    Parameters
    ----------
    values:
        1-D sequence of measured values (already cleaned/numeric).
    subgroup_labels:
        Optional parallel sequence assigning each value to a subgroup. When
        present and every subgroup has size >= 2 the data can be treated as
        ``SUBGROUPED``.
    spec:
        Optional specification limits for the characteristic.
    characteristic:
        Human-readable characteristic name.
    unit:
        Measurement unit metadata (mirrors ``spec.unit`` when both are given).
    """

    values: NDArray[np.float64]
    subgroup_labels: Optional[NDArray[np.object_]] = None
    spec: Optional[SpecLimits] = None
    characteristic: Optional[str] = None
    unit: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("Dataset requires at least one value.")
        if not np.isfinite(arr).all():
            raise ValueError(
                "Dataset.values must all be finite; clean NaN/inf via the "
                "validation layer before constructing a Dataset."
            )
        object.__setattr__(self, "values", arr)
        if self.subgroup_labels is not None:
            labels = np.asarray(self.subgroup_labels, dtype=object).ravel()
            if labels.size != arr.size:
                raise ValueError(
                    f"subgroup_labels length ({labels.size}) must match values "
                    f"length ({arr.size})."
                )
            object.__setattr__(self, "subgroup_labels", labels)

    @property
    def n(self) -> int:
        """Total number of observations."""
        return int(self.values.size)

    def subgroups(self) -> list[NDArray[np.float64]]:
        """Return values grouped by subgroup label, preserving first-seen order.

        Returns a single-group list containing all values when no labels are
        present.
        """
        if self.subgroup_labels is None:
            return [self.values]
        groups: dict[object, list[float]] = {}
        for label, val in zip(self.subgroup_labels, self.values):
            groups.setdefault(label, []).append(float(val))
        return [np.asarray(v, dtype=np.float64) for v in groups.values()]

    def subgroup_sizes(self) -> list[int]:
        """Return the size of each subgroup (single entry when unlabeled)."""
        return [int(g.size) for g in self.subgroups()]

    def is_balanced(self) -> bool:
        """True if all subgroups share the same size."""
        sizes = self.subgroup_sizes()
        return len(set(sizes)) == 1

    @classmethod
    def from_measurements(
        cls,
        measurements: Sequence[Measurement],
        *,
        spec: Optional[SpecLimits] = None,
        characteristic: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> "Dataset":
        """Build a :class:`Dataset` from a sequence of :class:`Measurement`.

        Subgroup labels are taken from ``Measurement.subgroup_id`` when *any*
        measurement carries one; otherwise the dataset is left unlabeled.
        """
        if not measurements:
            raise ValueError("Cannot build a Dataset from an empty measurement list.")
        values = np.asarray([m.value for m in measurements], dtype=np.float64)
        has_subgroups = any(m.subgroup_id is not None for m in measurements)
        labels: Optional[NDArray[np.object_]] = None
        if has_subgroups:
            labels = np.asarray(
                [m.subgroup_id for m in measurements], dtype=object
            )
        char = characteristic or next(
            (m.characteristic for m in measurements if m.characteristic), None
        )
        return cls(
            values=values,
            subgroup_labels=labels,
            spec=spec,
            characteristic=char,
            unit=unit or (spec.unit if spec else None),
        )
