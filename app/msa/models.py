"""Domain models for a crossed Gage R&R study and its results.

Design notes
------------
* **One array, fixed axes.** A balanced crossed study is stored as a 3-D array
  ``values[part, operator, trial]``. Balance is therefore structural: an
  unbalanced study cannot be represented, so no estimator downstream has to
  guard against it (the validation layer rejects such input with a message).
* **Sigmas, not 5.15/6 sigma spreads.** Every component is carried as a
  standard deviation (and variance). The study-variation multiplier ``k`` is a
  presentation convention applied only in :mod:`app.msa.metrics`, so the two
  conventions (6 vs 5.15) can never leak into the estimators.
* **Method-neutral result.** ANOVA and Average & Range both reduce to the same
  :class:`ComponentEstimates`; method-specific detail (the ANOVA table, the
  range chart check) travels alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GageMethod",
    "GageStudy",
    "rounding_floor",
    "ComponentEstimates",
    "SourceRow",
    "RangeCheck",
    "GageRRResult",
]


class GageMethod(str, Enum):
    """Estimation method for the variance components."""

    ANOVA = "anova"
    AVERAGE_RANGE = "average_range"


@dataclass(frozen=True, slots=True)
class GageStudy:
    """A balanced, crossed Gage R&R study.

    Parameters
    ----------
    values:
        Measurements with shape ``(parts, operators, trials)``. Copied and made
        read-only on construction.
    part_labels, operator_labels:
        Display labels in axis order. Lengths must match the array axes.
    unit:
        Optional measurement unit (metadata only).
    """

    values: NDArray[np.float64]
    part_labels: tuple[str, ...]
    operator_labels: tuple[str, ...]
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        arr = np.array(self.values, dtype=float, copy=True)
        if arr.ndim != 3:
            raise ValueError(
                f"values must be 3-D (parts, operators, trials); got {arr.ndim}-D."
            )
        p, o, r = arr.shape
        if p < 2 or o < 2 or r < 2:
            raise ValueError(
                "A Gage R&R study needs at least 2 parts, 2 operators and 2 "
                f"trials; got {p} x {o} x {r}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("All measurements must be finite numbers.")
        if len(self.part_labels) != p:
            raise ValueError(f"Expected {p} part labels, got {len(self.part_labels)}.")
        if len(self.operator_labels) != o:
            raise ValueError(
                f"Expected {o} operator labels, got {len(self.operator_labels)}."
            )
        arr.setflags(write=False)
        object.__setattr__(self, "values", arr)
        object.__setattr__(self, "part_labels", tuple(str(x) for x in self.part_labels))
        object.__setattr__(
            self, "operator_labels", tuple(str(x) for x in self.operator_labels)
        )

    @property
    def rounding_floor(self) -> float:
        """See :func:`rounding_floor`."""
        return rounding_floor(self.values)

    @property
    def n_parts(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_operators(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_trials(self) -> int:
        return int(self.values.shape[2])


def rounding_floor(values: NDArray[np.float64]) -> float:
    """Largest difference of means that floating-point rounding alone can create.

    Means of values of magnitude ``M`` carry an absolute rounding error of a
    few ``eps * M``; a difference or deviation smaller than this floor is
    numerically zero. Without it, a gauge with *no* measurement variation
    yields a GRR of ~1e-16 instead of 0 and an ndc of ~1e15. The factor 64 is
    generous headroom; the floor is still ~1e-14 relative to the data, far
    below any physically meaningful measurement variation.
    """
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return 64.0 * float(np.finfo(float).eps) * scale


@dataclass(frozen=True, slots=True)
class ComponentEstimates:
    """Variance-component estimates, as standard deviations.

    ``operator_sd`` and ``interaction_sd`` split reproducibility when the
    method estimates them: ANOVA always reports ``operator_sd`` and, unless the
    interaction was pooled, ``interaction_sd``. The Average & Range method
    reports neither (``None``); its AV is the reproducibility as a whole.

    ``clamped`` lists components whose raw estimate was negative and was set
    to zero (the usual convention; the raw value is kept for transparency).
    """

    repeatability_sd: float
    reproducibility_sd: float
    part_sd: float
    operator_sd: Optional[float] = None
    interaction_sd: Optional[float] = None
    clamped: tuple[str, ...] = ()
    raw_estimates: dict[str, float] = field(default_factory=dict)

    @property
    def grr_sd(self) -> float:
        return sqrt(self.repeatability_sd**2 + self.reproducibility_sd**2)

    @property
    def total_sd(self) -> float:
        return sqrt(self.grr_sd**2 + self.part_sd**2)


@dataclass(frozen=True, slots=True)
class SourceRow:
    """One line of the Gage R&R summary table.

    Percentages are ``None`` when undefined: ``pct_tolerance`` without a
    tolerance, and every percentage when total variation is zero.
    """

    source: str
    variance: float
    sd: float
    study_var: float
    pct_contribution: Optional[float]
    pct_study_var: Optional[float]
    pct_tolerance: Optional[float]


@dataclass(frozen=True, slots=True)
class RangeCheck:
    """Range chart across all part x operator cells (repeatability consistency).

    AIAG requires the ranges to be in statistical control before the study is
    interpreted: an out-of-control range means one appraiser was inconsistent
    on that part (a measurement-process problem, or a recording error).
    """

    r_bar: float
    ucl: float
    out_of_control: tuple[tuple[str, str, float], ...]  # (part, operator, range)

    @property
    def in_control(self) -> bool:
        return not self.out_of_control


@dataclass(frozen=True, slots=True)
class GageRRResult:
    """Complete result of one Gage R&R analysis (one method)."""

    method: GageMethod
    components: ComponentEstimates
    rows: tuple[SourceRow, ...]
    ndc: Optional[int]
    k: float
    tolerance: Optional[float]
    range_check: RangeCheck
    n_parts: int
    n_operators: int
    n_trials: int
    # Method-specific detail, kept as plain objects to avoid an import cycle.
    anova: Optional[object] = None
    average_range: Optional[object] = None

    def row(self, source: str) -> SourceRow:
        """Return the summary row for ``source`` (e.g. ``"Total Gage R&R"``)."""
        for r in self.rows:
            if r.source == source:
                return r
        raise KeyError(source)

    @property
    def pct_grr_study_var(self) -> Optional[float]:
        return self.row("Total Gage R&R").pct_study_var

    @property
    def pct_grr_tolerance(self) -> Optional[float]:
        return self.row("Total Gage R&R").pct_tolerance
