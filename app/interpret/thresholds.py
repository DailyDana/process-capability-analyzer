"""Configurable thresholds for the interpretation engine.

Every numeric cut-point the interpreter uses lives here so it is (a) documented
in one place, (b) overridable by the caller, and (c) never hard-coded inside a
rule. The defaults follow common manufacturing/AIAG interpretation conventions.

IMPORTANT: these capability categories are **engineering interpretation
conventions, not universal statistical laws**. Different industries and customer
standards use different cut-points (e.g. some require Cpk >= 1.33, automotive PPAP
often 1.67 for new processes). They are exposed as configuration precisely so a
site can align them to its own quality standard. Documented in FORMULAS.md §M5.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

__all__ = ["InterpretationThresholds", "CapabilityCategory"]


from enum import Enum


class CapabilityCategory(str, Enum):
    """Ordinal capability categories (interpretation convention, not law)."""

    INADEQUATE = "inadequate"        # index < marginal_min
    MARGINAL = "marginal"            # marginal_min <= index < capable_min
    CAPABLE = "capable"              # capable_min <= index < excellent_min
    EXCELLENT = "excellent"          # index >= excellent_min
    UNDEFINED = "undefined"          # index is None / not applicable


@dataclass(frozen=True, slots=True)
class InterpretationThresholds:
    """All interpretation cut-points, with documented defaults.

    Capability-index category bands (applied to Cpk/Ppk/Cp/Pp):

    * ``index <  marginal_min``            -> INADEQUATE
    * ``marginal_min <= index < capable_min``  -> MARGINAL
    * ``capable_min  <= index < excellent_min``-> CAPABLE
    * ``index >= excellent_min``           -> EXCELLENT

    Defaults (AIAG-style convention):
      marginal_min = 1.00, capable_min = 1.33, excellent_min = 1.67.

    Other cut-points:

    centering_gap_ratio:
        Cpk is considered "substantially" below Cp when
        ``(Cp - Cpk) / Cp >= centering_gap_ratio`` (default 0.10 = 10%). Flags a
        centering problem distinct from a pure variation problem.
    drift_gap_ratio:
        Pp/Ppk is considered "substantially" worse than Cp/Cpk when
        ``(Cpk - Ppk) / Cpk >= drift_gap_ratio`` (default 0.10 = 10%). Signals
        long-term drift / between-subgroup variation.
    spec_margin_tight_sigma:
        The two-sided spec width is flagged "tight relative to process spread"
        when it is less than ``spec_margin_tight_sigma * sigma_within`` (default
        6.0 -> narrower than the natural +/-3 sigma process band, i.e. Cp < 1).
    min_observations, min_subgroups:
        Mirror the M2 reliability-guard thresholds (default 30 / 20) so the
        interpreter's small-sample language stays consistent with the guards.
    """

    marginal_min: float = 1.00
    capable_min: float = 1.33
    excellent_min: float = 1.67

    centering_gap_ratio: float = 0.10
    drift_gap_ratio: float = 0.10
    spec_margin_tight_sigma: float = 6.0

    min_observations: int = 30
    min_subgroups: int = 20

    def __post_init__(self) -> None:
        """Reject cut-points that cannot express a coherent interpretation.

        :meth:`categorize` tests the bands in ascending order, so an out-of-order
        triple does not fail on its own -- it silently produces nonsense (a Cpk
        of 1.66 categorised INADEQUATE). Validating here makes the ordering an
        invariant of the type rather than a rule every caller must remember.

        Raises
        ------
        ValueError
            With a message naming the offending field and its value, suitable
            for showing directly to the user.
        """
        for name in (
            "marginal_min", "capable_min", "excellent_min",
            "centering_gap_ratio", "drift_gap_ratio", "spec_margin_tight_sigma",
        ):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")

        if not (self.marginal_min < self.capable_min < self.excellent_min):
            raise ValueError(
                "Capability thresholds must increase strictly "
                "(marginal_min < capable_min < excellent_min); got "
                f"marginal_min={self.marginal_min:g}, "
                f"capable_min={self.capable_min:g}, "
                f"excellent_min={self.excellent_min:g}."
            )

        for name in ("centering_gap_ratio", "drift_gap_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(
                    f"{name} is a proportion of an index and must lie in "
                    f"[0.0, 1.0); got {value!r}."
                )

        if self.spec_margin_tight_sigma <= 0.0:
            raise ValueError(
                "spec_margin_tight_sigma is a multiple of sigma and must be "
                f"positive; got {self.spec_margin_tight_sigma!r}."
            )

        for name in ("min_observations", "min_subgroups"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value!r}.")

    def categorize(self, index: float | None) -> CapabilityCategory:
        """Map a capability index value to its :class:`CapabilityCategory`.

        ``None`` -> UNDEFINED. A non-finite (+inf zero-variation sentinel) value
        is treated as EXCELLENT for banding purposes but the interpreter attaches
        an explicit zero-variation caveat separately.
        """
        if index is None:
            return CapabilityCategory.UNDEFINED
        if index != index:  # NaN
            return CapabilityCategory.UNDEFINED
        if index == float("inf"):
            return CapabilityCategory.EXCELLENT
        if index < self.marginal_min:
            return CapabilityCategory.INADEQUATE
        if index < self.capable_min:
            return CapabilityCategory.MARGINAL
        if index < self.excellent_min:
            return CapabilityCategory.CAPABLE
        return CapabilityCategory.EXCELLENT
