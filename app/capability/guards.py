"""Reliability guards for capability analysis.

These produce structured *warnings* that are attached to a capability result.
Per project decision they **never** suppress or refuse the indices — the numbers
are always reported, with the caveats made explicit. See FORMULAS.md §2.7.

Thresholds (configurable via :class:`GuardThresholds`):
* small sample:  total N < 30 individual observations
* few subgroups: number of rational subgroups < 20
* zero variation: sigma == 0
* non-normality: Shapiro-Wilk rejects at alpha = 0.05
* stability: not assessed until M3 -> "stability not verified" note
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from app.statistics.descriptive import to_float_array

__all__ = [
    "Severity",
    "ReliabilityWarning",
    "GuardThresholds",
    "NormalityResult",
    "assess_normality",
    "evaluate_guards",
]


class Severity(str, Enum):
    """Advisory severity of a reliability warning (never fatal)."""

    INFO = "info"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ReliabilityWarning:
    """A single structured reliability caveat.

    Attributes
    ----------
    code:
        Stable machine-readable identifier (e.g. ``"small_sample"``).
    severity:
        :class:`Severity` — advisory only.
    message:
        Human-readable explanation for UI/report.
    """

    code: str
    severity: Severity
    message: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class GuardThresholds:
    """Configurable thresholds for the reliability guards."""

    min_observations: int = 30
    min_subgroups: int = 20
    normality_alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class NormalityResult:
    """Outcome of the normality assessment.

    ``normal`` is ``None`` when the test could not be run (too few points).
    """

    normal: Optional[bool]
    shapiro_stat: Optional[float]
    shapiro_p: Optional[float]
    anderson_stat: Optional[float]
    anderson_crit_5pct: Optional[float]
    n: int
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "normal": self.normal,
            "shapiro_stat": self.shapiro_stat,
            "shapiro_p": self.shapiro_p,
            "anderson_stat": self.anderson_stat,
            "anderson_crit_5pct": self.anderson_crit_5pct,
            "n": self.n,
            "note": self.note,
        }


def assess_normality(values: ArrayLike, *, alpha: float = 0.05) -> NormalityResult:
    """Assess normality via Shapiro-Wilk (primary) + Anderson-Darling (reported).

    See FORMULAS.md §2.7. Shapiro-Wilk needs n >= 3; with fewer points, or with
    zero variance, the test cannot run and ``normal`` is ``None`` with a note.
    Rejection (``p < alpha``) sets ``normal = False``.
    """
    arr = to_float_array(values, nan_policy="omit")
    n = int(arr.size)

    if n < 3:
        return NormalityResult(
            normal=None,
            shapiro_stat=None,
            shapiro_p=None,
            anderson_stat=None,
            anderson_crit_5pct=None,
            n=n,
            note="Too few observations (n < 3) to test normality.",
        )
    if np.ptp(arr) == 0.0:
        return NormalityResult(
            normal=None,
            shapiro_stat=None,
            shapiro_p=None,
            anderson_stat=None,
            anderson_crit_5pct=None,
            n=n,
            note="Zero variance: normality is undefined/untestable.",
        )

    shapiro_stat: Optional[float]
    shapiro_p: Optional[float]
    try:
        sw = stats.shapiro(arr)
        shapiro_stat = float(sw.statistic)
        shapiro_p = float(sw.pvalue)
    except Exception as exc:  # pragma: no cover - scipy edge cases
        shapiro_stat = None
        shapiro_p = None
        note_sw = f"Shapiro-Wilk failed: {exc}. "
    else:
        note_sw = ""

    anderson_stat: Optional[float] = None
    anderson_crit: Optional[float] = None
    try:
        # scipy>=1.17 emits a FutureWarning asking callers to opt into a p-value
        # method; we only use the statistic + tabulated critical values here, so
        # the current interpolated behaviour is exactly what we want. Suppress
        # the advisory warning locally rather than globally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ad = stats.anderson(arr, dist="norm")
        anderson_stat = float(ad.statistic)
        # significance_level array includes 15,10,5,2.5,1; grab the 5% critical value.
        levels = list(ad.significance_level)
        if 5.0 in levels:
            anderson_crit = float(ad.critical_values[levels.index(5.0)])
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - scipy edge cases
        pass

    normal = None if shapiro_p is None else bool(shapiro_p >= alpha)
    note = note_sw
    if normal is False:
        note += (
            f"Shapiro-Wilk rejects normality (p={shapiro_p:.4g} < {alpha}); "
            "capability-to-PPM translation and the Cpk CI assume normality and "
            "should be treated as unreliable."
        )
    elif normal is True:
        note += f"Shapiro-Wilk does not reject normality (p={shapiro_p:.4g})."

    return NormalityResult(
        normal=normal,
        shapiro_stat=shapiro_stat,
        shapiro_p=shapiro_p,
        anderson_stat=anderson_stat,
        anderson_crit_5pct=anderson_crit,
        n=n,
        note=note.strip(),
    )


def evaluate_guards(
    *,
    n_observations: int,
    n_subgroups: Optional[int],
    sigma: float,
    normality: Optional[NormalityResult] = None,
    stability_state: Optional[bool] = None,
    stability_detail: str = "",
    thresholds: GuardThresholds = GuardThresholds(),
) -> list[ReliabilityWarning]:
    """Return the list of reliability warnings applicable to a result.

    None of these suppress the indices; they annotate them. ``n_subgroups`` may
    be ``None`` for unordered/individuals data where a subgroup count is not
    meaningful (the few-subgroups guard is then skipped).

    ``stability_state`` is three-valued:

    * ``None``  — stability not assessed yet -> INFO "stability not verified".
    * ``True``  — assessed and stable -> no stability warning.
    * ``False`` — assessed and UNSTABLE -> WARNING that capability indices are
      not trustworthy on an out-of-control process (indices are still reported).
    """
    warnings: list[ReliabilityWarning] = []

    if n_observations < thresholds.min_observations:
        warnings.append(
            ReliabilityWarning(
                code="small_sample",
                severity=Severity.WARNING,
                message=(
                    f"Only {n_observations} observations (< "
                    f"{thresholds.min_observations}). Capability indices have "
                    "wide sampling variability at this sample size; treat them "
                    "as indicative, not definitive."
                ),
            )
        )

    if n_subgroups is not None and n_subgroups < thresholds.min_subgroups:
        warnings.append(
            ReliabilityWarning(
                code="few_subgroups",
                severity=Severity.WARNING,
                message=(
                    f"Only {n_subgroups} subgroups (< "
                    f"{thresholds.min_subgroups}). Too few rational subgroups to "
                    "trust the within-subgroup sigma estimate or the stability "
                    "picture."
                ),
            )
        )

    if sigma == 0.0:
        warnings.append(
            ReliabilityWarning(
                code="zero_variation",
                severity=Severity.WARNING,
                message=(
                    "Zero observed variation: capability indices are reported as "
                    "+inf. This is almost always a measurement-resolution or "
                    "data-entry artifact rather than a real process; verify the "
                    "gauge and data."
                ),
            )
        )

    if normality is not None and normality.normal is False:
        warnings.append(
            ReliabilityWarning(
                code="non_normal",
                severity=Severity.WARNING,
                message=(
                    "Normality is rejected. Cp/Cpk/Pp/Ppk-to-PPM translation and "
                    "the Cpk confidence interval assume a normal distribution; "
                    "their probability interpretation is unreliable here. "
                    "Consider a distribution-appropriate capability method "
                    "(not implemented in this version)."
                ),
            )
        )

    if stability_state is None:
        warnings.append(
            ReliabilityWarning(
                code="stability_not_verified",
                severity=Severity.INFO,
                message=(
                    "Process stability has not been verified (SPC control rules "
                    "have not been evaluated). Capability indices are only "
                    "trustworthy for a statistically stable process; confirm "
                    "stability before quoting these numbers."
                ),
            )
        )
    elif stability_state is False:
        detail = f" ({stability_detail})" if stability_detail else ""
        warnings.append(
            ReliabilityWarning(
                code="unstable_process",
                severity=Severity.WARNING,
                message=(
                    "The process is statistically UNSTABLE: SPC control rules "
                    f"detected out-of-control signals{detail}. Capability/"
                    "performance indices assume a stable, predictable process, "
                    "so these numbers do NOT establish that the process is "
                    "capable — a high Cpk on an unstable process is not "
                    "trustworthy. The indices are reported unchanged; investigate "
                    "and remove the special causes, then re-assess capability."
                ),
            )
        )
    # stability_state is True -> stable, no stability warning.

    return warnings
