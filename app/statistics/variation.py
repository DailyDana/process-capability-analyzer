"""Within-subgroup and overall sigma estimation.

This module is the heart of the Cp/Cpk vs Pp/Ppk distinction. It exposes two
completely separate estimation paths that must never be confused:

* :func:`sigma_within` — short-term / within-subgroup variation, feeding
  **Cp, Cpu, Cpl, Cpk**. Estimated from subgroup ranges / standard deviations /
  moving ranges via the control-chart constants.
* :func:`sigma_overall` — long-term / overall variation, feeding
  **Pp, Ppu, Ppl, Ppk**. Estimated as the ordinary sample standard deviation of
  all pooled individual observations.

See ``docs/FORMULAS.md`` §2.2–2.3 for the full derivations, assumptions and
references. Nothing here is rounded; full float precision is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.statistics import constants as const
from app.statistics.descriptive import std_dev, to_float_array

__all__ = [
    "WithinEstimator",
    "WithinSigmaResult",
    "AUTO_S_SWITCH_N",
    "sigma_within",
    "sigma_within_from_subgroups",
    "sigma_within_from_individuals",
    "sigma_overall",
    "grand_mean",
    "moving_ranges",
]

# Subgroup size at/above which S-bar/c4 is auto-selected over R-bar/d2.
# Rationale documented in FORMULAS.md §2.2(b): range efficiency drops with n.
AUTO_S_SWITCH_N: int = 9


def _lookup_d2(n: int) -> float:
    """Return d2(n): the published table value when tabulated, else exact.

    The classic R-bar/d2 estimator is conventionally computed with the
    tabulated control-chart constants (matching textbook worked examples and
    SPC software). Beyond the tabulated range we fall back to the exact
    integral so the estimator still works for large subgroups.
    """
    if const.MIN_SUBGROUP_SIZE <= n <= const.MAX_TABULATED_N:
        return const.get_constants(n).d2
    return const.d2(n)


def _lookup_c4(n: int) -> float:
    """Return c4(n): the published table value when tabulated, else exact."""
    if const.MIN_SUBGROUP_SIZE <= n <= const.MAX_TABULATED_N:
        return const.get_constants(n).c4
    return const.c4(n)


class WithinEstimator(str, Enum):
    """Strategy for estimating within-subgroup sigma.

    * ``AUTO`` — R-bar/d2 for balanced subgroups with ``n < AUTO_S_SWITCH_N``,
      S-bar/c4 for balanced subgroups with ``n >= AUTO_S_SWITCH_N``, and pooled
      standard deviation when subgroups are unbalanced.
    * ``RBAR_D2`` — force average-range estimator (requires balanced n).
    * ``SBAR_C4`` — force average-standard-deviation estimator (requires balanced n).
    * ``POOLED`` — force pooled standard deviation (handles unbalanced n).
    * ``MR_D2`` — moving-range estimator for individuals (n = 1).
    """

    AUTO = "auto"
    RBAR_D2 = "rbar_d2"
    SBAR_C4 = "sbar_c4"
    POOLED = "pooled"
    MR_D2 = "mr_d2"


@dataclass(frozen=True, slots=True)
class WithinSigmaResult:
    """Result of a within-subgroup sigma estimation.

    Attributes
    ----------
    sigma:
        The estimated within-subgroup standard deviation. ``0.0`` when there is
        genuinely no within variation; never negative.
    estimator:
        The concrete estimator actually used (``AUTO`` is resolved to the
        underlying method).
    subgroup_size:
        The subgroup size ``n`` used for constant lookup, or ``None`` for the
        pooled estimator on unbalanced data (where a single ``n`` is not
        meaningful).
    n_subgroups:
        Number of subgroups the estimate was built from (moving-range pairs for
        the individuals estimator).
    detail:
        Free-form dict of intermediate quantities (e.g. ``R_bar``, ``S_bar``,
        ``MR_bar``, ``pooled_dof``) for transparency in reports/tests.
    """

    sigma: float
    estimator: WithinEstimator
    subgroup_size: Optional[int]
    n_subgroups: int
    detail: dict[str, float]


def _as_subgroup_arrays(
    subgroups: Sequence[ArrayLike],
) -> list[NDArray[np.float64]]:
    """Coerce and validate a sequence of subgroups into clean float arrays."""
    arrays: list[NDArray[np.float64]] = []
    for j, g in enumerate(subgroups):
        arr = to_float_array(g, nan_policy="omit")
        arrays.append(arr)
    if not arrays:
        raise ValueError("No subgroups provided.")
    return arrays


def sigma_within_from_subgroups(
    subgroups: Sequence[ArrayLike],
    *,
    estimator: WithinEstimator = WithinEstimator.AUTO,
) -> WithinSigmaResult:
    """Estimate within-subgroup sigma from a list of subgroups (each size >= 1).

    See FORMULAS.md §2.2. The AUTO strategy picks R-bar/d2, S-bar/c4, or pooled
    depending on subgroup size and balance. Explicit estimators are validated
    against the data shape (e.g. RBAR_D2/SBAR_C4 require equal subgroup sizes).

    Raises
    ------
    ValueError
        If an explicit balanced-only estimator is requested on unbalanced data,
        if subgroup sizes are < 2 where the estimator needs within-subgroup
        spread, or if ``n`` exceeds the tabulated constant range.
    """
    groups = _as_subgroup_arrays(subgroups)
    sizes = [g.size for g in groups]
    balanced = len(set(sizes)) == 1
    n = sizes[0] if balanced else None
    k = len(groups)

    # Resolve AUTO to a concrete estimator.
    resolved = estimator
    if estimator is WithinEstimator.AUTO:
        if not balanced:
            resolved = WithinEstimator.POOLED
        elif n is not None and n >= AUTO_S_SWITCH_N:
            resolved = WithinEstimator.SBAR_C4
        else:
            resolved = WithinEstimator.RBAR_D2

    if resolved in (WithinEstimator.RBAR_D2, WithinEstimator.SBAR_C4):
        if not balanced:
            raise ValueError(
                f"Estimator {resolved.value} requires equal subgroup sizes; got "
                f"sizes {sorted(set(sizes))}. Use POOLED for unbalanced data."
            )
        assert n is not None
        if n < 2:
            raise ValueError(
                f"Estimator {resolved.value} needs subgroup size n >= 2 to have "
                "within-subgroup spread; got n = 1 (use the individuals/MR path)."
            )

    if resolved is WithinEstimator.RBAR_D2:
        ranges = np.array([float(g.max() - g.min()) for g in groups])
        r_bar = float(ranges.mean())
        d2 = _lookup_d2(n)  # type: ignore[arg-type]
        sigma = r_bar / d2
        detail = {"R_bar": r_bar, "d2": d2}
    elif resolved is WithinEstimator.SBAR_C4:
        sds = np.array([std_dev(g, ddof=1) for g in groups])
        s_bar = float(sds.mean())
        c4 = _lookup_c4(n)  # type: ignore[arg-type]
        sigma = s_bar / c4
        detail = {"S_bar": s_bar, "c4": c4}
    elif resolved is WithinEstimator.POOLED:
        # Pooled variance across subgroups; needs at least one subgroup with n>=2.
        dof_total = sum(sz - 1 for sz in sizes)
        if dof_total <= 0:
            raise ValueError(
                "Pooled estimator requires total within-subgroup degrees of "
                "freedom > 0 (at least one subgroup of size >= 2)."
            )
        ss = sum(
            (sz - 1) * (std_dev(g, ddof=1) ** 2)
            for g, sz in zip(groups, sizes)
            if sz >= 2
        )
        s_pooled = float(np.sqrt(ss / dof_total))
        # Bias-correct via c4 evaluated at equivalent sample size (dof + 1).
        c4 = _lookup_c4(dof_total + 1)
        sigma = s_pooled / c4
        detail = {
            "s_pooled": s_pooled,
            "pooled_dof": float(dof_total),
            "c4": c4,
        }
    elif resolved is WithinEstimator.MR_D2:
        raise ValueError(
            "MR_D2 is the individuals estimator; call "
            "sigma_within_from_individuals() instead of passing subgroups."
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unhandled estimator {resolved!r}.")

    return WithinSigmaResult(
        sigma=float(sigma),
        estimator=resolved,
        subgroup_size=n,
        n_subgroups=k,
        detail=detail,
    )


def moving_ranges(values: ArrayLike) -> NDArray[np.float64]:
    """Return the moving ranges ``|x_i - x_{i-1}|`` of a time-ordered series."""
    arr = to_float_array(values, nan_policy="omit")
    if arr.size < 2:
        raise ValueError("Moving ranges require at least 2 observations.")
    return np.abs(np.diff(arr))


def sigma_within_from_individuals(values: ArrayLike) -> WithinSigmaResult:
    """Estimate within sigma from individuals via the average moving range.

    ``sigma_within = MR-bar / d2(2) = MR-bar / 1.128``. See FORMULAS.md §2.2(d).
    """
    mrs = moving_ranges(values)
    mr_bar = float(mrs.mean())
    d2_2 = _lookup_d2(2)
    sigma = mr_bar / d2_2
    arr = to_float_array(values, nan_policy="omit")
    return WithinSigmaResult(
        sigma=float(sigma),
        estimator=WithinEstimator.MR_D2,
        subgroup_size=2,
        n_subgroups=int(mrs.size),  # number of moving-range pairs
        detail={"MR_bar": mr_bar, "d2": d2_2, "n_observations": float(arr.size)},
    )


def sigma_within(
    subgroups: Optional[Sequence[ArrayLike]] = None,
    individuals: Optional[ArrayLike] = None,
    *,
    estimator: WithinEstimator = WithinEstimator.AUTO,
) -> WithinSigmaResult:
    """Dispatch to the correct within-sigma estimator based on data shape.

    Provide exactly one of ``subgroups`` (list of subgroup arrays, sizes >= 2)
    or ``individuals`` (a single time-ordered series, n = 1 per point).
    """
    if (subgroups is None) == (individuals is None):
        raise ValueError(
            "Provide exactly one of `subgroups` or `individuals`."
        )
    if individuals is not None:
        return sigma_within_from_individuals(individuals)
    assert subgroups is not None
    return sigma_within_from_subgroups(subgroups, estimator=estimator)


def sigma_overall(values: ArrayLike, *, ddof: int = 1) -> float:
    """Overall (long-term) sigma: sample std dev of all pooled observations.

    This is the *only* sigma that Pp/Ppk may use. See FORMULAS.md §2.3.
    Requires at least ``ddof + 1`` observations.
    """
    arr = to_float_array(values, nan_policy="omit")
    if arr.size <= ddof:
        raise ValueError(
            f"Overall sigma with ddof={ddof} requires more than {ddof} "
            f"observation(s); got {arr.size}."
        )
    return float(np.std(arr, ddof=ddof))


def grand_mean(values: ArrayLike) -> float:
    """Grand mean of all pooled individual observations (``x-double-bar``).

    Computed on the flat pooled data. For balanced subgroups this equals the
    mean of subgroup means; for unbalanced data the pooled mean is the correct
    process-location estimate.
    """
    arr = to_float_array(values, nan_policy="omit")
    return float(arr.mean())
