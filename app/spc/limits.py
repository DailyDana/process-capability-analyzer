"""Control-limit calculators for X-bar/R, X-bar/S, and I-MR charts.

Each calculator returns a :class:`ControlChart` describing one plotted series
(e.g. the X-bar chart or the R chart) with its center line and 3-sigma control
limits, the plotted point values, the per-point beyond-limit flags, and the
zone sigma used by the run-rule engine.

All limits come from the plotted statistic's own average (R-bar, S-bar, MR-bar)
via the validated M1 control-chart constants — deliberately independent of the
M2 capability sigma. See ``docs/FORMULAS.md`` §3.1-3.5.

Design guarantees
-----------------
* Missing/non-finite values are handled explicitly (raise or record), never
  silently dropped; individual-chart indices are preserved.
* Unequal subgroup sizes are rejected with a structured error (variable-limit
  charts are out of scope for v1).
* Insufficient data (k < 2 subgroups, or < 2 individuals) raises a clear error.
* Full float precision; a point exactly on a limit is NOT flagged (strict
  inequality triggers a beyond-limit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.statistics import constants as const
from app.statistics.descriptive import std_dev

__all__ = [
    "ChartType",
    "ControlChart",
    "ChartPair",
    "xbar_r_chart",
    "xbar_s_chart",
    "imr_chart",
]


class ChartType(str, Enum):
    """Identifies which chart a :class:`ControlChart` represents."""

    XBAR_R = "xbar_r"  # X-bar (location) paired with R
    XBAR_S = "xbar_s"  # X-bar (location) paired with S
    R = "r"  # range dispersion
    S = "s"  # std-dev dispersion
    INDIVIDUALS = "individuals"  # I chart
    MOVING_RANGE = "moving_range"  # MR chart


# Charts with a symmetric +/-3 sigma band get the full run-rule set; the
# dispersion charts (R, S, MR) are asymmetric and get only the beyond-limit test.
SYMMETRIC_CHARTS = frozenset({ChartType.XBAR_R, ChartType.XBAR_S, ChartType.INDIVIDUALS})


def _lookup(n: int) -> const.ChartConstants:
    """Return the tabulated constants for subgroup size ``n`` (table for 2..25)."""
    return const.get_constants(n)


@dataclass(frozen=True, slots=True)
class ControlChart:
    """One plotted control-chart series with its limits and flags.

    Attributes
    ----------
    chart_type:
        Which chart this is (:class:`ChartType`).
    points:
        The plotted statistic per position (subgroup means, ranges, individuals,
        moving ranges). Full precision, original order preserved.
    center_line, ucl, lcl:
        Center line and 3-sigma control limits.
    sigma_points:
        Standard deviation of the *plotted* statistic, ``(UCL - CL)/3``. Used by
        the run-rule zone construction. ``0.0`` when there is no variation.
    beyond_limits:
        Boolean array, ``True`` where a point is strictly outside [LCL, UCL].
    is_symmetric:
        Whether this chart supports the full run-rule set (X-bar/I) or only the
        beyond-limit test (R/S/MR).
    point_indices:
        Original observation/subgroup indices for each plotted point, so that
        violations map back to the user's rows even if positions were excluded.
    detail:
        Intermediate quantities (``R_bar``/``S_bar``/``MR_bar``, constants used).
    warnings:
        Structured advisory notes (e.g. autocorrelation caveat, excluded rows).
    """

    chart_type: ChartType
    points: NDArray[np.float64]
    center_line: float
    ucl: float
    lcl: float
    sigma_points: float
    beyond_limits: NDArray[np.bool_]
    is_symmetric: bool
    point_indices: NDArray[np.int_]
    detail: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return int(self.points.size)

    @property
    def n_beyond(self) -> int:
        return int(np.count_nonzero(self.beyond_limits))

    def as_dict(self) -> dict[str, object]:
        return {
            "chart_type": self.chart_type.value,
            "points": self.points.tolist(),
            "center_line": self.center_line,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "sigma_points": self.sigma_points,
            "beyond_limits": self.beyond_limits.tolist(),
            "is_symmetric": self.is_symmetric,
            "point_indices": self.point_indices.tolist(),
            "detail": self.detail,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ChartPair:
    """A location chart paired with its dispersion chart (X-bar+R, X-bar+S, I+MR).

    The dispersion chart is conventionally assessed *first*: if the spread is out
    of control the location-chart limits are not trustworthy. Both are returned
    so the caller/UI can present them together.
    """

    location: ControlChart
    dispersion: ControlChart

    def as_dict(self) -> dict[str, object]:
        return {
            "location": self.location.as_dict(),
            "dispersion": self.dispersion.as_dict(),
        }


def _validate_subgroups(
    subgroups: Sequence[ArrayLike],
) -> tuple[list[NDArray[np.float64]], int]:
    """Coerce, validate, and return balanced subgroup arrays plus their size.

    Raises
    ------
    ValueError
        If there are fewer than 2 subgroups, if any value is non-finite (missing
        data must be handled before charting), or if subgroup sizes are unequal.
    """
    if len(subgroups) < 2:
        raise ValueError(
            f"Control limits need at least 2 subgroups; got {len(subgroups)}."
        )
    arrays: list[NDArray[np.float64]] = []
    for j, g in enumerate(subgroups):
        arr = np.asarray(g, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError(f"Subgroup {j} is empty.")
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Subgroup {j} contains missing/non-finite values. Clean the "
                "data (validation layer) before building control charts; "
                "observations are never silently dropped here."
            )
        arrays.append(arr)
    sizes = {a.size for a in arrays}
    if len(sizes) != 1:
        raise ValueError(
            f"Subgroups must all have the same size for fixed control limits; "
            f"got sizes {sorted(sizes)}. Variable-limit charts are not supported "
            "in this version."
        )
    n = sizes.pop()
    if n < 2:
        raise ValueError(
            "X-bar/R and X-bar/S require subgroup size n >= 2; for n = 1 use the "
            "I-MR chart (imr_chart)."
        )
    if n > const.MAX_TABULATED_N:
        raise ValueError(
            f"Subgroup size n={n} exceeds the tabulated constant range "
            f"(<= {const.MAX_TABULATED_N})."
        )
    return arrays, n


def _make_chart(
    *,
    chart_type: ChartType,
    points: NDArray[np.float64],
    center_line: float,
    ucl: float,
    lcl: float,
    point_indices: Optional[NDArray[np.int_]] = None,
    detail: Optional[dict[str, float]] = None,
    warnings: tuple[str, ...] = (),
) -> ControlChart:
    """Assemble a ControlChart, computing sigma_points and beyond-limit flags."""
    pts = np.asarray(points, dtype=np.float64)
    # sigma of the plotted statistic; symmetric charts have (UCL-CL)==(CL-LCL).
    sigma_points = (ucl - center_line) / 3.0
    # Strict inequality: a point exactly on a limit is NOT out of control.
    beyond = (pts > ucl) | (pts < lcl)
    if point_indices is None:
        point_indices = np.arange(pts.size, dtype=int)
    return ControlChart(
        chart_type=chart_type,
        points=pts,
        center_line=float(center_line),
        ucl=float(ucl),
        lcl=float(lcl),
        sigma_points=float(sigma_points),
        beyond_limits=beyond,
        is_symmetric=chart_type in SYMMETRIC_CHARTS,
        point_indices=point_indices,
        detail=detail or {},
        warnings=warnings,
    )


def xbar_r_chart(subgroups: Sequence[ArrayLike]) -> ChartPair:
    """Build the X-bar and R control charts (FORMULAS.md §3.2).

    ``X-bar: x-double-bar +/- A2*R-bar``; ``R: [D3*R-bar, D4*R-bar]``.
    Requires >= 2 balanced subgroups of size 2..25.
    """
    groups, n = _validate_subgroups(subgroups)
    c = _lookup(n)

    means = np.array([g.mean() for g in groups], dtype=np.float64)
    ranges = np.array([g.max() - g.min() for g in groups], dtype=np.float64)
    x_dbar = float(means.mean())
    r_bar = float(ranges.mean())

    xbar = _make_chart(
        chart_type=ChartType.XBAR_R,
        points=means,
        center_line=x_dbar,
        ucl=x_dbar + c.a2 * r_bar,
        lcl=x_dbar - c.a2 * r_bar,
        detail={"x_double_bar": x_dbar, "R_bar": r_bar, "A2": c.a2, "n": float(n)},
    )
    rchart = _make_chart(
        chart_type=ChartType.R,
        points=ranges,
        center_line=r_bar,
        ucl=c.d4_factor * r_bar,
        lcl=c.d3_factor * r_bar,
        detail={"R_bar": r_bar, "D3": c.d3_factor, "D4": c.d4_factor, "n": float(n)},
    )
    return ChartPair(location=xbar, dispersion=rchart)


def xbar_s_chart(subgroups: Sequence[ArrayLike]) -> ChartPair:
    """Build the X-bar and S control charts (FORMULAS.md §3.3).

    ``X-bar: x-double-bar +/- A3*S-bar``; ``S: [B3*S-bar, B4*S-bar]``.
    Requires >= 2 balanced subgroups of size 2..25.
    """
    groups, n = _validate_subgroups(subgroups)
    c = _lookup(n)

    means = np.array([g.mean() for g in groups], dtype=np.float64)
    sds = np.array([std_dev(g, ddof=1) for g in groups], dtype=np.float64)
    x_dbar = float(means.mean())
    s_bar = float(sds.mean())

    xbar = _make_chart(
        chart_type=ChartType.XBAR_S,
        points=means,
        center_line=x_dbar,
        ucl=x_dbar + c.a3 * s_bar,
        lcl=x_dbar - c.a3 * s_bar,
        detail={"x_double_bar": x_dbar, "S_bar": s_bar, "A3": c.a3, "n": float(n)},
    )
    schart = _make_chart(
        chart_type=ChartType.S,
        points=sds,
        center_line=s_bar,
        ucl=c.b4 * s_bar,
        lcl=c.b3 * s_bar,
        detail={"S_bar": s_bar, "B3": c.b3, "B4": c.b4, "n": float(n)},
    )
    return ChartPair(location=xbar, dispersion=schart)


def imr_chart(
    values: ArrayLike,
    *,
    on_missing: str = "raise",
) -> ChartPair:
    """Build the Individuals and Moving-Range charts (FORMULAS.md §3.4).

    Moving-range window w = 2: ``MR_i = |x_i - x_{i-1}|``. Individuals limits are
    ``x-bar +/- E2*MR-bar`` (E2 = 3/d2(2)); MR limits are ``[0, D4(2)*MR-bar]``.

    Parameters
    ----------
    values:
        Time-ordered individual measurements (>= 2 usable points).
    on_missing:
        ``"raise"`` (default) raises on any NaN/inf; ``"exclude"`` drops
        non-finite points, records a warning, and **preserves original indices**
        for the retained points so violations still map to user rows.

    Raises
    ------
    ValueError
        If fewer than 2 usable observations remain, or on non-finite data under
        ``on_missing="raise"``.
    """
    raw = np.asarray(values, dtype=np.float64).ravel()
    if raw.size == 0:
        raise ValueError("I-MR chart requires at least 2 observations; got 0.")

    finite_mask = np.isfinite(raw)
    warnings: list[str] = []
    if not finite_mask.all():
        if on_missing == "raise":
            raise ValueError(
                f"I-MR input contains {int((~finite_mask).sum())} missing/"
                "non-finite value(s). Pass on_missing='exclude' to drop them "
                "explicitly (original indices are preserved)."
            )
        elif on_missing == "exclude":
            dropped = int((~finite_mask).sum())
            warnings.append(
                f"{dropped} non-finite observation(s) excluded from the I-MR "
                "chart; original indices are preserved for the retained points. "
                "Note: excluding points breaks the time-adjacency assumption of "
                "the moving range across the gaps."
            )
        else:
            raise ValueError(f"Unknown on_missing policy {on_missing!r}.")

    idx = np.nonzero(finite_mask)[0].astype(int)
    x = raw[finite_mask]
    if x.size < 2:
        raise ValueError(
            f"I-MR chart requires at least 2 usable observations; got {x.size}."
        )

    x_bar = float(x.mean())
    mr = np.abs(np.diff(x))  # N-1 moving ranges
    mr_bar = float(mr.mean())

    c2 = _lookup(2)  # d2(2)=1.128, D4(2)=3.267, D3(2)=0, E2(2)=2.660
    # E2 CONVENTION (single source of truth): use the TABULATED E2 from the
    # validated constants table, matching the tabulated d2(2) that
    # `statistics.variation.sigma_within_from_individuals` uses for MR_bar/d2.
    # Chart limits and the capability sigma therefore rest on the same
    # convention. (Computing 3/d2 inline instead would give 2.65957 from the
    # tabulated d2, or 2.65868 from the exact d2 -- a third and fourth value.
    # See FORMULAS.md Sec. 3.4.)
    e2 = c2.e2

    ichart = _make_chart(
        chart_type=ChartType.INDIVIDUALS,
        points=x,
        center_line=x_bar,
        ucl=x_bar + e2 * mr_bar,
        lcl=x_bar - e2 * mr_bar,
        point_indices=idx,
        detail={"x_bar": x_bar, "MR_bar": mr_bar, "E2": e2, "d2_2": c2.d2},
        warnings=tuple(warnings),
    )
    # MR chart has N-1 points; align its indices to the *second* member of each
    # consecutive pair (the point at which the MR is computed).
    mr_indices = idx[1:]
    mrchart = _make_chart(
        chart_type=ChartType.MOVING_RANGE,
        points=mr,
        center_line=mr_bar,
        ucl=c2.d4_factor * mr_bar,
        lcl=c2.d3_factor * mr_bar,  # D3(2) = 0
        point_indices=mr_indices,
        detail={"MR_bar": mr_bar, "D4": c2.d4_factor, "D3": c2.d3_factor},
        warnings=tuple(warnings),
    )
    return ChartPair(location=ichart, dispersion=mrchart)
