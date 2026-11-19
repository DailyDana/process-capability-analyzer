"""Phase I / Phase II control limits (``docs/FORMULAS.md`` §3.10).

In practice limits are set once, from a reference period in which the process
was stable (Phase I), and new data are then judged against those fixed limits
(Phase II). This module separates the two steps:

* :func:`estimate_limits` computes the limits from chosen baseline points
  only (the first ``n`` subgroups, minus any with an assignable cause);
* :func:`apply_limits` charts *all* the data against given limits, so the
  usual rule engine evaluates every point against them;
* :class:`FrozenLimits` carries the limits between the two and to and from a
  small CSV file, so a later session can reuse them.

The formulas are those of :mod:`app.spc.limits`, with the averages taken over
the baseline points only. For individuals a moving range is used when both of
its readings are consecutive baseline points; a range that spans an excluded
reading is left out rather than bridged.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.spc.limits import ChartPair, ChartType, _make_chart, _validate_subgroups, xbar_r_chart, xbar_s_chart
from app.statistics import constants as const
from app.statistics.descriptive import std_dev

__all__ = ["FrozenLimits", "estimate_limits", "apply_limits", "baseline_mask", "KINDS"]

KINDS = ("xbar_r", "xbar_s", "imr")
_FIELDS = ("location_cl", "location_ucl", "location_lcl", "dispersion_cl", "dispersion_ucl", "dispersion_lcl")


@dataclass(frozen=True, slots=True)
class FrozenLimits:
    """Fixed centre lines and limits for a location/dispersion chart pair."""

    kind: str  # "xbar_r", "xbar_s" or "imr"
    subgroup_size: int  # 1 for individuals
    location_cl: float
    location_ucl: float
    location_lcl: float
    dispersion_cl: float
    dispersion_ucl: float
    dispersion_lcl: float
    n_baseline: int = 0  # subgroups (or readings) the limits were estimated from; 0 if unknown
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown chart kind {self.kind!r}; expected one of {', '.join(KINDS)}.")
        if self.subgroup_size < 1 or (self.kind == "imr") != (self.subgroup_size == 1):
            raise ValueError("Subgroup size must be 1 for I-MR limits and at least 2 for X-bar limits.")
        values = [getattr(self, f) for f in _FIELDS]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Every limit must be a finite number.")
        if not self.location_lcl <= self.location_cl <= self.location_ucl:
            raise ValueError("The location limits must satisfy LCL <= centre line <= UCL.")
        if not 0.0 <= self.dispersion_lcl <= self.dispersion_cl <= self.dispersion_ucl:
            raise ValueError("The dispersion limits must satisfy 0 <= LCL <= centre line <= UCL.")

    def to_csv(self) -> str:
        """Two-column CSV at full precision (``repr`` of each float round-trips exactly)."""
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["field", "value"])
        w.writerow(["chart", self.kind])
        w.writerow(["subgroup_size", self.subgroup_size])
        for f in _FIELDS:
            w.writerow([f, repr(float(getattr(self, f)))])
        w.writerow(["n_baseline", self.n_baseline])
        w.writerow(["note", self.note])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str) -> "FrozenLimits":
        """Parse :meth:`to_csv` output. Raises ``ValueError`` with a plain message."""
        rows = {r[0].strip().lower(): r[1].strip() for r in csv.reader(io.StringIO(text)) if len(r) >= 2}
        missing = [f for f in ("chart", "subgroup_size", *_FIELDS) if f not in rows]
        if missing:
            raise ValueError(f"The limits file is missing: {', '.join(missing)}.")
        try:
            return cls(
                kind=rows["chart"].lower(),
                subgroup_size=int(float(rows["subgroup_size"])),
                **{f: float(rows[f]) for f in _FIELDS},
                n_baseline=int(float(rows.get("n_baseline") or 0)),
                note=rows.get("note", ""),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The limits file could not be read: {exc}") from exc


def baseline_mask(n_points: int, n_baseline: Optional[int] = None, exclude: Sequence[int] = ()) -> NDArray[np.bool_]:
    """Boolean mask of the points the limits are estimated from.

    ``n_baseline`` keeps the first that many points (all when ``None``);
    ``exclude`` holds zero-based positions removed from the estimate.
    """
    mask = np.ones(n_points, dtype=bool)
    if n_baseline is not None:
        if not 1 <= n_baseline <= n_points:
            raise ValueError(f"The baseline must be between 1 and {n_points} points; got {n_baseline}.")
        mask[n_baseline:] = False
    for pos in exclude:
        if not 0 <= int(pos) < n_points:
            raise ValueError(f"Excluded point {int(pos) + 1} does not exist (there are {n_points}).")
        mask[int(pos)] = False
    return mask


def _imr_arrays(values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError(f"I-MR chart requires at least 2 observations; got {x.size}.")
    if not np.isfinite(x).all():
        raise ValueError("I-MR input contains missing/non-finite values; clean the data first.")
    return x


def estimate_limits(kind: str, data: Sequence, mask: NDArray[np.bool_], note: str = "") -> FrozenLimits:
    """Limits from the baseline points selected by ``mask``.

    ``data`` is the list of subgroups (``xbar_r``, ``xbar_s``) or the series
    of individual readings (``imr``); ``mask`` has one entry per subgroup or
    reading.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown chart kind {kind!r}.")
    mask = np.asarray(mask, dtype=bool)
    if kind == "imr":
        x = _imr_arrays(data)
        if mask.size != x.size:
            raise ValueError("The baseline mask does not match the number of readings.")
        pair_ok = mask[1:] & mask[:-1]
        if mask.sum() < 2 or not pair_ok.any():
            raise ValueError(
                "The baseline needs at least two consecutive readings to estimate the moving range."
            )
        x_bar = float(x[mask].mean())
        mr_bar = float(np.abs(np.diff(x))[pair_ok].mean())
        c2 = const.get_constants(2)
        return FrozenLimits("imr", 1, x_bar, x_bar + c2.e2 * mr_bar, x_bar - c2.e2 * mr_bar,
                            mr_bar, c2.d4_factor * mr_bar, c2.d3_factor * mr_bar, int(mask.sum()), note)

    if mask.size != len(data):
        raise ValueError("The baseline mask does not match the number of subgroups.")
    chosen = [g for g, keep in zip(data, mask) if keep]
    if len(chosen) < 2:
        raise ValueError(f"The baseline needs at least 2 subgroups; {len(chosen)} left after exclusions.")
    pair = xbar_r_chart(chosen) if kind == "xbar_r" else xbar_s_chart(chosen)
    n = int(np.asarray(chosen[0]).size)
    loc, disp = pair.location, pair.dispersion
    return FrozenLimits(kind, n, loc.center_line, loc.ucl, loc.lcl, disp.center_line, disp.ucl, disp.lcl,
                        len(chosen), note)


def apply_limits(kind: str, data: Sequence, limits: FrozenLimits) -> ChartPair:
    """Chart every point of ``data`` against ``limits`` (nothing is re-estimated)."""
    if limits.kind != kind:
        raise ValueError(
            f"These limits are for a {limits.kind} chart, but the analysis is set to {kind}."
        )
    detail = {"frozen": 1.0, "n_baseline": float(limits.n_baseline)}
    if kind == "imr":
        x = _imr_arrays(data)
        idx = np.arange(x.size, dtype=int)
        loc = _make_chart(chart_type=ChartType.INDIVIDUALS, points=x, center_line=limits.location_cl,
                          ucl=limits.location_ucl, lcl=limits.location_lcl, point_indices=idx, detail=dict(detail))
        disp = _make_chart(chart_type=ChartType.MOVING_RANGE, points=np.abs(np.diff(x)),
                           center_line=limits.dispersion_cl, ucl=limits.dispersion_ucl,
                           lcl=limits.dispersion_lcl, point_indices=idx[1:], detail=dict(detail))
        return ChartPair(location=loc, dispersion=disp)

    groups, n = _validate_subgroups(data)
    if n != limits.subgroup_size:
        raise ValueError(
            f"These limits were set for subgroups of {limits.subgroup_size}, but the data are in subgroups of {n}. "
            "Limits depend on the subgroup size and cannot be reused across sizes."
        )
    means = np.array([g.mean() for g in groups], dtype=np.float64)
    if kind == "xbar_r":
        spread = np.array([g.max() - g.min() for g in groups], dtype=np.float64)
        loc_type, disp_type = ChartType.XBAR_R, ChartType.R
    else:
        spread = np.array([std_dev(g, ddof=1) for g in groups], dtype=np.float64)
        loc_type, disp_type = ChartType.XBAR_S, ChartType.S
    loc = _make_chart(chart_type=loc_type, points=means, center_line=limits.location_cl,
                      ucl=limits.location_ucl, lcl=limits.location_lcl, detail=dict(detail))
    disp = _make_chart(chart_type=disp_type, points=spread, center_line=limits.dispersion_cl,
                       ucl=limits.dispersion_ucl, lcl=limits.dispersion_lcl, detail=dict(detail))
    return ChartPair(location=loc, dispersion=disp)
