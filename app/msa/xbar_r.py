"""AIAG Average & Range (X-bar & R) method for Gage R&R.

For ``n`` parts, ``o`` appraisers and ``r`` trials (AIAG MSA 4th ed.)::

    R-bar-bar = average over appraisers of each appraiser's average range
    EV  = R-bar-bar * K1(r)                                  (repeatability)
    AV  = sqrt( (X-bar_diff * K2(o))^2 - EV^2 / (n r) )      (reproducibility)
          X-bar_diff = max - min of the appraiser averages;
          a negative radicand gives AV = 0
    GRR = sqrt(EV^2 + AV^2)
    PV  = R_p * K3(n),  R_p = max - min of the part averages
    TV  = sqrt(GRR^2 + PV^2)

All of these are standard-deviation estimates (the 4th edition dropped the
5.15 multiplier; see :mod:`app.msa.metrics` for study variation). The method
cannot separate the operator x part interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.msa.constants import k1, k2, k3
from app.msa.models import ComponentEstimates, GageStudy
from app.msa.ranges import cell_ranges

__all__ = ["AverageRangeDetail", "average_range_gage_rr"]


@dataclass(frozen=True, slots=True)
class AverageRangeDetail:
    """Intermediate quantities, exposed so a reviewer can check them by hand."""

    operator_avg_range: tuple[float, ...]
    r_bar_bar: float
    operator_averages: tuple[float, ...]
    x_bar_diff: float
    part_averages: tuple[float, ...]
    r_p: float
    k1: float
    k2: float
    k3: float
    av_radicand: float


def average_range_gage_rr(study: GageStudy) -> tuple[ComponentEstimates, AverageRangeDetail]:
    """Estimate EV, AV and PV with the Average & Range method.

    Raises
    ------
    ValueError
        When the study size is outside the AIAG K-tables (trials or appraisers
        not in {2, 3}, parts not in 2..10).
    """
    n, o, r = study.n_parts, study.n_operators, study.n_trials
    c1, c2, c3 = k1(r), k2(o), k3(n)

    y = study.values
    ranges = cell_ranges(study)  # (parts, operators)
    op_avg_range = ranges.mean(axis=0)
    r_bar_bar = float(op_avg_range.mean())
    op_avg = y.mean(axis=(0, 2))
    x_diff = float(op_avg.max() - op_avg.min())
    part_avg = y.mean(axis=(1, 2))
    r_p = float(part_avg.max() - part_avg.min())
    # Differences of means within rounding noise are exactly zero (see
    # app.msa.models.rounding_floor); cell ranges are exact differences.
    floor = study.rounding_floor
    if x_diff <= floor:
        x_diff = 0.0
    if r_p <= floor:
        r_p = 0.0

    ev = r_bar_bar * c1
    radicand = (x_diff * c2) ** 2 - ev**2 / (n * r)
    av = sqrt(radicand) if radicand > 0.0 else 0.0
    pv = r_p * c3

    components = ComponentEstimates(
        repeatability_sd=ev,
        reproducibility_sd=av,
        part_sd=pv,
        operator_sd=None,
        interaction_sd=None,
        clamped=("operator",) if radicand < 0.0 else (),
        raw_estimates={"av_radicand": radicand},
    )
    detail = AverageRangeDetail(
        operator_avg_range=tuple(float(x) for x in op_avg_range),
        r_bar_bar=r_bar_bar,
        operator_averages=tuple(float(x) for x in op_avg),
        x_bar_diff=x_diff,
        part_averages=tuple(float(x) for x in part_avg),
        r_p=r_p,
        k1=c1,
        k2=c2,
        k3=c3,
        av_radicand=radicand,
    )
    return components, detail
