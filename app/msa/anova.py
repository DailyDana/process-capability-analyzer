"""Two-way crossed random-effects ANOVA for Gage R&R.

Model (parts P, operators O, both random, crossed, ``r`` replicates)::

    y_ijk = mu + P_i + O_j + (PO)_ij + e_ijk

Sums of squares for ``p`` parts, ``o`` operators, ``r`` trials::

    SS_P  = o r sum_i (ybar_i.. - ybar)^2              df = p - 1
    SS_O  = p r sum_j (ybar_.j. - ybar)^2              df = o - 1
    SS_PO = r sum_ij (ybar_ij. - ybar_i.. - ybar_.j. + ybar)^2
                                                       df = (p - 1)(o - 1)
    SS_E  = sum_ijk (y_ijk - ybar_ij.)^2               df = p o (r - 1)

With the interaction in the model the F tests use the interaction mean square
as the denominator for the main effects (random-effects model)::

    F_P = MS_P / MS_PO,  F_O = MS_O / MS_PO,  F_PO = MS_PO / MS_E

and the expected-mean-square solutions are::

    s2_e  = MS_E
    s2_PO = (MS_PO - MS_E) / r
    s2_O  = (MS_O - MS_PO) / (p r)
    s2_P  = (MS_P - MS_PO) / (o r)

**Interaction pooling (statistical-convention flag).** When the interaction
p-value exceeds ``alpha_pool`` (default 0.25, the AIAG/Minitab convention) the
interaction is removed and its sum of squares pooled into error::

    MS_E' = (SS_PO + SS_E) / (df_PO + df_E)
    s2_e  = MS_E',  s2_O = (MS_O - MS_E') / (p r),  s2_P = (MS_P - MS_E') / (o r)

Negative component estimates are set to zero and reported as clamped.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np
from scipy.stats import f as f_dist

from app.msa.models import ComponentEstimates, GageStudy

__all__ = ["AnovaRow", "AnovaTable", "AnovaResult", "DEFAULT_ALPHA_POOL", "anova_gage_rr"]

DEFAULT_ALPHA_POOL: float = 0.25


@dataclass(frozen=True, slots=True)
class AnovaRow:
    source: str
    df: int
    ss: float
    ms: Optional[float]
    f: Optional[float] = None
    p: Optional[float] = None


@dataclass(frozen=True, slots=True)
class AnovaTable:
    rows: tuple[AnovaRow, ...]

    def row(self, source: str) -> AnovaRow:
        for r in self.rows:
            if r.source == source:
                return r
        raise KeyError(source)


@dataclass(frozen=True, slots=True)
class AnovaResult:
    """Full-model table, the reduced (pooled) table if used, and the components."""

    full: AnovaTable
    reduced: Optional[AnovaTable]
    interaction_p: Optional[float]
    interaction_pooled: bool
    alpha_pool: float
    components: ComponentEstimates

    @property
    def table(self) -> AnovaTable:
        """The table the components were derived from."""
        return self.reduced if self.reduced is not None else self.full


def _f_test(ms_num: float, ms_den: float, df_num: int, df_den: int) -> tuple[Optional[float], Optional[float]]:
    """F statistic and upper-tail p-value; ``(None, None)`` when undefined.

    A zero denominator with a positive numerator is an infinitely strong
    effect (p = 0); 0/0 carries no information at all.
    """
    if ms_den > 0.0:
        f = ms_num / ms_den
        return f, float(f_dist.sf(f, df_num, df_den))
    if ms_num > 0.0:
        return float("inf"), 0.0
    return None, None


def _sums_of_squares(y: np.ndarray, floor: float) -> dict[str, float]:
    """Sums of squares; any SS within rounding noise of zero is set to 0.

    ``floor`` is :func:`app.msa.models.rounding_floor` of the data: no SS term
    can exceed ``N * floor^2`` from rounding alone (N = number of readings).
    """
    p, o, r = y.shape
    y = y - y.mean()  # centring removes the offset before squaring
    grand = y.mean()
    part_mean = y.mean(axis=(1, 2))
    op_mean = y.mean(axis=(0, 2))
    cell_mean = y.mean(axis=2)
    ss_p = o * r * float(((part_mean - grand) ** 2).sum())
    ss_o = p * r * float(((op_mean - grand) ** 2).sum())
    inter = cell_mean - part_mean[:, None] - op_mean[None, :] + grand
    ss_po = r * float((inter**2).sum())
    ss_e = float(((y - cell_mean[:, :, None]) ** 2).sum())
    ss_t = float(((y - grand) ** 2).sum())
    noise = y.size * floor**2
    ss = {"P": ss_p, "O": ss_o, "PO": ss_po, "E": ss_e, "T": ss_t}
    return {k: (0.0 if v <= noise else v) for k, v in ss.items()}


def anova_gage_rr(study: GageStudy, alpha_pool: float = DEFAULT_ALPHA_POOL) -> AnovaResult:
    """Run the crossed Gage R&R ANOVA on ``study``.

    Parameters
    ----------
    alpha_pool:
        Interaction removal threshold. The interaction is pooled into error
        when its p-value is strictly greater than ``alpha_pool``. ``1.0``
        therefore always keeps it (the full model); ``0.0`` pools it unless
        p == 0. When the interaction F test is undefined (0/0) it is pooled.
    """
    if not 0.0 <= alpha_pool <= 1.0:
        raise ValueError(f"alpha_pool must be in [0, 1]; got {alpha_pool}.")

    y = study.values
    p, o, r = y.shape
    ss = _sums_of_squares(y, study.rounding_floor)
    df_p, df_o, df_po, df_e = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    df_t = p * o * r - 1
    ms_p, ms_o, ms_po, ms_e = ss["P"] / df_p, ss["O"] / df_o, ss["PO"] / df_po, ss["E"] / df_e

    f_p, p_p = _f_test(ms_p, ms_po, df_p, df_po)
    f_o, p_o = _f_test(ms_o, ms_po, df_o, df_po)
    f_po, p_po = _f_test(ms_po, ms_e, df_po, df_e)
    full = AnovaTable(
        rows=(
            AnovaRow("Part", df_p, ss["P"], ms_p, f_p, p_p),
            AnovaRow("Operator", df_o, ss["O"], ms_o, f_o, p_o),
            AnovaRow("Part x Operator", df_po, ss["PO"], ms_po, f_po, p_po),
            AnovaRow("Repeatability", df_e, ss["E"], ms_e),
            AnovaRow("Total", df_t, ss["T"], None),
        )
    )

    # p_po is None only for 0/0 (no interaction and no error variation): there
    # is nothing to keep, so pool.
    pooled = p_po is None or p_po > alpha_pool

    reduced: Optional[AnovaTable] = None
    if pooled:
        df_err = df_po + df_e
        ms_err = (ss["PO"] + ss["E"]) / df_err
        rf_p, rp_p = _f_test(ms_p, ms_err, df_p, df_err)
        rf_o, rp_o = _f_test(ms_o, ms_err, df_o, df_err)
        reduced = AnovaTable(
            rows=(
                AnovaRow("Part", df_p, ss["P"], ms_p, rf_p, rp_p),
                AnovaRow("Operator", df_o, ss["O"], ms_o, rf_o, rp_o),
                AnovaRow("Repeatability", df_err, ss["PO"] + ss["E"], ms_err),
                AnovaRow("Total", df_t, ss["T"], None),
            )
        )
        raw = {
            "repeatability": ms_err,
            "operator": (ms_o - ms_err) / (p * r),
            "interaction": 0.0,
            "part": (ms_p - ms_err) / (o * r),
        }
    else:
        raw = {
            "repeatability": ms_e,
            "operator": (ms_o - ms_po) / (p * r),
            "interaction": (ms_po - ms_e) / r,
            "part": (ms_p - ms_po) / (o * r),
        }

    clamped = tuple(name for name, v in raw.items() if v < 0.0)
    var = {name: max(0.0, v) for name, v in raw.items()}
    components = ComponentEstimates(
        repeatability_sd=sqrt(var["repeatability"]),
        reproducibility_sd=sqrt(var["operator"] + var["interaction"]),
        part_sd=sqrt(var["part"]),
        operator_sd=sqrt(var["operator"]),
        interaction_sd=None if pooled else sqrt(var["interaction"]),
        clamped=clamped,
        raw_estimates=raw,
    )
    return AnovaResult(
        full=full,
        reduced=reduced,
        interaction_p=p_po,
        interaction_pooled=pooled,
        alpha_pool=alpha_pool,
        components=components,
    )
