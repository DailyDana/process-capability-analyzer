"""Non-normal capability orchestrator -- the single entry point of its page.

validate -> fit every distribution the data allow -> chosen method (best fit,
a named distribution, or Box-Cox) -> normal-theory comparison -> observed
PPM -> interpretation. Ordinary user errors come back as ``ok=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from app.capability.nonnormal import (
    BoxCoxResult,
    Distribution,
    DistributionFit,
    NonNormalSpec,
    PercentileIndices,
    box_cox,
    fit_all,
    fit_distribution,
    percentile_indices,
)
from app.data.validation import RowStatus, clean_numeric_detailed
from app.interpret.models import InterpretationReport
from app.interpret.nonnormal import NonNormalThresholds, interpret_nonnormal

__all__ = ["BEST_FIT", "BOX_COX", "NonNormalRequest", "NonNormalSummary", "NonNormalOutcome", "run_nonnormal",
           "MIN_N", "RECOMMENDED_N"]

BEST_FIT = "best_fit"
BOX_COX = "box_cox"
Method = Union[str, Distribution]

MIN_N = 10
#: Tail percentiles need data: common guidance is at least 100 observations.
RECOMMENDED_N = 100


@dataclass(frozen=True, slots=True)
class NonNormalRequest:
    values: tuple[object, ...]
    usl: Optional[float] = None
    lsl: Optional[float] = None
    lsl_is_boundary: bool = False
    method: Method = BEST_FIT
    unbias_c4: bool = False
    round_lambda: bool = True
    thresholds: NonNormalThresholds = field(default_factory=NonNormalThresholds)


@dataclass(frozen=True, slots=True)
class NonNormalSummary:
    """Method-neutral result shown in the UI and the exports."""

    method_label: str
    n: int
    pp: Optional[float]
    ppu: Optional[float]
    ppl: Optional[float]
    ppk: float
    ppm_below: float
    ppm_above: float
    observed_ppm: float
    normal_ppk: float
    normal_ppm: float
    fit_statistic: float  # A^2 of the chosen model (Box-Cox: of the transformed data vs normal)

    @property
    def ppm_total(self) -> float:
        return self.ppm_below + self.ppm_above


@dataclass(frozen=True, slots=True)
class NonNormalOutcome:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    spec: Optional[NonNormalSpec] = None
    fits: tuple[DistributionFit, ...] = ()
    skipped: dict[Distribution, str] = field(default_factory=dict)
    chosen_fit: Optional[DistributionFit] = None
    percentile: Optional[PercentileIndices] = None
    boxcox: Optional[BoxCoxResult] = None
    summary: Optional[NonNormalSummary] = None
    interpretation: Optional[InterpretationReport] = None


def _observed_ppm(x: np.ndarray, spec: NonNormalSpec) -> float:
    out = np.zeros(x.size, dtype=bool)
    if spec.usl is not None:
        out |= x > spec.usl
    if spec.effective_lsl is not None:
        out |= x < spec.effective_lsl
    return float(out.mean() * 1e6)


def run_nonnormal(req: NonNormalRequest) -> NonNormalOutcome:
    cleaned = clean_numeric_detailed(list(req.values))
    warnings: list[str] = []
    dropped = [r for r in cleaned.records if r.status is not RowStatus.OK]
    if dropped:
        warnings.append(f"{len(dropped)} blank or non-numeric row(s) excluded.")
    x = np.asarray(cleaned.values, dtype=float)
    if x.size < MIN_N:
        return NonNormalOutcome(False, (f"At least {MIN_N} measurements are required (found {x.size}).",), tuple(warnings))
    if x.size < RECOMMENDED_N:
        warnings.append(
            f"Only {x.size} measurements: the 0.135 % and 99.865 % percentiles are extrapolated from the "
            f"fitted model; at least {RECOMMENDED_N} are recommended."
        )
    try:
        spec = NonNormalSpec(req.usl, req.lsl, req.lsl_is_boundary)
    except ValueError as exc:
        return NonNormalOutcome(False, (str(exc),), tuple(warnings))

    fits, skipped = fit_all(x, req.unbias_c4)
    normal = fit_distribution(x, Distribution.NORMAL, req.unbias_c4)
    normal_pi = percentile_indices(normal, spec)
    observed = _observed_ppm(x, spec)

    chosen: Optional[DistributionFit] = None
    pi: Optional[PercentileIndices] = None
    bc: Optional[BoxCoxResult] = None
    try:
        if req.method == BOX_COX:
            bc = box_cox(x, spec, round_to_half=req.round_lambda, unbias_c4=req.unbias_c4)
            label = f"Box-Cox (lambda = {bc.lambda_used:g})"
            summary = NonNormalSummary(label, int(x.size), bc.pp, bc.ppu, bc.ppl, bc.ppk, bc.ppm_below, bc.ppm_above,
                                       observed, normal_pi.ppk, normal_pi.ppm_total, bc.ad_statistic)
        else:
            if req.method == BEST_FIT:
                chosen = fits[0]
            else:
                dist = Distribution(req.method)
                matches = [f for f in fits if f.distribution is dist]
                if not matches:
                    raise ValueError(skipped.get(dist, f"The {dist.value} distribution could not be fitted."))
                chosen = matches[0]
            pi = percentile_indices(chosen, spec)
            label = f"{chosen.distribution.value.capitalize()} fit, percentile method"
            summary = NonNormalSummary(label, int(x.size), pi.pp, pi.ppu, pi.ppl, pi.ppk, pi.ppm_below, pi.ppm_above,
                                       observed, normal_pi.ppk, normal_pi.ppm_total, chosen.ad_statistic)
    except ValueError as exc:
        return NonNormalOutcome(False, (str(exc),), tuple(warnings), tuple(float(v) for v in x), spec,
                                tuple(fits), skipped)

    rep = interpret_nonnormal(summary, fits, chosen, bc, spec, req.thresholds)
    return NonNormalOutcome(True, (), tuple(warnings), tuple(float(v) for v in x), spec, tuple(fits), skipped,
                            chosen, pi, bc, summary, rep)


def columns_to_request(values: Sequence[object], **options: object) -> NonNormalRequest:
    return NonNormalRequest(values=tuple(values), **options)  # type: ignore[arg-type]
