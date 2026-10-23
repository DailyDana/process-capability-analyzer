"""Deterministic interpretation of a non-normal capability analysis.

Facts (codes ``nn_*``) and advice (``rec_nn_*``) are kept apart, as in
:mod:`app.interpret.engine`. The Ppk bands are the same engineering
conventions as for normal data (1.00 / 1.33 / 1.67, configurable).

What this interpretation adds is the comparison the user needs: how wrong
the normal assumption would have been. When normal-theory Ppk or PPM differ
materially from the non-normal result, that is stated with both numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from app.interpret.models import (
    ConclusionTopic,
    EngineeringRecommendation,
    InterpretationReport,
    RecommendationCategory,
    Severity,
    StatisticalConclusion,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.capability.nonnormal import BoxCoxResult, DistributionFit, NonNormalSpec
    from app.services.nonnormal import NonNormalSummary

__all__ = ["NonNormalThresholds", "interpret_nonnormal"]


@dataclass(frozen=True, slots=True)
class NonNormalThresholds:
    """Cut-points (engineering conventions, configurable).

    marginal_min, capable_min, excellent_min:
        Ppk bands, as for normal data.
    poor_fit_ad:
        Best A^2 above this is flagged as "no candidate fits well". 1.0 is a
        conservative round number: the 5 % critical values of A^2 with
        estimated parameters are about 0.75-0.8 for the families used.
    normal_gap_ratio:
        Normal-theory Ppk is "materially different" when the relative gap is at
        least this (default 0.10), or when its PPM differs by a factor >= 2.
    """

    marginal_min: float = 1.00
    capable_min: float = 1.33
    excellent_min: float = 1.67
    poor_fit_ad: float = 1.0
    normal_gap_ratio: float = 0.10

    def __post_init__(self) -> None:
        if not self.marginal_min < self.capable_min < self.excellent_min:
            raise ValueError("Require marginal_min < capable_min < excellent_min.")


def _band(ppk: float, t: NonNormalThresholds) -> tuple[str, Severity, str]:
    if ppk < t.marginal_min:
        return "inadequate", Severity.CRITICAL, "not capable"
    if ppk < t.capable_min:
        return "marginal", Severity.WARNING, "marginal"
    if ppk < t.excellent_min:
        return "capable", Severity.INFO, "capable"
    return "excellent", Severity.INFO, "capable with a substantial margin"


def interpret_nonnormal(
    s: "NonNormalSummary",
    fits: Sequence["DistributionFit"],
    chosen: Optional["DistributionFit"],
    bc: Optional["BoxCoxResult"],
    spec: "NonNormalSpec",
    t: NonNormalThresholds = NonNormalThresholds(),
) -> InterpretationReport:
    concl: list[StatisticalConclusion] = []
    recs: list[EngineeringRecommendation] = []

    # --- model ----------------------------------------------------------------
    best = fits[0] if fits else None
    if bc is not None:
        concl.append(StatisticalConclusion(
            "nn_model_boxcox", ConclusionTopic.NORMALITY, Severity.INFO,
            f"Box-Cox transformation with lambda = {bc.lambda_used:g} (maximum-likelihood {bc.lambda_mle:.3f}); "
            f"A^2 of the transformed data against a normal = {bc.ad_statistic:.3f}.",
            {"lambda": bc.lambda_used, "ad": bc.ad_statistic}))
    elif chosen is not None:
        others = ", ".join(f"{f.distribution.value} {f.ad_statistic:.2f}" for f in fits if f is not chosen)
        concl.append(StatisticalConclusion(
            "nn_model_fit", ConclusionTopic.NORMALITY, Severity.INFO,
            f"Model: {chosen.distribution.value} (Anderson-Darling A^2 = {chosen.ad_statistic:.3f}"
            + (f"; others: {others}" if others else "") + ").",
            {"distribution": chosen.distribution.value, "ad": chosen.ad_statistic}))
        if best is not None and chosen is not best:
            concl.append(StatisticalConclusion(
                "nn_not_best_fit", ConclusionTopic.NORMALITY, Severity.NOTICE,
                f"The selected {chosen.distribution.value} model is not the best fit: "
                f"{best.distribution.value} has a lower A^2 ({best.ad_statistic:.3f}).",
                {"best": best.distribution.value}))
    if s.fit_statistic > t.poor_fit_ad:
        concl.append(StatisticalConclusion(
            "nn_poor_fit", ConclusionTopic.NORMALITY, Severity.WARNING,
            f"The model fits poorly (A^2 = {s.fit_statistic:.2f} > {t.poor_fit_ad:g}): the tail percentiles, and so "
            "Ppk and the PPM, may be badly estimated.",
            {"ad": s.fit_statistic}))
        recs.append(EngineeringRecommendation(
            "rec_nn_check_fit", RecommendationCategory.CHECK_DISTRIBUTION, Severity.WARNING,
            "Look for a mixture of processes (shifts, cavities, machines) or a special cause before trusting a "
            "single distribution; stratify the data if needed.", "nn_poor_fit"))
    if spec.lsl_is_boundary:
        concl.append(StatisticalConclusion(
            "nn_lower_boundary", ConclusionTopic.ONE_SIDED_SPEC, Severity.INFO,
            f"The LSL ({spec.lsl:g}) is treated as a natural lower boundary, so only the upper side is assessed "
            "(no Pp, no Ppl).", {"lsl": spec.lsl}))

    # --- indices ----------------------------------------------------------------
    band, sev, words = _band(s.ppk, t)
    concl.append(StatisticalConclusion(
        f"nn_ppk_{band}", ConclusionTopic.PERFORMANCE, sev,
        f"Ppk = {s.ppk:.2f} ({s.method_label}): {words} by the {t.capable_min:g} convention. Expected "
        f"{s.ppm_total:,.0f} PPM out of specification (observed in the sample: {s.observed_ppm:,.0f} PPM).",
        {"ppk": s.ppk, "ppm": s.ppm_total, "observed_ppm": s.observed_ppm}))

    # --- the comparison that matters --------------------------------------------
    gap = abs(s.normal_ppk - s.ppk) / abs(s.ppk) if s.ppk else float("inf")
    ppm_ratio = (max(s.normal_ppm, s.ppm_total) / min(s.normal_ppm, s.ppm_total)
                 if min(s.normal_ppm, s.ppm_total) > 0 else float("inf"))
    if gap >= t.normal_gap_ratio or ppm_ratio >= 2.0:
        direction = "overstates" if s.normal_ppk > s.ppk else "understates"
        concl.append(StatisticalConclusion(
            "nn_normal_misleading", ConclusionTopic.NORMALITY, Severity.WARNING,
            f"Assuming normality would give Ppk = {s.normal_ppk:.2f} and {s.normal_ppm:,.0f} PPM: it {direction} "
            f"the capability compared with {s.ppk:.2f} and {s.ppm_total:,.0f} PPM.",
            {"normal_ppk": s.normal_ppk, "normal_ppm": s.normal_ppm}))
    else:
        concl.append(StatisticalConclusion(
            "nn_normal_close", ConclusionTopic.NORMALITY, Severity.INFO,
            f"The normal assumption would give a similar result (Ppk = {s.normal_ppk:.2f}).",
            {"normal_ppk": s.normal_ppk}))

    if s.ppk < t.capable_min:
        recs.append(EngineeringRecommendation(
            "rec_nn_improve", RecommendationCategory.REDUCE_VARIATION,
            Severity.CRITICAL if band == "inadequate" else Severity.WARNING,
            "Reduce the spread in the tail that faces the specification, or review the specification with the "
            "customer; re-run the study after the change.", f"nn_ppk_{band}"))
    elif s.fit_statistic <= t.poor_fit_ad and s.n >= 100:
        recs.append(EngineeringRecommendation(
            "rec_nn_maintain", RecommendationCategory.MAINTAIN, Severity.INFO,
            "Capable under a well-fitting model: keep the process under SPC and repeat the study after changes.",
            f"nn_ppk_{band}"))
    if s.n < 100:
        recs.append(EngineeringRecommendation(
            "rec_nn_more_data", RecommendationCategory.COLLECT_MORE_DATA, Severity.NOTICE,
            "Collect at least 100 measurements: tail percentiles from small samples rest mostly on the model.",
            f"nn_ppk_{band}"))

    headline = (f"Ppk = {s.ppk:.2f} ({words}), {s.ppm_total:,.0f} PPM expected [{s.method_label}]; "
                f"normal theory would say Ppk = {s.normal_ppk:.2f}.")
    return InterpretationReport(tuple(concl), tuple(recs), headline)
