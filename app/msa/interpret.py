"""Deterministic interpretation of a Gage R&R result.

Same contract as :mod:`app.interpret.engine`: statistical conclusions (facts,
with a stable ``code``) are kept apart from engineering recommendations
(advice, linked by ``trigger_code``). No LLM, no randomness.

Acceptance bands (engineering convention, AIAG MSA 4th ed. -- NOT a
statistical law, and configurable in :class:`GageThresholds`)::

    %GRR <  10%          acceptable
    10% <= %GRR <= 30%   may be acceptable, depending on the application
    %GRR >  30%          unacceptable
    ndc  >= 5            adequate discrimination

%GRR is judged against study (total) variation, and additionally against the
tolerance when one is given -- the first answers "can the gauge see process
variation" (process control), the second "can it sort good parts from bad"
(product control). The headline reports the worse of the two.

Measurement consistency comes first: if any part x operator range is beyond
the range chart's UCL, every number is reported with that caveat, mirroring
"stability before capability" elsewhere in the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.interpret.models import (
    ConclusionTopic,
    EngineeringRecommendation,
    InterpretationReport,
    RecommendationCategory,
    Severity,
    StatisticalConclusion,
)
from app.msa.anova import AnovaResult
from app.msa.models import GageMethod, GageRRResult

__all__ = ["GageThresholds", "GrrVerdict", "grr_verdict", "headline_verdict", "interpret_gage"]


@dataclass(frozen=True, slots=True)
class GageThresholds:
    """Cut-points used by the interpreter (all documented conventions).

    acceptable_max, marginal_max:
        %GRR bands (see module docstring). Defaults 10 / 30.
    ndc_min:
        Minimum number of distinct categories. Default 5.
    dominance_ratio:
        One GRR component "dominates" when its variance is at least this many
        times the other's. Default 2.0 (i.e. it carries >= 2/3 of the GRR
        variance). Used only to target recommendations.
    interaction_alpha:
        Significance level for reporting the operator x part interaction as
        significant. Default 0.05. (Distinct from the 0.25 *pooling* level,
        which decides only whether the term stays in the model.)
    """

    acceptable_max: float = 10.0
    marginal_max: float = 30.0
    ndc_min: int = 5
    dominance_ratio: float = 2.0
    interaction_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.acceptable_max < self.marginal_max:
            raise ValueError("Require 0 < acceptable_max < marginal_max.")
        if self.ndc_min < 1:
            raise ValueError("ndc_min must be >= 1.")
        if self.dominance_ratio < 1.0:
            raise ValueError("dominance_ratio must be >= 1.")
        if not 0.0 < self.interaction_alpha < 1.0:
            raise ValueError("interaction_alpha must be in (0, 1).")


class GrrVerdict(str, Enum):
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    UNACCEPTABLE = "unacceptable"


_VERDICT_ORDER = {GrrVerdict.ACCEPTABLE: 0, GrrVerdict.MARGINAL: 1, GrrVerdict.UNACCEPTABLE: 2}
_VERDICT_SEVERITY = {
    GrrVerdict.ACCEPTABLE: Severity.INFO,
    GrrVerdict.MARGINAL: Severity.WARNING,
    GrrVerdict.UNACCEPTABLE: Severity.CRITICAL,
}
_VERDICT_TEXT = {
    GrrVerdict.ACCEPTABLE: "acceptable",
    GrrVerdict.MARGINAL: "may be acceptable depending on the application",
    GrrVerdict.UNACCEPTABLE: "unacceptable",
}


def grr_verdict(pct: float, t: GageThresholds = GageThresholds()) -> GrrVerdict:
    """Classify a %GRR value into the AIAG band."""
    if pct < t.acceptable_max:
        return GrrVerdict.ACCEPTABLE
    if pct <= t.marginal_max:
        return GrrVerdict.MARGINAL
    return GrrVerdict.UNACCEPTABLE


def _fmt(x: float) -> str:
    return f"{x:.1f}%"


def interpret_gage(
    result: GageRRResult,
    thresholds: GageThresholds = GageThresholds(),
) -> InterpretationReport:
    t = thresholds
    concl: list[StatisticalConclusion] = []
    recs: list[EngineeringRecommendation] = []
    c = result.components

    # --- zero total variation: nothing else is meaningful -------------------
    if c.total_sd == 0.0:
        concl.append(
            StatisticalConclusion(
                code="msa_zero_variation",
                topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                severity=Severity.CRITICAL,
                text="All measurements are identical, so no variation component can be "
                "estimated and every percentage is undefined.",
            )
        )
        recs.append(
            EngineeringRecommendation(
                code="rec_msa_resolution",
                category=RecommendationCategory.VERIFY_GAUGE,
                severity=Severity.CRITICAL,
                text="Check the gauge resolution against the expected part-to-part "
                "variation, and select parts that span the process range.",
                trigger_code="msa_zero_variation",
            )
        )
        return InterpretationReport(
            conclusions=tuple(concl),
            recommendations=tuple(recs),
            headline="Measurement system: not assessable (no variation in the data).",
        )

    # --- measurement consistency (range chart) first ------------------------
    rc = result.range_check
    caveat = ""
    if not rc.in_control:
        cells = ", ".join(f"part {p} / operator {o} (R={r:.4g})" for p, o, r in rc.out_of_control[:5])
        more = len(rc.out_of_control) - 5
        if more > 0:
            cells += f" and {more} more"
        concl.append(
            StatisticalConclusion(
                code="msa_ranges_out_of_control",
                topic=ConclusionTopic.MEASUREMENT_CONSISTENCY,
                severity=Severity.WARNING,
                text=f"{len(rc.out_of_control)} part x operator range(s) exceed the range-chart "
                f"UCL ({rc.ucl:.4g}): {cells}. Repeatability is not consistent, so the "
                "figures below are not yet reliable.",
                evidence={"ucl": rc.ucl, "r_bar": rc.r_bar, "n_out": len(rc.out_of_control)},
            )
        )
        recs.append(
            EngineeringRecommendation(
                code="rec_msa_investigate_ranges",
                category=RecommendationCategory.VERIFY_GAUGE,
                severity=Severity.WARNING,
                text="Investigate the flagged readings (recording error, part handling, "
                "measurement location) and re-measure those cells before relying on the study.",
                trigger_code="msa_ranges_out_of_control",
            )
        )
        caveat = " (caveat: inconsistent repeat readings)"

    # --- %GRR verdicts --------------------------------------------------------
    verdicts: list[tuple[GrrVerdict, str]] = []
    pct_sv = result.pct_grr_study_var
    assert pct_sv is not None  # total_sd > 0 here
    v_sv = grr_verdict(pct_sv, t)
    verdicts.append((v_sv, f"{_fmt(pct_sv)} of study variation"))
    concl.append(
        StatisticalConclusion(
            code=f"msa_grr_{v_sv.value}",
            topic=ConclusionTopic.MEASUREMENT_SYSTEM,
            severity=_VERDICT_SEVERITY[v_sv],
            text=f"Gage R&R is {_fmt(pct_sv)} of total (study) variation: "
            f"{_VERDICT_TEXT[v_sv]} by the AIAG bands (<{t.acceptable_max:g}% / "
            f"{t.acceptable_max:g}-{t.marginal_max:g}% / >{t.marginal_max:g}%).",
            evidence={"pct_study_var": pct_sv},
        )
    )
    pct_tol = result.pct_grr_tolerance
    if pct_tol is not None:
        v_tol = grr_verdict(pct_tol, t)
        verdicts.append((v_tol, f"{_fmt(pct_tol)} of tolerance"))
        concl.append(
            StatisticalConclusion(
                code=f"msa_grr_tolerance_{v_tol.value}",
                topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                severity=_VERDICT_SEVERITY[v_tol],
                text=f"Gage R&R study variation ({result.k:g} sigma) is {_fmt(pct_tol)} of the "
                f"tolerance: {_VERDICT_TEXT[v_tol]} for sorting parts against the specification.",
                evidence={"pct_tolerance": pct_tol, "k": result.k, "tolerance": result.tolerance},
            )
        )
    worst, worst_basis = max(verdicts, key=lambda vb: _VERDICT_ORDER[vb[0]])

    # --- discrimination -----------------------------------------------------
    ndc = result.ndc
    ndc_ok = ndc is None or ndc >= t.ndc_min
    if ndc is None:
        concl.append(
            StatisticalConclusion(
                code="msa_grr_zero",
                topic=ConclusionTopic.DISCRIMINATION,
                severity=Severity.NOTICE,
                text="No measurement variation was detected (every repeat and every operator "
                "agreed exactly). This usually means the gauge resolution is too coarse to "
                "show its own error, not that the error is zero.",
            )
        )
        recs.append(
            EngineeringRecommendation(
                code="rec_msa_resolution",
                category=RecommendationCategory.VERIFY_GAUGE,
                severity=Severity.NOTICE,
                text="Confirm the gauge resolves at least one tenth of the tolerance or "
                "process variation.",
                trigger_code="msa_grr_zero",
            )
        )
    elif ndc_ok:
        concl.append(
            StatisticalConclusion(
                code="msa_ndc_adequate",
                topic=ConclusionTopic.DISCRIMINATION,
                severity=Severity.INFO,
                text=f"Number of distinct categories ndc = {ndc} (>= {t.ndc_min}): the gauge "
                "separates the parts adequately.",
                evidence={"ndc": ndc},
            )
        )
    else:
        concl.append(
            StatisticalConclusion(
                code="msa_ndc_low",
                topic=ConclusionTopic.DISCRIMINATION,
                severity=Severity.CRITICAL if ndc < 2 else Severity.WARNING,
                text=f"Number of distinct categories ndc = {ndc} (< {t.ndc_min}): the gauge "
                "cannot reliably distinguish between the parts in the study.",
                evidence={"ndc": ndc},
            )
        )
        recs.append(
            EngineeringRecommendation(
                code="rec_msa_discrimination",
                category=RecommendationCategory.VERIFY_GAUGE,
                severity=Severity.WARNING,
                text="Improve gauge discrimination, and check that the study parts span the "
                "full process range (parts that are too alike also give a low ndc).",
                trigger_code="msa_ndc_low",
            )
        )

    # --- operator x part interaction (ANOVA only) ----------------------------
    if result.method is GageMethod.ANOVA and isinstance(result.anova, AnovaResult):
        a = result.anova
        p_int = a.interaction_p
        if p_int is not None and p_int <= t.interaction_alpha:
            concl.append(
                StatisticalConclusion(
                    code="msa_interaction_significant",
                    topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                    severity=Severity.WARNING,
                    text=f"Operator x part interaction is significant (p = {p_int:.3g}): "
                    "operators do not differ consistently - some measure some parts differently.",
                    evidence={"p": p_int},
                )
            )
            recs.append(
                EngineeringRecommendation(
                    code="rec_msa_interaction",
                    category=RecommendationCategory.IMPROVE_REPRODUCIBILITY,
                    severity=Severity.WARNING,
                    text="Look for part features that make the method ambiguous (datum, "
                    "measurement location, form error) and for operator-specific technique.",
                    trigger_code="msa_interaction_significant",
                )
            )
        elif a.interaction_pooled:
            concl.append(
                StatisticalConclusion(
                    code="msa_interaction_pooled",
                    topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                    severity=Severity.INFO,
                    text="Operator x part interaction is not significant"
                    + (f" (p = {p_int:.3g} > {a.alpha_pool:g})" if p_int is not None else "")
                    + "; it was pooled into repeatability.",
                    evidence={"p": p_int, "alpha_pool": a.alpha_pool},
                )
            )
        else:
            concl.append(
                StatisticalConclusion(
                    code="msa_interaction_retained",
                    topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                    severity=Severity.NOTICE,
                    text=f"Operator x part interaction is kept in the model (p = {p_int:.3g} <= "
                    f"{a.alpha_pool:g}) but is not significant at {t.interaction_alpha:g}.",
                    evidence={"p": p_int, "alpha_pool": a.alpha_pool},
                )
            )

    if c.clamped:
        concl.append(
            StatisticalConclusion(
                code="msa_component_clamped",
                topic=ConclusionTopic.MEASUREMENT_SYSTEM,
                severity=Severity.NOTICE,
                text="Negative variance estimate(s) set to zero: "
                + ", ".join(c.clamped)
                + ". The true component is small relative to sampling noise.",
                evidence={"clamped": list(c.clamped)},
            )
        )

    # --- targeted recommendations -------------------------------------------
    ev2, av2 = c.repeatability_sd**2, c.reproducibility_sd**2
    if worst is not GrrVerdict.ACCEPTABLE:
        sev = _VERDICT_SEVERITY[worst]
        if ev2 >= t.dominance_ratio * av2:
            recs.append(
                EngineeringRecommendation(
                    code="rec_msa_repeatability",
                    category=RecommendationCategory.IMPROVE_REPEATABILITY,
                    severity=sev,
                    text="Repeatability (equipment variation) dominates: check gauge condition "
                    "and maintenance, fixturing and part location, and within-part form variation.",
                    trigger_code=f"msa_grr_{worst.value}",
                )
            )
        elif av2 >= t.dominance_ratio * ev2:
            recs.append(
                EngineeringRecommendation(
                    code="rec_msa_reproducibility",
                    category=RecommendationCategory.IMPROVE_REPRODUCIBILITY,
                    severity=sev,
                    text="Reproducibility (appraiser variation) dominates: standardise the "
                    "measurement method, train the appraisers, and check fixtures or readability "
                    "that let technique differ.",
                    trigger_code=f"msa_grr_{worst.value}",
                )
            )
        else:
            recs.append(
                EngineeringRecommendation(
                    code="rec_msa_both",
                    category=RecommendationCategory.IMPROVE_REPEATABILITY,
                    severity=sev,
                    text="Repeatability and reproducibility contribute comparably: address the "
                    "gauge/fixture and the measurement method together.",
                    trigger_code=f"msa_grr_{worst.value}",
                )
            )
    elif ndc_ok and rc.in_control:
        recs.append(
            EngineeringRecommendation(
                code="rec_msa_proceed",
                category=RecommendationCategory.PROCEED_TO_CAPABILITY,
                severity=Severity.INFO,
                text="The measurement system is fit for this characteristic: proceed with the "
                "capability study using this gauge and method.",
                trigger_code="msa_grr_acceptable",
            )
        )

    ndc_txt = f", ndc = {ndc}" if ndc is not None else ""
    headline = (
        f"Measurement system {worst.value.upper()}: %GRR = {worst_basis}{ndc_txt}{caveat}."
    )
    return InterpretationReport(
        conclusions=tuple(concl),
        recommendations=tuple(recs),
        headline=headline,
    )


def headline_verdict(result: GageRRResult, thresholds: GageThresholds = GageThresholds()) -> Optional[GrrVerdict]:
    """Worst %GRR verdict across study variation and tolerance (``None`` if undefined)."""
    vals = [v for v in (result.pct_grr_study_var, result.pct_grr_tolerance) if v is not None]
    if not vals:
        return None
    return max((grr_verdict(v, thresholds) for v in vals), key=lambda v: _VERDICT_ORDER[v])
