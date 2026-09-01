"""Deterministic, rule-based interpretation engine.

Consumes a :class:`~app.capability.report.CapabilityReport` (the analysis result
carrying capability/performance indices, sigma levels, stability, normality, CI,
and reliability warnings) and produces an :class:`InterpretationReport` of
strictly separated statistical conclusions and engineering recommendations.

Design principles
-----------------
* **Deterministic & rule-based.** No LLM, no randomness. Same input -> same
  output, always. Every branch is unit-testable.
* **Facts vs advice are separated by type.** Conclusions describe the numbers;
  recommendations propose actions and cite their trigger. They live in separate
  lists.
* **No invented conclusions.** A rule only fires when the data it needs is
  present (e.g. Cp-vs-Cpk centering logic requires a two-sided spec so Cp exists).
* **Stability has priority.** If the process is unstable, a CRITICAL conclusion
  states capability must not be read as reliable evidence of future performance —
  but the numeric indices are retained, never nulled, and the process is never
  called "capable" on Cpk alone.
* **Thresholds are documented and configurable** via
  :class:`~app.interpret.thresholds.InterpretationThresholds`.

M5 introduces NO new statistics. It only reads validated M1-M3 quantities.
"""

from __future__ import annotations

from math import isfinite, isinf
from typing import Optional

from app.capability.report import CapabilityReport
from app.interpret.models import (
    ConclusionTopic,
    EngineeringRecommendation,
    InterpretationReport,
    RecommendationCategory,
    Severity,
    StatisticalConclusion,
)
from app.interpret.thresholds import CapabilityCategory, InterpretationThresholds

__all__ = ["interpret", "interpret_capability_report"]


def _fmt(x: Optional[float]) -> str:
    """Format a number for conclusion text without losing meaning."""
    if x is None:
        return "n/a"
    if isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.3g}"


def interpret_capability_report(
    report: CapabilityReport,
    *,
    thresholds: InterpretationThresholds = InterpretationThresholds(),
) -> InterpretationReport:
    """Interpret a :class:`CapabilityReport` into structured conclusions/advice.

    The rules and their thresholds are documented in ``docs/FORMULAS.md`` §M5.
    """
    conclusions: list[StatisticalConclusion] = []
    recommendations: list[EngineeringRecommendation] = []

    t = thresholds
    cap = report.capability
    perf = report.performance
    spec = report.spec
    two_sided = spec.is_two_sided

    # Track high-level state for the headline and for stability-priority logic.
    unstable = report.stability is not None and not report.stability.is_stable
    stability_assessed = report.stability is not None

    # -- 1. STABILITY (priority) ------------------------------------------------
    if not stability_assessed:
        conclusions.append(
            StatisticalConclusion(
                code="stability_not_assessed",
                topic=ConclusionTopic.STABILITY,
                severity=Severity.NOTICE,
                text=(
                    "Process stability has not been assessed (no SPC control-rule "
                    "evaluation was supplied). Capability indices assume a stable "
                    "process, so their predictive validity is unconfirmed."
                ),
            )
        )
    elif unstable:
        rule_ids = ", ".join(report.stability.triggered_rule_ids) or "one or more rules"
        conclusions.append(
            StatisticalConclusion(
                code="process_unstable",
                topic=ConclusionTopic.STABILITY,
                severity=Severity.CRITICAL,
                text=(
                    f"The process is statistically UNSTABLE: SPC control rules "
                    f"({rule_ids}) detected out-of-control signals. Capability and "
                    "performance indices must NOT be interpreted as reliable "
                    "evidence of future process performance while the process is "
                    "out of control. The numeric indices are retained below but "
                    "are not trustworthy until the process is stabilized."
                ),
                evidence={
                    "triggered_rule_ids": list(report.stability.triggered_rule_ids),
                    "n_violations": report.stability.n_violations,
                },
            )
        )
        recommendations.append(
            EngineeringRecommendation(
                code="rec_investigate_stability",
                category=RecommendationCategory.INVESTIGATE_STABILITY,
                severity=Severity.CRITICAL,
                text=(
                    "Investigate and eliminate the assignable (special) causes "
                    "behind the SPC signals before optimizing or quoting "
                    "capability. Stabilize the process first, then re-assess "
                    "capability on in-control data."
                ),
                trigger_code="process_unstable",
            )
        )
    else:
        conclusions.append(
            StatisticalConclusion(
                code="process_stable",
                topic=ConclusionTopic.STABILITY,
                severity=Severity.INFO,
                text=(
                    "The process shows no SPC control-rule violations on the "
                    "evaluated chart(s); it is statistically stable, so capability "
                    "indices can be interpreted as evidence of process behaviour."
                ),
            )
        )

    # -- 2. ZERO VARIATION (guard before index banding) -------------------------
    zero_within = report.sigma_within is not None and report.sigma_within == 0.0
    zero_overall = report.sigma_overall == 0.0
    if zero_within or zero_overall:
        conclusions.append(
            StatisticalConclusion(
                code="zero_variation",
                topic=ConclusionTopic.ZERO_VARIATION,
                severity=Severity.WARNING,
                text=(
                    "Observed variation is zero, so capability indices are "
                    "mathematically +inf. Zero spread almost always indicates "
                    "insufficient measurement resolution or a data-entry issue "
                    "rather than a genuinely variation-free process. Note that "
                    "a flat series of 15 or more points also triggers control "
                    "rule N7 (15 consecutive points within 1 sigma): on "
                    "zero-variation data that signal means stratification / "
                    "inadequate gauge resolution, NOT an ordinary process shift "
                    "requiring a hunt for an assignable cause in the process "
                    "itself."
                ),
                evidence={
                    "sigma_within": report.sigma_within,
                    "sigma_overall": report.sigma_overall,
                },
            )
        )
        recommendations.append(
            EngineeringRecommendation(
                code="rec_verify_gauge",
                category=RecommendationCategory.VERIFY_GAUGE,
                severity=Severity.WARNING,
                text=(
                    "Verify the gauge resolution and the raw measurement records: "
                    "confirm the readings are not rounded or duplicated before "
                    "trusting any capability index computed from zero variation."
                ),
                trigger_code="zero_variation",
            )
        )

    # -- 3. ONE-SIDED SPEC note -------------------------------------------------
    if not two_sided:
        which = "upper (USL only)" if spec.usl is not None else "lower (LSL only)"
        conclusions.append(
            StatisticalConclusion(
                code="one_sided_spec",
                topic=ConclusionTopic.ONE_SIDED_SPEC,
                severity=Severity.INFO,
                text=(
                    f"The characteristic has a one-sided {which} specification. "
                    "Cp and Pp (which require both limits) are undefined and are "
                    "not reported; capability is expressed by the single-sided "
                    "index (Cpk = Cpu or Cpl as applicable)."
                ),
            )
        )

    # -- 4. CAPABILITY (Cpk, within) band --------------------------------------
    cpk = cap.cpk if cap else None
    cpk_cat = t.categorize(cpk)
    if cap is not None and cpk is not None:
        _emit_index_band(
            conclusions, topic=ConclusionTopic.CAPABILITY, label="Cpk", value=cpk,
            category=cpk_cat, thresholds=t, zero_variation=zero_within,
        )
    elif cap is None:
        # No within-subgroup sigma was supplied (e.g. unordered data with no
        # rational subgrouping). State this explicitly rather than silently
        # omitting the capability family.
        conclusions.append(
            StatisticalConclusion(
                code="within_capability_not_assessed",
                topic=ConclusionTopic.CAPABILITY,
                severity=Severity.NOTICE,
                text=(
                    "Within-subgroup capability (Cp/Cpk) was NOT assessed: no "
                    "valid within-subgroup variation estimate was available, "
                    "which is the case when the measurements are not in "
                    "production/time order and rational subgrouping cannot be "
                    "assumed. Only overall performance (Pp/Ppk) is reported "
                    "below; no statement about short-term process capability "
                    "can be made from these data."
                ),
            )
        )
        recommendations.append(
            EngineeringRecommendation(
                code="rec_collect_ordered_data",
                category=RecommendationCategory.COLLECT_MORE_DATA,
                severity=Severity.NOTICE,
                text=(
                    "To obtain Cp/Cpk and a stability verdict, collect (or "
                    "confirm) measurements in production/time order using "
                    "rational subgroups, so that within-subgroup variation "
                    "reflects common cause only."
                ),
                trigger_code="within_capability_not_assessed",
            )
        )

    # -- 5. PERFORMANCE (Ppk, overall) band ------------------------------------
    ppk = perf.cpk
    ppk_cat = t.categorize(ppk)
    if ppk is not None:
        _emit_index_band(
            conclusions, topic=ConclusionTopic.PERFORMANCE, label="Ppk", value=ppk,
            category=ppk_cat, thresholds=t, zero_variation=zero_overall,
        )

    # -- 6. CENTERING: Cp present and Cpk substantially below it ----------------
    if cap is not None and two_sided and cap.cp is not None and cpk is not None:
        cp = cap.cp
        if isfinite(cp) and isfinite(cpk) and cp > 0:
            gap = (cp - cpk) / cp
            if gap >= t.centering_gap_ratio:
                conclusions.append(
                    StatisticalConclusion(
                        code="off_center",
                        topic=ConclusionTopic.CENTERING,
                        severity=Severity.WARNING,
                        text=(
                            f"Cpk ({_fmt(cpk)}) is materially below Cp ({_fmt(cp)}) "
                            f"— a {gap * 100:.0f}% shortfall — which indicates the "
                            "process mean is off-centre within the specification, "
                            "not that the process is inherently too variable."
                        ),
                        evidence={"cp": cp, "cpk": cpk, "gap_ratio": gap},
                    )
                )
                recommendations.append(
                    EngineeringRecommendation(
                        code="rec_investigate_centering",
                        category=RecommendationCategory.INVESTIGATE_CENTERING,
                        severity=Severity.WARNING,
                        text=(
                            "The potential capability (Cp) is adequate but is not "
                            "being realized because the process is off-centre. "
                            "Investigate the process aim and consider shifting the "
                            "process mean toward the specification nominal to "
                            "recover capability."
                        ),
                        trigger_code="off_center",
                    )
                )
            else:
                conclusions.append(
                    StatisticalConclusion(
                        code="well_centered",
                        topic=ConclusionTopic.CENTERING,
                        severity=Severity.INFO,
                        text=(
                            f"Cpk ({_fmt(cpk)}) is close to Cp ({_fmt(cp)}); the "
                            "process is reasonably centred within the specification."
                        ),
                        evidence={"cp": cp, "cpk": cpk, "gap_ratio": gap},
                    )
                )

    # -- 7. EXCESSIVE VARIATION: low Cp AND low Cpk -----------------------------
    if cap is not None and two_sided and cap.cp is not None and cpk is not None:
        cp = cap.cp
        if (
            isfinite(cp) and isfinite(cpk)
            and cp < t.marginal_min and cpk < t.marginal_min
        ):
            conclusions.append(
                StatisticalConclusion(
                    code="excessive_variation",
                    topic=ConclusionTopic.SPEC_MARGIN,
                    severity=Severity.WARNING,
                    text=(
                        f"Both Cp ({_fmt(cp)}) and Cpk ({_fmt(cpk)}) are below "
                        f"{t.marginal_min:g}. Even perfectly centred, the process "
                        "spread is too wide for the specification: the limiting "
                        "factor is process variation (or a tight tolerance), not "
                        "centring."
                    ),
                    evidence={"cp": cp, "cpk": cpk},
                )
            )
            recommendations.append(
                EngineeringRecommendation(
                    code="rec_reduce_variation",
                    category=RecommendationCategory.REDUCE_VARIATION,
                    severity=Severity.WARNING,
                    text=(
                        "Because potential capability itself is inadequate, "
                        "centring adjustments alone cannot make the process "
                        "capable. Reduce common-cause variation (tooling, "
                        "fixturing, material consistency, machine condition) or, "
                        "with the customer, review whether the tolerance is "
                        "achievable for this process."
                    ),
                    trigger_code="excessive_variation",
                )
            )
            recommendations.append(
                EngineeringRecommendation(
                    code="rec_review_specification",
                    category=RecommendationCategory.REVIEW_SPECIFICATION,
                    severity=Severity.NOTICE,
                    text=(
                        "If the process variation reflects the true capability of "
                        "the equipment, evaluate whether the specification width is "
                        "appropriate for the process — an unachievably tight "
                        "tolerance is a design/tolerancing issue, not a process "
                        "defect."
                    ),
                    trigger_code="excessive_variation",
                )
            )

    # -- 8. WITHIN vs OVERALL (drift): Ppk substantially worse than Cpk ---------
    if (
        cap is not None and cpk is not None and ppk is not None
        and isfinite(cpk) and isfinite(ppk) and cpk > 0
    ):
        drift = (cpk - ppk) / cpk
        if drift >= t.drift_gap_ratio:
            conclusions.append(
                StatisticalConclusion(
                    code="long_term_drift",
                    topic=ConclusionTopic.WITHIN_VS_OVERALL,
                    severity=Severity.WARNING,
                    text=(
                        f"Overall performance Ppk ({_fmt(ppk)}) is materially worse "
                        f"than short-term capability Cpk ({_fmt(cpk)}) — a "
                        f"{drift * 100:.0f}% gap. Within-subgroup variation is "
                        "smaller than the total variation, which is the signature "
                        "of between-subgroup drift or shifts over the study period."
                    ),
                    evidence={"cpk": cpk, "ppk": ppk, "drift_ratio": drift},
                )
            )
            recommendations.append(
                EngineeringRecommendation(
                    code="rec_investigate_drift",
                    category=RecommendationCategory.INVESTIGATE_DRIFT,
                    severity=Severity.WARNING,
                    text=(
                        "The short-term process is tighter than its long-term "
                        "behaviour. Investigate sources of between-subgroup "
                        "variation: tool wear and setup drift, batch-to-batch "
                        "material differences, and machine/operator/shift or "
                        "environmental (temperature/humidity) effects over time."
                    ),
                    trigger_code="long_term_drift",
                )
            )
        else:
            conclusions.append(
                StatisticalConclusion(
                    code="within_overall_consistent",
                    topic=ConclusionTopic.WITHIN_VS_OVERALL,
                    severity=Severity.INFO,
                    text=(
                        f"Short-term Cpk ({_fmt(cpk)}) and long-term Ppk "
                        f"({_fmt(ppk)}) are close, indicating consistent behaviour "
                        "with little between-subgroup drift over the study period."
                    ),
                    evidence={"cpk": cpk, "ppk": ppk, "drift_ratio": drift},
                )
            )

    # -- 9. SPEC MARGIN tight relative to spread (two-sided) --------------------
    if (
        two_sided and report.sigma_within is not None
        and report.sigma_within > 0 and spec.tolerance_width is not None
    ):
        band = t.spec_margin_tight_sigma * report.sigma_within
        if spec.tolerance_width < band:
            conclusions.append(
                StatisticalConclusion(
                    code="spec_margin_tight",
                    topic=ConclusionTopic.SPEC_MARGIN,
                    severity=Severity.NOTICE,
                    text=(
                        f"The specification width ({_fmt(spec.tolerance_width)}) is "
                        f"narrower than the process's natural "
                        f"{t.spec_margin_tight_sigma:g}-sigma band "
                        f"({_fmt(band)}); the tolerance is tight relative to the "
                        "current process spread."
                    ),
                    evidence={
                        "tolerance_width": spec.tolerance_width,
                        "natural_band": band,
                        "sigma_within": report.sigma_within,
                    },
                )
            )

    # -- 10. SAMPLE-SIZE reliability -------------------------------------------
    if report.n_observations < t.min_observations:
        conclusions.append(
            StatisticalConclusion(
                code="small_sample",
                topic=ConclusionTopic.SAMPLE_SIZE,
                severity=Severity.WARNING,
                text=(
                    f"Only {report.n_observations} observations were used "
                    f"(< {t.min_observations}). Capability indices computed from "
                    "small samples have wide confidence intervals and can be "
                    "misleading."
                ),
                evidence={"n_observations": report.n_observations},
            )
        )
        recommendations.append(
            EngineeringRecommendation(
                code="rec_collect_more_data",
                category=RecommendationCategory.COLLECT_MORE_DATA,
                severity=Severity.WARNING,
                text=(
                    "Collect additional rational subgroups before drawing firm "
                    "capability conclusions; a larger, representative sample will "
                    "narrow the capability confidence interval."
                ),
                trigger_code="small_sample",
            )
        )
    if report.n_subgroups is not None and report.n_subgroups < t.min_subgroups:
        conclusions.append(
            StatisticalConclusion(
                code="few_subgroups",
                topic=ConclusionTopic.SAMPLE_SIZE,
                severity=Severity.NOTICE,
                text=(
                    f"Only {report.n_subgroups} rational subgroups were used "
                    f"(< {t.min_subgroups}); the within-subgroup sigma estimate and "
                    "the stability picture rest on limited data."
                ),
                evidence={"n_subgroups": report.n_subgroups},
            )
        )

    # -- 11. NORMALITY ----------------------------------------------------------
    if report.normality is not None and report.normality.normal is False:
        conclusions.append(
            StatisticalConclusion(
                code="non_normal",
                topic=ConclusionTopic.NORMALITY,
                severity=Severity.WARNING,
                text=(
                    "The normality test rejects the assumption of a normal "
                    "distribution. Cp/Cpk/Pp/Ppk and their PPM/sigma-level "
                    "translations assume normality, so their probability "
                    "interpretation is unreliable for these data."
                ),
                evidence={
                    "shapiro_p": report.normality.shapiro_p,
                },
            )
        )
        recommendations.append(
            EngineeringRecommendation(
                code="rec_check_distribution",
                category=RecommendationCategory.CHECK_DISTRIBUTION,
                severity=Severity.WARNING,
                text=(
                    "Investigate the distributional behaviour before relying on "
                    "normal-theory capability: check for mixtures, skew, or "
                    "boundedness, and consider a distribution-appropriate "
                    "capability method or a validated transformation."
                ),
                trigger_code="non_normal",
            )
        )

    # -- 12. CONFIDENCE INTERVAL (Cpk) -----------------------------------------
    if report.cpk_ci is not None:
        ci = report.cpk_ci
        conclusions.append(
            StatisticalConclusion(
                code="cpk_confidence_interval",
                topic=ConclusionTopic.CONFIDENCE_INTERVAL,
                severity=Severity.INFO,
                text=(
                    f"The approximate {ci.confidence * 100:.0f}% confidence "
                    f"interval for Cpk is [{_fmt(ci.lower)}, {_fmt(ci.upper)}]. "
                    "This is a large-sample normal approximation, not an exact "
                    "interval, and its assumptions (normality, stability) must "
                    "hold for it to be valid."
                ),
                evidence={
                    "lower": ci.lower, "upper": ci.upper,
                    "confidence": ci.confidence, "approximate": ci.approximate,
                },
            )
        )

    # -- 13. RULE VIOLATIONS detail (factual) -----------------------------------
    if unstable and report.stability is not None and report.stability.triggered_rule_ids:
        conclusions.append(
            StatisticalConclusion(
                code="rule_violations_detail",
                topic=ConclusionTopic.RULE_VIOLATIONS,
                severity=Severity.WARNING,
                text=(
                    "SPC control-rule violations were detected: "
                    f"{', '.join(report.stability.triggered_rule_ids)} "
                    f"({report.stability.n_violations} violation(s) total)."
                ),
                evidence={
                    "triggered_rule_ids": list(report.stability.triggered_rule_ids),
                    "n_violations": report.stability.n_violations,
                },
            )
        )

    # -- Maintain recommendation when everything looks good ---------------------
    # Gated on positive evidence across every axis (see _maybe_add_maintain):
    # zero variation, rejected normality and small samples all withhold the
    # advice rather than letting a flattering band speak louder than the data.
    _maybe_add_maintain(
        conclusions, recommendations, cap, cpk_cat, unstable, stability_assessed,
        thresholds=t,
        zero_variation=(zero_within or zero_overall),
        normality_rejected=(
            report.normality is not None and report.normality.normal is False
        ),
        insufficient_sample=(report.n_observations < t.min_observations),
    )

    headline = _build_headline(
        unstable=unstable, stability_assessed=stability_assessed,
        cpk=cpk, cpk_cat=cpk_cat, ppk=ppk, ppk_cat=ppk_cat,
        zero_variation=(zero_within or zero_overall),
    )

    return InterpretationReport(
        conclusions=tuple(conclusions),
        recommendations=tuple(recommendations),
        headline=headline,
    )


# Public alias: the CapabilityReport IS the analysis result M5 consumes.
interpret = interpret_capability_report


def _emit_index_band(
    conclusions: list[StatisticalConclusion],
    *,
    topic: ConclusionTopic,
    label: str,
    value: float,
    category: CapabilityCategory,
    thresholds: InterpretationThresholds,
    zero_variation: bool,
) -> None:
    """Emit a factual band conclusion for a capability/performance index."""
    if zero_variation or isinf(value):
        # Zero-variation already produced its own dedicated conclusion.
        return
    sev = {
        CapabilityCategory.INADEQUATE: Severity.WARNING,
        CapabilityCategory.MARGINAL: Severity.NOTICE,
        CapabilityCategory.CAPABLE: Severity.INFO,
        CapabilityCategory.EXCELLENT: Severity.INFO,
        CapabilityCategory.UNDEFINED: Severity.INFO,
    }[category]
    band_text = {
        CapabilityCategory.INADEQUATE: (
            f"below {thresholds.marginal_min:g}: the process does not meet the "
            "specification with adequate margin"
        ),
        CapabilityCategory.MARGINAL: (
            f"in the marginal band [{thresholds.marginal_min:g}, "
            f"{thresholds.capable_min:g}): barely adequate, with little safety "
            "margin"
        ),
        CapabilityCategory.CAPABLE: (
            f"in the capable band [{thresholds.capable_min:g}, "
            f"{thresholds.excellent_min:g}): the process meets the specification "
            "with reasonable margin"
        ),
        CapabilityCategory.EXCELLENT: (
            f"at or above {thresholds.excellent_min:g}: the process meets the "
            "specification with substantial margin"
        ),
    }[category]
    conclusions.append(
        StatisticalConclusion(
            code=f"{topic.value}_{category.value}",
            topic=topic,
            severity=sev,
            text=(
                f"{label} = {value:.3g} is {band_text}. (Category bands are an "
                "engineering interpretation convention, not a statistical law.)"
            ),
            evidence={"index": label, "value": value, "category": category.value},
        )
    )


def _maybe_add_maintain(
    conclusions: list[StatisticalConclusion],
    recommendations: list[EngineeringRecommendation],
    cap,
    cpk_cat: CapabilityCategory,
    unstable: bool,
    stability_assessed: bool,
    *,
    thresholds: InterpretationThresholds,
    zero_variation: bool = False,
    normality_rejected: bool = False,
    insufficient_sample: bool = False,
) -> None:
    """Add a MAINTAIN recommendation only when the evidence genuinely supports it.

    "Maintain current controls" is the strongest reassurance this tool emits, so
    it requires positive evidence on every axis, not merely a good-looking band:

    1. stability was **assessed** and the process is **stable**;
    2. a capability report exists with a **finite** Cpk in the CAPABLE or
       EXCELLENT band -- ``+inf`` from zero variation is NOT capability
       evidence, it is a measurement-resolution artifact (see below);
    3. the normality assumption was **not rejected** -- the Cpk-to-band mapping
       is normal-theory, so a rejected normality test undermines the very
       number the recommendation would rest on;
    4. the sample-size evidence is **sufficient** -- a flattering Cpk from a
       handful of observations has wide sampling variability.

    Conditions 2-4 were added during the hardening milestone. Previously a
    zero-variation dataset produced ``Cpk = +inf`` -> EXCELLENT -> "stable and
    capable, maintain current controls" **alongside** the contradictory
    "verify the gauge" warning. The recommendation must never be stronger than
    the evidence; when any gate fails we stay silent rather than advising, and
    the corresponding warning/conclusion carries the message instead.

    This gating changes only the ENGINEERING RECOMMENDATION. Statistical
    conclusions (including the index-band conclusions) are emitted elsewhere and
    are unaffected, preserving the strict conclusion/recommendation separation.
    """
    if not stability_assessed or unstable or cap is None:
        return
    if zero_variation:
        return  # +inf is an artifact, never capability evidence
    if normality_rejected or insufficient_sample:
        return  # advice would exceed the evidence
    cpk = getattr(cap, "cpk", None)
    if cpk is None or not isfinite(cpk):
        return
    if cpk_cat in (CapabilityCategory.CAPABLE, CapabilityCategory.EXCELLENT):
        recommendations.append(
            EngineeringRecommendation(
                code="rec_maintain",
                category=RecommendationCategory.MAINTAIN,
                severity=Severity.INFO,
                text=(
                    "The process is stable and capable. Maintain current controls "
                    "and continue SPC monitoring to detect any future shift or "
                    "drift early."
                ),
                trigger_code="process_stable",
            )
        )


def _build_headline(
    *,
    unstable: bool,
    stability_assessed: bool,
    cpk: Optional[float],
    cpk_cat: CapabilityCategory,
    ppk: Optional[float],
    ppk_cat: CapabilityCategory,
    zero_variation: bool,
) -> str:
    """Build a single-sentence factual headline (a conclusion, never advice)."""
    if zero_variation:
        return (
            "Zero observed variation — capability indices are undefined (+inf); "
            "verify gauge resolution and data before interpreting."
        )
    if unstable:
        return (
            "Process is statistically UNSTABLE; capability indices are retained "
            "but are not reliable evidence of future performance until the "
            "process is brought into control."
        )
    cpk_txt = "n/a" if cpk is None else f"{cpk:.3g} ({cpk_cat.value})"
    stab_txt = (
        "stable" if stability_assessed else "stability not assessed"
    )
    if cpk is None:
        return (
            f"Process {stab_txt}; within-subgroup capability (Cpk) not assessed "
            "— only overall performance (Ppk) is reported for these data."
        )
    return f"Process {stab_txt}; Cpk = {cpk_txt} on the evaluated data."
