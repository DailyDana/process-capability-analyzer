"""Interpretation of an attribute control chart: facts and recommendations, kept apart.

The thresholds are engineering rules of thumb, adjustable and documented in
``docs/FORMULAS.md`` §9: at least 20 baseline samples (Montgomery suggests
20-25), and an expected count per sample of at least 5 for the normal
approximation behind the 3-sigma limits to be reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.interpret.models import (
    ConclusionTopic,
    EngineeringRecommendation,
    InterpretationReport,
    RecommendationCategory,
    Severity,
    StatisticalConclusion,
)
from app.spc.attribute import AttributeChart, AttributeKind
from app.spc.rules import Direction, RuleEvaluation, Violation

__all__ = ["AttributeThresholds", "interpret_attribute"]

_NOUN = {
    AttributeKind.P: "fraction nonconforming",
    AttributeKind.NP: "number nonconforming",
    AttributeKind.C: "number of nonconformities",
    AttributeKind.U: "nonconformities per unit",
}


@dataclass(frozen=True, slots=True)
class AttributeThresholds:
    min_samples: int = 20
    min_expected_count: float = 5.0

    def __post_init__(self) -> None:
        if self.min_samples < 2 or self.min_expected_count <= 0:
            raise ValueError("Thresholds must be positive (at least 2 samples).")


def _samples(violations: Sequence[Violation]) -> str:
    nums = sorted({i + 1 for v in violations for i in v.indices})
    shown = ", ".join(str(i) for i in nums[:12])
    return shown + (f" and {len(nums) - 12} more" if len(nums) > 12 else "")


def interpret_attribute(
    chart: AttributeChart,
    evaluation: RuleEvaluation,
    thresholds: AttributeThresholds = AttributeThresholds(),
) -> InterpretationReport:
    conclusions: list[StatisticalConclusion] = []
    recs: list[EngineeringRecommendation] = []
    noun = _NOUN[chart.kind]

    high = [v for v in evaluation.violations if v.direction in (Direction.UPPER, Direction.UP)]
    low = [v for v in evaluation.violations if v.direction in (Direction.LOWER, Direction.DOWN)]
    other = [v for v in evaluation.violations if v.direction is Direction.NONE]

    if evaluation.is_stable:
        conclusions.append(StatisticalConclusion(
            "process_stable", ConclusionTopic.STABILITY, Severity.INFO,
            f"No out-of-control signal: the {noun} is consistent with a stable process at "
            f"a centre line of {chart.center_line:.6g}.",
            {"center_line": chart.center_line},
        ))
    else:
        conclusions.append(StatisticalConclusion(
            "process_unstable", ConclusionTopic.STABILITY, Severity.CRITICAL,
            f"{len(evaluation.violations)} out-of-control signal(s) (rules "
            f"{', '.join(evaluation.triggered_rule_ids)}): the {noun} is not stable.",
            {"n_violations": len(evaluation.violations), "rules": list(evaluation.triggered_rule_ids)},
        ))
    if high:
        conclusions.append(StatisticalConclusion(
            "attribute_signal_high", ConclusionTopic.RULE_VIOLATIONS, Severity.CRITICAL,
            f"Signals on the high side at sample(s) {_samples(high)}: the {noun} rose above "
            "what the centre line explains.",
            {"samples": _samples(high)},
        ))
        recs.append(EngineeringRecommendation(
            "rec_investigate_stability", RecommendationCategory.INVESTIGATE_STABILITY, Severity.CRITICAL,
            "Find the assignable cause at the flagged samples (material lot, operator, tool, set-up, "
            "inspection standard) and remove it. Once it is found and removed, leave those samples "
            "out of the baseline and recompute the limits.",
            "attribute_signal_high",
        ))
    if low:
        conclusions.append(StatisticalConclusion(
            "attribute_signal_low", ConclusionTopic.RULE_VIOLATIONS, Severity.WARNING,
            f"Signals on the low side at sample(s) {_samples(low)}: the {noun} fell below "
            "what the centre line explains.",
            {"samples": _samples(low)},
        ))
        recs.append(EngineeringRecommendation(
            "rec_verify_improvement", RecommendationCategory.INVESTIGATE_STABILITY, Severity.WARNING,
            "A drop is good news only if it is real: first rule out missed defects or a changed "
            "inspection standard. If the improvement is real, find what changed, make it permanent "
            "and set new limits from the improved period.",
            "attribute_signal_low",
        ))
    if other:
        conclusions.append(StatisticalConclusion(
            "attribute_signal_pattern", ConclusionTopic.RULE_VIOLATIONS, Severity.WARNING,
            f"A non-random pattern at sample(s) {_samples(other)} (alternation).",
            {"samples": _samples(other)},
        ))
        recs.append(EngineeringRecommendation(
            "rec_check_sampling", RecommendationCategory.INVESTIGATE_STABILITY, Severity.WARNING,
            "Check whether samples alternate between two sources (shifts, lines, inspectors).",
            "attribute_signal_pattern",
        ))

    if not chart.center_is_standard and chart.n_baseline < thresholds.min_samples:
        conclusions.append(StatisticalConclusion(
            "few_subgroups", ConclusionTopic.SAMPLE_SIZE, Severity.NOTICE,
            f"The centre line rests on {chart.n_baseline} samples (fewer than {thresholds.min_samples}); "
            "the limits are provisional.",
            {"n_baseline": chart.n_baseline},
        ))
        recs.append(EngineeringRecommendation(
            "rec_collect_more_data", RecommendationCategory.COLLECT_MORE_DATA, Severity.NOTICE,
            f"Collect at least {thresholds.min_samples} samples before treating the limits as final.",
            "few_subgroups",
        ))

    if chart.kind is AttributeKind.C:
        expected = np.full(chart.n_samples, chart.rate)
    else:
        expected = chart.rate * chart.sizes
    smallest = float(expected.min())
    if 0.0 < smallest < thresholds.min_expected_count:
        conclusions.append(StatisticalConclusion(
            "low_expected_count", ConclusionTopic.SAMPLE_SIZE, Severity.NOTICE,
            f"The expected count per sample is as low as {smallest:.3g} (below "
            f"{thresholds.min_expected_count:g}): the distribution is strongly skewed, so the 3-sigma "
            "limits give more false alarms on the high side than their nominal rate.",
            {"min_expected_count": smallest},
        ))
        recs.append(EngineeringRecommendation(
            "rec_increase_sample_size", RecommendationCategory.COLLECT_MORE_DATA, Severity.NOTICE,
            "Use larger samples (or combine consecutive samples) so that several nonconformities are "
            "expected in each.",
            "low_expected_count",
        ))
    if bool((chart.lcl <= 0).all()):
        conclusions.append(StatisticalConclusion(
            "lcl_at_zero", ConclusionTopic.SAMPLE_SIZE, Severity.INFO,
            "The lower control limit is 0, so a single sample cannot signal an improvement; only a "
            "run below the centre line can.",
            {},
        ))

    if evaluation.is_stable and not recs:
        recs.append(EngineeringRecommendation(
            "rec_maintain", RecommendationCategory.MAINTAIN, Severity.INFO,
            "Keep these limits and judge new samples against them. A stable process is not "
            "necessarily a good one: lowering the centre line needs a change to the process itself.",
            "process_stable",
        ))

    if evaluation.is_stable:
        headline = f"In control: {noun} centred at {chart.center_line:.6g}."
    else:
        headline = f"OUT OF CONTROL: {len(evaluation.violations)} signal(s) on the {chart.kind.value} chart."
    return InterpretationReport(tuple(conclusions), tuple(recs), headline)
