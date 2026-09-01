"""Structured objects for interpretation output.

The engine emits two strictly separated lists:

* :class:`StatisticalConclusion` — factual statements about the numbers.
* :class:`EngineeringRecommendation` — suggested actions tied to a condition.

Both carry a machine-readable ``code`` (the triggering condition), a
``category``/``severity``, and human-readable ``text``. The separation is a type
boundary, not a convention: recommendations live in a different class and a
different list, and the :class:`InterpretationReport` keeps them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Severity",
    "ConclusionTopic",
    "StatisticalConclusion",
    "RecommendationCategory",
    "EngineeringRecommendation",
    "InterpretationReport",
]


class Severity(str, Enum):
    """Severity of a conclusion or recommendation."""

    INFO = "info"           # neutral/positive factual note
    NOTICE = "notice"       # worth attention, not a problem yet
    WARNING = "warning"     # a real concern
    CRITICAL = "critical"   # blocks a "capable" verdict / requires action


class ConclusionTopic(str, Enum):
    """The subject a statistical conclusion addresses."""

    STABILITY = "stability"
    CAPABILITY = "capability"        # Cp/Cpk (within)
    PERFORMANCE = "performance"      # Pp/Ppk (overall)
    CENTERING = "centering"
    WITHIN_VS_OVERALL = "within_vs_overall"
    SPEC_MARGIN = "spec_margin"
    ONE_SIDED_SPEC = "one_sided_spec"
    SAMPLE_SIZE = "sample_size"
    NORMALITY = "normality"
    CONFIDENCE_INTERVAL = "confidence_interval"
    RULE_VIOLATIONS = "rule_violations"
    ZERO_VARIATION = "zero_variation"
    SIGMA_LEVEL = "sigma_level"


class RecommendationCategory(str, Enum):
    """The kind of action a recommendation proposes."""

    INVESTIGATE_STABILITY = "investigate_stability"
    INVESTIGATE_CENTERING = "investigate_centering"
    REDUCE_VARIATION = "reduce_variation"
    INVESTIGATE_DRIFT = "investigate_drift"
    COLLECT_MORE_DATA = "collect_more_data"
    CHECK_DISTRIBUTION = "check_distribution"
    REVIEW_SPECIFICATION = "review_specification"
    VERIFY_GAUGE = "verify_gauge"
    MAINTAIN = "maintain"            # process is good; keep monitoring


@dataclass(frozen=True, slots=True)
class StatisticalConclusion:
    """A factual, defensible statement about the analysis numbers.

    Attributes
    ----------
    code:
        Stable machine-readable identifier of the triggering condition
        (e.g. ``"capability_marginal"``). Tests assert on this, not on prose.
    topic:
        :class:`ConclusionTopic` the statement addresses.
    severity:
        :class:`Severity`.
    text:
        Human-readable factual statement. Never contains an action/advice.
    evidence:
        Optional dict of the numbers that triggered the conclusion, for
        traceability in reports.
    """

    code: str
    topic: ConclusionTopic
    severity: Severity
    text: str
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "statistical_conclusion",
            "code": self.code,
            "topic": self.topic.value,
            "severity": self.severity.value,
            "text": self.text,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class EngineeringRecommendation:
    """A suggested action tied to an identified condition.

    Explicitly NOT a statistical fact. Carries the ``trigger_code`` linking it to
    the condition that produced it.
    """

    code: str
    category: RecommendationCategory
    severity: Severity
    text: str
    trigger_code: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "engineering_recommendation",
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "text": self.text,
            "trigger_code": self.trigger_code,
        }


@dataclass(frozen=True, slots=True)
class InterpretationReport:
    """The interpretation output: conclusions and recommendations, kept apart.

    ``conclusions`` and ``recommendations`` are separate lists by design. The
    ``headline`` is a single short factual summary (a conclusion, never advice).
    """

    conclusions: tuple[StatisticalConclusion, ...]
    recommendations: tuple[EngineeringRecommendation, ...]
    headline: str

    @property
    def conclusion_codes(self) -> tuple[str, ...]:
        return tuple(c.code for c in self.conclusions)

    @property
    def recommendation_codes(self) -> tuple[str, ...]:
        return tuple(r.code for r in self.recommendations)

    def conclusions_for_topic(self, topic: ConclusionTopic) -> list[StatisticalConclusion]:
        return [c for c in self.conclusions if c.topic == topic]

    @property
    def max_severity(self) -> Severity:
        order = {Severity.INFO: 0, Severity.NOTICE: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}
        if not self.conclusions:
            return Severity.INFO
        return max((c.severity for c in self.conclusions), key=lambda s: order[s])

    def as_dict(self) -> dict[str, object]:
        return {
            "headline": self.headline,
            "conclusions": [c.as_dict() for c in self.conclusions],
            "recommendations": [r.as_dict() for r in self.recommendations],
            "max_severity": self.max_severity.value,
        }
