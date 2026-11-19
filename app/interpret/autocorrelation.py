"""Conclusion and recommendation for autocorrelated individuals data.

Positive autocorrelation (tool wear, thermal drift, slow material changes)
makes consecutive readings alike, so the average moving range understates the
process spread: the I-MR limits come out too narrow and the chart signals
without a special cause. Negative autocorrelation (over-adjustment,
alternating fixtures or streams) inflates the moving range and widens the
limits instead. Numbers are always reported; this only qualifies them.

``PRACTICAL_R1`` is an engineering rule, not a statistical constant: a
significant coefficient below it is noted, at or above it is a warning.
"""

from __future__ import annotations

from typing import Optional

from app.interpret.models import (
    ConclusionTopic,
    EngineeringRecommendation,
    RecommendationCategory,
    Severity,
    StatisticalConclusion,
)
from app.statistics.autocorrelation import AutocorrelationResult

__all__ = ["PRACTICAL_R1", "autocorrelation_findings"]

PRACTICAL_R1: float = 0.3


def autocorrelation_findings(
    ac: Optional[AutocorrelationResult], practical_r1: float = PRACTICAL_R1
) -> tuple[Optional[StatisticalConclusion], Optional[EngineeringRecommendation]]:
    """``(None, None)`` unless the lag-1 autocorrelation is significant."""
    if ac is None or not ac.significant or ac.r1 is None or ac.bound is None:
        return None, None
    positive = ac.r1 > 0
    severity = Severity.WARNING if abs(ac.r1) >= practical_r1 else Severity.NOTICE
    effect = (
        "Consecutive readings are alike, so the moving range understates the process "
        "spread: the I-MR limits are narrower than they should be and out-of-control "
        "signals may be false alarms. Cp/Cpk, which use the same within sigma, are "
        "overstated."
        if positive else
        "Consecutive readings alternate, so the moving range overstates the process "
        "spread: the I-MR limits are wider than they should be and real signals may "
        "be missed. Cp/Cpk are understated."
    )
    conclusion = StatisticalConclusion(
        code="autocorrelated_observations",
        topic=ConclusionTopic.INDEPENDENCE,
        severity=severity,
        text=(
            f"The individual readings are autocorrelated: lag-1 r1 = {ac.r1:.3f}, outside "
            f"+/-{ac.bound:.3f} (approximate {100 * (1 - ac.alpha):.0f}% bound for an independent "
            f"series of {ac.n}). {effect}"
        ),
        evidence={"r1": ac.r1, "bound": ac.bound, "p_value": ac.p_value, "n": ac.n},
    )
    recommendation = EngineeringRecommendation(
        code="rec_address_autocorrelation",
        category=RecommendationCategory.ADDRESS_AUTOCORRELATION,
        severity=severity,
        text=(
            "Find what links consecutive parts (tool wear, temperature, material lot, "
            "sampling faster than the process changes). Sampling less often, or charting "
            "subgroup averages over a longer interval, restores independence; otherwise "
            "treat the I-MR signals as provisional."
            if positive else
            "Look for over-adjustment after each reading, or for two alternating streams "
            "(fixtures, cavities, spindles) merged into one series; chart each stream on "
            "its own."
        ),
        trigger_code="autocorrelated_observations",
    )
    return conclusion, recommendation
