"""Fixed AnalysisResult (CapabilityReport) fixtures for interpretation tests.

Each builder returns a CapabilityReport in a specific, controlled state so tests
can assert exact conclusion/recommendation CODES (not prose). Sigmas and indices
are chosen so the resulting Cp/Cpk/Pp/Ppk land in known bands.
"""

from __future__ import annotations

from app.capability.guards import NormalityResult
from app.capability.report import StabilitySummary, build_capability_report
from app.data.models import SpecLimits

# Reusable two-sided spec centered at 10 with width 12 (USL16/LSL4).
SPEC = SpecLimits(usl=16.0, lsl=4.0, unit="mm")


def stable_capable():
    """Stable, well-centered, Cpk ~ 1.33 (capable), Ppk close to Cpk."""
    # sigma_within: Cpk = (16-10)/(3*sigma) = 1.33 -> sigma = 6/(3*1.33)=1.5038
    # Use sigma_within = 1.5 -> Cp = 12/(6*1.5)=1.333, Cpk = 6/(3*1.5)=1.333.
    return build_capability_report(
        mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
        normality=NormalityResult(normal=True, shapiro_stat=0.99, shapiro_p=0.6,
                                  anderson_stat=0.3, anderson_crit_5pct=0.75, n=60),
    )


def stable_incapable():
    """Stable but Cp<1 and Cpk<1 (excessive variation / too-tight tolerance)."""
    # sigma_within=2.5 -> Cp=12/(15)=0.8, Cpk=6/(7.5)=0.8 (both < 1).
    return build_capability_report(
        mean=10.0, sigma_within=2.5, sigma_overall=2.6, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
    )


def high_cp_low_cpk():
    """Stable, tight spread but off-center: Cp high, Cpk much lower."""
    # sigma_within=1.0, mean shifted to 13 (USL16/LSL4):
    # Cp=12/6=2.0 ; Cpu=(16-13)/3=1.0 ; Cpl=(13-4)/3=3.0 ; Cpk=1.0.
    return build_capability_report(
        mean=13.0, sigma_within=1.0, sigma_overall=1.05, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
    )


def low_cp_low_cpk():
    """Stable, centered, but excessive variation: Cp and Cpk both low."""
    return stable_incapable()


def high_cpk_unstable():
    """Excellent Cpk but the process is UNSTABLE (stability must dominate)."""
    # sigma_within=0.5 centered -> Cp=4.0, Cpk=4.0 (excellent), but unstable.
    return build_capability_report(
        mean=10.0, sigma_within=0.5, sigma_overall=0.55, spec=SPEC,
        n_observations=60, n_subgroups=24,
        stability=StabilitySummary(is_stable=False, triggered_rule_ids=("N1", "N2"),
                                   n_violations=4),
    )


def not_assessed_stability():
    """Good indices but stability never assessed (stability=None)."""
    return build_capability_report(
        mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=None,
    )


def small_sample():
    """Stable, capable-looking, but N below threshold."""
    return build_capability_report(
        mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=SPEC,
        n_observations=12, n_subgroups=4, stability=StabilitySummary(is_stable=True),
    )


def non_normal():
    """Stable but normality rejected."""
    return build_capability_report(
        mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
        normality=NormalityResult(normal=False, shapiro_stat=0.90, shapiro_p=0.001,
                                  anderson_stat=1.8, anderson_crit_5pct=0.75, n=60),
    )


def one_sided_upper():
    """One-sided upper spec (USL only) -> no Cp/Pp."""
    return build_capability_report(
        mean=10.0, sigma_within=1.0, sigma_overall=1.05,
        spec=SpecLimits(usl=16.0, unit="mm"),
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
    )


def zero_variation():
    """Zero variation -> indices +inf, verify-gauge recommendation."""
    return build_capability_report(
        mean=10.0, sigma_within=0.0, sigma_overall=0.0, spec=SPEC,
        n_observations=40, n_subgroups=20, stability=StabilitySummary(is_stable=True),
    )


def drift_ppk_worse():
    """Cpk good but Ppk substantially worse -> long-term drift."""
    # sigma_within=1.0 centered -> Cpk=(6)/3=2.0
    # sigma_overall=3.0 -> Ppk=(6)/9=0.667 ; drift gap huge.
    return build_capability_report(
        mean=10.0, sigma_within=1.0, sigma_overall=3.0, spec=SPEC,
        n_observations=60, n_subgroups=24, stability=StabilitySummary(is_stable=True),
    )
