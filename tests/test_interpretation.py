"""Tests for the interpretation engine (app.interpret.engine).

Assert on conclusion/recommendation CODES, topics, severities, and triggering
conditions — not on prose strings. Fixtures are fixed CapabilityReports in
controlled states (tests.fixtures.interpretation_cases).
"""

from __future__ import annotations

import pytest

from app.capability.guards import NormalityResult
from app.capability.report import StabilitySummary, build_capability_report
from app.interpret.engine import interpret
from app.interpret.models import (
    ConclusionTopic,
    RecommendationCategory,
    Severity,
)
from app.interpret.thresholds import CapabilityCategory, InterpretationThresholds
from tests.fixtures import interpretation_cases as cases


def codes(report):
    return set(report.conclusion_codes)


def rec_codes(report):
    return set(report.recommendation_codes)


class TestStableCapable:
    def test_capable_and_stable(self) -> None:
        r = interpret(cases.stable_capable())
        c = codes(r)
        assert "process_stable" in c
        assert "capability_capable" in c  # Cpk ~1.33 -> CAPABLE band
        assert "well_centered" in c
        assert "rec_maintain" in rec_codes(r)
        # No problem-level recommendations.
        assert "rec_reduce_variation" not in rec_codes(r)


class TestStableIncapable:
    def test_low_cp_low_cpk(self) -> None:
        r = interpret(cases.stable_incapable())
        c = codes(r)
        assert "process_stable" in c
        assert "excessive_variation" in c
        assert "capability_inadequate" in c  # Cpk 0.8 < 1
        rc = rec_codes(r)
        assert "rec_reduce_variation" in rc
        assert "rec_review_specification" in rc
        # Not a centering problem -> no centering rec.
        assert "rec_investigate_centering" not in rc


class TestHighCpLowCpk:
    def test_off_center(self) -> None:
        r = interpret(cases.high_cp_low_cpk())
        c = codes(r)
        assert "off_center" in c
        assert "rec_investigate_centering" in rec_codes(r)
        # Cp is high so this is NOT excessive_variation.
        assert "excessive_variation" not in c
        # The off_center conclusion is a centering topic.
        off = [x for x in r.conclusions if x.code == "off_center"][0]
        assert off.topic is ConclusionTopic.CENTERING
        assert off.severity is Severity.WARNING


class TestHighCpkUnstable:
    def test_stability_dominates(self) -> None:
        r = interpret(cases.high_cpk_unstable())
        c = codes(r)
        # Unstable conclusion present and CRITICAL.
        assert "process_unstable" in c
        unstable = [x for x in r.conclusions if x.code == "process_unstable"][0]
        assert unstable.severity is Severity.CRITICAL
        assert unstable.topic is ConclusionTopic.STABILITY
        # Indices RETAINED: capability band conclusion still emitted.
        assert "capability_excellent" in c
        # Never recommends "maintain" / never calls it capable-and-done.
        assert "rec_maintain" not in rec_codes(r)
        # Stability investigation recommended.
        assert "rec_investigate_stability" in rec_codes(r)
        assert r.max_severity is Severity.CRITICAL

    def test_indices_not_nulled(self) -> None:
        rep = cases.high_cpk_unstable()
        # The underlying numeric indices are intact regardless of interpretation.
        assert rep.capability is not None
        assert rep.capability.cpk is not None and rep.capability.cpk > 1.67


class TestNotAssessedStability:
    def test_says_so_explicitly(self) -> None:
        r = interpret(cases.not_assessed_stability())
        c = codes(r)
        assert "stability_not_assessed" in c
        assert "process_stable" not in c
        assert "process_unstable" not in c
        note = [x for x in r.conclusions if x.code == "stability_not_assessed"][0]
        assert note.topic is ConclusionTopic.STABILITY


class TestSmallSample:
    def test_small_sample_warns_and_collects(self) -> None:
        r = interpret(cases.small_sample())
        c = codes(r)
        assert "small_sample" in c
        assert "few_subgroups" in c
        assert "rec_collect_more_data" in rec_codes(r)
        sm = [x for x in r.conclusions if x.code == "small_sample"][0]
        assert sm.topic is ConclusionTopic.SAMPLE_SIZE
        assert sm.severity is Severity.WARNING


class TestNonNormal:
    def test_non_normal_warns_and_checks(self) -> None:
        r = interpret(cases.non_normal())
        c = codes(r)
        assert "non_normal" in c
        assert "rec_check_distribution" in rec_codes(r)
        nn = [x for x in r.conclusions if x.code == "non_normal"][0]
        assert nn.topic is ConclusionTopic.NORMALITY


class TestOneSided:
    def test_one_sided_no_cp_talk(self) -> None:
        r = interpret(cases.one_sided_upper())
        c = codes(r)
        assert "one_sided_spec" in c
        # Must NOT emit centering/excessive-variation conclusions that need Cp.
        assert "off_center" not in c
        assert "well_centered" not in c
        assert "excessive_variation" not in c
        assert "spec_margin_tight" not in c
        # Cpk band still reported (Cpk = Cpu here).
        assert any(code.startswith("capability_") for code in c)


class TestZeroVariation:
    def test_zero_variation_flagged(self) -> None:
        r = interpret(cases.zero_variation())
        c = codes(r)
        assert "zero_variation" in c
        assert "rec_verify_gauge" in rec_codes(r)
        # Band conclusions suppressed for +inf (dedicated zero-variation note).
        assert "capability_excellent" not in c
        assert "performance_excellent" not in c

    def test_headline_mentions_zero_variation(self) -> None:
        r = interpret(cases.zero_variation())
        assert "zero" in r.headline.lower()


class TestMaintainRecommendationEvidenceGating:
    """B3 regression: rec_maintain requires genuine, finite capability evidence.

    Previously sigma=0 gave Cpk=+inf -> EXCELLENT -> "stable and capable,
    maintain current controls", emitted alongside the contradictory
    "zero variation is almost always a gauge artifact" warning. Maintain is the
    strongest reassurance the tool gives and must never outrun the evidence.
    """

    def test_zero_variation_stable_has_verify_gauge_but_no_maintain(self) -> None:
        r = interpret(cases.zero_variation())
        assert "rec_verify_gauge" in rec_codes(r)
        assert "rec_maintain" not in rec_codes(r), (
            "zero variation must never produce a maintain recommendation"
        )

    def test_zero_variation_no_contradictory_pair(self) -> None:
        """The two contradictory strings must not coexist in one report."""
        r = interpret(cases.zero_variation())
        texts = " ".join(x.text for x in r.recommendations).lower()
        assert "gauge" in texts
        assert "maintain current controls" not in texts

    def test_zero_variation_unstable_has_no_maintain(self) -> None:
        rep = build_capability_report(
            mean=10.0, sigma_within=0.0, sigma_overall=0.0, spec=cases.SPEC,
            n_observations=40, n_subgroups=20,
            stability=StabilitySummary(
                is_stable=False, triggered_rule_ids=("N7",), n_violations=1,
            ),
        )
        r = interpret(rep)
        assert "rec_maintain" not in rec_codes(r)

    def test_finite_capable_stable_still_gets_maintain(self) -> None:
        """Regression guard: the normal, well-evidenced path is preserved."""
        r = interpret(cases.stable_capable())
        assert "rec_maintain" in rec_codes(r)

    def test_maintain_withheld_when_normality_rejected(self) -> None:
        rep = build_capability_report(
            mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=cases.SPEC,
            n_observations=60, n_subgroups=24,
            stability=StabilitySummary(is_stable=True),
            normality=NormalityResult(
                normal=False, shapiro_stat=0.90, shapiro_p=0.001,
                anderson_stat=1.5, anderson_crit_5pct=0.75, n=60,
            ),
        )
        r = interpret(rep)
        assert "non_normal" in codes(r)
        assert "rec_maintain" not in rec_codes(r), (
            "advice must not exceed evidence when normality is rejected"
        )

    def test_maintain_withheld_for_small_sample(self) -> None:
        rep = build_capability_report(
            mean=10.0, sigma_within=1.5, sigma_overall=1.55, spec=cases.SPEC,
            n_observations=12, n_subgroups=4,
            stability=StabilitySummary(is_stable=True),
        )
        r = interpret(rep)
        assert "small_sample" in codes(r)
        assert "rec_maintain" not in rec_codes(r)

    def test_conclusions_and_recommendations_stay_separated(self) -> None:
        """Gating touches recommendations only; conclusions are unaffected."""
        r = interpret(cases.zero_variation())
        assert "zero_variation" in codes(r)          # statistical conclusion kept
        assert all(
            c.code != "rec_maintain" for c in r.conclusions
        ), "recommendation codes must never appear among conclusions"


class TestZeroVariationFlatLineExplanation:
    """B4: the N7-on-flat-line signal is explained, not left contradictory."""

    def test_zero_variation_conclusion_explains_n7(self) -> None:
        r = interpret(cases.zero_variation())
        zv = [c for c in r.conclusions if c.code == "zero_variation"][0]
        text = zv.text.lower()
        assert "n7" in text
        assert "resolution" in text
        # It must frame N7 as stratification/resolution, not a process shift.
        assert "not an ordinary process shift" in text


class TestDrift:
    def test_ppk_worse_than_cpk(self) -> None:
        r = interpret(cases.drift_ppk_worse())
        c = codes(r)
        assert "long_term_drift" in c
        assert "rec_investigate_drift" in rec_codes(r)
        drift = [x for x in r.conclusions if x.code == "long_term_drift"][0]
        assert drift.topic is ConclusionTopic.WITHIN_VS_OVERALL
        # Consistency conclusion must NOT also be present.
        assert "within_overall_consistent" not in c


class TestOverlappingConditions:
    def test_multiple_conclusions_coexist(self) -> None:
        # Unstable + small sample + non-normal all at once.
        from app.capability.guards import NormalityResult
        from app.capability.report import StabilitySummary, build_capability_report
        from app.data.models import SpecLimits
        rep = build_capability_report(
            mean=13.0, sigma_within=1.0, sigma_overall=3.0,
            spec=SpecLimits(usl=16.0, lsl=4.0, unit="mm"),
            n_observations=10, n_subgroups=3,
            stability=StabilitySummary(is_stable=False, triggered_rule_ids=("N3",), n_violations=2),
            normality=NormalityResult(normal=False, shapiro_stat=0.9, shapiro_p=0.002,
                                      anderson_stat=1.5, anderson_crit_5pct=0.75, n=10),
        )
        r = interpret(rep)
        c = codes(r)
        # All independent conditions produce their own conclusions.
        assert {"process_unstable", "small_sample", "non_normal",
                "off_center", "long_term_drift"}.issubset(c)
        # Recommendations for each preserved.
        assert {"rec_investigate_stability", "rec_collect_more_data",
                "rec_check_distribution", "rec_investigate_centering",
                "rec_investigate_drift"}.issubset(rec_codes(r))


class TestSeparation:
    def test_conclusions_and_recommendations_are_separate_types(self) -> None:
        from app.interpret.models import (
            EngineeringRecommendation,
            StatisticalConclusion,
        )
        r = interpret(cases.high_cp_low_cpk())
        assert all(isinstance(x, StatisticalConclusion) for x in r.conclusions)
        assert all(isinstance(x, EngineeringRecommendation) for x in r.recommendations)
        # A recommendation carries its trigger; a conclusion does not advise.
        for rec in r.recommendations:
            assert rec.trigger_code  # every rec is tied to a condition

    def test_headline_is_a_conclusion_not_advice(self) -> None:
        r = interpret(cases.high_cpk_unstable())
        # Headline should state the fact of instability, not tell user what to do.
        assert "unstable" in r.headline.lower()
        assert "investigate" not in r.headline.lower()


class TestConfigurableThresholds:
    def test_custom_bands_change_category(self) -> None:
        rep = cases.stable_capable()  # Cpk ~1.333
        # Default: 1.333 -> CAPABLE. Raise capable_min so it becomes MARGINAL.
        strict = InterpretationThresholds(capable_min=1.5, excellent_min=2.0)
        r = interpret(rep, thresholds=strict)
        assert "capability_marginal" in codes(r)
        assert "capability_capable" not in codes(r)

    def test_categorize_helper(self) -> None:
        t = InterpretationThresholds()
        assert t.categorize(0.9) is CapabilityCategory.INADEQUATE
        assert t.categorize(1.1) is CapabilityCategory.MARGINAL
        assert t.categorize(1.4) is CapabilityCategory.CAPABLE
        assert t.categorize(1.8) is CapabilityCategory.EXCELLENT
        assert t.categorize(None) is CapabilityCategory.UNDEFINED

    def test_custom_centering_ratio(self) -> None:
        rep = cases.stable_capable()  # well-centered by default gap
        # Force a tiny centering ratio so any gap triggers off_center.
        loose = InterpretationThresholds(centering_gap_ratio=0.0)
        r = interpret(rep, thresholds=loose)
        # gap >= 0 always true -> off_center fires.
        assert "off_center" in codes(r)


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        rep = cases.high_cp_low_cpk()
        r1 = interpret(rep)
        r2 = interpret(rep)
        assert r1.conclusion_codes == r2.conclusion_codes
        assert r1.recommendation_codes == r2.recommendation_codes
        assert r1.headline == r2.headline


class TestThresholdValidation:
    """`InterpretationThresholds` must reject incoherent cut-points.

    `categorize()` walks the bands in ascending order, so an out-of-order triple
    does not fail on its own -- it silently mislabels. These tests pin that the
    dataclass refuses the invalid state instead of producing nonsense.
    """

    def test_defaults_are_valid(self) -> None:
        t = InterpretationThresholds()
        assert t.marginal_min < t.capable_min < t.excellent_min

    def test_reversed_bands_raise(self) -> None:
        with pytest.raises(ValueError, match="increase strictly"):
            InterpretationThresholds(
                marginal_min=1.67, capable_min=1.00, excellent_min=1.33
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"marginal_min": 1.33, "capable_min": 1.33},          # not strict
            {"capable_min": 1.67, "excellent_min": 1.67},         # not strict
            {"marginal_min": 2.0},                                # above capable
            {"excellent_min": 1.0},                               # below capable
        ],
    )
    def test_non_increasing_bands_raise(self, kwargs) -> None:
        with pytest.raises(ValueError, match="increase strictly"):
            InterpretationThresholds(**kwargs)

    @pytest.mark.parametrize("field", ["centering_gap_ratio", "drift_gap_ratio"])
    @pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
    def test_gap_ratios_must_be_proportions(self, field, bad) -> None:
        with pytest.raises(ValueError, match=field):
            InterpretationThresholds(**{field: bad})

    @pytest.mark.parametrize("field", ["centering_gap_ratio", "drift_gap_ratio"])
    @pytest.mark.parametrize("ok", [0.0, 0.5, 0.99])
    def test_gap_ratio_boundaries_accepted(self, field, ok) -> None:
        assert getattr(InterpretationThresholds(**{field: ok}), field) == ok

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_tight_sigma_must_be_positive(self, bad) -> None:
        with pytest.raises(ValueError, match="spec_margin_tight_sigma"):
            InterpretationThresholds(spec_margin_tight_sigma=bad)

    @pytest.mark.parametrize("field", ["min_observations", "min_subgroups"])
    def test_minimum_counts_must_be_at_least_one(self, field) -> None:
        with pytest.raises(ValueError, match=field):
            InterpretationThresholds(**{field: 0})

    def test_nan_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            InterpretationThresholds(capable_min=float("nan"))

    def test_ppap_style_configuration_is_accepted(self) -> None:
        """A real alternative standard must remain expressible."""
        t = InterpretationThresholds(
            marginal_min=1.33, capable_min=1.67, excellent_min=2.00
        )
        assert t.categorize(1.663) is CapabilityCategory.MARGINAL
