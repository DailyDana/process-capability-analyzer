"""Tests for the deterministic Gage R&R interpretation (app.msa.interpret).

Assertions are on stable codes, never on prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.interpret.models import RecommendationCategory, Severity
from app.msa.interpret import GageThresholds, GrrVerdict, grr_verdict, headline_verdict, interpret_gage
from app.msa.models import GageMethod, GageStudy
from app.services.gage_rr import analyze_study


def study_from(part_sd, op_shift, rep_sd, p=10, o=3, r=3, seed=3, inter_sd=0.0):
    rng = np.random.default_rng(seed)
    parts = np.linspace(-1.6, 1.6, p) * part_sd  # deterministic spread
    ops = np.linspace(-1, 1, o) * op_shift
    inter = rng.normal(0, inter_sd, (p, o)) if inter_sd else np.zeros((p, o))
    y = 50 + parts[:, None, None] + ops[None, :, None] + inter[:, :, None] + rng.normal(0, rep_sd, (p, o, r))
    return GageStudy(y, tuple(f"P{i}" for i in range(p)), tuple(f"O{j}" for j in range(o)))


def run(study, **kw):
    method = kw.pop("method", GageMethod.ANOVA)
    res = analyze_study(study, method=method, **kw)
    return res, interpret_gage(res)


class TestBands:
    @pytest.mark.parametrize(
        "pct,verdict",
        [(0.0, "acceptable"), (9.99, "acceptable"), (10.0, "marginal"), (30.0, "marginal"), (30.01, "unacceptable")],
    )
    def test_band_edges(self, pct, verdict):
        assert grr_verdict(pct) is GrrVerdict(verdict)

    def test_thresholds_validated(self):
        with pytest.raises(ValueError):
            GageThresholds(acceptable_max=30, marginal_max=10)


class TestVerdicts:
    def test_good_gauge_proceeds(self):
        res, rep = run(study_from(part_sd=1.0, op_shift=0.005, rep_sd=0.01))
        assert res.pct_grr_study_var < 10
        assert "msa_grr_acceptable" in rep.conclusion_codes
        assert "msa_ndc_adequate" in rep.conclusion_codes
        assert "rec_msa_proceed" in rep.recommendation_codes
        assert rep.headline.startswith("Measurement system ACCEPTABLE")

    def test_repeatability_dominated(self):
        res, rep = run(study_from(part_sd=0.3, op_shift=0.0, rep_sd=0.25))
        assert "msa_grr_unacceptable" in rep.conclusion_codes
        assert "rec_msa_repeatability" in rep.recommendation_codes
        assert "rec_msa_proceed" not in rep.recommendation_codes

    def test_reproducibility_dominated(self):
        res, rep = run(study_from(part_sd=0.5, op_shift=0.4, rep_sd=0.01))
        assert "rec_msa_reproducibility" in rep.recommendation_codes
        recs = {r.code: r for r in rep.recommendations}
        assert recs["rec_msa_reproducibility"].category is RecommendationCategory.IMPROVE_REPRODUCIBILITY

    def test_tolerance_verdict_and_worst_headline(self):
        # Good against process variation, poor against a tight tolerance.
        st = study_from(part_sd=1.0, op_shift=0.005, rep_sd=0.02)
        res, rep = run(st, tolerance=0.5)
        codes = rep.conclusion_codes
        assert "msa_grr_acceptable" in codes
        assert any(c.startswith("msa_grr_tolerance_") and c != "msa_grr_tolerance_acceptable" for c in codes)
        assert "of tolerance" in rep.headline
        assert headline_verdict(res) is not GrrVerdict.ACCEPTABLE
        assert "rec_msa_proceed" not in rep.recommendation_codes

    def test_low_ndc(self):
        res, rep = run(study_from(part_sd=0.2, op_shift=0.0, rep_sd=0.12))
        assert res.ndc is not None and res.ndc < 5
        assert "msa_ndc_low" in rep.conclusion_codes
        assert "rec_msa_discrimination" in rep.recommendation_codes


class TestConsistencyAndModel:
    def test_out_of_control_range_caveats_everything(self):
        st = study_from(part_sd=1.0, op_shift=0.005, rep_sd=0.01)
        y = np.array(st.values)
        y[2, 1, 2] += 0.5
        res, rep = run(GageStudy(y, st.part_labels, st.operator_labels))
        assert not res.range_check.in_control
        assert rep.conclusion_codes[0] == "msa_ranges_out_of_control"
        assert "rec_msa_investigate_ranges" in rep.recommendation_codes
        assert "rec_msa_proceed" not in rep.recommendation_codes
        assert "caveat" in rep.headline

    def test_significant_interaction(self):
        res, rep = run(study_from(part_sd=1.0, op_shift=0.0, rep_sd=0.02, inter_sd=0.15))
        assert res.anova.interaction_p < 0.05
        assert "msa_interaction_significant" in rep.conclusion_codes
        assert "rec_msa_interaction" in rep.recommendation_codes

    def test_pooled_interaction_reported(self):
        res, rep = run(study_from(part_sd=1.0, op_shift=0.01, rep_sd=0.05))
        assert res.anova.interaction_pooled
        assert "msa_interaction_pooled" in rep.conclusion_codes

    def test_average_range_has_no_interaction_conclusions(self):
        res, rep = run(study_from(part_sd=1.0, op_shift=0.01, rep_sd=0.05), method=GageMethod.AVERAGE_RANGE)
        assert not any(c.startswith("msa_interaction") for c in rep.conclusion_codes)

    def test_zero_variation(self):
        res, rep = run(GageStudy(np.ones((10, 3, 2)), tuple(map(str, range(10))), ("a", "b", "c")))
        assert rep.conclusion_codes == ("msa_zero_variation",)
        assert rep.max_severity is Severity.CRITICAL
        assert "not assessable" in rep.headline

    def test_perfect_gauge_ndc_undefined(self):
        y = np.repeat(np.linspace(1, 2, 10)[:, None, None], 3, axis=1).repeat(2, axis=2)
        res, rep = run(GageStudy(y, tuple(map(str, range(10))), ("a", "b", "c")))
        assert res.ndc is None
        assert "msa_grr_zero" in rep.conclusion_codes

    def test_conclusions_never_contain_recommendation_codes(self):
        _, rep = run(study_from(part_sd=0.3, op_shift=0.2, rep_sd=0.2))
        assert all(c.code.startswith("msa_") for c in rep.conclusions)
        assert all(r.code.startswith("rec_msa_") for r in rep.recommendations)
        assert all(r.trigger_code for r in rep.recommendations)
