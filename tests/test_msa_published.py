"""Gage R&R verified against PUBLISHED results (see examples/DATASETS.md).

1. ``examples/gage_rr_aiag.csv`` -- the AIAG MSA 4th ed. example
   (10 parts x 3 appraisers x 3 trials), as reproduced in
   * Minitab, "Example of Crossed Gage R&R Study" (ANOVA, tolerance 8, k = 6);
   * BPI Consulting (2015), "Three Ways to Analyze a Gage R&R Study"
     (Average & Range, Table 2).
2. ``examples/gage_rr_thermal_impedance.csv`` -- Houf & Burman (1988) power
   module thermal impedance (10 x 3 x 3), as analysed in SAS/STAT 14.1 PROC
   VARCOMP Example 121.2 (full model, interaction significant).

Tolerances are half a unit in the last published digit.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.msa.models import GageMethod
from app.statistics.constants import d4_factor
from app.services.gage_rr import columns_to_request, run_gage_study

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(name: str, **opts):
    with open(EXAMPLES / name, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = run_gage_study(
        columns_to_request(
            [r["part"] for r in rows],
            [r["operator"] for r in rows],
            [r["value"] for r in rows],
            [r["trial"] for r in rows],
            **opts,
        )
    )
    assert out.ok, out.errors
    return out


def dp(n: int) -> float:
    """Half a unit in the n-th decimal place."""
    return 0.5 * 10.0**-n


@pytest.fixture(scope="module")
def aiag_anova():
    return run("gage_rr_aiag.csv", tolerance=8.0)


@pytest.fixture(scope="module")
def aiag_xbar_r():
    return run("gage_rr_aiag.csv", method=GageMethod.AVERAGE_RANGE)


@pytest.fixture(scope="module")
def thermal():
    # SAS spec limits (18, 58) -> tolerance 40
    return run("gage_rr_thermal_impedance.csv", tolerance=40.0)


class TestAiagAnovaVsMinitab:
    def test_full_anova_table(self, aiag_anova):
        t = aiag_anova.result.anova.full
        exp = {
            "Part": (9, 88.3619, 9.81799, 492.291),
            "Operator": (2, 3.1673, 1.58363, 79.406),
            "Part x Operator": (18, 0.3590, 0.01994, 0.434),
        }
        for src, (df, ss, ms, f) in exp.items():
            row = t.row(src)
            assert row.df == df
            assert row.ss == pytest.approx(ss, abs=dp(4))
            assert row.ms == pytest.approx(ms, abs=dp(5))
            assert row.f == pytest.approx(f, abs=dp(3))
        assert t.row("Part x Operator").p == pytest.approx(0.974, abs=dp(3))
        assert t.row("Repeatability").ss == pytest.approx(2.7589, abs=dp(4))
        assert t.row("Repeatability").ms == pytest.approx(0.04598, abs=dp(5))
        assert t.row("Total").ss == pytest.approx(94.6471, abs=dp(4))

    def test_interaction_pooled_and_reduced_table(self, aiag_anova):
        a = aiag_anova.result.anova
        assert a.interaction_pooled
        t = a.reduced
        assert t.row("Part").f == pytest.approx(245.614, abs=dp(3))
        assert t.row("Operator").f == pytest.approx(39.617, abs=dp(3))
        assert t.row("Repeatability").df == 78
        assert t.row("Repeatability").ss == pytest.approx(3.1179, abs=dp(4))
        assert t.row("Repeatability").ms == pytest.approx(0.03997, abs=dp(5))

    def test_variance_components_and_gage_evaluation(self, aiag_anova):
        r = aiag_anova.result
        exp = {
            # source: (VarComp, %Contribution, SD, 6*SD, %StudyVar, %Tolerance)
            "Total Gage R&R": (0.09143, 7.76, 0.30237, 1.81423, 27.86, 22.68),
            "Repeatability": (0.03997, 3.39, 0.19993, 1.19960, 18.42, 14.99),
            "Reproducibility": (0.05146, 4.37, 0.22684, 1.36103, 20.90, 17.01),
            "Part-to-Part": (1.08645, 92.24, 1.04233, 6.25396, 96.04, 78.17),
            "Total Variation": (1.17788, 100.00, 1.08530, 6.51180, 100.00, 81.40),
        }
        for src, (var, pc, sd, sv, psv, ptol) in exp.items():
            row = r.row(src)
            assert row.variance == pytest.approx(var, abs=dp(5)), src
            assert row.pct_contribution == pytest.approx(pc, abs=dp(2)), src
            assert row.sd == pytest.approx(sd, abs=dp(5)), src
            assert row.study_var == pytest.approx(sv, abs=dp(5)), src
            assert row.pct_study_var == pytest.approx(psv, abs=dp(2)), src
            assert row.pct_tolerance == pytest.approx(ptol, abs=dp(2)), src
        assert r.ndc == 4

    def test_interpretation(self, aiag_anova):
        rep = aiag_anova.interpretation
        assert "msa_grr_marginal" in rep.conclusion_codes  # 27.86% of study variation
        assert "msa_grr_tolerance_marginal" in rep.conclusion_codes  # 22.68% of tolerance
        assert "msa_ndc_low" in rep.conclusion_codes  # ndc 4 < 5
        assert "msa_interaction_pooled" in rep.conclusion_codes

    def test_range_chart_flags_part4_appraiser_b(self, aiag_anova):
        # Our own finding (not a published figure): appraiser B's readings on
        # part 4 are 0.01, 1.03, 0.20 -> R = 1.02 > UCL = D4(3) * R-bar = 0.880.
        rc = aiag_anova.result.range_check
        assert rc.r_bar == pytest.approx(0.3417, abs=dp(4))
        assert rc.ucl == pytest.approx(d4_factor(3) * rc.r_bar)
        assert rc.ucl == pytest.approx(0.8796, abs=dp(4))
        assert [(p, o) for p, o, _ in rc.out_of_control] == [("4", "B")]
        assert rc.out_of_control[0][2] == pytest.approx(1.02)
        assert aiag_anova.interpretation.conclusion_codes[0] == "msa_ranges_out_of_control"


class TestAiagAverageRangeVsBpi:
    def test_intermediates(self, aiag_xbar_r):
        d = aiag_xbar_r.result.average_range
        assert (d.k1, d.k2, d.k3) == (0.5908, 0.5231, 0.3146)
        assert d.r_bar_bar == pytest.approx(0.342, abs=dp(3))
        assert d.x_bar_diff == pytest.approx(0.445, abs=dp(3))
        assert d.r_p == pytest.approx(3.511, abs=dp(3))

    def test_components(self, aiag_xbar_r):
        c = aiag_xbar_r.result.components
        assert c.repeatability_sd == pytest.approx(0.202, abs=dp(3))  # EV
        assert c.reproducibility_sd == pytest.approx(0.230, abs=dp(3))  # AV
        assert c.grr_sd == pytest.approx(0.306, abs=dp(3))  # R&R
        assert c.part_sd == pytest.approx(1.105, abs=dp(3))  # PV (text; Table 2 prints 1.104)
        assert c.total_sd == pytest.approx(1.146, abs=dp(3))  # TV

    def test_percentages(self, aiag_xbar_r):
        r = aiag_xbar_r.result
        assert r.row("Repeatability").pct_study_var == pytest.approx(17.61, abs=dp(2))
        assert r.row("Reproducibility").pct_study_var == pytest.approx(20.04, abs=dp(2))
        assert r.row("Total Gage R&R").pct_study_var == pytest.approx(26.68, abs=dp(2))
        # BPI prints 96.37; the unrounded value is 96.376 -> within 0.01.
        assert r.row("Part-to-Part").pct_study_var == pytest.approx(96.37, abs=0.01)


class TestThermalImpedanceVsSas:
    def test_anova_table(self, thermal):
        t = thermal.result.anova.full
        exp = {
            "Part": (9, 3935.955556, 437.328395),
            "Operator": (2, 39.266667, 19.633333),
            "Part x Operator": (18, 48.511111, 2.695062),
            "Repeatability": (60, 30.666667, 0.511111),
        }
        for src, (df, ss, ms) in exp.items():
            row = t.row(src)
            assert row.df == df and row.ss == pytest.approx(ss, abs=dp(6)) and row.ms == pytest.approx(ms, abs=dp(6))
        assert t.row("Total").ss == pytest.approx(4054.4, abs=dp(6))

    def test_variance_components(self, thermal):
        a = thermal.result.anova
        assert not a.interaction_pooled and a.interaction_p < 1e-6
        c = thermal.result.components
        assert c.part_sd**2 == pytest.approx(48.29259, abs=dp(5))  # Var(a)
        assert c.operator_sd**2 == pytest.approx(0.56461, abs=dp(5))  # Var(b)
        assert c.interaction_sd**2 == pytest.approx(0.72798, abs=dp(5))  # Var(a*b)
        assert c.repeatability_sd**2 == pytest.approx(0.51111, abs=dp(5))  # Var(Error)
        assert c.grr_sd**2 == pytest.approx(1.80370, abs=dp(5))  # Gamma M
        assert c.total_sd**2 == pytest.approx(50.09630, abs=dp(5))  # Gamma Y

    def test_precision_to_tolerance(self, thermal):
        # SAS PTR(18,58,6) = 6 * sqrt(Gamma M) / (58 - 18) = 0.20145
        assert thermal.result.pct_grr_tolerance / 100 == pytest.approx(0.20145, abs=dp(5))

    def test_interaction_reported_significant(self, thermal):
        assert "msa_interaction_significant" in thermal.interpretation.conclusion_codes
