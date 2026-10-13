"""Tests for the Gage R&R statistical core (app.msa).

Three independent lines of evidence:

1. a 2x2x2 study small enough to compute every sum of squares by hand
   (derivation in the comments below);
2. an independent implementation -- statsmodels OLS ANOVA, stored as
   ``tests/fixtures/gage_oracle.py`` by ``tools/make_gage_fixtures.py``;
3. Monte Carlo: the (unclamped) expected-mean-square estimators must be
   unbiased for known variance components.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from app.msa import constants
from app.msa.anova import anova_gage_rr
from app.msa.metrics import number_of_distinct_categories, summary_rows
from app.msa.models import ComponentEstimates, GageStudy
from app.msa.ranges import range_check
from app.msa.xbar_r import average_range_gage_rr
from tests.fixtures.gage_oracle import ORACLE

FIX = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Hand-computed 2 parts x 2 operators x 2 trials
#
#   P1/O1: 1, 3   P1/O2: 2, 4   P2/O1: 5, 7   P2/O2: 8, 8
#   cell means 2, 3, 6, 8; part means 2.5, 7; operator means 4, 5.5; grand 4.75
#   SS_P  = 2*2*((2.5-4.75)^2 + (7-4.75)^2)            = 40.5
#   SS_O  = 2*2*((4-4.75)^2 + (5.5-4.75)^2)             = 4.5
#   interaction residuals +-0.25 -> SS_PO = 2*4*0.0625  = 0.5
#   SS_E  = 2 + 2 + 2 + 0                               = 6
#   SS_T  = 51.5 (= sum of the above)
#   df 1, 1, 1, 4 -> MS 40.5, 4.5, 0.5, 1.5;  F_PO = 1/3, p ~ 0.595 > 0.25 -> pooled
#   pooled MS_E' = 6.5 / 5 = 1.3
#     s2_e = 1.3, s2_O = (4.5 - 1.3)/4 = 0.8, s2_P = (40.5 - 1.3)/4 = 9.8
#   full model (alpha_pool = 1):
#     s2_e = 1.5, s2_PO = (0.5 - 1.5)/2 = -0.5 -> 0 (clamped),
#     s2_O = (4.5 - 0.5)/4 = 1.0, s2_P = (40.5 - 0.5)/4 = 10.0
# ---------------------------------------------------------------------------
HAND = np.array([[[1, 3], [2, 4]], [[5, 7], [8, 8]]], dtype=float)


def hand_study() -> GageStudy:
    return GageStudy(HAND, ("P1", "P2"), ("O1", "O2"))


def load_csv_study(name: str) -> GageStudy:
    with open(FIX / name, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    parts = list(dict.fromkeys(r["part"] for r in rows))
    ops = list(dict.fromkeys(r["operator"] for r in rows))
    r = max(int(row["trial"]) for row in rows)
    arr = np.empty((len(parts), len(ops), r))
    for row in rows:
        arr[parts.index(row["part"]), ops.index(row["operator"]), int(row["trial"]) - 1] = float(row["value"])
    return GageStudy(arr, tuple(parts), tuple(ops))


class TestGageStudyModel:
    def test_shape_and_readonly(self):
        s = hand_study()
        assert (s.n_parts, s.n_operators, s.n_trials) == (2, 2, 2)
        with pytest.raises(ValueError):
            s.values[0, 0, 0] = 99.0

    def test_input_array_is_copied(self):
        arr = HAND.copy()
        s = GageStudy(arr, ("a", "b"), ("x", "y"))
        arr[0, 0, 0] = 1000.0
        assert s.values[0, 0, 0] == 1.0

    @pytest.mark.parametrize(
        "shape", [(1, 2, 2), (2, 1, 2), (2, 2, 1)], ids=["one-part", "one-operator", "one-trial"]
    )
    def test_minimum_sizes(self, shape):
        p, o, _ = shape
        with pytest.raises(ValueError, match="at least 2"):
            GageStudy(np.ones(shape), tuple("p" * p), tuple("o" * o))

    def test_non_finite_rejected(self):
        bad = HAND.copy()
        bad[1, 1, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            GageStudy(bad, ("a", "b"), ("x", "y"))

    def test_label_count_checked(self):
        with pytest.raises(ValueError, match="part labels"):
            GageStudy(HAND, ("a",), ("x", "y"))


class TestAnovaHandComputed:
    def test_sums_of_squares(self):
        t = anova_gage_rr(hand_study()).full
        assert t.row("Part").ss == pytest.approx(40.5, abs=1e-12)
        assert t.row("Operator").ss == pytest.approx(4.5, abs=1e-12)
        assert t.row("Part x Operator").ss == pytest.approx(0.5, abs=1e-12)
        assert t.row("Repeatability").ss == pytest.approx(6.0, abs=1e-12)
        assert t.row("Total").ss == pytest.approx(51.5, abs=1e-12)
        assert [t.row(s).df for s in ("Part", "Operator", "Part x Operator", "Repeatability", "Total")] == [1, 1, 1, 4, 7]

    def test_interaction_f_and_pooling(self):
        a = anova_gage_rr(hand_study())
        assert a.full.row("Part x Operator").f == pytest.approx(1.0 / 3.0)
        assert a.interaction_p == pytest.approx(0.5950, abs=5e-4)
        assert a.interaction_pooled
        assert a.reduced is not None
        assert a.reduced.row("Repeatability").df == 5
        assert a.reduced.row("Repeatability").ms == pytest.approx(1.3)

    def test_pooled_components(self):
        c = anova_gage_rr(hand_study()).components
        assert c.repeatability_sd**2 == pytest.approx(1.3)
        assert c.operator_sd**2 == pytest.approx(0.8)
        assert c.interaction_sd is None
        assert c.reproducibility_sd**2 == pytest.approx(0.8)
        assert c.part_sd**2 == pytest.approx(9.8)
        assert c.clamped == ()

    def test_full_model_components_and_clamping(self):
        a = anova_gage_rr(hand_study(), alpha_pool=1.0)
        assert not a.interaction_pooled and a.reduced is None
        c = a.components
        assert c.repeatability_sd**2 == pytest.approx(1.5)
        assert c.interaction_sd == 0.0
        assert c.raw_estimates["interaction"] == pytest.approx(-0.5)
        assert c.clamped == ("interaction",)
        assert c.operator_sd**2 == pytest.approx(1.0)
        assert c.part_sd**2 == pytest.approx(10.0)
        # main effects tested against MS_PO in the random-effects model
        assert a.full.row("Part").f == pytest.approx(40.5 / 0.5)
        assert a.full.row("Operator").f == pytest.approx(4.5 / 0.5)

    def test_alpha_pool_validated(self):
        with pytest.raises(ValueError):
            anova_gage_rr(hand_study(), alpha_pool=1.5)


class TestAnovaAgainstOracle:
    """statsmodels OLS ANOVA (independent implementation), stored fixtures."""

    @pytest.mark.parametrize("case", sorted(ORACLE))
    def test_matches_statsmodels(self, case):
        ref = ORACLE[case]
        a = anova_gage_rr(load_csv_study(ref["csv"]))
        for source in ("Part", "Operator", "Part x Operator", "Repeatability"):
            row = a.full.row(source)
            assert row.df == ref[source]["df"], source
            assert row.ss == pytest.approx(ref[source]["ss"], rel=1e-10), source
        inter = a.full.row("Part x Operator")
        assert inter.f == pytest.approx(ref["interaction_F"], rel=1e-10)
        assert a.interaction_p == pytest.approx(ref["interaction_p"], rel=1e-8)

    def test_fixture_cases_cover_both_branches(self):
        pooled = {c: anova_gage_rr(load_csv_study(ORACLE[c]["csv"])).interaction_pooled for c in ORACLE}
        assert set(pooled.values()) == {True, False}


class TestAnovaUnbiased:
    """E[MS] solutions are unbiased: the mean of the RAW (unclamped) estimates
    over many simulated studies must approach the true components."""

    def test_monte_carlo(self):
        rng = np.random.default_rng(12345)
        p, o, r = 10, 3, 3
        true = {"part": 1.0, "operator": 0.09, "interaction": 0.04, "repeatability": 0.0625}
        n_sim = 600
        acc = {k: 0.0 for k in true}
        for _ in range(n_sim):
            y = (
                rng.normal(0, math.sqrt(true["part"]), p)[:, None, None]
                + rng.normal(0, math.sqrt(true["operator"]), o)[None, :, None]
                + rng.normal(0, math.sqrt(true["interaction"]), (p, o))[:, :, None]
                + rng.normal(0, math.sqrt(true["repeatability"]), (p, o, r))
            )
            raw = anova_gage_rr(GageStudy(y, tuple(map(str, range(p))), tuple(map(str, range(o)))), alpha_pool=1.0).components.raw_estimates
            for k in acc:
                acc[k] += raw[k]
        for k, v in true.items():
            mean = acc[k] / n_sim
            # Tolerances are ~5 standard errors of the Monte Carlo mean, from
            # Var(MS) = 2 E[MS]^2 / df:  SE ~ 0.020 (part), 0.0039 (operator),
            # 0.00085 (interaction), 0.00047 (repeatability).
            tol = {"part": 0.10, "operator": 0.02, "interaction": 0.0045, "repeatability": 0.0025}[k]
            assert mean == pytest.approx(v, abs=tol), (k, mean, v)


class TestConstants:
    def test_aiag_k_tables_locked(self):
        # AIAG MSA 4th ed. values -- changing any of these is a methodology change.
        assert constants.K1 == {2: 0.8862, 3: 0.5908}
        assert constants.K2 == {2: 0.7071, 3: 0.5231}
        assert constants.K3 == {
            2: 0.7071, 3: 0.5231, 4: 0.4467, 5: 0.4030, 6: 0.3742,
            7: 0.3534, 8: 0.3375, 9: 0.3249, 10: 0.3146,
        }
        assert constants.NDC_FACTOR == 1.41

    def test_k1_matches_inverse_d2_for_many_subgroups(self):
        from app.statistics.constants import d2

        assert constants.K1[2] == pytest.approx(1 / d2(2), abs=5e-5)
        assert constants.K1[3] == pytest.approx(1 / d2(3), abs=5e-5)

    def test_k2_is_not_inverse_d2(self):
        # K2/K3 use d2* for ONE range (g=1); deriving them from d2 would be wrong.
        from app.statistics.constants import d2

        assert abs(constants.K2[2] - 1 / d2(2)) > 0.1

    @pytest.mark.parametrize("fn,n", [(constants.k1, 4), (constants.k2, 4), (constants.k3, 11)])
    def test_out_of_table_raises_with_hint(self, fn, n):
        with pytest.raises(ValueError, match="ANOVA"):
            fn(n)


class TestAverageRange:
    # Same 2x2x2 data. Ranges: 2, 2 (O1), 2, 0 (O2) -> operator avg ranges 2, 1
    #   R-bar-bar = 1.5, EV = 1.5 * 0.8862 = 1.3293
    #   operator averages 4, 5.5 -> X-bar_diff = 1.5
    #   AV^2 = (1.5 * 0.7071)^2 - EV^2/(2*2)
    #   part averages 2.5, 7 -> R_p = 4.5, PV = 4.5 * 0.7071
    def test_hand_example(self):
        c, d = average_range_gage_rr(hand_study())
        ev = 1.5 * 0.8862
        av2 = (1.5 * 0.7071) ** 2 - ev**2 / 4
        pv = 4.5 * 0.7071
        assert d.r_bar_bar == pytest.approx(1.5)
        assert d.x_bar_diff == pytest.approx(1.5)
        assert d.r_p == pytest.approx(4.5)
        assert c.repeatability_sd == pytest.approx(ev)
        assert c.reproducibility_sd == pytest.approx(math.sqrt(av2))
        assert c.part_sd == pytest.approx(pv)
        assert c.operator_sd is None and c.interaction_sd is None
        assert c.total_sd == pytest.approx(math.sqrt(ev**2 + av2 + pv**2))

    def test_negative_radicand_gives_zero_av(self):
        # Operators agree perfectly on average but repeat readings scatter.
        y = np.array([[[1.0, 3.0], [3.0, 1.0]], [[5.0, 7.0], [7.0, 5.0]]])
        c, d = average_range_gage_rr(GageStudy(y, ("a", "b"), ("x", "y")))
        assert d.x_bar_diff == 0.0
        assert d.av_radicand < 0.0
        assert c.reproducibility_sd == 0.0
        assert c.clamped == ("operator",)

    def test_out_of_table_study(self):
        y = np.random.default_rng(0).normal(size=(11, 2, 2))
        with pytest.raises(ValueError, match="2-10 parts"):
            average_range_gage_rr(GageStudy(y, tuple(map(str, range(11))), ("a", "b")))

    def test_agrees_with_anova_without_interaction(self):
        # Without interaction both methods estimate the same quantities; on the
        # pooled oracle case they agree to within sampling error.
        study = load_csv_study(ORACLE["10x3x2_pooled"]["csv"])
        xr, _ = average_range_gage_rr(study)
        an = anova_gage_rr(study).components
        assert xr.grr_sd == pytest.approx(an.grr_sd, rel=0.15)
        assert xr.part_sd == pytest.approx(an.part_sd, rel=0.15)


class TestRangeCheck:
    def test_flags_inconsistent_cell(self):
        y = np.full((5, 2, 2), 10.0)
        y[:, :, 1] += 0.1  # every cell range 0.1
        y[3, 1, 1] = 12.0  # one cell range 2.0
        s = GageStudy(y, tuple("abcde"), ("x", "y"))
        rc = range_check(s)
        assert not rc.in_control
        assert [(p, o) for p, o, _ in rc.out_of_control] == [("d", "y")]
        r_bar = (9 * 0.1 + 2.0) / 10
        assert rc.r_bar == pytest.approx(r_bar)
        assert rc.ucl == pytest.approx(3.266510868 * r_bar, rel=1e-8)

    def test_zero_ranges_in_control(self):
        rc = range_check(GageStudy(np.ones((2, 2, 2)), ("a", "b"), ("x", "y")))
        assert rc.in_control and rc.r_bar == 0.0


class TestMetrics:
    C = ComponentEstimates(repeatability_sd=0.3, reproducibility_sd=0.4, part_sd=2.0, operator_sd=0.4)

    def test_rows_and_percentages(self):
        rows = {r.source: r for r in summary_rows(self.C, k=6.0, tolerance=12.0)}
        grr = math.sqrt(0.09 + 0.16)  # 0.5
        tv = math.sqrt(0.25 + 4.0)
        assert list(rows) == [
            "Total Gage R&R", "Repeatability", "Reproducibility", "Operator", "Part-to-Part", "Total Variation",
        ]
        assert rows["Total Gage R&R"].sd == pytest.approx(grr)
        assert rows["Total Gage R&R"].pct_study_var == pytest.approx(100 * grr / tv)
        assert rows["Total Gage R&R"].pct_contribution == pytest.approx(100 * grr**2 / tv**2)
        assert rows["Total Gage R&R"].pct_tolerance == pytest.approx(100 * 6 * grr / 12.0)
        assert rows["Total Variation"].pct_study_var == pytest.approx(100.0)
        contrib = sum(rows[s].pct_contribution for s in ("Repeatability", "Reproducibility", "Part-to-Part"))
        assert contrib == pytest.approx(100.0)

    def test_k_affects_only_tolerance_and_study_var(self):
        a = {r.source: r for r in summary_rows(self.C, k=6.0, tolerance=10.0)}
        b = {r.source: r for r in summary_rows(self.C, k=5.15, tolerance=10.0)}
        for s in a:
            assert a[s].pct_study_var == pytest.approx(b[s].pct_study_var)
            assert a[s].pct_contribution == pytest.approx(b[s].pct_contribution)
        assert b["Total Gage R&R"].pct_tolerance == pytest.approx(a["Total Gage R&R"].pct_tolerance * 5.15 / 6)

    def test_no_tolerance_no_pct_tolerance(self):
        assert all(r.pct_tolerance is None for r in summary_rows(self.C))

    def test_interaction_row_only_when_estimated(self):
        with_int = ComponentEstimates(0.3, 0.4, 2.0, operator_sd=0.3, interaction_sd=0.2646)
        names = [r.source for r in summary_rows(with_int)]
        assert names.index("Operator") < names.index("Part x Operator") < names.index("Part-to-Part")
        xr = ComponentEstimates(0.3, 0.4, 2.0)
        assert "Operator" not in [r.source for r in summary_rows(xr)]

    def test_zero_total_variation(self):
        rows = summary_rows(ComponentEstimates(0.0, 0.0, 0.0, operator_sd=0.0))
        assert all(r.pct_study_var is None and r.pct_contribution is None for r in rows)

    @pytest.mark.parametrize("k,tol", [(0.0, None), (6.0, 0.0), (6.0, -1.0)])
    def test_bad_options(self, k, tol):
        with pytest.raises(ValueError):
            summary_rows(self.C, k=k, tolerance=tol)

    def test_ndc(self):
        # 1.41 * 2.0 / 0.5 = 5.64 -> 5 (truncated, not rounded)
        assert number_of_distinct_categories(self.C) == 5
        assert number_of_distinct_categories(ComponentEstimates(0.0, 0.0, 1.0)) is None
