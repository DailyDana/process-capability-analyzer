"""Attribute control charts: published limits, rules, service, exports and page.

The published values are the output printed in the documentation of the R
package ``qcc`` for the same data (see ``examples/DATASETS.md``); they are
quoted to the digits printed there.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from app.charts.attribute_render import render_attribute_chart
from app.interpret.attribute import AttributeThresholds, interpret_attribute
from app.reporting.attribute_report import attribute_csv_string, sample_rows, summary_rows
from app.services.attribute import AttributeRequest, run_attribute
from app.spc.attribute import APPLICABLE_RULES, AttributeKind, attribute_chart, evaluate_attribute_rules
from app.spc.rules import Direction, RuleSet

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "ui" / "attribute_page.py")
P, NP, C, U = AttributeKind.P, AttributeKind.NP, AttributeKind.C, AttributeKind.U
PRINTED = 5e-7  # qcc prints 7 significant digits: half a unit in the last one


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "examples" / f"attribute_{name}.csv")


@pytest.fixture(scope="module")
def oj() -> pd.DataFrame:
    return _load("orangejuice")


@pytest.fixture(scope="module")
def circuit() -> pd.DataFrame:
    return _load("circuit")


def _flagged(chart) -> list[int]:
    return (np.nonzero(chart.beyond_limits)[0] + 1).tolist()


class TestPublishedLimits:
    def test_orange_juice_p_chart_trial_samples(self, oj):
        t = oj[oj.trial]
        c = attribute_chart(P, t.D, t["size"])
        assert c.center_line == pytest.approx(0.2313333, rel=PRINTED)
        assert c.lcl[0] == pytest.approx(0.05242755, rel=PRINTED)
        assert c.ucl[0] == pytest.approx(0.4102391, rel=PRINTED)
        assert _flagged(c) == [15, 23]  # the two samples Montgomery investigates

    def test_orange_juice_np_chart(self, oj):
        t = oj[oj.trial]
        c = attribute_chart(NP, t.D, t["size"])
        assert c.center_line == pytest.approx(11.56667, rel=PRINTED)
        assert c.lcl[0] == pytest.approx(2.621377, rel=PRINTED)
        assert c.ucl[0] == pytest.approx(20.51196, rel=PRINTED)

    def test_orange_juice_revised_limits_applied_to_new_samples(self, oj):
        """Samples 15 and 23 removed from the baseline; samples 31-54 judged against it."""
        c = attribute_chart(P, oj.D, oj["size"], n_baseline=30, exclude=[14, 22])
        assert c.n_baseline == 28
        assert c.center_line == pytest.approx(0.215, rel=PRINTED)
        assert c.lcl[0] == pytest.approx(0.04070284, rel=PRINTED)
        assert c.ucl[0] == pytest.approx(0.3892972, rel=PRINTED)
        ev = evaluate_attribute_rules(c)
        runs = [v for v in ev.violations if v.rule_id == "N2"]
        # After the machine adjustment the fraction nonconforming dropped: a run below the centre line.
        assert runs and runs[0].direction is Direction.LOWER and min(runs[0].indices) >= 30

    def test_circuit_c_chart_trial_samples(self, circuit):
        t = circuit[circuit.trial]
        c = attribute_chart(C, t.x, t["size"])
        assert c.center_line == pytest.approx(19.84615, rel=PRINTED)
        assert c.lcl[0] == pytest.approx(6.481447, rel=PRINTED)
        assert c.ucl[0] == pytest.approx(33.21086, rel=PRINTED)
        assert _flagged(c) == [6, 20]

    def test_circuit_revised_c_and_u_charts(self, circuit):
        c = attribute_chart(C, circuit.x, None, n_baseline=26, exclude=[5, 19])
        assert c.center_line == pytest.approx(19.66667, rel=PRINTED)
        assert (c.lcl[0], c.ucl[0]) == pytest.approx((6.362532, 32.9708), rel=PRINTED)
        u = attribute_chart(U, circuit.x, circuit["size"], n_baseline=26, exclude=[5, 19])
        assert u.center_line == pytest.approx(0.1966667, rel=PRINTED)
        assert (u.lcl[0], u.ucl[0]) == pytest.approx((0.06362532, 0.329708), rel=PRINTED)

    def test_pc_manufacture_u_chart(self):
        d = _load("pcmanufact")
        c = attribute_chart(U, d.x, d["size"])
        assert c.center_line == pytest.approx(1.93, rel=1e-12)
        assert (c.lcl[0], c.ucl[0]) == pytest.approx((0.06613305, 3.793867), rel=PRINTED)
        assert evaluate_attribute_rules(c).is_stable

    def test_dyed_cloth_varying_sample_size(self):
        """153 defects in 107.5 units; each roll has its own limits ubar +- 3 sqrt(ubar / n_i)."""
        d = _load("dyedcloth")
        c = attribute_chart(U, d.x, d["size"])
        assert not c.constant_size
        assert c.center_line == pytest.approx(153 / 107.5, rel=1e-14)
        for i in (0, 1, 2):
            half = 3 * (153 / 107.5 / d["size"][i]) ** 0.5
            assert c.ucl[i] == pytest.approx(153 / 107.5 + half, rel=1e-13)
            assert c.lcl[i] == pytest.approx(153 / 107.5 - half, rel=1e-13)
        assert c.ucl[1] > c.ucl[0] > c.ucl[2]  # sizes 8 < 10 < 13: smaller sample, wider limits


class TestChartRules:
    def test_np_equals_p_times_n(self):
        x, n = [3, 5, 2, 8, 4], 40
        p, npc = attribute_chart(P, x, n), attribute_chart(NP, x, n)
        assert npc.center_line == pytest.approx(n * p.center_line)
        assert npc.ucl == pytest.approx(n * p.ucl) and npc.lcl == pytest.approx(n * p.lcl)

    def test_u_with_unit_size_equals_c(self):
        x = [4, 7, 3, 9, 5, 6]
        c, u = attribute_chart(C, x), attribute_chart(U, x, 1)
        assert (c.center_line, list(c.ucl), list(c.lcl)) == (u.center_line, list(u.ucl), list(u.lcl))

    def test_limits_are_clipped_to_the_possible_range(self):
        c = attribute_chart(P, [1, 0, 2, 1], 10)
        assert (c.lcl == 0).all()
        high = attribute_chart(P, [9, 10, 9, 10], 10)
        assert (high.ucl == 1.0).all()
        assert (attribute_chart(NP, [9, 10, 9, 10], 10).ucl == 10).all()

    def test_point_on_a_limit_is_not_flagged(self):
        c = attribute_chart(C, [4, 4, 4, 4, 10], center=4.0)  # UCL = 4 + 3*2 = 10 exactly
        assert c.ucl[0] == 10.0 and not c.beyond_limits.any()
        assert attribute_chart(C, [4, 4, 4, 4, 11], center=4.0).beyond_limits.tolist()[-1]

    def test_given_standard_estimates_nothing(self):
        c = attribute_chart(P, [5, 9, 30], 100, center=0.05)
        assert c.center_is_standard and c.n_baseline == 0 and c.center_line == 0.05
        assert _flagged(c) == [3]

    def test_excluded_samples_are_still_flagged(self):
        c = attribute_chart(C, [5, 6, 4, 40, 5, 6], exclude=[3])
        assert not c.in_baseline[3] and c.beyond_limits[3]
        assert c.center_line == pytest.approx(26 / 5)

    @pytest.mark.parametrize("kind,counts,sizes,fragment", [
        (P, [1, 2], None, "needs the sample size"),
        (P, [1, 60], [50, 50], "c or u chart"),
        (P, [1, 2], [50.5, 50], "whole numbers of items"),
        (NP, [1, 2], [50, 60], "use the p chart"),
        (C, [1, 2], [1, 2], "use the u chart"),
        (C, [1, -2], None, "negative"),
        (C, [1, 2.5], None, "whole numbers"),
        (U, [1, 2], [1, 0], "greater than zero"),
        (U, [1, 2], [1, 2, 3], "sample sizes"),
        (C, [], None, "No counts"),
        (C, [1, float("nan")], None, "finite"),
    ])
    def test_unusable_input_is_explained(self, kind, counts, sizes, fragment):
        with pytest.raises(ValueError, match=fragment):
            attribute_chart(kind, counts, sizes)

    @pytest.mark.parametrize("kwargs,fragment", [
        (dict(n_baseline=0), "baseline"), (dict(n_baseline=9), "baseline"),
        (dict(exclude=[7]), "does not exist"), (dict(exclude=[0, 1, 2, 3]), "nothing is left"),
        (dict(center=-1.0), "valid rate"),
    ])
    def test_bad_baseline_options(self, kwargs, fragment):
        with pytest.raises(ValueError, match=fragment):
            attribute_chart(C, [3, 4, 5, 6], **kwargs)

    def test_only_zone_free_rules_are_reported(self):
        rng = np.random.default_rng(5)
        for _ in range(60):
            x = rng.poisson(12, 60)
            x[20:28] += 9  # sustained shift: would trip the zone rules
            ev = evaluate_attribute_rules(attribute_chart(C, x), RuleSet.ALL)
            assert set(ev.triggered_rule_ids) <= APPLICABLE_RULES
            assert {s.rule_id for s in ev.skipped} <= APPLICABLE_RULES

    def test_varying_sizes_use_standardised_values_for_runs(self):
        """A run above the centre line is found although the raw limits differ per sample."""
        sizes = [50, 80] * 10
        counts = [round(0.10 * n) for n in sizes[:10]] + [round(0.14 * n) for n in sizes[10:]]
        c = attribute_chart(P, counts, sizes, n_baseline=10)
        ev = evaluate_attribute_rules(c)
        assert any(v.rule_id == "N2" and v.direction is Direction.UPPER for v in ev.violations)

    def test_beyond_limit_explanation_is_in_chart_units(self):
        c = attribute_chart(P, [5, 6, 4, 30, 5], [50, 80, 50, 80, 50])
        v = evaluate_attribute_rules(c).violations[0]
        assert v.rule_id == "N1" and "Sample 4 = 0.375" in v.explanation and "above its UCL" in v.explanation

    def test_false_alarm_rate_is_near_nominal(self):
        """Poisson(20), 3-sigma limits from a known standard: about 0.3% of points beyond."""
        rng = np.random.default_rng(9)
        c = attribute_chart(C, rng.poisson(20, 200_000), center=20.0)
        assert 0.001 < c.beyond_limits.mean() < 0.006


class TestInterpretation:
    def _rep(self, chart, **kw):
        return interpret_attribute(chart, evaluate_attribute_rules(chart), AttributeThresholds(**kw))

    def test_stable_chart(self):
        d = _load("pcmanufact")
        rep = self._rep(attribute_chart(U, d.x, d["size"]))
        assert rep.conclusion_codes == ("process_stable",)
        assert rep.recommendation_codes == ("rec_maintain",) and rep.headline.startswith("In control")

    def test_high_signals(self, oj):
        t = oj[oj.trial]
        rep = self._rep(attribute_chart(P, t.D, t["size"]))
        assert {"process_unstable", "attribute_signal_high"} <= set(rep.conclusion_codes)
        assert "rec_investigate_stability" in rep.recommendation_codes
        assert "15, 23" in next(c.text for c in rep.conclusions if c.code == "attribute_signal_high")

    def test_low_side_run_asks_to_verify_the_improvement(self, oj):
        rep = self._rep(attribute_chart(P, oj.D, oj["size"], n_baseline=30, exclude=[14, 22]))
        assert "attribute_signal_low" in rep.conclusion_codes
        assert "rec_verify_improvement" in rep.recommendation_codes

    def test_few_samples_and_low_expected_count(self):
        rep = self._rep(attribute_chart(P, [1, 0, 2, 1, 0, 1], 20))
        assert {"few_subgroups", "low_expected_count", "lcl_at_zero"} <= set(rep.conclusion_codes)
        assert {"rec_collect_more_data", "rec_increase_sample_size"} <= set(rep.recommendation_codes)
        assert "rec_maintain" not in rep.recommendation_codes

    def test_every_recommendation_names_its_trigger(self, oj):
        rep = self._rep(attribute_chart(P, oj.D, oj["size"], n_baseline=30))
        assert all(r.trigger_code in rep.conclusion_codes for r in rep.recommendations)

    def test_thresholds_validate(self):
        with pytest.raises(ValueError):
            AttributeThresholds(min_samples=1)


class TestService:
    def test_ok(self, oj):
        o = run_attribute(AttributeRequest(P, tuple(oj.D), tuple(oj["size"]), n_baseline=30, exclude=(15, 23)))
        assert o.ok and o.chart.n_baseline == 28 and o.chart.center_line == pytest.approx(0.215)

    def test_constant_size_and_text_numbers(self):
        o = run_attribute(AttributeRequest(P, ("3", " 5 ", 2, "4"), constant_size=50))
        assert o.ok and o.chart.center_line == pytest.approx(14 / 200)

    def test_trailing_blank_rows_are_dropped_with_a_warning(self):
        o = run_attribute(AttributeRequest(C, (3, 5, 2, None, float("nan"), "")))
        assert o.ok and o.chart.n_samples == 3 and "3 blank row(s)" in o.warnings[0]

    @pytest.mark.parametrize("req,fragment", [
        (AttributeRequest(C, (3, "abc", 2)), "row 2 ('abc')"),
        (AttributeRequest(C, (3, None, 2)), "Missing count or sample size in row(s) 2"),
        (AttributeRequest(P, (3, 4), (50, None)), "row(s) 2"),
        (AttributeRequest(P, (3, 4), (50,)), "2 counts but 1 sample sizes"),
        (AttributeRequest(C, (3,)), "At least 2 samples"),
        (AttributeRequest(P, (3, 4)), "needs the sample size"),
        (AttributeRequest(C, (3, 4), exclude=(5,)), "does not exist"),
        (AttributeRequest(C, (3, float("inf"))), "must be numbers"),
    ])
    def test_user_errors_come_back_as_messages(self, req, fragment):
        o = run_attribute(req)
        assert not o.ok and fragment in o.errors[0]

    def test_all_zero_baseline_warns(self):
        o = run_attribute(AttributeRequest(C, (0, 0, 0, 0)))
        assert o.ok and "No nonconformities" in o.warnings[0]


@pytest.fixture(scope="module")
def outcome(oj):
    return run_attribute(AttributeRequest(P, tuple(oj.D), tuple(oj["size"]), n_baseline=30, exclude=(15, 23)))


class TestOutput:

    def test_csv_carries_the_chart_at_full_precision(self, outcome):
        text = attribute_csv_string(outcome.chart, outcome.evaluation, outcome.interpretation)
        rows = list(csv.reader(io.StringIO(text)))
        flat = {r[0]: r[1] for r in rows if len(r) == 2}
        assert float(flat["UCL"]) == float(outcome.chart.ucl[0])
        assert float(flat["Centre line"]) == 0.215
        start = rows.index(["# Samples"]) + 1
        assert rows[start][:4] == ["Sample", "Count", "Size", "Plotted"]
        assert len(rows) - start - 1 == 54
        assert any(r and r[0] == "conclusion" and r[1] == "process_unstable" for r in rows)

    def test_sample_rows_list_the_rules_per_sample(self, outcome):
        rows = sample_rows(outcome.chart, outcome.evaluation)
        assert rows[14]["Rules"] == "N1" and rows[14]["In baseline"] is False
        assert rows[0]["Rules"] == "" and rows[0]["In baseline"] is True

    def test_summary_for_varying_sizes(self):
        d = _load("dyedcloth")
        rows = dict(summary_rows(attribute_chart(U, d.x, d["size"])))
        assert rows["Sample size"] == "varies, 8 to 13" and "per sample" in rows["UCL"]

    def test_figures_render(self, outcome):
        d = _load("dyedcloth")
        for fig in (render_attribute_chart(outcome.chart, outcome.evaluation),
                    render_attribute_chart(attribute_chart(U, d.x, d["size"])),
                    render_attribute_chart(attribute_chart(C, [3, 9, 4], center=4.0))):
            assert isinstance(fig, Figure) and fig.axes[0].has_data()


class TestPage:
    def test_default_example(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        assert not at.exception
        assert [h.value for h in at.main.header] == ["1 · Data", "2 · Chart", "3 · Interpretation", "4 · Export"]
        assert any("OUT OF CONTROL" in e.value for e in at.error)

    def test_every_example_runs(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        for label in at.radio(key="at_source").options[:-1]:
            at.radio(key="at_source").set_value(label)
            at.run()
            assert not at.exception, label
            assert len(at.main.header) == 4, label

    def test_wrong_chart_for_the_data_is_reported(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        at.radio(key="at_source").set_value("Example: dyed cloth (u chart, varying sample size)")
        at.run()
        kind = next(s for s in at.selectbox if s.label == "Chart")
        kind.set_value("c — nonconformities per sample")
        at.run()
        assert not at.exception
        assert any("use the u chart" in e.value for e in at.error)
