"""Phase I / Phase II: limits from a baseline, frozen limits, CSV round trip.

Reference: the piston-ring data of Montgomery, *Introduction to Statistical
Quality Control* (R package ``qcc``, ``pistonrings``): 25 preliminary samples
of 5 give x-double-bar = 74.001, R-bar = 0.023, X-bar limits 73.988 / 74.014
and an R-chart UCL of 0.049; the 15 later samples drift upward.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.data.models import SpecLimits
from app.services.analysis import AnalysisRequest, ChartKind, run_analysis
from app.spc.limits import imr_chart, xbar_r_chart, xbar_s_chart
from app.spc.phases import FrozenLimits, apply_limits, baseline_mask, estimate_limits

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rings() -> tuple[float, ...]:
    with open(ROOT / "examples" / "pistonrings.csv", newline="", encoding="utf-8") as fh:
        return tuple(float(r["measurement"]) for r in csv.DictReader(fh))


@pytest.fixture(scope="module")
def groups(rings) -> list[np.ndarray]:
    return [np.asarray(rings[i:i + 5]) for i in range(0, 200, 5)]


class TestEstimateAndApply:
    def test_phase_one_gives_the_published_limits(self, groups):
        lim = estimate_limits("xbar_r", groups, baseline_mask(40, 25))
        assert lim.n_baseline == 25 and lim.subgroup_size == 5
        assert round(lim.location_cl, 3) == 74.001
        assert round(lim.dispersion_cl, 3) == 0.023
        assert (round(lim.location_lcl, 3), round(lim.location_ucl, 3)) == (73.988, 74.014)
        assert round(lim.dispersion_ucl, 3) == 0.048  # Montgomery prints 0.049, from R-bar rounded to 0.023
        assert lim.dispersion_lcl == 0.0

    def test_baseline_of_everything_equals_the_ordinary_chart(self, groups):
        for kind, build in (("xbar_r", xbar_r_chart), ("xbar_s", xbar_s_chart)):
            lim = estimate_limits(kind, groups, baseline_mask(40))
            pair, ref = apply_limits(kind, groups, lim), build(groups)
            for a, b in ((pair.location, ref.location), (pair.dispersion, ref.dispersion)):
                assert (a.center_line, a.ucl, a.lcl) == (b.center_line, b.ucl, b.lcl)
                assert np.array_equal(a.points, b.points) and np.array_equal(a.beyond_limits, b.beyond_limits)

    def test_imr_baseline_of_everything_equals_the_ordinary_chart(self, rings):
        x = rings[:60]
        pair, ref = apply_limits("imr", x, estimate_limits("imr", x, baseline_mask(60))), imr_chart(x)
        assert (pair.location.ucl, pair.location.lcl) == pytest.approx((ref.location.ucl, ref.location.lcl), rel=1e-14)
        assert pair.dispersion.ucl == pytest.approx(ref.dispersion.ucl, rel=1e-14)
        assert np.array_equal(pair.dispersion.point_indices, ref.dispersion.point_indices)

    def test_phase_two_drift_is_caught_by_the_frozen_limits(self, groups):
        lim = estimate_limits("xbar_r", groups, baseline_mask(40, 25))
        pair = apply_limits("xbar_r", groups, lim)
        flagged = (np.nonzero(pair.location.beyond_limits)[0] + 1).tolist()
        assert flagged and min(flagged) > 25  # nothing in Phase I, signals in Phase II
        assert flagged == [37, 38, 39]

    def test_limits_from_all_the_data_hide_part_of_the_drift(self, groups):
        """The reason for freezing: the drifting samples widen and shift their own limits."""
        frozen = apply_limits("xbar_r", groups, estimate_limits("xbar_r", groups, baseline_mask(40, 25)))
        assert frozen.location.n_beyond > xbar_r_chart(groups).location.n_beyond

    def test_excluded_subgroup_changes_the_estimate_but_is_still_charted(self, groups):
        full = estimate_limits("xbar_r", groups, baseline_mask(40, 25))
        less = estimate_limits("xbar_r", groups, baseline_mask(40, 25, exclude=[13]))
        assert less.n_baseline == 24 and less.location_cl != full.location_cl
        assert apply_limits("xbar_r", groups, less).location.n_points == 40

    def test_imr_moving_range_does_not_bridge_an_excluded_reading(self):
        x = [10.0, 10.2, 50.0, 10.1, 10.3, 10.2]
        lim = estimate_limits("imr", x, baseline_mask(6, exclude=[2]))
        # Usable ranges: |10.2-10.0|, |10.3-10.1|, |10.2-10.3|; the two touching 50.0 are dropped.
        assert lim.dispersion_cl == pytest.approx((0.2 + 0.2 + 0.1) / 3)
        assert lim.location_cl == pytest.approx(np.mean([10.0, 10.2, 10.1, 10.3, 10.2]))
        assert apply_limits("imr", x, lim).location.beyond_limits.tolist() == [False, False, True, False, False, False]

    @pytest.mark.parametrize("call,fragment", [
        (lambda g: baseline_mask(40, 0), "baseline"),
        (lambda g: baseline_mask(40, 41), "baseline"),
        (lambda g: baseline_mask(40, 25, exclude=[40]), "does not exist"),
        (lambda g: estimate_limits("xbar_r", g, baseline_mask(40, 1)), "at least 2 subgroups"),
        (lambda g: estimate_limits("imr", [1.0, 2.0, 3.0], baseline_mask(3, exclude=[1])), "consecutive"),
        (lambda g: estimate_limits("nope", g, baseline_mask(40)), "Unknown chart kind"),
        (lambda g: apply_limits("xbar_s", g, estimate_limits("xbar_r", g, baseline_mask(40))), "xbar_r chart"),
        (lambda g: apply_limits("xbar_r", [x[:4] for x in g], estimate_limits("xbar_r", g, baseline_mask(40))),
         "subgroups of 5"),
    ])
    def test_unusable_requests_are_explained(self, groups, call, fragment):
        with pytest.raises(ValueError, match=fragment):
            call(groups)


class TestFrozenLimitsFile:
    def test_round_trip_is_exact(self, groups):
        lim = estimate_limits("xbar_r", groups, baseline_mask(40, 25), note="line 3, week 14")
        assert FrozenLimits.from_csv(lim.to_csv()) == lim

    def test_hand_written_file_with_extra_spaces(self):
        text = "field,value\nChart, IMR\nsubgroup_size,1\nlocation_cl,10\nlocation_ucl,13\nlocation_lcl,7\n" \
               "dispersion_cl,1.128\ndispersion_ucl,3.686\ndispersion_lcl,0\n"
        lim = FrozenLimits.from_csv(text)
        assert lim.kind == "imr" and lim.location_ucl == 13.0 and lim.n_baseline == 0

    @pytest.mark.parametrize("text,fragment", [
        ("field,value\nchart,xbar_r\n", "missing"),
        ("field,value\nchart,xbar_r\nsubgroup_size,5\nlocation_cl,a\nlocation_ucl,1\nlocation_lcl,0\n"
         "dispersion_cl,1\ndispersion_ucl,2\ndispersion_lcl,0\n", "could not be read"),
        ("field,value\nchart,xbar_r\nsubgroup_size,5\nlocation_cl,5\nlocation_ucl,1\nlocation_lcl,0\n"
         "dispersion_cl,1\ndispersion_ucl,2\ndispersion_lcl,0\n", "LCL <= centre line <= UCL"),
        ("field,value\nchart,imr\nsubgroup_size,5\nlocation_cl,1\nlocation_ucl,2\nlocation_lcl,0\n"
         "dispersion_cl,1\ndispersion_ucl,2\ndispersion_lcl,0\n", "Subgroup size"),
        ("", "missing"),
    ])
    def test_bad_files_are_explained(self, text, fragment):
        with pytest.raises(ValueError, match=fragment):
            FrozenLimits.from_csv(text)


class TestInTheAnalysis:
    SPEC = SpecLimits(usl=74.05, lsl=73.95)

    def _run(self, values, **kw):
        kw.setdefault("chart_kind", ChartKind.XBAR_R)
        return run_analysis(AnalysisRequest(values=tuple(values), subgroup_size=5, spec=self.SPEC, **kw))

    def test_default_is_unchanged_and_exposes_its_limits(self, rings):
        out = self._run(rings)
        assert out.phase.mode == "all" and out.phase.n_baseline == 40
        assert out.limits.location_ucl == out.chart_pair.location.ucl
        assert not any("Phase I" in w for w in out.warnings)

    def test_baseline_then_new_data(self, rings):
        out = self._run(rings, n_baseline=25)
        assert out.phase.mode == "baseline" and out.phase.n_baseline == 25
        assert out.phase.signals_in_baseline == 0 and out.phase.signals_after_baseline > 0
        assert round(out.chart_pair.location.ucl, 3) == 74.014
        assert any("estimated from 25 subgroup(s) among the first 25 (Phase I) and applied to all 40" in w
                   for w in out.warnings)
        assert not out.stability.is_stable

    def test_exclusion_reason_reaches_the_warnings(self, rings):
        out = self._run(rings, n_baseline=25, excluded=((14, "new operator, first shift"), (3, "")))
        assert out.phase.n_baseline == 23
        assert any("Subgroup 14 left out of the control limits. Reason: new operator, first shift." in w
                   for w in out.warnings)
        assert any("Subgroup 3 left out" in w and "none recorded" in w for w in out.warnings)

    def test_unstable_baseline_is_called_out(self, rings):
        shifted = list(rings)
        for i in range(50, 55):  # subgroup 11 pushed far out
            shifted[i] += 0.05
        out = self._run(shifted, n_baseline=25)
        assert out.phase.signals_in_baseline >= 1
        assert any("baseline is not stable yet" in w for w in out.warnings)

    def test_export_then_import_reproduces_the_phase_two_chart(self, rings):
        first = self._run(rings[:125])
        frozen = FrozenLimits.from_csv(first.limits.to_csv())
        later = self._run(rings[125:], frozen_limits=frozen)
        assert later.phase.mode == "imported" and later.phase.signals_after_baseline > 0
        assert later.chart_pair.location.ucl == first.chart_pair.location.ucl
        assert any("imported (set from 25 subgroups)" in w for w in later.warnings)
        both = self._run(rings, n_baseline=25)
        assert np.array_equal(later.chart_pair.location.beyond_limits, both.chart_pair.location.beyond_limits[25:])

    def test_imported_limits_for_another_chart_are_an_error(self, rings):
        frozen = self._run(rings[:125]).limits
        out = self._run(rings, frozen_limits=frozen, chart_kind=ChartKind.XBAR_S)
        assert not out.ok and "xbar_r chart" in out.error
        out = run_analysis(AnalysisRequest(values=tuple(rings), chart_kind=ChartKind.XBAR_R, subgroup_size=4,
                                           frozen_limits=frozen))
        assert not out.ok and "subgroups of 5" in out.error

    def test_bad_baseline_is_an_error_not_an_exception(self, rings):
        assert "baseline" in self._run(rings, n_baseline=99).error
        assert "does not exist" in self._run(rings, excluded=((77, "x"),)).error

    def test_imr_baseline(self, rings):
        out = self._run(rings[:80], chart_kind=ChartKind.IMR, n_baseline=50, excluded=((7, "gauge reset"),))
        assert out.ok and out.phase.n_baseline == 49
        assert any("Reading 7 left out" in w for w in out.warnings)

    def test_signal_indices_still_point_at_source_rows(self, rings):
        """Frozen-limit charts go through the same original-row mapping as ordinary ones."""
        idx = tuple(range(10, 210))  # pretend 10 header/blank rows were dropped upstream
        out = self._run(rings, n_baseline=25, original_indices=idx)
        firsts = {v.indices[0] for v in out.location_eval.violations}
        assert firsts and all((i - 10) % 5 == 0 and i >= 10 + 25 * 5 for i in firsts)


class TestPage:
    """The capability page: Phase I controls, the Phase II summary and the limits download."""

    PAGE = str(ROOT / "app" / "ui" / "streamlit_app.py")

    def _app(self):
        from tests.test_streamlit_app import _run_with_data

        with open(ROOT / "examples" / "pistonrings.csv", newline="", encoding="utf-8") as fh:
            values = [float(r["measurement"]) for r in csv.DictReader(fh)]
        return _run_with_data(values, timeout=120.0)

    def test_default_run_has_no_phase_banner(self):
        at = self._app()
        assert not at.exception
        assert not any("Phase II view" in i.value for i in at.info)

    def test_baseline_period_shows_the_phase_two_summary(self):
        at = self._app()
        at.radio(key="limits_source").set_value("A baseline period (Phase I)")
        at.run()
        box = next(n for n in at.number_input if n.label.startswith("Baseline: first N"))
        assert int(box.value) == 40
        box.set_value(25)
        at.run()
        assert not at.exception
        banner = next(i.value for i in at.info if "Phase II view" in i.value)
        assert "limits from 25 baseline point(s)" in banner and "Signals inside the baseline: **0**" in banner
        assert any("Phase I" in w.value for w in at.warning)

    def test_methodology_row_describes_the_limits(self):
        from app.services.analysis import PhaseSummary

        assert PhaseSummary("all", 40, 40, (), 0, 0).describe() is None
        text = PhaseSummary("baseline", 40, 23, ((14, "new operator"), (3, " ")), 0, 5).describe()
        assert text == ("estimated from 23 baseline point(s), applied to all 40; left out of the baseline: "
                        "14 (new operator); 3 (no reason recorded)")
        assert "imported" in PhaseSummary("imported", 15, 0, (), 0, 4).describe()
