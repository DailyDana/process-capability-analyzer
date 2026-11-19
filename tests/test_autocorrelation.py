"""Lag-1 autocorrelation: the statistic, the approximate test, and how it reaches the user."""

from __future__ import annotations

import numpy as np
import pytest

from app.data.models import SpecLimits
from app.interpret.autocorrelation import PRACTICAL_R1, autocorrelation_findings
from app.interpret.models import ConclusionTopic, Severity
from app.services.analysis import AnalysisRequest, ChartKind, run_analysis
from app.statistics.autocorrelation import MIN_OBSERVATIONS, lag1_autocorrelation


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return 10.0 + x


class TestStatistic:
    def test_hand_computed(self) -> None:
        # x = 1,2,3,4,5: deviations -2..2, numerator (4+2+0+2)... = 4, denominator 10
        assert lag1_autocorrelation([1, 2, 3, 4, 5]).r1 == pytest.approx(0.4)

    def test_alternating_series_is_strongly_negative(self) -> None:
        assert lag1_autocorrelation([1, -1] * 20).r1 == pytest.approx(-39 / 40)

    def test_invariant_to_shift_and_scale(self) -> None:
        x = _ar1(0.5, 200, 1)
        r = lag1_autocorrelation(x).r1
        assert lag1_autocorrelation(1e6 + 3.0 * x).r1 == pytest.approx(r, rel=1e-9)
        assert lag1_autocorrelation(-x).r1 == pytest.approx(r, rel=1e-12)

    def test_constant_series_has_no_coefficient(self) -> None:
        res = lag1_autocorrelation([5.0] * 30)
        assert res.r1 is None and res.significant is None

    def test_bound_and_p_value(self) -> None:
        res = lag1_autocorrelation(_ar1(0.0, 100, 2))
        assert res.bound == pytest.approx(1.959963984540054 / 10.0)
        assert res.z == pytest.approx(res.r1 * 10.0)
        assert (res.p_value < 0.05) == res.significant

    def test_too_few_observations_reports_r1_without_a_test(self) -> None:
        res = lag1_autocorrelation(_ar1(0.0, MIN_OBSERVATIONS - 1, 3))
        assert res.r1 is not None and res.significant is None and res.bound is None

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, float("nan"), 2.0, 3.0]])
    def test_rejects_unusable_input(self, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            lag1_autocorrelation(bad)

    def test_false_alarm_rate_of_the_approximate_test(self) -> None:
        """Independent normal series: about alpha of them should be flagged."""
        rng = np.random.default_rng(11)
        flagged = sum(bool(lag1_autocorrelation(rng.normal(size=100)).significant) for _ in range(4000))
        assert 0.03 < flagged / 4000 < 0.065

    def test_detects_an_ar1_process(self) -> None:
        res = lag1_autocorrelation(_ar1(0.6, 300, 4))
        assert res.significant and res.r1 == pytest.approx(0.6, abs=0.12)


class TestFindings:
    def test_nothing_when_not_significant(self) -> None:
        assert autocorrelation_findings(lag1_autocorrelation(_ar1(0.0, 100, 2))) == (None, None)
        assert autocorrelation_findings(None) == (None, None)

    def test_positive_is_a_warning_about_narrow_limits(self) -> None:
        c, r = autocorrelation_findings(lag1_autocorrelation(_ar1(0.7, 200, 5)))
        assert c.code == "autocorrelated_observations" and c.topic is ConclusionTopic.INDEPENDENCE
        assert c.severity is Severity.WARNING and "narrower" in c.text
        assert r.code == "rec_address_autocorrelation" and r.trigger_code == c.code

    def test_negative_says_limits_are_too_wide(self) -> None:
        c, r = autocorrelation_findings(lag1_autocorrelation([1.0, -1.0] * 30))
        assert "wider" in c.text and "over-adjustment" in r.text

    def test_weak_but_significant_is_only_a_notice(self) -> None:
        x = _ar1(0.12, 4000, 6)
        res = lag1_autocorrelation(x)
        assert res.significant and abs(res.r1) < PRACTICAL_R1
        assert autocorrelation_findings(res)[0].severity is Severity.NOTICE


class TestInTheAnalysis:
    SPEC = SpecLimits(usl=20.0, lsl=0.0)

    def _run(self, x, **kw):
        return run_analysis(AnalysisRequest(values=tuple(float(v) for v in x), **kw))

    def test_imr_with_spec_gets_conclusion_and_recommendation(self) -> None:
        out = self._run(_ar1(0.7, 200, 5), chart_kind=ChartKind.IMR, spec=self.SPEC)
        assert out.autocorrelation.significant
        assert "autocorrelated_observations" in out.interpretation.conclusion_codes
        assert "rec_address_autocorrelation" in out.interpretation.recommendation_codes

    def test_imr_without_spec_still_warns(self) -> None:
        out = self._run(_ar1(0.7, 200, 5), chart_kind=ChartKind.IMR)
        assert out.interpretation is None
        assert any("autocorrelated" in w for w in out.warnings)

    def test_numbers_are_unchanged_by_the_check(self) -> None:
        out = self._run(_ar1(0.7, 200, 5), chart_kind=ChartKind.IMR, spec=self.SPEC)
        assert out.capability.capability.cpk is not None and out.chart_pair is not None

    def test_independent_series_adds_nothing(self) -> None:
        out = self._run(_ar1(0.0, 100, 2), chart_kind=ChartKind.IMR, spec=self.SPEC)
        assert out.autocorrelation.significant is False
        assert "autocorrelated_observations" not in out.interpretation.conclusion_codes

    def test_not_run_for_subgroups_or_unordered_data(self) -> None:
        x = _ar1(0.7, 200, 5)
        assert self._run(x, chart_kind=ChartKind.XBAR_R, subgroup_size=5, spec=self.SPEC).autocorrelation is None
        assert self._run(x, chart_kind=ChartKind.IMR, spec=self.SPEC, time_ordered=False).autocorrelation is None
