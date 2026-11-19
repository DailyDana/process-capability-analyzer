"""Regression tests for defects found in the v0.4.0 code review.

Each service promises ``ok=False`` with a plain message for ordinary user
input; these are the inputs that used to escape as exceptions or as NaN.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

import app
from app.capability.guards import NormalityResult, evaluate_guards
from app.data.models import SpecLimits
from app.services.analysis import AnalysisRequest, ChartKind, run_analysis
from app.services.nonnormal import BOX_COX, NonNormalRequest, run_nonnormal
from app.statistics.variation import WithinEstimator


class TestNonNormalDegenerateInput:
    @pytest.mark.parametrize("method", ["best_fit", BOX_COX, "normal", "weibull"])
    def test_constant_data_is_a_user_error_not_a_crash(self, method: str) -> None:
        out = run_nonnormal(NonNormalRequest(values=tuple([5.0] * 20), usl=6.0, method=method))
        assert not out.ok
        assert "variation" in out.errors[0].lower()

    def test_box_cox_rejects_a_non_positive_usl(self) -> None:
        x = tuple(np.random.default_rng(1).lognormal(0.0, 0.5, 50))
        out = run_nonnormal(NonNormalRequest(values=x, usl=-1.0, method=BOX_COX))
        assert not out.ok
        assert "USL" in out.errors[0]

    def test_a_successful_result_is_never_nan(self) -> None:
        x = tuple(np.random.default_rng(2).lognormal(0.0, 0.5, 80))
        for method in ("best_fit", BOX_COX, "lognormal", "gamma", "weibull", "exponential", "normal"):
            out = run_nonnormal(NonNormalRequest(values=x, usl=6.0, method=method))
            assert out.ok and out.summary is not None
            assert math.isfinite(out.summary.ppk) and math.isfinite(out.summary.ppm_total)


class TestAnalysisService:
    def test_individuals_estimator_on_subgroups_is_reported_not_raised(self) -> None:
        x = tuple(float(v) for v in np.random.default_rng(3).normal(10, 1, 50))
        out = run_analysis(AnalysisRequest(values=x, chart_kind=ChartKind.XBAR_R, subgroup_size=5,
                                           spec=SpecLimits(usl=13, lsl=7),
                                           within_estimator=WithinEstimator.MR_D2))
        assert not out.ok
        assert "individuals" in out.error.lower()

    def test_descriptive_summary_covers_the_analysed_measurements(self) -> None:
        """Trailing readings that do not fill a subgroup are excluded everywhere."""
        x = tuple(float(v) for v in np.random.default_rng(7).normal(10, 1, 103))
        out = run_analysis(AnalysisRequest(values=x, chart_kind=ChartKind.XBAR_R, subgroup_size=5,
                                           spec=SpecLimits(usl=13, lsl=7)))
        assert out.ok and out.n_used == 100
        assert out.descriptive.n == 100
        assert out.descriptive.mean == pytest.approx(out.capability.performance.mean, rel=1e-12)


def test_non_normal_guard_points_to_the_non_normal_page() -> None:
    rejected = NormalityResult(False, 0.9, 0.001, 1.5, 0.75, 50)
    ws = evaluate_guards(n_observations=50, n_subgroups=None, sigma=1.0, normality=rejected,
                         stability_state=True)
    msg = next(w.message for w in ws if w.code == "non_normal")
    assert "not implemented" not in msg
    assert "Capability, non-normal" in msg


def test_package_version_matches_pyproject() -> None:
    text = (Path(app.__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    assert declared is not None
    assert app.__version__ == declared.group(1)
