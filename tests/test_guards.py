"""Tests for reliability guards (app.capability.guards)."""

from __future__ import annotations

import numpy as np
import pytest

from app.capability.guards import (
    GuardThresholds,
    Severity,
    assess_normality,
    evaluate_guards,
)


class TestNormality:
    def test_normal_data_not_rejected(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=200)
        res = assess_normality(data)
        assert res.normal is True
        assert res.shapiro_p is not None and res.shapiro_p >= 0.05

    def test_clearly_nonnormal_rejected(self) -> None:
        rng = np.random.default_rng(7)
        # Heavily right-skewed exponential data.
        data = rng.exponential(1.0, size=200)
        res = assess_normality(data)
        assert res.normal is False
        assert "reject" in res.note.lower()

    def test_too_few_points(self) -> None:
        res = assess_normality([1.0, 2.0])
        assert res.normal is None
        assert "Too few" in res.note

    def test_zero_variance_untestable(self) -> None:
        res = assess_normality([5.0, 5.0, 5.0, 5.0])
        assert res.normal is None
        assert "Zero variance" in res.note

    def test_anderson_reported(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 1.0, size=100)
        res = assess_normality(data)
        assert res.anderson_stat is not None
        assert res.anderson_crit_5pct is not None


class TestGuards:
    def test_small_sample_warns(self) -> None:
        ws = evaluate_guards(
            n_observations=20, n_subgroups=None, sigma=1.0
        )
        codes = {w.code for w in ws}
        assert "small_sample" in codes

    def test_no_small_sample_at_threshold(self) -> None:
        ws = evaluate_guards(
            n_observations=30, n_subgroups=None, sigma=1.0, stability_state=True
        )
        codes = {w.code for w in ws}
        assert "small_sample" not in codes

    def test_few_subgroups_warns(self) -> None:
        ws = evaluate_guards(n_observations=100, n_subgroups=10, sigma=1.0)
        codes = {w.code for w in ws}
        assert "few_subgroups" in codes

    def test_few_subgroups_skipped_when_none(self) -> None:
        ws = evaluate_guards(n_observations=100, n_subgroups=None, sigma=1.0)
        codes = {w.code for w in ws}
        assert "few_subgroups" not in codes

    def test_zero_variation_warns(self) -> None:
        ws = evaluate_guards(n_observations=100, n_subgroups=30, sigma=0.0)
        codes = {w.code for w in ws}
        assert "zero_variation" in codes

    def test_stability_not_verified_info(self) -> None:
        ws = evaluate_guards(n_observations=100, n_subgroups=30, sigma=1.0)
        stab = [w for w in ws if w.code == "stability_not_verified"]
        assert stab and stab[0].severity is Severity.INFO

    def test_stability_verified_suppresses_note(self) -> None:
        ws = evaluate_guards(
            n_observations=100, n_subgroups=30, sigma=1.0, stability_state=True
        )
        codes = {w.code for w in ws}
        assert "stability_not_verified" not in codes
        assert "unstable_process" not in codes

    def test_unstable_state_warns(self) -> None:
        ws = evaluate_guards(
            n_observations=100,
            n_subgroups=30,
            sigma=1.0,
            stability_state=False,
            stability_detail="rules: N1",
        )
        unstable = [w for w in ws if w.code == "unstable_process"]
        assert unstable and unstable[0].severity is Severity.WARNING
        assert "N1" in unstable[0].message

    def test_custom_thresholds(self) -> None:
        th = GuardThresholds(min_observations=50, min_subgroups=25)
        ws = evaluate_guards(
            n_observations=40, n_subgroups=30, sigma=1.0, thresholds=th
        )
        codes = {w.code for w in ws}
        assert "small_sample" in codes  # 40 < 50
        assert "few_subgroups" not in codes  # 30 >= 25

    def test_all_guards_are_advisory_not_exceptions(self) -> None:
        # Even the worst case returns warnings, never raises.
        ws = evaluate_guards(n_observations=1, n_subgroups=1, sigma=0.0)
        assert isinstance(ws, list)
        assert all(w.severity in (Severity.INFO, Severity.WARNING) for w in ws)
