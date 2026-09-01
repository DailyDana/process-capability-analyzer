"""Tests for within/overall sigma estimation (app.statistics.variation).

Expected values are the independently hand-computed constants in
tests.fixtures.reference_data (derivations shown there), never produced by the
implementation under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.statistics.variation import (
    AUTO_S_SWITCH_N,
    WithinEstimator,
    grand_mean,
    moving_ranges,
    sigma_overall,
    sigma_within,
    sigma_within_from_individuals,
    sigma_within_from_subgroups,
)
from tests.fixtures import reference_data as ref


class TestRbarD2:
    def test_default_rbar_on_small_n(self) -> None:
        res = sigma_within_from_subgroups(ref.DRIFT_SUBGROUPS)
        assert res.estimator is WithinEstimator.RBAR_D2  # n=3 < 9 -> R-bar/d2
        assert res.sigma == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)
        assert res.detail["R_bar"] == pytest.approx(ref.DRIFT_RBAR)
        assert res.n_subgroups == 3
        assert res.subgroup_size == 3

    def test_forced_rbar(self) -> None:
        res = sigma_within_from_subgroups(
            ref.DRIFT_SUBGROUPS, estimator=WithinEstimator.RBAR_D2
        )
        assert res.sigma == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)


class TestSbarC4:
    def test_forced_sbar(self) -> None:
        res = sigma_within_from_subgroups(
            ref.DRIFT_SUBGROUPS, estimator=WithinEstimator.SBAR_C4
        )
        assert res.estimator is WithinEstimator.SBAR_C4
        assert res.sigma == pytest.approx(ref.DRIFT_SIGMA_WITHIN_SBAR, rel=1e-9)
        assert res.detail["S_bar"] == pytest.approx(ref.DRIFT_SBAR)

    def test_auto_switch_to_sbar_at_n9(self) -> None:
        # Build 3 balanced subgroups of size 9 -> AUTO must pick S-bar/c4.
        rng = np.arange(9, dtype=float)
        groups = [rng, rng + 10.0, rng + 20.0]
        res = sigma_within_from_subgroups(groups)  # AUTO
        assert AUTO_S_SWITCH_N == 9
        assert res.estimator is WithinEstimator.SBAR_C4

    def test_auto_stays_rbar_below_switch(self) -> None:
        rng = np.arange(8, dtype=float)  # n=8 < 9
        groups = [rng, rng + 10.0]
        res = sigma_within_from_subgroups(groups)
        assert res.estimator is WithinEstimator.RBAR_D2


class TestPooled:
    def test_pooled_unbalanced(self) -> None:
        res = sigma_within_from_subgroups(
            ref.POOLED_SUBGROUPS, estimator=WithinEstimator.POOLED
        )
        assert res.estimator is WithinEstimator.POOLED
        assert res.detail["s_pooled"] == pytest.approx(ref.POOLED_S, rel=1e-9)
        assert res.detail["pooled_dof"] == pytest.approx(3.0)
        assert res.sigma == pytest.approx(ref.POOLED_SIGMA, rel=1e-4)

    def test_auto_uses_pooled_when_unbalanced(self) -> None:
        res = sigma_within_from_subgroups(ref.POOLED_SUBGROUPS)  # AUTO
        assert res.estimator is WithinEstimator.POOLED

    def test_rbar_rejects_unbalanced(self) -> None:
        with pytest.raises(ValueError, match="equal subgroup sizes"):
            sigma_within_from_subgroups(
                ref.POOLED_SUBGROUPS, estimator=WithinEstimator.RBAR_D2
            )

    def test_rbar_rejects_n1(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            sigma_within_from_subgroups(
                [np.array([1.0]), np.array([2.0])],
                estimator=WithinEstimator.RBAR_D2,
            )


class TestMovingRange:
    def test_moving_ranges_values(self) -> None:
        mrs = moving_ranges(ref.MR_VALUES)
        assert list(mrs) == [2.0, 1.0, 3.0]

    def test_individuals_sigma(self) -> None:
        res = sigma_within_from_individuals(ref.MR_VALUES)
        assert res.estimator is WithinEstimator.MR_D2
        assert res.detail["MR_bar"] == pytest.approx(ref.MR_BAR)
        assert res.sigma == pytest.approx(ref.MR_SIGMA, rel=1e-4)
        assert res.subgroup_size == 2
        assert res.n_subgroups == 3  # three MR pairs

    def test_moving_ranges_need_two_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            moving_ranges([5.0])


class TestOverallSigma:
    def test_overall_sigma_value(self) -> None:
        assert sigma_overall(ref.DRIFT_ALL) == pytest.approx(
            ref.DRIFT_SIGMA_OVERALL, rel=1e-9
        )

    def test_grand_mean(self) -> None:
        assert grand_mean(ref.DRIFT_ALL) == pytest.approx(ref.DRIFT_MEAN)

    def test_overall_requires_two_points(self) -> None:
        with pytest.raises(ValueError, match="requires more than"):
            sigma_overall([5.0])


class TestWithinNotEqualOverall:
    """Requirement 1: within sigma must differ from overall on drifting data."""

    def test_within_much_smaller_than_overall(self) -> None:
        within = sigma_within_from_subgroups(ref.DRIFT_SUBGROUPS).sigma
        overall = sigma_overall(ref.DRIFT_ALL)
        # within ~1.18, overall ~8.70: overall is far larger due to drift.
        assert within == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)
        assert overall == pytest.approx(ref.DRIFT_SIGMA_OVERALL, rel=1e-9)
        assert overall > 5.0 * within


class TestDispatch:
    def test_dispatch_subgroups(self) -> None:
        res = sigma_within(subgroups=ref.DRIFT_SUBGROUPS)
        assert res.sigma == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)

    def test_dispatch_individuals(self) -> None:
        res = sigma_within(individuals=ref.MR_VALUES)
        assert res.sigma == pytest.approx(ref.MR_SIGMA, rel=1e-4)

    def test_dispatch_requires_exactly_one(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            sigma_within()
        with pytest.raises(ValueError, match="exactly one"):
            sigma_within(subgroups=ref.DRIFT_SUBGROUPS, individuals=ref.MR_VALUES)
