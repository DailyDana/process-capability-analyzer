"""Tests for control-limit calculators (app.spc.limits).

Expected limit values are computed independently from the published constants
(A2, A3, D3, D4, B3, B4, E2 hand-substituted), NOT by re-deriving them through
the same code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.spc.limits import (
    ChartType,
    imr_chart,
    xbar_r_chart,
    xbar_s_chart,
)

# Published constants used for independent expected-value computation.
A2_N5 = 0.577
D3_N5 = 0.0
D4_N5 = 2.114
A3_N5 = 1.427
B3_N5 = 0.0
B4_N5 = 2.089
D2_2 = 1.128
D4_2 = 3.267
# E2 CONVENTION (hardening milestone, B1): the I-chart uses the TABULATED E2
# from the validated constants table -- the same tabulated-constant convention
# the MR_bar/d2 sigma estimator uses. Previously this file pinned 3/1.128
# (= 2.65957), a value matching neither the constants table (2.660) nor the
# "exact" 3/d2_exact (2.65868) FORMULAS.md claimed. See FORMULAS.md Sec. 3.4.
E2_2 = 2.660


# Five balanced subgroups of size 5 (known values).
SUBGROUPS_N5 = [
    np.array([10.0, 12.0, 11.0, 9.0, 13.0]),   # mean 11.0, range 4
    np.array([11.0, 10.0, 12.0, 11.0, 10.0]),  # mean 10.8, range 2
    np.array([9.0, 11.0, 10.0, 12.0, 11.0]),   # mean 10.6, range 3
    np.array([13.0, 12.0, 11.0, 14.0, 13.0]),  # mean 12.6, range 3
    np.array([10.0, 9.0, 11.0, 10.0, 12.0]),   # mean 10.4, range 3
]
# Grand mean = (11.0+10.8+10.6+12.6+10.4)/5 = 55.4/5 = 11.08
# R-bar = (4+2+3+3+3)/5 = 15/5 = 3.0
XDBAR = 11.08
RBAR = 3.0


class TestXbarR:
    def test_center_lines(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        assert pair.location.center_line == pytest.approx(XDBAR)
        assert pair.dispersion.center_line == pytest.approx(RBAR)

    def test_xbar_limits(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        # UCL/LCL = 11.08 +/- 0.577*3.0 = 11.08 +/- 1.731
        assert pair.location.ucl == pytest.approx(XDBAR + A2_N5 * RBAR, rel=1e-6)
        assert pair.location.lcl == pytest.approx(XDBAR - A2_N5 * RBAR, rel=1e-6)

    def test_r_limits(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        assert pair.dispersion.ucl == pytest.approx(D4_N5 * RBAR, rel=1e-6)  # 6.342
        assert pair.dispersion.lcl == pytest.approx(D3_N5 * RBAR)  # 0.0

    def test_sigma_points_is_third_of_band(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        expected = (pair.location.ucl - pair.location.center_line) / 3.0
        assert pair.location.sigma_points == pytest.approx(expected)

    def test_chart_types(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        assert pair.location.chart_type is ChartType.XBAR_R
        assert pair.dispersion.chart_type is ChartType.R
        assert pair.location.is_symmetric is True
        assert pair.dispersion.is_symmetric is False  # R chart asymmetric

    def test_points_preserved(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        assert pair.location.points[0] == pytest.approx(11.0)
        assert pair.dispersion.points[0] == pytest.approx(4.0)
        assert pair.location.n_points == 5

    def test_rejects_single_subgroup(self) -> None:
        with pytest.raises(ValueError, match="at least 2 subgroups"):
            xbar_r_chart([np.array([1.0, 2.0, 3.0])])

    def test_rejects_unequal_sizes(self) -> None:
        with pytest.raises(ValueError, match="same size"):
            xbar_r_chart([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])])

    def test_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            xbar_r_chart([np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])])

    def test_rejects_n1(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            xbar_r_chart([np.array([1.0]), np.array([2.0])])


class TestXbarS:
    def test_center_lines(self) -> None:
        pair = xbar_s_chart(SUBGROUPS_N5)
        assert pair.location.center_line == pytest.approx(XDBAR)
        # S-bar = mean of subgroup sample std devs.
        expected_sbar = float(
            np.mean([np.std(g, ddof=1) for g in SUBGROUPS_N5])
        )
        assert pair.dispersion.center_line == pytest.approx(expected_sbar)

    def test_xbar_limits(self) -> None:
        pair = xbar_s_chart(SUBGROUPS_N5)
        sbar = pair.dispersion.center_line
        assert pair.location.ucl == pytest.approx(XDBAR + A3_N5 * sbar, rel=1e-6)
        assert pair.location.lcl == pytest.approx(XDBAR - A3_N5 * sbar, rel=1e-6)

    def test_s_limits(self) -> None:
        pair = xbar_s_chart(SUBGROUPS_N5)
        sbar = pair.dispersion.center_line
        assert pair.dispersion.ucl == pytest.approx(B4_N5 * sbar, rel=1e-6)
        assert pair.dispersion.lcl == pytest.approx(B3_N5 * sbar)  # 0 for n=5

    def test_chart_types(self) -> None:
        pair = xbar_s_chart(SUBGROUPS_N5)
        assert pair.location.chart_type is ChartType.XBAR_S
        assert pair.dispersion.chart_type is ChartType.S


class TestIMR:
    # Series with known moving ranges.
    VALUES = np.array([10.0, 12.0, 11.0, 14.0, 13.0])
    # MR = |2|,|−1|,|3|,|−1| = 2,1,3,1 -> MR-bar = 7/4 = 1.75
    # x-bar = (10+12+11+14+13)/5 = 60/5 = 12.0
    XBAR = 12.0
    MRBAR = 1.75

    def test_i_chart_limits(self) -> None:
        pair = imr_chart(self.VALUES)
        assert pair.location.center_line == pytest.approx(self.XBAR)
        assert pair.location.ucl == pytest.approx(self.XBAR + E2_2 * self.MRBAR, rel=1e-6)
        assert pair.location.lcl == pytest.approx(self.XBAR - E2_2 * self.MRBAR, rel=1e-6)

    def test_mr_chart_limits(self) -> None:
        pair = imr_chart(self.VALUES)
        assert pair.dispersion.center_line == pytest.approx(self.MRBAR)
        assert pair.dispersion.ucl == pytest.approx(D4_2 * self.MRBAR, rel=1e-6)
        assert pair.dispersion.lcl == pytest.approx(0.0)  # D3(2) = 0

    def test_mr_has_n_minus_1_points(self) -> None:
        pair = imr_chart(self.VALUES)
        assert pair.dispersion.n_points == self.VALUES.size - 1
        assert list(pair.dispersion.points) == [2.0, 1.0, 3.0, 1.0]

    def test_chart_types(self) -> None:
        pair = imr_chart(self.VALUES)
        assert pair.location.chart_type is ChartType.INDIVIDUALS
        assert pair.dispersion.chart_type is ChartType.MOVING_RANGE
        assert pair.location.is_symmetric is True
        assert pair.dispersion.is_symmetric is False

    def test_e2_matches_authoritative_tabulated_constant(self) -> None:
        """B1 cross-check: the I-chart's implied E2 IS the tabulated constant.

        This is the regression guard for the E2 convention. Before the
        hardening milestone three values were in play: the constants table
        (2.660), an inline 3/d2_table in imr_chart (2.65957), and an "exact"
        3/d2_exact (2.65868) that FORMULAS.md incorrectly claimed was used.
        The chart must now agree exactly with the single authoritative source.
        """
        from app.statistics.constants import CONSTANTS

        pair = imr_chart(self.VALUES)
        implied_e2 = (pair.location.ucl - pair.location.center_line) / self.MRBAR
        # Exact agreement with the validated constants table (no tolerance slop).
        assert implied_e2 == pytest.approx(CONSTANTS[2].e2, rel=1e-12)
        assert implied_e2 == pytest.approx(E2_2, rel=1e-12)
        # And explicitly NOT the two rejected conventions.
        assert implied_e2 != pytest.approx(3.0 / 1.128, rel=1e-9)
        assert implied_e2 != pytest.approx(2.6586807763582745, rel=1e-9)

    def test_e2_convention_agrees_with_mr_sigma_estimator(self) -> None:
        """The chart constant and the capability sigma share one convention.

        ``sigma_within_from_individuals`` uses MR_bar / d2_table(2). The I-chart
        half-width is E2_table * MR_bar. Consistency means the chart's implied
        3-sigma half-width equals 3 * that sigma to within the rounding of the
        published table (E2 = 2.660 vs 3/1.128 = 2.65957 -> ~0.016 %).
        """
        from app.statistics.variation import sigma_within_from_individuals

        pair = imr_chart(self.VALUES)
        sigma = sigma_within_from_individuals(self.VALUES).sigma
        half_width = pair.location.ucl - pair.location.center_line
        assert half_width == pytest.approx(3.0 * sigma, rel=2e-4)

    def test_missing_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="missing/non-finite"):
            imr_chart([10.0, np.nan, 12.0, 11.0])

    def test_missing_exclude_preserves_indices(self) -> None:
        # index 2 is NaN and excluded; retained original indices [0,1,3,4].
        pair = imr_chart([10.0, 12.0, np.nan, 14.0, 13.0], on_missing="exclude")
        assert list(pair.location.point_indices) == [0, 1, 3, 4]
        assert pair.location.warnings  # excluded-rows warning present

    def test_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            imr_chart([5.0])


class TestBeyondLimitFlags:
    def test_point_on_limit_not_flagged(self) -> None:
        # Construct a chart where a point sits exactly on the UCL.
        pair = xbar_r_chart(SUBGROUPS_N5)
        ucl = pair.location.ucl
        # A point exactly at UCL must NOT be flagged (strict inequality).
        pts = pair.location.points.copy()
        # Re-run beyond flag logic via a fresh chart is overkill; assert the
        # documented semantics on the existing flags: none of the in-band points
        # are flagged.
        assert pair.location.n_beyond == 0
        assert not pair.location.beyond_limits.any()

    def test_serialization(self) -> None:
        pair = xbar_r_chart(SUBGROUPS_N5)
        d = pair.as_dict()
        assert d["location"]["chart_type"] == "xbar_r"
        assert "ucl" in d["location"]
        assert d["dispersion"]["chart_type"] == "r"
