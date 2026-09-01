"""Tests for descriptive statistics (app.statistics.descriptive).

Known-answer datasets are used throughout. Where a value is hand-checkable it is
written out in the test so the expectation is auditable, not opaque.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.statistics import descriptive as d


# ---------------------------------------------------------------------------
# Known-answer basics
# ---------------------------------------------------------------------------
class TestKnownAnswers:
    def test_mean_simple(self) -> None:
        # mean of 1..5 = 3
        assert d.mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_median_odd(self) -> None:
        assert d.median([3, 1, 2, 5, 4]) == pytest.approx(3.0)

    def test_median_even_interpolates(self) -> None:
        # median of 1,2,3,4 = (2+3)/2 = 2.5
        assert d.median([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_sample_variance_bessel(self) -> None:
        # data 2,4,4,4,5,5,7,9: mean=5, sum sq dev=32, sample var=32/7
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        assert d.variance(data) == pytest.approx(32.0 / 7.0)

    def test_population_variance(self) -> None:
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        # population var = 32/8 = 4 -> std = 2
        assert d.variance(data, ddof=0) == pytest.approx(4.0)
        assert d.std_dev(data, ddof=0) == pytest.approx(2.0)

    def test_std_dev_is_sqrt_variance(self) -> None:
        data = [10.1, 10.3, 9.8, 10.0, 10.2]
        assert d.std_dev(data) == pytest.approx(math.sqrt(d.variance(data)))

    def test_min_max_range(self) -> None:
        data = [3.5, 1.2, 8.9, 4.4]
        assert d.minimum(data) == pytest.approx(1.2)
        assert d.maximum(data) == pytest.approx(8.9)
        assert d.value_range(data) == pytest.approx(8.9 - 1.2)


# ---------------------------------------------------------------------------
# NaN / missing / non-finite policy
# ---------------------------------------------------------------------------
class TestNanPolicy:
    def test_raise_on_nan_by_default(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            d.mean([1.0, 2.0, np.nan])

    def test_omit_nan_when_requested(self) -> None:
        assert d.mean([1.0, 2.0, np.nan], nan_policy="omit") == pytest.approx(1.5)

    def test_inf_always_raises(self) -> None:
        with pytest.raises(ValueError, match="inf"):
            d.mean([1.0, 2.0, np.inf], nan_policy="omit")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no values"):
            d.mean([])

    def test_all_nan_with_omit_raises(self) -> None:
        with pytest.raises(ValueError, match="All values were NaN"):
            d.mean([np.nan, np.nan], nan_policy="omit")


# ---------------------------------------------------------------------------
# Edge cases: tiny samples & zero variation
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_variance_single_value_raises(self) -> None:
        with pytest.raises(ValueError, match="requires more than"):
            d.variance([42.0])  # ddof=1 needs n>1

    def test_population_variance_single_value_ok(self) -> None:
        assert d.variance([42.0], ddof=0) == pytest.approx(0.0)

    def test_zero_variation(self) -> None:
        data = [10.0, 10.0, 10.0, 10.0]
        assert d.variance(data) == pytest.approx(0.0)
        assert d.std_dev(data) == pytest.approx(0.0)
        assert d.value_range(data) == pytest.approx(0.0)

    def test_duplicate_values_are_kept(self) -> None:
        # duplicates are legitimate; mean/variance must reflect all of them
        data = [5.0, 5.0, 5.0, 11.0]
        assert d.mean(data) == pytest.approx(6.5)
        assert d.count(data) == 4


# ---------------------------------------------------------------------------
# describe() convenience aggregator
# ---------------------------------------------------------------------------
class TestDescribe:
    def test_describe_full(self) -> None:
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        s = d.describe(data)
        assert s.n == 8
        assert s.mean == pytest.approx(5.0)
        assert s.variance == pytest.approx(32.0 / 7.0)
        assert s.minimum == pytest.approx(2.0)
        assert s.maximum == pytest.approx(9.0)
        assert s.value_range == pytest.approx(7.0)
        assert s.ddof == 1

    def test_describe_omits_nan_and_counts_remaining(self) -> None:
        data = [1.0, 2.0, np.nan, 4.0]
        s = d.describe(data)
        assert s.n == 3  # NaN dropped
        assert s.mean == pytest.approx(7.0 / 3.0)

    def test_describe_tiny_sample_sets_nan_std_not_raise(self) -> None:
        s = d.describe([42.0])  # n=1, ddof=1 -> std undefined
        assert s.n == 1
        assert math.isnan(s.std_dev)
        assert math.isnan(s.variance)
        assert s.mean == pytest.approx(42.0)

    def test_describe_as_dict_roundtrip(self) -> None:
        s = d.describe([1.0, 2.0, 3.0])
        dd = s.as_dict()
        assert dd["n"] == 3
        assert dd["mean"] == pytest.approx(2.0)
        assert "range" in dd and dd["range"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Precision preservation (no internal rounding)
# ---------------------------------------------------------------------------
class TestPrecision:
    def test_no_internal_rounding(self) -> None:
        data = [1.0000000001, 1.0000000002, 1.0000000003]
        # if the code rounded, these would collapse; the spread must survive
        assert d.std_dev(data) > 0.0
        assert d.mean(data) == pytest.approx(1.0000000002, abs=1e-15)
