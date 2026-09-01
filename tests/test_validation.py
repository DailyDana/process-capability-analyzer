"""Tests for input validation / cleaning (app.data.validation)."""

from __future__ import annotations

import numpy as np
import pytest

from app.data.validation import (
    RowStatus,
    clean_numeric,
    clean_numeric_detailed,
    validate_spec_limits,
)


class TestCleanNumeric:
    def test_all_clean(self) -> None:
        vals, rep = clean_numeric([1.0, 2.0, 3.0])
        assert list(vals) == [1.0, 2.0, 3.0]
        assert rep.is_clean
        assert rep.n_valid == 3
        assert rep.n_dropped == 0

    def test_numeric_strings_parsed(self) -> None:
        vals, rep = clean_numeric(["1.5", "2.5", "3"])
        assert list(vals) == [1.5, 2.5, 3.0]
        assert rep.is_clean

    def test_missing_tokens_dropped(self) -> None:
        vals, rep = clean_numeric([1.0, "", "N/A", None, np.nan, "  ", 2.0])
        assert list(vals) == [1.0, 2.0]
        assert rep.dropped_missing == 5
        assert rep.n_valid == 2

    def test_non_numeric_dropped_by_default(self) -> None:
        vals, rep = clean_numeric([1.0, "bad", "12x", 3.0])
        assert list(vals) == [1.0, 3.0]
        assert rep.dropped_non_numeric == 2
        assert "'bad'" in rep.invalid_examples or "'12x'" in rep.invalid_examples

    def test_non_numeric_raises_when_requested(self) -> None:
        with pytest.raises(ValueError, match="Non-numeric"):
            clean_numeric([1.0, "bad", 3.0], on_invalid="raise")

    def test_inf_dropped_as_non_finite(self) -> None:
        vals, rep = clean_numeric([1.0, float("inf"), 2.0, float("-inf")])
        assert list(vals) == [1.0, 2.0]
        assert rep.dropped_non_finite == 2

    def test_duplicates_counted_not_removed(self) -> None:
        vals, rep = clean_numeric([5.0, 5.0, 5.0, 7.0])
        assert list(vals) == [5.0, 5.0, 5.0, 7.0]  # kept
        assert rep.n_duplicates == 2  # two repeats of 5.0
        assert rep.n_valid == 4

    def test_empty_input(self) -> None:
        vals, rep = clean_numeric([])
        assert vals.size == 0
        assert rep.n_input == 0
        assert rep.n_valid == 0

    def test_all_invalid(self) -> None:
        vals, rep = clean_numeric(["a", "b", None])
        assert vals.size == 0
        assert rep.dropped_non_numeric == 2
        assert rep.dropped_missing == 1

    def test_summary_string(self) -> None:
        _, rep = clean_numeric([1.0, "", "bad", 5.0, 5.0])
        s = rep.summary()
        assert "usable" in s
        assert "missing" in s
        assert "non-numeric" in s

    def test_precision_preserved(self) -> None:
        vals, _ = clean_numeric(["10.123456789012345"])
        assert vals[0] == pytest.approx(10.123456789012345, abs=1e-15)


class TestDetailedCleaning:
    def test_per_row_records_preserve_original_indices(self) -> None:
        raw = [10.0, "bad", None, 10.5, float("inf"), 9.8]
        det = clean_numeric_detailed(raw)
        assert len(det.records) == 6  # one record per INPUT row
        assert det.used_indices == (0, 3, 5)
        assert list(det.values) == [10.0, 10.5, 9.8]

    def test_statuses_are_specific(self) -> None:
        det = clean_numeric_detailed([10.0, "bad", None, float("inf")])
        statuses = [r.status for r in det.records]
        assert statuses == [
            RowStatus.OK, RowStatus.NON_NUMERIC, RowStatus.MISSING,
            RowStatus.NON_FINITE,
        ]

    def test_excluded_rows_carry_reasons(self) -> None:
        det = clean_numeric_detailed([10.0, "oops", None])
        excluded = det.excluded
        assert len(excluded) == 2
        assert all(r.reason for r in excluded)
        assert any("number" in r.reason for r in excluded)

    def test_report_matches_wrapper(self) -> None:
        raw = [1.0, "x", None, 2.0, 2.0]
        det = clean_numeric_detailed(raw)
        values, report = clean_numeric(raw)
        assert list(det.values) == list(values)
        assert det.report == report

    def test_ok_rows_expose_parsed_value(self) -> None:
        det = clean_numeric_detailed(["10.5", 3])
        assert [r.value for r in det.records] == [10.5, 3.0]
        assert all(r.is_used for r in det.records)


class TestValidateSpecLimits:
    def test_valid_two_sided(self) -> None:
        r = validate_spec_limits(usl=10.0, lsl=8.0, nominal=9.0)
        assert r.ok
        assert not r.errors

    def test_valid_one_sided(self) -> None:
        assert validate_spec_limits(usl=10.0, lsl=None).ok
        assert validate_spec_limits(usl=None, lsl=1.0).ok

    def test_both_none(self) -> None:
        r = validate_spec_limits(usl=None, lsl=None)
        assert not r.ok
        assert any("at least one" in e.lower() for e in r.errors)

    def test_reversed_limits(self) -> None:
        r = validate_spec_limits(usl=8.0, lsl=10.0)
        assert not r.ok
        assert any("below LSL" in e for e in r.errors)

    def test_equal_limits(self) -> None:
        r = validate_spec_limits(usl=5.0, lsl=5.0)
        assert not r.ok
        assert any("zero" in e.lower() for e in r.errors)

    def test_nominal_out_of_range(self) -> None:
        assert not validate_spec_limits(usl=10.0, lsl=8.0, nominal=11.0).ok
        assert not validate_spec_limits(usl=10.0, lsl=8.0, nominal=7.0).ok

    def test_nominal_on_bound_warns_but_ok(self) -> None:
        r = validate_spec_limits(usl=10.0, lsl=8.0, nominal=10.0)
        assert r.ok
        assert r.warnings

    def test_non_finite_usl(self) -> None:
        r = validate_spec_limits(usl=float("inf"), lsl=0.0)
        assert not r.ok
        assert any("finite" in e for e in r.errors)

    def test_nan_lsl(self) -> None:
        r = validate_spec_limits(usl=10.0, lsl=float("nan"))
        assert not r.ok
