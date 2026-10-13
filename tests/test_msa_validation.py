"""Tests for long-format Gage R&R input validation (app.msa.validation)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.msa.validation import MAX_LISTED, build_gage_study


def long_format(p: int = 10, o: int = 3, r: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    parts, ops, trials, vals = [], [], [], []
    for i, j, k in itertools.product(range(p), range(o), range(r)):
        parts.append(f"P{i + 1}")
        ops.append(f"Op{j + 1}")
        trials.append(k + 1)
        vals.append(float(10 + i + rng.normal(0, 0.1)))
    return parts, ops, vals, trials


class TestValidStudies:
    def test_builds_balanced_array(self):
        parts, ops, vals, trials = long_format(10, 3, 3)
        v = build_gage_study(parts, ops, vals, trials, unit="mm")
        assert v.ok and v.errors == ()
        s = v.study
        assert (s.n_parts, s.n_operators, s.n_trials) == (10, 3, 3)
        assert s.part_labels[0] == "P1" and s.operator_labels == ("Op1", "Op2", "Op3")
        assert s.unit == "mm"
        assert s.values[0, 0, 0] == vals[0]

    def test_first_appearance_order_and_shuffled_rows(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        order = np.random.default_rng(1).permutation(len(vals))
        sh = [[col[i] for i in order] for col in (parts, ops, vals, trials)]
        v = build_gage_study(*sh)
        assert v.ok
        ref = build_gage_study(parts, ops, vals, trials).study
        # Same cell contents regardless of row order (labels may be reordered).
        for pi, pl in enumerate(v.study.part_labels):
            for oi, ol in enumerate(v.study.operator_labels):
                rp, ro = ref.part_labels.index(pl), ref.operator_labels.index(ol)
                assert list(v.study.values[pi, oi]) == list(ref.values[rp, ro])

    def test_trials_sorted_numerically(self):
        parts = ["A", "A", "A", "B", "B", "B"] * 2
        ops = ["x"] * 6 + ["y"] * 6
        trials = ["10", "2", "1"] * 4
        vals = [3.0, 2.0, 1.0] * 4
        v = build_gage_study(parts, ops, vals, trials)
        assert v.ok
        assert list(v.study.values[0, 0]) == [1.0, 2.0, 3.0]

    def test_without_trial_column(self):
        parts, ops, vals, _ = long_format(10, 3, 2)
        v = build_gage_study(parts, ops, vals)
        assert v.ok and v.study.n_trials == 2

    def test_numeric_strings_accepted(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        v = build_gage_study(parts, ops, [str(x) for x in vals], trials)
        assert v.ok

    def test_typical_study_has_no_warnings(self):
        v = build_gage_study(*long_format(10, 3, 3))
        assert v.warnings == ()


class TestWarnings:
    def test_few_parts_and_operators(self):
        v = build_gage_study(*long_format(5, 2, 3))
        assert v.ok
        text = " ".join(v.warnings)
        assert "Only 5 parts" in text and "Only 2 operators" in text

    def test_few_repeatability_df(self):
        v = build_gage_study(*long_format(3, 2, 2))
        assert any("degrees of freedom" in w for w in v.warnings)

    def test_zero_variation_warned(self):
        parts, ops, _, trials = long_format(10, 3, 2)
        v = build_gage_study(parts, ops, [5.0] * len(parts), trials)
        assert v.ok and any("identical" in w for w in v.warnings)


class TestErrors:
    def test_length_mismatch(self):
        v = build_gage_study(["a"], ["x", "y"], [1.0])
        assert not v.ok and "different lengths" in v.errors[0]

    def test_empty(self):
        v = build_gage_study([], [], [])
        assert not v.ok and "No measurements" in v.errors[0]

    def test_bad_values_are_errors_not_drops(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        vals[4] = "n/a"
        vals[7] = None
        vals[9] = float("inf")
        v = build_gage_study(parts, ops, vals, trials)
        assert not v.ok and v.study is None
        assert "row 5" in v.errors[0] and "row 8" in v.errors[0] and "row 10" in v.errors[0]
        assert "balanced" in v.errors[0]

    def test_blank_labels(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        parts[2] = "  "
        ops[3] = float("nan")
        v = build_gage_study(parts, ops, vals, trials)
        assert not v.ok
        assert any("Missing part label: row 3" in e for e in v.errors)
        assert any("Missing operator label: row 4" in e for e in v.errors)

    def test_duplicate_trial(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        trials[1] = 1  # P1/Op1 now has trial 1 twice
        v = build_gage_study(parts, ops, vals, trials)
        assert not v.ok and "Duplicated" in v.errors[0] and "rows 1 and 2" in v.errors[0]

    def test_missing_cell(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        keep = [i for i in range(len(parts)) if not (parts[i] == "P4" and ops[i] == "Op2")]
        cols = [[c[i] for i in keep] for c in (parts, ops, vals, trials)]
        v = build_gage_study(*cols)
        assert not v.ok and "part P4 x operator Op2" in v.errors[0] and "Crossed" in v.errors[0]

    def test_unbalanced(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        parts.append("P1"); ops.append("Op1"); vals.append(10.0); trials.append(3)  # noqa: E702
        v = build_gage_study(parts, ops, vals, trials)
        assert not v.ok and "Unbalanced" in v.errors[0] and "part P1 x operator Op1 (3)" in v.errors[0]

    def test_single_trial(self):
        v = build_gage_study(*long_format(10, 3, 1))
        assert not v.ok and "2 trials" in v.errors[0]

    def test_single_operator_and_part(self):
        v = build_gage_study(*long_format(1, 1, 2))
        assert not v.ok
        assert any("2 parts" in e for e in v.errors) and any("2 operators" in e for e in v.errors)

    def test_listing_is_capped(self):
        parts, ops, vals, trials = long_format(10, 3, 2)
        for i in range(20):
            vals[i] = "x"
        v = build_gage_study(parts, ops, vals, trials)
        assert f"and {20 - MAX_LISTED} more" in v.errors[0]
