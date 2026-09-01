"""Tests for domain models (app.data.models)."""

from __future__ import annotations

import numpy as np
import pytest

from app.data.models import DataRegime, Dataset, Measurement, SpecLimits


class TestSpecLimits:
    def test_two_sided_ok(self) -> None:
        s = SpecLimits(usl=10.5, lsl=9.5, nominal=10.0, unit="mm")
        assert s.is_two_sided
        assert s.tolerance_width == pytest.approx(1.0)
        assert s.implied_nominal() == pytest.approx(10.0)
        assert s.unit == "mm"

    def test_midpoint_nominal_when_unset(self) -> None:
        s = SpecLimits(usl=10.0, lsl=8.0)
        assert s.implied_nominal() == pytest.approx(9.0)

    def test_one_sided_upper(self) -> None:
        s = SpecLimits(usl=5.0)
        assert not s.is_two_sided
        assert s.tolerance_width is None
        assert s.implied_nominal() is None

    def test_one_sided_lower(self) -> None:
        s = SpecLimits(lsl=1.0)
        assert not s.is_two_sided

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SpecLimits()

    def test_usl_not_above_lsl_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            SpecLimits(usl=5.0, lsl=5.0)
        with pytest.raises(ValueError, match="strictly greater"):
            SpecLimits(usl=4.0, lsl=5.0)

    def test_nominal_outside_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="above USL"):
            SpecLimits(usl=10.0, lsl=8.0, nominal=11.0)
        with pytest.raises(ValueError, match="below LSL"):
            SpecLimits(usl=10.0, lsl=8.0, nominal=7.0)

    def test_non_finite_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            SpecLimits(usl=float("inf"), lsl=0.0)


class TestMeasurement:
    def test_minimal(self) -> None:
        m = Measurement(value=10.0)
        assert m.value == 10.0
        assert m.sample_id is None

    def test_with_metadata(self) -> None:
        m = Measurement(
            value=9.8,
            sample_id="P-001",
            machine="CNC-3",
            operator="A. Yilmaz",
            characteristic="bore_dia",
            subgroup_id="sg1",
        )
        assert m.machine == "CNC-3"
        assert m.subgroup_id == "sg1"

    def test_non_finite_value_raises(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            Measurement(value=float("nan"))
        with pytest.raises(ValueError, match="finite number"):
            Measurement(value=float("inf"))


class TestDataset:
    def test_basic_unlabeled(self) -> None:
        ds = Dataset(values=np.array([1.0, 2.0, 3.0]))
        assert ds.n == 3
        assert ds.subgroup_sizes() == [3]
        assert ds.is_balanced()

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            Dataset(values=np.array([]))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Dataset(values=np.array([1.0, np.nan]))

    def test_subgroups_grouped_in_order(self) -> None:
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array(["a", "a", "b", "b"], dtype=object)
        ds = Dataset(values=vals, subgroup_labels=labels)
        groups = ds.subgroups()
        assert len(groups) == 2
        assert list(groups[0]) == [1.0, 2.0]
        assert list(groups[1]) == [3.0, 4.0]
        assert ds.subgroup_sizes() == [2, 2]
        assert ds.is_balanced()

    def test_label_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            Dataset(
                values=np.array([1.0, 2.0, 3.0]),
                subgroup_labels=np.array(["a", "b"], dtype=object),
            )

    def test_unbalanced_detected(self) -> None:
        vals = np.array([1.0, 2.0, 3.0])
        labels = np.array(["a", "a", "b"], dtype=object)
        ds = Dataset(values=vals, subgroup_labels=labels)
        assert ds.subgroup_sizes() == [2, 1]
        assert not ds.is_balanced()

    def test_from_measurements_infers_subgroups(self) -> None:
        ms = [
            Measurement(value=1.0, subgroup_id="s1", characteristic="len"),
            Measurement(value=2.0, subgroup_id="s1"),
            Measurement(value=3.0, subgroup_id="s2"),
            Measurement(value=4.0, subgroup_id="s2"),
        ]
        ds = Dataset.from_measurements(ms, spec=SpecLimits(usl=5.0, lsl=0.0, unit="mm"))
        assert ds.n == 4
        assert ds.subgroup_sizes() == [2, 2]
        assert ds.characteristic == "len"
        assert ds.unit == "mm"

    def test_from_measurements_without_subgroups(self) -> None:
        ms = [Measurement(value=float(i)) for i in range(5)]
        ds = Dataset.from_measurements(ms)
        assert ds.subgroup_labels is None
        assert ds.subgroup_sizes() == [5]

    def test_from_empty_measurements_raises(self) -> None:
        with pytest.raises(ValueError, match="empty measurement list"):
            Dataset.from_measurements([])


class TestDataRegime:
    def test_enum_values(self) -> None:
        assert DataRegime.SUBGROUPED.value == "subgrouped"
        assert DataRegime.INDIVIDUALS.value == "individuals"
        assert DataRegime.UNORDERED.value == "unordered"
