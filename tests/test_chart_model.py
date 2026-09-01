"""Tests for the chart MODEL layer (app.charts.model).

The central guarantee tested here: the ChartModel is a faithful, verbatim view
of the M3 ControlChart / RuleEvaluation. It recomputes nothing. Limit values,
point values, original indices, and violations must match M3 bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.charts.model import (
    ChartModel,
    PointClass,
    build_chart_model,
    build_pair_model,
    x_axis_label_for,
)
from app.spc.limits import ChartType, imr_chart, xbar_r_chart, xbar_s_chart
from app.spc.rules import RuleSet, evaluate_rules


def _stable_subgroups(n_groups: int = 6):
    rng = np.random.default_rng(0)
    return [np.array([10.0, 10.1, 9.9, 10.05, 9.95]) + rng.normal(0, 0.01, 5)
            for _ in range(n_groups)]


def _ooc_subgroups():
    sg = [np.array([10.0, 10.2, 9.8, 10.1, 9.9]) for _ in range(9)]
    sg.append(np.array([15.0, 15.2, 14.8, 15.1, 14.9]))  # big shift
    return sg


class TestVerbatimLimits:
    """Model limits must equal M3 limits exactly (==, not approx)."""

    def test_xbar_r_limits_exact(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        assert m.center_line == pair.location.center_line
        assert m.ucl == pair.location.ucl
        assert m.lcl == pair.location.lcl
        assert m.sigma_points == pair.location.sigma_points

    def test_r_chart_limits_exact(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.dispersion)
        assert m.center_line == pair.dispersion.center_line
        assert m.ucl == pair.dispersion.ucl
        assert m.lcl == pair.dispersion.lcl

    def test_xbar_s_limits_exact(self) -> None:
        pair = xbar_s_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        assert m.center_line == pair.location.center_line
        assert m.ucl == pair.location.ucl
        assert m.lcl == pair.location.lcl

    def test_imr_limits_exact(self) -> None:
        pair = imr_chart(np.array([10.0, 12.0, 11.0, 14.0, 13.0]))
        mi = build_chart_model(pair.location)
        mm = build_chart_model(pair.dispersion)
        assert mi.ucl == pair.location.ucl and mi.lcl == pair.location.lcl
        assert mi.center_line == pair.location.center_line
        assert mm.ucl == pair.dispersion.ucl and mm.lcl == pair.dispersion.lcl


class TestVerbatimPoints:
    def test_point_values_exact(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        assert m.values == list(pair.location.points)
        assert len(m.points) == pair.location.points.size

    def test_original_indices_preserved(self) -> None:
        # I-MR with an excluded value -> non-trivial original indices.
        pair = imr_chart([10.0, 12.0, np.nan, 14.0, 13.0], on_missing="exclude")
        mi = build_chart_model(pair.location)
        assert mi.original_indices == list(pair.location.point_indices)
        assert mi.original_indices == [0, 1, 3, 4]

    def test_mr_indices_align_to_second_member(self) -> None:
        pair = imr_chart(np.array([10.0, 12.0, 11.0, 14.0, 13.0]))
        mm = build_chart_model(pair.dispersion)
        # MR has N-1 points, indices aligned to the 2nd member of each pair.
        assert len(mm.points) == 4
        assert mm.original_indices == [1, 2, 3, 4]


class TestPointClassification:
    def test_beyond_limit_tagged(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        # The shifted subgroup (index 9) must be beyond limits.
        beyond = m.points_in_class(PointClass.BEYOND_LIMITS)
        assert any(p.original_index == 9 for p in beyond)
        # And it matches the M3 flag exactly.
        assert list(pair.location.beyond_limits) == [
            PointClass.BEYOND_LIMITS in p.classes for p in m.points
        ]

    def test_normal_points_have_normal_class(self) -> None:
        pair = xbar_r_chart(_stable_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        # Stable data: all points normal, none OOC.
        assert all(PointClass.NORMAL in p.classes for p in m.points)
        assert m.ooc_points == []
        assert m.is_stable

    def test_rule_violation_tagged_and_overlaps_preserved(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        # The run of 9 low points then a high shift triggers multiple rules.
        violated = m.points_in_class(PointClass.RULE_VIOLATION)
        assert violated
        # At least one point carries multiple overlapping rule IDs.
        assert any(len(p.violation_ids) >= 2 for p in violated)

    def test_no_evaluation_means_no_rule_tags(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location, None)  # no evaluation
        assert m.points_in_class(PointClass.RULE_VIOLATION) == []
        # Beyond-limit still tagged from the chart flag.
        assert m.points_in_class(PointClass.BEYOND_LIMITS)


class TestViolationMapping:
    def test_violations_carried_verbatim(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        assert m.violations == ev.violations  # same objects, structured & separate
        assert m.is_stable == ev.is_stable

    def test_lookup_violations_for_index(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        # Index 9 (the shift) should be implicated in >= 1 violation.
        vs = m.violations_for_original_index(9)
        assert vs
        assert all(9 in v.indices for v in vs)

    def test_point_violation_ids_match_violations(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        for p in m.points:
            expected = []
            for v in ev.violations:
                if p.original_index in v.indices and v.rule_id not in expected:
                    expected.append(v.rule_id)
            assert list(p.violation_ids) == expected


class TestLabels:
    def test_xbar_uses_subgroup_axis(self) -> None:
        pair = xbar_r_chart(_stable_subgroups())
        m = build_chart_model(pair.location, unit="mm")
        assert m.x_label == "Subgroup"
        assert "mm" in m.y_label

    def test_individuals_uses_observation_axis(self) -> None:
        pair = imr_chart(np.array([10.0, 11.0, 12.0, 10.5]))
        m = build_chart_model(pair.location, unit="inch")
        assert m.x_label == "Observation"
        assert "inch" in m.y_label

    def test_unit_absent_no_parens(self) -> None:
        pair = imr_chart(np.array([10.0, 11.0, 12.0, 10.5]))
        m = build_chart_model(pair.location)  # no unit
        assert "(" not in m.y_label

    def test_x_axis_label_helper(self) -> None:
        assert x_axis_label_for(ChartType.XBAR_R) == "Subgroup"
        assert x_axis_label_for(ChartType.INDIVIDUALS) == "Observation"
        assert x_axis_label_for(ChartType.MOVING_RANGE) == "Observation"


class TestPairModel:
    def test_pair_model_independent_evaluations(self) -> None:
        pair = imr_chart(np.array([10, 12, 11, 14, 13, 10, 30.0, 11, 12, 10]))
        loc_ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        disp_ev = evaluate_rules(pair.dispersion, rule_set=RuleSet.NELSON)
        pm = build_pair_model(pair, location_eval=loc_ev, dispersion_eval=disp_ev, unit="mm")
        assert pm.location.chart_type is ChartType.INDIVIDUALS
        assert pm.dispersion.chart_type is ChartType.MOVING_RANGE
        # Individuals has N points, MR has N-1.
        assert len(pm.location.points) == 10
        assert len(pm.dispersion.points) == 9

    def test_pair_serialization(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        pm = build_pair_model(
            pair,
            location_eval=evaluate_rules(pair.location, rule_set=RuleSet.NELSON),
            dispersion_eval=evaluate_rules(pair.dispersion, rule_set=RuleSet.NELSON),
            unit="mm",
        )
        d = pm.as_dict()
        assert d["location"]["ucl"] == pair.location.ucl
        assert d["dispersion"]["chart_type"] == "r"


class TestInsufficientAndEdge:
    def test_two_point_imr_model(self) -> None:
        pair = imr_chart(np.array([10.0, 12.0]))
        mi = build_chart_model(pair.location)
        mm = build_chart_model(pair.dispersion)
        assert len(mi.points) == 2
        assert len(mm.points) == 1  # N-1

    def test_zero_variation_flat_model(self) -> None:
        pair = xbar_r_chart([np.array([5.0, 5.0, 5.0]) for _ in range(4)])
        ev = evaluate_rules(pair.location, rule_set=RuleSet.ALL)
        m = build_chart_model(pair.location, ev)
        # Flat line: sigma_points 0, no OOC, model still builds.
        assert m.sigma_points == 0.0
        assert m.ooc_points == []
        assert isinstance(m, ChartModel)
