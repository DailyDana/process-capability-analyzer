"""Tests for the WE / Nelson run-rule engine (app.spc.rules).

Every rule is triggered by a hand-crafted deterministic sequence whose expected
violation (rule ID, affected indices, direction) is written out explicitly here,
NOT produced by running the rule engine. In-control sequences prove there are no
false positives.

Synthetic charts are built directly with CL = 0 and sigma_points = 1, so the
control limits are +/-3 and zone boundaries fall at integer sigma values, making
every crafted point's zone unambiguous by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.spc.limits import ChartType, ControlChart
from app.spc.rules import (
    Direction,
    NELSON_RULES,
    RuleSet,
    WESTERN_ELECTRIC_RULES,
    evaluate_rules,
)


def mk(points, *, symmetric: bool = True, chart_type: ChartType = ChartType.INDIVIDUALS):
    """Build a synthetic chart: CL=0, sigma=1 -> UCL=3, LCL=-3.

    Zone boundaries: |z|=1 (C/B), |z|=2 (B/A), |z|=3 (limit).
    """
    pts = np.asarray(points, dtype=float)
    return ControlChart(
        chart_type=chart_type,
        points=pts,
        center_line=0.0,
        ucl=3.0,
        lcl=-3.0,
        sigma_points=1.0,
        beyond_limits=(pts > 3.0) | (pts < -3.0),
        is_symmetric=symmetric,
        point_indices=np.arange(pts.size, dtype=int),
    )



def mk_flat(value: float, n: int):
    """Build a genuine ZERO-VARIATION chart: every point identical.

    This is what `imr_chart`/`xbar_r_chart` produce for constant data:
    MR_bar = 0 -> UCL == LCL == CL and sigma_points == 0. Distinct from `mk`,
    whose CL=0/sigma=1 frame would treat a constant series as an offset run.
    """
    pts = np.full(n, float(value))
    return ControlChart(
        chart_type=ChartType.INDIVIDUALS,
        points=pts,
        center_line=float(value),
        ucl=float(value),
        lcl=float(value),
        sigma_points=0.0,
        beyond_limits=np.zeros(n, dtype=bool),
        is_symmetric=True,
        point_indices=np.arange(n, dtype=int),
    )


def ids_of(ev, rule_id):
    return [v for v in ev.violations if v.rule_id == rule_id]


# ---------------------------------------------------------------------------
# Western Electric rules
# ---------------------------------------------------------------------------
class TestWesternElectric:
    def test_WE1_beyond_3sigma_upper(self) -> None:
        # Index 3 = 3.5 > UCL(3). Others in-band.
        ev = evaluate_rules(mk([0.1, -0.2, 0.3, 3.5, 0.0]), rule_set=RuleSet.WECO)
        v = ids_of(ev, "WE1")
        assert len(v) == 1
        assert v[0].indices == (3,)
        assert v[0].direction is Direction.UPPER

    def test_WE1_beyond_3sigma_lower(self) -> None:
        ev = evaluate_rules(mk([0.1, -3.4, 0.3, 0.2]), rule_set=RuleSet.WECO)
        v = ids_of(ev, "WE1")
        assert v[0].indices == (1,)
        assert v[0].direction is Direction.LOWER

    def test_WE2_two_of_three_beyond_2sigma(self) -> None:
        # points at z = 2.5, 0.1, 2.6 -> 2 of 3 in Zone A upper (indices 0,2).
        ev = evaluate_rules(mk([2.5, 0.1, 2.6]), rule_set=RuleSet.WECO)
        v = ids_of(ev, "WE2")
        assert len(v) >= 1
        assert v[0].indices == (0, 2)
        assert v[0].direction is Direction.UPPER

    def test_WE3_four_of_five_beyond_1sigma(self) -> None:
        # z = 1.5,1.6,0.1,1.7,1.8 -> 4 of 5 in Zone B upper (indices 0,1,3,4).
        ev = evaluate_rules(mk([1.5, 1.6, 0.1, 1.7, 1.8]), rule_set=RuleSet.WECO)
        v = ids_of(ev, "WE3")
        assert len(v) >= 1
        assert v[0].indices == (0, 1, 3, 4)
        assert v[0].direction is Direction.UPPER

    def test_WE4_eight_on_one_side(self) -> None:
        # 8 consecutive points above CL (all +0.5). Indices 0..7.
        ev = evaluate_rules(mk([0.5] * 8), rule_set=RuleSet.WECO)
        v = ids_of(ev, "WE4")
        assert len(v) == 1
        assert v[0].indices == tuple(range(8))
        assert v[0].direction is Direction.UPPER

    def test_WE4_seven_on_one_side_does_not_trigger(self) -> None:
        # Boundary: only 7 on one side -> WE4 (needs 8) must NOT fire.
        ev = evaluate_rules(mk([0.5] * 7), rule_set=RuleSet.WECO)
        assert ids_of(ev, "WE4") == []


# ---------------------------------------------------------------------------
# Nelson rules
# ---------------------------------------------------------------------------
class TestNelson:
    def test_N1_beyond_3sigma(self) -> None:
        ev = evaluate_rules(mk([0.0, 0.1, 4.0]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N1")
        assert v[0].indices == (2,)
        assert v[0].direction is Direction.UPPER

    def test_N2_nine_on_one_side(self) -> None:
        ev = evaluate_rules(mk([-0.5] * 9), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N2")
        assert len(v) == 1
        assert v[0].indices == tuple(range(9))
        assert v[0].direction is Direction.LOWER

    def test_N2_eight_on_one_side_does_not_trigger(self) -> None:
        # Boundary: Nelson needs 9, so 8 must not fire N2 (but WE4 would).
        ev = evaluate_rules(mk([0.5] * 8), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N2") == []

    def test_N3_six_increasing(self) -> None:
        ev = evaluate_rules(
            mk([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]), rule_set=RuleSet.NELSON
        )
        v = ids_of(ev, "N3")
        assert v[0].indices == (0, 1, 2, 3, 4, 5)
        assert v[0].direction is Direction.UP

    def test_N3_six_decreasing(self) -> None:
        ev = evaluate_rules(
            mk([1.0, 0.6, 0.2, -0.2, -0.6, -1.0]), rule_set=RuleSet.NELSON
        )
        v = ids_of(ev, "N3")
        assert v[0].direction is Direction.DOWN

    def test_N3_five_increasing_does_not_trigger(self) -> None:
        ev = evaluate_rules(mk([-1.0, -0.5, 0.0, 0.5, 1.0]), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N3") == []

    def test_N4_fourteen_alternating(self) -> None:
        ev = evaluate_rules(mk([0.5, -0.5] * 7), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N4")
        assert len(v) == 1
        assert len(v[0].indices) == 14
        assert v[0].direction is Direction.NONE

    def test_N5_two_of_three_zone_a(self) -> None:
        ev = evaluate_rules(mk([-2.5, -0.1, -2.7]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N5")
        assert v[0].indices == (0, 2)
        assert v[0].direction is Direction.LOWER

    def test_N6_four_of_five_zone_b(self) -> None:
        ev = evaluate_rules(mk([1.2, 1.3, 0.0, 1.4, 1.5]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")
        assert v[0].indices == (0, 1, 3, 4)

    def test_N7_fifteen_within_1sigma(self) -> None:
        # 15 points all |z| < 1, not monotone/alternating enough for others.
        pts = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.15,
               0.25, -0.25, 0.05, -0.05, 0.2, -0.2, 0.1]
        ev = evaluate_rules(mk(pts), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N7")
        assert len(v) == 1
        assert len(v[0].indices) == 15
        assert v[0].direction is Direction.NONE

    def test_N8_eight_outside_1sigma(self) -> None:
        # 8 points all |z| >= 1, both sides present (mixture).
        ev = evaluate_rules(
            mk([1.5, -1.5, 1.6, -1.6, 1.7, -1.7, 1.8, -1.8]), rule_set=RuleSet.NELSON
        )
        v = ids_of(ev, "N8")
        assert len(v) == 1
        assert len(v[0].indices) == 8
        assert v[0].direction is Direction.NONE


class TestN8BothSidesRequirement:
    """A2 regression: Nelson rule 8 requires BOTH sides of the center line.

    Nelson (1984) rule 8: "Eight points in a row on both sides of the
    centerline with none in Zone C." The original implementation tested only
    ``|z| >= 1``, so a purely one-sided run (a SHIFT, which is N2's/WE4's
    signal) was misreported as a MIXTURE. These tests pin the both-sides
    requirement; cases 2 and 3 fail against the original implementation.
    """

    def test_1_alternating_fires(self) -> None:
        ev = evaluate_rules(mk([1.5, -1.5] * 4), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N8")
        assert len(v) == 1
        assert v[0].indices == (0, 1, 2, 3, 4, 5, 6, 7)
        assert "both sides" in v[0].explanation

    def test_2_eight_positive_does_not_fire(self) -> None:
        """One-sided run above CL is a shift, NOT a mixture."""
        ev = evaluate_rules(mk([1.5] * 8), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N8") == []

    def test_3_eight_negative_does_not_fire(self) -> None:
        """One-sided run below CL is a shift, NOT a mixture."""
        ev = evaluate_rules(mk([-1.5] * 8), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N8") == []

    def test_4_zone_c_point_breaks_the_run(self) -> None:
        # 4 outside, one Zone-C point, 4 outside: no 8-long outside run exists.
        pts = [1.5, -1.5, 1.5, -1.5, 0.2, 1.5, -1.5, 1.5, -1.5]
        ev = evaluate_rules(mk(pts), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N8") == []

    def test_5_boundary_values_exactly_at_one_sigma(self) -> None:
        """|z| == 1 is outside Zone C (>= convention preserved)."""
        ev = evaluate_rules(mk([1.0, -1.0] * 4), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N8")
        assert len(v) == 1
        assert len(v[0].indices) == 8

    def test_6_longer_sequence_merges_overlapping_windows(self) -> None:
        """12 alternating points = 5 candidate windows -> ONE merged violation."""
        ev = evaluate_rules(mk([1.5, -1.5] * 6), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N8")
        assert len(v) == 1, "overlapping windows must merge into one violation"
        assert v[0].indices == tuple(range(12))

    def test_7_side_change_late_in_run_still_qualifies(self) -> None:
        """A window spanning both sides fires even if 7 points share a side."""
        ev = evaluate_rules(mk([1.5] * 7 + [-1.5]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N8")
        assert len(v) == 1
        assert v[0].indices == tuple(range(8))

    def test_8_stays_distinct_from_N2_and_WE4(self) -> None:
        """A one-sided shift must be reported as a shift, never as a mixture."""
        ev = evaluate_rules(mk([1.5] * 9), rule_set=RuleSet.ALL)
        assert ids_of(ev, "N8") == [], "shift must not be labelled a mixture"
        assert ids_of(ev, "N2"), "N2 (9 same side) should fire"
        assert ids_of(ev, "WE4"), "WE4 (8 same side) should fire"

    def test_9_zero_length_guard_short_series(self) -> None:
        ev = evaluate_rules(mk([1.5, -1.5] * 3), rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N8") == []


# ---------------------------------------------------------------------------
# No false positives on a known in-control sequence
# ---------------------------------------------------------------------------
class TestInControl:
    def test_in_control_no_violations(self) -> None:
        # Small, non-monotone, non-alternating, all within 2 sigma, both sides,
        # no run of 8/9. Deliberately benign.
        pts = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15]
        for rs in (RuleSet.WECO, RuleSet.NELSON, RuleSet.ALL):
            ev = evaluate_rules(mk(pts), rule_set=rs)
            assert ev.is_stable, f"unexpected violations for {rs}: {ev.violations}"
            assert ev.violations == ()

    def test_in_control_longer_series(self) -> None:
        rng = np.random.default_rng(12345)
        # Draw within +/-2 sigma to avoid WE1; shuffle sign to avoid runs.
        pts = rng.uniform(-1.5, 1.5, size=20)
        ev = evaluate_rules(mk(pts), rule_set=RuleSet.NELSON)
        # This particular seed is checked to be in-control for the run rules.
        for v in ev.violations:
            # If anything triggers it must not be a beyond-limit (all within 1.5).
            assert v.rule_id != "N1"


# ---------------------------------------------------------------------------
# Multiple simultaneous violations
# ---------------------------------------------------------------------------
class TestMultipleViolations:
    def test_beyond_and_run_together(self) -> None:
        # 9 points all clearly positive (N2) with the last one beyond 3 sigma (N1).
        ev = evaluate_rules(
            mk([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.5]), rule_set=RuleSet.NELSON
        )
        rule_ids = {v.rule_id for v in ev.violations}
        assert "N1" in rule_ids  # beyond limit
        assert "N2" in rule_ids  # nine on one side
        # Both preserved, not collapsed.
        assert not ev.is_stable

    def test_all_ruleset_preserves_both_families(self) -> None:
        # 8 on one side triggers WE4 (8) but not N2 (needs 9).
        ev = evaluate_rules(mk([0.5] * 8), rule_set=RuleSet.ALL)
        rule_ids = {v.rule_id for v in ev.violations}
        assert "WE4" in rule_ids
        assert "N2" not in rule_ids  # distinct run lengths preserved


# ---------------------------------------------------------------------------
# Asymmetric charts: only beyond-limit test applies
# ---------------------------------------------------------------------------
class TestAsymmetricCharts:
    def test_dispersion_chart_only_beyond_limit(self) -> None:
        # A run of 9 on one side on an R-chart must NOT trigger N2 (asymmetric).
        chart = mk([0.5] * 9, symmetric=False, chart_type=ChartType.R)
        ev = evaluate_rules(chart, rule_set=RuleSet.NELSON)
        assert ids_of(ev, "N2") == []
        # But a beyond-limit point still triggers N1.
        chart2 = mk([0.5, 0.5, 4.0], symmetric=False, chart_type=ChartType.R)
        ev2 = evaluate_rules(chart2, rule_set=RuleSet.NELSON)
        assert ids_of(ev2, "N1")


# ---------------------------------------------------------------------------
# Insufficient data / boundary conditions
# ---------------------------------------------------------------------------
class TestInsufficientData:
    def test_short_series_skips_long_rules(self) -> None:
        ev = evaluate_rules(mk([0.5, 0.6, 0.7]), rule_set=RuleSet.NELSON)
        skipped_ids = {s.rule_id for s in ev.skipped}
        # N2(9), N3(6), N4(14), N7(15), N8(8) all need > 3 points.
        assert {"N2", "N3", "N4", "N7", "N8"}.issubset(skipped_ids)
        # N1 (needs 1) is NOT skipped.
        assert "N1" not in skipped_ids

    def test_skip_records_reason(self) -> None:
        ev = evaluate_rules(mk([0.5, 0.6]), rule_set=RuleSet.NELSON)
        n2 = [s for s in ev.skipped if s.rule_id == "N2"]
        assert n2 and "at least 9" in n2[0].reason

    def test_no_crash_on_single_point(self) -> None:
        ev = evaluate_rules(mk([0.5]), rule_set=RuleSet.ALL)
        assert isinstance(ev.is_stable, bool)


# ---------------------------------------------------------------------------
# Zero-variation chart
# ---------------------------------------------------------------------------
class TestZeroVariation:
    def test_flat_line_no_run_rules(self) -> None:
        # sigma_points = 0 -> all z = 0; no run rule can fire, no beyond-limit.
        flat = ControlChart(
            chart_type=ChartType.INDIVIDUALS,
            points=np.full(12, 5.0),
            center_line=5.0,
            ucl=5.0,
            lcl=5.0,
            sigma_points=0.0,
            beyond_limits=np.zeros(12, dtype=bool),
            is_symmetric=True,
            point_indices=np.arange(12),
        )
        ev = evaluate_rules(flat, rule_set=RuleSet.ALL)
        assert ev.is_stable


# ---------------------------------------------------------------------------
# Original-index mapping (violations map back to user rows)
# ---------------------------------------------------------------------------
class TestIndexMapping:
    def test_indices_mapped_through_point_indices(self) -> None:
        # Chart whose plotted points correspond to original rows [10..18].
        pts = np.array([0.5] * 9)
        chart = ControlChart(
            chart_type=ChartType.INDIVIDUALS,
            points=pts,
            center_line=0.0,
            ucl=3.0,
            lcl=-3.0,
            sigma_points=1.0,
            beyond_limits=np.zeros(9, dtype=bool),
            is_symmetric=True,
            point_indices=np.arange(10, 19, dtype=int),
        )
        ev = evaluate_rules(chart, rule_set=RuleSet.NELSON)
        n2 = [v for v in ev.violations if v.rule_id == "N2"][0]
        assert n2.indices == tuple(range(10, 19))  # mapped, not 0..8


# ---------------------------------------------------------------------------
# Registry sanity: rule sets are distinct and complete
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_weco_has_four_rules(self) -> None:
        ids = {r.rule_id for r in WESTERN_ELECTRIC_RULES}
        assert ids == {"WE1", "WE2", "WE3", "WE4"}

    def test_nelson_has_eight_rules(self) -> None:
        ids = {r.rule_id for r in NELSON_RULES}
        assert ids == {"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"}

    def test_we4_and_n2_have_distinct_run_lengths(self) -> None:
        # Documented difference: WE4 = 8, N2 = 9.
        ev8 = evaluate_rules(mk([0.5] * 8), rule_set=RuleSet.ALL)
        assert {v.rule_id for v in ev8.violations} & {"WE4"}
        assert not ({v.rule_id for v in ev8.violations} & {"N2"})
        ev9 = evaluate_rules(mk([0.5] * 9), rule_set=RuleSet.ALL)
        got = {v.rule_id for v in ev9.violations}
        assert "WE4" in got and "N2" in got


class TestRuleSetAllDeduplication:
    """B2: RuleSet.ALL must not double-count logically identical detections.

    WE1<->N1, WE2<->N5 and WE3<->N6 are the same test in both families. Running
    ALL previously emitted two Violations per physical event, inflating
    n_violations and the violations tables in every export.
    """

    def test_beyond_limit_reported_once_with_both_ids(self) -> None:
        ev = evaluate_rules(mk([4.0]), rule_set=RuleSet.ALL)
        beyond = [v for v in ev.violations if v.matches_rule("WE1")]
        assert len(beyond) == 1, "one physical event -> one violation"
        v = beyond[0]
        assert v.all_rule_ids == ("WE1", "N1")
        assert v.matches_rule("N1") and v.matches_rule("WE1")

    def test_both_families_remain_attributable(self) -> None:
        ev = evaluate_rules(mk([4.0]), rule_set=RuleSet.ALL)
        assert "WE1" in ev.triggered_rule_ids
        assert "N1" in ev.triggered_rule_ids

    def test_two_of_three_pair_merged(self) -> None:
        ev = evaluate_rules(mk([2.5, 2.5, 0.0]), rule_set=RuleSet.ALL)
        merged = [v for v in ev.violations if v.matches_rule("WE2")]
        assert len(merged) == 1
        assert merged[0].all_rule_ids == ("WE2", "N5")

    def test_four_of_five_pair_merged(self) -> None:
        ev = evaluate_rules(mk([1.5, 1.5, 1.5, 1.5, 0.0]), rule_set=RuleSet.ALL)
        merged = [v for v in ev.violations if v.matches_rule("WE3")]
        assert len(merged) == 1
        assert merged[0].all_rule_ids == ("WE3", "N6")

    def test_distinct_rules_are_not_merged(self) -> None:
        """WE4 (8) and N2 (9) make different claims and must stay separate."""
        ev = evaluate_rules(mk([0.5] * 9), rule_set=RuleSet.ALL)
        ids = sorted({v.rule_id for v in ev.violations})
        assert "WE4" in ids and "N2" in ids
        we4 = [v for v in ev.violations if v.rule_id == "WE4"][0]
        n2 = [v for v in ev.violations if v.rule_id == "N2"][0]
        assert we4.equivalent_rule_ids == ()
        assert n2.equivalent_rule_ids == ()

    def test_n_violations_not_inflated(self) -> None:
        """ALL must not report more physical events than a single family."""
        pts = [4.0]  # one point beyond the UCL: one physical event
        n_weco = len(evaluate_rules(mk(pts), rule_set=RuleSet.WECO).violations)
        n_nelson = len(evaluate_rules(mk(pts), rule_set=RuleSet.NELSON).violations)
        n_all = len(evaluate_rules(mk(pts), rule_set=RuleSet.ALL).violations)
        assert n_weco == 1 and n_nelson == 1
        assert n_all == 1, "equivalent detections must not double-count"

    def test_single_family_runs_have_no_equivalents(self) -> None:
        for rs in (RuleSet.WECO, RuleSet.NELSON):
            ev = evaluate_rules(mk([4.0]), rule_set=rs)
            for v in ev.violations:
                assert v.equivalent_rule_ids == ()

    def test_nelson_only_rules_pass_through_under_all(self) -> None:
        """N3/N4/N7/N8 have no WE counterpart and must survive ALL untouched."""
        ev = evaluate_rules(mk([0.0] * 15), rule_set=RuleSet.ALL)
        assert any(v.rule_id == "N7" for v in ev.violations)


class TestSlidingWindowPhysicalEvents:
    """C1: one continuous excursion is one violation, not one per window."""

    def test_minimal_two_of_three_is_one_violation(self) -> None:
        ev = evaluate_rules(mk([2.5, 2.5, 0.0, 0.0, 0.0]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N5")
        assert len(v) == 1
        assert v[0].indices == (0, 1)

    def test_sustained_four_of_five_run_is_one_event(self) -> None:
        """10 points beyond 1 sigma = 6 sliding windows -> ONE physical event."""
        ev = evaluate_rules(mk([1.5] * 10), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")
        assert len(v) == 1, f"expected 1 merged event, got {len(v)}"
        assert v[0].indices == tuple(range(10))
        assert "one event" in v[0].explanation

    def test_two_separated_excursions_stay_distinct(self) -> None:
        """A gap closes an event; the methodology requires both be reported."""
        pts = [1.5] * 5 + [0.0] * 5 + [1.5] * 5
        ev = evaluate_rules(mk(pts), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")
        assert len(v) == 2, "distinct excursions must not be merged"
        assert v[0].indices == (0, 1, 2, 3, 4)
        assert v[1].indices == (10, 11, 12, 13, 14)

    def test_opposite_sides_stay_distinct(self) -> None:
        """Upper and lower excursions are different physical events."""
        pts = [1.5] * 5 + [0.0] * 5 + [-1.5] * 5
        ev = evaluate_rules(mk(pts), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")
        assert len(v) == 2
        assert {x.direction for x in v} == {Direction.UPPER, Direction.LOWER}

    def test_merge_preserves_all_affected_indices(self) -> None:
        """Merging must not lose points (union, not first-window-only)."""
        ev = evaluate_rules(mk([1.5] * 8), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")[0]
        assert v.indices == tuple(range(8))

    def test_non_contiguous_hits_within_window_preserved(self) -> None:
        """4-of-5 allows one interior miss; the miss is not flagged."""
        ev = evaluate_rules(mk([1.5, 1.5, 0.2, 1.5, 1.5]), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N6")
        assert len(v) == 1
        assert v[0].indices == (0, 1, 3, 4), "the in-zone point must not be flagged"

    def test_two_of_three_sustained_run_merges(self) -> None:
        ev = evaluate_rules(mk([2.5] * 6), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N5")
        assert len(v) == 1
        assert v[0].indices == tuple(range(6))


class TestFlatLineSemantics:
    """B4: flat-line behavior is documented and asserted, not contradictory.

    Decision: N7 IS retained for exactly-flat data. An exactly flat series is
    the limiting case of stratification and in practice means the gauge cannot
    resolve the process variation -- a real, actionable signal. The former
    docstring claim that "no run rule can fire on a flat line" was false and has
    been corrected rather than the behavior being suppressed.
    """

    def test_flat_line_of_15_triggers_n7(self) -> None:
        ev = evaluate_rules(mk_flat(5.0, 15), rule_set=RuleSet.NELSON)
        v = ids_of(ev, "N7")
        assert len(v) == 1
        assert len(v[0].indices) == 15
        assert not ev.is_stable

    def test_flat_line_below_threshold_is_clean(self) -> None:
        ev = evaluate_rules(mk_flat(5.0, 14), rule_set=RuleSet.NELSON)
        assert ev.violations == ()
        assert ev.is_stable

    def test_flat_line_triggers_only_n7(self) -> None:
        """Every other rule must stay silent, as the docstring now states."""
        ev = evaluate_rules(mk_flat(5.0, 30), rule_set=RuleSet.ALL)
        ids = {v.rule_id for v in ev.violations}
        assert ids == {"N7"}, f"only N7 may fire on a flat line, got {ids}"

    def test_flat_line_no_beyond_limit_violation(self) -> None:
        ev = evaluate_rules(mk_flat(5.0, 20), rule_set=RuleSet.ALL)
        assert not any(v.matches_rule("WE1") for v in ev.violations)
