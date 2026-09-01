"""Western Electric (WECO) and Nelson run-rule engines.

Operates on a :class:`~app.spc.limits.ControlChart` and returns structured
:class:`Violation` objects. Every violation records the rule ID, rule name,
affected observation indices (mapped to the chart's original point indices),
the direction (upper/lower/up/down/none), and a human-readable explanation.

The two rule sets are kept DISTINCT (own IDs, own run lengths); see
``docs/FORMULAS.md`` §3.6-3.8. Rules that need more points than are available
are reported as :class:`RuleSkipped` rather than silently ignored. If several
rules fire on overlapping points, ALL applicable violations are preserved.

Zone construction (FORMULAS.md §3.5): zones are exact thirds of the control-
limit band, ``sigma = (UCL - CL)/3``. Signed zone level of a point:
``z = (point - CL) / sigma``; ``|z|`` in [0,1)=C, [1,2)=B, [2,3)=A, >=3 beyond.
Boundary convention: a point exactly on a zone boundary counts as being in the
*outer* zone (>= comparison), matching standard SPC practice; a point exactly on
a control limit is NOT "beyond" it (strict inequality), consistent with limits.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from app.spc.limits import ControlChart

__all__ = [
    "RuleSet",
    "Direction",
    "Violation",
    "RuleSkipped",
    "RuleEvaluation",
    "evaluate_rules",
    "WESTERN_ELECTRIC_RULES",
    "NELSON_RULES",
]


class RuleSet(str, Enum):
    """Which family of run rules to evaluate."""

    WECO = "western_electric"
    NELSON = "nelson"
    ALL = "all"


class Direction(str, Enum):
    """Direction of a violation relative to the center line."""

    UPPER = "upper"
    LOWER = "lower"
    UP = "up"  # increasing trend
    DOWN = "down"  # decreasing trend
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Violation:
    """A single detected rule violation.

    Attributes
    ----------
    rule_id:
        Stable rule identifier (e.g. ``"WE1"``, ``"N3"``).
    rule_name:
        Human-readable rule name.
    rule_set:
        Which set the rule belongs to.
    indices:
        Affected points as ORIGINAL observation indices (mapped through the
        chart's ``point_indices``), ascending.
    direction:
        :class:`Direction` of the violation.
    explanation:
        Human-readable description of what triggered and why.
    equivalent_rule_ids:
        Other rule IDs that detect this SAME physical event, populated only
        under :attr:`RuleSet.ALL` where the Western Electric and Nelson sets
        contain logically identical tests (WE1<->N1, WE2<->N5, WE3<->N6).
        The violation is reported ONCE (so physical event counts are not
        inflated) while both IDs remain available for attribution. Empty for
        single-family evaluations and for rules with no equivalent.
    """

    rule_id: str
    rule_name: str
    rule_set: RuleSet
    indices: tuple[int, ...]
    direction: Direction
    explanation: str
    equivalent_rule_ids: tuple[str, ...] = ()

    @property
    def all_rule_ids(self) -> tuple[str, ...]:
        """Every rule ID that detected this event (primary first)."""
        return (self.rule_id,) + self.equivalent_rule_ids

    def matches_rule(self, rule_id: str) -> bool:
        """True if ``rule_id`` detected this event (primary or equivalent)."""
        return rule_id in self.all_rule_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_set": self.rule_set.value,
            "indices": list(self.indices),
            "direction": self.direction.value,
            "explanation": self.explanation,
            "equivalent_rule_ids": list(self.equivalent_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class RuleSkipped:
    """Record that a rule could not be evaluated (insufficient data)."""

    rule_id: str
    rule_name: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"rule_id": self.rule_id, "rule_name": self.rule_name, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """Result of evaluating a rule set against a chart.

    Attributes
    ----------
    violations:
        All detected violations (may overlap; all preserved).
    skipped:
        Rules not evaluated due to insufficient data.
    is_stable:
        ``True`` iff no violations were found. This is the SPC stability verdict
        the capability layer consumes.
    """

    violations: tuple[Violation, ...]
    skipped: tuple[RuleSkipped, ...]

    @property
    def is_stable(self) -> bool:
        return len(self.violations) == 0

    @property
    def triggered_rule_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for v in self.violations:
            # Include equivalent IDs so a de-duplicated WE1/N1 event is still
            # attributable to BOTH families (RuleSet.ALL, FORMULAS.md 3.8).
            for rid in v.all_rule_ids:
                if rid not in seen:
                    seen.append(rid)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        return {
            "violations": [v.as_dict() for v in self.violations],
            "skipped": [s.as_dict() for s in self.skipped],
            "is_stable": self.is_stable,
            "triggered_rule_ids": list(self.triggered_rule_ids),
        }


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------
def _signed_zone_levels(chart: ControlChart) -> NDArray[np.float64]:
    """Return signed sigma distance of each point from CL: ``(x - CL)/sigma``.

    Zero-variation (flat-line) charts
    ---------------------------------
    When ``sigma_points == 0`` every point sits exactly on the CL, so the signed
    level is defined as 0 everywhere (avoiding 0/0). This has a deliberate and
    documented consequence: **N7 (15 consecutive points within 1 sigma) DOES
    fire on a flat line of 15 or more points**, because ``|z| = 0 < 1`` for
    every point.

    That is the intended semantics, not an accident. N7 detects
    *stratification* -- points hugging the center line far more tightly than the
    control limits imply. An exactly flat series is the limiting case of that
    pattern, and in manufacturing it almost always means the measurement
    resolution is too coarse for the process (the gauge cannot resolve the
    variation that exists), or the data were rounded/duplicated. Both are real,
    actionable special causes, so signalling them is correct: see
    Montgomery SQC 7th ed. Sec. 5.3 on control limits that are too wide relative
    to the plotted statistic's true dispersion.

    Rules that CANNOT fire on a flat line, for reference:

    * the beyond-limit test (WE1/N1): UCL == LCL == CL, and the comparison is
      strict, so no point is "beyond";
    * same-side runs (WE4/N2): points exactly on the CL belong to neither side
      and break the run;
    * trend (N3) and alternation (N4): ties break both;
    * 2-of-3 / 4-of-5 beyond a zone (WE2/WE3/N5/N6): ``|z| = 0`` clears no zone;
    * N8 (outside Zone C, both sides): ``|z| = 0`` is inside Zone C.

    So a flat line of >= 15 points yields exactly ONE violation (N7), which the
    interpretation layer explains as a resolution/stratification signal rather
    than as ordinary process instability.
    """
    if chart.sigma_points == 0.0:
        return np.zeros(chart.points.size, dtype=np.float64)
    return (chart.points - chart.center_line) / chart.sigma_points


def _orig(chart: ControlChart, positions: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Map plotted positions to original observation indices, ascending."""
    mapped = sorted(int(chart.point_indices[p]) for p in positions)
    return tuple(mapped)


# ---------------------------------------------------------------------------
# Individual rule detectors. Each returns a list of Violation.
# They receive the chart, the signed zone levels z, and the configured RuleSet
# tag so the emitted Violation carries the correct provenance.
# ---------------------------------------------------------------------------
def _rule_beyond_3sigma(
    chart: ControlChart, z: NDArray[np.float64], rule_id: str, rule_set: RuleSet
) -> list[Violation]:
    """One point beyond a control limit (WE1 / N1). Uses limits.py beyond flags."""
    out: list[Violation] = []
    for p in np.nonzero(chart.beyond_limits)[0]:
        direction = Direction.UPPER if chart.points[p] > chart.ucl else Direction.LOWER
        out.append(
            Violation(
                rule_id=rule_id,
                rule_name="One point beyond 3 sigma (outside a control limit)",
                rule_set=rule_set,
                indices=_orig(chart, [int(p)]),
                direction=direction,
                explanation=(
                    f"Point at index {int(chart.point_indices[p])} = "
                    f"{chart.points[p]:.6g} lies "
                    f"{'above the UCL' if direction is Direction.UPPER else 'below the LCL'} "
                    f"({chart.ucl:.6g}/{chart.lcl:.6g}); a single point beyond 3 sigma "
                    "indicates a special cause."
                ),
            )
        )
    return out


def _rule_k_of_m_beyond_zone(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    k: int,
    m: int,
    zone_threshold: float,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """Generic 'k of m consecutive points beyond +/- zone_threshold, same side'.

    Covers WE2/N5 (2 of 3 in Zone A: threshold 2) and WE3/N6 (4 of 5 in Zone B:
    threshold 1). Same side means all counted points share the sign.

    Physical-event semantics (C1, hardening milestone)
    --------------------------------------------------
    A sliding window is the DETECTOR, not the event. A sustained excursion
    satisfies the k-of-m test in several consecutive window positions; emitting
    one violation per window inflates ``n_violations`` and makes one physical
    excursion look like several independent special causes.

    Overlapping qualifying windows **on the same side** are therefore merged
    into a single violation covering the union of their flagged points. Merging
    is by window adjacency/overlap, not by ``set(indices)``: distinct events
    remain distinct because
      * a gap (a window position that does not qualify) closes the event, and
      * the upper and lower sides are tracked separately, so a genuine
        excursion above followed by one below yields two violations.
    """
    out: list[Violation] = []
    npts = z.size
    if npts < m:
        return out

    # Collect qualifying windows per side, then merge overlapping/adjacent ones
    # into single physical events (see docstring).
    for sign, direction in ((1.0, Direction.UPPER), (-1.0, Direction.LOWER)):
        events: list[dict] = []
        for start in range(npts - m + 1):
            hits = [
                i
                for i in range(start, start + m)
                if (z[i] * sign) >= zone_threshold
            ]
            if len(hits) < k:
                continue
            if events and start <= events[-1]["win_end"]:
                # Overlaps the previous qualifying window -> same event.
                events[-1]["win_end"] = start + m - 1
                events[-1]["hits"].update(hits)
            else:
                events.append(
                    {
                        "win_start": start,
                        "win_end": start + m - 1,
                        "hits": set(hits),
                    }
                )

        for ev in events:
            hits_sorted = sorted(ev["hits"])
            n_windows = ev["win_end"] - ev["win_start"] - m + 2
            span = (
                f"positions {ev['win_start']}-{ev['win_end']}"
                if n_windows > 1
                else f"positions {ev['win_start']}-{ev['win_start'] + m - 1}"
            )
            detail = (
                f"{len(hits_sorted)} points across {span} are beyond "
                f"{zone_threshold:.0f} sigma on the "
                f"{'upper' if direction is Direction.UPPER else 'lower'} "
                "side of the center line"
            )
            if n_windows > 1:
                detail += (
                    f" (a sustained excursion satisfying the {k}-of-{m} test in "
                    f"{n_windows} consecutive windows, reported as one event)"
                )
            else:
                detail = (
                    f"{len(hits_sorted)} of {m} consecutive points ({span}) are "
                    f"beyond {zone_threshold:.0f} sigma on the "
                    f"{'upper' if direction is Direction.UPPER else 'lower'} "
                    "side of the center line"
                )
            out.append(
                Violation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    rule_set=rule_set,
                    indices=_orig(chart, hits_sorted),
                    direction=direction,
                    explanation=detail + ".",
                )
            )

    # Keep a deterministic order: by first affected index, then direction.
    out.sort(key=lambda v: (v.indices[0], v.direction.value))
    return out


def _rule_run_one_side(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    run_length: int,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """'run_length consecutive points on the same side of CL' (WE4=8 / N2=9).

    Points exactly on the CL (z == 0) break the run (they belong to neither
    side), matching the standard treatment.
    """
    out: list[Violation] = []
    npts = z.size
    if npts < run_length:
        return out
    i = 0
    while i < npts:
        if z[i] > 0:
            side, direction = 1, Direction.UPPER
        elif z[i] < 0:
            side, direction = -1, Direction.LOWER
        else:
            i += 1
            continue
        j = i
        while j < npts and (z[j] > 0) == (side == 1) and z[j] != 0:
            j += 1
        run = list(range(i, j))
        if len(run) >= run_length:
            # Emit one violation per maximal run (all points of the run).
            out.append(
                Violation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    rule_set=rule_set,
                    indices=_orig(chart, run),
                    direction=direction,
                    explanation=(
                        f"{len(run)} consecutive points (positions {i}-{j - 1}) "
                        f"lie on the {'upper' if side == 1 else 'lower'} side of "
                        f"the center line (>= {run_length} required)."
                    ),
                )
            )
        i = j if j > i else i + 1
    return out


def _rule_trend(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    run_length: int,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """'run_length consecutive points steadily increasing or decreasing' (N3=6).

    Uses the raw plotted values (not zones). Equal consecutive values break the
    monotone run (standard treatment: a tie is neither up nor down).
    """
    out: list[Violation] = []
    x = chart.points
    npts = x.size
    if npts < run_length:
        return out
    # Build direction of each step: +1 up, -1 down, 0 tie.
    for want, direction in ((1, Direction.UP), (-1, Direction.DOWN)):
        i = 0
        while i < npts - 1:
            j = i
            while j < npts - 1 and np.sign(x[j + 1] - x[j]) == want:
                j += 1
            length = j - i + 1  # number of points in the monotone run
            if length >= run_length:
                out.append(
                    Violation(
                        rule_id=rule_id,
                        rule_name=rule_name,
                        rule_set=rule_set,
                        indices=_orig(chart, list(range(i, j + 1))),
                        direction=direction,
                        explanation=(
                            f"{length} consecutive points (positions {i}-{j}) are "
                            f"steadily {'increasing' if want == 1 else 'decreasing'} "
                            f"(>= {run_length} required), indicating a trend."
                        ),
                    )
                )
            i = j + 1 if j > i else i + 1
    return out


def _rule_alternating(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    run_length: int,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """'run_length consecutive points alternating up and down' (N4=14).

    A run of length L means L points, i.e. L-1 alternating steps.
    """
    out: list[Violation] = []
    x = chart.points
    npts = x.size
    if npts < run_length:
        return out
    diffs = np.sign(np.diff(x))  # step directions
    i = 0
    while i < diffs.size:
        # Extend while signs strictly alternate and are non-zero.
        j = i
        while (
            j < diffs.size
            and diffs[j] != 0
            and (j == i or diffs[j] == -diffs[j - 1])
        ):
            j += 1
        n_steps = j - i
        n_points = n_steps + 1
        if n_points >= run_length:
            out.append(
                Violation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    rule_set=rule_set,
                    indices=_orig(chart, list(range(i, i + n_points))),
                    direction=Direction.NONE,
                    explanation=(
                        f"{n_points} consecutive points (positions {i}-"
                        f"{i + n_points - 1}) alternate up and down "
                        f"(>= {run_length} required), indicating over-adjustment "
                        "or systematic oscillation."
                    ),
                )
            )
        i = j if j > i else i + 1
    return out


def _rule_within_1sigma(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    run_length: int,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """'run_length consecutive points within 1 sigma of CL, both sides' (N7=15).

    Stratification / limits too wide. |z| < 1 for every point in the run.
    """
    out: list[Violation] = []
    npts = z.size
    if npts < run_length:
        return out
    within = np.abs(z) < 1.0
    i = 0
    while i < npts:
        if not within[i]:
            i += 1
            continue
        j = i
        while j < npts and within[j]:
            j += 1
        if (j - i) >= run_length:
            out.append(
                Violation(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    rule_set=rule_set,
                    indices=_orig(chart, list(range(i, j))),
                    direction=Direction.NONE,
                    explanation=(
                        f"{j - i} consecutive points (positions {i}-{j - 1}) lie "
                        f"within 1 sigma of the center line (>= {run_length} "
                        "required), suggesting stratification or control limits "
                        "that are too wide."
                    ),
                )
            )
        i = j if j > i else i + 1
    return out


def _rule_outside_1sigma(
    chart: ControlChart,
    z: NDArray[np.float64],
    *,
    run_length: int,
    rule_id: str,
    rule_name: str,
    rule_set: RuleSet,
) -> list[Violation]:
    """'run_length consecutive points outside Zone C, on BOTH sides' (N8=8).

    Canonical Nelson rule 8: "Eight points in a row on both sides of the
    centerline with none in Zone C." Two conditions must hold together:

    1. every point in the run is outside Zone C (``|z| >= 1``), and
    2. the run spans BOTH sides of the center line (at least one point above
       and at least one below).

    Condition 2 is what makes this a *mixture* signal and keeps it distinct
    from a one-sided shift (which is N2's / WE4's job). A point inside Zone C
    breaks the run.

    Detection scans each maximal outside-Zone-C run for windows of
    ``run_length`` consecutive points that contain both signs, then merges
    contiguous/overlapping qualifying windows into a single violation covering
    their union -- so one physical mixture pattern is reported once rather than
    once per sliding window.
    """
    out: list[Violation] = []
    npts = z.size
    if npts < run_length:
        return out

    outside = np.abs(z) >= 1.0
    # Points exactly on the center line (z == 0) are inside Zone C and are
    # therefore already excluded by `outside`; sign() is only consulted for
    # points that passed that filter.
    signs = np.sign(z)

    # Collect qualifying window spans across the whole series, merging any that
    # touch or overlap so one pattern yields one violation.
    spans: list[list[int]] = []

    i = 0
    while i < npts:
        if not outside[i]:
            i += 1
            continue
        # [i, j) is a maximal run of consecutive points outside Zone C.
        j = i
        while j < npts and outside[j]:
            j += 1

        # Within this run, find windows of `run_length` spanning both sides.
        for start in range(i, j - run_length + 1):
            window = signs[start:start + run_length]
            if (window > 0).any() and (window < 0).any():
                end = start + run_length
                if spans and spans[-1][1] >= start:
                    spans[-1][1] = max(spans[-1][1], end)
                else:
                    spans.append([start, end])

        i = j if j > i else i + 1

    for start, end in spans:
        positions = list(range(start, end))
        n_above = int((signs[start:end] > 0).sum())
        n_below = int((signs[start:end] < 0).sum())
        out.append(
            Violation(
                rule_id=rule_id,
                rule_name=rule_name,
                rule_set=rule_set,
                indices=_orig(chart, positions),
                direction=Direction.NONE,
                explanation=(
                    f"{end - start} consecutive points (positions {start}-"
                    f"{end - 1}) lie outside 1 sigma with none in Zone C, "
                    f"occurring on both sides of the center line "
                    f"({n_above} above, {n_below} below; >= {run_length} "
                    "consecutive points spanning both sides required), "
                    "suggesting a mixture of two distributions."
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rule registry. Each entry: (rule_id, rule_name, min_points, detector-callable).
# The detector closes over its parameters via functools-free lambdas below.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _RuleSpec:
    rule_id: str
    rule_name: str
    min_points: int
    detector: Callable[[ControlChart, NDArray[np.float64], str, str, RuleSet], list[Violation]]


def _spec(
    rule_id: str,
    rule_name: str,
    min_points: int,
    fn: Callable[..., list[Violation]],
    **kwargs: object,
) -> _RuleSpec:
    """Bind detector kwargs into a uniform (chart, z, id, name, set) signature."""

    def detector(
        chart: ControlChart,
        z: NDArray[np.float64],
        rid: str,
        rname: str,
        rset: RuleSet,
    ) -> list[Violation]:
        if fn is _rule_beyond_3sigma:
            return _rule_beyond_3sigma(chart, z, rid, rset)
        return fn(chart, z, rule_id=rid, rule_name=rname, rule_set=rset, **kwargs)  # type: ignore[arg-type]

    return _RuleSpec(rule_id=rule_id, rule_name=rule_name, min_points=min_points, detector=detector)


WESTERN_ELECTRIC_RULES: tuple[_RuleSpec, ...] = (
    _spec("WE1", "One point beyond 3 sigma", 1, _rule_beyond_3sigma),
    _spec(
        "WE2", "Two of three consecutive points beyond 2 sigma (same side)", 3,
        _rule_k_of_m_beyond_zone, k=2, m=3, zone_threshold=2.0,
    ),
    _spec(
        "WE3", "Four of five consecutive points beyond 1 sigma (same side)", 5,
        _rule_k_of_m_beyond_zone, k=4, m=5, zone_threshold=1.0,
    ),
    _spec(
        "WE4", "Eight consecutive points on one side of the center line", 8,
        _rule_run_one_side, run_length=8,
    ),
)


NELSON_RULES: tuple[_RuleSpec, ...] = (
    _spec("N1", "One point beyond 3 sigma", 1, _rule_beyond_3sigma),
    _spec("N2", "Nine consecutive points on one side of the center line", 9,
          _rule_run_one_side, run_length=9),
    _spec("N3", "Six consecutive points steadily increasing or decreasing", 6,
          _rule_trend, run_length=6),
    _spec("N4", "Fourteen consecutive points alternating up and down", 14,
          _rule_alternating, run_length=14),
    _spec("N5", "Two of three consecutive points beyond 2 sigma (same side)", 3,
          _rule_k_of_m_beyond_zone, k=2, m=3, zone_threshold=2.0),
    _spec("N6", "Four of five consecutive points beyond 1 sigma (same side)", 5,
          _rule_k_of_m_beyond_zone, k=4, m=5, zone_threshold=1.0),
    _spec("N7", "Fifteen consecutive points within 1 sigma of the center line", 15,
          _rule_within_1sigma, run_length=15),
    _spec("N8", "Eight consecutive points with none within 1 sigma", 8,
          _rule_outside_1sigma, run_length=8),
)


# Logically identical tests appearing in BOTH rule families. Under RuleSet.ALL
# these detect the same physical event with the same indices and direction, so
# reporting both would double-count. The pairs are (WECO id, Nelson id):
#
#   WE1 <-> N1   one point beyond 3 sigma
#   WE2 <-> N5   two of three consecutive beyond 2 sigma, same side
#   WE3 <-> N6   four of five consecutive beyond 1 sigma, same side
#
# WE4 (8 on one side) and N2 (9 on one side) are deliberately NOT paired: the
# run lengths differ, so they are different claims about the data and each must
# be reported on its own (FORMULAS.md Sec. 3.8).
_EQUIVALENT_RULES: dict[str, str] = {
    "WE1": "N1",
    "WE2": "N5",
    "WE3": "N6",
}


def _deduplicate_equivalents(violations: list[Violation]) -> list[Violation]:
    """Merge logically identical WE/Nelson violations into single events.

    Two violations merge only when they are the SAME physical event: an
    equivalent rule pair, identical affected indices, and identical direction.
    The Western Electric record is kept as primary (it is the older, more widely
    cited formulation) and the Nelson ID is preserved in
    ``equivalent_rule_ids``, so attribution to either family is still possible
    and ``n_violations`` counts physical events rather than detectors.

    Anything not matching an equivalent pair is passed through untouched.
    """
    by_key: dict[tuple[str, tuple[int, ...], str], Violation] = {}
    order: list[tuple[str, tuple[int, ...], str]] = []
    passthrough: list[Violation] = []

    # Canonical key: the WECO id of the pair (or the id itself), indices, dir.
    nelson_to_weco = {v: k for k, v in _EQUIVALENT_RULES.items()}

    for v in violations:
        canonical = (
            v.rule_id
            if v.rule_id in _EQUIVALENT_RULES
            else nelson_to_weco.get(v.rule_id)
        )
        if canonical is None:
            passthrough.append(v)
            continue
        key = (canonical, v.indices, v.direction.value)
        if key not in by_key:
            by_key[key] = v
            order.append(key)
        else:
            existing = by_key[key]
            # Prefer the WECO record as primary; record the partner ID.
            weco_id, nelson_id = canonical, _EQUIVALENT_RULES[canonical]
            primary = existing if existing.rule_id == weco_id else v
            partner_id = nelson_id if primary.rule_id == weco_id else weco_id
            by_key[key] = Violation(
                rule_id=primary.rule_id,
                rule_name=primary.rule_name,
                rule_set=primary.rule_set,
                indices=primary.indices,
                direction=primary.direction,
                explanation=primary.explanation,
                equivalent_rule_ids=(partner_id,),
            )

    merged = [by_key[k] for k in order] + passthrough
    merged.sort(key=lambda v: (v.indices[0] if v.indices else -1, v.rule_id))
    return merged


def _rules_for(rule_set: RuleSet) -> list[tuple[_RuleSpec, RuleSet]]:
    if rule_set is RuleSet.WECO:
        return [(r, RuleSet.WECO) for r in WESTERN_ELECTRIC_RULES]
    if rule_set is RuleSet.NELSON:
        return [(r, RuleSet.NELSON) for r in NELSON_RULES]
    # ALL: both sets, each under its own provenance and IDs.
    return [(r, RuleSet.WECO) for r in WESTERN_ELECTRIC_RULES] + [
        (r, RuleSet.NELSON) for r in NELSON_RULES
    ]


def evaluate_rules(
    chart: ControlChart,
    *,
    rule_set: RuleSet = RuleSet.NELSON,
) -> RuleEvaluation:
    """Evaluate the selected run-rule set against a control chart.

    Parameters
    ----------
    chart:
        A :class:`~app.spc.limits.ControlChart`. Symmetric charts (X-bar, I) get
        the full rule set; asymmetric dispersion charts (R, S, MR) get only the
        beyond-limit test (WE1/N1), matching standard SPC practice
        (FORMULAS.md §3.5).
    rule_set:
        :class:`RuleSet` to evaluate (default NELSON).

    Returns
    -------
    RuleEvaluation
        All violations (overlaps preserved), plus rules skipped for insufficient
        data. ``is_stable`` is True iff there are no violations.
    """
    z = _signed_zone_levels(chart)
    npts = chart.points.size

    violations: list[Violation] = []
    skipped: list[RuleSkipped] = []

    for spec, provenance in _rules_for(rule_set):
        is_beyond_rule = spec.rule_name.startswith("One point beyond")
        # Asymmetric charts: only the beyond-limit test applies.
        if not chart.is_symmetric and not is_beyond_rule:
            continue
        if npts < spec.min_points:
            skipped.append(
                RuleSkipped(
                    rule_id=spec.rule_id,
                    rule_name=spec.rule_name,
                    reason=(
                        f"Needs at least {spec.min_points} points; chart has "
                        f"{npts}."
                    ),
                )
            )
            continue
        found = spec.detector(chart, z, spec.rule_id, spec.rule_name, provenance)
        violations.extend(found)

    if rule_set is RuleSet.ALL:
        # Both families ran; merge the logically identical detections so
        # n_violations counts physical events, not detectors (FORMULAS.md 3.8).
        violations = _deduplicate_equivalents(violations)

    return RuleEvaluation(violations=tuple(violations), skipped=tuple(skipped))
