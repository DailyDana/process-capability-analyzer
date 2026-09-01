"""Pure chart model — the single source of truth for what a chart displays.

A :class:`ChartModel` is assembled directly from the M3
:class:`~app.spc.limits.ControlChart` (limits, point values, original indices,
beyond-limit flags) and :class:`~app.spc.rules.RuleEvaluation` (violations). It
performs **no statistical computation** — it only *reads* the M3 output and
organizes it for display. This module deliberately does not import matplotlib,
so the Streamlit UI (M6) can consume the model without any plotting dependency.

Point display classification
----------------------------
Each plotted point gets a set of display tags derived (not recomputed) from M3:

* ``BEYOND_LIMITS`` — the M3 ``beyond_limits[i]`` flag is True.
* ``RULE_VIOLATION`` — the point's ORIGINAL index appears in >= 1 M3 violation.
* ``NORMAL`` — neither of the above.

A point can be both BEYOND_LIMITS and RULE_VIOLATION (e.g. WE1/N1 fires on the
same point). Overlapping violations are preserved: :attr:`PointView.violation_ids`
lists every rule that implicates the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.spc.limits import ChartType, ControlChart
from app.spc.rules import RuleEvaluation, Violation

__all__ = [
    "PointClass",
    "PointView",
    "ChartModel",
    "ChartPairModel",
    "build_chart_model",
    "build_pair_model",
    "x_axis_label_for",
]


class PointClass(str, Enum):
    """Display classification of a plotted point (derived from M3, not recomputed)."""

    NORMAL = "normal"
    BEYOND_LIMITS = "beyond_limits"
    RULE_VIOLATION = "rule_violation"


def x_axis_label_for(chart_type: ChartType) -> str:
    """Return the conventional X-axis label for a chart type."""
    if chart_type in (ChartType.XBAR_R, ChartType.XBAR_S, ChartType.R, ChartType.S):
        return "Subgroup"
    return "Observation"


def _y_axis_label_for(chart_type: ChartType, unit: Optional[str]) -> str:
    """Return a Y-axis label including the measurement unit when available."""
    base = {
        ChartType.XBAR_R: "Subgroup mean",
        ChartType.XBAR_S: "Subgroup mean",
        ChartType.R: "Subgroup range",
        ChartType.S: "Subgroup std dev",
        ChartType.INDIVIDUALS: "Individual value",
        ChartType.MOVING_RANGE: "Moving range",
    }[chart_type]
    if unit:
        return f"{base} ({unit})"
    return base


def _default_title_for(chart_type: ChartType) -> str:
    return {
        ChartType.XBAR_R: "X-bar chart (X-bar / R)",
        ChartType.XBAR_S: "X-bar chart (X-bar / S)",
        ChartType.R: "R chart",
        ChartType.S: "S chart",
        ChartType.INDIVIDUALS: "Individuals chart (I-MR)",
        ChartType.MOVING_RANGE: "Moving range chart (I-MR)",
    }[chart_type]


@dataclass(frozen=True, slots=True)
class PointView:
    """Display view of one plotted point.

    Attributes
    ----------
    position:
        Plot position (0-based, along the chart's own x-axis).
    original_index:
        Original observation/subgroup index from the M3 chart ``point_indices``
        — what the user's row maps to. May differ from ``position`` when
        observations were excluded (I-MR with missing values) or for the MR
        chart (aligned to the second member of each pair).
    value:
        The plotted statistic value (verbatim from M3).
    beyond_limits:
        M3 beyond-limit flag for this point.
    violation_ids:
        Rule IDs (e.g. ``("N1", "N2")``) whose violation includes this point's
        original index. Empty when the point is in no violation. Order-preserving,
        de-duplicated; overlapping violations are all represented.
    classes:
        Set of :class:`PointClass` display tags.
    """

    position: int
    original_index: int
    value: float
    beyond_limits: bool
    violation_ids: tuple[str, ...]
    classes: frozenset[PointClass]

    @property
    def is_ooc(self) -> bool:
        """True if the point is out of control by any signal (beyond or rule)."""
        return (
            PointClass.BEYOND_LIMITS in self.classes
            or PointClass.RULE_VIOLATION in self.classes
        )


@dataclass(frozen=True, slots=True)
class ChartModel:
    """Complete, plot-agnostic description of one control chart.

    Every numeric value here is taken verbatim from the M3 ``ControlChart`` /
    ``RuleEvaluation`` — this class computes no statistics. It is the single
    source of truth the renderer and the UI both consume.
    """

    chart_type: ChartType
    title: str
    x_label: str
    y_label: str
    unit: Optional[str]

    # --- verbatim from the M3 ControlChart ---
    center_line: float
    ucl: float
    lcl: float
    sigma_points: float
    is_symmetric: bool
    points: tuple[PointView, ...]

    # --- verbatim from the M3 RuleEvaluation (structured, separate from visuals) ---
    violations: tuple[Violation, ...]
    is_stable: bool

    # Zone boundaries for optional shading (derived from CL +/- k*sigma_points).
    zone_1sigma: tuple[float, float]
    zone_2sigma: tuple[float, float]

    @property
    def positions(self) -> list[int]:
        return [p.position for p in self.points]

    @property
    def values(self) -> list[float]:
        return [p.value for p in self.points]

    @property
    def original_indices(self) -> list[int]:
        return [p.original_index for p in self.points]

    def points_in_class(self, cls: PointClass) -> list[PointView]:
        return [p for p in self.points if cls in p.classes]

    @property
    def ooc_points(self) -> list[PointView]:
        return [p for p in self.points if p.is_ooc]

    def violations_for_original_index(self, original_index: int) -> list[Violation]:
        """All violations whose affected indices include ``original_index``."""
        return [v for v in self.violations if original_index in v.indices]

    def as_dict(self) -> dict[str, object]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "unit": self.unit,
            "center_line": self.center_line,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "sigma_points": self.sigma_points,
            "is_symmetric": self.is_symmetric,
            "points": [
                {
                    "position": p.position,
                    "original_index": p.original_index,
                    "value": p.value,
                    "beyond_limits": p.beyond_limits,
                    "violation_ids": list(p.violation_ids),
                    "classes": sorted(c.value for c in p.classes),
                }
                for p in self.points
            ],
            "violations": [v.as_dict() for v in self.violations],
            "is_stable": self.is_stable,
        }


def _violations_by_original_index(
    violations: tuple[Violation, ...],
) -> dict[int, list[str]]:
    """Map each original observation index -> ordered, de-duplicated rule IDs."""
    mapping: dict[int, list[str]] = {}
    for v in violations:
        for idx in v.indices:
            bucket = mapping.setdefault(idx, [])
            if v.rule_id not in bucket:
                bucket.append(v.rule_id)
    return mapping


def build_chart_model(
    chart: ControlChart,
    evaluation: Optional[RuleEvaluation] = None,
    *,
    unit: Optional[str] = None,
    title: Optional[str] = None,
) -> ChartModel:
    """Assemble a :class:`ChartModel` from an M3 chart (+ optional evaluation).

    Parameters
    ----------
    chart:
        The M3 :class:`~app.spc.limits.ControlChart`. Its limits, point values,
        and original indices are used verbatim.
    evaluation:
        Optional :class:`~app.spc.rules.RuleEvaluation` for this chart. When
        provided, its violations drive the RULE_VIOLATION tagging and are carried
        (structured) on the model. When ``None``, no rule tags are applied and
        ``is_stable`` reflects only whether any beyond-limit point exists.
    unit:
        Measurement unit for the Y-axis label, when available.
    title:
        Optional chart title override.

    Notes
    -----
    This function performs NO statistical computation. It reads M3 fields and
    organizes them; every numeric value is passed through unchanged.
    """
    violations: tuple[Violation, ...] = tuple(evaluation.violations) if evaluation else ()
    viol_map = _violations_by_original_index(violations)

    points: list[PointView] = []
    for pos in range(chart.points.size):
        orig = int(chart.point_indices[pos])
        beyond = bool(chart.beyond_limits[pos])
        vids = tuple(viol_map.get(orig, ()))
        classes: set[PointClass] = set()
        if beyond:
            classes.add(PointClass.BEYOND_LIMITS)
        if vids:
            classes.add(PointClass.RULE_VIOLATION)
        if not classes:
            classes.add(PointClass.NORMAL)
        points.append(
            PointView(
                position=pos,
                original_index=orig,
                value=float(chart.points[pos]),
                beyond_limits=beyond,
                violation_ids=vids,
                classes=frozenset(classes),
            )
        )

    if evaluation is not None:
        is_stable = evaluation.is_stable
    else:
        is_stable = not bool(chart.beyond_limits.any())

    cl = chart.center_line
    s = chart.sigma_points
    return ChartModel(
        chart_type=chart.chart_type,
        title=title or _default_title_for(chart.chart_type),
        x_label=x_axis_label_for(chart.chart_type),
        y_label=_y_axis_label_for(chart.chart_type, unit),
        unit=unit,
        center_line=cl,
        ucl=chart.ucl,
        lcl=chart.lcl,
        sigma_points=s,
        is_symmetric=chart.is_symmetric,
        points=tuple(points),
        violations=violations,
        is_stable=is_stable,
        zone_1sigma=(cl - s, cl + s),
        zone_2sigma=(cl - 2.0 * s, cl + 2.0 * s),
    )


@dataclass(frozen=True, slots=True)
class ChartPairModel:
    """A location + dispersion pair of :class:`ChartModel` (X-bar/R, X-bar/S, I-MR)."""

    location: ChartModel
    dispersion: ChartModel

    def as_dict(self) -> dict[str, object]:
        return {
            "location": self.location.as_dict(),
            "dispersion": self.dispersion.as_dict(),
        }


def build_pair_model(
    pair,
    *,
    location_eval: Optional[RuleEvaluation] = None,
    dispersion_eval: Optional[RuleEvaluation] = None,
    unit: Optional[str] = None,
) -> ChartPairModel:
    """Build a :class:`ChartPairModel` from an M3 :class:`~app.spc.limits.ChartPair`.

    Each chart's rule evaluation is optional and attached independently, so the
    dispersion chart correctly carries only its (beyond-limit) violations.
    """
    return ChartPairModel(
        location=build_chart_model(pair.location, location_eval, unit=unit),
        dispersion=build_chart_model(pair.dispersion, dispersion_eval, unit=unit),
    )
