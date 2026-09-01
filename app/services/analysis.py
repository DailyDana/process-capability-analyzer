"""Analysis orchestrator — the single entry point the UI calls.

Enforces the sequence-of-analysis contract established in the architecture:

    validate -> assign regime -> control charts -> STABILITY FIRST ->
    sigma estimation -> capability (with stability caveats) -> interpretation

It composes the validated M1-M5 layers and computes nothing itself. The UI calls
``run_analysis`` and renders the returned :class:`AnalysisOutcome`; it never
imports the statistical primitives directly.

Ordinary user errors (bad spec, too few points, wrong subgroup size) come back
as ``ok=False`` with a plain-language message — no tracebacks reach the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from app.capability.guards import GuardThresholds, NormalityResult, assess_normality
from app.capability.report import CapabilityReport, StabilitySummary, build_capability_report
from app.charts.model import ChartPairModel, build_pair_model
from app.data.models import DataRegime, SpecLimits
from app.data.validation import validate_spec_limits
from app.interpret.engine import interpret
from app.interpret.models import InterpretationReport
from app.interpret.thresholds import InterpretationThresholds
from app.services.data_loader import SubgroupResult, build_subgroups
from app.services.stability import stability_from_chart_pair
from app.spc.limits import ChartPair, ControlChart, imr_chart, xbar_r_chart, xbar_s_chart
from app.spc.rules import RuleEvaluation, RuleSet, evaluate_rules
from app.statistics.descriptive import DescriptiveSummary, describe
from app.statistics.variation import (
    WithinEstimator,
    grand_mean,
    sigma_overall,
    sigma_within_from_individuals,
    sigma_within_from_subgroups,
)

__all__ = [
    "ChartKind",
    "AnalysisRequest",
    "AnalysisOutcome",
    "run_analysis",
    "make_spec",
]


class ChartKind(str, Enum):
    """User-selectable control-chart family."""

    XBAR_R = "xbar_r"
    XBAR_S = "xbar_s"
    IMR = "imr"


def make_spec(
    usl: Optional[float],
    lsl: Optional[float],
    nominal: Optional[float] = None,
    unit: Optional[str] = None,
) -> tuple[Optional[SpecLimits], list[str], list[str]]:
    """Validate raw spec input and build a :class:`SpecLimits` if valid.

    Returns ``(spec_or_None, errors, warnings)`` using the M1 validation layer.
    Never raises for ordinary user input.
    """
    if usl is None and lsl is None:
        return None, [], []  # "no specification" is a legitimate state
    v = validate_spec_limits(usl, lsl, nominal)
    if not v.ok:
        return None, list(v.errors), list(v.warnings)
    return (
        SpecLimits(usl=usl, lsl=lsl, nominal=nominal, unit=unit),
        [],
        list(v.warnings),
    )


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Everything needed to run one analysis (a hashable-ish description).

    ``time_ordered`` is the user's ATTESTATION that the measurements are in
    production/time order. It is authoritative input, never inferred from the
    data (a sortedness heuristic would be both unreliable and unfalsifiable).
    When ``False`` the analysis is treated as :attr:`DataRegime.UNORDERED`:
    within-subgroup sigma has no rational-subgroup basis, so Cp/Cpk are not
    reported and the control charts are not treated as a stability verdict.

    ``original_indices`` carries the ORIGINAL source-row index of each value in
    ``values``, i.e. ``DetailedCleaning.used_indices`` from the cleaning layer.
    ``values`` has already had missing/non-numeric rows removed, so without this
    mapping every downstream index (chart points, violations, exports) would
    refer to a position in the CLEANED array and would silently point at the
    wrong source row whenever anything was excluded. When ``None`` the values
    are assumed to be positionally identical to the source rows.
    """

    values: tuple[float, ...]
    chart_kind: ChartKind
    subgroup_size: int = 5
    spec: Optional[SpecLimits] = None
    unit: Optional[str] = None
    rule_set: RuleSet = RuleSet.NELSON
    within_estimator: WithinEstimator = WithinEstimator.AUTO
    ci_alpha: float = 0.05
    time_ordered: bool = True
    original_indices: Optional[tuple[int, ...]] = None
    guard_thresholds: GuardThresholds = field(default_factory=GuardThresholds)
    interp_thresholds: InterpretationThresholds = field(
        default_factory=InterpretationThresholds
    )


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Result of a full analysis pass.

    ``ok=False`` carries a user-facing ``error`` (and no results). Warnings are
    non-fatal notes (e.g. trailing measurements excluded from subgrouping).
    """

    ok: bool
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    descriptive: Optional[DescriptiveSummary] = None
    chart_pair: Optional[ChartPair] = None
    chart_models: Optional[ChartPairModel] = None
    location_eval: Optional[RuleEvaluation] = None
    dispersion_eval: Optional[RuleEvaluation] = None
    stability: Optional[StabilitySummary] = None
    capability: Optional[CapabilityReport] = None
    interpretation: Optional[InterpretationReport] = None
    normality: Optional[NormalityResult] = None

    sigma_within: Optional[float] = None
    sigma_overall_value: Optional[float] = None
    within_estimator_used: Optional[str] = None
    n_subgroups: Optional[int] = None
    subgroup_size: Optional[int] = None
    n_used: int = 0
    data_regime: Optional[DataRegime] = None

    @property
    def all_violations(self) -> tuple:
        """Flat tuple of violations from both charts (overlaps preserved)."""
        out = []
        if self.location_eval:
            out.extend(self.location_eval.violations)
        if self.dispersion_eval:
            out.extend(self.dispersion_eval.violations)
        return tuple(out)


def _fail(msg: str) -> AnalysisOutcome:
    return AnalysisOutcome(ok=False, error=msg)


def _remap_to_original_indices(
    pair: ChartPair,
    original_indices: Sequence[int],
    *,
    subgroup_size: int,
    is_imr: bool,
) -> ChartPair:
    """Rewrite a chart pair's ``point_indices`` to ORIGINAL source-row indices.

    The chart builders index their points over the CLEANED value array. This
    maps those positions back through ``original_indices``
    (``DetailedCleaning.used_indices``) so violations, chart models and exports
    all cite the row numbers the user actually supplied.

    Semantics per chart type:

    * **I-MR** -- one point per observation, so position ``p`` maps directly to
      ``original_indices[p]``. The MR chart already aligns each moving range to
      the second member of its pair, so its existing (cleaned) indices are
      mapped the same way.
    * **X-bar/R/S** -- one point per SUBGROUP, which spans several original
      rows. The subgroup is labelled with the original index of its FIRST
      member (``original_indices[p * subgroup_size]``), the conventional
      identifier for a subgroup, and the only choice that stays a single
      integer. Callers needing the full membership can reconstruct it as
      ``original_indices[p*n : (p+1)*n]``.

    Purely an identity relabelling: no point value, limit or statistic is
    touched, so no statistical result can change.
    """
    orig = np.asarray(original_indices, dtype=int)

    def _map(chart: ControlChart) -> ControlChart:
        pos = np.asarray(chart.point_indices, dtype=int)
        if is_imr:
            safe = pos[pos < orig.size]
            if safe.size != pos.size:  # defensive: never silently mis-map
                return chart
            new_idx = orig[pos]
        else:
            starts = pos * subgroup_size
            if starts.size and int(starts.max()) >= orig.size:
                return chart
            new_idx = orig[starts]
        return replace(chart, point_indices=new_idx)

    return ChartPair(location=_map(pair.location), dispersion=_map(pair.dispersion))


def run_analysis(request: AnalysisRequest) -> AnalysisOutcome:
    """Run the full sequence-of-analysis and return a structured outcome.

    Order is deliberate and matches the documented contract: control charts and
    the SPC rule engine run BEFORE capability, so the capability report can carry
    the stability verdict rather than being quoted in isolation.
    """
    values = np.asarray(request.values, dtype=np.float64)
    n = int(values.size)
    warnings: list[str] = []

    if n < 2:
        return _fail(
            f"At least 2 valid measurements are required to run an analysis; "
            f"got {n}."
        )
    if not np.isfinite(values).all():
        return _fail(
            "The measurement series still contains missing or non-finite values. "
            "Clean the data before running the analysis."
        )

    # --- Descriptive statistics (M1) ---
    descriptive = describe(values, nan_policy="omit")

    # --- Control charts (M3) ---
    chart_pair: ChartPair
    n_subgroups: Optional[int]
    subgroup_size: Optional[int]
    n_used = n

    if request.chart_kind is ChartKind.IMR:
        if n < 2:
            return _fail("The I-MR chart needs at least 2 observations.")
        try:
            chart_pair = imr_chart(values)
        except ValueError as exc:
            return _fail(str(exc))
        n_subgroups = None
        subgroup_size = 1
    else:
        sg: SubgroupResult = build_subgroups(values, request.subgroup_size)
        if not sg.ok:
            return _fail(sg.error)
        if sg.warning:
            warnings.append(sg.warning)
        n_used = sg.n_used
        try:
            if request.chart_kind is ChartKind.XBAR_R:
                chart_pair = xbar_r_chart(list(sg.subgroups))
            else:
                chart_pair = xbar_s_chart(list(sg.subgroups))
        except ValueError as exc:
            return _fail(str(exc))
        n_subgroups = sg.n_subgroups
        subgroup_size = request.subgroup_size

    # --- Original-row index traceability (C1) -------------------------------
    # `values` is the CLEANED series; chart builders number their points 0..N-1
    # over that array. If any source row was dropped during cleaning, those
    # positions are NOT the user's row numbers. Remap `point_indices` here --
    # once, at the single point where charts are created -- so that every
    # downstream consumer that already reads `point_indices` (rules._orig,
    # ChartModel.original_index, all three exporters, the UI) reports genuine
    # ORIGINAL row indices with no further changes.
    if request.original_indices is not None:
        chart_pair = _remap_to_original_indices(
            chart_pair,
            request.original_indices,
            subgroup_size=(1 if request.chart_kind is ChartKind.IMR else request.subgroup_size),
            is_imr=(request.chart_kind is ChartKind.IMR),
        )

    # --- SPC rules + stability FIRST (M3) ---
    location_eval = evaluate_rules(chart_pair.location, rule_set=request.rule_set)
    dispersion_eval = evaluate_rules(chart_pair.dispersion, rule_set=request.rule_set)
    stability = stability_from_chart_pair(chart_pair, rule_set=request.rule_set)

    # --- Chart models for rendering (M4) ---
    chart_models = build_pair_model(
        chart_pair,
        location_eval=location_eval,
        dispersion_eval=dispersion_eval,
        unit=request.unit,
    )

    # --- Sigma estimation (M2) ---
    analysed = values[:n_used]
    if request.chart_kind is ChartKind.IMR:
        within_res = sigma_within_from_individuals(analysed)
    else:
        sg2 = build_subgroups(analysed, request.subgroup_size)
        within_res = sigma_within_from_subgroups(
            list(sg2.subgroups), estimator=request.within_estimator
        )
    sigma_w = within_res.sigma
    sigma_o = sigma_overall(analysed)

    # --- Normality (M2 guard) ---
    normality = assess_normality(analysed)

    # --- Data regime: the attestation is authoritative (never inferred) ------
    # Without a time-order attestation there is no rational subgrouping, so the
    # within-subgroup sigma has no valid interpretation and the charts are not a
    # stability verdict. We still compute both sigmas for transparency, but the
    # UNORDERED regime withholds sigma_within (and stability) from the
    # capability report, which is the single place that maps sigma -> index
    # family. No formula changes; the existing suppression path is reused.
    if request.time_ordered:
        regime = (
            DataRegime.INDIVIDUALS
            if request.chart_kind is ChartKind.IMR
            else DataRegime.SUBGROUPED
        )
    else:
        regime = DataRegime.UNORDERED
        warnings.append(
            "Measurements are NOT attested to be in production/time order, so "
            "they are treated as an unordered sample: Cp/Cpk (within-subgroup "
            "capability) are not reported because rational subgrouping cannot "
            "be assumed, and the control charts are shown for reference only "
            "rather than as a stability verdict. Pp/Ppk (overall performance) "
            "remain valid. Confirm the time order to enable full SPC analysis."
        )

    # --- Capability WITH stability attached (M2) ---
    capability: Optional[CapabilityReport] = None
    interpretation: Optional[InterpretationReport] = None
    if request.spec is not None:
        unordered = regime is DataRegime.UNORDERED
        capability = build_capability_report(
            mean=grand_mean(analysed),
            sigma_within=None if unordered else sigma_w,
            sigma_overall=sigma_o,
            spec=request.spec,
            n_observations=int(analysed.size),
            n_subgroups=None if unordered else n_subgroups,
            normality=normality,
            stability=None if unordered else stability,
            ci_alpha=request.ci_alpha,
            thresholds=request.guard_thresholds,
        )
        interpretation = interpret(capability, thresholds=request.interp_thresholds)
    else:
        warnings.append(
            "No specification limits were provided, so capability/performance "
            "indices and the engineering interpretation cannot be computed. "
            "SPC charts and stability analysis are still available."
        )

    return AnalysisOutcome(
        ok=True,
        warnings=tuple(warnings),
        descriptive=descriptive,
        chart_pair=chart_pair,
        chart_models=chart_models,
        location_eval=location_eval,
        dispersion_eval=dispersion_eval,
        # Stability is only a verdict for time-ordered data; for an unordered
        # sample the charts are reference-only and no verdict is claimed.
        stability=None if regime is DataRegime.UNORDERED else stability,
        capability=capability,
        interpretation=interpretation,
        normality=normality,
        # sigma_within is still surfaced for transparency, but it did NOT feed
        # any capability index under the UNORDERED regime.
        sigma_within=sigma_w,
        sigma_overall_value=sigma_o,
        within_estimator_used=within_res.estimator.value,
        n_subgroups=n_subgroups,
        subgroup_size=subgroup_size,
        n_used=n_used,
        data_regime=regime,
    )
