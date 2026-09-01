"""Adapter from the SPC rule engine to the capability layer's StabilitySummary.

This is the seam that lets capability consume an SPC stability verdict without
the capability package importing the spc package (dependency stays one-way:
services -> {spc, capability}).

Stability distinction enforced here (see FORMULAS.md §3.1):

* **Capability** answers "can the stable process meet specifications?"
* **SPC** answers "is the process statistically stable/predictable?"

A process is only treated as stable when the location AND dispersion charts are
both free of violations — an out-of-control dispersion chart invalidates the
location-chart limits, so both must pass.
"""

from __future__ import annotations

from typing import Iterable

from app.capability.report import StabilitySummary
from app.spc.limits import ChartPair
from app.spc.rules import RuleEvaluation, RuleSet, evaluate_rules

__all__ = [
    "stability_from_evaluations",
    "stability_from_chart_pair",
]


def stability_from_evaluations(
    evaluations: Iterable[RuleEvaluation],
    *,
    detail: str = "",
) -> StabilitySummary:
    """Combine one or more :class:`RuleEvaluation` into a :class:`StabilitySummary`.

    The process is stable only if *every* evaluation is stable. Triggered rule
    IDs are unioned across evaluations (order-preserving, de-duplicated).
    """
    evals = list(evaluations)
    all_stable = all(e.is_stable for e in evals)
    total_violations = sum(len(e.violations) for e in evals)

    seen: list[str] = []
    for e in evals:
        for rid in e.triggered_rule_ids:
            if rid not in seen:
                seen.append(rid)

    return StabilitySummary(
        is_stable=all_stable,
        triggered_rule_ids=tuple(seen),
        n_violations=total_violations,
        detail=detail,
    )


def stability_from_chart_pair(
    pair: ChartPair,
    *,
    rule_set: RuleSet = RuleSet.NELSON,
) -> StabilitySummary:
    """Evaluate both charts of a :class:`ChartPair` and summarize stability.

    The location chart gets the full rule set; the dispersion chart gets only the
    beyond-limit test (enforced inside :func:`evaluate_rules` via the chart's
    symmetry flag). Both must be in control for the process to be stable.
    """
    loc_eval = evaluate_rules(pair.location, rule_set=rule_set)
    disp_eval = evaluate_rules(pair.dispersion, rule_set=rule_set)
    detail = ""
    if not (loc_eval.is_stable and disp_eval.is_stable):
        parts = []
        if not loc_eval.is_stable:
            parts.append(
                f"location chart: {', '.join(loc_eval.triggered_rule_ids)}"
            )
        if not disp_eval.is_stable:
            parts.append(
                f"dispersion chart: {', '.join(disp_eval.triggered_rule_ids)}"
            )
        detail = "; ".join(parts)
    return stability_from_evaluations([loc_eval, disp_eval], detail=detail)
