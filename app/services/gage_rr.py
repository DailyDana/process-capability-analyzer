"""Gage R&R orchestrator -- the single entry point the Gage R&R page calls.

Sequence (mirrors :mod:`app.services.analysis`)::

    validate long-format input -> balanced study -> range-chart consistency
    -> variance components (ANOVA or Average & Range) -> summary table / ndc
    -> interpretation

It composes :mod:`app.msa` and computes nothing itself. Ordinary user errors
(unbalanced data, a study outside the Average & Range tables, a bad
tolerance) come back as ``ok=False`` with plain-language messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.interpret.models import InterpretationReport
from app.msa.anova import DEFAULT_ALPHA_POOL, anova_gage_rr
from app.msa.interpret import GageThresholds, interpret_gage
from app.msa.metrics import DEFAULT_K, number_of_distinct_categories, summary_rows
from app.msa.models import GageMethod, GageRRResult, GageStudy
from app.msa.ranges import range_check
from app.msa.validation import build_gage_study
from app.msa.xbar_r import average_range_gage_rr

__all__ = ["GageRequest", "GageOutcome", "analyze_study", "run_gage_study"]


@dataclass(frozen=True, slots=True)
class GageRequest:
    """Everything needed for one Gage R&R analysis.

    ``tolerance`` is ``USL - LSL`` (optional); %Tolerance is reported only when
    it is given.
    """

    parts: tuple[Any, ...]
    operators: tuple[Any, ...]
    values: tuple[Any, ...]
    trials: Optional[tuple[Any, ...]] = None
    method: GageMethod = GageMethod.ANOVA
    k: float = DEFAULT_K
    tolerance: Optional[float] = None
    alpha_pool: float = DEFAULT_ALPHA_POOL
    unit: Optional[str] = None
    thresholds: GageThresholds = field(default_factory=GageThresholds)


@dataclass(frozen=True, slots=True)
class GageOutcome:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    study: Optional[GageStudy] = None
    result: Optional[GageRRResult] = None
    interpretation: Optional[InterpretationReport] = None


def analyze_study(
    study: GageStudy,
    method: GageMethod = GageMethod.ANOVA,
    k: float = DEFAULT_K,
    tolerance: Optional[float] = None,
    alpha_pool: float = DEFAULT_ALPHA_POOL,
) -> GageRRResult:
    """Run one method on a validated study (raises ``ValueError`` on bad options)."""
    anova = detail = None
    if method is GageMethod.ANOVA:
        anova = anova_gage_rr(study, alpha_pool=alpha_pool)
        components = anova.components
    elif method is GageMethod.AVERAGE_RANGE:
        components, detail = average_range_gage_rr(study)
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"Unknown method {method!r}.")
    return GageRRResult(
        method=method,
        components=components,
        rows=summary_rows(components, k=k, tolerance=tolerance),
        ndc=number_of_distinct_categories(components),
        k=k,
        tolerance=tolerance,
        range_check=range_check(study),
        n_parts=study.n_parts,
        n_operators=study.n_operators,
        n_trials=study.n_trials,
        anova=anova,
        average_range=detail,
    )


def run_gage_study(req: GageRequest) -> GageOutcome:
    """Validate, analyse and interpret. Never raises for ordinary user input."""
    v = build_gage_study(req.parts, req.operators, req.values, req.trials, unit=req.unit)
    if not v.ok or v.study is None:
        return GageOutcome(ok=False, errors=v.errors, warnings=v.warnings)
    try:
        result = analyze_study(
            v.study,
            method=req.method,
            k=req.k,
            tolerance=req.tolerance,
            alpha_pool=req.alpha_pool,
        )
    except ValueError as exc:
        return GageOutcome(ok=False, errors=(str(exc),), warnings=v.warnings, study=v.study)
    return GageOutcome(
        ok=True,
        warnings=v.warnings,
        study=v.study,
        result=result,
        interpretation=interpret_gage(result, req.thresholds),
    )


def columns_to_request(
    parts: Sequence[Any],
    operators: Sequence[Any],
    values: Sequence[Any],
    trials: Optional[Sequence[Any]] = None,
    **options: Any,
) -> GageRequest:
    """Convenience constructor from column sequences (e.g. DataFrame columns)."""
    return GageRequest(
        parts=tuple(parts),
        operators=tuple(operators),
        values=tuple(values),
        trials=tuple(trials) if trials is not None else None,
        **options,
    )
