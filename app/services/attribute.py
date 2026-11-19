"""Attribute-chart orchestrator -- the single entry point of its page.

validate raw columns -> chart (:mod:`app.spc.attribute`) -> run rules ->
interpretation. Ordinary user errors come back as ``ok=False`` with
plain-language messages; no row is ever dropped silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from app.interpret.attribute import AttributeThresholds, interpret_attribute
from app.interpret.models import InterpretationReport
from app.spc.attribute import AttributeChart, AttributeKind, attribute_chart, evaluate_attribute_rules
from app.spc.rules import RuleEvaluation, RuleSet

__all__ = ["AttributeRequest", "AttributeOutcome", "run_attribute"]


@dataclass(frozen=True, slots=True)
class AttributeRequest:
    """``sizes`` is one value per sample; ``constant_size`` one value for all.

    ``exclude`` holds 1-based sample numbers, as the user sees them.
    """

    kind: AttributeKind
    counts: tuple[object, ...]
    sizes: Optional[tuple[object, ...]] = None
    constant_size: Optional[float] = None
    n_baseline: Optional[int] = None
    exclude: tuple[int, ...] = ()
    center: Optional[float] = None
    rule_set: RuleSet = RuleSet.NELSON
    thresholds: AttributeThresholds = field(default_factory=AttributeThresholds)


@dataclass(frozen=True, slots=True)
class AttributeOutcome:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    chart: Optional[AttributeChart] = None
    evaluation: Optional[RuleEvaluation] = None
    interpretation: Optional[InterpretationReport] = None


def _blank(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()


def _number(v: object) -> Optional[float]:
    try:
        f = float(v.strip().replace(",", ".")) if isinstance(v, str) else float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _column(raw: Sequence[object], name: str, errors: list[str]) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    bad: list[str] = []
    for i, v in enumerate(raw):
        if _blank(v):
            out.append(None)
            continue
        f = _number(v)
        if f is None:
            bad.append(f"row {i + 1} ({v!r})")
        out.append(f)
    if bad:
        shown = ", ".join(bad[:8]) + (f" (and {len(bad) - 8} more)" if len(bad) > 8 else "")
        errors.append(f"{name} must be numbers: {shown}.")
    return out


def run_attribute(req: AttributeRequest) -> AttributeOutcome:
    """Validate, chart, evaluate and interpret. Never raises for ordinary user input."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = _column(req.counts, "Counts", errors)
    sizes = _column(req.sizes, "Sample sizes", errors) if req.sizes is not None else None
    if sizes is not None and len(sizes) != len(counts):
        errors.append(f"{len(counts)} counts but {len(sizes)} sample sizes.")
    if errors:
        return AttributeOutcome(False, tuple(errors))

    # Trailing blank rows (an empty tail in a spreadsheet) are dropped; a blank
    # anywhere else would shift every later sample, so it is an error.
    n = len(counts)
    while n > 0 and counts[n - 1] is None and (sizes is None or sizes[n - 1] is None):
        n -= 1
    if n < len(counts):
        warnings.append(f"{len(counts) - n} blank row(s) at the end ignored.")
    counts = counts[:n]
    sizes = sizes[:n] if sizes is not None else None
    gaps = [i + 1 for i in range(n) if counts[i] is None or (sizes is not None and sizes[i] is None)]
    if gaps:
        shown = ", ".join(str(g) for g in gaps[:8]) + (f" (and {len(gaps) - 8} more)" if len(gaps) > 8 else "")
        return AttributeOutcome(
            False,
            (f"Missing count or sample size in row(s) {shown}. Fill or remove them: samples are in "
             "time order and cannot be skipped.",),
            tuple(warnings),
        )
    if n < 2:
        return AttributeOutcome(False, (f"At least 2 samples are required (found {n}).",), tuple(warnings))

    size_arg: Union[Sequence[float], float, None]
    if sizes is not None:
        size_arg = [float(s) for s in sizes]  # type: ignore[arg-type]
    else:
        size_arg = req.constant_size
    try:
        chart = attribute_chart(
            req.kind,
            [float(c) for c in counts],  # type: ignore[arg-type]
            size_arg,
            n_baseline=req.n_baseline,
            exclude=tuple(int(e) - 1 for e in req.exclude),
            center=req.center,
        )
    except ValueError as exc:
        return AttributeOutcome(False, (str(exc),), tuple(warnings))
    if chart.rate == 0.0:
        warnings.append(
            "No nonconformities in the baseline: the limits collapse onto 0 and the chart says nothing yet."
        )
    evaluation = evaluate_attribute_rules(chart, req.rule_set)
    return AttributeOutcome(
        True, (), tuple(warnings), chart, evaluation, interpret_attribute(chart, evaluation, req.thresholds)
    )
