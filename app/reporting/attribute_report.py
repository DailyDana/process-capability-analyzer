"""CSV export of an attribute control chart (full precision, no recomputation)."""

from __future__ import annotations

import csv
import io
from typing import Optional

from app.interpret.models import InterpretationReport
from app.reporting.bundle import ExportMeta
from app.reporting.csv_report import _num as _csv_num
from app.spc.attribute import AttributeChart, AttributeKind
from app.spc.rules import RuleEvaluation

__all__ = ["summary_rows", "sample_rows", "attribute_csv_string"]

_RATE_LABEL = {
    AttributeKind.P: "p-bar (fraction nonconforming)",
    AttributeKind.NP: "p-bar (fraction nonconforming)",
    AttributeKind.C: "c-bar (nonconformities per sample)",
    AttributeKind.U: "u-bar (nonconformities per unit)",
}


def summary_rows(chart: AttributeChart) -> list[tuple[str, object]]:
    """(label, value) rows shared by the CSV and the UI."""
    rows: list[tuple[str, object]] = [
        ("Chart", chart.kind.value),
        ("Samples", chart.n_samples),
        ("Centre line from", "given standard" if chart.center_is_standard else f"{chart.n_baseline} sample(s)"),
        (_RATE_LABEL[chart.kind], chart.rate),
        ("Centre line", chart.center_line),
        ("Sample size", float(chart.sizes[0]) if chart.constant_size
         else f"varies, {chart.sizes.min():g} to {chart.sizes.max():g}"),
    ]
    if chart.constant_size:
        rows += [("LCL", float(chart.lcl[0])), ("UCL", float(chart.ucl[0]))]
    else:
        rows += [("LCL", "per sample, see the sample table"), ("UCL", "per sample, see the sample table")]
    rows.append(("Samples beyond a limit", int(chart.beyond_limits.sum())))
    return rows


def sample_rows(chart: AttributeChart, evaluation: Optional[RuleEvaluation] = None) -> list[dict[str, object]]:
    """One row per sample: what was plotted, its limits, and the rules it triggered."""
    rules: dict[int, list[str]] = {}
    if evaluation is not None:
        for v in evaluation.violations:
            for i in v.indices:
                for rid in v.all_rule_ids:
                    if rid not in rules.setdefault(i, []):
                        rules[i].append(rid)
    return [
        {
            "Sample": i + 1,
            "Count": float(chart.counts[i]),
            "Size": float(chart.sizes[i]),
            "Plotted": float(chart.points[i]),
            "LCL": float(chart.lcl[i]),
            "CL": chart.center_line,
            "UCL": float(chart.ucl[i]),
            "In baseline": bool(chart.in_baseline[i]),
            "Rules": " ".join(rules.get(i, [])),
        }
        for i in range(chart.n_samples)
    ]


def attribute_csv_string(
    chart: AttributeChart,
    evaluation: Optional[RuleEvaluation] = None,
    interpretation: Optional[InterpretationReport] = None,
    meta: ExportMeta = ExportMeta(),
    warnings: tuple[str, ...] = (),
) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["# Metadata"])
    w.writerow(["Field", "Value"])
    for k, v in meta.as_rows():
        w.writerow([k, v])
    w.writerow([])
    w.writerow(["# Attribute control chart"])
    w.writerow(["Field", "Value"])
    for k, v in summary_rows(chart):
        w.writerow([k, _csv_num(v) if isinstance(v, float) else v])
    if warnings:
        w.writerow([])
        w.writerow(["# Data warnings"])
        for m in warnings:
            w.writerow([m])
    if evaluation is not None:
        w.writerow([])
        w.writerow(["# Signals"])
        w.writerow(["Rule", "Samples", "Direction", "Explanation"])
        for v in evaluation.violations:
            w.writerow(["/".join(v.all_rule_ids), " ".join(str(i + 1) for i in v.indices),
                        v.direction.value, v.explanation])
    if interpretation is not None:
        w.writerow([])
        w.writerow(["# Interpretation"])
        w.writerow(["Headline", interpretation.headline])
        w.writerow(["Kind", "Code", "Severity", "Text"])
        for c in interpretation.conclusions:
            w.writerow(["conclusion", c.code, c.severity.value, c.text])
        for x in interpretation.recommendations:
            w.writerow(["recommendation", x.code, x.severity.value, x.text])
    w.writerow([])
    w.writerow(["# Samples"])
    rows = sample_rows(chart, evaluation)
    w.writerow(list(rows[0]))
    for r in rows:
        w.writerow([_csv_num(v) if isinstance(v, float) else v for v in r.values()])
    return buf.getvalue()
