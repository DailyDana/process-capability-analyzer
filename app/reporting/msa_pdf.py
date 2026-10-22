"""PDF reports for the measurement system analyses (Gage R&R, Type-1 gauge).

Serialization only, like :mod:`app.reporting.pdf_report` (whose styles and
helpers are reused): every number comes from the result objects, charts come
from the existing renderers, nothing is recalculated.

Dynamic text is XML-escaped before it reaches a ReportLab ``Paragraph``: the
interpretation texts contain ``<``, ``>`` and ``&`` (e.g. "<10%", "Gage R&R"),
which ReportLab would otherwise parse as markup. Only characters available in
the standard Helvetica font are used (">=", "R-bar", not their Unicode forms).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

from app.charts.gage_model import build_components_model, build_interaction_model, build_range_model
from app.charts.gage_render import render_components, render_interaction, render_range_by_operator
from app.interpret.models import InterpretationReport
from app.msa.anova import AnovaResult, AnovaTable
from app.msa.models import GageRRResult, GageStudy
from app.msa.xbar_r import AverageRangeDetail
from app.reporting.bundle import ExportMeta
from app.reporting.gage_report import methodology_rows
from app.reporting.pdf_report import (
    _BODY,
    _BRAND,
    _CELL,
    _H1,
    _H2,
    _SMALL,
    _TABLE_STYLE,
    _chart_image,
    _num,
    _styles,
)

__all__ = ["gage_rr_pdf_bytes", "pdf_text", "plain_text", "build_document"]

_TITLE = ParagraphStyle("MsaTitle", parent=_styles["Title"], textColor=_BRAND)
_SUB = ParagraphStyle("MsaSub", parent=_styles["Heading2"], textColor=colors.HexColor("#444444"))


_GLYPHS = (("≥", ">="), ("≤", "<="), ("×", "x"), ("σ", "sigma"), ("R̄", "R-bar"), ("−", "-"), ("—", "-"))


def plain_text(text: object) -> str:
    """For plain table cells (drawn verbatim): replace glyphs Helvetica lacks."""
    s = str(text)
    for a, b in _GLYPHS:
        s = s.replace(a, b)
    return s


def pdf_text(text: object) -> str:
    """For Paragraph markup: plain_text plus XML escaping of < > &."""
    return escape(plain_text(text))


def _pe(text: object, style: ParagraphStyle = _BODY) -> Paragraph:
    return Paragraph(pdf_text(text), style)


def _fmt(v: object, decimals: int = 4) -> str:
    if isinstance(v, bool) or v is None:
        return "n/a" if v is None else ("yes" if v else "no")
    if isinstance(v, float):
        return _num(v, decimals)
    return str(v)


def _kv(rows: list[tuple[str, object]]) -> Table:
    data = [["Field", "Value"]] + [[plain_text(k), _pe(_fmt(v), _CELL)] for k, v in rows]
    t = Table(data, colWidths=[62 * mm, 108 * mm], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    return t


def _grid(header: list[str], rows: list[list[object]], widths: list[float]) -> Table:
    data: list[list[object]] = [[plain_text(h) for h in header]]
    data += [[_pe(_fmt(c), _CELL) if isinstance(c, str) and len(c) > 18 else plain_text(_fmt(c)) for c in row] for row in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    return t


def _cover(story: list, title: str, meta: ExportMeta, key_rows: list[tuple[str, object]],
           interp: Optional[InterpretationReport], caution: Optional[str]) -> None:
    story.append(Spacer(1, 8 * mm))
    story.append(_pe(title, _TITLE))
    bits = [b for b in (meta.characteristic, meta.part_id) if b]
    if bits:
        story.append(_pe(" - ".join(bits), _SUB))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", color=_BRAND, thickness=1.2))
    story.append(Spacer(1, 4 * mm))
    rows = meta.as_rows() + [("Report generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))] + key_rows
    story.append(_kv(rows))
    if interp is not None:
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("<b>Executive summary</b>", _H2))
        story.append(_pe(interp.headline))
    if caution:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(f"<font color='#B00020'><b>Caution:</b> {pdf_text(caution)}</font>", _BODY))


def _interpretation(story: list, interp: InterpretationReport, first_no: int) -> None:
    story.append(_pe(f"{first_no}. Statistical Conclusions", _H1))
    story.append(_pe("Factual statements from the deterministic rule-based interpreter; no advice.", _SMALL))
    story.append(Spacer(1, 2 * mm))
    data: list[list[object]] = [["Code", "Severity", "Conclusion"]]
    data += [[_pe(c.code, _CELL), c.severity.value, _pe(c.text, _CELL)] for c in interp.conclusions]
    t = Table(data, colWidths=[42 * mm, 18 * mm, 110 * mm], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 5 * mm))
    story.append(_pe(f"{first_no + 1}. Engineering Recommendations", _H1))
    story.append(_pe("Suggested actions; engineering judgment, not statistical fact.", _SMALL))
    story.append(Spacer(1, 2 * mm))
    if not interp.recommendations:
        story.append(_pe("No specific actions are indicated."))
        return
    data = [["Code", "Trigger", "Action"]]
    data += [[_pe(r.code, _CELL), _pe(r.trigger_code, _CELL), _pe(r.text, _CELL)] for r in interp.recommendations]
    t = Table(data, colWidths=[40 * mm, 38 * mm, 92 * mm], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    story.append(t)


def build_document(story: list, title: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
        bottomMargin=16 * mm, title=title, author="SPC / Process Capability Analyzer",
    )
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gage R&R
# ---------------------------------------------------------------------------
def _anova_rows(t: AnovaTable) -> list[list[object]]:
    return [[r.source, r.df, _num(r.ss, 5), _num(r.ms, 5), _num(r.f, 3), _num(r.p, 3)] for r in t.rows]


def _measurement_table(study: GageStudy) -> Table:
    o, r = study.n_operators, study.n_trials
    if o * r <= 12:
        header = ["Part"] + [f"{op} / {k + 1}" for op in study.operator_labels for k in range(r)]
        rows = [
            [part] + [_num(float(study.values[i, j, k]), 4) for j in range(o) for k in range(r)]
            for i, part in enumerate(study.part_labels)
        ]
        first = 22.0
        rest = (170.0 - first) / (o * r)
        return _grid(header, rows, [first] + [rest] * (o * r))
    rows = [
        [part, op, k + 1, _num(float(study.values[i, j, k]), 4)]
        for i, part in enumerate(study.part_labels)
        for j, op in enumerate(study.operator_labels)
        for k in range(r)
    ]
    return _grid(["Part", "Operator", "Trial", "Value"], rows, [40, 50, 30, 50])


def gage_rr_pdf_bytes(
    study: GageStudy,
    result: GageRRResult,
    interpretation: InterpretationReport,
    meta: ExportMeta = ExportMeta(),
    warnings: tuple[str, ...] = (),
    acceptable_max: float = 10.0,
    marginal_max: float = 30.0,
) -> bytes:
    """Gage R&R report: summary, tables, charts, interpretation, raw data."""
    title = "Gage R&R (Measurement System Analysis) Report"
    story: list = []
    key = [
        ("Method", methodology_rows(result)[0][1]),
        ("Study", f"{result.n_parts} parts x {result.n_operators} operators x {result.n_trials} trials"),
        ("%GRR of study variation", f"{_num(result.pct_grr_study_var, 2)}%"),
        ("%GRR of tolerance", f"{_num(result.pct_grr_tolerance, 2)}%" if result.pct_grr_tolerance is not None else None),
        ("ndc", result.ndc),
        ("Ranges above UCL", len(result.range_check.out_of_control)),
    ]
    caution = None
    if not result.range_check.in_control:
        caution = (
            "one or more part x operator ranges exceed the range-chart UCL; the repeatability "
            "estimate and everything derived from it are not yet reliable."
        )
    _cover(story, title, meta, key, interpretation, caution)
    for w in warnings:
        story.append(Spacer(1, 1 * mm))
        story.append(_pe(f"Data warning: {w}", _SMALL))
    story.append(PageBreak())

    story.append(_pe("1. Method", _H1))
    story.append(_kv(methodology_rows(result)))
    story.append(Spacer(1, 2 * mm))
    story.append(_pe(
        "Acceptance bands (AIAG convention, configurable): %GRR < 10% acceptable, 10-30% may be "
        "acceptable depending on the application, > 30% unacceptable; ndc >= 5. The bands apply to "
        "%Study Var (a ratio of standard deviations), not to %Contribution. Formulas: docs/FORMULAS.md "
        "section 6.", _SMALL))

    story.append(_pe("2. Gage R&R", _H1))
    rows = [
        [r.source, _num(r.variance, 5), _num(r.sd, 5), _num(r.study_var, 5), _num(r.pct_contribution, 2),
         _num(r.pct_study_var, 2), _num(r.pct_tolerance, 2)]
        for r in result.rows
    ]
    story.append(_grid(
        ["Source", "Variance", "StdDev", f"Study Var ({result.k:g} SD)", "% Contrib.", "% Study Var", "% Tol."],
        rows, [34, 22, 22, 26, 22, 22, 22],
    ))
    if isinstance(result.anova, AnovaResult):
        a = result.anova
        story.append(_pe("ANOVA, full model (with operator x part interaction)", _H2))
        story.append(_grid(["Source", "DF", "SS", "MS", "F", "P"], _anova_rows(a.full), [40, 14, 30, 30, 28, 28]))
        if a.reduced is not None:
            story.append(_pe(
                f"ANOVA, interaction pooled into repeatability (p = {_num(a.interaction_p, 3)} > {a.alpha_pool:g})", _H2))
            story.append(_grid(["Source", "DF", "SS", "MS", "F", "P"], _anova_rows(a.reduced), [40, 14, 30, 30, 28, 28]))
    elif isinstance(result.average_range, AverageRangeDetail):
        d = result.average_range
        story.append(_pe("Average & Range intermediates", _H2))
        story.append(_kv([
            ("R-bar-bar (average range)", d.r_bar_bar), ("X-bar diff (operator averages)", d.x_bar_diff),
            ("Rp (part averages)", d.r_p), ("K1 / K2 / K3", f"{d.k1} / {d.k2} / {d.k3}"),
        ]))
    story.append(PageBreak())

    story.append(_pe("3. Charts", _H1))
    figs: list[Figure] = [
        render_components(build_components_model(result, acceptable_max, marginal_max)),
        render_range_by_operator(build_range_model(study, result)),
        render_interaction(build_interaction_model(study)),
    ]
    for fig in figs:
        story.append(_chart_image(fig))
        story.append(Spacer(1, 3 * mm))
        plt.close(fig)
    story.append(PageBreak())

    _interpretation(story, interpretation, first_no=4)
    story.append(PageBreak())
    story.append(_pe("6. Measurements", _H1))
    story.append(_pe(
        "Columns are operator / trial." if study.n_operators * study.n_trials <= 12 else "One row per reading.",
        _SMALL))
    story.append(_measurement_table(study))
    return build_document(story, title)
