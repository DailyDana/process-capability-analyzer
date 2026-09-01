"""Professional PDF report via ReportLab (M7).

Consumes an :class:`~app.reporting.bundle.AnalysisBundle` and (optionally) the
M4 :class:`~app.charts.model.ChartPairModel`. Charts are rendered through the
EXISTING M4 renderer (``app.charts.render``) and embedded as PNG at their native
aspect ratio — there is no second chart implementation and NO statistic is
recalculated here.

Report structure (per the M7 specification):

  Cover / Executive summary
  1. Data & Methodology
  2. Descriptive Statistics
  3. SPC Analysis (embedded M4 charts + violations table)
  4. Capability & Performance (within vs overall, strictly separated)
  5. Statistical Conclusions   (verbatim M5 objects)
  6. Engineering Recommendations (verbatim M5 objects, visually separate)
  7. Technical Appendix (references the controlled methodology document)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.charts import render as chart_render
from app.charts.model import ChartPairModel
from app.reporting.bundle import AnalysisBundle

__all__ = ["build_pdf_bytes", "write_pdf", "NA"]

NA = "n/a"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
_BRAND = colors.HexColor("#1F3B6F")
_LIMIT_RED = colors.HexColor("#B00020")
_OK_GREEN = colors.HexColor("#2C7A2C")

_styles = getSampleStyleSheet()
_H1 = ParagraphStyle("H1x", parent=_styles["Heading1"], textColor=_BRAND, spaceBefore=14)
_H2 = ParagraphStyle("H2x", parent=_styles["Heading2"], textColor=_BRAND, spaceBefore=10)
_BODY = ParagraphStyle("Bodyx", parent=_styles["BodyText"], alignment=TA_LEFT, leading=13)
_SMALL = ParagraphStyle("Smallx", parent=_styles["BodyText"], fontSize=8, leading=10,
                        textColor=colors.HexColor("#444444"))
_CELL = ParagraphStyle("Cellx", parent=_styles["BodyText"], fontSize=8, leading=10)

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), _BRAND),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C2D0")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FB")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])


def _num(x: Optional[float], decimals: int = 4) -> str:
    """Display formatting; sentinels readable; missing -> NA. Display-only."""
    if x is None:
        return NA
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return NA
    if fx != fx:
        return "NaN"
    if fx == float("inf"):
        return "+inf"
    if fx == float("-inf"):
        return "-inf"
    return f"{fx:,.{decimals}f}"


def _p(text: str, style: ParagraphStyle = _BODY) -> Paragraph:
    return Paragraph(text, style)


def _kv_table(rows: list[tuple[str, object]], col_widths=(55 * mm, 105 * mm)) -> Table:
    data = [["Field", "Value"]] + [[str(k), _p(str(v), _CELL)] for k, v in rows]
    t = Table(data, colWidths=list(col_widths), repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    return t


def _chart_image(fig, max_width_mm: float = 170.0) -> Image:
    """Embed a matplotlib figure as a PNG Image at its NATIVE aspect ratio.

    The PNG comes from the existing M4 exporter at 200 dpi (print-readable).
    Width is fitted to the printable area; height follows the true pixel aspect
    ratio so the chart is never stretched or distorted.
    """
    png = chart_render.figure_to_png_bytes(fig, dpi=200)
    reader = ImageReader(io.BytesIO(png))
    px_w, px_h = reader.getSize()
    width = max_width_mm * mm
    height = width * (px_h / px_w)
    return Image(io.BytesIO(png), width=width, height=height)


# ---------------------------------------------------------------------------
# Section builders (each consumes bundle fields verbatim)
# ---------------------------------------------------------------------------
def _cover(story: list, bundle: AnalysisBundle) -> None:
    meta = bundle.meta
    cap = bundle.capability
    interp = bundle.interpretation

    story.append(Spacer(1, 8 * mm))
    story.append(_p("SPC / Process Capability Analysis Report",
                    ParagraphStyle("Title2", parent=_styles["Title"], textColor=_BRAND)))
    subtitle_bits = [b for b in (meta.characteristic, meta.part_id) if b]
    if subtitle_bits:
        story.append(_p(" — ".join(subtitle_bits),
                        ParagraphStyle("Sub", parent=_styles["Heading2"],
                                       textColor=colors.HexColor("#444444"))))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", color=_BRAND, thickness=1.2))
    story.append(Spacer(1, 4 * mm))

    stable_txt = (
        "NOT ASSESSED" if cap.stability is None
        else ("STABLE" if cap.stability.is_stable else "UNSTABLE")
    )
    cpk = cap.capability.cpk if cap.capability else None
    rows: list[tuple[str, object]] = [
        ("Report generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("Sample size (N)", cap.n_observations),
        ("Process stability", stable_txt),
        ("Cpk (within)", _num(cpk, 3)),
        ("Ppk (overall)", _num(cap.performance.cpk, 3)),
    ]
    rows = meta.as_rows() + rows
    story.append(_kv_table(rows))
    story.append(Spacer(1, 5 * mm))

    story.append(_p("<b>Executive summary</b>", _H2))
    # Headline comes verbatim from the deterministic M5 engine.
    story.append(_p(interp.headline, _BODY))
    if cap.stability is not None and not cap.stability.is_stable:
        story.append(Spacer(1, 2 * mm))
        story.append(_p(
            "<font color='#B00020'><b>Stability caution:</b> the process is out "
            "of statistical control. Capability and performance indices in this "
            "report are reported unchanged but must not be interpreted as "
            "reliable evidence of future process performance until the special "
            "causes are removed.</font>", _BODY))
    story.append(Spacer(1, 2 * mm))
    story.append(_p(
        f"Overall interpretation severity: "
        f"<b>{interp.max_severity.value.upper()}</b>. Full statistical "
        "conclusions and engineering recommendations appear in sections 5-6.",
        _SMALL))


def _methodology(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("1. Data &amp; Methodology", _H1))
    cap = bundle.capability
    spec = cap.spec

    rows: list[tuple[str, object]] = []
    if bundle.methodology is not None:
        rows.extend(bundle.methodology.as_rows())
    else:
        rows.append(("Observations analysed", cap.n_observations))
        if cap.n_subgroups is not None:
            rows.append(("Number of subgroups", cap.n_subgroups))

    rows.extend([
        ("USL", _num(spec.usl)),
        ("LSL", _num(spec.lsl)),
        ("Nominal / target", _num(spec.nominal)),
        ("Measurement unit", spec.unit or NA),
        ("Specification type", "two-sided" if spec.is_two_sided else "one-sided"),
    ])
    story.append(_kv_table(rows))

    if bundle.methodology is not None and bundle.methodology.exclusion_notes:
        story.append(Spacer(1, 3 * mm))
        story.append(_p("<b>Excluded observations</b>", _H2))
        for note in bundle.methodology.exclusion_notes:
            story.append(_p(f"• {note}", _SMALL))


def _descriptive(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("2. Descriptive Statistics", _H1))
    d = bundle.descriptive
    if d is None:
        story.append(_p("Descriptive statistics were not provided.", _BODY))
        return
    unit = bundle.meta.unit or ""
    u = f" {unit}" if unit else ""
    rows = [
        ("N", d.n),
        (f"Mean{u}", _num(d.mean)),
        (f"Median{u}", _num(d.median)),
        (f"Standard deviation (s){u}", _num(d.std_dev)),
        ("Variance", _num(d.variance)),
        (f"Minimum{u}", _num(d.minimum)),
        (f"Maximum{u}", _num(d.maximum)),
        (f"Range{u}", _num(d.value_range)),
    ]
    story.append(_kv_table(rows))
    story.append(Spacer(1, 2 * mm))
    story.append(_p(
        "Sample statistics with Bessel's correction (ddof = 1). Values are "
        "rounded for display; all calculations retain full precision.", _SMALL))


def _spc(story: list, bundle: AnalysisBundle,
         chart_models: Optional[ChartPairModel]) -> None:
    story.append(_p("3. SPC Analysis", _H1))
    cap = bundle.capability

    if cap.stability is None:
        story.append(_p(
            "Process stability was not assessed for this analysis.", _BODY))
    elif cap.stability.is_stable:
        story.append(_p(
            "<font color='#2C7A2C'><b>Process is statistically STABLE</b> — no "
            "control-rule violations detected.</font>", _BODY))
    else:
        story.append(_p(
            f"<font color='#B00020'><b>Process is statistically UNSTABLE</b> — "
            f"{cap.stability.n_violations} violation(s); rules triggered: "
            f"{', '.join(cap.stability.triggered_rule_ids)}.</font>", _BODY))

    if chart_models is not None:
        for model in (chart_models.location, chart_models.dispersion):
            story.append(Spacer(1, 3 * mm))
            story.append(_p(f"<b>{model.title}</b>", _H2))
            rows = [
                ("Center line (CL)", _num(model.center_line)),
                ("Upper control limit (UCL)", _num(model.ucl)),
                ("Lower control limit (LCL)", _num(model.lcl)),
                ("Points plotted", len(model.points)),
                ("Points beyond limits",
                 sum(1 for pt in model.points if pt.beyond_limits)),
            ]
            story.append(_kv_table(rows))
            story.append(Spacer(1, 2 * mm))
            fig = chart_render.render_chart(model)
            try:
                story.append(_chart_image(fig))
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)

    story.append(Spacer(1, 3 * mm))
    story.append(_p("<b>Rule violations</b>", _H2))
    if not bundle.violations:
        story.append(_p("No SPC rule violations detected.", _BODY))
    else:
        header = ["Rule ID", "Rule name", "Rule set", "Direction",
                  "Affected obs.", "Explanation"]
        data: list[list] = [header]
        for v in bundle.violations:
            data.append([
                v.rule_id,
                _p(v.rule_name, _CELL),
                v.rule_set.value,
                v.direction.value,
                _p(", ".join(str(i) for i in v.indices), _CELL),
                _p(v.explanation, _CELL),
            ])
        t = Table(data,
                  colWidths=[14 * mm, 34 * mm, 20 * mm, 16 * mm, 24 * mm, 62 * mm],
                  repeatRows=1)
        t.setStyle(_TABLE_STYLE)
        story.append(t)


def _capability(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("4. Capability &amp; Performance", _H1))
    cap = bundle.capability

    if cap.stability is not None and not cap.stability.is_stable:
        story.append(_p(
            "<font color='#B00020'><b>Important:</b> the process is unstable; "
            "the indices below do not constitute trustworthy evidence of future "
            "performance (see section 3).</font>", _BODY))
        story.append(Spacer(1, 2 * mm))

    story.append(_p("<b>Within-subgroup capability (short-term)</b>", _H2))
    story.append(_p(
        "Estimated from within-subgroup variation only — the process's inherent "
        "capability if it held still. Question answered: <i>can the stable "
        "process meet the specification?</i>", _SMALL))
    if cap.capability is None:
        story.append(_p(
            "No within-subgroup sigma was available for this data regime; "
            "Cp/Cpk are not reported.", _BODY))
    else:
        c = cap.capability
        rows = [
            ("sigma (within)", _num(cap.sigma_within, 6)),
            ("Cp", _num(c.cp, 3)),
            ("Cpk", _num(c.cpk, 3)),
            ("Cpu", _num(c.cpu, 3)),
            ("Cpl", _num(c.cpl, 3)),
        ]
        if cap.sigma_level_within is not None:
            rows += [
                ("Sigma level / Z-bench (within)", _num(cap.sigma_level_within.z_bench, 3)),
                ("DPMO (within)", _num(cap.sigma_level_within.dpmo, 1)),
            ]
        story.append(_kv_table(rows))

    story.append(Spacer(1, 3 * mm))
    story.append(_p("<b>Overall process performance (long-term)</b>", _H2))
    story.append(_p(
        "Estimated from the total variation of all observations, including "
        "drift and shifts between subgroups. Question answered: <i>how did the "
        "process actually perform over the whole study?</i>", _SMALL))
    p = cap.performance
    rows = [
        ("sigma (overall)", _num(cap.sigma_overall, 6)),
        ("Pp", _num(p.cp, 3)),
        ("Ppk", _num(p.cpk, 3)),
        ("Ppu", _num(p.cpu, 3)),
        ("Ppl", _num(p.cpl, 3)),
        ("Sigma level / Z-bench (overall)", _num(cap.sigma_level_overall.z_bench, 3)),
        ("DPMO (overall)", _num(cap.sigma_level_overall.dpmo, 1)),
    ]
    story.append(_kv_table(rows))

    if cap.cpk_ci is not None:
        ci = cap.cpk_ci
        story.append(Spacer(1, 3 * mm))
        story.append(_p("<b>Cpk confidence interval</b>", _H2))
        story.append(_p(
            f"Approximate {ci.confidence * 100:.0f}% interval: "
            f"[{_num(ci.lower, 3)}, {_num(ci.upper, 3)}] "
            "(Bissell 1990 / Montgomery large-sample normal approximation — "
            "<b>not an exact interval</b>).", _BODY))
        for cav in ci.caveats:
            story.append(_p(f"• {cav}", _SMALL))

    if cap.warnings:
        story.append(Spacer(1, 3 * mm))
        story.append(_p("<b>Reliability warnings</b>", _H2))
        sev_hex = {
            "critical": "#B00020",
            "warning": "#B45F04",
            "notice": "#7F6000",
            "info": "#2C7A2C",
        }
        for w in cap.warnings:
            hex_color = sev_hex.get(w.severity.value, "#000000")
            story.append(_p(
                f"<font color='{hex_color}'>"
                f"[{w.severity.value.upper()}] <b>{w.code}</b></font> — {w.message}",
                _SMALL))

    if cap.normality is not None:
        story.append(Spacer(1, 2 * mm))
        verdict = {True: "not rejected", False: "REJECTED",
                   None: "not testable"}[cap.normality.normal]
        story.append(_p(
            f"<b>Normality:</b> {verdict}"
            + (f" (Shapiro-Wilk p = {_num(cap.normality.shapiro_p, 4)})"
               if cap.normality.shapiro_p is not None else "")
            + (f". {cap.normality.note}" if cap.normality.note else ""),
            _SMALL))


def _conclusions(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("5. Statistical Conclusions", _H1))
    story.append(_p(
        "Factual statements produced by the deterministic rule-based "
        "interpretation engine. These describe what the numbers say; they "
        "contain no advice.", _SMALL))
    story.append(Spacer(1, 2 * mm))
    header = ["Code", "Topic", "Severity", "Conclusion"]
    data: list[list] = [header]
    for c in bundle.interpretation.conclusions:
        data.append([
            _p(c.code, _CELL),
            c.topic.value,
            c.severity.value,
            _p(c.text, _CELL),
        ])
    t = Table(data, colWidths=[30 * mm, 26 * mm, 16 * mm, 98 * mm], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    story.append(t)


def _recommendations(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("6. Engineering Recommendations", _H1))
    story.append(_p(
        "Suggested actions tied to the identified conditions. These are "
        "engineering judgments, <b>not statistical facts</b>; each cites the "
        "conclusion that triggered it.", _SMALL))
    story.append(Spacer(1, 2 * mm))
    if not bundle.interpretation.recommendations:
        story.append(_p("No specific actions are indicated.", _BODY))
        return
    header = ["Code", "Trigger", "Severity", "Recommended action"]
    data: list[list] = [header]
    for r in bundle.interpretation.recommendations:
        data.append([
            _p(r.code, _CELL),
            _p(r.trigger_code, _CELL),
            r.severity.value,
            _p(r.text, _CELL),
        ])
    t = Table(data, colWidths=[34 * mm, 28 * mm, 16 * mm, 92 * mm], repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    story.append(t)


def _appendix(story: list, bundle: AnalysisBundle) -> None:
    story.append(_p("7. Technical Appendix", _H1))
    story.append(_p(
        "The complete controlled methodology — every formula, constant, "
        "assumption, convention flag, and literature reference used by this "
        "analysis — is maintained in the project document "
        "<b>docs/FORMULAS.md</b>. That document is the single authoritative "
        "source; this appendix summarizes the conventions relevant to reading "
        "this report and deliberately does not duplicate the formula "
        "derivations.", _BODY))
    story.append(Spacer(1, 2 * mm))
    bullets = [
        "<b>Cp/Cpk vs Pp/Ppk:</b> Cp/Cpk use within-subgroup (short-term) sigma "
        "(R-bar/d2, S-bar/c4, pooled, or MR-bar/d2 depending on regime); Pp/Ppk "
        "use the overall (long-term) sample standard deviation of all pooled "
        "observations. They are never interchanged.",
        "<b>Stability before capability:</b> SPC control rules are evaluated "
        "before capability. Capability indices computed on an unstable process "
        "are reported but explicitly caveated.",
        "<b>Control limits:</b> Shewhart 3-sigma limits from the published "
        "control-chart constants (Montgomery Appendix VI / ASTM STP-15D), "
        "computed by the validated limit calculators.",
        "<b>Run rules:</b> Western Electric (4 rules) and Nelson (8 rules) are "
        "implemented as distinct sets with their canonical run lengths "
        "(e.g. WE4 = 8 points on one side; Nelson N2 = 9).",
        "<b>Cpk confidence interval:</b> Bissell (1990) / Montgomery eq. 8.24 "
        "large-sample normal approximation; approximate by construction and "
        "invalid under non-normality or instability (caveats attached).",
        "<b>Capability bands</b> (inadequate &lt; 1.00 ≤ marginal &lt; 1.33 ≤ "
        "capable &lt; 1.67 ≤ excellent) are configurable engineering "
        "interpretation conventions, not universal statistical laws.",
        "<b>Zero variation:</b> indices are reported as +inf sentinels with a "
        "gauge-resolution warning; they are never silently dropped.",
    ]
    for b in bullets:
        story.append(_p(f"• {b}", _SMALL))
        story.append(Spacer(1, 1 * mm))
    story.append(Spacer(1, 3 * mm))
    story.append(_p(
        "Primary references: Montgomery, <i>Introduction to Statistical Quality "
        "Control</i>, 7th ed. (Wiley, 2013); AIAG <i>SPC Reference Manual</i>, "
        "2nd ed. (2005); NIST/SEMATECH <i>e-Handbook of Statistical Methods</i> "
        "§6; David &amp; Nagaraja, <i>Order Statistics</i>, 3rd ed. (Wiley, "
        "2003); Bissell, \"How reliable is your capability index?\", "
        "<i>Applied Statistics</i> 39(3), 1990.", _SMALL))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def build_pdf_bytes(
    bundle: AnalysisBundle,
    chart_models: Optional[ChartPairModel] = None,
    *,
    title: str = "SPC / Process Capability Analysis Report",
) -> bytes:
    """Build the full PDF report and return it as bytes.

    ``chart_models`` is the M4 :class:`ChartPairModel`; when provided, both
    charts are rendered via the existing M4 renderer and embedded. When absent
    the SPC section still reports limits/violations from the bundle.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="SPC / Process Capability Analyzer",
    )
    story: list = []
    _cover(story, bundle)
    story.append(PageBreak())
    _methodology(story, bundle)
    story.append(Spacer(1, 4 * mm))
    _descriptive(story, bundle)
    story.append(PageBreak())
    _spc(story, bundle, chart_models)
    story.append(PageBreak())
    _capability(story, bundle)
    story.append(PageBreak())
    _conclusions(story, bundle)
    story.append(Spacer(1, 5 * mm))
    _recommendations(story, bundle)
    story.append(PageBreak())
    _appendix(story, bundle)

    doc.build(story)
    return buf.getvalue()


def write_pdf(
    bundle: AnalysisBundle,
    path: str,
    chart_models: Optional[ChartPairModel] = None,
) -> str:
    """Build and save the PDF report to ``path``. Returns the path."""
    data = build_pdf_bytes(bundle, chart_models)
    with open(path, "wb") as f:
        f.write(data)
    return path
