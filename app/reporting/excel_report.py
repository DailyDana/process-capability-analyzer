"""Professional multi-sheet .xlsx export of an :class:`AnalysisBundle`.

Sheets: Summary, Measurements, Statistics, Capability, SPC, Violations,
Interpretation. All values come from the tested analysis layer — NO statistic is
recomputed in Excel (per the M5 requirement). A few Excel formulas are used only
for trivially re-derivable display quantities (e.g. count of violations) and are
clearly not substitutes for the Python-computed statistics.

Missing/unavailable values are written explicitly as ``"n/a"``. The workbook is
returned as an openpyxl ``Workbook`` (and can be saved or streamed to bytes).
"""

from __future__ import annotations

import io
from typing import Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.reporting.bundle import AnalysisBundle

__all__ = ["build_workbook", "write_xlsx", "workbook_to_bytes", "NA"]

NA = "n/a"

_HEADER_FILL = PatternFill("solid", fgColor="1F3B6F")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_TITLE_FONT = Font(bold=True, size=13, color="1F3B6F")
_SECTION_FONT = Font(bold=True, size=11, color="1F3B6F")
_SEV_FILL = {
    "critical": PatternFill("solid", fgColor="F4CCCC"),
    "warning": PatternFill("solid", fgColor="FCE5CD"),
    "notice": PatternFill("solid", fgColor="FFF2CC"),
    "info": PatternFill("solid", fgColor="D9EAD3"),
}


def _num(x: Optional[float]) -> object:
    """Return a real number for Excel, or the ``NA`` string when missing.

    Non-finite sentinels are written as readable strings ("inf"/"-inf"/"NaN")
    because Excel cannot store them as numbers.
    """
    if x is None:
        return NA
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return NA
    if fx != fx:
        return "NaN"
    if fx == float("inf"):
        return "inf"
    if fx == float("-inf"):
        return "-inf"
    return fx


def _s(x: object) -> object:
    return NA if x is None else x


def _header_row(ws: Worksheet, row: int, headers: list[str]) -> None:
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="left")


def _autosize(ws: Worksheet, max_width: int = 60) -> None:
    for col in ws.columns:
        length = 0
        letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value is not None:
                length = max(length, len(str(cell.value)))
        ws.column_dimensions[letter].width = min(max(length + 2, 10), max_width)


def _build_summary(ws: Worksheet, bundle: AnalysisBundle) -> None:
    cap = bundle.capability
    spec = cap.spec
    ws.cell(row=1, column=1, value="SPC / Process Capability Report").font = _TITLE_FONT
    ws.cell(row=2, column=1, value=bundle.interpretation.headline).alignment = Alignment(
        wrap_text=True
    )
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=4)

    row = 4
    ws.cell(row=row, column=1, value="Metadata").font = _SECTION_FONT
    row += 1
    _header_row(ws, row, ["Field", "Value"])
    row += 1
    for label, value in bundle.meta.as_rows():
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=_s(value))
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Specification").font = _SECTION_FONT
    row += 1
    _header_row(ws, row, ["Field", "Value"])
    row += 1
    for label, value in (
        ("USL", _num(spec.usl)), ("LSL", _num(spec.lsl)),
        ("Nominal", _num(spec.nominal)), ("Unit", _s(spec.unit)),
    ):
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=value)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Key results").font = _SECTION_FONT
    row += 1
    _header_row(ws, row, ["Metric", "Value"])
    row += 1
    cpk = cap.capability.cpk if cap.capability else None
    key = [
        ("Cpk (within)", _num(cpk)),
        ("Ppk (overall)", _num(cap.performance.cpk)),
        ("sigma_within", _num(cap.sigma_within)),
        ("sigma_overall", _num(cap.sigma_overall)),
        ("Stable", _s(cap.stability.is_stable) if cap.stability else NA),
    ]
    for label, value in key:
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=value)
        row += 1
    _autosize(ws)


def _build_measurements(ws: Worksheet, bundle: AnalysisBundle) -> None:
    # "Original row" is the index identity used everywhere else in the report
    # (violations, charts, UI preview): the row number in the SOURCE data, not
    # the position in the cleaned array. See AnalysisBundle.measurement_indices.
    _header_row(ws, 1, ["Original row", "Subgroup", "Value"])
    if bundle.measurements is not None:
        labels = bundle.subgroup_labels
        indices = bundle.measurement_indices
        for i, val in enumerate(bundle.measurements):
            original = indices[i] if indices is not None and i < len(indices) else i
            ws.cell(row=i + 2, column=1, value=int(original))
            ws.cell(row=i + 2, column=2,
                    value=_s(labels[i]) if labels is not None else NA)
            ws.cell(row=i + 2, column=3, value=_num(val))
    else:
        ws.cell(row=2, column=1, value="(measurements not provided)")
    _autosize(ws)


def _build_statistics(ws: Worksheet, bundle: AnalysisBundle) -> None:
    _header_row(ws, 1, ["Statistic", "Value"])
    d = bundle.descriptive
    rows: list[tuple[str, object]]
    if d is not None:
        rows = [
            ("n", _num(d.n)), ("mean", _num(d.mean)), ("median", _num(d.median)),
            ("variance", _num(d.variance)), ("std_dev", _num(d.std_dev)),
            ("minimum", _num(d.minimum)), ("maximum", _num(d.maximum)),
            ("range", _num(d.value_range)),
        ]
    else:
        rows = [("(not provided)", NA)]
    rows += [
        ("sigma_within (short-term)", _num(bundle.capability.sigma_within)),
        ("sigma_overall (long-term)", _num(bundle.capability.sigma_overall)),
        ("grand_mean", _num(bundle.capability.mean)),
    ]
    for i, (label, value) in enumerate(rows, start=2):
        ws.cell(row=i, column=1, value=label)
        ws.cell(row=i, column=2, value=value)
    _autosize(ws)


def _build_capability(ws: Worksheet, bundle: AnalysisBundle) -> None:
    cap = bundle.capability
    _header_row(ws, 1, ["Index", "Family", "Value", "Notes"])
    r = 2

    def put(name: str, family: str, value: object, note: str = "") -> None:
        nonlocal r
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=family)
        ws.cell(row=r, column=3, value=value)
        ws.cell(row=r, column=4, value=note)
        r += 1

    if cap.capability is not None:
        c = cap.capability
        put("Cp", "within (short-term)", _num(c.cp), "potential; requires 2-sided spec")
        put("Cpk", "within (short-term)", _num(c.cpk), "actual, centering-aware")
        put("Cpu", "within (short-term)", _num(c.cpu))
        put("Cpl", "within (short-term)", _num(c.cpl))
    else:
        put("Cp/Cpk", "within (short-term)", NA, "no within-subgroup sigma available")
    p = cap.performance
    put("Pp", "overall (long-term)", _num(p.cp), "potential; requires 2-sided spec")
    put("Ppk", "overall (long-term)", _num(p.cpk), "actual, centering-aware")
    put("Ppu", "overall (long-term)", _num(p.cpu))
    put("Ppl", "overall (long-term)", _num(p.cpl))

    if cap.sigma_level_within is not None:
        put("Z_bench", "within (short-term)", _num(cap.sigma_level_within.z_bench))
        put("DPMO", "within (short-term)", _num(cap.sigma_level_within.dpmo))
    put("Z_bench", "overall (long-term)", _num(cap.sigma_level_overall.z_bench))
    put("DPMO", "overall (long-term)", _num(cap.sigma_level_overall.dpmo))

    if cap.cpk_ci is not None:
        ci = cap.cpk_ci
        put("Cpk CI lower", "within", _num(ci.lower),
            f"{ci.confidence * 100:.0f}% approx (not exact)")
        put("Cpk CI upper", "within", _num(ci.upper),
            f"{ci.confidence * 100:.0f}% approx (not exact)")
    _autosize(ws)


def _build_spc(ws: Worksheet, bundle: AnalysisBundle) -> None:
    cap = bundle.capability
    _header_row(ws, 1, ["Field", "Value"])
    rows: list[tuple[str, object]]
    if cap.stability is not None:
        st = cap.stability
        rows = [
            ("Stability assessed", True),
            ("Is stable", _s(st.is_stable)),
            ("Number of violations", _num(st.n_violations)),
            ("Triggered rules", ", ".join(st.triggered_rule_ids) or NA),
        ]
    else:
        rows = [("Stability assessed", False)]
    if cap.normality is not None:
        rows += [("Normal (test)", _s(cap.normality.normal)),
                 ("Shapiro-Wilk p", _num(cap.normality.shapiro_p))]
    for i, (label, value) in enumerate(rows, start=2):
        ws.cell(row=i, column=1, value=label)
        ws.cell(row=i, column=2, value=value)
    _autosize(ws)


def _build_violations(ws: Worksheet, bundle: AnalysisBundle) -> None:
    _header_row(ws, 1, ["Rule ID", "Rule name", "Rule set", "Direction", "Affected indices"])
    if bundle.violations:
        for i, v in enumerate(bundle.violations, start=2):
            ws.cell(row=i, column=1, value=v.rule_id)
            ws.cell(row=i, column=2, value=v.rule_name)
            ws.cell(row=i, column=3, value=v.rule_set.value)
            ws.cell(row=i, column=4, value=v.direction.value)
            ws.cell(row=i, column=5, value=" ".join(str(x) for x in v.indices))
    else:
        ws.cell(row=2, column=1, value="(no violations)")
    _autosize(ws)


def _build_interpretation(ws: Worksheet, bundle: AnalysisBundle) -> None:
    interp = bundle.interpretation
    ws.cell(row=1, column=1, value="Headline").font = _SECTION_FONT
    ws.cell(row=1, column=2, value=interp.headline).alignment = Alignment(wrap_text=True)

    row = 3
    ws.cell(row=row, column=1, value="Statistical conclusions").font = _SECTION_FONT
    row += 1
    _header_row(ws, row, ["Code", "Topic", "Severity", "Statement"])
    row += 1
    for c in interp.conclusions:
        ws.cell(row=row, column=1, value=c.code)
        ws.cell(row=row, column=2, value=c.topic.value)
        sev = ws.cell(row=row, column=3, value=c.severity.value)
        if c.severity.value in _SEV_FILL:
            sev.fill = _SEV_FILL[c.severity.value]
        ws.cell(row=row, column=4, value=c.text).alignment = Alignment(wrap_text=True)
        row += 1

    row += 1
    # Recommendations kept in a clearly separate block.
    ws.cell(row=row, column=1, value="Engineering recommendations").font = _SECTION_FONT
    row += 1
    _header_row(ws, row, ["Code", "Category", "Severity", "Trigger", "Action"])
    row += 1
    for rec in interp.recommendations:
        ws.cell(row=row, column=1, value=rec.code)
        ws.cell(row=row, column=2, value=rec.category.value)
        sev = ws.cell(row=row, column=3, value=rec.severity.value)
        if rec.severity.value in _SEV_FILL:
            sev.fill = _SEV_FILL[rec.severity.value]
        ws.cell(row=row, column=4, value=rec.trigger_code)
        ws.cell(row=row, column=5, value=rec.text).alignment = Alignment(wrap_text=True)
        row += 1
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 22
    ws.column_dimensions["E"].width = 70


def build_workbook(bundle: AnalysisBundle) -> Workbook:
    """Build the multi-sheet workbook from an :class:`AnalysisBundle`."""
    wb = Workbook()
    # The default sheet becomes Summary.
    ws_summary = wb.active
    ws_summary.title = "Summary"
    _build_summary(ws_summary, bundle)
    _build_measurements(wb.create_sheet("Measurements"), bundle)
    _build_statistics(wb.create_sheet("Statistics"), bundle)
    _build_capability(wb.create_sheet("Capability"), bundle)
    _build_spc(wb.create_sheet("SPC"), bundle)
    _build_violations(wb.create_sheet("Violations"), bundle)
    _build_interpretation(wb.create_sheet("Interpretation"), bundle)
    return wb


def write_xlsx(bundle: AnalysisBundle, path: str) -> str:
    """Build and save the workbook to ``path``. Returns the path."""
    wb = build_workbook(bundle)
    wb.save(path)
    return path


def workbook_to_bytes(bundle: AnalysisBundle) -> bytes:
    """Return the .xlsx as bytes (for in-memory download/streaming)."""
    wb = build_workbook(bundle)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
