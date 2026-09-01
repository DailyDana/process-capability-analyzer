"""Tests for CSV and Excel export (app.reporting).

Validate actual serialized content — sheet names, cell values, precision,
explicit missing-value handling, determinism — not merely that a file exists.
"""

from __future__ import annotations

import io
import os
import tempfile

import pytest
from openpyxl import load_workbook

from app.interpret.engine import interpret
from app.reporting import csv_report, excel_report
from app.reporting.bundle import AnalysisBundle, ExportMeta
from app.statistics.descriptive import describe
from tests.fixtures import interpretation_cases as cases


def _bundle(rep=None, *, with_desc=True, with_meas=True):
    rep = rep or cases.stable_capable()
    interp = interpret(rep)
    desc = describe([9.8, 10.1, 10.2, 9.9, 10.0, 10.3]) if with_desc else None
    meas = [9.8, 10.1, 10.2, 9.9, 10.0, 10.3] if with_meas else None
    return AnalysisBundle(
        meta=ExportMeta(characteristic="bore_dia", unit="mm", machine="CNC-3",
                        timestamp="2026-09-01T10:00:00"),
        capability=rep, interpretation=interp, descriptive=desc, measurements=meas,
    )


class TestCsvContent:
    def test_sections_present(self) -> None:
        s = csv_report.to_csv_string(_bundle())
        for section in (
            "# Metadata", "# Specification", "# Descriptive statistics",
            "# Sigma estimates", "# Capability indices",
            "# Performance indices", "# Cpk confidence interval",
            "# SPC stability summary", "# Normality", "# SPC rule violations",
            "# Statistical conclusions", "# Engineering recommendations",
        ):
            assert section in s, f"missing section {section}"

    def test_metadata_values(self) -> None:
        s = csv_report.to_csv_string(_bundle())
        assert "bore_dia" in s
        assert "CNC-3" in s
        assert "2026-09-01T10:00:00" in s

    def test_capability_values_full_precision(self) -> None:
        rep = cases.stable_capable()
        s = csv_report.to_csv_string(_bundle(rep))
        # Cpk value must appear at full repr precision (round-trippable).
        cpk = rep.capability.cpk
        assert repr(cpk) in s

    def test_missing_values_explicit_na(self) -> None:
        # No descriptive, no measurements -> explicit n/a, not blank.
        s = csv_report.to_csv_string(_bundle(with_desc=False))
        assert csv_report.NA in s

    def test_one_sided_spec_na(self) -> None:
        s = csv_report.to_csv_string(_bundle(cases.one_sided_upper()))
        # LSL is None -> represented as n/a; Cp is None -> n/a.
        assert "LSL" in s
        assert csv_report.NA in s

    def test_zero_variation_inf_token(self) -> None:
        s = csv_report.to_csv_string(_bundle(cases.zero_variation(), with_meas=False))
        assert "inf" in s  # +inf capability sentinel rendered readably

    def test_conclusions_and_recommendations_separate(self) -> None:
        s = csv_report.to_csv_string(_bundle(cases.high_cpk_unstable()))
        # The two interpretation blocks are distinct sections.
        assert s.index("# Statistical conclusions") < s.index("# Engineering recommendations")
        assert "process_unstable" in s
        assert "rec_investigate_stability" in s

    def test_violations_section_none(self) -> None:
        s = csv_report.to_csv_string(_bundle())
        assert "# SPC rule violations" in s
        assert "(none)" in s

    def test_deterministic(self) -> None:
        b = _bundle()
        assert csv_report.to_csv_string(b) == csv_report.to_csv_string(b)

    def test_write_file(self) -> None:
        d = tempfile.mkdtemp()
        path = csv_report.write_csv(_bundle(), os.path.join(d, "r.csv"))
        with open(path, encoding="utf-8") as f:
            assert "# Metadata" in f.read()


class TestCsvWithViolations:
    def test_violations_rendered(self) -> None:
        import numpy as np
        from app.spc.limits import imr_chart
        from app.spc.rules import RuleSet, evaluate_rules
        pair = imr_chart(np.array([10., 10.1, 9.9, 10., 10.1, 9.9, 10., 10.1, 30.0, 10.]))
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        rep = cases.high_cpk_unstable()
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="x"),
            capability=rep, interpretation=interpret(rep),
            evaluations=[ev],
        )
        s = csv_report.to_csv_string(bundle)
        assert "N1" in s  # the outlier rule id
        # Affected index appears in the indices column.
        assert "8" in s


class TestExcelStructure:
    def test_all_sheets_present(self) -> None:
        data = excel_report.workbook_to_bytes(_bundle())
        wb = load_workbook(io.BytesIO(data))
        assert wb.sheetnames == [
            "Summary", "Measurements", "Statistics", "Capability",
            "SPC", "Violations", "Interpretation",
        ]

    def test_summary_headline(self) -> None:
        rep = cases.high_cpk_unstable()
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle(rep))))
        ws = wb["Summary"]
        # Headline (row 2) mentions instability.
        assert "unstable" in str(ws.cell(row=2, column=1).value).lower()

    def test_measurements_count(self) -> None:
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle())))
        ws = wb["Measurements"]
        # 6 measurements -> rows 2..7 have values.
        values = [ws.cell(row=r, column=3).value for r in range(2, 8)]
        assert all(v is not None for v in values)
        assert values[0] == pytest.approx(9.8)

    def test_capability_sheet_has_indices(self) -> None:
        rep = cases.stable_capable()
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle(rep))))
        ws = wb["Capability"]
        found = {ws.cell(row=r, column=1).value for r in range(1, ws.max_row + 1)}
        assert {"Cp", "Cpk", "Pp", "Ppk"}.issubset(found)

    def test_capability_values_are_numbers_not_recomputed(self) -> None:
        rep = cases.stable_capable()
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle(rep))))
        ws = wb["Capability"]
        # Find the Cpk row and assert its numeric value equals the Python result.
        cpk_val = None
        for r in range(1, ws.max_row + 1):
            if ws.cell(row=r, column=1).value == "Cpk":
                cpk_val = ws.cell(row=r, column=3).value
                break
        assert cpk_val == pytest.approx(rep.capability.cpk)
        # Cell holds a literal number, not an Excel formula string.
        assert not (isinstance(cpk_val, str) and cpk_val.startswith("="))

    def test_interpretation_sheet_separates_blocks(self) -> None:
        rep = cases.high_cpk_unstable()
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle(rep))))
        ws = wb["Interpretation"]
        col_a = [ws.cell(row=r, column=1).value for r in range(1, ws.max_row + 1)]
        assert "Statistical conclusions" in col_a
        assert "Engineering recommendations" in col_a
        # Conclusions block comes before recommendations block.
        assert col_a.index("Statistical conclusions") < col_a.index("Engineering recommendations")

    def test_violations_sheet_none(self) -> None:
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle())))
        ws = wb["Violations"]
        assert "no violations" in str(ws.cell(row=2, column=1).value).lower()

    def test_missing_value_na_in_excel(self) -> None:
        # One-sided spec -> Cp is None -> represented as n/a string in Capability.
        rep = cases.one_sided_upper()
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle(rep))))
        ws = wb["Capability"]
        # LSL-dependent Cp row should carry n/a (one-sided => cp None).
        texts = [ws.cell(row=r, column=3).value for r in range(1, ws.max_row + 1)]
        assert excel_report.NA in texts

    def test_write_xlsx_file(self) -> None:
        d = tempfile.mkdtemp()
        path = excel_report.write_xlsx(_bundle(), os.path.join(d, "r.xlsx"))
        assert os.path.getsize(path) > 2000
        with open(path, "rb") as f:
            assert f.read(2) == b"PK"  # xlsx = zip

    def test_unit_propagates(self) -> None:
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(_bundle())))
        ws = wb["Summary"]
        cells = [ws.cell(row=r, column=2).value for r in range(1, ws.max_row + 1)]
        assert "mm" in cells
