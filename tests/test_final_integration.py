"""M7 final integration tests — the complete pipeline, end to end.

CSV -> cleaning -> analysis -> SPC -> capability -> interpretation -> chart ->
CSV export -> Excel export -> PDF export.

PDF content is verified STRUCTURALLY by extracting text with pypdf and asserting
on section titles, values, rule IDs, and warnings — not merely file existence.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app.reporting import csv_report, excel_report, pdf_report
from app.reporting.bundle import AnalysisBundle, ExportMeta, MethodologyInfo
from app.services.analysis import AnalysisRequest, ChartKind, make_spec, run_analysis
from app.services.data_loader import clean_column, load_table


# ---------------------------------------------------------------------------
# Pipeline helper: raw CSV bytes -> everything
# ---------------------------------------------------------------------------
def full_pipeline(
    raw_values,
    *,
    chart_kind=ChartKind.XBAR_R,
    subgroup_size=5,
    usl=12.0,
    lsl=8.0,
    unit="mm",
):
    """Run the complete pipeline from raw CSV bytes; returns (outcome, bundle)."""
    csv_bytes = pd.DataFrame({"measurement": raw_values}).to_csv(index=False).encode()
    loaded = load_table(csv_bytes, "pipeline.csv")
    assert loaded.ok, loaded.error
    cleaning = clean_column(loaded.frame["measurement"].tolist())
    spec, errors, _ = make_spec(usl=usl, lsl=lsl, unit=unit)
    assert not errors, errors
    outcome = run_analysis(AnalysisRequest(
        values=tuple(cleaning.values),
        chart_kind=chart_kind,
        subgroup_size=subgroup_size,
        spec=spec,
        unit=unit,
    ))
    assert outcome.ok, outcome.error
    bundle = AnalysisBundle.from_evaluations(
        meta=ExportMeta(characteristic="test_char", unit=unit,
                        machine="M-01", source_file="pipeline.csv"),
        capability=outcome.capability,
        interpretation=outcome.interpretation,
        descriptive=outcome.descriptive,
        measurements=list(cleaning.values),
        evaluations=[outcome.location_eval, outcome.dispersion_eval],
        methodology=MethodologyInfo(
            data_regime=("individuals" if chart_kind is ChartKind.IMR else "subgrouped"),
            chart_type=chart_kind.value,
            subgroup_size=outcome.subgroup_size,
            n_subgroups=outcome.n_subgroups,
            within_estimator=outcome.within_estimator_used,
            rule_set="nelson",
            n_input_rows=cleaning.report.n_input,
            n_used=outcome.n_used,
            n_excluded=cleaning.report.n_dropped,
            exclusion_notes=tuple(
                f"Row {r.original_index}: {r.reason}" for r in cleaning.excluded
            ),
        ),
    )
    return outcome, bundle


def pdf_text(bundle, chart_models=None) -> str:
    data = pdf_report.build_pdf_bytes(bundle, chart_models)
    assert data[:5] == b"%PDF-"
    reader = PdfReader(io.BytesIO(data))
    return " ".join(page.extract_text() for page in reader.pages)


def _stable(n=60, seed=42):
    return list(np.random.default_rng(seed).normal(10.0, 0.4, n))


# ---------------------------------------------------------------------------
# Scenario: stable, capable process
# ---------------------------------------------------------------------------
class TestStableCapable:
    def test_pipeline(self) -> None:
        outcome, bundle = full_pipeline(_stable())
        assert outcome.stability.is_stable          # no false instability
        cpk = outcome.capability.capability.cpk
        assert 1.0 < cpk < 3.0                       # sensible capability
        assert "process_stable" in outcome.interpretation.conclusion_codes

    def test_all_three_exports(self) -> None:
        outcome, bundle = full_pipeline(_stable())
        csv_s = csv_report.to_csv_string(bundle)
        xlsx = excel_report.workbook_to_bytes(bundle)
        text = pdf_text(bundle, outcome.chart_models)
        assert "process_stable" in csv_s
        assert xlsx[:2] == b"PK"
        assert "STABLE" in text
        assert "No SPC rule violations detected." in text

    def test_pdf_sections_complete(self) -> None:
        outcome, bundle = full_pipeline(_stable())
        text = pdf_text(bundle, outcome.chart_models)
        for section in (
            "Executive summary", "Data & Methodology", "Descriptive Statistics",
            "SPC Analysis", "Capability & Performance", "Statistical Conclusions",
            "Engineering Recommendations", "Technical Appendix",
        ):
            assert section in text, f"PDF missing section: {section}"

    def test_pdf_carries_metadata_and_unit(self) -> None:
        outcome, bundle = full_pipeline(_stable())
        text = pdf_text(bundle, outcome.chart_models)
        assert "test_char" in text
        assert "M-01" in text
        assert "mm" in text


# ---------------------------------------------------------------------------
# Scenario: unstable process
# ---------------------------------------------------------------------------
class TestUnstable:
    def _values(self):
        vals = _stable()
        vals[30] = 14.0  # gross outlier
        return vals

    def test_pipeline_flags_unstable(self) -> None:
        outcome, _ = full_pipeline(self._values())
        assert not outcome.stability.is_stable
        # Capability numerically retained.
        assert outcome.capability.capability.cpk is not None
        assert any(w.code == "unstable_process" for w in outcome.capability.warnings)
        assert "process_unstable" in outcome.interpretation.conclusion_codes

    def test_pdf_contains_stability_warning(self) -> None:
        outcome, bundle = full_pipeline(self._values())
        text = pdf_text(bundle, outcome.chart_models)
        assert "UNSTABLE" in text
        assert "must not be interpreted as reliable" in text or "not be interpreted" in text
        # Indices still present in the PDF.
        assert "Cpk" in text and "Ppk" in text

    def test_pdf_contains_rule_ids_and_indices(self) -> None:
        outcome, bundle = full_pipeline(self._values())
        text = pdf_text(bundle, outcome.chart_models)
        triggered = outcome.stability.triggered_rule_ids
        assert triggered  # sanity
        for rid in triggered:
            assert rid in text, f"rule {rid} missing from PDF"
        # Affected subgroup index appears in the violations table (subgroup 6
        # contains observation 30 for n=5).
        first_violation = outcome.all_violations[0]
        assert str(first_violation.indices[0]) in text


# ---------------------------------------------------------------------------
# Scenario: dirty data
# ---------------------------------------------------------------------------
class TestDirtyData:
    def test_exclusions_visible_and_indices_preserved(self) -> None:
        vals = _stable(58)
        raw = vals[:20] + ["bad", None] + vals[20:]  # rows 20,21 excluded
        csv_bytes = pd.DataFrame({"measurement": raw}).to_csv(index=False).encode()
        loaded = load_table(csv_bytes, "dirty.csv")
        cleaning = clean_column(loaded.frame["measurement"].tolist())
        assert cleaning.report.n_dropped == 2
        # Original indices preserved: rows 20/21 are the excluded ones.
        excluded_idx = [r.original_index for r in cleaning.excluded]
        assert excluded_idx == [20, 21]

    def test_report_reflects_cleaning(self) -> None:
        vals = _stable(58)
        raw = vals[:20] + ["bad", None] + vals[20:]
        outcome, bundle = full_pipeline(raw)
        assert bundle.methodology.n_excluded == 2
        text = pdf_text(bundle, outcome.chart_models)
        assert "Excluded observations" in text
        assert "Row 20" in text and "Row 21" in text
        # CSV export also carries the analysis unimpeded.
        csv_s = csv_report.to_csv_string(bundle)
        assert "Cpk" in csv_s


# ---------------------------------------------------------------------------
# Scenario: one-sided specification
# ---------------------------------------------------------------------------
class TestOneSided:
    def test_no_false_cp_pp(self) -> None:
        outcome, bundle = full_pipeline(_stable(), usl=12.0, lsl=None)
        cap = outcome.capability
        assert cap.capability.cp is None      # no Cp for one-sided
        assert cap.performance.cp is None     # no Pp for one-sided
        assert cap.capability.cpk is not None
        assert "one_sided_spec" in outcome.interpretation.conclusion_codes

    def test_exports_handle_one_sided(self) -> None:
        outcome, bundle = full_pipeline(_stable(), usl=12.0, lsl=None)
        csv_s = csv_report.to_csv_string(bundle)
        assert csv_report.NA in csv_s          # Cp -> explicit n/a
        text = pdf_text(bundle, outcome.chart_models)
        assert "one-sided" in text
        # Excel does not crash and carries NA.
        assert excel_report.workbook_to_bytes(bundle)[:2] == b"PK"


# ---------------------------------------------------------------------------
# Scenario: zero variation
# ---------------------------------------------------------------------------
class TestZeroVariation:
    def test_pipeline_zero_variation(self) -> None:
        outcome, bundle = full_pipeline([10.0] * 20, chart_kind=ChartKind.IMR)
        cap = outcome.capability
        assert cap.capability.cpk == float("inf")   # +inf sentinel
        assert any(w.code == "zero_variation" for w in cap.warnings)

    def test_exports_do_not_crash(self) -> None:
        outcome, bundle = full_pipeline([10.0] * 20, chart_kind=ChartKind.IMR)
        csv_s = csv_report.to_csv_string(bundle)
        assert "inf" in csv_s
        assert excel_report.workbook_to_bytes(bundle)[:2] == b"PK"
        text = pdf_text(bundle, outcome.chart_models)
        assert "+inf" in text
        assert "zero_variation" in text.replace("\n", "")


# ---------------------------------------------------------------------------
# Scenario: non-normal data
# ---------------------------------------------------------------------------
class TestNonNormal:
    def _values(self):
        # Heavily skewed (exponential) but in-control enough to isolate the
        # normality warning; individuals chart avoids subgroup-mean averaging
        # (which would re-normalize the plotted points).
        rng = np.random.default_rng(3)
        return list(rng.exponential(0.4, 80) + 9.0)

    def test_warning_propagates(self) -> None:
        outcome, bundle = full_pipeline(
            self._values(), chart_kind=ChartKind.IMR, usl=13.0, lsl=8.5,
        )
        assert outcome.normality.normal is False
        assert "non_normal" in outcome.interpretation.conclusion_codes
        assert any(w.code == "non_normal" for w in outcome.capability.warnings)

    def test_pdf_contains_normality_warning(self) -> None:
        outcome, bundle = full_pipeline(
            self._values(), chart_kind=ChartKind.IMR, usl=13.0, lsl=8.5,
        )
        text = pdf_text(bundle, outcome.chart_models)
        assert "REJECTED" in text
        assert "non_normal" in text.replace("\n", "")


# ---------------------------------------------------------------------------
# PDF chart embedding quality
# ---------------------------------------------------------------------------
class TestPdfChartEmbedding:
    def test_pdf_embeds_images(self) -> None:
        outcome, bundle = full_pipeline(_stable())
        data = pdf_report.build_pdf_bytes(bundle, outcome.chart_models)
        reader = PdfReader(io.BytesIO(data))
        image_count = 0
        for page in reader.pages:
            resources = page.get("/Resources", {})
            xobjects = resources.get("/XObject", {})
            for obj in xobjects.values():
                if obj.get("/Subtype") == "/Image":
                    image_count += 1
        assert image_count >= 2  # location + dispersion charts embedded

    def test_pdf_without_charts_still_valid(self) -> None:
        _, bundle = full_pipeline(_stable())
        text = pdf_text(bundle, None)  # no chart models supplied
        assert "SPC Analysis" in text
        assert "Rule violations" in text

    def test_chart_native_aspect_ratio(self) -> None:
        # The embed helper must preserve the PNG's aspect ratio (no distortion).
        import matplotlib.pyplot as plt
        from app.charts import render as chart_render
        from app.reporting.pdf_report import _chart_image
        outcome, _ = full_pipeline(_stable())
        fig = chart_render.render_chart(outcome.chart_models.location)
        img = _chart_image(fig, max_width_mm=170.0)
        plt.close(fig)
        png = chart_render.figure_to_png_bytes(fig, dpi=200)
        from reportlab.lib.utils import ImageReader
        pw, ph = ImageReader(io.BytesIO(png)).getSize()
        assert img.drawHeight / img.drawWidth == pytest.approx(ph / pw, rel=1e-6)


# ---------------------------------------------------------------------------
# All chart kinds through the full pipeline + PDF
# ---------------------------------------------------------------------------
class TestAllChartKindsPdf:
    @pytest.mark.parametrize(
        "kind,size", [(ChartKind.XBAR_R, 5), (ChartKind.XBAR_S, 10), (ChartKind.IMR, 1)]
    )
    def test_pdf_for_each_chart_kind(self, kind, size) -> None:
        outcome, bundle = full_pipeline(_stable(60), chart_kind=kind, subgroup_size=size)
        text = pdf_text(bundle, outcome.chart_models)
        assert "Capability & Performance" in text
        assert "Center line (CL)" in text
