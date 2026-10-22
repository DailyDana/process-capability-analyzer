"""MSA PDF reports: content verified by text extraction (pypdf)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.msa.models import GageMethod
from app.reporting.bundle import ExportMeta
from app.reporting.msa_pdf import gage_rr_pdf_bytes, pdf_text, plain_text
from app.services.gage_rr import columns_to_request, run_gage_study

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "gage_rr_aiag.csv"


def _text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


@pytest.fixture(scope="module", params=list(GageMethod), ids=lambda m: m.value)
def report(request):
    with open(EXAMPLE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = [[r[k] for r in rows] for k in ("part", "operator", "value", "trial")]
    out = run_gage_study(columns_to_request(*cols, method=request.param, tolerance=8.0))
    pdf = gage_rr_pdf_bytes(out.study, out.result, out.interpretation, ExportMeta(characteristic="Bore diameter"))
    return out, pdf, _text(pdf)


class TestGageRrPdf:
    def test_is_a_pdf_with_sections(self, report):
        _, pdf, text = report
        assert pdf.startswith(b"%PDF")
        for heading in ("Gage R&R (Measurement System Analysis) Report", "1. Method", "2. Gage R&R",
                        "3. Charts", "4. Statistical Conclusions", "5. Engineering Recommendations",
                        "6. Measurements"):
            assert heading in text

    def test_published_figures_and_headline(self, report):
        out, _, text = report
        expected = {GageMethod.ANOVA: "27.86", GageMethod.AVERAGE_RANGE: "26.68"}[out.result.method]
        assert expected in text
        assert "Bore diameter" in text
        assert "Caution" in text  # the out-of-control range on part 4 / appraiser B
        assert out.interpretation.headline.split(":")[0] in text

    def test_no_markup_leaks(self, report):
        _, _, text = report
        assert "&amp;" not in text and "&lt;" not in text
        assert "Total Gage R&R" in text
        assert "< 10% acceptable" in text  # an unescaped "<" would have broken the paragraph

    def test_all_measurements_listed(self, report):
        out, _, text = report
        assert "-2.16" in text and "2.26" in text  # extremes of the AIAG data


def test_text_helpers():
    assert plain_text("ndc ≥ 5 · σ") == "ndc >= 5 · sigma"
    assert pdf_text("<10% & R̄") == "&lt;10% &amp; R-bar"
