"""Non-normal capability: service, interpretation, exports and the page."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from app.interpret.nonnormal import NonNormalThresholds
from app.reporting.bundle import ExportMeta
from app.reporting.nonnormal_report import nonnormal_csv_string, nonnormal_pdf_bytes
from app.services.nonnormal import BEST_FIT, BOX_COX, NonNormalRequest, run_nonnormal

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "ui" / "nonnormal_page.py")


def cycle_values():
    with open(ROOT / "examples" / "nonnormal_cycle_time.csv", newline="", encoding="utf-8") as fh:
        return tuple(r["cycle_time_days"] for r in csv.DictReader(fh))


def run(**kw):
    base = dict(usl=30.0, lsl=0.0, lsl_is_boundary=True)
    base.update(kw)
    return run_nonnormal(NonNormalRequest(cycle_values(), **base))


class TestService:
    def test_best_fit_is_weibull_with_published_ppm(self):
        o = run()
        assert o.ok and o.chosen_fit.distribution.value == "weibull"
        assert o.summary.ppm_total == pytest.approx(233244.81, abs=0.005)
        assert o.summary.pp is None  # one-sided because of the boundary

    def test_box_cox_with_c4_matches_published(self):
        o = run(method=BOX_COX, unbias_c4=True)
        assert o.summary.ppm_total == pytest.approx(227113.29, abs=0.005)
        assert o.boxcox.lambda_used == 0.5

    def test_named_distribution_and_unfittable(self):
        assert run(method="gamma").chosen_fit.distribution.value == "gamma"
        o = run_nonnormal(NonNormalRequest(tuple([-1.0] + [float(i) for i in range(1, 30)]), usl=40.0,
                                           method="weibull"))
        assert not o.ok and "positive" in o.errors[0]

    def test_observed_ppm(self):
        vals = [float(v) for v in cycle_values()]
        expected = 1e6 * sum(v > 30 for v in vals) / len(vals)
        assert run().summary.observed_ppm == pytest.approx(expected)

    def test_errors_and_warnings(self):
        assert not run_nonnormal(NonNormalRequest(tuple(range(5)), usl=10.0)).ok
        o = run_nonnormal(NonNormalRequest(tuple(np.random.default_rng(0).weibull(2, 40)), usl=3.0))
        assert o.ok and any("100 are recommended" in w for w in o.warnings)
        bad = run_nonnormal(NonNormalRequest(tuple(range(1, 30)), usl=1.0, lsl=5.0))
        assert not bad.ok and "USL must be greater" in bad.errors[0]


class TestInterpretation:
    def test_cycle_time_codes(self):
        rep = run().interpretation
        assert {"nn_model_fit", "nn_lower_boundary", "nn_ppk_inadequate"} <= set(rep.conclusion_codes)
        assert "rec_nn_improve" in rep.recommendation_codes

    def test_normal_misleading_flagged_on_skewed_data(self):
        x = np.random.default_rng(1).lognormal(0.0, 0.6, 400)
        o = run_nonnormal(NonNormalRequest(tuple(x), usl=float(np.quantile(x, 0.999)) * 1.4))
        assert "nn_normal_misleading" in o.interpretation.conclusion_codes

    def test_normal_data_close(self):
        x = np.random.default_rng(2).normal(10, 1, 300)
        o = run_nonnormal(NonNormalRequest(tuple(x), usl=15.0, lsl=5.0, method="normal"))
        assert "nn_normal_close" in o.interpretation.conclusion_codes
        assert "nn_ppk_capable" in o.interpretation.conclusion_codes or "nn_ppk_excellent" in o.interpretation.conclusion_codes

    def test_poor_fit_and_not_best(self):
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(5, 0.2, 150), rng.normal(8, 0.2, 150)])  # a mixture
        o = run_nonnormal(NonNormalRequest(tuple(x), usl=10.0, method="exponential"))
        codes = o.interpretation.conclusion_codes
        assert "nn_poor_fit" in codes and "nn_not_best_fit" in codes
        assert "rec_nn_check_fit" in o.interpretation.recommendation_codes

    def test_thresholds_validated(self):
        with pytest.raises(ValueError):
            NonNormalThresholds(marginal_min=1.5, capable_min=1.33)


class TestExports:
    @pytest.mark.parametrize("method", [BEST_FIT, BOX_COX])
    def test_csv_and_pdf(self, method):
        o = run(method=method)
        text = nonnormal_csv_string(o, ExportMeta(characteristic="Cycle time"))
        rows = list(csv.reader(io.StringIO(text)))
        ppk = next(r for r in rows if r and r[0] == "Ppk")
        assert float(ppk[1]) == o.summary.ppk
        pdf = nonnormal_pdf_bytes(o, ExportMeta(characteristic="Cycle time", unit="days"))
        t = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
        for needle in ("Process Capability Report (non-normal data)", "Cycle time", "2. Distribution fits",
                       "4. Statistical Conclusions"):
            assert needle in t
        assert "&amp;" not in t


class TestPage:
    def test_example(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        assert not at.exception
        assert [h.value for h in at.main.header] == [
            "1 · Data and specification", "2 · Result", "3 · Charts", "4 · Interpretation", "5 · Export"]
        vals = {m.label: m.value for m in at.metric}
        assert vals["Expected PPM"] == "233,245"
        assert vals["Ppk"] == "0.21"

    def test_box_cox_method(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        at.sidebar.radio(key="nn_method").set_value("Box-Cox transformation")
        at.sidebar.checkbox(key="nn_c4").check()
        at.run()
        assert not at.exception
        assert {m.label: m.value for m in at.metric}["Expected PPM"] == "227,113"
