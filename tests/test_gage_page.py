"""Streamlit AppTest coverage for the Gage R&R page and the page router.

Executes the real page scripts in-process (same harness as
``tests/test_streamlit_app.py``): imports, widget wiring and every section must
render without an exception, and the displayed headline must come from the
service layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parent.parent / "app" / "ui"
PAGE = str(UI / "gage_rr_page.py")
ROUTER = str(UI / "streamlit_app.py")


def _run() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    return at


def _metric(at: AppTest, label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return m.value
    raise KeyError(label)


class TestGagePageExample:
    def test_example_renders_all_sections(self):
        at = _run()
        assert not at.exception
        headers = [h.value for h in at.main.header]
        assert headers == ["1 · Study data", "2 · Result", "3 · Charts", "4 · Interpretation", "5 · Export"]
        # Minitab's published figures for this data (tolerance 8).
        assert _metric(at, "% GRR of study variation") == "27.86%"
        assert _metric(at, "% GRR of tolerance") == "22.68%"
        assert _metric(at, "ndc") == "4"
        assert _metric(at, "Ranges above UCL") == "1"
        assert any("Measurement system MARGINAL" in w.value for w in at.warning)

    def test_average_range_method(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        at.sidebar.radio(key="gage_method").set_value("Average & Range (AIAG)")
        at.run()
        assert not at.exception
        assert _metric(at, "% GRR of study variation") == "26.68%"  # BPI Consulting, Table 2

    def test_without_tolerance(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        at.sidebar.checkbox(key="gage_use_tol").uncheck()
        at.run()
        assert not at.exception
        assert _metric(at, "% GRR of tolerance") == "—"

    def test_bad_thresholds_are_reported(self):
        at = AppTest.from_file(PAGE, default_timeout=90)
        at.run()
        at.sidebar.number_input(key="gage_acc").set_value(40.0)
        at.run()
        assert not at.exception
        assert any("acceptable_max" in e.value for e in at.error)


class TestRouter:
    def test_router_runs_default_page(self):
        at = AppTest.from_file(ROUTER, default_timeout=90)
        at.run()
        assert not at.exception
        assert at.title and "Process Capability" in at.title[0].value


@pytest.mark.parametrize("page", ["capability_page.py", "gage_rr_page.py", "type1_page.py"])
def test_pages_do_not_set_page_config(page):
    # Only the router may call st.set_page_config.
    assert "set_page_config" not in (UI / page).read_text(encoding="utf-8")
