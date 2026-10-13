"""Tests for the Gage R&R orchestrator (app.services.gage_rr)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.msa.models import GageMethod
from app.services.gage_rr import GageRequest, columns_to_request, run_gage_study

FIX = Path(__file__).parent / "fixtures"


def columns(name="gage_synthetic_10x3x3_interaction.csv"):
    with open(FIX / name, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return (
        [r["part"] for r in rows],
        [r["operator"] for r in rows],
        [r["value"] for r in rows],
        [r["trial"] for r in rows],
    )


class TestRunGageStudy:
    @pytest.mark.parametrize("method", list(GageMethod))
    def test_happy_path(self, method):
        p, o, v, t = columns()
        out = run_gage_study(columns_to_request(p, o, v, t, method=method, tolerance=10.0, unit="mm"))
        assert out.ok and out.errors == ()
        assert out.result.method is method
        assert out.result.n_parts == 10 and out.result.n_trials == 3
        assert out.result.pct_grr_tolerance is not None
        assert out.interpretation.headline.startswith("Measurement system")
        assert out.study.unit == "mm"

    def test_validation_errors_returned_not_raised(self):
        p, o, v, t = columns()
        v[0] = "oops"
        out = run_gage_study(columns_to_request(p, o, v, t))
        assert not out.ok and out.result is None and "row 1" in out.errors[0]

    def test_average_range_outside_tables_is_a_message(self):
        p, o, v, t = columns()
        # Re-label so there are 4 operators... simplest: use trials as operators
        # is not balanced; instead build an 11-part study.
        p2 = p + [f"X{x}" for x in p[:9]]
        o2 = o + o[:9]
        v2 = v + v[:9]
        t2 = t + t[:9]
        out = run_gage_study(columns_to_request(p2, o2, v2, t2, method=GageMethod.AVERAGE_RANGE))
        assert not out.ok and "ANOVA" in out.errors[0]
        assert out.study is not None  # data were valid; only the method is unavailable

    def test_bad_tolerance_is_a_message(self):
        p, o, v, t = columns()
        out = run_gage_study(columns_to_request(p, o, v, t, tolerance=-1.0))
        assert not out.ok and "Tolerance" in out.errors[0]

    def test_warnings_propagate(self):
        p, o, v, t = columns()
        keep = [i for i, part in enumerate(p) if part in {"P01", "P02", "P03", "P04", "P05"}]
        cols = [[c[i] for i in keep] for c in (p, o, v, t)]
        out = run_gage_study(columns_to_request(*cols))
        assert out.ok and any("Only 5 parts" in w for w in out.warnings)

    def test_request_defaults(self):
        req = GageRequest(parts=(), operators=(), values=())
        assert req.method is GageMethod.ANOVA and req.k == 6.0 and req.alpha_pool == 0.25
