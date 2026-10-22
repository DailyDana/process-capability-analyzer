"""Type-1 gauge study (Cg/Cgk) verified against published results.

1. ``examples/type1_bosch.csv`` -- Robert Bosch GmbH, Booklet 10 "Capability
   of Measurement and Test Processes", worked example of Procedure 1
   (part W450, x_m = 6.002, T = 0.060, resolution 0.001, n = 50).
2. ``examples/type1_spcforexcel.csv`` -- BPI Consulting, "Type 1 Gage Study"
   (x_m = 12.306, T = 0.05, n = 50).

Tolerances are half a unit in the last published digit, except where noted.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from app.msa.interpret import Type1Thresholds, interpret_type1
from app.msa.type1 import Type1Request, type1_study, validate_type1

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def readings(name: str) -> tuple[float, ...]:
    with open(EXAMPLES / name, newline="", encoding="utf-8") as fh:
        return tuple(float(r["value"]) for r in csv.DictReader(fh))


def dp(n: int) -> float:
    return 0.5 * 10.0**-n


@pytest.fixture(scope="module")
def bosch():
    return type1_study(Type1Request(readings("type1_bosch.csv"), reference=6.002, tolerance=0.060, resolution=0.001))


@pytest.fixture(scope="module")
def bpi():
    return type1_study(Type1Request(readings("type1_spcforexcel.csv"), reference=12.306, tolerance=0.05))


class TestBoschBooklet10:
    def test_statistics(self, bosch):
        assert bosch.n == 50
        assert bosch.mean == pytest.approx(6.00090, abs=dp(5))
        assert bosch.sd == pytest.approx(0.00099488, abs=dp(8))
        assert abs(bosch.bias) == pytest.approx(0.001100, abs=dp(6))

    def test_indices_and_confidence_intervals(self, bosch):
        assert bosch.cg == pytest.approx(2.01, abs=dp(2))
        assert bosch.cgk == pytest.approx(1.64, abs=dp(2))
        assert bosch.cg_ci == pytest.approx((1.61, 2.41), abs=dp(2))
        assert bosch.cgk_ci == pytest.approx((1.30, 1.98), abs=dp(2))

    def test_resolution_and_bias(self, bosch):
        assert bosch.pct_resolution == pytest.approx(1.67, abs=dp(2))
        assert bosch.bias_significant is True  # "Test for Bias: significant"
        assert bosch.bias_criterion == pytest.approx(0.284, abs=dp(3))  # Bosch's table value for n = 50

    def test_minimum_tolerances(self, bosch):
        tmin = bosch.min_tolerances()
        assert tmin["Cgk"] == pytest.approx(0.050696, abs=dp(6))
        assert tmin["RE"] == pytest.approx(0.020000, abs=dp(6))
        # Bosch prints 0.039701; the exact value is 0.039696 (0.013% apart).
        # The source of the difference is not stated; locked at that margin.
        assert tmin["Cg"] == pytest.approx(0.039701, rel=2e-4)

    def test_verdict(self, bosch):
        rep = interpret_type1(bosch)
        # Bosch: "Measurement system capable (RE, Cg, Cgk)" with a significant bias.
        assert rep.headline.startswith("Gauge CAPABLE")
        assert {"type1_cg_ok", "type1_cgk_ok", "type1_resolution_ok", "type1_bias_significant"} <= set(rep.conclusion_codes)
        assert "rec_type1_proceed" in rep.recommendation_codes


class TestSpcForExcel:
    def test_statistics_and_bias_test(self, bpi):
        assert bpi.mean == pytest.approx(12.3075, abs=dp(4))
        assert bpi.sd == pytest.approx(0.00485, abs=dp(5))
        assert bpi.bias == pytest.approx(0.0015, abs=dp(4))
        assert bpi.t_stat == pytest.approx(2.185, abs=dp(3))
        assert bpi.p_value == pytest.approx(0.0337, abs=dp(4))

    def test_indices_and_percentages(self, bpi):
        assert bpi.cg == pytest.approx(0.343, abs=dp(3))
        assert bpi.cgk == pytest.approx(0.240, abs=dp(3))
        # BPI prints 58.2, computed from s rounded to 0.00485 (6 * 0.00485 / 0.05
        # = 58.20); the unrounded s gives 58.25.
        assert bpi.pct_var_rep == pytest.approx(58.25, abs=dp(2))
        assert 100 * 6 * round(bpi.sd, 5) / 0.05 == pytest.approx(58.2, abs=1e-9)
        assert bpi.pct_var_rep_bias == pytest.approx(83.22, abs=dp(2))

    def test_verdict(self, bpi):
        rep = interpret_type1(bpi)
        assert rep.headline.startswith("Gauge NOT CAPABLE")
        assert "type1_cg_low" in rep.conclusion_codes
        assert "rec_type1_repeatability" in rep.recommendation_codes


class TestFormulas:
    def test_defaults_reduce_to_bosch_form(self):
        vals = (10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.1, 10.0, 10.2, 9.9)
        r = type1_study(Type1Request(vals, reference=10.05, tolerance=6.0))
        s = r.sd
        assert r.cg == pytest.approx(0.2 * 6.0 / (6 * s))
        assert r.cgk == pytest.approx((0.1 * 6.0 - abs(r.bias)) / (3 * s))
        assert r.pct_var_rep == pytest.approx(20.0 / r.cg)
        assert r.pct_var_rep_bias == pytest.approx(20.0 / r.cgk)

    def test_bias_criterion_equals_t_test(self):
        # |bias|/s >= t(0.975, n-1)/sqrt(n)  <=>  p <= 0.05
        vals = tuple(10 + 0.01 * ((i * 37) % 11 - 5) for i in range(30))
        for ref in (10.0, 9.995, 9.99, 9.98):
            r = type1_study(Type1Request(vals, reference=ref, tolerance=1.0))
            assert r.bias_significant == (r.p_value <= 0.05)

    def test_k_and_l_options(self):
        vals = (1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99)
        a = type1_study(Type1Request(vals, 1.0, 2.0, k_percent=20, l_sigma=6))
        b = type1_study(Type1Request(vals, 1.0, 2.0, k_percent=15, l_sigma=4))
        assert b.cg == pytest.approx(a.cg * (15 / 20) * (6 / 4))

    def test_zero_spread(self):
        r = type1_study(Type1Request((5.0,) * 20, reference=5.0, tolerance=1.0, resolution=0.1))
        assert r.sd == 0.0 and math.isinf(r.cg) and r.cg_ci is None and r.p_value is None
        rep = interpret_type1(r)
        assert "type1_zero_spread" in rep.conclusion_codes
        assert "not assessable" in rep.headline

    def test_bias_beyond_band_gives_negative_cgk(self):
        vals = (1.10, 1.11, 1.09, 1.10, 1.12, 1.08, 1.10, 1.11, 1.09, 1.10)
        r = type1_study(Type1Request(vals, reference=1.0, tolerance=0.5))  # band +-0.05, bias 0.10
        assert r.cgk < 0 and r.pct_var_rep_bias is None
        assert "rec_type1_repeatability" in interpret_type1(r).recommendation_codes or r.cg >= 1.33

    def test_resolution_bands(self):
        vals = tuple(1 + 0.001 * (i % 5) for i in range(50))
        codes = {}
        for res in (0.02, 0.07, 0.15):
            rep = interpret_type1(type1_study(Type1Request(vals, 1.002, 1.0, resolution=res)))
            codes[res] = rep.conclusion_codes[0]
        assert codes == {0.02: "type1_resolution_ok", 0.07: "type1_resolution_marginal", 0.15: "type1_resolution_poor"}

    def test_thresholds_validated(self):
        with pytest.raises(ValueError):
            Type1Thresholds(resolution_ok_pct=10, resolution_max_pct=5)


class TestValidation:
    def test_counts(self):
        assert not validate_type1([1.0] * 5, 1.0, 1.0).ok
        v = validate_type1([1.0, 1.1] * 10, 1.0, 1.0)
        assert v.ok and any("at least 25" in w for w in v.warnings)
        v = validate_type1([1.0, 1.1] * 15, 1.0, 1.0)
        assert v.ok and any("recommends 50" in w for w in v.warnings)

    def test_bad_inputs_are_errors(self):
        v = validate_type1([1.0, "x", 1.1] + [1.0] * 30, 1.0, 1.0)
        assert not v.ok and "row 2" in v.errors[0]
        assert not validate_type1([1.0] * 30, "a", 1.0).ok
        assert not validate_type1([1.0] * 30, 1.0, 0.0).ok
        assert not validate_type1([1.0] * 30, 1.0, 1.0, resolution=-1).ok

    def test_blank_rows_ignored_with_warning(self):
        v = validate_type1([1.0, None, 1.1] + [1.0] * 30, 1.0, 1.0)
        assert v.ok and len(v.values) == 32 and any("blank" in w for w in v.warnings)
