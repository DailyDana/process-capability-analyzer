"""Tests for control-chart constants (app.statistics.constants).

The central assertion of this module: the *exact* computation functions
(c4, d2, d3, and the derived chart factors) must reproduce the *canonical
published table values* to within their printed precision. This cross-validates
both the table (guards against transcription typos) and the math.
"""

from __future__ import annotations

import pytest

from app.statistics import constants as c


# Tolerance: published tables are rounded to 3-4 significant decimals, so we
# compare against them with a matching absolute tolerance.
TABLE_TOL = 1.5e-3


class TestExactMatchesTable:
    """Exact functions reproduce the tabulated canonical values."""

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_c4(self, n: int) -> None:
        assert c.c4(n) == pytest.approx(c.CONSTANTS[n].c4, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_d2(self, n: int) -> None:
        assert c.d2(n) == pytest.approx(c.CONSTANTS[n].d2, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_d3(self, n: int) -> None:
        assert c.d3(n) == pytest.approx(c.CONSTANTS[n].d3, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_a2(self, n: int) -> None:
        assert c.a2(n) == pytest.approx(c.CONSTANTS[n].a2, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_a3(self, n: int) -> None:
        assert c.a3(n) == pytest.approx(c.CONSTANTS[n].a3, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_d4(self, n: int) -> None:
        assert c.d4_factor(n) == pytest.approx(c.CONSTANTS[n].d4_factor, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_d3_factor(self, n: int) -> None:
        assert c.d3_factor(n) == pytest.approx(c.CONSTANTS[n].d3_factor, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_b4(self, n: int) -> None:
        assert c.b4(n) == pytest.approx(c.CONSTANTS[n].b4, abs=TABLE_TOL)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_b3(self, n: int) -> None:
        assert c.b3(n) == pytest.approx(c.CONSTANTS[n].b3, abs=TABLE_TOL)


class TestLandmarkValues:
    """Spot-check the most-cited textbook values by hand."""

    def test_d2_n2(self) -> None:
        assert c.d2(2) == pytest.approx(1.128, abs=1e-3)

    def test_d2_n5(self) -> None:
        assert c.d2(5) == pytest.approx(2.326, abs=1e-3)

    def test_c4_n5(self) -> None:
        assert c.c4(5) == pytest.approx(0.9400, abs=1e-4)

    def test_c4_n10(self) -> None:
        assert c.c4(10) == pytest.approx(0.9727, abs=1e-4)

    def test_e2_n2_is_2_66(self) -> None:
        # The single most-used individuals-chart constant. The published value
        # 2.660 is 3/1.128 using the already-rounded d2; the exact 3/d2(2) =
        # 3/1.128379 = 2.6587, so we allow a 2e-3 window around the table value.
        assert c.e2(2) == pytest.approx(2.660, abs=2e-3)

    def test_e2_n2_exact_value(self) -> None:
        # Pin the exact (unrounded) value so a regression in d2 is caught.
        assert c.e2(2) == pytest.approx(3.0 / c.d2(2), rel=1e-12)
        assert c.e2(2) == pytest.approx(2.6587, abs=1e-3)

    def test_a2_n5(self) -> None:
        assert c.a2(5) == pytest.approx(0.577, abs=1e-3)

    def test_d4_n2(self) -> None:
        assert c.d4_factor(2) == pytest.approx(3.267, abs=1e-3)


class TestMathematicalIdentities:
    """Internal consistency relations that must hold exactly (not just vs table)."""

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_a2_definition(self, n: int) -> None:
        # A2 = 3 / (d2 * sqrt(n)) by construction
        assert c.a2(n) == pytest.approx(3.0 / (c.d2(n) * n**0.5), rel=1e-9)

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_d3_d4_symmetry_about_one(self, n: int) -> None:
        # D3 and D4 are 1 -/+ 3*d3/d2; their deviations from 1 are equal in size
        # whenever D3 is not truncated at zero.
        spread = 3.0 * c.d3(n) / c.d2(n)
        assert c.d4_factor(n) == pytest.approx(1.0 + spread, rel=1e-9)
        if 1.0 - spread > 0:
            assert c.d3_factor(n) == pytest.approx(1.0 - spread, rel=1e-9)
        else:
            assert c.d3_factor(n) == 0.0

    @pytest.mark.parametrize("n", list(range(2, 26)))
    def test_c4_between_zero_and_one(self, n: int) -> None:
        assert 0.0 < c.c4(n) < 1.0

    def test_c4_increases_toward_one(self) -> None:
        # c4 -> 1 as n grows (bias vanishes)
        vals = [c.c4(n) for n in range(2, 26)]
        assert all(a < b for a, b in zip(vals, vals[1:]))
        assert vals[-1] > 0.98


class TestLookupAndGuards:
    def test_get_constants_returns_bundle(self) -> None:
        cc = c.get_constants(5)
        assert cc.n == 5
        assert cc.d2 == pytest.approx(2.326, abs=1e-3)
        assert cc.e2 == pytest.approx(1.290, abs=1e-3)

    def test_get_constants_rejects_n_below_2(self) -> None:
        with pytest.raises(ValueError, match="below the minimum"):
            c.get_constants(1)

    def test_get_constants_rejects_n_above_table(self) -> None:
        with pytest.raises(ValueError, match="exceeds the tabulated maximum"):
            c.get_constants(26)

    def test_c4_rejects_n_below_2(self) -> None:
        with pytest.raises(ValueError):
            c.c4(1)

    def test_exact_functions_extend_beyond_table(self) -> None:
        # Exact functions must work past n=25 even though the table stops there.
        assert 0.98 < c.c4(30) < 1.0
        assert c.d2(30) > c.d2(25)
