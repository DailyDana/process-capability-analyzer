"""Tests for capability indices, sigma level, and Cpk CI (app.capability.indices).

Covers the required proofs:
  3. perfectly centered process -> Cp ~= Cpk
  4. off-center process -> Cpk < Cp
  5. one-sided specification handling
  7. zero variation handled (indices -> +inf, no crash)
  8. Cpk CI valid and documented

Expected values are hand-computed in tests.fixtures.reference_data.
"""

from __future__ import annotations

import math

import pytest

from app.capability.indices import (
    CpkConfidenceInterval,
    compute_index_set,
    compute_sigma_level,
    cpk_confidence_interval,
)
from app.data.models import SpecLimits
from tests.fixtures import reference_data as ref


class TestTwoSidedCentered:
    """Requirement 3: centered process has Cp == Cpk."""

    def test_centered_cp_equals_cpk(self) -> None:
        spec = SpecLimits(usl=ref.CENTERED_USL, lsl=ref.CENTERED_LSL)
        idx = compute_index_set(
            mean=ref.SYMM_MEAN, sigma=ref.SYMM_SIGMA, spec=spec, family="capability"
        )
        assert idx.cp == pytest.approx(ref.CENTERED_CP, rel=1e-9)
        assert idx.cpk == pytest.approx(ref.CENTERED_CPK, rel=1e-9)
        assert idx.cp == pytest.approx(idx.cpk, rel=1e-12)  # exactly equal
        assert idx.cpu == pytest.approx(idx.cpl, rel=1e-12)


class TestTwoSidedOffCenter:
    """Requirement 4: off-center process has Cpk < Cp."""

    def test_offcenter_cpk_less_than_cp(self) -> None:
        spec = SpecLimits(usl=ref.OFFCENTER_USL, lsl=ref.OFFCENTER_LSL)
        idx = compute_index_set(
            mean=ref.SYMM_MEAN, sigma=ref.SYMM_SIGMA, spec=spec, family="capability"
        )
        assert idx.cp == pytest.approx(ref.OFFCENTER_CP, rel=1e-9)
        assert idx.cpu == pytest.approx(ref.OFFCENTER_CPU, rel=1e-9)
        assert idx.cpl == pytest.approx(ref.OFFCENTER_CPL, rel=1e-9)
        assert idx.cpk == pytest.approx(ref.OFFCENTER_CPK, rel=1e-9)
        assert idx.cpk < idx.cp  # strictly less when off-center
        assert idx.cpk == pytest.approx(idx.cpl)  # limited by nearer bound


class TestOneSided:
    """Requirement 5: one-sided spec handling."""

    def test_upper_only(self) -> None:
        spec = SpecLimits(usl=ref.ONESIDED_UPPER_USL)
        idx = compute_index_set(
            mean=ref.SYMM_MEAN, sigma=ref.SYMM_SIGMA, spec=spec, family="capability"
        )
        assert idx.cp is None  # potential undefined for one-sided
        assert idx.cpl is None
        assert idx.cpu == pytest.approx(ref.ONESIDED_UPPER_CPK, rel=1e-9)
        assert idx.cpk == pytest.approx(ref.ONESIDED_UPPER_CPK, rel=1e-9)

    def test_lower_only(self) -> None:
        spec = SpecLimits(lsl=ref.ONESIDED_LOWER_LSL)
        idx = compute_index_set(
            mean=ref.SYMM_MEAN, sigma=ref.SYMM_SIGMA, spec=spec, family="capability"
        )
        assert idx.cp is None
        assert idx.cpu is None
        assert idx.cpl == pytest.approx(ref.ONESIDED_LOWER_CPK, rel=1e-9)
        assert idx.cpk == pytest.approx(ref.ONESIDED_LOWER_CPK, rel=1e-9)


class TestZeroVariation:
    """Requirement 7: zero variation -> +inf sentinel, no crash."""

    def test_two_sided_zero_sigma(self) -> None:
        spec = SpecLimits(usl=ref.ZERO_USL, lsl=ref.ZERO_LSL)
        idx = compute_index_set(mean=10.0, sigma=0.0, spec=spec, family="capability")
        assert idx.cp == math.inf
        assert idx.cpk == math.inf
        assert idx.cpu == math.inf
        assert idx.cpl == math.inf

    def test_zero_sigma_mean_on_bound(self) -> None:
        # mean exactly on USL -> Cpu numerator 0 -> 0.0, Cpk = min(0, +inf)=0
        spec = SpecLimits(usl=10.0, lsl=4.0)
        idx = compute_index_set(mean=10.0, sigma=0.0, spec=spec, family="capability")
        assert idx.cpu == 0.0
        assert idx.cpk == 0.0

    def test_negative_sigma_rejected(self) -> None:
        spec = SpecLimits(usl=10.0, lsl=4.0)
        with pytest.raises(ValueError, match="non-negative"):
            compute_index_set(mean=7.0, sigma=-1.0, spec=spec, family="capability")


class TestSigmaLevel:
    def test_z_bench_symmetric_3sigma(self) -> None:
        spec = SpecLimits(usl=ref.SIGLEVEL_USL, lsl=ref.SIGLEVEL_LSL)
        sl = compute_sigma_level(
            mean=ref.SIGLEVEL_MEAN, sigma=ref.SIGLEVEL_SIGMA, spec=spec
        )
        assert sl.z_usl == pytest.approx(3.0, rel=1e-9)
        assert sl.z_lsl == pytest.approx(3.0, rel=1e-9)
        assert sl.p_above == pytest.approx(ref.SIGLEVEL_P_ABOVE, rel=1e-6)
        assert sl.p_total == pytest.approx(ref.SIGLEVEL_P_TOTAL, rel=1e-6)
        assert sl.dpmo == pytest.approx(ref.SIGLEVEL_DPMO, rel=1e-6)
        assert sl.z_bench == pytest.approx(ref.SIGLEVEL_ZBENCH, abs=1e-4)
        assert sl.with_shift is False

    def test_1p5_shift_is_opt_in_and_flagged(self) -> None:
        spec = SpecLimits(usl=ref.SIGLEVEL_USL, lsl=ref.SIGLEVEL_LSL)
        base = compute_sigma_level(
            mean=ref.SIGLEVEL_MEAN, sigma=ref.SIGLEVEL_SIGMA, spec=spec
        )
        shifted = compute_sigma_level(
            mean=ref.SIGLEVEL_MEAN,
            sigma=ref.SIGLEVEL_SIGMA,
            spec=spec,
            apply_1p5_shift=True,
        )
        assert shifted.with_shift is True
        assert shifted.z_bench == pytest.approx(base.z_bench + 1.5, rel=1e-9)

    def test_zero_sigma_sigma_level(self) -> None:
        spec = SpecLimits(usl=16.0, lsl=4.0)
        sl = compute_sigma_level(mean=10.0, sigma=0.0, spec=spec)
        assert sl.p_total == 0.0
        assert sl.z_bench == math.inf
        assert sl.dpmo == 0.0


class TestZeroSigmaLevelMeanPosition:
    """A1 regression: sigma=0 sigma-level MUST depend on the mean's position.

    The original implementation unconditionally reported p_total=0, DPMO=0 and
    Z_bench=+inf whenever sigma==0, regardless of where the mean sat. That let
    the report claim Cpk=0 (or -inf) alongside DPMO=0 / Z=+inf for a process
    producing 100% nonconforming parts.
    """

    TWO_SIDED = SpecLimits(usl=16.0, lsl=4.0)

    def test_1_inside_two_sided(self) -> None:
        sl = compute_sigma_level(mean=10.0, sigma=0.0, spec=self.TWO_SIDED)
        assert sl.p_total == 0.0
        assert sl.dpmo == 0.0
        assert sl.z_bench == math.inf
        assert sl.z_usl == math.inf and sl.z_lsl == math.inf

    def test_2_exactly_at_usl_is_undefined(self) -> None:
        sl = compute_sigma_level(mean=16.0, sigma=0.0, spec=self.TWO_SIDED)
        assert math.isnan(sl.p_total)
        assert math.isnan(sl.dpmo)
        assert math.isnan(sl.z_bench)
        assert math.isnan(sl.z_usl)
        # The conforming side stays well-defined.
        assert sl.z_lsl == math.inf

    def test_3_exactly_at_lsl_is_undefined(self) -> None:
        sl = compute_sigma_level(mean=4.0, sigma=0.0, spec=self.TWO_SIDED)
        assert math.isnan(sl.p_total)
        assert math.isnan(sl.dpmo)
        assert math.isnan(sl.z_bench)
        assert math.isnan(sl.z_lsl)
        assert sl.z_usl == math.inf

    def test_4_above_usl_is_fully_nonconforming(self) -> None:
        sl = compute_sigma_level(mean=20.0, sigma=0.0, spec=self.TWO_SIDED)
        assert sl.p_above == 1.0
        assert sl.p_total == 1.0
        assert sl.dpmo == pytest.approx(1e6)
        assert sl.z_bench == -math.inf
        assert sl.z_usl == -math.inf

    def test_5_below_lsl_is_fully_nonconforming(self) -> None:
        sl = compute_sigma_level(mean=1.0, sigma=0.0, spec=self.TWO_SIDED)
        assert sl.p_below == 1.0
        assert sl.p_total == 1.0
        assert sl.dpmo == pytest.approx(1e6)
        assert sl.z_bench == -math.inf
        assert sl.z_lsl == -math.inf

    def test_6_usl_only(self) -> None:
        spec = SpecLimits(usl=16.0)
        inside = compute_sigma_level(mean=10.0, sigma=0.0, spec=spec)
        assert inside.p_total == 0.0 and inside.z_bench == math.inf
        assert inside.z_lsl is None  # absent bound stays None
        outside = compute_sigma_level(mean=20.0, sigma=0.0, spec=spec)
        assert outside.p_total == 1.0 and outside.dpmo == pytest.approx(1e6)
        assert outside.z_bench == -math.inf
        assert outside.z_lsl is None
        on_bound = compute_sigma_level(mean=16.0, sigma=0.0, spec=spec)
        assert math.isnan(on_bound.p_total)

    def test_7_lsl_only(self) -> None:
        spec = SpecLimits(lsl=4.0)
        inside = compute_sigma_level(mean=10.0, sigma=0.0, spec=spec)
        assert inside.p_total == 0.0 and inside.z_bench == math.inf
        assert inside.z_usl is None
        outside = compute_sigma_level(mean=1.0, sigma=0.0, spec=spec)
        assert outside.p_total == 1.0 and outside.dpmo == pytest.approx(1e6)
        assert outside.z_bench == -math.inf
        assert outside.z_usl is None
        on_bound = compute_sigma_level(mean=4.0, sigma=0.0, spec=spec)
        assert math.isnan(on_bound.p_total)

    def test_no_contradiction_between_cpk_and_dpmo(self) -> None:
        """The exact contradiction the audit flagged must be impossible."""
        cases = [
            (10.0, self.TWO_SIDED), (16.0, self.TWO_SIDED), (4.0, self.TWO_SIDED),
            (20.0, self.TWO_SIDED), (1.0, self.TWO_SIDED),
            (10.0, SpecLimits(usl=16.0)), (20.0, SpecLimits(usl=16.0)),
            (10.0, SpecLimits(lsl=4.0)), (1.0, SpecLimits(lsl=4.0)),
        ]
        for mean, spec in cases:
            sl = compute_sigma_level(mean=mean, sigma=0.0, spec=spec)
            idx = compute_index_set(
                mean=mean, sigma=0.0, spec=spec, family="capability"
            )
            # A non-positive Cpk must never coexist with a zero-defect claim.
            if idx.cpk is not None and idx.cpk <= 0.0:
                assert not (sl.dpmo == 0.0 and sl.z_bench == math.inf), (
                    f"contradiction at mean={mean}, spec={spec}"
                )

    def test_nonzero_sigma_behavior_unchanged(self) -> None:
        """Guard: the fix must not perturb the ordinary sigma > 0 path."""
        sl = compute_sigma_level(mean=10.0, sigma=2.0, spec=self.TWO_SIDED)
        assert sl.z_usl == pytest.approx(3.0)
        assert sl.z_lsl == pytest.approx(3.0)
        assert sl.p_total == pytest.approx(2 * 0.00134989803, rel=1e-6)
        assert sl.z_bench == pytest.approx(2.78215, abs=1e-4)

    def test_zero_sigma_shift_flag_truthful(self) -> None:
        """1.5-sigma shift cannot be added to inf/NaN; flag stays truthful."""
        sl = compute_sigma_level(
            mean=10.0, sigma=0.0, spec=self.TWO_SIDED, apply_1p5_shift=True
        )
        assert sl.with_shift is True
        assert sl.z_bench == math.inf  # inf + 1.5 is still inf, not corrupted


class TestCpkConfidenceInterval:
    """Requirement 8: CI valid and documented."""

    def test_ci_case_a(self) -> None:
        ci = cpk_confidence_interval(ref.CI_A_CPK, ref.CI_A_N)
        assert ci.lower == pytest.approx(ref.CI_A_LOWER, rel=1e-6)
        assert ci.upper == pytest.approx(ref.CI_A_UPPER, rel=1e-6)
        assert ci.confidence == pytest.approx(0.95)
        assert ci.approximate is True  # never exact
        assert ci.lower < ci.cpk < ci.upper

    def test_ci_case_b(self) -> None:
        ci = cpk_confidence_interval(ref.CI_B_CPK, ref.CI_B_N)
        assert ci.lower == pytest.approx(ref.CI_B_LOWER, rel=1e-6)
        assert ci.upper == pytest.approx(ref.CI_B_UPPER, rel=1e-6)

    def test_ci_custom_confidence(self) -> None:
        ci90 = cpk_confidence_interval(1.0, 50, alpha=0.10)
        ci95 = cpk_confidence_interval(1.0, 50, alpha=0.05)
        assert ci90.confidence == pytest.approx(0.90)
        # 90% interval is narrower than 95%
        assert ci90.half_width < ci95.half_width

    def test_small_sample_caveat_attached(self) -> None:
        ci = cpk_confidence_interval(1.0, 10)  # N < 30
        assert any("Small sample" in c for c in ci.caveats)

    def test_extra_caveats_preserved(self) -> None:
        ci = cpk_confidence_interval(
            1.0, 50, extra_caveats=("within-sigma dof caveat",)
        )
        assert "within-sigma dof caveat" in ci.caveats

    def test_ci_rejects_infinite_cpk(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            cpk_confidence_interval(math.inf, 50)

    def test_ci_rejects_n_below_2(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            cpk_confidence_interval(1.0, 1)

    def test_ci_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            cpk_confidence_interval(1.0, 50, alpha=1.5)

    def test_ci_dataclass_as_dict(self) -> None:
        ci = cpk_confidence_interval(1.0, 50)
        d = ci.as_dict()
        assert d["approximate"] is True
        assert "caveats" in d
        assert isinstance(ci, CpkConfidenceInterval)
