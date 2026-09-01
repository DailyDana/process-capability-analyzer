"""Tests for the capability report builder (app.capability.report).

This is where the end-to-end M2 requirements are proven:
  1. within sigma != overall sigma on drifting data
  2. Cp/Cpk and Pp/Ppk therefore differ
  6. insufficient sample/subgroup counts -> warnings, not failures
  + the cardinal invariant: Pp/Ppk NEVER reuse the within-subgroup sigma.

Expected values come from tests.fixtures.reference_data (hand-computed).
"""

from __future__ import annotations

from app.capability.report import build_capability_report
from app.data.models import SpecLimits
from app.statistics.variation import (
    grand_mean,
    sigma_overall,
    sigma_within_from_subgroups,
)
import pytest

from tests.fixtures import reference_data as ref


def _drift_report(**overrides):
    """Build a report for the drift dataset with correctly separated sigmas."""
    within = sigma_within_from_subgroups(ref.DRIFT_SUBGROUPS).sigma
    overall = sigma_overall(ref.DRIFT_ALL)
    mean = grand_mean(ref.DRIFT_ALL)
    spec = SpecLimits(usl=ref.DRIFT_SPEC_USL, lsl=ref.DRIFT_SPEC_LSL)
    kwargs = dict(
        mean=mean,
        sigma_within=within,
        sigma_overall=overall,
        spec=spec,
        n_observations=len(ref.DRIFT_ALL),
        n_subgroups=len(ref.DRIFT_SUBGROUPS),
    )
    kwargs.update(overrides)
    return build_capability_report(**kwargs)


class TestWithinVsOverallDrivesIndices:
    """Requirements 1 & 2."""

    def test_within_and_overall_sigma_differ(self) -> None:
        rep = _drift_report()
        assert rep.sigma_within == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)
        assert rep.sigma_overall == pytest.approx(ref.DRIFT_SIGMA_OVERALL, rel=1e-9)
        assert rep.sigma_overall > 5.0 * rep.sigma_within

    def test_capability_uses_within(self) -> None:
        rep = _drift_report()
        assert rep.capability is not None
        assert rep.capability.sigma == pytest.approx(ref.DRIFT_SIGMA_WITHIN, rel=1e-9)
        assert rep.capability.cp == pytest.approx(ref.DRIFT_CP, rel=1e-9)
        assert rep.capability.cpk == pytest.approx(ref.DRIFT_CPK, rel=1e-9)
        assert rep.capability.cpu == pytest.approx(ref.DRIFT_CPU, rel=1e-9)
        assert rep.capability.cpl == pytest.approx(ref.DRIFT_CPL, rel=1e-9)

    def test_performance_uses_overall(self) -> None:
        rep = _drift_report()
        assert rep.performance.sigma == pytest.approx(ref.DRIFT_SIGMA_OVERALL, rel=1e-9)
        assert rep.performance.cp == pytest.approx(ref.DRIFT_PP, rel=1e-9)
        assert rep.performance.cpk == pytest.approx(ref.DRIFT_PPK, rel=1e-9)
        assert rep.performance.cpu == pytest.approx(ref.DRIFT_PPU, rel=1e-9)
        assert rep.performance.cpl == pytest.approx(ref.DRIFT_PPL, rel=1e-9)

    def test_capability_and_performance_differ(self) -> None:
        rep = _drift_report()
        # Cp >> Pp and Cpk >> Ppk because within << overall on drifting data.
        assert rep.capability.cp > 5.0  # ~5.64
        assert rep.performance.cp < 1.0  # ~0.77
        assert rep.capability.cpk > rep.performance.cpk

    def test_performance_never_reuses_within_sigma(self) -> None:
        """Cardinal invariant: swap within to a bogus value; Pp/Ppk must not move."""
        overall = sigma_overall(ref.DRIFT_ALL)
        mean = grand_mean(ref.DRIFT_ALL)
        spec = SpecLimits(usl=ref.DRIFT_SPEC_USL, lsl=ref.DRIFT_SPEC_LSL)
        rep_a = build_capability_report(
            mean=mean, sigma_within=0.01, sigma_overall=overall, spec=spec,
            n_observations=9, n_subgroups=3,
        )
        rep_b = build_capability_report(
            mean=mean, sigma_within=999.0, sigma_overall=overall, spec=spec,
            n_observations=9, n_subgroups=3,
        )
        # Performance identical regardless of within sigma...
        assert rep_a.performance.cpk == pytest.approx(rep_b.performance.cpk)
        assert rep_a.performance.cp == pytest.approx(rep_b.performance.cp)
        assert rep_a.performance.cpk == pytest.approx(ref.DRIFT_PPK, rel=1e-9)
        # ...while capability DID change with within sigma (sanity of the test).
        assert rep_a.capability.cpk != pytest.approx(rep_b.capability.cpk)


class TestInsufficientDataWarnsNotFails:
    """Requirement 6: small N / few subgroups warn, indices still produced."""

    def test_small_data_still_returns_indices(self) -> None:
        rep = _drift_report()  # N=9 (<30), 3 subgroups (<20)
        codes = {w.code for w in rep.warnings}
        assert "small_sample" in codes
        assert "few_subgroups" in codes
        # Crucially, indices are NOT suppressed:
        assert rep.capability is not None and rep.capability.cpk is not None
        assert rep.performance.cpk is not None

    def test_cpk_ci_present_and_flagged_approximate(self) -> None:
        rep = _drift_report()
        assert rep.cpk_ci is not None
        assert rep.cpk_ci.approximate is True
        # within-sigma dof caveat must be attached
        assert any("within-subgroup sigma" in c for c in rep.cpk_ci.caveats)

    def test_stability_note_present_by_default(self) -> None:
        rep = _drift_report()
        assert any(w.code == "stability_not_verified" for w in rep.warnings)


class TestOneSidedReport:
    def test_one_sided_upper_report(self) -> None:
        spec = SpecLimits(usl=ref.ONESIDED_UPPER_USL)
        rep = build_capability_report(
            mean=ref.SYMM_MEAN,
            sigma_within=ref.SYMM_SIGMA,
            sigma_overall=ref.SYMM_SIGMA,
            spec=spec,
            n_observations=5,
            n_subgroups=None,
        )
        assert rep.capability.cp is None  # one-sided -> no potential index
        assert rep.capability.cpk == pytest.approx(ref.ONESIDED_UPPER_CPK, rel=1e-9)
        assert rep.performance.cp is None


class TestZeroVariationReport:
    def test_zero_variation_report_no_ci(self) -> None:
        spec = SpecLimits(usl=ref.ZERO_USL, lsl=ref.ZERO_LSL)
        rep = build_capability_report(
            mean=10.0,
            sigma_within=0.0,
            sigma_overall=0.0,
            spec=spec,
            n_observations=4,
            n_subgroups=None,
        )
        import math
        assert rep.capability.cpk == math.inf
        assert rep.performance.cpk == math.inf
        # CI must be skipped (Cpk is +inf), not crash.
        assert rep.cpk_ci is None
        assert any(w.code == "zero_variation" for w in rep.warnings)


class TestUnorderedNoWithinSigma:
    def test_no_within_sigma_only_performance(self) -> None:
        # Unordered data: no within sigma -> capability omitted, Pp/Ppk reported.
        spec = SpecLimits(usl=ref.OFFCENTER_USL, lsl=ref.OFFCENTER_LSL)
        rep = build_capability_report(
            mean=ref.SYMM_MEAN,
            sigma_overall=ref.SYMM_SIGMA,
            spec=spec,
            n_observations=5,
            sigma_within=None,
            n_subgroups=None,
        )
        assert rep.capability is None
        assert rep.sigma_level_within is None
        assert rep.cpk_ci is None
        assert rep.performance.cpk == pytest.approx(ref.OFFCENTER_CPK, rel=1e-9)


class TestReportSerialization:
    def test_as_dict_roundtrip(self) -> None:
        rep = _drift_report()
        d = rep.as_dict()
        assert d["capability"]["cpk"] == pytest.approx(ref.DRIFT_CPK, rel=1e-9)
        assert d["performance"]["cpk"] == pytest.approx(ref.DRIFT_PPK, rel=1e-9)
        assert isinstance(d["warnings"], list)
        assert d["cpk_ci"]["approximate"] is True
