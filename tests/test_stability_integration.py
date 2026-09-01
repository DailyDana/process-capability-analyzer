"""Tests for stability integration into the capability report.

Proves the key requirement: an unstable process (SPC rule violation) attaches an
explicit instability WARNING to the capability result but does NOT suppress or
alter the numerical indices — a high Cpk on an unstable process is caveated, not
hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.capability.report import StabilitySummary, build_capability_report
from app.data.models import SpecLimits
from app.services.stability import (
    stability_from_chart_pair,
    stability_from_evaluations,
)
from app.spc.limits import imr_chart, xbar_r_chart
from app.spc.rules import RuleSet, evaluate_rules


def _spec():
    return SpecLimits(usl=20.0, lsl=0.0)


class TestStabilityStates:
    def test_not_assessed_gives_info_note(self) -> None:
        rep = build_capability_report(
            mean=10.0, sigma_within=1.0, sigma_overall=1.0, spec=_spec(),
            n_observations=50, n_subgroups=25, stability=None,
        )
        codes = {w.code for w in rep.warnings}
        assert "stability_not_verified" in codes
        assert "unstable_process" not in codes

    def test_stable_gives_no_stability_warning(self) -> None:
        stab = StabilitySummary(is_stable=True)
        rep = build_capability_report(
            mean=10.0, sigma_within=1.0, sigma_overall=1.0, spec=_spec(),
            n_observations=50, n_subgroups=25, stability=stab,
        )
        codes = {w.code for w in rep.warnings}
        assert "stability_not_verified" not in codes
        assert "unstable_process" not in codes

    def test_unstable_warns_but_keeps_indices(self) -> None:
        stab = StabilitySummary(
            is_stable=False, triggered_rule_ids=("N1", "N2"), n_violations=3
        )
        rep = build_capability_report(
            mean=10.0, sigma_within=1.0, sigma_overall=1.0, spec=_spec(),
            n_observations=50, n_subgroups=25, stability=stab,
        )
        unstable = [w for w in rep.warnings if w.code == "unstable_process"]
        assert unstable
        assert "N1" in unstable[0].message and "N2" in unstable[0].message
        # Indices MUST still be present and unaltered.
        assert rep.capability is not None
        assert rep.capability.cpk is not None
        assert rep.performance.cpk is not None
        # The stability summary is carried on the report.
        assert rep.stability is stab
        assert rep.as_dict()["stability"]["is_stable"] is False

    def test_high_cpk_unstable_not_declared_capable(self) -> None:
        # Deliberately high Cpk but unstable -> warning must be present.
        stab = StabilitySummary(is_stable=False, triggered_rule_ids=("N1",))
        rep = build_capability_report(
            mean=10.0, sigma_within=0.5, sigma_overall=0.5, spec=_spec(),
            n_observations=50, n_subgroups=25, stability=stab,
        )
        assert rep.capability.cpk > 2.0  # excellent-looking Cpk
        assert any(w.code == "unstable_process" for w in rep.warnings)

    def test_unstable_adds_ci_caveat(self) -> None:
        stab = StabilitySummary(is_stable=False, triggered_rule_ids=("N3",))
        rep = build_capability_report(
            mean=10.0, sigma_within=1.0, sigma_overall=1.0, spec=_spec(),
            n_observations=50, n_subgroups=25, stability=stab,
        )
        assert rep.cpk_ci is not None
        assert any("stability is not confirmed" in c.lower() for c in rep.cpk_ci.caveats)


class TestServicesAdapter:
    def test_stability_from_evaluations_stable(self) -> None:
        # Two in-control I-charts.
        p1 = imr_chart(np.array([10.0, 10.1, 9.9, 10.05, 9.95, 10.0, 10.1]))
        ev = evaluate_rules(p1.location, rule_set=RuleSet.NELSON)
        stab = stability_from_evaluations([ev])
        assert stab.is_stable
        assert stab.n_violations == 0

    def test_stability_from_chart_pair_detects_instability(self) -> None:
        # A clear upward run + outlier -> unstable.
        vals = np.array(
            [10.0, 10.1, 9.9, 10.0, 10.05, 9.95, 10.0, 10.1, 9.9, 25.0]
        )
        pair = imr_chart(vals)
        stab = stability_from_chart_pair(pair, rule_set=RuleSet.NELSON)
        assert not stab.is_stable
        assert "N1" in stab.triggered_rule_ids  # the outlier
        assert stab.detail  # human-readable detail present

    def test_end_to_end_unstable_flows_into_capability(self) -> None:
        # Subgroups with an out-of-control mean -> X-bar violation -> unstable.
        base = [np.array([10.0, 10.2, 9.8, 10.1, 9.9]) for _ in range(9)]
        base.append(np.array([30.0, 30.2, 29.8, 30.1, 29.9]))  # huge shift
        pair = xbar_r_chart(base)
        stab = stability_from_chart_pair(pair, rule_set=RuleSet.NELSON)
        assert not stab.is_stable
        rep = build_capability_report(
            mean=12.0, sigma_within=0.2, sigma_overall=6.0, spec=SpecLimits(usl=40, lsl=0),
            n_observations=50, n_subgroups=10, stability=stab,
        )
        assert any(w.code == "unstable_process" for w in rep.warnings)
        assert rep.capability is not None  # indices retained
