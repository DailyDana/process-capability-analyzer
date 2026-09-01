"""Capability report builder — wires the correct sigma to each index family.

This is the single place that couples sigma sources to index families, so the
Cp/Cpk-vs-Pp/Ppk invariant lives in exactly one auditable spot:

* ``sigma_within``  -> capability family (Cp, Cpu, Cpl, Cpk)
* ``sigma_overall`` -> performance family (Pp, Ppu, Ppl, Ppk)

It also attaches the sigma-level (short- and long-term), the Cpk confidence
interval, and the reliability guard warnings. See FORMULAS.md §2.4–2.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

from app.capability.guards import (
    GuardThresholds,
    NormalityResult,
    ReliabilityWarning,
    evaluate_guards,
)
from app.capability.indices import (
    CpkConfidenceInterval,
    IndexSet,
    SigmaLevel,
    compute_index_set,
    compute_sigma_level,
    cpk_confidence_interval,
    DEFAULT_CI_ALPHA,
)
from app.data.models import SpecLimits

__all__ = [
    "CapabilityReport",
    "StabilitySummary",
    "build_capability_report",
]


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    """Compact, framework-agnostic view of an SPC stability verdict.

    The capability layer consumes this *structural* summary rather than importing
    the spc package directly, keeping the dependency direction clean. Build it
    from an :class:`~app.spc.rules.RuleEvaluation` via
    :meth:`from_rule_evaluation` (defined in the services layer) or construct it
    directly.

    Attributes
    ----------
    is_stable:
        ``True`` iff the process shows no control-rule violations.
    triggered_rule_ids:
        IDs of the rules that fired (empty when stable).
    n_violations:
        Total number of violations detected.
    detail:
        Optional free-text note (e.g. which chart was assessed).
    """

    is_stable: bool
    triggered_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    n_violations: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "is_stable": self.is_stable,
            "triggered_rule_ids": list(self.triggered_rule_ids),
            "n_violations": self.n_violations,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Full capability/performance analysis for one characteristic.

    Attributes
    ----------
    capability:
        Cp-family indices from within-subgroup sigma (``None`` if no within
        sigma was available, e.g. unordered data).
    performance:
        Pp-family indices from overall sigma.
    sigma_level_within, sigma_level_overall:
        Z-bench / DPMO from each sigma (``None`` when that sigma is absent).
    cpk_ci:
        Approximate CI for the within-sigma Cpk (``None`` when not computable —
        no within sigma, non-finite Cpk, or n < 2).
    warnings:
        Reliability caveats (advisory; indices are never suppressed).
    mean, sigma_within, sigma_overall, n_observations, n_subgroups:
        The inputs, echoed for transparency/reporting.
    """

    mean: float
    sigma_within: Optional[float]
    sigma_overall: float
    n_observations: int
    n_subgroups: Optional[int]
    spec: SpecLimits
    capability: Optional[IndexSet]
    performance: IndexSet
    sigma_level_within: Optional[SigmaLevel]
    sigma_level_overall: SigmaLevel
    cpk_ci: Optional[CpkConfidenceInterval]
    normality: Optional[NormalityResult]
    stability: Optional["StabilitySummary"] = None
    warnings: tuple[ReliabilityWarning, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "sigma_within": self.sigma_within,
            "sigma_overall": self.sigma_overall,
            "n_observations": self.n_observations,
            "n_subgroups": self.n_subgroups,
            "capability": self.capability.as_dict() if self.capability else None,
            "performance": self.performance.as_dict(),
            "sigma_level_within": (
                self.sigma_level_within.as_dict() if self.sigma_level_within else None
            ),
            "sigma_level_overall": self.sigma_level_overall.as_dict(),
            "cpk_ci": self.cpk_ci.as_dict() if self.cpk_ci else None,
            "normality": self.normality.as_dict() if self.normality else None,
            "stability": self.stability.as_dict() if self.stability else None,
            "warnings": [
                {"code": w.code, "severity": w.severity.value, "message": w.message}
                for w in self.warnings
            ],
        }


def build_capability_report(
    *,
    mean: float,
    sigma_overall: float,
    spec: SpecLimits,
    n_observations: int,
    sigma_within: Optional[float] = None,
    n_subgroups: Optional[int] = None,
    normality: Optional[NormalityResult] = None,
    stability: Optional[StabilitySummary] = None,
    ci_alpha: float = DEFAULT_CI_ALPHA,
    apply_1p5_shift: bool = False,
    thresholds: GuardThresholds = GuardThresholds(),
) -> CapabilityReport:
    """Assemble a :class:`CapabilityReport`, binding each sigma to its family.

    ``sigma_within`` is optional: when ``None`` (e.g. unordered data with no
    valid subgroup structure) the capability family and its sigma level / Cpk CI
    are omitted, and only overall performance (Pp/Ppk) is reported.

    ``stability`` is an optional :class:`StabilitySummary` from the SPC layer:

    * ``None``            -> stability not assessed (INFO note attached).
    * ``is_stable=True``  -> stable (no stability warning).
    * ``is_stable=False`` -> UNSTABLE: a WARNING is attached and the Cpk CI gains
      an explicit instability caveat. The capability indices are **never**
      suppressed or altered — only caveated (per project requirement:
      "do not automatically declare a process capable ... do not delete or
      suppress the numerical capability indices").
    """
    # Three-valued stability state for the guards.
    stability_state: Optional[bool] = None if stability is None else stability.is_stable
    stability_detail = "" if stability is None else (
        stability.detail
        or (
            "rules: " + ", ".join(stability.triggered_rule_ids)
            if stability.triggered_rule_ids
            else ""
        )
    )
    stability_ok_for_ci = stability is not None and stability.is_stable

    # --- Performance family: OVERALL sigma only. Never the within sigma. ---
    performance = compute_index_set(
        mean=mean, sigma=sigma_overall, spec=spec, family="performance"
    )
    sigma_level_overall = compute_sigma_level(
        mean=mean, sigma=sigma_overall, spec=spec, apply_1p5_shift=apply_1p5_shift
    )

    # --- Capability family: WITHIN sigma only (if available). ---
    capability: Optional[IndexSet] = None
    sigma_level_within: Optional[SigmaLevel] = None
    cpk_ci: Optional[CpkConfidenceInterval] = None
    if sigma_within is not None:
        capability = compute_index_set(
            mean=mean, sigma=sigma_within, spec=spec, family="capability"
        )
        sigma_level_within = compute_sigma_level(
            mean=mean, sigma=sigma_within, spec=spec, apply_1p5_shift=apply_1p5_shift
        )
        # Cpk CI is built on the within-sigma Cpk when it is finite and n allows.
        if (
            capability.cpk is not None
            and isfinite(capability.cpk)
            and n_observations >= 2
        ):
            extra: list[str] = [
                "Cpk uses a within-subgroup sigma estimator, so the effective "
                "degrees of freedom differ from N-1; this interval is a "
                "first-order approximation only.",
            ]
            if normality is not None and normality.normal is False:
                extra.append(
                    "Normality is rejected for these data; the normal-theory "
                    "interval is not valid and must not be read as exact."
                )
            if not stability_ok_for_ci:
                extra.append(
                    "Process stability is not confirmed; the interval assumes a "
                    "stable, in-control process."
                )
            cpk_ci = cpk_confidence_interval(
                capability.cpk,
                n_observations,
                alpha=ci_alpha,
                extra_caveats=tuple(extra),
            )

    # --- Reliability guards (advisory). Use within sigma if present, else overall. ---
    guard_sigma = sigma_within if sigma_within is not None else sigma_overall
    warnings = evaluate_guards(
        n_observations=n_observations,
        n_subgroups=n_subgroups,
        sigma=guard_sigma,
        normality=normality,
        stability_state=stability_state,
        stability_detail=stability_detail,
        thresholds=thresholds,
    )

    return CapabilityReport(
        mean=float(mean),
        sigma_within=None if sigma_within is None else float(sigma_within),
        sigma_overall=float(sigma_overall),
        n_observations=int(n_observations),
        n_subgroups=n_subgroups,
        spec=spec,
        capability=capability,
        performance=performance,
        sigma_level_within=sigma_level_within,
        sigma_level_overall=sigma_level_overall,
        cpk_ci=cpk_ci,
        normality=normality,
        stability=stability,
        warnings=tuple(warnings),
    )
