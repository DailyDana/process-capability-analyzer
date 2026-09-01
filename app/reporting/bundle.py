"""AnalysisBundle — the single typed input for all exporters.

Gathers everything an export needs, all of it already computed by the tested
analysis layers. The bundle itself computes nothing; it only holds references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.capability.report import CapabilityReport
from app.interpret.models import InterpretationReport
from app.spc.rules import RuleEvaluation, Violation
from app.statistics.descriptive import DescriptiveSummary

__all__ = ["AnalysisBundle", "ExportMeta", "MethodologyInfo"]


@dataclass(frozen=True, slots=True)
class MethodologyInfo:
    """How the analysis was performed (for the report's methodology section).

    All fields are echoes of configuration/results already produced by the
    validated layers; nothing here is computed.
    """

    data_regime: Optional[str] = None          # "subgrouped" / "individuals"
    chart_type: Optional[str] = None           # "X-bar/R", "X-bar/S", "I-MR"
    subgroup_size: Optional[int] = None
    n_subgroups: Optional[int] = None
    within_estimator: Optional[str] = None     # e.g. "rbar_d2"
    rule_set: Optional[str] = None             # e.g. "nelson"
    n_input_rows: Optional[int] = None
    n_used: Optional[int] = None
    n_excluded: Optional[int] = None
    exclusion_notes: tuple[str, ...] = field(default_factory=tuple)

    def as_rows(self) -> list[tuple[str, object]]:
        """Return (label, value) rows for tabular display, omitting None."""
        candidates = [
            ("Data regime", self.data_regime),
            ("Control chart", self.chart_type),
            ("Subgroup size (n)", self.subgroup_size),
            ("Number of subgroups", self.n_subgroups),
            ("Within-sigma estimator", self.within_estimator),
            ("SPC rule set", self.rule_set),
            ("Input rows", self.n_input_rows),
            ("Observations used", self.n_used),
            ("Observations excluded", self.n_excluded),
        ]
        return [(k, v) for k, v in candidates if v is not None]


@dataclass(frozen=True, slots=True)
class ExportMeta:
    """Traceability metadata for a report (all optional)."""

    characteristic: Optional[str] = None
    unit: Optional[str] = None
    part_id: Optional[str] = None
    machine: Optional[str] = None
    operator: Optional[str] = None
    timestamp: Optional[str] = None
    source_file: Optional[str] = None
    notes: Optional[str] = None

    def as_rows(self) -> list[tuple[str, object]]:
        """Return (label, value) rows for tabular export, omitting None."""
        candidates = [
            ("Characteristic", self.characteristic),
            ("Unit", self.unit),
            ("Part ID", self.part_id),
            ("Machine", self.machine),
            ("Operator", self.operator),
            ("Timestamp", self.timestamp),
            ("Source file", self.source_file),
            ("Notes", self.notes),
        ]
        return [(k, v) for k, v in candidates if v is not None]


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Everything an exporter needs, already computed elsewhere.

    Attributes
    ----------
    meta:
        Traceability metadata.
    capability:
        The M2 :class:`CapabilityReport` (indices, sigmas, sigma level, CI,
        stability, normality, warnings).
    interpretation:
        The M5 :class:`InterpretationReport` (conclusions + recommendations).
    descriptive:
        Optional M1 :class:`DescriptiveSummary` of the raw measurements.
    measurements:
        Optional raw measurement values (for the Measurements sheet).
    measurement_indices:
        Optional ORIGINAL source-row index for each entry of ``measurements``
        (i.e. ``DetailedCleaning.used_indices``). Exporters use this for the
        "Original row" column so that a measurement's identity matches the
        indices cited in the violations tables. When ``None``, exporters fall
        back to the enumeration position, which is correct only when nothing
        was excluded during cleaning.
    subgroup_labels:
        Optional per-measurement subgroup labels, parallel to ``measurements``.
    violations:
        Optional flat list of SPC :class:`Violation` objects (already detected).
    """

    meta: ExportMeta
    capability: CapabilityReport
    interpretation: InterpretationReport
    descriptive: Optional[DescriptiveSummary] = None
    measurements: Optional[Sequence[float]] = None
    measurement_indices: Optional[Sequence[int]] = None
    subgroup_labels: Optional[Sequence[object]] = None
    violations: tuple[Violation, ...] = field(default_factory=tuple)
    methodology: Optional[MethodologyInfo] = None

    @classmethod
    def from_evaluations(
        cls,
        *,
        meta: ExportMeta,
        capability: CapabilityReport,
        interpretation: InterpretationReport,
        descriptive: Optional[DescriptiveSummary] = None,
        measurements: Optional[Sequence[float]] = None,
        measurement_indices: Optional[Sequence[int]] = None,
        subgroup_labels: Optional[Sequence[object]] = None,
        evaluations: Optional[Sequence[RuleEvaluation]] = None,
        methodology: Optional[MethodologyInfo] = None,
    ) -> "AnalysisBundle":
        """Convenience builder that flattens rule evaluations into violations."""
        violations: list[Violation] = []
        for ev in evaluations or ():
            violations.extend(ev.violations)
        return cls(
            meta=meta,
            capability=capability,
            interpretation=interpretation,
            descriptive=descriptive,
            measurements=measurements,
            measurement_indices=measurement_indices,
            subgroup_labels=subgroup_labels,
            violations=tuple(violations),
            methodology=methodology,
        )
