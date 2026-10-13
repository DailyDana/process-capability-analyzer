"""Pure chart models for a Gage R&R study (no matplotlib here).

Three charts, the standard minimum for reading a crossed study:

* **Components of variation** -- %Contribution / %Study Var / %Tolerance per
  source, taken verbatim from the result's summary rows.
* **R chart by operator** -- every part x operator range, grouped by operator,
  with R-bar and UCL from the result's range check (never recomputed here).
* **Operator x part interaction** -- the average of each operator on each part;
  roughly parallel lines mean no interaction, crossing lines mean some
  operators measure some parts differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.msa.models import GageRRResult, GageStudy
from app.msa.ranges import cell_ranges

__all__ = [
    "ComponentsModel",
    "RangeByOperatorModel",
    "InteractionModel",
    "build_components_model",
    "build_range_model",
    "build_interaction_model",
    "COMPONENT_SOURCES",
]

#: Sources shown in the components chart, in display order.
COMPONENT_SOURCES = ("Total Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part")


@dataclass(frozen=True, slots=True)
class ComponentsModel:
    sources: tuple[str, ...]
    pct_contribution: tuple[Optional[float], ...]
    pct_study_var: tuple[Optional[float], ...]
    pct_tolerance: Optional[tuple[Optional[float], ...]]
    acceptable_max: float
    marginal_max: float


@dataclass(frozen=True, slots=True)
class RangeByOperatorModel:
    operators: tuple[str, ...]
    parts: tuple[str, ...]
    ranges: tuple[tuple[float, ...], ...]  # [operator][part]
    out_of_control: tuple[tuple[bool, ...], ...]  # [operator][part]
    r_bar: float
    ucl: float
    unit: Optional[str]


@dataclass(frozen=True, slots=True)
class InteractionModel:
    operators: tuple[str, ...]
    parts: tuple[str, ...]
    averages: tuple[tuple[float, ...], ...]  # [operator][part]
    unit: Optional[str]


def build_components_model(
    result: GageRRResult, acceptable_max: float = 10.0, marginal_max: float = 30.0
) -> ComponentsModel:
    rows = [result.row(s) for s in COMPONENT_SOURCES]
    has_tol = result.tolerance is not None
    return ComponentsModel(
        sources=COMPONENT_SOURCES,
        pct_contribution=tuple(r.pct_contribution for r in rows),
        pct_study_var=tuple(r.pct_study_var for r in rows),
        pct_tolerance=tuple(r.pct_tolerance for r in rows) if has_tol else None,
        acceptable_max=acceptable_max,
        marginal_max=marginal_max,
    )


def build_range_model(study: GageStudy, result: GageRRResult) -> RangeByOperatorModel:
    ranges = cell_ranges(study)  # (parts, operators)
    flagged = {(p, o) for p, o, _ in result.range_check.out_of_control}
    ops, parts = study.operator_labels, study.part_labels
    return RangeByOperatorModel(
        operators=ops,
        parts=parts,
        ranges=tuple(tuple(float(ranges[i, j]) for i in range(len(parts))) for j in range(len(ops))),
        out_of_control=tuple(tuple((p, o) in flagged for p in parts) for o in ops),
        r_bar=result.range_check.r_bar,
        ucl=result.range_check.ucl,
        unit=study.unit,
    )


def build_interaction_model(study: GageStudy) -> InteractionModel:
    cell_means = np.asarray(study.values.mean(axis=2))  # (parts, operators)
    return InteractionModel(
        operators=study.operator_labels,
        parts=study.part_labels,
        averages=tuple(tuple(float(x) for x in cell_means[:, j]) for j in range(study.n_operators)),
        unit=study.unit,
    )
