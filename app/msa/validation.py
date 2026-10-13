"""Validation of long-format Gage R&R input into a balanced :class:`GageStudy`.

Input is one row per measurement: ``part``, ``operator``, ``value`` and an
optional ``trial`` label. Unlike the capability path, **nothing is dropped**:
a missing or non-numeric measurement makes the study unbalanced, and a crossed
Gage R&R on an unbalanced study is a different (and here unsupported)
analysis. So every such row is an *error* that names the row, and the engineer
decides whether to re-measure or remove the whole part.

Errors (study rejected)
-----------------------
* columns of different length;
* a blank part/operator label, or a value that is not a finite number;
* a duplicated (part, operator, trial) when trial labels are given;
* a part x operator cell with no measurements;
* unequal numbers of trials across cells;
* fewer than 2 parts, 2 operators or 2 trials.

Warnings (study accepted)
-------------------------
* fewer than 10 parts (AIAG's typical study; PV and ndc become imprecise);
* fewer than 3 operators;
* few repeatability degrees of freedom (``parts x operators x (trials - 1)
  < 15``);
* no variation at all in the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from app.data.validation import RowStatus, clean_numeric_detailed
from app.msa.models import GageStudy

__all__ = ["GageDataValidation", "build_gage_study", "MAX_LISTED"]

#: At most this many offending rows/cells are listed individually per error.
MAX_LISTED = 8


@dataclass(frozen=True, slots=True)
class GageDataValidation:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    study: Optional[GageStudy]


def _label(token: Any) -> Optional[str]:
    if token is None:
        return None
    if isinstance(token, float) and math.isnan(token):
        return None
    text = str(token).strip()
    return text or None


def _listing(items: list[str]) -> str:
    shown = ", ".join(items[:MAX_LISTED])
    extra = len(items) - MAX_LISTED
    return shown + (f" (and {extra} more)" if extra > 0 else "")


def _trial_sort_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def build_gage_study(
    parts: Sequence[Any],
    operators: Sequence[Any],
    values: Sequence[Any],
    trials: Optional[Sequence[Any]] = None,
    unit: Optional[str] = None,
) -> GageDataValidation:
    """Validate long-format columns and assemble a balanced study.

    Parts and operators keep their order of first appearance. Within a cell,
    measurements are ordered by trial label when given (numeric labels sort
    numerically), otherwise by row order. Neither order affects any estimate;
    it only makes tables and charts reproducible.
    """
    errors: list[str] = []
    warnings: list[str] = []

    n = len(values)
    if len(parts) != n or len(operators) != n or (trials is not None and len(trials) != n):
        lens = f"part={len(parts)}, operator={len(operators)}, value={n}"
        if trials is not None:
            lens += f", trial={len(trials)}"
        return GageDataValidation(False, (f"Columns have different lengths ({lens}).",), (), None)
    if n == 0:
        return GageDataValidation(False, ("No measurements were provided.",), (), None)

    cleaned = clean_numeric_detailed(list(values))
    bad_values = [
        f"row {rec.original_index + 1} ({rec.raw!r})"
        for rec in cleaned.records
        if rec.status is not RowStatus.OK
    ]
    if bad_values:
        errors.append(
            "Every measurement must be a finite number - a crossed Gage R&R needs a "
            f"balanced study, so rows cannot be dropped. Fix or remove: {_listing(bad_values)}."
        )

    part_labels = [_label(t) for t in parts]
    op_labels = [_label(t) for t in operators]
    trial_labels = [_label(t) for t in trials] if trials is not None else None
    for name, labels in (("part", part_labels), ("operator", op_labels), ("trial", trial_labels)):
        if labels is None:
            continue
        blank = [f"row {i + 1}" for i, lab in enumerate(labels) if lab is None]
        if blank:
            errors.append(f"Missing {name} label: {_listing(blank)}.")
    if errors:
        return GageDataValidation(False, tuple(errors), (), None)

    part_order: list[str] = list(dict.fromkeys(part_labels))  # type: ignore[arg-type]
    op_order: list[str] = list(dict.fromkeys(op_labels))  # type: ignore[arg-type]

    cells: dict[tuple[str, str], list[tuple[str, float]]] = {}
    seen: dict[tuple[str, str, str], int] = {}
    duplicates: list[str] = []
    for i, rec in enumerate(cleaned.records):
        key = (part_labels[i], op_labels[i])  # type: ignore[index]
        trial = trial_labels[i] if trial_labels is not None else str(len(cells.get(key, [])) + 1)
        assert rec.value is not None and trial is not None
        if trial_labels is not None:
            tkey = (key[0], key[1], trial)
            if tkey in seen:
                duplicates.append(
                    f"rows {seen[tkey] + 1} and {i + 1} (part {key[0]}, operator {key[1]}, trial {trial})"
                )
            else:
                seen[tkey] = i
        cells.setdefault(key, []).append((trial, rec.value))
    if duplicates:
        errors.append(f"Duplicated part/operator/trial: {_listing(duplicates)}.")

    p, o = len(part_order), len(op_order)
    if p < 2:
        errors.append(f"At least 2 parts are required (found {p}).")
    if o < 2:
        errors.append(
            f"At least 2 operators are required to estimate reproducibility (found {o})."
        )

    missing = [f"part {pt} x operator {op}" for pt in part_order for op in op_order if (pt, op) not in cells]
    if missing:
        errors.append(
            "Crossed design required: every operator must measure every part. "
            f"Missing: {_listing(missing)}."
        )
    if errors:
        return GageDataValidation(False, tuple(errors), (), None)

    counts = {key: len(v) for key, v in cells.items()}
    distinct = sorted(set(counts.values()))
    if len(distinct) > 1:
        r_mode = max(distinct, key=lambda c: sum(1 for v in counts.values() if v == c))
        off = [f"part {k[0]} x operator {k[1]} ({c})" for k, c in counts.items() if c != r_mode]
        return GageDataValidation(
            False,
            (
                f"Unbalanced study: trials per cell range from {distinct[0]} to {distinct[-1]}. "
                f"Every operator must measure every part the same number of times; "
                f"cells differing from {r_mode}: {_listing(off)}.",
            ),
            (),
            None,
        )
    r = distinct[0]
    if r < 2:
        return GageDataValidation(
            False,
            (f"At least 2 trials per part and operator are required to estimate repeatability (found {r}).",),
            (),
            None,
        )

    arr = np.empty((p, o, r), dtype=float)
    for i, pt in enumerate(part_order):
        for j, op in enumerate(op_order):
            entries = cells[(pt, op)]
            if trial_labels is not None:
                entries = sorted(entries, key=lambda e: _trial_sort_key(e[0]))
            arr[i, j, :] = [v for _, v in entries]

    if p < 10:
        warnings.append(
            f"Only {p} parts: AIAG's typical study uses 10 parts spanning the process range. "
            "Part variation (PV) and ndc are imprecise with fewer."
        )
    if o < 3:
        warnings.append(f"Only {o} operators: reproducibility rests on {o - 1} degree(s) of freedom.")
    df_e = p * o * (r - 1)
    if df_e < 15:
        warnings.append(
            f"Repeatability has only {df_e} degrees of freedom (parts x operators x (trials - 1)); "
            "consider more trials or parts."
        )
    if float(arr.max() - arr.min()) == 0.0:
        warnings.append(
            "All measurements are identical: the gauge cannot resolve these parts "
            "(check resolution) - percentages are undefined."
        )

    study = GageStudy(values=arr, part_labels=tuple(part_order), operator_labels=tuple(op_order), unit=unit)
    return GageDataValidation(True, (), tuple(warnings), study)
