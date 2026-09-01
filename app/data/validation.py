"""Validation and coercion of raw measurement input.

This layer sits between untrusted input (CSV cells, Excel cells, manual grid
entries) and the clean :class:`~app.data.models.Dataset`. It never silently
discards data: every coercion decision is recorded in a structured
:class:`CleaningReport` so the UI/report can tell the engineer exactly what was
dropped and why.

Guarantees
----------
* Non-numeric entries (strings like ``"N/A"``, ``"bad"``, empty cells) are
  detected and either dropped (recorded) or cause a raise, per ``on_invalid``.
* NaN / blank cells are treated as missing and dropped (recorded).
* ``inf`` values are treated as invalid (a measurement can never be infinite).
* Duplicate values are *not* removed — repeated readings are legitimate in
  measurement data — but they are counted and reported for awareness.
* Full precision is preserved; no rounding occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CleaningReport",
    "clean_numeric",
    "clean_numeric_detailed",
    "RowStatus",
    "RowRecord",
    "DetailedCleaning",
    "validate_spec_limits",
    "SpecValidation",
]

OnInvalid = Literal["drop", "raise"]


class RowStatus(str, Enum):
    """Per-row outcome of numeric cleaning.

    ``OK`` rows are used in the analysis; all other statuses are excluded and
    the reason is retained so the UI can show exactly what happened to which
    original row. Nothing is ever silently dropped.
    """

    OK = "ok"
    MISSING = "missing"
    NON_NUMERIC = "non_numeric"
    NON_FINITE = "non_finite"


@dataclass(frozen=True, slots=True)
class RowRecord:
    """One input row's cleaning outcome, keyed by its ORIGINAL index.

    Attributes
    ----------
    original_index:
        Position in the user's input (0-based), preserved regardless of drops.
    raw:
        The raw token as supplied.
    status:
        :class:`RowStatus`.
    value:
        The parsed float when ``status is OK``, else ``None``.
    reason:
        Human-readable explanation for an excluded row ("" when OK).
    """

    original_index: int
    raw: Any
    status: RowStatus
    value: Optional[float] = None
    reason: str = ""

    @property
    def is_used(self) -> bool:
        return self.status is RowStatus.OK


@dataclass(frozen=True, slots=True)
class DetailedCleaning:
    """Per-row cleaning outcome plus the aggregate :class:`CleaningReport`."""

    records: tuple[RowRecord, ...]
    report: "CleaningReport"

    @property
    def values(self) -> NDArray[np.float64]:
        """The retained values, in original order."""
        return np.asarray(
            [r.value for r in self.records if r.is_used], dtype=np.float64
        )

    @property
    def used_indices(self) -> tuple[int, ...]:
        """Original indices of the retained rows."""
        return tuple(r.original_index for r in self.records if r.is_used)

    @property
    def excluded(self) -> tuple[RowRecord, ...]:
        """Records for every excluded row, with reasons."""
        return tuple(r for r in self.records if not r.is_used)


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Structured record of what happened while cleaning a value column.

    Attributes
    ----------
    n_input:
        Number of raw entries received.
    n_valid:
        Number of finite numeric values retained.
    dropped_missing:
        Count of blank/None/NaN entries dropped.
    dropped_non_numeric:
        Count of entries that could not be parsed as numbers.
    dropped_non_finite:
        Count of ``inf``/``-inf`` entries dropped.
    n_duplicates:
        Number of values that are duplicates of an earlier value (informational
        only; duplicates are retained).
    invalid_examples:
        Up to a few example raw tokens that failed numeric parsing, to help the
        user locate data-entry problems.
    """

    n_input: int
    n_valid: int
    dropped_missing: int = 0
    dropped_non_numeric: int = 0
    dropped_non_finite: int = 0
    n_duplicates: int = 0
    invalid_examples: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_dropped(self) -> int:
        """Total number of entries removed for any reason."""
        return self.dropped_missing + self.dropped_non_numeric + self.dropped_non_finite

    @property
    def is_clean(self) -> bool:
        """True if nothing had to be dropped."""
        return self.n_dropped == 0

    def summary(self) -> str:
        """Human-readable one-line summary for UI/report display."""
        parts = [f"{self.n_valid}/{self.n_input} values usable"]
        if self.dropped_missing:
            parts.append(f"{self.dropped_missing} missing")
        if self.dropped_non_numeric:
            parts.append(f"{self.dropped_non_numeric} non-numeric")
        if self.dropped_non_finite:
            parts.append(f"{self.dropped_non_finite} non-finite")
        if self.n_duplicates:
            parts.append(f"{self.n_duplicates} duplicate value(s)")
        return "; ".join(parts)


def _looks_missing(token: Any) -> bool:
    """Return True if a raw token represents a missing value."""
    if token is None:
        return True
    if isinstance(token, float) and np.isnan(token):
        return True
    if isinstance(token, str):
        t = token.strip().lower()
        return t in {"", "na", "n/a", "nan", "null", "none", "-", "--"}
    return False


def clean_numeric_detailed(
    raw: Sequence[Any],
    *,
    on_invalid: OnInvalid = "drop",
) -> DetailedCleaning:
    """Clean a raw sequence, retaining a PER-ROW record keyed by original index.

    Same parsing rules as :func:`clean_numeric` (which delegates here), plus a
    :class:`RowRecord` for every input row so callers can show exactly which
    observations were excluded and why. Original indices are preserved.

    Raises
    ------
    ValueError
        Under ``on_invalid="raise"`` when a non-missing, non-numeric token is
        encountered.
    """
    records: list[RowRecord] = []
    dropped_missing = 0
    dropped_non_numeric = 0
    dropped_non_finite = 0
    invalid_examples: list[str] = []
    kept: list[float] = []

    for i, token in enumerate(raw):
        if _looks_missing(token):
            dropped_missing += 1
            records.append(
                RowRecord(
                    original_index=i, raw=token, status=RowStatus.MISSING,
                    reason="Blank or missing value.",
                )
            )
            continue
        try:
            val = float(token)
        except (TypeError, ValueError):
            dropped_non_numeric += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(repr(token))
            if on_invalid == "raise":
                raise ValueError(
                    f"Non-numeric measurement encountered: {token!r}."
                )
            records.append(
                RowRecord(
                    original_index=i, raw=token, status=RowStatus.NON_NUMERIC,
                    reason=f"Could not be parsed as a number: {token!r}.",
                )
            )
            continue
        if not np.isfinite(val):
            dropped_non_finite += 1
            records.append(
                RowRecord(
                    original_index=i, raw=token, status=RowStatus.NON_FINITE,
                    reason="Value is infinite (not a valid measurement).",
                )
            )
            continue
        kept.append(val)
        records.append(
            RowRecord(
                original_index=i, raw=token, status=RowStatus.OK, value=val,
            )
        )

    values = np.asarray(kept, dtype=np.float64)
    n_duplicates = int(values.size - np.unique(values).size) if values.size else 0

    report = CleaningReport(
        n_input=len(raw),
        n_valid=int(values.size),
        dropped_missing=dropped_missing,
        dropped_non_numeric=dropped_non_numeric,
        dropped_non_finite=dropped_non_finite,
        n_duplicates=n_duplicates,
        invalid_examples=tuple(invalid_examples),
    )
    return DetailedCleaning(records=tuple(records), report=report)


def clean_numeric(
    raw: Sequence[Any],
    *,
    on_invalid: OnInvalid = "drop",
) -> tuple[NDArray[np.float64], CleaningReport]:
    """Coerce a raw sequence into a clean ``float64`` array plus a report.

    Thin wrapper over :func:`clean_numeric_detailed` (single implementation of
    the parsing rules).

    Parameters
    ----------
    raw:
        Sequence of raw cell values (mixed types allowed).
    on_invalid:
        ``"drop"`` (default) removes non-numeric entries and records them;
        ``"raise"`` raises ``ValueError`` on the first unparseable entry.

    Returns
    -------
    (values, report):
        ``values`` is the cleaned finite array (may be empty if everything was
        dropped); ``report`` describes every coercion decision.

    Raises
    ------
    ValueError
        Under ``on_invalid="raise"`` when a non-missing, non-numeric token is
        encountered.
    """
    detailed = clean_numeric_detailed(raw, on_invalid=on_invalid)
    return detailed.values, detailed.report


@dataclass(frozen=True, slots=True)
class SpecValidation:
    """Result of validating specification limits.

    ``ok`` is True when the limits are usable. ``errors`` are hard problems
    (limits unusable); ``warnings`` are advisory (limits usable but noteworthy).
    """

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_spec_limits(
    usl: Optional[float],
    lsl: Optional[float],
    nominal: Optional[float] = None,
) -> SpecValidation:
    """Validate raw specification inputs *before* constructing :class:`SpecLimits`.

    This mirrors the invariants enforced by :class:`SpecLimits` but returns a
    structured result instead of raising, so the UI can show friendly messages
    on the specification-input screen. Constructing ``SpecLimits`` remains the
    authoritative gate.

    Rules
    -----
    * At least one of USL/LSL must be provided (error otherwise).
    * If both given, ``USL`` must be strictly greater than ``LSL`` (error).
    * Any provided value must be finite (error).
    * A ``nominal`` outside the given bounds is an error (matches SpecLimits).
    * A ``nominal`` exactly on a bound is allowed but warned.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for name, val in (("USL", usl), ("LSL", lsl), ("nominal", nominal)):
        if val is not None and not np.isfinite(val):
            errors.append(f"{name} must be a finite number (got {val!r}).")

    if usl is None and lsl is None:
        errors.append("At least one specification limit (USL or LSL) is required.")

    if usl is not None and lsl is not None and np.isfinite(usl) and np.isfinite(lsl):
        if usl < lsl:
            errors.append(f"USL ({usl}) is below LSL ({lsl}); limits are reversed.")
        elif usl == lsl:
            errors.append(f"USL equals LSL ({usl}); tolerance width is zero.")

    if nominal is not None and np.isfinite(nominal):
        if usl is not None and np.isfinite(usl) and nominal > usl:
            errors.append(f"Nominal ({nominal}) lies above USL ({usl}).")
        if lsl is not None and np.isfinite(lsl) and nominal < lsl:
            errors.append(f"Nominal ({nominal}) lies below LSL ({lsl}).")
        if usl is not None and nominal == usl:
            warnings.append("Nominal sits exactly on the USL.")
        if lsl is not None and nominal == lsl:
            warnings.append("Nominal sits exactly on the LSL.")

    return SpecValidation(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
