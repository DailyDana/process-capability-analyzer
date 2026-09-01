"""Data loading service: files and manual entry -> cleaned measurement series.

Wraps the M1 cleaning/validation layer with file parsing so the UI never touches
low-level statistics. Returns structured results (never raises for ordinary user
error) so the presentation layer can render friendly messages.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from app.data.validation import (
    DetailedCleaning,
    clean_numeric_detailed,
)

__all__ = [
    "LoadResult",
    "load_table",
    "clean_column",
    "build_subgroups",
    "SubgroupResult",
]


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Outcome of loading a tabular file.

    ``ok`` is False with a user-facing ``error`` when the file could not be
    parsed; no exception escapes to the UI.
    """

    ok: bool
    frame: Optional[pd.DataFrame] = None
    error: str = ""
    source_name: str = ""

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns) if self.frame is not None else []

    @property
    def n_rows(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))


def load_table(
    data: bytes | io.BytesIO,
    filename: str,
) -> LoadResult:
    """Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    data:
        Raw file bytes (as delivered by a Streamlit uploader).
    filename:
        Used to pick the parser by extension (.csv/.txt vs .xlsx/.xls).

    Returns
    -------
    LoadResult
        ``ok=False`` with a plain-language ``error`` on any parse failure.
    """
    name = (filename or "").lower()
    raw = data.getvalue() if isinstance(data, io.BytesIO) else data
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            frame = pd.read_excel(io.BytesIO(raw))
        elif name.endswith((".csv", ".txt", ".tsv")):
            # NOTE: do NOT use sep=None (dialect sniffing) — on a single-column
            # file the sniffer can pick a letter from the header as the
            # delimiter and shred the column names. Use explicit separators.
            if name.endswith(".tsv"):
                frame = pd.read_csv(io.BytesIO(raw), sep="\t")
            else:
                frame = pd.read_csv(io.BytesIO(raw))
        else:
            return LoadResult(
                ok=False,
                error=(
                    f"Unsupported file type '{filename}'. Please upload a CSV "
                    "(.csv/.txt/.tsv) or Excel (.xlsx/.xls) file."
                ),
                source_name=filename,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        return LoadResult(
            ok=False,
            error=f"Could not read '{filename}': {exc}",
            source_name=filename,
        )

    if frame is None or frame.shape[1] == 0:
        return LoadResult(
            ok=False,
            error=f"'{filename}' contains no columns.",
            source_name=filename,
        )
    if len(frame) == 0:
        return LoadResult(
            ok=False,
            error=f"'{filename}' contains no data rows.",
            source_name=filename,
        )
    return LoadResult(ok=True, frame=frame, source_name=filename)


def clean_column(values: Sequence[Any]) -> DetailedCleaning:
    """Clean a raw measurement column via the M1 layer, keeping per-row records."""
    return clean_numeric_detailed(list(values), on_invalid="drop")


@dataclass(frozen=True, slots=True)
class SubgroupResult:
    """Outcome of splitting a clean series into equal-size subgroups."""

    ok: bool
    subgroups: tuple[np.ndarray, ...] = field(default_factory=tuple)
    error: str = ""
    warning: str = ""
    n_used: int = 0
    n_trailing_dropped: int = 0

    @property
    def n_subgroups(self) -> int:
        return len(self.subgroups)


def build_subgroups(
    values: np.ndarray,
    subgroup_size: int,
    *,
    allow_partial_tail: bool = True,
) -> SubgroupResult:
    """Split a clean series into consecutive equal-size subgroups.

    The M3 chart calculators require balanced subgroups. When the series length
    is not an exact multiple of ``subgroup_size``, the trailing partial subgroup
    is excluded and reported (``allow_partial_tail=True``), or the split is
    refused with an explanatory error.

    Returns a structured result — never raises for ordinary user error.
    """
    n = int(np.asarray(values).size)
    if subgroup_size < 2:
        return SubgroupResult(
            ok=False,
            error=(
                f"Subgroup size must be at least 2 for X-bar charts (got "
                f"{subgroup_size}). Use the I-MR chart for individual "
                "measurements."
            ),
        )
    if n < subgroup_size * 2:
        return SubgroupResult(
            ok=False,
            error=(
                f"Need at least {subgroup_size * 2} measurements to form 2 "
                f"subgroups of size {subgroup_size}; only {n} are available."
            ),
        )

    k = n // subgroup_size
    used = k * subgroup_size
    remainder = n - used
    if remainder and not allow_partial_tail:
        return SubgroupResult(
            ok=False,
            error=(
                f"{n} measurements do not divide evenly into subgroups of "
                f"{subgroup_size} ({remainder} left over). Adjust the subgroup "
                "size or the data range."
            ),
        )

    arr = np.asarray(values, dtype=np.float64)[:used]
    groups = tuple(arr[i * subgroup_size:(i + 1) * subgroup_size] for i in range(k))
    warning = ""
    if remainder:
        warning = (
            f"{remainder} trailing measurement(s) were excluded because they do "
            f"not fill a complete subgroup of {subgroup_size}."
        )
    return SubgroupResult(
        ok=True,
        subgroups=groups,
        warning=warning,
        n_used=used,
        n_trailing_dropped=remainder,
    )
