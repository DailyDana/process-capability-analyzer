"""Reader for the NIST Statistical Reference Datasets (univariate) in ``nist_strd/``.

Each ``.dat`` file is the unmodified file from
https://www.itl.nist.gov/div898/strd/univ/homepage.html and carries its own
certified values in the header, so data and expectations cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

DIR = Path(__file__).parent / "nist_strd"
NAMES = ("PiDigits", "Lottery", "Lew", "Mavro", "Michelso", "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4")


@dataclass(frozen=True)
class NistDataset:
    name: str
    values: tuple[float, ...]
    mean: float
    sd: float
    r1: float


def _certified(text: str, label: str) -> float:
    m = re.search(rf"{label}:\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)", text)
    if m is None:
        raise ValueError(f"certified value {label!r} not found")
    return float(Decimal(m.group(1)))


def load(name: str) -> NistDataset:
    text = (DIR / f"{name}.dat").read_text(encoding="ascii")
    lines = text.splitlines()
    m = re.search(r"Data\s*:\s*lines\s+(\d+)\s+to\s+(\d+)", text)
    if m is None:
        raise ValueError("data line range not found")
    first, last = int(m.group(1)), int(m.group(2))
    values = tuple(float(s) for s in (ln.strip() for ln in lines[first - 1:last]) if s)
    n = int(re.search(r"Number of Observations:\s*(\d+)", text).group(1))  # type: ignore[union-attr]
    if len(values) != n:
        raise ValueError(f"{name}: read {len(values)} values, header says {n}")
    return NistDataset(name, values, _certified(text, "ybar"), _certified(text, "s"), _certified(text, r"r\(1\)"))
