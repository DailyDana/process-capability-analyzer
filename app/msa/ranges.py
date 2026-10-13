"""Range chart across the part x operator cells of a Gage R&R study.

Each cell (one operator measuring one part ``r`` times) yields a range. With
``R-bar`` the average of all cell ranges, the chart's upper limit is
``UCL = D4(r) * R-bar`` (LCL is 0 for r <= 6). A range above the UCL means an
operator was not consistent on that part, so the repeatability estimate is not
trustworthy until it is explained -- the measurement-system analogue of
"stability before capability".

``D4`` comes from :func:`app.statistics.constants.d4_factor`, computed from the
exact moments of the relative range (3.2665 for r=2, 2.5746 for r=3) rather
than the rounded table values 3.27 / 2.57; the difference only matters for a
range lying within 0.1% of the limit.
"""

from __future__ import annotations

import numpy as np

from app.msa.models import GageStudy, RangeCheck
from app.statistics.constants import d4_factor

__all__ = ["cell_ranges", "range_check"]


def cell_ranges(study: GageStudy) -> np.ndarray:
    """Ranges over trials, shape ``(parts, operators)``."""
    v = study.values
    return np.asarray(v.max(axis=2) - v.min(axis=2), dtype=float)


def range_check(study: GageStudy) -> RangeCheck:
    """Evaluate every cell range against ``D4 * R-bar``."""
    ranges = cell_ranges(study)
    r_bar = float(ranges.mean())
    ucl = float(d4_factor(study.n_trials) * r_bar)
    flagged: list[tuple[str, str, float]] = []
    if r_bar > 0.0:
        for i, part in enumerate(study.part_labels):
            for j, op in enumerate(study.operator_labels):
                rng = float(ranges[i, j])
                if rng > ucl:
                    flagged.append((part, op, rng))
    return RangeCheck(r_bar=r_bar, ucl=ucl, out_of_control=tuple(flagged))
