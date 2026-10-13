"""Summary table, percentages and ndc for a Gage R&R result.

For each source ``x`` with standard deviation ``s_x`` and total ``s_TV``::

    %Contribution = 100 * s_x^2 / s_TV^2      (variances add to 100%)
    %Study Var    = 100 * s_x / s_TV          (does NOT add to 100%)
    %Tolerance    = 100 * k * s_x / (USL - LSL)
    Study Var     = k * s_x

``k`` is the study-variation multiplier (statistical-convention flag): 6 by
default (AIAG MSA 4th ed., Minitab), 5.15 for the 3rd-edition convention. The
two percentages that are ratios of sigmas (%Contribution, %Study Var) do not
depend on ``k``; %Tolerance does.

``ndc = trunc(1.41 * s_part / s_GRR)`` (AIAG). It is ``None`` when
``s_GRR == 0`` (a perfect gauge has unbounded discrimination).
"""

from __future__ import annotations

from math import floor
from typing import Optional

from app.msa.constants import NDC_FACTOR
from app.msa.models import ComponentEstimates, SourceRow

__all__ = ["DEFAULT_K", "summary_rows", "number_of_distinct_categories"]

DEFAULT_K: float = 6.0


def _row(
    source: str,
    sd: float,
    total_sd: float,
    k: float,
    tolerance: Optional[float],
) -> SourceRow:
    if total_sd > 0.0:
        pct_c: Optional[float] = 100.0 * sd**2 / total_sd**2
        pct_sv: Optional[float] = 100.0 * sd / total_sd
    else:
        pct_c = pct_sv = None
    pct_tol = 100.0 * k * sd / tolerance if tolerance else None
    return SourceRow(
        source=source,
        variance=sd**2,
        sd=sd,
        study_var=k * sd,
        pct_contribution=pct_c,
        pct_study_var=pct_sv,
        pct_tolerance=pct_tol,
    )


def summary_rows(
    c: ComponentEstimates,
    k: float = DEFAULT_K,
    tolerance: Optional[float] = None,
) -> tuple[SourceRow, ...]:
    """Build the standard Gage R&R table, in the conventional row order.

    ``Operator`` and ``Part x Operator`` rows appear only when the method
    estimates them (ANOVA without pooling shows both; pooled ANOVA shows
    Operator; Average & Range shows neither, since AV *is* reproducibility).
    This matches the layout of common commercial output.
    """
    if k <= 0.0:
        raise ValueError(f"Study-variation multiplier k must be > 0; got {k}.")
    if tolerance is not None and not tolerance > 0.0:
        raise ValueError(f"Tolerance (USL - LSL) must be > 0; got {tolerance}.")
    tv = c.total_sd
    rows = [
        _row("Total Gage R&R", c.grr_sd, tv, k, tolerance),
        _row("Repeatability", c.repeatability_sd, tv, k, tolerance),
        _row("Reproducibility", c.reproducibility_sd, tv, k, tolerance),
    ]
    if c.operator_sd is not None:
        rows.append(_row("Operator", c.operator_sd, tv, k, tolerance))
    if c.interaction_sd is not None:
        rows.append(_row("Part x Operator", c.interaction_sd, tv, k, tolerance))
    rows.append(_row("Part-to-Part", c.part_sd, tv, k, tolerance))
    rows.append(_row("Total Variation", tv, tv, k, tolerance))
    return tuple(rows)


def number_of_distinct_categories(c: ComponentEstimates) -> Optional[int]:
    """AIAG ndc, truncated; ``None`` when GRR is zero."""
    grr = c.grr_sd
    if grr <= 0.0:
        return None
    return int(floor(NDC_FACTOR * c.part_sd / grr))
