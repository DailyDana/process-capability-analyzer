"""Attribute control charts: p, np, c and u (``docs/FORMULAS.md`` §9).

For sample ``i`` with size ``n_i`` (items inspected, or inspection units) and
count ``x_i`` (nonconforming items, or nonconformities)::

    p   plotted x_i / n_i     CL = pbar = sum x / sum n     sigma_i = sqrt(pbar (1 - pbar) / n_i)
    np  plotted x_i           CL = n pbar                   sigma   = sqrt(n pbar (1 - pbar))
    c   plotted x_i           CL = cbar = mean x            sigma   = sqrt(cbar)
    u   plotted x_i / n_i     CL = ubar = sum x / sum n     sigma_i = sqrt(ubar / n_i)

Limits are ``CL +- 3 sigma_i``, the lower one not below 0 and (p, np) the
upper one not above 1 (or ``n``). The np and c charts need a constant sample
size; p and u accept varying sizes and then have one pair of limits per sample.

**Baseline.** By default the centre line comes from every sample. With
``n_baseline`` it comes from the first ``n_baseline`` samples only (the trial
period) and the rest are judged against it; ``exclude`` removes samples with
an assignable cause from the estimate. They are still plotted and still
flagged. ``center`` instead fixes the centre line at a known standard.

**Run rules (statistical-convention flag).** The binomial and Poisson
distributions are skewed, so the zone rules built on normal thirds of the
band are not applied. Like Minitab, only the four tests that do not depend
on zones are used: a point beyond a limit, a run on one side of the centre
line, a trend, and alternation. With varying sample sizes the rules run on
the standardised values ``z_i = (plotted_i - CL) / sigma_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from app.spc.limits import ChartType, ControlChart
from app.spc.rules import Direction, RuleEvaluation, RuleSet, Violation, evaluate_rules

__all__ = [
    "AttributeKind",
    "AttributeChart",
    "attribute_chart",
    "evaluate_attribute_rules",
    "APPLICABLE_RULES",
]


class AttributeKind(str, Enum):
    P = "p"
    NP = "np"
    C = "c"
    U = "u"


#: Rules that do not rely on normal-theory zones (see the module docstring).
APPLICABLE_RULES = frozenset({"WE1", "WE4", "N1", "N2", "N3", "N4"})

_NEEDS_CONSTANT_SIZE = frozenset({AttributeKind.NP, AttributeKind.C})
_BINOMIAL = frozenset({AttributeKind.P, AttributeKind.NP})


@dataclass(frozen=True, slots=True)
class AttributeChart:
    """One attribute chart. Arrays have one entry per sample, in the order given."""

    kind: AttributeKind
    counts: NDArray[np.float64]
    sizes: NDArray[np.float64]
    points: NDArray[np.float64]
    center_line: float
    ucl: NDArray[np.float64]
    lcl: NDArray[np.float64]
    sigma: NDArray[np.float64]
    beyond_limits: NDArray[np.bool_]
    in_baseline: NDArray[np.bool_]  # samples the centre line was estimated from
    rate: float  # pbar (p, np), cbar per sample (c) or ubar (u)
    constant_size: bool
    center_is_standard: bool

    @property
    def n_samples(self) -> int:
        return int(self.points.size)

    @property
    def n_baseline(self) -> int:
        return int(np.count_nonzero(self.in_baseline))

    @property
    def standardized(self) -> NDArray[np.float64]:
        """``(plotted - CL) / sigma_i``; 0 where sigma is 0."""
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.points - self.center_line) / self.sigma
        return np.where(self.sigma > 0, z, 0.0)


def _as_counts(values: Sequence[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"No {name} were provided.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Every {name[:-1]} must be a finite number.")
    return arr


def attribute_chart(
    kind: AttributeKind,
    counts: Sequence[float],
    sizes: Optional[Sequence[float] | float] = None,
    *,
    n_baseline: Optional[int] = None,
    exclude: Sequence[int] = (),
    center: Optional[float] = None,
) -> AttributeChart:
    """Build a p, np, c or u chart.

    Parameters
    ----------
    counts:
        Nonconforming items (p, np) or nonconformities (c, u) per sample.
    sizes:
        Sample size per sample, or one number for all. Required for p, np and
        u. For c it is optional and only checked to be constant.
    n_baseline:
        Estimate the centre line from the first ``n_baseline`` samples only.
    exclude:
        Zero-based positions left out of the estimate (still plotted).
    center:
        Known standard for the plotted rate (``p``, ``c`` or ``u``); when
        given nothing is estimated and ``n_baseline``/``exclude`` are ignored.

    Raises
    ------
    ValueError
        With a plain-language message for unusable input.
    """
    x = _as_counts(counts, "counts")
    k = int(x.size)
    if (x < 0).any():
        raise ValueError("Counts cannot be negative.")
    if not np.allclose(x, np.round(x)):
        raise ValueError("Counts must be whole numbers.")

    if sizes is None:
        if kind is not AttributeKind.C:
            raise ValueError(f"The {kind.value} chart needs the sample size of every sample.")
        n = np.ones(k, dtype=np.float64)
    else:
        n = np.asarray(sizes, dtype=np.float64).ravel()
        if n.size == 1:
            n = np.full(k, float(n[0]))
        if n.size != k:
            raise ValueError(f"{k} counts but {n.size} sample sizes.")
        if not np.isfinite(n).all() or (n <= 0).any():
            raise ValueError("Every sample size must be greater than zero.")
    constant = bool(np.all(n == n[0]))

    if kind in _BINOMIAL:
        if not np.allclose(n, np.round(n)):
            raise ValueError("Sample sizes must be whole numbers of items for p and np charts.")
        if (x > n).any():
            i = int(np.nonzero(x > n)[0][0])
            raise ValueError(
                f"Sample {i + 1} has {x[i]:g} nonconforming items in a sample of {n[i]:g}. "
                "If these are defects per unit, not defective items, use a c or u chart."
            )
    if kind in _NEEDS_CONSTANT_SIZE and not constant:
        other = "p" if kind is AttributeKind.NP else "u"
        raise ValueError(
            f"The {kind.value} chart needs the same sample size in every sample; "
            f"use the {other} chart for varying sizes."
        )

    in_base = np.ones(k, dtype=bool)
    if center is None:
        if n_baseline is not None:
            if not 1 <= n_baseline <= k:
                raise ValueError(f"The baseline must be between 1 and {k} samples; got {n_baseline}.")
            in_base[n_baseline:] = False
        for pos in exclude:
            if not 0 <= int(pos) < k:
                raise ValueError(f"Excluded sample {int(pos) + 1} does not exist (there are {k}).")
            in_base[int(pos)] = False
        if not in_base.any():
            raise ValueError("Every baseline sample is excluded; nothing is left to estimate the centre line from.")
        rate = float(x[in_base].sum() / n[in_base].sum()) if kind is not AttributeKind.C else float(x[in_base].mean())
    else:
        rate = float(center)
        upper = 1.0 if kind in _BINOMIAL else np.inf
        if not np.isfinite(rate) or rate < 0 or rate > upper:
            raise ValueError("The standard value must be a valid rate (between 0 and 1 for p and np charts).")
        in_base[:] = False

    if kind is AttributeKind.P:
        points, cl = x / n, rate
        sigma = np.sqrt(rate * (1.0 - rate) / n)
        cap: Optional[NDArray[np.float64]] = np.ones(k)
    elif kind is AttributeKind.NP:
        points, cl = x.copy(), float(n[0]) * rate
        sigma = np.full(k, sqrt(float(n[0]) * rate * (1.0 - rate)))
        cap = n.copy()
    elif kind is AttributeKind.C:
        points, cl = x.copy(), rate
        sigma = np.full(k, sqrt(rate))
        cap = None
    else:
        points, cl = x / n, rate
        sigma = np.sqrt(rate / n)
        cap = None

    ucl = cl + 3.0 * sigma
    if cap is not None:
        ucl = np.minimum(ucl, cap)
    lcl = np.maximum(cl - 3.0 * sigma, 0.0)
    # Strict inequality, as on the variables charts: a point on a limit is in control.
    beyond = (points > ucl) | (points < lcl)
    return AttributeChart(
        kind=kind, counts=x, sizes=n, points=points, center_line=float(cl), ucl=ucl, lcl=lcl, sigma=sigma,
        beyond_limits=beyond, in_baseline=in_base, rate=rate, constant_size=constant,
        center_is_standard=center is not None,
    )


def _rule_chart(chart: AttributeChart) -> ControlChart:
    """The chart in the form the run-rule engine takes.

    Constant sample size: the plotted values with their single pair of limits.
    Varying sizes: standardised values against +-3. In both cases the
    beyond-limit flags are the attribute chart's own (its limits are clipped
    at 0 and at the sample size, which the symmetric band does not know).
    """
    idx = np.arange(chart.n_samples, dtype=int)
    if chart.constant_size:
        s = float(chart.sigma[0])
        return ControlChart(ChartType.INDIVIDUALS, chart.points, chart.center_line, chart.center_line + 3.0 * s,
                            chart.center_line - 3.0 * s, s, chart.beyond_limits, True, idx)
    return ControlChart(ChartType.INDIVIDUALS, chart.standardized, 0.0, 3.0, -3.0, 1.0, chart.beyond_limits, True, idx)


def evaluate_attribute_rules(chart: AttributeChart, rule_set: RuleSet = RuleSet.NELSON) -> RuleEvaluation:
    """Run the applicable rules (:data:`APPLICABLE_RULES`) on an attribute chart."""
    raw = evaluate_rules(_rule_chart(chart), rule_set=rule_set)
    violations: list[Violation] = []
    for v in raw.violations:
        ids = [rid for rid in v.all_rule_ids if rid in APPLICABLE_RULES]
        if not ids:
            continue
        if v.rule_id in ("WE1", "N1"):
            i = v.indices[0]
            upper = chart.points[i] > chart.ucl[i]
            v = Violation(
                rule_id=v.rule_id, rule_name=v.rule_name, rule_set=v.rule_set, indices=v.indices,
                direction=Direction.UPPER if upper else Direction.LOWER,
                explanation=(
                    f"Sample {i + 1} = {chart.points[i]:.6g} lies "
                    f"{'above its UCL' if upper else 'below its LCL'} "
                    f"({chart.ucl[i]:.6g} / {chart.lcl[i]:.6g}); a point beyond 3 sigma indicates a special cause."
                ),
                equivalent_rule_ids=v.equivalent_rule_ids,
            )
        violations.append(v)
    skipped = tuple(s for s in raw.skipped if s.rule_id in APPLICABLE_RULES)
    return RuleEvaluation(violations=tuple(violations), skipped=skipped)
