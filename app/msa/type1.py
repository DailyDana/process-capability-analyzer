"""Type-1 gauge study (Bosch "Procedure 1", VDA 5 "Verfahren 1"): Cg and Cgk.

One appraiser measures one reference part (known value ``x_m``) ``n`` times,
typically 50 (at least 25). With ``T`` the tolerance, ``K`` the percentage of
the tolerance allowed for the gauge (default 20) and ``L`` the number of
standard deviations it must cover (default 6)::

    x-bar, s    mean and sample standard deviation of the readings
    bias        = x-bar - x_m
    Cg          = (K/100 * T) / (L * s)
    Cgk         = (K/200 * T - |bias|) / (L/2 * s)
    %Var(Rep)          = 100 * L * s / T      (= K / Cg)
    %Var(Rep + Bias)   = K / Cgk
    t           = bias / (s / sqrt(n)),  two-sided p from t(n - 1)
    %RE         = 100 * resolution / T

With the defaults ``Cg = 0.2 T / 6s`` and ``Cgk = (0.1 T - |bias|) / 3s``
(Bosch Booklet 10; the same defaults as Minitab). Bosch's bias criterion
``|bias| / s < t(0.975, n-1) / sqrt(n)`` (0.284 for n = 50, 0.413 for n = 25)
is the same test as the two-sided t-test at alpha = 0.05.

Confidence intervals (both reproduce Bosch Booklet 10's worked example):

* Cg is proportional to 1/s, so its exact interval follows from the
  chi-square distribution of s: ``Cg * sqrt(chi2(alpha/2, n-1) / (n-1))`` to
  ``Cg * sqrt(chi2(1-alpha/2, n-1) / (n-1))``.
* Cgk uses the Bissell (1990) normal approximation already used for Cpk
  (``app.capability.indices``), generalised to the ``L/2`` denominator
  ``c``: ``Cgk +- z * sqrt(1/(c^2 n) + Cgk^2 / (2(n-1)))`` (c = 3 -> 1/(9n)).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, sqrt
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

from app.data.validation import RowStatus, clean_numeric_detailed

__all__ = [
    "Type1Request",
    "Type1Result",
    "Type1Validation",
    "validate_type1",
    "type1_study",
    "MIN_READINGS",
    "RECOMMENDED_READINGS",
]

#: Fewer readings than this are rejected; Bosch requires at least 25.
MIN_READINGS = 10
#: Bosch Booklet 10 asks for 50 (25 as a minimum).
RECOMMENDED_READINGS = 50


@dataclass(frozen=True, slots=True)
class Type1Request:
    values: tuple[float, ...]
    reference: float
    tolerance: float
    k_percent: float = 20.0
    l_sigma: float = 6.0
    resolution: Optional[float] = None
    alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class Type1Validation:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    values: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Type1Result:
    n: int
    reference: float
    tolerance: float
    k_percent: float
    l_sigma: float
    mean: float
    sd: float
    bias: float
    cg: float
    cgk: float
    pct_var_rep: float
    pct_var_rep_bias: Optional[float]  # None when Cgk <= 0 (bias uses the whole band)
    t_stat: Optional[float]
    p_value: Optional[float]
    bias_significant: Optional[bool]
    bias_criterion: float  # t(1-alpha/2, n-1)/sqrt(n), Bosch's |bias|/s limit
    cg_ci: Optional[tuple[float, float]]
    cgk_ci: Optional[tuple[float, float]]
    alpha: float
    resolution: Optional[float]
    pct_resolution: Optional[float]
    values: tuple[float, ...]

    @property
    def band(self) -> tuple[float, float]:
        """Reference +- K/200 * T: the band the run chart shows (+-0.1 T by default)."""
        half = self.k_percent / 200.0 * self.tolerance
        return self.reference - half, self.reference + half

    def min_tolerances(
        self, index_min: float = 1.33, resolution_max_pct: float = 5.0
    ) -> dict[str, Optional[float]]:
        """Smallest tolerance for which this gauge would still pass (Bosch's T_min).

        Solving each criterion for T, with the observed s and bias::

            T_min(Cg)  = index_min * L * s / (K/100)
            T_min(Cgk) = (index_min * L/2 * s + |bias|) / (K/200)
            T_min(RE)  = resolution / (resolution_max_pct / 100)

        These reproduce Bosch Booklet 10's worked example (0.039701, 0.050696,
        0.020000 with index_min = 1.33). ``None`` when undefined.
        """
        k = self.k_percent / 100.0
        return {
            "Cg": index_min * self.l_sigma * self.sd / k,
            "Cgk": (index_min * self.l_sigma / 2.0 * self.sd + abs(self.bias)) / (k / 2.0),
            "RE": self.resolution / (resolution_max_pct / 100.0) if self.resolution else None,
        }


def validate_type1(
    raw_values: Sequence[object],
    reference: object,
    tolerance: object,
    resolution: object = None,
) -> Type1Validation:
    """Check the readings and settings; never drops a reading silently."""
    errors: list[str] = []
    warnings: list[str] = []
    cleaned = clean_numeric_detailed(list(raw_values))
    bad = [f"row {r.original_index + 1} ({r.raw!r})" for r in cleaned.records
           if r.status is not RowStatus.OK and r.status is not RowStatus.MISSING]
    if bad:
        shown = ", ".join(bad[:8]) + (f" (and {len(bad) - 8} more)" if len(bad) > 8 else "")
        errors.append(f"Non-numeric or non-finite readings: {shown}.")
    values = tuple(float(v) for v in cleaned.values)
    n_missing = sum(1 for r in cleaned.records if r.status is RowStatus.MISSING)
    if n_missing:
        warnings.append(f"{n_missing} blank row(s) ignored.")
    for name, v in (("Reference value", reference), ("Tolerance", tolerance)):
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number.")
            continue
        if not np.isfinite(fv):
            errors.append(f"{name} must be finite.")
        elif name == "Tolerance" and fv <= 0:
            errors.append("Tolerance (USL - LSL) must be greater than zero.")
    if resolution is not None:
        try:
            fr = float(resolution)  # type: ignore[arg-type]
            if not fr > 0:
                errors.append("Resolution must be greater than zero.")
        except (TypeError, ValueError):
            errors.append("Resolution must be a number.")
    if len(values) < MIN_READINGS and not bad:
        errors.append(f"At least {MIN_READINGS} readings are required (found {len(values)}).")
    elif len(values) < 25:
        warnings.append(f"Only {len(values)} readings: Bosch Booklet 10 requires at least 25 (50 recommended).")
    elif len(values) < RECOMMENDED_READINGS:
        warnings.append(f"{len(values)} readings: Bosch Booklet 10 recommends 50.")
    if values and float(np.ptp(values)) == 0.0:
        warnings.append("All readings are identical: the resolution is too coarse to show the gauge's spread.")
    return Type1Validation(not errors, tuple(errors), tuple(warnings), values)


def type1_study(req: Type1Request) -> Type1Result:
    """Compute the Type-1 study. Inputs are assumed validated (see validate_type1)."""
    if req.tolerance <= 0 or req.k_percent <= 0 or req.l_sigma <= 0:
        raise ValueError("Tolerance, K and L must be positive.")
    if not 0.0 < req.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    x = np.asarray(req.values, dtype=float)
    n = int(x.size)
    if n < 2:
        raise ValueError("At least 2 readings are required.")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    # Same rounding-floor idea as app.msa.models: a spread within float noise is zero.
    if sd <= 64.0 * np.finfo(float).eps * float(np.max(np.abs(x))):
        sd = 0.0
    bias = mean - req.reference
    allowed = req.k_percent / 100.0 * req.tolerance
    c = req.l_sigma / 2.0
    crit = float(t_dist.ppf(1.0 - req.alpha / 2.0, n - 1)) / sqrt(n)

    if sd > 0.0:
        cg = allowed / (req.l_sigma * sd)
        cgk = (allowed / 2.0 - abs(bias)) / (c * sd)
        t_stat: Optional[float] = bias / (sd / sqrt(n))
        p_value: Optional[float] = float(2.0 * t_dist.sf(abs(t_stat), n - 1))
        significant: Optional[bool] = abs(bias) / sd >= crit
        lo = sqrt(float(chi2.ppf(req.alpha / 2.0, n - 1)) / (n - 1))
        hi = sqrt(float(chi2.ppf(1.0 - req.alpha / 2.0, n - 1)) / (n - 1))
        cg_ci: Optional[tuple[float, float]] = (cg * lo, cg * hi)
        z = float(norm.ppf(1.0 - req.alpha / 2.0))
        se = sqrt(1.0 / (c * c * n) + cgk * cgk / (2.0 * (n - 1)))
        cgk_ci: Optional[tuple[float, float]] = (cgk - z * se, cgk + z * se)
    else:
        cg = inf
        cgk = inf if abs(bias) < allowed / 2.0 else -inf
        t_stat = p_value = significant = None
        cg_ci = cgk_ci = None

    pct_rep = 100.0 * req.l_sigma * sd / req.tolerance
    pct_rep_bias = req.k_percent / cgk if isfinite(cgk) and cgk > 0 else (0.0 if cgk == inf else None)
    pct_res = 100.0 * req.resolution / req.tolerance if req.resolution else None
    return Type1Result(
        n=n, reference=req.reference, tolerance=req.tolerance, k_percent=req.k_percent,
        l_sigma=req.l_sigma, mean=mean, sd=sd, bias=bias, cg=cg, cgk=cgk,
        pct_var_rep=pct_rep, pct_var_rep_bias=pct_rep_bias, t_stat=t_stat, p_value=p_value,
        bias_significant=significant, bias_criterion=crit, cg_ci=cg_ci, cgk_ci=cgk_ci,
        alpha=req.alpha, resolution=req.resolution, pct_resolution=pct_res, values=tuple(float(v) for v in x),
    )
