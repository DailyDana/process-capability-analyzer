"""Process performance (Pp/Ppk) for non-normal data.

Two methods (``docs/FORMULAS.md`` §8):

**Percentile method** (ISO 22514-2 / Clements 1989). Fit a distribution, then
replace the normal-theory ``mean +- 3 sigma`` by its 0.135th, 50th and
99.865th percentiles::

    Pp  = (USL - LSL) / (X_99.865 - X_0.135)
    Ppu = (USL - X_50) / (X_99.865 - X_50)
    Ppl = (X_50 - LSL) / (X_50 - X_0.135)
    Ppk = min(Ppu, Ppl)

For a normal fit this is exactly the classic Pp/Ppk. Expected PPM come from
the fitted CDF. Distributions: normal, lognormal, Weibull, gamma (2-parameter,
lower bound 0) and exponential (1-parameter); exact maximum-likelihood fits
(closed form, or the score equations solved to machine precision), except that the normal fit uses the mean and the overall sigma
convention below so that it reproduces the normal-theory indices.

**Box-Cox transformation.** ``y = (x^lambda - 1) / lambda`` (``ln x`` at
lambda = 0), lambda by maximum likelihood, optionally rounded to the nearest
0.5 as common software does. The specification limits are transformed the
same way and normal-theory Pp/Ppk computed on the transformed scale. (The
scaled form keeps the order of values for negative lambda, so USL and LSL
never swap; it differs from ``x^lambda`` only by a positive affine map, which
leaves Pp/Ppk unchanged.)

**Overall sigma (convention flag).** The rest of this application defines
Pp/Ppk on ``s`` (AIAG, §2.3). Minitab by default divides by the unbiasing
constant ``c4(n)``. ``unbias_c4=True`` applies ``s / c4`` to the normal fit
and to Box-Cox; it has no effect on the MLE-fitted distributions.

**Lower boundary.** Many non-normal characteristics (flatness, runout,
roughness, cycle time) have a natural lower bound of 0 rather than a lower
specification. With ``lsl_is_boundary=True`` the LSL is not used for Ppl or
for PPM below; the analysis becomes one-sided (upper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf, sqrt
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special, stats

from app.statistics.constants import c4

__all__ = [
    "Distribution",
    "DistributionFit",
    "PercentileIndices",
    "BoxCoxResult",
    "NonNormalSpec",
    "fit_distribution",
    "fit_all",
    "percentile_indices",
    "anderson_darling",
    "box_cox",
    "boxcox_transform",
    "P_LOW",
    "P_MID",
    "P_HIGH",
]

#: Phi(-3), 0.5 and Phi(+3): the "0.135 % / 99.865 %" percentiles, kept exact so
#: that a normal fit reduces to the classic 6-sigma indices without rounding.
P_LOW, P_MID, P_HIGH = float(stats.norm.cdf(-3.0)), 0.5, float(stats.norm.cdf(3.0))


class Distribution(str, Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"


#: Distributions that require strictly positive data (lower bound 0).
POSITIVE_ONLY = frozenset({Distribution.LOGNORMAL, Distribution.WEIBULL, Distribution.GAMMA, Distribution.EXPONENTIAL})


@dataclass(frozen=True, slots=True)
class NonNormalSpec:
    usl: Optional[float] = None
    lsl: Optional[float] = None
    lsl_is_boundary: bool = False

    def __post_init__(self) -> None:
        if self.usl is None and self.lsl is None:
            raise ValueError("At least one specification limit is required.")
        if self.usl is not None and self.lsl is not None and not self.usl > self.lsl:
            raise ValueError("USL must be greater than LSL.")
        if self.lsl_is_boundary and self.usl is None:
            raise ValueError("A lower boundary needs a USL: there is nothing else to assess.")

    @property
    def effective_lsl(self) -> Optional[float]:
        return None if self.lsl_is_boundary else self.lsl


@dataclass(frozen=True, slots=True)
class DistributionFit:
    distribution: Distribution
    params: dict[str, float]
    frozen: object = field(repr=False, compare=False)  # scipy frozen distribution
    ad_statistic: float = float("nan")
    log_likelihood: float = float("nan")

    def ppf(self, q: float) -> float:
        return float(self.frozen.ppf(q))  # type: ignore[attr-defined]

    def cdf(self, x: float) -> float:
        return float(self.frozen.cdf(x))  # type: ignore[attr-defined]

    def sf(self, x: float) -> float:
        return float(self.frozen.sf(x))  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class PercentileIndices:
    x_low: float
    x_mid: float
    x_high: float
    pp: Optional[float]
    ppu: Optional[float]
    ppl: Optional[float]
    ppk: float
    ppm_below: float
    ppm_above: float

    @property
    def ppm_total(self) -> float:
        return self.ppm_below + self.ppm_above


def _overall_sd(x: NDArray[np.float64], unbias_c4: bool) -> float:
    s = float(np.std(x, ddof=1))
    return s / c4(int(x.size)) if unbias_c4 else s


def anderson_darling(x: NDArray[np.float64], cdf) -> float:
    """Anderson-Darling A^2 of ``x`` against a fully specified CDF.

    ``A^2 = -n - (1/n) * sum_i (2i - 1) [ln F(x_(i)) + ln(1 - F(x_(n+1-i)))]``.
    Used to rank fits; with estimated parameters the null distribution depends
    on the family, so A^2 values are compared, not turned into p-values.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    f = np.clip(np.asarray(cdf(xs), dtype=float), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(f) + np.log1p(-f[::-1]))) / n)


def _weibull_mle(x: NDArray[np.float64]) -> tuple[float, float]:
    """Two-parameter Weibull MLE from its score equation, solved to machine precision.

    The shape k solves ``sum(x^k ln x) / sum(x^k) - 1/k - mean(ln x) = 0``
    (monotone in k), then ``scale = mean(x^k)^(1/k)``. A general-purpose
    optimiser stops short of this: on the cycle-time reference data it is
    0.65 PPM away from the published (and exact) 233244.81.
    """
    lx = np.log(x)
    lx_mean = float(lx.mean())

    def score(k: float) -> float:
        z = k * (lx - lx.max())  # scaled for overflow safety; the ratio is unchanged
        w = np.exp(z)
        return float(np.sum(w * lx) / np.sum(w)) - 1.0 / k - lx_mean

    lo, hi = 1e-3, 1.0
    while score(hi) < 0 and hi < 1e4:
        hi *= 2.0
    k = float(optimize.brentq(score, lo, hi, xtol=1e-14, rtol=1e-15))
    scale = float(np.mean(np.exp(k * lx)) ** (1.0 / k))
    return k, scale


def _gamma_mle(x: NDArray[np.float64]) -> tuple[float, float]:
    """Two-parameter gamma MLE: shape solves ``ln k - digamma(k) = ln(mean) - mean(ln x)``."""
    target = float(np.log(x.mean()) - np.log(x).mean())
    if target <= 0:
        raise ValueError("The gamma fit is degenerate (no variation).")

    def f(k: float) -> float:
        return float(np.log(k) - special.digamma(k)) - target

    lo, hi = 1e-6, 1.0
    while f(hi) > 0 and hi < 1e8:
        hi *= 2.0
    k = float(optimize.brentq(f, lo, hi, xtol=1e-14, rtol=1e-15))
    return k, float(x.mean()) / k


def fit_distribution(x: NDArray[np.float64], dist: Distribution, unbias_c4: bool = False) -> DistributionFit:
    """Fit one distribution. Raises ValueError when the data are outside its support."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        raise ValueError("At least 3 observations are required.")
    if dist in POSITIVE_ONLY and not np.all(x > 0):
        raise ValueError(f"The {dist.value} distribution needs strictly positive data.")
    if dist is Distribution.NORMAL:
        mu, sd = float(x.mean()), _overall_sd(x, unbias_c4)
        if sd <= 0:
            raise ValueError("No variation in the data.")
        frozen = stats.norm(mu, sd)
        params = {"mean": mu, "sd": sd}
    elif dist is Distribution.LOGNORMAL:
        lx = np.log(x)
        mu, sigma = float(lx.mean()), float(lx.std(ddof=0))  # closed-form MLE
        frozen = stats.lognorm(sigma, 0, np.exp(mu))
        params = {"mu (log)": mu, "sigma (log)": sigma}
    elif dist is Distribution.WEIBULL:
        shape, scale = _weibull_mle(x)
        frozen = stats.weibull_min(shape, 0, scale)
        params = {"shape": shape, "scale": scale}
    elif dist is Distribution.GAMMA:
        shape, scale = _gamma_mle(x)
        frozen = stats.gamma(shape, 0, scale)
        params = {"shape": shape, "scale": scale}
    elif dist is Distribution.EXPONENTIAL:
        scale = float(x.mean())  # closed-form MLE
        frozen = stats.expon(0, scale)
        params = {"scale": scale}
    else:  # pragma: no cover
        raise ValueError(dist)
    ll = float(np.sum(frozen.logpdf(x)))
    return DistributionFit(dist, params, frozen, anderson_darling(x, frozen.cdf), ll)


def fit_all(x: NDArray[np.float64], unbias_c4: bool = False) -> tuple[list[DistributionFit], dict[Distribution, str]]:
    """Fit every distribution the data allow; ranked by A^2 (best first)."""
    fits: list[DistributionFit] = []
    skipped: dict[Distribution, str] = {}
    for d in Distribution:
        try:
            fits.append(fit_distribution(x, d, unbias_c4))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            skipped[d] = str(exc)
    fits.sort(key=lambda f: f.ad_statistic)
    return fits, skipped


def percentile_indices(fit: DistributionFit, spec: NonNormalSpec) -> PercentileIndices:
    lo, mid, hi = fit.ppf(P_LOW), fit.ppf(P_MID), fit.ppf(P_HIGH)
    usl, lsl = spec.usl, spec.effective_lsl
    ppu = (usl - mid) / (hi - mid) if usl is not None else None
    ppl = (mid - lsl) / (mid - lo) if lsl is not None else None
    pp = (usl - lsl) / (hi - lo) if usl is not None and lsl is not None else None
    ppk = min(v for v in (ppu, ppl) if v is not None)
    below = fit.cdf(lsl) * 1e6 if lsl is not None else 0.0
    above = fit.sf(usl) * 1e6 if usl is not None else 0.0
    return PercentileIndices(lo, mid, hi, pp, ppu, ppl, ppk, below, above)


# ---------------------------------------------------------------------------
# Box-Cox
# ---------------------------------------------------------------------------
def boxcox_transform(v, lam: float):
    v = np.asarray(v, dtype=float)
    return np.log(v) if lam == 0 else (np.power(v, lam) - 1.0) / lam


@dataclass(frozen=True, slots=True)
class BoxCoxResult:
    lambda_mle: float
    lambda_used: float
    mean_t: float
    sd_t: float
    usl_t: Optional[float]
    lsl_t: Optional[float]
    pp: Optional[float]
    ppu: Optional[float]
    ppl: Optional[float]
    ppk: float
    ppm_below: float
    ppm_above: float
    ad_statistic: float

    @property
    def ppm_total(self) -> float:
        return self.ppm_below + self.ppm_above


def box_cox(x: NDArray[np.float64], spec: NonNormalSpec, *, lam: Optional[float] = None,
            round_to_half: bool = True, unbias_c4: bool = False) -> BoxCoxResult:
    """Box-Cox capability. ``lam`` fixes lambda; otherwise MLE (optionally rounded to 0.5)."""
    x = np.asarray(x, dtype=float)
    if not np.all(x > 0):
        raise ValueError("Box-Cox needs strictly positive data.")
    lam_mle = float(stats.boxcox_normmax(x, method="mle"))
    if lam is None:
        lam = round(lam_mle * 2.0) / 2.0 if round_to_half else lam_mle
    y = boxcox_transform(x, lam)
    m, s = float(y.mean()), _overall_sd(y, unbias_c4)
    if s <= 0:
        raise ValueError("No variation in the data.")
    usl, lsl = spec.usl, spec.effective_lsl
    if lsl is not None and lsl <= 0:
        raise ValueError("Box-Cox cannot transform an LSL <= 0; treat it as a lower boundary instead.")
    usl_t = float(boxcox_transform(usl, lam)) if usl is not None else None
    lsl_t = float(boxcox_transform(lsl, lam)) if lsl is not None else None
    ppu = (usl_t - m) / (3 * s) if usl_t is not None else None
    ppl = (m - lsl_t) / (3 * s) if lsl_t is not None else None
    pp = (usl_t - lsl_t) / (6 * s) if usl_t is not None and lsl_t is not None else None
    ppk = min(v for v in (ppu, ppl) if v is not None)
    below = float(stats.norm.cdf(lsl_t, m, s)) * 1e6 if lsl_t is not None else 0.0
    above = float(stats.norm.sf(usl_t, m, s)) * 1e6 if usl_t is not None else 0.0
    ad = anderson_darling(y, stats.norm(m, s).cdf)
    return BoxCoxResult(lam_mle, float(lam), m, s, usl_t, lsl_t, pp, ppu, ppl, ppk, below, above, ad)


def normal_theory_ppk(x: NDArray[np.float64], spec: NonNormalSpec, unbias_c4: bool = False) -> float:
    """Classic Ppk assuming normality -- reported beside the non-normal result."""
    fit = fit_distribution(np.asarray(x, dtype=float), Distribution.NORMAL, unbias_c4)
    return percentile_indices(fit, spec).ppk


_ = (inf, sqrt)  # keep the math imports explicit for readers of the formulas
