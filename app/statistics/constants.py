"""Control-chart constants for Shewhart charts.

This module provides the standard control-chart constants used to convert
subgroup ranges / standard deviations into unbiased estimates of the process
standard deviation and into 3-sigma control limits.

Two access paths are provided:

1. A tabulated dictionary :data:`CONSTANTS` of the *canonical published values*
   (rounded as they appear in ASTM / AIAG / Montgomery tables). These are what
   engineers cross-check by eye and what most textbooks print.

2. Exact computation functions (:func:`c4`, :func:`d2`, :func:`d3`,
   :func:`a2`, ...) that derive the constants from their mathematical
   definitions. ``c4`` has a closed form; ``d2`` and ``d3`` are defined via
   integrals of the standard-normal range distribution and are evaluated
   numerically. The unit tests assert that the exact functions reproduce the
   tabulated values, which validates both.

References
----------
- Montgomery, D.C. (2013). *Introduction to Statistical Quality Control*,
  7th ed., Wiley. Appendix Table VI (control-chart constants).
- AIAG (2005). *Statistical Process Control (SPC) Reference Manual*, 2nd ed.
- NIST/SEMATECH e-Handbook of Statistical Methods, Section 6.3.2
  (control-chart constants).
- ASTM Manual on Presentation of Data and Control Chart Analysis (STP-15D).

Definitions of the derived constants
------------------------------------
Let ``n`` be the subgroup size. With W = R / sigma (relative range of a normal
sample) the classic results are::

    d2 = E[W]                       # mean of the relative range
    d3 = sqrt(Var[W])               # std dev of the relative range
    c4 = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)   # E[s]/sigma bias factor

and the chart factors are derived from them::

    A2 = 3 / (d2 * sqrt(n))         # X-bar limits from R-bar
    A3 = 3 / (c4 * sqrt(n))         # X-bar limits from S-bar
    D3 = max(0, 1 - 3*d3/d2)        # R chart lower factor
    D4 = 1 + 3*d3/d2                # R chart upper factor
    B3 = max(0, 1 - 3/c4*sqrt(1-c4**2))   # S chart lower factor
    B4 = 1 + 3/c4*sqrt(1-c4**2)           # S chart upper factor
    E2 = 3 / d2                     # I-chart limits from moving range (n=2 -> 2.66)

All formulas above are the standard published definitions; none are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import lgamma, log, sqrt
from typing import Final

import numpy as np
from scipy.stats import norm

__all__ = [
    "ChartConstants",
    "CONSTANTS",
    "c4",
    "d2",
    "d3",
    "a2",
    "a3",
    "d3_factor",
    "d4_factor",
    "b3",
    "b4",
    "e2",
    "get_constants",
    "MIN_SUBGROUP_SIZE",
    "MAX_TABULATED_N",
]

# The moving-range and range machinery is undefined for n < 2.
MIN_SUBGROUP_SIZE: Final[int] = 2
# We tabulate canonical values up to this subgroup size.
MAX_TABULATED_N: Final[int] = 25


@dataclass(frozen=True, slots=True)
class ChartConstants:
    """Bundle of control-chart constants for one subgroup size.

    Attributes
    ----------
    n:
        Subgroup size.
    d2, d3:
        Mean and standard deviation of the relative range W = R / sigma.
    c4:
        Unbiasing constant for the sample standard deviation, E[s] = c4 * sigma.
    a2, a3:
        X-bar chart factors for use with R-bar and S-bar respectively.
    d3_factor, d4_factor:
        R chart lower/upper control-limit factors (named ``D3``/``D4`` in the
        literature; suffixed ``_factor`` here to avoid clashing with the
        dispersion constant ``d3``).
    b3, b4:
        S chart lower/upper control-limit factors.
    e2:
        Individuals-chart factor for use with the average moving range.
    """

    n: int
    d2: float
    d3: float
    c4: float
    a2: float
    a3: float
    d3_factor: float
    d4_factor: float
    b3: float
    b4: float
    e2: float


# ---------------------------------------------------------------------------
# Canonical published values (Montgomery Table VI / ASTM STP-15D).
# Stored to the precision commonly printed in the standards. The exact
# computation functions below reproduce these to within rounding, which the
# unit tests assert.
# ---------------------------------------------------------------------------
# Fields: n -> (d2, d3, A2, A3, D3, D4, B3, B4, c4, E2)
_TABLE: Final[dict[int, tuple[float, ...]]] = {
    2: (1.128, 0.853, 1.880, 2.659, 0.000, 3.267, 0.000, 3.267, 0.7979, 2.660),
    3: (1.693, 0.888, 1.023, 1.954, 0.000, 2.574, 0.000, 2.568, 0.8862, 1.772),
    4: (2.059, 0.880, 0.729, 1.628, 0.000, 2.282, 0.000, 2.266, 0.9213, 1.457),
    5: (2.326, 0.864, 0.577, 1.427, 0.000, 2.114, 0.000, 2.089, 0.9400, 1.290),
    6: (2.534, 0.848, 0.483, 1.287, 0.000, 2.004, 0.030, 1.970, 0.9515, 1.184),
    7: (2.704, 0.833, 0.419, 1.182, 0.076, 1.924, 0.118, 1.882, 0.9594, 1.109),
    8: (2.847, 0.820, 0.373, 1.099, 0.136, 1.864, 0.185, 1.815, 0.9650, 1.054),
    9: (2.970, 0.808, 0.337, 1.032, 0.184, 1.816, 0.239, 1.761, 0.9693, 1.010),
    10: (3.078, 0.797, 0.308, 0.975, 0.223, 1.777, 0.284, 1.716, 0.9727, 0.975),
    11: (3.173, 0.787, 0.285, 0.927, 0.256, 1.744, 0.321, 1.679, 0.9754, 0.945),
    12: (3.258, 0.778, 0.266, 0.886, 0.283, 1.717, 0.354, 1.646, 0.9776, 0.921),
    13: (3.336, 0.770, 0.249, 0.850, 0.307, 1.693, 0.382, 1.618, 0.9794, 0.899),
    14: (3.407, 0.763, 0.235, 0.817, 0.328, 1.672, 0.406, 1.594, 0.9810, 0.881),
    15: (3.472, 0.756, 0.223, 0.789, 0.347, 1.653, 0.428, 1.572, 0.9823, 0.864),
    16: (3.532, 0.750, 0.212, 0.763, 0.363, 1.637, 0.448, 1.552, 0.9835, 0.849),
    17: (3.588, 0.744, 0.203, 0.739, 0.378, 1.622, 0.466, 1.534, 0.9845, 0.836),
    18: (3.640, 0.739, 0.194, 0.718, 0.391, 1.608, 0.482, 1.518, 0.9854, 0.824),
    19: (3.689, 0.734, 0.187, 0.698, 0.403, 1.597, 0.497, 1.503, 0.9862, 0.813),
    20: (3.735, 0.729, 0.180, 0.680, 0.415, 1.585, 0.510, 1.490, 0.9869, 0.803),
    21: (3.778, 0.724, 0.173, 0.663, 0.425, 1.575, 0.523, 1.477, 0.9876, 0.794),
    22: (3.819, 0.720, 0.167, 0.647, 0.434, 1.566, 0.534, 1.466, 0.9882, 0.785),
    23: (3.858, 0.716, 0.162, 0.633, 0.443, 1.557, 0.545, 1.455, 0.9887, 0.778),
    24: (3.895, 0.712, 0.157, 0.619, 0.451, 1.548, 0.555, 1.445, 0.9892, 0.770),
    25: (3.931, 0.708, 0.153, 0.606, 0.459, 1.541, 0.565, 1.435, 0.9896, 0.763),
}


CONSTANTS: Final[dict[int, ChartConstants]] = {
    n: ChartConstants(
        n=n,
        d2=row[0],
        d3=row[1],
        a2=row[2],
        a3=row[3],
        d3_factor=row[4],
        d4_factor=row[5],
        b3=row[6],
        b4=row[7],
        c4=row[8],
        e2=row[9],
    )
    for n, row in _TABLE.items()
}


def get_constants(n: int) -> ChartConstants:
    """Return the canonical :class:`ChartConstants` for subgroup size ``n``.

    Parameters
    ----------
    n:
        Subgroup size (must be in ``[MIN_SUBGROUP_SIZE, MAX_TABULATED_N]``).

    Raises
    ------
    ValueError
        If ``n`` is outside the tabulated range.
    """
    if n < MIN_SUBGROUP_SIZE:
        raise ValueError(
            f"Subgroup size n={n} is below the minimum {MIN_SUBGROUP_SIZE}; "
            "range/moving-range constants are undefined for n < 2."
        )
    if n > MAX_TABULATED_N:
        raise ValueError(
            f"Subgroup size n={n} exceeds the tabulated maximum "
            f"{MAX_TABULATED_N}. For large subgroups prefer the S-bar/c4 "
            "estimator and compute constants via the exact functions."
        )
    return CONSTANTS[n]


# ---------------------------------------------------------------------------
# Exact computation of the constants from first principles.
# These validate the table and allow extension beyond n=25.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def c4(n: int) -> float:
    """Exact ``c4`` bias constant: ``E[s] = c4 * sigma`` for a normal sample.

    Formula
    -------
    ``c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)``.

    Computed in log-gamma space for numerical stability.

    Reference: Montgomery (2013), eq. for c4; NIST e-Handbook 6.3.2.
    """
    if n < 2:
        raise ValueError("c4 is undefined for n < 2.")
    log_c4 = 0.5 * log(2.0 / (n - 1)) + lgamma(n / 2.0) - lgamma((n - 1) / 2.0)
    return float(np.exp(log_c4))


@lru_cache(maxsize=None)
def _range_moments(n: int) -> tuple[float, float]:
    """Return ``(E[W], E[W^2])`` for the relative range ``W = R/sigma``.

    ``W`` is the range of ``n`` i.i.d. standard-normal variates. Two classic
    results are used, both evaluated on a fixed high-resolution grid (fast,
    deterministic, and comfortably within the 1e-3 table tolerance):

    * First moment (single integral, no nesting)::

          E[W] = integral_{-inf}^{inf} (1 - Phi(x)^n - (1 - Phi(x))^n) dx

      This is the standard expected-range identity (David & Nagaraja, *Order
      Statistics*): ``E[R] = E[X_(n)] - E[X_(1)]`` expanded via the extreme
      order-statistic survival/CDF forms.

    * Second moment via the survival form of the range CDF::

          F_W(w) = n * integral phi(x) [Phi(x+w) - Phi(x)]^(n-1) dx
          E[W^2] = integral_0^inf 2 w (1 - F_W(w)) dw

    The x-integration for ``F_W`` is vectorised over the whole ``w`` grid at
    once, so the whole thing is a couple of numpy reductions rather than a
    nested adaptive quadrature.

    Reference: David, H.A. & Nagaraja (2003), *Order Statistics*, Ch. 3.
    """
    if n < 2:
        raise ValueError("The range distribution is undefined for n < 2.")

    # --- First moment: single-integral identity on a fixed grid ---
    # The standard normal is negligible beyond +/- 8 sigma; the range of n
    # normals extends a little further, so integrate x over [-9, 9].
    x = np.linspace(-9.0, 9.0, 8001)
    Phi_x = norm.cdf(x)
    integrand_m1 = 1.0 - Phi_x**n - (1.0 - Phi_x) ** n
    m1 = float(np.trapezoid(integrand_m1, x))

    # --- Second moment: survival form, vectorised over the w-grid ---
    # F_W(w) = n * integral phi(u) [Phi(u+w) - Phi(u)]^(n-1) du
    u = np.linspace(-9.0, 9.0, 2001)
    phi_u = norm.pdf(u)
    Phi_u = norm.cdf(u)
    w = np.linspace(0.0, 18.0, 2001)
    # Broadcast: rows indexed by w, columns by u.
    diff = norm.cdf(u[None, :] + w[:, None]) - Phi_u[None, :]
    inner = phi_u[None, :] * diff ** (n - 1)
    F_W = n * np.trapezoid(inner, u, axis=1)
    F_W = np.clip(F_W, 0.0, 1.0)
    m2 = float(np.trapezoid(2.0 * w * (1.0 - F_W), w))

    return m1, m2


def _range_moment(n: int, order: int) -> float:
    """Return ``E[W^order]`` for ``order`` in {1, 2} (thin accessor)."""
    m1, m2 = _range_moments(n)
    if order == 1:
        return m1
    if order == 2:
        return m2
    raise ValueError("Only the first two range moments are implemented.")


def d2(n: int) -> float:
    """Exact ``d2 = E[R/sigma]`` (mean of the relative range).

    Reference: Montgomery (2013), Appendix Table VI; David (1981).
    """
    return _range_moment(n, 1)


def d3(n: int) -> float:
    """Exact ``d3 = sqrt(Var[R/sigma])`` (std dev of the relative range).

    Reference: Montgomery (2013), Appendix Table VI.
    """
    m1 = _range_moment(n, 1)
    m2 = _range_moment(n, 2)
    return float(sqrt(max(0.0, m2 - m1 * m1)))


def a2(n: int) -> float:
    """X-bar chart factor for use with R-bar: ``A2 = 3 / (d2 * sqrt(n))``."""
    return 3.0 / (d2(n) * sqrt(n))


def a3(n: int) -> float:
    """X-bar chart factor for use with S-bar: ``A3 = 3 / (c4 * sqrt(n))``."""
    return 3.0 / (c4(n) * sqrt(n))


def d4_factor(n: int) -> float:
    """R chart upper factor: ``D4 = 1 + 3 * d3 / d2``."""
    return 1.0 + 3.0 * d3(n) / d2(n)


def d3_factor(n: int) -> float:
    """R chart lower factor: ``D3 = max(0, 1 - 3 * d3 / d2)``."""
    return max(0.0, 1.0 - 3.0 * d3(n) / d2(n))


def b4(n: int) -> float:
    """S chart upper factor: ``B4 = 1 + 3/c4 * sqrt(1 - c4**2)``."""
    c = c4(n)
    return 1.0 + 3.0 / c * sqrt(1.0 - c * c)


def b3(n: int) -> float:
    """S chart lower factor: ``B3 = max(0, 1 - 3/c4 * sqrt(1 - c4**2))``."""
    c = c4(n)
    return max(0.0, 1.0 - 3.0 / c * sqrt(1.0 - c * c))


def e2(n: int = 2) -> float:
    """Individuals-chart factor: ``E2 = 3 / d2``.

    For the standard moving range of two consecutive observations ``n=2`` and
    ``E2 = 3 / 1.128 = 2.660``.
    """
    return 3.0 / d2(n)
