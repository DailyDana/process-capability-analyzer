"""Lag-1 autocorrelation of a time-ordered series.

Control limits for individuals assume independent observations. The lag-1
sample autocorrelation coefficient measures how far that holds
(``docs/FORMULAS.md`` §3.9)::

    r1 = sum_{i=1}^{n-1} (x_i - xbar)(x_{i+1} - xbar) / sum_{i=1}^{n} (x_i - xbar)^2

This is the definition NIST certifies in its Statistical Reference Datasets,
against which the implementation is tested.

**Approximate test.** For an independent series ``r1`` is approximately normal
with mean 0 and standard deviation ``1 / sqrt(n)`` (Bartlett; Box, Jenkins &
Reinsel), so ``z = r1 * sqrt(n)`` and the series is called autocorrelated when
``|r1| > z_{1 - alpha/2} / sqrt(n)``. It is a large-sample approximation and
is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

__all__ = ["AutocorrelationResult", "lag1_autocorrelation", "MIN_OBSERVATIONS"]

#: Below this the approximate test says nothing useful, so it is not run.
MIN_OBSERVATIONS = 15


@dataclass(frozen=True, slots=True)
class AutocorrelationResult:
    """Lag-1 autocorrelation and its approximate significance.

    ``r1`` is ``None`` for a series with no variation. ``significant`` is
    ``None`` when the test was not run (too few observations, or no ``r1``).
    """

    n: int
    r1: Optional[float]
    bound: Optional[float]  # |r1| above this is significant at ``alpha``
    z: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]
    alpha: float

    def as_dict(self) -> dict[str, object]:
        return {"n": self.n, "r1": self.r1, "bound": self.bound, "z": self.z,
                "p_value": self.p_value, "significant": self.significant, "alpha": self.alpha}


def lag1_autocorrelation(values: ArrayLike, *, alpha: float = 0.05) -> AutocorrelationResult:
    """Lag-1 autocorrelation of ``values`` (in time order) with the approximate test."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    x = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(x).all():
        raise ValueError("Autocorrelation needs finite values; clean the data first.")
    n = int(x.size)
    if n < 3:
        raise ValueError(f"Autocorrelation needs at least 3 observations; got {n}.")

    d = x - x.mean()
    d = d - d.mean()  # second pass removes the rounding left in the first mean
    denom = float(np.dot(d, d))
    # A spread within floating-point noise of the magnitude of the data is no variation.
    if denom <= n * (64.0 * np.finfo(float).eps * float(np.max(np.abs(x)))) ** 2:
        return AutocorrelationResult(n, None, None, None, None, None, alpha)
    r1 = float(np.dot(d[:-1], d[1:]) / denom)

    if n < MIN_OBSERVATIONS:
        return AutocorrelationResult(n, r1, None, None, None, None, alpha)
    bound = float(norm.ppf(1.0 - alpha / 2.0)) / sqrt(n)
    z = r1 * sqrt(n)
    p = float(2.0 * norm.sf(abs(z)))
    return AutocorrelationResult(n, r1, bound, z, p, bool(abs(r1) > bound), alpha)
