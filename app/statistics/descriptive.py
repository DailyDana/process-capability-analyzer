"""Descriptive statistics for measurement data.

Pure functions over 1-D numeric arrays. Every function:

* accepts array-like input and coerces to ``float`` ndarray,
* ignores nothing silently — NaN handling is explicit via the ``nan_policy``
  argument (default ``"raise"`` so callers make a conscious choice),
* preserves full floating-point precision (no internal rounding).

Statistical conventions
-----------------------
* **Sample variance / standard deviation** use Bessel's correction (``ddof=1``)
  by default, because SPC works with *samples* drawn from an ongoing process,
  not the full population. This is the estimator behind Pp/Ppk (overall sigma)
  and the S-chart. The population form (``ddof=0``) is available via ``ddof``.
* **Median** uses linear interpolation between the two central order statistics
  for even ``n`` (numpy default), matching standard statistical software.

References
----------
* Montgomery, D.C. (2013). *Introduction to Statistical Quality Control*, 7th
  ed., Wiley — Chapter 3 (describing the sample statistics used in SPC).
* NIST/SEMATECH e-Handbook of Statistical Methods, Section 1.3.5
  (summary statistics definitions).
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_DDOF",
    "to_float_array",
    "count",
    "mean",
    "median",
    "variance",
    "std_dev",
    "minimum",
    "maximum",
    "value_range",
    "DescriptiveSummary",
    "describe",
]

# Bessel-corrected (sample) estimator is the SPC default.
DEFAULT_DDOF: Final[int] = 1

NanPolicy = Literal["raise", "omit"]


def to_float_array(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> NDArray[np.float64]:
    """Coerce array-like input to a 1-D ``float64`` ndarray.

    Parameters
    ----------
    values:
        Array-like of numbers.
    nan_policy:
        ``"raise"`` (default) raises if NaNs are present; ``"omit"`` drops them.
        Non-finite values (``inf``) always raise — they indicate a computation
        or data error rather than a legitimate measurement.

    Raises
    ------
    ValueError
        If the array is empty, contains ``inf``, or contains NaN under the
        ``"raise"`` policy.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Input contains no values.")
    if np.isinf(arr).any():
        raise ValueError("Input contains non-finite (inf) values.")
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        if nan_policy == "raise":
            raise ValueError(
                f"Input contains {int(nan_mask.sum())} NaN value(s); "
                "pass nan_policy='omit' to drop them explicitly."
            )
        arr = arr[~nan_mask]
        if arr.size == 0:
            raise ValueError("All values were NaN; nothing to summarize.")
    return arr


def count(values: ArrayLike, *, nan_policy: NanPolicy = "omit") -> int:
    """Return the number of usable (finite, non-NaN) observations."""
    return int(to_float_array(values, nan_policy=nan_policy).size)


def mean(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> float:
    """Arithmetic mean ``x-bar = (1/n) * sum(x_i)``."""
    return float(np.mean(to_float_array(values, nan_policy=nan_policy)))


def median(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> float:
    """Median (50th percentile) with linear interpolation for even ``n``."""
    return float(np.median(to_float_array(values, nan_policy=nan_policy)))


def variance(
    values: ArrayLike, *, ddof: int = DEFAULT_DDOF, nan_policy: NanPolicy = "raise"
) -> float:
    """Variance.

    ``ddof=1`` (default) gives the unbiased sample variance
    ``s^2 = sum((x_i - x-bar)^2) / (n - 1)``.
    ``ddof=0`` gives the population variance (divide by ``n``).

    Raises
    ------
    ValueError
        If fewer than ``ddof + 1`` observations remain (variance undefined).
    """
    arr = to_float_array(values, nan_policy=nan_policy)
    if arr.size <= ddof:
        raise ValueError(
            f"Variance with ddof={ddof} requires more than {ddof} observation(s); "
            f"got {arr.size}."
        )
    return float(np.var(arr, ddof=ddof))


def std_dev(
    values: ArrayLike, *, ddof: int = DEFAULT_DDOF, nan_policy: NanPolicy = "raise"
) -> float:
    """Standard deviation (square root of :func:`variance`)."""
    arr = to_float_array(values, nan_policy=nan_policy)
    if arr.size <= ddof:
        raise ValueError(
            f"Standard deviation with ddof={ddof} requires more than {ddof} "
            f"observation(s); got {arr.size}."
        )
    return float(np.std(arr, ddof=ddof))


def minimum(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> float:
    """Minimum observation."""
    return float(np.min(to_float_array(values, nan_policy=nan_policy)))


def maximum(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> float:
    """Maximum observation."""
    return float(np.max(to_float_array(values, nan_policy=nan_policy)))


def value_range(values: ArrayLike, *, nan_policy: NanPolicy = "raise") -> float:
    """Range ``R = max - min``."""
    arr = to_float_array(values, nan_policy=nan_policy)
    return float(np.max(arr) - np.min(arr))


class DescriptiveSummary:
    """Immutable bundle of descriptive statistics for one measurement series.

    All fields preserve full floating-point precision. Presentation rounding is
    a UI/reporting concern and is deliberately not applied here.
    """

    __slots__ = (
        "n",
        "mean",
        "median",
        "variance",
        "std_dev",
        "minimum",
        "maximum",
        "value_range",
        "ddof",
    )

    def __init__(
        self,
        *,
        n: int,
        mean: float,
        median: float,
        variance: float,
        std_dev: float,
        minimum: float,
        maximum: float,
        value_range: float,
        ddof: int,
    ) -> None:
        self.n = n
        self.mean = mean
        self.median = median
        self.variance = variance
        self.std_dev = std_dev
        self.minimum = minimum
        self.maximum = maximum
        self.value_range = value_range
        self.ddof = ddof

    def as_dict(self) -> dict[str, float | int]:
        """Return the summary as a plain dictionary (for reporting/export)."""
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "range": self.value_range,
            "ddof": self.ddof,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DescriptiveSummary(n={self.n}, mean={self.mean:.6g}, "
            f"median={self.median:.6g}, std_dev={self.std_dev:.6g}, "
            f"range={self.value_range:.6g}, ddof={self.ddof})"
        )


def describe(
    values: ArrayLike, *, ddof: int = DEFAULT_DDOF, nan_policy: NanPolicy = "omit"
) -> DescriptiveSummary:
    """Compute a full :class:`DescriptiveSummary` in one pass-through.

    ``nan_policy`` defaults to ``"omit"`` here because ``describe`` is the
    convenience entry point most likely to be handed raw, imperfect data; the
    dropped-count is recoverable by comparing ``n`` against the input length.
    Standard deviation / variance require n > ddof; if that fails they are set
    to ``nan`` rather than raising, so a summary is still produced for tiny
    samples (the capability layer flags these separately in a later milestone).
    """
    arr = to_float_array(values, nan_policy=nan_policy)
    n = int(arr.size)
    if n > ddof:
        var = float(np.var(arr, ddof=ddof))
        sd = float(np.std(arr, ddof=ddof))
    else:
        var = float("nan")
        sd = float("nan")
    return DescriptiveSummary(
        n=n,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        variance=var,
        std_dev=sd,
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        value_range=float(np.max(arr) - np.min(arr)),
        ddof=ddof,
    )
