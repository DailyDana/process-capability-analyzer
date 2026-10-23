"""Charts for the non-normal capability analysis (read-only, Agg backend).

* Histogram of the data with the fitted density, the normal density for
  comparison, and the specification limits.
* Probability plot: ordered data against the model's quantiles at Benard's
  plotting positions ``(i - 0.3) / (n + 0.4)``; points on the line mean the
  model describes the data, including the tails that Ppk depends on.

For Box-Cox the model is the back-transformed normal: density
``phi((T(x) - m) / s) / s * x^(lambda - 1)`` and quantiles ``T^-1(m + s z)``.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from app.capability.nonnormal import BoxCoxResult, DistributionFit, NonNormalSpec, boxcox_transform

__all__ = ["render_histogram", "render_probability_plot"]

_C_BAR = "#c9d4e6"
_C_MODEL = "#0072B2"
_C_NORMAL = "#E69F00"
_C_SPEC = "#b00020"


def _inverse_boxcox(y: np.ndarray, lam: float) -> np.ndarray:
    return np.exp(y) if lam == 0 else np.power(np.maximum(lam * y + 1.0, 1e-300), 1.0 / lam)


def _model_pdf(xs: np.ndarray, fit: Optional[DistributionFit], bc: Optional[BoxCoxResult]) -> np.ndarray:
    if bc is not None:
        pos = xs > 0
        out = np.zeros_like(xs)
        y = boxcox_transform(xs[pos], bc.lambda_used)
        out[pos] = stats.norm.pdf(y, bc.mean_t, bc.sd_t) * np.power(xs[pos], bc.lambda_used - 1.0)
        return out
    return np.asarray(fit.frozen.pdf(xs))  # type: ignore[union-attr,attr-defined]


def _model_ppf(q: np.ndarray, fit: Optional[DistributionFit], bc: Optional[BoxCoxResult]) -> np.ndarray:
    if bc is not None:
        return _inverse_boxcox(bc.mean_t + bc.sd_t * stats.norm.ppf(q), bc.lambda_used)
    return np.asarray(fit.frozen.ppf(q))  # type: ignore[union-attr,attr-defined]


def render_histogram(values, spec: NonNormalSpec, fit: Optional[DistributionFit], bc: Optional[BoxCoxResult],
                     label: str, normal: DistributionFit, *, unit: Optional[str] = None,
                     figsize=(10.0, 4.2), dpi: int = 150) -> Figure:
    x = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(x, bins="auto", density=True, color=_C_BAR, edgecolor="white")
    lo = min(x.min(), spec.lsl if spec.lsl is not None else x.min())
    hi = max(x.max(), spec.usl if spec.usl is not None else x.max())
    pad = 0.08 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 600)
    ax.plot(xs, _model_pdf(xs, fit, bc), color=_C_MODEL, linewidth=2.0, label=label)
    ax.plot(xs, normal.frozen.pdf(xs), color=_C_NORMAL, linewidth=1.3, linestyle="--",  # type: ignore[attr-defined]
            label="Normal (for comparison)")
    for v, name, style in ((spec.usl, "USL", "-"), (spec.lsl, "LB" if spec.lsl_is_boundary else "LSL",
                                                      ":" if spec.lsl_is_boundary else "-")):
        if v is not None:
            ax.axvline(v, color=_C_SPEC, linestyle=style, linewidth=1.4)
            ax.annotate(name, (v, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=8, color=_C_SPEC)
    ax.set_xlabel(f"Value ({unit})" if unit else "Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of the data", pad=14)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_probability_plot(values, fit: Optional[DistributionFit], bc: Optional[BoxCoxResult], label: str,
                            *, figsize=(6.0, 4.6), dpi: int = 150) -> Figure:
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    q = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    model = _model_ppf(q, fit, bc)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(model, x, s=10, color=_C_MODEL, zorder=3)
    lim = [min(model.min(), x.min()), max(model.max(), x.max())]
    ax.plot(lim, lim, color="#555555", linewidth=1.0)
    ax.set_xlabel(f"Model quantile ({label})")
    ax.set_ylabel("Observed value")
    ax.set_title("Probability plot")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
