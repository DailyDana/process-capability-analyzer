"""Run chart for a Type-1 gauge study (reads the result only; Agg backend).

Readings in measurement order, with the reference value, the band
reference +- K/200 * T (+-0.1 T by default -- the region Cgk is measured
against) and the mean +- (L/2) s spread that Cg compares with it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from app.msa.type1 import Type1Result

__all__ = ["render_type1_run_chart"]

_C_POINT = "#1f3b6f"
_C_REF = "#2c7a2c"
_C_BAND = "#b00020"
_C_SPREAD = "#777777"


def render_type1_run_chart(result: Type1Result, *, unit: str | None = None,
                           figsize=(10.0, 4.2), dpi: int = 150) -> Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(1, result.n + 1)
    lo, hi = result.band
    ax.plot(x, result.values, color=_C_POINT, marker="o", markersize=3.5, linewidth=0.8, zorder=3)
    lines = [
        (result.reference, _C_REF, "-", f"x_m = {result.reference:.6g}"),
        (lo, _C_BAND, "--", f"x_m - {result.k_percent / 2:g}% T"),
        (hi, _C_BAND, "--", f"x_m + {result.k_percent / 2:g}% T"),
        (result.mean, _C_POINT, ":", f"mean = {result.mean:.6g}"),
    ]
    half = result.l_sigma / 2.0 * result.sd
    if result.sd > 0:
        lines += [
            (result.mean - half, _C_SPREAD, "--", f"mean - {result.l_sigma / 2:g}s"),
            (result.mean + half, _C_SPREAD, "--", f"mean + {result.l_sigma / 2:g}s"),
        ]
    for y, color, style, label in lines:
        ax.axhline(y, color=color, linestyle=style, linewidth=1.1)
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=7.5, color=color, xytext=(4, 0), textcoords="offset points")
    pad = 0.15 * (hi - lo)
    vmin, vmax = min(min(result.values), lo), max(max(result.values), hi)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Reading")
    ax.set_ylabel(f"Value ({unit})" if unit else "Value")
    ax.set_title(f"Type-1 run chart  (Cg = {result.cg:.2f}, Cgk = {result.cgk:.2f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
