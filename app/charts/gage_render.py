"""Matplotlib renderers for the Gage R&R chart models.

Same rules as :mod:`app.charts.render`: read the model only, never recompute;
``Agg`` backend; out-of-control points carry shape + outline, not colour alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from app.charts.gage_model import ComponentsModel, InteractionModel, RangeByOperatorModel

__all__ = ["render_components", "render_range_by_operator", "render_interaction"]

_C_POINT = "#1f3b6f"
_C_CL = "#2c7a2c"
_C_LIMIT = "#b00020"
# Okabe-Ito palette: distinguishable with common colour-vision deficiencies.
_SERIES = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def render_components(model: ComponentsModel, *, figsize=(9.0, 4.2), dpi: int = 150) -> Figure:
    """Grouped bars per source; dotted guides at the acceptance bands."""
    series = [("% Contribution", model.pct_contribution), ("% Study Var", model.pct_study_var)]
    if model.pct_tolerance is not None:
        series.append(("% Tolerance", model.pct_tolerance))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(model.sources))
    width = 0.8 / len(series)
    for k, (label, vals) in enumerate(series):
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * width, heights, width, label=label,
                      color=_SERIES[k], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7, xytext=(0, 1), textcoords="offset points")
    for level in (model.acceptable_max, model.marginal_max):
        ax.axhline(level, color="#777777", linestyle=":", linewidth=0.9)
        ax.annotate(f"{level:g}%", (1.0, level), xycoords=("axes fraction", "data"), ha="left",
                    va="center", fontsize=7, color="#555555", xytext=(4, 0), textcoords="offset points")
    ax.set_xticks(x, model.sources)
    ax.set_ylabel("Percent")
    ax.set_title("Components of variation")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_range_by_operator(model: RangeByOperatorModel, *, figsize=(10.0, 4.2), dpi: int = 150) -> Figure:
    """Ranges grouped by operator, with R-bar and UCL."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n = len(model.parts)
    for j, op in enumerate(model.operators):
        xs = np.arange(n) + j * (n + 1)
        ys = np.array(model.ranges[j])
        ooc = np.array(model.out_of_control[j])
        ax.plot(xs, ys, color=_C_POINT, linewidth=0.8, marker="o", markersize=4, zorder=2)
        if ooc.any():
            ax.scatter(xs[ooc], ys[ooc], marker="s", s=60, facecolor="white", edgecolor=_C_LIMIT,
                       linewidth=1.6, zorder=3)
        ax.annotate(op, (xs.mean(), 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        if j:
            ax.axvline(xs[0] - 1, color="#cccccc", linewidth=0.8)
    ax.axhline(model.r_bar, color=_C_CL, linewidth=1.2)
    ax.axhline(model.ucl, color=_C_LIMIT, linewidth=1.2, linestyle="--")
    ax.annotate(f"R̄ = {model.r_bar:.4g}", (1.0, model.r_bar), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=8, color=_C_CL, xytext=(4, 0), textcoords="offset points")
    ax.annotate(f"UCL = {model.ucl:.4g}", (1.0, model.ucl), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=8, color=_C_LIMIT, xytext=(4, 0), textcoords="offset points")
    ax.set_xticks([])
    ax.set_xlabel("Parts, grouped by operator")
    ax.set_ylabel(f"Range ({model.unit})" if model.unit else "Range")
    ax.set_title("R chart by operator", pad=18)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=_C_POINT, marker="o", markersize=4, linewidth=0.8, label="Range (part x operator)"),
        Line2D([], [], color=_C_LIMIT, marker="s", markerfacecolor="white", linestyle="none", markersize=7,
               label="Above UCL"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def render_interaction(model: InteractionModel, *, figsize=(10.0, 4.2), dpi: int = 150) -> Figure:
    """One line per operator across parts."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(model.parts))
    for j, op in enumerate(model.operators):
        ax.plot(x, model.averages[j], color=_SERIES[j % len(_SERIES)], marker=_MARKERS[j % len(_MARKERS)],
                markersize=5, linewidth=1.2, label=op)
    ax.set_xticks(x, model.parts)
    ax.set_xlabel("Part")
    ax.set_ylabel(f"Average ({model.unit})" if model.unit else "Average")
    ax.set_title("Operator x part interaction")
    ax.legend(title="Operator", fontsize=8, title_fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
