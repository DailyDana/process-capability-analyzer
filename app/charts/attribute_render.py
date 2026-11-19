"""Attribute control chart figure (reads the chart and its evaluation only; Agg backend).

Limits are drawn as steps so that varying sample sizes show as varying
limits. Samples the centre line was estimated from are filled, the others
hollow (crossed when they were excluded from the baseline); signalling
samples are red.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from app.spc.attribute import AttributeChart, AttributeKind
from app.spc.rules import RuleEvaluation

__all__ = ["render_attribute_chart", "Y_LABEL"]

_C_POINT = "#1f3b6f"
_C_LIMIT = "#b00020"
_C_CENTER = "#2c7a2c"

Y_LABEL = {
    AttributeKind.P: "Fraction nonconforming (p)",
    AttributeKind.NP: "Number nonconforming (np)",
    AttributeKind.C: "Nonconformities (c)",
    AttributeKind.U: "Nonconformities per unit (u)",
}


def render_attribute_chart(
    chart: AttributeChart,
    evaluation: Optional[RuleEvaluation] = None,
    *,
    figsize=(10.0, 4.4),
    dpi: int = 150,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    k = chart.n_samples
    x = np.arange(1, k + 1)
    flagged = chart.beyond_limits.copy()
    if evaluation is not None:
        for v in evaluation.violations:
            flagged[list(v.indices)] = True

    base = chart.in_baseline
    last_base = int(np.nonzero(base)[0].max()) if base.any() else -1
    if chart.center_is_standard:
        filled = np.ones(k, dtype=bool)
        excluded = np.zeros(k, dtype=bool)
    else:
        filled = base
        excluded = ~base & (np.arange(k) <= last_base)

    ax.plot(x, chart.points, color=_C_POINT, linewidth=0.9, zorder=2)
    for is_flagged, color, size in ((False, _C_POINT, 22), (True, _C_LIMIT, 36)):
        sel = flagged == is_flagged
        m = sel & filled
        ax.scatter(x[m], chart.points[m], s=size, facecolors=color, edgecolors=color, zorder=3)
        m = sel & ~filled & ~excluded
        ax.scatter(x[m], chart.points[m], s=size, facecolors="white", edgecolors=color, zorder=3)
        m = sel & excluded
        ax.scatter(x[m], chart.points[m], s=size + 20, marker="x", color=color, zorder=3)

    edges = np.arange(0.5, k + 1.0, 1.0)
    for y in (chart.ucl, chart.lcl):
        ax.stairs(y, edges, color=_C_LIMIT, linestyle="--", linewidth=1.1, baseline=None)
    ax.axhline(chart.center_line, color=_C_CENTER, linewidth=1.1)
    if not chart.center_is_standard and 0 <= last_base < k - 1:
        ax.axvline(last_base + 1.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.annotate("baseline | new", (last_base + 1.5, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color="#555555")
    varying = "" if chart.constant_size else " (varies)"
    labels = (
        (float(chart.ucl[-1]), _C_LIMIT, f"UCL = {chart.ucl[-1]:.5g}{varying}"),
        (chart.center_line, _C_CENTER, f"CL = {chart.center_line:.5g}"),
        (float(chart.lcl[-1]), _C_LIMIT, f"LCL = {chart.lcl[-1]:.5g}{varying}"),
    )
    for y, color, label in labels:
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=7.5, color=color, xytext=(4, 0), textcoords="offset points")
    ax.set_xlim(0.5, k + 0.5)
    ax.set_xlabel("Sample")
    ax.set_ylabel(Y_LABEL[chart.kind])
    ax.set_title(f"{chart.kind.value} chart", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
