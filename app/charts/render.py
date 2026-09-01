"""Matplotlib renderer for :class:`~app.charts.model.ChartModel`.

Consumes a pure :class:`ChartModel` and produces a professional SPC chart. It
reads only the model — it never touches the M3 calculators or recomputes a
limit. Uses the non-interactive ``Agg`` backend so it is headless/test-safe.

Visual design
-------------
* CL solid, UCL/LCL dashed, each labeled; a faint 1-/2-sigma zone shading on
  symmetric charts.
* Normal points: small filled circles connected by a thin line.
* Out-of-control points are distinguished by **marker shape + outline + color**
  (not color alone, for accessibility): beyond-limit points use a red-edged
  square, rule-violation points use a red-edged triangle; a point that is both
  gets the square (beyond-limit dominates) plus a rule annotation.
* Rule violations are annotated sparingly: only OOC points carry a small rule-ID
  label; normal points are never labeled (keeps dense charts readable).
* A legend explains every symbol and the limit lines.
* The measurement unit (when present) is in the Y-axis label.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, deterministic; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from app.charts.model import ChartModel, ChartPairModel, PointClass

__all__ = [
    "render_chart",
    "render_pair",
    "save_png",
    "save_svg",
    "figure_to_png_bytes",
    "figure_to_svg_bytes",
]

# Colors kept colorblind-aware; shape/outline carry the same information.
_COLOR_NORMAL = "#1f3b6f"
_COLOR_CL = "#2c7a2c"
_COLOR_LIMIT = "#b00020"
_COLOR_OOC = "#b00020"
_COLOR_ZONE1 = "#e8eef7"
_COLOR_ZONE2 = "#f2f5fb"


def render_chart(
    model: ChartModel,
    *,
    figsize: tuple[float, float] = (10.0, 4.5),
    dpi: int = 150,
    show_zones: bool = True,
    annotate_rules: bool = True,
) -> Figure:
    """Render a single :class:`ChartModel` to a matplotlib :class:`Figure`.

    The caller owns the returned figure (close it with ``plt.close(fig)`` when
    done, or use the ``save_*`` helpers which do not auto-close).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    positions = model.positions
    values = model.values

    # --- zone shading (symmetric charts only; zones are meaningful there) ---
    if show_zones and model.is_symmetric and model.sigma_points > 0:
        ax.axhspan(model.zone_2sigma[0], model.zone_2sigma[1], color=_COLOR_ZONE2, zorder=0)
        ax.axhspan(model.zone_1sigma[0], model.zone_1sigma[1], color=_COLOR_ZONE1, zorder=0)

    # --- limit + center lines ---
    ax.axhline(model.center_line, color=_COLOR_CL, lw=1.6, zorder=2,
               label=f"CL = {model.center_line:.4g}")
    ax.axhline(model.ucl, color=_COLOR_LIMIT, lw=1.3, ls="--", zorder=2,
               label=f"UCL = {model.ucl:.4g}")
    ax.axhline(model.lcl, color=_COLOR_LIMIT, lw=1.3, ls="--", zorder=2,
               label=f"LCL = {model.lcl:.4g}")

    # --- connecting line + normal points ---
    ax.plot(positions, values, color=_COLOR_NORMAL, lw=1.0, zorder=3)
    normal = model.points_in_class(PointClass.NORMAL)
    if normal:
        ax.scatter(
            [p.position for p in normal],
            [p.value for p in normal],
            marker="o", s=28, facecolor=_COLOR_NORMAL, edgecolor="white",
            linewidths=0.6, zorder=4, label="In-control observation",
        )

    # --- beyond-limit points: red-edged SQUARE ---
    beyond = [p for p in model.points if PointClass.BEYOND_LIMITS in p.classes]
    if beyond:
        ax.scatter(
            [p.position for p in beyond],
            [p.value for p in beyond],
            marker="s", s=70, facecolor=_COLOR_OOC, edgecolor="black",
            linewidths=1.1, zorder=6, label="Beyond control limit",
        )

    # --- rule-violation-only points (not also beyond limit): red-edged TRIANGLE ---
    rule_only = [
        p for p in model.points
        if PointClass.RULE_VIOLATION in p.classes
        and PointClass.BEYOND_LIMITS not in p.classes
    ]
    if rule_only:
        ax.scatter(
            [p.position for p in rule_only],
            [p.value for p in rule_only],
            marker="^", s=64, facecolor="none", edgecolor=_COLOR_OOC,
            linewidths=1.4, zorder=5, label="Rule violation",
        )

    # --- sparse rule-ID annotations on OOC points only ---
    if annotate_rules:
        for p in model.ooc_points:
            if p.violation_ids:
                label = ",".join(p.violation_ids)
                ax.annotate(
                    label,
                    (p.position, p.value),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=_COLOR_OOC, zorder=7,
                )

    ax.set_title(model.title, fontsize=11, fontweight="bold")
    ax.set_xlabel(model.x_label, fontsize=10)
    ax.set_ylabel(model.y_label, fontsize=10)
    ax.margins(x=0.02)
    ax.grid(True, axis="y", alpha=0.25, zorder=1)

    # De-duplicate legend entries while preserving order.
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Line2D] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return fig


def render_pair(
    pair: ChartPairModel,
    *,
    figsize: tuple[float, float] = (10.0, 7.5),
    dpi: int = 150,
    show_zones: bool = True,
    annotate_rules: bool = True,
) -> Figure:
    """Render a location+dispersion pair stacked in one figure (shared x-axis)."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.8]},
    )
    _draw_on_axis(ax_top, pair.location, show_zones=show_zones, annotate_rules=annotate_rules)
    _draw_on_axis(ax_bot, pair.dispersion, show_zones=show_zones, annotate_rules=annotate_rules)
    ax_top.set_xlabel("")  # shared; only bottom shows it
    fig.tight_layout()
    return fig


def _draw_on_axis(ax, model: ChartModel, *, show_zones: bool, annotate_rules: bool) -> None:
    """Render a ChartModel onto an existing axis (used by render_pair)."""
    positions = model.positions
    values = model.values

    if show_zones and model.is_symmetric and model.sigma_points > 0:
        ax.axhspan(model.zone_2sigma[0], model.zone_2sigma[1], color=_COLOR_ZONE2, zorder=0)
        ax.axhspan(model.zone_1sigma[0], model.zone_1sigma[1], color=_COLOR_ZONE1, zorder=0)

    ax.axhline(model.center_line, color=_COLOR_CL, lw=1.6, zorder=2,
               label=f"CL = {model.center_line:.4g}")
    ax.axhline(model.ucl, color=_COLOR_LIMIT, lw=1.3, ls="--", zorder=2,
               label=f"UCL = {model.ucl:.4g}")
    ax.axhline(model.lcl, color=_COLOR_LIMIT, lw=1.3, ls="--", zorder=2,
               label=f"LCL = {model.lcl:.4g}")
    ax.plot(positions, values, color=_COLOR_NORMAL, lw=1.0, zorder=3)

    normal = model.points_in_class(PointClass.NORMAL)
    if normal:
        ax.scatter([p.position for p in normal], [p.value for p in normal],
                   marker="o", s=26, facecolor=_COLOR_NORMAL, edgecolor="white",
                   linewidths=0.6, zorder=4, label="In-control observation")
    beyond = [p for p in model.points if PointClass.BEYOND_LIMITS in p.classes]
    if beyond:
        ax.scatter([p.position for p in beyond], [p.value for p in beyond],
                   marker="s", s=66, facecolor=_COLOR_OOC, edgecolor="black",
                   linewidths=1.1, zorder=6, label="Beyond control limit")
    rule_only = [p for p in model.points
                 if PointClass.RULE_VIOLATION in p.classes
                 and PointClass.BEYOND_LIMITS not in p.classes]
    if rule_only:
        ax.scatter([p.position for p in rule_only], [p.value for p in rule_only],
                   marker="^", s=60, facecolor="none", edgecolor=_COLOR_OOC,
                   linewidths=1.4, zorder=5, label="Rule violation")
    if annotate_rules:
        for p in model.ooc_points:
            if p.violation_ids:
                ax.annotate(",".join(p.violation_ids), (p.position, p.value),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=7, color=_COLOR_OOC, zorder=7)

    ax.set_title(model.title, fontsize=10, fontweight="bold")
    ax.set_xlabel(model.x_label, fontsize=9)
    ax.set_ylabel(model.y_label, fontsize=9)
    ax.margins(x=0.02)
    ax.grid(True, axis="y", alpha=0.25, zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def save_png(fig: Figure, path: str, *, dpi: int = 200) -> str:
    """Save a figure as high-resolution PNG. Returns the path."""
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    return path


def save_svg(fig: Figure, path: str) -> str:
    """Save a figure as SVG (vector, ideal for PDF embedding). Returns the path."""
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def figure_to_png_bytes(fig: Figure, *, dpi: int = 200) -> bytes:
    """Return the figure as PNG bytes (for in-memory embedding / Streamlit)."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def figure_to_svg_bytes(fig: Figure) -> bytes:
    """Return the figure as SVG bytes."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()
