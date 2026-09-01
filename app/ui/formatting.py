"""Presentation-only formatting helpers for the UI.

IMPORTANT: these functions round for DISPLAY only. The underlying analysis
objects always retain full floating-point precision; nothing here feeds back
into a calculation.
"""

from __future__ import annotations

from math import isinf, isnan
from typing import Optional

__all__ = ["fmt_num", "fmt_index", "with_unit", "severity_icon", "NA_DISPLAY"]

NA_DISPLAY = "—"


def fmt_num(x: Optional[float], decimals: int = 4) -> str:
    """Format a number for display; missing -> em dash, sentinels readable."""
    if x is None:
        return NA_DISPLAY
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return NA_DISPLAY
    if isnan(fx):
        return "NaN"
    if isinf(fx):
        return "∞" if fx > 0 else "−∞"
    return f"{fx:,.{decimals}f}"


def fmt_index(x: Optional[float], decimals: int = 3) -> str:
    """Format a capability index (fewer decimals than raw measurements)."""
    return fmt_num(x, decimals)


def with_unit(text: str, unit: Optional[str]) -> str:
    """Append a unit in parentheses when one is available."""
    return f"{text} ({unit})" if unit else text


def severity_icon(severity: str) -> str:
    """Return a leading glyph for a severity level (visual, not color-only)."""
    return {
        "critical": "🛑",
        "warning": "⚠️",
        "notice": "ℹ️",
        "info": "✅",
    }.get(severity, "•")
