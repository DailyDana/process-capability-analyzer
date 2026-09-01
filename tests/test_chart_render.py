"""Tests for the chart RENDERER (app.charts.render).

These verify the actual plotted data and annotations on the matplotlib figure —
axis labels, the CL/UCL/LCL line y-values, the scatter offsets for normal vs
out-of-control points, rule annotations, legend presence — not merely that a
file was produced. Export tests check real PNG/SVG byte output.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from app.charts.model import build_chart_model, build_pair_model
from app.charts import render
from app.spc.limits import imr_chart, xbar_r_chart, xbar_s_chart
from app.spc.rules import RuleSet, evaluate_rules


def _ooc_subgroups():
    sg = [np.array([10.0, 10.2, 9.8, 10.1, 9.9]) for _ in range(9)]
    sg.append(np.array([15.0, 15.2, 14.8, 15.1, 14.9]))
    return sg


def _hlines_y(ax):
    """Return the y-values of all axhline (infinite horizontal) lines on ax."""
    ys = []
    for line in ax.get_lines():
        xd = line.get_xdata()
        yd = line.get_ydata()
        # axhline lines have constant y across x [0,1] in axes coords.
        if len(yd) == 2 and yd[0] == yd[1]:
            ys.append(float(yd[0]))
    return ys


class TestLimitLinesDrawn:
    def test_cl_ucl_lcl_lines_match_model(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location, evaluate_rules(pair.location, rule_set=RuleSet.NELSON))
        fig = render.render_chart(m)
        ax = fig.axes[0]
        ys = _hlines_y(ax)
        for expected in (m.center_line, m.ucl, m.lcl):
            assert any(abs(y - expected) < 1e-9 for y in ys), f"{expected} not drawn"
        plt.close(fig)

    def test_axis_labels_and_unit(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location, unit="mm")
        fig = render.render_chart(m)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Subgroup"
        assert "mm" in ax.get_ylabel()
        assert ax.get_title()  # non-empty title
        plt.close(fig)

    def test_individuals_axis_label(self) -> None:
        pair = imr_chart(np.array([10.0, 11.0, 12.0, 10.5, 11.5]))
        m = build_chart_model(pair.location, unit="inch")
        fig = render.render_chart(m)
        assert fig.axes[0].get_xlabel() == "Observation"
        assert "inch" in fig.axes[0].get_ylabel()
        plt.close(fig)


class TestPlottedPointData:
    def test_connecting_line_matches_values(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        fig = render.render_chart(m)
        ax = fig.axes[0]
        # The series line is the one whose ydata equals the model values.
        series = None
        for line in ax.get_lines():
            yd = list(line.get_ydata())
            if len(yd) == len(m.values) and np.allclose(yd, m.values):
                series = line
                break
        assert series is not None, "plotted series does not match model values"
        assert list(series.get_xdata()) == m.positions
        plt.close(fig)

    def test_ooc_points_have_distinct_markers(self) -> None:
        # Accessibility: OOC must be distinguished by more than color.
        # Use a single-outlier I-MR series so the chart has BOTH normal points
        # (circles) and a beyond-limit point (square) -> distinct marker groups.
        vals = np.array([10.0, 10.1, 9.9, 10.05, 9.95, 10.0, 10.1, 9.9, 30.0, 10.0, 9.95])
        pair = imr_chart(vals)
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        fig = render.render_chart(m)
        ax = fig.axes[0]
        collections = ax.collections
        # Normal circles + beyond square (+ possibly rule triangle) = >= 2 groups.
        assert len(collections) >= 2
        # The beyond-limit scatter must contain the outlier's position.
        beyond_positions = {p.position for p in m.points if p.beyond_limits}
        assert beyond_positions  # sanity: there is a beyond point
        found = False
        for col in collections:
            offs = col.get_offsets()
            xs = {int(round(x)) for x, _ in offs}
            if beyond_positions.issubset(xs):
                found = True
        assert found
        plt.close(fig)

    def test_rule_annotations_only_on_ooc(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        fig = render.render_chart(m, annotate_rules=True)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # There is at least one rule-ID annotation, and no annotation is empty.
        assert texts
        assert all(t.strip() for t in texts)
        # Number of annotations must not exceed the number of OOC points
        # (normal points are never labeled -> keeps dense charts readable).
        assert len(texts) <= len(m.ooc_points)
        plt.close(fig)

    def test_legend_present(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        ev = evaluate_rules(pair.location, rule_set=RuleSet.NELSON)
        m = build_chart_model(pair.location, ev)
        fig = render.render_chart(m)
        leg = fig.axes[0].get_legend()
        assert leg is not None
        labels = [t.get_text() for t in leg.get_texts()]
        assert any("CL" in l for l in labels)
        assert any("UCL" in l for l in labels)
        plt.close(fig)


class TestPairRender:
    def test_pair_two_axes(self) -> None:
        pair = imr_chart(np.array([10, 12, 11, 14, 13, 10, 30.0, 11, 12, 10]))
        pm = build_pair_model(
            pair,
            location_eval=evaluate_rules(pair.location, rule_set=RuleSet.NELSON),
            dispersion_eval=evaluate_rules(pair.dispersion, rule_set=RuleSet.NELSON),
            unit="mm",
        )
        fig = render.render_pair(pm)
        assert len(fig.axes) == 2
        # Top axis = individuals (N points connecting line), bottom = MR (N-1).
        top, bot = fig.axes
        assert "mm" in top.get_ylabel()
        plt.close(fig)


class TestExport:
    def test_png_file(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        fig = render.render_chart(m)
        d = tempfile.mkdtemp()
        path = render.save_png(fig, os.path.join(d, "c.png"), dpi=200)
        assert os.path.getsize(path) > 1000
        # PNG magic bytes.
        with open(path, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
        plt.close(fig)

    def test_svg_file(self) -> None:
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location)
        fig = render.render_chart(m)
        d = tempfile.mkdtemp()
        path = render.save_svg(fig, os.path.join(d, "c.svg"))
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(256)
        assert "<svg" in head
        plt.close(fig)

    def test_png_bytes(self) -> None:
        pair = imr_chart(np.array([10.0, 11.0, 12.0, 10.5, 11.5]))
        m = build_chart_model(pair.location)
        fig = render.render_chart(m)
        data = render.figure_to_png_bytes(fig)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        plt.close(fig)

    def test_svg_bytes(self) -> None:
        pair = imr_chart(np.array([10.0, 11.0, 12.0, 10.5, 11.5]))
        m = build_chart_model(pair.location)
        fig = render.render_chart(m)
        data = render.figure_to_svg_bytes(fig)
        assert b"<svg" in data[:512]
        plt.close(fig)

    def test_exported_svg_contains_labels_and_limits(self) -> None:
        # Exported chart must retain readable labels/legend/limit text.
        pair = xbar_r_chart(_ooc_subgroups())
        m = build_chart_model(pair.location, unit="mm")
        fig = render.render_chart(m)
        svg = render.figure_to_svg_bytes(fig).decode("utf-8")
        assert "Subgroup" in svg  # x label
        assert "mm" in svg  # unit in y label
        assert "CL" in svg and "UCL" in svg  # legend/limit labels
        plt.close(fig)
