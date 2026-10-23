"""CSV and PDF export of a non-normal capability analysis (serialization only)."""

from __future__ import annotations

import csv
import io

import matplotlib.pyplot as plt
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from app.capability.nonnormal import Distribution, fit_distribution
from app.charts.nonnormal_render import render_histogram, render_probability_plot
from app.reporting.bundle import ExportMeta
from app.reporting.csv_report import _num as _csv_num
from app.reporting.csv_report import _s as _csv_s
from app.reporting.msa_pdf import _H1, _SMALL, _cover, _grid, _interpretation, _kv, _pe, build_document
from app.reporting.pdf_report import _chart_image, _num
from app.services.nonnormal import NonNormalOutcome

__all__ = ["summary_rows", "fit_rows", "nonnormal_csv_string", "nonnormal_pdf_bytes"]


def summary_rows(o: NonNormalOutcome) -> list[tuple[str, object]]:
    s, spec = o.summary, o.spec
    assert s is not None and spec is not None
    rows: list[tuple[str, object]] = [
        ("Method", s.method_label),
        ("Observations (n)", s.n),
        ("USL", spec.usl),
        ("LSL" + (" (treated as lower boundary)" if spec.lsl_is_boundary else ""), spec.lsl),
        ("Pp", s.pp),
        ("Ppu", s.ppu),
        ("Ppl", s.ppl),
        ("Ppk", s.ppk),
        ("Expected PPM below LSL", s.ppm_below),
        ("Expected PPM above USL", s.ppm_above),
        ("Expected PPM total", s.ppm_total),
        ("Observed PPM", s.observed_ppm),
        ("Ppk assuming normality", s.normal_ppk),
        ("PPM assuming normality", s.normal_ppm),
        ("Anderson-Darling A^2 of the model", s.fit_statistic),
    ]
    if o.percentile is not None:
        rows += [("X 0.135 %", o.percentile.x_low), ("X 50 %", o.percentile.x_mid), ("X 99.865 %", o.percentile.x_high)]
    if o.boxcox is not None:
        b = o.boxcox
        rows += [("Lambda (MLE)", b.lambda_mle), ("Lambda used", b.lambda_used), ("Transformed mean", b.mean_t),
                 ("Transformed sigma", b.sd_t), ("Transformed USL", b.usl_t), ("Transformed LSL", b.lsl_t)]
    return rows


def fit_rows(o: NonNormalOutcome) -> list[list[object]]:
    return [[f.distribution.value, "; ".join(f"{k} = {v:.6g}" for k, v in f.params.items()), f.ad_statistic]
            for f in o.fits]


def _v(v: object) -> str:
    return _csv_num(v) if isinstance(v, float) else _csv_s(v)


def nonnormal_csv_string(o: NonNormalOutcome, meta: ExportMeta = ExportMeta()) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["# Metadata"])
    w.writerow(["Field", "Value"])
    for k, v in meta.as_rows():
        w.writerow([k, _csv_s(v)])
    w.writerow([])
    w.writerow(["# Non-normal capability"])
    w.writerow(["Field", "Value"])
    for k, v in summary_rows(o):
        w.writerow([k, _v(v)])
    w.writerow([])
    w.writerow(["# Distribution fits (ranked by A^2)"])
    w.writerow(["Distribution", "Parameters", "A^2"])
    for d, p, a in fit_rows(o):
        w.writerow([d, p, _v(a)])
    for d, why in o.skipped.items():
        w.writerow([d.value, f"not fitted: {why}", ""])
    if o.warnings:
        w.writerow([])
        w.writerow(["# Data warnings"])
        for m in o.warnings:
            w.writerow([m])
    if o.interpretation is not None:
        w.writerow([])
        w.writerow(["# Interpretation"])
        w.writerow(["Headline", o.interpretation.headline])
        w.writerow(["Kind", "Code", "Severity", "Text"])
        for c in o.interpretation.conclusions:
            w.writerow(["conclusion", c.code, c.severity.value, c.text])
        for r in o.interpretation.recommendations:
            w.writerow(["recommendation", r.code, r.severity.value, r.text])
    w.writerow([])
    w.writerow(["# Measurements"])
    w.writerow(["#", "Value"])
    for i, v in enumerate(o.values, start=1):
        w.writerow([i, _csv_num(v)])
    return buf.getvalue()


def nonnormal_pdf_bytes(o: NonNormalOutcome, meta: ExportMeta = ExportMeta()) -> bytes:
    s = o.summary
    assert s is not None and o.spec is not None and o.interpretation is not None
    title = "Process Capability Report (non-normal data)"
    story: list = []
    key = [("Method", s.method_label), ("Observations", s.n), ("Ppk", _num(s.ppk, 2)),
           ("Expected PPM", _num(s.ppm_total, 0)), ("Ppk assuming normality", _num(s.normal_ppk, 2))]
    _cover(story, title, meta, key, o.interpretation, None)
    for w in o.warnings:
        story.append(Spacer(1, 1 * mm))
        story.append(_pe(f"Data warning: {w}", _SMALL))
    story.append(PageBreak())
    story.append(_pe("1. Results", _H1))
    story.append(_kv(summary_rows(o)))
    story.append(Spacer(1, 2 * mm))
    story.append(_pe(
        "Percentile method (ISO 22514-2): Ppk = min((USL - X50)/(X99.865 - X50), (X50 - LSL)/(X50 - X0.135)) "
        "on the fitted distribution; Box-Cox: normal-theory indices on the transformed scale. Formulas and "
        "conventions: docs/FORMULAS.md section 8.", _SMALL))
    story.append(_pe("2. Distribution fits", _H1))
    story.append(_grid(["Distribution", "Parameters", "A^2"],
                       [[d, p, _num(a, 3)] for d, p, a in fit_rows(o)], [30, 110, 30]))
    story.append(PageBreak())
    story.append(_pe("3. Charts", _H1))
    normal = fit_distribution(__import__("numpy").asarray(o.values), Distribution.NORMAL)
    figs = [render_histogram(o.values, o.spec, o.chosen_fit, o.boxcox, s.method_label, normal, unit=meta.unit),
            render_probability_plot(o.values, o.chosen_fit, o.boxcox, s.method_label)]
    for f in figs:
        story.append(_chart_image(f, max_width_mm=170.0 if f is figs[0] else 120.0))
        story.append(Spacer(1, 3 * mm))
        plt.close(f)
    story.append(PageBreak())
    _interpretation(story, o.interpretation, first_no=4)
    return build_document(story, title)
