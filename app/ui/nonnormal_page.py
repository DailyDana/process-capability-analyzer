"""Non-normal capability page of the Streamlit application.

Thin presentation layer over ``app.services.nonnormal``; charts from
``app.charts.nonnormal_render``; exports from ``app.reporting.nonnormal_report``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.capability.nonnormal import Distribution, fit_distribution
from app.charts.nonnormal_render import render_histogram, render_probability_plot
from app.interpret.nonnormal import NonNormalThresholds
from app.reporting.bundle import ExportMeta
from app.reporting.nonnormal_report import fit_rows, nonnormal_csv_string, nonnormal_pdf_bytes, summary_rows
from app.services.data_loader import load_table
from app.services.nonnormal import BEST_FIT, BOX_COX, NonNormalRequest, run_nonnormal
from app.ui.formatting import fmt_num, severity_icon

EXAMPLE = _ROOT / "examples" / "nonnormal_cycle_time.csv"
EXAMPLE_LABEL = "Example: cycle time, 207 orders (iSixSigma)"
UPLOAD = "📁 Upload a file"
METHODS = {
    "Best-fitting distribution (percentile method)": BEST_FIT,
    "Box-Cox transformation": BOX_COX,
    **{f"{d.value.capitalize()} (percentile method)": d.value for d in Distribution},
}

st.title("📈 Capability for non-normal data")
st.caption(
    "Flatness, runout, roughness, cycle times and many other characteristics are skewed, and a normal-theory "
    "Ppk can be badly wrong for them. This page fits candidate distributions and computes Pp/Ppk from their "
    "percentiles (ISO 22514-2), or uses a Box-Cox transformation, and shows what normal theory would have said. "
    "Checked against a published cycle-time study."
)

st.header("1 · Data and specification")
src = st.radio("Data", [EXAMPLE_LABEL, UPLOAD], key="nn_source")
raw: list = []
source_name = ""
if src == EXAMPLE_LABEL:
    raw = pd.read_csv(EXAMPLE)["cycle_time_days"].tolist()
    source_name = EXAMPLE.name
    st.caption("Order cycle times in days; upper limit 30, natural lower bound 0. Source: `examples/DATASETS.md`.")
else:
    up = st.file_uploader("Measurements", type=["csv", "txt", "tsv", "xlsx", "xls"], key="nn_file")
    if up is not None:
        loaded = load_table(up.getvalue(), up.name)
        if not loaded.ok:
            st.error(loaded.error)
        else:
            cols = [str(c) for c in loaded.frame.columns]
            col = st.selectbox("Measurement column", cols, index=len(cols) - 1, key="nn_col")
            raw = loaded.frame[col].tolist()
            source_name = up.name
if not raw:
    st.info("Choose the example or upload a file with one measurement per row.")
    st.stop()

is_ex = src == EXAMPLE_LABEL
c1, c2, c3, c4 = st.columns(4)
use_usl = c1.checkbox("USL", value=True, key=f"nn_use_usl_{src}")
usl = c1.number_input("USL value", value=30.0 if is_ex else 1.0, format="%.6g", key=f"nn_usl_{src}",
                      label_visibility="collapsed") if use_usl else None
use_lsl = c2.checkbox("LSL", value=is_ex, key=f"nn_use_lsl_{src}")
lsl = c2.number_input("LSL value", value=0.0, format="%.6g", key=f"nn_lsl_{src}",
                      label_visibility="collapsed") if use_lsl else None
boundary = c3.checkbox("LSL is a natural lower bound", value=is_ex, key=f"nn_bound_{src}", disabled=lsl is None,
                       help="For characteristics that cannot go below a value (e.g. 0 for flatness): only the "
                            "upper side is assessed.") if lsl is not None else False
unit = c4.text_input("Unit (optional)", value="days" if is_ex else "", key=f"nn_unit_{src}") or None

with st.sidebar:
    st.header("Non-normal settings")
    method_label = st.radio("Method", list(METHODS), key="nn_method")
    round_lambda = st.checkbox("Round lambda to 0.5", value=True, key="nn_round",
                               help="Box-Cox only. Common software rounds the MLE lambda to the nearest 0.5.")
    unbias = st.checkbox("Unbias sigma with c4 (Minitab default)", value=False, key="nn_c4",
                         help="Applies to the normal model and Box-Cox. The rest of this app uses s, as AIAG does.")
    with st.expander("Interpretation thresholds"):
        cap_min = st.number_input("Capable Ppk", min_value=0.5, value=1.33, step=0.01, key="nn_cap")
        poor_ad = st.number_input("Poor fit above A^2", min_value=0.1, value=1.0, step=0.1, key="nn_ad")

try:
    thresholds = NonNormalThresholds(capable_min=float(cap_min), poor_fit_ad=float(poor_ad))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

o = run_nonnormal(NonNormalRequest(tuple(raw), usl, lsl, bool(boundary), METHODS[method_label], bool(unbias),
                                   bool(round_lambda), thresholds))
for w in o.warnings:
    st.warning(w)
if not o.ok:
    for e in o.errors:
        st.error(e)
    st.stop()
s, rep = o.summary, o.interpretation

st.header("2 · Result")
(st.success if s.ppk >= thresholds.capable_min else st.error)(f"{severity_icon(rep.max_severity.value)} {rep.headline}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ppk", fmt_num(s.ppk, 2))
m2.metric("Expected PPM", fmt_num(s.ppm_total, 0))
m3.metric("Observed PPM", fmt_num(s.observed_ppm, 0))
m4.metric("Ppk if assumed normal", fmt_num(s.normal_ppk, 2), delta=fmt_num(s.normal_ppk - s.ppk, 2),
          delta_color="off")
st.dataframe(pd.DataFrame([{"Quantity": k, "Value": f"{v:.6g}" if isinstance(v, float) else ("—" if v is None else str(v))}
                             for k, v in summary_rows(o)]), width="stretch", hide_index=True, height=420)
st.markdown("**Distribution fits** (lower A² = better)")
st.dataframe(pd.DataFrame([{"Distribution": d, "Parameters": p, "A²": f"{a:.3f}"} for d, p, a in fit_rows(o)]),
                width="stretch", hide_index=True)
for d, why in o.skipped.items():
    st.caption(f"{d.value}: {why}")

st.header("3 · Charts")
normal = fit_distribution(np.asarray(o.values), Distribution.NORMAL, bool(unbias))
h = render_histogram(o.values, o.spec, o.chosen_fit, o.boxcox, s.method_label, normal, unit=unit)
st.pyplot(h, width="stretch")
plt.close(h)
pc, _ = st.columns([3, 2])
pp = render_probability_plot(o.values, o.chosen_fit, o.boxcox, s.method_label)
pc.pyplot(pp, width="stretch")
plt.close(pp)

st.header("4 · Interpretation")
i1, i2 = st.columns(2)
with i1:
    st.subheader("📊 Statistical conclusions")
    for c in rep.conclusions:
        st.markdown(f"{severity_icon(c.severity.value)} {c.text}  \n`{c.code}`")
with i2:
    st.subheader("🔧 Engineering recommendations")
    if not rep.recommendations:
        st.markdown("No action indicated.")
    for r in rep.recommendations:
        st.markdown(f"{severity_icon(r.severity.value)} {r.text}  \n`{r.code}`")

st.header("5 · Export")
meta = ExportMeta(unit=unit, source_file=source_name)
e1, e2 = st.columns(2)
e1.download_button("⬇️ PDF report", data=lambda: nonnormal_pdf_bytes(o, meta), file_name="nonnormal_capability.pdf",
                   mime="application/pdf", width="stretch")
e2.download_button("⬇️ CSV (full precision)", data=nonnormal_csv_string(o, meta).encode("utf-8"),
                   file_name="nonnormal_capability.csv", mime="text/csv", width="stretch")
