"""Type-1 gauge study (Cg/Cgk) page of the Streamlit application.

Thin presentation layer: numbers from ``app.msa.type1`` and
``app.msa.interpret``, the chart from ``app.charts.type1_render``, exports
from ``app.reporting``. Recomputed on every rerun (milliseconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.charts.type1_render import render_type1_run_chart
from app.msa.interpret import Type1Thresholds, interpret_type1
from app.msa.type1 import Type1Request, type1_study, validate_type1
from app.reporting.bundle import ExportMeta
from app.reporting.msa_pdf import type1_pdf_bytes
from app.reporting.type1_report import result_rows, type1_csv_string, type1_workbook_bytes
from app.services.data_loader import load_table
from app.ui.formatting import NA_DISPLAY, fmt_num, severity_icon

EXAMPLES = {
    "Example: Bosch Booklet 10, Procedure 1 (capable)": (
        _ROOT / "examples" / "type1_bosch.csv", 6.002, 0.060, 0.001,
        "Robert Bosch GmbH, Booklet 10, worked example of Procedure 1 (part W450).",
    ),
    "Example: BPI Consulting (not capable)": (
        _ROOT / "examples" / "type1_spcforexcel.csv", 12.306, 0.05, None,
        "BPI Consulting (SPC for Excel), Type 1 Gage Study example.",
    ),
}
UPLOAD = "📁 Upload a file"

st.title("🎯 Type-1 gauge study")
st.caption(
    "One appraiser measures one reference part of known value many times (50 recommended). "
    "Cg compares the gauge's spread with the tolerance, Cgk adds its bias. Run it before the "
    "Gage R&R study. Checked against the Bosch Booklet 10 and BPI Consulting worked examples."
)

st.header("1 · Readings and reference")
src = st.radio("Data", [*EXAMPLES, UPLOAD], key="t1_source")

raw: list = []
source_name = ""
defaults = (0.0, 1.0, None)
if src in EXAMPLES:
    path, ref0, tol0, res0, note = EXAMPLES[src]
    raw = pd.read_csv(path)["value"].tolist()
    source_name = path.name
    defaults = (ref0, tol0, res0)
    st.caption(f"{note} Source and published results: `examples/DATASETS.md`.")
else:
    up = st.file_uploader("Readings file", type=["csv", "txt", "tsv", "xlsx", "xls"], key="t1_file")
    if up is not None:
        loaded = load_table(up.getvalue(), up.name)
        if not loaded.ok:
            st.error(loaded.error)
        else:
            cols = [str(c) for c in loaded.frame.columns]
            guess = next((i for i, c in enumerate(cols) if c.strip().lower() in ("value", "reading", "measurement", "y")), len(cols) - 1)
            col = st.selectbox("Measurement column", cols, index=guess, key="t1_col")
            raw = loaded.frame[col].tolist()
            source_name = up.name
if not raw:
    st.info("Choose an example or upload a file with one reading per row.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
reference = c1.number_input("Reference value x_m", value=float(defaults[0]), format="%.6g", key=f"t1_ref_{src}")
tolerance = c2.number_input("Tolerance T (USL − LSL)", value=float(defaults[1]), min_value=0.0, format="%.6g", key=f"t1_tol_{src}")
use_res = c3.checkbox("Resolution known", value=defaults[2] is not None, key=f"t1_use_res_{src}")
resolution = c3.number_input("Resolution", value=float(defaults[2] or 0.001), min_value=0.0, format="%.6g",
                             key=f"t1_res_{src}", disabled=not use_res) if use_res else None
unit = c4.text_input("Unit (optional)", value="", key="t1_unit") or None

with st.sidebar:
    st.header("Type-1 settings")
    k_pct = st.number_input("K: % of tolerance for the gauge", min_value=1.0, max_value=100.0, value=20.0, key="t1_k")
    l_sig = st.number_input("L: sigma multiplier", min_value=1.0, max_value=12.0, value=6.0, key="t1_l",
                            help="Bosch and Minitab default: 6 (Cg = 0.2T / 6s).")
    with st.expander("Advanced"):
        index_min = st.number_input("Required Cg / Cgk", min_value=0.5, value=1.33, step=0.01, key="t1_min")
        conf = st.number_input("Confidence level", min_value=0.5, max_value=0.999, value=0.95, step=0.01, key="t1_conf")

v = validate_type1(raw, reference, tolerance, resolution)
for w in v.warnings:
    st.warning(w)
if not v.ok:
    for e in v.errors:
        st.error(e)
    st.stop()

result = type1_study(Type1Request(v.values, float(reference), float(tolerance), float(k_pct), float(l_sig),
                                  float(resolution) if resolution else None, 1.0 - float(conf)))
rep = interpret_type1(result, Type1Thresholds(index_min=float(index_min)))

st.header("2 · Result")
box = st.success if rep.headline.startswith("Gauge CAPABLE") else (st.info if "not assessable" in rep.headline else st.error)
box(f"{severity_icon(rep.max_severity.value)} {rep.headline}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cg", fmt_num(result.cg, 2))
m2.metric("Cgk", fmt_num(result.cgk, 2))
m3.metric("Bias", fmt_num(result.bias, 5), help="Mean − reference value")
m4.metric("Bias p-value", NA_DISPLAY if result.p_value is None
          else ("< 0.0001" if result.p_value < 1e-4 else fmt_num(result.p_value, 4)))
st.dataframe(
    pd.DataFrame([{"Quantity": k, "Value": f"{x:.6g}" if isinstance(x, float) else ("—" if x is None else str(x))}
                  for k, x in result_rows(result)]),
    width="stretch", hide_index=True, height=420,
)

st.header("3 · Run chart")
fig = render_type1_run_chart(result, unit=unit)
st.pyplot(fig, width="stretch")
plt.close(fig)

st.header("4 · Interpretation")
i1, i2 = st.columns(2)
with i1:
    st.subheader("📊 Statistical conclusions")
    for c in rep.conclusions:
        st.markdown(f"{severity_icon(c.severity.value)} {c.text}  \n`{c.code}`")
with i2:
    st.subheader("🔧 Engineering recommendations")
    for r in rep.recommendations:
        st.markdown(f"{severity_icon(r.severity.value)} {r.text}  \n`{r.code}`")

st.header("5 · Export")
meta = ExportMeta(unit=unit, source_file=source_name)
e1, e2, e3 = st.columns(3)
e1.download_button("⬇️ PDF report", data=lambda: type1_pdf_bytes(result, rep, meta, v.warnings),
                   file_name="type1_gauge_study.pdf", mime="application/pdf", width="stretch")
e2.download_button("⬇️ CSV (full precision)", data=type1_csv_string(result, rep, meta, v.warnings).encode("utf-8"),
                   file_name="type1_gauge_study.csv", mime="text/csv", width="stretch")
e3.download_button("⬇️ Excel workbook", data=type1_workbook_bytes(result, rep, meta, v.warnings),
                   file_name="type1_gauge_study.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
