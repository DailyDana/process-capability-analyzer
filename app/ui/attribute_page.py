"""Attribute control charts (p, np, c, u) page of the Streamlit application.

Thin presentation layer over ``app.services.attribute``. Recomputed on every
rerun (milliseconds).
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

from app.charts.attribute_render import render_attribute_chart
from app.interpret.attribute import AttributeThresholds
from app.reporting.attribute_report import attribute_csv_string, sample_rows, summary_rows
from app.reporting.bundle import ExportMeta
from app.services.attribute import AttributeRequest, run_attribute
from app.services.data_loader import load_table
from app.spc.attribute import AttributeKind
from app.spc.rules import RuleSet
from app.ui.formatting import severity_icon

KINDS = {
    "p — fraction nonconforming": AttributeKind.P,
    "np — number nonconforming": AttributeKind.NP,
    "c — nonconformities per sample": AttributeKind.C,
    "u — nonconformities per unit": AttributeKind.U,
}
# name -> (file, count column, size column, kind label, baseline samples, note)
EXAMPLES = {
    "Example: orange juice cans (p chart, 30 trial + 24 new samples)": (
        "attribute_orangejuice.csv", "D", "size", "p — fraction nonconforming", 30,
        "Nonconforming cans in samples of 50 (Montgomery; R package qcc, `orangejuice`).",
    ),
    "Example: circuit boards (c chart, 26 trial + 20 new samples)": (
        "attribute_circuit.csv", "x", "size", "c — nonconformities per sample", 26,
        "Nonconformities in samples of 100 boards (Montgomery; qcc, `circuit`).",
    ),
    "Example: personal computers (u chart)": (
        "attribute_pcmanufact.csv", "x", "size", "u — nonconformities per unit", None,
        "Nonconformities in samples of 5 computers (Montgomery; qcc, `pcmanufact`).",
    ),
    "Example: dyed cloth (u chart, varying sample size)": (
        "attribute_dyedcloth.csv", "x", "size", "u — nonconformities per unit", None,
        "Defects per 50 square metres of cloth, rolls of different size (Montgomery; qcc, `dyedcloth`).",
    ),
}
UPLOAD = "📁 Upload a file"
RULE_SETS = {"Nelson (tests 1-4)": RuleSet.NELSON, "Western Electric (tests 1 and 4)": RuleSet.WECO}

st.title("🔢 Attribute control charts")
st.caption(
    "For counted data: nonconforming items (p, np) or nonconformities (c, u). Limits can come from "
    "a baseline period and be applied to later samples. Checked against the published limits of "
    "four textbook datasets."
)

st.header("1 · Data")
src = st.radio("Data", [*EXAMPLES, UPLOAD], key="at_source")

frame = None
source_name = ""
count_default = size_default = None
kind_default = next(iter(KINDS))
baseline_default = None
if src in EXAMPLES:
    file, count_default, size_default, kind_default, baseline_default, note = EXAMPLES[src]
    frame = pd.read_csv(_ROOT / "examples" / file)
    source_name = file
    st.caption(f"{note} Source and published limits: `examples/DATASETS.md`.")
else:
    up = st.file_uploader("File with one row per sample, in time order",
                          type=["csv", "txt", "tsv", "xlsx", "xls"], key="at_file")
    if up is not None:
        loaded = load_table(up.getvalue(), up.name)
        if not loaded.ok:
            st.error(loaded.error)
        else:
            frame = loaded.frame
            source_name = up.name
if frame is None:
    st.info("Choose an example or upload a file with a count column (and a sample-size column).")
    st.stop()

cols = [str(c) for c in frame.columns]
frame.columns = cols
c1, c2, c3 = st.columns(3)
kind_label = c1.selectbox("Chart", list(KINDS), index=list(KINDS).index(kind_default), key=f"at_kind_{src}")
kind = KINDS[kind_label]
count_col = c2.selectbox("Count column", cols, key=f"at_count_{src}",
                         index=cols.index(count_default) if count_default in cols else 0)
SAME = "(same size for every sample)"
size_options = [*cols, SAME]
size_col = c3.selectbox("Sample-size column", size_options, key=f"at_size_{src}",
                        index=size_options.index(size_default) if size_default in cols else len(cols))
constant_size = None
if size_col == SAME:
    constant_size = c3.number_input("Sample size", min_value=0.0, value=1.0 if kind is AttributeKind.C else 50.0,
                                    format="%.6g", key=f"at_n_{src}")

n_rows = len(frame)
with st.sidebar:
    st.header("Attribute chart settings")
    rule_label = st.radio("Run rules", list(RULE_SETS), key="at_rules",
                          help="Only the tests that do not rely on normal zones are applied, as in Minitab.")
    source = st.radio("Centre line from", ["Baseline samples", "A known standard"], key="at_center_src")
    n_baseline = None
    exclude: tuple[int, ...] = ()
    center = None
    if source == "Baseline samples":
        n_baseline = st.number_input(
            "Baseline: first N samples", min_value=1, max_value=max(n_rows, 1),
            value=int(baseline_default or n_rows), step=1, key=f"at_base_{src}",
            help="The centre line is estimated from these; later samples are judged against it.",
        )
        picked = st.multiselect(
            "Leave out of the baseline (assignable cause found)", list(range(1, int(n_baseline) + 1)),
            key=f"at_excl_{src}",
            help="Still plotted and still flagged, but not used for the centre line.",
        )
        exclude = tuple(int(p) for p in picked)
    else:
        center = st.number_input("Standard value (p, c or u)", min_value=0.0, value=0.1, format="%.6g",
                                 key="at_std")
    with st.expander("Interpretation thresholds"):
        min_samples = st.number_input("Min baseline samples", min_value=2, value=20, step=1, key="at_min_s")
        min_expected = st.number_input("Min expected count per sample", min_value=0.1, value=5.0, step=0.5,
                                       key="at_min_e")

o = run_attribute(AttributeRequest(
    kind=kind,
    counts=tuple(frame[count_col].tolist()),
    sizes=None if size_col == SAME else tuple(frame[size_col].tolist()),
    constant_size=constant_size,
    n_baseline=None if n_baseline is None else int(n_baseline),
    exclude=exclude,
    center=center,
    rule_set=RULE_SETS[rule_label],
    thresholds=AttributeThresholds(int(min_samples), float(min_expected)),
))
for w in o.warnings:
    st.warning(w)
if not o.ok:
    for e in o.errors:
        st.error(e)
    st.stop()
chart, ev, rep = o.chart, o.evaluation, o.interpretation

st.header("2 · Chart")
(st.success if ev.is_stable else st.error)(f"{severity_icon(rep.max_severity.value)} {rep.headline}")
fig = render_attribute_chart(chart, ev)
st.pyplot(fig, width="stretch")
plt.close(fig)
st.caption("Filled: samples the centre line was estimated from. Hollow: judged against it. "
           "× : left out of the baseline. Red: signal.")
left, right = st.columns([1, 2])
left.dataframe(
    pd.DataFrame([{"Quantity": k, "Value": f"{v:.6g}" if isinstance(v, float) else str(v)}
                  for k, v in summary_rows(chart)]),
    width="stretch", hide_index=True,
)
if ev.violations:
    right.dataframe(
        pd.DataFrame([{"Rule": "/".join(v.all_rule_ids), "Samples": " ".join(str(i + 1) for i in v.indices),
                       "What happened": v.explanation} for v in ev.violations]),
        width="stretch", hide_index=True,
    )
else:
    right.info("No rule was triggered.")
with st.expander("Samples and their limits"):
    st.dataframe(pd.DataFrame(sample_rows(chart, ev)), width="stretch", hide_index=True)

st.header("3 · Interpretation")
i1, i2 = st.columns(2)
with i1:
    st.subheader("📊 Statistical conclusions")
    for c in rep.conclusions:
        st.markdown(f"{severity_icon(c.severity.value)} {c.text}  \n`{c.code}`")
with i2:
    st.subheader("🔧 Engineering recommendations")
    for r in rep.recommendations:
        st.markdown(f"{severity_icon(r.severity.value)} {r.text}  \n`{r.code}`")

st.header("4 · Export")
st.download_button(
    "⬇️ CSV (full precision)",
    data=attribute_csv_string(chart, ev, rep, ExportMeta(source_file=source_name), o.warnings).encode("utf-8"),
    file_name=f"{chart.kind.value}_chart.csv", mime="text/csv",
)
