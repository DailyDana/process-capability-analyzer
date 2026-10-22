"""Gage R&R (MSA) page of the Streamlit application.

Thin presentation layer, like the capability page: every number comes from
``app.services.gage_rr``; charts from ``app.charts.gage_*``; exports from
``app.reporting.gage_report``. Nothing is computed here.

The analysis is recomputed on every rerun. It takes milliseconds, and it means
a displayed result can never be stale relative to the widgets.
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

from app.charts.gage_model import build_components_model, build_interaction_model, build_range_model
from app.charts.gage_render import render_components, render_interaction, render_range_by_operator
from app.charts.render import figure_to_png_bytes
from app.msa.anova import AnovaResult, AnovaTable
from app.msa.interpret import GageThresholds, GrrVerdict, headline_verdict
from app.msa.models import GageMethod
from app.reporting.bundle import ExportMeta
from app.reporting.gage_report import gage_csv_string, gage_workbook_bytes
from app.reporting.msa_pdf import gage_rr_pdf_bytes
from app.services.data_loader import load_table
from app.services.gage_rr import columns_to_request, run_gage_study
from app.ui.formatting import NA_DISPLAY, fmt_num, severity_icon, with_unit

EXAMPLE_FILE = _ROOT / "examples" / "gage_rr_aiag.csv"
EXAMPLE_LABEL = "Example: AIAG MSA 4th ed. study (10 parts x 3 appraisers x 3 trials)"
NO_TRIAL = "(none - use row order)"
METHOD_LABELS = {
    "ANOVA (recommended)": GageMethod.ANOVA,
    "Average & Range (AIAG)": GageMethod.AVERAGE_RANGE,
}
K_LABELS = {"6 (AIAG 4th ed.)": 6.0, "5.15 (AIAG 3rd ed.)": 5.15}

# Column-name guesses for an uploaded file (lower-cased, exact match).
_GUESS = {
    "part": ("part", "parts", "part id", "part_id", "sample", "item"),
    "operator": ("operator", "operators", "appraiser", "inspector", "op"),
    "value": ("value", "measurement", "measure", "reading", "result", "y"),
    "trial": ("trial", "trials", "replicate", "repeat", "run", "rep"),
}


def _guess(columns: list[str], role: str, fallback: int) -> int:
    lowered = [str(c).strip().lower() for c in columns]
    for name in _GUESS[role]:
        if name in lowered:
            return lowered.index(name)
    return min(fallback, len(columns) - 1)


st.title("📏 Gage R&R (crossed)")
st.caption(
    "Measurement system analysis for variable data: how much of the observed variation "
    "comes from the gauge (repeatability) and the appraisers (reproducibility)? "
    "Validate the measurement system **before** a capability study. Balanced, crossed "
    "studies only; checked against published AIAG, Minitab and SAS results."
)

# ===========================================================================
# 1. DATA
# ===========================================================================
st.header("1 · Study data")
src = st.radio("Data", [EXAMPLE_LABEL, "📁 Upload a file"], key="gage_source")

frame = None
source_name = ""
if src == EXAMPLE_LABEL:
    frame = pd.read_csv(EXAMPLE_FILE, dtype=str)
    source_name = EXAMPLE_FILE.name
    st.caption(
        "Data as reproduced from the AIAG MSA Reference Manual, 4th ed. "
        "(source and published results: `examples/DATASETS.md`)."
    )
else:
    st.caption(
        "Long format: one row per measurement, with columns for part, operator and value "
        "(a trial column is optional). Every operator must measure every part the same "
        "number of times."
    )
    up = st.file_uploader("Study file", type=["csv", "txt", "tsv", "xlsx", "xls"], key="gage_file")
    if up is not None:
        loaded = load_table(up.getvalue(), up.name)
        if not loaded.ok:
            st.error(loaded.error)
        else:
            frame = loaded.frame
            source_name = up.name

if frame is None:
    st.info("Choose the example or upload a study file to begin.")
    st.stop()

cols = [str(c) for c in frame.columns]
if len(cols) < 3:
    st.error("The file needs at least three columns: part, operator and measurement.")
    st.stop()
c1, c2, c3, c4 = st.columns(4)
part_col = c1.selectbox("Part column", cols, index=_guess(cols, "part", 0), key="gage_part_col")
op_col = c2.selectbox("Operator column", cols, index=_guess(cols, "operator", 1), key="gage_op_col")
val_col = c3.selectbox("Measurement column", cols, index=_guess(cols, "value", len(cols) - 1), key="gage_val_col")
trial_like = [i for i, c in enumerate(cols) if c.strip().lower() in _GUESS["trial"]]
trial_col = c4.selectbox(
    "Trial column (optional)", [NO_TRIAL, *cols],
    index=trial_like[0] + 1 if trial_like else 0, key="gage_trial_col",
)

chosen = [part_col, op_col, val_col] + ([trial_col] if trial_col != NO_TRIAL else [])
if len(set(chosen)) != len(chosen):
    st.error("Each role needs a different column.")
    st.stop()

# ---------------------------------------------------------------------------
# Settings (sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Gage R&R settings")
    method = METHOD_LABELS[st.radio("Method", list(METHOD_LABELS), key="gage_method")]
    use_tol = st.checkbox("Report % of tolerance", value=True, key="gage_use_tol")
    tolerance = None
    if use_tol:
        tolerance = st.number_input(
            "Tolerance (USL − LSL)", min_value=0.0, value=8.0 if src == EXAMPLE_LABEL else 1.0,
            format="%.6g", key="gage_tol",
            help="The example uses 8, as in Minitab's published analysis of this data.",
        )
        if tolerance <= 0:
            tolerance = None
    unit = st.text_input("Unit (optional)", value="", key="gage_unit") or None
    k = K_LABELS[st.selectbox("Study variation multiplier k", list(K_LABELS), key="gage_k",
                              help="Affects Study Var and % Tolerance only, never % Study Var.")]
    with st.expander("Advanced"):
        alpha_pool = st.number_input(
            "Pool interaction when p >", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            key="gage_alpha", help="AIAG/Minitab convention 0.25. Set to 1 to always keep it.",
        )
        acc = st.number_input("Acceptable below % GRR", min_value=0.1, value=10.0, key="gage_acc")
        marg = st.number_input("Unacceptable above % GRR", min_value=0.2, value=30.0, key="gage_marg")
        ndc_min = st.number_input("Minimum ndc", min_value=1, value=5, step=1, key="gage_ndc")
    try:
        thresholds = GageThresholds(acceptable_max=acc, marginal_max=marg, ndc_min=int(ndc_min))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

request = columns_to_request(
    frame[part_col].tolist(),
    frame[op_col].tolist(),
    frame[val_col].tolist(),
    frame[trial_col].tolist() if trial_col != NO_TRIAL else None,
    method=method,
    k=k,
    tolerance=tolerance,
    alpha_pool=float(alpha_pool),
    unit=unit,
    thresholds=thresholds,
)
outcome = run_gage_study(request)

for w in outcome.warnings:
    st.warning(w)
if not outcome.ok:
    for e in outcome.errors:
        st.error(e)
    st.stop()

study, result, rep = outcome.study, outcome.result, outcome.interpretation
st.caption(
    f"**{source_name}** · {study.n_parts} parts × {study.n_operators} operators × "
    f"{study.n_trials} trials = {study.values.size} measurements"
)

# ===========================================================================
# 2. RESULT
# ===========================================================================
st.header("2 · Result")
verdict = headline_verdict(result, thresholds)
box = {GrrVerdict.ACCEPTABLE: st.success, GrrVerdict.MARGINAL: st.warning,
       GrrVerdict.UNACCEPTABLE: st.error}.get(verdict, st.info)
box(f"{severity_icon(rep.max_severity.value)} {rep.headline}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("% GRR of study variation", fmt_num(result.pct_grr_study_var, 2) + ("%" if result.pct_grr_study_var is not None else ""))
m2.metric("% GRR of tolerance", fmt_num(result.pct_grr_tolerance, 2) + ("%" if result.pct_grr_tolerance is not None else ""))
m3.metric("ndc", NA_DISPLAY if result.ndc is None else str(result.ndc))
m4.metric("Ranges above UCL", len(result.range_check.out_of_control),
          help=f"R̄ = {fmt_num(result.range_check.r_bar)}, UCL = {fmt_num(result.range_check.ucl)}")

table = pd.DataFrame(
    [
        {
            "Source": r.source,
            "Variance": fmt_num(r.variance, 5),
            with_unit("StdDev", unit): fmt_num(r.sd, 5),
            f"Study Var ({k:g}·SD)": fmt_num(r.study_var, 5),
            "% Contribution": fmt_num(r.pct_contribution, 2),
            "% Study Var": fmt_num(r.pct_study_var, 2),
            "% Tolerance": fmt_num(r.pct_tolerance, 2),
        }
        for r in result.rows
    ]
)
st.dataframe(table, width="stretch", hide_index=True)
st.caption(
    "AIAG bands apply to **% Study Var** (a ratio of standard deviations). % Contribution "
    "(a ratio of variances) adds to 100% and always looks smaller — do not judge it "
    "against the 10% / 30% bands."
)


def _anova_frame(t: AnovaTable) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Source": r.source, "DF": r.df, "SS": fmt_num(r.ss, 5), "MS": fmt_num(r.ms, 5),
             "F": fmt_num(r.f, 3), "P": fmt_num(r.p, 3)}
            for r in t.rows
        ]
    )


if isinstance(result.anova, AnovaResult):
    a = result.anova
    with st.expander("ANOVA tables", expanded=False):
        st.markdown("**Full model (with operator × part interaction)**")
        st.dataframe(_anova_frame(a.full), width="stretch", hide_index=True)
        if a.reduced is not None:
            st.markdown(
                f"**Interaction pooled into repeatability** (p = {fmt_num(a.interaction_p, 3)} > "
                f"{a.alpha_pool:g})"
            )
            st.dataframe(_anova_frame(a.reduced), width="stretch", hide_index=True)
elif result.average_range is not None:
    d = result.average_range
    with st.expander("Average & Range intermediates", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Quantity": "R̄̄ (average range)", "Value": fmt_num(d.r_bar_bar, 5)},
                    {"Quantity": "X̄ diff (operator averages)", "Value": fmt_num(d.x_bar_diff, 5)},
                    {"Quantity": "Rp (part averages)", "Value": fmt_num(d.r_p, 5)},
                    {"Quantity": "K1 / K2 / K3", "Value": f"{d.k1} / {d.k2} / {d.k3}"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("The Average & Range method cannot separate the operator × part interaction.")

# ===========================================================================
# 3. CHARTS
# ===========================================================================
st.header("3 · Charts")
figs = {
    "components": render_components(build_components_model(result, thresholds.acceptable_max, thresholds.marginal_max)),
    "range": render_range_by_operator(build_range_model(study, result)),
    "interaction": render_interaction(build_interaction_model(study)),
}
for fig in figs.values():
    st.pyplot(fig, width="stretch")

# ===========================================================================
# 4. INTERPRETATION
# ===========================================================================
st.header("4 · Interpretation")
ic1, ic2 = st.columns(2)
with ic1:
    st.subheader("📊 Statistical conclusions")
    for c in rep.conclusions:
        st.markdown(f"{severity_icon(c.severity.value)} {c.text}  \n`{c.code}`")
with ic2:
    st.subheader("🔧 Engineering recommendations")
    if not rep.recommendations:
        st.markdown("No action indicated.")
    for r in rep.recommendations:
        st.markdown(f"{severity_icon(r.severity.value)} {r.text}  \n`{r.code}`")

# ===========================================================================
# 5. EXPORT
# ===========================================================================
st.header("5 · Export")
meta = ExportMeta(unit=unit, source_file=source_name)
e0, e1, e2, e3 = st.columns(4)
e0.download_button(
    "⬇️ PDF report",
    # A callable defers the ~1 s build until the button is clicked.
    data=lambda: gage_rr_pdf_bytes(
        study, result, rep, meta, outcome.warnings, thresholds.acceptable_max, thresholds.marginal_max
    ),
    file_name="gage_rr_report.pdf", mime="application/pdf", width="stretch",
)
e1.download_button(
    "⬇️ CSV (full precision)",
    data=gage_csv_string(study, result, rep, meta, outcome.warnings).encode("utf-8"),
    file_name="gage_rr.csv", mime="text/csv", width="stretch",
)
e2.download_button(
    "⬇️ Excel workbook",
    data=gage_workbook_bytes(study, result, rep, meta, outcome.warnings),
    file_name="gage_rr.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)
e3.download_button(
    "⬇️ Components chart (PNG)",
    data=figure_to_png_bytes(figs["components"]),
    file_name="gage_rr_components.png", mime="image/png", width="stretch",
)
for fig in figs.values():
    plt.close(fig)
