"""SPC / Process Capability Analyzer — Streamlit application.

Run with:  streamlit run app/ui/streamlit_app.py

This module is a THIN presentation layer. Every number it displays comes from
the validated domain/service layers:

    app.services.data_loader  — file parsing + M1 cleaning
    app.services.analysis     — the full sequence-of-analysis orchestrator
    app.charts.render         — M4 chart rendering
    app.reporting.*           — M5 CSV/Excel export

No statistics, SPC rules, interpretation, or report construction happen here.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Allow `streamlit run app/ui/streamlit_app.py` from the project root.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.charts import render as chart_render
from app.interpret.thresholds import InterpretationThresholds
from app.reporting import csv_report, excel_report, pdf_report
from app.reporting.bundle import AnalysisBundle, ExportMeta, MethodologyInfo
from app.services.analysis import (
    AnalysisRequest,
    ChartKind,
    make_spec,
    run_analysis,
)
from app.services.data_loader import clean_column, load_table
from app.spc.rules import RuleSet
from app.statistics.variation import WithinEstimator
from app.ui.formatting import NA_DISPLAY, fmt_index, fmt_num, severity_icon, with_unit

st.set_page_config(
    page_title="SPC / Process Capability Analyzer",
    page_icon="📐",
    layout="wide",
)

CHART_LABELS = {
    "X̄–R  (subgroups, n ≤ 8)": ChartKind.XBAR_R,
    "X̄–S  (subgroups, larger n)": ChartKind.XBAR_S,
    "I–MR  (individuals, n = 1)": ChartKind.IMR,
}
RULESET_LABELS = {
    "Nelson (8 rules)": RuleSet.NELSON,
    "Western Electric (4 rules)": RuleSet.WECO,
    "Both (distinct IDs)": RuleSet.ALL,
}
ESTIMATOR_LABELS = {
    "Auto (R̄/d₂, S̄/c₄ for n ≥ 9)": WithinEstimator.AUTO,
    "R̄ / d₂ (average range)": WithinEstimator.RBAR_D2,
    "S̄ / c₄ (average std dev)": WithinEstimator.SBAR_C4,
    "Pooled standard deviation": WithinEstimator.POOLED,
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("raw_values", None)      # list of raw tokens (untouched)
    ss.setdefault("source_name", "")
    ss.setdefault("cleaning", None)        # DetailedCleaning
    ss.setdefault("analysis_version", 0)   # bumped on every recompute
    ss.setdefault("last_signature", None)  # detects stale results
    ss.setdefault("outcome", None)
    # Data-source ownership (C3): which input is authoritative, and the identity
    # of the last file load we actually accepted (so a file lingering in the
    # uploader widget cannot silently reprocess on an unrelated rerun).
    ss.setdefault("active_source", None)    # "file" | "manual" | None
    ss.setdefault("last_load_id", None)     # (name, size, column)


def _signature(values, chart_kind, subgroup_size, spec, rule_set, estimator, unit,
               time_ordered, interp_thresholds, original_indices=None):
    """Identity of the inputs an analysis depends on (detects staleness).

    MUST cover every ``AnalysisRequest`` field that can change
    ``AnalysisOutcome``; ``tests/test_ui_services.py`` asserts this
    correspondence field-by-field so a future request field cannot silently go
    stale (C2).
    """
    spec_key = (
        None if spec is None
        else (spec.usl, spec.lsl, spec.nominal, spec.unit)
    )
    return (
        hash(tuple(values)),
        chart_kind.value,
        int(subgroup_size),
        spec_key,
        rule_set.value,
        estimator.value,
        unit,
        bool(time_ordered),
        # astuple(), not a hand-listed subset: every InterpretationThresholds
        # field is user-adjustable, so listing three of them by name is exactly
        # how the original staleness bug arose. A field added to the dataclass
        # is covered here automatically.
        dataclasses.astuple(interp_thresholds),
        tuple(original_indices) if original_indices is not None else None,
    )


_init_state()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📐 SPC / Process Capability Analyzer")
st.caption(
    "Statistical Process Control and process-capability analysis for "
    "manufacturing engineering. Stability is assessed **before** capability; "
    "Cp/Cpk (within-subgroup) and Pp/Ppk (overall) are never interchanged."
)

# ===========================================================================
# 1. DATA INPUT
# ===========================================================================
st.header("1 · Data Input")

# Data-source ownership (C3)
# --------------------------
# Streamlit reruns this script top-to-bottom on EVERY widget interaction, and
# `st.file_uploader` keeps returning the previously uploaded file. Writing to
# session state unconditionally from the upload block therefore silently
# clobbered manual entry whenever any unrelated widget changed. Ownership is now
# explicit and sticky:
#   * an explicit radio selects which source is authoritative;
#   * the file block only writes when File is selected AND the file identity
#     (name, size) actually changed since the last accepted load;
#   * manual entry only writes on an explicit button press.
# Switching source is an explicit act and may discard the other source's data;
# an unrelated rerun never can.
source = st.radio(
    "Data source",
    ["📁 File", "✍️ Manual entry"],
    key="data_source",
    horizontal=True,
    help=(
        "Which input owns the analysis. The inactive source is left untouched "
        "and will never silently replace the active one."
    ),
)
is_file_source = source.endswith("File")

if is_file_source:
    uploaded = st.file_uploader(
        "Measurement file",
        type=["csv", "txt", "tsv", "xlsx", "xls"],
        help="One column must contain the measured values.",
    )
    if uploaded is not None:
        result = load_table(uploaded.getvalue(), uploaded.name)
        if not result.ok:
            st.error(result.error)
        else:
            st.success(f"Loaded **{result.n_rows}** rows from `{uploaded.name}`.")
            col = st.selectbox(
                "Measurement column", result.columns,
                help="Select the column holding the measured values.",
            )
            # Identity of THIS accepted load. Re-processing only happens when
            # something the user actually changed differs from last time; a file
            # merely sitting in the widget across reruns is not a change.
            load_id = (uploaded.name, uploaded.size, col)
            # Two independent conditions must BOTH hold before we overwrite the
            # active dataset:
            #   1. the file identity changed since the last accepted load, and
            #   2. we are not about to silently steal ownership from manual
            #      entry -- the user must have explicitly selected File as the
            #      source (which this branch guarantees) AND either nothing owns
            #      the data yet or the file already owns it.
            # Condition 2 is what makes a lingering upload harmless: re-running
            # the script cannot hand ownership back to the file on its own.
            identity_changed = load_id != st.session_state.get("last_load_id")
            may_take_ownership = st.session_state.get("active_source") in (None, "file")
            if col is not None and identity_changed and may_take_ownership:
                st.session_state.raw_values = list(result.frame[col].tolist())
                st.session_state.source_name = uploaded.name
                st.session_state.cleaning = clean_column(st.session_state.raw_values)
                st.session_state.last_load_id = load_id
                st.session_state.active_source = "file"
            elif col is not None and identity_changed and not may_take_ownership:
                # A genuinely new file while manual data is active: offer the
                # switch explicitly instead of performing it behind the user.
                st.warning(
                    f"`{uploaded.name}` is loaded but **not** active — your "
                    "manually entered data is still being analysed. Click below "
                    "to switch."
                )
                if st.button(f"Use `{uploaded.name}` instead", type="secondary"):
                    st.session_state.raw_values = list(result.frame[col].tolist())
                    st.session_state.source_name = uploaded.name
                    st.session_state.cleaning = clean_column(
                        st.session_state.raw_values
                    )
                    st.session_state.last_load_id = load_id
                    st.session_state.active_source = "file"
                    st.rerun()
else:
    st.caption("Enter or paste measurements. Blank rows are ignored.")
    seed = pd.DataFrame({"measurement": [None] * 10})
    edited = st.data_editor(
        seed, num_rows="dynamic", use_container_width=True, key="manual_editor",
        column_config={
            "measurement": st.column_config.NumberColumn(
                "Measurement", format="%.6f", help="Measured value."
            )
        },
    )
    if st.button("Use manual data", type="secondary"):
        vals = [v for v in edited["measurement"].tolist()]
        if not any(v is not None and str(v).strip() != "" for v in vals):
            st.warning("No measurements were entered.")
        else:
            st.session_state.raw_values = vals
            st.session_state.source_name = "manual entry"
            st.session_state.cleaning = clean_column(vals)
            st.session_state.active_source = "manual"
            # Clear the accepted-file identity so returning to the File tab
            # deliberately re-loads that file rather than being a no-op.
            st.session_state.last_load_id = None
            st.success(f"Accepted {len(vals)} manually entered rows.")

cleaning = st.session_state.cleaning
if cleaning is None:
    st.info("Upload a file or enter data manually to begin.")
    st.stop()

# Always show which dataset the numbers below actually come from.
_active = st.session_state.get("active_source")
st.caption(
    f"**Active data source:** {st.session_state.get('source_name') or 'unknown'}"
    + (" *(manual entry)*" if _active == "manual" else "")
)
if _active == "manual" and is_file_source:
    st.info(
        "The analysis below still uses your **manually entered** data. Upload a "
        "file (or re-select a column) to switch the active source to a file."
    )
elif _active == "file" and not is_file_source:
    st.info(
        f"The analysis below still uses the uploaded file "
        f"**{st.session_state.get('source_name')}**. Click *Use manual data* to "
        "switch the active source to your manual entries."
    )

rep = cleaning.report
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows loaded", rep.n_input)
c2.metric("Valid measurements", rep.n_valid)
c3.metric("Missing", rep.dropped_missing)
c4.metric("Non-numeric", rep.dropped_non_numeric)
c5.metric("Non-finite", rep.dropped_non_finite)
c6.metric("Duplicate values", rep.n_duplicates,
          help="Repeated values are legitimate and are RETAINED; shown for awareness.")

if rep.n_dropped:
    st.warning(
        f"{rep.n_dropped} row(s) were excluded from the analysis. "
        "Your original data is never modified — see Data Preview for details."
    )
if rep.invalid_examples:
    st.caption("Examples of unparseable entries: " + ", ".join(rep.invalid_examples))

if rep.n_valid < 2:
    st.error(
        f"Only {rep.n_valid} valid measurement(s) available. At least 2 are "
        "required for any analysis."
    )
    st.stop()

# ===========================================================================
# 2. SPECIFICATIONS & METADATA
# ===========================================================================
st.header("2 · Specifications & Metadata")

sc1, sc2 = st.columns([2, 3])
with sc1:
    st.subheader("Specification limits")
    use_usl = st.checkbox("Upper specification limit (USL)", value=True)
    usl = st.number_input("USL", value=0.0, format="%.6f",
                          disabled=not use_usl, key="usl_in") if use_usl else None
    use_lsl = st.checkbox("Lower specification limit (LSL)", value=True)
    lsl = st.number_input("LSL", value=0.0, format="%.6f",
                          disabled=not use_lsl, key="lsl_in") if use_lsl else None
    use_nom = st.checkbox("Nominal / target", value=False)
    nominal = st.number_input("Nominal", value=0.0, format="%.6f",
                              disabled=not use_nom, key="nom_in") if use_nom else None

with sc2:
    st.subheader("Characteristic & metadata")
    m1, m2 = st.columns(2)
    characteristic = m1.text_input("Characteristic name", value="")
    unit = m2.text_input("Measurement unit", value="",
                         help="Free text (mm, inch, °C, …). Never interpreted numerically.")
    m3, m4 = st.columns(2)
    part_id = m3.text_input("Part ID", value="")
    machine = m4.text_input("Machine", value="")
    m5, m6 = st.columns(2)
    operator = m5.text_input("Operator", value="")
    timestamp = m6.text_input("Date / time", value="")
    description = st.text_area("Description / notes", value="", height=68)

spec, spec_errors, spec_warnings = make_spec(
    usl if use_usl else None,
    lsl if use_lsl else None,
    nominal if use_nom else None,
    unit or None,
)
for e in spec_errors:
    st.error(f"Specification error: {e}")
for w in spec_warnings:
    st.warning(f"Specification note: {w}")

if spec is None and not spec_errors:
    st.info(
        "**No specification** entered. SPC charts and stability analysis will "
        "run; capability indices require at least one specification limit."
    )
elif spec is not None:
    kind = (
        "Two-sided (USL and LSL)" if spec.is_two_sided
        else ("Upper only (USL)" if spec.usl is not None else "Lower only (LSL)")
    )
    st.success(
        f"**{kind}** specification accepted"
        + (f" · tolerance width {fmt_num(spec.tolerance_width)} {unit}"
           if spec.is_two_sided else "")
    )

# ===========================================================================
# Analysis configuration (drives sections 5-7)
# ===========================================================================
st.header("3 · Data Preview")

pv1, pv2 = st.columns([3, 2])
with pv1:
    rows = [
        {
            "Original row": r.original_index,
            "Raw value": r.raw,
            "Status": r.status.value,
            "Used in analysis": "Yes" if r.is_used else "No",
            "Reason excluded": r.reason or "",
        }
        for r in cleaning.records
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
with pv2:
    st.metric("Observations analysed", len(cleaning.used_indices))
    if cleaning.excluded:
        st.warning(f"{len(cleaning.excluded)} row(s) excluded:")
        for r in cleaning.excluded[:10]:
            st.caption(f"• Row {r.original_index}: {r.reason}")
        if len(cleaning.excluded) > 10:
            st.caption(f"… and {len(cleaning.excluded) - 10} more.")
    else:
        st.success("All rows are valid and included in the analysis.")

values = cleaning.values

# --- Sidebar: analysis configuration ---
with st.sidebar:
    st.header("Analysis configuration")
    chart_label = st.selectbox("Control chart", list(CHART_LABELS.keys()))
    chart_kind = CHART_LABELS[chart_label]
    subgroup_size = 1
    if chart_kind is not ChartKind.IMR:
        subgroup_size = st.number_input(
            "Subgroup size (n)", min_value=2, max_value=25, value=5, step=1,
            help="Measurements per rational subgroup. Must be balanced.",
        )
        st.caption(
            f"{len(values)} measurements → "
            f"{len(values) // int(subgroup_size)} subgroup(s) of {int(subgroup_size)}"
        )
    ruleset_label = st.selectbox("SPC rule set", list(RULESET_LABELS.keys()))
    rule_set = RULESET_LABELS[ruleset_label]
    est_label = st.selectbox("Within-σ estimator", list(ESTIMATOR_LABELS.keys()))
    estimator = ESTIMATOR_LABELS[est_label]

    st.divider()
    time_ordered = st.checkbox(
        "Measurements are in production/time order",
        value=True,
        help=(
            "Required for within-subgroup capability. SPC charts and Cp/Cpk "
            "assume rational subgrouping, which only exists when the data are "
            "in the order produced. Uncheck for an unordered sample (e.g. rows "
            "sorted by value, or a pooled batch of parts): Cp/Cpk and the "
            "stability verdict will then be withheld and only Pp/Ppk reported."
        ),
    )
    if not time_ordered:
        st.warning(
            "Unordered sample: Cp/Cpk and stability are not assessed.",
            icon="⚠️",
        )
    with st.expander("Interpretation thresholds"):
        st.caption(
            "Engineering interpretation conventions — not universal statistical "
            "laws. Adjust to your customer standard. These change the WORDING "
            "and severity of section 7 only; no statistic is recomputed."
        )

        st.markdown("**Capability bands**")
        cap_min = st.number_input("Capable ≥", value=1.33, step=0.01, format="%.2f")
        exc_min = st.number_input("Excellent ≥", value=1.67, step=0.01, format="%.2f")
        marg_min = st.number_input("Marginal ≥", value=1.00, step=0.01, format="%.2f")
        st.caption("Must increase strictly: Marginal < Capable < Excellent.")

        st.markdown("**Gap triggers**")
        centering_gap = st.number_input(
            "Centering gap ratio", value=0.10, step=0.01, format="%.2f",
            min_value=0.0, max_value=0.99,
            help=(
                "Flag a centering problem when Cpk falls this far below Cp, as "
                "a proportion of Cp. 0.10 = 10%."
            ),
        )
        drift_gap = st.number_input(
            "Drift gap ratio", value=0.10, step=0.01, format="%.2f",
            min_value=0.0, max_value=0.99,
            help=(
                "Flag long-term drift when Ppk falls this far below Cpk, as a "
                "proportion of Cpk. 0.10 = 10%."
            ),
        )
        tight_sigma = st.number_input(
            "Tight-spec sigma multiple", value=6.0, step=0.5, format="%.1f",
            min_value=0.1,
            help=(
                "Call the specification tight when its width is narrower than "
                "this many within-sigma. 6.0 = the natural ±3σ process band."
            ),
        )

        st.markdown("**Small-sample language**")
        min_obs = st.number_input(
            "Min observations", value=30, step=1, min_value=1,
            help="Below this count the interpreter qualifies its conclusions.",
        )
        min_sg = st.number_input(
            "Min subgroups", value=20, step=1, min_value=1,
            help="Below this count the control limits are called provisional.",
        )
    decimals = st.slider("Display decimals", 2, 8, 4)

# The dataclass validates its own invariants (ordering, ranges). A typo in the
# sidebar is ordinary user input, not a bug, so it must not take the page down:
# report it and fall back to the documented defaults, saying so explicitly.
try:
    interp_thresholds = InterpretationThresholds(
        marginal_min=marg_min,
        capable_min=cap_min,
        excellent_min=exc_min,
        centering_gap_ratio=centering_gap,
        drift_gap_ratio=drift_gap,
        spec_margin_tight_sigma=tight_sigma,
        min_observations=int(min_obs),
        min_subgroups=int(min_sg),
    )
except ValueError as exc:
    interp_thresholds = InterpretationThresholds()
    st.sidebar.error(
        f"**Invalid interpretation thresholds:** {exc}"
        f"\n\nFalling back to the defaults "
        "(1.00 / 1.33 / 1.67) for this run."
    )

# --- Run analysis (with staleness detection) ---
sig = _signature(values, chart_kind, subgroup_size, spec, rule_set, estimator,
                 unit or None, time_ordered, interp_thresholds,
                 tuple(cleaning.used_indices))
if sig != st.session_state.last_signature:
    st.session_state.outcome = run_analysis(
        AnalysisRequest(
            values=tuple(values),
            chart_kind=chart_kind,
            subgroup_size=int(subgroup_size),
            spec=spec,
            unit=unit or None,
            rule_set=rule_set,
            within_estimator=estimator,
            time_ordered=time_ordered,
            original_indices=tuple(cleaning.used_indices),
            interp_thresholds=interp_thresholds,
        )
    )
    st.session_state.last_signature = sig
    st.session_state.analysis_version += 1

outcome = st.session_state.outcome
st.sidebar.caption(
    f"Analysis v{st.session_state.analysis_version} · "
    f"{len(values)} obs · {chart_label.split()[0]}"
)

if not outcome.ok:
    st.error(f"**Analysis could not run:** {outcome.error}")
    st.stop()
for w in outcome.warnings:
    st.warning(w)

# ===========================================================================
# 4. DESCRIPTIVE STATISTICS
# ===========================================================================
st.header("4 · Descriptive Statistics")
d = outcome.descriptive
dc = st.columns(4)
dc[0].metric("N", d.n)
dc[1].metric(with_unit("Mean", unit), fmt_num(d.mean, decimals))
dc[2].metric(with_unit("Median", unit), fmt_num(d.median, decimals))
dc[3].metric(with_unit("Std dev (s)", unit), fmt_num(d.std_dev, decimals))
dc2 = st.columns(4)
dc2[0].metric("Variance", fmt_num(d.variance, decimals))
dc2[1].metric(with_unit("Minimum", unit), fmt_num(d.minimum, decimals))
dc2[2].metric(with_unit("Maximum", unit), fmt_num(d.maximum, decimals))
dc2[3].metric(with_unit("Range", unit), fmt_num(d.value_range, decimals))
st.caption(
    "Sample statistics (ddof = 1). Values are rounded for display only; "
    "full precision is retained in all calculations and exports."
)

# ===========================================================================
# 5. SPC ANALYSIS
# ===========================================================================
st.header("5 · SPC Analysis")

models = outcome.chart_models
loc, disp = models.location, models.dispersion

if outcome.stability is None:
    st.info(
        "**Stability not assessed.** These measurements are not attested to be "
        "in production/time order, so the control charts below are shown for "
        "reference only — without a known time order they do not constitute a "
        "statistical stability verdict. Tick *'Measurements are in production/"
        "time order'* in the sidebar to enable stability analysis."
    )
elif outcome.stability.is_stable:
    st.success(
        "✅ **Process is statistically STABLE** — no control-rule violations "
        f"detected using the {ruleset_label} rule set."
    )
else:
    st.error(
        "🛑 **Process is statistically UNSTABLE** — "
        f"{outcome.stability.n_violations} violation(s); rules triggered: "
        f"**{', '.join(outcome.stability.triggered_rule_ids)}**. "
        "Capability indices below must not be read as reliable evidence of "
        "future performance until special causes are removed."
    )

lc, dcol = st.columns(2)
for column, model, name in ((lc, loc, "Location chart"), (dcol, disp, "Dispersion chart")):
    with column:
        st.subheader(model.title)
        m1, m2, m3 = st.columns(3)
        m1.metric("UCL", fmt_num(model.ucl, decimals))
        m2.metric("CL", fmt_num(model.center_line, decimals))
        m3.metric("LCL", fmt_num(model.lcl, decimals))

fig = chart_render.render_pair(models)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Rule violations")
violations = outcome.all_violations
if not violations:
    st.success("No SPC rule violations detected.")
else:
    vrows = [
        {
            "Rule ID": v.rule_id,
            "Rule name": v.rule_name,
            "Rule set": v.rule_set.value,
            "Direction": v.direction.value,
            "Affected observations": ", ".join(str(i) for i in v.indices),
        }
        for v in violations
    ]
    st.dataframe(pd.DataFrame(vrows), use_container_width=True)
    with st.expander("Detailed explanations"):
        for v in violations:
            st.markdown(f"**{v.rule_id} — {v.rule_name}**")
            st.caption(v.explanation)

skipped = list(outcome.location_eval.skipped)
if skipped:
    with st.expander(f"Rules not evaluated ({len(skipped)}) — insufficient data"):
        for s in skipped:
            st.caption(f"• **{s.rule_id}** {s.rule_name}: {s.reason}")

# ===========================================================================
# 6. CAPABILITY ANALYSIS
# ===========================================================================
st.header("6 · Capability Analysis")

if outcome.capability is None:
    st.info(
        "Capability indices require at least one specification limit. "
        "Enter a USL and/or LSL in section 2."
    )
else:
    cap = outcome.capability
    if outcome.stability is not None and not outcome.stability.is_stable:
        st.error(
            "🛑 **Stability warning:** the process is out of statistical control. "
            "The indices below are reported unchanged, but they do **not** "
            "establish that the process is capable."
        )
    if cap.sigma_within is None:
        st.warning(
            "⚠️ **Cp/Cpk are not available for this analysis.** Within-subgroup "
            "capability requires rational subgrouping, which assumes the "
            "measurements are in production/time order. Because that has not "
            "been attested, no valid within-subgroup variation estimate exists, "
            "so reporting Cp/Cpk would be statistically indefensible. "
            "**Pp/Ppk (overall performance) remain valid** and are shown below."
        )

    sg1, sg2, sg3 = st.columns(3)
    sg1.metric(with_unit("σ within (short-term)", unit), fmt_num(cap.sigma_within, decimals),
               help=f"Estimator: {outcome.within_estimator_used}. Drives Cp/Cpk.")
    sg2.metric(with_unit("σ overall (long-term)", unit), fmt_num(cap.sigma_overall, decimals),
               help="Sample std dev of all observations. Drives Pp/Ppk.")
    sg3.metric(with_unit("Process mean", unit), fmt_num(cap.mean, decimals))

    capcol, perfcol = st.columns(2)
    with capcol:
        st.subheader("Capability — within variation")
        st.caption("Short-term potential: *can the stable process meet spec?*")
        c = cap.capability
        if c is None:
            st.info("No within-subgroup σ available for this data regime.")
        else:
            r1 = st.columns(2)
            r1[0].metric("Cp", fmt_index(c.cp) if c.cp is not None else NA_DISPLAY,
                         help="Potential capability; requires a two-sided spec.")
            r1[1].metric("Cpk", fmt_index(c.cpk))
            r2 = st.columns(2)
            r2[0].metric("Cpu", fmt_index(c.cpu) if c.cpu is not None else NA_DISPLAY)
            r2[1].metric("Cpl", fmt_index(c.cpl) if c.cpl is not None else NA_DISPLAY)
        if cap.sigma_level_within:
            s1 = st.columns(2)
            s1[0].metric("Sigma level (Z-bench)", fmt_index(cap.sigma_level_within.z_bench))
            s1[1].metric("DPMO", fmt_num(cap.sigma_level_within.dpmo, 1))
    with perfcol:
        st.subheader("Performance — overall variation")
        st.caption("Long-term actual: *how did it perform over the whole run?*")
        p = cap.performance
        r1 = st.columns(2)
        r1[0].metric("Pp", fmt_index(p.cp) if p.cp is not None else NA_DISPLAY,
                     help="Potential performance; requires a two-sided spec.")
        r1[1].metric("Ppk", fmt_index(p.cpk))
        r2 = st.columns(2)
        r2[0].metric("Ppu", fmt_index(p.cpu) if p.cpu is not None else NA_DISPLAY)
        r2[1].metric("Ppl", fmt_index(p.cpl) if p.cpl is not None else NA_DISPLAY)
        s2 = st.columns(2)
        s2[0].metric("Sigma level (Z-bench)", fmt_index(cap.sigma_level_overall.z_bench))
        s2[1].metric("DPMO", fmt_num(cap.sigma_level_overall.dpmo, 1))

    if cap.cpk_ci is not None:
        ci = cap.cpk_ci
        st.info(
            f"**Cpk {ci.confidence * 100:.0f}% confidence interval:** "
            f"[{fmt_index(ci.lower)}, {fmt_index(ci.upper)}] — "
            "approximate (large-sample normal method), **not exact**."
        )
        with st.expander("Confidence-interval caveats"):
            for cav in ci.caveats:
                st.caption(f"• {cav}")

    if cap.warnings:
        st.subheader("Reliability warnings")
        for w in cap.warnings:
            (st.warning if w.severity.value == "warning" else st.info)(
                f"{severity_icon(w.severity.value)} **{w.code}** — {w.message}"
            )

    if cap.normality is not None:
        with st.expander("Normality assessment"):
            verdict = {True: "not rejected", False: "REJECTED", None: "not testable"}[
                cap.normality.normal
            ]
            st.write(f"Normality {verdict} · Shapiro–Wilk p = "
                     f"{fmt_num(cap.normality.shapiro_p, 4)}")
            if cap.normality.note:
                st.caption(cap.normality.note)

# ===========================================================================
# 7. INTERPRETATION
# ===========================================================================
st.header("7 · Interpretation")

if outcome.interpretation is None:
    st.info("Interpretation requires specification limits (see section 2).")
else:
    interp = outcome.interpretation
    st.markdown(f"### {interp.headline}")
    st.caption(
        "Generated by a deterministic rule engine — no language model is "
        "involved in producing these conclusions."
    )

    icol1, icol2 = st.columns(2)
    with icol1:
        st.subheader("📊 Statistical Conclusions")
        st.caption("What the numbers say — factual statements.")
        for c in interp.conclusions:
            with st.container(border=True):
                st.markdown(
                    f"{severity_icon(c.severity.value)} **{c.code}** "
                    f"· `{c.topic.value}` · *{c.severity.value}*"
                )
                st.write(c.text)
                if c.evidence:
                    with st.expander("Evidence"):
                        st.json(c.evidence, expanded=False)
    with icol2:
        st.subheader("🔧 Engineering Recommendations")
        st.caption("Suggested actions — **not** statistical facts.")
        if not interp.recommendations:
            st.info("No specific actions are indicated.")
        for r in interp.recommendations:
            with st.container(border=True):
                st.markdown(
                    f"{severity_icon(r.severity.value)} **{r.code}** "
                    f"· `{r.category.value}` · *{r.severity.value}*"
                )
                st.write(r.text)
                st.caption(f"Triggered by: `{r.trigger_code}`")

# ===========================================================================
# 8. EXPORT / REPORT
# ===========================================================================
st.header("8 · Export / Report")

if outcome.interpretation is None or outcome.capability is None:
    st.info(
        "Full CSV/Excel reports require specification limits so that capability "
        "and interpretation can be included. Chart images are available below."
    )
else:
    meta = ExportMeta(
        characteristic=characteristic or None,
        unit=unit or None,
        part_id=part_id or None,
        machine=machine or None,
        operator=operator or None,
        timestamp=timestamp or None,
        source_file=st.session_state.source_name or None,
        notes=description or None,
    )
    bundle = AnalysisBundle.from_evaluations(
        meta=meta,
        capability=outcome.capability,
        interpretation=outcome.interpretation,
        descriptive=outcome.descriptive,
        measurements=list(values),
        measurement_indices=list(cleaning.used_indices),
        evaluations=[outcome.location_eval, outcome.dispersion_eval],
        methodology=MethodologyInfo(
            data_regime=(
                outcome.data_regime.value if outcome.data_regime else None
            ),
            chart_type=chart_label.split()[0],
            subgroup_size=outcome.subgroup_size,
            n_subgroups=outcome.n_subgroups,
            within_estimator=outcome.within_estimator_used,
            rule_set=rule_set.value,
            n_input_rows=cleaning.report.n_input,
            n_used=outcome.n_used,
            n_excluded=cleaning.report.n_dropped,
            exclusion_notes=tuple(
                f"Row {r.original_index}: {r.reason}" for r in cleaning.excluded
            ),
        ),
    )
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "⬇️ Download CSV results",
        data=csv_report.to_csv_string(bundle).encode("utf-8"),
        file_name="spc_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e2.download_button(
        "⬇️ Download Excel report",
        data=excel_report.workbook_to_bytes(bundle),
        file_name="spc_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    e3.download_button(
        "⬇️ Download PDF report",
        data=pdf_report.build_pdf_bytes(bundle, outcome.chart_models),
        file_name="spc_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.subheader("Chart images")
fig_export = chart_render.render_pair(models)
g1, g2 = st.columns(2)
g1.download_button(
    "🖼️ Download charts (PNG, 200 dpi)",
    data=chart_render.figure_to_png_bytes(fig_export, dpi=200),
    file_name="spc_charts.png", mime="image/png", use_container_width=True,
)
g2.download_button(
    "🖼️ Download charts (SVG, vector)",
    data=chart_render.figure_to_svg_bytes(fig_export),
    file_name="spc_charts.svg", mime="image/svg+xml", use_container_width=True,
)
plt.close(fig_export)

st.caption(
    "Reports: CSV (full precision), Excel (7 sheets), PDF (engineering report "
    "with embedded charts), PNG/SVG chart images."
)
