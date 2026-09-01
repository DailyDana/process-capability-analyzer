"""Chart construction and rendering (presentation layer).

M4 turns the validated M3 statistical results into professional SPC charts
WITHOUT duplicating any statistical logic. The layer is split so that plotting
is cleanly separated from the numbers:

Modules
-------
model
    Pure, plot-library-agnostic ``ChartModel`` assembled from the M3
    ``ControlChart`` and ``RuleEvaluation``. Single source of truth: every
    limit, point value, index, and violation is taken verbatim from M3. No
    matplotlib import here, so the M6 Streamlit UI can consume the model
    directly.
render
    Renders a ``ChartModel`` to a matplotlib figure and exports PNG/SVG.

The chart layer NEVER recomputes control limits or re-evaluates rules.
See ``docs/FORMULAS.md`` §M4.
"""
