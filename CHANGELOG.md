# Changelog

## Unreleased

### Added
- **Example data on the capability page**: a third data source loads one of
  three bundled datasets (piston rings Phase I; piston rings with limits from
  the first 25 subgroups; a synthetic individuals series) and presets the chart,
  subgroup size, specification limits and limit source. It never replaces data
  the user loaded without asking.

## 0.5.0 (2026-11-20)

### Added
- **Attribute control charts** (p, np, c, u) on their own page: limits per
  sample for varying sample sizes, a baseline period or a known standard, the
  run rules that do not rely on normal zones, interpretation and CSV export.
  Reproduces the limits printed in the `qcc` documentation for four of
  Montgomery's datasets. Methodology: `docs/FORMULAS.md` §9.
- **Phase I / Phase II control limits**: limits from a baseline period, points
  left out with a recorded reason, later data judged against them, and frozen
  limits exported to and imported from CSV (§3.11).
- **Lag-1 autocorrelation check** before the I-MR chart, with an approximate
  significance test, a conclusion and a recommendation (§3.10).
- **Show your work**: formula, numbers and source for every capability index
  and Gage R&R quantity, in the UI and as an appendix of both PDF reports (§10).
- NIST StRD univariate datasets in the test suite (mean, standard deviation,
  lag-1 autocorrelation against certified values).
- A scheduled workflow that visits the hosted demos so they stay awake.

### Fixed
- Non-normal capability: identical measurements now give a plain error instead of an
  unhandled exception, and Box-Cox rejects a USL at or below zero instead of returning NaN.
- The descriptive summary now covers the same measurements as the indices when trailing
  readings are left out of the last subgroup.
- The analysis service reports an estimator that does not fit the data shape as an error.
- The non-normality warning points to the non-normal capability page.

## 0.4.0 (2026-10-23)

### Added
- **Capability for non-normal data** on its own page: percentile method
  (ISO 22514-2) on normal, lognormal, Weibull, gamma and exponential fits
  ranked by Anderson-Darling, or a Box-Cox transformation; expected and
  observed PPM; a natural lower bound in place of the LSL; the normal-theory
  result shown alongside. Exact maximum-likelihood fits. Reproduces the three
  model results of a published cycle-time study to the printed decimal.
  Methodology: `docs/FORMULAS.md` §8.

## 0.3.0 (2026-10-23)

### Added
- **Type-1 gauge study** (Bosch Procedure 1 / VDA 5 Verfahren 1) on its own
  page: Cg, Cgk, %Var, the bias t-test, resolution as % of tolerance,
  confidence intervals for Cg and Cgk, and the smallest tolerance the gauge
  would still pass. Verified against the Bosch Booklet 10 and BPI Consulting
  worked examples. Methodology: `docs/FORMULAS.md` §7.
- **PDF reports** for Gage R&R and Type-1 studies (built when the download
  button is clicked).

## 0.2.0 (2026-10-13)

### Added
- **Gage R&R (crossed, balanced)** on its own page: ANOVA (random-effects
  F tests, interaction pooled at p > 0.25 by convention) and the AIAG Average &
  Range method. Summary table with %Contribution, %Study Var and %Tolerance
  (k = 6 or 5.15), ndc, a range-chart consistency check, deterministic
  interpretation and CSV/Excel export. Methodology: `docs/FORMULAS.md` §6.
- Two reference studies in `examples/` (AIAG MSA 4th ed.; Houf & Burman 1988).
  The results match the published Minitab, SAS and BPI analyses of them to the
  printed decimal.
- `tools/make_gage_fixtures.py`, which regenerates the statsmodels ANOVA oracle
  used by the tests.

### Changed
- `app/ui/streamlit_app.py` is now a page router (`st.navigation`). The
  capability page moved to `app/ui/capability_page.py`. The entry point and the
  live-demo URL are unchanged.
- `use_container_width` replaced by `width="stretch"`. It has been deprecated
  since 2025 and is being removed from Streamlit.
- Requires Streamlit 1.62 or later (the tested version).

## 0.1.0 (2026-09-01)

First release: X̄–R, X̄–S and I–MR charts with Western Electric and Nelson
rules, Cp/Cpk and Pp/Ppk with stability-first interpretation, Cpk confidence
interval, CSV/Excel/PDF reports, Streamlit UI.
