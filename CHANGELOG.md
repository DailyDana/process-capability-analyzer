# Changelog

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
