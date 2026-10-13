"""Streamlit presentation layer.

Thin UI only. All statistics, SPC, interpretation, and export logic live in the
validated ``app.*`` domain/service layers; this package renders their results.

Modules
-------
formatting
    Presentation-only number/label helpers (rounding for DISPLAY only — the
    underlying results always retain full precision).
streamlit_app
    The application entry point and page router
    (``streamlit run app/ui/streamlit_app.py``).
capability_page
    SPC charts, stability and Cp/Cpk vs Pp/Ppk.
gage_rr_page
    Crossed Gage R&R (measurement system analysis).
"""
