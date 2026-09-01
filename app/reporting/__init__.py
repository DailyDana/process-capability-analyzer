"""Report/export layer: CSV, Excel (M5), and PDF (M7).

This layer SERIALIZES already-computed analysis results. It performs no
statistical computation — every value comes from the tested M1-M3 analysis
layer and the M5 interpretation engine. Keeping it separate from the statistics
preserves the one-directional dependency and keeps the numbers a single source
of truth.

Modules
-------
bundle
    ``AnalysisBundle`` — a small container gathering everything an export needs
    (metadata, spec, descriptive stats, capability report, SPC violations,
    interpretation) so the exporters have one typed input.
csv_report
    Deterministic flat CSV export.
excel_report
    Multi-sheet professional .xlsx export.
"""
