"""Statistical utilities: descriptive statistics and control-chart constants.

This subpackage is pure (no I/O) and depends only on numpy/scipy so that every
statistical quantity is independently unit-testable.

Modules
-------
descriptive
    Mean, median, variance, standard deviation, range, min/max.
constants
    Authoritative control-chart constant table (d2, d3, A2, A3, D3, D4, B3, B4,
    c4, E2) with lookup helpers.
"""
