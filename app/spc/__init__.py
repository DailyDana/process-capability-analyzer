"""SPC control charts and the run-rule engine.

This subpackage is pure (no plotting, no I/O) so limit math and rule logic stay
fully unit-testable; rendering is M4's concern and consumes these results.

Modules
-------
limits
    Control-limit calculators for X-bar/R, X-bar/S, and I-MR charts. Each
    returns a :class:`ControlChart` bundling CL/UCL/LCL, the plotted points, the
    zone sigma, and per-point out-of-limit flags.
rules
    The Western Electric and Nelson run-rule engines operating on a
    :class:`ControlChart`, returning structured :class:`Violation` objects.

The control-chart limits are built from the plotted statistic's own dispersion
(R-bar, S-bar, MR-bar) via the validated M1 constants — never from the M2
capability sigma. See ``docs/FORMULAS.md`` §3.
"""
