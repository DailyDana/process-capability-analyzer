"""SPC / Process Capability Analyzer.

A modular, technically credible Statistical Process Control (SPC) and process
capability analysis tool for manufacturing engineering.

The package is organized in layers with one-directional dependencies:

    ui  ->  services  ->  (interpret, capability, spc, charts, reporting)
                       ->  statistics  ->  data
                       ->  data

The core domain (``data``, ``statistics``, ``spc``, ``capability``,
``interpret``) is pure and I/O-free so it stays fully unit-testable.
"""

__version__ = "0.1.0"
