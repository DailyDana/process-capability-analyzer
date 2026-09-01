"""Data layer: domain models, validation, and I/O.

The models here are the vocabulary the whole application speaks. They are pure
data holders (frozen dataclasses) with no behaviour beyond light structural
validation, so that the statistics / SPC / capability layers can depend on them
without pulling in any I/O.

Modules
-------
models
    ``SpecLimits``, ``Measurement``, ``Dataset`` and the ``DataRegime`` enum.
validation
    Coercion of raw/imperfect input into clean numeric arrays plus spec-limit
    sanity checks, returning structured reports rather than raising blindly.
"""
