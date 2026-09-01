"""Deterministic, rule-based engineering interpretation of analysis results.

This subpackage turns the structured numeric results from M2 (capability) and M3
(stability / SPC rules) into two STRICTLY SEPARATED kinds of output:

* :class:`StatisticalConclusion` — a factual, defensible statement about what the
  numbers say (e.g. "Cpk 1.18 < Cp 1.42, so centering limits capability").
* :class:`EngineeringRecommendation` — a suggested action tied to an identified
  condition (e.g. "investigate process centering"). Never presented as fact.

The engine is 100% rule-based and deterministic — no LLM, no randomness — so it
is fully reproducible and unit-testable. An optional natural-language layer may
consume these structured objects later, but it must not compute statistics.

Every threshold is documented in ``docs/FORMULAS.md`` §M5 and is configurable via
:class:`~app.interpret.thresholds.InterpretationThresholds`. M5 introduces NO new
statistical formulas; it only reads validated M1-M3 quantities.
"""
