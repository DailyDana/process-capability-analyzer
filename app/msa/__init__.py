"""Measurement system analysis (MSA): crossed Gage R&R.

Scope (deliberately narrow, see ``docs/FORMULAS.md`` §6):

* balanced, crossed studies only (every operator measures every part the same
  number of times);
* two estimation methods: two-way ANOVA (with the interaction term) and the
  AIAG Average & Range method;
* variable (continuous) data only.

Out of scope: attribute agreement, nested/unbalanced designs, Type-1 gauge
studies, linearity and bias.

Like the rest of the statistical core this package has no UI or I/O
dependency; :mod:`app.services.gage_rr` orchestrates it for the UI.
"""
