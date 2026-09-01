"""Process capability and performance indices.

This subpackage computes the capability metrics and their reliability guards.
The cardinal rule enforced here:

* **Cp, Cpu, Cpl, Cpk** are computed from *within-subgroup* sigma only.
* **Pp, Ppu, Ppl, Ppk** are computed from *overall* sigma only.

The two families never share a sigma. See ``docs/FORMULAS.md`` §2.4–2.7.

Modules
-------
indices
    Index formulas, sigma level / Z-bench, and the Cpk confidence interval.
guards
    Reliability warnings (small sample, few subgroups, zero variation,
    non-normality, stability-not-verified).
"""
