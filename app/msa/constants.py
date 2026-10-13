"""Average & Range method constants (AIAG MSA Reference Manual, 4th ed.).

``K1 = 1/d2*`` for the number of trials, ``K2 = 1/d2*`` for the number of
appraisers and ``K3 = 1/d2*`` for the number of parts. They are tabulated
values of ``d2*`` (the bias-correction constant for the *average range of g
subgroups*), **not** the control-chart ``d2`` used elsewhere in this
application:

* K1 assumes many subgroups (parts x appraisers > 15), where d2* ~= d2, so
  K1(2) = 1/1.128 and K1(3) = 1/1.693 coincide with 1/d2.
* K2 and K3 use a single range (g = 1), where d2* differs from d2
  noticeably (d2*(2, g=1) = 1.41 vs d2(2) = 1.128). Deriving them from d2
  would therefore be wrong; the tables are reproduced verbatim instead.

The manual tabulates K1 only for 2-3 trials and K2 only for 2-3 appraisers.
Studies outside these ranges can still be analysed with the ANOVA method.
"""

from __future__ import annotations

__all__ = ["K1", "K2", "K3", "NDC_FACTOR", "k1", "k2", "k3"]

#: Trials -> K1.
K1: dict[int, float] = {2: 0.8862, 3: 0.5908}

#: Appraisers -> K2.
K2: dict[int, float] = {2: 0.7071, 3: 0.5231}

#: Parts -> K3.
K3: dict[int, float] = {
    2: 0.7071,
    3: 0.5231,
    4: 0.4467,
    5: 0.4030,
    6: 0.3742,
    7: 0.3534,
    8: 0.3375,
    9: 0.3249,
    10: 0.3146,
}

#: AIAG number-of-distinct-categories factor, ndc = 1.41 * (PV / GRR),
#: truncated. (1.41 ~= sqrt(2); some software uses sqrt(2) exactly, which can
#: change ndc only when the product lands within 0.3% of an integer.)
NDC_FACTOR: float = 1.41


def _lookup(table: dict[int, float], n: int, what: str) -> float:
    try:
        return table[n]
    except KeyError:
        lo, hi = min(table), max(table)
        raise ValueError(
            f"The Average & Range method is tabulated for {lo}-{hi} {what} "
            f"(AIAG MSA 4th ed.); got {n}. Use the ANOVA method instead."
        ) from None


def k1(trials: int) -> float:
    return _lookup(K1, trials, "trials")


def k2(appraisers: int) -> float:
    return _lookup(K2, appraisers, "appraisers")


def k3(parts: int) -> float:
    return _lookup(K3, parts, "parts")
