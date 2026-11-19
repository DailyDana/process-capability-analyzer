"""Show-your-work derivations for the reported indices.

Each :class:`Derivation` gives one reported number as a formula, the same
formula with the inputs filled in, the value that arithmetic gives, and where
the formula comes from. The value is **recomputed here from the displayed
inputs** with plain arithmetic, not copied from the result, so that a reader
can follow it with a calculator and a test can check that the explanation and
the results table agree (``tests/test_derivations.py``).

Sources are the sections of ``docs/FORMULAS.md``, which carry the full
references ([M] Montgomery, [AIAG] SPC manual, [AIAG-MSA] MSA manual).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, inf, isfinite, sqrt
from typing import Mapping, Optional

from scipy.stats import norm

from app.capability.report import CapabilityReport
from app.msa.constants import NDC_FACTOR
from app.msa.models import GageMethod, GageRRResult

__all__ = ["Derivation", "capability_derivations", "gage_derivations", "ascii_rows"]

_SRC_SIGMA_W = "FORMULAS.md §2.2; [M] ch. 6, [AIAG] ch. II"
_SRC_SIGMA_O = "FORMULAS.md §2.3; [AIAG] ch. IV"
_SRC_INDEX = "FORMULAS.md §2.4; [M] §8.3, [AIAG] ch. IV"
_SRC_PPM = "FORMULAS.md §2.5"
_SRC_CI = "FORMULAS.md §2.6; Bissell (1990), [M] eq. 8.24"
_SRC_AR = "FORMULAS.md §6.4; [AIAG-MSA] ch. III sect. B"
_SRC_ANOVA = "FORMULAS.md §6.3; [AIAG-MSA] app. A, [BBM]"
_SRC_TABLE = "FORMULAS.md §6.6; [AIAG-MSA] ch. III sect. B"


@dataclass(frozen=True, slots=True)
class Derivation:
    """One reported number, shown as formula -> numbers -> value."""

    key: str
    label: str
    formula: str
    substitution: str
    value: Optional[float]
    source: str

    def as_row(self) -> dict[str, str]:
        return {
            "Quantity": self.label,
            "Formula": self.formula,
            "With the numbers": self.substitution,
            "Value": _g(self.value),
            "Source": self.source,
        }


#: The PDF reports use the built-in Helvetica, which has no Greek or minus-sign glyphs.
_ASCII = str.maketrans({"σ": "sigma", "Φ": "Phi", "−": "-", "⁶": "^6", "²": "^2", "×": "x", "—": "-"})


def ascii_rows(derivations: "list[Derivation]") -> list[list[str]]:
    """Rows (quantity, formula, numbers, value, source) in plain ASCII, for the PDF appendix."""
    return [[str(v).translate(_ASCII) for v in d.as_row().values()] for d in derivations]


def _g(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if not isfinite(x):
        return "+inf" if x > 0 else ("-inf" if x < 0 else "NaN")
    return f"{x:.6g}"


def _div(num: float, den: float) -> float:
    """``num / den`` with the application's zero-variation convention (FORMULAS.md §2.4)."""
    if den == 0.0:
        return inf if num > 0 else (-inf if num < 0 else 0.0)
    return num / den


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------
def _sigma_within(estimator: Optional[str], d: Mapping[str, float]) -> Optional[Derivation]:
    if estimator == "rbar_d2" and {"R_bar", "d2"} <= set(d):
        return Derivation("sigma_within", "σ within", "R-bar / d2(n)", f"{_g(d['R_bar'])} / {_g(d['d2'])}",
                          d["R_bar"] / d["d2"], _SRC_SIGMA_W)
    if estimator == "sbar_c4" and {"S_bar", "c4"} <= set(d):
        return Derivation("sigma_within", "σ within", "S-bar / c4(n)", f"{_g(d['S_bar'])} / {_g(d['c4'])}",
                          d["S_bar"] / d["c4"], _SRC_SIGMA_W)
    if estimator == "pooled" and {"s_pooled", "c4", "pooled_dof"} <= set(d):
        return Derivation("sigma_within", "σ within", "s_pooled / c4(dof + 1)",
                          f"{_g(d['s_pooled'])} / {_g(d['c4'])}   (dof = {_g(d['pooled_dof'])})",
                          d["s_pooled"] / d["c4"], _SRC_SIGMA_W)
    if estimator == "mr_d2" and {"MR_bar", "d2"} <= set(d):
        return Derivation("sigma_within", "σ within", "MR-bar / d2(2)", f"{_g(d['MR_bar'])} / {_g(d['d2'])}",
                          d["MR_bar"] / d["d2"], _SRC_SIGMA_W)
    return None


def _index_family(prefix: str, sigma_name: str, sigma: float, mean: float,
                  usl: Optional[float], lsl: Optional[float]) -> list[Derivation]:
    """Cp/Cpu/Cpl/Cpk (prefix ``C``) or Pp/Ppu/Ppl/Ppk (prefix ``P``) from one sigma."""
    out: list[Derivation] = []
    s = _g(sigma)
    upper = lower = None
    if usl is not None and lsl is not None:
        out.append(Derivation(f"{prefix.lower()}p", f"{prefix}p", f"(USL − LSL) / (6 {sigma_name})",
                              f"({_g(usl)} − {_g(lsl)}) / (6 × {s})", _div(usl - lsl, 6.0 * sigma), _SRC_INDEX))
    if usl is not None:
        upper = _div(usl - mean, 3.0 * sigma)
        out.append(Derivation(f"{prefix.lower()}pu", f"{prefix}pu", f"(USL − x-bar) / (3 {sigma_name})",
                              f"({_g(usl)} − {_g(mean)}) / (3 × {s})", upper, _SRC_INDEX))
    if lsl is not None:
        lower = _div(mean - lsl, 3.0 * sigma)
        out.append(Derivation(f"{prefix.lower()}pl", f"{prefix}pl", f"(x-bar − LSL) / (3 {sigma_name})",
                              f"({_g(mean)} − {_g(lsl)}) / (3 × {s})", lower, _SRC_INDEX))
    sides = [v for v in (upper, lower) if v is not None]
    if len(sides) == 2:
        formula, numbers = f"min({prefix}pu, {prefix}pl)", f"min({_g(upper)}, {_g(lower)})"
    elif upper is not None:
        formula, numbers = f"{prefix}pu (upper specification only)", _g(upper)
    else:
        formula, numbers = f"{prefix}pl (lower specification only)", _g(lower)
    out.append(Derivation(f"{prefix.lower()}pk", f"{prefix}pk", formula, numbers, min(sides), _SRC_INDEX))
    return out


def capability_derivations(
    report: CapabilityReport,
    within_estimator: Optional[str] = None,
    within_detail: Optional[Mapping[str, float]] = None,
) -> list[Derivation]:
    """Derivations for everything in a capability report.

    ``within_estimator`` and ``within_detail`` are the estimator name and the
    intermediate quantities of the within-sigma estimate
    (:class:`app.statistics.variation.WithinSigmaResult`); without them the
    within sigma is listed as an input.
    """
    usl, lsl, mean = report.spec.usl, report.spec.lsl, report.mean
    out: list[Derivation] = [
        Derivation("mean", "x-bar", "sum(x) / N", f"N = {report.n_observations}", mean, "FORMULAS.md §1.2"),
    ]
    if report.sigma_within is not None and report.capability is not None:
        sw = _sigma_within(within_estimator, within_detail or {})
        if sw is None:
            sw = Derivation("sigma_within", "σ within", "within-subgroup estimate", "see the methodology section",
                            report.sigma_within, _SRC_SIGMA_W)
        out.append(sw)
        out += _index_family("C", "σ within", float(sw.value), mean, usl, lsl)  # type: ignore[arg-type]
    out.append(Derivation("sigma_overall", "σ overall", "s = sqrt( sum (x − x-bar)² / (N − 1) )",
                          f"N = {report.n_observations}", report.sigma_overall, _SRC_SIGMA_O))
    so = report.sigma_overall
    out += _index_family("P", "σ overall", so, mean, usl, lsl)

    if so > 0.0:
        terms, numbers, p = [], [], 0.0
        if usl is not None:
            z = (usl - mean) / so
            p += float(norm.sf(z))
            terms.append("Φ(−(USL − x-bar)/σ overall)")
            numbers.append(f"Φ(−{_g(z)})")
        if lsl is not None:
            z = (mean - lsl) / so
            p += float(norm.sf(z))
            terms.append("Φ(−(x-bar − LSL)/σ overall)")
            numbers.append(f"Φ(−{_g(z)})")
        out.append(Derivation("ppm_overall", "Expected PPM (overall)", f"10⁶ × [{' + '.join(terms)}]",
                              f"10⁶ × [{' + '.join(numbers)}]", 1e6 * p, _SRC_PPM))

    ci = report.cpk_ci
    if ci is not None:
        z = float(norm.ppf(1.0 - (1.0 - ci.confidence) / 2.0))
        half = z * sqrt(1.0 / (9.0 * ci.n) + ci.cpk**2 / (2.0 * (ci.n - 1)))
        pct = f"{100 * ci.confidence:.0f}%"
        numbers = f"{_g(ci.cpk)} {{sign}} {_g(z)} × sqrt(1/(9 × {ci.n}) + {_g(ci.cpk)}² / (2 × {ci.n - 1}))"
        out.append(Derivation("cpk_ci_lower", f"Cpk {pct} CI, lower", "Cpk − z sqrt(1/(9N) + Cpk²/(2(N − 1)))",
                              numbers.format(sign="−"), ci.cpk - half, _SRC_CI))
        out.append(Derivation("cpk_ci_upper", f"Cpk {pct} CI, upper", "Cpk + z sqrt(1/(9N) + Cpk²/(2(N − 1)))",
                              numbers.format(sign="+"), ci.cpk + half, _SRC_CI))
    return out


# ---------------------------------------------------------------------------
# Gage R&R
# ---------------------------------------------------------------------------
def _clamp_note(raw: float) -> str:
    return "   (negative, set to 0)" if raw < 0 else ""


def gage_derivations(result: GageRRResult) -> list[Derivation]:
    """Derivations for a Gage R&R result (either method)."""
    p, o, r = result.n_parts, result.n_operators, result.n_trials
    out: list[Derivation] = []

    if result.method is GageMethod.AVERAGE_RANGE:
        d = result.average_range
        ev = d.r_bar_bar * d.k1
        out.append(Derivation("ev", "Repeatability (EV)", "R-double-bar × K1",
                              f"{_g(d.r_bar_bar)} × {_g(d.k1)}", ev, _SRC_AR))
        radicand = (d.x_bar_diff * d.k2) ** 2 - ev**2 / (p * r)
        av = sqrt(radicand) if radicand > 0 else 0.0
        out.append(Derivation("av", "Reproducibility (AV)", "sqrt( (X-bar_diff × K2)² − EV² / (n r) )",
                              f"sqrt( ({_g(d.x_bar_diff)} × {_g(d.k2)})² − {_g(ev)}² / ({p} × {r}) )"
                              + _clamp_note(radicand), av, _SRC_AR))
        pv = d.r_p * d.k3
        out.append(Derivation("pv", "Part-to-part (PV)", "R_p × K3", f"{_g(d.r_p)} × {_g(d.k3)}", pv, _SRC_AR))
    else:
        a = result.anova
        t = a.table
        ms_p, ms_o = t.row("Part").ms, t.row("Operator").ms
        ms_e = t.row("Repeatability").ms
        err_name = "MS_error (interaction pooled)" if a.interaction_pooled else "MS_E"
        ev = sqrt(ms_e)
        out.append(Derivation("ev", "Repeatability (EV)", f"sqrt({err_name})", f"sqrt({_g(ms_e)})", ev, _SRC_ANOVA))
        if a.interaction_pooled:
            v_o_raw = (ms_o - ms_e) / (p * r)
            v_po = 0.0
            den_name, den = err_name, ms_e
        else:
            ms_po = t.row("Part x Operator").ms
            v_po_raw = (ms_po - ms_e) / r
            v_po = max(0.0, v_po_raw)
            out.append(Derivation("interaction", "Part × Operator", "sqrt( (MS_PO − MS_E) / r )",
                                  f"sqrt( ({_g(ms_po)} − {_g(ms_e)}) / {r} )" + _clamp_note(v_po_raw),
                                  sqrt(v_po), _SRC_ANOVA))
            v_o_raw = (ms_o - ms_po) / (p * r)
            den_name, den = "MS_PO", ms_po
        v_o = max(0.0, v_o_raw)
        out.append(Derivation("operator", "Operator", f"sqrt( (MS_O − {den_name}) / (p r) )",
                              f"sqrt( ({_g(ms_o)} − {_g(den)}) / ({p} × {r}) )" + _clamp_note(v_o_raw),
                              sqrt(v_o), _SRC_ANOVA))
        av = sqrt(v_o + v_po)
        out.append(Derivation("av", "Reproducibility (AV)", "sqrt( σ²_operator + σ²_part×operator )",
                              f"sqrt( {_g(sqrt(v_o))}² + {_g(sqrt(v_po))}² )", av, _SRC_ANOVA))
        v_p_raw = (ms_p - den) / (o * r)
        pv = sqrt(max(0.0, v_p_raw))
        out.append(Derivation("pv", "Part-to-part (PV)", f"sqrt( (MS_P − {den_name}) / (o r) )",
                              f"sqrt( ({_g(ms_p)} − {_g(den)}) / ({o} × {r}) )" + _clamp_note(v_p_raw),
                              pv, _SRC_ANOVA))

    grr = sqrt(ev**2 + av**2)
    out.append(Derivation("grr", "Total Gage R&R (GRR)", "sqrt( EV² + AV² )",
                          f"sqrt( {_g(ev)}² + {_g(av)}² )", grr, _SRC_TABLE))
    tv = sqrt(grr**2 + pv**2)
    out.append(Derivation("tv", "Total variation (TV)", "sqrt( GRR² + PV² )",
                          f"sqrt( {_g(grr)}² + {_g(pv)}² )", tv, _SRC_TABLE))
    if tv > 0.0:
        out.append(Derivation("pct_study_var", "%Study Var, GRR", "100 × GRR / TV",
                              f"100 × {_g(grr)} / {_g(tv)}", 100.0 * grr / tv, _SRC_TABLE))
        out.append(Derivation("pct_contribution", "%Contribution, GRR", "100 × GRR² / TV²",
                              f"100 × {_g(grr)}² / {_g(tv)}²", 100.0 * grr**2 / tv**2, _SRC_TABLE))
    if result.tolerance:
        out.append(Derivation("pct_tolerance", "%Tolerance, GRR", "100 × k × GRR / (USL − LSL)",
                              f"100 × {_g(result.k)} × {_g(grr)} / {_g(result.tolerance)}",
                              100.0 * result.k * grr / result.tolerance, _SRC_TABLE))
    if grr > 0.0:
        out.append(Derivation("ndc", "ndc", f"trunc( {NDC_FACTOR:g} × PV / GRR )",
                              f"trunc( {NDC_FACTOR:g} × {_g(pv)} / {_g(grr)} ) = trunc({_g(NDC_FACTOR * pv / grr)})",
                              float(floor(NDC_FACTOR * pv / grr)), _SRC_TABLE))
    return out
