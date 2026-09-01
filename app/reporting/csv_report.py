"""Deterministic CSV export of an :class:`~app.reporting.bundle.AnalysisBundle`.

The CSV is a sectioned key/value + table layout (one file, human-readable and
diff-friendly). It:

* preserves full numeric precision (``repr``-grade float formatting),
* represents missing/unavailable values explicitly as ``"n/a"`` (never blank-by-
  accident, never silently dropped),
* is fully deterministic — same bundle -> byte-identical output,
* performs no statistical computation (values come straight from the bundle).
"""

from __future__ import annotations

import csv
import io
from typing import Optional

from app.reporting.bundle import AnalysisBundle

__all__ = ["to_csv_string", "write_csv", "NA"]

NA = "n/a"


def _num(x: Optional[float]) -> str:
    """Format a number at full precision, or ``NA`` when missing.

    Uses ``repr`` so no precision is lost (round-trippable). ``+inf``/``-inf``
    (zero-variation sentinel) and ``NaN`` are rendered as readable tokens.
    """
    if x is None:
        return NA
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return NA
    if fx != fx:  # NaN
        return "NaN"
    if fx == float("inf"):
        return "inf"
    if fx == float("-inf"):
        return "-inf"
    return repr(fx)


def _s(x: object) -> str:
    """Format a non-numeric value, or ``NA`` when None."""
    return NA if x is None else str(x)


def _section(writer: "csv.writer", title: str) -> None:
    writer.writerow([])
    writer.writerow([f"# {title}"])


def to_csv_string(bundle: AnalysisBundle) -> str:
    """Serialize an :class:`AnalysisBundle` to a deterministic CSV string."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")

    cap = bundle.capability
    spec = cap.spec

    # --- Metadata ---
    w.writerow(["# Metadata"])
    w.writerow(["Field", "Value"])
    for label, value in bundle.meta.as_rows():
        w.writerow([label, _s(value)])

    # --- Specification ---
    _section(w, "Specification")
    w.writerow(["Field", "Value"])
    w.writerow(["USL", _num(spec.usl)])
    w.writerow(["LSL", _num(spec.lsl)])
    w.writerow(["Nominal", _num(spec.nominal)])
    w.writerow(["Unit", _s(spec.unit)])
    w.writerow(["Two-sided", _s(spec.is_two_sided)])
    w.writerow(["Tolerance width", _num(spec.tolerance_width)])

    # --- Descriptive statistics ---
    _section(w, "Descriptive statistics")
    w.writerow(["Statistic", "Value"])
    if bundle.descriptive is not None:
        d = bundle.descriptive
        w.writerow(["n", _num(d.n)])
        w.writerow(["mean", _num(d.mean)])
        w.writerow(["median", _num(d.median)])
        w.writerow(["variance", _num(d.variance)])
        w.writerow(["std_dev", _num(d.std_dev)])
        w.writerow(["minimum", _num(d.minimum)])
        w.writerow(["maximum", _num(d.maximum)])
        w.writerow(["range", _num(d.value_range)])
    else:
        w.writerow(["(not provided)", NA])

    # --- Sigma estimates ---
    _section(w, "Sigma estimates")
    w.writerow(["Estimate", "Value"])
    w.writerow(["sigma_within (short-term)", _num(cap.sigma_within)])
    w.writerow(["sigma_overall (long-term)", _num(cap.sigma_overall)])
    w.writerow(["grand_mean", _num(cap.mean)])
    w.writerow(["n_observations", _num(cap.n_observations)])
    w.writerow(["n_subgroups", _num(cap.n_subgroups)])

    # --- Capability indices (within) ---
    _section(w, "Capability indices (within-subgroup, short-term)")
    w.writerow(["Index", "Value"])
    if cap.capability is not None:
        c = cap.capability
        w.writerow(["Cp", _num(c.cp)])
        w.writerow(["Cpk", _num(c.cpk)])
        w.writerow(["Cpu", _num(c.cpu)])
        w.writerow(["Cpl", _num(c.cpl)])
    else:
        w.writerow(["(no within-subgroup sigma available)", NA])
    if cap.sigma_level_within is not None:
        w.writerow(["Z_bench (within)", _num(cap.sigma_level_within.z_bench)])
        w.writerow(["DPMO (within)", _num(cap.sigma_level_within.dpmo)])

    # --- Performance indices (overall) ---
    _section(w, "Performance indices (overall, long-term)")
    w.writerow(["Index", "Value"])
    p = cap.performance
    w.writerow(["Pp", _num(p.cp)])
    w.writerow(["Ppk", _num(p.cpk)])
    w.writerow(["Ppu", _num(p.cpu)])
    w.writerow(["Ppl", _num(p.cpl)])
    w.writerow(["Z_bench (overall)", _num(cap.sigma_level_overall.z_bench)])
    w.writerow(["DPMO (overall)", _num(cap.sigma_level_overall.dpmo)])

    # --- Cpk confidence interval ---
    _section(w, "Cpk confidence interval")
    w.writerow(["Field", "Value"])
    if cap.cpk_ci is not None:
        ci = cap.cpk_ci
        w.writerow(["confidence", _num(ci.confidence)])
        w.writerow(["lower", _num(ci.lower)])
        w.writerow(["upper", _num(ci.upper)])
        w.writerow(["approximate", _s(ci.approximate)])
    else:
        w.writerow(["(not computed)", NA])

    # --- SPC / stability summary ---
    _section(w, "SPC stability summary")
    w.writerow(["Field", "Value"])
    if cap.stability is not None:
        st = cap.stability
        w.writerow(["is_stable", _s(st.is_stable)])
        w.writerow(["n_violations", _num(st.n_violations)])
        w.writerow(["triggered_rule_ids", _s(", ".join(st.triggered_rule_ids) or NA)])
    else:
        w.writerow(["stability_assessed", _s(False)])

    # --- Normality ---
    _section(w, "Normality")
    w.writerow(["Field", "Value"])
    if cap.normality is not None:
        nm = cap.normality
        w.writerow(["normal", _s(nm.normal)])
        w.writerow(["shapiro_p", _num(nm.shapiro_p)])
    else:
        w.writerow(["(not tested)", NA])

    # --- Rule violations table ---
    _section(w, "SPC rule violations")
    w.writerow(["rule_id", "rule_name", "rule_set", "direction", "indices"])
    if bundle.violations:
        for v in bundle.violations:
            w.writerow([
                v.rule_id,
                v.rule_name,
                v.rule_set.value,
                v.direction.value,
                " ".join(str(i) for i in v.indices),
            ])
    else:
        w.writerow(["(none)", NA, NA, NA, NA])

    # --- Interpretation: statistical conclusions ---
    _section(w, "Statistical conclusions")
    w.writerow(["code", "topic", "severity", "text"])
    for c in bundle.interpretation.conclusions:
        w.writerow([c.code, c.topic.value, c.severity.value, c.text])

    # --- Interpretation: engineering recommendations (kept separate) ---
    _section(w, "Engineering recommendations")
    w.writerow(["code", "category", "severity", "trigger_code", "text"])
    for r in bundle.interpretation.recommendations:
        w.writerow([r.code, r.category.value, r.severity.value, r.trigger_code, r.text])

    return buf.getvalue()


def write_csv(bundle: AnalysisBundle, path: str) -> str:
    """Write the CSV to ``path`` (UTF-8). Returns the path."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(to_csv_string(bundle))
    return path
