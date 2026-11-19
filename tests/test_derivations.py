"""Show-your-work: the value in every explanation equals the value in the results.

The derivations recompute each number from the inputs they display, so these
tests compare two independent paths: the analysis pipeline and the formula a
reader would follow by hand.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from app.data.models import SpecLimits
from app.explain.derivations import ascii_rows, capability_derivations, gage_derivations
from app.msa.models import GageMethod
from app.services.analysis import AnalysisRequest, ChartKind, run_analysis
from app.services.gage_rr import columns_to_request, run_gage_study
from app.statistics.variation import WithinEstimator

ROOT = Path(__file__).resolve().parent.parent
REL = 1e-12


def _close(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return a == pytest.approx(b, rel=REL, abs=1e-300)


def _by_key(derivations) -> dict:
    keys = [d.key for d in derivations]
    assert len(keys) == len(set(keys)), "duplicate derivation keys"
    return {d.key: d for d in derivations}


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------
def _values(seed: int = 4, n: int = 100) -> tuple[float, ...]:
    return tuple(float(v) for v in np.random.default_rng(seed).normal(10.0, 0.5, n))


SPECS = [SpecLimits(usl=12.0, lsl=8.0), SpecLimits(usl=11.5), SpecLimits(lsl=8.5)]
SETUPS = [
    (ChartKind.XBAR_R, 5, WithinEstimator.AUTO),
    (ChartKind.XBAR_R, 5, WithinEstimator.POOLED),
    (ChartKind.XBAR_S, 10, WithinEstimator.AUTO),
    (ChartKind.XBAR_R, 4, WithinEstimator.SBAR_C4),
    (ChartKind.IMR, 1, WithinEstimator.AUTO),
]


@pytest.mark.parametrize("spec,setup", list(itertools.product(SPECS, SETUPS)))
def test_capability_explanations_equal_the_reported_values(spec, setup):
    kind, size, estimator = setup
    out = run_analysis(AnalysisRequest(values=_values(), chart_kind=kind, subgroup_size=size, spec=spec,
                                       within_estimator=estimator))
    assert out.ok
    rep = out.capability
    d = _by_key(capability_derivations(rep, out.within_estimator_used, out.within_detail))

    assert _close(d["mean"].value, rep.mean)
    assert _close(d["sigma_within"].value, rep.sigma_within)
    assert "see the methodology" not in d["sigma_within"].substitution  # the estimator was actually explained
    assert _close(d["sigma_overall"].value, rep.sigma_overall)
    for family, prefix in ((rep.capability, "c"), (rep.performance, "p")):
        for name, reported in (("p", family.cp), ("pu", family.cpu), ("pl", family.cpl), ("pk", family.cpk)):
            key = prefix + name
            if reported is None:
                assert key not in d
            else:
                assert _close(d[key].value, reported), key
    assert _close(d["ppm_overall"].value, rep.sigma_level_overall.dpmo)
    assert _close(d["cpk_ci_lower"].value, rep.cpk_ci.lower)
    assert _close(d["cpk_ci_upper"].value, rep.cpk_ci.upper)


def test_unordered_data_explains_performance_only():
    out = run_analysis(AnalysisRequest(values=_values(), chart_kind=ChartKind.IMR, spec=SPECS[0], time_ordered=False))
    d = _by_key(capability_derivations(out.capability, out.within_estimator_used, out.within_detail))
    assert "sigma_within" not in d and "cpk" not in d
    assert _close(d["ppk"].value, out.capability.performance.cpk)


def test_zero_variation_uses_the_same_sentinels():
    out = run_analysis(AnalysisRequest(values=tuple([10.0] * 40), chart_kind=ChartKind.XBAR_R, subgroup_size=5,
                                       spec=SPECS[0]))
    d = _by_key(capability_derivations(out.capability, out.within_estimator_used, out.within_detail))
    assert d["cpk"].value == out.capability.capability.cpk == math.inf
    assert d["ppk"].value == math.inf and "ppm_overall" not in d
    assert d["cpk"].as_row()["Value"] == "+inf"


def test_without_estimator_detail_sigma_within_is_an_input():
    out = run_analysis(AnalysisRequest(values=_values(), chart_kind=ChartKind.XBAR_R, subgroup_size=5, spec=SPECS[0]))
    d = _by_key(capability_derivations(out.capability))
    assert d["sigma_within"].value == out.capability.sigma_within
    assert _close(d["cpk"].value, out.capability.capability.cpk)


def test_substitution_shows_the_numbers_used():
    out = run_analysis(AnalysisRequest(values=_values(), chart_kind=ChartKind.XBAR_R, subgroup_size=5, spec=SPECS[0]))
    d = _by_key(capability_derivations(out.capability, out.within_estimator_used, out.within_detail))
    assert d["cp"].formula == "(USL − LSL) / (6 σ within)"
    assert d["cp"].substitution == f"(12 − 8) / (6 × {out.capability.sigma_within:.6g})"
    assert d["sigma_within"].formula == "R-bar / d2(n)" and "/ 2.326" in d["sigma_within"].substitution
    assert all(x.source.startswith("FORMULAS.md §") for x in d.values())


# ---------------------------------------------------------------------------
# Gage R&R
# ---------------------------------------------------------------------------
def _gage(name: str, method: GageMethod, alpha_pool: float = 0.25, tolerance=None):
    with open(ROOT / "examples" / name, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k.lower(): k for k in rows[0]}
    part = next(cols[k] for k in cols if k.startswith("part"))
    op = next(cols[k] for k in cols if k.startswith(("operator", "appraiser", "inspector")))
    val = next(cols[k] for k in cols if k in ("value", "measurement", "reading", "impedance"))
    out = run_gage_study(columns_to_request([r[part] for r in rows], [r[op] for r in rows], [r[val] for r in rows],
                                            method=method, alpha_pool=alpha_pool, tolerance=tolerance))
    assert out.ok, out.errors
    return out.result


GAGE_CASES = [
    ("gage_rr_aiag.csv", GageMethod.ANOVA, 0.25, None),
    ("gage_rr_aiag.csv", GageMethod.ANOVA, 1.0, 8.0),  # interaction kept
    ("gage_rr_aiag.csv", GageMethod.ANOVA, 0.0, 8.0),  # interaction pooled
    ("gage_rr_aiag.csv", GageMethod.AVERAGE_RANGE, 0.25, 8.0),
    ("gage_rr_thermal_impedance.csv", GageMethod.ANOVA, 0.25, 30.0),
    ("gage_rr_thermal_impedance.csv", GageMethod.ANOVA, 1.0, None),
    ("gage_rr_thermal_impedance.csv", GageMethod.AVERAGE_RANGE, 0.25, None),
]


@pytest.mark.parametrize("name,method,alpha_pool,tolerance", GAGE_CASES)
def test_gage_explanations_equal_the_results_table(name, method, alpha_pool, tolerance):
    res = _gage(name, method, alpha_pool, tolerance)
    d = _by_key(gage_derivations(res))
    c = res.components
    assert _close(d["ev"].value, c.repeatability_sd)
    assert _close(d["av"].value, c.reproducibility_sd)
    assert _close(d["pv"].value, c.part_sd)
    assert _close(d["grr"].value, c.grr_sd)
    assert _close(d["tv"].value, c.total_sd)
    grr_row = res.row("Total Gage R&R")
    assert _close(d["pct_study_var"].value, grr_row.pct_study_var)
    assert _close(d["pct_contribution"].value, grr_row.pct_contribution)
    if tolerance is None:
        assert "pct_tolerance" not in d
    else:
        assert _close(d["pct_tolerance"].value, grr_row.pct_tolerance)
    assert d["ndc"].value == res.ndc
    if c.operator_sd is not None:
        assert _close(d["operator"].value, c.operator_sd)
    if c.interaction_sd is not None:
        assert _close(d["interaction"].value, c.interaction_sd)
    else:
        assert "interaction" not in d


def test_clamped_component_is_said_to_be_clamped():
    """A study with no operator effect: the negative estimate is shown and noted, the value is 0."""
    parts = [i for i in range(10) for _ in range(6)]
    ops = [j for _ in range(10) for j in range(3) for _ in range(2)]
    for seed in range(40):
        rng = np.random.default_rng(seed)
        vals = [p + rng.normal(0, 0.5) for p in parts]
        res = run_gage_study(columns_to_request(parts, ops, vals, method=GageMethod.ANOVA)).result
        if "operator" in res.components.clamped:
            d = _by_key(gage_derivations(res))
            assert d["operator"].value == 0.0 and "negative, set to 0" in d["operator"].substitution
            return
    pytest.fail("no clamped case generated")


# ---------------------------------------------------------------------------
# PDF appendix
# ---------------------------------------------------------------------------
def test_ascii_rows_have_no_glyphs_missing_from_the_pdf_font():
    out = run_analysis(AnalysisRequest(values=_values(), chart_kind=ChartKind.XBAR_R, subgroup_size=5, spec=SPECS[0]))
    rows = ascii_rows(capability_derivations(out.capability, out.within_estimator_used, out.within_detail))
    rows += ascii_rows(gage_derivations(_gage("gage_rr_aiag.csv", GageMethod.ANOVA, 1.0, 8.0)))
    text = " ".join(cell for row in rows for cell in row)
    text.encode("cp1252")  # Helvetica's WinAnsi repertoire; raises if a glyph is missing
    assert "sigma within" in text and "σ" not in text


def test_gage_pdf_has_the_calculations_appendix():
    from app.msa.interpret import interpret_gage
    from app.reporting.msa_pdf import gage_rr_pdf_bytes
    from app.services.gage_rr import run_gage_study as run

    with open(ROOT / "examples" / "gage_rr_aiag.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keys = list(rows[0])
    out = run(columns_to_request([r[keys[0]] for r in rows], [r[keys[1]] for r in rows], [r[keys[-1]] for r in rows]))
    assert out.ok, out.errors
    pdf = gage_rr_pdf_bytes(out.study, out.result, interpret_gage(out.result))
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert "7. Calculations" in text and "trunc( 1.41 x" in text
    assert f"{out.result.components.grr_sd:.6g}" in text
