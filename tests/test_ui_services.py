"""M6 service/UI-integration tests.

These exercise the service layer the Streamlit UI calls — file upload -> clean ->
analyse -> interpret -> export — without needing a browser. They verify real
behaviour (values, staleness, warnings, export bytes), not screenshots.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.data.models import DataRegime
from app.data.validation import RowStatus
from app.interpret.thresholds import InterpretationThresholds
from app.reporting import csv_report, excel_report, pdf_report
from app.reporting.bundle import AnalysisBundle, ExportMeta
from app.statistics.variation import WithinEstimator
from app.reporting import pdf_report
from app.services.analysis import (
    AnalysisRequest,
    ChartKind,
    make_spec,
    run_analysis,
)
from app.services.data_loader import build_subgroups, clean_column, load_table
from app.spc.rules import RuleSet


def _stable_values(n: int = 60, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(10.0, 0.5, n)


def _csv_bytes(values, column="measurement") -> bytes:
    return pd.DataFrame({column: values}).to_csv(index=False).encode("utf-8")


def _xlsx_bytes(values, column="measurement") -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({column: values}).to_excel(buf, index=False)
    return buf.getvalue()


def _spec():
    spec, errors, _ = make_spec(usl=12.0, lsl=8.0, nominal=10.0, unit="mm")
    assert not errors
    return spec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class TestCsvUploadToAnalysis:
    def test_csv_upload_produces_results(self) -> None:
        vals = _stable_values()
        res = load_table(_csv_bytes(vals), "data.csv")
        assert res.ok and res.n_rows == 60
        cleaning = clean_column(res.frame["measurement"].tolist())
        assert cleaning.report.n_valid == 60
        out = run_analysis(AnalysisRequest(
            values=tuple(cleaning.values), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(), unit="mm",
        ))
        assert out.ok
        assert out.capability is not None
        assert out.interpretation is not None
        assert out.n_subgroups == 12

    def test_csv_with_dirty_data_reports_counts(self) -> None:
        raw = [10.0, "bad", None, 10.5, "", float("inf"), 9.8, 10.0]
        res = load_table(_csv_bytes(raw), "dirty.csv")
        assert res.ok
        cleaning = clean_column(res.frame["measurement"].tolist())
        r = cleaning.report
        # 4 usable (10.0, 10.5, 9.8, 10.0); 'inf' becomes a parsed non-finite
        assert r.n_valid == 4
        assert r.dropped_non_numeric >= 1
        assert r.n_duplicates == 1  # 10.0 twice
        # Original indices preserved for the used rows.
        assert cleaning.used_indices == (0, 3, 6, 7)

    def test_single_column_csv_preserves_header_regression(self) -> None:
        """Regression: dialect sniffing (sep=None) shredded single-column headers.

        pandas' sniffer guessed a letter from the header as the delimiter, so
        'measurement' became ['m','asur','m.1','nt']. The loader now uses
        explicit separators. This test pins the fixed behaviour.
        """
        res = load_table(_csv_bytes([1.0, 2.0, 3.0], column="measurement"), "one.csv")
        assert res.ok
        assert res.columns == ["measurement"]
        assert res.n_rows == 3

    def test_tsv_still_parses(self) -> None:
        content = "measurement\tlabel\n1.0\ta\n2.0\tb\n".encode("utf-8")
        res = load_table(content, "data.tsv")
        assert res.ok
        assert res.columns == ["measurement", "label"]

    def test_multi_column_csv_selects_column(self) -> None:
        frame = pd.DataFrame({"id": [1, 2, 3], "measurement": [10.0, 10.1, 9.9]})
        res = load_table(frame.to_csv(index=False).encode("utf-8"), "multi.csv")
        assert res.ok
        assert res.columns == ["id", "measurement"]

    def test_unsupported_file_type_message(self) -> None:
        res = load_table(b"whatever", "notes.docx")
        assert not res.ok
        assert "Unsupported file type" in res.error
        assert "traceback" not in res.error.lower()

    def test_corrupt_file_returns_friendly_error(self) -> None:
        res = load_table(b"\x00\x01\x02not-an-excel", "broken.xlsx")
        assert not res.ok
        assert res.error.startswith("Could not read")


class TestExcelUploadToAnalysis:
    def test_excel_upload_produces_results(self) -> None:
        vals = _stable_values(40)
        res = load_table(_xlsx_bytes(vals), "data.xlsx")
        assert res.ok and res.n_rows == 40
        cleaning = clean_column(res.frame["measurement"].tolist())
        out = run_analysis(AnalysisRequest(
            values=tuple(cleaning.values), chart_kind=ChartKind.IMR, spec=_spec(),
        ))
        assert out.ok
        assert out.capability.capability is not None


class TestManualEntry:
    def test_manual_values_analyse(self) -> None:
        raw = [10.0, 10.2, 9.8, 10.1, None, 9.9, 10.3, 9.7, 10.0, 10.1,
               9.9, 10.2, 10.0, 9.8, 10.1]
        cleaning = clean_column(raw)
        assert cleaning.report.dropped_missing == 1
        out = run_analysis(AnalysisRequest(
            values=tuple(cleaning.values), chart_kind=ChartKind.IMR, spec=_spec(),
        ))
        assert out.ok
        assert out.descriptive.n == 14


# ---------------------------------------------------------------------------
# Analysis reactivity (no stale results)
# ---------------------------------------------------------------------------
class TestSpecChangeUpdatesCapability:
    def test_changing_spec_changes_indices(self) -> None:
        vals = tuple(_stable_values())
        tight, _, _ = make_spec(usl=11.0, lsl=9.0, unit="mm")
        wide, _, _ = make_spec(usl=14.0, lsl=6.0, unit="mm")
        a = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=5, spec=tight))
        b = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=5, spec=wide))
        assert a.capability.capability.cpk != pytest.approx(b.capability.capability.cpk)
        assert b.capability.capability.cpk > a.capability.capability.cpk

    def test_no_spec_yields_no_capability_but_still_spc(self) -> None:
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values()), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=None,
        ))
        assert out.ok
        assert out.capability is None
        assert out.interpretation is None
        assert out.stability is not None  # SPC still available
        assert any("No specification" in w for w in out.warnings)


class TestSubgroupSizeUpdatesSpc:
    def test_changing_subgroup_size_changes_limits(self) -> None:
        vals = tuple(_stable_values())
        a = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=5, spec=_spec()))
        b = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=6, spec=_spec()))
        assert a.n_subgroups == 12 and b.n_subgroups == 10
        assert a.chart_pair.location.ucl != pytest.approx(b.chart_pair.location.ucl)
        # Within-sigma estimate also changes with subgrouping.
        assert a.sigma_within != pytest.approx(b.sigma_within)

    def test_subgroup_size_switches_estimator_at_9(self) -> None:
        vals = tuple(_stable_values(90))
        small = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                             subgroup_size=5, spec=_spec()))
        large = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_S,
                                             subgroup_size=9, spec=_spec()))
        assert small.within_estimator_used == "rbar_d2"
        assert large.within_estimator_used == "sbar_c4"

    def test_trailing_partial_subgroup_warned_not_silent(self) -> None:
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values(52)), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(),
        ))
        assert out.ok
        assert out.n_used == 50
        assert any("trailing" in w.lower() for w in out.warnings)


# ---------------------------------------------------------------------------
# Error handling (user-facing messages, no tracebacks)
# ---------------------------------------------------------------------------
class TestUserFacingErrors:
    def test_too_few_observations(self) -> None:
        out = run_analysis(AnalysisRequest(values=(1.0,), chart_kind=ChartKind.IMR))
        assert not out.ok
        assert "at least 2" in out.error.lower()

    def test_incompatible_subgroup_size(self) -> None:
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values(8)), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(),
        ))
        assert not out.ok
        assert "at least 10 measurements" in out.error

    def test_subgroup_size_below_two(self) -> None:
        res = build_subgroups(np.arange(20.0), 1)
        assert not res.ok
        assert "at least 2" in res.error

    def test_invalid_spec_returns_message(self) -> None:
        spec, errors, _ = make_spec(usl=5.0, lsl=10.0)
        assert spec is None
        assert errors and "below LSL" in errors[0]

    def test_equal_spec_limits_rejected(self) -> None:
        spec, errors, _ = make_spec(usl=5.0, lsl=5.0)
        assert spec is None and errors

    def test_no_spec_is_not_an_error(self) -> None:
        spec, errors, _ = make_spec(usl=None, lsl=None)
        assert spec is None and errors == []

    def test_one_sided_spec_accepted(self) -> None:
        spec, errors, _ = make_spec(usl=12.0, lsl=None, unit="mm")
        assert spec is not None and not errors
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values()), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=spec,
        ))
        assert out.ok
        assert out.capability.capability.cp is None  # one-sided: no Cp
        assert out.capability.capability.cpk is not None


# ---------------------------------------------------------------------------
# Stability surfacing
# ---------------------------------------------------------------------------
class TestStabilitySurfacing:
    def test_unstable_process_flagged(self) -> None:
        vals = list(_stable_values(40))
        vals[20] = 30.0  # gross outlier
        out = run_analysis(AnalysisRequest(
            values=tuple(vals), chart_kind=ChartKind.IMR, spec=_spec(),
        ))
        assert out.ok
        assert not out.stability.is_stable
        assert out.stability.triggered_rule_ids
        # Capability report carries the instability warning...
        codes = {w.code for w in out.capability.warnings}
        assert "unstable_process" in codes
        # ...and the interpretation makes it CRITICAL.
        assert "process_unstable" in out.interpretation.conclusion_codes
        # Indices are RETAINED, not nulled.
        assert out.capability.capability.cpk is not None

    def test_stable_process_reports_stable(self) -> None:
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values()), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(),
        ))
        assert out.stability.is_stable
        assert "process_stable" in out.interpretation.conclusion_codes


# ---------------------------------------------------------------------------
# Interpretation separation preserved through the service layer
# ---------------------------------------------------------------------------
class TestInterpretationSeparation:
    def test_conclusions_and_recommendations_separate(self) -> None:
        from app.interpret.models import (
            EngineeringRecommendation,
            StatisticalConclusion,
        )
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values()), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(),
        ))
        interp = out.interpretation
        assert all(isinstance(c, StatisticalConclusion) for c in interp.conclusions)
        assert all(isinstance(r, EngineeringRecommendation)
                   for r in interp.recommendations)
        assert interp.headline

    def test_custom_thresholds_flow_through(self) -> None:
        vals = tuple(_stable_values())
        strict = InterpretationThresholds(capable_min=5.0, excellent_min=9.0)
        out = run_analysis(AnalysisRequest(
            values=vals, chart_kind=ChartKind.XBAR_R, subgroup_size=5,
            spec=_spec(), interp_thresholds=strict,
        ))
        # With an unreachable "capable" bar nothing is categorised capable+.
        assert "capability_capable" not in out.interpretation.conclusion_codes


# ---------------------------------------------------------------------------
# Export wiring from the analysis outcome
# ---------------------------------------------------------------------------
class TestExportsFromOutcome:
    def _bundle(self):
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values()), chart_kind=ChartKind.XBAR_R,
            subgroup_size=5, spec=_spec(), unit="mm",
        ))
        return out, AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="bore", unit="mm", source_file="d.csv"),
            capability=out.capability, interpretation=out.interpretation,
            descriptive=out.descriptive, measurements=list(out.chart_pair.location.points),
            evaluations=[out.location_eval, out.dispersion_eval],
        )

    def test_csv_export_from_analysis(self) -> None:
        _, bundle = self._bundle()
        s = csv_report.to_csv_string(bundle)
        assert "# Statistical conclusions" in s
        assert "# Engineering recommendations" in s
        assert "bore" in s

    def test_excel_export_from_analysis(self) -> None:
        _, bundle = self._bundle()
        data = excel_report.workbook_to_bytes(bundle)
        assert data[:2] == b"PK"

    def test_chart_export_from_outcome(self) -> None:
        import matplotlib.pyplot as plt
        from app.charts import render
        out, _ = self._bundle()
        fig = render.render_pair(out.chart_models)
        png = render.figure_to_png_bytes(fig)
        svg = render.figure_to_svg_bytes(fig)
        plt.close(fig)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"<svg" in svg[:512]


# ---------------------------------------------------------------------------
# Staleness: recomputation on input change
# ---------------------------------------------------------------------------
class TestNoStaleResults:
    def test_same_inputs_same_outputs(self) -> None:
        req = AnalysisRequest(values=tuple(_stable_values()),
                              chart_kind=ChartKind.XBAR_R, subgroup_size=5,
                              spec=_spec())
        a, b = run_analysis(req), run_analysis(req)
        assert a.capability.capability.cpk == b.capability.capability.cpk
        assert a.interpretation.conclusion_codes == b.interpretation.conclusion_codes

    def test_changing_values_changes_results(self) -> None:
        v1 = tuple(_stable_values(60, seed=1))
        v2 = tuple(_stable_values(60, seed=2))
        a = run_analysis(AnalysisRequest(values=v1, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=5, spec=_spec()))
        b = run_analysis(AnalysisRequest(values=v2, chart_kind=ChartKind.XBAR_R,
                                         subgroup_size=5, spec=_spec()))
        assert a.sigma_within != pytest.approx(b.sigma_within)

    def test_changing_chart_kind_changes_regime(self) -> None:
        vals = tuple(_stable_values())
        sub = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.XBAR_R,
                                           subgroup_size=5, spec=_spec()))
        ind = run_analysis(AnalysisRequest(values=vals, chart_kind=ChartKind.IMR,
                                           spec=_spec()))
        assert sub.n_subgroups == 12
        assert ind.n_subgroups is None
        assert sub.within_estimator_used != ind.within_estimator_used


# ---------------------------------------------------------------------------
# Chart-kind coverage
# ---------------------------------------------------------------------------
class TestAllChartKinds:
    @pytest.mark.parametrize(
        "kind,size", [(ChartKind.XBAR_R, 5), (ChartKind.XBAR_S, 10), (ChartKind.IMR, 1)]
    )
    def test_each_chart_kind_runs(self, kind, size) -> None:
        out = run_analysis(AnalysisRequest(
            values=tuple(_stable_values(60)), chart_kind=kind,
            subgroup_size=size, spec=_spec(), unit="mm",
        ))
        assert out.ok
        assert out.chart_models is not None
        # Unit propagates into the chart model's axis label.
        assert "mm" in out.chart_models.location.y_label


# ===========================================================================
# A3: data-order attestation / DataRegime guardrail
# ===========================================================================
class TestDataOrderAttestation:
    """A3 regression: unordered data must not yield Cp/Cpk.

    Within-subgroup sigma presupposes rational subgrouping, which only exists
    for time-ordered production data. The attestation is authoritative user
    input (never inferred from sortedness). When it is absent the service layer
    -- not the UI -- must withhold Cp/Cpk and the stability verdict while
    keeping Pp/Ppk, which depend only on the overall sigma.
    """

    SPEC = make_spec(usl=12.0, lsl=8.0, unit="mm")[0]

    def _run(self, ordered: bool, kind: ChartKind = ChartKind.XBAR_R):
        return run_analysis(
            AnalysisRequest(
                values=tuple(_stable_values()),
                chart_kind=kind,
                subgroup_size=5,
                spec=self.SPEC,
                unit="mm",
                time_ordered=ordered,
            )
        )

    def test_default_is_ordered(self) -> None:
        """Existing callers keep the ordered workflow with no code change."""
        req = AnalysisRequest(values=(1.0, 2.0), chart_kind=ChartKind.IMR)
        assert req.time_ordered is True

    def test_ordered_preserves_existing_behaviour(self) -> None:
        out = self._run(ordered=True)
        assert out.data_regime is DataRegime.SUBGROUPED
        cap = out.capability
        assert cap.capability is not None
        assert cap.capability.cpk is not None
        assert cap.sigma_within is not None
        assert out.stability is not None

    def test_unordered_suppresses_within_capability(self) -> None:
        out = self._run(ordered=False)
        assert out.data_regime is DataRegime.UNORDERED
        cap = out.capability
        assert cap.capability is None, "Cp/Cpk must not be reported"
        assert cap.sigma_within is None
        assert cap.cpk_ci is None
        assert cap.sigma_level_within is None

    def test_unordered_retains_performance_indices(self) -> None:
        out = self._run(ordered=False)
        perf = out.capability.performance
        assert perf is not None
        assert perf.cp is not None and perf.cpk is not None
        assert out.capability.sigma_overall is not None

    def test_performance_identical_ordered_and_unordered(self) -> None:
        """The guardrail must not perturb the overall-sigma path at all."""
        a = self._run(ordered=True).capability.performance
        b = self._run(ordered=False).capability.performance
        assert a.cp == b.cp and a.cpk == b.cpk
        assert a.cpu == b.cpu and a.cpl == b.cpl

    def test_unordered_withholds_stability_verdict(self) -> None:
        out = self._run(ordered=False)
        assert out.stability is None, "charts are reference-only when unordered"

    def test_unordered_emits_explanatory_warning(self) -> None:
        out = self._run(ordered=False)
        joined = " ".join(out.warnings).lower()
        assert "time order" in joined
        assert "cp/cpk" in joined

    def test_unordered_imr_also_guarded(self) -> None:
        out = self._run(ordered=False, kind=ChartKind.IMR)
        assert out.data_regime is DataRegime.UNORDERED
        assert out.capability.capability is None
        assert out.capability.performance is not None

    def test_ordered_imr_regime_is_individuals(self) -> None:
        out = self._run(ordered=True, kind=ChartKind.IMR)
        assert out.data_regime is DataRegime.INDIVIDUALS

    def test_sigma_within_surfaced_but_not_used(self) -> None:
        """Transparency: the number is visible, but it fed no index."""
        out = self._run(ordered=False)
        assert out.sigma_within is not None
        assert out.capability.sigma_within is None

    def test_interpretation_cannot_claim_capability_from_cpk(self) -> None:
        out = self._run(ordered=False)
        codes = out.interpretation.conclusion_codes
        assert "within_capability_not_assessed" in codes
        assert not [c for c in codes if c.startswith("capability_")], (
            "no Cpk band conclusion may be emitted without a within sigma"
        )
        assert "rec_maintain" not in out.interpretation.recommendation_codes
        assert "Cpk" not in out.interpretation.headline.split("—")[0] or             "not assessed" in out.interpretation.headline

    def test_interpretation_recommends_ordered_collection(self) -> None:
        out = self._run(ordered=False)
        assert "rec_collect_ordered_data" in out.interpretation.recommendation_codes

    def test_exports_render_na_for_suppressed_indices(self) -> None:
        out = self._run(ordered=False)
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="unordered", unit="mm"),
            descriptive=out.descriptive,
            capability=out.capability,
            interpretation=out.interpretation,
            measurements=list(_stable_values()),
            evaluations=[out.location_eval, out.dispersion_eval],
        )
        text = csv_report.to_csv_string(bundle)
        # Within sigma is explicitly n/a, not blank and not a silent omission.
        assert "sigma_within (short-term),n/a" in text
        assert "(no within-subgroup sigma available),n/a" in text
        # Performance indices are still exported with real values.
        assert any(l.startswith("Ppk,") for l in text.splitlines())
        ppk_line = [l for l in text.splitlines() if l.startswith("Ppk,")][0]
        assert ppk_line.split(",")[1] not in ("n/a", "")
        # No Cp/Cpk value rows may be emitted.
        assert not [
            l for l in text.splitlines()
            if l.startswith(("Cp,", "Cpk,", "Cpu,", "Cpl,"))
        ]
        # Excel and PDF must also build without error for this state.
        assert len(excel_report.workbook_to_bytes(bundle)) > 0
        assert len(pdf_report.build_pdf_bytes(bundle, out.chart_models)) > 0

    def test_attestation_participates_in_request_identity(self) -> None:
        """The UI staleness signature depends on this field being part of the
        request; two requests differing only in attestation must not be equal."""
        base = dict(
            values=(1.0, 2.0, 3.0, 4.0),
            chart_kind=ChartKind.IMR,
            spec=self.SPEC,
        )
        assert AnalysisRequest(**base, time_ordered=True) != AnalysisRequest(
            **base, time_ordered=False
        )


# ===========================================================================
# C1: original-row index traceability, end to end
# ===========================================================================
class TestOriginalIndexTraceability:
    """C1 regression: every layer must cite the ORIGINAL source-row index.

    `cleaning.values` has had bad rows removed, so a chart point at cleaned
    position p is NOT source row p once anything was excluded. Before this fix
    `run_analysis` never consulted `cleaning.used_indices`, so violations (and
    the Excel Measurements sheet) pointed at the wrong rows.
    """

    # Rows 2 and 4 are non-numeric and get excluded; the outlier is the LAST
    # row (original index 19, cleaned position 17) -> a 2-row discrepancy.
    RAW = [10.0, 10.1, "bad", 10.2, "n/a", 10.3, 10.0, 9.9, 10.1, 10.2,
           9.8, 10.0, 10.4, 9.7, 10.1, 10.0, 9.9, 10.2, 10.1, 13.9]
    OUTLIER_ORIGINAL_ROW = 19
    OUTLIER_CLEANED_POS = 17

    def _cleaned(self):
        return clean_column(list(self.RAW))

    def _run(self, *, with_mapping: bool):
        cl = self._cleaned()
        spec, _, _ = make_spec(usl=12.0, lsl=8.0, unit="mm")
        return cl, run_analysis(
            AnalysisRequest(
                values=tuple(cl.values),
                chart_kind=ChartKind.IMR,
                spec=spec,
                unit="mm",
                original_indices=tuple(cl.used_indices) if with_mapping else None,
            )
        )

    def test_fixture_actually_has_an_index_discrepancy(self) -> None:
        """Guard: the fixture must exercise the bug, not a trivial case."""
        cl = self._cleaned()
        assert len(cl.values) == len(self.RAW) - 2
        assert cl.used_indices[self.OUTLIER_CLEANED_POS] == self.OUTLIER_ORIGINAL_ROW
        assert self.OUTLIER_CLEANED_POS != self.OUTLIER_ORIGINAL_ROW

    def test_violation_cites_original_row(self) -> None:
        _, out = self._run(with_mapping=True)
        beyond = [v for v in out.location_eval.violations if v.matches_rule("N1")]
        assert beyond, "expected a beyond-limits violation for the outlier"
        assert beyond[0].indices == (self.OUTLIER_ORIGINAL_ROW,)

    def test_without_mapping_it_reports_the_cleaned_position(self) -> None:
        """Documents the old behavior, proving the mapping is what fixes it."""
        _, out = self._run(with_mapping=False)
        beyond = [v for v in out.location_eval.violations if v.matches_rule("N1")]
        assert beyond[0].indices == (self.OUTLIER_CLEANED_POS,)

    def test_chart_model_points_carry_original_indices(self) -> None:
        cl, out = self._run(with_mapping=True)
        originals = [p.original_index for p in out.chart_models.location.points]
        assert originals == list(cl.used_indices)

    def test_chart_model_violation_lookup_uses_original_index(self) -> None:
        _, out = self._run(with_mapping=True)
        loc = out.chart_models.location
        assert loc.violations_for_original_index(self.OUTLIER_ORIGINAL_ROW)
        assert not loc.violations_for_original_index(self.OUTLIER_CLEANED_POS)

    def test_csv_export_cites_original_row(self) -> None:
        cl, out = self._run(with_mapping=True)
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="trace", unit="mm"),
            descriptive=out.descriptive, capability=out.capability,
            interpretation=out.interpretation, measurements=list(cl.values),
            measurement_indices=list(cl.used_indices),
            evaluations=[out.location_eval, out.dispersion_eval],
        )
        text = csv_report.to_csv_string(bundle)
        vio_lines = [l for l in text.splitlines() if l.startswith(("N1,", "WE1,"))]
        assert vio_lines, "no violation rows in CSV"
        assert str(self.OUTLIER_ORIGINAL_ROW) in " ".join(vio_lines)

    def test_excel_measurements_sheet_uses_original_rows(self) -> None:
        import io

        from openpyxl import load_workbook

        cl, out = self._run(with_mapping=True)
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="trace", unit="mm"),
            descriptive=out.descriptive, capability=out.capability,
            interpretation=out.interpretation, measurements=list(cl.values),
            measurement_indices=list(cl.used_indices),
            evaluations=[out.location_eval, out.dispersion_eval],
        )
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(bundle)))
        ws = wb["Measurements"]
        assert ws.cell(row=1, column=1).value == "Original row"
        exported = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1)]
        assert exported == list(cl.used_indices)
        # Rows 2 and 4 were excluded, so they must not appear.
        assert 2 not in exported and 4 not in exported
        # The last measurement is the outlier and must carry row 19.
        assert exported[-1] == self.OUTLIER_ORIGINAL_ROW

    def test_pdf_export_mentions_original_row(self) -> None:
        import io

        from pypdf import PdfReader

        cl, out = self._run(with_mapping=True)
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="trace", unit="mm"),
            descriptive=out.descriptive, capability=out.capability,
            interpretation=out.interpretation, measurements=list(cl.values),
            measurement_indices=list(cl.used_indices),
            evaluations=[out.location_eval, out.dispersion_eval],
        )
        pdf = pdf_report.build_pdf_bytes(bundle, out.chart_models)
        text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf)).pages)
        assert str(self.OUTLIER_ORIGINAL_ROW) in text

    def test_all_layers_agree_on_one_index_identity(self) -> None:
        """CSV, Excel, chart model and violations must cite the SAME row."""
        import io

        from openpyxl import load_workbook

        cl, out = self._run(with_mapping=True)
        bundle = AnalysisBundle.from_evaluations(
            meta=ExportMeta(characteristic="trace", unit="mm"),
            descriptive=out.descriptive, capability=out.capability,
            interpretation=out.interpretation, measurements=list(cl.values),
            measurement_indices=list(cl.used_indices),
            evaluations=[out.location_eval, out.dispersion_eval],
        )
        vio_row = [v for v in out.location_eval.violations
                   if v.matches_rule("N1")][0].indices[0]
        wb = load_workbook(io.BytesIO(excel_report.workbook_to_bytes(bundle)))
        ws = wb["Measurements"]
        rows = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1)]
        vals = [ws.cell(row=r, column=3).value for r in range(2, ws.max_row + 1)]
        # The Excel row carrying the outlier value must be the violation's row.
        assert rows[vals.index(max(vals))] == vio_row
        model_rows = [p.original_index for p in out.chart_models.location.points]
        assert vio_row in model_rows

    def test_subgrouped_chart_labels_subgroups_by_first_original_row(self) -> None:
        """X-bar points span several rows; they carry the FIRST member's row."""
        raw = ["bad"] + [10.0, 10.1, 9.9, 10.2, 10.0] * 4
        cl = clean_column(raw)
        spec, _, _ = make_spec(usl=12.0, lsl=8.0, unit="mm")
        out = run_analysis(
            AnalysisRequest(
                values=tuple(cl.values), chart_kind=ChartKind.XBAR_R,
                subgroup_size=5, spec=spec, unit="mm",
                original_indices=tuple(cl.used_indices),
            )
        )
        # Row 0 was dropped, so subgroup 0 starts at original row 1.
        originals = [p.original_index for p in out.chart_models.location.points]
        assert originals[0] == 1
        assert originals == [1, 6, 11, 16]

    def test_mapping_does_not_change_any_statistic(self) -> None:
        """Index relabelling must not perturb a single number."""
        _, a = self._run(with_mapping=False)
        _, b = self._run(with_mapping=True)
        assert a.capability.capability.cpk == b.capability.capability.cpk
        assert a.capability.performance.cpk == b.capability.performance.cpk
        assert a.sigma_within == b.sigma_within
        assert a.sigma_overall_value == b.sigma_overall_value
        assert a.chart_pair.location.ucl == b.chart_pair.location.ucl
        assert a.stability.n_violations == b.stability.n_violations


# ===========================================================================
# C2: signature completeness (future-proofing)
# ===========================================================================
class TestSignatureCompleteness:
    """C2: every AnalysisRequest field that changes the outcome must be in the
    UI staleness signature, or results silently go stale.

    This is the regression guard for the original `interp_thresholds` omission:
    it enumerates the request fields and fails when a NEW one appears that the
    signature does not cover, forcing a deliberate decision at that time.
    """

    #: Fields deliberately excluded from the signature, with justification.
    #: `values` is covered via a hash; the rest are covered explicitly.
    KNOWN_FIELDS = {
        "values", "chart_kind", "subgroup_size", "spec", "unit", "rule_set",
        "within_estimator", "ci_alpha", "time_ordered", "original_indices",
        "guard_thresholds", "interp_thresholds",
    }

    def test_no_unreviewed_request_fields(self) -> None:
        """A new AnalysisRequest field must be consciously reviewed."""
        import dataclasses

        actual = {f.name for f in dataclasses.fields(AnalysisRequest)}
        new = actual - self.KNOWN_FIELDS
        assert not new, (
            f"AnalysisRequest gained field(s) {sorted(new)}. Add them to "
            "_signature() (or document why they cannot affect the outcome) and "
            "extend this test."
        )

    @staticmethod
    def _load_signature():
        """Import `_signature` WITHOUT executing the Streamlit app script.

        `app/ui/capability_page.py` is a script: importing it normally runs the
        whole page (and `st.stop()` is a no-op outside a script runner). We
        therefore compile only the function definitions we need.
        """
        import ast
        import pathlib

        src = pathlib.Path("app/ui/capability_page.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        keep = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or (isinstance(node, ast.FunctionDef) and node.name == "_signature")
        ]
        module = ast.Module(body=keep, type_ignores=[])
        ns: dict = {}
        exec(compile(module, "streamlit_app_signature", "exec"), ns)  # noqa: S102
        return ns["_signature"]

    def _sig(self, **overrides):
        _signature = self._load_signature()

        base = dict(
            values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            chart_kind=ChartKind.XBAR_R,
            subgroup_size=3,
            spec=make_spec(usl=12.0, lsl=8.0, unit="mm")[0],
            rule_set=RuleSet.NELSON,
            estimator=WithinEstimator.AUTO,
            unit="mm",
            time_ordered=True,
            interp_thresholds=InterpretationThresholds(),
            original_indices=(0, 1, 2, 3, 4, 5),
        )
        base.update(overrides)
        return _signature(**base)

    def test_baseline_signature_is_stable(self) -> None:
        assert self._sig() == self._sig()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("values", (9.0, 9.1, 9.2, 9.3, 9.4, 9.5)),
            ("chart_kind", ChartKind.IMR),
            ("subgroup_size", 2),
            ("spec", make_spec(usl=99.0, lsl=1.0, unit="mm")[0]),
            ("rule_set", RuleSet.WECO),
            ("estimator", WithinEstimator.RBAR_D2),
            ("unit", "inch"),
            ("time_ordered", False),
            ("interp_thresholds", InterpretationThresholds(capable_min=1.5)),
            ("original_indices", (0, 1, 2, 3, 4, 7)),
        ],
    )
    def test_each_input_changes_the_signature(self, field, value) -> None:
        assert self._sig() != self._sig(**{field: value}), (
            f"changing {field!r} left the signature unchanged -> stale results"
        )

    #: A DIFFERENT, still-valid value for every InterpretationThresholds field.
    #: All eight are user-adjustable in the sidebar, so all eight must reach the
    #: signature. Adding a field to the dataclass without adding it here fails
    #: `test_every_threshold_field_is_covered`.
    THRESHOLD_ALTERNATES = {
        "marginal_min": 0.90,
        "capable_min": 1.50,
        "excellent_min": 1.80,
        "centering_gap_ratio": 0.25,
        "drift_gap_ratio": 0.25,
        "spec_margin_tight_sigma": 8.0,
        "min_observations": 40,
        "min_subgroups": 25,
    }

    def test_every_threshold_field_has_an_alternate(self) -> None:
        """A new threshold field must be consciously reviewed for staleness."""
        import dataclasses

        actual = {f.name for f in dataclasses.fields(InterpretationThresholds)}
        missing = actual - set(self.THRESHOLD_ALTERNATES)
        assert not missing, (
            f"InterpretationThresholds gained field(s) {sorted(missing)}. The "
            "sidebar exposes these thresholds, so confirm _signature() covers "
            "the new field and add an alternate value here."
        )

    @pytest.mark.parametrize("field", sorted(THRESHOLD_ALTERNATES))
    def test_every_threshold_field_changes_the_signature(self, field) -> None:
        """Regression guard for the original three-fields-by-name signature.

        The signature once listed marginal/capable/excellent explicitly; the
        other five cut-points could therefore be edited in the sidebar without
        invalidating the cached outcome.
        """
        alt = InterpretationThresholds(**{field: self.THRESHOLD_ALTERNATES[field]})
        assert self._sig() != self._sig(interp_thresholds=alt), (
            f"changing InterpretationThresholds.{field} left the signature "
            "unchanged -> section 7 would keep showing the previous verdict"
        )

    def test_unexposed_fields_are_documented(self) -> None:
        """guard_thresholds/ci_alpha are not user-adjustable yet.

        They exist on AnalysisRequest but have no UI widget, so they cannot go
        stale from the user's perspective today. If either is ever exposed, the
        signature must gain it -- this test records that contract.
        """
        import inspect
        import pathlib

        src = pathlib.Path("app/ui/capability_page.py").read_text(encoding="utf-8")
        for field in ("guard_thresholds", "ci_alpha"):
            exposed = f"{field}=" in src.split("def _signature")[0]
            assert not exposed, (
                f"{field} appears to be user-adjustable now; add it to "
                "_signature() and to this test."
            )
