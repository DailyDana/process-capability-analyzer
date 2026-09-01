"""End-to-end smoke test of the actual Streamlit app.

Uses Streamlit's official AppTest harness, which EXECUTES the real
``app/ui/streamlit_app.py`` script in-process and exposes the rendered elements.
This verifies the app actually runs (imports, widget wiring, section rendering)
rather than just checking that a server port is open.

The app stops early when no data is loaded, so the first assertions cover the
initial state; then we inject a representative dataset into session_state and
re-run to exercise the full analysis path through all 8 sections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "ui" / "streamlit_app.py")


def _representative_values(n: int = 58, seed: int = 2026) -> list[float]:
    """Stable process with one special-cause outlier and a sustained tail shift."""
    rng = np.random.default_rng(seed)
    vals = list(rng.normal(10.0, 0.35, n))
    vals[41] = 12.4          # single gross outlier -> beyond-limit rule
    for i in range(50, n):   # sustained shift -> run rule
        vals[i] += 0.55
    return vals


def _run_with_data(values, timeout: float = 90.0) -> AppTest:
    """Run the app with measurements pre-loaded into session state."""
    from app.services.data_loader import clean_column

    at = AppTest.from_file(APP, default_timeout=timeout)
    at.session_state["raw_values"] = list(values)
    at.session_state["source_name"] = "smoke_test.csv"
    at.session_state["cleaning"] = clean_column(list(values))
    at.run()
    return at


class TestAppLaunches:
    def test_app_runs_without_data(self) -> None:
        """The app must render its input section and stop cleanly with no data."""
        at = AppTest.from_file(APP, default_timeout=90).run()
        assert not at.exception, f"App raised: {at.exception}"
        titles = [t.value for t in at.title]
        assert any("SPC" in t for t in titles)
        headers = [h.value for h in at.header]
        assert any("Data Input" in h for h in headers)
        # With no data the app shows guidance and stops.
        assert any("Upload a file" in i.value for i in at.info)


class TestEndToEndSmoke:
    def test_full_analysis_renders_all_sections(self) -> None:
        at = _run_with_data(_representative_values())
        assert not at.exception, f"App raised: {at.exception}"
        headers = " | ".join(h.value for h in at.header)
        for section in (
            "Data Input", "Specifications", "Data Preview",
            "Descriptive Statistics", "SPC Analysis", "Capability Analysis",
            "Interpretation", "Export",
        ):
            assert section in headers, f"missing section: {section}"

    def test_metrics_rendered_for_cleaning_summary(self) -> None:
        at = _run_with_data(_representative_values())
        labels = [m.label for m in at.metric]
        for expected in ("Rows loaded", "Valid measurements", "Missing",
                         "Non-numeric", "Non-finite", "Duplicate values"):
            assert expected in labels

    def test_descriptive_metrics_present(self) -> None:
        at = _run_with_data(_representative_values())
        labels = [m.label for m in at.metric]
        assert "N" in labels
        assert any(l.startswith("Mean") for l in labels)
        assert any(l.startswith("Median") for l in labels)
        assert any(l.startswith("Range") for l in labels)

    def test_control_limits_metrics_present(self) -> None:
        at = _run_with_data(_representative_values())
        labels = [m.label for m in at.metric]
        assert "UCL" in labels and "CL" in labels and "LCL" in labels

    def test_unstable_data_surfaces_stability_error(self) -> None:
        """The representative dataset is deliberately out of control."""
        at = _run_with_data(_representative_values())
        errors = " ".join(e.value for e in at.error)
        assert "UNSTABLE" in errors.upper()

    def test_dirty_data_reports_exclusions(self) -> None:
        vals = _representative_values()
        dirty = list(vals) + ["bad", None, float("inf")]
        at = _run_with_data(dirty)
        assert not at.exception
        warnings = " ".join(w.value for w in at.warning)
        assert "excluded" in warnings.lower()

    def test_no_traceback_leaks_to_ui(self) -> None:
        at = _run_with_data(_representative_values())
        blob = " ".join(
            [e.value for e in at.error] + [w.value for w in at.warning]
        )
        assert "Traceback" not in blob
        assert "File \"" not in blob

    def test_insufficient_data_shows_friendly_error(self) -> None:
        at = _run_with_data([10.0])
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "valid measurement" in errors.lower()


class TestDataOrderAttestationUI:
    """A3: the sidebar attestation must reach run_analysis and change output.

    This drives the real widgets (not injected session state), so it verifies
    the UI -> service wiring the service-level tests cannot see.
    """

    def _find_order_checkbox(self, at: AppTest):
        for cb in at.checkbox:
            if "production/time order" in (cb.label or ""):
                return cb
        raise AssertionError(
            "attestation checkbox not found; labels="
            + repr([c.label for c in at.checkbox])
        )

    def _app_with_spec(self) -> AppTest:
        """Load data AND a valid spec so the capability section renders."""
        at = _run_with_data(_representative_values())
        for n in at.number_input:
            if n.label == "USL":
                n.set_value(13.0)
            elif n.label == "LSL":
                n.set_value(8.0)
        at.run()
        assert not at.exception, f"App raised: {at.exception}"
        return at

    def test_checkbox_exists_and_defaults_to_ordered(self) -> None:
        at = _run_with_data(_representative_values())
        cb = self._find_order_checkbox(at)
        assert cb.value is True, "ordered workflow must be the default"

    def test_unchecking_suppresses_cpk_in_ui(self) -> None:
        at = self._app_with_spec()
        # Baseline: Cpk IS reported for time-ordered data.
        assert any("Cpk" == (m.label or "") for m in at.metric)

        self._find_order_checkbox(at).set_value(False).run()
        assert not at.exception, f"App raised after unchecking: {at.exception}"

        # Cp/Cpk must disappear; Ppk must remain.
        labels = [m.label for m in at.metric]
        assert "Cpk" not in labels, "Cpk must not be shown for unordered data"
        assert "Cp" not in labels
        assert "Ppk" in labels, "overall performance must remain available"

        # The user must be told WHY.
        warn_text = " ".join(w.value for w in at.warning)
        assert "Cp/Cpk are not available" in warn_text
        assert "time order" in warn_text
        # Stability is reported as not assessed, not as a verdict.
        info_text = " ".join(i.value for i in at.info)
        assert "Stability not assessed" in info_text

    def test_rechecking_restores_capability(self) -> None:
        at = self._app_with_spec()
        self._find_order_checkbox(at).set_value(False).run()
        self._find_order_checkbox(at).set_value(True).run()
        assert not at.exception
        assert any("Cpk" == (m.label or "") for m in at.metric)
        warn_text = " ".join(w.value for w in at.warning)
        assert "Cp/Cpk are not available" not in warn_text


class TestDataSourceOwnership:
    """C3 regression: a lingering file upload must not clobber the active data.

    These drive the REAL widgets (file uploader, source radio, buttons) instead
    of injecting `session_state`, because injection bypasses the exact code path
    where the bug lived -- which is how it survived 699 tests.

    Original bug: `st.file_uploader` keeps returning the previously uploaded
    file on every rerun, so the unconditional `with tab_file:` block rewrote
    `raw_values`/`cleaning` on ANY later widget interaction, silently discarding
    whatever the user had switched to.

    Note on manual entry: Streamlit's AppTest harness exposes no `data_editor`
    driver, so manual *entry* cannot be typed through the harness. The manual
    dataset is therefore installed exactly as the app's own button handler
    installs it (same session-state keys), and the regression -- survival across
    unrelated reruns -- is then driven entirely through real widgets.
    """

    CSV = b"measurement\n10.0\n10.1\n9.9\n10.2\n10.0\n9.8\n10.3\n10.1\n"
    CSV2 = b"measurement\n20.0\n20.1\n19.9\n20.2\n20.0\n19.8\n20.3\n20.1\n"
    MANUAL = [5.0, 5.1, 4.9, 5.2, 5.0, 4.8, 5.3, 5.1, 5.0, 4.95]

    def _app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=90).run()
        assert not at.exception, f"App raised: {at.exception}"
        return at

    def _radio(self, at: AppTest):
        return at.radio(key="data_source")

    def _upload(self, at: AppTest, content: bytes, name: str = "meas.csv") -> AppTest:
        at.file_uploader[0].upload(name, content, "text/csv")
        at.run()
        assert not at.exception, f"App raised on upload: {at.exception}"
        return at

    def _adopt_manual(self, at: AppTest) -> AppTest:
        """Install manual data exactly as the "Use manual data" handler does."""
        from app.services.data_loader import clean_column

        at.session_state["raw_values"] = list(self.MANUAL)
        at.session_state["source_name"] = "manual entry"
        at.session_state["cleaning"] = clean_column(list(self.MANUAL))
        at.session_state["active_source"] = "manual"
        at.session_state["last_load_id"] = None
        at.run()
        assert not at.exception
        return at

    def _nudge_unrelated(self, at: AppTest, label: str, value: float) -> AppTest:
        """Change a widget that has nothing to do with data loading."""
        for n in at.number_input:
            if n.label == label:
                n.set_value(value)
                break
        else:  # pragma: no cover - defensive
            raise AssertionError(f"number_input {label!r} not found")
        at.run()
        return at

    # -- source selector ----------------------------------------------------
    def test_source_radio_exists_and_defaults_to_file(self) -> None:
        at = self._app()
        assert self._radio(at).value.endswith("File")

    # -- 1. upload loads data ----------------------------------------------
    def test_upload_loads_data(self) -> None:
        at = self._upload(self._app(), self.CSV)
        assert at.session_state["active_source"] == "file"
        assert at.session_state["source_name"] == "meas.csv"
        vals = list(at.session_state["cleaning"].values)
        assert vals == pytest.approx([10.0, 10.1, 9.9, 10.2, 10.0, 9.8, 10.3, 10.1])

    # -- 2. manual becomes active ------------------------------------------
    def test_manual_data_becomes_active(self) -> None:
        at = self._adopt_manual(self._upload(self._app(), self.CSV))
        assert at.session_state["active_source"] == "manual"
        assert list(at.session_state["cleaning"].values) == pytest.approx(self.MANUAL)

    # -- 3. THE regression --------------------------------------------------
    def test_manual_survives_unrelated_widget_change(self) -> None:
        """The C3 regression: an unrelated rerun must not resurrect the file.

        Under the original implementation the lingering uploader re-ran its
        block here and silently restored the file's values.
        """
        at = self._adopt_manual(self._upload(self._app(), self.CSV))
        before = list(at.session_state["cleaning"].values)

        self._nudge_unrelated(at, "USL", 11.0)

        after = list(at.session_state["cleaning"].values)
        assert after == pytest.approx(before), (
            "manual data was silently replaced by the lingering file upload"
        )
        assert at.session_state["active_source"] == "manual"
        assert at.session_state["source_name"] == "manual entry"

    def test_manual_survives_several_unrelated_reruns(self) -> None:
        at = self._adopt_manual(self._upload(self._app(), self.CSV))
        before = list(at.session_state["cleaning"].values)

        self._nudge_unrelated(at, "USL", 11.0)
        self._nudge_unrelated(at, "LSL", 2.0)
        for cb in at.checkbox:
            if "production/time order" in (cb.label or ""):
                cb.set_value(False).run()
                break
        at.run()

        assert not at.exception
        assert list(at.session_state["cleaning"].values) == pytest.approx(before)
        assert at.session_state["active_source"] == "manual"

    def test_file_data_survives_unrelated_change_too(self) -> None:
        """Symmetric guarantee: no spurious reprocessing of the file either."""
        at = self._upload(self._app(), self.CSV)
        before = list(at.session_state["cleaning"].values)
        load_id = at.session_state["last_load_id"]
        self._nudge_unrelated(at, "USL", 11.0)
        assert list(at.session_state["cleaning"].values) == pytest.approx(before)
        assert at.session_state["last_load_id"] == load_id

    # -- 4. switching back to File is explicit ------------------------------
    def test_switching_source_back_to_file_is_explicit(self) -> None:
        at = self._adopt_manual(self._upload(self._app(), self.CSV))
        self._radio(at).set_value("\U0001F4C1 File").run()
        assert not at.exception
        # Selecting the radio alone must not swap the dataset...
        assert at.session_state["active_source"] == "manual"
        info = " ".join(i.value for i in at.info)
        assert "manually entered" in info
        # ...but uploading a file offers an explicit switch, which we take.
        self._upload(at, self.CSV2, name="other.csv")
        assert at.session_state["active_source"] == "manual", (
            "a new upload must not seize ownership without confirmation"
        )
        switch = [b for b in at.button if "instead" in (b.label or "")]
        assert switch, "expected an explicit switch button"
        switch[0].click().run()
        assert at.session_state["active_source"] == "file"
        assert list(at.session_state["cleaning"].values)[0] == pytest.approx(20.0)

    def test_new_file_identity_triggers_reload(self) -> None:
        at = self._upload(self._app(), self.CSV)
        first = list(at.session_state["cleaning"].values)
        self._upload(at, self.CSV2, name="other.csv")
        second = list(at.session_state["cleaning"].values)
        assert second != pytest.approx(first)
        assert second[0] == pytest.approx(20.0)

    # -- indicator ----------------------------------------------------------
    def test_active_source_indicator_is_shown(self) -> None:
        at = self._adopt_manual(self._upload(self._app(), self.CSV))
        caps = " ".join(c.value for c in at.caption)
        assert "Active data source" in caps
        assert "manual entry" in caps

    # -- 5. version/signature only bumps on real change --------------------
    def test_analysis_version_stable_on_idle_rerun(self) -> None:
        at = self._upload(self._app(), self.CSV)
        v1 = at.session_state["analysis_version"]
        at.run()  # nothing changed
        assert at.session_state["analysis_version"] == v1

    def test_analysis_version_bumps_on_real_parameter_change(self) -> None:
        at = self._upload(self._app(), self.CSV)
        v1 = at.session_state["analysis_version"]
        self._nudge_unrelated(at, "USL", 11.25)
        assert at.session_state["analysis_version"] > v1

    def test_analysis_version_bumps_when_dataset_changes(self) -> None:
        at = self._upload(self._app(), self.CSV)
        v1 = at.session_state["analysis_version"]
        self._upload(at, self.CSV2, name="other.csv")
        assert at.session_state["analysis_version"] > v1


class TestInterpretationThresholdControls:
    """All eight interpretation cut-points are adjustable from the sidebar.

    Three of them (the capability bands) were already exposed; the other five
    were reachable only from code. These tests pin that every field now has a
    widget, that editing any of them actually re-runs the analysis, and that an
    incoherent combination is reported instead of taking the page down.
    """

    #: Widget label -> InterpretationThresholds field it drives.
    LABELS = {
        "Marginal ≥": "marginal_min",
        "Capable ≥": "capable_min",
        "Excellent ≥": "excellent_min",
        "Centering gap ratio": "centering_gap_ratio",
        "Drift gap ratio": "drift_gap_ratio",
        "Tight-spec sigma multiple": "spec_margin_tight_sigma",
        "Min observations": "min_observations",
        "Min subgroups": "min_subgroups",
    }

    @staticmethod
    def _set(at: AppTest, label: str, value) -> AppTest:
        for n in at.number_input:
            if n.label == label:
                n.set_value(value)
                at.run()
                return at
        raise AssertionError(
            f"number_input {label!r} not found; labels="
            + repr([n.label for n in at.number_input])
        )

    @staticmethod
    def _version(at: AppTest) -> int:
        """The analysis counter the sidebar renders, e.g. 'Analysis v2 - ...'."""
        for c in at.caption:
            if c.value.startswith("Analysis v"):
                return int(c.value.split("Analysis v")[1].split(" ")[0])
        raise AssertionError("analysis version caption not rendered")

    def test_every_threshold_field_has_a_widget(self) -> None:
        import dataclasses

        from app.interpret.thresholds import InterpretationThresholds

        at = _run_with_data(_representative_values())
        assert not at.exception, f"App raised: {at.exception}"
        rendered = {n.label for n in at.number_input}
        missing = [lbl for lbl in self.LABELS if lbl not in rendered]
        assert not missing, f"threshold widgets missing from the sidebar: {missing}"

        fields = {f.name for f in dataclasses.fields(InterpretationThresholds)}
        assert fields == set(self.LABELS.values()), (
            "InterpretationThresholds fields and sidebar widgets have diverged: "
            f"{sorted(fields ^ set(self.LABELS.values()))}"
        )

    @pytest.mark.parametrize(
        "label,value",
        [
            ("Centering gap ratio", 0.25),
            ("Drift gap ratio", 0.25),
            ("Tight-spec sigma multiple", 8.0),
            ("Min observations", 40),
            ("Min subgroups", 25),
        ],
    )
    def test_newly_exposed_thresholds_invalidate_the_cached_analysis(
        self, label, value
    ) -> None:
        """Editing a threshold must re-run the analysis, not show a stale one.

        These five reach `_signature` only because it hashes the whole dataclass;
        with the previous hand-listed three-field tuple the version would not
        move and section 7 would keep the earlier verdict.
        """
        at = _run_with_data(_representative_values())
        before = self._version(at)
        at = self._set(at, label, value)
        assert not at.exception, f"App raised: {at.exception}"
        assert self._version(at) == before + 1, (
            f"changing {label!r} did not re-run the analysis (version stayed at "
            f"{before}) -> section 7 is stale"
        )

    def test_incoherent_bands_are_reported_and_defaults_used(self) -> None:
        """A typo is ordinary user input: report it, keep the page alive."""
        at = _run_with_data(_representative_values())
        at = self._set(at, "Capable ≥", 0.50)  # below Marginal (1.00)
        assert not at.exception, f"App crashed on invalid thresholds: {at.exception}"
        errors = [e.value for e in at.error]
        assert any("Invalid interpretation thresholds" in e for e in errors), (
            f"no error surfaced for reversed bands; errors={errors}"
        )
        assert any("increase strictly" in e for e in errors), (
            "the error must say WHY it is invalid"
        )
        # The analysis still rendered, using the documented defaults.
        assert any("Interpretation" in h.value for h in at.header)

    def test_valid_alternative_standard_produces_no_error(self) -> None:
        """PPAP-style bands (1.33 / 1.67 / 2.00) must remain expressible."""
        at = _run_with_data(_representative_values())
        at = self._set(at, "Excellent ≥", 2.00)
        at = self._set(at, "Capable ≥", 1.67)
        at = self._set(at, "Marginal ≥", 1.33)
        assert not at.exception, f"App raised: {at.exception}"
        assert not [
            e.value for e in at.error
            if "Invalid interpretation thresholds" in e.value
        ]

