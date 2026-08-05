"""Offscreen project-persistence and version baseline tests."""

from __future__ import annotations

import json
import os
import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from raypath_scpt import (
    APP_VERSION,
    DeviationPoint,
    GruImportOptionsDialog,
    InversionResult,
    InversionBatchResult,
    InversionWorker,
    PICK_SEQUENCE,
    PROJECT_SCHEMA_VERSION,
    RayPathMainWindow,
    SurveyGeometry,
    SurveyGeometryDialog,
    WaveformPickerDialog,
    add_suggested_picks,
    apply_application_theme,
    parse_gru,
    forward_model,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROJECT_ROOT = FIXTURES.parent.parent


class ProjectStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_manual_project_round_trip_records_versions(self) -> None:
        source = RayPathMainWindow()
        loaded = RayPathMainWindow()
        try:
            source.offset_spin.setValue(3.25)
            source.reg_slider.setValue(42)
            source._set_all_pick_rows(
                [
                    (
                        2.0,
                        {"first_peak": 12.1, "crossover": 12.4, "zero_cross": 12.6, "max_peak": 13.0},
                    ),
                    (
                        4.0,
                        {"first_peak": 21.3, "crossover": 21.8, "zero_cross": 22.0, "max_peak": 22.5},
                    ),
                ]
            )

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "round_trip.rpscpt"
                source.project_path = path
                self.assertTrue(source.save_project())

                raw_payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(raw_payload["application_version"], APP_VERSION)
                self.assertEqual(raw_payload["schema_version"], PROJECT_SCHEMA_VERSION)
                self.assertEqual(raw_payload["version"], PROJECT_SCHEMA_VERSION)

                loaded._load_project(path)
                self.assertAlmostEqual(loaded.offset_spin.value(), 3.25)
                self.assertEqual(loaded.reg_slider.value(), 42)
                self.assertEqual(loaded.input_table.item(0, 0).text(), "2.000")
                self.assertEqual(loaded.input_table.item(0, 1).text(), "12.100")
                self.assertEqual(loaded.input_table.item(0, 2).text(), "12.400")
                self.assertEqual(loaded.input_table.item(1, 4).text(), "22.500")
        finally:
            source._set_dirty(False)
            loaded._set_dirty(False)
            source.close()
            loaded.close()

    def test_picker_exposes_the_seven_step_workflow(self) -> None:
        records = parse_gru(FIXTURES / "minimal.GRU")
        dialog = WaveformPickerDialog(records, "minimal.GRU")
        try:
            self.assertEqual(len(PICK_SEQUENCE), 7)
            self.assertEqual(len(dialog.marker_buttons), 7)
            self.assertIn("[0/7]", dialog.record_list.item(0).text())
            self.assertEqual(dialog.active_kind, "first_peak")
            self.assertEqual(dialog.active_channel, 17)
            self.assertEqual(dialog.accept_current_button.text(), "Accept Current Picks")
            self.assertAlmostEqual(dialog.zoom_half_width_spin.value(), 20.0)
            self.assertAlmostEqual(dialog.canvas.axes.get_xlim()[1] - dialog.canvas.axes.get_xlim()[0], 40.0)

            dialog.zoom_half_width_spin.setValue(10.0)

            self.assertAlmostEqual(dialog.max_peak_half_width_ms, 10.0)
            self.assertAlmostEqual(dialog.canvas.axes.get_xlim()[1] - dialog.canvas.axes.get_xlim()[0], 20.0)
        finally:
            dialog.close()

    def test_light_theme_is_default_and_switch_updates_all_canvases(self) -> None:
        apply_application_theme(self.app, dark_mode=False)
        window = RayPathMainWindow()
        try:
            self.assertFalse(window.dark_mode_toggle.isChecked())
            self.assertFalse(window.dark_mode)
            self.assertEqual(window.velocity_canvas.axes_color, "#ffffff")

            window.dark_mode_toggle.setChecked(True)

            self.assertTrue(window.dark_mode)
            self.assertTrue(bool(self.app.property("dark_mode")))
            for canvas in (
                window.velocity_canvas,
                window.ray_canvas,
                window.fit_canvas,
                window.waterfall_canvas,
                window.vs30_canvas,
            ):
                self.assertTrue(canvas.dark_mode)
                self.assertEqual(canvas.axes_color, "#0d1117")
        finally:
            window.dark_mode_toggle.setChecked(False)
            window._set_dirty(False)
            window.close()

    def test_ts_method1_is_primary_and_experimental_weighting_is_isolated(self) -> None:
        window = RayPathMainWindow()
        try:
            depths = np.asarray([3.5, 10.0, 20.0, 25.0])
            velocities = np.asarray([160.0, 210.0, 280.0, 360.0])
            count = depths.size
            result = InversionResult(
                depths_m=depths,
                thicknesses_m=np.diff(np.r_[0.0, depths]),
                velocities_mps=velocities,
                observed_times_s=np.linspace(0.02, 0.10, count),
                calculated_times_s=np.linspace(0.02, 0.10, count),
                residuals_s=np.zeros(count),
                ray_parameters=np.zeros(count),
                ray_x_segments=[np.zeros(index + 1) for index in range(count)],
                receiver_offsets_m=np.full(count, 2.4),
                rmse_s=0.0,
                success=True,
                message="Synthetic GUI regression",
                iterations=1,
            )
            window.comparison_results = {"first_peak": result}
            window._update_all_vs30_results(record_history=True)
            primary_neutral = window.comparison_vs30["first_peak"]
            experimental_neutral = window.comparison_experimental_vs30["first_peak"]

            window.extrapolation_weight_slider.blockSignals(True)
            window.extrapolation_weight_slider.setValue(100)
            window.extrapolation_weight_slider.blockSignals(False)
            window._update_all_vs30_results(record_history=False)
            primary_deep = window.comparison_vs30["first_peak"]
            experimental_deep = window.comparison_experimental_vs30["first_peak"]

            self.assertAlmostEqual(primary_neutral.value_mps, primary_deep.value_mps, places=12)
            self.assertNotAlmostEqual(experimental_neutral.value_mps, experimental_deep.value_mps, places=6)
            self.assertEqual(primary_deep.uncertainty_factor, 0.05)
            self.assertIn("TS 1170.5:2025 Method 1", window._project_payload()["vs30_primary_method"])
        finally:
            window._set_dirty(False)
            window.close()

    def test_wp04_settings_round_trip_and_worker_returns_repeatable_ensemble(self) -> None:
        source = RayPathMainWindow()
        loaded = RayPathMainWindow()
        try:
            source.auto_regularization_check.setChecked(True)
            source.robust_loss_combo.setCurrentIndex(source.robust_loss_combo.findData("huber"))
            source.manual_uncertainty_spin.setValue(0.75)
            source.ensemble_size_spin.setValue(6)
            source.uncertainty_seed_spin.setValue(24680)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "wp04_settings.rpscpt"
                path.write_text(json.dumps(source._project_payload()), encoding="utf-8")
                loaded._load_project(path)

            self.assertTrue(loaded.auto_regularization_check.isChecked())
            self.assertEqual(loaded.robust_loss_combo.currentData(), "huber")
            self.assertAlmostEqual(loaded.manual_uncertainty_spin.value(), 0.75)
            self.assertEqual(loaded.ensemble_size_spin.value(), 6)
            self.assertEqual(loaded.ensemble_preset_combo.currentData(), -1)
            self.assertEqual(loaded.uncertainty_seed_spin.value(), 24680)

            depths = np.asarray([3.5, 7.0, 11.0, 16.0, 21.0, 25.0])
            velocities = np.asarray([170.0, 195.0, 225.0, 260.0, 300.0, 350.0])
            times, _ = forward_model(np.diff(np.r_[0.0, depths]), velocities, 2.4)
            worker = InversionWorker(
                depths,
                {"crossover": times},
                2.4,
                np.full(depths.size, 2.4),
                0.35,
                np.full(depths.size, 0.0002),
                "huber",
                "crossover",
                True,
                6,
                24680,
            )
            batches = []
            failures = []
            worker.finished.connect(batches.append)
            worker.failed.connect(lambda message, detail: failures.append((message, detail)))
            worker.run()

            self.assertFalse(failures)
            self.assertEqual(len(batches), 1)
            self.assertIsInstance(batches[0], InversionBatchResult)
            self.assertIn("crossover", batches[0].uncertainty_results)
            self.assertEqual(batches[0].uncertainty_results["crossover"].random_seed, 24680)
            self.assertIsNotNone(batches[0].regularization_selection)
        finally:
            source._set_dirty(False)
            loaded._set_dirty(False)
            source.close()
            loaded.close()

    def test_uncertainty_presets_default_off_and_select_preview_or_final_counts(self) -> None:
        window = RayPathMainWindow()
        try:
            self.assertEqual(window.ensemble_preset_combo.currentData(), 0)
            self.assertEqual(window.ensemble_size_spin.value(), 0)

            window.ensemble_preset_combo.setCurrentIndex(window.ensemble_preset_combo.findData(20))
            self.assertEqual(window.ensemble_size_spin.value(), 20)
            self.assertTrue(window.ensemble_size_spin.isHidden())

            window.ensemble_preset_combo.setCurrentIndex(window.ensemble_preset_combo.findData(200))
            self.assertEqual(window.ensemble_size_spin.value(), 200)

            window.ensemble_preset_combo.setCurrentIndex(window.ensemble_preset_combo.findData(-1))
            self.assertEqual(window.ensemble_size_spin.value(), 50)
            self.assertFalse(window.ensemble_size_spin.isHidden())
        finally:
            window._set_dirty(False)
            window.close()

    def test_accept_current_picks_marks_reviewed_and_advances(self) -> None:
        records = parse_gru(FIXTURES / "minimal.GRU")
        add_suggested_picks(records)
        records[0].review_comment = "Synthetic fixture warnings reviewed."
        dialog = WaveformPickerDialog(records, "minimal.GRU")
        try:
            dialog._accept_current_picks()

            self.assertEqual(records[0].review_state, "accepted_with_comment")
            self.assertIsNotNone(records[0].pick_uncertainty_ms)
            self.assertEqual(dialog.record_list.currentRow(), 1)
            self.assertIn("ACCEPTED + NOTE", dialog.record_list.item(0).text())
        finally:
            dialog.close()

    def test_waveform_review_timing_and_geometry_round_trip_in_schema_seven(self) -> None:
        source = RayPathMainWindow()
        loaded = RayPathMainWindow()
        try:
            source.waveform_records = parse_gru(FIXTURES / "minimal.GRU", pre_trigger_ms=25.0)
            add_suggested_picks(source.waveform_records)
            source.gru_path = (FIXTURES / "minimal.GRU").resolve()
            source.gru_pre_trigger_ms = 25.0
            source.survey_geometry = SurveyGeometry(
                source_offset_uncertainty_m=0.02,
                coordinate_system="NZTM2000",
                vertical_datum="NZVD2016",
                elevations_enabled=True,
                source_elevation_m=10.0,
                receiver_reference_elevation_m=10.0,
                receiver_depth_reference="CPT ground reference",
                depth_basis="along_rods",
                source_to_receiver_bearing_deg=0.0,
                source_block_axis_bearing_deg=90.0,
                channel_17_strike_bearing_deg=90.0,
                channel_18_strike_bearing_deg=270.0,
                receiver_orientation_bearing_deg=90.0,
                deviation_points=[DeviationPoint(2.0), DeviationPoint(4.0)],
            )
            source.waveform_records[0].review_state = "accepted_with_comment"
            source.waveform_records[0].review_comment = "Reviewed despite a synthetic-fixture warning."
            source.waveform_records[0].pick_uncertainty_ms = 0.4
            source.waveform_records[1].review_state = "rejected"
            source.waveform_records[1].review_comment = "Synthetic rejection for persistence coverage."
            source._populate_table_from_picks()

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "qc_round_trip.rpscpt"
                source.project_path = path
                self.assertTrue(source.save_project())
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("qc_metrics", payload["picks"][0])
                self.assertEqual(payload["gru_pre_trigger_ms"], 25.0)
                self.assertEqual(payload["pick_time_reference"], "relative_to_trigger")
                self.assertIn("picks_trigger_relative_ms", payload["picks"][0])
                self.assertIn("picks_recorded_ms", payload["picks"][0])
                self.assertEqual(payload["survey_geometry"]["coordinate_system"], "NZTM2000")
                self.assertEqual(payload["geometry_audit"]["corrected_vertical_depths_m"], [2.0, 4.0])
                self.assertEqual(payload["geometry_audit"]["applicability_warnings"], [])
                trigger_pick = payload["picks"][0]["picks_trigger_relative_ms"]["first_peak_17"]
                recorded_pick = payload["picks"][0]["picks_recorded_ms"]["first_peak_17"]
                self.assertAlmostEqual(recorded_pick - trigger_pick, 25.0)
                loaded._load_project(path)

            self.assertEqual(loaded.waveform_records[0].review_state, "accepted_with_comment")
            self.assertEqual(
                loaded.waveform_records[0].review_comment,
                "Reviewed despite a synthetic-fixture warning.",
            )
            self.assertAlmostEqual(loaded.waveform_records[0].pick_uncertainty_ms, 0.4)
            self.assertAlmostEqual(loaded.gru_pre_trigger_ms, 25.0)
            self.assertEqual(loaded.survey_geometry.coordinate_system, "NZTM2000")
            self.assertEqual(len(loaded.survey_geometry.deviation_points), 2)
            np.testing.assert_array_equal(
                loaded.waveform_records[0].recorded_time_ms,
                [0.0, 25.0, 50.0, 75.0, 100.0],
            )
            self.assertTrue(loaded.waveform_records[1].is_excluded)
            self.assertTrue(loaded.review_waveforms_button.isEnabled())
        finally:
            source._set_dirty(False)
            loaded._set_dirty(False)
            source.close()
            loaded.close()

    def test_gru_import_dialog_exposes_the_fifty_ms_default(self) -> None:
        dialog = GruImportOptionsDialog("minimal.GRU")
        try:
            self.assertAlmostEqual(dialog.pre_trigger_ms, 50.0)
            self.assertIn("trigger-relative 0.000 ms", dialog.audit_label.text())
        finally:
            dialog.close()

    def test_geometry_dialog_calculates_and_accepts_complete_survey(self) -> None:
        geometry = SurveyGeometry(
            source_offset_uncertainty_m=0.02,
            coordinate_system="NZTM2000",
            vertical_datum="NZVD2016",
            elevations_enabled=True,
            source_elevation_m=10.0,
            receiver_reference_elevation_m=10.0,
            receiver_depth_reference="CPT ground reference",
            source_to_receiver_bearing_deg=0.0,
            source_block_axis_bearing_deg=90.0,
            channel_17_strike_bearing_deg=90.0,
            channel_18_strike_bearing_deg=270.0,
            receiver_orientation_bearing_deg=90.0,
            deviation_points=[DeviationPoint(2.0), DeviationPoint(4.0)],
        )
        dialog = SurveyGeometryDialog(geometry, [2.0, 4.0], 2.4)
        try:
            dialog._save_geometry()

            self.assertIsNotNone(dialog.corrected_geometry)
            self.assertEqual(dialog.corrected_geometry.warnings, ())
            np.testing.assert_allclose(dialog.corrected_geometry.vertical_depths_m, [2.0, 4.0])
            np.testing.assert_allclose(dialog.corrected_geometry.receiver_offsets_m, [2.4, 2.4])
        finally:
            dialog.close()

    def test_rejected_waveform_depth_is_omitted_from_inversion_input(self) -> None:
        window = RayPathMainWindow()
        try:
            records = parse_gru(FIXTURES / "minimal.GRU")
            third = copy.deepcopy(records[1])
            third.test_number = 3
            third.depth_m = 6.0
            records.append(third)
            add_suggested_picks(records)
            for record in records:
                record.review_state = "accepted"
            records[1].review_state = "rejected"
            window.waveform_records = records
            window._populate_table_from_picks()

            depths, times = window._read_all_pick_rows()

            self.assertEqual(depths.tolist(), [2.0, 6.0])
            self.assertEqual(set(times), {"first_peak", "crossover", "zero_cross", "max_peak"})
            self.assertTrue(window.input_table.item(1, 0).font().strikeOut())
        finally:
            window._set_dirty(False)
            window.close()

    def test_waveform_qc_companion_csv_records_exclusions(self) -> None:
        window = RayPathMainWindow()
        try:
            records = parse_gru(FIXTURES / "minimal.GRU")
            add_suggested_picks(records)
            records[0].review_state = "accepted"
            records[1].review_state = "rejected"
            records[1].review_comment = "Excluded in CSV regression test."
            window.waveform_records = records

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "waveform_qc.csv"
                window._export_waveform_qc_csv(path)
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["Included in Inversion"], "Yes")
            self.assertEqual(rows[1]["Included in Inversion"], "No")
            self.assertEqual(rows[1]["Analyst Comment"], "Excluded in CSV regression test.")
        finally:
            window._set_dirty(False)
            window.close()

    def test_saved_rejection_survives_when_raw_gru_is_unavailable(self) -> None:
        window = RayPathMainWindow()
        try:
            inputs = []
            for depth, state in ((2.0, "accepted"), (4.0, "rejected"), (6.0, "accepted")):
                inputs.append(
                    {
                        "depth_m": str(depth),
                        "first_peak_ms": str(5.0 + depth),
                        "crossover_ms": str(5.2 + depth),
                        "zero_cross_ms": str(5.4 + depth),
                        "max_peak_ms": str(5.6 + depth),
                        "review_state": state,
                        "review_comment": "Unavailable-source exclusion" if state == "rejected" else "",
                        "pick_uncertainty_ms": 0.2,
                    }
                )
            payload = {
                "format": "RayPath SCPT Project",
                "version": PROJECT_SCHEMA_VERSION,
                "schema_version": PROJECT_SCHEMA_VERSION,
                "units": "SI",
                "arrival_estimator": "crossover",
                "gru_source": "Z:/unavailable/raw.GRU",
                "inputs": inputs,
                "picks": [],
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "missing_raw.rpscpt"
                path.write_text(json.dumps(payload), encoding="utf-8")
                window._load_project(path)

            depths, _times = window._read_all_pick_rows()
            self.assertEqual(depths.tolist(), [2.0, 6.0])
            self.assertEqual(window.observation_review[4.0]["review_state"], "rejected")
            self.assertTrue(window.input_table.item(1, 0).font().strikeOut())
        finally:
            window._set_dirty(False)
            window.close()

    def test_legacy_version_field_remains_supported(self) -> None:
        window = RayPathMainWindow()
        try:
            payload = {
                "format": "RayPath SCPT Project",
                "version": 3,
                "units": "SI",
                "source_offset_m": 2.4,
                "regularization": 0.35,
                "inputs": [
                    {
                        "depth_m": "2.0",
                        "first_peak_ms": "10.0",
                        "first_cross_ms": "10.1",
                        "max_peak_ms": "10.2",
                    },
                    {
                        "depth_m": "4.0",
                        "first_peak_ms": "20.0",
                        "first_cross_ms": "20.1",
                        "max_peak_ms": "20.2",
                    },
                ],
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "legacy.rpscpt"
                path.write_text(json.dumps(payload), encoding="utf-8")
                window._load_project(path)

            self.assertEqual(window.input_table.item(1, 2).text(), "")
            self.assertEqual(window.input_table.item(1, 3).text(), "20.100")
            depths, times = window._read_all_pick_rows()
            self.assertEqual(depths.tolist(), [2.0, 4.0])
            self.assertEqual(set(times), {"first_peak", "zero_cross", "max_peak"})
        finally:
            window._set_dirty(False)
            window.close()

    def test_reviewed_gru_project_fixture_preserves_picks(self) -> None:
        window = RayPathMainWindow()
        previous_directory = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT)
            window._load_project(FIXTURES / "reviewed_minimal_v3.rpscpt")

            self.assertEqual(len(window.waveform_records), 2)
            self.assertAlmostEqual(window.waveform_records[0].get_pick("zero_cross", 17), 12.3)
            self.assertIsNone(window.waveform_records[0].get_pair_pick("crossover"))
            self.assertEqual(window.waveform_records[0].review_state, "not_reviewed")
            self.assertAlmostEqual(window.waveform_records[1].get_pick("max_peak", 18), 23.2)
            self.assertEqual(window.input_table.item(0, 2).text(), "")
            self.assertEqual(window.input_table.item(0, 3).text(), "12.500")
        finally:
            os.chdir(previous_directory)
            window._set_dirty(False)
            window.close()

    def test_version_one_raw_gru_picks_are_migrated_to_trigger_time(self) -> None:
        window = RayPathMainWindow()
        try:
            payload = {
                "format": "RayPath SCPT Project",
                "version": 1,
                "units": "SI",
                "source_offset_m": 2.4,
                "regularization": 0.35,
                "arrival_estimator": "first_cross",
                "gru_source": str(FIXTURES / "minimal.GRU"),
                "inputs": [
                    {"depth_m": "2.0", "arrival_time_ms": "62.5"},
                    {"depth_m": "4.0", "arrival_time_ms": "72.5"},
                ],
                "picks": [
                    {
                        "test_number": 1,
                        "depth_m": 2.0,
                        "picks_ms": {
                            "first_peak_17": 61.8,
                            "first_peak_18": 62.2,
                            "first_cross_17": 62.3,
                            "first_cross_18": 62.7,
                            "max_peak_17": 62.8,
                            "max_peak_18": 63.2,
                        },
                    },
                    {
                        "test_number": 2,
                        "depth_m": 4.0,
                        "picks_ms": {
                            "first_peak_17": 71.8,
                            "first_peak_18": 72.2,
                            "first_cross_17": 72.3,
                            "first_cross_18": 72.7,
                            "max_peak_17": 72.8,
                            "max_peak_18": 73.2,
                        },
                    },
                ],
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "version_one.rpscpt"
                path.write_text(json.dumps(payload), encoding="utf-8")
                window._load_project(path)

            self.assertAlmostEqual(window.waveform_records[0].get_pick("zero_cross", 17), 12.3)
            self.assertIsNone(window.waveform_records[0].get_pair_pick("crossover"))
            self.assertEqual(window.input_table.item(0, 2).text(), "")
            self.assertEqual(window.input_table.item(0, 3).text(), "12.500")
        finally:
            window._set_dirty(False)
            window.close()

    def test_unsupported_schema_is_rejected(self) -> None:
        window = RayPathMainWindow()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "future.rpscpt"
                path.write_text(
                    json.dumps(
                        {
                            "format": "RayPath SCPT Project",
                            "version": PROJECT_SCHEMA_VERSION,
                            "schema_version": PROJECT_SCHEMA_VERSION + 1,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "not a supported"):
                    window._load_project(path)
        finally:
            window._set_dirty(False)
            window.close()


if __name__ == "__main__":
    unittest.main()
