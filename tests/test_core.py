"""Headless numerical and GRU baseline tests."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raypath_scpt import (
    DeviationPoint,
    GRU_PRE_TRIGGER_MS,
    GruFormatError,
    SurveyGeometry,
    WaveformRecord,
    add_suggested_picks,
    calculate_corrected_geometry,
    calculate_ts1170_5_method1_vs30,
    calculate_vs30,
    calculate_waveform_qc,
    depth_aware_regularization_operator,
    forward_model,
    generate_velocity_uncertainty_ensemble,
    gru_deviation_points,
    invert_velocity_profile,
    parse_gru,
    pseudo_interval_velocities,
    solve_direct_ray,
    select_regularization_lcurve,
    suggest_pick_uncertainty_ms,
    suggest_pair_crossover,
    ts1170_5_vs30_band,
    uncertainty_ensemble_classification,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class DirectRayTests(unittest.TestCase):
    def test_homogeneous_profile_matches_straight_line(self) -> None:
        thicknesses = np.asarray([2.0, 3.0, 5.0])
        velocity = 250.0
        offset = 2.4

        ray = solve_direct_ray(thicknesses, np.full(thicknesses.size, velocity), offset)

        expected_time = math.hypot(float(np.sum(thicknesses)), offset) / velocity
        self.assertAlmostEqual(ray.travel_time_s, expected_time, places=12)
        self.assertAlmostEqual(float(np.sum(ray.horizontal_segments_m)), offset, places=10)
        np.testing.assert_allclose(ray.angles_rad, ray.angles_rad[0], rtol=0.0, atol=1.0e-12)

    def test_zero_offset_is_vertical(self) -> None:
        ray = solve_direct_ray([2.0, 3.0], [200.0, 400.0], 0.0)

        self.assertEqual(ray.ray_parameter, 0.0)
        np.testing.assert_array_equal(ray.angles_rad, np.zeros(2))
        np.testing.assert_array_equal(ray.horizontal_segments_m, np.zeros(2))
        self.assertAlmostEqual(ray.travel_time_s, 2.0 / 200.0 + 3.0 / 400.0, places=14)

    def test_high_contrast_near_critical_solution_remains_finite(self) -> None:
        ray = solve_direct_ray([1.0, 4.0], [120.0, 900.0], 40.0)

        self.assertTrue(math.isfinite(ray.travel_time_s))
        self.assertLess(ray.ray_parameter, 1.0 / 900.0)
        self.assertAlmostEqual(float(np.sum(ray.horizontal_segments_m)), 40.0, places=7)

    def test_invalid_models_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_direct_ray([1.0, -1.0], [200.0, 300.0], 2.4)
        with self.assertRaises(ValueError):
            solve_direct_ray([1.0], [200.0], -0.1)


class ForwardAndInversionTests(unittest.TestCase):
    def test_forward_model_returns_every_receiver(self) -> None:
        times, rays = forward_model([2.0, 2.0, 2.0], [180.0, 220.0, 300.0], 2.4)

        self.assertEqual(times.shape, (3,))
        self.assertEqual(len(rays), 3)
        self.assertTrue(np.all(np.diff(times) > 0.0))
        for ray in rays:
            self.assertAlmostEqual(float(np.sum(ray.horizontal_segments_m)), 2.4, places=8)

    def test_exact_synthetic_profile_has_negligible_misfit(self) -> None:
        depths = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])
        velocities = np.asarray([180.0, 210.0, 240.0, 280.0, 320.0])
        thicknesses = np.diff(np.r_[0.0, depths])
        observed, _ = forward_model(thicknesses, velocities, 2.4)

        result = invert_velocity_profile(depths, observed, 2.4, regularization=0.0)

        self.assertLess(result.rmse_s, 1.0e-6)
        np.testing.assert_allclose(result.calculated_times_s, observed, rtol=0.0, atol=2.0e-6)

    def test_pseudo_interval_rejects_nonpositive_time_increment(self) -> None:
        values = pseudo_interval_velocities([1.0, 2.0, 3.0], [0.01, 0.01, 0.03])

        self.assertAlmostEqual(values[0], 100.0)
        self.assertTrue(math.isnan(values[1]))
        self.assertAlmostEqual(values[2], 50.0)

    def test_receiver_specific_offsets_are_used_by_forward_model(self) -> None:
        thicknesses = np.asarray([2.0, 3.0, 4.0])
        velocities = np.asarray([180.0, 240.0, 320.0])
        offsets = np.asarray([2.0, 2.5, 3.0])

        times, rays = forward_model(thicknesses, velocities, 2.0, offsets)

        for index, offset in enumerate(offsets):
            expected = solve_direct_ray(thicknesses[: index + 1], velocities[: index + 1], offset)
            self.assertAlmostEqual(times[index], expected.travel_time_s, places=12)
            self.assertAlmostEqual(float(np.sum(rays[index].horizontal_segments_m)), offset, places=9)

    def test_inversion_fits_receiver_specific_geometry(self) -> None:
        depths = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])
        velocities = np.asarray([180.0, 210.0, 240.0, 280.0, 320.0])
        offsets = np.asarray([2.0, 2.1, 2.25, 2.4, 2.6])
        thicknesses = np.diff(np.r_[0.0, depths])
        observed, _ = forward_model(thicknesses, velocities, 2.0, offsets)

        result = invert_velocity_profile(
            depths,
            observed,
            2.0,
            regularization=0.0,
            receiver_offsets_m=offsets,
        )

        self.assertLess(result.rmse_s, 1.0e-6)
        np.testing.assert_allclose(result.receiver_offsets_m, offsets)


class QuantifiedInversionTests(unittest.TestCase):
    def test_ensemble_counts_are_labelled_by_intended_use(self) -> None:
        self.assertEqual(uncertainty_ensemble_classification(0), "Off")
        self.assertIn("preliminary", uncertainty_ensemble_classification(20).lower())
        self.assertIn("report-quality", uncertainty_ensemble_classification(100).lower())
        self.assertIn("report-quality", uncertainty_ensemble_classification(200).lower())

    def test_depth_aware_curvature_annihilates_linear_log_velocity_on_irregular_centres(self) -> None:
        depths = np.asarray([1.2, 3.9, 7.1, 12.8, 20.0])
        thicknesses = np.diff(np.r_[0.0, depths])
        centres = depths - 0.5 * thicknesses
        log_velocity = 5.1 + 0.025 * centres

        regularized = depth_aware_regularization_operator(depths).dot(log_velocity)

        np.testing.assert_allclose(regularized, 0.0, atol=1.0e-12)

    def test_irregular_spacing_recovers_equivalent_smooth_profile(self) -> None:
        profiles = []
        for depths in (
            np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0]),
            np.asarray([1.5, 3.7, 6.4, 9.8, 12.2, 15.9, 19.6, 22.1, 25.0]),
        ):
            thicknesses = np.diff(np.r_[0.0, depths])
            centres = depths - 0.5 * thicknesses
            expected = np.exp(math.log(170.0) + 0.025 * centres)
            times, _ = forward_model(thicknesses, expected, 2.4)
            model = invert_velocity_profile(depths, times, 2.4, regularization=0.55)
            profiles.append((centres, model.velocities_mps))
            np.testing.assert_allclose(model.velocities_mps, expected, rtol=0.025, atol=2.0)

        comparison_depths = np.linspace(2.0, 23.0, 15)
        first = np.interp(comparison_depths, profiles[0][0], np.log(profiles[0][1]))
        second = np.interp(comparison_depths, profiles[1][0], np.log(profiles[1][1]))
        np.testing.assert_allclose(first, second, atol=0.025)

    def test_large_observation_uncertainty_reduces_outlier_influence(self) -> None:
        depths = np.asarray([3.0, 6.0, 9.0, 12.0, 15.0, 20.0, 25.0])
        expected = np.asarray([180.0, 190.0, 210.0, 230.0, 250.0, 280.0, 320.0])
        times, _ = forward_model(np.diff(np.r_[0.0, depths]), expected, 2.4)
        contaminated = times.copy()
        contaminated[3] += 0.004
        equal = invert_velocity_profile(
            depths,
            contaminated,
            2.4,
            regularization=0.35,
            observation_std_s=np.full(depths.size, 0.0005),
        )
        uncertainties = np.full(depths.size, 0.0005)
        uncertainties[3] = 0.005
        weighted = invert_velocity_profile(
            depths,
            contaminated,
            2.4,
            regularization=0.35,
            observation_std_s=uncertainties,
        )

        equal_error = float(np.linalg.norm(np.log(equal.velocities_mps / expected)))
        weighted_error = float(np.linalg.norm(np.log(weighted.velocities_mps / expected)))
        self.assertLess(weighted_error, equal_error)
        self.assertTrue(math.isfinite(weighted.data_cost_ms2))
        self.assertTrue(math.isfinite(weighted.regularization_cost))
        self.assertEqual(weighted.observation_std_s.size, depths.size)

    def test_huber_loss_flags_but_does_not_delete_outlier(self) -> None:
        depths = np.asarray([3.0, 6.0, 10.0, 15.0, 20.0, 25.0])
        velocities = np.asarray([180.0, 200.0, 220.0, 250.0, 280.0, 320.0])
        times, _ = forward_model(np.diff(np.r_[0.0, depths]), velocities, 2.4)
        times[2] += 0.006

        model = invert_velocity_profile(
            depths,
            times,
            2.4,
            regularization=0.45,
            observation_std_s=np.full(depths.size, 0.0005),
            robust_loss="huber",
        )

        self.assertEqual(model.observed_times_s.size, depths.size)
        self.assertTrue(model.outlier_flags[2])
        self.assertEqual(model.robust_loss, "huber")

    def test_uncertainty_ensemble_is_repeatable_and_propagates_vs30(self) -> None:
        depths = np.asarray([3.5, 7.0, 11.0, 16.0, 21.0, 25.0])
        velocities = np.asarray([170.0, 195.0, 225.0, 260.0, 300.0, 350.0])
        times, _ = forward_model(np.diff(np.r_[0.0, depths]), velocities, 2.4)
        central = invert_velocity_profile(
            depths,
            times,
            2.4,
            regularization=0.35,
            observation_std_s=np.full(depths.size, 0.0002),
        )

        first = generate_velocity_uncertainty_ensemble(central, 2.4, 0.35, ensemble_size=8, random_seed=42)
        second = generate_velocity_uncertainty_ensemble(central, 2.4, 0.35, ensemble_size=8, random_seed=42)

        np.testing.assert_allclose(first.velocity_lower_mps, second.velocity_lower_mps)
        np.testing.assert_allclose(first.velocity_upper_mps, second.velocity_upper_mps)
        np.testing.assert_allclose(first.vs30_samples_mps, second.vs30_samples_mps)
        self.assertIsNotNone(first.vs30_lower_mps)
        self.assertLess(first.vs30_lower_mps, first.vs30_upper_mps)

    def test_lcurve_selection_is_a_repeatable_interior_candidate(self) -> None:
        depths = np.asarray([3.0, 6.0, 10.0, 15.0, 20.0, 25.0])
        velocities = np.asarray([180.0, 205.0, 230.0, 260.0, 295.0, 335.0])
        times, _ = forward_model(np.diff(np.r_[0.0, depths]), velocities, 2.4)
        candidates = (0.0, 0.05, 0.2, 0.6, 1.0)

        selection, model = select_regularization_lcurve(
            depths,
            times,
            2.4,
            observation_std_s=np.full(depths.size, 0.0005),
            candidate_factors=candidates,
        )

        self.assertIn(selection.selected_factor, candidates[1:-1])
        self.assertEqual(model.depths_m.size, depths.size)
        self.assertEqual(selection.candidate_factors.size, len(candidates))


class SurveyGeometryTests(unittest.TestCase):
    @staticmethod
    def complete_geometry() -> SurveyGeometry:
        return SurveyGeometry(
            source_offset_uncertainty_m=0.02,
            coordinate_system="NZTM2000",
            vertical_datum="NZVD2016",
            elevations_enabled=True,
            source_elevation_m=100.0,
            receiver_reference_elevation_m=99.0,
            receiver_depth_reference="CPT ground reference",
            depth_basis="along_rods",
            source_to_receiver_bearing_deg=90.0,
            source_block_axis_bearing_deg=180.0,
            channel_17_strike_bearing_deg=180.0,
            channel_18_strike_bearing_deg=0.0,
            receiver_orientation_bearing_deg=180.0,
            deviation_points=[
                DeviationPoint(10.0, 0.0, None),
                DeviationPoint(20.0, 30.0, 90.0),
            ],
        )

    def test_inclination_elevation_and_azimuth_correct_receiver_coordinates(self) -> None:
        corrected = calculate_corrected_geometry([10.0, 20.0], 2.0, self.complete_geometry())

        np.testing.assert_allclose(
            corrected.vertical_depths_m,
            [11.0, 11.0 + 10.0 * math.cos(math.radians(30.0))],
        )
        np.testing.assert_allclose(corrected.receiver_east_m, [2.0, 7.0], atol=1.0e-12)
        np.testing.assert_allclose(corrected.receiver_north_m, [0.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(corrected.receiver_offsets_m, [2.0, 7.0], atol=1.0e-12)
        self.assertEqual(corrected.warnings, ())
        self.assertTrue(corrected.correction_applied)

    def test_missing_deviation_azimuth_is_visible_and_not_invented(self) -> None:
        geometry = self.complete_geometry()
        geometry.deviation_points[1] = DeviationPoint(20.0, 30.0, None)

        corrected = calculate_corrected_geometry([10.0, 20.0], 2.0, geometry)

        np.testing.assert_allclose(corrected.receiver_offsets_m, [2.0, 2.0], atol=1.0e-12)
        self.assertTrue(any("azimuth is missing" in warning for warning in corrected.warnings))


class Vs30BaselineTests(unittest.TestCase):
    def test_full_profile_is_clipped_at_30_metres(self) -> None:
        result = calculate_vs30([10.0, 20.0, 40.0], [200.0, 400.0, 800.0])

        expected = 30.0 / (10.0 / 200.0 + 10.0 / 400.0 + 10.0 / 800.0)
        self.assertAlmostEqual(result.value_mps, expected, places=10)
        self.assertAlmostEqual(float(np.sum(result.included_thicknesses_m)), 30.0, places=10)
        self.assertEqual(result.extrapolated_thickness_m, 0.0)

    def test_current_experimental_weighting_order_is_preserved(self) -> None:
        depths = [5.0, 10.0, 15.0, 20.0, 25.0]
        velocities = [150.0, 200.0, 250.0, 300.0, 400.0]
        shallow = calculate_vs30(depths, velocities, 0.25)
        neutral = calculate_vs30(depths, velocities, 1.0)
        deep = calculate_vs30(depths, velocities, 4.0)

        self.assertLess(shallow.extrapolated_velocity_mps, neutral.extrapolated_velocity_mps)
        self.assertLess(neutral.extrapolated_velocity_mps, deep.extrapolated_velocity_mps)
        self.assertLess(shallow.value_mps, neutral.value_mps)
        self.assertLess(neutral.value_mps, deep.value_mps)

    def test_profile_shallower_than_25_metres_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 25"):
            calculate_vs30([10.0, 20.0, 24.9], [200.0, 300.0, 400.0])


class Ts1170Method1Vs30Tests(unittest.TestCase):
    def test_scpt_shallow_adjustment_last_layer_extension_and_bounds(self) -> None:
        depths = [2.5, 3.0, 3.5, 10.0, 20.0, 25.0]
        velocities = [100.0, 140.0, 180.0, 200.0, 250.0, 300.0]

        result = calculate_ts1170_5_method1_vs30(depths, velocities)

        # The depth-average from 2.5-3.5 m is (0.5*140 + 0.5*180)/1 = 160 m/s.
        self.assertAlmostEqual(result.shallow_reference_velocity_mps, 160.0, places=12)
        expected_time = 3.0 / 160.0 + 0.5 / 180.0 + 6.5 / 200.0 + 10.0 / 250.0 + 10.0 / 300.0
        expected = 30.0 / expected_time
        self.assertAlmostEqual(result.value_mps, expected, places=12)
        self.assertAlmostEqual(result.lower_bound_mps, expected / 1.05, places=12)
        self.assertAlmostEqual(result.upper_bound_mps, expected * 1.05, places=12)
        self.assertAlmostEqual(result.extrapolated_thickness_m, 5.0, places=12)
        self.assertAlmostEqual(result.extrapolated_velocity_mps, 300.0, places=12)
        self.assertAlmostEqual(float(np.sum(result.included_thicknesses_m)), 30.0, places=12)

    def test_profile_beyond_30_metres_is_clipped_exactly(self) -> None:
        result = calculate_ts1170_5_method1_vs30([3.5, 15.0, 40.0], [180.0, 300.0, 600.0])

        expected = 30.0 / (3.0 / 180.0 + 0.5 / 180.0 + 11.5 / 300.0 + 15.0 / 600.0)
        self.assertAlmostEqual(result.value_mps, expected, places=12)
        self.assertEqual(result.extrapolated_thickness_m, 0.0)
        self.assertAlmostEqual(float(np.sum(result.included_thicknesses_m)), 30.0, places=12)

    def test_exactly_25_metres_is_accepted_but_shallower_profile_is_not(self) -> None:
        accepted = calculate_ts1170_5_method1_vs30([3.5, 25.0], [200.0, 400.0])
        self.assertAlmostEqual(accepted.measured_depth_m, 25.0)
        with self.assertRaisesRegex(ValueError, "at least 25"):
            calculate_ts1170_5_method1_vs30([3.5, 24.999], [200.0, 400.0])

    def test_uncertainty_range_reports_every_intersected_numerical_band(self) -> None:
        result = calculate_ts1170_5_method1_vs30([3.5, 25.0], [190.0, 200.0])

        self.assertEqual(ts1170_5_vs30_band(150.0), "VII")
        self.assertEqual(ts1170_5_vs30_band(200.0), "VI")
        self.assertEqual(ts1170_5_vs30_band(200.01), "V")
        self.assertTrue(result.spans_multiple_vs30_bands)
        self.assertEqual(result.indicative_vs30_bands, ("VI", "V"))

    def test_slow_upper_profile_is_exposed_for_additional_class_vi_review(self) -> None:
        result = calculate_ts1170_5_method1_vs30(
            [3.5, 12.0, 20.0, 25.0],
            [140.0, 140.0, 300.0, 350.0],
        )

        self.assertAlmostEqual(result.slow_velocity_thickness_top20_m, 12.0)
        self.assertTrue(any("Site Class VI" in note for note in result.notes))


class GruImportBaselineTests(unittest.TestCase):
    def test_gru_records_are_sorted_and_trigger_corrected(self) -> None:
        records = parse_gru(FIXTURES / "minimal.GRU")

        self.assertEqual([record.test_number for record in records], [1, 2])
        self.assertEqual([record.depth_m for record in records], [2.0, 4.0])
        np.testing.assert_array_equal(records[0].time_ms, [-50.0, -25.0, 0.0, 25.0, 50.0])
        self.assertEqual(records[0].pre_trigger_ms, GRU_PRE_TRIGGER_MS)
        np.testing.assert_array_equal(records[0].recorded_time_ms, [0.0, 25.0, 50.0, 75.0, 100.0])
        self.assertEqual(records[0].pick_uncertainty_ms, 12.5)
        np.testing.assert_array_equal(records[0].left, [0.0, 2.0, 4.0, 2.0, 0.0])
        np.testing.assert_array_equal(records[0].right, [0.0, -2.0, -4.0, -2.0, 0.0])

    def test_configured_pretrigger_must_be_spanned(self) -> None:
        with self.assertRaises(GruFormatError):
            parse_gru(FIXTURES / "minimal.GRU", pre_trigger_ms=150.0)

    def test_nondefault_pretrigger_preserves_both_time_references(self) -> None:
        record = parse_gru(FIXTURES / "minimal.GRU", pre_trigger_ms=25.0)[0]

        np.testing.assert_array_equal(record.time_ms, [-25.0, 0.0, 25.0, 50.0, 75.0])
        np.testing.assert_array_equal(record.recorded_time_ms, [0.0, 25.0, 50.0, 75.0, 100.0])
        record.set_pick("first_peak", 17, 12.5)
        self.assertAlmostEqual(record.recorded_picks_ms()["first_peak_17"], 37.5)

    def test_negative_pretrigger_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_gru(FIXTURES / "minimal.GRU", pre_trigger_ms=-1.0)

    def test_empty_acquisition_blocks_are_skipped_and_reported(self) -> None:
        content = """[SEISMIC TEST=23  DEPTH=20.01]
[SEISMIC TEST END]
[SEISMIC TEST=27  DEPTH=20.01]
*E:0#17:0#18:0!
*E:50#17:1#18:-1!
*E:100#17:0#18:0!
[SEISMIC TEST END]
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty-block.GRU"
            source.write_text(content, encoding="utf-8")
            messages: list[str] = []

            records = parse_gru(source, import_messages=messages)

        self.assertEqual([record.test_number for record in records], [27])
        self.assertEqual(len(messages), 1)
        self.assertIn("test 23", messages[0])
        self.assertIn("20.01 m", messages[0])

    def test_partially_populated_acquisition_block_remains_an_error(self) -> None:
        content = """[SEISMIC TEST=23  DEPTH=20.01]
*E:0#17:0#18:0!
[SEISMIC TEST END]
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "truncated-block.GRU"
            source.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(GruFormatError, "too few samples"):
                parse_gru(source)

    def test_gru_xy_tilt_is_imported_as_interval_vertical_correction(self) -> None:
        content = """*D:0.5#30:-3#31:4!
*D:1.0#30:-3#31:4!
*D:1.5#30:0#31:0!
*D:2.0#30:0#31:0!
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "inclination.GRU"
            source.write_text(content, encoding="utf-8")
            points = gru_deviation_points(source, [1.0, 2.0])

        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].inclination_deg, 5.0)
        self.assertAlmostEqual(points[1].inclination_deg, 0.0)
        self.assertIsNone(points[0].azimuth_deg)
        corrected = calculate_corrected_geometry(
            [1.0, 2.0],
            2.4,
            SurveyGeometry(deviation_points=points),
        )
        self.assertAlmostEqual(corrected.vertical_depths_m[0], math.cos(math.radians(5.0)))
        self.assertAlmostEqual(corrected.vertical_depths_m[1], math.cos(math.radians(5.0)) + 1.0)


class ArrivalDefinitionTests(unittest.TestCase):
    def test_pair_crossover_is_first_intersection_after_initial_lobes(self) -> None:
        time_ms = np.linspace(-50.0, 45.0, 191)
        phase = np.pi * (time_ms - 5.0) / 10.0
        left = np.where(time_ms >= 5.0, np.sin(phase), 0.0)
        right = -left

        crossover = suggest_pair_crossover(time_ms, left, right)

        self.assertAlmostEqual(crossover, 15.0, delta=0.75)

    def test_suggestions_store_one_pair_crossover_and_trace_specific_picks(self) -> None:
        time_ms = np.linspace(-50.0, 45.0, 191)
        phase = np.pi * (time_ms - 5.0) / 10.0
        left = np.where(time_ms >= 5.0, np.sin(phase), 0.0)
        right = -left
        record = WaveformRecord(1, 2.0, time_ms, left, right)

        add_suggested_picks([record])

        self.assertIsNotNone(record.get_pair_pick("crossover"))
        for kind in ("first_peak", "zero_cross", "max_peak"):
            self.assertIsNotNone(record.get_pick(kind, 17))
            self.assertIsNotNone(record.get_pick(kind, 18))
        self.assertFalse(any(key.startswith("first_cross") for key in record.picks_ms))


class WaveformQcTests(unittest.TestCase):
    @staticmethod
    def reversed_pair() -> WaveformRecord:
        time_ms = np.linspace(-50.0, 80.0, 261)
        phase = np.pi * (time_ms - 5.0) / 10.0
        left = np.where(time_ms >= 5.0, np.sin(phase) * np.exp(-(time_ms - 5.0) / 60.0), 0.0)
        right = -0.8 * left
        record = WaveformRecord(1, 2.0, time_ms, left, right, pick_uncertainty_ms=0.25)
        add_suggested_picks([record])
        return record

    def test_clean_reversed_pair_passes_advisory_qc(self) -> None:
        qc = calculate_waveform_qc(self.reversed_pair())

        self.assertTrue(qc.passes_minimum)
        self.assertTrue(qc.polarity_reversed)
        self.assertGreater(qc.sign_reversed_correlation, 0.99)
        self.assertTrue(math.isinf(qc.snr_left_db))
        self.assertEqual(qc.noise_rms_left, 0.0)
        self.assertEqual(qc.noise_rms_right, 0.0)
        self.assertTrue(qc.sample_interval_consistent)

    def test_same_polarity_pair_raises_visible_warnings_without_rejecting(self) -> None:
        record = self.reversed_pair()
        record.right = record.left.copy()
        add_suggested_picks([record], overwrite=True)

        qc = calculate_waveform_qc(record)

        self.assertIn("first peak/trough polarity is not reversed", qc.warnings)
        self.assertTrue(any("correlation" in warning for warning in qc.warnings))
        self.assertFalse(record.is_excluded)

    def test_qc_uncertainty_suggestion_has_auditable_floor_and_basis(self) -> None:
        time_ms = np.linspace(-10.0, 30.0, 161)
        signal = np.sin(2.0 * np.pi * (time_ms - 4.0) / 8.0) * (time_ms >= 4.0)
        record = WaveformRecord(1, 2.0, time_ms, signal, signal.copy())
        add_suggested_picks([record])

        suggestion, basis = suggest_pick_uncertainty_ms(record)

        self.assertGreaterEqual(suggestion, 0.5 * float(np.median(np.diff(time_ms))))
        self.assertIn("one standard deviation", basis)
        self.assertEqual(record.pick_uncertainty_source, "automatic_qc_default")

    def test_rejection_is_an_explicit_analyst_state(self) -> None:
        record = self.reversed_pair()
        self.assertFalse(record.is_excluded)

        record.review_state = "rejected"

        self.assertTrue(record.is_excluded)


if __name__ == "__main__":
    unittest.main()
