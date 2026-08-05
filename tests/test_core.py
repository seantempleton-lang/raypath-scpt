"""Headless numerical and GRU baseline tests."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from raypath_scpt import (
    GRU_PRE_TRIGGER_MS,
    GruFormatError,
    WaveformRecord,
    add_suggested_picks,
    calculate_vs30,
    calculate_waveform_qc,
    forward_model,
    invert_velocity_profile,
    parse_gru,
    pseudo_interval_velocities,
    solve_direct_ray,
    suggest_pair_crossover,
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

    def test_rejection_is_an_explicit_analyst_state(self) -> None:
        record = self.reversed_pair()
        self.assertFalse(record.is_excluded)

        record.review_state = "rejected"

        self.assertTrue(record.is_excluded)


if __name__ == "__main__":
    unittest.main()
