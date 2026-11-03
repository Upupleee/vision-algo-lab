from __future__ import annotations

import sys
import unittest
from inspect import signature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from traditional.template_matching.ncc_matching import (
    find_best_matches,
    match_template_ncc,
    parabolic_subpixel,
)
from traditional.template_matching.pose_ncc import match_pose_ncc


def make_template(shape=(33, 41), seed=731):
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(size=shape), 1.1)
    return 110.0 + 28.0 * texture / texture.std()


def affine_truth(template, center, angle=0.0, scale=1.0):
    theta = np.deg2rad(angle)
    linear = scale * np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])
    h, w = template.shape
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = np.asarray(center) - linear @ np.array([(w - 1) / 2, (h - 1) / 2])
    return transform


def render_target(image, template, transform):
    yy, xx = np.indices(image.shape, dtype=float)
    points = np.stack([xx.ravel(), yy.ravel(), np.ones(image.size)])
    source = np.linalg.solve(transform, points)
    h, w = template.shape
    inside = ((source[0] >= 0) & (source[0] <= w - 1)
              & (source[1] >= 0) & (source[1] <= h - 1))
    sampled = map_coordinates(template, [source[1], source[0]], order=1, mode="nearest")
    result = image.copy()
    result.flat[np.flatnonzero(inside)] = sampled[inside]
    return result


class PoseNCCTest(unittest.TestCase):
    def setUp(self):
        self.template = make_template()
        self.background = np.random.default_rng(932).normal(105.0, 4.0, (145, 181))

    def scene(self, center=(86.0, 70.0), angle=0.0, scale=1.0):
        truth = affine_truth(self.template, center, angle, scale)
        return render_target(self.background, self.template, truth), truth

    def assert_pose(self, match, truth, angle=0.0, scale=1.0, tolerance=0.8):
        h, w = self.template.shape
        points = np.array([[0, 0, 1], [w - 1, 0, 1],
                           [w - 1, h - 1, 1], [0, h - 1, 1]])
        expected_corners = (points @ truth.T)[:, :2]
        expected_center = truth @ [(w - 1) / 2, (h - 1) / 2, 1]
        actual = np.asarray(match["transform"])
        self.assertEqual(actual.shape, (3, 3))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual[2], [0, 0, 1], atol=0, rtol=0)
        np.testing.assert_allclose(actual[:2, :2], truth[:2, :2], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual[:2, 2], truth[:2, 2], atol=tolerance, rtol=0)
        np.testing.assert_allclose(match["corners"], expected_corners, atol=tolerance, rtol=0)
        np.testing.assert_allclose([match["center_x"], match["center_y"]],
                                   expected_center[:2], atol=tolerance, rtol=0)
        np.testing.assert_allclose([match["x"], match["y"]], actual[:2, 2], atol=1e-12)
        np.testing.assert_allclose([match["bbox_x"], match["bbox_y"]],
                                   expected_corners.min(axis=0), atol=tolerance, rtol=0)
        self.assertAlmostEqual(match["angle_deg"], angle, places=10)
        self.assertAlmostEqual(match["scale"], scale, places=10)
        self.assertTrue(np.isfinite(match["score"]))
        self.assertGreaterEqual(match["score"], -1.0)
        self.assertLessEqual(match["score"], 1.0)

    def assert_matched(self, result, count=1):
        self.assertTrue(result["matched"], result)
        self.assertEqual(result["reason"], "matched")
        self.assertEqual(len(result["matches"]), count)
        self.assertTrue(np.isfinite(result["elapsed_ms"]))
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertGreater(result["searched_poses"], 0)
        self.assertAlmostEqual(result["best_score"], result["matches"][0]["score"])
        if result["second_score"] is not None:
            self.assertGreaterEqual(result["best_score"], result["second_score"])
            self.assertAlmostEqual(result["score_gap"], result["best_score"] - result["second_score"])

    def assert_rejected(self, result, reason):
        self.assertFalse(result["matched"], result)
        self.assertEqual(result["reason"], reason)
        self.assertEqual(result["matches"], [])

    def test_default_interface_and_identity(self):
        image, truth = self.scene()
        result = match_pose_ncc(image, self.template)
        self.assertEqual(set(result), {"matched", "reason", "matches", "best_score",
                                       "second_score", "score_gap", "elapsed_ms", "searched_poses"})
        self.assert_matched(result)
        self.assertEqual(result["searched_poses"], 1)
        self.assertGreater(result["best_score"], 0.999999)
        self.assert_pose(result["matches"][0], truth, tolerance=0.1)

    def test_rotation_and_scale_search(self):
        for angle, scale in [(-30.0, 0.8), (-15.0, 1.2), (15.0, 0.8), (30.0, 1.2)]:
            with self.subTest(angle=angle, scale=scale):
                image, truth = self.scene(angle=angle, scale=scale)
                result = match_pose_ncc(image, self.template, angle_range=(-30, 30),
                                        angle_step=15, scale_range=(0.8, 1.2), scale_step=0.2)
                self.assert_matched(result)
                self.assertEqual(result["searched_poses"], 15)
                self.assertGreater(result["best_score"], 0.85)
                self.assert_pose(result["matches"][0], truth, angle, scale)

    def test_grid_includes_nondivisible_upper_bounds(self):
        image, truth = self.scene(angle=17, scale=1.15)
        result = match_pose_ncc(image, self.template, angle_range=(0, 17), angle_step=10,
                                scale_range=(0.9, 1.15), scale_step=0.2)
        self.assert_matched(result)
        self.assertEqual(result["searched_poses"], 9)
        self.assert_pose(result["matches"][0], truth, 17, 1.15)

    def test_fractional_translation_subpixel(self):
        image, truth = self.scene(center=(86.3, 70.35))
        refined = match_pose_ncc(image, self.template, pyramid_levels=1)
        integer = match_pose_ncc(image, self.template, pyramid_levels=1, subpixel=False)
        self.assert_matched(refined)
        self.assert_matched(integer)
        self.assert_pose(refined["matches"][0], truth, tolerance=0.2)
        a = np.asarray(refined["matches"][0]["transform"])[:2, 2]
        b = np.asarray(integer["matches"][0]["transform"])[:2, 2]
        self.assertLess(np.linalg.norm(a - truth[:2, 2]), np.linalg.norm(b - truth[:2, 2]))
        for key in ("bbox_x", "bbox_y"):
            self.assertEqual(integer["matches"][0][key] % 1, 0)

    def test_affine_illumination_and_noise(self):
        clean, truth = self.scene(angle=-15, scale=1.2)
        for gain, offset, noise in [(0.55, 70.0, 0.0), (1.6, -35.0, 5.0)]:
            with self.subTest(gain=gain, offset=offset, noise=noise):
                image = gain * clean + offset + np.random.default_rng(17).normal(0, noise, clean.shape)
                result = match_pose_ncc(image, self.template, angle_range=(-30, 0),
                                        angle_step=15, scale_range=(1.0, 1.2), scale_step=0.2)
                self.assert_matched(result)
                self.assertGreater(result["best_score"], 0.9)
                self.assert_pose(result["matches"][0], truth, -15, 1.2)

    def test_constant_neighbor_windows_keep_subpixel_pose_finite(self):
        self.template = np.zeros((17, 19))
        h, w = self.template.shape
        x, y = 15, 13
        truth = affine_truth(self.template, (x + (w - 1) / 2, y + (h - 1) / 2))
        for row, col in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            self.template.fill(0)
            self.template[row, col] = 1
            image = np.zeros((45, 49))
            image[y:y + h, x:x + w] = self.template
            dy = 1 if row == 0 else -1
            dx = 1 if col == 0 else -1
            self.assertEqual(np.ptp(image[y + dy:y + dy + h, x:x + w]), 0)
            self.assertEqual(np.ptp(image[y:y + h, x + dx:x + dx + w]), 0)
            for levels in (1, 3):
                with self.subTest(row=row, col=col, levels=levels):
                    with np.errstate(invalid="raise", divide="raise", over="raise"):
                        refined = match_pose_ncc(image, self.template, pyramid_levels=levels)
                        integer = match_pose_ncc(image, self.template, pyramid_levels=levels,
                                                 subpixel=False)
                    for result in (refined, integer):
                        self.assert_matched(result)
                        self.assertGreater(result["best_score"], 0.999999)
                        self.assert_pose(result["matches"][0], truth, tolerance=1e-10)
                    self.assertEqual(refined["matches"], integer["matches"])

    def test_large_offset_and_tiny_gain(self):
        clean, truth = self.scene()
        for gain, offset in [(1.0, 1e10), (1e-10, 0.0)]:
            for transform_template in (False, True):
                for levels in (1, 3):
                    with self.subTest(gain=gain, offset=offset,
                                      transform_template=transform_template, levels=levels):
                        image = gain * clean + offset
                        template = gain * self.template + offset if transform_template else self.template
                        with np.errstate(invalid="raise", divide="raise", over="raise"):
                            result = match_pose_ncc(image, template, pyramid_levels=levels,
                                                    subpixel=False)
                        self.assert_matched(result)
                        self.assertGreater(result["best_score"], 0.999999)
                        self.assert_pose(result["matches"][0], truth, tolerance=1e-10)

    def test_huge_amplitude_self_match(self):
        template = self.template * 1e100
        for levels in (1, 3):
            with self.subTest(levels=levels):
                with np.errstate(invalid="raise", divide="raise", over="raise"):
                    result = match_pose_ncc(template.copy(), template, pyramid_levels=levels)
                self.assert_matched(result)
                self.assertGreater(result["best_score"], 0.999999)
                self.assert_pose(result["matches"][0], np.eye(3), tolerance=1e-10)
                self.assertIsNone(result["second_score"])
                self.assertIsNone(result["score_gap"])

    def test_unchanged_target_on_extremely_bright_background(self):
        h, w = self.template.shape
        x, y = 17, 15
        truth = affine_truth(self.template, (x + (w - 1) / 2, y + (h - 1) / 2))
        for brightness in (1e10, 1e100):
            image = np.full((65, 77), brightness)
            image[y:y + h, x:x + w] = self.template
            np.testing.assert_array_equal(image[y:y + h, x:x + w], self.template)
            for levels in (1, 3):
                with self.subTest(brightness=brightness, levels=levels):
                    with np.errstate(invalid="raise", divide="raise", over="raise"):
                        result = match_pose_ncc(image, self.template, pyramid_levels=levels,
                                                subpixel=False)
                    self.assert_matched(result)
                    self.assertGreater(result["best_score"], 0.999999)
                    self.assert_pose(result["matches"][0], truth, tolerance=1e-10)

    def test_partial_occlusion(self):
        image, truth = self.scene()
        image[63:72, 79:91] = 105.0
        result = match_pose_ncc(image, self.template, min_score=0.8)
        self.assert_matched(result)
        self.assertLess(result["best_score"], 0.99)
        self.assert_pose(result["matches"][0], truth, tolerance=0.3)

    def test_no_target_noise(self):
        result = match_pose_ncc(self.background, self.template, min_score=0.8)
        self.assert_rejected(result, "below_min_score")
        self.assertLess(result["best_score"], 0.8)
        self.assertEqual(result["searched_poses"], 1)

    def test_no_target_constant_image(self):
        for value in (0.0, 115.0):
            with self.subTest(value=value):
                result = match_pose_ncc(np.full_like(self.background, value), self.template)
                self.assert_rejected(result, "below_min_score")
                for key in ("best_score", "second_score", "score_gap"):
                    self.assertIsNone(result[key])

    def two_targets(self):
        first = affine_truth(self.template, (47, 45))
        second = affine_truth(self.template, (133, 101))
        image = render_target(self.background, self.template, first)
        return render_target(image, self.template, second), [first, second]

    def close_periodic_targets(self):
        tile = np.random.default_rng(731).normal(110, 28, (33, 4))
        self.template = np.tile(tile, (1, 11))[:, :41]
        image = self.background.copy()
        image[50:83, 50:95] = np.tile(tile, (1, 12))[:, :45]
        truths = [affine_truth(self.template, (70, 66)),
                  affine_truth(self.template, (74, 66))]
        for x in (50, 54):
            np.testing.assert_array_equal(image[50:83, x:x + 41], self.template)
        return image, truths

    def test_small_nms_preserves_close_target_ambiguity(self):
        image, _ = self.close_periodic_targets()
        for levels in (1, 3):
            with self.subTest(levels=levels):
                result = match_pose_ncc(image, self.template, pyramid_levels=levels,
                                        nms_radius=1, candidate_limit=2, subpixel=False)
                self.assert_rejected(result, "ambiguous")
                self.assertGreater(result["best_score"], 0.999999)
                self.assertGreater(result["second_score"], 0.999999)
                self.assertAlmostEqual(result["score_gap"], 0.0, places=10)

    def test_small_nms_returns_both_close_targets(self):
        image, truths = self.close_periodic_targets()
        for levels in (1, 3):
            with self.subTest(levels=levels):
                result = match_pose_ncc(image, self.template, pyramid_levels=levels,
                                        nms_radius=1, candidate_limit=2, max_count=2,
                                        min_score_gap=0, min_score=0.999, subpixel=False)
                self.assert_matched(result, 2)
                ordered = sorted(result["matches"], key=lambda item: item["center_x"])
                for match, truth in zip(ordered, truths):
                    self.assertGreater(match["score"], 0.999999)
                    self.assert_pose(match, truth, tolerance=1e-10)
                self.assertAlmostEqual(result["score_gap"], 0.0, places=10)

    def test_equal_targets_are_ambiguous(self):
        image, _ = self.two_targets()
        result = match_pose_ncc(image, self.template)
        self.assert_rejected(result, "ambiguous")
        self.assertGreater(result["best_score"], 0.999999)
        self.assertGreater(result["second_score"], 0.999999)
        self.assertAlmostEqual(result["score_gap"], 0.0, places=10)
        accepted = match_pose_ncc(image, self.template, min_score_gap=0)
        self.assert_matched(accepted)

    def test_multiple_targets_and_output_limit(self):
        image, truths = self.two_targets()
        for limit in (1, 2, 3):
            with self.subTest(max_count=limit):
                result = match_pose_ncc(image, self.template, max_count=limit,
                                        min_score_gap=0, min_score=0.95, subpixel=False)
                self.assert_matched(result, min(limit, 2))
                scores = [item["score"] for item in result["matches"]]
                self.assertEqual(scores, sorted(scores, reverse=True))
                if limit >= 2:
                    ordered = sorted(result["matches"], key=lambda item: item["center_x"])
                    for match, truth in zip(ordered, truths):
                        self.assert_pose(match, truth, tolerance=1e-10)

    def test_multiple_rotated_scaled_targets(self):
        poses = [((46, 43), -15, 0.8), ((133, 102), 30, 1.2)]
        image = self.background.copy()
        truths = []
        for center, angle, scale in poses:
            truth = affine_truth(self.template, center, angle, scale)
            truths.append(truth)
            image = render_target(image, self.template, truth)
        result = match_pose_ncc(image, self.template, max_count=2, min_score_gap=0,
                                angle_range=(-15, 30), angle_step=15,
                                scale_range=(0.8, 1.2), scale_step=0.2)
        self.assert_matched(result, 2)
        ordered = sorted(result["matches"], key=lambda item: item["center_x"])
        for match, truth, (_, angle, scale) in zip(ordered, truths, poses):
            self.assert_pose(match, truth, angle, scale)

    def test_nms_suppresses_alternative_poses(self):
        image, truth = self.scene()
        result = match_pose_ncc(image, self.template, angle_range=(-2, 2), angle_step=2,
                                scale_range=(0.98, 1.02), scale_step=0.02,
                                min_score=0.9, min_score_gap=0, max_count=5)
        self.assert_matched(result)
        self.assert_pose(result["matches"][0], truth, tolerance=0.1)

    def test_mask_excludes_corrupted_pixels(self):
        mask = np.ones_like(self.template, dtype=bool)
        mask[7:25, 10:29] = False
        corrupted = self.template.copy()
        corrupted[~mask] = np.random.default_rng(51).normal(110, 180, (~mask).sum())
        truth = affine_truth(self.template, (86, 70))
        image = render_target(self.background, corrupted, truth)
        masked = match_pose_ncc(image, self.template, mask=mask, subpixel=False)
        unmasked = match_pose_ncc(image, self.template, min_score=0.8)
        self.assert_matched(masked)
        self.assertGreater(masked["best_score"], 0.999999)
        self.assert_pose(masked["matches"][0], truth, tolerance=1e-10)
        self.assert_rejected(unmasked, "below_min_score")

    def test_rotated_scaled_mask(self):
        mask = np.ones_like(self.template)
        mask[8:22, 12:27] = 0
        corrupted = self.template.copy()
        corrupted[mask == 0] = 500
        truth = affine_truth(self.template, (86, 70), 30, 1.2)
        image = render_target(self.background, corrupted, truth)
        result = match_pose_ncc(image, self.template, mask=mask, angle_range=(15, 30),
                                angle_step=15, scale_range=(1.0, 1.2), scale_step=0.2)
        self.assert_matched(result)
        self.assert_pose(result["matches"][0], truth, 30, 1.2)

    def test_all_ones_mask_matches_unmasked(self):
        image, _ = self.scene()
        baseline = match_pose_ncc(image, self.template)
        for dtype in (bool, np.uint8, np.float64):
            with self.subTest(dtype=dtype):
                result = match_pose_ncc(image, self.template, mask=np.ones(self.template.shape, dtype=dtype))
                for key in baseline.keys() - {"elapsed_ms"}:
                    self.assertEqual(result[key], baseline[key])

    def test_image_edges_and_corners(self):
        h, w = self.template.shape
        height, width = self.background.shape
        for x, y in [(0, 0), (width - w, 0), (0, height - h),
                     (width - w, height - h), (0, 51), (67, height - h)]:
            for levels in (1, 3):
                with self.subTest(x=x, y=y, levels=levels):
                    image, truth = self.scene(center=(x + (w - 1) / 2, y + (h - 1) / 2))
                    result = match_pose_ncc(image, self.template, pyramid_levels=levels)
                    self.assert_matched(result)
                    self.assert_pose(result["matches"][0], truth, tolerance=0.1)

    def test_template_fills_image(self):
        for levels in (1, 3):
            with self.subTest(levels=levels):
                result = match_pose_ncc(self.template.copy(), self.template, pyramid_levels=levels)
                self.assert_matched(result)
                self.assert_pose(result["matches"][0], np.eye(3), tolerance=1e-10)
                self.assertIsNone(result["second_score"])
                self.assertIsNone(result["score_gap"])

    def test_oversized_template_and_pose(self):
        cases = [(self.background[:20, :25], {}),
                 (self.background, {"scale_range": (10, 10)})]
        for image, kwargs in cases:
            with self.subTest(kwargs=kwargs, shape=image.shape):
                result = match_pose_ncc(image, self.template, **kwargs)
                self.assert_rejected(result, "no_valid_pose")
                self.assertEqual(result["searched_poses"], 0)
                self.assertIsNone(result["best_score"])

    def test_search_skips_only_oversized_poses(self):
        result = match_pose_ncc(self.template.copy(), self.template, scale_range=(1, 2), scale_step=1)
        self.assert_matched(result)
        self.assertEqual(result["searched_poses"], 1)

    def test_small_template_pyramid_fallback(self):
        self.template = make_template((9, 11))
        image, truth = self.scene()
        result = match_pose_ncc(image, self.template, pyramid_levels=8, subpixel=False)
        single = match_pose_ncc(image, self.template, pyramid_levels=1, subpixel=False)
        self.assert_matched(result)
        self.assert_matched(single)
        self.assert_pose(result["matches"][0], truth, tolerance=1e-10)
        self.assertEqual(result["matches"], single["matches"])

    def test_pyramid_and_single_level_consistency(self):
        for angle, scale in [(0, 1), (-15, 0.8), (30, 1.2)]:
            with self.subTest(angle=angle, scale=scale):
                image, truth = self.scene(angle=angle, scale=scale)
                kwargs = dict(angle_range=(-15, 30), angle_step=15,
                              scale_range=(0.8, 1.2), scale_step=0.2, subpixel=False)
                single = match_pose_ncc(image, self.template, pyramid_levels=1, **kwargs)
                pyramid = match_pose_ncc(image, self.template, pyramid_levels=3, **kwargs)
                for result in (single, pyramid):
                    self.assert_matched(result)
                    self.assert_pose(result["matches"][0], truth, angle, scale)
                self.assertEqual(single["searched_poses"], pyramid["searched_poses"])
                np.testing.assert_allclose(single["matches"][0]["transform"],
                                           pyramid["matches"][0]["transform"], atol=1e-10)
                self.assertAlmostEqual(single["best_score"], pyramid["best_score"], places=10)

    def test_exhaustive_verification_defaults_to_true(self):
        parameter = signature(match_pose_ncc).parameters["exhaustive_verification"]
        self.assertIs(parameter.default, True)
        self.assertEqual(parameter.kind, parameter.KEYWORD_ONLY)
        image, truth = self.scene()
        default = match_pose_ncc(image, self.template)
        explicit = match_pose_ncc(image, self.template, exhaustive_verification=True)
        for result in (default, explicit):
            self.assert_matched(result)
            self.assert_pose(result["matches"][0], truth, tolerance=0.1)
        for key in default.keys() - {"elapsed_ms"}:
            self.assertEqual(default[key], explicit[key])

    def test_high_frequency_odd_phase_pyramid_recovers_exact_target(self):
        for shape in [(32, 40), (33, 41)]:
            with self.subTest(shape=shape):
                rng = np.random.default_rng(731)
                yy, xx = np.indices(shape)
                envelope = gaussian_filter(rng.normal(size=shape), 1)
                self.template = (110 + 28 * (-1.0) ** (xx + yy) * (2 + envelope)
                                 + 0.01 * rng.normal(size=shape))
                h, w = shape
                x, y = 67, 55
                image = self.background.copy()
                image[y:y + h, x:x + w] = self.template
                truth = affine_truth(self.template, (x + (w - 1) / 2, y + (h - 1) / 2))
                kwargs = dict(candidate_limit=2, subpixel=False, min_score=0.99)
                single = match_pose_ncc(image, self.template, pyramid_levels=1, **kwargs)
                default = match_pose_ncc(image, self.template, pyramid_levels=3, **kwargs)
                explicit = match_pose_ncc(image, self.template, pyramid_levels=3,
                                          exhaustive_verification=True, **kwargs)
                for result in (single, default, explicit):
                    self.assert_matched(result)
                    self.assertGreater(result["best_score"], 0.999999)
                    self.assert_pose(result["matches"][0], truth, tolerance=1e-10)
                    self.assertEqual(result["matches"], single["matches"])
                    self.assertEqual(result["searched_poses"], single["searched_poses"])

    def test_approximate_mode_runs_with_and_without_pyramid(self):
        image, truth = self.scene()
        for levels in (1, 3):
            with self.subTest(levels=levels):
                with np.errstate(invalid="raise", divide="raise", over="raise"):
                    result = match_pose_ncc(image, self.template, pyramid_levels=levels,
                                            exhaustive_verification=False)
                    no_target = match_pose_ncc(self.background, self.template,
                                               pyramid_levels=levels, exhaustive_verification=False)
                self.assert_matched(result)
                self.assertGreater(result["best_score"], 0.999999)
                self.assert_pose(result["matches"][0], truth, tolerance=0.1)
                self.assert_rejected(no_target, "below_min_score")
                self.assertTrue(np.isfinite(no_target["best_score"]))
                self.assertLess(no_target["best_score"], 0.8)
                self.assertGreater(no_target["searched_poses"], 0)
                self.assertTrue(np.isfinite(no_target["elapsed_ms"]))

    def test_invalid_image_and_template(self):
        invalid = [np.empty((0, 3)), np.empty((3, 0)), np.ones(5), np.ones((3, 4, 1)),
                   np.array(1.0), np.full((8, 9), np.nan), np.full((8, 9), np.inf)]
        for name in ("image", "template"):
            for index, value in enumerate(invalid):
                with self.subTest(name=name, case=index):
                    kwargs = {"image": self.background, "template": self.template, name: value}
                    with self.assertRaises(ValueError):
                        match_pose_ncc(**kwargs)

    def test_constant_or_insufficient_template(self):
        for template in (np.ones((8, 9)), np.zeros((8, 9)), np.array([[1., 2., 3.]])):
            with self.subTest(shape=template.shape, value=template[0, 0]):
                with self.assertRaises(ValueError):
                    match_pose_ncc(self.background, template)

    def test_invalid_masks(self):
        sparse = np.zeros_like(self.template)
        sparse.flat[:3] = 1
        invalid = [np.ones((3, 4)), np.full_like(self.template, 0.5),
                   np.full_like(self.template, -1), np.full_like(self.template, 2),
                   np.full_like(self.template, np.nan), np.full_like(self.template, np.inf),
                   np.zeros_like(self.template), sparse, np.ones((33, 41, 1))]
        for index, mask in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    match_pose_ncc(self.background, self.template, mask=mask)
        template = self.template.copy()
        template[4:10, 5:11] = 70
        mask = np.zeros_like(template)
        mask[4:10, 5:11] = 1
        with self.assertRaises(ValueError):
            match_pose_ncc(self.background, template, mask=mask)

    def test_invalid_search_configuration(self):
        cases = []
        for name in ("angle_range", "scale_range"):
            cases.extend({name: value} for value in [(2, 1), (np.nan, 1), (1, np.inf), (1,), (1, 2, 3)])
        cases.extend({"scale_range": value} for value in [(0, 1), (-1, 1)])
        for name in ("angle_step", "scale_step", "nms_radius"):
            cases.extend({name: value} for value in [0, -1, np.nan, np.inf])
        for name in ("max_count", "pyramid_levels", "candidate_limit"):
            cases.extend({name: value} for value in [0, -1, 1.5, True, np.bool_(True)])
        for name, values in [("min_score", [-1.1, 1.1, np.nan, np.inf]),
                             ("min_score_gap", [-0.1, 2.1, np.nan, np.inf])]:
            cases.extend({name: value} for value in values)
        cases.extend([
            {"pyramid_levels": 9}, {"max_count": 2}, {"candidate_limit": 1},
            {"max_count": 3, "candidate_limit": 2, "min_score_gap": 0},
            {"angle_range": (0, 1), "angle_step": 0.00001},
            {"scale_range": (1, 2), "scale_step": 0.00001},
            {"angle_range": (0, 100), "angle_step": 1,
             "scale_range": (1, 2), "scale_step": 0.01},
        ])
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    match_pose_ncc(self.background, self.template, **kwargs)

    def test_inputs_are_not_modified(self):
        image, _ = self.scene()
        mask = np.ones_like(self.template)
        originals = [array.copy() for array in (image, self.template, mask)]
        for array in (image, self.template, mask):
            array.flags.writeable = False
        result = match_pose_ncc(image, self.template, mask=mask)
        self.assert_matched(result)
        for actual, original in zip((image, self.template, mask), originals):
            np.testing.assert_array_equal(actual, original)

    def test_array_dtypes_and_noncontiguous_inputs(self):
        for dtype in (np.uint8, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                template = self.template.astype(dtype)
                image = self.background.astype(dtype)
                image[54:87, 66:107] = template
                image_storage = np.zeros((145, 362), dtype=dtype)
                template_storage = np.zeros((33, 82), dtype=dtype)
                image_storage[:, ::2] = image
                template_storage[:, ::2] = template
                result = match_pose_ncc(image_storage[:, ::2], template_storage[:, ::2], subpixel=False)
                self.assert_matched(result)
                self.assertGreater(result["best_score"], 0.999999)
                self.assert_pose(result["matches"][0], affine_truth(template, (86, 70)), tolerance=1e-10)

    def test_legacy_ncc_interfaces(self):
        image, truth = self.scene(center=(86.3, 70.35))
        score = match_template_ncc(image, self.template)
        self.assertEqual(score.shape, (113, 141))
        y, x = np.unravel_index(np.argmax(score), score.shape)
        dy, dx = parabolic_subpixel(score, y, x)
        old = find_best_matches(image, self.template, 0.8, 8, 1, True)
        self.assertIsInstance(old, list)
        self.assertEqual(len(old), 1)
        self.assertEqual(set(old[0]), {"x", "y", "score"})
        np.testing.assert_allclose([old[0]["x"], old[0]["y"]], [x + dx, y + dy], atol=1e-12)
        np.testing.assert_allclose([old[0]["x"], old[0]["y"]], truth[:2, 2], atol=0.2, rtol=0)
        result = match_pose_ncc(image, self.template, pyramid_levels=1, min_score_gap=0)
        self.assert_matched(result)
        for key in ("x", "y", "score"):
            self.assertAlmostEqual(result["matches"][0][key], old[0][key], places=9)
        integer = find_best_matches(image, self.template, 0.8, 8, 1, False)
        self.assertEqual((integer[0]["x"], integer[0]["y"]), (x, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
