from __future__ import annotations

import json
import platform

import cv2
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from tests.metrics import localization_error
from tests.test_pose_ncc import affine_truth, make_template, render_target
from traditional.template_matching.pose_ncc import match_pose_ncc


def run_benchmark() -> dict:
    rows = []
    totals = {"targets": 0, "missed": 0, "detections": 0, "false_positive": 0}
    for scenario in ("translation", "rotation_scale", "off_grid", "off_grid_fine", "illumination_noise",
                     "occlusion", "no_target", "similar_distractor", "ambiguous", "multiple"):
        timings, errors, angle_errors, scale_errors, scores, reasons = [], [], [], [], [], []
        counts = {"targets": 0, "missed": 0, "detections": 0, "false_positive": 0}
        for seed in (731, 732, 733):
            template = make_template(seed=seed)
            rng = np.random.default_rng(seed + 1000)
            image = rng.normal(105.0, 4.0, (145, 181))
            poses = [((86.3, 70.35), 0.0, 1.0)]
            if scenario in ("rotation_scale", "illumination_noise"):
                poses = [((86.3, 70.35), 15.0, 1.2)]
            elif scenario in ("off_grid", "off_grid_fine"):
                poses = [((86.3, 70.35), 12.0, 1.13)]
            elif scenario == "no_target":
                poses = []
            elif scenario in ("ambiguous", "multiple"):
                poses = [((47.0, 45.0), 0.0, 1.0), ((133.0, 101.0), 0.0, 1.0)]
            elif scenario == "similar_distractor":
                poses = [((47.0, 45.0), 0.0, 1.0)]
            for center, angle, scale in poses:
                image = render_target(image, template, affine_truth(template, center, angle, scale))
            if scenario == "illumination_noise":
                image = 0.7 * image + 45 + rng.normal(0, 3, image.shape)
            elif scenario == "occlusion":
                image[63:72, 79:91] = 105.0
            elif scenario == "similar_distractor":
                distractor = template.copy()
                distractor[:, 20:] = distractor[::-1, 20:]
                image = render_target(image, distractor, affine_truth(template, (133, 101)))
            kwargs = dict(angle_range=(-30, 30), angle_step=15,
                          scale_range=(0.8, 1.2), scale_step=0.2)
            if scenario == "multiple":
                kwargs.update(max_count=2, min_score_gap=0)
            elif scenario == "off_grid_fine":
                kwargs.update(angle_range=(5, 20), angle_step=2,
                              scale_range=(1.05, 1.2), scale_step=0.025)
            match_pose_ncc(image, template, **kwargs)
            for _ in range(3):
                result = match_pose_ncc(image, template, **kwargs)
                timings.append(result["elapsed_ms"])
            reasons.append(result["reason"])
            if result["best_score"] is not None:
                scores.append(result["best_score"])
            expected = [] if scenario == "ambiguous" else poses
            matches = result["matches"]
            counts["targets"] += len(expected)
            counts["detections"] += len(matches)
            accepted = 0
            if matches and expected:
                predicted_xy = np.array([[m["center_x"], m["center_y"]] for m in matches])
                expected_xy = np.array([p[0] for p in expected])
                distances = np.linalg.norm(predicted_xy[:, None] - expected_xy[None], axis=2)
                ii, jj = linear_sum_assignment(distances)
                for i, j in zip(ii, jj):
                    m, (_, angle, scale) = matches[i], expected[j]
                    angle_error = abs((m["angle_deg"] - angle + 180) % 360 - 180)
                    scale_error = abs(m["scale"] - scale)
                    error = float(localization_error(predicted_xy[i:i + 1], expected_xy[j:j + 1])[0])
                    if error <= 2 and angle_error <= 7.5 and scale_error <= 0.1:
                        accepted += 1
                        errors.append(error)
                        angle_errors.append(angle_error)
                        scale_errors.append(scale_error)
            counts["missed"] += len(expected) - accepted
            counts["false_positive"] += len(matches) - accepted
        for key, value in counts.items():
            totals[key] += value
        rows.append({
            "scenario": scenario, "seeds": 3, "counts": counts, "reasons": reasons,
            "center_error_max_px": max(errors) if errors else None,
            "angle_error_max_deg": max(angle_errors) if angle_errors else None,
            "scale_error_max": max(scale_errors) if scale_errors else None,
            "best_score_min": min(scores) if scores else None,
            "median_ms": float(np.median(timings)),
        })
    return {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "protocol": {"seeds": [731, 732, 733], "warmup": 1, "repeat": 3,
                     "position_tolerance_px": 2, "angle_tolerance_deg": 7.5,
                     "scale_tolerance": 0.1, "exhaustive_verification": True},
        "totals": totals, "scenarios": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, allow_nan=False))
