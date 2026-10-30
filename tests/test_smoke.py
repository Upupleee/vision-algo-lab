"""冒烟测试：验证传统算法核心模块在合成数据上的精度。

运行（仓库根目录）:
    python -m tests.test_smoke
或
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from data.synthetic import (
    make_corner_scene,
    make_edge_profile,
    make_edge_scene,
    make_focus_stack,
    make_template_scene,
)
from tests.metrics import unimodality


def test_ncc_matching() -> str:
    from traditional.template_matching.ncc_matching import find_best_matches

    img, tpl, gt_xy = make_template_scene(noise=3.0)
    m = find_best_matches(img, tpl, min_score=0.6)[0]
    err = np.hypot(m["x"] - gt_xy[0], m["y"] - gt_xy[1])
    # 双三次重采样 + 噪声使满分 1.0 不可达，0.9 已表明匹配唯一且稳定
    assert m["score"] > 0.9, f"NCC 得分过低: {m['score']}"
    assert err < 0.25, f"NCC 亚像素误差过大: {err:.3f} px"
    return f"score={m['score']:.4f} err={err:.4f}px"


def test_shape_based_matching() -> str:
    from traditional.template_matching.shape_based import match

    img, tpl, gt_xy = make_template_scene(noise=2.0)
    x, y, s = match(img, tpl)
    err = np.hypot(x - gt_xy[0], y - gt_xy[1])
    assert s > 0.85, f"形状匹配得分过低: {s}"
    assert err < 2.0, f"形状匹配整像素误差过大: {err:.2f} px"
    return f"score={s:.4f} err={err:.2f}px（整像素）"


def test_caliper() -> str:
    from traditional.caliper_measurement.caliper import Caliper

    img, gt_fn = make_edge_scene(noise=1.0)
    cal = Caliper(center=(61.5, 80), length=128, width=11, angle_deg=90.0, num_probes=40)
    res = cal.measure(img)
    pts = res.points
    err = np.abs(pts[:, 0] - gt_fn(pts[:, 1]))
    assert np.nanmean(err) < 0.1, f"卡尺亚像素误差过大: {np.nanmean(err):.4f} px"
    assert res.rms < 0.15, f"直线拟合 RMS 过大: {res.rms:.4f} px"
    return f"mean_err={np.nanmean(err):.4f}px line_rms={res.rms:.4f}px"


def test_subpixel_profile() -> str:
    from traditional.subpixel.edge_subpixel import gaussian_fit_subpixel, parabolic_gradient_subpixel

    profile, gt = make_edge_profile(sigma=1.2, noise=0.0)
    p1, _ = parabolic_gradient_subpixel(profile)
    p2, _, _ = gaussian_fit_subpixel(profile, x0=p1)
    assert abs(p1 - gt) < 0.15, f"抛物线法误差过大: {abs(p1 - gt):.4f} px"
    assert abs(p2 - gt) < 0.02, f"高斯拟合法误差过大: {abs(p2 - gt):.4f} px"
    return f"parabolic={abs(p1 - gt):.4f}px gaussian_fit={abs(p2 - gt):.4f}px"


def test_devernay_edges() -> str:
    from traditional.subpixel.edge_subpixel import devernay_edges

    img, gt_fn = make_edge_scene(sigma=1.0, noise=0.5)
    pts = devernay_edges(img, high=25.0)
    assert len(pts) > 50, f"Devernay 边缘点过少: {len(pts)}"
    dist = np.abs(pts[:, 0] - gt_fn(pts[:, 1])) / np.sqrt(1 + 0.02**2)
    med = float(np.median(dist))
    assert med < 0.15, f"Devernay 中位法向误差过大: {med:.4f} px"
    return f"points={len(pts)} median_err={med:.4f}px"


def test_corner_subpixel() -> str:
    from traditional.subpixel.corner_subpixel import refine_corners_gradient, refine_corners_opencv

    img, gt_pts = make_corner_scene()
    resp = cv2.cornerHarris(np.float32(img), 3, 5, 0.04)
    ys, xs = np.where(resp > 0.01 * resp.max())
    cand = np.stack([xs, ys], axis=1).astype(np.float32)
    picked = []
    for gx_, gy_ in gt_pts:
        d = np.hypot(cand[:, 0] - gx_, cand[:, 1] - gy_)
        j = int(np.argmin(d))
        if d[j] < 6:
            picked.append(cand[j])
    picked = np.array(picked)
    r1 = refine_corners_opencv(img, picked)
    r2 = refine_corners_gradient(img, picked)
    e0 = float(np.hypot(*(picked - gt_pts).T).mean())
    e1 = float(np.hypot(*(r1 - gt_pts).T).mean())
    e2 = float(np.hypot(*(r2 - gt_pts).T).mean())
    assert e1 < 0.3, f"OpenCV 角点精化误差过大: {e1:.3f} px"
    assert e2 < 0.5, f"手写角点精化误差过大: {e2:.3f} px"
    return f"init={e0:.3f}px opencv={e1:.3f}px manual={e2:.3f}px"


def test_focus_measures() -> str:
    from traditional.autofocus.focus_measures import ALL_MEASURES, monotonic_ratio

    stack, gt_best, positions, z_focus = make_focus_stack()
    detail = []
    for name, fn in ALL_MEASURES.items():
        scores = np.array([fn(f) for f in stack])
        peak = int(np.argmax(scores))
        assert abs(peak - gt_best) <= 1, f"{name} 峰值帧偏差过大: {peak} vs {gt_best}"
        u = unimodality(scores)
        detail.append(f"{name}(峰偏={abs(peak - gt_best)},单峰={u:.2f})")
    return " ".join(detail)


def test_focus_search() -> str:
    from traditional.autofocus.focus_measures import tenengrad
    from traditional.autofocus.focus_search import gaussian_fit, golden_section, hill_climb

    stack, gt_best, positions, z_focus = make_focus_stack()
    scores = np.array([tenengrad(f) for f in stack])

    p_fit = gaussian_fit(positions, scores)
    frame_of = {round(p, 4): f for p, f in zip(positions, stack)}

    def score_at(p: float) -> float:
        key = min(frame_of, key=lambda q: abs(q - p))
        return tenengrad(frame_of[key])

    p_hill, n_hill = hill_climb(score_at, positions[0], positions[-1])
    p_gold, n_gold = golden_section(score_at, positions[0], positions[-1])
    assert abs(p_fit - z_focus) < 0.5, f"高斯拟合误差过大: {abs(p_fit - z_focus):.3f}"
    assert abs(p_hill - z_focus) < 1.0, f"爬山法误差过大: {abs(p_hill - z_focus):.3f}"
    assert abs(p_gold - z_focus) < 1.0, f"黄金分割误差过大: {abs(p_gold - z_focus):.3f}"
    return (
        f"gauss_fit_err={abs(p_fit - z_focus):.3f} hill_err={abs(p_hill - z_focus):.3f}"
        f"({n_hill}次) golden_err={abs(p_gold - z_focus):.3f}({n_gold}次)"
    )


TESTS = [
    test_ncc_matching,
    test_shape_based_matching,
    test_caliper,
    test_subpixel_profile,
    test_devernay_edges,
    test_corner_subpixel,
    test_focus_measures,
    test_focus_search,
]


def main() -> int:
    n_fail = 0
    print("=" * 72)
    for t in TESTS:
        try:
            detail = t()
            print(f"[PASS] {t.__name__:26s} {detail}")
        except AssertionError as e:
            n_fail += 1
            print(f"[FAIL] {t.__name__:26s} {e}")
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[ERROR] {t.__name__:25s} {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"{len(TESTS) - n_fail}/{len(TESTS)} 通过")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
