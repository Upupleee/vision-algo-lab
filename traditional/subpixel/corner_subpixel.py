"""亚像素角点：OpenCV cornerSubPix 封装 + 自实现的最小化梯度正交误差法。"""
from __future__ import annotations

import cv2
import numpy as np


def refine_corners_opencv(
    gray: np.ndarray, corners: np.ndarray, win: tuple[int, int] = (5, 5)
) -> np.ndarray:
    """OpenCV 经典实现：最小化梯度和窗口内灰度变化的正交残差。

    corners: (N, 2) 初始整像素/浮点角点 (x, y)。返回同形状的精化坐标。
    """
    pts = corners.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(
        np.asarray(gray, dtype=np.uint8),
        pts,
        winSize=win,
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    return refined.reshape(-1, 2)


def refine_corners_gradient(
    gray: np.ndarray, corners: np.ndarray, win: int = 4
) -> np.ndarray:
    """手写实现（供研究与 OpenCV 对比）。

    对每个角点，在窗口内解最小二乘：
        min Σ [∇I(p) · (c - p)]^2
    即求解结构张量 (Σ g g^T) c = Σ g g^T p，g 为该点梯度。
    """
    img = np.asarray(gray, dtype=np.float64)
    h, w = img.shape
    gy, gx = np.gradient(img)
    out = corners.astype(np.float64).copy()

    for i, (cx, cy) in enumerate(corners):
        x0, x1 = int(cx) - win, int(cx) + win + 1
        y0, y1 = int(cy) - win, int(cy) + win + 1
        if x0 < 1 or y0 < 1 or x1 > w - 1 or y1 > h - 1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        ix, iy = gx[y0:y1, x0:x1].ravel(), gy[y0:y1, x0:x1].ravel()
        px, py = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
        # 对比度加权（Zhang 摄标定法）：边缘点才参与
        mag2 = ix**2 + iy**2
        keep = mag2 > (mag2.max() * 1e-3 if mag2.max() > 0 else 1)
        ix, iy, px, py = ix[keep], iy[keep], px[keep], py[keep]
        A = np.stack([ix, iy], axis=1)          # (M, 2)
        b = (A * np.stack([px, py], axis=1)).sum(axis=1)  # b_i = g_i · p_i
        M = A.T @ A
        if np.linalg.cond(M) < 1e8:
            out[i] = np.linalg.solve(M, A.T @ b)
    return out


if __name__ == "__main__":
    from data.synthetic import make_corner_scene

    img, gt_pts = make_corner_scene()
    resp = cv2.cornerHarris(np.float32(img), 3, 5, 0.04)
    ys, xs = np.where(resp > 0.01 * resp.max())
    cand = np.stack([xs, ys], axis=1).astype(np.float32)
    # 每个真值附近只留一个候选
    picked = []
    for gx_, gy_ in gt_pts:
        d = np.hypot(cand[:, 0] - gx_, cand[:, 1] - gy_)
        j = int(np.argmin(d))
        if d[j] < 6:
            picked.append(cand[j])
    picked = np.array(picked)
    r1 = refine_corners_opencv(img, picked)
    r2 = refine_corners_gradient(img, picked)
    print(f"初检误差   : {np.hypot(*(picked - gt_pts).T).mean():.3f} px")
    print(f"OpenCV 精化: {np.hypot(*(r1 - gt_pts).T).mean():.3f} px")
    print(f"手写精化   : {np.hypot(*(r2 - gt_pts).T).mean():.3f} px")
