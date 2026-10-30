"""基于梯度方向的形状匹配（Steger / Halcon shape-based matching 的简化实现）。

原理：
1. 对模板提取 Sobel 梯度，保留幅值最大的前 n 个点，作为特征点集
   {p_i, 方向 φ_i, 权重 w_i}。
2. 匹配时在候选位置处同样提取梯度，评分为方向单位向量内积的加权和：
       S(u,v) = Σ w_i * max(0, cos(φ_i - φ(u+v+p_i)))   （Σw_i = 1）
3. 得分对均匀光照变化鲁棒（只依赖梯度方向，不依赖幅值/绝对灰度）。

当前实现为平移匹配版本；旋转/缩放可通过在候选点处采样旋转后的模板方向扩展。
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_features(
    gray: np.ndarray, top_k: int = 200, min_mag: float = 20.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取梯度特征点。返回 (相对坐标 (N,2) xy, 单位方向 (N,2), 权重 (N,))。"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    ys, xs = np.nonzero(mag > min_mag)
    if len(ys) == 0:
        raise ValueError("未找到足够梯度特征，请降低 min_mag")
    order = np.argsort(-mag[ys, xs])[:top_k]
    ys, xs = ys[order], xs[order]
    m = mag[ys, xs]
    return (
        np.stack([xs, ys], axis=1).astype(np.float64),
        np.stack([gx[ys, xs] / m, gy[ys, xs] / m], axis=1),
        m / m.sum(),
    )


def match(
    image: np.ndarray,
    template: np.ndarray,
    min_score: float = 0.7,
    top_k: int = 200,
) -> tuple[float, float, float]:
    """全图平移匹配，返回 (x, y, score)，即模板左上角的最佳整像素位置。"""
    pts, dirs, wts = extract_features(template, top_k=top_k)
    xs, ys = pts[:, 0], pts[:, 1]
    u_min = max(1, int(-xs.min()) + 1)
    v_min = max(1, int(-ys.min()) + 1)
    u_max = image.shape[1] - int(np.ceil(xs.max())) - 1
    v_max = image.shape[0] - int(np.ceil(ys.max())) - 1
    if u_max < u_min or v_max < v_min:
        raise ValueError("模板过大，无法在图像内移动")

    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    best = (-1.0, 0.0, 0.0)
    for v in range(v_min, v_max + 1):
        for u in range(u_min, u_max + 1):
            px = (u + xs).astype(int)
            py = (v + ys).astype(int)
            norm = mag[py, px]
            norm[norm == 0] = 1.0
            ux, uy = gx[py, px] / norm, gy[py, px] / norm
            dots = ux * dirs[:, 0] + uy * dirs[:, 1]
            s = float(np.dot(wts, np.clip(dots, 0.0, None)))
            if s > best[0]:
                best = (s, float(u), float(v))
        if best[0] >= max(min_score, 0.98):
            break  # 已接近满分，提前结束
    return best[1], best[2], best[0]


if __name__ == "__main__":
    from data.synthetic import make_template_scene

    img, tpl, gt_xy = make_template_scene(noise=2.0)
    x, y, s = match(img, tpl)
    err = np.hypot(x - gt_xy[0], y - gt_xy[1])
    print(f"形状匹配得分={s:.4f}  定位误差={err:.2f} px（整像素精度）")
