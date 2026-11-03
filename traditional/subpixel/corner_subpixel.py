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
    import matplotlib.pyplot as plt

    from data.synthetic import make_corner_scene
    from traditional.viz import AUX, DET, GT, MUTED, metric_text, save, setup

    setup()
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

    e0 = np.hypot(*(picked - gt_pts).T).mean()
    e1 = np.hypot(*(r1 - gt_pts).T).mean()
    e2 = np.hypot(*(r2 - gt_pts).T).mean()
    print(f"初检误差   : {e0:.3f} px")
    print(f"OpenCV 精化: {e1:.3f} px")
    print(f"手写精化   : {e2:.3f} px")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) 全图：真值 / 初检 / 两种精化
    axes[0].imshow(img, cmap="gray", interpolation="nearest")
    axes[0].set_title("棋盘格角点：初检与两种精化", pad=6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal")
    axes[0].plot(gt_pts[:, 0], gt_pts[:, 1], linestyle="none", marker="o",
                 ms=9, markerfacecolor="none", markeredgecolor=GT, markeredgewidth=1.5,
                 label="真值")
    axes[0].plot(picked[:, 0], picked[:, 1], linestyle="none", marker="x",
                 ms=5, color=MUTED, label="Harris 初检")
    axes[0].plot(r1[:, 0], r1[:, 1], linestyle="none", marker="+",
                 ms=7, color=AUX, markeredgewidth=1.5, label="OpenCV 精化")
    axes[0].plot(r2[:, 0], r2[:, 1], linestyle="none", marker=".",
                 ms=4, color=DET, label="手写精化")
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)
    metric_text(
        axes[0],
        [
            f"初检     {e0:.3f} px",
            f"OpenCV   {e1:.3f} px",
            f"手写     {e2:.3f} px",
            f"角点数   {len(gt_pts)}",
        ],
    )

    # (b) 单个角点局部放大 —— 看清精化前后差在哪
    k = int(np.argmin(np.hypot(*(r2 - gt_pts).T)))
    gx, gy = gt_pts[k]
    rad = 7
    x0c, x1c = int(gx - rad), int(gx + rad)
    y0c, y1c = int(gy - rad), int(gy + rad)
    x0c, y0c = max(0, x0c), max(0, y0c)
    x1c, y1c = min(img.shape[1], x1c), min(img.shape[0], y1c)
    axes[1].imshow(
        img[y0c:y1c, x0c:x1c], cmap="gray", interpolation="nearest",
        extent=(x0c - 0.5, x1c - 0.5, y1c - 0.5, y0c - 0.5),
    )
    axes[1].set_title("单个角点局部放大", pad=6)
    axes[1].set_xlabel("x（px）")
    axes[1].set_ylabel("y（px）")
    axes[1].plot(gx, gy, marker="o", ms=11, markerfacecolor="none",
                 markeredgecolor=GT, markeredgewidth=1.8, label="真值")
    axes[1].plot(picked[k, 0], picked[k, 1], marker="x", ms=7, color=MUTED,
                 markeredgewidth=1.6, label="初检")
    axes[1].plot(r1[k, 0], r1[k, 1], marker="+", ms=10, color=AUX,
                 markeredgewidth=1.7, label="OpenCV")
    axes[1].plot(r2[k, 0], r2[k, 1], marker=".", ms=9, color=DET, label="手写")
    axes[1].legend(loc="upper right", fontsize=6.5, borderpad=0.3)

    # (c) 每个角点的误差对比
    idx = np.arange(len(gt_pts))
    w = 0.27
    axes[2].bar(idx - w, np.hypot(*(picked - gt_pts).T), w, color=MUTED, label="初检")
    axes[2].bar(idx, np.hypot(*(r1 - gt_pts).T), w, color=AUX, label="OpenCV")
    axes[2].bar(idx + w, np.hypot(*(r2 - gt_pts).T), w, color=DET, label="手写")
    axes[2].set_title("各角点定位误差对比", pad=6)
    axes[2].set_xlabel("角点序号")
    axes[2].set_ylabel("误差（px）")
    axes[2].set_yscale("log")
    axes[2].set_xticks(idx)
    axes[2].tick_params(axis="x", labelsize=6.5)
    axes[2].grid(True, axis="y", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=7)
    axes[2].set_ylim(top=axes[2].get_ylim()[1] * 2.2)
    metric_text(
        axes[2],
        [f"均值 {e0:.3f} → {e1:.3f} → {e2:.3f} px", "纵轴为对数刻度"],
        loc="upper left",
        size=7,
    )

    fig.tight_layout()
    print("图片 →", save(fig, "corner_subpixel"))
