"""亚像素边缘定位：一维剖面方法 + 二维 Devernay 梯度抛物面方法。

一维方法（配合卡尺剖面使用）：
- parabolic_gradient_subpixel: |梯度| 峰抛物线插值，最快，精度 ~0.05-0.1 px
- gaussian_fit_subpixel:       误差函数（积分高斯）剖面拟合，最精确 ~0.01 px

二维方法：
- devernay_edges: Devernay (1995) 方法。Canny 非极大值抑制得到边缘像素后，
  沿梯度方向对梯度幅值采样三点，抛物线拟合求亚像素偏移。精度 ~0.05 px，
  对边缘模糊度不敏感，是工业视觉中的常用基线。

TODO: Zernike 矩法（Ghosal-Mehrotra 1993）需要 A31 矩做模糊宽度补偿才能在
离焦边缘上达到高精度，待研究后在 docs/ 补充笔记并实现。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.special import erf


def parabolic_gradient_subpixel(profile: np.ndarray) -> tuple[float, float]:
    """|一阶差分| 最大处抛物线插值。返回 (亚像素位置, 峰值强度)。最快的方法。"""
    f = np.abs(np.gradient(np.asarray(profile, dtype=np.float64)))
    k = int(np.argmax(f))
    if 0 < k < len(f) - 1:
        denom = f[k - 1] - 2 * f[k] + f[k + 1]
        delta = 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (f[k - 1] - f[k + 1]) / denom, -0.5, 0.5))
    else:
        delta = 0.0
    return k + delta, float(f[k])


def gaussian_fit_subpixel(
    profile: np.ndarray, x0: float | None = None
) -> tuple[float, float, float]:
    """用误差函数拟合阶跃边缘模型：I(x) = a·erf((x-c)/(σ√2)) + b。

    返回 (边缘中心 c, 拟合 RMS, σ)。一维方法中最精确，适合卡尺剖面。
    """
    xs = np.arange(len(profile), dtype=np.float64)
    if x0 is None:
        x0, _ = parabolic_gradient_subpixel(profile)

    def model(x, a, c, sigma, b):
        return a * erf((x - c) / (sigma * np.sqrt(2.0))) + b

    amp = float(np.ptp(profile)) / 2.0 or 1.0
    p0 = [amp, x0, 1.0, float(np.mean(profile))]
    popt, _ = curve_fit(model, xs, profile, p0=p0, maxfev=5000)
    a, c, sigma, b = popt
    rms = float(np.sqrt(np.mean((model(xs, *popt) - profile) ** 2)))
    return float(c), rms, float(abs(sigma))


def devernay_edges(
    gray: np.ndarray,
    sigma_grad: float = 1.0,
    high: float = 30.0,
) -> np.ndarray:
    """Devernay 亚像素边缘检测。返回边缘点坐标 (N, 2)，列 [x, y]。

    步骤：
    1. 高斯偏导求梯度幅值；
    2. 沿梯度方向非极大值抑制（阈值 high，同 Canny）；
    3. 对每个边缘像素，沿梯度方向在 ±1 采样梯度幅值，三点抛物线拟合，
       偏移 -b/(2a) 截断到 [-0.5, 0.5]，得到亚像素边缘点。
    """
    import cv2

    img = np.asarray(gray, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("需要灰度图")
    sm = cv2.GaussianBlur(img, (0, 0), sigma_grad)
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)

    h, w = mag.shape
    ys, xs = np.nonzero(mag >= high)
    keep = np.ones(len(xs), dtype=bool)
    ux = np.cos(theta[ys, xs])
    uy = np.sin(theta[ys, xs])
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            if sx == 0 and sy == 0:
                continue
            nx = np.clip(xs + sx, 0, w - 1)
            ny = np.clip(ys + sy, 0, h - 1)
            align = np.abs(ux * sx + uy * sy) > 0.5  # 邻居位于梯度方向 ±45° 内
            keep &= ~(align & (mag[ny, nx] > mag[ys, xs]))
    fy, fx = ys[keep], xs[keep]

    # 亚像素：沿梯度方向三点抛物线
    dirx = np.cos(theta[fy, fx])
    diry = np.sin(theta[fy, fx])
    g0 = mag[fy, fx]
    gm = map_coordinates(mag, [fy - diry, fx - dirx], order=1, mode="nearest")
    gp = map_coordinates(mag, [fy + diry, fx + dirx], order=1, mode="nearest")
    denom = gm - 2 * g0 + gp
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (gm - gp) / denom, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    return np.stack([fx + delta * dirx, fy + delta * diry], axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from data.synthetic import make_edge_profile, make_edge_scene
    from traditional.viz import AUX, DET, GT, image_ax, metric_text, save, setup

    setup()
    profile, gt = make_edge_profile(sigma=1.2, noise=1.0)
    p1, _ = parabolic_gradient_subpixel(profile)
    c, rms, sig = gaussian_fit_subpixel(profile, x0=p1)
    e_par, e_gau = abs(p1 - gt), abs(c - gt)
    print(f"[1D] 抛物线法  : 误差={e_par:.4f} px")
    print(f"[1D] 高斯拟合法: 误差={e_gau:.4f} px  sigma={sig:.3f}  RMS={rms:.3f}")

    img, gt_fn = make_edge_scene(sigma=1.0, noise=0.5)
    pts = devernay_edges(img)
    if len(pts):
        err = np.abs(pts[:, 0] - gt_fn(pts[:, 1])) / np.sqrt(1 + 0.08**2)
        med = float(np.median(err))
        print(f"[2D] Devernay  : {len(pts)} 点  中位法向误差={med:.4f} px")
    else:
        med = float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) 一维阶跃剖面：真值 vs 两种亚像素估计
    xs = np.arange(len(profile))
    axes[0].plot(xs, profile, "o", color="#9AA0A6", ms=3.2, alpha=0.75, label="采样剖面（含噪声）")
    axes[0].axvline(gt, color=GT, ls="--", lw=1.8, label=f"真值 {gt:.3f}")
    axes[0].axvline(p1, color=AUX, ls=":", lw=1.8, label=f"抛物线 {p1:.3f}")
    axes[0].axvline(c, color=DET, ls="-", lw=1.5, alpha=0.9, label=f"高斯拟合 {c:.3f}")
    axes[0].set_title("一维阶跃边缘的亚像素定位", pad=6)
    axes[0].set_xlabel("像素坐标（px）")
    axes[0].set_ylabel("灰度")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=7.5)
    metric_text(
        axes[0],
        [
            f"抛物线误差  {e_par:.4f} px",
            f"高斯拟合误差 {e_gau:.4f} px",
            f"拟合 sigma   {sig:.3f}",
            f"拟合 RMS     {rms:.4f}",
        ],
    )

    # (a-inset) 放大边缘过渡区，三条竖线才分得开
    zx0, zx1 = gt - 2.5, gt + 2.5
    axin = axes[0].inset_axes([0.60, 0.10, 0.37, 0.30])
    axin.plot(xs, profile, "o", color="#9AA0A6", ms=3.0, alpha=0.85)
    for v, col, st, lw in ((gt, GT, "--", 1.8), (p1, AUX, ":", 1.8), (c, DET, "-", 1.4)):
        axin.axvline(v, color=col, ls=st, lw=lw)
    axin.set_xlim(zx0, zx1)
    lo = np.min(profile[(xs >= zx0) & (xs <= zx1)])
    hi = np.max(profile[(xs >= zx0) & (xs <= zx1)])
    pad = (hi - lo) * 0.15
    axin.set_ylim(lo - pad, hi + pad)
    axin.tick_params(labelsize=6.5)
    axin.grid(True, alpha=0.4)
    axin.set_title("过渡区放大", fontsize=7, pad=2)
    axes[0].indicate_inset_zoom(axin, edgecolor="#9AA0A6", alpha=0.7)

    # (b) 二维 Devernay 边缘点 —— 裁到边缘附近，否则斜率 0.02 看不出倾斜
    x0c, x1c, y0c, y1c = 50, 76, 0, 64
    crop = img[y0c:y1c, x0c:x1c]
    axes[1].imshow(
        crop, cmap="gray", interpolation="nearest", aspect="auto",
        extent=(x0c - 0.5, x1c - 0.5, y1c - 0.5, y0c - 0.5),
    )
    axes[1].set_title("二维 Devernay 亚像素边缘（局部放大）", pad=6)
    axes[1].set_xlabel("x（px）")
    axes[1].set_ylabel("y（px）")
    yy = np.linspace(y0c, y1c, 100)
    axes[1].plot(gt_fn(yy), yy, color=GT, ls="--", lw=1.6, label="真值边缘")
    if len(pts):
        sel = ((pts[:, 0] >= x0c) & (pts[:, 0] < x1c)
               & (pts[:, 1] >= y0c) & (pts[:, 1] < y1c))
        axes[1].plot(pts[sel, 0], pts[sel, 1], linestyle="none", marker="o",
                     ms=3.0, color=DET, alpha=0.85,
                     label=f"边缘点 {int(sel.sum())}")
    axes[1].legend(loc="lower right", fontsize=7)
    metric_text(
        axes[1],
        [f"全图 {len(pts)} 点", f"中位法向误差 {med:.4f} px"],
        loc="upper left",
        size=7.5,
    )

    fig.tight_layout()
    print("图片 →", save(fig, "edge_subpixel"))
