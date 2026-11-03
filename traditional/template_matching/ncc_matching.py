"""归一化互相关（NCC / ZNCC）模板匹配，FFT 加速 + 亚像素抛物线精化。

ZNCC 定义：
    R(u,v) = Σ (I(u+x,v+y)-μ_I) (T(x,y)-μ_T)
             / (σ_I(u,v) * sqrt(Σ (T-μ_T)^2))
其中 μ_I、σ_I 为模板窗口内图像的局部均值与标准差。

亚像素：对峰值附近的评分曲面沿 x、y 各自做三点抛物线插值。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def match_template_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """计算零均值归一化互相关评分图，取值 [-1, 1]。

    返回尺寸 (H-h+1, W-w+1)，R[y, x] 对应模板左上角放在 (y, x) 处的得分。
    """
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if template.ndim != 2 or image.ndim != 2:
        raise ValueError("仅支持灰度图（2D 数组）")
    h, w = template.shape
    if h > image.shape[0] or w > image.shape[1]:
        raise ValueError("模板尺寸大于图像尺寸")

    t = template - template.mean()
    t_energy = np.sqrt((t * t).sum())
    if t_energy == 0:
        raise ValueError("模板为常量图，无法匹配")

    # 分子：图像与去均值模板的互相关
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")

    # 分母：窗口局部能量（利用积分图加速局部和与局部平方和）
    integral = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    integral_sq = np.pad(image * image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s1 = (
        integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    )
    s2 = (
        integral_sq[h:, w:] - integral_sq[:-h, w:] - integral_sq[h:, :-w] + integral_sq[:-h, :-w]
    )
    n = h * w
    var = s2 - s1 * s1 / n
    var[var < 1e-12] = 1e-12  # 常量窗口防除零
    local_std = np.sqrt(var)

    return num / (local_std * t_energy)


def parabolic_subpixel(score: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """对 (y, x) 处的峰值做 x/y 两方向的抛物线插值，返回亚像素峰位偏移。

    三点抛物线峰值：delta = 0.5*(f[-] - f[+]) / (f[-] - 2f[0] + f[+])。
    """
    def _axis(f_m: float, f_0: float, f_p: float) -> float:
        denom = f_m - 2.0 * f_0 + f_p
        if abs(denom) < 1e-12:
            return 0.0
        delta = 0.5 * (f_m - f_p) / denom
        return float(np.clip(delta, -0.5, 0.5))

    dy = dx = 0.0
    if 1 <= y < score.shape[0] - 1:
        dy = _axis(score[y - 1, x], score[y, x], score[y + 1, x])
    if 1 <= x < score.shape[1] - 1:
        dx = _axis(score[y, x - 1], score[y, x], score[y, x + 1])
    return dy, dx


def find_best_matches(
    image: np.ndarray,
    template: np.ndarray,
    min_score: float = 0.9,
    nms_radius: int = 8,
    max_count: int = 10,
    subpixel: bool = True,
) -> list[dict]:
    """找出评分超过 min_score 的所有实例（非极大值抑制），含亚像素坐标。

    返回字典列表：{x, y, score}，按得分降序。(x, y) 为模板左上角亚像素坐标。
    """
    score = match_template_ncc(image, template)
    cand = np.argwhere(score >= min_score)
    order = np.argsort(-score[cand[:, 0], cand[:, 1]])
    cand = cand[order]

    picked: list[tuple[int, int]] = []
    for y, x in cand:
        if all((y - py) ** 2 + (x - px) ** 2 > nms_radius**2 for py, px in picked):
            picked.append((int(y), int(x)))
        if len(picked) >= max_count:
            break

    results = []
    for y, x in picked:
        if subpixel:
            dy, dx = parabolic_subpixel(score, y, x)
        else:
            dy = dx = 0.0
        results.append(
            {"x": x + dx, "y": y + dy, "score": float(score[y, x])}
        )
    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from data.synthetic import make_template_scene
    from traditional.viz import (
        AUX,
        DET,
        GT,
        draw_box,
        draw_cross,
        image_ax,
        metric_text,
        save,
        setup,
    )

    setup()
    img, tpl, gt_xy = make_template_scene(noise=3.0)
    score = match_template_ncc(img, tpl)
    matches = find_best_matches(img, tpl, min_score=0.6)
    best = matches[0]
    err = np.hypot(best["x"] - gt_xy[0], best["y"] - gt_xy[1])
    print(f"匹配得分={best['score']:.4f}  定位误差={err:.4f} px")

    h, w = tpl.shape
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.45, 1.15, 1]}
    )

    # (a) 场景 + 真值框 vs 检测框
    image_ax(axes[0], img, "场景与匹配结果")
    draw_box(axes[0], gt_xy[0], gt_xy[1], w, h, color=GT, ls="--", lw=1.8, label="真值")
    draw_box(axes[0], best["x"], best["y"], w, h, color=DET, ls="-", lw=1.6, label="检测")
    draw_cross(axes[0], best["x"], best["y"], color=DET, size=5, label="亚像素角点")
    axes[0].legend(loc="lower right")
    metric_text(
        axes[0],
        [
            f"ZNCC 得分  {best['score']:.4f}",
            f"定位误差   {err:.4f} px",
            f"模板尺寸   {w}×{h}",
        ],
    )

    # (b) 完整 NCC 评分曲面（vmin 取实际最小值，拉开背景对比）
    im = axes[1].imshow(score, cmap="magma", vmin=float(score.min()), vmax=1.0)
    axes[1].set_title("ZNCC 评分曲面", pad=6)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_aspect("equal")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("相关系数", fontsize=8)
    axes[1].plot(
        gt_xy[0], gt_xy[1], linestyle="none", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="#7BE0B0", markeredgewidth=1.5,
        label="真值",
    )
    axes[1].plot(
        best["x"], best["y"], marker="+", markersize=10, color="white",
        markeredgewidth=1.8, label="亚像素峰值",
    )
    axes[1].legend(loc="upper right", fontsize=7)

    # (c) 峰值邻域的两条评分剖面 —— 亚像素插值的直接依据
    # 横轴统一为「相对整像素峰值的偏移」，两条曲线才可比较
    yi = int(round(best["y"]))
    xi = int(round(best["x"]))
    half = 4
    offs = np.arange(-half, half + 1)
    if 0 <= xi - half and xi + half < score.shape[1] and 0 <= yi - half and yi + half < score.shape[0]:
        prof_x = score[yi, xi - half: xi + half + 1]
        prof_y = score[yi - half: yi + half + 1, xi]
        axes[2].plot(offs, prof_x, "o-", color=AUX, lw=1.7, ms=4.5, label="沿 x 剖面")
        axes[2].plot(offs, prof_y, "s-", color=DET, lw=1.7, ms=4.5, label="沿 y 剖面")
        axes[2].axvline(best["x"] - xi, color=AUX, ls=":", lw=1.4)
        axes[2].axvline(best["y"] - yi, color=DET, ls=":", lw=1.4)
        axes[2].set_xlabel("相对整像素峰值的偏移（px）")
        axes[2].set_ylabel("ZNCC")
        axes[2].legend(loc="lower center")
    else:
        axes[2].text(0.5, 0.5, "峰值过于贴近边界\n无法取 9 点剖面",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("峰值邻域评分剖面", pad=6)
    axes[2].grid(True, alpha=0.5)
    metric_text(
        axes[2],
        [
            f"Δx = {best['x'] - gt_xy[0]:+.3f} px",
            f"Δy = {best['y'] - gt_xy[1]:+.3f} px",
        ],
        loc="upper left",
        size=7.5,
    )

    fig.tight_layout()
    print("图片 →", save(fig, "ncc_matching"))
