"""对焦搜索策略：给定逐帧评价函数得分，估计最佳对焦位置。

包含：
- gaussian_fit: 对峰值附近三点做高斯拟合（亚帧精度定位）
- hill_climb:   爬山法（模拟镜头粗扫 + 精调，限定步数）
- golden_section: 黄金分割搜索（要求单峰，最少评测次数）
"""
from __future__ import annotations

import numpy as np


def gaussian_fit(positions: np.ndarray, scores: np.ndarray) -> float:
    """对 (positions, scores) 中最大三点做对数抛物线（高斯）拟合。

    ln s(p) ≈ a(p - p*)² + c 的峰位即最佳对焦位置。返回 p*。
    """
    i = int(np.argmax(scores))
    if i == 0 or i == len(scores) - 1:
        return float(positions[i])
    x_m, x_0, x_p = positions[i - 1], positions[i], positions[i + 1]
    y_m, y_0, y_p = np.log(np.maximum(scores[[i - 1, i, i + 1]], 1e-12))
    denom = (y_m - 2 * y_0 + y_p)
    if abs(denom) < 1e-12:
        return float(x_0)
    delta = 0.5 * (y_m - y_p) / denom
    # 步长可能不均匀时按线性内插修正
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(x_0 + delta * (x_p - x_m) / 2.0)  # delta 单位是"步"，半间隔换算


def hill_climb(
    score_fn, lo: float, hi: float, coarse_steps: int = 8, fine_iters: int = 6
) -> tuple[float, int]:
    """两阶段爬山：粗扫全范围 → 在最佳点邻域逐步缩半步长。

    score_fn(p) 为给定位置的评价得分。返回 (最佳位置, 总评测次数)。
    """
    n_eval = 0

    def probe(p):
        nonlocal n_eval
        n_eval += 1
        return score_fn(p)

    ps = np.linspace(lo, hi, coarse_steps)
    vals = [probe(p) for p in ps]
    k = int(np.argmax(vals))
    best, best_v = float(ps[k]), vals[k]

    step = (hi - lo) / (coarse_steps - 1)
    for _ in range(fine_iters):
        step /= 2
        for p in (best - step, best + step):
            v = probe(p)
            if v > best_v:
                best, best_v = p, v
    return best, n_eval


def golden_section(
    score_fn, lo: float, hi: float, tol: float = 1e-2
) -> tuple[float, int]:
    """黄金分割搜索。要求 scores 单峰。返回 (最佳位置, 评测次数)。"""
    invphi = (np.sqrt(5.0) - 1) / 2.0
    n_eval = 0

    def probe(p):
        nonlocal n_eval
        n_eval += 1
        return score_fn(p)

    a, b = lo, hi
    x1 = b - invphi * (b - a)
    x2 = a + invphi * (b - a)
    f1, f2 = probe(x1), probe(x2)
    while (b - a) > tol:
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + invphi * (b - a)
            f2 = probe(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - invphi * (b - a)
            f1 = probe(x1)
    return float((a + b) / 2), n_eval


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from data.synthetic import make_focus_stack
    from traditional.autofocus.focus_measures import tenengrad
    from traditional.viz import AUX, DET, GT, MUTED, metric_text, save, setup

    setup()
    stack, gt_best, positions, z_focus = make_focus_stack()
    scores = np.array([tenengrad(f) for f in stack])

    p_gauss = gaussian_fit(positions, scores)
    p_frame = float(positions[int(np.argmax(scores))])
    print(f"真值对焦位置: {z_focus}")
    print(f"整帧峰值位置: {p_frame}")
    print(f"高斯拟合位置: {p_gauss:.3f}")

    frame_of = {round(p, 4): f for p, f in zip(positions, stack)}

    def score_at(p: float) -> float:
        key = min(frame_of, key=lambda q: abs(q - p))
        return tenengrad(frame_of[key])

    p1, n1 = hill_climb(score_at, positions[0], positions[-1])
    p2, n2 = golden_section(score_at, positions[0], positions[-1])
    print(f"爬山法  : 位置={p1:.3f}  评测次数={n1}")
    print(f"黄金分割: 位置={p2:.3f}  评测次数={n2}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})

    # (a) 评分曲线 + 三点高斯拟合 + 三种定位结果
    ax = axes[0]
    ax.plot(positions, scores, "o-", color=MUTED, lw=1.2, ms=3.4, label="逐帧评分")
    i = int(np.argmax(scores))
    if 0 < i < len(scores) - 1:
        xs3 = positions[i - 1: i + 2]
        ys3 = np.log(np.maximum(scores[i - 1: i + 2], 1e-12))
        coef = np.polyfit(xs3, ys3, 2)
        xf = np.linspace(xs3[0], xs3[-1], 80)
        ax.plot(xf, np.exp(np.polyval(coef, xf)), "-", color=DET, lw=1.9,
                label="三点高斯拟合")
    ax.axvline(z_focus, color=GT, ls="--", lw=1.9, label=f"真值 z={z_focus:.2f}")
    for val, col, name, mk in (
        (p_frame, MUTED, f"整帧峰值 {p_frame:.2f}", "s"),
        (p_gauss, DET, f"高斯拟合 {p_gauss:.2f}", "^"),
        (p1, AUX, f"爬山 {p1:.2f}", "D"),
        (p2, "#9C27B0", f"黄金分割 {p2:.2f}", "v"),
    ):
        ax.axvline(val, color=col, ls=":", lw=1.3, alpha=0.85)
        ax.plot(val, np.interp(val, positions, scores), marker=mk, ms=7,
                color=col, markeredgecolor="white", markeredgewidth=0.8,
                zorder=7, label=name)
    ax.set_title("对焦评分曲线与四种定位结果", pad=6)
    ax.set_xlabel("z 位置（对焦扫描坐标）")
    ax.set_ylabel("Tenengrad 评分")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left", fontsize=6.4, borderpad=0.32, handlelength=1.5,
              labelspacing=0.32)
    ax.set_xlim(positions[0] - 2, positions[-1] + 2)
    ax.set_ylim(-max(scores) * 0.04, max(scores) * 1.16)
    metric_text(
        ax,
        [
            f"真值         {z_focus:.3f}",
            f"高斯拟合误差 {abs(p_gauss - z_focus):.3f}",
            f"爬山误差     {abs(p1 - z_focus):.3f}",
            f"黄金分割误差 {abs(p2 - z_focus):.3f}",
        ],
        loc="lower right",
        size=7.2,
    )

    # (b) 评测次数对比 —— 搜索策略的核心成本指标
    ax2 = axes[1]
    labels = ["整帧峰值\n(全扫)", "高斯拟合\n(仅峰值3点)", "爬山法", "黄金分割"]
    counts = [len(positions), 3, n1, n2]
    colors = [MUTED, DET, AUX, "#9C27B0"]
    bars = ax2.bar(range(len(labels)), counts, color=colors, alpha=0.88, width=0.6)
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2, c + 0.6, str(c),
                 ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("评价函数调用次数")
    ax2.set_title("搜索策略的评测成本", pad=6)
    ax2.grid(True, axis="y", alpha=0.4)
    ax2.set_ylim(0, max(counts) * 1.22)

    fig.tight_layout()
    print("图片 →", save(fig, "focus_search"))
