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
    from data.synthetic import make_focus_stack

    from traditional.autofocus.focus_measures import tenengrad

    stack, gt_best, positions, z_focus = make_focus_stack()
    scores = np.array([tenengrad(f) for f in stack])

    print(f"真值对焦位置: {z_focus}")
    print(f"整帧峰值位置: {positions[int(np.argmax(scores))]}")
    print(f"高斯拟合位置: {gaussian_fit(positions, scores):.3f}")

    frame_of = {round(p, 4): f for p, f in zip(positions, stack)}

    def score_at(p: float) -> float:
        key = min(frame_of, key=lambda q: abs(q - p))
        return tenengrad(frame_of[key])

    p1, n1 = hill_climb(score_at, positions[0], positions[-1])
    p2, n2 = golden_section(score_at, positions[0], positions[-1])
    print(f"爬山法  : 位置={p1:.3f}  评测次数={n1}")
    print(f"黄金分割: 位置={p2:.3f}  评测次数={n2}")
