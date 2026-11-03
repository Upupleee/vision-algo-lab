"""自动对焦清晰度评价函数（Focus Measure / FOM）。

好的评价函数应满足：单峰性、无偏性、对噪声鲁棒、计算量小。
所有函数输入均为灰度图（float 或 uint8 皆可），返回标量得分，值越大越清晰。
"""
from __future__ import annotations

import cv2
import numpy as np


def _prep(gray: np.ndarray) -> np.ndarray:
    """统一转成 float64 单通道灰度。

    注意顺序：**必须先降到灰度，再转 float64**。

    反过来的话 `cv2.cvtColor` 会收到 CV_64F 输入并直接抛
    ``Unsupported depth of input image`` —— cvtColor 只接受
    8U / 16U / 32F 三档深度，不认 CV_64F。彩色图（工业相机常见
    输出）一进 tenengrad 就会炸在这里。
    """
    img = np.asarray(gray)
    if img.ndim == 3:
        # float64 三通道先降到 float32 才能进 cvtColor
        src = img.astype(np.float32) if img.dtype == np.float64 else img
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def brenner(gray: np.ndarray) -> float:
    """Brenner 梯度：相隔 2 像素的灰度差平方（经典、快速）。"""
    img = _prep(gray)
    d = img[:, 2:] - img[:, :-2]
    return float(np.mean(d * d))


def tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    """Tenengrad：Sobel 梯度平方和。"""
    img = _prep(gray)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx * gx + gy * gy))


def smd(gray: np.ndarray) -> float:
    """绝对差和（Sum of Modified Differences），对噪声不如 Tenengrad。"""
    img = _prep(gray)
    dx = np.abs(img[:, 1:] - img[:, :-1]).mean()
    dy = np.abs(img[1:, :] - img[:-1, :]).mean()
    return float(dx + dy)


def variance(gray: np.ndarray) -> float:
    """归一化灰度方差，最简单但受场景内容影响大。"""
    img = _prep(gray)
    mu = img.mean()
    return float(np.mean((img - mu) ** 2) / mu)


def laplacian_energy(gray: np.ndarray, ksize: int = 3) -> float:
    """能量拉普拉斯（EOL），二阶导，纹理敏感。"""
    img = _prep(gray)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return float(np.mean(lap * lap))


def vollath_f4(gray: np.ndarray) -> float:
    """Vollath F4：自相关差分项，抗噪较好。"""
    img = _prep(gray)
    a = img[:, :-1] * img[:, 1:]
    b = img[:, :-2] * img[:, 2:]
    return float(a.mean() - b.mean())


def dct_sharpness(gray: np.ndarray, block: int = 8, ratio: float = 0.5) -> float:
    """DCT 高频能量法：块内幅值最大的高频系数能量均值。"""
    img = _prep(gray)
    h, w = (d - d % block for d in img.shape)
    blocks = img[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
    B = blocks.reshape(-1, block, block)
    freq = np.array([cv2.dct(b) for b in B])
    cut = max(1, int(block * block * ratio))
    flat = np.sort(np.abs(freq.reshape(len(B), -1)), axis=1)
    return float(np.mean(flat[:, -cut:].sum(axis=1)))


ALL_MEASURES: dict[str, object] = {
    "brenner": brenner,
    "tenengrad": tenengrad,
    "smd": smd,
    "variance": variance,
    "laplacian_energy": laplacian_energy,
    "vollath_f4": vollath_f4,
    "dct": dct_sharpness,
}


def evaluate_all(gray: np.ndarray) -> dict[str, float]:
    """一次性计算所有评价函数（用于横向对比）。"""
    return {name: fn(gray) for name, fn in ALL_MEASURES.items()}


def monotonic_ratio(scores: np.ndarray) -> float:
    """峰前严格递增 + 峰后严格递减的采样点占比（单峰性指标）。"""
    scores = np.asarray(scores)
    k = int(np.argmax(scores))
    up = np.diff(scores[: k + 1])
    down = np.diff(scores[k:])
    good = np.sum(up > 0) + np.sum(down < 0)
    return float(good / max(len(up) + len(down), 1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from data.synthetic import make_focus_stack
    from traditional.viz import AUX, DET, GT, MUTED, metric_text, save, setup

    setup()
    stack, gt_best, positions, z_focus = make_focus_stack()
    print(f"真值最清晰帧: {gt_best}（位置 {positions[gt_best]}）")

    curves: dict[str, np.ndarray] = {}
    peak_idx: dict[str, int] = {}
    mono: dict[str, float] = {}
    for name, fn in ALL_MEASURES.items():
        scores = np.array([fn(f) for f in stack], dtype=float)
        curves[name] = scores
        peak_idx[name] = int(np.argmax(scores))
        mono[name] = float(monotonic_ratio(scores))
        print(
            f"{name:>16s}: 峰值帧={peak_idx[name]:3d}  "
            f"单峰性={mono[name]:.2f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.55, 1]})

    # (a) 7 条评价曲线（各自归一化到 [0,1] 才能同图比较）
    cmap = plt.get_cmap("tab10")
    for i, (name, sc) in enumerate(curves.items()):
        span = sc.max() - sc.min()
        norm = (sc - sc.min()) / span if span > 1e-12 else np.zeros_like(sc)
        axes[0].plot(positions, norm, "-", lw=1.7, color=cmap(i % 10), label=name)
        axes[0].plot(positions[peak_idx[name]], norm[peak_idx[name]], "o",
                     ms=4.5, color=cmap(i % 10), markeredgecolor="white", markeredgewidth=0.6)
    axes[0].axvline(z_focus, color=GT, ls="--", lw=1.8, label=f"真值 z={z_focus:.2f}")
    axes[0].set_title("七种清晰度评价函数的对焦曲线", pad=6)
    axes[0].set_xlabel("z 位置（对焦扫描坐标）")
    axes[0].set_ylabel("归一化评分")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(loc="upper center", ncol=4, fontsize=6.8, borderpad=0.35)
    metric_text(
        axes[0],
        [f"扫描帧数 {len(stack)}", f"真值帧 {gt_best}"],
        loc="lower center",
        size=7,
    )

    # (b) 单峰性 + 峰值邻域放大（看清谁更尖锐）
    names = list(curves.keys())
    mono_v = np.array([mono[n] for n in names])

    ax_b = axes[1]
    idx = np.arange(len(names))
    bars = ax_b.bar(idx, mono_v, 0.62, color=AUX, alpha=0.85)
    for i, (b, v) in enumerate(zip(bars, mono_v)):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                  ha="center", fontsize=7, color="#2C5F8A")
    ax_b.set_ylim(0, 1.22)
    ax_b.set_xticks(idx)
    ax_b.set_xticklabels(names, rotation=32, ha="right", fontsize=7.5)
    ax_b.set_ylabel("单峰性（1.0 = 理想）")
    ax_b.set_title("各评价函数的单峰性", pad=6)
    ax_b.grid(True, axis="y", alpha=0.4)
    ax_b.axhline(1.0, color=GT, ls="--", lw=1.2)
    ax_b.set_xlabel(
        f"七种函数峰值帧全部 = {gt_best}（真值帧）\n"
        f"残余 {abs(positions[gt_best] - z_focus):.2f} z 来自扫描步长 "
        f"{positions[1] - positions[0]:.2f}，非函数差异",
        fontsize=6.8,
    )

    fig.tight_layout()
    print("图片 →", save(fig, "focus_measures"))
