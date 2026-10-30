"""自动对焦清晰度评价函数（Focus Measure / FOM）。

好的评价函数应满足：单峰性、无偏性、对噪声鲁棒、计算量小。
所有函数输入均为灰度图（float 或 uint8 皆可），返回标量得分，值越大越清晰。
"""
from __future__ import annotations

import cv2
import numpy as np


def _prep(gray: np.ndarray) -> np.ndarray:
    img = np.asarray(gray, dtype=np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


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
    from data.synthetic import make_focus_stack

    stack, gt_best, positions, _ = make_focus_stack()
    print(f"真值最清晰帧: {gt_best}（位置 {positions[gt_best]}）")
    for name, fn in ALL_MEASURES.items():
        scores = [fn(f) for f in stack]
        print(
            f"{name:>16s}: 峰值帧={int(np.argmax(scores)):3d}  "
            f"单峰性={monotonic_ratio(scores):.2f}"
        )
