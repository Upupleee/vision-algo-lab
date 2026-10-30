"""三条路线（传统/机器学习/深度学习）共用的评测指标。

评测口径统一后，不同方法的结果才能放进同一张表里对比。
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """亚像素定位均方根误差（像素）。自动忽略 NaN。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    mask = ~(np.isnan(pred) | np.isnan(gt))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def localization_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """逐点欧氏定位误差（像素）。输入 (N, 2)。"""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    return np.hypot(pred_xy[:, 0] - gt_xy[:, 0], pred_xy[:, 1] - gt_xy[:, 1])


def unimodality(scores: np.ndarray) -> float:
    """评分曲线单峰性：峰前递增 + 峰后递减的占比（0~1）。"""
    scores = np.asarray(scores, dtype=np.float64)
    k = int(np.argmax(scores))
    up, down = np.diff(scores[: k + 1]), np.diff(scores[k:])
    good = np.sum(up > 0) + np.sum(down < 0)
    return float(good / max(len(up) + len(down), 1))


def bench_ms(fn: Callable, repeat: int = 20, warmup: int = 3) -> float:
    """毫秒级耗时（取中位数），用于记录推理耗时/帧。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))
