"""卡尺（Caliper）测量工具：一维投影剖面 + 亚像素边缘提取 + 直线拟合。

工作流程（与 Halcon metrology / Cognex 卡尺一致）：
1. 沿给定方向（矩形 ROI 长轴）布置 num_probes 个"卡尺"，每个卡尺是垂直于
   长轴、宽为 roi_width 的短线段；
2. 对每个卡尺沿线段做亚像素双线性采样并平均，得到一维灰度剖面 profile；
3. 在剖面上求带符号梯度峰 → 抛物线亚像素插值 → 得到该卡尺处边缘的
   亚像素位置（长轴坐标 + 横向偏移）；
4. 对所有卡尺的边缘点做直线（PCA）拟合，输出 RMS 残差。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from traditional.subpixel.edge_subpixel import gaussian_fit_subpixel


@dataclass
class CaliperResult:
    positions: np.ndarray  # 边缘点 x 坐标（N,）
    strengths: np.ndarray  # 对应梯度峰值强度（N,）
    points: np.ndarray | None = None  # 边缘点世界坐标 (N, 2)，x/y 各一列
    line: tuple[float, float, float, float] | None = None  # (vx, vy, x0, y0)
    rms: float = float("nan")  # 直线拟合 RMS 残差


@dataclass
class Caliper:
    """矩形卡尺 ROI。

    center: ROI 中心 (x, y)；length: 长轴像素数（卡尺排列方向）；
    width: 每个卡尺的采样宽度（垂直于长轴）；angle_deg: 长轴与 x 轴夹角。
    """

    center: tuple[float, float]
    length: int
    width: int = 9
    angle_deg: float = 0.0
    num_probes: int = 50
    min_strength: float = 5.0
    sigma_smooth: float = 1.0
    results: list[CaliperResult] = field(default_factory=list, repr=False)

    def measure(self, gray: np.ndarray, polarity: int = 0) -> CaliperResult:
        """执行测量。polarity: 0=任意, +1=暗到亮, -1=亮到暗。"""
        gray = np.asarray(gray, dtype=np.float64)
        if gray.ndim != 2:
            raise ValueError("需要灰度图")
        half_w = self.width // 2
        theta = np.deg2rad(self.angle_deg)
        axis = np.array([np.cos(theta), np.sin(theta)])  # 长轴单位向量 (x, y)
        normal = np.array([-np.sin(theta), np.cos(theta)])  # 垂直方向
        ts = np.linspace(0, self.length - 1, self.num_probes)  # 各卡尺沿长轴位置
        ss = np.arange(-half_w, half_w + 1)  # 卡尺内采样偏移
        half_s = (len(ss) - 1) / 2.0

        profiles = np.empty((len(ts), len(ss)))
        for i, t in enumerate(ts):
            base = np.array(self.center) + axis * (t - (self.length - 1) / 2.0)
            pts = base[None, :] + normal[None, :] * ss[:, None]  # (width, 2)
            profiles[i] = _sample_points(gray, pts)

        if self.sigma_smooth > 0:
            k = cv2.getGaussianKernel(int(np.ceil(self.sigma_smooth * 3)) * 2 + 1, self.sigma_smooth)
            profiles = cv2.filter2D(profiles, -1, k.T, borderType=cv2.BORDER_REFLECT)  # k.T: 沿剖面方向平滑

        grad = np.gradient(profiles, axis=1)
        if polarity != 0:
            grad = grad * float(polarity)
        elif np.mean(grad) < 0:
            grad = -grad  # 自动判别极性，统一为"正峰"

        pos_xy = np.empty((len(ts), 2))
        strengths = np.empty(len(ts))
        for i in range(len(ts)):
            k = int(np.argmax(grad[i]))
            strengths[i] = grad[i, k]
            if strengths[i] < self.min_strength:
                pos_xy[i] = np.nan
                continue
            # 亚像素：优先误差函数拟合（精度 ~0.01px），失败时退回抛物线插值
            try:
                s_idx, _, _ = gaussian_fit_subpixel(profiles[i], x0=float(k))
            except Exception:  # noqa: BLE001
                s_idx = k + _parabolic_peak(grad[i], k)
            s_centered = s_idx - half_s  # 卡尺内横向亚像素偏移
            t = ts[i] - (self.length - 1) / 2.0
            pos_xy[i] = np.array(self.center) + axis * t + normal * s_centered

        valid = ~np.isnan(pos_xy[:, 0])
        res = CaliperResult(positions=pos_xy[:, 0], strengths=strengths, points=pos_xy)
        if valid.sum() >= 2 and np.ptp(pos_xy[valid], axis=0).max() > 5:
            (vx, vy, x0, y0), rms = _fit_line(pos_xy[valid])
            res.line = (vx, vy, x0, y0)
            res.rms = rms
        self.results.append(res)
        return res


def _sample_points(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """双线性亚像素采样：pts (N, 2) 为 (x, y) 坐标。"""
    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    h, w = gray.shape
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    wx, wy = x - x0, y - y0
    val = (
        gray[y0, x0] * (1 - wx) * (1 - wy)
        + gray[y0, x1] * wx * (1 - wy)
        + gray[y1, x0] * (1 - wx) * wy
        + gray[y1, x1] * wx * wy
    )
    val[~ok] = np.nan
    if np.isnan(val).any():
        val = np.where(np.isnan(val), np.nanmean(val), val)
    return val


def _parabolic_peak(f: np.ndarray, k: int) -> float:
    """三点抛物线插值亚像素峰位（返回相对 k 的偏移）。"""
    if k <= 0 or k >= len(f) - 1:
        return 0.0
    denom = f[k - 1] - 2 * f[k] + f[k + 1]
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (f[k - 1] - f[k + 1]) / denom, -0.5, 0.5))


def _fit_line(pts: np.ndarray) -> tuple[tuple[float, float, float, float], float]:
    """PCA 直线拟合。返回 ((vx, vy, x0, y0), rms)。"""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    v = vt[0]
    resid = q - np.outer(q @ v, v)
    rms = float(np.sqrt((resid**2).sum(axis=1).mean()))
    return (float(v[0]), float(v[1]), float(c[0]), float(c[1])), rms


if __name__ == "__main__":
    from data.synthetic import make_edge_scene

    img, gt_fn = make_edge_scene(noise=1.0)
    # 长轴沿 +y（angle=90），卡尺横向采样沿 x，边缘满足 x = gt_fn(y)
    cal = Caliper(center=(61.5, 80), length=128, width=11, angle_deg=90.0, num_probes=40)
    res = cal.measure(img)
    pts = res.points
    err = np.abs(pts[:, 0] - gt_fn(pts[:, 1]))
    print(f"边缘位置误差: mean={np.nanmean(err):.4f} px, max={np.nanmax(err):.4f} px")
    print(f"直线拟合 RMS={res.rms:.4f} px")
