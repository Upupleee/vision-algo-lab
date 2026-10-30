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
    from data.synthetic import make_template_scene

    img, tpl, gt_xy = make_template_scene(noise=3.0)
    matches = find_best_matches(img, tpl, min_score=0.6)
    err = np.hypot(matches[0]["x"] - gt_xy[0], matches[0]["y"] - gt_xy[1])
    print(f"匹配得分={matches[0]['score']:.4f}  定位误差={err:.4f} px")
