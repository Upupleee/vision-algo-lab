"""合成测试数据生成器：每类场景都附带解析真值（亚像素位置 / 对焦位置）。

所有函数固定随机种子（可通过 seed 参数改变），保证实验可复现。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.special import erf


def _smooth_texture(size: tuple[int, int], seed: int = 7, blur: float = 2.0,
                    lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    """平滑随机纹理（模糊噪声），作为模板 / 对焦底图。"""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(lo, hi * 0.4, size)
    tex = cv2.GaussianBlur(noise, (0, 0), blur)
    tex = (tex - tex.min()) / max(np.ptp(tex), 1e-9)
    return tex * (hi - lo) + lo


def make_template_scene(
    img_size: tuple[int, int] = (220, 260),
    tpl_size: tuple[int, int] = (56, 72),
    gt_xy: tuple[float, float] = (121.37, 84.63),
    noise: float = 0.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """模板匹配场景。

    返回 (image, template, gt_xy)：template 被以亚像素平移量 gt_xy
    （左上角坐标）粘贴进 image，噪声为高斯标准差。
    """
    tpl = _smooth_texture(tpl_size, seed=seed).astype(np.float32)
    canvas = np.zeros(img_size, dtype=np.float32)
    h, w = tpl_size
    canvas[:h, :w] = tpl
    # 低对比度背景纹理，避免假峰
    bg = _smooth_texture(img_size, seed=seed + 1, blur=6.0) * 0.25
    M = np.float32([[1, 0, gt_xy[0]], [0, 1, gt_xy[1]]])
    dsize = (img_size[1], img_size[0])  # cv2 的 dsize 是 (W, H)
    warped = cv2.warpAffine(canvas, M, dsize, flags=cv2.INTER_CUBIC)
    mask = cv2.warpAffine(np.ones(img_size, np.float32), M, dsize, flags=cv2.INTER_NEAREST)
    img = np.where(mask > 0, warped, bg.astype(np.float32))
    if noise > 0:
        rng = np.random.default_rng(seed + 2)
        img = img + rng.normal(0, noise, img_size)
    return img.astype(np.float64), tpl.astype(np.float64), gt_xy


def make_edge_profile(
    n: int = 64,
    center: float | None = None,
    sigma: float = 1.2,
    contrast: float = 160.0,
    base: float = 40.0,
    noise: float = 0.0,
    seed: int = 3,
) -> tuple[np.ndarray, float]:
    """一维阶跃边缘剖面（积分高斯模型）。返回 (profile, 真值中心)。"""
    if center is None:
        center = n / 2.0 + 0.37
    x = np.arange(n, dtype=np.float64)
    prof = base + contrast * 0.5 * (1.0 + erf((x - center) / (sigma * np.sqrt(2.0))))
    if noise > 0:
        prof = prof + np.random.default_rng(seed).normal(0, noise, n)
    return prof, center


def make_edge_scene(
    size: tuple[int, int] = (160, 128),
    slope: float = 0.02,
    intercept: float = 60.0,
    sigma: float = 1.0,
    noise: float = 0.0,
    seed: int = 5,
) -> tuple[np.ndarray, object]:
    """二维斜直边场景：边缘满足 x = slope·y + intercept（左暗右亮）。

    返回 (image, gt_fn)，gt_fn(y) 给出该行的边缘亚像素 x 坐标。
    默认斜率很小：y∈[0,160] 上边缘横向移动 ~3.2px，可完整落入
    width=11 的竖直卡尺采样窗内（配合 center_x≈61.5 使用）。
    """
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    d = x - (slope * y + intercept)
    img = 40.0 + 160.0 * 0.5 * (1.0 + erf(d / (sigma * np.sqrt(2.0))))
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0, noise, size)
    return img, (lambda yy: slope * yy + intercept)


def make_corner_scene(
    squares: tuple[int, int] = (6, 5),
    cell: int = 30,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """棋盘格角点场景（经小角度透视变换，角点亚像素真值已知）。

    返回 (image, gt_pts (N,2) xy)。
    """
    rows, cols = squares
    board = np.zeros(((rows + 1) * cell, (cols + 1) * cell), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 255.0

    rng = np.random.default_rng(seed)
    ang = np.deg2rad(4.0 + rng.uniform(0, 2))
    cos, sin = np.cos(ang), np.sin(ang)
    H = np.float32(
        [
            [cos * 0.98, -sin * 0.98, 38.5],
            [sin * 0.98, cos * 0.98, 31.7],
            [1e-5, 2e-5, 1.0],
        ]
    )
    size = (board.shape[1] + 80, board.shape[0] + 80)
    img = cv2.warpPerspective(
        # 注意：不要用 INTER_CUBIC——过冲截断（>255 clip）会使基于梯度的
        # 亚像素精化产生 ~0.9px 系统偏差，见 docs/research_notes.md
        board, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=25.0
    )

    gt = [
        [(c * cell - 0.5, r * cell - 0.5)]  # 半像素约定：4 方块交界在像素角上
        for r in range(1, rows)
        for c in range(1, cols)
    ]
    gt_pts = cv2.perspectiveTransform(np.array(gt, dtype=np.float32), H).reshape(-1, 2)
    return img.astype(np.float64), gt_pts


def make_focus_stack(
    size: tuple[int, int] = (96, 96),
    n_frames: int = 41,
    z_focus: float = 23.37,
    z_range: tuple[float, float] = (0.0, 60.0),
    sigma_floor: float = 0.3,
    sigma_slope: float = 0.25,
    noise: float = 0.3,
    seed: int = 13,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """对焦扫描序列：模拟镜头在不同 z 位置的离焦模糊。

    采用几何离焦模型（无尖点，物理平滑）：
        sigma(z) = sqrt(sigma_floor² + sigma_slope²·(z - z_focus)²)
    注意底图纹理自身模糊 σ_tex 会与离焦模糊按方差叠加（σ_tot² = σ_tex² +
    σ_defocus²），σ_tex 过大会抹平近焦区评分差异，见 docs/research_notes.md。
    返回 (stack (N,H,W), 最清晰帧下标, 各帧 z 位置, 真值对焦位置 z_focus)。
    """
    base = _smooth_texture(size, seed=seed, blur=0.6)
    positions = np.linspace(z_range[0], z_range[1], n_frames)
    stack = np.empty((n_frames, *size))
    for i, z in enumerate(positions):
        sig = np.sqrt(sigma_floor**2 + (sigma_slope * (z - z_focus)) ** 2)
        frame = cv2.GaussianBlur(base, (0, 0), sig)
        if noise > 0:
            frame = frame + np.random.default_rng(seed + 1000 + i).normal(0, noise, size)
        stack[i] = frame
    gt_best = int(np.argmin(np.abs(positions - z_focus)))
    return stack, gt_best, positions, z_focus
