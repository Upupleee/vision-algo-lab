"""对焦数据源抽象：把「离线查表」和「真实拍摄」统一成一个 score_fn(p)。

为什么需要这一层
----------------
`focus_measures` 是纯函数（f(gray) -> float），`focus_search` 只依赖
`score_fn(p) -> float` 回调 —— 两者都不关心 p 背后是查表还是驱动电机。
真正缺的是「帧从哪来」。把这件事抽象出来，同一套搜索策略就能同时跑
合成基线和真实数据集。

真实数据集约定（RTImg/）
-----------------------
目录结构 ``RTImg/<组号>/<NNN>-<DDDD>-<ZZZZ>.<ext>``：

- ``NNN``  目标编号（同一组内一致）
- ``DDDD`` 目标实际距离，单位 mm（Distance To Subject）
- ``ZZZZ`` 拍摄时的镜头对焦距离，单位 mm

同一组内按 ``ZZZZ`` 升序即扫描顺序。

真值定义
--------
数据集**没有**相机自动对焦的 ``-0000`` 参考帧（RTImg/README 里那条约定
在这批数据中不成立）。可用的真值是：**对焦距离最接近目标距离的那一帧**，
即 ``argmin |ZZZZ - DDDD|``。实测 001（DDDD=1260）与 007（DDDD=1040）
两组的评分峰值帧与该定义完全吻合。

性能约定
--------
6016×4016 的 JPEG 解码约 190 ms，是硬成本（降采样解码虽快，但会摧毁
对焦评价的灵敏度，实测 1/8 降采样使相邻帧区分度从 0.05:1.00 塌到
0.52:1.00）。因此本模块**始终全尺寸解码，靠 ROI 裁剪降计算量**，
并在实例内缓存最近若干帧，避免搜索过程中重复读盘。
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

# NNN-DDDD-ZZZZ.ext，扩展名大小写混用（.jpg/.JPG/.png/.PNG）
_NAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.(jpe?g|png|bmp|tiff?)$", re.IGNORECASE)


@dataclass(frozen=True)
class Frame:
    """一帧对焦扫描图像及其元数据。"""

    path: Path
    subject_id: int      # NNN
    subject_dist: float  # DDDD，目标实际距离 mm
    focus_dist: float    # ZZZZ，拍摄时的对焦距离 mm

    @property
    def name(self) -> str:
        return self.path.name


def parse_frame(path: Path) -> Frame | None:
    """从文件名解析元数据；不符合约定返回 None。"""
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    return Frame(
        path=path,
        subject_id=int(m.group(1)),
        subject_dist=float(m.group(2)),
        focus_dist=float(m.group(3)),
    )


class FolderSource:
    """一个目录（或一组目录）里的对焦扫描序列。

    参数
    ----
    paths : 目录路径列表。每个目录视为一个场景，目录内的图按 ZZZZ 排序。
    roi   : 可选 ROI，``(x, y, w, h)`` 绝对像素；或 ``(fx, fy, fw, fh)``
            形式的比例（各值 ≤ 1 时按比例解释）。对焦评价只在 ROI 内进行，
            既提速也排除背景干扰。
    cache_size : 缓存最近读入的帧数，避免搜索时反复读盘。
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        roi: tuple[float, float, float, float] | None = None,
        cache_size: int = 4,
    ) -> None:
        self.roi = roi
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # 单个 str/Path 会被当成字符序列迭代，必须先包成列表
        if isinstance(paths, (str, Path)):
            paths = [paths]

        frames: list[Frame] = []
        for p in paths:
            d = Path(p)
            if not d.is_dir():
                raise NotADirectoryError(f"不是目录：{d}")
            # 先扫本层；本层没有符合命名的图，再往下扫一层子目录。
            # 这样 RTImg/ 和 RTImg/001/ 两种传法都能工作。
            found = [
                fr for fr in (parse_frame(f) for f in sorted(d.iterdir()))
                if fr is not None
            ]
            if found:
                frames.extend(found)
                continue
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    frames.extend(
                        fr for fr in (parse_frame(f) for f in sorted(sub.iterdir()))
                        if fr is not None
                    )
        if not frames:
            raise ValueError(f"目录中未找到符合 NNN-DDDD-ZZZZ 命名的图片：{paths}")

        # 按 (组号, 对焦距离) 排序，保证同一场景内扫描顺序正确
        frames.sort(key=lambda fr: (fr.subject_id, fr.focus_dist))
        self.frames: list[Frame] = frames

        groups: dict[int, list[Frame]] = {}
        for fr in frames:
            groups.setdefault(fr.subject_id, []).append(fr)
        self.groups: dict[int, list[Frame]] = groups

    # ── 基本查询 ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def z_positions(self, subject_id: int | None = None) -> np.ndarray:
        """返回对焦距离序列（mm）。指定 subject_id 则只取该组。"""
        fr = self.frames if subject_id is None else self.groups[subject_id]
        return np.array([f.focus_dist for f in fr], dtype=float)

    def subject_ids(self) -> list[int]:
        return sorted(self.groups)

    # ── 真值 ────────────────────────────────────────────────────
    def gt_frame(self, subject_id: int) -> Frame:
        """真值帧：对焦距离最接近目标距离的那一帧。"""
        fr = self.groups[subject_id]
        return min(fr, key=lambda f: abs(f.focus_dist - f.subject_dist))

    def gt_z(self, subject_id: int) -> float:
        return self.gt_frame(subject_id).focus_dist

    # ── 读图 ────────────────────────────────────────────────────
    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return img
        h, w = img.shape[:2]
        x, y, rw, rh = self.roi
        # 各分量都 ≤ 1 时按比例解释
        if max(x, y, rw, rh) <= 1.0:
            x, y, rw, rh = x * w, y * h, rw * w, rh * h
        x0 = int(round(np.clip(x, 0, w - 1)))
        y0 = int(round(np.clip(y, 0, h - 1)))
        x1 = int(round(np.clip(x + rw, x0 + 1, w)))
        y1 = int(round(np.clip(y + rh, y0 + 1, h)))
        return img[y0:y1, x0:x1]

    def load(self, frame: Frame) -> np.ndarray:
        """读取一帧并裁 ROI。带 LRU 缓存。

        始终用 ``IMREAD_COLOR`` 全尺寸解码：降采样解码（IMREAD_REDUCED_*）
        虽然快几十毫秒，但等效低通滤波，会抹平相邻帧的清晰度差异 ——
        对焦评价恰恰是在测模糊，这条路不能走。
        """
        key = str(frame.path)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit

        img = cv2.imread(key, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"无法读取图片：{frame.path}")
        img = self._apply_roi(img)

        self._cache[key] = img
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def load_at(self, z: float, subject_id: int) -> np.ndarray:
        """取对焦距离最接近 z 的那一帧（真实场景即「移动到 z 再拍一张」）。"""
        fr = self.groups[subject_id]
        best = min(fr, key=lambda f: abs(f.focus_dist - z))
        return self.load(best)

    # ── 交给搜索策略 ────────────────────────────────────────────
    def score_fn(
        self, measure: Callable[[np.ndarray], float], subject_id: int
    ) -> Callable[[float], float]:
        """构造 ``score_fn(z) -> float``，可直接喂给 hill_climb / golden_section。

        注意：返回的是「取最接近 z 的那一帧」的评分，不是插值 ——
        真实镜头也只能停在离散位置上。要亚帧精度，用 gaussian_fit 后处理。
        """

        def _fn(z: float) -> float:
            return float(measure(self.load_at(z, subject_id)))

        return _fn

    def curve(
        self, measure: Callable[[np.ndarray], float], subject_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """一次性算出整组的 (对焦距离, 评分)，用于离线评估与绘图。"""
        fr = self.groups[subject_id]
        zs = np.array([f.focus_dist for f in fr], dtype=float)
        sc = np.array([measure(self.load(f)) for f in fr], dtype=float)
        return zs, sc


class SyntheticSource:
    """合成数据源：包 ``make_focus_stack``，保持回归基线可用。

    接口与 FolderSource 对齐，搜索策略无需区分二者。
    """

    def __init__(self) -> None:
        from data.synthetic import make_focus_stack

        self.stack, self.gt_best, self.positions, self.z_focus = make_focus_stack()
        self._index = {round(float(p), 6): i for i, p in enumerate(self.positions)}

    def __len__(self) -> int:
        return len(self.stack)

    def z_positions(self) -> np.ndarray:
        return np.asarray(self.positions, dtype=float)

    @property
    def gt_z(self) -> float:
        return float(self.z_focus)

    def load_at(self, z: float) -> np.ndarray:
        key = min(self._index, key=lambda q: abs(q - z))
        return self.stack[self._index[key]]

    def score_fn(self, measure: Callable[[np.ndarray], float]) -> Callable[[float], float]:
        return lambda z: float(measure(self.load_at(z)))

    def curve(self, measure: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray]:
        zs = self.z_positions()
        return zs, np.array([measure(f) for f in self.stack], dtype=float)
