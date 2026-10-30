"""学习式特征点匹配：SuperPoint 检测/描述 + 匹配器，替代传统模板匹配。

优势：对旋转、尺度、光照、局部遮挡鲁棒，可直接得到亚像素级配准。
实现路径（三选一，骨架只提供接口约定）：
1. 预训练 SuperPoint + LightGlue（推荐起步，权重公开）；
2. LoFTR 无检测器匹配（弱纹理场景）；
3. 自训练：用 data/synthetic 生成几何增广对，蒸馏训练。

接口约定（与 traditional/template_matching 对齐）：
    match(image, template) -> (x, y, score, H)
    其中 H 为 3x3 单应矩阵，score 为内点率，(x, y) 为模板角点映射均值。
"""
from __future__ import annotations

import numpy as np


def match(image: np.ndarray, template: np.ndarray, min_inliers: int = 30) -> dict:
    """TODO: 接入 SuperPoint/LightGlue 权重后实现。

    步骤：
    1. 两图提取特征点与描述子（亚像素热图 softmax + NMS）；
    2. 描述子匹配（LightGlue 或互近邻 + ratio test）；
    3. cv2.findHomography(RANSAC) 估计单应，内点率即 score；
    4. 模板四角经 H 映射到图像坐标，取均值/中点作为定位输出。
    """
    raise NotImplementedError(
        "请安装 torch 并下载 SuperPoint/LightGlue 权重后实现，"
        "参考 https://github.com/magicleap/SuperPoint"
    )
