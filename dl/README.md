# 深度学习路线（dl/）

端到端神经网络。每个文件是一个研究方向的模型骨架（定义网络结构、损失、
训练循环入口），需要 `pip install -r requirements-dl.txt`。

| 文件 | 方向 | 参考 |
|------|------|------|
| `autofocus_cnn.py` | 自动对焦 | CNN 从单帧回归离焦量（Pertuz et al. 2019 思路） |
| `learned_matching.py` | 模板匹配 | SuperPoint 特征 + SuperGlue/LoFTR 匹配 |
| `subpixel_edge_net.py` | 亚像素边缘 | U-Net 边缘距离回归（0-1 连续标签） |

## 研究约定

- 数据集构建必须复用 `data/synthetic.py`（含真值）或记录真实采集参数。
- 与传统基线对比实验放在 `docs/experiments.md`，指标用 `tests/metrics.py`。
- 训练脚本固定 seed，记录 checkpoint 到 `runs/`（已 gitignore）。
