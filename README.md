# Vision Algo Lab — 工业视觉算法研究仓库

研究**模板匹配、卡尺测量、亚像素处理、自动对焦**等工业视觉核心算法。
三大技术路线**物理隔离**，互不依赖：

```
traditional/   传统算法（图像处理/几何优化，无学习成分）
ml/            机器学习（手工特征 + 浅层学习器）
dl/            深度学习（端到端神经网络）
```

> 规则：`traditional/` 内的代码**禁止** import `ml/`、`dl/` 及任何学习框架；
> 三条路线只通过 `data/` 中的统一测试数据和相同的评测指标（`tests/metrics.py`）对话。

## 目录结构

```
vision-algo-lab/
├── traditional/                 # ── 传统算法 ──
│   ├── template_matching/
│   │   ├── ncc_matching.py      # 归一化互相关（FFT 加速）+ 抛物线亚像素
│   │   └── shape_based.py       # 基于梯度方向的形状匹配（Halcon 风格简化版）
│   ├── caliper_measurement/
│   │   └── caliper.py           # 卡尺工具：投影剖面 + 边缘提取 + 亚像素定位
│   ├── subpixel/
│   │   ├── edge_subpixel.py     # 高斯剖面拟合 / 梯度抛物线 / Zernike 矩边缘
│   │   └── corner_subpixel.py   # 亚像素角点精化
│   └── autofocus/
│       ├── focus_measures.py    # 清晰度评价函数（Tenengrad/Brenner/方差/...）
│       └── focus_search.py      # 对焦搜索：高斯拟合 / 爬山 / 黄金分割
├── ml/                          # ── 机器学习 ──
│   └── hog_svm_matching.py      # HOG 特征 + SVM 模板验证
├── dl/                          # ── 深度学习 ──
│   ├── autofocus_cnn.py         # CNN 对焦评分回归（骨架）
│   ├── learned_matching.py      # 学习式特征匹配（SuperPoint/LoFTR 思路，骨架）
│   └── subpixel_edge_net.py     # U-Net 亚像素边缘回归（骨架）
├── data/
│   └── synthetic.py             # 合成测试数据（含亚像素/对焦真值）
├── tests/
│   ├── metrics.py               # 三路线共用的评测指标
│   └── test_smoke.py            # 冒烟测试：python tests/test_smoke.py
└── docs/                        # 论文笔记 / 实验记录
```

## 快速开始

```bash
# 传统算法部分
pip install -r requirements.txt
python -m tests.test_smoke                          # 全部冒烟测试
python -m traditional.autofocus.focus_measures      # 单模块 demo（python -m 方式运行）

# 机器学习部分（可选）
pip install -r requirements-ml.txt
python -m ml.hog_svm_matching

# 深度学习部分（可选）
pip install -r requirements-dl.txt
```

## 各模块研究要点

| 方向 | 传统路线 | 机器学习路线 | 深度学习路线 |
|------|---------|-------------|-------------|
| 模板匹配 | NCC、形状匹配（梯度方向投票） | HOG + SVM 验证、随机森林关键点筛选 | SuperPoint/LoFTR、度量学习 |
| 卡尺测量 | 投影剖面 + 边缘对拟合 | 学习型分类器区分真假边缘 | 分割网络提取边缘区域 |
| 亚像素 | 梯度抛物线、高斯拟合、Zernike 矩、灰度矩 | 回归模型修正传统结果 | 亚像素边缘/角点回归网络 |
| 自动对焦 | Tenengrad 等评分曲线 + 拟合搜索 | 特征 → 对焦位置回归 | CNN 直接从图像预测离焦量 |

## 评测约定

- 亚像素精度：以合成数据的解析真值计算 **RMSE（像素）**，目标 < 0.05 px。
- 模板匹配：定位误差（px）+ 重复性 + 耗时（ms/帧）。
- 自动对焦：评分曲线单峰性、搜索步数、最终离焦量。
- 所有结果记录到 `docs/experiments.md`，实验脚本必须可复现（固定随机种子）。
