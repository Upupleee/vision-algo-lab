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

## 文档入口

[工业视觉文档导航](docs/README.md)：按 12 个领域查阅传统 / ML / DL 路线、应用排障、已知状态与现有资料，并提供横向基础、评测目录和阅读路径。

[演进节点与算法卡覆盖核对](docs/evolution_coverage.md)：逐主时间线节点链接具体卡片，区分已有、部分、缺卡与非算法，列出五层／问答、证据及实现边界，并提供跨域重复说明和去重缺口；不代表算法复现完成。

2026-11-06文档同步：**255卡不增；113节点＝83已有＋2部分＋4缺卡＋24非算法，剩6项覆盖任务**。G44机制已有，但证明一致性与采样安全[独立P0保留](docs/evolution_coverage.md#g44-independent-p0)，不降级、不升级为实测；[原材料最小缺页清单](docs/evolution_coverage.md#minimum-source-pages)与[笔记撤回追踪](docs/research_notes.md#zernike-retraction-20261106)分别列明。

## 快速开始

### 环境准备（首次执行一次）

本机没有系统级 Python，唯一可用解释器是 WorkBuddy 自带的
`C:\Users\Administrator\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe`。
它的 `Scripts\` 不在 PATH 里，**所以裸敲 `pip` 会报「无法将 pip 项识别为...」**。

在项目根建虚拟环境（`.venv/` 已在 `.gitignore` 中）：

```powershell
# 1. 建环境（只做一次）
& C:\Users\Administrator\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe -m venv .venv

# 2. 激活
.\.venv\Scripts\Activate.ps1
```

> 若激活被拦截，先执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

激活后提示符前会出现 `(.venv)`，此后 `python` / `pip` 都指向环境内，
裸敲 `pip install xxx` 正常可用。

### 安装依赖

```powershell
pip install -r requirements.txt          # 传统算法线（numpy/opencv/scipy/matplotlib）
```

**镜像源注意：** 清华源 `pypi.tuna.tsinghua.edu.cn` 对 pip 26.x 的 UA 返回 403，
表现为 `ERROR: No matching distribution found for numpy`。改用阿里云源：

```powershell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

环境内已放置 `.venv\pip.ini` 固化阿里云源，重建环境后需重新放置。

### 运行

**统一用 `-m` 模块方式运行，不要用脚本路径直跑。**

各模块 `__main__` 里用的是 `from data.synthetic import ...` 这类包内绝对导入。
直跑脚本时 `sys.path[0]` 取的是**脚本所在目录**，不是项目根，于是 `data` 包
根本不在搜索路径里，报 `ModuleNotFoundError: No module named 'data'`。

> 关键：`cwd` 是不是项目根，跟 import 能不能找到包**没有关系**。
> 决定这件事的是 `sys.path`，而 `sys.path[0]` 在两种模式下取值不同：
>
> | 运行方式 | `sys.path[0]` |
> |---|---|
> | `python traditional/xxx/ncc_matching.py` | `...\traditional\template_matching`（脚本目录）|
> | `python -m traditional.template_matching.ncc_matching` | 项目根（= cwd）|

#### 完整入口清单（2026-10-30 实测）

| 模块 | 用途 | 状态 |
|---|---|---|
| `tests.test_smoke` | 冒烟测试，8 项 | ✅ 8/8 通过 |
| `tests.test_pose_ncc` | pose_ncc 回归测试，37 项 | ✅ OK（5.8s）|
| `tests.benchmark_pose_ncc` | 场景评测（输出 JSON）| ✅ 可运行 |
| `traditional.template_matching.ncc_matching` | NCC 匹配 demo | ✅ 误差 0.2013 px |
| `traditional.template_matching.shape_based` | 形状匹配 demo | ✅ 误差 0.73 px |
| `traditional.caliper_measurement.caliper` | 卡尺测量 demo | ✅ 直线 RMS 0.0112 px |
| `traditional.subpixel.edge_subpixel` | 亚像素边缘 demo | ✅ 中位误差 0.0128 px |
| `traditional.subpixel.corner_subpixel` | 亚像素角点 demo | ✅ 手写精化 0.047 px |
| `traditional.autofocus.focus_measures` | 清晰度评价函数（8 种）| ✅ 单峰性均 ≥0.93 |
| `traditional.autofocus.focus_search` | 对焦搜索 | ✅ 黄金分割 21 次 |
| `traditional.autofocus.real_dataset` | **真实数据集 RTImg 评估** | ✅ 12 组 × 7 函数 |
| `ml.hog_svm_matching` | HOG + SVM | ⛔ 缺 scikit-learn |
| `dl.autofocus_cnn` | CNN 对焦回归 | ⛔ 缺 torch |
| `dl.subpixel_edge_net` | U-Net 边缘回归 | ⛔ 缺 torch |

```powershell
# 当前环境（.venv，仅装 requirements.txt）可直接跑
python -m tests.test_smoke
python -m unittest tests.test_pose_ncc
python -m traditional.template_matching.ncc_matching

# 机器学习线（可选，需先 pip install -r requirements-ml.txt）
python -m ml.hog_svm_matching

# 深度学习线（可选，需单独环境，见下）
python -m dl.autofocus_cnn
```

#### VS Code

`.vscode/launch.json` 已配好常用入口，按 `F5` 选择即可。

> **右上角绿色三角「Run Python File」不要点。** 它执行的是
> `python <文件完整路径>`，绕过 `launch.json`，必然触发
> `ModuleNotFoundError`。请用 `F5`，或点绿三角旁的下拉箭头选
> `Debug Python File`。

#### 可视化输出

每个 demo 跑完都会把结果图存到 `data/output/`（已在 `.gitignore` 中忽略）：

| 图片 | 内容 |
|---|---|
| `ncc_matching.png` | 场景+真值/检测框 · ZNCC 评分曲面 · 峰值邻域评分剖面 |
| `shape_based.png` | 场景+匹配框 · 模板梯度特征点与方向 |
| `caliper.png` | 卡尺 ROI+边缘点+拟合线 · 逐点偏差 · 单条卡尺剖面与梯度峰 |
| `edge_subpixel.png` | 1D 阶跃剖面三法对比（含过渡区放大）· 2D Devernay 边缘点 |
| `corner_subpixel.png` | 角点四色对比 · 单角点放大 · 逐点误差（对数刻度）|
| `focus_measures.png` | 七种清晰度函数对焦曲线 · 单峰性对比 |
| `focus_search.png` | 评分曲线+高斯拟合+四种定位结果 · 评测成本对比 |

绘图统一走 `traditional/viz.py`：中文字体（雅黑）、固定配色语义
（真值青绿虚线 / 检测橙红实线 / 辅助蓝）、白底。默认只存盘，
加 `--show` 可弹窗交互查看：

```powershell
python -m traditional.template_matching.ncc_matching --show
```

### 真实数据集（RTImg/）

`RTImg/` 是真实拍摄的对焦扫描数据，12 个场景共 110 帧，6016×4016 BGR。

**命名约定** `<NNN>-<DDDD>-<ZZZZ>.<ext>`：

| 段 | 含义 |
|---|---|
| `NNN` | 目标编号（同组一致） |
| `DDDD` | 目标实际距离 mm（DTS） |
| `ZZZZ` | 拍摄时的镜头对焦距离 mm |

**真值定义**：数据集**没有** `-0000` 自动对焦参考帧（`RTImg/README` 里那条
约定在这批数据中不成立）。采用「对焦距离最接近目标距离的那一帧」，
即 `argmin |ZZZZ − DDDD|`。

**数据源抽象** `traditional/autofocus/source.py`：

- `FolderSource` —— 读真实目录，`score_fn(measure, subject_id)` 返回
  `score_fn(z) -> float`，可直接喂给 `hill_climb` / `golden_section`
- `SyntheticSource` —— 包 `make_focus_stack`，保持回归基线

```powershell
python -m traditional.autofocus.real_dataset                     # 默认中心 70% ROI
python -m traditional.autofocus.real_dataset --roi none          # 全图
python -m traditional.autofocus.real_dataset --roi 0.3,0.15,0.4,0.4
```

**性能现实**（6016×4016，实测）：

| 项 | 耗时 |
|---|---|
| JPEG 解码（硬成本） | ~190 ms/帧 |
| Tenengrad 全图 | ~335 ms |
| Tenengrad 中心 70% ROI | ~24 ms |
| 110 帧 × 7 个函数（ROI） | **175 s** |

> **降采样不可行。** `IMREAD_REDUCED_*` 虽能把解码降到 110 ms，但等效
> 低通滤波会摧毁对焦灵敏度 —— 实测 1/8 降采样使相邻帧区分度从
> `0.05:1.00` 塌到 `0.52:1.00`。评价函数测的正是模糊，不能先用模糊把
> 信号吃掉。**优化只能靠 ROI，不能靠降采样。**

**评估结论**（12 组命中数，2026-10-30 实测）：

| 函数 | 命中 | 平均误差 mm | 备注 |
|---|---|---|---|
| `brenner` | **11/12** | 2.9 | 最稳 |
| `vollath_f4` | **11/12** | 2.9 | 最稳 |
| `tenengrad` | 10/12 | 3.2 | |
| `laplacian_energy` | 10/12 | 4.6 | |
| `smd` | 9/12 | 4.8 | |
| `variance` | 7/12 | **441** | 对内容敏感，不可靠 |
| `dct` | **2/12** | **816** | **在大图上失效** |

> `dct_sharpness` 按 8×8 块取高频能量**均值**，在 2400 万像素图上有约
> 47 万个块，目标细节被大量平坦背景块稀释，峰值基本随机。要用它得改成
> 取块能量的高分位数，或限定到更小的分析窗。

### 深度学习线单独建环境

`requirements-dl.txt` 的 `torch>=2.2` 要求 `numpy<2`，而传统线用的是
numpy 2.5.x。两者不能共处一个环境，dl 线单独建：

```powershell
& C:\Users\Administrator\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe -m venv .venv-dl
.\.venv-dl\Scripts\Activate.ps1
pip install -r requirements-dl.txt -i https://mirrors.aliyun.com/pypi/simple
```

> `.venv-dl/` 尚未加入 `.gitignore`，需要时自行补上。

## 各模块研究要点

### 旋转／尺度 ZNCC 基线

新增 `traditional/template_matching/pose_ncc.py`，支持旋转／尺度离散搜索、空间金字塔、二值掩膜、多目标与无目标拒识。原 `ncc_matching.py` 和形状匹配接口保持不变。

- [接口、坐标与调参说明](docs/pose_ncc.md)
- [合成场景实测与已知边界](docs/experiments.md)
- 回归测试：`python -B -W error -m unittest tests.test_pose_ncc`
- 场景评测：`python -B -W error -m tests.benchmark_pose_ncc`

默认开启全分辨率复核，不能视为实时优化版本。角度和尺度精度受采样步长限制；深度学习模块仍为骨架，以下表格包含研究方向而非全部已实现功能。

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
