# 自动对焦与成像清晰度：方法演进与工程实践双线总览

[文档导航](README.md) · [编写与学习规范](algorithm_learning_standard.md) · [测量与对焦卡片索引](algorithm_cards/README.md#测量与对焦) · [独立测量总览](measurement_evolution.md) · [实验记录](experiments.md)

2026-11-06覆盖回链：[SF94](algorithm_cards/sff_haralick_extensions.md#sf94)承接1994门控SML、逐像素焦栈、相邻三点／查表与台位移到高度，[G12／11-06](evolution_coverage.md#11-对焦成像质量)部分转已有／P2；核／核／部／核（限定等距重写、尺度还原与条件敏感性）／部、答5。SFF系统未实现，查表离散、有效域、F10非等距／零分及设备标定缺口保留；1994证据不回填1989全部细节，不把全局gaussian_fit或RTImg当逐点深度验收。主表八节点及原来源不改，G09／G10／G11仍P0缺卡，有限检查仅历史。

## 1. 范围与证据口径

本章以基于图像的自动对焦为主：从清晰度评分到焦点曲线、采样搜索、离焦回归与运动闭环。**方法演进线**解释评分／模型／控制分别改变什么；**工程实践线**连接现有函数、数据代理真值、选型、排障和验收。Shape-from-focus（SFF）提供局部评分与峰位插值的相关证据，但其输出是逐点深度，不等于全图自动对焦；图像复原、相位检测 AF、测距硬件和景深合成也不是当前仓库已具备的能力。

核验日期：**2026-11-04**。本章区分原始提出／早期应用、比较研究、综述归纳和具体实现变体，不为所有指标指定唯一首创年。正式出版、预印本、技术报告、在线化和软件版本分别记录；仅填来源实际给出的日期。全文入口返回截断时只记“全文片段”。本文没有统一性能实验，不给通用指标排名、固定帧数收敛或设备速度承诺。

七种评分仅为 [F01–F07](algorithm_cards/measurement_focus.md#f01-brenner)；DCT-II 是变换，单峰性是诊断，三点高斯是峰位模型，爬山／黄金分割／全扫描是搜索，FocusNet 是网络与训练循环骨架，不能把 F01–F14 称为十四种清晰度评分。建立总览不升级卡片 L1–L5 或实现状态。

## 2. 核心演进维度与双线连接

| 核心维度 | 持续解决的问题 | 演进机制与代价 | 工程必须观察什么 |
|---|---|---|---|
| 清晰度评价 | 如何把离焦引起的信息变化变成可比较标量 | 差分／梯度、二阶能量、自相关、统计与频域选频并行；频带和聚合定义决定敏感性 | 原分数、响应支持、ROI、曝光、灰度范围、饱和与锐化 |
| 纹理与噪声鲁棒性 | 内容、方向、噪声可能比焦差更影响评分 | 双方向、多尺度、相关统计、内容选区或学习特征；仍依赖假设和样本 | 峰邻域分差、重复采集波动、方向盲区、多 ROI 一致性 |
| 焦点曲线模型 | 离散最大帧不足以估连续焦点，且并非总单峰 | 局部高斯／二次模型、置信检查；平底、偏斜、多深度会失效 | 原始位置轴、是否等距、内部峰、平台、负值及拟合残差 |
| 采样与搜索 | 少拍少算又不漏焦点 | 全扫、粗细搜索、区间复用、受控探索；少调用以可用峰形和初值换取 | 请求位置、实际位置／帧 ID、独立曝光数、错峰与超时 |
| 运动闭环 | 评分最优如何转成可靠设备位置 | 到位检测、回差、稳定等待、同步、再评分和安全回退 | 时间戳、限位、运动方向、实际到位误差、闭环成功／失败 |
| 耗时与学习成本 | 算分快不等于系统快，回归快不等于无训练成本 | ROI／缓存／复用，或离焦学习；减少在线扫描同时增加训练与部署管理 | 解码、评分、传输、运动、等待和尾延迟；数据与权重版本 |

梯度、自相关与综述比较支撑评分分支；Nayar–Nakagawa 支撑局部峰模型；Pinkard 等说明成像设计可以和学习回归组合，而不是依赖网络从不可辨识图像中猜方向。[cite:1](#af-source-1)[cite:3](#af-source-3)[cite:4](#af-source-4)[cite:5](#af-source-5)[cite:6](#af-source-6)[cite:7](#af-source-7)[cite:8](#af-source-8)

| 工程环节 | 输入／输出及可替换角色 | 仓库状态 |
|---|---|---|
| 图像和位置来源 | 固定场景、位置与采集条件 → 帧 | `FolderSource`／`SyntheticSource` 是文件／合成数据，不是相机电机驱动 |
| 评分 | 灰度／BGR 图像 → 标量 | 七种函数已有实现，但无统一空图／非有限值拒绝 |
| 曲线诊断 | 按位置排序的分数 → 峰形与可信性 | 有严格增减比例；没有完整多峰／噪声可信度决策 |
| 搜索／回归 | 评分回调或网络输入 → 位置／离焦建议 | 爬山、黄金分割、局部高斯已实现；FocusNet 无已验证权重 |
| 执行与验证 | 限幅动作、到位采图、复核 → 成功／拒绝／回退 | **真实硬件闭环未实现**；回调签名不能证明闭环存在 |

## 3. 方法演进线：代表性里程碑

下表列 **8 个代表节点**。综述和比较研究是“评价方法”的里程碑，不是其全部被比较指标的发明时间；单列同作者相邻两年的不同论文，避免用一个年份覆盖所有变体。

| 年份／版本与分支 | 先前瓶颈 → 核心改进 | 新增代价与遗留问题 | 场景与卡片对应 | 原始来源／仓库状态 |
|---|---|---|---|---|
| 1970，Tenenbaum 学位论文／AIM-134 报告 | 固定采集不能满足任务 → 将相机调节纳入视觉任务，讨论梯度清晰度与对焦测距 | 内容、动态范围、噪声及控制过程共同影响结果；不等于现代 Tenengrad 变体逐式一致 | 梯度 AF 与主动成像；[F02](algorithm_cards/measurement_focus.md#f02-tenengrad) | [cite:1](#af-source-1)；**双向 Sobel 能量已实现**，原系统未复现 |
| 1976，Brenner 等／显微早期应用 | 自动显微系统需要自动采集与质量控制 → 计算机、相机和伺服平台的系统实践，后世将跨距差评分关联于此文 | 本次摘要不足以确认跨距差公式首创；需原文逐页核对 | 显微图像评分；[F01](algorithm_cards/measurement_focus.md#f01-brenner) | [cite:2](#af-source-2)；**横向二像素差平方均值已实现**，论文系统未实现 |
| 1985，Groen–Young–Ligthart／比较研究 | 指标选择靠经验 → 在明确图像和采集条件下比较已有对焦函数 | 测试条件有限，结果不构成所有图像的排序；文章明确比较既有函数 | 梯度／Laplacian／统计评分选型；[F02](algorithm_cards/measurement_focus.md#f02-tenengrad)、[F04](algorithm_cards/measurement_focus.md#f04-归一化方差-variance)、[F05](algorithm_cards/measurement_focus.md#f05-laplacian-energy) | [cite:3](#af-source-3)；**有相关评分变体**，未复现实验 |
| 1987，Vollath／相关评分 | 微分评分受噪声影响 → 从自相关构造对焦准则，分析不同对比度响应 | 抗噪依赖噪声相关性，有限支持和纹理仍影响曲线 | 相关统计分支；[F06](algorithm_cards/measurement_focus.md#f06-vollath-f4-自相关差) | [cite:4](#af-source-4)；**两个滞后相关均值之差已实现**，原公式全文未取得 |
| 1988，Vollath／场景与噪声分析 | 工作范围、唯一焦点与噪声行为需统一评价 → 系统研究准则性质并讨论新准则 | 不应把该文与 1987 同题混合；没有支持任意噪声下 F4 最优 | 噪声／工作范围选型与[F09](algorithm_cards/measurement_focus.md#f09-单峰性比例)诊断关联，非其首创 | [cite:5](#af-source-5)；**未复现论文比较**；2011 在线登记不是提出年 |
| 1989 报告／1994 期刊，Nayar／Nayar–Nakagawa／SFF | 局部焦栈的离散峰限制深度估计 → SML 局部评分与高斯峰插值 | 需纹理、焦栈与成像假设；局部深度不等于全图 AF；不支持任意曲线必高斯 | 峰邻域精化；[F10](algorithm_cards/measurement_focus.md#f10-三点高斯峰位)、[F13](algorithm_cards/measurement_focus.md#f13-全扫描与数据源) | [cite:6](#af-source-6)；**仅同类局部拟合已有**，SML 与 SFF 深度链未实现 |
| 2012 在线／2013 卷期，Pertuz–Puig–Garcia／综述及比较 | 多类评分缺少一致条件比较 → 按工作原理归类并考察噪声、对比度、饱和、窗口 | 部分 AF 指标经修改用于 SFF，不能直接沿用其结论到整图 AF；不是各指标首创 | [F01–F07](algorithm_cards/measurement_focus.md#f01-brenner)定义审计与选型 | [cite:7](#af-source-7)；**七个本地变体**，不是论文完整方法集／统一复现 |
| 2019，Pinkard 等／成像设计＋DL 回归 | 多帧扫描慢、普通单帧方向可能歧义 → 离轴相干照明＋傅里叶特征网络预测离焦 | 需相应光路、训练焦栈、场景泛化与执行复核；不是普通灰度 CNN 自动拥有的能力 | [F14](algorithm_cards/measurement_focus.md#f14-focusnet-有符号离焦回归)为概念对照，无 FCFNN 专卡 | [cite:8](#af-source-8)；**原方法未实现**；本地 FocusNet 是网络＋训练循环骨架 |

## 4. 指标起源、综述与版本边界

### 4.1 早期来源不等于所有现代实现的首创证明

Tenenbaum 的原扫描件内页为 **1970 年 9 月学位论文**，报告封面为 **1970 年 10 月 AIM-134／CS 182**；检索中也出现后续 1971 书目。此处按实读封面选择 1970，并保留学位论文与报告月份不同。目录出现“梯度幅值累积前需阈值”的内容，但正文相应公式未完整取得，不能声称本地无阈值 `mean(gx²+gy²)` 与原文逐式相同，也不把 Sobel 算子的历史当作 Tenengrad 的唯一首创证据。[cite:1](#af-source-1)

Brenner等1976论文书目可核，不据此声称跨二像素评分首创。初建总览只知二手1971年份冲突；2026-11-06同步[G10逐字段证据](algorithm_cards/focus_origins.md#g10-citation-1971-closed)：Santos p.272完整条目已与1976题名／卷页／作者对应，确认该条错年，且有作者拼写差异。最早误引成因仍未确定，但不再把完整参考表列为缺页；G10整体缺卡／P0仍保留，1976评分／搜索正文未取得，F01现代定义不替代原机制。[cite:2](#af-source-2)

G09／G11亦仍缺卡P0，历史目标五层全部分；[最新最小原页清单](algorithm_cards/focus_origins.md#audit-20261106)区分1970目标章节、1971另一论文及1987／1988版本。临时同页图不归因用户，不增独立来源；本地定义算术检查不是原算法、焦栈性能或设备实测，本轮同步未重跑。

Vollath **1987** 的 *Automatic focusing by correlative methods* 与 **1988** 的 *The influence of the scene parameters and of noise on the behaviour of automatic focusing algorithms* 是不同论文。前者已读出版页摘要，后者仅获得检索摘要和登记书目；没有逐式核验原文 F4／F5 的全部定义。因此本地 `vollath_f4` 只称**滞后 1、2 的相关均值差变体**，不假设它和任意原文求和式在不同窗口上等价。[cite:4](#af-source-4)[cite:5](#af-source-5)

### 4.2 比较研究与综述不能反向认领指标

Groen 等 1985 摘要明确说比较文献中已有的 **11 个函数**；它可以支持条件化选型和相关指标已被研究，不能作为梯度、平方 Laplacian、所有归一化统计量的唯一首创。摘要中的 normalized image standard deviation 也不能不看公式就替换成当前 `Var(I)/mean(I)`。[cite:3](#af-source-3)

Pertuz 等论文的期刊首页明确 **2012-11-16 online**，正式卷期为 **Pattern Recognition 46(5)，1415–1432，2013**；镜像前置 ResearchGate 封面的“November 2012”不是独立论文版本。第 2.1 节按梯度、Laplacian、小波、统计、DCT、其他六类组织，并明确一些指标由 AF 修改用于 SFF。因而“Pertuz 2013 提出了 Brenner／Tenengrad／Vollath”等表述不成立；综述适合做导航，具体起源须回原论文。[cite:7](#af-source-7)

本章没有给 SMD、当前归一化方差、当前 Laplacian energy 或 DCT top-k 变体强行填写原始提出年；“尚未核验首创”是实际结论，不是待补的猜测日期。DCT 基础变换的论文年代也不能直接充当当前评分的年代。

### 4.3 SFF 峰拟合与学习对焦的版本辨析

*Shape from Focus* 有 **Shree K. Nayar 的 1989 年 11 月 CMU-RI-TR-89-27 报告**，也有 **Nayar 与 Yasuo Nakagawa 的 1994 年 TPAMI 16(8)，824–831 期刊文**。报告封面与期刊作者表不同，不能拼成“二作者 1989 期刊版”。1994 文的 SML 与高斯插值面向局部深度；当前 `gaussian_fit` 是已有全局分数数组上的等距简式，不代表 SML、SFF 或完整论文已实现。[cite:6](#af-source-6)

2019 单帧显微对焦原文作者是 **Henry Pinkard、Zachary Phillips、Arman Babakhani、Daniel A. Fletcher、Laura Waller**，不是旧 [DL 说明](../dl/README.md)中未具完整书目的“Pertuz et al. 2019”。正式文为 Optica 6(6)，794–797，DOI `10.1364/OPTICA.6.000794`；Berkeley 另有 **2019-12-01 的 UCB/EECS-2019-161** 档案，内稿日期为 2019-05-13。后者不等于正式期刊首次出版日期，也不是仓库 FocusNet 的原论文。[cite:8](#af-source-8)

## 5. 传统、ML、DL 并行及实现状态

### 5.1 七种评分的实际定义

以下全部来自 [focus_measures.py](../traditional/autofocus/focus_measures.py)函数体，表示**已有实现**而非全输入域健壮性已验收。共同预处理是 BGR 先转灰度、再 float64；不是 RGB 协议，也没有统一检查空图、NaN／Inf 和所有参数。

| 卡片／函数 | 当前实际定义 | 容易混淆之处及工程边界 |
|---|---|---|
| [F01 Brenner](algorithm_cards/measurement_focus.md#f01-brenner)／`brenner` | 仅横向跨 2 px 差的平方均值 | 非二阶导、无纵向项；纹理方向盲区和高频零点存在；宽度不足可空均值 |
| [F02 Tenengrad](algorithm_cards/measurement_focus.md#f02-tenengrad)／`tenengrad` | 双向 Sobel 平方和的全图均值 | 无梯度阈值；核尺寸／增益和边界改变数值，不能跨配置搬阈值 |
| [F03 SMD](algorithm_cards/measurement_focus.md#f03-smd)／`smd` | 横／纵邻差绝对值各自均值之和 | 非乘积 SMD、非 SML；不同方向支持数不同；源码抗噪排序不是普遍结论 |
| [F04 variance](algorithm_cards/measurement_focus.md#f04-归一化方差-variance)／`variance` | `Var(I)/mean(I)` | 非除均值平方；不具一般曝光不变性；纯黑图可除零，内容重排不改分数 |
| [F05 Laplacian energy](algorithm_cards/measurement_focus.md#f05-laplacian-energy)／`laplacian_energy` | `mean(Laplacian(I)²)` | 非 Laplacian 方差、非 SML；噪声／振铃会升分，无额外预平滑 |
| [F06 Vollath F4](algorithm_cards/measurement_focus.md#f06-vollath-f4-自相关差)／`vollath_f4` | 横向滞后 1 与 2 的乘积各自均值之差 | 两支持／分母不同，不等价统一求和；可为负，不能直接套对数高斯 |
| [F07 DCT 当前评分](algorithm_cards/measurement_focus.md#f07-dct_sharpness-当前评分)／`dct_sharpness` | 整块 DCT 后取系数绝对值 top-k 和，再跨块均值 | **不排 DC、按幅值不按频率、不是平方能量**；裁掉右／下余边，小图可能无整块 |

`dct_sharpness` 的“高频能量”注释与函数体冲突。亮的常量块也有 DC 幅值贡献；选择大系数不等于选择高频。排 DC、加频率掩膜、改平方能量或块高分位聚合都属于**新评分定义**，不能当作原函数的等价提速，也不能用当前 RTImg 失败推断所有 DCT 评分失效。[F08](algorithm_cards/measurement_focus.md#f08-dct-ii-变换)讲的是变换本身，不新增一个评分器。

### 5.2 搜索、数据与网络状态

| 卡片／源码 | 已有内容 | 未实现／不应声称的内容 |
|---|---|---|
| [F09 单峰性](algorithm_cards/measurement_focus.md#f09-单峰性比例)／[focus_measures.py](../traditional/autofocus/focus_measures.py)、[metrics.py](../tests/metrics.py) | 首个最大值两侧严格增减的计数比例 | 不是连续单峰证明；端点单调曲线也可得 1，平台不算通过 |
| [F10 高斯峰](algorithm_cards/measurement_focus.md#f10-三点高斯峰位)／[focus_search.py](../traditional/autofocus/focus_search.py) | 全数组取峰，再取相邻三点 log 拟合等距简式 | 平均左右间距不能修正一般非等距输入；“三点”不包含发现峰所需采集成本 |
| [F11 爬山](algorithm_cards/measurement_focus.md#f11-粗扫加缩步爬山)／[focus_search.py](../traditional/autofocus/focus_search.py) | 全范围粗扫＋固定轮数缩步，返回请求位置与调用次数 | 无限位、无噪声决策、无帧去重；细探可越出 lo/hi，不能直接接电机 |
| [F12 黄金分割](algorithm_cards/measurement_focus.md#f12-黄金分割)／[focus_search.py](../traditional/autofocus/focus_search.py) | 比较后复用一点，区间宽不大于 tol 时返中点 | 依赖单峰和正确比较；无最大次数／非法 tol 完整保护，查表平台不满足连续严格单峰 |
| [F13 全扫描](algorithm_cards/measurement_focus.md#f13-全扫描与数据源)／[source.py](../traditional/autofocus/source.py)、[real_dataset.py](../traditional/autofocus/real_dataset.py) | 文件解析、按组排序、ROI、图像 LRU、逐帧评分与统计 | 无独立 `full_scan` 函数；`load_at` 最近邻，不是移动设备；缓存图像不缓存评分 |
| ML／[路线说明](../ml/README.md) | 仅匹配 HOG＋SVM 示例，与对焦任务不同 | 手工清晰度特征＋RF／SVR 等对焦回归**未实现**；通用分类器卡不是对焦模型 |
| [F14 FocusNet](algorithm_cards/measurement_focus.md#f14-focusnet-有符号离焦回归)／[autofocus_cnn.py](../dl/autofocus_cnn.py) | 卷积、全局池化、标量头；Adam＋MSE 的真实参数更新循环 | 缺完整 Dataset、验证、随机状态、设备管理、权重持久化及闭环；不是仅有空接口，也不是训练完成 |

### 5.3 路线组合与方向可辨识性

传统评分＋搜索不需要学习权重，适合作基线和最终复核；ML 可将手工梯度／频带／曲线特征输入浅层回归器或方向分类器，但需要按场景划分数据和可靠标签；DL 可学习更复杂成像先验，付出训练、权重、域偏移与推理环境成本。三者可组合为“回归给初值／范围 → 少量受控采样 → 传统评分复核”，但这是**拟议方案**，不是现有端到端能力。

普通单帧并不总能提供有符号离焦。若指定成像模型中正负离焦的 PSF 相同，同一目标在两侧产生相同观测，则不存在仅凭该图像唯一恢复方向的函数；MSE 回归会趋向条件均值。增加网络深度不增加观测信息。需要单侧范围先验、已知受控位移的第二帧、当前位置信息，或改变照明／光学编码等条件；这些改变均需重新验证。[F14](algorithm_cards/measurement_focus.md#f14-focusnet-有符号离焦回归)给出条件风险解释。

Pinkard 等 2019 方法使用**离轴相干照明与傅里叶域全连接网络 FCFNN**，不是普通灰度输入三层卷积的本地 FocusNet；训练标签由另一类照明焦栈生成。原文可读段落也明确展示跨样本类型可能失效并需补训练数据。因此不能把该文单帧预测结果、旧注释的“1–2 帧收敛”或网络输出标量当作本地闭环证据。[cite:8](#af-source-8)

## 6. 工程实践线：选型、排障与验收

### 6.1 先选可观测目标，再压缩采样

| 场景／约束 | 建议起点 | 必要条件与权衡 | 仓库可做／缺口 |
|---|---|---|---|
| 新场景、峰形未知 | 固定 ROI 全扫描＋多个评分并看曲线 | 成本较高但可暴露多峰、端点、方向盲区；只保证采样点最优 | 已有离线曲线评估；无实机全扫闭环 |
| 内部单峰、邻域近似高斯且等距 | 离散峰＋局部三点拟合 | 正分、负曲率、峰邻域可信；更少局部计算不省去先前采图 | 现函数仅等距简式；非等距需另实现真实坐标拟合 |
| 峰区已知且评分稳定 | 粗扫缩步／黄金分割对照 | 前者可能越界／漏窄峰，后者依赖单峰；运动回差可破坏回调一致性 | 算法函数已有，安全门和闭环未建 |
| 大平坦背景、少量细纹理 | 任务 ROI，比较双向梯度与明确频带评分 | 小 ROI 快但统计波动大；不能裁掉测量目标，不能只按全图均值 | ROI 已有；真正 DCT 高频能量需新增定义和回归 |
| 噪声大、曝光漂移 | 先锁采集，再比较梯度／相关／统计曲线 | 自相关抗噪需假设；平滑可同时损失焦差；无万能指标 | 七函数可作基线，未统一噪声置信／曝光校正 |
| 多深度／倾斜样品 | 任务优先 ROI 或局部分区策略 | 全图单标量可选错深度；SFF 局部深度是另一任务 | 无深度图／场景最优焦平面融合实现 |
| 扫描预算极紧且有稳定训练域 | ML／DL 初值＋采样复核 | 必须可辨方向、训练和实际成像同域、越界可拒绝 | 对焦 ML 未实现；DL 只有骨架，不能报告加速收益 |

### 6.2 按症状排障

| 症状 | 先看证据 | 动作与避免的误判 |
|---|---|---|
| 所有分数非有限／黑图异常 | 图像尺寸、均值、NaN／Inf、块数、通道顺序 | 先拒绝无效输入；variance 黑图除零、SMD/Brenner 小图空支持、DCT 无整块不是搜索失败 |
| 峰在端点且单峰性为 1 | 原始位置轴、曲线、范围与任务目标 | 扩大合法范围或声明未包峰；高单峰比例不证明正确对焦 |
| 评分高而肉眼／下游仍失焦 | 曝光、锐化、噪声、反光及纹理 ROI | 比较同采集条件；分数绝对值不跨内容比较，不能靠调搜索容差修评分 |
| DCT 平坦背景主导／亮度越高越清晰 | DC、频率坐标、系数 top-k、完整块支持 | 先承认当前定义非高频能量；改定义需新基线，不能只归咎平均稀释 |
| Vollath 出负分，高斯拟合异常 | 两个相关项、支持与纹理周期 | 可负并不必然为 bug；不要靠把负分夹到极小正数假装高斯模型成立 |
| RTImg 连续拟合偏、图与返回值不同 | 非等距 ZZZZ、峰相邻间隔、demo 的 `polyfit` | 原函数平均间距不是一般解；真实坐标二次拟合或回退离散帧，绘图平滑不证明返回值正确 |
| 黄金分割很快“收敛”但落错帧 | 请求位置与实际帧 ID、平台和比较分支 | 最近邻查表去重后用离散搜索；tol 变小不增加可用帧信息 |
| 爬山请求出界 | 每次 probe、粗扫端点最优、限位 | 加独立硬件限位及失败协议后才接实机；本地函数无此保护，本次不修 |
| 耗时高／缓存仍占很多内存 | JPEG 解码、ROI 时机、评分重算、`imgs` 列表 | 分项剖析；全尺寸解码后裁 ROI 不减少解码本身，整组列表可超过 LRU 容量 |
| CNN 训练下降但方向错／泛化失败 | 同图正负标签、B 与 B1 广播、场景泄漏、照明域 | 先查可辨识性和数据契约，后调模型；权重或训练 loss 不替代闭环验证 |
| 同一位置往返分数不同 | 到位反馈、回差、温漂、稳定时间和曝光时间戳 | 统一运动方向／稳定条件再比较；评分回调必须代表同一物理测量过程 |

与几何测量联合使用时，还要检查离焦变化带来的边界偏差、倍率和 ROI 映射，参阅[测量工程线](measurement_evolution.md)。不能为提升清晰度分数任意增强图像而不复核尺寸误差。

### 6.3 RTImg 代理真值与验收协议

`FolderSource` 从 `NNN-DDDD-ZZZZ` 解析目标编号、目标距离与拍摄时镜头对焦距离；按组和 ZZZZ 排序。当前 `gt_frame` 选择**对焦距离最接近目标距离的可用帧**。这是文件标签构造的代理真值，不是独立测量的光学最佳焦点；不能把 ZZZZ 的 mm 当作 Z 轴电机位移，更不能将对焦距离误差当成工件尺寸误差。当前不依赖假想的 `-0000` 自动对焦参考帧。[source.py](../traditional/autofocus/source.py)

`load_at(z)` 返回最近已有帧，不插值、不移动电机；多个请求可能命中同帧。`real_dataset` 运行时按实际发现的组数统计，文件头固定场景数不是当前资源完整性的证据。结果中的命中率应明确是“命中代理真值帧”，与真正焦平面误差分开。[real_dataset.py](../traditional/autofocus/real_dataset.py)

- **数据与划分**：保存原图、实际位置轴、帧 ID、场景／工件／设备、曝光／增益、光路、ROI 与真值定义；训练／验证／验收按完整场景隔离，不随机打散相邻焦栈帧造成泄漏。
- **评分与峰**：固定定义和支持；报告无效比例、内部峰／端点、多峰／平台、重复噪声及峰邻域分差。单峰性比例只是诊断，不能代替峰位正确率。
- **精度与闭环**：分报代理帧命中、实际焦差、方向正确率、重采重复性和闭环失败率；离散采样间距不是无条件半步长误差界，连续搜索 tol 也不是物理精度。
- **成本**：同时记评分调用数、独立帧数、实际运动次数、解码与评分成本、移动／稳定／曝光／传输成本及尾延迟；局部三点使用量不等于端到端三次曝光。
- **回退与安全**：非法范围、低纹理、全无效、超时、域外、多峰、端点和设备未到位必须有明确拒绝／扩搜／回到已验证位置策略；学习输出先限幅并复核，不直接当运动命令。

现有 [test_smoke.py](../tests/test_smoke.py) 的评分测试只对一种合成焦栈断言峰帧偏差，并计算单峰性用于展示；搜索测试基于最近邻帧回调，不是电机闭环。历史合成和 RTImg 结果见[实验记录](experiments.md)、[仓库真实数据说明](../README.md#真实数据集rtimg)及[卡片验证边界](algorithm_cards/measurement_focus.md#本次验证未覆盖项与来源状态)。本章未重跑 RTImg 全量、训练网络或采集硬件数据，不重复引用历史速度／精度为本次结果。当前文档验证仅覆盖结构、链接、引用、源码对应及两文件变更范围。

## 7. 来源与逐源访问记录

访问层级按“全文／全文片段／摘要／书目／受限”记录实际取得内容；搜索摘录不称全文。**[CONFIRMED]** 仅表示指定书目或版本关系已交叉支持，不表示性能有三个独立复现；方法机制主要来自一篇论文时标 **[SINGLE-SOURCE]**。来源缺口标 **[INSUFFICIENT DATA]**；同文镜像、索引、登记元数据不重复计算为独立实验。

<a id="af-source-1"></a>

**[cite:1] Jay Martin Tenenbaum，1970。** *Accommodation in Computer Vision*，Stanford 学位论文及 AIM-134／CS 182 报告。[原扫描件镜像](http://bitsavers.trailing-edge.com/pdf/stanford/Stanford_CS_TR_Collection_2025-12-12/PDF/1970/CS-TR-70-182.pdf)：**全文片段已读**，实际取得封面、学位内页、摘要、目录与图表目录，核对论文 September 1970、报告 October 1970、Focus Ranging 章节和梯度阈值图题；未完整读到第 VI 章公式。[Stanford 1971 项目报告目录](http://i.stanford.edu/pub/cstr/reports/cs/tr/71/209/CS-TR-71-209.pdf)：**检索书目**，列 AIM-134／September 1970；[ACM 馆藏](https://dl.acm.org/doi/10.5555/905569)：**检索书目**；[Stanford 原 PDF 地址](https://i.stanford.edu/pub/cstr/reports/cs/tr/70/182/CS-TR-70-182.pdf)：**受限，抓取为空**。1970 来源可核，不从扫描存档路径日期推方法年；现代 Tenengrad 与原公式等价性 [INSUFFICIENT DATA]。

<a id="af-source-2"></a>

**[cite:2] J. F. Brenner、B. S. Dew、J. B. Horton、T. King、P. W. Neurath、W. D. Selles，1976。** *An automated microscope for cytologic research a preliminary evaluation*，Journal of Histochemistry & Cytochemistry 24(1)，100–111，DOI `10.1177/24.1.1254907`。[PubMed](https://pubmed.ncbi.nlm.nih.gov/1254907/)：**摘要＋书目已读**，1976 年 1 月与系统构成；[Ovid 期刊记录](https://www.ovid.com/journals/jhcy/abstract/10.1177/24.1.1254907~an-automated-microscope-for-cytologic-research-a-preliminary?redirectionsource=fulltextview)：**检索书目**，同年卷页。[Evaluation of autofocus functions in molecular cytogenetic analysis 的 CORE 镜像](https://core.ac.uk/download/pdf/30046041.pdf)：**仅检索正文片段**，将跨二像素差归为 Brenner et al., 1971；这是**二手引文年份冲突 [DISPUTED]**，未核完该文参考表，**尚未确认存在独立的 1971 年早期文献**，不能将该年份当作已确认的原文或更早首创证据。1976 条目 [CONFIRMED] 仅指书目可查；1976 全文未取得，不声称该文首创评分，首创归属及年份仍为 [INSUFFICIENT DATA]。

<a id="af-source-3"></a>

**[cite:3] Frans C. A. Groen、Ian T. Young、Guido Ligthart，1985。** *A comparison of different focus functions for use in autofocus algorithms*，Cytometry 6(2)，81–91，DOI `10.1002/cyto.990060202`。[Wiley 出版页](https://onlinelibrary.wiley.com/doi/10.1002/cyto.990060202)：**摘要＋书目／参考文献已读**，March 1985，摘要明确比较已有 11 个函数；页面标 Free Access，但本次返回未含实验正文，不能写全文已读。只作为条件比较，不采用其摘要排名为仓库结论。[SINGLE-SOURCE]

<a id="af-source-4"></a>

**[cite:4] D. Vollath，1987。** *Automatic focusing by correlative methods*，Journal of Microscopy 147(3)，279–288，DOI `10.1111/j.1365-2818.1987.tb02839.x`。[Wiley 原文页面](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2818.1987.tb02839.x)：**摘要＋书目／参考文献已读**，September 1987；支持自相关准则和噪声／对比度讨论，未读 PDF 正文。F4／F5 原式和本地均值变体的严格映射尚未核验。[SINGLE-SOURCE]

<a id="af-source-5"></a>

**[cite:5] D. Vollath，1988。** *The influence of the scene parameters and of noise on the behaviour of automatic focusing algorithms*，Journal of Microscopy 151(2)，133–146，DOI `10.1111/j.1365-2818.1988.tb04620.x`。[Wiley 页面](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2818.1988.tb04620.x)：**检索摘要可见，直接访问受限于安全验证**；[Crossref 登记](https://api.crossref.org/works/10.1111/j.1365-2818.1988.tb04620.x)：**书目 JSON 已读**，印刷 1988 年 8 月，在线登记 2011-08-02；后者不改写提出年。未取得全文；不把摘要的噪声措辞当任意条件保证。[SINGLE-SOURCE]

<a id="af-source-6"></a>

**[cite:6] Shape from Focus 的两个版本。** [CMU-RI-TR-89-27 原报告](https://ri.cmu.edu/pub_files/pub3/nayar_shree_k_1989_1/nayar_shree_k_1989_1.pdf)：**书目封面已读**，作者 Shree K. Nayar，November 1989；未取得报告正文。Shree K. Nayar、Yasuo Nakagawa，*Shape from Focus*，IEEE TPAMI 16(8)，824–831，1994，DOI `10.1109/34.308479`：[Columbia 作者原文 PDF](https://www1.cs.columbia.edu/CAVE/publications/pdfs/Nayar_PAMI94.pdf)，**全文片段已读**，期刊页首、摘要及前四节部分，明确 SML、局部高斯插值、固定光学系统与纹理前提；[IEEE Computer Society 后续引文](https://www.computer.org/csdl/journal/tp/2001/01/i0085/13rRUwhpBOU)，**检索书目**交叉年卷页。1994 文收稿 1992／修订 1993 不是出版年。来源 7 的正文另有高斯三点说明，不据此将本地等距函数称为全套复现。

<a id="af-source-7"></a>

**[cite:7] Said Pertuz、Domenec Puig、Miguel Angel Garcia，2012 在线／2013 卷期。** *Analysis of focus measure operators for shape-from-focus*，Pattern Recognition 46(5)，1415–1432，DOI `10.1016/j.patcog.2012.11.011`。[论文 PDF 镜像](https://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf)：**全文片段已读**，前置封面、正式首页、第 1–3.1 节部分和方法表；首页明确 Available online 16 November 2012，正式卷页 2013，正文说明六类及 AF→SFF 修改；附录未完整取得，不宣称全部公式已读。[Crossref 登记](https://api.crossref.org/works/10.1016/j.patcog.2012.11.011)：**书目 JSON 已读**，印刷 2013 年 5 月，在线字段为空，在线日期取论文首页而非猜 DOI。[独立 IEEE 论文引文](https://xplorestaging.ieee.org/document/8237371/references)：**检索书目**交叉年卷页；[作者课题组页面](https://deim.urv.cat/~rivi/pub-analysis-of-focus-measure-operators-for-shape-from-focus/)：**检索摘要可见，直接访问 404**；[出版 DOI](https://doi.org/10.1016/j.patcog.2012.11.011)：**受限，验证码页**。版本 [CONFIRMED]；没有把综述作为被归纳指标首创或把其 SFF 结果直接外推 AF。

<a id="af-source-8"></a>

**[cite:8] Henry Pinkard、Zachary Phillips、Arman Babakhani、Daniel A. Fletcher、Laura Waller，2019。** *Deep learning for single-shot autofocus microscopy*，Optica 6(6)，794–797，DOI `10.1364/OPTICA.6.000794`。[Crossref 登记](https://api.crossref.org/works/10.1364/OPTICA.6.000794)：**书目 JSON 已读**，在线 2019-06-05、印刷 2019-06-20，页字段仅 794；[作者配套 Pycro-Manager 教程](https://pycro-manager.readthedocs.io/en/latest/application_notebooks/Single_shot_autofocus_pycromanager.html)：**教程全文与引文已读**，补充 794–797、离轴照明、成对焦栈和硬件前提；教程不是原论文全文。[Berkeley 档案页](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-161.html)：**摘要＋书目已读**，2019-12-01；[档案 PDF](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/Archive/EECS-2019-161.pdf)：**全文片段已读**，含 2019-05-13 内稿、成像／训练流程、傅里叶网络和跨样本失败讨论，后文截断。[Optica 正文入口](https://opg.optica.org/optica/fulltext.cfm?uri=optica-6-6-794&id=412060)：**受限，验证码**。出版／档案版本分清，不以报告日期替代期刊日期；同文多入口不是独立性能验证。[SINGLE-SOURCE] 方法说明。

## 8. 本地证据与未覆盖项

本章静态核对 [focus_measures.py](../traditional/autofocus/focus_measures.py)、[focus_search.py](../traditional/autofocus/focus_search.py)、[source.py](../traditional/autofocus/source.py)、[real_dataset.py](../traditional/autofocus/real_dataset.py)、[autofocus_cnn.py](../dl/autofocus_cnn.py) 的相关函数体，以及 [F01–F14](algorithm_cards/measurement_focus.md#f01-brenner)、[学习组件](algorithm_cards/learning_components.md)、[研究笔记](research_notes.md)与[测试](../tests/test_smoke.py)。以函数体而非注释中的“最快”“最少次数”“拟合无偏”判断能力；平滑成像模型也不自动保证真实评分恰为高斯或峰估计无偏。

尚未完成的内容包括各评分唯一首创的逐式追溯、原始 Brenner／Vollath 全文、本地 DCT 语义修订、一般非等距拟合、安全搜索、对焦专用 ML、可复现 DL 训练、独立光学真值和相机／电机闭环。本文只记录现状与可检验方案，不修改这些实现，也不把文档完成等同于上述缺口已解决。
