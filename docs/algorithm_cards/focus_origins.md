# 对焦起源独立卡：Tenenbaum 1970、Brenner 1976、Vollath 1987

[学习规范](../algorithm_learning_standard.md) · [演进覆盖矩阵](../evolution_coverage.md#11-对焦成像质量) · [对焦总览](../autofocus_evolution.md) · [现代评分卡](measurement_focus.md#对焦共同数据契约)

## 范围、证据与升级口径

核验日期：**2026-11-05**。本文件包含三个独立历史目标卡，键为 G09、G10、G11；不是把 F02、F01、F06 改名重算。每卡分别展开 L1–L5、五层问答、来源、实现和验证范围。本文只补此文件，不修改矩阵、卡片索引、总数或源码；建立卡片载体不自动消除历史机制缺口。

**原文机制尚未闭合，三项均不可销项。** 现有证据足以讲清现代评分和若干二手公式的差异，但不足以逐式复现三个历史目标。层级状态针对历史目标，统一保守标“部分”；其中已核验的局部事实在正文单列，不继承现代卡的“已核验”。

| 缺口／独立卡 | 新增可确认内容 | 历史机制仍缺什么 | 矩阵升级建议 |
|---|---|---|---|
| [G09](#g09)／11-01 | 1970 封面月份、梯度阈值图题、主动调节任务；现代无阈值 Sobel 与二手公式的条件映射 | VI.3 原评分的范数、阈值比较及支持；VI.6 搜索、范围细化和停止规则 | 不升“已有”；保留缺卡验收项，注明已有未闭合独立卡 |
| [G10](#g10)／11-02 | 1976 原系统正文片段；二手 1971 引文及门控／双向修改；F01 与明确现代求和式的映射 | 1976 评分及调焦正文；1971 完整参考条目和误引归因 | 不升“已有”；原文评分与年份冲突未闭合，不销项 |
| [G11](#g11)／11-04 | 1987 原摘要；Santos 明标 1988 的 F4/F5 求和式；F06 不同支持均值差的精确代数差别 | 1987 原 F4/F5 的定义、式号、支持、归一化及二值化机制 | 不升“已有”；不能用 1988 的二手转述补成 1987 原式 |

访问层级使用“全文／全文片段／摘要／书目／受限”。取得 PDF 链接不等于取得完整内容；本次没有任何一篇目标原文达到完整方法阅读。`[SINGLE-SOURCE]` 表示结论只来自指定文献；同文作者版、CORE、SciSpace 是同一证据，不作独立交叉验证。`[DISPUTED]` 表示版本／引文冲突；`[INSUFFICIENT DATA]` 是实际缺证结论，而不是用常见公式填空的许可。

数组统一记为 `I[y,x]`，尺寸 `(H,W)`，零基坐标；灰度值单位为 ADU 或调用方规定的线性单位，不默认为物理辐亮度。本文数值退化来自已有卡中的解析例及其代数延伸，不是新增样本数据、仿真焦栈或论文实测。分数一般不能跨图像内容、ROI、曝光、核增益直接比较；未提供通用理论焦差界。

<a id="g09"></a>

## G09：Tenenbaum 1970 梯度对焦与主动调节

### 任务、演进与状态

历史对象是 Jay Martin Tenenbaum 的 *Accommodation in Computer Vision*：学位内页 September 1970，AIM-134／CS 182 报告封面 October 1970。不能因 ACM 检索字段出现后续年份，或文件存档路径带 2025，就改写原封面年代。原摘要把相机参数调节置于识别任务内部，第 VI 章为 Focus Ranging；目标不仅是“图像→清晰度分数”，还涉及“任务与范围先验→传感器调节→焦点／距离信息”。[cite:1](#source-1)

前序瓶颈是固定采集不能同时满足不同视觉任务；改变的是任务驱动的相机调节与可用信息，而不只是更换一个卷积核。新增代价包括控制、采样、曝光与量化协调；内容、噪声和距离可辨识性仍限制结果。这里只能确认原摘要与目录所述主线，不能把现代全扫、黄金分割或三点高斯写成原算法。[cite:1](#source-1)

| 字段 | 本卡状态与边界 |
|---|---|
| 输入／输出 | 历史系统：场景、任务及可调相机→任务所需信息，焦距测距分支细节未取；本地 F02：灰度／BGR 数组→标量 |
| 前提 | 固定比较任务、可观测纹理及受控采集；原报告更具体的范围约束尚未逐页核对 |
| L1／L2／L3／L4／L5 | 部分／部分／部分／部分／部分；现代局部公式核验不升级历史方法 |
| 实现 | `tenengrad` 已有；无原报告主动相机系统复现 |
| 验证 | 原扫描件封面、摘要、目录、图题；二手正文与本地算子解析检查；无原系统／现场验证 |

### L1 步骤：区分已知流程与缺失规则

原摘要可支持“根据任务信息需要调节相机；任务失败且能归因为具体图像缺陷时，再细化有关参数”。目录定位 VI.3 Criteria of Focus（印刷 p.311 起）、VI.6 Focus Ranging Program（p.336 起），图题定位梯度幅值与累积前阈值（pp.315–316）以及程序流程图（p.347）。这些是**定位线索，不是相应页正文已读**。[cite:1](#source-1)

目前不能填写原评分算子、逐点聚合、阈值数值及相等处理，也不能填写原搜索步长更新、测距不确定度细化和停止分支。原方法伪代码因此不伪造；可执行学习对象限定为本地 F02：预处理→两个 Sobel 响应→平方和→全图均值，单次遍历完成即返回，没有搜索停止条件。

Santos 1997 §2.1.1(4)，印刷 p.265，给出双向 Sobel 平方和的求和式，并明确其比较采用不设阈值的版本，转引 Krotkov 1987；它不是 Tenenbaum 1970 的公式原页。[cite:4](#source-4)

### L2 机制：阈值、范数与任务不是同一层

有符号梯度直接相加会抵消，幅值或平方能量避免这一问题。阈值可减少弱响应参与，但也可能删除弱纹理；核、阈值和聚合范数共同决定曲线，不能仅凭“threshold magnitude”图题判定原式是累积幅值还是累积平方。[cite:1](#source-1)[cite:4](#source-4)

F02 双向 Sobel 比单方向差分减少方向盲区，平方会放大强边缘和噪声；它既不选择任务目标，也不驱动相机。删除阈值不总是等价简化，自动曝光或逐帧自适应阈值还会改变跨焦位的可比性。Osibote 2010 也明说使用包含所有像素位置的 Tenengrad 变体；该文引用 Santos，因此不是独立的起源证明。[cite:7](#source-7)

### L3 工程权衡：先冻结成像契约

先固定 BGR／灰度约定、ROI、曝光／增益、锐化、Sobel 核和边界，再看重复采集的波动与峰邻域分差，最后选择焦位采样和拒绝规则。默认 `ksize=3` 来自本地源码，不是 1970 参数；不能为原阈值编一个通用初值。

若另建现代门控版：梯度幅值阈值 \(\tau\) 与梯度同单位，平方响应阈值 \(t\) 与能量同单位，在相同非负条件及比较符号下才有 \(t=\tau^2\)。阈值应由同采集配置下的弱纹理与噪声对照确定；这属于工程方案，不声称原文如此标定。应记录阈下比例、饱和、有效像素数、焦点是否在扫描端点；全零或平台不能当作成功对焦。

改变核尺寸可能同时改变滤波频带与增益；增益变化不能靠直接沿用分数阈值解决。粗采样可漏窄峰，小 ROI 可节约评分成本但丢失目标；运动稳定等待和曝光时间不能从算子耗时推导。

### L4 数学：现代式已知，1970 原式不填空

令 \(g_x=K_x*I\)、\(g_y=K_y*I\)、\(q=g_x^2+g_y^2\)。在同核、同边界、同支持 \(\Omega\) 下，Santos 的无阈值求和与本地均值满足：

\[
T_{\mathrm{Santos}}=\sum_{p\in\Omega}q(p),\qquad
T_{\mathrm{repo}}=\frac{1}{HW}\sum_{p\in\Omega}q(p),\quad |\Omega|=HW.
\]

因此仅在固定尺寸及上述全部约定相同的焦栈内，二者相差固定正比例并保留排序；文献未明的边界不能默认为 OpenCV 默认边界。[cite:4](#source-4)[cite:9](#source-9)

为解释现代变体，可**另行定义** \(T_t=\sum q\,\mathbf1[q>t]\)；这是本文比较用的门控式，不标作 Tenenbaum 原式。它与无门控式的差恰是被剔除项之和。若原目标累积的是 \(\sqrt q\)，则不能用乘一个归一化常数变成平方能量；原目标究竟是哪一种仍缺证。

无饱和、同边界且零和导数核消去常量时，\(I'=aI+b\) 给出 \(T'_{\mathrm{repo}}=a^2T_{\mathrm{repo}}\)。对固定支持的周期模型，Parseval 将能量写为带非负权 \(|\widehat K_x|^2+|\widehat K_y|^2\) 的频谱和；高斯模糊衰减可推出随模糊尺度不增，但不证明真实焦位曲线单峰，也不覆盖变化的 ROI、裁边、噪声或原主动系统。

默认 3×3 Sobel 的两个核平方和分别为 12；独立零均值白噪声在内部给出能量期望贡献 \(24\sigma_n^2\)。反射边界复用像素时不能把此值当整图精确噪声底。固定核计算量为 \(O(HW)\)，不是搜索或硬件总耗时的界。

### L5 实现、具体映射与数值退化

| 原目标／二手定义 | 本地对应 | 等价条件或缺口 |
|---|---|---|
| 原报告梯度幅值、阈值图题 | [F02](measurement_focus.md#f02-tenengrad)／[tenengrad](../../traditional/autofocus/focus_measures.py#L37-L42) | 本地无阈值；原核、范数及门控未核实，不认定原式实现 |
| Santos 无阈值双向 Sobel 求和 | `mean(gx*gx+gy*gy)` | 同支持同核时除以 HW；仅固定配置下保序 |
| 原任务驱动调节／Focus Ranging | 本地评分函数无对应闭环 | 不能把 [focus_search.py](../../traditional/autofocus/focus_search.py) 的存在当原控制流程已承接 |

以下为 **F02 数学等价伪代码＋拟议输入防护**，不是原方法复现；拒绝分支尚未加入仓库函数。

```text
I = prep_BGR_then_float64(image)
if I is not nonempty_2d or any_nonfinite(I): return INVALID
gx = Sobel(I, dx=1, dy=0, ksize=3, border=REFLECT_101)
gy = Sobel(I, dx=0, dy=1, ksize=3, border=REFLECT_101)
q = gx*gx + gy*gy
if any_nonfinite(q): return INVALID
return sum_float64(q)/(H*W)
```

F02 既有解析例：3×3 每行 `(0,1,2)`，默认反射边界下每行 gx 为 `(0,8,0)`、gy 为零；全图均值 \(64/3\)，只取中心有效支持则为 64。两者并非同一分数。常量图为零但不提供焦位信息；提高曝光增益可平方增分，不能据此认定变清晰。

实现应先转灰度再转 float64，避免 uint8 差分下溢和不支持的彩色 float64 转换；本地 `_prep` 对 float64 彩图先降到 float32，会引入该精度环节。共享梯度或分块时须保留 halo，内部块边界不能各自反射。降采样、换核或加门控属于定义修改，须独立回归。

本次执行既有边界例及偏置／增益关系断言通过；没有运行 Tenenbaum 的评分、焦点测距程序或硬件闭环。

### 五层问答与证据入口

| 层 | 问题 | 答案要点与成立条件 | 易错答案／反例 | 证据 |
|---|---|---|---|---|
| L1 | 给图像能否输出原报告的距离？ | F02 只给标量；原 VI.6 的程序、相机及映射未取得 | 把最大分数直接解释为毫米距离 | [1](#source-1)、[9](#source-9) |
| L2 | 为何不能把阈值图题直接写成平方门控式？ | 图题不确定累积范数、比较符和支持；需原正文 | 幅值与平方和只差常数 | [1](#source-1)、本卡 L4 |
| L3 | 换核后分数上涨，先调哪项？ | 先查核增益、边界、曝光及 ROI，再标定阈值 | 直接缩小搜索步长能修正评分漂移 | [9](#source-9)、本卡 L3 |
| L4 | 求和换均值何时保序？ | 相同算子、支持，且 HW 固定正值 | 所有裁剪和自适应 ROI 下均保序 | [4](#source-4)、本卡 L4 |
| L5 | 为什么解析例得到 64/3 而不是 64？ | 全图反射支持与仅中心有效支持不同 | 认定卷积实现有错 | [9](#source-9)、本卡 L5 |

**未闭合验收：**取得 VI.3 pp.311–317、关联算子定义及 VI.6 pp.336–351 的可核原页；逐项记录核、范数、阈值、支持、范围／搜索／停止分支，再映射 F02。只有目录与二手阈值说法不足以升级 G09。

<a id="g10"></a>

## G10：Brenner 等 1976 显微系统与跨距差评分

### 任务、演进与状态

原文为 J. F. Brenner、B. S. Dew、J. B. Horton、T. King、P. W. Neurath、W. D. Selles，*An automated microscope for cytologic research a preliminary evaluation*，Journal of Histochemistry & Cytochemistry 24(1)，100–111，January 1976，DOI `10.1177/24.1.1254907`。PubMed 与原 PDF 首页及系统正文片段支持作者和系统身份；原 PDF 返回在照明／滤镜段落后截断，**未取得评分和调焦方法部分**。[cite:2](#source-2)[cite:3](#source-3)

前序瓶颈是细胞研究的自动寻找、定位、采集与分析；SCANIT 将显微镜、TV 相机、自动寻细胞与计算机伺服台结合，区分全自动和交互模式。代价包括采集、标定、控制与系统同步；现代三行评分不能代表整套 SCANIT。原文还回指 Dew 等先前的设计报告，故本卡不宣布 1976 是整套系统或评分的唯一首创年。[cite:3](#source-3)

| 字段 | 本卡状态与边界 |
|---|---|
| 输入／输出 | 原系统：载玻片、任务及操作模式→寻细胞、居中、对焦、记录／分析；F01：数组→横向二像素差平方均值 |
| 前提 | 目标处于可采集区域、采集标定稳定；原评分的前提与阈值尚未核实 |
| L1／L2／L3／L4／L5 | 部分／部分／部分／部分／部分 |
| 实现 | `brenner` 已实现现代单方向无阈值均值版；SCANIT 和原评分未复现 |
| 验证 | 原系统片段与书目、二手方法正文、本地评分解析断言；无原系统实验 |

### L1 步骤：系统流程与评分分开

已读系统正文的全自动示例是：在指定区域寻找白细胞→居中并对焦→分类计数→继续寻找，直至区域覆盖完毕或达到指定计数。交互示例则由操作者找细胞和分类，按钮触发自动对焦及多滤镜记录。这些是**原文工作模式示例，不是已取得的内部对焦算法**。[cite:3](#source-3)

本地 F01 步骤是：灰度浮点预处理→逐行计算相隔 2 px 的差→平方→有效支持均值→返回。它不找细胞、不移动平台、不搜索峰；遍历完即终止。原调焦如何选择采样位置、如何判定停止仍为 `[INSUFFICIENT DATA]`。

### L2 机制：跨二像素不是二阶导

跨距差 \(I[y,x+2]-I[y,x]\) 仍是一阶差，不包含二阶差分的中心系数。常量亮度偏置可被抵消，平方避免符号抵消；仅横向取差对只随 y 变化的目标失明，纹理“高频”也不保证得分非零。

二手版本不能混合：Santos 1997 §2.1.1(3) 在平方项用 `j+2`，紧接的门控却写 `|g(i,j+1)-g(i,j)| ≥ v`，又说明前三种函数在实验中改为横纵两方向取最大。本文保留这个字面差别，不擅自把 `j+1` 修成 `j+2`。Sun 等作者稿的 F-3 则对**二距差的平方**施阈值；Mir 等 2014 §2.1 明列无阈值横向和纵向两个求和版本。[cite:4](#source-4)[cite:6](#source-6)[cite:8](#source-8)

### L3 工程权衡：年份误引与参数协议

**1971／1976 必须分两层判断。** Santos 方法正文确实写 `Brenner et al., 1971`；1976 原论文的书目身份可确认，但本次各镜像返回都未覆盖 Santos 完整参考表，无法逐项证明该“1971”条目就是与 1976 相同作者、题名、卷页的年份误写。故称“二手 1971 误引嫌疑／引文冲突”，不把疑似误引升级为已证实笔误，也不据此发明独立的 1971 原论文。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

参数方面，2 px 是采样网格跨度，不是固定物理距离。F01 至少要求 H≥1、W≥3。先看纹理方向、图像缩放／采样、曝光和噪声，再选择 ROI 和焦栈范围。若增加门控，须同时固定“比较一距还是二距”“幅值还是平方”“大于还是大于等于”“双向取最大还是求和”，否则无法复现比较。

幅值门槛单位为 ADU，平方门槛单位为 ADU²；没有可靠的原始门槛数值可搬用。应按重复采集波动、弱细胞纹理保留率和峰邻域稳定性确定现代参数，而不是从论文年代推一个默认阈值。平方增益关系说明曝光变化能混淆焦点变化。

### L4 数学：明确版本后才谈等价

令 \(\Omega_2=\{0\le y<H,0\le x<W-2\}\)，\(N_2=H(W-2)\)，\(d_2=I[y,x+2]-I[y,x]\)。本地式和 Mir 横向无门控求和式满足：

\[
B_{\mathrm{repo}}=\frac{1}{N_2}\sum_{\Omega_2}d_2^2,
\qquad B_{\mathrm{Mir}}=N_2B_{\mathrm{repo}}.
\]

这是**现代版本之间**同支持、固定尺寸时的正比例关系，不是已核实的 1976 原式。[cite:8](#source-8)[cite:9](#source-9)

Santos 所显示的字面门控可在上述有效域上解释为 \(\sum d_2^2\mathbf1[|d_1|\ge v]\)，其中 \(d_1=I[y,x+1]-I[y,x]\)；该有效域是本文为可执行性明确化的约定，文献双向最大修改的边界没有在已读段落完全说明。Sun F-3 的门控是 \(d_2^2\ge\theta\)。二者不只差阈值开方，因为比较的差分跨度也不同；只能在门控均不剔除任何非零贡献等特殊情况下回到同一无门控总和。[cite:4](#source-4)[cite:6](#source-6)

在周期或无限网格上，差分的频率权为：

\[
|e^{2i\omega_x}-1|^2=4\sin^2\omega_x.
\]

DC 与 Nyquist 频率均为零；因此“越高频越大”不成立。无量化／饱和时 \(B(aI+b)=a^2B(I)\)。对与信号独立的零均值平稳噪声，二距差的期望能量贡献为 \(2\sigma_n^2-2C_n(2)\)，白噪声时才简化为 \(2\sigma_n^2\)；门控与噪声相关时不能直接套该无门控结论。时间 \(O(HW)\)，流式累计额外空间可为常数，不提供通用焦位误差界。

### L5 实现、具体映射与数值退化

| 来源／定义 | F01 的关系 | 不能宣称什么 |
|---|---|---|
| Brenner 1976 系统正文片段 | [F01](measurement_focus.md#f01-brenner)只可能作为评分组件对照 | 未读到原评分，不能宣布公式逐式一致或系统已实现 |
| Mir 2014 横向跨二距无门控求和 | [brenner](../../traditional/autofocus/focus_measures.py#L30-L34)除以 H(W−2) | 不自动等价纵向、双向最大、门控版本 |
| Santos 1997 的字面门控及双向修改 | 本地既无门控也无纵向项 | 不称该研究完整复现，且其 1971 年归属未闭合 |
| Sun F-3 二距平方门控 | 本地仅与门控全部放行的情况相接 | 不把平方阈值直接填成幅值阈值 |

以下为 **F01 等价算子＋拟议防护**，不是 1976 原调焦伪代码。

```text
I = prep_BGR_then_float64(image)
if I is not finite_2d or H < 1 or W < 3: return INVALID
total = 0.0
for y in range(H):
    for x in range(W-2):
        d = I[y,x+2] - I[y,x]
        total += d*d
if not finite(total): return INVALID
return total/(H*(W-2))
```

既有解析例每行 `(0,1,2,3)` 得均值 4，求和则为每行 8；每行 `(0,1,0,1)` 二距差全零，尽管并非常量仍得零。只随 y 变化的图像也得零。宽度 2 会产生空支持，本地 `mean` 返回 NaN 而不是主动拒绝；不得把 NaN 替为零后宣称对焦成功。

本次执行上述斜坡、交替纹理和宽度 2 断言通过。减法必须在有符号／浮点域执行；分块要包括跨块的二距像素对并保留实际计数，块均值不能不加权平均。改变跨度、加入纵向或取最大都是评分定义修改，不是等价加速。

### 五层问答与证据入口

| 层 | 问题 | 答案要点与成立条件 | 易错答案／反例 | 证据 |
|---|---|---|---|---|
| L1 | 原系统何时终止，F01 何时终止？ | 原自动计数示例以区域／计数为终止；F01 遍历一次；内部调焦停止未取得 | 把计数停止条件当调焦停止条件 | [3](#source-3)、[9](#source-9) |
| L2 | 跨二距为何会漏掉清晰条纹？ | 一阶跨距差在周期 2 的 x 纹理处为零，另有方向盲区 | “二阶导一定强调所有高频” | 本卡 L4、[9](#source-9) |
| L3 | 1971 已能认定误写吗？ | 可证正文使用该年；缺完整参考表对应，不能证实误写来源 | 猜测另有 1971 首创文或宣布已完成纠错 | [2](#source-2)、[4](#source-4) |
| L4 | 二手门控仅需把 v 平方吗？ | 先确认比较的是 d1 还是 d2；跨度不同不能只换单位 | 默改 Santos 的 j+1 为 j+2 | [4](#source-4)、[6](#source-6) |
| L5 | 分块后偏低或小图 NaN 怎么查？ | 查二距跨界对、计数、dtype、W≥3 与有限性 | 统一补零或丢掉 NaN 后继续选峰 | 本卡 L5、[9](#source-9) |

**未闭合验收：**取得 1976 原文对焦／评分段落及相关图式，确认方向、跨度、门控、聚合与搜索；取得 Santos 最终参考表，把 1971 的题名／作者／卷页与 1976 原条目逐字段比对。在此之前 G10 不可销项；即使确认误引，也不能代替原评分机制核验。

<a id="g11"></a>

## G11：Vollath 1987 相关对焦与 F4/F5 版本边界

### 任务、演进与状态

D. Vollath，*Automatic focusing by correlative methods*，Journal of Microscopy 147(3)，279–288，September 1987，DOI `10.1111/j.1365-2818.1987.tb02839.x`。原出版页摘要说明自相关对焦、不同对比度响应、随机遍历计量下的二值图计算，以及简单分割后二值准则可选择亮度层级；PDF／epdf 未取得方法正文。[cite:5](#source-5)

前序瓶颈是差分评分受噪声扰动；改变的是用相关统计刻画清晰度，并讨论降低计算量的二值化途径。代价是相关性、窗口与亮度选择的条件约束；不能把摘要“较少受噪声影响”改成任意噪声下都无偏、最优或曲线单峰。

1988 *The influence of the scene parameters and of noise on the behaviour of automatic focusing algorithms* 是另一研究节点 N23。Santos §2.1.5 的介绍合称 1987／1988，但两个具体公式标题均写 **Vollath, 1988**；本卡只称“1997 二手文献所标 1988 的 F4/F5 式”。不从合称的引言反推 1987 已有完全相同的式号、求和边界和归一化。[cite:4](#source-4)

| 字段 | 本卡状态与边界 |
|---|---|
| 输入／输出 | 图像→相关准则；原二值化参数与聚合细节未取；本地 F06 输出不同支持均值差 |
| 前提 | 灰度、方向、支持及相关统计约定固定；抗噪推导另需独立性与平稳性 |
| L1／L2／L3／L4／L5 | 部分／部分／部分／部分／部分；下述二手式与源码差异可精确核验，但不升级 1987 |
| 实现 | `vollath_f4` 已有现代变体；F5、1987 二值化和原实验未实现／未复现 |
| 验证 | 1987 摘要、Santos 两式正文、本地有限支持代数与数值断言；没有原式复现测试 |

### L1 步骤：两种相关支持分别计算

F06 先将图像转为 float64 灰度，分别计算横向滞后 1 和 2 的像素乘积，对两组**分别求均值**，最后相减。单次评分结束即返回，没有焦位搜索。Santos 二手 F4 则对两组分别求和后相减；F5 只用滞后 1 的求和再减全图像素数乘全图均值平方。[cite:4](#source-4)[cite:9](#source-9)

原摘要的二值随机遍历计量与简单分割是另有机制的路线；未取得具体阈值、概率关系和算法步骤，不能补成“先 Otsu 再调 F06”，也不能假设二值化无损等价于灰度评分。[cite:5](#source-5)

### L2 机制：非零滞后不等于噪声消失

对于信号加独立零均值白噪声，不同像素噪声乘积的期望为零，所以非零滞后相关可避免差分平方中的直接白噪声能量底。但单帧仍有波动，固定图样、条纹、去噪引起的相关噪声也可残留；“均值期望不变”不保证方差小或最佳焦位不变。

F4 的两个支持大小不同，F06 则改变两个项的权重。F5 的全图均值平方是第三种统计，不是将 F4 改一个函数名。常量图在 F06 上为零，不意味着所有文献 F4/F5 的有限支持求和也为零；边界项会保留背景贡献。

### L3 工程权衡：先看两项，再看差

至少 H≥1、W≥3 才能计算本地 F06 两组非空均值。应同时记录两项值、各自计数、原差值与 ROI 边界，不仅记录归一化后曲线。调整顺序为：排查空支持／非有限值→检查周期与方向→检查曝光／边界→估计重复采集波动→设计焦位搜索与拒绝策略。

lag 是像素单位，改变像素间距会改变物理尺度。负分可能是正常结构响应；不得夹成极小正数再送给对数高斯拟合，以此掩盖模型不适用。宽 ROI 的边界比例虽下降，也不能在任意内容下保证和原求和式只差固定尺度；边缘高亮可能使边界修正显著。

相关乘积与差的单位是灰度²，求和另随有效像素数变化。大背景下相减可能损失有效位；用 float64、稳定累计、记录相对差与重复波动，比机械增加 epsilon 有意义。重新中心化、统一支持或改分母均须作为新定义验收。

### L4 数学：逐式区分二手 F4、F5 与本地 F06

为对应 `I[y,x]`，将 Santos 式的递增索引 i 映射到 x，j 映射到 y，即 M→W、N→H。这是明确的轴重标记；若把原 i 固定解释成行，则对应的是本地转置输入，不能把横纵版本默认等价。

设：

\[
A=\sum_{y=0}^{H-1}\sum_{x=0}^{W-2}I[y,x]I[y,x+1],\quad
B=\sum_{y=0}^{H-1}\sum_{x=0}^{W-3}I[y,x]I[y,x+2],
\]

\[
N_1=H(W-1),\quad N_2=H(W-2),\quad
N=HW,\quad\mu=N^{-1}\sum I.
\]

Santos 印刷 p.266 §2.1.5(12)(13) 的两式，经该轴重标记后为：

\[
S_4=A-B,\qquad S_5=A-N\mu^2.
\]

这里 S4/S5 是本文为**二手转述式**使用的符号，不称为“1987 原式 (4)/(5)”。F06 的实际式是：

\[
V=\frac{A}{N_1}-\frac{B}{N_2}.
\]

直接消元得到：

\[
S_4=N_1V+\frac{B}{W-2}
     =N_2V+\frac{A}{W-1}.
\]

额外项随图像变化，所以即便 H、W 固定，也不能用统一乘一个常数恢复 S4 或保证排序。若有意把 lag 1 裁到 lag 2 的共同支持，令 \(A_c\) 为该裁剪后的和、\(E=\sum_yI[y,W-2]I[y,W-1]\)，则 \(A=A_c+E\)、\(S_4=A_c-B+E\)；共同支持的平均差 \((A_c-B)/N_2\) 仍不是本地 V。

S5 也不能写成 F06：\(S_5/N=A/N-\mu^2\)，而“lag 1 均值减全图均值平方”是 \(A/N_1-\mu^2\)，相差 \(A(1/N_1-1/N)\)。对常量 c，精确有 \(S_4=Hc^2\)、\(S_5=-Hc^2\)、\(V=0\)。这些是二手式有限支持的代数结果，不应擅自通过补边把它们修成零。[cite:4](#source-4)

有限支持亮度偏置也不总消去。定义 \(C_k=\sum_{\Omega_k}(I[y,x]+I[y,x+k])\)，则：

\[
V(aI+b)=a^2V(I)+ab\left(\frac{C_1}{N_1}-\frac{C_2}{N_2}\right).
\]

两个均值各自的 \(b^2\) 项抵消，但信号交叉项未必抵消；纯增益 b=0 时才总满足平方尺度关系。平稳期望模型下 \(E[V]=R_S(1)-R_S(2)+R_n(1)-R_n(2)\)，白噪声非零滞后项为零。频率权 \(\cos\omega_x-\cos2\omega_x\) 可负，因此不能套用非负梯度能量的模糊单调性论证。

各式计算时间 \(O(HW)\)；循环累计的额外存储可为常数，本地向量化会分配两个乘积数组。大数相消属于数值误差问题，不是原统计抗噪证明。

### L5 实现、具体映射与数值退化

| 目标／版本 | 源码关系 | 判定 |
|---|---|---|
| Vollath 1987 原准则与二值化 | [F06](measurement_focus.md#f06-vollath-f4-自相关差)仅作现代相关变体对照 | 原正文未读，无法确认原 F4/F5 命名及式号，不认领复现 |
| Santos 标 1988 的 S4=A−B | [vollath_f4](../../traditional/autofocus/focus_measures.py#L67-L72)为 A/N1−B/N2 | 相差两个不等权重及图像依赖修正，非统一归一化 |
| Santos 标 1988 的 S5=A−Nμ² | 本地无 F5 | 不等于 F06，也不等于 A/N1−μ² 的统一缩放 |
| 共同支持、循环补边、去均值相关 | 本地均未采用这些替换 | 都须单列版本，不能以“优化”名义改变定义 |

下列为 **已核本地 V 与二手 S4/S5 的并列核算伪代码＋拟议防护**，不是 1987 原文伪代码。

```text
I = prep_BGR_then_float64(image)
if I is not finite_2d or H < 1 or W < 3: return INVALID
A = 0.0
B = 0.0
for y in range(H):
    for x in range(W-1): A += I[y,x]*I[y,x+1]
    for x in range(W-2): B += I[y,x]*I[y,x+2]
mu = sum_float64(I)/(H*W)
V_repo = A/(H*(W-1)) - B/(H*(W-2))
S4_secondary = A-B
S5_secondary = A-H*W*mu*mu
if any_nonfinite(A,B,mu,V_repo,S4_secondary,S5_secondary): return INVALID
return V_repo,S4_secondary,S5_secondary
```

以下为既有 F06 解析例的代数延伸，已调用实际函数核对 V 并核验映射；不是焦栈测量结果。

| 一行输入 | A／B | 本地 V | 二手 S4 | 二手 S5 | 说明 |
|---|---|---|---|---|---|
| `(1,2,3)` | 8／3 | 1 | 5 | −4 | 三种统计不能混叫同一数值 |
| `(1,0,1)` | 0／1 | −1 | −1 | −4/3 | 负 V 本身不是程序错误 |
| `(2,1,2)` | 4／4 | −2 | 0 | −13/3 | 给上一行加常量 1，V 仍改变；S4 与 V 的排序还可相反 |
| 常量 c，H 行 W 列 | N1c²／N2c² | 0 | Hc² | −Hc² | 不同支持边界项不可忽略 |

W=2 时本地 lag 2 均值为空，实际返回 NaN；有限大浮点输入的乘积也可能溢出。分块必须保留每个 lag 的跨块像素对及独立计数，不能以块均值平均或统一分母替换。用中心化降低背景相消会改变有限支持交叉项，必须保留校正后才可能保持等价，不能默认去均值无损。

### 五层问答与证据入口

| 层 | 问题 | 答案要点与成立条件 | 易错答案／反例 | 证据 |
|---|---|---|---|---|
| L1 | 1987 原 F4/F5 是否已经读到？ | 没有；取得原摘要，明确求和边界来自 Santos 标注的 1988 转述 | 把两个年份合并成同篇原文 | [4](#source-4)、[5](#source-5) |
| L2 | 为什么抗白噪声不等于抗所有噪声？ | 只在独立零均值非零滞后期望为零的模型下；相关噪声残留且单帧有方差 | “噪声被完全消除” | 本卡 L4、[5](#source-5) |
| L3 | 负分是否加 epsilon 再拟合？ | 先查方向、周期、两相关项、支持及模型；拒绝不适用的 log 拟合 | 强制正分就恢复高斯峰假设 | 本卡 L3、L5 |
| L4 | S4 是否等于 N1 乘 V？ | 还差 B/(W−2)，图像依赖；常量例直接反证 | 固定尺寸即可按常数互换 | 本卡 L4、[4](#source-4)、[9](#source-9) |
| L5 | 将两个均值改求和能否作为优化？ | 不能；保留各自计数和跨界像素对；需同时测常量、负值、偏置和小宽度 | 只看一个焦栈峰位相同就认定等价 | 本卡 L5、[9](#source-9) |

**未闭合验收：**取得 1987 pp.279–288 方法原页，记录每个准则名称／式号、轴、求和支持、归一化、均值域、二值化转换及噪声假设，再与 1988 区别核对。当前代数证明闭合的是“已读二手式→本地变体”的差异，不是“1987 原式→本地”的历史承接；G11 不可销项。

## 逐源证据与访问记录

本节每条只支持其实际取得的内容。原文身份、方法转述和数学推导分层记录；二手间一致不能替代原页。没有把检索失败当作文献不存在的证明。

<a id="source-1"></a>

**[cite:1] Tenenbaum，1970，原扫描件／P0 内容。** [Bitsavers 镜像](http://bitsavers.trailing-edge.com/pdf/stanford/Stanford_CS_TR_Collection_2025-12-12/PDF/1970/CS-TR-70-182.pdf)：本次读取到封面、学位内页、摘要、目录与图表目录，报告 October 1970、学位 September 1970；能定位 VI.3、VI.6 和 pp.315–316 图题，但正文公式没有返回。状态：**全文片段，[SINGLE-SOURCE] 原机制线索**。[Stanford 原址](https://i.stanford.edu/pub/cstr/reports/cs/tr/70/182/CS-TR-70-182.pdf)及[另一扫描镜像](https://archive.decromancer.ca/bitsavers.org/pdf/stanford/Stanford_CS_TR_Collection_2025-12-12/PDF/1970/CS-TR-70-182.pdf)均返回空；直接 HTTP 下载镜像为 403。[ACM 馆藏](https://dl.acm.org/doi/10.5555/905569)仅取得检索书目，不从出版年聚合字段反推原式。缺口：核／范数／阈值／支持／程序停止未核。

<a id="source-2"></a>

**[cite:2] Brenner 等，1976，PubMed／书目与摘要。** [PMID 1254907](https://pubmed.ncbi.nlm.nih.gov/1254907/) 已读作者、1976 Jan、24(1):100–111、DOI 与系统摘要；[Ovid 记录](https://www.ovid.com/journals/jhcy/abstract/10.1177/24.1.1254907~an-automated-microscope-for-cytologic-research-a-preliminary?redirectionsource=fulltextview)取得检索书目，同年卷页。与来源 3 交叉确认的是 **1976 条目身份**，不是评分首创或 1971 误引原因；不把索引当独立方法实验。

<a id="source-3"></a>

**[cite:3] Brenner 等，1976，出版 PDF／P0。** [SAGE PDF](https://journals.sagepub.com/doi/pdf/10.1177/24.1.1254907) 本次返回标题、作者、摘要、系统描述、全自动／交互示例、伺服台、扫描数字化、照明滤镜的正文片段；能核验 SCANIT、既有设计报告回指及系统／评分的边界。状态：**全文片段，[SINGLE-SOURCE] 系统细节**。返回在滤镜段后截断，未取得对焦评分；full、epdf、带页定位入口均为空，直接下载超时。没有原评分式号／页码可引用，不声称已完整阅读 100–111 页。

<a id="source-4"></a>

**[cite:4] Santos 等，1997，方法比较／P1。** *Evaluation of autofocus functions in molecular cytogenetic analysis*，Journal of Microscopy 188，264–272。[作者站 PDF](https://www2.die.upm.es/im/papers/Autofocus.pdf) 返回正式首页及 pp.265–266 方法正文：§2.1.1(3) 的 1971 年与 j+1 门控、(4) 的无阈值 Tenengrad、跨页双向最大修改、§2.1.5(12)(13) 标注 Vollath 1988 的 F4/F5。状态：**全文片段，[SINGLE-SOURCE] 二手定义，[DISPUTED] 1971 引文**。[CORE 镜像](https://core.ac.uk/download/pdf/30046041.pdf)与[SciSpace 镜像](https://scispace.com/pdf/evaluation-of-autofocus-functions-in-molecular-cytogenetic-bi13zjgq2l.pdf)支持相同方法段，但均未返回最终参考表；三个入口不算三个独立来源。作者站直接下载因本机证书链校验失败，未禁用 TLS 验证。未据此证实 1971 是哪一条书目的笔误。

<a id="source-5"></a>

**[cite:5] Vollath，1987，原出版页／P0 摘要。** [Wiley 摘要页](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2818.1987.tb02839.x) 已读 September 1987、147(3):279–288、DOI、摘要及参考文献；只支持自相关、对比度响应、二值随机遍历计量和简单分割路线。状态：**摘要＋书目，原公式 [INSUFFICIENT DATA]**。[全文页](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2818.1987.tb02839.x)和[epdf](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-2818.1987.tb02839.x)为安全验证，[pdf](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1365-2818.1987.tb02839.x)为空；原题名及机构库方向检索未提供可用方法原页。不可借来源 4 的 1988 标签补写 1987 原式。

<a id="source-6"></a>

**[cite:6] Sun、Duthaler、Nelson，作者稿／P1 比较。** [Autofocusing Algorithm Selection in Computer Microscopy](https://amnl.mie.utoronto.ca/data/J7.pdf) 本次已读 §III、F-3/F-4/F-12/F-13 等正文片段；F-3 使用二距差平方门控，F-4 为无阈值 Sobel 平方和。状态：**全文片段，版本比较 [SINGLE-SOURCE]**。未读到最终参考表和完整出版页，本文不凭文件名补会议年、卷页或原文归属；只用于暴露门控量及方向的版本差异，不引用其性能排名。

<a id="source-7"></a>

**[cite:7] Osibote、Dendere、Krishnan、Douglas，2010／P1。** [Automated focusing in bright-field microscopy for tuberculosis detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC2965442/)，J Microsc 240(2):155–163，DOI `10.1111/j.1365-2818.2010.03389.x`，PMC 可用日期 2011 不是正式出版年。取得摘要、采集与 Focus measures、部分搜索正文；明确 Brenner 关联 1976、m=2，Tenengrad 采用所有像素的变体，F4 引 Vollath 1988。状态：**全文片段；公式在当前文本返回中缺失**。仅采用可读文字，不从缺失公式图猜精确边界；其转引 Santos 的内容不算独立起源证据。

<a id="source-8"></a>

**[cite:8] Mir、Xu、van Beek，2014／P1。** [An extensive empirical evaluation of focus measures for digital photography](https://cs.uwaterloo.ca/~vanbeek/Publications/spie2014.pdf)，已读作者及 §2.1 横／纵 Brenner 有限支持求和式、双向组合区别、Sobel 算子与对 Tenenbaum 的间接归属说明。状态：**全文片段**，参考表未返回；不把 Krotkov credits Tenenbaum 的表述当作 1970 原页。支持本卡 F01 正比例映射及不同方向不是同一配置，不将摄影比较结论外推显微场景。

<a id="source-9"></a>

**[cite:9] 本地规范、卡片与实际函数体。** 已读[学习规范](../algorithm_learning_standard.md)、[G09–G11 清单](../evolution_coverage.md#去重缺口清单)、[对焦总览](../autofocus_evolution.md)、[F01/F02/F06](measurement_focus.md#对焦共同数据契约)及[focus_measures.py](../../traditional/autofocus/focus_measures.py)。具体入口为 `_prep` L12–27、`brenner` L30–34、`tenengrad` L37–42、`vollath_f4` L67–72。源码是本地变体定义证据，不提供历史起源证明。原现代卡里的历史验证不自动算本次测试。

<a id="source-10"></a>

**[cite:10] Pertuz、Puig、Garcia，2012 在线／2013 卷期／P1。** [Analysis of focus measure operators for shape-from-focus](https://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf)，已读封面、首页及前段，确认 online 16 November 2012，Pattern Recognition 46(2013):1415–1432，以及部分 AF 指标为 SFF 改写的声明。状态：**全文片段**；本次返回未到附录，不用它补三个目标的原式。作为版本警戒证据：综述采用同名算子不保证原始 AF 与局部 SFF 支持一致。

## 验证记录与后续验收

本次用仓库现有 `.venv` 解释器、NumPy **2.5.4**、OpenCV **5.0.0**，通过标准输入执行临时核算，启用 `-B` 不写字节码，不新增测试脚本或下载附件。已执行：F02 既有反射边界例与偏置／增益关系；F01 既有斜坡、周期二纹理和小支持；F06 三个解析行输入、S4 映射恒等式和小支持。断言全部通过，W=2 的 F01/F06 确认返回 NaN。这只是本地定义核算，不是原论文复现、抗噪性能实验或实际焦差测量。

未执行：原文算法实现、原论文实验、全量 RTImg、硬件采集、速度剖析及焦点误差验收。以后验证至少需要固定版本与支持，分开报告数值等价、焦栈排序、重复采集波动、峰位正确性和运动闭环；不能用一个峰位碰巧相同证明两个评分等价。

最终验收边界：**G09 原评分与主动流程未闭合；G10 原评分及二手 1971 误引归因未闭合；G11 1987 原 F4/F5 未闭合。** 三张卡已建立，但三项均不具备升级“已有”或销项的证据。下一次补证应直接定位缺失原页与完整参考条目，而不是继续复制现代同名公式。

<a id="audit-20261106"></a>

## 2026-11-06 追加核验：原机制、引文闭合与精确页限制

本节是对 2026-11-05 记录的**增量补证**，前文及旧访问失败记录原样保留。前文“本次”“未取得最终参考表”等均描述旧轮次；本轮已取得 Santos 完整 PDF 并核验参考表，不能继续将该子项列作未获取。本轮仍未取得 Tenenbaum 1970 VI.3／VI.6、Brenner 1976 评分／搜索正文或 Vollath 1987／1988 的公式定义原页。三项总体仍不销项，历史目标五层仍为“部分”；不是要求全系统复现后才允许销项，而是目前尚缺覆盖矩阵要求的**具体核心机制及其承接证据**。[cite:9](#source-9)

### 本轮结论与最小剩余证据

| 项目 | 已闭合的子项／新增依据 | 最小剩余证据包 | 总体判定 |
|---|---|---|---|
| G09／11-01 | IJCAI 1971 作者原论文 p.3 的主动调焦描述，p.7 参考条目 6 明指 AIM-134，1970；这是不同论文的官方学术入口，不是同 PDF 换镜像 | 1970 VI.3 的准则定义页及其引用的算子定义；VI.6 的范围初始化、采样／更新、细化及停止分支原页，含 p.347 流程图及解释 | 原评分／测距流程缺证；不销项，不用 1971 边缘跟踪替代 VI.6 |
| G10／11-02 | Santos PDF 第 9 页＝印刷 p.272 的完整 Brenner 引文已取；与 PubMed、Ovid 的 1976 条目逐字段对应，关闭“1971 指向何文及年份错引”子项 | 1976 pp.100–111 中评分定义、方向／跨度／门控／聚合与调焦搜索的原段落、图式及前后文；确切方法页尚不能定位 | 误引对应已闭合，原评分与搜索仍缺；G10 不销项 |
| G11／11-04 | Santos PDF 第 3 页＝p.266 的 F4/F5 支持、无分母求和与 1988 标签经页图复核；p.272 明列两篇不同文献；本地不同支持均值非统一归一化再次核验 | 1987 pp.279–288 的准则名称／式号、支持、均值与归一化、二值化定义；1988 pp.133–146 的相应定义与假设页用于版本对照 | 原 F4/F5 未读，不将二手式或曲线图升级为原定义；不销项 |

只需补齐能承载这些规则的页及必要上下文，不要求无目的重读全部论文。页码未能确认时保留“整篇卷页＋目标段落”，**不猜具体页、不将 PDF 序号等同印刷页码**。对应依据见[本轮来源与访问台账](#sources-20261106)。

<a id="g10-citation-1971-closed"></a>

### G10：1971 引文逐字段闭合

Santos p.265 §2.1.1(3) 正文写 `Brenner et al., 1971`，p.272 References 确实印有下列条目；页图与 PDF 文本层相符，不是本轮 OCR 单独产生的年份错误：[cite:11](#source-11)

> Brenner, J.F., Dew, B.S., Horton, J.B., King, J.B., Neirath, P.W. & Sellers, W.D. (1971) An automated microscope for cytologic research. J. Histochem. Cytochem. 24, 100–111.

| 字段 | Santos p.272 字面内容 | 1976 正式条目 | 判断 |
|---|---|---|---|
| 前三作者 | Brenner J.F.；Dew B.S.；Horton J.B. | 相同 | 对应 |
| 第四作者 | King J.B. | King T. | 首字母不一致，不默改引文 |
| 第五、六作者 | Neirath P.W.；Sellers W.D. | Neurath P.W.；Selles W.D. | 姓氏拼写不一致 |
| 年份 | 1971 | 1976 Jan | 所引卷页对应 1976，不是另一篇 1971 原论文的证据 |
| 题名 | An automated microscope for cytologic research | 同一主体，另有 a preliminary evaluation | 二手条目省略副标题 |
| 期刊／卷／页 | J. Histochem. Cytochem. 24, 100–111 | 同刊 24(1), 100–111 | 卷页及题名、前三作者共同锁定同一条目 |
| DOI | 未列 | 10.1177/24.1.1254907 | DOI 来自正式索引，不回填为 Santos 已印内容 |

**结论限定：**可确认 Santos 的这条参考文献将 1976 论文错标为 1971，并伴作者字段差异；“正文年份→本篇参考表→1976 正式条目”的对应链已闭合。尚不能确定错误由作者、排版还是更早转引引入，不能宣称已找出最早误引者，也不能由此排除所有可能的 1971 相关研究。该生成原因不是取得原评分的替代品；本轮不再把“缺 Santos 完整条目”作为阻塞项。[cite:11](#source-11)[cite:12](#source-12)

原评分仍不能由 Santos 反推。p.265 的平方项写 `j+2`，门控仍写 `|g(i,j+1)-g(i,j)| ≥ v`；p.265 末至 p.266 首说明前三种函数改为横纵梯度取最大再累加。这些字面差别已核对页图，但既不是 1976 原规则的确认，也没有依据擅自“修正”为二距门控。[cite:11](#source-11)

<a id="g09-original-mapping-20261106"></a>

### G09：原机制 L1–L5 与问答映射

Pingle 与 Tenenbaum 的 *An Accommodating Edge Follower* 是 IJCAI 1971 的**另一篇作者原论文**。p.3 §5 的 Focus 段说明：离焦展宽边缘过渡，降低固定间隔两点的强度差；在边界对比度足够但没有清晰梯度峰时怀疑离焦；需要对焦时，相对固定镜头移动 vidicon，直至对焦准则最大化，以局部梯度作为自然的清晰度准则。p.7 参考条目 6 指向 1970 AIM-134，条目 7 指向 Horn 1968 *Focusing*。这可补足任务与主动执行器的原作者描述，却不能提供 VI.3 的聚合范数／阈值或 VI.6 的搜索程序。[cite:13](#source-13)

| 层／状态 | 本轮原机制映射与问答 | 成立条件、易错点与剩余证据 |
|---|---|---|
| L1／部分 | 问：从什么异常触发何种动作？答：1971 p.3 是足够对比但无明显梯度峰→怀疑离焦→移动 vidicon→最大化局部准则；本地 F02 只输出评分 | 这是另一任务的已读步骤；“最大化”是目标，不等于已知搜索迭代与停止容差。VI.6 范围测距分支仍缺 |
| L2／部分 | 问：为何不能只换边缘算子？答：作者把离焦导致的过渡展宽与局部梯度降低联系起来；主动改变采集可恢复适合任务的信息 | 原描述针对局部边缘；不证明全图平方和、任意纹理或任意噪声下单峰，也不把局部梯度直接写成 Sobel 能量 |
| L3／部分 | 问：弱边缘时先提高哪个量？答：先区分对比度、量化与离焦诊断；p.3 将 target voltage、Quantization Window、Focus 分开讨论 | 对历史 vidicon 的控制约束不能变成现代相机通用参数。没有原 VI.3 阈值初值、焦位步长或停止容差可搬用 |
| L4／部分 | 问：原准则累加幅值还是平方？答：仍不能回答；1971 的局部强度差描述与 1970 梯度／阈值图题都不足以指定核、范数和支持 | 必须取得 VI.3 定义及算子上下文。前文 F02 增益／边界推导仍仅对现代定义成立 |
| L5／部分 | 问：调用 tenengrad 加 focus_search 是否复现？答：不是；原评分契约、执行器标定、范围映射和 VI.6 分支未接入 | 核验 F02 只检验当前函数。后续需把原页每条分支映射到明确参考流程，再谈实现／未实现，不复用 1971 跟踪停止规则 |

**精确页限制：**沿用旧目录定位的 VI.3 pp.311–317、阈值相关图题 pp.315–316、VI.6 pp.336–351 及流程图 p.347，均为**待取得的印刷页**。本轮没有该扫描 PDF 的实体页序映射，不能声称 `#page=311` 指向印刷 p.311。可优先提供上述定义／流程图连同前后解释页；若评分回指更早的算子章节，还需被引用定义页。[cite:1](#source-1)[cite:14](#source-14)

<a id="g10-original-mapping-20261106"></a>

### G10：原机制 L1–L5 与问答映射

| 层／状态 | 本轮原机制映射与问答 | 成立条件、易错点与剩余证据 |
|---|---|---|
| L1／部分 | 问：1976 怎样评分、选下一焦位并停止？答：尚未取到该段；原系统工作模式与计数终止已在旧轮次记录，但不是内部调焦步骤 | 本轮 SAGE PDF 为 403，Ovid 为 PDF Only 书目／摘要；不能拿 Santos 焦栈评估流程填原搜索 |
| L2／部分 | 问：跨二距及门控是原机制吗？答：跨二距平方是已读二手版本与本地组件的共同部分，1976 的方向、门控和聚合尚缺原页 | p.265 的一距门控与二距平方并存，不能依据名称消去这个差别 |
| L3／部分 | 问：1971 是否仍需另找一篇原论文？答：这条引文已对应到 1976 论文，优先获取该文评分／搜索页，而非凭错年发明独立节点 | 只关闭此条书目误引对应，不声称找到最早误引原因或评分唯一首创年 |
| L4／部分 | 问：F01 与原式只差像素数吗？答：仅可对明确的现代横向无门控和写出乘数 H(W−2)；原式未取得不能认定等价 | 搜索曲线排序、阈值比较量、是否归一化均待原定义；固定尺寸也不能消除门控差异 |
| L5／部分 | 问：已有 F01 通过测试能否销项？答：不能；dtype、二距支持和空支持测试只覆盖该现代组件 | 最小实现映射需原评分→支持／单位→原搜索状态更新／停止，不要求先造整套 SCANIT，却不能缺失核心流程 |

**精确页限制：**目前只知正式篇幅 pp.100–111；本轮未取得原 PDF，不能给出评分式号、搜索图号或确切所在页。应在合法取得的全文中检索 `focus`／`focusing`／`sharpness`，保留命中页、跨页续文及被引用的定义；若原文仅转引评分／搜索，还需其直接来源。PubMed、Ovid 不含这些方法页。[cite:12](#source-12)[cite:15](#source-15)

<a id="g11-original-mapping-20261106"></a>

### G11：原机制 L1–L5 与问答映射

| 层／状态 | 本轮原机制映射与问答 | 成立条件、易错点与剩余证据 |
|---|---|---|
| L1／部分 | 问：1987、1988 与 Santos 是否同一个定义入口？答：不是；1987 出版摘要只支持相关准则及二值路线，Santos p.266 的两个具体标题标 1988，p.272 分列两篇 | 需要两年的定义页分别核对；不可把 1988 性质分析节点 N23 合入 G11，也不凭二手命名补原式号 |
| L2／部分 | 问：相关法如何抗噪、二值化如何省算？答：原摘要提出较低噪声敏感性及随机遍历计量的二值计算路线，但其概率关系和阈值转换尚缺 | 前文独立零均值白噪声的期望推导是明确条件模型，不是已经读到原证明；简单分割路线不自动等于 Otsu |
| L3／部分 | 问：曲线纵轴归一到 1 是否证明评分除以支持数？答：不能，展示尺度、评分定义分母和各滞后的有效计数是三回事 | 需原图题及定义。只有 Relative Weight 或 σ 标记的裁图，不能确定频率权的解析式、阈值或归一化协议 |
| L4／部分 | 问：F06 是否为 F4 的统一归一化？答：对已核 Santos 式不是，两个 lag 的分母不同，差含图像依赖修正项；对原版仍待定义页 | 本轮严格有理数重验前文恒等式；不从二手有限支持倒推 1987 或 1988 原支持 |
| L5／部分 | 问：如何逐项对接原实现？答：先记录原轴、每项支持、分母、均值域、边界与二值映射，再对照 F06 两个独立均值；F5 本地未实现 | 常量、负分、偏置、支持退化是必要测试，但不能替代原定义或实验；本轮不实现任何原版评分 |

**支持与归一化复核。** Santos p.266 经明确的轴重标记 M→W、N→H 后，仍是前文的 \(S_4=A-B\)、\(S_5=A-HW\mu^2\)；第一项 lag 1 支持为 \(H(W-1)\)，第二项 lag 2 为 \(H(W-2)\)，两个求和式没有各自均值分母。F06 是 \(V=A/[H(W-1)]-B/[H(W-2)]\)，因此：

\[
S_4=H(W-1)V+\frac{B}{W-2}.
\]

即使 H、W 固定，附加项仍随图像变化；不能叫“只做归一化”。对常量 c，Santos 式的 \(S_4=Hc^2\)、\(S_5=-Hc^2\)，而 F06 为零。式中的 \(\mu\) 是全图均值，不能无说明地替换成配对支持的均值。以上均是**Santos 二手式→本地定义**的精确比较，不标作 Vollath 原式。[cite:11](#source-11)[cite:9](#source-9)

**精确页限制：**1987 只定位篇幅 pp.279–288，1988 只定位 pp.133–146，尚无公式页／式号／PDF 序号。NSTL 的 1988 页面虽写“全文图片”，本轮实际打开的前三个资源仅为曲线裁图：Relative Weight 的 1,2／3／4／5 标记，以及两张带 σ 标记的 Intensity 曲线；均无页码、图题和公式上下文。它们不是完整原页，不能按网页排列推成 pp.133–135，也不能从曲线外形拟合回原 F4/F5。临时证据来源中的同类三图不构成另一个独立来源，不归因用户。[cite:16](#source-16)

<a id="sources-20261106"></a>

### 本轮来源与访问台账

下列编号续接旧来源；同文文本层、渲染页图及临时证据来源中的相同页图是同一内容的校对，不计多个独立来源，也不归因用户。下载／核算均沿用原取证轮次记录，本次同步未重跑；完整文件下载成功不等于全篇方法逐段阅读。

<a id="source-11"></a>

**[cite:11] Santos 等，1997；来源 4 的本轮获取升级。** [UPM 作者站 PDF](https://www2.die.upm.es/im/papers/Autofocus.pdf)：保持 TLS 验证，下载成功，2,642,116 bytes，共 9 页，SHA-256 `6d9b286ad95ab50bf4864b4db47eae06c4e1e380a9d9f09f993e5c6821168d5c`。按页提取文本后检索 Brenner／Vollath／1971／References，核对 PDF 第 2、3、9 页的完整文本与渲染页图，印刷页分别为 265、266、272；临时证据来源中的三张完整页图与这些页对应，不归因用户。**完整 PDF 已得；关键页已核，非全篇实验重审。** 对自身印刷的参考条目／公式是直接内容证据，对 1970／1976／1987／1988 原机制仍是 P1 二手。页码可用 [p.265（PDF 2）](https://www2.die.upm.es/im/papers/Autofocus.pdf#page=2)、[p.266（PDF 3）](https://www2.die.upm.es/im/papers/Autofocus.pdf#page=3)、[p.272（PDF 9）](https://www2.die.upm.es/im/papers/Autofocus.pdf#page=9) 定位；fragment 是否由阅读器执行不影响已核页序。

<a id="source-12"></a>

**[cite:12] Brenner 1976 正式条目本轮复核。** [PubMed PMID 1254907](https://pubmed.ncbi.nlm.nih.gov/1254907/)与[Ovid 出版记录](https://www.ovid.com/journals/jhcy/abstract/10.1177/24.1.1254907~an-automated-microscope-for-cytologic-research-a-preliminary?redirectionsource=fulltextview)：均读到六作者、题名、January 1976、24(1):100–111、DOI 和系统摘要；Ovid 明标 `Journal Article: PDF Only`，返回内容不含评分和搜索。两入口与来源 11 的参考表对照支持“该 1971 条目是对 1976 论文的错误引用”这一有限结论，**不构成三个独立方法实验**。

<a id="source-13"></a>

**[cite:13] Pingle、Tenenbaum，1971；IJCAI 官方原论文。** [An Accommodating Edge Follower](https://ijcai.org/Proceedings/71/Papers/001.pdf)，完整 PDF 7 页，SHA-256 `356720de6cfaa8b5bf719988d0ba75c1c5ecbf9081ce927149327812ad734dcd`。逐页提取并检索 Focus／gradient／AIM-134，重点核对 [p.3 §5 Focus（PDF 3）](https://ijcai.org/Proceedings/71/Papers/001.pdf#page=3)和 [p.7 References 6、7（PDF 7）](https://ijcai.org/Proceedings/71/Papers/001.pdf#page=7)的文本／页图；临时证据来源中的 pp.3、7 同页校对一致，不归因用户。状态：**P0 作者原论文关键页，[SINGLE-SOURCE] 1971 局部调焦描述**；不是 1970 目标章节，不以作者重叠冒充独立验证 VI.3／VI.6。p.3 的相机电压式不是所缺的对焦评分式，不混用式号。

<a id="source-14"></a>

**[cite:14] Tenenbaum 1970 本轮原页获取限制。** [Stanford 原 PDF](https://i.stanford.edu/pub/cstr/reports/cs/tr/70/182/CS-TR-70-182.pdf)直接下载返回 `SSL: UNEXPECTED_EOF_WHILE_READING`，未禁用证书验证；[Bitsavers PDF](https://bitsavers.org/pdf/stanford/Stanford_CS_TR_Collection_2025-12-12/PDF/1970/CS-TR-70-182.pdf)返回 HTTP 403，停止该入口，不加伪装请求或继续枚举绕行路径。[Stanford SearchWorks 查询](https://searchworks.stanford.edu/?q=%22Accommodation+in+computer+vision%22)提取为空；[Stanford 馆藏 AIM 清单](https://stacks.stanford.edu/file/druid:qy276zh7089/qy276zh7089.pdf)返回为清单文本片段，非目标方法页。[SAILDART 学位目录](https://www.saildart.org/allow/pdf/THESIS/)读到 AIM-134、Jay Martin Tenenbaum、September 1970，但该行没有 PDF 超链接；只作馆藏线索。**本轮未取得 VI.3／VI.6 原页**；未获取不能解释为原文不存在。

<a id="source-15"></a>

**[cite:15] Brenner 1976 本轮原 PDF 获取限制。** [SAGE PDF](https://journals.sagepub.com/doi/pdf/10.1177/24.1.1254907)直接下载返回 HTTP 403；已停止，不换同站路径或设法绕过。按 DOI 查询 [NSTL](https://www.nstl.gov.cn/paper_detail.html?doi=10.1177%2F24.1.1254907)只返回导航文字，没有取得条目正文或图式。不同官方学术入口的 PubMed／Ovid 仅到来源 12 的书目摘要层。故旧来源 3 的系统正文片段仍是历史记录，本轮不能声称重新取得或读全 pp.100–111。

<a id="source-16"></a>

**[cite:16] Vollath 1987／1988 官方入口与图像限制。** 本轮重读 [1987 Wiley 摘要](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2818.1987.tb02839.x)，确认题名、作者、147(3):279–288、September 1987、自相关与两种二值路线；[1987 epdf](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-2818.1987.tb02839.x)及 [1988 Wiley 摘要入口](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2818.1988.tb04620.x)显示安全验证，均停止，未尝试解验证码、获取受限 PDF 或付费访问。[1987 NSTL DOI 查询](https://www.nstl.gov.cn/paper_detail.html?doi=10.1111%2Fj.1365-2818.1987.tb02839.x)只返回导航；[1988 NSTL 条目](https://www.nstl.gov.cn/paper_detail.html?doi=10.1111%2Fj.1365-2818.1988.tb04620.x)读到题名、D. Vollath、DOI `10.1111/j.1365-2818.1988.tb04620.x`、151(2):133–146 和摘要片段。页面导出的前三个图像资源为 [Relative Weight 裁图](https://aka.doubaocdn.com/s/W12cW3K9hu)、[Intensity 裁图一](https://aka.doubaocdn.com/s/dz4S94zxPL)、[Intensity 裁图二](https://aka.doubaocdn.com/s/kKimrYjtlV)，已打开核对，仅作图像片段；CDN 地址不是出版社稳定页码标识，不保证长期可用。未按“全文图片”标签声称已得全文，也未从裁图补写公式。状态：**1987 摘要、1988 书目／摘要片段／三张裁图；两年原方法定义均 [INSUFFICIENT DATA]**。

### 本轮核验记录与交付边界

本轮通过标准输入、`.venv` 解释器的 `-B` 模式执行核算，没有新增仓库测试脚本。NumPy **2.5.4**、OpenCV **5.0.0**；重新调用当前 `brenner`、`tenengrad`、`vollath_f4`，并用 `Fraction` 严格有理数独立检查支持映射。F02 的 64/3 反射边界例及增益／偏置关系、F01 的斜坡 4／周期二纹理 0、F01/F06 的 W=2→NaN 均通过。F06 对 `(1,2,3)`、`(1,0,1)`、`(2,1,2)`、`(2,2,2)`、`(0,1,2,3)` 的实际返回与 \(A/N_1-B/N_2\) 一致；两种 S4 消元恒等式全部通过。这里的输入是定义核算例，不是合成实验数据或论文实测。

文献核验包括 Santos 页数／哈希／页码对应／完整 1971 条目和两年 Vollath 条目，以及 IJCAI p.3／p.7 的图文对应。PDF 与逐页提取、渲染产物仅放系统临时目录 `focus_origins_20261106`，不纳入仓库；为渲染 PDF，已在现有 `.venv` 安装 PyMuPDF 及其依赖，未改项目依赖清单。交付的文档变更仅为本文件追加；覆盖矩阵、索引、总数、现代卡与算法源码均不调整。未执行原算法复现、原实验、焦栈性能、硬件闭环或速度验收，不把数值恒等式检查写成原机制通过。

**最终状态：G09 不销项；G10 仅关闭 Santos 1971 错引对应子项，整体不销项；G11 不销项。** 下一轮最有价值的输入是带可核题名／页码的 1970 VI.3／VI.6、1976 评分／搜索、1987／1988 定义与二值化页；三张无图题曲线图或更多现代同名公式不能减少这些剩余证据。

<a id="audit-20261106-r2"></a>

## 2026-11-06 再续核验：新入口与指定原页获取结果

本轮先读来源 1–16 的访问记录及[最小剩余证据表](#audit-20261106)，再检查 TEMP 文件与现有 PDF 工具。旧轮次结论、L1–L5 和题目范围不改；**本轮没有取得新的目标原规则，不增加历史机制条目，也不升级层级。** Santos 的 1971 错引对应已闭合，保持关闭，不重新列为待补证。

### 指定页与逐项销项判定

| 项目／指定印刷页 | 本轮实际取得与渲染 | 尚未取得的最小证据 | 能否销项 |
|---|---|---|---|
| G09：1970 pp.311–317、336–351 | 未取得目标 PDF；两组指定页均未渲染。只复核现有 1971 作者论文与新馆藏入口 | VI.3 原评分及回指算子定义；VI.6 初始化、采样／更新、细化、停止分支和 p.347 流程图上下文 | **不能**；1971 局部调焦不是 1970 原评分／测距程序。[cite:17](#source-17)[cite:18](#source-18) |
| G10：1976 pp.100–111 | 未取得目标 PDF，指定页未渲染；新 Ovid 目录及其明确提供的全文入口仍只到书目／摘要 | 评分的方向、跨度、门控、聚合及调焦搜索原段落、图式和必要直接来源 | **不能**；Santos 错引对应子项维持已关闭，但原评分／搜索未闭合。[cite:17](#source-17)[cite:19](#source-19) |
| G11：1987 pp.279–288；1988 pp.133–146 | 两年目标 PDF 均未取得、未渲染；Santos p.266／p.272 仍只是二手定义和参考表 | 两年分别可核的准则名称／式号、支持、归一化、均值域、二值转换及噪声假设 | **不能**；不把二手 F4/F5 或 1988 曲线裁图算作原定义，也不把 1988 独立节点并入 G11。[cite:17](#source-17)[cite:20](#source-20) |

以上“未渲染”是缺少相应原文件，不是已执行渲染却失败；没有猜测任何目标原文的 PDF 页序。三项原规则均为 **[INSUFFICIENT DATA]**，没有新增可销的核心机制子项。

### TEMP 与渲染能力复核

<a id="source-17"></a>

**[cite:17] 本地现有文件／本轮运行证据。** 递归列出当前系统 TEMP 的 `.pdf`，并核查仓库内 `.pdf`。`C:\Users\Administrator\AppData\Local\Temp\focus_origins_20261106` 中只有下列两份完整 PDF；其他 TEMP PDF 文件名属于其他取证任务，未冒充本题原文。没有找到以四篇目标原文身份保存的本地 PDF；此结论限于本轮检查范围，不声称全盘不存在。

| 现有文件 | 字节／页数／SHA-256 复核 | 本轮内存渲染 |
|---|---|---|
| `Santos.pdf` | 2,642,116 bytes；9 页；`6d9b286ad95ab50bf4864b4db47eae06c4e1e380a9d9f09f993e5c6821168d5c`，与来源 11 相同 | PDF 2、3、9，各 596×842 像素，成功 |
| `Pingle_Tenenbaum_1971.pdf` | 3,829,909 bytes；7 页；`356720de6cfaa8b5bf719988d0ba75c1c5ecbf9081ce927149327812ad734dcd`，与来源 13 相同 | PDF 3、7，各 566×750 像素，成功 |

沿用 `.venv\Scripts\python.exe -B`，已存在的 PyMuPDF `VersionBind` 实测输出 `2.1`；两文件 `needs_pass=0`。使用 `get_pixmap(matrix=Matrix(1,1), alpha=False)` 在内存中渲染，没有写入新 PDF、页图或脚本。另检索既有逐页文本中的 `Brenner|Vollath|AIM-134|Focus|References`，打开既有 Santos p.266／p.272、Pingle p.3 页图校对，分别确认 1988 标签、1971 错引条目与局部调焦描述；它们仍不是四篇指定原文。没有重新下载这两份 PDF，也没有将本轮仅做内存渲染的其余页写成重新目视核读。

### 新入口、参数与失败台账

本轮未再次请求旧失败的 Stanford 原 PDF、Bitsavers／decromancer 目标 PDF、SAGE PDF／full／epdf、Wiley 安全验证入口或 NSTL DOI 参数。检索服务再次列出这些旧链接时，仅识别为已知入口，不点击、不追加 `#page`、查询参数或请求头重试。以下新入口也不作为绕过登录、付费或安全验证的手段。

<a id="source-18"></a>

**[cite:18] G09：SAILDART 总目录与 DTIC／Internet Archive 新路径。**

| 新尝试／精确地址 | 实际结果 | 证据边界与停止点 |
|---|---|---|
| [SAILDART 扫描总目录](https://www.saildart.org/allow/pdf/) | 返回扫描目录及 AIM 编号表；134 是纯文字，相邻若干编号有 PDF 超链接 | 不同于旧 `/allow/pdf/THESIS/`；没有 AIM-134 下载链接，不根据邻号猜本站 PDF 路径 |
| [Internet Archive 题名查询](https://archive.org/advancedsearch.php?q=title%3A%22Accommodation+in+computer+vision%22&output=json) | 内容提取返回 `local crawler returned empty content` | 未取得 JSON，不能写成查询结果为零；同地址不重试 |
| [书目检索命中](https://scispace.com/pdf/bibliography-of-stanford-computer-science-reports-1963-1983-4wgc4bn9l2.pdf) | 检索摘要列出 `STAN-CS-70-182 (AIM-134, AD748565)`，题名及 452 页 | 只得到书目摘要；`AD748565` 是待原页核实的检索线索，不是已获报告 |
| [Internet Archive 候选元数据](https://archive.org/metadata/DTIC_AD0748565) | 内容提取返回 `local crawler returned empty content` | 标识是依据上述编号构造的候选，未验证馆藏条目存在，不称为已发现镜像 |
| [DTIC 候选 PDF](https://apps.dtic.mil/sti/pdfs/AD0748565.pdf) | 标准库 `urllib.request.urlopen`、默认 TLS 校验、30 秒超时：`URLError <urlopen error [Errno 11002] getaddrinfo failed>` | DNS 解析阶段失败，未收到 HTTP 状态或 PDF 字节；不是 403、404 或文件不存在的证明；未改主机、参数重试 |

未将索引中的 452 页直接换算成指定印刷页的 PDF 序号。DTIC 候选编号仍需成功取得题名／封面才能确认身份。

<a id="source-19"></a>

**[cite:19] G10：开放位置索引与 Ovid 卷期目录链。** [OpenAlex DOI 查询](https://api.openalex.org/works/https://doi.org/10.1177/24.1.1254907)返回 `W2023597557`，`oa_status=closed`、`best_oa_location=null`、`any_repository_has_fulltext=false`、`has_content.pdf=false`；两个 `locations` 是 DOI 和 PubMed，`pdf_url` 都为 `null`。这是该索引当前记录没有提供开放全文位置，**不能推断所有合法镜像都不存在**。

[Ovid 1976 卷 24 第 1 期目录](https://www.ovid.com/journals/jhcy/toc/1976/24010)返回 100–111 页目标条目及实际超链接；沿该链接访问[不带参数的全文入口](https://www.ovid.com/journals/jhcy/fulltext/10.1177/24.1.1254907~an-automated-microscope-for-cytologic-research-a-preliminary)，只得到 `Journal Article: PDF Only`、六作者、卷页、DOI 和系统摘要，没有评分式或 PDF 下载内容。它是目录明确给出的新地址，不是旧 `abstract?...redirectionsource=fulltextview` 的重复请求；未再枚举 Ovid 下载路径。URL 含 `fulltext` 不等于已取得全文。

<a id="source-20"></a>

**[cite:20] G11：两年分别查询 OpenAlex 与作者机构 KITopen。**

| 新入口／参数 | 实际返回 | 是否取得原定义 |
|---|---|---|
| [1987 OpenAlex DOI 查询](https://api.openalex.org/works/https://doi.org/10.1111/j.1365-2818.1987.tb02839.x) | `W2158133371`；147(3):279–288；唯一 location 是 DOI，`pdf_url=null` | 否 |
| [1988 OpenAlex DOI 查询](https://api.openalex.org/works/https://doi.org/10.1111/j.1365-2818.1988.tb04620.x) | `W2081760847`；151(2):133–146；唯一 location 是 DOI，`pdf_url=null` | 否 |
| [KITopen 主页](https://publikationen.bibliothek.kit.edu/) | 返回年度导航；标准库读取 HTML 为 HTTP 200，未发现带 `name` 的检索输入／选择框、form 或匹配 search／suche／Vollath 的链接 | 否；不意味着网站没有其他检索服务 |
| [KITopen 1987 年度首屏](https://publikationen.bibliothek.kit.edu/index.php?year=1987&hitsperpage=100)、[1988 年度首屏](https://publikationen.bibliothek.kit.edu/index.php?year=1988&hitsperpage=100) | 返回年度条目及分页链接；本次返回片段没有目标两篇或相应 PDF | 否；没有遍历全部分页，不得写成机构库全库零命中 |
| [KITopen 候选检索地址](https://publikationen.bibliothek.kit.edu/search.php) | 标准库请求返回 HTTP 404 | 地址是尝试而非已知官方搜索端点；停止，不继续猜参数 |

两个 OpenAlex 记录均为 `oa_status=closed`、`best_oa_location=null`、`any_repository_has_fulltext=false`、`has_content.pdf=false`。该结果仅约束索引所知入口，不是全文法律状态或全球可获得性的最终判断。机构库导航与现代转引不能补原 F4/F5 定义。

<a id="source-21"></a>

**[cite:21] 检索过程记录，不作原机制证据。** 本轮按题名、作者报告和机构域名检索；有用的新路径只有来源 18–20，其他结果为旧入口、二手引用或无关命中。用于避免下一轮原样重复的检索式如下：

| 目标 | 本轮检索式 | 结果限定 |
|---|---|---|
| G09 题名／文件 | `"Tenenbaum" "Accommodation in computer vision" pdf`；`"Accommodation in computer vision" "pdf" "1970" Stanford`；`"CS-TR-70-182.pdf"`；`"Tenenbaum" "AIM-134" download` | 重返旧扫描链接、AIM 清单及 1971 作者论文；没有新目标 PDF |
| G09 报告号 | `"Accommodation in computer vision" DTIC`；`"Tenenbaum" "1970" "pdf" "sail"`；`"AD748565"`；`"AD0748565" Tenenbaum` | 第二式返回含报告号的书目摘要；第三式为无关结果，第四式明确无结果；不将其当作 DTIC 原页 |
| G10 | `"Brenner" "An automated microscope" pdf 1976`；`"An automated microscope for cytologic" "pdf" Brenner -site:semanticscholar.org -site:ovid.com -site:wiley.com -site:arxiv.org`；`"1254907" "pdf" microscope`；`"An automated microscope" "Brenner" "report" 1976 1975` | 主要为二手论文；最后一式提供 Ovid 卷期目录，正文结果见来源 19 |
| G11 通用／作者 | `"Vollath" "1987" "autofocus" pdf`；`"Vollath" "1988" "autofocusing" pdf`；`"Vollath" "Automatic focusing" Karlsruhe report`；`"The influence of the scene parameters" Vollath pdf`；`"Vollath" "Scharfeinstellung" "1987"` | 旧 Wiley 页面、二手转引与专利线索；未据专利或其他系统报告扩题 |
| G11 机构域名 | `site:publikationen.bibliothek.kit.edu Vollath "focusing"`；`"Vollath" "correlative" site:kit.edu`；`"Vollath" "scene parameters" site:kit.edu` | 第一式返回无关条目，后两式明确无结果；不等于全库不存在 |

### 本轮验证与交付边界

本轮实际执行的是文件盘点、两份既有 PDF 的字节数／页数／哈希校验、五页内存渲染、三个既有页图校对及新网络入口查询。未安装任何依赖，未使用失效的全局 Python 应用别名，没有新增下载文件、提取文件或取证脚本；只追加本文件。没有运行原算法、现代评分数值测试、硬件实验或全量测试，也不沿用上一轮测试结果冒充本轮执行。

**最终判定：G09 不能销；G10 错引对应子项保持已闭合、整体不能销；G11 不能销。** 后续仍须取得表中指定原页及其必要上下文；本轮新增的是可复核的访问边界与去重台账，不是原规则或原算法验证。
