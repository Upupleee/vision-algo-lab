# 定位、匹配与配准：方法演进与工程落地双线样章

[文档导航](README.md) · [旋转／尺度 ZNCC 旧专题](pose_ncc.md) · [实验记录](experiments.md) · [研究笔记](research_notes.md)

[算法卡片总索引](algorithm_cards/README.md) · [匹配51卡索引](algorithm_cards/README.md#定位匹配配准) · [L1–L5正文](algorithm_cards/matching.md) · [保留旧稿](algorithms/matching_l1_l5.md)

51卡含具体实现、度量、搜索拒识与评测组件，数量不是独立算法数；族级基线不是所有算法全史。五层覆盖沿用原卡K／R／M协议，本次导航更新不升级验证状态。

## 1. 范围与阅读约定

本章用两条相互对照的线索回答“为什么出现这类方法”和“在本仓库中能用到哪一步”。**方法演进线**沿问题、表示、优化与计算代价展开；**工程落地线**沿成像条件、变换模型、候选产生、几何验证、精化与拒识展开。这里不是把传统算法排成被深度学习逐代淘汰的单链，也不是按发表年份给精度排名。

覆盖 LK、RANSAC、相位相关、Chamfer、FFT-NCC、SIFT、ORB、ECC、SuperPoint、SuperGlue、LoFTR、LightGlue 十二个代表分支。所列年份是选定代表论文的年份，不据此宣称整个思想的首创；选到 LightGlue 不表示它在核验日仍是“最新方法”。本章没有做十二方法的统一性能实验，不提供跨硬件速度倍数或工业精度承诺。

在保留上述十二分支及原有编号的基础上，本次增补 [匹配滤波理论源头](#matched-filter-history) 与 [1972 年 SSDA](#ssda-history)：前者解释已知信号在指定噪声模型下为何使用相关，后者解释如何减少差候选的比较工作。North 的报告是匹配滤波／相关检测路线的重要早期理论来源，**不是所有定位、匹配与配准方法的唯一起源**；也不据此认定相关运算或 ZNCC 由 North 首创。[cite:14](#source-14)[cite:15](#source-15)

- **定位**：寻找已知模板实例的位置，可能同时估计旋转和尺度。
- **对应匹配**：找到两图中的点对；点对本身不是目标姿态，也不是单应矩阵。
- **配准**：在选定几何模型下估计图像间变换；平移、相似、仿射、单应与非刚性模型不能混用。
- **实现状态**：“已实现”仅指本仓库存在可执行算法；“骨架”指有接口或路线文字但不能执行算法；“未实现”指本次检查未见对应算法路径。依赖库可能提供某 API，不等于仓库已接入。
- **工业用途**：各条列的是满足条件时可评估的工程用途，属于选型推论，不是本仓库已有客户部署或论文已经验证的产线结果。

联网核验日期为 **2026-11-04**。引用编号是本章的来源索引；原论文、作者／会议页面与现代库文档分开列出。年份版本差异、单一来源与全文未取得项见第 7 节。

## 2. 两条线的总图

### 2.1 方法演进线：并行分支与连接点

| 代表年份 | 分支 | 主要改变的环节 | 不应据此推导的能力 | 来源 |
|---|---|---|---|---|
| 1943 报告／1963 重印 | 匹配滤波理论源头 | 已知信号、白噪声条件下的线性滤波输出 SNR 最大化；与相关运算相连 | 所有匹配的唯一起源、ZNCC 首创或任意噪声下检测最优 | [cite:14](#source-14)[cite:16](#source-16) |
| 1972 | SSDA（逐次相似性检测） | 逐次累积比较量，提前淘汰差候选，减少完整求和 | 所有早停都无损、直接等于 FFT-NCC 或任意场景固定加速 | [cite:15](#source-15)[cite:5](#source-5) |
| 1975 | 相位相关 | 用频域相位关系估计平移 | 一次基础相位相关同时估计任意旋转／尺度 | [cite:1](#source-1) |
| 1977 | Chamfer | 用边缘距离场代替严格边缘重合 | 对杂乱边缘、遮挡或任意形变无条件鲁棒 | [cite:2](#source-2) |
| 1981 | Lucas–Kanade（LK） | 梯度线性化与局部迭代，减少候选枚举 | 无初值的全局搜索，或自动获得所有后续 LK 变体 | [cite:3](#source-3) |
| 1981 | RANSAC | 从含外点的数据中估计几何模型 | 自动生成图像特征或判定零件语义 | [cite:4](#source-4) |
| 1995 | FFT-NCC 代表实现 | 变换域相关分子与局部归一化统计结合 | FFT 本身带来旋转／尺度不变性 | [cite:5](#source-5) |
| 1999／2004 | SIFT | 尺度空间关键点与局部描述子 | 完整仿射／透视不变，或无需几何验证 | [cite:6](#source-6) |
| 2008 | ECC | 对相关系数目标进行连续参数配准 | 全局最优、任意光照不变或无初始化要求 | [cite:7](#source-7) |
| 2011 | ORB | 面向计算与存储开销的二值局部特征 | 在任何工件上都比 SIFT 更准确 | [cite:8](#source-8) |
| 2018；预印本 2017 | SuperPoint | 学习兴趣点与描述子 | 单独输出完整可靠配准 | [cite:9](#source-9) |
| 2020；预印本 2019 | SuperGlue | 对两组局部特征做上下文匹配与拒配 | 取代特征提取或几何模型检验 | [cite:10](#source-10) |
| 2021 | LoFTR | 绕过独立关键点检测，粗到细建立对应 | 完全无纹理区域一定可观测，或等于稠密光流 | [cite:11](#source-11) |
| 2023 | LightGlue | 自适应计算量的稀疏局部特征匹配 | 固定最坏延迟，或对所有场景统一领先 | [cite:12](#source-12) |

相位相关和 FFT-NCC 共享傅里叶工具，但归一化对象不同；LK 和 ECC 都可作局部直接配准，但目标函数与更新推导不同；Chamfer 是距离场形状路线，不是把灰度相关简单改名。SIFT、ORB、SuperPoint 提供局部特征，SuperGlue、LightGlue 着重对应关系，LoFTR 则绕过独立检测器。RANSAC 横跨特征路线，负责鲁棒几何估计，而不是这些表示方法的后一代。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)[cite:7](#source-7)[cite:9](#source-9)[cite:10](#source-10)[cite:11](#source-11)[cite:12](#source-12)

### 2.2 工程落地线：同一条生产任务链

| 阶段 | 需要回答的问题 | 可选方法角色 | 仓库现状 |
|---|---|---|---|
| 成像与任务定义 | 是否饱和、失焦、低重叠？寻找一个还是多个目标？ | 所有路线共有前提 | 有局部处理与理论文档，无统一现场采集闭环 |
| 粗定位／初值 | 合理的平移、角度、尺度范围是什么？ | NCC、姿态网格、相位相关、形状搜索或特征匹配 | 有平移 NCC 与旋转／尺度 ZNCC |
| 对应生成 | 可重复的纹理／边缘在哪里？ | Chamfer、SIFT、ORB、SuperPoint + 匹配器、LoFTR | 有梯度方向平移基线；学习匹配仅骨架 |
| 几何验证 | 点对是否支持同一个物理模型？ | RANSAC + 相似／仿射／单应等模型 | 未接入；不能把 NCC 的 NMS 当作 RANSAC |
| 连续精化 | 初值已近且有效支持稳定吗？ | LK、ECC 或专门峰值拟合 | 只有 NCC 平移评分峰的三点抛物线精化 |
| 拒识与下游 | 是否歧义？坐标方向正确吗？是否允许驱动设备？ | 分数、分差、残差、覆盖率、物理先验 | 姿态 ZNCC 有分数／空间分差拒识，无设备安全闭环 |

“特征匹配 → RANSAC → ECC”是可评估的组合，不是固定必经链，也不是已经实现的流水线。已经对齐的小 ROI 不一定需要复杂特征；大位移且缺少初值时不宜直接把局部优化当作全局定位。先匹配再测量仍需标定和不确定度分析，见 [标定坐标](calibration_coordinates.md)。

## 3. 灰度、频域、形状与局部优化分支

### 3.0 前史：匹配滤波理论与逐次比较加速

<a id="matched-filter-history"></a>

**匹配滤波：North 1943 年报告／1963 年重印。** D. O. North 的 *An Analysis of the Factors Which Determine Signal/Noise Discrimination in Pulsed-Carrier Systems* 原为 RCA Laboratories Report PTR-6C，日期为 **1943-06-25**；1963 年 7 月在 *Proceedings of the IEEE* 51(7)、1016–1027 页重印。IEEE 同期 *Preface to North paper* 明确给出报告编号、原始日期和重印关系；不能把 1963 写成该报告首次形成的年份，也不能把前言的 1015 页或 DOI 当作 North 正文的信息。[cite:14](#source-14)

**方法演进线。** 从已知波形的噪声中检测问题出发，匹配滤波把“像不像模板”连接到一个明确的优化目标：在指定采样时刻，使线性滤波器输出的信号分量平方与噪声方差之比最大。以下用现代有限维记号解释条件，不冒充 North 原文公式的逐字转录。[cite:14](#source-14)[cite:16](#source-16)

- **问题**：已知、非零、有限能量的确定信号受到加性噪声干扰，需要有效汇集其证据，而非只看单个样本。
- **改进／白噪声 SNR 条件**：令观测向量为 `x = s + n`，噪声满足 `E[n] = 0`、`E[n nᴴ] = σ²I`、`σ² > 0`，线性输出为 `z = wᴴx`（`ᴴ` 表示共轭转置）。则 `SNR(w) = abs(wᴴs)² / (σ² wᴴw) ≤ (sᴴs) / σ²`；由 Cauchy–Schwarz 不等式，非零 `w` 与 `s` 成比例时取等号。写成卷积即 `h(t) = k s*(t₀-t)`，在 `t₀` 采样，相当于与已知信号做相关；`*` 表示复共轭，实信号只需反转。[cite:16](#source-16)
- **代价／边界**：这里的“最优”仅针对给定信号、观测支持、采样时刻与二阶噪声模型下的**线性输出 SNR**；该证明本身不要求高斯分布，白噪声也不等于独立高斯噪声。若要声称给定虚警率下检测概率最优，还须另列检测假设、噪声分布（如高斯）、未知参数和判决规则。有色噪声的正定协方差为 `C` 时，权重变为 `w ∝ C⁻¹s`，可理解为先白化再匹配；不能把未经白化的普通相关称为一般有色噪声下最优。[cite:16](#source-16)
- **工业用途**：可评估已知脉冲／条纹响应的检测与定位，或作为图像模板相关的理论参照；真实纹理背景、照明漂移、饱和与几何变形不自动符合该信号模型。
- **实现状态**：**未实现独立的经典匹配滤波检测器或噪声白化链路**。现有 ZNCC 分子确实使用相关运算，但这不等于仓库已经实现 North 报告的完整检测系统，也不证明其评分满足上述最优条件。
- **来源状态**：IEEE 书目、摘要与重印前言已读取，MIT Lincoln Laboratory 历史文章交叉确认 1943／1963 关系；未取得 North 正文全文，理论条件另由现代教学材料和本节代数说明支撑。[cite:14](#source-14)[cite:16](#source-16)

**工程落地线：相关不等于 ZNCC。** 对实值图像窗口 `Iᵤ` 与模板 `T`，未归一化相关为 `c(u) = Σ Iᵤ(i) T(i)`；它保留亮度、能量和模板大小的影响。固定支持下 `SSD(u) = Σ Iᵤ(i)² + Σ T(i)² - 2c(u)`，只有候选窗口能量也保持不变时，最小 SSD 与最大原始相关才有相同排序，不能省略这一条件。[cite:5](#source-5)

ZNCC 则为 `ρ(u) = Σ[(Iᵤ(i)-μᵤ)(T(i)-μT)] / sqrt(Σ(Iᵤ(i)-μᵤ)² × Σ(T(i)-μT)²)`，均值和求和均取相同有效支持。它在两侧方差非零时范围为 `[-1, 1]`；若 `Iᵤ = aT + b` 且 `a > 0`，理想评分为 1，`a < 0` 时为 -1，常量窗口则退化。只除以未去均值能量的归一化相关不能消除常数偏置。Lewis 将这个去均值版本称为 NCC，本仓库用 ZNCC 明确其公式；窗口相关的归一化分母使它不再是固定线性滤波输出，**不能直接继承白噪声匹配滤波的 SNR 最优性**。FFT 只是计算相关的一种手段，不会自动完成去均值、能量归一化或噪声白化。[cite:5](#source-5)[cite:16](#source-16)

<a id="ssda-history"></a>

**SSDA：Barnea／Silverman 1972。** D. I. Barnea 与 H. F. Silverman 的 *A Class of Algorithms for Fast Digital Image Registration* 发表于 *IEEE Transactions on Computers* C-21(2)、179–186 页，1972 年 2 月。这里把它列为逐次相似性检测算法（Sequential Similarity Detection Algorithm，SSDA）的早期代表文献，不扩写成所有快速模板匹配或所有提前退出策略的唯一首创。[cite:15](#source-15)

**方法演进线。** 不必为每个差候选都算完整块模板。固定一个候选位置后，按选定次序逐点比较模板与图像，累积误差或部分相关证据；每增加一些项就检查终止条件，足以判差便停止该候选并转向下一位置，把较完整的计算留给有希望的候选。误差阈值式描述与 Lewis 的随机部分相关描述说明了同一类“逐次分配计算”的思路，但终止保证取决于具体评分和界，不能仅凭 SSDA 名称判定。[cite:15](#source-15)[cite:5](#source-5)

- **问题**：空间域逐位置、逐像素完整求和会把大量时间花在明显不匹配区域。
- **改进**：逐次累积并提前淘汰，减少差候选实际访问的像素数；它改变的是候选评分的计算策略，不是 FFT，也不是自动的旋转／尺度搜索。
- **代价／边界**：收益取决于模板、搜索区、访问顺序、阈值及干扰样本；困难候选仍可能需要完整比较。统计早停可能漏掉正确峰，严格界剪枝则需满足下表的条件，二者都没有任意场景固定加速的保证。
- **工业用途**：可评估大搜索区、差候选较多的空间域模板比较；应在同一评分目标下比较访问像素量、尾延迟和漏检情况，不能把 SAD／SSD 与 ZNCC 的差异都归因于早停。
- **实现状态**：**未实现 SSDA，也未实现本节所述非负误差严格界剪枝**。[ncc_matching.py](../traditional/template_matching/ncc_matching.py) 与 [pose_ncc.py](../traditional/template_matching/pose_ncc.py) 计算完整评分图或局部评分图后再筛峰，没有对单候选逐像素累计并淘汰。
- **来源状态**：IEEE 已核验作者、年份、页码、DOI 和摘要；Lewis 扩展原文已读取其 SSDA 段落。1972 论文全文未取得，误差累积阈值机制另有 Brown 综述搜索摘录支持；dblp 条目被反爬验证阻挡，不计为核验成功。[cite:15](#source-15)[cite:5](#source-5)

**工程落地线：统计早停与非负误差严格界。** 下表的严格界是对固定目标函数的直接数学说明，不宣称是 1972 原文的逐项复现；也不把 Lewis 所评述的随机估计变体当作所有 SSDA 实现的共同保证。

| 机制 | 逐次计算与终止条件 | 保证与必要限制 |
|---|---|---|
| 统计／估计式早停 | Lewis 描述按随机顺序累积相关项，以部分计算作 Monte Carlo 估计；估计该位置不接近最大峰便提前结束 | 不是确定性界，不保证找到完整相关面的最大值；样本数和阈值需要设置，多个窄峰／重复纹理可能误导估计。[cite:5](#source-5) |
| 非负误差严格界 | 固定完整误差 `E_N = Σ eᵢ`，且每项 `eᵢ ≥ 0`；例如绝对差 SAD 或平方差 SSD。无放回访问各项时，部分和 `E_k ≤ E_N` | 若已有完整候选误差 `B`，`E_k > B` 即证明该候选不能优于当前最佳；仅需一个最优解且允许舍弃并列解时可用 `E_k ≥ B`。这是下界剪枝，不是统计猜测 |
| 固定误差阈值拒绝 | 接受规则为 `E_N ≤ τ` 时，`E_k > τ` 可立即拒绝该候选 | 只保证拒绝正确；未越界不等于最终通过，也不能只按“越界前检查了多少点”宣称获得精确最优排序 |

严格界要在**同一固定目标**下使用：像素次序可以改变，但完整支持、非负权重、归一化常数与均值必须已确定；如果用部分样本均值重定义残差，或用 `E_k/k` 外推完整均值，就不再能直接引用该部分和下界。保留所有并列最优解时不能用 `≥ B` 丢弃相等候选；浮点实现还需保守处理舍入和溢出。最大化原始相关／ZNCC 时，应推导剩余贡献的合法上界，不能照搬最小化 SAD／SSD 的非负部分和规则：去均值乘积可正可负，ZNCC 分母又涉及完整窗口统计。

[shape_based.match](../traditional/template_matching/shape_based.py) 的现有早停是在一行候选均已完整评分后，若全局最佳分数达到 `max(min_score, 0.98)` 就退出后续行；它不是 SSDA 的候选内部逐次误差淘汰，也不是证明剩余候选不可能更好的严格界。现有 NMS、候选数量截断和金字塔粗筛同样不能改称 SSDA。未来接入应分别标明“统计近似”与“精确剪枝”，用同一评分的完整搜索作对照；当前不报告 SSDA 或经典匹配滤波的仓库性能。

### 3.1 相位相关：1975

**方法演进线。** Kuglin 与 Hines 的 *The Phase Correlation Image Alignment Method* 是这里选用的 1975 年代表文献。针对主要为平移的两幅图，傅里叶位移定理将平移变成频域相位差；归一化互功率谱经逆变换得到位移峰。它对频谱幅值做归一化，而 ZNCC 对图像窗口的去均值能量做归一化，二者不能因都使用 FFT 就视为同一算法。[cite:1](#source-1)[cite:5](#source-5)

- **问题**：逐位置比较整幅或大 ROI 图像代价高，且幅值变化会干扰普通相关。
- **改进**：利用整体频域相位关系寻找平移峰，避免为每个位移显式计算全部像素差。
- **代价／边界**：周期边界导致模图像尺寸的位移歧义；边界不连续、低重叠、噪声及多个独立运动影响峰。相位归一化可能放大弱频率噪声；窗函数、补边、掩膜和亚像素 DFT 是需要另行说明的工程处理，不能全部归给 1975 原文。[cite:1](#source-1)
- **工业用途**：可评估相邻显微帧漂移、平台小范围位移、扫描拼接的平移初值；前提是有效重叠区域共享同一运动。
- **实现状态**：**未实现**；当前源码没有相位谱归一化或 `phaseCorrelate` 调用。`pose_ncc._score` 的 `fftconvolve` 不是相位相关。
- **来源状态**：年份获官方库参考文献及独立综述／论文引文支持，但本次未取得 1975 原文全文；不据二手引文认定首创，也不补造 DOI，详见 [cite:1](#source-1)。

**工程落地线。** 两图只差平移时可把它列为新增基线；出现旋转／尺度时，不能只改变阈值期待恢复。对数极坐标等扩展需要额外模型与独立验证，当前仓库的姿态枚举也不是该扩展。

### 3.2 Chamfer：1977

**方法演进线。** Barrow、Tenenbaum、Bolles、Wolf 的 *Parametric Correspondence and Chamfer Matching: Two New Techniques for Image Matching*（IJCAI 1977）讨论将预测轮廓与图像边缘对应。严格二值边缘重合对轻微错位过于敏感；先计算到最近图像边缘的距离场，再累计变换后模板边缘点的距离，可提供“离正确位置还有多远”的评分。[cite:2](#source-2)

- **问题**：亮度外观可能变化，二值曲线又难以像灰度块那样平滑比较。
- **改进**：把逐点最近边缘搜索转成距离场查表；边缘距离通常越小越好。原文的 chamfering 使用离散近似距离，不应直接称为精确欧氏距离实现。
- **代价／边界**：需要可靠边缘提取和变换搜索；背景杂边可给出低距离，重复平行线会混淆，多条模板边可能靠向同一条图像边。丢弃灰度还可能丢失辨别纹理；原文已经讨论这些局限。[cite:2](#source-2)
- **工业用途**：可评估背光零件轮廓、孔形、外形定位，以及灰度变化大而轮廓稳定的工件；需要检验边缘完整性和背景干扰。
- **实现状态**：**未实现 Chamfer**。[shape_based.py](../traditional/template_matching/shape_based.py) 选取强 Sobel 梯度点并加权累计非负方向内积，没有距离场或最近边缘距离，因此只能作为相关的形状路线对照。
- **来源状态**：已读取 IJCAI 官方原文 PDF；1977 与题名另有大学档案引文交叉核对，见 [cite:2](#source-2)。

**工程落地线。** 不要把 `shape_based.match(min_score=...)` 当作 Chamfer 距离阈值：其分数越高越好，且当前 `min_score` 只参与提前退出条件，低分仍可能返回三元组。该实现仅平移、整数位置，不是商业形状匹配系统的等价复现。

### 3.3 Lucas–Kanade：1981

**方法演进线。** Lucas 与 Kanade 的 *An Iterative Image Registration Technique with an Application to Stereo Vision*（IJCAI 1981）利用空间强度梯度，以局部线性化和迭代代替大量候选比较。可用加权平方残差的局部最小化理解其基本思想：当前变换附近做小参数更新，逐步对齐。后来的金字塔跟踪、逆组合更新和鲁棒损失是扩展，不应一并标成 1981 版本的能力。[cite:3](#source-3)

- **问题**：穷举搜索昂贵，而连续帧或已有粗定位时，真实变换可能已接近初值。
- **改进**：使用梯度提供更新方向，在局部求解小规模参数增量；可配合选定变换模型。
- **代价／边界**：需要亮度一致性近似、足够梯度和合理初值；单边缘有孔径问题，平坦窗口缺少约束，大位移、遮挡及局部极值会导致失锁。金字塔扩大实用捕获范围，但不保证全局最优。[cite:3](#source-3)[cite:7](#source-7)
- **工业用途**：可评估连续帧标志点跟踪、已粗定位后的局部对齐、稳定工况下小运动估计；必须有失锁重定位策略。
- **实现状态**：**未实现匹配／跟踪 LK**。NCC 的三点评分峰拟合不是 LK，仓库单帧姿态搜索也不是光流跟踪。
- **来源状态**：CMU 官方条目核验 IJCAI 1981、674–679 页；同年 DARPA Workshop 版本为 121–130 页，不能混拼会名和页码。[cite:3](#source-3)

**工程落地线。** 对 `match_pose_ncc` 给出的初值再接 LK 是潜在方案，但还须明确采样方向、支持区域、亮度模型与迭代失败协议。现有 `subpixel=True` 只细化平移峰位，绝不隐含连续优化角度和尺度。

### 3.4 FFT-NCC：1995 年代表论文与仓库延伸

**方法演进线。** Lewis 在 Vision Interface 1995 的正式论文是 *Fast Template Matching*，作者提供的扩展修订版题为 *Fast Normalized Cross-Correlation*。原文把变换域相关分子与图像和／平方和的积分表结合，避免局部归一化抵消相关加速收益。这里以 1995 标记这一代表方案，而不是宣称 NCC 或傅里叶相关于该年诞生。[cite:5](#source-5)

- **问题**：未归一化相关受局部亮度能量影响；逐窗口计算均值和能量又很昂贵。
- **改进**：FFT 计算相关分子，预计算矩形窗口局部统计；对非退化窗口，去均值归一化评分可消除正增益和常数偏置的影响。
- **代价／边界**：FFT 有变换、补边及内存开销，小模板或小搜索区未必更快。旋转、尺度、透视和遮挡不会因 FFT 而消失；非均匀照明、饱和及反相不受上述正增益结论保护。[cite:5](#source-5)
- **工业用途**：可评估印刷标记、固定视角纹理工件、贴装前粗定位；有限旋转／尺度可由显式姿态网格扩展，代价随搜索组合增加。
- **实现状态**：**已实现相关路线，非逐项论文复现**。[ncc_matching.py](../traditional/template_matching/ncc_matching.py) 用 FFT 分子和积分图分母；[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 为处理掩膜，分子、局部和、平方和都使用 FFT 卷积，并增加姿态网格、支持处理、候选搜索及拒识。
- **来源状态**：已读作者托管的 1995 原文及扩展网页；原文页首明确年份、正式题名和扩展版关系，不能将两个题名不加说明地合并。[cite:5](#source-5)

**工程落地线。** 仓库采用的是去均值版本 ZNCC；文献对 NCC 命名有差异，应看公式。`pose_ncc` 默认仅平移，必须显式设置角度／尺度范围；空间金字塔不是姿态自动层级优化。默认 `exhaustive_verification=True` 保留每个姿态的原图完整评分复核，不能宣称“使用金字塔所以已加速”。详见 [旧专题](pose_ncc.md) 的第二、三章。

### 3.5 ECC：2008

**方法演进线。** Evangelidis 与 Psarakis 的 *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*（IEEE TPAMI 2008）选择归一化相关目标，并推导前向加法及逆组合迭代方案。相比直接用原始灰度 SSD，它面向亮度偏置、对比度变化下的参数配准；相比姿态网格搜索，它追求连续参数的局部更新。[cite:7](#source-7)

- **问题**：亮度恒常假设不足，纯网格高精度搜索又可能代价高。
- **改进**：最大化变换后图像与参考图像的去均值归一化相关，通过可求解的局部近似迭代参数。
- **代价／边界**：仍是局部优化，需要可观测纹理和良好初始化；不保证抗任意阴影、反光、遮挡或非刚性形变。变换自由度越多越需约束，迭代不收敛必须作为失败处理。[cite:7](#source-7)[cite:13](#source-13)
- **工业用途**：可评估粗定位后的印刷对齐、检测差分前的小幅配准；不应让过度自由的形变把真实缺陷“配准掉”。
- **实现状态**：**未实现**，未见 `findTransformECC` 调用或 ECC 参数更新。现有 ZNCC 评分与 ECC 有归一化相关层面的联系，但三点抛物线峰拟合不是 ECC。
- **来源状态**：已读取 HAL 托管作者稿，封面期刊引文为 2008、30(10)、1858–1865；2013 是 HAL 入库日期，不是论文发表年份。[cite:7](#source-7)

**工程落地线。** 若后续接入 OpenCV ECC，需固定变换方向、模型、终止条件、初值与有效 ROI，并对不收敛和非有限矩阵拒识；不能只接收一个较高相关系数。现代 OpenCV 的 API 能力不自动等同于本仓库或 2008 原文版本。[cite:13](#source-13)

## 4. 局部特征、鲁棒几何与学习匹配分支

### 4.1 RANSAC：1981，横跨匹配路线的几何估计器

**方法演进线。** Fischler 与 Bolles 的 *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*（Communications of the ACM 1981）处理数据中含有大量粗差的问题。反复从小样本拟合候选模型，以与模型一致的数据形成共识，避免直接对全部点做最小二乘而被外点拖偏。[cite:4](#source-4)

- **问题**：即便描述子相似，点对仍可能错误，全部一起拟合会破坏几何变换。
- **改进**：将采样、模型估计与残差共识检验分开；适用于匹配之后的几何验证，而非图像特征检测。
- **代价／边界**：结果受内点比例、最小样本规模、退化样本、残差阈值和迭代预算影响。对于独立抽样的理想模型，目标成功概率为 p、内点率为 w、每次抽 s 点时，可用 `N >= log(1-p) / log(1-w^s)` 估计次数；低内点率或大样本规模会增加成本，实际相关数据和退化情况不受此公式保证。[cite:4](#source-4)
- **工业用途**：可评估平面标签／板件单应、拼接几何校验、机器人视觉点对应的外点剔除；先选符合场景的模型，不能为三维视差强行拟合单个平面 H。
- **实现状态**：**未实现**；[learned_matching.py](../dl/learned_matching.py) 的说明文字提及 `findHomography(RANSAC)`，但函数直接抛出 `NotImplementedError`，不存在运行中的几何估计。
- **来源状态**：ACM 官方页面核验 1981 年 6 月、24(6)、381–395 及 DOI；本次读取的是书目与摘要，不宣称逐页复核原文推导。[cite:4](#source-4)

**工程落地线。** 内点数量、内点率、残差和空间覆盖率应共同检查。重复纹理可能支持一个一致但错误的变换，高内点率也不是目标存在概率；应加 ROI、尺寸、方向、投影轮廓及机械先验。NCC 的峰分差解决空间候选竞争，不是点对共识估计。

### 4.2 SIFT：1999 会议版与 2004 期刊版

**方法演进线。** Lowe 的 *Object Recognition from Local Scale-Invariant Features*（ICCV 1999）与 *Distinctive Image Features from Scale-Invariant Keypoints*（IJCV 2004）是两个相关但不应混淆的条目。尺度空间检测、主方向与局部梯度描述，使分散的局部区域在旋转、尺度变化和部分遮挡下仍可能建立对应；2004 文献给出系统阐述。[cite:6](#source-6)

- **问题**：整块模板面对大尺度／方向变化或部分遮挡容易整体失配。
- **改进**：用尺度与方向归一化的局部表示，把整块一致性要求转为多个可见局部对应的集合。
- **代价／边界**：尺度空间、多维描述子和匹配有计算与存储成本；低纹理、重复纹理、显著透视与视角变化仍困难。尺度／旋转适应不能扩大为完整仿射或透视不变。
- **工业用途**：可评估有纹理平面工件、标签、包装和离线巡检图像的跨视角粗配准；需足够分散且可重复的特征。
- **实现状态**：**未实现**；未见 SIFT 特征提取与匹配流水线。现有 HOG + SVM 候选重排序并不是 SIFT。
- **来源状态**：作者 UBC 页面同时列出 1999、2004 原论文链接与书目信息；大学获奖记录与课程资料提供交叉年份依据。[cite:6](#source-6)

**工程落地线。** 先评估特征覆盖、最近邻歧义和几何残差，再比较检测阈值与特征数量；点很多但集中在一小块区域，并不能保证模板四角映射可靠。几何输出还要统一为“模板到图像”的方向。

### 4.3 ORB：2011

**方法演进线。** Rublee、Rabaud、Konolige、Bradski 的 *ORB: An Efficient Alternative to SIFT or SURF*（ICCV 2011）面向局部特征的开销问题，将带方向的 FAST 关键点与经旋转处理、选择的 BRIEF 二值测试结合。二值描述子可用位运算与 Hamming 距离匹配，形成与浮点描述子不同的资源取舍。[cite:8](#source-8)

- **问题**：局部特征检测、描述和比较的成本影响资源受限设备。
- **改进**：采用紧凑二值表示及方向补偿，降低描述与匹配开销；可配合图像金字塔处理尺度变化。
- **代价／边界**：二值强度比较仍受模糊、弱对比、重复结构和大视角变化影响；尺度覆盖取决于金字塔配置，不应视为任意尺度不变。原论文的速度对比不能直接变成本仓库倍数。
- **工业用途**：可评估嵌入式或 CPU 预算紧的粗定位、移动巡检、视觉里程计前端；需在真实工件上与 SIFT／模板基线比较。
- **实现状态**：**未实现**，未见 `ORB_create` 或二值描述子匹配链路。
- **来源状态**：出版记录核验 ICCV 2011、2564–2571、DOI `10.1109/ICCV.2011.6126544`；语义学术索引及独立论文引文同年，见 [cite:8](#source-8)。

**工程落地线。** 特征预算既影响时间也影响几何覆盖；不能仅按总点数优化。上线前仍需要固定实现版本、距离类型、过滤策略、模型残差与失败输出。

### 4.4 SuperPoint：2018，预印本 2017

**方法演进线。** DeTone、Malisiewicz、Rabinovich 的 *SuperPoint: Self-Supervised Interest Point Detection and Description* 发表于 CVPR Workshops 2018，arXiv v1 为 2017-12-20。共享卷积编码器同时输出兴趣点与描述子，Homographic Adaptation 用多单应视图提高检测可重复性并辅助从合成到真实图像的适配。[cite:9](#source-9)

- **问题**：手工检测与描述组合未必能在目标数据分布上同时获得稳定重复点和可区分描述子。
- **改进**：学习检测和描述，并借助几何变换形成训练监督；“自监督”不代表无需训练数据或域适配。
- **代价／边界**：引入模型权重、推理框架、输入尺度／归一化契约与训练域偏差；兴趣点输出仍受纹理和检测阈值限制。
- **工业用途**：可评估有自然纹理的跨帧／跨视角定位前端；不能直接保证镜面金属、重复网格或新成像模态的效果。
- **实现状态**：**骨架，算法未接入**；[learned_matching.py](../dl/learned_matching.py) 仅描述路线，没有加载权重、网络前向或特征结果。
- **来源状态**：CVF 明确是 **Workshops** 而非 CVPR 主会；arXiv 提交历史核验 2017 预印本，两种年份应并列解释。[cite:9](#source-9)

**工程落地线。** SuperPoint 是检测／描述前端，不是 SuperGlue 的别名；后续仍需描述子匹配或学习匹配器及几何估计。骨架中“可直接得到亚像素级配准”的文字不能当作运行结果或定位准确度保证。

### 4.5 SuperGlue：2020，预印本 2019

**方法演进线。** Sarlin、DeTone、Malisiewicz、Rabinovich 的 *SuperGlue: Learning Feature Matching with Graph Neural Networks*（CVPR 2020；arXiv v1 2019-11-26）在两组局部特征之间建立对应。注意力聚合图内／图间上下文，结合可微最优传输求匹配分配，并允许点不被匹配。[cite:10](#source-10)

- **问题**：逐描述子最近邻主要利用局部相似性，面对歧义和不可匹配点时缺少整体上下文。
- **改进**：联合考虑点的位置、描述和上下文，在匹配与拒配之间做学习式分配。
- **代价／边界**：需要与特征前端兼容的模型；注意力和分配计算随点数增加而变重。检测器没有提供的真实点不能被稀疏匹配器凭空恢复；训练域偏差仍存在。
- **工业用途**：可评估带局部纹理、遮挡或视角差异的离线配准与视觉定位；资源预算允许时与简单匹配加几何验证比较。
- **实现状态**：**未实现**；当前学习匹配骨架没有 SuperGlue 模型或推理路径，其他学习模型的名字不代表已具备该模型。
- **来源状态**：CVF 主会条目为 2020、4938–4947；arXiv 初次提交为 2019，见 [cite:10](#source-10)。

**工程落地线。** 网络拒配不是最终安全拒识；输出点对仍需选定相似、仿射、单应或其他几何模型。不能把匹配置信分数、RANSAC 内点率和 ZNCC 分数使用同一阈值。

### 4.6 LoFTR：2021

**方法演进线。** Sun、Shen、Wang、Bao、Zhou 的 *LoFTR: Detector-Free Local Feature Matching with Transformers*（CVPR 2021）绕开“先独立检测关键点再匹配”的瓶颈。在粗特征图上用自注意力和交叉注意力形成两图条件化表示，再细化选出的良好对应。[cite:11](#source-11)

- **问题**：低纹理区域难以独立产生足够可重复关键点，稀疏匹配前端已丢失潜在对应。
- **改进**：利用更大范围的上下文，先在粗层建立对应，再在细层精化；弱纹理条件下可作为检测器路线的替代候选。
- **代价／边界**：计算与显存受输入分辨率、特征图与匹配处理影响；低纹理不等于无信息，完全平坦、重复纹理或低重叠仍会歧义。粗到细对应不等于每个像素都有可信流，也不意味着任意非刚性配准完成。
- **工业用途**：可评估特征点不足但仍有上下文结构的表面图像、弱纹理平面粗配准；需按成像域和权重分别验收。
- **实现状态**：**骨架仅列研究路线，算法未接入**；没有特征提取、Transformer、粗细匹配或权重加载实现。
- **来源状态**：CVF 官方条目核验 2021、8922–8931；本次主要依赖该一手论文页面，不声称已完成三方独立性能复核。[cite:11](#source-11)

**工程落地线。** 若未来为了节拍缩图，必须将对应点正确映回原图并重新评估小结构定位误差。匹配数增加不等于均匀几何覆盖，更不能把论文数据集效果直接转换为工件精度。

### 4.7 LightGlue：2023

**方法演进线。** Lindenberger、Sarlin、Pollefeys 的 *LightGlue: Local Feature Matching at Light Speed*（ICCV 2023）重新设计稀疏学习匹配中的若干计算环节，并根据图像对难度调整推理工作量。自适应深度与点筛除等设计旨在让容易匹配的图像对少做计算，而非固定执行同等工作。[cite:12](#source-12)

- **问题**：较强的稀疏学习匹配器也可能有较高计算、内存及训练成本。
- **改进**：优化局部特征匹配结构，利用早停和点筛除等自适应机制降低不必要计算。
- **代价／边界**：仍依赖可靠特征前端、兼容权重和推理环境；困难样本工作量较大，平均加速不构成最坏时间上界。精度／延迟取决于点数、阈值和设备配置。
- **工业用途**：可评估节拍敏感而又需要学习式匹配的定位／配准任务；应关注困难帧延迟和拒识率，而不只展示容易样本。
- **实现状态**：**骨架，算法未接入**；[learned_matching.py](../dl/learned_matching.py) 提到 SuperPoint + LightGlue，但实际函数直接报未实现。
- **来源状态**：CVF 官方条目核验 ICCV 2023、17627–17638；不把它称为核验日最新方案，不将论文相对改进写成本仓库实测。[cite:12](#source-12)

**工程落地线。** 与 SuperGlue 一样，LightGlue 是稀疏局部特征匹配器，不取代 SuperPoint 等提取器。部署检查还应覆盖代码与权重许可证、模型版本、预处理、动态计算策略和超时回退；公开下载不自动等于允许所有商用方式。

## 5. 从旧专题到源码的能力对照

### 5.1 已有实现与不可混称项

| 入口 | 实际算法／输出 | 关键边界 | 旧专题入口 |
|---|---|---|---|
| [match_template_ncc](../traditional/template_matching/ncc_matching.py) | FFT 分子 + 积分图局部统计，输出平移评分图 | 无姿态网格、无掩膜；退化处理不等同于新接口 | [ZNCC 原理与边界](pose_ncc.md#第二章zncc掩膜与数值边界) |
| [find_best_matches](../traditional/template_matching/ncc_matching.py) | 评分阈值、空间 NMS、可选平移抛物线精化，返回列表 | 无新接口的 `reason` 与分差协议 | [返回协议](pose_ncc.md#返回协议与坐标) |
| [match_pose_ncc](../traditional/template_matching/pose_ncc.py) | 有效支持 ZNCC、角度／统一尺度网格、空间金字塔、复核与拒识 | 非 LK／ECC；无透视、非刚性、出画目标支持 | [姿态搜索](pose_ncc.md#第三章姿态网格与空间金字塔) |
| [shape_based.match](../traditional/template_matching/shape_based.py) | 强梯度点的方向内积；平移整数位置三元组 | 非 Chamfer；低于 `min_score` 不保证拒识 | [旧接口定位说明](pose_ncc.md#定位) |
| [verify_candidates](../ml/hog_svm_matching.py) | HOG + SVM，增加 `svm_score` 并排序 | 每次调用生成合成样本并训练；不修正位置，也未用分类阈值过滤候选 | [ML 路线边界](../ml/README.md) |
| [learned_matching.match](../dl/learned_matching.py) | 只有函数签名和方案文字 | 直接 `NotImplementedError`；说明中的元组与签名 `dict` 尚无可执行返回协议 | [DL 完成度](../dl/README.md) |

本次实现状态检查读取上述函数体，并搜索 `calcOpticalFlow`、`findHomography`、`estimateAffine`、`phaseCorrelate`、`distanceTransform`、`SIFT_create`、`ORB_create`、`findTransformECC` 与学习模型名。出现于说明文字的调用不算实现；对第三方安装目录不作“仓库算法已实现”的认定。

本次前史增补另检查仓库 Python 源码中的 `SSDA`、`matched_filter`、匹配滤波与逐次相似性检测关键词，并核对三个模板匹配模块的求和和退出位置：未见独立经典匹配滤波检测／白化链路、SSDA 或 SAD／SSD 部分和严格界实现；已有相关分子和其他 `break` 不能作为这些算法已落地的证据。

### 5.2 pose_ncc 的工程约束

1. **输入与支持**：二维、非空、有限数值灰度；掩膜同形且仅 0/1，有效模板至少四像素且不能为常量。旋转时排除补边及掩膜混合边界；掩膜并不允许完整包围画布越出图像。
2. **数值与成本**：`_score` 使用 FFT 局部统计；方差不可靠时，从原始像素按批做局部直接评分。该兜底不能恢复量化丢失的纹理，还可能使低对比／极端量纲输入显著变慢。
3. **搜索与精化**：角度和尺度仍是离散网格；抛物线只细化平移，精化后不重新计算 `score`。默认全分辨率复核避免粗层截断漏掉某姿态的最佳原图峰，但每姿态候选上限及跨姿态 NMS 仍限制召回。
4. **拒识范围**：`ambiguous` 比较不同空间候选，不识别同中心旋转对称歧义。`second_score=None` 时分差为空，不触发分差拒识；“通过”不代表已穷尽所有竞争假设。
5. **多目标**：必须显式 `max_count>1, min_score_gap=0`，且候选容量足够；这是关闭唯一性拒识，不是关闭评分阈值。半径过大会合并邻近真实目标。
6. **坐标协议**：`transform` 将原模板像素中心映到图像；`x/y` 是原模板原点映射，`center_x/y` 是模板中心，`bbox_x/y` 是变换包围画布左上角。ROI 裁剪后要组合偏移矩阵；它们不是可互换的定位真值。

以上来自 [pose_ncc 源码](../traditional/template_matching/pose_ncc.py)，详细推导与边界见 [坐标与变换](pose_ncc.md#第一章坐标与变换)、[平移亚像素精化](pose_ncc.md#第四章平移亚像素精化)、[拒识与多目标](pose_ncc.md#第五章拒识与多目标)。相关测试入口为 [test_pose_ncc.py](../tests/test_pose_ncc.py)，测试容差不是工业准确度保证。

## 6. 工程选型、排障与验收

### 6.1 先选模型与证据，再选算法

| 场景／约束 | 优先评估路线 | 应先确认 | 本仓库可以直接做什么 |
|---|---|---|---|
| 固定视角、已知模板、纹理稳定 | ZNCC；有限旋转／尺度用姿态网格 | ROI、姿态范围、负样本和重复纹理 | 运行现有传统基线与拒识回归 |
| 整幅／大 ROI 主要平移、重叠充分 | 相位相关与 ZNCC 对照 | 周期边界、位移符号、多运动干扰 | 只有 ZNCC；相位相关需新增实现 |
| 外形稳定、灰度变化大 | Chamfer 或梯度方向形状路线 | 轮廓提取、背景杂边、方向极性 | 只有简化梯度方向平移基线 |
| 跨视角、局部遮挡且有分散纹理 | SIFT／ORB + 几何验证 | 平面假设、内点覆盖、外点和重复图案 | 无对应端到端实现 |
| 粗定位已近、要连续参数对齐 | LK／ECC | 初值、支持、光度模型、可观测性 | 只有平移评分峰精化，不能等价替代 |
| 手工点匹配不足且有学习推理预算 | SuperPoint + SuperGlue／LightGlue | 数据域、权重兼容、许可证与尾延迟 | 学习入口仅骨架 |
| 独立关键点不足但有上下文结构 | LoFTR 与稀疏路线对照 | 输入分辨率、显存、几何覆盖和低重叠 | 路线文字存在，算法未实现 |
| 明显三维视差／非刚性变化 | 重新定义多视几何或形变模型 | 单一 H 是否物理有效、缺陷是否被形变吞掉 | 现有姿态 ZNCC 不支持 |

这些是基于各方法机制和限制的工程建议，不是统一基准得出的胜负。照明、失焦、畸变及运动模糊先参阅 [预处理增强](preprocessing.md)、[标定坐标](calibration_coordinates.md)；需要从区域构造 ROI／掩膜时参阅 [分割区域](segmentation_regions.md)。改善成像往往比放宽阈值更能保住几何可观测性，但仍须用现场数据验证。

### 6.2 按症状排障

| 症状 | 优先假设与检查证据 | 针对动作 | 不应采取的捷径 |
|---|---|---|---|
| `ValueError`／没有有效姿态 | 数组维度、NaN/Inf、常量、掩膜、变换画布尺寸 | 先修输入与合法姿态范围，记录 `searched_poses` | 把异常吞成“未找到”或先降阈值 |
| 目标可见但 ZNCC 低分 | 网格间姿态、反相、饱和、模糊、遮挡、模板老化 | 固定图像比较粗细网格和受控成像条件 | 用单张正例把 `min_score` 降到刚通过 |
| 高分但 `ambiguous` | 重复纹理、真实多目标、旁瓣与 NMS 半径 | 看候选空间关系；单目标改善 ROI／模板，多目标才关闭分差 | 把所有歧义都当作误报并禁用拒识 |
| 同中心角度不稳定 | 旋转对称、角度网格与纹理方向信息不足 | 增加非对称可见结构或声明角度不可观测 | 认为已有 `score_gap` 就覆盖姿态歧义 |
| 多层近似模式漏检／错误放行唯一性 | 粗层高频损失、奇数相位、候选截断 | 恢复原图复核，以单层作对照 | 只展示成功样本的近似模式速度 |
| 形状方法低分仍给位置 | `shape_based.match` 无低分拒识返回协议 | 调用侧核对分数，单独设计拒识；注明当前边界 | 套用 `match_pose_ncc` 的返回保证 |
| LK／ECC 漂移、不收敛 | 初值远、低纹理、支持变化、模型错配 | 检查局部残差、梯度可观测性及变换合理性，失败回到粗定位 | 无限制增加迭代或输出最后一个矩阵 |
| 特征点很多但 H 错 | 重复图案、点聚集、退化构型、三维视差 | 同时查看内点分布、投影四角、残差与物理范围 | 只凭点数或内点率放行 |
| 相位相关位移跳到另一侧 | 周期模糊、边界突变、低重叠 | 检查位移方向及回绕，限定物理范围并验证重叠 | 直接把峰下标作为绝对设备位移 |
| 学习匹配线上失效 | 域变化、预处理／缩放回填不一致、权重不兼容 | 保存输入、版本、点对和残差，按工况分组验收 | 将公开权重或论文结果当成现场认证 |
| 坐标固定偏移／亚像素抖动 | 原点与中心、ROI、半像素约定、边界峰、粗姿态 | 用同一 H 变换同一参考点，比较精化开关 | 把小数位数当定位准确度 |
| 时延偶发突增 | 网格数量、FFT 直接评分兜底、特征数、学习自适应计算 | 记录分阶段耗时及困难样本尾延迟，先限制合理搜索范围 | 用平均时间替代设备截止时间验收 |

尚未实现的方法行是后续接入时的排查方案，不表示已采集其运行时证据。仓库现有排障细节及测试名称见 [现场排障与验证方法](pose_ncc.md#第六章现场排障与验证方法)；半像素和插值陷阱见 [研究笔记](research_notes.md)。

### 6.3 最小验收协议

- **数据**：分开现场正例、无目标、近似干扰、对称／重复结构、遮挡、离焦与跨批次样本；训练／调参／验收按工件或批次隔离，避免相邻帧泄漏。
- **几何**：声明模板到图像方向、ROI 偏移、像素中心定义、允许变换、真值来源；像素指标和毫米指标分开，后者需要标定及其不确定度。
- **检测与拒识**：同时报告漏检、误检、歧义拒识、成功率和失败原因；多目标用一对一匹配，不只统计成功样本的定位误差。
- **精度**：位置、圆周角差、尺度误差分别统计；同图重复运算不等于重新采集的重复性。输出亚像素坐标、离散步长或测试容差都不是现场精度证书。
- **资源**：说明图像与模板尺寸、ROI、姿态／特征预算、权重和软件版本、硬件、预热、端到端与分阶段耗时；报告中位数及尾延迟，不混用不同设备的论文数据。
- **回退**：预先约定不收敛、低重叠、超时、歧义和模型越界时的拒识或重定位；设备动作需独立安全策略，不能只凭匹配分数驱动。

既有 [benchmark_pose_ncc.py](../tests/benchmark_pose_ncc.py) 对每场景使用三个固定种子，各预热一次再计时三次；检测与误差取每种子的最后结果，计时重复不算独立采集。`off_grid` 故意保留过粗姿态的失败例，`ambiguous` 按拒识任务评价；它不是十二路线统一评测，也不覆盖现场尾延迟。历史结果继续以 [experiments.md](experiments.md) 为准，本章不新增性能数字。

## 7. 来源与核验记录

### 7.1 核验分级与差异处理

**[CONFIRMED]** 表示书目或版本差异得到原文／官方记录直接支持；并不自动表示已有三个独立来源或算法性能已复现。**[MAJORITY]** 表示多个二手引文一致但仍存在原文获取缺口。**[SINGLE-SOURCE]** 表示本次主要依赖一个来源体系，不冒充独立交叉验证。作者稿、同文预印本、会议网页和作者仓库不能凑成三份独立实验。

| 项目 | 采用年份与来源核验结论 | 获取深度／证据限制 |
|---|---|---|
| 匹配滤波理论源头 | 1943-06-25 报告／1963 年 7 月重印；[CONFIRMED]，IEEE 前言与书目、MIT 历史文章一致 | 已读前言、摘要与书目，未取得 North 正文全文；现代 SNR 推导不冒充原文，不认定所有匹配的唯一起源 |
| SSDA | 1972；[CONFIRMED] IEEE 书目；Lewis 扩展原文与 Brown 综述摘录支持方法关联 | 未取得 1972 全文；dblp 反爬阻挡，未完成该站交叉核验；统计早停说明依赖 Lewis，严格界为显式条件下的数学推导 |
| 相位相关 | 1975；[MAJORITY]，官方库文献表、Brown 综述搜索摘录及独立应用论文引文一致 | 原文全文未取得；Brown 镜像全文抓取未返回正文，不能称已阅读全文 |
| Chamfer | 1977；[CONFIRMED]，IJCAI 原文与大学档案引文相符 | 已读原文方法与讨论；历史扫描含 OCR 拼写误差，不用 OCR 细节作精确数值结论 |
| LK | 1981；[CONFIRMED]，CMU 明列 IJCAI 与同年 DARPA 两版 | 已核验官方书目／摘要和扩展项目页；不用 2004 综述年份替代原始年份 |
| RANSAC | 1981；[CONFIRMED]，ACM 出版记录与独立课程引文相符 | 本次读取书目和摘要；没有逐页验证原文公式或复现实验 |
| FFT-NCC | 1995；[CONFIRMED]，作者原文页首与独立后续论文引文一致 | 原文题名 Fast Template Matching；Fast Normalized Cross-Correlation 为扩展修订版，不为扩展网页另猜发表年 |
| SIFT | 1999／2004；[CONFIRMED]，作者两篇原文书目、大学记录与课程资料一致 | 区分会议提出与期刊系统阐述，不写成两个互斥的“发明年” |
| ECC | 2008；[CONFIRMED]，作者稿期刊引文与独立论文文献表一致 | HAL 入库 2013；其他早期引用中的 2007 不替代本章指定的 2008 期刊版 |
| ORB | 2011；[CONFIRMED]，出版记录、学术索引及独立引文同年 | 已核验书目与摘要；不转述论文倍数为本仓库实测 |
| SuperPoint | 2018／预印本 2017；[CONFIRMED] 版本日期，[SINGLE-SOURCE] 原论文体系 | CVF Workshops 与 arXiv 历史一致；不是主会论文，亦非独立性能交叉验证 |
| SuperGlue | 2020／预印本 2019；[CONFIRMED] 版本日期，[SINGLE-SOURCE] 原论文体系 | CVF 主会与 arXiv 历史；现代 GPU 的论文描述不是本仓库节拍证据 |
| LoFTR | 2021；[SINGLE-SOURCE]，CVF 原论文页面书目明确 | 已读摘要与论文入口；未进行独立数据集复现 |
| LightGlue | 2023；[SINGLE-SOURCE]，CVF 原论文页面书目明确 | 已读摘要与论文入口；未作核验日全领域“最新”检索或性能排名 |

初次尝试的 CMU 不带 `-ijcai` 后缀旧地址及 UBC 论文目录页返回 404，已改用有效论文条目／作者专题页。检索发现的非正规论文镜像不作为最终证据。下列链接用于可追溯阅读；联网成功不保证未来永久可访问，现代文档也不证明本机依赖版本或接口可用性。

### 7.2 原论文与权威入口

<a id="source-1"></a>

**[cite:1] 相位相关。** C. D. Kuglin, D. C. Hines, *The Phase Correlation Image Alignment Method*, Proceedings of the IEEE International Conference on Cybernetics and Society, 1975, pp. 163–165。已读 [scikit-image 官方 registration 文档与参考文献](https://scikit-image.org/docs/0.24.x/api/skimage.registration.html)，作为原论文引文及现代算法边界依据。交叉年份证据为 Brown, *A Survey of Image Registration Techniques* 的 [大学镜像搜索摘录](https://www.geo.uzh.ch/microsite/rsl-documents/research/SARlab/GMTILiterature/PDF/Bro92.pdf) 及 [相位相关稳像应用论文的参考文献页](http://www.chineseoptics.net.cn/en/article/id/8528)；后两者本次仅取得搜索摘录。原文全文未取得，原会议城市的二手写法不一致，本章不采用城市信息；1970 FFT 配准、1975 相位相关和现代亚像素算法不可混年。

<a id="source-2"></a>

**[cite:2] Chamfer。** H. G. Barrow, J. M. Tenenbaum, R. C. Bolles, H. C. Wolf, *Parametric Correspondence and Chamfer Matching: Two New Techniques for Image Matching*, IJCAI 1977, pp. 659–663。[IJCAI 官方原文 PDF](https://ijcai.org/Past%20Proceedings/IJCAI-77-VOL2/PDF/024.pdf) 已读取；[中京大学档案引文](http://isl.sist.chukyo-u.ac.jp/Archives/ChamferMatching.html) 搜索摘录交叉确认年份与页码。原文讨论边缘距离场、离散距离近似、杂乱边缘与平行线歧义。

<a id="source-3"></a>

**[cite:3] LK。** B. D. Lucas, T. Kanade, *An Iterative Image Registration Technique with an Application to Stereo Vision*, IJCAI 1981, Vol. 2, pp. 674–679。[CMU 原论文条目](https://www.ri.cmu.edu/publications/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai/) 已读取并包含原文入口；[CMU Lucas–Kanade 20 Years On](https://www.ri.cmu.edu/project/lucas-kanade-20-years-on/) 已读取，用于区分后续扩展与 1981 两版。独立 [课程引文](https://cse.taylor.edu/~jdenning/classes/cos351/slides/15_opticflow.html) 搜索摘录同样列 IJCAI 1981。

<a id="source-4"></a>

**[cite:4] RANSAC。** M. A. Fischler, R. C. Bolles, *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*, Communications of the ACM, 24(6), 1981, pp. 381–395。[ACM 官方出版页](https://cacm.acm.org/research/random-sample-consensus/) 已读取，DOI 为 `10.1145/358669.358692`；[ENS 图像对应课程资料](https://www.di.ens.fr/willow/teaching/recvis11/slides/lecture03_part1_image_alignment.pdf) 搜索摘录交叉年份。次数公式是独立抽样模型下的通用解释，不是实测成功率。

<a id="source-5"></a>

**[cite:5] FFT-NCC 代表方案。** J. P. Lewis, *Fast Template Matching*, Vision Interface 1995, pp. 120–123。[作者托管原文 PDF](https://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf) 和 [扩展修订版 Fast Normalized Cross-Correlation](https://scribblethink.org/Work/nvisionInterface/) 均已读取；[后续独立论文的出版页引文](https://ascelibrary.org/doi/10.1061/(ASCE)CO.1943-7862.0002256) 搜索摘录同样列 1995 与正式题名。只采用可跨版本确认的机制，不移用历史硬件上的耗时数字。

<a id="source-6"></a>

**[cite:6] SIFT。** D. G. Lowe, *Object Recognition from Local Scale-Invariant Features*, ICCV 1999, pp. 1150–1157；*Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 60(2), 2004, pp. 91–110。[作者 SIFT 专题及两篇原文入口](https://www.cs.ubc.ca/~lowe/keypoints/) 已读取；[UBC 1999 论文获奖记录](https://www.cs.ubc.ca/award/2011/11/iccv-test-time-award) 与 [奥克兰大学 SIFT 课程资料](https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/COMPSCI%20773%20SIFT.pdf) 的搜索摘录交叉确认。本章不把作者旧网页上的“up-to-date”用语当作当前综述结论。

<a id="source-7"></a>

**[cite:7] ECC。** G. D. Evangelidis, E. Z. Psarakis, *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*, IEEE TPAMI 30(10), 2008, pp. 1858–1865。[HAL 托管作者稿](https://hal.inria.fr/hal-00864385/file/PAMI_2008_preprint.pdf) 已读取，DOI `10.1109/TPAMI.2008.113`；[独立后续论文文献表](https://pdfs.semanticscholar.org/b375/63a750eef10830a03aeceb9c9d32e836b7b9.pdf) 搜索摘录同样标 2008。作者稿明确局部迭代、光度模型与直接搜索的不同代价。

<a id="source-8"></a>

**[cite:8] ORB。** E. Rublee, V. Rabaud, K. Konolige, G. Bradski, *ORB: An Efficient Alternative to SIFT or SURF*, ICCV 2011, pp. 2564–2571。[出版记录与摘要](https://dl.acm.org/doi/10.1109/ICCV.2011.6126544) 已读取；[Semantic Scholar 书目](https://www.semanticscholar.org/paper/ORB:-An-efficient-alternative-to-SIFT-or-SURF-Rublee-Rabaud/976e29fe6c9baeb39732bca0e35f66f84d5bdd90) 及 [独立机器人论文引文](https://pure.hw.ac.uk/ws/files/50621350/IROS_Hazel_3.pdf) 搜索摘录交叉确认年份。主页面 DOI 为 `10.1109/ICCV.2011.6126544`。

<a id="source-9"></a>

**[cite:9] SuperPoint。** D. DeTone, T. Malisiewicz, A. Rabinovich, *SuperPoint: Self-Supervised Interest Point Detection and Description*, CVPR Workshops 2018, pp. 224–236。[CVF 官方论文页](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html) 与 [arXiv:1712.07629 提交历史](https://arxiv.org/abs/1712.07629) 均已读取；v1 为 2017-12-20。两页面属于同一论文证据体系，不构成独立性能验证。

<a id="source-10"></a>

**[cite:10] SuperGlue。** P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich, *SuperGlue: Learning Feature Matching with Graph Neural Networks*, CVPR 2020, pp. 4938–4947。[CVF 官方论文页](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) 与 [arXiv:1911.11763 提交历史](https://arxiv.org/abs/1911.11763) 均已读取；v1 为 2019-11-26。原论文定位是局部特征匹配与不可匹配点拒配。

<a id="source-11"></a>

**[cite:11] LoFTR。** J. Sun, Z. Shen, Y. Wang, H. Bao, X. Zhou, *LoFTR: Detector-Free Local Feature Matching with Transformers*, CVPR 2021, pp. 8922–8931。[CVF 官方书目、摘要及原文入口](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) 已读取。作为粗到细无检测器对应路线的一手说明，不采用论文当时榜单名次作为当前排名。

<a id="source-12"></a>

**[cite:12] LightGlue。** P. Lindenberger, P.-E. Sarlin, M. Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023, pp. 17627–17638。[CVF 官方书目、摘要及原文入口](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) 已读取。依据其自适应计算设计讨论平均与困难样本成本，不声称固定延迟或最新地位。

<a id="source-13"></a>

**[cite:13] 现代 API 边界参考。** [OpenCV tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html) 已读取，涵盖金字塔 LK 与 `findTransformECC` 等接口。该文档用于解释现代接口角色，不用于替代原论文年份、不证明仓库已经调用这些函数，也不推断本机安装版本具有页面列出的全部能力。

<a id="source-14"></a>

**[cite:14] North 报告与重印关系。** D. O. North, *An Analysis of the Factors Which Determine Signal/Noise Discrimination in Pulsed-Carrier Systems*, RCA Laboratories Report PTR-6C, 1943-06-25；重印于 *Proceedings of the IEEE*, 51(7), July 1963, pp. 1016–1027，DOI `10.1109/PROC.1963.2383`。[IEEE 正文出版记录与摘要](https://xplorestaging.ieee.org/document/1444313/metrics) 已读取；[IEEE 同期 Preface to North paper](https://xplorestaging.ieee.org/document/1444312) 明确注明报告日期、编号和重印关系，该前言为 p. 1015、DOI `10.1109/PROC.1963.2382`，不是 North 正文。独立交叉依据为 W. W. Ward, *The NOMAC and Rake Systems*, Lincoln Laboratory Journal 5(3), 1992 的 [MIT 原文 PDF](https://archive.ll.mit.edu/publications/journal/pdf/vol05_no3/5.3.2.nomacrakesystems.pdf)，已读其 p. 354 “Matched Filters and Correlators” 框及参考文献。IEEE 主站正文入口本次抓取为空，改由 IEEE 的 `xplorestaging.ieee.org` 页面取得书目；没有取得 North 正文全文，不据书目或历史回顾判定唯一首创。

<a id="source-15"></a>

**[cite:15] SSDA 代表文献与核验限制。** D. I. Barnea, H. F. Silverman, *A Class of Algorithms for Fast Digital Image Registration*, *IEEE Transactions on Computers*, C-21(2), February 1972, pp. 179–186，DOI `10.1109/TC.1972.5008923`。[IEEE 出版记录与摘要](https://xplorestaging.ieee.org/document/5008923/citations) 已读取；所用年份、页码以此为准，不采用个别二手来源的 2009 或 179–193。已尝试 [dblp 条目](https://dblp.org/rec/journals/tc/BarneaS72)、卷目录及其 Trier 镜像，但均返回反爬验证；作者页抓取只返回较近期条目，未取得 1972 项，因此 **dblp 未核验成功**，不作为已确认信息的证据。算法交叉说明为 [Lewis 扩展原文的 SSDA 段落](https://scribblethink.org/Work/nvisionInterface/)（已读随机次序、Monte Carlo 估计、早停及最大峰无保证的讨论）和 [Brown 1992 综述](https://www.geo.uzh.ch/microsite/rsl-documents/research/SARlab/GMTILiterature/PDF/Bro92.pdf) 的搜索摘录（逐窗口累积误差并在越阈时停止，全文仍未取得）。1972 原论文全文未取得；IEEE 摘要中的历史计算收益不转述为固定加速或仓库实测。

<a id="source-16"></a>

**[cite:16] 匹配滤波条件的现代解释，非历史首创证据。** [MIT 6.02 Lecture 8: Noise](https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/resources/lecture-8-noise/) 已读模型部分的逐字稿，并取得约 49:50–50:09 的检索摘录，说明已知样本加权、信号能量与噪声方差的关系。[Introduction to Stochastic Signal Processing，第 9 章 The Matched Filter](https://socratic-software.github.io/stochastica/Chap_9.html) 已读指定时刻输出、Cauchy–Schwarz 推导、噪声谱加权及白噪声时反转滤波器的内容；该教材采用的幅度 SNR 和频谱约定不直接搬成本章数值。本章明确采用有限维功率 SNR `abs(wᴴs)² / (wᴴCw)`，其等式、白噪声特例及非高斯边界由所列协方差条件直接推导；与 Lewis 原文的窗口去均值归一化公式分开，不作为 North 已逐页复核或 ZNCC 检测最优的证据。

### 7.3 本章验证边界

**原稿验证记录（保留）。** 下一段的“十二个方法”“十三个来源”“91 处链接”“仅新增本文件”和测试结果均属于本次前史增补前的记录，不是修订后的统计，也不表示本轮重跑了算法测试。

2026-11-04 本次检查确认十二个方法小节均具备双线及逐项字段，十三个来源条目完整，91 处仓库内文件链接／锚点有效，基础 Markdown 与空白检查通过。使用已有虚拟环境执行 `.\.venv\Scripts\python.exe -B -W error -m unittest tests.test_pose_ncc`，37 项测试通过；未重跑完整性能基准，不新增吞吐量或精度结果。变更范围按写入操作及前后 Git 状态核对，仅新增本文件，保留原有未提交改动。尝试跨命令保存哈希基线时，终端变量未保留，因此不将该哈希比较计为有效验证。

本章只新增文档，不改变旧专题、算法、依赖或测试。源码核对及本地链接检查用于确认描述与现有仓库一致；来源核验用于确认代表年份、题名和方法角色。两者都不能替代新增算法的运行验收、论文复现或工业现场验证。后续扩展应先统一失败协议与坐标语义，再在独立数据上比较候选质量、几何可靠性和资源成本，而不是仅按发表年份决定替换基线。

**本次前史增补验证（2026-11-04）。** 与写入前副本逐行比较，原稿 367 行按原顺序全部保留，仅插入增补；保留原十二分支和来源 1–13，新增前史小节、总表两行及来源 14–16。已检查 118 处仓库内文件链接／锚点、16 个来源条目、锚点唯一性、标题层级、表格列数、代码围栏、行尾空白、文件末尾换行与冲突标记，均通过。本文件在 Git 中仍为未跟踪文件，因此 `git diff --check` 不能覆盖它，格式与原文保留结论以直接读取文件的检查为准。本轮仅修订本文件，不改算法、依赖或测试，也未重跑单元测试或性能基准；不将原稿测试结果当作新增算法的验证。IEEE 与 Lewis 获取深度、dblp 阻挡及历史全文缺口见来源记录。
