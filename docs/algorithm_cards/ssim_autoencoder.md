# SA01 SSIM Autoencoder：正常重建、局部结构损失与异常残差

[学习规范](../algorithm_learning_standard.md) · [覆盖清单 G18／06-04](../evolution_coverage.md) · [缺陷异常总览](../defect_evolution.md) · [Q02 单尺度 SSIM 指标](preprocessing.md#q02-ssim-单尺度结构相似度)

<a id="sa01"></a>
<a id="g18"></a>

## 任务、演进与覆盖状态

**任务：**仅用经确认的正常图训练卷积自编码器（AE），测试时重建输入，通过输入与重建的局部结构差异产生异常热图、候选掩膜和区域证据。它不是将待检图与一张外部良品模板比较，也不是只调用一次全图 SSIM。输入需要固定通道、灰度尺度、采样分辨率和有效检测域；正常训练必须覆盖合法变化。正常重建差、异常重建好都可能发生，异常分数不是缺陷概率。[cite:1](#source-1)[cite:2](#source-2)

**演进定位：**逐像素 L2 重建误差容易放大正常边缘的轻微错位，却未必突出强度近似的结构缺陷 → 保留正常重建路线，改用局部亮度、对比度及结构统计作为训练目标和推理比较量 → 增加网络训练、窗口与容量选择成本 → 仍不保证异常无法通过瓶颈或被解码器重建。这里的“较不敏感”不是平移不变，更不能免除输入／重建坐标一致性。[cite:1](#source-1)[cite:2](#source-2)

方法版本分别为 **2018-07-05 arXiv v1** 与 **2019-02-01 arXiv v3**；正式书目为 **VISIGRAPP 2019，Volume 5: VISAPP，372–380**，不是 CVPR Workshops。v3 是本卡主线，v1 差异单列。核验日期 **2026-11-05**，本文件仅 SA01 一卡。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

| 字段 | 状态与边界 |
|---|---|
| L1 步骤 | 已核验：原文网络、训练／推理流程与具体尺寸；原文未交代的运行参数明确另作设计 |
| L2 机制 | 已核验：瓶颈、局部比较、重建融合和阈值的作用及反例 |
| L3 工程权衡 | 部分：尺度、定参、数据隔离和排障方案齐备，未在工业数据上定参 |
| L4 数学 | 已核验：限定于固定非负归一化窗口、正稳定常数的 SSIM；解析梯度及有限算术检查，不含网络全局收敛或检测保证 |
| L5 实现 | 部分：固定版本外部源码静态对应、可实施伪代码、文内可执行微检查；没有本仓库 AE 实现、训练权重或框架前向验证 |
| 面试自测 | 答 5：L1–L5 各一问，含答案、条件、易错点、证据入口 |
| 作者实现锁定 | **SSIM 原作者指标源码已锁内容；Bergmann 等人的 AE 作者实现未确认取得**。第三方 AE 锁提交，不冒充作者代码 |
| G18 销项口径 | 具备将“缺具体方法卡”转“已有”的正文依据；不表示全部五层已验收，不表示作者 AE 代码／完整复现子项关闭 |

本次仅写本文件，不改覆盖表、目录、总览、依赖或代码。现有 Q02 只承接指标，不能承接 AE 网络和训练链；本卡也不反向升级 Q02 的历史测试状态。仓库已有 PyTorch 教学网络和 `requirements-dl.txt`，未发现 SSIM-AE 专用 Python 实现；不把其前向示例算作 SA01 的训练或验证。

## L1 步骤：训练、融合重建与局部决策

### 输入输出与数据契约

- 训练输入：同一产品／材料域的正常灰度图，按原图、实物及批次先划分训练／验证／验收，再裁 patch；同图裁片不得跨折。监督目标是输入自身，无真实缺陷训练标签，但不是没有训练目标。
- 网络输入：每个 patch 为 `128×128×1`；数学与层表用 HWC，拟议 PyTorch 实现必须显式转 `B×1×128×128`。原文未指定张量内存布局，不将框架布局倒写为论文规定。
- 输出：浮点重建图、SSIM 图、异常图 `R=1-S`、有效支持掩膜、阈值后候选。保留 ROI 原点、缩放、patch 起点、融合权重和阈值版本；无覆盖／非有限／域外输入返回 UNKNOWN，不默认为正常。
- 本卡可实施补充协议将灰度除以 255 映射到 `[0,1]`，令动态范围 `L=1`；这是明确选定的实现尺度，不声称原文给出了这项归一化细节。输出层仍保持论文的线性激活，不暗中换 sigmoid 或裁剪。

### 具体 AE 层配置

编码器按 v3 Table I（亦见 v1 Fig. 3(b)）。每行给出输出尺寸、核、步幅及每侧 padding；所有未注明的 dilation 为本卡实现选择 1。[cite:1](#source-1)[cite:2](#source-2)

| 层 | 输出 H×W×C | 核 k | 步幅 s | padding p | 激活 |
|---|---|---|---|---|---|
| 输入 | 128×128×1 | — | — | — | — |
| E1 Conv | 64×64×32 | 4×4 | 2 | 1 | LeakyReLU，负斜率 0.2 |
| E2 Conv | 32×32×32 | 4×4 | 2 | 1 | 同上 |
| E3 Conv | 32×32×32 | 3×3 | 1 | 1 | 同上 |
| E4 Conv | 16×16×64 | 4×4 | 2 | 1 | 同上 |
| E5 Conv | 16×16×64 | 3×3 | 1 | 1 | 同上 |
| E6 Conv | 8×8×128 | 4×4 | 2 | 1 | 同上 |
| E7 Conv | 8×8×64 | 3×3 | 1 | 1 | 同上 |
| E8 Conv | 8×8×32 | 3×3 | 1 | 1 | 同上 |
| E9 Conv | 1×1×d | 8×8 | 1 | 0 | 线性 |

原文称解码器为反向结构。本卡将该描述具体化为如下转置卷积序列；它是可核对尺寸的实现解释，不是已取得的作者网络类。stride=1 的 3×3 层也可用独立普通卷积表达同形映射，但不能因此声称权重逐项相同；不与编码器绑权重。[cite:2](#source-2)

| 层 | 输出 H×W×C | k／s／p | 激活 |
|---|---|---|---|
| D1 ConvTranspose | 8×8×32 | 8／1／0 | LeakyReLU(0.2) |
| D2 ConvTranspose | 8×8×64 | 3／1／1 | 同上 |
| D3 ConvTranspose | 8×8×128 | 3／1／1 | 同上 |
| D4 ConvTranspose | 16×16×64 | 4／2／1 | 同上 |
| D5 ConvTranspose | 16×16×64 | 3／1／1 | 同上 |
| D6 ConvTranspose | 32×32×32 | 4／2／1 | 同上 |
| D7 ConvTranspose | 32×32×32 | 3／1／1 | 同上 |
| D8 ConvTranspose | 64×64×32 | 4／2／1 | 同上 |
| D9 ConvTranspose | 128×128×1 | 4／2／1 | **线性** |

此解释取 `output_padding=0`；空间关系为卷积 `floor((n+2p-k)/s)+1`，转置卷积 `(n-1)s-2p+k`。表中不含池化、跳连、BatchNorm、VAE 重参数化或预训练特征。bias、初始化和 Adam 内部参数未在所读论文配置中完整给出，不由现代框架默认值替作者补定。

### 训练配置与停止

| 项目 | v3 明文 | v1 差异／未知项 |
|---|---|---|
| 数据 | 两种 woven fabrics，各 100 正常图供训练／验证、50 缺陷图；另有 NanoTWICE 的 5 正常图、40 缺陷图 | v1 实验只针对 NanoTWICE，明确 4 正常图训练、1 图可作阈值验证 |
| 裁片 | 每网络 10,000 个正常 128×128 patch；裁片前图像缩为 256×256 | v1 写 20,000 patch；所读步骤未写 v3 的 256×256 预缩放，不能倒灌 |
| 潜变量 | 织物 d=100，纳米纤维 d=500 | v1 给 d 的敏感性比较，不把单个最优配置当跨域默认 |
| 优化 | Adam，初始学习率 2×10^-4，weight decay 10^-5，200 epochs | 两版这三项一致；batch size、随机种子、初始化、β、ε、学习率日程及 weight decay 的具体软件语义未充分披露 |
| SSIM | K=11，α=β=γ=1，训练与推理都使用结构度量 | 窗权重、裁边、数据范围及稳定常数歧义见 L4；不能仅凭 K=11 推定所有库等价 |

表内实验设定来自原文，不是本仓库已下载数据或已训练的记录。[cite:1](#source-1)[cite:2](#source-2)

本卡训练方案为：确认分组 → 固定 patch 清单／种子 → 前向 `y=D(E(x))` → 对完整有效局部窗求 `mean(1-SSIM(x,y))` → 反向传播 → Adam 更新 → 独立正常验证 → 保存模型与全部协议。原文的 200 epochs 是固定训练预算，不证明收敛到全局最优。若采用验证早停，须另记 patience、最小改进、选择准则及是否恢复最佳权重，属于工程变体；非有限 loss／梯度、数据为空或预算超时应中止并标失败。若需从零执行，可预注册 `batch=32, seed=0, β=(0.9,0.999), ε=10^-8, bias=True` 及初始化和 coupled L2 正则语义，但这些只是补齐方案，**本次未运行，亦非论文参数**。

### 推理、重建融合与阈值

1. 冻结模型，使用与训练一致的成像尺度、灰度变换和 ROI。对测试图取 128×128 patch，原文水平／垂直步幅 30 px，以减轻网络对 patch 位置的依赖。[cite:2](#source-2)
2. 对每个 patch 前向重建；同一全局像素累加多个重建值及覆盖次数，先得完整融合重建，再比较输入与融合重建。**原文平均的是重建像素，不是平均二值掩膜，也不是先算残差再平均。**[cite:1](#source-1)[cite:2](#source-2)
3. 本卡补齐末端起点：`starts(n)=sorted(unique(range(0,n-P+1,s) ∪ {n-P}))`，`n<P` 拒绝。这是避免末条带无覆盖的工程规则；原文没有精确给出尾块实现，不冒称其原代码。
4. 在融合重建上计算固定 SSIM 局部图，再取 `R=1-S`。本卡主实现用完整有效 11×11 窗，输出 `(H-10)×(W-10)`，坐标对应输入中心 `(row+5,col+5)`；原图四边 5 px 不作有效检测。ROI 洞须使其邻域中心失效，不能填零后当正常。
5. `M=valid AND (R>τ)` 得候选。原文扫描阈值得 ROC，没有跨产品通用数值 τ；v1 明说可用第 5 张正常图按目标假阳率设置阈值。工程上从独立正常验证有效像素分数的分位数选 τ，并固定分位数算法及 `>` 的并列值规则；只能估该正常分布内的像素误报，不等于未来整件误报率或缺陷召回保证。[cite:1](#source-1)[cite:2](#source-2)
6. v3 明文使用**直径 4 px 的圆形结构元开运算**去小候选；v1 另有每区域凸包填洞，v3 所读步骤未保留此项。偶数直径的离散化／锚点需实现时明确，不能未经核对换成 `disk(4)`（半径 4、通常 9×9 支持）。保持原始热图和处理前掩膜以审计细缺陷是否被删掉。[cite:1](#source-1)[cite:2](#source-2)

具体输入走查：按本卡尾块规则，256×256 灰度 ROI 每轴起点为 `0,30,60,90,120,128`，共 36 patch；每 patch 经 `128→64→32→32→16→16→8→8→8→1` 编码及对称解码。融合图仍 256×256；valid SSIM 为 246×246。是否 NG 必须看实际分数、冻结 τ 及区域规格，**不能从这些尺寸推得缺陷结果**。原图缩放回去仅是坐标回投，不恢复被缩图或开运算删掉的缺陷。

## L2 机制：为什么保留这些步骤

| 机制 | 解决的问题与成立条件 | 删除／替换的后果与反例 |
|---|---|---|
| 只学正常重建 | 用正常样本的可压缩结构作为间接先验 | 污染训练会教会模型重建缺陷；罕见但合法纹理未覆盖也会高分 |
| 瓶颈与局部 patch | 限制直接复制、控制模型容量和有限数据训练 | d 太小正常也模糊；d 太大或有强跳连可能保留异常；低维不构成异常必然离流形的证明 |
| 局部亮度／对比／结构 | 联合比较均值、能量和协方差，不只逐像素差 | 若灰度相近但排列改变，协方差项可突出差别；但大窗也会把小缺陷稀释，SSIM 不严格平移不变 |
| 训练及推理同类度量 | 让网络优化的保真与检测比较对齐 | 仅训练用 SSIM、测试仍 L2 不等同论文方法；只更换测试度量也必须单列消融 |
| 重叠重建平均 | 缓解不同 patch 上下文产生的接缝，覆盖计数归一化 | 不除覆盖次数造成中心亮、边缘暗；先平均残差通常不等价，见 L4 |
| 阈值与区域规则分离 | 将连续异常证据变为给定风险下的候选 | AUROC 是排序表现，不是操作点；开运算后的“干净图”可能恰是漏掉细划痕 |

这些机制与反例由原文及其目标函数支持，但没有三份独立复现实验证据，性能结论仍为 `[SINGLE-SOURCE]`。SSIM 是相似度，`1-SSIM` 一般不满足距离度量的全部公理；论文称 distance 不构成三角不等式保证。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)

## L3 工程权衡：尺度、数值和操作点

| 参数 | 单位／起点依据 | 代价、症状与诊断 |
|---|---|---|
| 图像分辨率／P=128 | px；v3 的训练预缩放及 patch 尺度 | 先换算最小缺陷宽度占几像素；缩没了的细节不能靠降低 τ 恢复 |
| K=11，σ=1.5 | px；K 来自 AE 原文，Gaussian σ 来自 SSIM 原作者实现，是本卡选定补充 | 大窗更稳定但扩展异常响应支持；小窗更受噪声影响；同时看正常纹理周期与缺陷宽度 |
| d=100／500 | 潜变量维数；对应论文织物／纳米纤维 | 不是压缩率保证；同时看正常重建和真实缺陷重建，不能只最小化正常 loss |
| patch 步幅 30 | px；原文重建融合 | 步幅变小增加重叠与计算量；先检查覆盖图和接缝，不默认残差融合等价 |
| L、C1、C2 | L 为灰度单位，C 为灰度平方 | 归一化后沿用 L=255 会让常数过大；逐图估 L 会改变异常尺度；近常量纹理对 C2 特别敏感 |
| loss reduction／正则 | loss 无量纲，λ 随目标定义 | mean 与 sum 不仅影响梯度大小，还改变相对正则强度；不能直接继承学习率和 weight decay |
| τ、分位数 q | τ 与残差同尺度、q 无量纲 | 正常像素 q 分位数不约束最大像素聚合后的整图误报；换图尺寸／有效域需重校准 |
| 形态学尺寸 | 工作分辨率 px，原文直径 4 | 先保存前后候选，按最小缺陷做漏检检查；不能把直径与半径混用 |

调参顺序：**划分与污染审计 → 分辨率／可检性 → 重建及覆盖坐标 → SSIM 常数和支持 → 模型容量／优化 → 固定阈值 → 后处理 → 性能优化**。正常验证足以暴露误报、漂移和数值问题，不足以确定未知缺陷召回最优模型。

| 症状 | 应收集的证据 | 调整或回退 |
|---|---|---|
| 强边缘大量高分 | 输入／重建叠图、patch 接缝、SSIM 三因子、光照 | 先排缩放／坐标和欠拟合，不先降灵敏度 |
| 缺陷也重建得很好 | 对照重建、训练清单、d、污染与重复帧 | 冻结训练库、审计污染；考虑嵌入或判别路线，不保证缩 d 就能修好 |
| 训练低损失、推理图异常 | train/test 的窗权重、总体／样本矩、量化、边界 | 统一协议后再对照；第三方示例的异构 SSIM 见 L5 |
| 平坦区域 NaN／极大梯度 | 二阶矩相消、C2、半精度、输出范围、分母 | 提升统计精度并用中心化矩定位；不要只给总分母加任意 ε 掩盖错误 |
| 换批次后普遍高分 | 光照、原料、正常模式、逐批误报／UNKNOWN | 返回复检并重新标定，不让低分模型自行确认新训练良品 |

工程验收应分开图像级、像素级、区域级和按缺陷尺寸分层的指标；冻结验收集，明确阈值是否用过缺陷标签。SSIM-AE 不提供通用定位误差界、漏检率上界或产线节拍保证。

## L4 数学：局部损失、梯度与数值边界

### 统计支持与常数不能混写

对一个完整支持的窗口，参考像素记 `x_i`，重建记 `y_i`，固定权重 `w_i≥0, Σw_i=1`。定义总体加权矩：

\[
\mu_x=\sum_iw_ix_i,\quad \mu_y=\sum_iw_iy_i,\quad
v_x=\sum_iw_i(x_i-\mu_x)^2,\quad v_y=\sum_iw_i(y_i-\mu_y)^2,\quad
c_{xy}=\sum_iw_i(x_i-\mu_x)(y_i-\mu_y).
\]

本卡计算方案用 11×11 归一化 Gaussian、σ=1.5、valid 支持、总体矩，无 `n/(n-1)` 补偿；继承的是 Wang `ssim_index.m` 的计算约定，不是声称 Bergmann 文本明确列出所有这些实现细节。[cite:4](#source-4)

令 `α=β=γ=1`，亮度因子 `l=(2μxμy+C1)/(μx²+μy²+C1)`，对比因子 `(2√vx√vy+C2)/(vx+vy+C2)`，结构因子 `(2cxy+C2)/(2√vx√vy+C2)`，相乘消去中间项：

\[
S=\frac{AB}{CD},\qquad
A=2\mu_x\mu_y+C_1,\quad B=2c_{xy}+C_2,\quad
C=\mu_x^2+\mu_y^2+C_1,\quad D=v_x+v_y+C_2.
\]

即使平坦窗 `vx=vy=0`，合并式也不需要对零方差开方求导。训练损失取

\[
\mathcal L_{\mathrm{rec}}(\theta)=\frac{1}{Z}\sum_{b,u\in V_b}\bigl(1-S(x_{b,u},f_\theta(x_b)_u)\bigr),\qquad Z=\sum_b|V_b|>0.
\]

这里 `u` 是窗口中心，不是每个独立像素对。`1-S` 的显式 reduction 是本卡可实施约定，与“最大化平均 SSIM”等价；`(1-S)/2` 只差比例却会改变固定 τ、学习率与正则相对权重，不能无记录切换。

| 口径 | 稳定常数 | 证据与判定 |
|---|---|---|
| Bergmann v1 §3.2、v3 §III-B 的字面描述 | `c1=0.01, c2=0.03` 直接出现在合并式 | **确实读到，不能静默改成平方值**；未取得作者 AE 代码确认实际实验采用哪一套 |
| Wang 作者 `ssim_index.m` v1.0 | `C1=(K1 L)^2, C2=(K2 L)^2`，`K1=.01,K2=.03`，默认 L=255 | L=1 时是 `0.0001,0.0009`，不等于左侧论文的字面 C |
| TF 2.2.0 `tf.image.ssim(max_val=1)` | 同平方规则；11×11 Gaussian，σ=1.5，总体矩、VALID | 第三方 AE 训练走这条路径，不是作者实际实验选择的证明 |

来源冲突标为 **[DISPUTED：论文文字与标准实现参数口径不一致]**；无法确认是符号简写、排印问题还是实验设置差异，不擅自判为已纠正的笔误。本卡后续数值微检查选 Wang 平方规则，同时保留论文 literal 模式；若重新复现，应分别报告两模式并向作者确认，而非只写“默认 SSIM”。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)[cite:6](#source-6)

### 对重建像素的解析梯度

固定 x、固定窗口及有效域，对窗口内第 j 个重建像素：

\[
\frac{\partial\mu_y}{\partial y_j}=w_j,\qquad
\frac{\partial v_y}{\partial y_j}=2w_j(y_j-\mu_y),\qquad
\frac{\partial c_{xy}}{\partial y_j}=w_j(x_j-\mu_x).
\]

其中中心化项求导后的求和项因 `Σw_i(y_i-μy)=0` 消失。由商法则，记 `A'=2μx wj`、`B'=2wj(xj-μx)`、`C'=2μy wj`、`D'=2wj(yj-μy)`，得到

\[
\frac{\partial S}{\partial y_j}
=\frac{A'B+AB'}{CD}-\frac{AB(C'D+CD')}{(CD)^2}
=\frac{2w_j[\mu_x B+A(x_j-\mu_x)]}{CD}
-S\left[\frac{2\mu_yw_j}{C}+\frac{2w_j(y_j-\mu_y)}{D}\right].
\]

该式不除以 A 或 B；若负相关恰使 `B=0`，仍能计算，而用 `S·∂log(S)` 或 `S(A'/A+B'/B-…)` 会误入零分母／负数对数。整图重叠窗口必须累加所有包含该像素的贡献：

\[
\frac{\partial\mathcal L_{\mathrm{rec}}}{\partial y_p}
=-\frac1Z\sum_{u:p\in W_u}\frac{\partial S_u}{\partial y_p},\qquad
\nabla_\theta\mathcal L_{\mathrm{rec}}=J_{f_\theta}^{T}\nabla_y\mathcal L_{\mathrm{rec}}.
\]

若把 weight decay 实现为目标中 `λ||θ||²/2`，参数梯度另加 `λθ`；AdamW 的解耦衰减不是同一更新，论文给一个 weight decay 数值不足以恢复全部软件行为。LeakyReLU 在零点的框架次梯度约定也需记录，局部 SSIM 光滑不代表整个非凸 AE 全局光滑／可辨识；潜变量可有多种等价表示，没有唯一正常流形恢复保证。

### 范围、退化、融合与复杂度

- 精确总体统计下 `C≥C1>0,D≥C2>0`；由 `2|μxμy|≤μx²+μy²`、`|cxy|≤√vxvy≤(vx+vy)/2`，得 `|A|≤C,|B|≤D`，故 `-1≤S≤1`、`0≤R≤2`。这是代数范围，不是缺陷概率界；数值略超界先诊断，不在训练中任意 clamp 掩盖问题。
- `x=y` 时 `S=1` 且局部梯度为零。两常量 a、b 时 `S=(2ab+C1)/(a²+b²+C1)`；零常量对一常量且 L=1 的平方规则给 `0.0001/1.0001`，论文 literal C 则给 `0.01/1.01`。负相关可令 S<0，不能截到 `[0,1]` 后宣称等价。
- 同时将 x、y、L 乘正数 a，平方常数模式 S 不变；仅改变像素而忘改 C 不成立。加相同亮度偏置一般也不保持 S 不变。L 不得逐图按观测极差重估。
- 稳定常数不消除精度问题：`E[y²]-E[y]²` 的相消会产生负方差；FP16 下近黑窗的乘积分母可能下溢。使用至少 FP32 统计、必要时 FP64 中心化矩作对照；先分因子求比值可减少乘积下溢。对近零负方差裁零会改梯度，只能限定误差容差并验证，明显负值或非有限须失败退出。
- 融合公式为 `ŷ(p)=Σt a_t(p)ŷ_t(p)/Σt a_t(p)`，原文等权，权重和必须大于零。SSIM 非线性，`1-S(x,mean ŷ_t)` 一般不等于 `mean[1-S(x,ŷ_t)]`。例如常量 x=.5，两重建常量 .2、.8，先平均重建得到完全一致、R=0；先算两残差再平均大于零。这是算术反例，不是检测实验。
- 单个像素变化能影响其周围多个窗口中心；在固定重建下 K=11 的直接统计支持半径 5 px，网络重建传播还可更广，故热图不是缺陷真边界。忽略无效支持或开运算抹除细区域都会改变有效检测范围。
- 直接局部统计复杂度 `O(BHWK²)`，Gaussian 可分离后为 `O(BHWK)`；AE 卷积约为 `Σl H_l W_l C_in,l C_out,l k_l²` 次乘加，训练另需激活及反传。T 个推理 patch 的网络成本随 T 增长，融合约 `O(TP²)`；本卡不报未经测量的 GPU 时延。批处理／可分离滤波优化须逐图逐像素对照，不能只比较全图 loss。

## L5 实现：固定版本对应、伪代码与有限检查

### 原文、作者指标源码及第三方 AE 的边界

**作者链核验结论：**MVTec 出版页确认题名、作者、VISAPP 书目并提供 textures 补充材料入口；合著者 Sindy Löwe 本人页面为该论文提供 arXiv，但未列代码链接。其 GitHub 账号 `loeweX` 的公开仓库列表本次未找到对应 AE 工程。有限搜索不能证明作者从未发布过代码，结论只能是**本次未确认取得，无法锁定 Bergmann 作者 AE commit／权重**。不把 MVTec AD 的评测脚本当 AE 作者训练实现。[cite:3](#source-3)[cite:8](#source-8)

取得并静态读取的 AE 是 `plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-`；README 明写 **third party implementation**。锁定提交 **`1bd6651a77d796445156d2a3e08d11d805079367`**，提交时间 **2020-07-27T10:01:15Z**。这只锁源码，不锁训练环境；README 指定 `tensorflow==2.2.0`，`skimage` 等未全部锁版本，树中未发现 LICENSE，不能据公开访问认定可任意商用。本卡仅分析，不复制整套代码。[cite:5](#source-5)

| 证据位置 | 实际行为 | 与原文／本卡主线的差异 |
|---|---|---|
| `network.py:AutoEncoder` | HWC；128 模式编码层数／通道与表吻合；256 模式增加下／上采样层；解码 stride=1 用普通卷积 | 输出是 **sigmoid**，而论文输出线性；默认 `grayscale=False` 为 3 通道，不是论文单灰度 |
| `options.py` | 默认 P=128、d=100、batch=128、epochs=200、stride=32、percent=98 | 这些是第三方默认，不倒写为论文 30 步幅或未报告的 batch |
| `train.py:data_flow,ssim_loss` | `/255`；`1-reduce_mean(tf.image.ssim(...,max_val=1))`；可选 SSIM+L1 | TF 2.2.0 核为 Gaussian／总体矩／VALID／平方 C；混合 L1 是变体 |
| `train.py` 优化及验证 | `Adam(lr,decay)`，EarlyStopping(patience=20)，保存验证最优；增强后文件清单末 20% 验证 | TF 2.2.0 的 `decay` 是学习率 `lr/(1+decay·step)`，**不是论文 weight decay**；裁片级分割不能保证原图级隔离 |
| `utils.py:get_patch,patch2img` | 栅格裁片，重建累加／计数平均；不追加最后起点 | `(n-P)%stride!=0` 时可能留无覆盖尾条并除零；不能机械将32改30而不补覆盖 |
| `test.py:get_residual_map` | 重建先 `*255` 转 uint8，再用 `skimage.metrics.structural_similarity(win_size=11,full=True)` | 量化影响残差；没有显式指定 Gaussian、总体矩或 data_range，不能当训练核等价 |
| `test.py:get_threshold` | 原始 train/good 文件末 20% 的像素残差分别取98分位数 | 与训练增强裁片验证集非同一显式分组；阈值标定尚未乘边缘抑制，应用时则乘了，协议不一致 |
| `test.py:get_results` | SSIM 和 L1 各自过阈值后 **OR**，边缘5 px乘0.2；可选黑／白背景掩膜 | 不是原文单 SSIM；不是 SSIM 三因子加权和，也不是重建融合的替代 |
| `test.py` 后处理 | `morphology.disk(4)` 后开运算 | 4 是半径，不是论文直径4；不能用作者实验结论为该变体背书 |

第三方源码的训练／推理失配可用 **scikit-image v0.17.2** 作固定版本解释：该版本默认 `gaussian_weights=False,use_sample_covariance=True`，`win_size=11` 时是 11×11 均匀统计、121/120 协方差补偿；`full=True` 返回整图 S，只有标量均值会裁边。**v0.17.2 是本卡选择的审阅版本，不是已证实第三方作者安装版本**；依赖未锁使其实际环境不可还原。现代版本的 `channel_axis` 迁移也不等于旧 `multichannel` 默认语义已验证。[cite:7](#source-7)

### 可实施伪代码

此处是原文主线加显式工程契约，非已执行网络。保持线性输出、重建后融合及完整支持；SSIM 常数模式必须保存到模型元数据，不能靠函数名猜测。

```text
TRAIN(normal_images, split_manifest, config):
    verify_group_disjointness(split_manifest)
    verify_normal_labels_and_finite_images(normal_images)
    patches = make_frozen_training_patches(normal_images, config)
    model = build_encoder_decoder(layer_tables, linear_output=True)
    optimizer = configured_adam_with_explicit_regularization(config)
    for epoch in 1..config.epochs:
        for batch in patches:
            x = fixed_gray_transform(batch)
            y = model(x)
            S, valid = local_ssim(x, y, config.window, config.constants)
            require(count(valid) > 0 and finite(S[valid]))
            loss = mean(1-S[valid]) + explicit_regularizer(model)
            clear_gradients()
            backward(loss)
            require(finite_gradients())
            optimizer.step()
        evaluate_frozen_normal_validation()
        save_checkpoint_with_protocol()
    return chosen_checkpoint_with_stopping_reason()

INFER(image, model, protocol, tau):
    x, input_valid, coordinates = fixed_preprocess(image, protocol)
    require(min_shape(x) >= 128 and finite(x))
    sum_image = zeros_like(x)
    count_image = zeros_like(x)
    for origin in cartesian(starts(H,128,30), starts(W,128,30)):
        patch = crop(x, origin, 128)
        require(all_valid(input_valid, origin, 128))
        y = model_in_eval_mode(patch)
        require(shape(y) == shape(patch) and finite(y))
        add_at(sum_image, origin, y)
        add_at(count_image, origin, ones_like(y))
    require(all(count_image > 0))
    reconstruction = sum_image / count_image
    S, valid = local_ssim(x, reconstruction, protocol.window, protocol.constants)
    R = 1-S
    raw_mask = valid AND (R > tau)
    mask = opening_with_frozen_diameter_and_anchor(raw_mask)
    mask = mask AND valid
    return reconstruction, R, valid, raw_mask, mask, coordinates
```

严格版伪代码遇不完整 patch 直接拒绝当前 ROI；若改为跳过并保留剩余有效区域，必须显式处理零覆盖和新的检测域，不能悄悄返回完整 OK。模型 `eval` 不改变本卡无 BN／Dropout 网络的数学映射，但属于冻结推理状态的契约。SSIM 函数必须返回局部图；只返回每图标量的 `tf.image.ssim` 不能直接用于像素定位。

### 文内可执行微检查

以下 **PowerShell 5.1／System.Double** 代码仅检验公式、梯度及形状算术，无第三方依赖，不创建文件。确定性数组是代数单元测试输入，不是合成缺陷数据集，更不是模型输出。它没有执行上述网络伪代码、MATLAB、TensorFlow、PyTorch 或 scikit-image。

```powershell
$ErrorActionPreference = 'Stop'
$script:checks = 0
function Assert-SA([bool]$ok, [string]$name) {
    if (-not $ok) { throw $name }
    $script:checks++
}
function Near-SA([double]$a, [double]$b, [double]$tol, [string]$name) {
    Assert-SA (-not [double]::IsNaN($a) -and -not [double]::IsInfinity($a) -and [math]::Abs($a-$b) -le $tol) $name
}
function Window-SA([double[]]$x, [double[]]$y, [double[]]$w, [double]$c1=0.0001, [double]$c2=0.0009) {
    if ($x.Count -eq 0 -or $x.Count -ne $y.Count -or $x.Count -ne $w.Count -or $c1 -le 0 -or $c2 -le 0) { throw 'invalid shape/constants' }
    $mx=0.0; $my=0.0; $sw=0.0
    for ($i=0; $i -lt $x.Count; $i++) {
        foreach ($v in @($x[$i],$y[$i],$w[$i])) {
            if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) { throw 'nonfinite' }
        }
        if ($w[$i] -lt 0) { throw 'negative weight' }
        $sw+=$w[$i]; $mx+=$w[$i]*$x[$i]; $my+=$w[$i]*$y[$i]
    }
    if ([math]::Abs($sw-1) -gt 1e-12) { throw 'weights not normalized' }
    $vx=0.0; $vy=0.0; $cov=0.0
    for ($i=0; $i -lt $x.Count; $i++) {
        $dx=$x[$i]-$mx; $dy=$y[$i]-$my
        $vx+=$w[$i]*$dx*$dx; $vy+=$w[$i]*$dy*$dy; $cov+=$w[$i]*$dx*$dy
    }
    $a=2*$mx*$my+$c1; $b=2*$cov+$c2
    $c=$mx*$mx+$my*$my+$c1; $d=$vx+$vy+$c2
    $s=($a/$c)*($b/$d)
    $g=New-Object double[] $x.Count
    for ($j=0; $j -lt $x.Count; $j++) {
        $g[$j]=2*$w[$j]*($mx*$b+$a*($x[$j]-$mx))/($c*$d)-$s*(2*$my*$w[$j]/$c+2*$w[$j]*($y[$j]-$my)/$d)
    }
    [pscustomobject]@{ S=$s; G=$g }
}
function Starts-SA([int]$n, [int]$p, [int]$stride) {
    if ($n -lt $p -or $p -le 0 -or $stride -le 0) { throw 'invalid tile shape' }
    $a=@(); for ($i=0; $i -le $n-$p; $i+=$stride) { $a+=$i }
    @($a+@($n-$p) | Sort-Object -Unique)
}
$w=New-Object double[] 121
$x=New-Object double[] 121
$y=New-Object double[] 121
$sum=0.0
for ($r=-5; $r -le 5; $r++) {
    for ($c=-5; $c -le 5; $c++) {
        $i=($r+5)*11+$c+5
        $w[$i]=[math]::Exp(-($r*$r+$c*$c)/4.5); $sum+=$w[$i]
        $x[$i]=(($i*17)%101)/100.0; $y[$i]=(($i*13+7)%103)/102.0
    }
}
for ($i=0; $i -lt 121; $i++) { $w[$i]/=$sum }
$same=Window-SA $x $x $w
Near-SA $same.S 1 1e-12 'identity'
Near-SA (($same.G | ForEach-Object {[math]::Abs($_)} | Measure-Object -Maximum).Maximum) 0 1e-10 'identity gradient'
$zero=[double[]](@(0.0)*121); $one=[double[]](@(1.0)*121)
Near-SA (Window-SA $zero $zero $w).S 1 1e-12 'zero variance'
Near-SA (Window-SA $zero $one $w).S (0.0001/1.0001) 1e-12 'standard constants'
Near-SA (Window-SA $zero $one $w 0.01 0.03).S (0.01/1.01) 1e-12 'literal constants'
$negative=Window-SA @(0.0,1.0) @(1.0,0.0) @(0.5,0.5)
Assert-SA ($negative.S -lt 0 -and $negative.S -ge -1) 'negative covariance'
$z=Window-SA @(-1.0,1.0) @(0.00045,-0.00045) @(0.5,0.5)
Near-SA $z.S 0 1e-12 'zero numerator'
Assert-SA (-not [double]::IsNaN($z.G[0])) 'zero numerator gradient finite'
$ref=Window-SA $x $y $w
Near-SA (Window-SA $y $x $w).S $ref.S 1e-12 'symmetry'
$xs=[double[]]($x | ForEach-Object {255*$_}); $ys=[double[]]($y | ForEach-Object {255*$_})
Near-SA (Window-SA $xs $ys $w 6.5025 58.5225).S $ref.S 1e-12 'range scaling'
$h=1e-6; $maxError=0.0
for ($j=0; $j -lt 121; $j++) {
    $yp=[double[]]$y.Clone(); $ym=[double[]]$y.Clone()
    $yp[$j]+=$h; $ym[$j]-=$h
    $fd=((Window-SA $x $yp $w).S-(Window-SA $x $ym $w).S)/(2*$h)
    $maxError=[math]::Max($maxError,[math]::Abs($fd-$ref.G[$j]))
    Near-SA $fd $ref.G[$j] 1e-7 "gradient $j"
}
$bad=[double[]]$y.Clone(); $bad[0]=[double]::NaN
$rejected=$false; try { Window-SA $x $bad $w | Out-Null } catch { $rejected=$true }
Assert-SA $rejected 'NaN rejected'
$n=128; $ks=@(4,4,3,4,3,4,3,3,8); $ss=@(2,2,1,2,1,2,1,1,1); $ps=@(1,1,1,1,1,1,1,1,0)
$expected=@(64,32,32,16,16,8,8,8,1)
for ($i=0; $i -lt 9; $i++) {
    $n=[int]([math]::Floor(($n+2*$ps[$i]-$ks[$i])/$ss[$i])+1)
    Assert-SA ($n -eq $expected[$i]) "encoder $i"
}
for ($i=8; $i -ge 0; $i--) { $n=($n-1)*$ss[$i]-2*$ps[$i]+$ks[$i] }
Assert-SA ($n -eq 128) 'decoder size'
$starts=@(Starts-SA 256 128 30)
Assert-SA (($starts -join ',') -eq '0,30,60,90,120,128') 'tail starts'
$cover=New-Object int[] 256
foreach ($s in $starts) { for ($i=$s; $i -lt $s+128; $i++) { $cover[$i]++ } }
Assert-SA (($cover | Measure-Object -Minimum).Minimum -gt 0) 'coverage'
Assert-SA ($starts.Count*$starts.Count -eq 36 -and 256-11+1 -eq 246) 'map sizes'
$fused=1-(Window-SA @(0.5) @(0.5) @(1.0)).S
$averaged=((1-(Window-SA @(0.5) @(0.2) @(1.0)).S)+(1-(Window-SA @(0.5) @(0.8) @(1.0)).S))/2
Near-SA $fused 0 1e-12 'reconstruction fusion'
Assert-SA ($averaged -gt 0.1) 'residual fusion differs'
"PASS $script:checks assertions; max_gradient_abs_error=$maxError"
```

执行方法：从本文件提取唯一 `powershell` 围栏块，以 `[scriptblock]::Create(...)` 在 PowerShell 5.1 内执行。结果与文档静态检查记录见末节；不把算术运行等同于作者网络复现。

## 五层面试自测：贯穿同一正常织物场景

| 层级／问题 | 参考答案与成立条件 | 易错答案 | 证据入口 |
|---|---|---|---|
| L1：256×256 正常织物域的待检图如何得到候选？ | 按固定灰度尺度，128 patch／30 步幅加尾块，共36次重建；像素累加除覆盖数，融合后做SSIM，取1-S，valid域过冻结阈值再开运算；无效域UNKNOWN | 直接对整图算一个SSIM，或者把patch二值掩膜拼接 | L1推理；v3 §IV-B |
| L2：为什么仅降低逐像素残差阈值未必能救结构缺陷？ | 正常边缘错位与结构缺陷可能在L2上重叠；SSIM比较局部均值／方差／协方差，但仅在重建确实改变异常时有证据；原样重建的缺陷仍低分 | SSIM天然平移不变、任何异常必高分 | L2；v3 §III-B |
| L3：换更细纹理后误报高，先改τ还是K？ | 先核分辨率、纹理周期、正常库与训练／推理统计一致性，查看原图／重建／热图，再在分组验证上比较K、d，最后校准τ；仅正常验证不能评价漏检 | 98分位数保证整件2%误报、参数越大越好 | L3；v1 §4.1；第三方阈值边界 |
| L4：如何对一个重建像素求梯度，B=0怎么办？ | 先写三个矩的导数，再商法则；累加所有包含该像素的窗口并除Z、取负；使用不除A/B的表达式，C/D因正稳定项非零；0方差也用合并式 | SSIM图各中心独立，或者对S取log后求导 | L4解析梯度与微检查 |
| L5：网络结构看起来一样，为什么不能说已复现论文？ | 作者AE未锁；第三方sigmoid、Adam decay、窗／矩／量化、SSIM与L1 OR、disk半径均与主线有差异；还缺框架前向、训练数据划分、权重、指标和真实缺陷验收 | 下载公开代码／算术通过即官方完整复现 | L5版本表；来源5–7 |

## 来源、访问深度与版本锁定

访问日期均为 **2026-11-05**。`T` 为实际读取的方法／源码正文，`A` 为作者页面说明，`B` 为书目；并非看到链接便算全文。arXiv HTML 初次返回截断，随后通过内存 HTTP 获取并提取 v1 §3.2–4.3、v3 §III-B–IV-C，读到层表、训练及推理段落；未作逐图人工复核、PDF版式核对或全部版本逐字 diff。同一论文、出版商及作者网页只交叉支持归属，不视为独立性能实验。

- <a id="source-1"></a>**[cite:1] Bergmann、Löwe、Fauser、Sattlegger、Steger，2018 v1。** [提交历史／摘要](https://arxiv.org/abs/1807.02011)，[固定 v1 正文](https://arxiv.org/html/1807.02011v1)。T＋B：已读方法、Fig.3(b)层表、§4训练／测试与评估段落，核对20,000裁片、4+1正常图、步幅30、开运算与凸包；实验结果仅为作者报告 `[SINGLE-SOURCE]`，不转录为本站表现。
- <a id="source-2"></a>**[cite:2] 同论文，2019 v3。** [固定 v3 正文](https://arxiv.org/html/1807.02011v3)。T：已读目标、SSIM公式、Table I、§IV-A/B/C的配置及部分结果讨论；核对10,000裁片、256预缩放、d=100/500、训练200轮、重建平均和直径4开运算。未取得作者训练程序来消除常数／优化器歧义。
- <a id="source-3"></a>**[cite:3] 原作者机构与出版商书目。** [MVTec academic publications](https://www.mvtec.com/company/research/publications)，[SciTePress DOI入口](https://www.scitepress.org/Link.aspx?doi=10.5220/0007364503720380)。A＋B：两页均已读，确认VISAPP 2019、372–380、DOI `10.5220/0007364503720380`。MVTec 的 `textures.zip` 仅看到补充材料链接，未下载；不声称内含作者代码。总览旧有错误CVPR入口失败记录仍保留在原文件，本卡不沿用错误归属。
- <a id="source-4"></a>**[cite:4] SSIM 指标原作者实现，不是 Bergmann AE。** [Wang作者页](https://ece.uwaterloo.ca/~z70wang/research/ssim/)，[ssim_index.m](https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m)。A＋T：页面及完整函数已读；头部 **Version 1.0，Copyright 2003 Zhou Wang**，正式论文由作者页确认为 TIP 13(4),600–612,2004。锁本次HTTP内容解码后UTF-8编码的 SHA-256：`73a32c4e49c19e0fb880675733af8d8bfa9c66052ec31b8435630c4a614676e1`；ETag `"1717-48c414dddff80"`、Last-Modified `2010-07-26 02:40:14 GMT` 是服务器资源元信息，不是论文年或Git版本。原码 Gaussian／VALID／总体矩／平方C已核；未运行MATLAB。网页宽泛使用说明与源码头的教学科研／商业需许可条款不同，复用须核实；本卡微检查独立按公式编写。
- <a id="source-5"></a>**[cite:5] 第三方 AE 固定提交。** [README](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/README.md)，[network.py](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/network.py)，[train.py](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/train.py)，[test.py](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/test.py)，[utils.py](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/utils.py)，[options.py](https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/1bd6651a77d796445156d2a3e08d11d805079367/options.py)。T：五个Python文件均从该SHA的raw入口完整读取，提交与树经GitHub API确认；README滚动入口已读其自称第三方及依赖说明，固定README链接供后续复查。首次猜测的 `model.py` 返回404，随后按树读取实际 `network.py`。未运行、未下载权重，不引用README准确率为本卡证据。
- <a id="source-6"></a>**[cite:6] TensorFlow v2.2.0 官方源代码。** [image_ops_impl.py](https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/image_ops_impl.py)，[optimizer_v2.py](https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/optimizer_v2.py)。T（指定函数节选）：实际读取 `_ssim_helper/_fspecial_gauss/_ssim_per_channel/ssim` 及 `_decayed_lr`；确认平方C、总体矩、Gaussian、VALID、通道／空间平均和学习率decay。锁发布标签，不宣称已安装或审遍整个框架。
- <a id="source-7"></a>**[cite:7] scikit-image v0.17.2 官方源码。** [完整 structural_similarity 文件](https://github.com/scikit-image/scikit-image/blob/v0.17.2/skimage/metrics/_structural_similarity.py)。T：raw正文已读，核对默认均匀窗、样本矩、uint8范围推断、full图与均值裁边差别。是本卡解释版本，不伪装为第三方仓库未披露的安装版本；SciPy滤波底层未完整审阅／运行。
- <a id="source-8"></a>**[cite:8] 作者代码定位与访问限制。** [Sindy Löwe本人页面](https://sindylowe.com/)、[账号资料API](https://api.github.com/users/loeweX)、[公开仓库列表API](https://api.github.com/users/loeweX/repos?per_page=100)、[MVTec AD页面](https://www.mvtec.com/company/research/datasets/mvtec-ad)。A／列表已读：作者身份由姓名、主页互证，对应论文只见arXiv入口；MVTec AD提供的是evaluation code。另查 GitHub 题名／SSIM-AE仓库，候选或自称第三方、或仅README；错误猜测账号查询返回422不作不存在的证明。结论为 `[INSUFFICIENT DATA] 作者AE发布身份与版本未闭合`，不是“证明没有官方实现”。

## 有限验证记录与 G18 销项依据

验证严格分三类：原文／源码静态核验、文内确定性算术、文档结构及写入范围。Windows 的 `python --version` 本次返回9009（WindowsApps占位入口，非可用Python），没有因此安装环境或新增依赖；以现有PowerShell执行文内微检查，不能把未执行的Python／框架对照写成通过。

文内算术检查覆盖同图及零梯度、平坦窗、两套常数、负相关、零分子、对称性、共同缩放、121个中心差分梯度、NaN拒绝、编码／解码尺寸、尾块覆盖和融合非交换性。仍未执行：真实图SSIM对照、二维整图梯度／掩膜、框架自动微分、AE前向／训练、数据集指标、模型导出、量化、性能剖析及现场验收。

| 本次实际执行项目 | 结果及解释 |
|---|---|
| 文内PowerShell块 | **147条断言通过**；121点梯度中心差分步长10^-6、绝对容差10^-7，最大绝对差 **5.16100773229056×10^-11**；只适用于这组确定性算术输入 |
| 标题／引用／本地链接 | L1–L5各1个二级标题，4条围栏配对，10个显式锚点无重复，35处本地链接及锚点引用（含重复）通过；Q02目标标题存在 |
| 检查器失败与修正 | 首次链接正则把代码内数组语法误当链接，检查失败；排除围栏代码后重跑通过，不隐瞒首轮失败 |
| 固定README复查 | 额外读取固定SHA的README raw正文，确认自称third party及tensorflow==2.2.0；补足来源5最初滚动README读取的边界 |
| 空白／Git检查 | 新文件逐行无尾空白；`git diff --check`无差异格式错误，但未跟踪的新卡不在该命令检查范围内，故另做新文件检查 |
| 写入范围 | 编辑操作只新增本文件；前后Git摘要保留原有修改。尝试记录7894个现有文件哈希，但内存基线未跨命令保留，**全工作树哈希比对未完成**，不声称其他文件逐字节比对通过；未写覆盖表或目录 |

**销项依据：**按覆盖清单“已有”的正文口径，SA01已独立承接06-04的具体AE网络、正常训练、局部结构目标、融合重建与异常决策，并有L1–L5、答5、固定版本分析和明确实现边界，因此可作为G18“缺具体方法卡”的销项材料。**作者AE实现锁定和完整复现仍未完成**，不以第三方工程、指标作者源码或微检查替代；若G18验收额外要求作者AE身份／版本闭合，则该子项保留未销。覆盖表本身未修改，计数／状态不会因本文件自动更新。
