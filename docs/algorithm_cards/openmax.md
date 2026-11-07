# OM01 OpenMax：MAV、距离尾部与未知激活重分配

[学习规范](../algorithm_learning_standard.md) · [分类装配总览](../classification_assembly_evolution.md#36-e06openmax-将未知输入从闭集置信中分离) · [覆盖清单 G24](../evolution_coverage.md) · [卡片索引](README.md) · [C08 Softmax](learning_components.md#c08-softmax) · [C21 交叉熵](learning_components.md#c21-多类交叉熵-cross-entropy)

<a id="om01"></a>
<a id="g24"></a>

## 任务、演进与覆盖状态

**任务：**给定一个已训练的已知类分类网络，以及这些类别的有标签训练样本，为新输入输出已知类与一个聚合未知类的归一化分数，并允许拒识。OpenMax 不命名新类别、不自动增量训练，也不把未知零件直接判成装配 NG；已知件错位、不可见和未知型号仍须走不同业务规则。它是监督分类之后的开放集校准，不是仅正常样本训练的单类异常检测。[cite:1](#source-1)

**演进定位：**闭集 Softmax 把全部概率分给已知类，未知输入仍可能高置信 → 在分类前激活向量空间引入正确训练样本的类中心与距离尾部模型 → 用 rank 加权的 Weibull 尾部分数重分配激活，再对包含未知类的新向量做 Softmax → 增加模型缓存、尾部拟合与拒识校准成本 → 仍有多模态、近邻未知、负激活和分布漂移问题。方法采用 CVPR **2016**；arXiv v1 为 **2015-11-19**；本卡读取的作者代码提交为 **2018-04-30**，三个日期不混用。[cite:1](#source-1)[cite:3](#source-3)[cite:10](#source-10)

核验日期：**2026-11-05**。本文件只有 **OM01** 一张卡，承接 **G24／07-06 E06**；libMR、Softmax、CAP 风险条件是该卡组件和推导，不另计卡。

| 字段 | 状态与范围 |
|---|---|
| L1 步骤 | 已核验：正确分类筛选、逐类逐 crop MAV、距离右尾、重分配与拒识；论文／代码差异另列 |
| L2 机制 | 已核验：全向量距离、CDF 方向、rank、激活守恒及反例；非概率直接转移 |
| L3 工程权衡 | 部分：参数、分组验证、排障和回退设计；无业务定参／真实未知类验收 |
| L4 数学 | 部分：均值、平移后 Weibull 似然、重分配及有限反例已推导；未重证任意网络上的完整风险保证 |
| L5 实现 | 部分：作者固定源码静态对应、参考伪代码和有限算术检查；未实现 OpenMax 或调用 libMR |
| 面试自测 | 答 5：逐层问题、答案要点、成立条件、易错点与证据入口 |
| 本仓库状态 | **未实现**；没有本卡分类权重、MAV／尾部模型、开放集数据集或推理接口 |
| 验证口径 | 原始资料与固定源码核验、文档检查、标量／向量算术微检查；不代表拟合、训练、图像推理或现场验证 |

本地 HOG-SVM 的候选重排以及 C08 的 Softmax 公式均不包含这条距离尾部链，不能反向升级为 OpenMax 已实现。

## L1 步骤：训练、校准、推理与拒识

### 输入输出和数据布局

令已知类别数为 \(K\)，crop 数为 \(C\)。网络对输入图像 \(x\) 输出分类 Softmax **之前**的全向量 \(v_c(x)\in\mathbb R^K\)。论文称其 penultimate layer，具体采用 AlexNet **FC8**，不是默认指 FC7 embedding，更不是后 Softmax 概率。作者数据为 `fc8[C,K]`，原实验 \(C=10,K=1000\)；这里的 channel 指 Caffe 的裁剪视图，不是 RGB 或卷积通道。[cite:1](#source-1)[cite:3](#source-3)

单 crop 的闭集概率为 \(p_c=\operatorname{softmax}(v_c)\)，作者 `scores` 为各 crop 概率的平均 \(\bar p=C^{-1}\sum_c p_c\)。已知标签 \(y\in\{1,\ldots,K\}\)；本卡数学把未知类记作 0，作者返回数组把未知类放在**末尾** `K`（已知类代码索引为 `0..K-1`）。任何接口转换必须显式保存该映射。[cite:3](#source-3)[cite:6](#source-6)

### 建模与停止条件

1. 先用有标签已知类数据训练／选择闭集分类器，冻结权重、类别次序、预处理、crop 和激活层。常规监督 CE 是基础网络的训练目标；OpenMax 原流程不把未知类当成第 \(K+1\) 个有监督输出重新反传训练。
2. 对每个类别 \(j\)，只保留 **真值为 j 且冻结分类器 top-1 也为 j** 的训练图像，集合记为 \(I_j\)。作者用图像 `scores.argmax()` 筛选，一幅图通过后，其全部 crop 都参与，不是每个 crop 再分别筛选。
3. 对每个 \((j,c)\)，求这些图像的完整 FC8 向量均值 \(\mu_{jc}\in\mathbb R^K\)。不是只平均第 j 个 logit，也不是把所有类别或所有 crop 混成一个中心。
4. 对同一 \(I_j\) 计算 \(d_{ijc}=D(v_c(x_i),\mu_{jc})\)，按距离取最大的 \(\eta\) 个，即**右尾／高尾**。每个 \((j,c)\) 新建 `libmr.MR()` 并 `fit_high(tail, len(tail))`，持久化完整模型状态。
5. 所有类／crop 模型有效、距离契约一致，才结束校准并发布。空正确集合、样本不足、全相同尾部、非有限距离或拟合失败均为 `INVALID_MODEL`，不是未知类别预测；不能用全类平均或伪造 Weibull 参数悄悄补齐。
6. 在独立已知验证与开发用未知样本上选 \(\eta,\alpha,\epsilon\)，冻结配置后再做独立验收。OpenMax 不存在测试时反复拟合到“拒掉此图”为止的循环。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)

论文／作者默认尾长为 20；作者 Python 切片在 \(n_j<\eta\) 时实际使用全部可用距离，但这不证明少量点拟合有效。本卡参考契约更严格：至少 \(\eta\ge2\)、\(n_j\ge\eta\)，且尾部有足够数值变化；下界 2 只是接口最低防护，不是统计充分样本量。达不到时停止发布，或另行标记并验证一种降级方案。

### 新图像的完整计算

1. 同样提取 `fc8[C,K]` 和图像级 `scores[K]`。对 `scores` 降序排列，获得共享 rank；单 crop 时 Softmax 与 logit 排名相同，多 crop 时平均概率排名不一定等于平均 logit 排名。
2. 前 \(\alpha\) 类依名次获得权重 \(a_{s(r)}=(\alpha+1-r)/\alpha\)，\(r=1,\ldots,\alpha\)；其余权重为 0。这是**固定作者代码口径**，其与印刷算法框的差异见 L4。
3. 每个 crop、候选类求全向量到该类 MAV 的距离，再读 \(F_{jc}=\mathrm{MR}_{jc}.\mathrm{w\_score}(d_{jc})\)。该分数随高尾距离增大而增大，语义是相对该类中心的离群程度，不是已校准的 \(P(y=0\mid x)\)。
4. 令 \(q_{jc}=a_jF_{jc}\)，修改已知激活 \(\tilde v_{jc}=v_{jc}(1-q_{jc})\)；未知激活 \(u_c=\sum_j v_{jc}q_{jc}\)。负值保留，不能裁成 0 后仍称逐值复现。
5. 对每个 crop 的 \([\tilde v_{1c},\ldots,\tilde v_{Kc},u_c]\) 做一次 \(K+1\) 类稳定 Softmax，再平均 crop 概率得到 \(\bar P\)。**不能先平均修改后的激活再 Softmax**。
6. 若未知类为最大项，拒为 `UNKNOWN_CLASS`；否则若获胜已知类概率 \(<\epsilon\)，拒为 `LOW_CONFIDENCE`；否则返回该已知类。阈值相等时按论文严格小于规则接受；本卡实现设计对最大值并列选择 `AMBIGUOUS` 复检，不冒充作者既有 tie 规则。输入非有限／维度不符是 `INVALID_INPUT`。[cite:1](#source-1)[cite:6](#source-6)

### 一个可逐步走查的输入

作者 README 给出的真实数据接口是 `data/train_features/n01440764/n01440764_9981.JPEG.mat`，其 FC8 形状为 `(10,1000)`；对应类别原有 1300 幅训练图，正确分类 1224 幅，预计算距离形状为 `(10,1224)`。复现者应从这 1224 个完整向量逐 crop 求 MAV，取每个 crop 的最大 20 距离拟合，而不是从 1000 个类别 logit 中选 20 个。然后按前述六步处理待测 `.mat` 文件，输出长度 1001 的向量。上述数字是**作者说明，不是本仓库重算结果**，本次未下载特征包。[cite:3](#source-3)

为明确重分配算术，另设一组**代数检查输入而非训练／性能数据**：单 crop，\(v=(4,2,1)\)，\(\alpha=2\)，拟合器输出暂代为 \(F=(0.75,0.50,0)\)。则 \(a=(1,0.5,0)\)，\(q=(0.75,0.25,0)\)，\(\tilde v=(1,1.5,1)\)，\(u=3.5\)。四类 Softmax 约为 \((0.0632,0.1041,0.0632,0.7695)\)，应由未知类获胜而拒识。这里并未声称这些 F 来自实际 `fit_high`。

## L2 机制：必要性、因果关系与反例

| 机制 | 为什么需要 | 删除／替换后的后果 |
|---|---|---|
| 正确分类训练子集 | 刻画模型已学会的该类激活模式；真值与预测同时满足 | 混入误分类会污染中心；只按预测分组还会引入其他真值类；但正确筛选也可能遗漏困难合法模式 |
| 完整 AV 的 MAV 与距离 | 利用“相关类别一起响应”的模式，而非孤立最大分数 | 仅看第 j 个分量会忽略其他 K−1 维的异常响应；高 Softmax 不等于靠近训练分布 |
| 逐类尾部模型 | 不同类别激活尺度／离散程度不同，尾部关注罕见大距离 | 共用半径或拟合全体距离改变统计问题；Weibull 近似不是任意距离总体必然严格服从 Weibull |
| rank 权重 | 重点修改最可能的少数类别，降低大量低位激活的影响 | 给所有类权重 1 是另一变体；按离 MAV 最近的类排名也不是作者的高分类分数排名 |
| 激活重分配后统一归一化 | 为未知类建立 pseudo-logit，已知＋未知概率最终和为 1 | 对旧概率做 \(p'_j=p_j(1-q_j),p'_0=\sum p_jq_j\) 是另一算法，不等于 OpenMax |
| 未知获胜＋置信阈值 | 分开“模式像未知”与“已知间仍不确定” | 只取已知 argmax 会删掉未知决策；只阈值 Softmax 又退回闭集置信拒识基线 |

关键不变量是 \(\sum_j\tilde v_{jc}+u_c=\sum_jv_{jc}\)，不是保留原来的概率质量或指数和。即使 \(a_j=0\)，未选中类的 **logit** 不变，其最终概率仍因分母改变而变化，甚至上升。若所有 \(q=0\)，未知 logit 为 0，仍贡献 \(e^0=1\)，因此输出并不严格退回原 K 类 Softmax；关闭 OpenMax 必须旁路整个 \(K+1\) 类归一化。[cite:6](#source-6)，这些不变量是公式直接推论。

## L3 工程权衡：尺度、拒识验证与排障

### 参数与选择次序

| 参数／契约 | 单位、起点及依据 | 代价与边界 |
|---|---|---|
| 激活层／权重／crop | FC8 为 K 维分类激活；原配置 10 crops | 换 FC7、换骨干、温度或 crop 都需重新建 MAV／距离／尾部和阈值，不是只改维度 |
| 尾长 \(\eta\) | 样本个数，不是尾比例；原补充比较 10、20、25、30、40、50，选 20 | 太少方差大；太多纳入主体、增加已知拒绝；这些是原实验现象，不是跨域最优值 |
| rank 范围 \(\alpha\) | 整数 \(1\le\alpha\le K\)；原实验选 10，也比较 5 | 增大后低 rank 负 logit 可参与；并不保证未知概率单调增大 |
| `eucos` 距离 | \(D(v,\mu)=\|v-\mu\|_2/200+1-\frac{v^T\mu}{\|v\|\|\mu\|}\) | 200 是作者激活尺度折中，不是像素／毫米单位；欧氏项保幅值、余弦项看方向 |
| 拒识阈值 \(\epsilon\) | 无量纲，作用于获胜已知概率；\(0\le\epsilon\le1\) | 固定模型下增大阈值只会扩大置信拒绝集合，不改变未知 argmax；不得称其真实缺陷率 |
| libMR 平移与数值参数 | 本提交默认 `translate_amount=10000`；独立于 OpenMax rank 的 `alpha` | 不能照搬其他 libMR 版本的默认平移；库内 `alpha=5.0` 用于参数置信区间，并非 top-5 |

参数依据来自原补充、作者距离源码和该提交自带 libMR。[cite:2](#source-2)[cite:7](#source-7)[cite:8](#source-8)

推荐次序：**已知／未知定义及划分 → 标签表和预处理 → 正确样本数／MAV 与距离分布 → 距离实现及尺度 → 尾拟合有效性 → 联合选择 \(\eta,\alpha\) → \(\epsilon\) 操作点 → 缓存与加速**。不通过调阈值掩盖类别错位或单位不一致。

### 训练与验证分账

- 基础网络监督训练、MAV／尾部校准、超参数选择、最终验收分别记录来源。MAV 与尾部使用训练样本是原方法设计，不把重复使用同件图像的训练结果冒充泛化效果。
- 原文用开发用已知图像和采样开放集图像校准，并称未用 fooling 图像选阈值；它明确不能穷举 unknown unknowns。本卡工程建议按实物／批次分组，开发未知类别与最终未知类别进一步隔离；此更严格协议不冒充原文已完整执行。[cite:1](#source-1)
- 验证已知类时保留全部样本，包括闭集误分类者；“仅正确分类”是建模筛选，不是评测过滤。分别报告闭集 top-1、已知正确接收率、已知误拒率、未知误接收率及原因分布。
- 如用未知分数画 AUROC，明确未知为正类、分数方向为 \(\bar P_0\) 越大越未知；如用 F1，写清正类、计数定义及类别混合比例，不只报一个无分母的指标。\(\bar P_0\) 曲线不替代同时包含置信阈值的最终决策验收。
- 只有已知验证时可以选已知误拒操作点，不能推断未知召回。与最大 Softmax 阈值基线应使用同一骨干、划分和开发预算；同一测试集反复择阈值后不再是盲验。现场无通用漏放率或误报率保证。

### 症状与证据

| 症状 | 优先检查 | 回退／限制 |
|---|---|---|
| 某类几乎都拒绝 | \(n_j\)、正确筛选偏差、训练／新批距离尾部和模型有效性 | 补困难合法样本，重新冻结全链；多中心是扩展，不是原单 MAV |
| 近似未知被高置信接收 | 高 rank 类距离很小，细粒度未知与已知共享 AV | 增强可分表示或加入复检；补充原文已有此类失败，不能宣传全面抗对抗 |
| 切换 euclidean 后异常高接收 | 预处理欧氏距离未除 200，而推理函数除 200 | 固定提交有此静态不一致；重新统一距离并重拟合，不能沿用不匹配缓存 |
| 调 CLI 参数输出不变 | `main()` 是否把解析值传入实际调用 | 本提交 tail／rank／distance 参数未完整传递，须检查生效配置而非相信帮助文本 |
| 未知激活负值或概率反常 | 参与重分配的 logit 符号、rank 及 crop 排名 | 负值是原公式允许的结果；先按原语义解释，裁剪／平移是需要重校准的新变体 |
| NaN／概率和错误 | 零范数余弦、无效 MR 返回 −9999、指数溢出、未知索引 | 返回无效状态；不把 NaN 当 0，不把无效尾分数裁为“完全正常” |

## L4 数学推导：均值、尾部、重分配与风险条件

### MAV 与距离模型

正确训练集合 \(I_j=\{i:y_i=j,\arg\max_k\bar p_k(x_i)=j\}\)，\(n_j=|I_j|>0\)。逐 crop 中心：

\[
\mu_{jc}=\frac1{n_j}\sum_{i\in I_j}v_c(x_i)
=\arg\min_m\sum_{i\in I_j}\|v_c(x_i)-m\|_2^2.
\]

对目标求导得 \(2n_jm-2\sum_iv_c(x_i)=0\)，Hessian 为 \(2n_jI\)，故均值唯一。它是平方欧氏目标的解，**不是宣称 arithmetic MAV 同时最小化 eucos**。单中心假设可能把多簇中间的低密度区当典型，正确训练筛选不能消除这个建模偏差。

单 crop 的常规分类训练目标可写 \(L_{CE}=-\sum_i\log\operatorname{softmax}(v(x_i))_{y_i}\)，其梯度为 \(p_j-\mathbf1[j=y_i]\)。该目标只区分已知标签，不约束未知在 AV 中远离所有 MAV；OpenMax 后拟合没有改写这条梯度。不能由低 CE 推出良好的开放集几何。

### FitHigh 的平移、CDF 和似然

本提交 `MR()` 默认不丢最高分，令右尾排序结果为 \(d_1,\ldots,d_\eta\)，其最小值为 \(b\)。`FitHigh` 设置 `sign=+1`，对尾部执行

\[
t_i=d_i+T-b>0,\qquad T=10000.
\]

随后对 \(t_i\) 拟合二参数 Weibull，保存 scale \(\lambda>0\)、shape \(\kappa>0\) 及平移状态。新距离 \(d\) 用同一变换 \(t=d+T-b\)；默认 `complement_reject` 分支的 `w_score` 返回 CDF 而非 survival：

\[
F(d)=\begin{cases}0,&t\le0,\\1-\exp[-(t/\lambda)^\kappa],&t>0.\end{cases}
\]

有效位置参数可记 \(\tau=b-T\)，使 \(t=d-\tau\)。**τ 是距离轴平移，不是 K 维 MAV**；更不能把 \(\|v-\tau\|\) 当距离定义。MAV、尾部最小值、T、scale／shape、sign／fit-type 都有独立作用。库源码 `parmhat[0]` 是 scale，`parmhat[1]` 是 shape；`fit_high` 的 Python 包装不返回 C++ 拟合状态，之后必须检查 `mr.is_valid`。无效对象的 `w_score` 可返回 −9999，不是概率。[cite:5](#source-5)[cite:8](#source-8)

固定平移后，Weibull 密度给出对数似然：

\[
\ell(\kappa,\lambda)=\eta\log\kappa-\eta\kappa\log\lambda
+(\kappa-1)\sum_i\log t_i-\sum_i(t_i/\lambda)^\kappa.
\]

令 \(\partial\ell/\partial\lambda=0\)，得到

\[
\lambda^\kappa=\frac1\eta\sum_i t_i^\kappa,\qquad
\frac1\kappa+\frac1\eta\sum_i\log t_i
-\frac{\sum_i t_i^\kappa\log t_i}{\sum_i t_i^\kappa}=0.
\]

第二式通常数值求根；应在 log 域计算幂和，检查收敛与参数正值。全相同的 t 不能稳定辨识有限形状，极小离散度也会病态。这是**条件于所选尾部及固定平移的二参数似然推导**，不是重新实现 libMR 的求解／置信区间，也不是严格的删失尾似然。样本共享估计 MAV、尾部选择及有限样本相关性，均不允许把渐近 EVT 当作任意小样本精确分布定理。用 `scipy.stats.weibull_min.fit` 默认自由 location 拟合整个距离集，不会自动与这里等价。

### Rank、激活守恒与统一 Softmax

采用作者代码的 \(a_{s(r)}=(\alpha+1-r)/\alpha\)，定义

\[
q_{jc}=a_jF_{jc},\quad \omega_{jc}=1-q_{jc},\quad
\tilde v_{jc}=v_{jc}\omega_{jc},\quad
u_c=\sum_j(v_{jc}-\tilde v_{jc})=\sum_jv_{jc}q_{jc}.
\]

因此 \(\sum_j\tilde v_{jc}+u_c=\sum_jv_{jc}\)，但指数不保持线性，归一化必须重新计算：

\[
P_{jc}=\frac{e^{\tilde v_{jc}}}{e^{u_c}+\sum_ke^{\tilde v_{kc}}},\quad
P_{0c}=\frac{e^{u_c}}{e^{u_c}+\sum_ke^{\tilde v_{kc}}},\quad
\bar P_j=\frac1C\sum_cP_{jc}.
\]

由非负性与逐 crop 归一化，\(\sum_{j=0}^K\bar P_j=1\)。这仅证明合法归一化分数，不证明频率意义的概率校准。[cite:6](#source-6)

**印刷公式与源码不可混写：**CVPR 2016 算法 2 行 3 印为 rank 因子 \((\alpha-i)/\alpha\) 乘生存函数形式的指数，且表达式使用 τ；正文 §2.3 却明确要求距离到 MAV 的 **Weibull CDF**。作者代码采用 `1 - w_score * ranked_alpha`，rank 从 1 到 \(1/\alpha\)，而非 \((\alpha-1)/\alpha\) 到 0。§2.4 的“CDF 递增，因此 \(1-\omega\) 递减”也与该变量约定不一致。本卡以已核的作者代码实现语义展开，**保留出版差异，不把手工改写公式称为原文逐字等式或作者勘误**。特别地，\(\alpha=1\) 在代码中仍完整修正第一名，不会因 rank=0 而失效。[cite:1](#source-1)[cite:6](#source-6)

### 负 logit 和平移反例

当 \(v_j<0\) 时，\(v_j(1-q_j)\) 向 0 移动，即数值**增大**；转给未知的 \(v_jq_j\) 为负。固定其他量，对一个 \(q_j\) 求导，令 \(P_j,P_0\) 为同一 crop 的重分配后概率：

\[
\frac{\partial P_0}{\partial q_j}=P_0v_j(1-P_0+P_j).
\]

有限激活时括号为正，因此负 logit 的尾部分数增大反而降低未知概率。只有固定其余变量且 \(v_j\ge0\) 时，才能说该项的增加不降低未知概率；真实输入变化还会同时改变 v、rank、距离，不能把这个偏导推广为全局单调。

代数反例：\(K=2,\alpha=2,v=(-1,-2),F=(1,1)\) 给 \(\tilde v=(0,-1),u=-2\)，结果 \((P_1,P_2,P_0)\approx(0.6652,0.2447,0.0900)\)。即使极端尾分数为 1，未知也不获胜。F=1 在此是饱和／极限检查，不冒充一组实测拟合值。

原 K 类 Softmax 对 \(v\mapsto v+c\mathbf1\) 不变，OpenMax 不具有这一自由度：即使暂时固定 q，已知修改量是 \(c(1-q_j)\)，未知修改量是 \(c\sum_jq_j\)，不是所有 \(K+1\) 分量共同加一个常数；实际 eucos 的余弦项也可能变化。因此“先减原 logit 最大值再做重分配”会改变算法。**稳定化只能在完成重分配后，对全部 K+1 个 logit 共同减最大值。** ReLU、负值截断、平移至正区间都属于需重拟合／重验的变体。

### 开空间风险：可以推导的部分和不保证的部分

在固定 AV 特征空间内，令有限训练集附近的球并集以外为开空间 \(\mathcal O\)，\(S_o\) 为有限研究区域；对可测非负识别函数 \(f\) 且分母为正，风险定义为

\[
R_{\mathcal O}(f)=\frac{\int_{\mathcal O}f(z)\,dz}{\int_{S_o}f(z)\,dz}.
\]

它是“在离已知数据远处赋正标签”的相对体积／质量，不是未知样本分布上的误接收概率。2016 定理引用的 2015 Theorem 1 依赖有限、非负加权的 CAP（紧致衰减）模型及合适阈值，Theorem 2 讨论**线性**特征变换；阈值存在不等于该阈值同时有良好已知召回。[cite:1](#source-1)[cite:9](#source-9)

对有效 Weibull 生存函数 \(S(d)=1-F(d)\)，取 \(0<\delta<1\)，则

\[
S(d)>\delta\quad\Rightarrow\quad
d<\tau+\lambda[-\log\delta]^{1/\kappa}.
\]

若该上界为正且距离含正系数欧氏项，则超水平集被有界球包含；例如 eucos 的非负余弦项保证 \(\|v-\mu\|_2\le200D(v,\mu)\)。有限中心组合可形成紧致的阈值接受区域。这说明 **距离生存模型** 的衰减基础；不能把对 MAV 的紧致性不加检查地等同于对全部训练邻域零风险。纯余弦在同方向射线 \(v=t\mu,t>0\) 上恒为 0，不限制幅值远离；原补充明确指出纯 cosine 不具有所需 CAP 性质。[cite:2](#source-2)

对完整代码还须检查而不能默认成立：当 \(0<a_j<1\) 时 \(\omega_j\to1-a_j\)，不是每项都衰减至 0；rank 依赖输入且会切换；负 logit 使加权激活不再是非负 CAP 组合；Softmax 分母耦合所有类；任意非线性图像→AV 映射也不是 2015 线性定理的直接对象。

具体地，在无界带符号 AV 域取 \(v(t)=(-t,-2t)\)、固定有限 MAV、有效带欧氏距离尾模型、\(\alpha=2\)。随 \(t\to\infty\)，\(F_1,F_2\to1\)，\(\tilde v_1\to0,\tilde v_2\sim-t,u\sim-2t\)，故 \(P_1\to1,P_0\to0\)。任意 \(\epsilon<1\) 的普通决策仍可接受离所有 MAV 任意远的向量。这个反例否定“任意带符号 logits 都无条件远处衰减”的扩张解释；它不宣称这些向量必为原 AlexNet 可达输入，也不冒充完整风险积分反证。

所以本卡记录原文的 AV 空间风险主张，同时将完整保证保持为**部分核验**。非负性、距离远离控制、阈值、组合方式和可达特征域必须逐项证明；不能由单个 CDF 单调性推出最终 \(P_0\) 全局单调，更不能推出全部未知件／对抗样本必拒。原补充已有细粒度未知错接收与已知误拒示例。[cite:2](#source-2)

### 复杂度与误差边界

设正确校准图像总数 \(M=\sum_jn_j\)，AV 维度 \(D=K\)。不计网络前向，MAV 与训练距离约 \(O(MCD)\)；逐类完整排序约 \(O(C\sum_jn_j\log n_j)\)，尾模型拟合另依赖迭代次数。持久化中心 \(O(KCD)\)，尾模型 \(O(KC)\)，不必部署全部训练距离。

作者推理遍历全部 K 类，距离约 \(O(CKD)\)，类别排序 \(O(K\log K)\)。只算前 α 类的距离可降至 \(O(C\alpha D)\)，另加 \(O(CK)\) 归一化；仅在所有模型已有效、rank／距离不变时才与原有限值计算等价。近似距离、量化、中心压缩或修改裁剪顺序均需对照。尾部估计没有本卡可证的通用误拒／误接收误差界，复杂度减少不构成检测精度保证。

## L5 实现：固定源码对应、伪代码与数值边界

### 源码对应和版本差异

作者仓库统一锁定 `abhijitbendale/OSDN@c13b7697c52f6748dc29f4387deced4daa6d6239`。本卡不把移动的 master 当版本，也不把该后续提交当 2016 发表快照。[cite:3](#source-3)

| 固定提交内路径／函数 | 核验的职责 | 复现时必须保留的限制 |
|---|---|---|
| `preprocessing/MAV_Compute.py::compute_mean_vector` | 按真值目录与 `scores.argmax()` 筛选，逐 crop 平均完整激活 | 空集直接访问 `correct_features[0]`；参考实现应显式拒绝 |
| `preprocessing/compute_distances.py::compute_channel_distances` | 正确样本到 MAV；保存 eucos／euclidean／cosine | 保存的纯欧氏是未缩放 L2；eucos 为 L2/200＋cosine |
| `evt_fitting.py::weibull_tailfitting` | 类别／crop 右尾切片；新 MR；`fit_high` | 原代码没有有效性检查；`tailsize=0` 的 `[-0:]` 会变成全数组，不是空尾 |
| `openmax_utils.py::compute_distance` | 推理距离 | eucos 与训练一致，但纯 euclidean 推理为 L2/200，和上述训练输出不一致 |
| `compute_openmax.py::recalibrate_scores` | 平均概率排名、rank 权重、逐类 `w_score`、激活差 | 有符号激活不截断；硬编码 10 crops／1000 类 |
| `computeOpenMaxProbability` | 每 crop 未知激活先求和再 exp；最后平均概率 | 原实现直接 exp，未减最大值；不是对每类剩余量分别 exp 后求和 |
| `compute_openmax.py::main` | 加载数据和调用上述函数 | 已解析的 `weibull_tailsize` 未用，调用仍传常量 20；`alpha_rank`／`distance` 未传入校准调用，实际走默认值 |
| `libMR/libmr.pyx`、`MetaRecognition.cpp/.h`、`weibull.c` | Python→C++ 包装、FitHigh 平移、CDF、Weibull 求解 | 默认 T=10000；保存 sign／类型／平移；检查有效性；未编译运行此版本 |

上述是外部代码**静态事实**，不是已执行 bug 复现。参考伪代码选择训练／推理共享 eucos 并显式传参；扩展 K、C、错误退出和稳定 Softmax 都标为工程实现设计，不宣称与原脚本逐条执行完全相同。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)

### 可实现的参考伪代码

`D` 是同一个冻结距离函数；`new_MR` 使用已锁版本的新对象；`is_valid` 为属性。`assert` 在此表示显式契约检查，生产实现须返回带原因错误，不依赖可被关闭的断言。排序并列固定按类别 ID；最终最大值并列复检，是本卡策略。

```text
CALIBRATE(train, frozen_net, labels, eta, D):
    require integer(eta) and eta >= 2
    grouped = collect_full_AVs_where_truth_equals_image_top1(train, frozen_net)
    for j in labels:
        require len(grouped[j]) >= eta
        for c in crops:
            mu[j,c] = mean(grouped[j][:,c,:], axis=0, precision=float64)
            ds = [D(v[c,:], mu[j,c]) for v in grouped[j]]
            require all_finite_nonnegative(ds)
            tail = sorted(ds)[-eta:]
            require numerically_nonconstant(tail)
            mr[j,c] = new_MR(translate_amount=10000)
            mr[j,c].fit_high(tail, len(tail))
            require mr[j,c].is_valid
            require valid_monotone_scores_on_distance_grid(mr[j,c])
    return frozen_bundle(mu, mr, net_version, labels, crops, D, eta)

PREDICT(x, bundle, alpha, epsilon):
    require integer(alpha) and 1 <= alpha <= K and 0 <= epsilon <= 1
    V = extract_frozen_AV(x)
    require shape(V) == (C,K) and all_finite(V) and valid_bundle(bundle)
    closed = mean([stable_softmax(V[c,:]) for c in crops], axis=0)
    order = stable_descending_sort(closed, tie_key=class_id)
    a = zeros(K)
    for r in 1..alpha:
        a[order[r-1]] = (alpha + 1 - r) / alpha
    crop_probs = []
    for c in crops:
        q = zeros(K)
        for j in order[:alpha]:
            d = D(V[c,:], mu[j,c])
            require finite(d) and d >= 0
            F = mr[j,c].w_score(d)
            require finite(F) and 0 <= F <= 1
            q[j] = a[j] * F
        known = V[c,:] * (1-q)
        unknown = accurate_sum(V[c,:] * q)
        z = concatenate(known, [unknown])
        require all_finite(z)
        crop_probs.append(stable_softmax(z))
    P = mean(crop_probs, axis=0)
    require finite_normalized_probabilities(P)
    if maximum_is_tied(P): return AMBIGUOUS, P
    winner = argmax(P)
    if winner == K: return UNKNOWN_CLASS, P
    if P[winner] < epsilon: return LOW_CONFIDENCE, P
    return labels[winner], P
```

训练退出是完整 bundle 验证成功；任何 crop／类缺模型时不发布不完整结果。推理没有迭代停止问题，但应设置网络／拟合调用超时；超时和配置不匹配均进入复检，不默认为已知。

### 数值边界与必须测试的分支

| 输入／边界 | 正确处理或待验证断言 |
|---|---|
| \(n_j=0\)、\(n_j<\eta\)、尾部全等／近常量 | 拒绝发布；不得复制距离凑样本数；拟合成功标志还需有限参数与分数检查 |
| cosine 零向量／零 MAV | 余弦未定义，返回无效；若设计 L2 回退须训练、推理、校准同步，不能只在预测加 epsilon |
| \(\alpha=1\)／\(\alpha=K\)、非法 0 或 K+1 | 第一名权重必须 1；最后选中项为 1/α；非法参数拒绝 |
| F=0／F=1、F<0／NaN | 合法端点按公式；无效值报错；−9999 不能用 clip 掩盖 |
| 负 logit、全零 logit | 允许负未知激活；全零且合法距离时 K+1 类均匀，按并列策略复检 |
| 很大有限 logit | 对完成重分配的 K+1 向量共同减 max；先检查求和是否已溢出；稳定 softmax 不修复此前 Inf |
| 很小／很大 \((t/\lambda)^\kappa\) | CDF 稳定参考用 `-expm1(-h)`；大 h 饱和为 1，小 h 防相消；不要擅改已锁 libMR 后声称逐位等价 |
| q=0 | 未知仍贡献 exp(0)，不是原 K 类 Softmax；单独旁路测试关闭路径 |
| crop 聚合 | 验证 mean(softmax(z_c)) 与 softmax(mean(z_c)) 一般不同；MAV 布局／crop 次序不能错位 |
| 阈值相等／最大值并列 | 原式严格 `< epsilon`；本卡并列复检优先，记录不是库默认行为 |
| 浮点守恒 | 容差按激活总绝对量缩放；同时检查未知有符号和，不强制每项非负 |
| 序列化／加速 | 完整 MR 状态往返、标量与批量、全类与 top-α 路径数值对照；真实 libMR 测试仍未执行 |

### 有限算术检查器

以下 Python 只用标准库，检查**给定 q 的重分配与归一化**。它不是 OpenMax 实现，不拟合 Weibull，不创建分类数据，不验证 libMR ABI 或模型效果；其中向量是明确指定的代数反例。

```python
import math

def stable_softmax(z):
    if not z or not all(math.isfinite(t) for t in z):
        raise ValueError("invalid logits")
    m = max(z)
    e = [math.exp(t - m) for t in z]
    den = math.fsum(e)
    return [t / den for t in e]

def redistribute(v, q):
    if len(v) != len(q) or not v:
        raise ValueError("shape")
    if not all(math.isfinite(t) for t in v + q):
        raise ValueError("nonfinite")
    if not all(0 <= t <= 1 for t in q):
        raise ValueError("invalid tail weight")
    z = [t * (1 - w) for t, w in zip(v, q)]
    z.append(math.fsum(t * w for t, w in zip(v, q)))
    return z, stable_softmax(z)

def run_checks():
    z, p = redistribute([4., 2., 1.], [.75, .25, 0.])
    assert z == [1., 1.5, 1., 3.5]
    assert math.isclose(math.fsum(z), 7.)
    assert p.index(max(p)) == 3 and math.isclose(math.fsum(p), 1.)
    z, p = redistribute([-1., -2.], [1., .5])
    assert z == [0., -1., -2.] and p[0] > p[2]
    _, farther = redistribute([-10., -20.], [1., .5])
    assert farther[0] > .9999 and farther[2] < p[2]
    _, zero_q = redistribute([4., 2., 1.], [0., 0., 0.])
    assert zero_q[-1] > 0 and zero_q[0] != stable_softmax([4., 2., 1.])[0]
    assert stable_softmax([0., 0., 0.]) == [1/3, 1/3, 1/3]
    for alpha in [1, 2, 10]:
        a = [(alpha + 1 - r) / alpha for r in range(1, alpha + 1)]
        assert a[0] == 1 and a[-1] == 1 / alpha
    _, large = redistribute([1000., 999.], [.5, .25])
    assert all(math.isfinite(t) for t in large)
    assert math.isclose(math.fsum(large), 1.)
    p1, p2 = stable_softmax([4., 0.]), stable_softmax([0., 0.])
    mean_p = [(a+b)/2 for a, b in zip(p1, p2)]
    assert abs(mean_p[0] - stable_softmax([2., 0.])[0]) > .1
    for q in [[-9999., 0.], [float("nan"), 0.], [1.1, 0.]]:
        try:
            redistribute([1., 2.], q)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid q accepted")
    print("OM01 arithmetic checks passed; no libMR fitting or network inference")

run_checks()
```

## 面试自测：贯穿 L1–L5 的五问五答

| 层级／问题 | 答案要点 | 成立条件与易错答案 | 证据入口 |
|---|---|---|---|
| L1：给你 `(10,1000)` FC8 和标签，如何从训练得到一个拒识结果？ | 真值＋图像 top-1 同类筛选；每类每 crop 求完整 MAV；取类内距离右尾 FitHigh；图像排名，crop 内重分配并 Softmax，平均后未知获胜或低置信拒识 | 不是平均单个 logit，不把 crop 当类，不在测试重新拟合；未知在代码末尾 | L1、[cite:4](#source-4)[cite:6](#source-6) |
| L2：为什么不能把旧 Softmax 概率的一部分直接搬给未知类？ | OpenMax 保持的是带符号激活和；exp 非线性，分母必须重算；未改 logit 的类别概率也会变；q=0 时未知仍有 exp(0) | \(p(1-q)\) 是另一方法；“未知分数就是 Weibull CDF”也错 | L2、L4 重分配 |
| L3：换骨干后未知误接收增多，先改 tail=20 还是 ε？ | 先查类别次序、FC8 对应层、crop、距离尺度与正确样本覆盖，再查拟合／开发未知，联合选 η、α 后定 ε；报告已知误拒和未知误接收 | 20／10 是原实验选择；仅已知验证无法优化真实未知召回；代码 CLI 参数未必生效 | L3、[cite:2](#source-2)[cite:7](#source-7) |
| L4：CDF 随距离增大，是否就证明未知概率单调、所有未知必拒？ | 推导 \(u=\sum vq\) 和 \(\partial P_0/\partial q_j=P_0v_j(1-P_0+P_j)\)；负 logit 反号；CAP 需非负衰减模型、有限组合、合适距离与阈值，且风险在 AV 空间 | 不用单一 CDF 单调性替代完整复合函数证明；纯 cosine 射线和负 AV 反例都要会举 | L4 风险节、[cite:9](#source-9) |
| L5：公式正确却输出 NaN／概率错误，怎样定位和优化？ | 查零范数、MR.is_valid／−9999、scale/shape／平移状态、eucos 契约、末尾未知索引；重分配后 K+1 共减 max；先验算守恒与 crop 聚合，再缓存尾模型、只算 top-α 距离 | 不先平移原 v，不裁负未知，不把 mean-softmax 顺序交换；微检查不等于真实拟合通过 | L5 检查表、[cite:5](#source-5)[cite:8](#source-8) |

## 来源、版本与获取深度

以下均于 **2026-11-05** 联网读取。论文、补充、作者代码是同一研究链，不能冒充三项独立复现实验。机制采用 **[SINGLE-SOURCE：原作者链]**；论文与代码可交叉核对实现语义，但不据此声称性能独立验证。数学检查是本卡推论；未列出跨算法性能排名。

- <a id="source-1"></a>**[cite:1] Bendale、Boult，Towards Open Set Deep Networks，CVPR 2016，1563–1572。** [CVF 书目](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)、[会议原文 PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)。已读方法正文、算法 1／2、§2.3／2.4 风险主张及 §3 训练／测试协议；初始网页提取截断，后用内存 PDF 文本提取补取算法 2、定理及实验协议命中页。不是逐图逐表完整实验复核；印刷 CDF／rank／ω 差异按正文记录。
- <a id="source-2"></a>**[cite:2] 同文 CVPR 2016 Supplemental。** [官方 PDF](https://openaccess.thecvf.com/content_cvpr_2016/supplemental/Bendale_Towards_Open_Set_2016_CVPR_supplemental.pdf)。已读参数分析、纯 cosine 的 CAP 限制、细粒度未知及已知样本失败文字；未复算图中性能。
- <a id="source-3"></a>**[cite:3] 作者 OSDN 仓库版本。** [固定提交](https://github.com/abhijitbendale/OSDN/commit/c13b7697c52f6748dc29f4387deced4daa6d6239)、[固定 README](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/README.md)。GitHub API 实读提交 SHA 与 UTC 日期 `2018-04-30T00:08:32Z`；README 全部相关流程已读。它是后续快照，不称原发表时源码；README 预计算样本数量未本地复算，数据包未下载。
- <a id="source-4"></a>**[cite:4] 固定预处理源码。** [MAV_Compute.py](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/preprocessing/MAV_Compute.py)、[compute_distances.py](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/preprocessing/compute_distances.py)。已读正确图像筛选、crop 均值和三种距离实现；未跑 Caffe 或预处理脚本。
- <a id="source-5"></a>**[cite:5] 固定尾部拟合调用。** [evt_fitting.py](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/evt_fitting.py)。已读 `weibull_tailfitting`／`query_weibull` 完整函数，确认右尾、逐类逐 crop 新 MR 和调用参数。
- <a id="source-6"></a>**[cite:6] 固定重分配实现。** [compute_openmax.py](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/compute_openmax.py)。已读排名、rank 权重、带符号差值、未知指数、概率平均、常量和 main 参数传递；只有源码证据，未运行原 Python 2 脚本。
- <a id="source-7"></a>**[cite:7] 固定推理距离。** [openmax_utils.py](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/openmax_utils.py)。已读 eucos、euclidean、cosine 分支，与 [cite:4] 静态交叉发现纯欧氏尺度不一致；不能据此断言作者预计算包一定含同样错配。
- <a id="source-8"></a>**[cite:8] 同一提交内置 libMR，不替换成 PyPI 或另一仓库的最新版。** [libmr.pyx](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/libMR/libmr.pyx)、[MetaRecognition.cpp](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/libMR/MetaRecognition.cpp)、[MetaRecognition.h](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/libMR/MetaRecognition.h)、[weibull.c](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/libMR/weibull.c)。已读包装、默认值、FitHigh／EvtGeneric、CDF／W_score、有效性和序列化相关段，以及 `weibull_cdf`；未完整审计 Weibull 求解器／置信区间。Git tree 确认不是 submodule；`MetaRecognition.cpp` blob 为 `b423be078bb77d697369eab719e592bd0e13302f`，`libmr.pyx` blob 为 `e0d588b2e2c27d2c4927926990f75e2501d3cb9c`。未编译、安装或调用 libMR。
- <a id="source-9"></a>**[cite:9] Bendale、Boult，Towards Open World Recognition，CVPR 2015，1893–1902。** [书目](https://openaccess.thecvf.com/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html)、[原文 PDF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Bendale_Towards_Open_World_2015_CVPR_paper.pdf)。已读命中页中的 §4 Theorem 1／2、非负 CAP 组合、线性变换和阈值泛化警告；作为 2016 引用的理论前提，不声称完整重证其所有测度论步骤或更早 CAP 原始论文。
- <a id="source-10"></a>**[cite:10] arXiv:1511.06233v1，2015-11-19。** [指定版本 HTML](https://arxiv.org/html/1511.06233v1)。已读版本页及定向提取 §2.3／2.4／§3、补充参数段，辅助确认文本；正式方法日期以 [cite:1] 为准，不混成“2016 v1”。

**许可边界：**固定 OSDN 文件头允许附条件再分发，不表示内置 libMR 相同。实读 [COPYRIGHT_Libmr.txt](https://github.com/abhijitbendale/OSDN/blob/c13b7697c52f6748dc29f4387deced4daa6d6239/libMR/COPYRIGHT_Libmr.txt) 含非商业／商业用途限制，部署前需核准实际采用版本及授权；不把本卡静态分析当法律许可或推断所有后续 libMR 版本均同一许可。本次仅编写解释与独立算术检查，未复制库到仓库。

## 验证记录与 G24 承接边界

本次已执行本卡标准库算术检查器，覆盖 rank 端点、激活守恒、概率归一化、负 logit／更远负向量反例、q=0 的额外未知分母、大有限值稳定 Softmax、crop 聚合非交换，以及非法 q 拒绝。通过仅表示这些有限数学输入上的断言成立；替代给定的 F／q **没有**证明任何 Weibull 拟合有效。文档同时检查五层／五问、来源锚点、本地链接及唯一编号；只新增本文件，不改覆盖矩阵、索引、依赖、测试或其他卡片。

**仍未完成：**真实 `fit_high`／`w_score` 与序列化往返、作者脚本运行、已训练骨干／真实样本特征、MAV 数值重建、训练和推理距离缓存对照、未知类别独立验收、全风险保证重证，以及任何工业漏放／误拒／节拍实测。因此 L3／L4／L5 均保留“部分”，本仓库算法状态为“未实现”。

**按覆盖清单的“具体方法缺卡”口径，G24 可销项为已有卡**：正确分类样本→MAV→距离右尾拟合→rank／Weibull→未知激活重分配→Softmax→拒识的关键机制已由原文及固定作者源码承接，五层正文与答 5 齐备；不再只由 C08 Softmax 组件代替。销项不表示完整论文复现、所有 L 层已核验或现场验收；后续深度／复现工作保留 P2。由于本次仅允许写本文件，**总览与索引中的 G24 台账未同步**，历史 ID 保留，不在此擅改计数。
