# OC01 One-Class SVM：正常支持域、ν 与核二次规划

[学习规范](../algorithm_learning_standard.md) · [缺陷异常总览](../defect_evolution.md#33-e03从两类分界改为正常支持域) · [覆盖清单 G17](../evolution_coverage.md) · [两类 SVC：C02](learning_components.md#c02-支持向量机-svmrbf-svc) · [两类 SVC 与概率：26](matching.md#26-svmrbf-svc-与概率)

<a id="oc01"></a>
<a id="g17"></a>

## 任务、演进与覆盖状态

**任务：**给定同一特征空间的训练样本，学习一个接受区域；新样本输出正常侧分数、异常侧分数与支持域内外判定。原方法不需要类别标签；工业 novelty detection 另外假定训练样本经过确认、主要代表正常总体。统计域外不等于违反质量规格，域内也不保证无缺陷；整图特征不直接产生缺陷位置。[cite:1](#source-1)[cite:2](#source-2)[cite:6](#source-6)

**演进定位：**已知正负样本的两类分界难以覆盖未知异常 → 在核特征空间把样本与原点分离 → 用正则化半空间的原像描述正常支持域 → 付出核求解、支持向量推理和数据维护成本 → 仍受特征、污染、罕见正常模式和跨域漂移限制。1999 NIPS 方法、1999-11-27 技术报告及其 2000-09-18 修订、2001-07-01 期刊版分别记录，不把下载路径中的 2016 当方法年，也不称其精确最小输入空间体积估计。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:8](#source-8)

核验日期：**2026-11-05**。本文件只含 OC01 一张卡，关联 G17／06-03；SVDD、两类 SVC 仅作边界对照，不另计新卡。

| 字段 | 状态与范围 |
|---|---|
| L1 步骤 | 已核验：输入、训练／推理契约、停止与异常退出、参考伪代码 |
| L2 机制 | 已核验：原点分离、核几何、slack、支持向量及反例 |
| L3 工程权衡 | 部分：参数、污染、阈值与排障设计齐备，无产线定参或验收 |
| L4 数学 | 已核验：限定为等权 PSD 核模型；原始／对偶／KKT、ρ 恢复、有限样本 ν 界、两坐标更新；不表示完整重证论文统计理论 |
| L5 实现 | 部分：外部指定版本源码静态对应、可实现伪代码、测试设计；本仓库无专用实现或已训练模型 |
| 面试自测 | 答 5：逐层问题、答案、成立条件、易错点与证据入口 |
| 验证口径 | 文献与源码核验、文档检查和数学一致性检查；非缺陷召回／误报／节拍实测 |

本地 [hog_svm_matching.py](../../ml/hog_svm_matching.py#L57-L80) 实际训练 `StandardScaler + SVC(kernel="rbf", C=10, probability=True)`，对候选作概率排序，未按分类阈值拒绝；它不是 OC01。两处既有 SVC 卡不替代本卡，也不因本卡而升级实现状态。

## L1 步骤：从正常训练到新样本评分

### 输入输出与符号

训练矩阵 \(X\in\mathbb R^{n\times d}\)，每行一件样本或一个固定语义窗口，\(n,d>0\)。同件不同帧／同图 patch 必须按原件分组，特征列顺序、单位和窗口坐标保持一致。变换后记 \(z_i=T(x_i)\)，核 \(K_{ij}=k(z_i,z_j)\) 有限、对称、半正定（PSD）。本卡使用等权 \(0<\nu\le1\)，记 \(C_\nu=1/(\nu n)\)，不要与两类 SVC 的独立参数 C 混淆。

定义未平移的正常分数、决策分数、异常分数：

\[
g(z)=\sum_i\alpha_i k(z_i,z),\qquad f(z)=g(z)-\rho,\qquad a(z)=\rho-g(z)=-f(z).
\]

\(f\) 越大越偏正常侧；\(a\) 越大越偏异常侧。数学接受域采用 \(S=\{z:f(z)\ge0\}\)，严格域外为 \(f<0\)。这不是概率；若 \(w\ne0\)，真正的特征空间有符号几何距离是 \(f/\|w\|\)，不是直接把 API 分数当成单位距离。[cite:2](#source-2)[cite:5](#source-5)

### 训练、停止与推理

1. 确认正常定义、特征可检性、分组划分；训练集拟合预处理，验证／验收／推理仅变换，不重新拟合。
2. 选择核及 \(\nu,\gamma\)，生成核矩阵或列缓存；不向两类 `SVC.fit` 传全一标签冒充单类算法。
3. 解 L4 的盒约束＋和约束 QP；检查可行性、KKT 违反量、对偶间隙和有限值，而非只看优化器返回 `success`。
4. 用自由支持向量恢复 \(\rho\)；不存在自由 SV 时用 KKT 区间，\(\nu=1\) 按端点规则单独处理。保留 SV、系数、预处理、核参数、阈值与版本。
5. 独立正常验证集可定误报操作点；有真实缺陷验证时才能评估相应召回。改变业务阈值后，不能继续把原模型的 ν 界当新阈值保证。
6. 新样本走同一特征及变换，算 \(g,f,a\)；按明确的零值／灰区规则判定。非有限输入、未知特征版本、未收敛模型、超时或失效成像应返回 UNKNOWN／复检，不强行输出 OK。

正常退出须完整集满足收敛条件；达到迭代／时间上限是 `NOT_CONVERGED`，不是合格模型。多个窗口映射回原图须保存 ROI 原点、步幅、缩放和有效域；上采样分数图不是像素级定位保证。[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)

## L2 机制：每一步解决什么问题

| 机制 | 必要性与不变量 | 删除／替换后的后果 |
|---|---|---|
| 与特征空间原点分离 | \(-\rho\) 鼓励提高接受门槛，\(\|w\|^2/2\) 控制函数复杂度；原点不是一批真实负例 | 两类 SVC 需要两类监督，不能把原点当真实异常总体 |
| 核映射 | 半空间在输入空间的原像可为非线性、多块接受域 | 线性核只给半空间，不自动包围多模态正常簇；RBF 也不保证支持域拓扑正确 |
| slack 与 \(C_\nu\) | 允许部分训练点在门槛外，避免为容纳所有点任意扭曲边界 | 允许舍弃不等于能识别污染标签；稀有正常模式也可能被舍弃 |
| \(\sum\alpha_i=1\) 与盒约束 | 使 \(w\) 成为特征像的凸组合；单个样本影响系数受上界限制 | 单坐标独立裁剪会破坏和约束，不能用无约束梯度法后逐点截断替代 QP |
| 标准化与固定表示 | 按训练统计定义距离几何，防止大单位列主导 RBF | 用验收集 fit 会泄漏；缺陷在特征提取时已被抹掉，后续核无法恢复 |
| ρ 与业务阈值分开 | ρ 是训练 KKT 门槛；业务阈值是在独立验证协议下作代价决策 | 无依据直接移动门槛可能改善误报却增加漏检，不再继承训练 ν 界 |

一个重要反例是：线性核先把训练特征严格中心化，则其均值为零，\(\alpha_i=1/n\) 可行并给 \(w=0\)，故最优对偶目标为零；“把正常点远离原点”的几何退化。RBF 的距离平移不变，减同一均值本身不改变核值，但逐列缩放会改变核值；**不要把特征标准化与核矩阵中心化混为一谈**。RBF 特征像满足 \(k(z,z)=1\)，核空间原点与输入坐标零向量不是同一对象。[cite:2](#source-2)，中心化反例为本卡模型直接推论。

## L3 工程权衡：尺度、污染与操作点

### 参数与调参顺序

| 参数／量 | 单位及起点依据 | 验证和代价 |
|---|---|---|
| \(\nu\in(0,1]\) | 无量纲；由正常覆盖需求、训练污染审计和可接受舍弃预算选候选，不等于真实缺陷率 | 与 γ 联合验证；增大 ν 不保证所有新样本分数或接受域单调变化 |
| RBF \(\gamma>0\) | 特征距离平方的倒数；\(k=e^{-\gamma\|z-z'\|^2}\)，论文 \(e^{-\|z-z'\|^2/c}\) 对应 \(\gamma=1/c\)，若用 σ 则 \(\gamma=1/(2\sigma^2)\) | 可用仅训练集非零距离中位数 \(m\) 的倒数作对数搜索中心；这是启发式，不是论文推荐最优值；\(m=0\) 先查重复／常量特征 |
| `gamma='scale'` | scikit-learn 为 \(1/(d\operatorname{Var}(Z))\)，方差对整个传入矩阵求，不是每列各算一个 γ；`auto` 为 \(1/d\) | 全部非恒定列严格中心化标准化时才化为 \(1/d\)；若只有 r 个非恒定列，理想情况下化为 \(1/r\)；全常量需判退化 |
| 标准化 \(\mu_j,s_j\) | 与原特征同单位；\(z_j=(x_j-\mu_j)/s_j\)，ddof=0，常量列 scale=1 | 只 fit 训练折；标准化不是白化；均值／方差可被异常值污染，稀疏输入不要盲目中心化 |
| 收敛容差 | α 容差按盒上界尺度选；梯度容差按核／系数尺度选；独立设置可行性与间隙容差 | 官方 API `tol=1e-3` 是默认值，不是现场精度；收紧容差提高数值可靠性但增耗时 |
| 核缓存／SV 数量 | 缓存 MB，支持向量数 m | 内存不足导致核列重算；预测每样本约 \(O(md)\)，不是只由特征维度决定 |

以上 γ 与 API 定义来源于原论文和官方文档；距离中位数搜索、分组验收与调参顺序为工程设计，未提供通用误报／漏检界。[cite:1](#source-1)[cite:5](#source-5)[cite:7](#source-7)

推荐顺序为：**划分／正常标签审计 → 成像与特征尺度 → 核矩阵数值／距离分布 → ν 与 γ 联合验证 → 数值容差 → 业务阈值 → 缓存和加速**。只用正常验证可比较正常拒绝率、分数稳定性和资源，却不能据此挑出保证真实缺陷召回最好的模型。测试缺陷反复参与调参后不能仍称盲验。

### 症状、证据与回退

| 症状 | 优先证据 | 处理与限制 |
|---|---|---|
| 训练分数几乎相同 | γ 太小导致 K 接近全一、重复特征、常量列、线性中心化退化 | 先查表示与距离；γ=0 的常量核不作为有效 RBF 模型 |
| 新正常样本普遍域外 | γ 太大使不同样本间核接近零，或新批次尺度偏移 | 分组检查近邻距离、scaler 和合法正常模式覆盖；不要直接调高异常阈值掩盖漂移 |
| 缺陷进入正常接受域 | 污染样本、模糊／池化损失、缺陷与正常特征重合 | 冻结更新、回滚可信训练集；即使污染比例小于 ν，也无自动清洗保证 |
| 罕见正常结构被拒 | 正常采样比例失衡、簇间密度差、窗口混类 | 独立评估罕见正常；分产品／位置建模是新协议，不能把频次低等同缺陷 |
| `predict=-1` 比例超过 ν | 边界零值约定、求解容差、未收敛、权重、ρ／系数尺度不一致 | 分别统计 \(f<-\epsilon_f\)、\(\lvert f\rvert\le\epsilon_f\) 和负标签，不用标签计数直接证伪严格数学界 |
| 换批分数尺度不一致 | 训练 n、ν、scaler、核宽度、系数归一化及聚合改变 | 固化模型／门槛版本；不同模型原始分数不可直接混用 |

官方指南明确 One-Class SVM 对污染敏感。ν 允许训练误差但不提供污染识别保证；标准化污染还会先改变全部样本的距离几何。业务“正常”与统计“高密度”必须分开。[cite:6](#source-6)[cite:7](#source-7)

## L4 数学推导：目标、边界与求解

### 原始问题与对偶

在核特征空间中，\(\phi_i=\phi(z_i)\)，\(\rho\in\mathbb R\) 不强加正值：

\[
\min_{w,\rho,\xi}\ P=\frac12\|w\|^2+C_\nu\sum_i\xi_i-\rho,
\qquad w^T\phi_i\ge\rho-\xi_i,\quad \xi_i\ge0.
\]

引入 \(\alpha_i,\beta_i\ge0\)：

\[
\mathcal L=\tfrac12\|w\|^2+C_\nu\sum_i\xi_i-\rho
-\sum_i\alpha_i(w^T\phi_i-\rho+\xi_i)-\sum_i\beta_i\xi_i.
\]

分别对 \(w,\rho,\xi_i\) 求驻点，得到：

\[
w=\sum_i\alpha_i\phi_i,\qquad \sum_i\alpha_i=1,\qquad
\beta_i=C_\nu-\alpha_i.
\]

故最大化形式对偶是 \(D=-\alpha^TK\alpha/2\)，等价的最小化 QP 为：

\[
\min_\alpha Q(\alpha)=\tfrac12\alpha^TK\alpha,
\qquad 0\le\alpha_i\le C_\nu,\qquad \mathbf1^T\alpha=1.
\]

原论文对应 1999 式 (3)–(6)、2001 式 (3.4)–(3.12)。注意最小化 Q 的数值不是最大化 D；计算原对偶间隙时不能写成 \(P-Q\)。PSD 保证凸性；严格可行原始点存在，有限样本核张成空间下可用强对偶及 KKT。\(\alpha_i=1/n\) 总可行；若 \(\nu>1\)，盒容量 \(nC_\nu<1\) 不可行。[cite:1](#source-1)[cite:2](#source-2)

PSD 不保证系数唯一：重复点可在相同核列间分摊权重；若 K 在可行差分子空间 \(\{v:\mathbf1^Tv=0\}\) 上正定，则 α 唯一。\(w\) 的二次正则严格凸使最优 w 唯一，但 ρ 可以不唯一。一般核下不保证 \(\rho>0\) 或与原点可分，不能把图解当定理。

### KKT 与自由／无自由 SV 的 ρ

除可行性和驻点条件，还须：

\[
\alpha_i(g_i-\rho+\xi_i)=0,\qquad
(C_\nu-\alpha_i)\xi_i=0,\qquad g_i=(K\alpha)_i.
\]

| α 状态 | 由 KKT 得到 | 不能反推的结论 |
|---|---|---|
| \(\alpha_i=0\) | \(\xi_i=0,\ g_i\ge\rho\) | 可能恰在边界，不必严格内部 |
| \(0<\alpha_i<C_\nu\) | \(\xi_i=0,\ g_i=\rho\)，称自由 SV | 全部 SV 不都自由 |
| \(\alpha_i=C_\nu\) | \(g_i=\rho-\xi_i\le\rho\) | 上界 SV 可能 \(\xi_i=0\)，不一定严格异常 |

**有自由 SV：**集合 \(F=\{i:0<\alpha_i<C_\nu\}\)，精确解任选其 \(g_i\) 都相同；浮点实现可用 \(\rho=|F|^{-1}\sum_{i\in F}g_i\)，并报告自由点最大偏离，不能用平均掩盖未收敛。[cite:2](#source-2)[cite:4](#source-4)

**无自由 SV 且 \(0<\nu<1\)：**正系数全在上界，其余为零；令：

\[
\rho_{\rm lo}=\max_{\alpha_i=C_\nu}g_i,\qquad
\rho_{\rm hi}=\min_{\alpha_i=0}g_i.
\]

KKT 要求 \(\rho_{\rm lo}\le\rho\le\rho_{\rm hi}\)，可取两端中点。这时两类集合都非空；严格无自由解要求上界点数 \(=\nu n\) 为整数。若 νn 非整数却被浮点分类成无自由点，应首先收紧 α 状态容差／重算梯度，而不是静默伪造自由 SV。若区间反转超过容差，说明数值或最优性检查失败。[cite:4](#source-4)，区间与整数条件由 KKT 和和约束直接推出。

**\(\nu=1\) 特例：**唯一可行 α 是 \(\alpha_i=1/n=C_\nu\)，没有零系数集合；\(\rho\in[\max_i g_i,+\infty)\) 都满足 KKT。选择有限 \(\rho=\max_i g_i\) 是本卡参考实现的端点约定，不是宣称 ρ 唯一。不能取“无穷上界与有限下界的中点”，也不能对空自由集合求均值。LIBSVM v336 的通用 `calculate_rho` 无自由时直接 `(ub+lb)/2`，本卡静态核对到此路径在全上界时保留 `ub=INF`；因此参考伪代码显式防护，**未实跑此版本确认其最终行为，不外推所有 scikit-learn 版本均有同样问题**。[cite:4](#source-4)

### ν 的有限样本界：计数对象和条件

定义严格训练域外点 \(E=\{i:g_i<\rho\}=\{i:\xi_i>0\}\)，支持向量 \(V=\{i:\alpha_i>0\}\)，均按样本行计数，重复点仍是多行。精确最优等权模型中：

\[
|E|C_\nu\le\sum_i\alpha_i=1\quad\Rightarrow\quad |E|\le\lfloor\nu n\rfloor,
\]

\[
1=\sum_{i\in V}\alpha_i\le |V|C_\nu\quad\Rightarrow\quad |V|\ge\lceil\nu n\rceil.
\]

第一个用“严格域外必在上界”，第二个甚至只需对偶可行性；**不要求有限样本 i.i.d.，也不要求必有自由 SV**。1999 Proposition 1 原陈述另写了 \(\rho\ne0\)；忠实引用定理时保留该条件。本卡对“严格 \(f<0\)”的上述代数证明不需要除以 ρ，故在 \(\rho=0\) 仍成立；它不是把原文所有渐近结论一起放宽。[cite:1](#source-1)[cite:2](#source-2)

重要限制：

- \(f=0\) 的点不属于 E。LIBSVM v336 用 `sum > 0 ? +1 : -1`，零值记负类；因此负标签比例可以超过 ν 而不违反此处的严格界。浮点下同时报告边界带，不机械断言 `mean(predict == -1) <= nu`。[cite:4](#source-4)
- 松弛是优化意义的误差，不是已知缺陷标签；正常训练数据的域外点可能是合法少见样本。模型不保证把实际污染全部分到 E。[cite:6](#source-6)
- `sample_weight` 改变各行盒上界后，原始等权计数证明不能照搬；应重新推导加权界，不以重复样本／权重暗改统计单位。[cite:5](#source-5)
- 渐近“两个比例趋于 ν”在 1999 Proposition 1 还要求独立采样、分布不含离散分量、核解析且非常量，并保留原命题的非退化前提；它不是任意有限 n 的等式。量化／重复特征及固定有限支持数据不能自动满足这些条件。[cite:1](#source-1)
- 新样本拒绝概率、真实缺陷漏检率、现场误报率不由上述计数式保证。论文另有依赖函数类容量、间隔、样本量和置信水平的统计界；此卡未重证完整统计章节，不把 API 指南的直观概率描述当无条件定理。[cite:1](#source-1)[cite:6](#source-6)

### 两坐标 SMO 更新及数值停止

保持其他 α 不动，令 \(s=\alpha_i+\alpha_j\)，以 \(t=\alpha_i'\)、\(\alpha_j'=s-t\) 参数化：

\[
L=\max(0,s-C_\nu),\quad H=\min(C_\nu,s),\quad
\eta=K_{ii}+K_{jj}-2K_{ij}=\|\phi_i-\phi_j\|^2\ge0.
\]

当前梯度 \(g=K\alpha\)，令 \(\delta=t-\alpha_i\)，局部目标变化为：

\[
\Delta Q=\delta(g_i-g_j)+\tfrac12\eta\delta^2.
\]

若 \(\eta>0\)，精确最优更新为：

\[
\alpha_i'=\operatorname{clip}\!\left(\alpha_i-\frac{g_i-g_j}{\eta},L,H\right),
\quad \alpha_j'=s-\alpha_i',\quad
g'=g+\delta(K_{:i}-K_{:j}).
\]

若 \(\eta=0\)，比较两端及当前点的 \(\Delta Q\)，取不增目标的最优候选；斜率也为零时保持当前点即可。小正 η 仍有内部最优候选，不能只比端点；若直接除法不稳定，使用有界一维凸求解或更高精度。明显负 η 应拒绝该核／重算，不把它当合法非凸核仍声称全局最优。LIBSVM 的 `TAU=1e-12` 替代非正二次系数是实现防护，与本卡端点求解不是逐步相同算法。[cite:3](#source-3)[cite:4](#source-4)

一个透明的工作集选择是：从 \(A=\{i:\alpha_i<C_\nu\}\) 取最小 \(g_i\) 的可增加点，从 \(B=\{j:\alpha_j>0\}\) 取最大 \(g_j\) 的可减少点。最大违反量：

\[
v=\max\left(0,\max_{j\in B}g_j-\min_{i\in A}g_i\right).
\]

\(v>0\) 时沿此对转移质量有下降方向；\(v=0\) 加对偶可行性即满足该凸 QP 的 KKT。\(\nu=1\) 时 A 为空，直接处理唯一可行 α，不能对空集取最小值。全量重算梯度后检查 \(v\le\epsilon_{\rm KKT}\)、\(|\sum\alpha-1|\le\epsilon_{\rm eq}\) 和盒残差。再取 \(\xi_i=\max(0,\rho-g_i)\)：

\[
P=\tfrac12\alpha^TK\alpha+C_\nu\sum_i\xi_i-\rho,\quad
D=-\tfrac12\alpha^TK\alpha,\quad
P-D=\alpha^TK\alpha+C_\nu\sum_i\xi_i-\rho\ge0.
\]

间隙应小于绝对＋相对容差，负到超过舍入误差说明尺度、可行性或公式有错。PSD 与精确子问题给目标单调不增且有下界；这本身只说明目标值可收敛，不足以宣称任意选对策略都到最优。最终完整 KKT 才是凸模型的全局最优证书；不能给任意数据承诺固定迭代数。启用 shrinking 后应恢复完整梯度及活动集再检查，避免把暂时移除的点遗漏。[cite:4](#source-4)

稠密 RBF 建核约 \(O(n^2d)\)、存储 \(O(n^2)\)；缓存已有列时，一次完整梯度两坐标更新约 \(O(n)\)，选对扫描另为 \(O(n)\)，列未命中时另计核计算。总训练成本依赖迭代数和核条件，不能一概写成固定 \(O(n^2)\)。通用稠密 QP 可作小规模参照，不是无需说明容差的真值机。

### 与 SVDD 和两类 SVC 的区别

| 方法 | 目标／约束 | 与 OC01 的边界 |
|---|---|---|
| OC01 | 半空间与原点分离；\(\min\|w\|^2/2+C_\nu\sum\xi-\rho\)；对偶 \(\sum\alpha=1\) | 无真实负类；输出支持域侧分数 |
| 两类 C-SVC | \(\min\|w\|^2/2+C\sum\xi\)，\(y_i(w^T\phi_i+b)\ge1-\xi_i\)；对偶 \(\sum\alpha_i y_i=0\) | 依赖正负标签，决策 \(\sum\alpha_i y_i k+b\)；C02／26 及本地重排属于此路 |
| SVDD | 特征球 \(\min R^2+C_\nu\sum\xi\)，\(\|\phi_i-c\|^2\le R^2+\xi_i\) | 对偶最小化 \(\alpha^TK\alpha-\sum_i\alpha_iK_{ii}\)，同样盒约束和和约束；一般核不与 OC01 等价 |

若 **整个输入域** 上 \(k(z,z)=\kappa\) 恒定，如 RBF，则 SVDD 的线性项为常数，最优 α 集与 OC01 相同；配对门槛满足 \(R^2=\kappa-2\rho+\|w\|^2\)，球内分数为：

\[
R^2-\|\phi(z)-c\|^2=2(g(z)-\rho).
\]

因此在相同 ν、核、数据和兼容门槛约定下，两者接受域一致；没有自由 SV 时必须协调非唯一门槛，不能任取两个解后要求边界一致。仅训练对角线相同却测试 \(k(z,z)\) 改变，不足以推出预测等价；一般线性核／多项式核亦不能照搬。这里的 SVDD 依据 2001 论文 §3 球模型及技术报告，不声称完整核验 Tax–Duin 原始论文。[cite:2](#source-2)[cite:3](#source-3)

## L5 实现：源码映射、伪代码与测试

### 官方实现对应与尺度

| 证据入口 | 实际核到的内容 | 本卡复现须遵守 |
|---|---|---|
| LIBSVM v336 `solve_one_class` | 前 \(\lfloor\nu n\rfloor\) 个系数置 1，下一项置剩余质量，其余 0；p=0、y 全一、上下类盒界均 1；调用通用 `Solver` | 内部系数记 q，\(\sum q_i=\nu n\)，不是论文 \(\sum\alpha=1\)；不是 `Solver_NU` 的两类 ν-SVC |
| `Solver::Solve` 同号分支 | 保持两坐标和，夹到盒内，更新梯度；全量恢复后再判最优；迭代上限会警告 | 本卡最违反对伪代码不是现代 LIBSVM 二阶工作集策略的逐行移植 |
| `Solver::calculate_rho` | 自由点 yG 平均，无自由点用 ub/lb 中点 | y 全一才退化为本卡的 g 关系；ν=1 空侧边界须额外防护 |
| `svm_predict_values` | 系数加权核和减 `model->rho[0]`，严格大于零返回 +1 | 零值返回 -1，边界约定不同于本卡闭接受域；先统一再比较标签 |
| scikit-learn `_classes.py`，`afe5bcd80` | `OneClassSVM.fit` 传内部全一标签、`offset_=-_intercept_`；`score_samples=decision_function+offset_` | API 的 y 被忽略；不要沿用 SVC 的概率列或 `classes_` 解释单类分数 |

等权条件下 LIBSVM 尺度换算为 \(q_i=(\nu n)\alpha_i\)、\(\rho_{\rm lib}=(\nu n)\rho\)，故其原始决策分数为本卡 \(f\) 的 \(\nu n\) 倍；正倍数不改符号，但比较数值、容差、阈值与间隙前必须统一尺度。官方 `OneClassSVM` 的 `offset_` 与 `intercept_` 的负号关系已核，后端和版本应在真正运行比较时再次确认，不能直接假定任何包装库返回系数之和都是 1。[cite:4](#source-4)[cite:5](#source-5)[cite:9](#source-9)

### 参考伪代码

下面为**等权、稠密、无 shrinking、最违反对的教学方案，非仓库已实现函数**。核／预处理／ρ 恢复与判定约定均按本卡；可用通用 QP 取代循环，但仍执行同样的残差检查。

```text
FIT_OC01(X_train, nu, gamma, tolerances, max_iter):
    require nonempty finite X_train and 0 < nu <= 1 and gamma > 0
    T = fit_scaler_on_training_only(X_train)
    Z = T.transform(X_train)
    K = finite_symmetric_psd_rbf(Z, gamma)
    n = rows(Z)
    C = 1 / (nu * n)
    alpha = vector(n, 1 / n)
    g = K @ alpha
    if nu == 1:
        rho = max(g)
        return validate_and_package(T, Z, alpha, rho, K, tolerances)
    for iteration in range(max_iter):
        A = indices(alpha < C)
        B = indices(alpha > 0)
        i = argmin(g[A]) mapped_to_original_index
        j = argmax(g[B]) mapped_to_original_index
        if g[j] - g[i] <= eps_KKT:
            g = K @ alpha
            if full_kkt_violation(alpha, g, C) <= eps_KKT:
                rho = recover_rho_by_free_mean_or_bounded_interval(alpha, g, C)
                if feasible_and_gap_small(alpha, rho, K, C, tolerances):
                    return validate_and_package(T, Z, alpha, rho, K, tolerances)
                return NOT_CONVERGED
            continue
        s = alpha[i] + alpha[j]
        L = max(0, s - C)
        H = min(C, s)
        eta = recomputed_nonnegative_pair_curvature(K, i, j)
        if eta > 0:
            t = stable_bounded_quadratic_minimizer(alpha[i], g[i]-g[j], eta, L, H)
        else:
            t = best_of_endpoints_and_current_by_delta_Q(L, H, alpha[i], g[i]-g[j])
        delta = t - alpha[i]
        require feasible_pair_and_nonincreasing_Q(t, s-t)
        if numerically_stalled(delta):
            return NOT_CONVERGED
        alpha[i] = t
        alpha[j] = s - t
        g = g + delta * (K[:, i] - K[:, j])
        periodically_recompute_full_gradient_and_residuals()
    return NOT_CONVERGED

SCORE_OC01(model, X_new, anomaly_threshold, boundary_band):
    require finite_input_and_matching_feature_schema(X_new, model)
    Z_new = model.T.transform(X_new)
    g = kernel(Z_new, model.support_vectors) @ model.support_coefficients
    f = g - model.rho
    a = -f
    return f, a, decision_with_explicit_boundary_policy(a, anomaly_threshold, boundary_band)
```

上述 `stable_bounded_quadratic_minimizer` 在可稳定除法时就是 L4 的 clip 公式，否则使用有界一维凸求解；ρ helper 按 L4 三种情形处理并验证区间，不是隐藏“调用 SVM 即完成”的黑盒。α 状态容差仅用于诊断与恢复自由集合，不可把接近边界的值独立截断后破坏和约束。低于保存阈值的小系数若被丢弃，需重新验证分数误差，不能假装无损稀疏化。

### 数据布局、稳定性与优化

- 密集参考计算使用 float64 的 \(n\times d\)、\(n\times n\)；预测核块为 \(n_{\rm new}\times m\)。预计算核接口通常要求 \(n_{\rm new}\times n_{\rm train}\)，必须保留原训练列次序，不能误传 SV 子矩阵。[cite:5](#source-5)
- RBF 的 \(\|z\|^2+\|z'\|^2-2z^Tz'\) 有大数相消；小规模参考可直接累加差平方，大规模对微小负距离只在舍入阈值内截零，明显负值重算。高 γ 指数下溢到零可能是合法极限，也可能意味着尺度错误。
- LIBSVM v336 核缓存元素是 `Qfloat=float`，α／梯度为 double；不能声称源码全链路 float64。增加缓存、批量预测、并行核计算、shrinking 可改变舍入与耗时，必须对照完整残差、分数和边界带，而非只对照非零系数个数。[cite:4](#source-4)
- 非 PSD 核不能靠对称化就修好；加 \(\lambda I\)、低秩近似、随机特征或 SGDOneClassSVM 都改变优化问题／近似误差。先保留精确小规模参考，再评估省内存方案，不默认与本卡核 QP 等价。[cite:6](#source-6)

### 测试设计与判定标准

以下为应执行的设计，不是测试通过清单；所有模型训练指标与真实任务验收必须另记来源。

| 测试 | 数据／操作 | 必须断言或记录 |
|---|---|---|
| 输入与契约 | 空矩阵、单行、NaN/Inf、错列、非法 ν、零 γ、全常量列 | 明确拒绝／退化状态；不产出伪正常标签 |
| 原对偶与 KKT | 真实样本特征的小规模 PSD 核，通用 QP 与两坐标法独立求解 | 盒／和残差、自由点偏差、全量 v、P-D；目标与分数在约定容差内一致 |
| 自由 SV／无自由 SV | 从真实特征及 ν 扫描记录两类解；无自由解仅在实际满足条件时计覆盖 | 有自由时 ρ 平均；无自由时区间非空；未出现该分支就标未覆盖，不能编造命中 |
| ν=1／零曲率 | 同一真实记录的重复输入、ν=1、重复核列 | α 固定时 ρ 有限；η=0 不除零；重复点非唯一系数不导致错误失败 |
| 有限样本界 | 等权精确／足够收敛解，记录边界带与 α 判零规则 | 严格域外数上界、SV 下界；同时保留近零不确定计数，负标签不代替严格 E |
| 系数尺度及符号 | 固定版本外部库与归一化参考对照 | q/(νn)、ρlib/(νn)、分数倍数；offset=-intercept；零值约定单独比较 |
| 变换与泄漏 | 固定训练统计，改变验证集顺序／批次／数量；序列化后再读 | scaler 不变、同样本分数一致；特征列错序被识别；模型与阈值同版本 |
| 污染与域偏移 | 有标签审计的真实正常／污染批次，按工件隔离 | 污染前后 scaler、核距离、分数、罕见正常拒绝率；不以 ν 充当污染标签 |
| 优化等价性 | 相同真实输入下缓存大小、批量、shrinking、核近似对照 | 精确加速比残差／分数；近似方法另报误差与尾延迟，不要求 α 逐位一致 |
| 缺陷验收 | 独立真实正常和缺陷保留集，固定门槛，分缺陷尺寸／类型／批次 | 整件误报／漏检、UNKNOWN 比例、定位与资源分开；不以训练界或 AUROC 替代操作点 |

## 五层面试自测：5 问与答案

| 层级／问题 | 答案与成立条件 | 易错答案／证据入口 |
|---|---|---|
| L1：只有正常工件特征，如何训练并判断新件？ | 工件分组隔离，训练 fit 变换，求等权核 QP，恢复 ρ，固定模型算 g-ρ；收敛与异常退出分开；域外不自动是质量 NG | “给 SVC 全一标签即可”；见 L1、两类 SVC 对照及来源 5、9 |
| L2：为什么既要核又要缩放？线性中心化会怎样？ | 核允许非线性接受域；缩放改变距离几何而中心化不改变 RBF 距离；线性中心化使原点在样本凸包中，可产生 w=0 退化 | “标准化等于核中心化，任何核都能把原点分开”；见 L2、来源 2、7 |
| L3：换批大量误报，调 ν 就能修好吗？ | 先核 scaler、采集、距离及正常模式覆盖，再联合验证 ν/γ 和业务门槛；污染会改变预处理与边界，纯正常验证不能评缺陷召回 | “ν 就是缺陷率，设为污染比例便自动去污”；见 L3、来源 5、6 |
| L4：推导 ν 界；ρ 没有自由 SV 时怎么办？ | 驻点给和约束及盒约束；严格域外 α=Cν 推上界，非零 α 的容量推下界；无自由用上界点最大 g 与零点最小 g 的区间，ν=1 用有限端点；原命题与严格计数推论条件分开 | “上界 SV 都是异常”“总能找自由点”“未来误报率≤ν”；见 L4、来源 1–4 |
| L5：理论一致却与 LIBSVM 分数不同，怎样查？ | 查 q=(νn)α、ρ 的尺度、减号、零标签约定、核列顺序、缓存精度及完整 KKT；两坐标保持和约束，η=0 不除零；优化前保留小规模对照 | “系数必定和为 1”“max_iter 停止就收敛”；见 L5、来源 4、5、9 |

## 来源层级、版本与获取深度

来源优先级：**P0 原论文／作者稿／官方实现与 API > P1 权威综述 > P2 社区解析 > P3 通用摘要**。本卡数学主干使用 P0，未用博客承担公式或性能结论。原论文不同版本属于同一研究谱系，scikit-learn 与 LIBSVM 也有实现继承关系，不能凑成三份独立实验；理论条目标为 **[SINGLE-SOURCE：原研究谱系]**，实现条目标为 **[CONFIRMED：指定源码行为]**，不是三方独立复现。工程推论明确标成设计，未核验性能不作量化宣称。

- <a id="source-1"></a>**[cite:1] P0／原论文，Schölkopf、Williamson、Smola、Shawe-Taylor、Platt，Support Vector Method for Novelty Detection。** [NIPS 1999 官方 PDF](https://papers.nips.cc/paper/1999/file/8725fb777f25776ffa9076e44fcfd776-Paper.pdf)。已读取 §2 式 (1)–(6)、自由 SV 门槛、§3 Proposition 1 的 ρ≠0／ν／渐近前提及后续统计界说明；返回到 §4 实验开头截断。使用官方 1999 归档口径，不以此断言纸质论文集精确发行日。未复现实验、未重证全部泛化定理。
- <a id="source-2"></a>**[cite:2] P0／作者托管期刊稿，Estimating the Support of a High-Dimensional Distribution。** [Smola 作者站 PDF](https://alex.smola.org/papers/2001/SchPlaShaSmoetal01.pdf)。已读 §1–3，尤其 (3.4)–(3.12) 原始／拉格朗日／对偶／ρ，及 (3.13)–(3.17) 球模型；返回在该段后截断。未据此声称读完 §4–5 或全部 29 页；ν 命题条件采用已实际读到的来源 1。
- <a id="source-3"></a>**[cite:3] P0／技术报告，MSR-TR-99-87。** [Microsoft 官方 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf)。首页明确 **27 November 1999; revised 18 September 2000**；已读算法、恒定核对角线与球模型等价说明、§4 两变量优化至式 (23) 附近，返回截断。数学抽取有字符丢失，目标符号与来源 1、2 交叉核对，更新公式另由本卡直接推导与来源 4 核对；不是完整 1999 初稿与 2001 终稿逐式比对。
- <a id="source-4"></a>**[cite:4] P0／官方源码，LIBSVM v336。** [svm.cpp](https://github.com/cjlin1/libsvm/blob/v336/svm.cpp)、[原始文本](https://raw.githubusercontent.com/cjlin1/libsvm/v336/svm.cpp)、[svm.h](https://raw.githubusercontent.com/cjlin1/libsvm/v336/svm.h)。`LIBSVM_VERSION 336` 已核。读取核／缓存／Solver 同号更新、停止及梯度恢复；对首次返回截断后又定向取得 `solve_one_class`、`Solver::calculate_rho`、`svm_predict_values` 函数体。版本固定为 tag，不称核验日最新版本；未编译运行库、未逐行审计整个工程。
- <a id="source-5"></a>**[cite:5] P0／官方 API。** [scikit-learn OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)。已读 γ、ν、tol、sample_weight、fit_status_、n_iter_、offset_、分数与预计算核形状。stable 是可变页面，核验日源码入口为 `afe5bcd80`；不当作本机已安装版本。
- <a id="source-6"></a>**[cite:6] P0／官方指南。** [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)。已读正常训练与污染检测区别、One-Class SVM 污染敏感性、核宽度及规模化说明。指南对 ν 与新正常样本概率的直观描述比原论文定理条件宽，本卡以来源 1 的有限／渐近条件为准；通用 `contamination` 段落不代表 OneClassSVM 构造器存在该参数。
- <a id="source-7"></a>**[cite:7] P0／官方 API。** [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。已读训练统计持久化、离群值敏感、常量列 scale=1、ddof=0、稀疏中心化限制。其 NaN 处理约定不能替代 OC01 输入有限值检查。
- <a id="source-8"></a>**[cite:8] P0／出版书目。** [ACM 收录 DOI 10.1162/089976601750264965](https://dl.acm.org/doi/10.1162/089976601750264965)。已读书目／摘要／参考文献，核对 Neural Computation **13(7)，1443–1471，2001-07-01**、技术报告前身；不是另一份独立理论证明或实验复现。
- <a id="source-9"></a>**[cite:9] P0／官方包装源码。** [scikit-learn `_classes.py`，afe5bcd80](https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/svm/_classes.py)。定向读取 `OneClassSVM.fit`、`decision_function`、`score_samples`、`predict`；核对内部全一标签、offset 负号和评分关系，未编译其 vendored LIBSVM，也不把官方上游 v336 冒充此提交内嵌版本。

## 验证范围与遗留项

本卡已完成原文指定章节、外部指定函数和本地两类 SVC 边界的静态核验。数学检查对象是原对偶符号、两坐标可行区间、目标变化、ρ 区间、严格计数界、系数尺度及恒定对角核下 SVDD 的配对门槛；不生成或引用虚构检测数据。

**本次实际运行核验：**使用现有 `.venv` 中 NumPy 2.5.4、SciPy 1.18.1、OpenCV 5.0.0，在内存中读取仓库 `RTImg/*/*.JPG` 排序后的前 8 条真实图像记录，缩为 8×8 灰度特征后按训练统计标准化，以非零距离中位数倒数设 γ。此特征仅供数学检查，不是建议的缺陷表征，也没有为这些记录补造正常／缺陷标签。对 ν=0.37、0.5 用 SLSQP 求同一等权 QP，对 ν=1 使用唯一可行 α 与有限 ρ 端点；三组可行性、KKT、原对偶间隙、带容差严格计数、系数尺度和 SVDD 分数恒等式检查通过。最大 KKT 违反量约 1.24×10⁻⁸，最大间隙约 1.40×10⁻⁹；另外完成 84 次两坐标目标变化、和／盒约束及梯度更新恒等检查。没有执行完整 SMO 循环或外部库训练对照，未将无自由 SV 的所有分支记为数值覆盖。检查脚本通过标准输入临时执行，未新增脚本、数据、模型或测试文件。

**文档核验：**五层、5 问答、代码围栏、数学定界符、49 处本地链接／锚点引用和 9 个来源锚点检查通过。当前环境未安装 scikit-learn，本次没有安装或运行它；指定外部实现的结论仍属源码静态核验。

**仍未实现：**本仓库 OC01 训练／推理接口、模型持久化、分组调参、窗口定位、线上阈值校准和污染回滚。L5 伪代码与测试设计不等于专用算法实现；临时数学一致性核验即使通过，也不提升为缺陷任务复现。真实正常／异常标签与独立批次不足时，不能报告缺陷检出率、现场误报率或通过验收。

**后续边界：**完整统计章节重证、Tax–Duin 原始 SVDD 文献独立核验、加权 ν 界、外部库固定版本运行对照、全部数值分支测试与真实缺陷验收。总览／索引仍是旧状态；本文件不修改其计数，不宣称 G17 已在台账实际销项。
