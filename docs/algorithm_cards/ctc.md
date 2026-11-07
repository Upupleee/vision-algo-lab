# CTC 算法卡

[学习规范](../algorithm_learning_standard.md) · [OCR 总览 O-H2](../ocr_codes_evolution.md#3-历史演进线-aocr-识别与检测) · [覆盖表 G27／08-02](../evolution_coverage.md#08-字符编码) · [O01 切字基线](remaining_domains.md#o01-切字加模板相关-ocr)

## 范围、状态与销项口径

核验日期：**2026-11-05**；仓库基准 HEAD `918a03e` 加已有工作树。唯一卡键为 **`ctc.md:CT01`**，承接 G27“CTC 前后向、数值稳定、重复字符与解码近似”，不顺带承接 G28 CRNN 架构、检测、整形或完整 OCR 系统。

| 项目 | 覆盖与边界 |
|---|---|
| 演进定位 | ICML 2006 的无逐帧对齐监督训练目标；由显式切字／帧标签转向对合法单调路径边缘化，不是字符检测器 |
| L1 | 已核验：输入契约、训练与推理解耦、折叠、动态规划、完整小例和停止条件 |
| L2 | 已核验：blank、重复标签 skip 禁止、求和而非取最大、双后缀状态的必要性 |
| L3 | 部分：参数尺度、长度与字符集、不可行／空目标、排障和选型协议完整；无产线调参与耗时剖析 |
| L4 | 已核验：限定模型下前后向边界、后验、logits 梯度、最小帧长和复杂度；无通用识别误差界 |
| L5 | 部分：固定官方源码对应、文档内可执行伪代码及穷举／差分检查；不是生产模块或 OCR 接入 |
| 面试自测 | L1–L5 共5问，包含答案、成立条件、易错项与证据入口 |
| 来源 | 原论文 P0／T部分，官方固定版本源码 P0／T，Distill 权威解释 P1／T相关部分；可信度与获取层级分开 |
| 实现状态 | **本仓库 CTC／OCR 生产实现未实现**；Python 源码检索未见 `CTCLoss`、`ctc_loss`、`ctc_decode`。依赖声明含 torch 不等于已有算法 |
| 验证范围 | 仅离散算法与数值检查，不训练、不读取 OCR 数据、不报告 CER／现场精度；实际记录见 L5 |
| 销项 | 本卡提供 G27 内容缺口的承接依据；只写本文件，覆盖表、索引、OCR 总览的历史“缺卡”文字保持原样，**不声称总表已销账或所有层级完成工程验收** |

“已核验”只表示以下指定资料、推导和有限测试范围成立。没有三份独立实验支撑通用性能结论；原论文与其引用实现不能算独立实验复现。本文小矩阵是公开的确定性单元测试输入，不是识别数据或模拟性能成绩。

<a id="ct01"></a>
## CT01 Connectionist Temporal Classification

### 任务、输入输出与历史分支

输入是编码器对给定序列 x 产生的逐步 logits \(a\in\mathbb R^{T\times C}\)，或其已归一化 log-probabilities \(\ell_{t,k}=\log p_t(k\mid x)\)。C 包含一个 blank 类 b；非 blank 词元集合记为 \(\Sigma\)。训练另需目标 \(y=(y_0,\ldots,y_{U-1})\in\Sigma^U\)，**目标内不得含 blank**。空格可以是普通词元，不能与 blank、padding 或 EOS 混为一谈。T 是编码器输出步数，既不是原图宽度，也不是标签字符数；若采用词元化，U/R 以词元 ID 而非 Unicode 字节计数。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

训练输出单样本负对数似然 \(\mathcal L=-\log P(y\mid x)\) 及对 logits 的梯度；无路径时返回不可行状态和无穷损失，不伪造有限梯度。推理没有目标 y，输出贪心字符串或 beam 候选、分数与另行定义的拒识结果。前后向得到的是“指定目标有多大概率”，不是直接解出最优未知目标。

历史链是：逐字／逐帧边界标注昂贵 → 增加 blank、定义多对一折叠、对全部合法路径求和 → 用动态规划和梯度优化整串似然 → 增加单调时间轴、数值稳定、长度预算及解码搜索的代价。Graves、Fernández、Gomez、Schmidhuber 的 *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*，ICML 2006，是本卡原始依据；原文实验是 TIMIT 语音，不是工业 OCR。CTC 不限定必须使用 RNN；CRNN 把图像变成列序列再使用 CTC，属于另一个架构节点。[cite:1](#source-1)[cite:4](#source-4)

### L1 步骤：训练与解码分别走通

1. 固定字符表、blank ID、每个样本有效 T、目标 U，拒绝非法 ID、含 blank 的有效目标和不合法概率。先计算相邻相同目标对数 R，检查 \(T\ge U+R\)。
2. 编码器产生 logits，在类别轴使用 `log_softmax`。只消费有效时间步；padding 不参与路径、损失或解码。
3. 训练将目标扩展为 \(z=(b,y_0,b,y_1,\ldots,b,y_{U-1},b)\)，长度 \(S=2U+1\)。按 L4 的初始化及 stay／advance／skip 转移，逐帧累计前向 log 概率。
4. 在最终允许状态求 `logsumexp` 得 \(\log P(y\mid x)\)，取负作为损失。空目标只用唯一 blank 状态；不可行时先报告原因，不继续计算后验。
5. 若需要训练梯度，初始化后向终态并倒序递推，按“同一词元的所有状态”汇总后验，得到 \(p_t(k)-\gamma_{t,k}\)，继续反传编码器。一次前后向在 T 帧处理完后停止；模型训练轮数／早停不由 CTC 算法本身决定。
6. 推理改走贪心或前缀 beam：贪心对每帧 argmax 后**先合并连续相同 ID，再去 blank**；beam 按每帧扩展、同前缀聚合、剪枝，处理完 T 帧停止。需要业务格式／语言模型时另记目标变化及修改前结果。

**折叠走例。** `A,A,blank,A → A,blank,A → AA`；`A,A,A → A`；`blank,blank → 空串`；`A,B,A → ABA`。连续重复指原路径上的相邻重复，不是输出中所有相同字符只保留一个。[cite:1](#source-1)[cite:4](#source-4)

**完整概率走例。** 取 C=2（`blank=0,A=1`），两帧均为 `(0.6,0.4)`，训练目标 `A`。合法路径是 `(b,A)`、`(A,b)`、`(A,A)`，概率分别为 0.24、0.24、0.16，总和 **0.64**。扩展串 `(b,A,b)` 的概率域前向表为：

| 时间 | blank 起态 | A 状态 | blank 终态 |
|---|---:|---:|---:|
| 第1帧 | 0.6 | 0.4 | 0 |
| 第2帧 | 0.36 | 0.40 | 0.24 |

最终 `0.40+0.24=0.64`，损失 \(-\log(0.64)\)。贪心却选 `(b,b)`，输出空串，概率 0.36。无剪枝的前缀聚合返回 `A` 为最高概率输出；这也说明最大路径并不等于最大字符串概率。对目标 `AA`，两帧不够，三帧最短合法路径是 `(A,b,A)`；其概率由具体三帧输入决定，不能继承上例两帧损失。

### L2 机制：不可省略的状态与聚合

| 机制 | 解决的困难与不变量 | 错误替换后的反例 |
|---|---|---|
| blank 是额外路径符号 | 可在帧上不输出标签，并隔开目标的相邻重复；并非强制对应图像背景或语音静音 | 先去 blank 再合并，把 `A,b,A` 错变成 `A`，永远丢掉某些双字符 |
| 扩展目标状态 | 状态不仅保存符号，还保存目标中的位置；所有路径都按目标顺序前进，不能跳过目标标签 | 把两个 `A` 状态合并成同一个状态，会丢掉“已经完成几个 A”的信息 |
| skip 只跨非重复标签间 blank | 从 s−2 到 s 仅在目标状态为非 blank 且标签与 s−2 不同允许；连续同字必须过中间 blank | 对 `AA` 放开 skip，会把 `(A,A)` 错算成 `AA`；对 `AB` 禁掉所有 skip，会错误要求3帧 |
| 路径边缘化用加法 | 同一字符串的互斥路径事件求和；前向值是路径概率和，不是最好路径分数 | 用 max 替换 sum 得 Viterbi／最佳对齐目标，不再是 CTC 边缘似然 |
| beam 分 blank／非 blank 结尾 | 前缀相同，但下一帧同字可能是“延长当前字”或“新增重复字” | 只存一个总分并无条件扩展同字，会漏算或重复计算 `A`／`AA` |

概率模型的分解是给定完整输入／编码器条件后 \(P(\pi\mid x)=\prod_t p_t(\pi_t\mid x)\)，并不要求原始像素或声音帧相互独立；双向编码器可以看完整输入。限制在于标准 CTC 路径输出不显式以之前已输出的标签串为自回归条件，且对齐必须单调。改变阅读顺序、任意二维文本或语义翻译不能仅靠放宽 blank 解决。[cite:1](#source-1)[cite:4](#source-4)

### L3 工程权衡：长度、数值与拒识

| 参数／协议 | 单位与起始依据 | 权衡、症状、证据和改动顺序 |
|---|---|---|
| T、U、R 与宽度步幅 | 步／词元；逐层按实际卷积／池化尺寸公式算 T，再记录余量 `T−U−R` | 无穷损失先查负余量、截断和实际有效宽；增大 T 增加算力，不保证保留字形，不给通用宽度倍数 |
| blank、词表、目标长度 | 整数 ID；训练、解码与数据侧同一映射 | 大量空串先查 blank 位置、类别轴、未知词元及 padding 长度，不先降低拒识阈值 |
| log 域与精度 | 自然对数；差分检查用 float64，部署先对照 float32 | `softmax` 后取 log 可先下溢；使用稳定 log_softmax，跟踪 NaN／Inf；低精度下不把大负常数当严格不可达 |
| beam 宽 K | 每帧保留的前缀个数；先比较贪心、K=1、2、4、8 的验证集收益和延迟 | 这些是对照预算，不是论文推荐默认值；K=1 前缀 beam 也不等同逐帧贪心；词元裁剪进一步改变近似 |
| `reduction` | 单样本损失单位为 nat；核验使用 `none`／单样本 `sum` | PyTorch mean 不是直接样本均值，而是先除以 `max(U,1)` 再批平均；空串和长串占比改变优化权重 |
| `zero_infinity` | 策略开关，默认 false | 置零损失和梯度相当于让不可行样本不提供训练信号，不修复长度、标签或概率错误；必须独立记录计数与比例 |
| 词典／语言模型／格式 | 对数分数权重、长度奖励需验证集标定 | 不再只优化 CTC 概率；随机编号可能被“纠正”为常见合法错串，保存原串和约束后串 |

PyTorch v2.5.1 的布局、默认值及 mean 的长度下限来自固定源码，不把当前网页／其他后端契约直接套用。其 cuDNN 路线有额外条件，例如 blank=0、拼接目标、整型参数类型和有效长度要求；本卡不验证 GPU 后端。[cite:2](#source-2)[cite:3](#source-3)

**空目标与不可行必须分开。** 对 U=0、T>0，唯一合法路径是全部 blank，\(\mathcal L=-\sum_t\ell_{t,b}\)，并不是自动零损失。数学上 T=0、U=0 的空乘积为1，损失0；T=0、U>0 不可行。长度可行不保证概率非零：若外部 mask 把所有合法路径的概率都置0，仍有 \(P=0\)。此时 \(\log P=-\infty\)、损失为正无穷，后验与普通有限梯度未定义；本卡返回 `None`，可选置零是另一个明确策略。某个框架支持 `input_length=0` 不保证支持物理张量时间维度为0，后者必须独立实测。[cite:2](#source-2)

**调参顺序。** 先固定失败样本和目标真值 → 检查图像裁切／方向、词元映射、T/U/R 和 padding → 核对概率归一化、loss reduction 与梯度 → 对比贪心和不同 beam 及是否带语言先验 → 才改编码器、训练数据和拒识阈值。对 `11→1`，增 beam 不能补回训练时已经不可能的对齐。最终验证应分别统计 CER、整串正确、拒识、空参考和重复字符子集，按设备／字体／批次隔离；本卡没有执行这些 OCR 验收。[cite:5](#source-5)

### L4 数学：前后向、稳定计算与梯度

#### 1. 路径模型与最小帧长

令 \(B\) 先压缩相邻重复，再删 blank，路径 \(\pi\in(\Sigma\cup\{b\})^T\)。则

\[
P(y\mid x)=\sum_{\pi:B(\pi)=y}\prod_{t=0}^{T-1}p_t(\pi_t\mid x),\qquad
\mathcal L=-\log P(y\mid x).
\]

所有长度 T 的路径通过 B 唯一分到各输出串，故在每帧归一化时，全部输出串的概率之和为1。定义 \(R=\sum_{u=1}^{U-1}\mathbf1[y_u=y_{u-1}]\)。每个目标至少占一帧，每对相邻重复至少需要一帧 blank，所以 \(T\ge U+R\) 必要。反过来，按目标依次各放一帧，只在相邻重复之间插一帧 blank，就构成 U+R 帧合法路径；多出的帧可放 blank。因此该条件对**路径集合非空**充要，对每个词元概率严格正的模型也保证 P>0；对硬零概率输入、识别正确和图像可辨识性均不保证。空目标令 R=0。[cite:1](#source-1)[cite:4](#source-4)

#### 2. 前向：包含当前帧

全卡采用0基 t、s。\(\alpha_t(s)\) 为在第 t 帧到达扩展目标状态 s、且包含该帧发射的合法前缀概率。定义前驱集合

\[
\operatorname{Pred}(s)=\{s\}\cup\{s-1:s\ge1\}
\cup\{s-2:s\ge2,\ z_s\ne b,\ z_s\ne z_{s-2}\}.
\]

对 T>0，\(\alpha_0(0)=p_0(b)\)，U>0 时 \(\alpha_0(1)=p_0(y_0)\)，其余0。递推与终止为

\[
\alpha_t(s)=p_t(z_s)\sum_{r\in\operatorname{Pred}(s)}\alpha_{t-1}(r),\qquad
P=\begin{cases}\alpha_{T-1}(0)&U=0,\\
\alpha_{T-1}(S-1)+\alpha_{T-1}(S-2)&U>0.\end{cases}
\]

所有越界状态视为0，不能用 Python 负下标回绕。终态是最后一个目标标签或其后的 blank，不能对全部状态求和；否则把没输出完目标的前缀也计入。空目标只有一个终态，不能重复相加。[cite:1](#source-1)[cite:2](#source-2)

#### 3. 后向：本文不包含当前帧

定义 \(\beta_t(s)\) 为已在 t 帧状态 s 的条件下，剩余 **t+1…T−1** 帧走到合法终态的概率。终态集合 E 在 U=0 时为 `{0}`，否则为 `{S−2,S−1}`：

\[
\beta_{T-1}(s)=\mathbf1[s\in E],\qquad
\beta_t(s)=\sum_{r:s\in\operatorname{Pred}(r)}p_{t+1}(z_r)\beta_{t+1}(r).
\]

后继为 s、s+1，以及满足 \(s+2<S,z_{s+2}\ne b,z_{s+2}\ne z_s\) 的 s+2；其他后继不存在。对任何有效 t 有不变量

\[
P=\sum_s\alpha_t(s)\beta_t(s),\qquad
\gamma_{t,k}=\frac{\sum_{s:z_s=k}\alpha_t(s)\beta_t(s)}{P},\qquad
\sum_k\gamma_{t,k}=1.
\]

**与原文／固定源码的约定桥接。** Graves 式(9)–(11)与 PyTorch `log_beta` 使用包含第 t 帧的后向量 \(\widetilde\beta_t(s)=p_t(z_s)\beta_t(s)\)，所以其终态为当前发射概率，而不是1；后验需以 \(\alpha\widetilde\beta/p_t(z_s)\) 汇总。本文代码使用排除当前帧的版本，避免硬零概率下的 `−inf−(−inf)`。两套约定都成立，但**不能用前者初始化、后者后验式混算**。零发射时用无除法的定义，不直接除零。[cite:1](#source-1)[cite:2](#source-2)

#### 4. log 域、不可达状态与梯度

令 \(A=\log\alpha,D=\log\beta\)。概率0对应 `−inf`，概率1对应0，乘法改加法、求和改为

\[
\operatorname{LSE}(v)=m+\log\sum_i\exp(v_i-m),\quad m=\max_i v_i.
\]

输入为空或全部 `−inf` 时，LSE **直接返回 `−inf`**，不能计算 `−inf−(−inf)`；NaN、正无穷 logits 或整行全被 mask 应拒绝。前向为 \(A_t(s)=\ell_{t,z_s}+\operatorname{LSE}_{r\in Pred(s)}A_{t-1}(r)\)，后向为 \(D_t(s)=\operatorname{LSE}_{r:s\in Pred(r)}(\ell_{t+1,z_r}+D_{t+1}(r))\)。先算有限 \(\log P\)，再用 \(\gamma_{t,k}=\exp(\operatorname{LSE}_{s:z_s=k}(A_t(s)+D_t(s))-\log P)\)。不要先指数化整串 P 后再取 log，也不要用 epsilon 将本应不可行的样本变成有限损失。[cite:2](#source-2)

原文还讨论逐帧缩放抵抗下溢，并不是原文已经使用本文的 log 域实现。若用缩放法，必须记录缩放常数和最终合法终态剩余质量；若未像原文那样剔除不能完成目标的状态，不能机械把“所有状态归一化常数的 log 之和”当目标概率。[cite:1](#source-1)

从路径乘积对单帧概率求导，对于 P>0 且 \(p_t(k)>0\)：

\[
\frac{\partial\mathcal L}{\partial p_t(k)}=-\frac{\gamma_{t,k}}{p_t(k)},\qquad
\frac{\partial\mathcal L}{\partial\ell_{t,k}}=-\gamma_{t,k}
\quad\text{（把各 log 权重视作独立变量）}.
\]

再用 \(\ell_{t,j}=a_{t,j}-\log\sum_k e^{a_{t,k}}\) 和 \(\partial\ell_{t,j}/\partial a_{t,k}=\mathbf1[j=k]-p_t(k)\)，得到

\[
\frac{\partial\mathcal L}{\partial a_{t,k}}
=-\gamma_{t,k}+p_t(k)\sum_j\gamma_{t,j}
=p_t(k)-\gamma_{t,k}.
\]

同一标签可能对应多个扩展状态，必须全部求和；每帧 logits 梯度和应接近0。PyTorch v2.5.1 native backward 返回形式为 `exp(lp)−posterior`，按归一化 log-probabilities 的契约使用；不要把其输入当任意独立 log 权重，离开概率单纯形做有限差分后宣称公式矛盾。本卡差分对象始终是 **logits → log_softmax → CTC** 的组合。批量 reduction 和上游 `grad_out` 还会缩放这些梯度。[cite:1](#source-1)[cite:2](#source-2)

#### 5. 贪心与前缀 beam 的不同目标

贪心得 \(B(\arg\max_\pi P(\pi\mid x))\)，期望求的却是 \(\arg\max_y\sum_{B(\pi)=y}P(\pi\mid x)\)，max 与 sum 不能互换。对前缀 q 保存 \(p_b(q),p_n(q)\)，分别表示当前所有折叠到 q、末帧为 blank／非 blank 的路径概率和。初始空前缀 \(p_b(\epsilon)=1,p_n(\epsilon)=0\)，其余0。每帧创建新的累加表，所有更新都用 `+=`：

\[
p'_b(q)\mathrel{+}=(p_b(q)+p_n(q))p_t(b).
\]

对于每个非 blank c：若 q 非空且末字等于 c，做

\[
p'_n(q)\mathrel{+}=p_n(q)p_t(c),\qquad
p'_n(qc)\mathrel{+}=p_b(q)p_t(c).
\]

否则 \(p'_n(qc)\mathrel{+}=(p_b(q)+p_n(q))p_t(c)\)。同帧所有来源聚合后，按 \(p'_b+p'_n\) 保留前 K 个，不能扩展一个来源就抢先剪枝。实现用 log 域 LSE 累加。空前缀没有末字，不应访问 `q[-1]`；空输出本身是合法候选，不能凭“至少输出一字”的偏好删除。[cite:4](#source-4)

**历史与保证边界。** Graves 2006 §3.2 的 prefix search 是按扩展前缀概率搜索、在完成序列超过剩余前缀上界时终止；原文另用 blank 分段启发式。这里的逐帧固定宽度 prefix beam 是现代常用近似，不能冒称原文的同一程序。不剪枝的前缀聚合等价于全部路径求和，但前缀数可指数增长；固定 K 丢弃的质量不会自动恢复，不保证全局 MAP，也不保证每个样本的识别正确率随 K 单调上升。[cite:1](#source-1)[cite:4](#source-4)

#### 6. 复杂度、可辨识性和误差边界

已知目标的前向／后向各为 \(O(T(U+1))\)，完整状态存储 \(O(T(U+1))\)；只求 loss 可滚动两行至 \(O(U+1)\)，输出全部 C 类梯度额外需要 \(O(TC)\)。参考代码按各类扫描全部状态汇总后验，教学版耗时为 \(O(TC(U+1))\)；生产实现可一次遍历状态按标签聚合恢复 \(O(T(U+1)+TC)\)。贪心扫描概率表为 \(O(TC)\)。beam 不计前缀拷贝时每帧至多 KC 次扩展，完全排序约 \(O(TKC\log(KC))\)，前缀字符串拷贝和词典／语言模型另计。

CTC 边缘化不会辨识唯一真实字边界，多条对齐可能同样合理；尖峰位置不等于物理字符框。动态规划的精确性只针对既定离散概率模型；浮点舍入、beam 剪枝、编码器错误是不同误差源。**未提供通用识别误差界、固定 beam 最优差距界或神经网络训练全局收敛保证**。

### L5 实现：源码对应、可执行参考与核验

#### 1. 固定实现与数据布局

固定参照 **PyTorch v2.5.1**，不是浮动 `main`；实际本地交叉运行版本另记。源码读取涉及以下位置：[cite:2](#source-2)[cite:3](#source-3)

| 源码位置／符号 | 与本卡的对应及注意点 |
|---|---|
| `aten/src/ATen/native/LossCTC.cpp:get_target_prime` | 偶数状态为 blank，奇数取目标 `(s/2)`；状态索引不是类别 ID |
| `ctc_loss_allocate_outputs` | `log_probs` 为 `[T,N,C]`；目标 `[N,Smax]` 或拼接的一维目标；有效长度按样本给出；有 stride／offset 逻辑 |
| `ctc_loss_cpu_template` | 初态、三项 logsumexp、重复标签禁 skip、末尾两态／空目标一态、有效 T=0 分支；无需显式 `current!=blank` 是因为偶数状态与 s−2 同为 blank |
| `ctc_loss_backward_cpu_template` | 使用含当前帧的 beta，按标签累加 alpha+beta 后减回当前 log 概率；尾部 padding 梯度清零；无穷损失按策略置零 |
| `ctc_loss_impl`／`torch/nn/modules/loss.py:CTCLoss` | reduction、`zero_infinity`、无批次输入包装及 API 契约；mean 的分母使用目标长度下限1 |

不能仅凭 native 代码缺少某项检查就把非法目标送进内核。调用方应验证 ID 范围、目标不含 blank、有效长度和归一化；批次 N/T 轴交换、目标拼接偏移错误和错误 padding mask 都可能产生“能算但学错”。T=0 的数学契约由下面参考定义；官方某后端的物理零时间张量是否接受不由源码中的单样本零长度分支推断。

#### 2. 可执行伪代码／教学参考

以下 Python 只依赖标准库，固定 `blank=0`，单样本 `[T,C]`；C 单独传入以便表达 T=0。`ctc` 返回 `(logP, logits_gradient, occupancy)`；P=0 返回 `(-inf,None,None)`，不自动启用置零。代码存放在文档，不新增 `.py`、训练器或解码产品接口。

```python
import math
from itertools import product, groupby


NEG = float('-inf')


def lse(values):
    values = list(values)
    m = max(values, default=NEG)
    return NEG if m == NEG else m + math.log(math.fsum(math.exp(v - m) for v in values))


def logsoftmax(a):
    if any(not row or any(not math.isfinite(v) for v in row) for row in a):
        raise ValueError('finite nonempty logits rows required')
    return [[v - lse(row) for v in row] for row in a]


def validate(lp, c):
    if type(c) is not int or c < 1:
        raise ValueError('invalid class count')
    for row in lp:
        if len(row) != c or any(math.isnan(v) or v == math.inf for v in row):
            raise ValueError('invalid log probabilities')
        if not math.isclose(lse(row), 0.0, abs_tol=1e-9):
            raise ValueError('rows must be log-normalized')


def collapse(path):
    return tuple(k for k, _ in groupby(path) if k != 0)


def ctc(lp, y, c):
    validate(lp, c)
    y = tuple(y)
    if any(type(k) is not int or not 0 < k < c for k in y):
        raise ValueError('target must exclude blank and invalid IDs')
    tmax, u = len(lp), len(y)
    repeats = sum(y[i] == y[i - 1] for i in range(1, u))
    if tmax < u + repeats:
        return NEG, None, None
    if tmax == 0:
        return 0.0, [], []
    z = [0]
    for k in y:
        z.extend([k, 0])
    size = len(z)
    pred = []
    succ = [[] for _ in z]
    for s in range(size):
        p = [s]
        if s > 0:
            p.append(s - 1)
        if s > 1 and z[s] != 0 and z[s] != z[s - 2]:
            p.append(s - 2)
        pred.append(p)
        for r in p:
            succ[r].append(s)
    alpha = [[NEG] * size for _ in lp]
    alpha[0][0] = lp[0][0]
    if u:
        alpha[0][1] = lp[0][y[0]]
    for t in range(1, tmax):
        for s in range(size):
            alpha[t][s] = lp[t][z[s]] + lse(alpha[t - 1][r] for r in pred[s])
    ends = [size - 1] if not u else [size - 2, size - 1]
    logp = lse(alpha[-1][s] for s in ends)
    if logp == NEG:
        return NEG, None, None
    beta = [[NEG] * size for _ in lp]
    for s in ends:
        beta[-1][s] = 0.0
    for t in range(tmax - 2, -1, -1):
        for s in range(size):
            beta[t][s] = lse(lp[t + 1][z[r]] + beta[t + 1][r] for r in succ[s])
    gamma, grad = [], []
    for t in range(tmax):
        total = lse(alpha[t][s] + beta[t][s] for s in range(size))
        assert math.isclose(total, logp, abs_tol=1e-9)
        q = [math.exp(lse(alpha[t][s] + beta[t][s]
                         for s in range(size) if z[s] == k) - logp) for k in range(c)]
        gamma.append(q)
        grad.append([math.exp(lp[t][k]) - q[k] for k in range(c)])
    return logp, grad, gamma


def prefix_beam(lp, c, width=None):
    validate(lp, c)
    if width is not None and (type(width) is not int or width < 1):
        raise ValueError('positive beam width required')
    beam = {(): [0.0, NEG]}
    for row in lp:
        nxt = {}
        def add(q, end, value):
            if value != NEG:
                pair = nxt.setdefault(q, [NEG, NEG])
                pair[end] = lse([pair[end], value])
        for q, (pb, pn) in beam.items():
            total = lse([pb, pn])
            add(q, 0, total + row[0])
            for k in range(1, c):
                if q and q[-1] == k:
                    add(q, 1, pn + row[k])
                    add(q + (k,), 1, pb + row[k])
                else:
                    add(q + (k,), 1, total + row[k])
        ranked = sorted(nxt.items(), key=lambda item: (-lse(item[1]), item[0]))
        beam = dict(ranked if width is None else ranked[:width])
    return {q: lse(pair) for q, pair in beam.items()}


def greedy(lp, c):
    validate(lp, c)
    return collapse([max(range(c), key=lambda k: row[k]) for row in lp])
```

#### 3. 穷举、有限差分与官方库交叉检查

下一块接在上一块后执行。穷举使用所有路径，折叠用独立的逐帧“与原路径前一符号比较”实现，避免与 `groupby` 共享同一错误；另汇总每帧类别占用，对照后向。确定性 logits、T=0…5、U=0…4、词表 `{A,B}`；不剪枝前缀表还核对所有可达输出串，包含 U>4 的情形。差分步长 `1e-5`、阈值是本次数值检查预算，不是理论误差界。Torch 可选部分使用已有环境，不在这里安装或修改项目依赖。

```python
def oracle(lp, c):
    out = {}
    for path in product(range(c), repeat=len(lp)):
        q = tuple(k for t, k in enumerate(path) if k and (t == 0 or k != path[t - 1]))
        p = math.prod(math.exp(lp[t][k]) for t, k in enumerate(path))
        mass, occ = out.setdefault(q, [0.0, [[0.0] * c for _ in lp]])
        out[q][0] = mass + p
        for t, k in enumerate(path):
            occ[t][k] += p
        assert collapse(path) == q
    return out


def logits(tmax, c=3):
    return [[(((t + 1) * (k + 2)) % 7 - 3) / 3 for k in range(c)] for t in range(tmax)]


def check_reference():
    targets = [q for u in range(5) for q in product((1, 2), repeat=u)]
    cases, paths, finite, max_mass, max_occ = 0, 0, 0, 0.0, 0.0
    for tmax in range(6):
        lp = logsoftmax(logits(tmax))
        ref = oracle(lp, 3)
        paths += 3 ** tmax
        beam = prefix_beam(lp, 3)
        assert set(beam) == set(ref)
        assert math.isclose(sum(v[0] for v in ref.values()), 1, abs_tol=1e-12)
        for q in ref:
            assert math.isclose(math.exp(beam[q]), ref[q][0], abs_tol=1e-12)
        for q in targets:
            cases += 1
            mass, occ = ref.get(q, (0.0, None))
            logp, grad, gamma = ctc(lp, q, 3)
            feasible = tmax >= len(q) + sum(q[i] == q[i - 1] for i in range(1, len(q)))
            assert (mass > 0) == feasible
            if mass == 0:
                assert logp == NEG and grad is None and gamma is None
                continue
            finite += 1
            max_mass = max(max_mass, abs(math.exp(logp) - mass))
            for t in range(tmax):
                assert abs(sum(grad[t])) < 1e-12
                for k in range(3):
                    max_occ = max(max_occ, abs(gamma[t][k] - occ[t][k] / mass))
    assert max_mass < 1e-12 and max_occ < 1e-12
    step, coordinates, max_diff = 1e-5, 0, 0.0
    for tmax, q in [(3, ()), (3, (1, 1)), (3, (1, 2)), (4, (1, 2, 1)), (1, (1,))]:
        a = logits(tmax)
        _, grad, _ = ctc(logsoftmax(a), q, 3)
        for t in range(tmax):
            for k in range(3):
                plus, minus = [row[:] for row in a], [row[:] for row in a]
                plus[t][k] += step
                minus[t][k] -= step
                numeric = (-ctc(logsoftmax(plus), q, 3)[0] + ctc(logsoftmax(minus), q, 3)[0]) / (2 * step)
                max_diff = max(max_diff, abs(numeric - grad[t][k]))
                coordinates += 1
    assert max_diff < 1e-8
    lp = [[math.log(.6), math.log(.4)]] * 2
    scores = prefix_beam(lp, 2)
    assert greedy(lp, 2) == () and max(scores, key=scores.get) == (1,)
    assert math.isclose(math.exp(scores[(1,)]), .64, abs_tol=1e-12)
    narrow = prefix_beam(lp, 2, 1)
    assert max(narrow, key=narrow.get) == ()
    other = [[math.log(v) for v in row] for row in [[.2, .6, .2], [.25, .35, .4]]]
    narrow = prefix_beam(other, 3, 1)
    assert greedy(other, 3) == (1, 2) and max(narrow, key=narrow.get) == (1,)
    logp, _, _ = ctc([[math.log(.5), math.log(.5)]] * 2000, (), 2)
    assert math.isfinite(logp) and abs(logp + 2000 * math.log(2)) < 1e-8
    assert math.exp(logp) == 0.0
    assert ctc([[0.0, NEG]], (1,), 2) == (NEG, None, None)
    assert ctc([[0.0, NEG]], (), 2)[0] == 0.0
    assert prefix_beam([[0.0, NEG]], 2) == {(): 0.0}
    assert ctc([], (), 2) == (0.0, [], [])
    assert ctc([], (1,), 2) == (NEG, None, None)
    assert lse([NEG, NEG]) == NEG and lse([]) == NEG
    bad = [([[0.0, NEG]], (0,), 2), ([[0.0, NEG]], (2,), 2),
           ([[NEG, NEG]], (), 2), ([[math.nan, 0.0]], (), 2),
           ([[math.inf, 0.0]], (), 2), ([[0.0, 0.0]], (), 2),
           ([[0.0]], (), 2)]
    for args in bad:
        try:
            ctc(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    print('reference', cases, 'cases', paths, 'paths', finite, 'feasible')
    print('max_mass', max_mass, 'max_occupancy', max_occ)
    print('finite_difference', coordinates, 'coordinates', max_diff)
    print('edge_cases PASS; long_empty_logP', logp)


def check_torch():
    import torch
    import torch.nn.functional as F
    max_loss, max_grad, count = 0.0, 0.0, 0
    for tmax in range(1, 6):
        for u in range(5):
            for q in product((1, 2), repeat=u):
                logp, grad, _ = ctc(logsoftmax(logits(tmax)), q, 3)
                if logp == NEG:
                    continue
                a = torch.tensor(logits(tmax), dtype=torch.float64, requires_grad=True)
                loss = F.ctc_loss(a.log_softmax(-1).unsqueeze(1), torch.tensor(q, dtype=torch.long),
                                  [tmax], [u], blank=0, reduction='sum')
                loss.backward()
                max_loss = max(max_loss, abs(loss.item() + logp))
                max_grad = max(max_grad, (a.grad - torch.tensor(grad, dtype=torch.float64)).abs().max().item())
                count += 1
    assert max_loss < 1e-10 and max_grad < 1e-10
    for zero in (False, True):
        a = torch.tensor(logits(2), dtype=torch.float64, requires_grad=True)
        loss = F.ctc_loss(a.log_softmax(-1).unsqueeze(1), torch.tensor([1, 1]),
                          [2], [2], reduction='sum', zero_infinity=zero)
        assert (loss.item() == 0.0) if zero else math.isinf(loss.item())
        if zero:
            loss.backward()
            assert torch.equal(a.grad, torch.zeros_like(a.grad))
    a = torch.tensor(logits(4), dtype=torch.float64, requires_grad=True)
    loss = F.ctc_loss(a.log_softmax(-1).unsqueeze(1), torch.tensor([1]), [2], [1], reduction='sum')
    loss.backward()
    assert torch.equal(a.grad[2:], torch.zeros_like(a.grad[2:]))
    for q in [(), (1,)]:
        a = torch.tensor(logits(1), dtype=torch.float64)
        loss = F.ctc_loss(a.log_softmax(-1).unsqueeze(1), torch.tensor(q, dtype=torch.long),
                          [0], [len(q)], reduction='sum')
        assert (loss.item() == 0.0) if not q else math.isinf(loss.item())
    print('torch', torch.__version__, count, 'cases', 'max_loss', max_loss, 'max_grad', max_grad)
    print('torch zero_infinity/padding/effective_zero_length PASS')


check_reference()
check_torch()
```

复跑时用 Python 读取本文件两个 `python` 代码块，按顺序 `exec` 即可，不需要将脚本落盘。没有 torch 的环境只执行 `check_reference()`；其结果不能冒充官方库差分。该参考没有批量加速、词典、语言模型、CUDA、模型训练或真实图像输入。优化前应保持穷举总质量、后验、损失和 logits 梯度等价，再剖析状态聚合、滚动数组、分桶与硬件内核；不先以剪枝换速度再沿用“精确等价”措辞。

#### 4. 实际执行记录与未验证项

2026-11-05 在 Windows、Python **3.13.14**、现有 **PyTorch 2.14.1+cpu** 环境中，按文档提取两个代码块执行，最终退出码0。CPU／float64／单样本 sum；**本地运行版本不是静态参照的 v2.5.1**，仅作另一现有版本的有限行为交叉，不声称已在 v2.5.1 复现。未安装新依赖，未生成测试脚本文件。以下均为实际输出：

| 检查 | 实际结果 | 范围 |
|---|---|---|
| 全路径与目标组合 | 364 条路径；186 个 `(T,y)` 组合，其中56个可行、130个不可行，全部通过 | T=0…5、C=3；每个 T 的31个目标，覆盖空串、AA／BB 等重复与非重复 |
| 前向概率／后向占用 | 最大绝对误差分别为 `8.326672684688674e-17`、`8.881784197001252e-16` | 独立穷举概率与类别占用；每帧后验和1、logits梯度和0 |
| 无剪枝前缀聚合 | 全部可达输出的键集合和概率与穷举一致 | 包含空前缀、不同路径合并、连续重复、宽度1剪枝反例；不是任意 K 的最优保证 |
| logits 中心差分 | 42个坐标，最大绝对误差 `3.3868713389395566e-11` | 空目标、重复、非重复、单帧；步长 `1e-5`，阈值 `1e-8` |
| 长序列稳定性 | 2000帧空目标 `logP=-1386.2943611198225`，有限；`exp(logP)=0.0` | 展示 log 域保留损失而概率域下溢，不是识别成绩 |
| 输入与边界 | 全部通过 | 含 blank／越界目标、全−inf／NaN／+inf／未归一化／错列数输入拒绝；硬零路径、数学 T=0、空目标、空 LSE |
| PyTorch CPU 交叉 | 55例；最大 loss 差 `1.7763568394002505e-15`，最大 logits 梯度差 `1.6653345369377348e-15` | 非零物理 T 的全部可行目标；显式 float64 双方比较 |
| 库策略与长度 | 全部通过 | `AA,T=2` 的 Inf 与置零策略／零梯度；padding 梯度为0；物理 T=1 而有效长度0的空／非空目标 |

第一次执行发现检查器误把无剪枝赢家 `A` 当作上例 K=1 的预期，已修正并重跑：K=1 在第1帧删掉 A，最终为空；另用两帧 `(.2,.6,.2)`、`(.25,.35,.4)` 检验 K=1 与贪心确实可不同，分别返回 A 和 AB。此修正针对测试预期，不掩盖剪枝丢质量，也没有把 beam 近似包装为精确搜索。

复跑命令（在仓库根目录；`python` 替换为具有 torch 的实际解释器）：

```powershell
@'
from pathlib import Path
import re
p = Path('docs/algorithm_cards/ctc.md')
blocks = re.findall(r'```python\n(.*?)\n```', p.read_text(encoding='utf-8'), re.S)
assert len(blocks) == 2
exec(compile('\n\n'.join(blocks), '<CT01 extracted checks>', 'exec'))
'@ | python -X utf8 -
```

本次未运行物理张量 T=0 的库接口、批量拼接／二维填充互转、非零 blank 的库用例、mean 的数值比对、CUDA／cuDNN／半精度、词典／语言模型、OCR 数据、训练收敛、CER 或现场验收。相关公式／契约说明不冒充这些测试通过。公式、确定性测试和外部实现比对只支持算法内核的有限正确性；生产实现状态仍为**未实现**。

### 面试自测：贯穿 L1–L5 的5问

| 层级／问题 | 参考答案与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：两帧 `(b=.6,A=.4)`，训练 `A` 与推理各得到什么？ | 合法路径三条，总概率 .64；损失 `−log(.64)`；贪心为空，无剪枝前缀聚合选 A；训练指定目标，推理无目标 | 把最好路径 .36 当 `P(A)`，或认为训练时运行 beam | L1 完整走例；L5 检查器 |
| L2：`A,b,A` 为什么是 AA？为什么两个 A 之间不能 skip？ | 先合并原路径相邻重复再删 blank；扩展目标中两 A 状态不同，需要中间 blank；最短3帧 | 先删 blank，或只看当前符号相同便合并目标状态 | L2 机制表；原文§3.1、§4.1 [cite:1](#source-1) |
| L3：长编号训练 Inf，开 `zero_infinity`、加 beam 能解决吗？ | 先记录 T/U/R、有效宽与目标映射，判断结构不可行或硬零；置零只是屏蔽训练贡献；推理 beam 不改变训练路径存在性；空目标不是不可行 | 只检查 T≥U；认为损失0意味着识别成功；把 blank 当空格 | L3 参数表；固定源码 [cite:2](#source-2)[cite:3](#source-3) |
| L4：从 alpha/beta 推到 logits 梯度，何时要除当前概率？ | 本文 beta 不含当前帧，`alpha*beta/P` 得占用；原文 beta 含当前帧则除一次当前概率；按同类所有状态求和，再由 log_softmax 得 `p−gamma`；仅 P>0 可用 | 混合后向定义、重复发射；对独立 log 权重求导仍答 `p−gamma` | L4 第3–4节；原文式(14)–(16) |
| L5：loss 与库相近但梯度不符，如何核验和优化？ | 固定 float64、CPU、单样本 sum，差分 logits 的组合函数；查 beta 终态、skip、同类聚合、归一化、dtype、有效长度和 reduction；先过穷举／占用／梯度，再优化存储 | 只比较非重复非空目标；拿 GPU 默认 mean 对比单样本 sum；用 epsilon 隐藏 Inf | L5 固定源码表、可复跑检查与实测范围 |

## 来源层级与读取范围

可信度采用 P0 原始／官方、P1 权威二手；获取层级用 B 书目、A 摘要、T 正文／源码，**T部分不等于全文通读**。核验日期均为2026-11-05。来源只支持各自声明的结论，不把同论文镜像或官方实现的引用关系算成独立效果证据。

- <a id="source-1"></a>**[cite:1] P0，Graves 等，ICML 2006 原论文。** [作者 PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)。实际读取：网页抽取的§1–3及§4开头；随后从同一 URL 内存读取8页 PDF，抽取第4–6页，核对§4.1–4.2、式(6)–(16)、边界、重复标签、缩放与 softmax 梯度。属于 **T相关部分**，未逐页核验全部实验与图表，不声称 TIMIT 复现。折叠／路径模型是原始证据；现代记号、最小帧长充要论证及排除当前帧 beta 为本卡显式重述推导。
- <a id="source-2"></a>**[cite:2] P0，PyTorch v2.5.1 官方固定实现。** [LossCTC.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/LossCTC.cpp) · [raw](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/LossCTC.cpp)。T 源码：读取 `get_target_prime`、分配／长度、CPU alpha/beta、logsumexp、梯度和包装 reduction 相关段。网络正文初次截断后另按段提取后向与包装；只核对 CPU 路线，不声称编译运行了 v2.5.1 或核验 CUDA 内核。
- <a id="source-3"></a>**[cite:3] P0，PyTorch v2.5.1 API 契约源码。** [loss.py:CTCLoss](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py) · [raw](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/nn/modules/loss.py)。T 对应类正文／docstring：核对 `[T,N,C]`／`[T,C]`、目标无 blank、拼接／填充、reduction、`zero_infinity` 与 cuDNN 约束。版本 API 网页抓取为空，已改读固定源码，不将失败网页计为已读资料。
- <a id="source-4"></a>**[cite:4] P1，Awni Hannun，Sequence Modeling With CTC，Distill，2017-11-27。** [正文](https://distill.pub/2017/ctc/)，DOI `10.23915/distill.00008`。T相关部分：读取折叠、单调性、动态规划、推理；初次结果截断后另提取 prefix beam 段，核对同字延长／新重复、blank／非 blank 双概率和先聚合后排名。用于现代解释与实现机制交叉，非新的独立识别实验，也不是 Graves2006 原代码。
- <a id="source-5"></a>**[cite:5] 本仓库规范与交接依据，不充当外部学术证据。** [学习规范](../algorithm_learning_standard.md)、[OCR 总览](../ocr_codes_evolution.md)、[覆盖表](../evolution_coverage.md)、[requirements-dl.txt](../../requirements-dl.txt)。实际读取 L1–L5、来源／实现诚实边界、O-H2、T/U/R、G27／08-02 和依赖声明；确定只补单卡，不升级 CRNN 或 OCR 实现状态。

**证据结论。** 路径求和、重复规则和前向机制由原文、官方实现、权威解释交叉核对；后向／梯度由原文与固定源码及有限数值检查核验。前缀 beam 具体组织以 Distill 解释和本卡穷举为依据，不冒称三份独立原始证明。L3 的部署收益、GPU 后端、真实数据、训练稳定性与拒识阈值仍未验证；这些保留项不影响本卡承接 G27 的具体内容缺口，但阻止把它登记为完整 OCR 实现或现场验收通过。
