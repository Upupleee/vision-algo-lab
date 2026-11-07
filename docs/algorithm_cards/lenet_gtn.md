# LG01 LeNet-5／GTN：原字符网络与图变换网络的全局训练

[学习规范](../algorithm_learning_standard.md) · [OCR 总览 O-H1](../ocr_codes_evolution.md#3-历史演进线-aocr-识别与检测) · [覆盖台账 G26／08-01](../evolution_coverage.md) · [卷积组件 C05](learning_components.md#c05-二维卷积-conv2d)

<a id="lg01"></a>
<a id="g26"></a>

## 任务、演进与覆盖状态

本文件只有 **LG01 一张卡**，承接 **G26／08-01／O-H1 LeNet-5／GTN**，包含两个分别展开 L1–L5 的子方法。LeNet-5 学习字符图像到类别代价的映射；Graph Transformer Networks（GTN）组织切分、识别、约束与路径评分，并把整串监督传回可训练模块。二者可组合，不是两个别名，也不是“CNN 后接一个名字叫 GTN 的分类层”。此处 GTN 不是现代注意力 Graph Transformer，也不是图节点分类 GNN。[cite:1](#source-1)

**演进定位：**手工字符特征难覆盖字形变化 → 局部连接、空间权重共享与下采样学习表示 → 仍需训练数据、归一化及拒识；切字和识别独立优化难保证整串最优 → 图保留多种切分／标签解释，以全局目标联合调节评分 → 增加图构造、路径搜索、梯度溯源与资源管理成本，候选遗漏和成像信息缺失仍不能靠反传补回。字符网络、过切分、SDNN 滑动识别、语言约束是有组合及替代关系的模块，不是一条无条件替代链。[cite:1](#source-1)

核验日：**2026-11-05**。方法书目为 LeCun、Bottou、Bengio、Haffner，*Proceedings of the IEEE* 86(11):2278–2324，1998，DOI `10.1109/5.726791`。本卡的精确页码、式号和 Table I **以作者网站的 46 页投稿稿为准**，其页眉为 `SUBMITTED TO PROCEEDINGS OF THE IEEE, 1998`；正式版的 GTN 全局训练为 §VI、图转换为 §VIII，作者稿相应为 §VII、§IX，不能直接混引。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

| 独立字段 | LeNet-5 子方法 | GTN 子方法 |
|---|---|---|
| 任务／输入输出 | 已定位、归一化的单字符灰度图 → 10 个数字类别的平方距离代价，训练另需标签 | 字符串图像及候选图、字符集／语法 → 最优路径、原始字符串及代价；训练另需整串标签 |
| 适用前提 | 字形可辨识，输入协议和字符集固定，训练分布有覆盖 | 目标至少有一条合法候选路径，图有限、评分可计算，可训练数值映射可微或分段可微 |
| L1 步骤 | 已核验：逐层、部分连接、训练／推理契约 | 已核验：图构造、识别变换、组合、路径选择、评分与训练流程 |
| L2 机制 | 已核验：共享、可训练下采样、缩放 tanh、RBF 与竞争目标 | 已核验：保留候选、隐切分、全局竞争、图与梯度的区别 |
| L3 工程权衡 | 部分：来源明确的结构常数、参数尺度和诊断；无任务数据定参 | 部分：候选预算、代价尺度、约束与剪枝诊断；无真实数据定参 |
| L4 数学推导 | 已核验：限定为结构代数、损失及局部梯度 | 已核验：限定有限 DAG 的路径目标、动态规划、占用率及组合梯度 |
| L5 实现 | 部分：原机制伪代码、参数算术及局部梯度检查；未取得原训练源码／权重 | 部分：固定现代官方源码静态核对、伪代码及局部公式检查；未构建完整 GTN |
| 面试自测 | 与 GTN 共用贯穿 L1–L5 的 5 问，每问覆盖两者 | 不是 LeNet 答5、GTN 仅提名；也不将共享5问登记为答10 |
| 仓库实现 | Python 源码关键词检索未发现 LeNet／GTN 专用实现 | C05 通用卷积不能代替这两个子方法 |
| 验证范围 | 参数求和、缩放激活及下采样／RBF 局部有限差分 | 全路径与目标路径占用率差的局部有限差分；非完整图程序验证 |

“已核验”限定文献与列明的数学范围，不是已完成训练、前向一致性或现场验收。L3/L5 的工程实证缺口保留；两子方法正文完整展开不等于所有状态都应填“核”。

## 子方法 A：LeNet-5

### L1 步骤：输入、分层与训练推理

**输入契约。**教学布局记为 NCHW，输入 `[N,1,32,32]`；原文是大致尺度归一化并居中的字符，而不是直接把任意 ROI 拉成正方形。作者稿 §II-B 指出数据库字符至多约 20×20、位于 28×28 场内，而网络接收 32×32，以给高层感受野保留边界余量。白背景为 −0.1，黑前景为 1.175；若原灰度 g 在 `[0,255]` 且 255 为白，线性映射可写 `−0.1+1.275×(1−g/255)`，这是按端点推导的实现表达，不是原始读取代码。该映射只对应约零均值、约单位方差的原数据统计，不保证任意图像都满足。[cite:1](#source-1)

所有 C 层用 5×5 valid、stride=1；S 层用不重叠 2×2、stride=2。到 F6 为止，在加偏置后应用同一缩放激活；输出 RBF 不再加 tanh。层名保留 C1/S2/C3/S4/C5/F6，不能按今天“只数卷积层”的方式猜结构。[cite:1](#source-1)

| 层 | 计算与依赖 | 输出，不含 N | 可训练标量 |
|---|---|---|---|
| C1 | 单输入通道 → 6 个 5×5 核，逐图一个 bias，再激活 | 6×28×28 | 6×(25+1)=156 |
| S2 | 对 C1 每图 2×2 求和，乘该图可训练系数 a，再加 b 和激活 | 6×14×14 | 6×2=12 |
| C3 | 按 Table I 只连接指定 S2 图；对所连图分别卷积并求和，一个输出图一个 bias，再激活 | 16×10×10 | 60×25+16=1516 |
| S4 | 与 S2 同机制，每图各自 a、b | 16×5×5 | 16×2=32 |
| C5 | 每个输出图连接全部16个 S4 图，各5×5卷积，再激活 | 120×1×1 | 120×(16×25+1)=48120 |
| F6 | 120维 → 84维全连接，含84个bias，再激活 | 84 | 84×(120+1)=10164 |
| 输出 RBF | 每类一个84维原型，输出与 F6 的平方欧氏距离 | 10 | 本卡固定原型，0；另存840个常量 |

到 F6 **合计60000个可训练参数**。固定 RBF 原型不计入可训练参数；若实验选择学习原型则增加840，但已改变参数协议，并必须避免下述无竞争目标的塌缩。C5 在32×32输入下看起来等价于一次全连接，但原定义是卷积；扩大输入后空间输出不再1×1，F6及输出也需相应改为滑动应用，不能只改输入尺寸后照旧 flatten。[cite:1](#source-1)

**C3 完整连接表，索引从0开始：**每行列出一个 C3 输出图读取的 S2 图，不是输出位置或固定随机分组。

| C3 图 | S2 图集合 | C3 图 | S2 图集合 |
|---|---|---|---|
| 0 | 0,1,2 | 8 | 2,3,4,5 |
| 1 | 1,2,3 | 9 | 0,3,4,5 |
| 2 | 2,3,4 | 10 | 0,1,4,5 |
| 3 | 3,4,5 | 11 | 0,1,2,5 |
| 4 | 0,4,5 | 12 | 0,1,3,4 |
| 5 | 0,1,5 | 13 | 1,2,4,5 |
| 6 | 0,1,2,3 | 14 | 0,2,3,5 |
| 7 | 1,2,3,4 | 15 | 0,1,2,3,4,5 |

表来自作者稿 p.7 Table I、p.8 配套说明：前6图每图3输入，随后6图为连续4输入，随后3图为非连续4输入，最后1图为全部6输入，故输入图／输出图对数 `6×3+6×4+3×4+6=60`。普通 `Conv2d(6,16,5)` 有96对，**不是原 C3**；groups 也不能直接表达这些重叠、不等长集合。[cite:1](#source-1)

**激活和输出。**原非线性为 `f(u)=1.7159×tanh((2/3)u)`，不是 ReLU，也不是默认幅度1的 tanh。F6 的84来自7×12字符位图编码，每类原型分量为 ±1；原文 Fig.3 展示整套可打印字符编码，数字任务只取相应10类。原型向量是分布式目标码，不是10维 one-hot；RBF 输出是距离代价，越小越好，不是直接输出 `exp(−distance)`，也不是现代线性10类 logits＋softmax 头。[cite:1](#source-1)

**训练和停止。**先锁数据划分、字符集、裁切及原型码表，再构建一次参数并跨样本复用；每步清梯度 → 前向全部层 → 按声明的原文 MSE/MLE 或 MAP 目标算损失 → 反向并累加共享权重梯度 → 更新可训练参数。原文附录描述逐样本随机更新及对角 Gauss–Newton 近似预条件的 stochastic diagonal Levenberg–Marquardt，并非现代 Adam 配方。本卡给出可复现目标和梯度，不声称恢复完整原训练器。使用预先冻结的训练预算和验证选择规则；非有限数、尺寸错误、码表缺失立即终止该运行，预算耗尽不代表达到全局最优。[cite:1](#source-1)

**推理。**同预处理前向得到10个代价，取最小者并保留全部代价及最小／次小差。拒识阈值需独立校准；MAP 中的固定背景项可以解释拒识方向，但不保证工业未知字符识别。字符输出不等于文本定位、整串识别或 GTN 已建立。

### L2 机制：原设计与替换后果

| 设计 | 解决的困难及因果机制 | 删除／替换的后果与边界 |
|---|---|---|
| 局部连接＋空间共享 | 用同一组检测器扫描各处，减少自由参数并利用二维局部结构 | 全连接不具备同样结构先验；卷积的平移等变还受 valid 边界、后续stride与裁切限制 |
| C3 部分连接 | 控制参数，并让不同输出图接收不同输入子集，打破结构对称、鼓励互补特征 | 换密集卷积多900参数且改变函数类；不是仅更换存储格式 |
| 可训练下采样 | 降低位置精度，a/b控制局部汇聚后的工作点和非线性程度 | 固定平均池化没有这些自由度，最大池化的选择导数也不同；无法承诺任意平移不变 |
| 缩放 tanh | 有符号、零附近较大斜率；±1目标在非饱和工作区而不是输出极限 | ReLU改变符号域与曲率；未缩放tanh的±1位于渐近极限，码表和优化关系改变 |
| 固定分布式 RBF 码 | F6学习接近正确字形原型；近似字符允许近似目标码 | 固定原型限制所有类中心完全合并；仍不保证没有坏驻点、未知类高匹配或码冲突 |
| MAP 竞争项 | 同时压低正确类代价、相对推高其他类代价，考虑背景候选 | 只降低正确类代价不评价竞争解释；原型可训练时可出现所有原型及表示相同的零损失塌缩 |

原文 S2 写的是“四输入**相加**后乘可训练系数”。1999 作者后续章用“average”描述相同思想；若用均值替求和，需把系数改为 `a_mean=4a_sum` 才前向等价，初始化和优化器参数也要随重参数化核对。不可将两份表述混成“原层就是无参数 AvgPool”。[cite:1](#source-1)[cite:4](#source-4)

### L3 工程权衡：尺度、调参及诊断

| 参数／协议 | 单位、依据与起点 | 失效证据、调节顺序与代价 |
|---|---|---|
| 输入尺度与居中 | 像素，固定32×32及原归一化端点 | 先保存归一化图检查截笔、极性和偏心；直接拉伸可能改变字符，不能先增模型容量掩盖 |
| C3掩码／核尺寸／通道 | 离散结构，按 L1 固定，不是性能调参范围 | 对照1516及60000参数；改成密集层须另标现代变体并重新训练 |
| tanh 的 A/S | 无量纲，1.7159及2/3来自附录A | 检查pre-activation分布、饱和比例、导数；再查输入量程和初始化 |
| S2/S4 的 a/b | 归一化特征域内的增益／偏置，每图2个 | 先确认sum/mean系数，再看饱和及梯度；原始逐参数初值未取得，不编造默认1/0 |
| 权重初始尺度 | 作者稿附录A使用与fan-in相关的均匀范围，明确写到±2.4/F_i | 不静默替成He初始化或±2.4/√F_i；共享权重fan-in、输入相关性和最终实现须另核对 |
| RBF原型／背景代价 J | 原型±1；J>0，与平方距离同为无量纲代价 | 先核码表、是否固定，再在验证集校准拒识；本次未取得可直接加载码表或任务专用J |
| 优化步长／预条件 | 参数更新尺度依梯度和曲率；不是统一学习率处方 | 小批可学习性→梯度有限值→步长，再评估预算；无原完整配置，不报“原论文精确复现” |
| 字符数量及数据覆盖 | 数字10类；扩充字符集需增加原型且验证可分性 | O/0等近似码与近似字形同时存在时，必须保留候选并考虑整串约束及拒识 |

排障顺序为 **真实图／标签与划分 → 预处理 → shape和C3连接 → 码表／损失 → 局部梯度 → 小样本学习 → 泛化／拒识 → 性能剖析**。训练损失下降而错误类别仍最低时，优先查能量方向、MAP竞争项和标签索引，不直接提高阈值。无通用分类误差、收敛速度或现场误放行上界。[cite:1](#source-1)

### L4 数学推导：层级、原目标与梯度

**卷积及共享。**采用现代库常见的不翻核相关记号，对输出图 j 的连接集合 M(j)：

\[
u_{j,p,q}=b_j+\sum_{i\in M(j)}\sum_{r=0}^{4}\sum_{s=0}^{4}
W_{j,i,r,s}x_{i,p+r,q+s},\qquad h_{j,p,q}=f(u_{j,p,q}).
\]

这固定了实现坐标协议；若移植使用数学卷积的权重，需要翻核对齐，不能只看名字。valid 输出每轴为 `H−5+1`，故32→28→14→10→5→1。令初始感受野 r=1、采样步距 j=1，各层递推 `r'=r+(k−1)j`、`j'=sj`，得到 C1/S2/C3/S4/C5 的 `(r,j)` 为 `(5,1)/(6,2)/(14,2)/(16,4)/(32,4)`。F6在固定入口组合C5全部单元，覆盖32×32输入；这些是支持域代数，不保证每个像素实际梯度都非零。

若 \(\delta_{j,p,q}=\partial L/\partial u_{j,p,q}\)，则共享核梯度为所有使用位置及batch之和：

\[
\frac{\partial L}{\partial W_{j,i,r,s}}=
\sum_{n,p,q}\delta_{n,j,p,q}x_{n,i,p+r,q+s},\qquad
\frac{\partial L}{\partial b_j}=\sum_{n,p,q}\delta_{n,j,p,q}.
\]

不在 M(j) 的核不存在；若用密集参数乘固定掩码实现，掩码为0处梯度及有效权重须恒为0，并区分“分配的参数槽”和“自由参数”。结构依据原 Table I，反传为链式法则直接推导。[cite:1](#source-1)

**可训练下采样。**令对应2×2块的和为 \(s_{j,p,q}\)：

\[
u_{j,p,q}=a_js_{j,p,q}+b_j,\quad z_{j,p,q}=f(u_{j,p,q}),\quad
f'(u)=AS\,[1-\tanh^2(Su)].
\]

上游为 g，令 \(\delta=g f'(u)\)，则 \(\partial L/\partial a_j=\sum\delta s\)、\(\partial L/\partial b_j=\sum\delta\)，块内每个输入都得到 \(a_j\delta\)。求和跨空间及batch；mean实现则输入导数多1/4且增益需相应重参数化。代入 A=1.7159、S=2/3，原点斜率约1.14393，`f(1)≈0.9999973`；大幅输入仍会饱和，没有“缩放后梯度永不消失”的保证。[cite:1](#source-1)

**RBF 输出与原文两个目标。**令 F6 为 \(h\in\mathbb R^{84}\)，类别原型 \(c_k\)，标签 t：

\[
E_k(h)=\sum_{j=1}^{84}(h_j-c_{k,j})^2.
\]

作者稿式(7)称 RBF 输出 \(y_i\)，此处改记 E 防止与真实标签混淆。式(8)的 MLE/MSE 目标为 \(L_{\rm MSE}=E_t\)，训练集再取平均；不擅自添加1/2或除以84。它在固定方差高斯能量解释下省略常数；不等于10类softmax交叉熵。[cite:1](#source-1)

作者稿式(9)的判别 MAP 目标，保留固定“rubbish”背景代价 J：

\[
Z=e^{-J}+\sum_{k=0}^{9}e^{-E_k},\qquad
L_{\rm MAP}=E_t+\log Z=-\log\frac{e^{-E_t}}{Z}.
\]

于是 \(p_k=e^{-E_k}/Z\)，背景质量为 \(p_{\rm bg}=e^{-J}/Z\)，有

\[
\frac{\partial L_{\rm MAP}}{\partial E_k}=\mathbf1[k=t]-p_k,
\qquad
\frac{\partial L}{\partial h_j}=2\sum_k(\mathbf1[k=t]-p_k)(h_j-c_{k,j}).
\]

正确类导数非负，下降更新趋向降低其代价；错误类导数非正，趋向增加其代价，再经 h 的雅可比调节网络。若原型也学习，\(\partial L/\partial c_{k,j}=-2(\mathbf1[k=t]-p_k)(h_j-c_{k,j})\)；固定原型不更新。对 MSE 只有目标类项，若全部 h 与全部可训练 c 同为常量，损失可为0，给出忽略输入的塌缩反例。MAP引入竞争以反制这一机制，但不构成非凸训练全局收敛定理。[cite:1](#source-1)

数值上用 `E_t+logsumexp([-J, -E_0, …, -E_9])`；去掉背景项后确实能写成负距离上的交叉熵，但**这不允许把原模型改写成自由线性logits的现代softmax头**。J影响梯度和拒识；固定背景还使各类代价共同平移不再完全抵消。

**参数、连接数与复杂度口径。**60000是可训练参数，不是空间展开连接数。卷积乘加数为 \(H_oW_o k^2\sum_j|M(j)|\)，不含bias、激活及RBF。C1/C3/C5/F6分别为117600/150000/48000/10080，共 **325680 MAC/图**；S层额外需要每块3次加法、一次增益乘法及bias，RBF另需840次差值／平方及归约。不同FLOP口径不能直接混比，内存主要为batch激活及参数状态。

原作者稿 p.8 写 C3 “156000 connections”，但同页1516参数及10×10展开给151600（含bias）；按各层同口径并加固定RBF的840连接，合计340908，而 p.7 总数写345308，恰差4400。**保留该文内计数不一致，不把345308当已复算结论**；自由参数60000与 Table I 一致，不受这一展开连接数差异影响。[cite:1](#source-1)

参数通常不可唯一辨识，例如某些隐藏通道重排并补偿相邻权重可保持函数；下采样不可逆，标签不充分时也不能恢复真实字形。局部导数检查不证明学得正确分类器。

### L5 实现：伪代码、布局与复现边界

以下是**按原机制编写的控制流伪代码**，不是仓库已有模块。参数构造只执行一次；`C3Connections` 为 L1 完整表，`SparseValidCorrelation` 为每个允许图对一个5×5核、每个输出图一个bias。核共享发生在空间，不发生在不同图对。原始数字原型码表必须从可核验资源转录并另行校验；本次未提供或猜造该840标量文件。

```text
ScaledTanh(u):
    return 1.7159 * tanh((2 / 3) * u)

Subsample(x, a, b):
    require height(x) % 2 == 0 and width(x) % 2 == 0
    s = x[:,:,0::2,0::2] + x[:,:,0::2,1::2]
    s = s + x[:,:,1::2,0::2] + x[:,:,1::2,1::2]
    return ScaledTanh(a[None,:,None,None] * s + b[None,:,None,None])

LeNet5Energy(X, parameters, fixed_codes):
    require shape(X) == [N,1,32,32] and N > 0 and Finite(X)
    require shape(fixed_codes) == [10,84] and ValuesArePlusOrMinusOne(fixed_codes)
    h = ScaledTanh(ValidCorrelation(X, W1, b1))
    h = Subsample(h, a2, b2)
    h = ScaledTanh(SparseValidCorrelation(h, W3, b3, C3Connections))
    h = Subsample(h, a4, b4)
    h = ScaledTanh(ValidCorrelation(h, W5, b5))
    require shape(h) == [N,120,1,1]
    h = ScaledTanh(Linear(Reshape(h,[N,120]), W6, b6))
    E = SumOverLastAxis((h[:,None,:] - fixed_codes[None,:,:]) ** 2)
    require shape(E) == [N,10] and Finite(E)
    return E

TrainStep(X, target, J, optimizer):
    require IntegerLabelsInRange(target,0,10) and J > 0
    optimizer.ZeroGrad()
    E = LeNet5Energy(X, parameters, fixed_codes)
    logZ = LogSumExp(ConcatClassAxis(ConstantColumn(-J,N), -E))
    loss = Mean(E[SampleIndex,target] + logZ)
    Backward(loss)
    require AllGradientsFinite(parameters)
    optimizer.StepWithDeclaredProtocol()
```

`2/3`必须用浮点除法，避免某些语言整数除法变0；N=1时不能用无轴 `squeeze` 丢batch。推理返回 `Argmin(E,class_axis)`；固定码表不入优化器。训练目标为明确选择的 MAP，不宣称所有原文实验都用同一目标、同一J或同一原型学习策略。

本次没有取得可确认与1998实验一致的 LeNet 原训练源码、完整权重、数字码表和优化器配置。因此不拿教程中“两卷积＋最大池化＋ReLU＋softmax”冒充源码对应。L5依据是原文公式／Table I与上述参考流程；完整逐层前向、共享核梯度和训练仍须实现后验证。

## 子方法 B：Graph Transformer Networks

### L1 步骤：图模块、组合与训练推理

**图契约。**基础实例为有限有向无环多重图 `G=(V,A,start,accept)`。节点表示候选切分边界或组合状态；弧有独立ID、端点、数据载荷及标量代价，可含字符输入／输出标签。相同端点、相同字符仍可能是两条不同弧，不能随意去重。完整路径代表一种切分和标签解释，路径代价为沿途弧代价之和，越小越好。原 GTN 的载荷可为图像、向量、符号，并不只限今天的标量 WFST。[cite:1](#source-1)

| 模块 | 输入→输出及具体变换 | 训练依赖／反传内容 |
|---|---|---|
| 候选切分／segmenter | 字符串图像→切分DAG；候选切口为节点，合法片段为携图像的弧 | 可有固定启发式或可训练片段代价；硬阈值选切口不自动获得导数 |
| Recognition Transformer | 每条图像弧运行共享字符识别器→按类别展开为并行标签弧，附类别代价及片段代价 | 保存每次识别状态、图像弧与类别输出对应；所有调用共享同一网络参数 |
| Composition Transformer | 识别图＋词典／语法变换图→保留标签匹配的乘积状态图，组合弧代价相加 | 记录输出弧来自哪两条输入弧；反向将贡献累加回两图 |
| Path Selector | 解释图＋真实字符串→只接受该串的约束图 G_y，但保留该串全部切分路径 | 可用目标串线性接受器与解释图组合；复制弧的梯度需散加回原弧 |
| Viterbi Transformer＋Path Scorer | 图→最小代价路径→路径代价 | 反传到被选弧，非选弧0；平局需固定规则，只是分段导数 |
| Forward Scorer | 图→全部完整路径的负log-sum-exp代价 | 反传按路径后验占用率分配，不只选一条路径 |
| Global Criterion | 约束图与全解释图的评分→单个整串损失 | 两路梯度在共享解释图累加，再经过组合、识别回到参数 |

这是作者稿 Fig.16–21、24、27–29 的角色细化。语言约束可以不接，但目标 Path Selector 只在训练用；推理时不得泄漏真实字符串。原文没有声称所有段的几何坐标和启发式切口都可被同一梯度自动优化。[cite:1](#source-1)

**具体输入走一遍。**原文 Fig.20 用标签串“34”，候选切分有“3／4”及错误拆出的细竖笔画，识别图同时容纳“34”和“341”等解释。训练不需要人工指定哪条弧是真正的字符边界：Path Selector保留所有输出为“34”的路径；最佳正确路径的代价与全图最佳路径比较，或用Forward Scorer对两边分别聚合所有路径。若错误的“341”更优，损失把正确解释相关代价下压，把错误解释相关代价上推；同时出现在两条最佳路径的弧，其 Viterbi 判别梯度抵消。[cite:1](#source-1)

**推理顺序。**图像→候选DAG→识别并行标签图→可选语法组合→Viterbi回溯→原始串、路径和代价→独立拒识／业务核对。推理Viterbi选择最佳**路径**；若目标是总概率最大的**字符串**，必须聚合同串所有路径再比较，二者不保证相同。保存未被词典修正的候选证据，不能将“语法合法”等同“图像读对”。

**训练顺序与停止。**按样本构图并评分，检查全图与目标图均有完整路径；计算本卡选定的判别Forward损失，反传，汇总共享参数梯度后更新。按预算或验证停止；目标图为空、图循环不符合算法约束、分数非有限、资源超限都应显式报告原因，不将 `∞−∞` 置零继续训练。候选构造若漏掉真实串，必须先修候选／标签／语法，而非宣称端到端学习能从空图恢复目标。

**SDNN 并行入口。**原文还讨论空间滑动字符网络产生沿位置的输出图，再与字符模型变换器组合，用多位置证据和转换规则解释重复响应／非字符区域；这可替换HOS候选生成，但仍需要字符模型及全局目标，不等于“把滑窗argmax直接拼起来”。不能倒称其为后来CTC的具体实现。[cite:1](#source-1)

### L2 机制：全局优化为何不等于局部分类

| 机制 | 必要性及不变量 | 删除或替换后的后果 |
|---|---|---|
| 保留多条切分路径 | 不在识别前过早决定边界；HOS完整路径按次序覆盖输入墨迹 | 若先硬切错，正确串可能根本不在候选空间；再好的字符器也无法选中 |
| 共享识别参数 | 每段相同映射，各段梯度统一更新 | 各段独立权重改变模型且无法自然处理新图；不能每次调用都初始化 |
| 语法／目标图组合 | 将字符级候选约束为合法串，同时保留来源关系 | 词典缺真实编号会系统性错读；训练目标图不应仅留一个人工选中切分 |
| 判别全局目标 | 正确路径集合与所有竞争路径集合比较，处理相互关联的模块错误 | 只降正确代价存在塌缩，局部分类准确率高也不保证整串最优 |
| 全路径聚合 | 同串不同切分的证据可累加，反传无需固定单一对齐 | 只用最优路径会忽略其他解释；但也不能重复枚举同一推导来凭空增加概率质量 |
| 弧来源追踪 | 一个输入弧可复制成多个输出弧，梯度必须按来源求和 | 按“端点＋标签”合并或最后写入覆盖，会丢梯度／混淆不同推导 |

从机制上应区分**符号结构**与**数值状态**：图构造决定当前计算依赖；固定构造内的代价由连续函数产生，可按链式法则反传。离散建弧条件、阈值剪枝和拓扑切换不因此变成处处光滑。原文广义图变换可用 `check/fprop/bprop` 表述，不意味着任何图操作、任意外部识别器都有正确梯度。[cite:1](#source-1)

### L3 工程权衡：候选预算、代价尺度与失效

| 参数／条件 | 尺度与依据 | 证据与调参顺序 |
|---|---|---|
| 切口数量、最小／最大片段宽度 | 切口个数、图像像素；依真实字高和连笔程度，无通用原论文阈值 | 首先测真实串候选可达率，再查识别；过少漏真解，过多增加片段网络调用及图弧数 |
| 每片段保留类别数 | 类别个数；无剪枝基线保留全部K类 | 先与全类别结果比较；top-k砍掉真类会令目标图为空，损失不能修复 |
| 字符、切分、语法代价比例 | 必须在统一可加代价尺度；RBF平方距离不是已校准log概率 | 查长度偏好和各来源代价分布，再调校准／比例；不能混用越大越好分数与越小越好代价 |
| 每字符／弧的常量偏置 | 代价／弧；影响路径长度，不是无害常数 | 短串偏好先查累加项；不同长度路径加同一弧偏置并不共同平移 |
| beam宽度、图节点／弧预算 | 代价差、节点／弧个数、内存字节 | 建小图精确基线后逐步剪枝；剪枝改变路径配分与梯度，不宣称与全图目标等价 |
| 词典、字符模型和空标签 | 离散协议；需记录字符编码、epsilon及起止状态 | 单列罕见串、随机编号、重复字符和空串；epsilon不是可显示的空格，也不是CTC blank |
| 训练粒度及优化器 | 每串或batch；共享网络参数不随图大小复制 | 正确／错误路径竞争及梯度范数先查，再调学习率；按弧数平均会改变原整串目标权重 |

典型症状：字符准确而整串差，先查候选可达率／图覆盖；loss恒0先查是否错误地把全图也约束成目标图，以及是否仅用零间隔判别Viterbi；loss为Inf先查空目标图；多一条无关epsilon路就改变结果，先查重复推导计数；梯度随副本数异常，查scatter-add及损失归一化。训练和部署语法版本不同必须单独记录。上述均为诊断设计，没有真实OCR评测或通用速度／精度保证。

### L4 数学推导：图目标、动态规划与梯度

**路径建模。**令解释图的完整路径集合为 \(\Pi\)，输出等于目标串y的集合为 \(\Pi_y\subseteq\Pi\)。弧 a 的代价 \(c_a(\theta)\) 可由片段代价、字符RBF代价及语法代价相加构成；\(n_a(\pi)\) 是该路径使用该弧的次数，在DAG中为0或1：

\[
C_\pi=\sum_a n_a(\pi)c_a,\quad
V(G)=\min_{\pi\in\Pi} C_\pi,\quad
Z(G)=\sum_{\pi\in\Pi}e^{-C_\pi},\quad F(G)=-\log Z(G).
\]

这里不要求每个局部模块先输出归一化概率；有限图上的 Z 给出全局归一化。`F`是代价形式的soft-min，可能为负，即使每条弧代价非负；它不是平均路径代价。[cite:1](#source-1)

| 原文目标 | 单样本表达 | 梯度与局限 |
|---|---|---|
| Viterbi training，式(11) | \(L_v=V(G_y)\) | 只降最佳正确路径；缺竞争，有塌缩风险 |
| Discriminative Viterbi，式(12) | \(L_{dv}=V(G_y)-V(G)\) | 正确最佳路径计数减全图最佳路径计数；最佳正确时可为0，没有正间隔保证 |
| Forward training，§VII-C | \(L_f=F(G_y)\) | 聚合全部正确路径，但仍无竞争归一化 |
| Discriminative Forward，式(17) | \(L_{df}=F(G_y)-F(G)\) | 正确集合与全部集合比较，为本卡主训练目标 |

式号均属作者稿。原文判别Viterbi不是默认带margin的hinge，不擅自加一个m再称原式；\(G_y\subseteq G\) 给两个判别目标非负。对判别Forward：

\[
L_{df}=-\log Z(G_y)+\log Z(G)
=-\log P(y\mid x),\qquad P(y\mid x)=\frac{Z(G_y)}{Z(G)}.
\]

这是对所有输出为y的隐切分求和，不是强制选中一条边界对齐。候选空间定义了概率事件；错误构图、重复计数或剪枝均改变该目标，不得把有限候选后验解释成无条件真实正确概率。[cite:1](#source-1)

**Viterbi与Forward递推。**设单起点s、终点t（多起止可加零代价epsilon虚节点），按拓扑序：

\[
v(s)=0,\quad v(v)=\min_{a:u\to v}\{v(u)+c_a\};
\qquad f(s)=0,\quad
f(v)=-\log\sum_{a:u\to v}\exp[-f(u)-c_a].
\]

无到达路径的值为+∞。Viterbi保存最优入弧索引并从t回溯；Forward最终为F(G)。稳定soft-min对有限候选 q 取 m=min(q)，算 `m−log(sum(exp(−(q−m))))`；若没有有限候选，直接返回+∞，不能执行 `∞−∞`。原文式(10)、(13)、(14)分别给最小递推及此负log-sum-exp形式。[cite:1](#source-1)

**梯度由配分求导，而非口头说“能端到端”。**固定路径集合，记 \(p_G(\pi)=e^{-C_\pi}/Z(G)\)，则

\[
\frac{\partial F(G)}{\partial c_a}
=-\frac1Z\sum_\pi e^{-C_\pi}(-n_a(\pi))
=\mathbb E_{p_G}[n_a(\pi)]\equiv\gamma_G(a).
\]

因此

\[
\boxed{\frac{\partial L_{df}}{\partial c_a}=\gamma_{G_y}(a)-\gamma_G(a)},\qquad
\nabla_\theta L_{df}=\sum_a[\gamma_{G_y}(a)-\gamma_G(a)]\nabla_\theta c_a.
\]

目标图中不存在的弧其目标占用率为0。正确串特有弧倾向下压，错误串特有弧倾向上推；共享弧的作用取决于两种后验占用率，不能统一设±1。若多个图弧对应同一次字符网络输出，先把弧梯度相加，再经 RBF 及 LeNet-5 反传；若多片段调用共享θ，再累加各次网络梯度。这是作者稿式(15)/(16)的前向评分反传及 p.22 参数共享说明的现代记号推导。[cite:1](#source-1)

可用log域前后向计算占用率：\(\alpha(s)=0\)，\(\alpha(v)=\operatorname{LSE}_{a:u\to v}(\alpha(u)-c_a)\)；\(\beta(t)=0\)，\(\beta(u)=\operatorname{LSE}_{a:u\to v}(-c_a+\beta(v))\)。对有效弧有

\[
\gamma_G(a)=\exp\{\alpha(u)-c_a+\beta(v)-\log Z(G)\}.
\]

分别对G和G_y计算并按弧来源散加；无效弧占用率0。Viterbi是把soft-min换min，唯一最优路径局部有 \(\partial V/\partial c_a=n_a(\pi^*)\)；平局处不可微，约定一个分支不等于所有方向导数都成立。原论文Fig.20共享路径抵消即上述计数差的例子。

**图组合及广义变换。**标准无epsilon情况，图A的弧a输出标签等于图B弧b输入标签时，生成乘积状态弧

\[
(u_A,u_B)\to(v_A,v_B),\quad c'_e=c^A_a+c^B_b.
\]

输出继承A的输入标签和B的输出标签；仅接受两图都可完成的轨迹。若输出弧上游梯度为 \(\delta'_e\)，则

\[
\frac{\partial L}{\partial c^A_a}=\sum_{e:\operatorname{srcA}(e)=a}\delta'_e,
\qquad
\frac{\partial L}{\partial c^B_b}=\sum_{e:\operatorname{srcB}(e)=b}\delta'_e.
\]

epsilon转移可只推进一侧，需要防止等价epsilon推进顺序产生重复路径；多条真正不同推导则应保留。原作者稿 p.31–32 把广义组合分为 `check(arc1,arc2)` 决定是否建边，`fprop(...)` 生成一条／多条弧甚至子图的数值，`bprop(...)` 把输出梯度传回载荷及内部参数。若 \(z'=\phi(z_A,z_B;\theta)\)，反向分别为 \(J_{\phi,z_A}^Tg'\)、\(J_{\phi,z_B}^Tg'\)、\(J_{\phi,\theta}^Tg'\)。它推广的不只是“代价相加”，还包括图像到类别弧的识别变换。[cite:1](#source-1)

**成立条件、复杂度和退化。**在有限DAG且代价有限、目标可达、弧数值映射可微时，Viterbi／前后向为 O(|V|+|A|)，额外存储同量级；不枚举指数数量的完整路径。组合最坏有 \(|V_A||V_B|\) 个状态，朴素弧匹配可达 \(|A_A||A_B|\)，实际按可达状态和标签索引减少；R段K类识别展开约RK条弧，网络开销另加。固定后验梯度不是全局训练最优保证。

有环图不适用上述拓扑递推；最短路可能有负环，全路径指数和也可能发散，需要另设收敛条件／算法，不能只重复迭代到“看起来不变”。若所有路径共享常量代价平移，判别Forward可不变；同一中间状态的势函数重参数化也能保持完整路径排序，说明单弧代价一般不可辨识。给每条弧加常数则改变不同长度路径的偏好。硬剪枝后的梯度只针对剩余图，未提供未剪枝目标的无偏保证。

### L5 实现：固定源码、伪代码与数值边界

**外部实现对应。**核对官方 `facebookresearch/gtn` 固定提交 **`6b8f731e99fba67585bcf7175bd8b43c52c598b1`，2021-06-03 UTC**。该仓库README已标弃用并指向 `gtn-org/gtn`；本卡只声明所读快照。它是现代加权自动机自动微分实现，**不是1998年原训练程序，也不是LeNet-5权重来源**。[cite:5](#source-5)

| 文件／符号 | 已静态核对内容 | 使用边界 |
|---|---|---|
| `gtn/functions.cpp`：`forwardScore/viterbiScore/viterbiPath` | 分别转到 `shortestDistance`、tropical分支及`shortestPath` | 名字有shortest，但源码是越大越好的log-score约定 |
| `gtn/functions/shortest.cpp`：`shortestDistance` | 拓扑入度队列；log-sum-exp累加或max；记录节点分数及最大项缓存 | 部分环／断连结构会触发异常，但只检查accept节点剩余入度，不是全图无环校验；调用前独立验证DAG，不能当通用循环图评分器 |
| 同文件：`shortestDistanceGrad` | log-sum-exp导数分配到入弧并回传节点；tropical仅向选定索引传递 | `float`实现，数值结果不等于本卡FP64公式检验 |
| `gtn/functions/compose.cpp` | 匹配第一图olabel与第二图ilabel、可达性筛选、epsilon处理、弧权重相加，`gradInfo`保留来源 | 源码与文献支持该机制；没有运行完整组合／epsilon回归，不能宣称所有边界已验证 |
| `gtn/functions.cpp`：`subtract`／`remove` | 相减两路反传分别为正／负；所读`remove`梯度显式未实现 | 不得把任意图操作塞入训练后声称自动可微；不依赖`remove`实现本卡目标 |

`compose.cpp`网页正文在末尾截断；补取固定文件API提供的完整base64载荷核对末尾 `gradInfo` 的累加与 `addGrad`。此为静态核对，不是编译运行结果。[cite:6](#source-6)[cite:7](#source-7)

环检测反例：起点0、终点1，弧0→1，另有不通向终点的2→2自环；按固定源码分支，终点剩余入度为0，因此不会由该异常检查报告孤立自环。不能将“评分调用未抛异常”作为全图DAG证书。本卡参考路径算法的显式无环检查仍需保留；此结论来自源码分支核对，非编译运行结果。

**符号转换必须一致。**库弧权重 w=−c，`forward_score(G)=log Z(G)=−F(G)`，`viterbi_score(G)=max sum(w)=−V(G)`；故在库约定下损失为 `forward_score(G_all)−forward_score(G_target)`，不能照抄代价公式的书写顺序。库权重梯度为 `gamma_all−gamma_target`，再经 w=−c 传回代价时再乘−1。[cite:6](#source-6)

以下为**代价约定**的实现伪代码，图操作需维护节点／弧ID和完整反向映射；无库安装或调用完成的含义。`ComposeExact` 包括可达性、起止状态和epsilon去重协议；不能以不处理epsilon的两层循环代替其完整语义。

```text
BuildInterpretation(image, shared_recognizer, grammar):
    segments = BuildSegmentationDAG(image)
    RequireAcyclicAndValidCoverage(segments)
    G = EmptyGraphWithSameNodes(segments)
    for segment_arc in segments.arcs:
        crop = NormalizeCharacterWithSavedMapping(segment_arc.image)
        energies, cache = shared_recognizer.Forward(crop)
        for k in CharacterSet:
            a = G.AddArc(segment_arc.src, segment_arc.dst, label=k,
                         cost=segment_arc.cost + energies[k])
            SaveRecognitionOrigin(a, segment_arc, cache, k)
    if grammar exists:
        G = ComposeExact(G, grammar, save_arc_origins=True)
    return G

ForwardCostDAG(G):
    RequireFiniteCostsAndAcyclic(G)
    f = FillNodes(+infinity)
    f[G.start] = 0
    for v in TopologicalOrder(G):
        candidates = [f[src(a)] + cost(a) for a in Incoming(v)]
        if v == G.start:
            candidates.Append(0)
        f[v] = StableSoftMinOrInfinity(candidates)
    return f[G.accept], f

GlobalTrainStep(image, target, grammar, optimizer):
    optimizer.ZeroGrad()
    G_all = BuildInterpretation(image, shared_recognizer, grammar)
    G_target = SelectAllPathsSpelling(G_all, target, save_arc_origins=True)
    require HasCompletePath(G_all) and HasCompletePath(G_target)
    F_all, state_all = ForwardCostDAG(G_all)
    F_target, state_target = ForwardCostDAG(G_target)
    loss = F_target - F_all
    delta_all = BackwardForwardCost(G_all, state_all, seed=-1)
    delta_target = BackwardForwardCost(G_target, state_target, seed=+1)
    ScatterAddByOrigin(delta_target, delta_all)
    BackwardCompositionThenSharedRecognition(delta_all)
    require Finite(loss) and AllGradientsFinite(parameters)
    optimizer.StepWithDeclaredProtocol()
```

`BackwardForwardCost`实现L4的前后向占用率或等价逆拓扑链式法则。梯度缓存必须属于本次样本／图版本；两个损失分支共享同一识别图时先累加再更新，不能一支反传后立即更新参数。片段批量前向可以加速，但必须保留 `(image_id, segment_arc_id, class_id)` 映射；从浮点张量提取普通标量再构图会切断自动微分，需要明确的桥接backward，不能只调用图端 `backward` 就认为 CNN 已得到梯度。

优化先以小图无剪枝结果为基准，再评估按标签排序、缓存共享片段、批量识别及惰性组合。前两项在固定图与算子协议下可保持目标，beam/top-k会改变路径集；混合精度需核对log-sum-exp、近似平局和梯度。去掉重复epsilon推导与删掉真实不同路径不是一回事，须以手工可枚举图验证。

## 验证矩阵与可复查结果

以下区分来源事实、确定性结构推导、局部数学检查与真实识别实验。局部数值输入仅为**公式单元检查**，不生成识别数据集、不冒充论文复现性能。检查使用仓库已有 `.venv` Python及NumPy，不安装训练依赖；PDF渲染依赖只放系统临时目录。

| 检查项 | 断言／方法 | 本次状态及边界 |
|---|---|---|
| C3及总参数 | 全连接表60图对；逐层156/12/1516/32/48120/10164，合计60000 | 已执行算术断言；密集C3多900参数 |
| shape／MAC／感受野 | 按valid卷积、2×2 stride2推导；conv/FC为325680 MAC，C5感受野32 | 文内代数；需后续真实模型逐层shape测试 |
| 缩放tanh | `1.7159*tanh(2/3)`应约0.9999972559 | 已执行；不能写成浮点精确1 |
| 下采样局部梯度 | FP64，步长1e−6，2×2块 `[0.1,0.2,−0.3,0.4]`，a/b为0.2/0.3，中心差分对a/b | 已执行，最大绝对差约3.36e−11；非全网络梯度测试 |
| RBF MAP局部梯度 | FP64，h=`[0.1,−0.2,0.3]`，两个互反±1原型，J=3，目标0；中心差分对h | 已执行，最大绝对差约1.28e−10；为低维同公式检查，非原84维码表验证 |
| GTN目标局部梯度 | 三条互不共享弧的双弧路径，弧代价`[0.2,0.3,−0.1,0.4,0.5,0.7]`，前两条为目标集合；枚举配分对六弧中心差分 | 已执行，占用率差与差分最大绝对差约2.93e−11；尚未验证DAG实现、共享弧或组合 |
| C3稀疏算子及共享梯度 | 对允许图对逐一扰动，禁用图对无作用；空间梯度应求和 | 设计，未运行 |
| 原LeNet前向一致性 | 码表、预处理、权重全部固定，逐层比较输出与梯度 | 未取得原完整资产，未运行 |
| 图评分及组合端到端 | 小DAG枚举对照Viterbi/Forward，含共享弧、平局、重复字符、epsilon、空图、无合法路径、有环拒绝 | 设计，未运行外部GTN或完整本地实现 |
| 组合梯度 | 输出弧复制时输入梯度求和；两图参数都获正确梯度，来源ID不可覆盖 | 原文／源码静态核对；未运行完整数值测试 |
| OCR任务验收 | 字符误差、CER、整串正确率、拒识率、候选可达率、延迟／显存分开报告 | 无训练、权重推理、数据集复现或现场实测 |

正式回归应包含 N=1、多batch、边界笔画、极性错误、相似字符、错误切分、词典外随机串以及目标不可达；训练／验证／测试按书写者或生产批次隔离。CER以参考字符数为分母，整串正确率及拒识另计，不用单字符准确率替代系统评价。[OCR 验收约定](../ocr_codes_evolution.md#52-验收设计与追溯输出)

## 面试自测：五问贯穿两子方法 L1–L5

| 层级问题 | 答案要点与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：给单字图及标签“34”的整串图，两者分别如何前向和训练？ | 单字按32→28→14→10→5→1→84→10代价；整串建候选图、共享识别、可选组合，目标路径选择只在训练，判别目标比较约束图和全图 | 把字串直接送固定flatten；推理使用目标Path Selector；只输出逐块argmax拼串 | A-L1／B-L1，原Fig.2、17–21：[cite:1](#source-1) |
| L2：AvgPool＋ReLU＋softmax为何不是原LeNet？GTN只加词典够不够？ | 原S层有a/b及tanh，C3部分连接、84维RBF码和能量竞争；词典只限制路径，GTN全局训练还需评分目标及可追踪反传 | sum改mean不改a；词典合法即正确；只降正确路径导致塌缩 | A-L2／B-L2，原式(7)–(9)、§VII：[cite:1](#source-1)[cite:4](#source-4) |
| L3：网络不学习、整串频繁漏字，先改什么？ | 单字先核输入极性／量程、C3、激活饱和、原型与损失符号；整串先查目标可达率、片段跨度、代价长度偏好和剪枝；有依据再调步长／预算 | 所有错误都靠提高网络容量；空目标图loss置0；盲加大词典权重 | 两子方法L3、验证矩阵；[cite:1](#source-1) |
| L4：推导RBF MAP与GTN判别Forward梯度，并说明何时无效？ | RBF对能量为`1[k=t]−p_k`再乘距离导数；GTN对弧代价为目标占用率减全图占用率，再链式到共享CNN；要求目标非空、配分有限、数值映射可微 | 把符号写反；只用最佳路径冒充求和；有环图无条件套拓扑递推；局部梯度正确即全局收敛 | A-L4／B-L4，原式(9)、(13)–(17)：[cite:1](#source-1) |
| L5：原权重数量吻合、现代GTN的backward能运行，就复现了吗？ | 还需核C3掩码、sum系数、tanh缩放、码表／预处理、原优化器；图端核w=−c、epsilon语义、来源scatter-add及CNN桥接，逐层／枚举／有限差分后再真实OCR评测 | 60000参数即原模型；`forward_score(target)−forward_score(all)`照搬代价公式；`.item()`后自动接通CNN；把2021库当1998源码 | 两子方法L5、来源与未完成项：[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7) |

此为知识库自测设计，答5不代表训练工程或真实面试／现场验收已经完成。

## 来源、层级与获取深度

所有访问均在 **2026-11-05**。历史机制以原作者稿为 **P0／[SINGLE-SOURCE]**；正式版书目、同作者后续章节和多个镜像不算独立实验。现代官方GTN源码是其自身实现的P0，可交叉检查图组合及自动微分机制，但不能证明1998所有程序细节。没有三份独立实验证据支持通用性能结论，本卡不作准确率、速度或优劣排名。

- <a id="source-1"></a>**[cite:1] LeCun、Bottou、Bengio、Haffner，Gradient-Based Learning Applied to Document Recognition，1998，作者投稿稿。** [作者PDF](https://leon.bottou.org/publications/pdf/ieee-1998.pdf)。取得完整46页PDF；网页解析仅到早期学习章节，本地pypdf/pdftotext遇Type3字体乱码，不能冒称文本提取成功。随后渲染并直接核对p.7–9（Table I、C1–F6、RBF、式5–9）、p.18–25（GTN、HOS、共享识别、Viterbi／Forward目标及式10–17）、p.28–32（SDNN、组合、check/fprop/bprop）、p.40–41（附录A激活／初始化、随机更新及对角预条件说明）。未逐页核验全部实验及附录曲率算法；引用这些PDF页码，不混作期刊页码。Table I与连接数矛盾保留在A-L4。
- <a id="source-2"></a>**[cite:2] 作者书目记录。** [Bottou论文页](https://bottou.org/papers/lecun-98h)。已读书目及全文入口，核对作者、1998、86(11)、2278–2324；书目不是第二份实验，也不以网站修改日期充当算法年份。
- <a id="source-3"></a>**[cite:3] 正式版的HAL存档。** [HAL PDF入口](https://hal.science/hal-03926082/file/lecun1998.pdf)、[带v1入口](https://hal.science/hal-03926082v1/file/lecun1998.pdf)。不带v1网页解析取得存档封面、摘要、引言及章节导航，核对DOI和正式版章节编号；后段截断。v1入口返回反自动化验证，本地获取也未取得可解析PDF，不绕过验证。HAL封面2023-01-06是存档日期，不是1998论文的新版本发表日期；正式版后段公式／表格未逐项与投稿稿比对。
- <a id="source-4"></a>**[cite:4] 同作者体系，Object Recognition with Gradient-Based Learning，1999。** [作者PDF](https://leon.bottou.org/publications/pdf/features-1999.pdf)。网页正文读取到§2.1卷积与下采样描述，后续截断；用于交叉辨析“average＋可训练系数”和1998“sum＋可训练系数”，不作为独立复现或覆盖GTN原式的替代证据。
- <a id="source-5"></a>**[cite:5] Facebook Research官方GTN仓库及固定版本记录。** [README](https://github.com/facebookresearch/gtn)、[固定提交元数据](https://api.github.com/repos/facebookresearch/gtn/commits/6b8f731e99fba67585bcf7175bd8b43c52c598b1)。已读README、查询main提交／目录树及固定提交时间，锁定`6b8f731e99fba67585bcf7175bd8b43c52c598b1`、2021-06-03T05:12:02Z。README说明C++及Python绑定、加权自动机自动微分、弃用和迁移，不是1998 LeNet/GTN资产发布说明。
- <a id="source-6"></a>**[cite:6] 同提交评分与算子源码。** [gtn/functions.cpp](https://raw.githubusercontent.com/facebookresearch/gtn/6b8f731e99fba67585bcf7175bd8b43c52c598b1/gtn/functions.cpp)、[gtn/functions/shortest.cpp](https://raw.githubusercontent.com/facebookresearch/gtn/6b8f731e99fba67585bcf7175bd8b43c52c598b1/gtn/functions/shortest.cpp)。网页取得两份完整正文；定向核对score包装、符号、log-sum-exp／max、拓扑队列、反向及`remove`的未实现梯度。未编译、未安装gtn、未运行官方单测。
- <a id="source-7"></a>**[cite:7] 同提交图组合源码。** [compose.cpp](https://raw.githubusercontent.com/facebookresearch/gtn/6b8f731e99fba67585bcf7175bd8b43c52c598b1/gtn/functions/compose.cpp)、[固定文件API](https://api.github.com/repos/facebookresearch/gtn/contents/gtn/functions/compose.cpp?ref=6b8f731e99fba67585bcf7175bd8b43c52c598b1)。读到可达性、标签匹配、乘积状态、epsilon推进、`gradInfo`，API补齐文件末尾，核对按来源`+=`及`addGrad`。文件blob为`d88593e97442746c216b85c5ff8ef9a09bf412ce`；属于静态实现证据，无完整epsilon正确性或跨框架数值验证结论。

## G26 承接结论与未完成项

**文档承接结论：**LG01已分别覆盖LeNet-5和GTN两子方法的L1–L5正文，不再只是C05卷积或GTN提名。原C3部分连接、可训练下采样、缩放tanh、RBF与原损失，以及GTN图模块、组合、正确／全路径全局目标和梯度均有实际读取的原始机制证据；可建议把 **G26／08-01的“缺卡”改为“已有卡，工程未完成”**，承接键为`lenet_gtn.md:LG01`，仍计一张卡。每个子方法建议独立登记 **核／核／部／核（限定推导）／部**，共享面试答5。

**不可据此整项关闭：**原1998训练源码、可直接加载的原型码表和完整权重／训练配置未取得；正式版后段与投稿稿未逐式比对；LeNet完整前向／反向、GTN组合与CNN梯度桥接、官方库运行、真实数据训练、整串评测和现场拒识均未完成。原文展开连接数存在已披露矛盾，不宣称消解。若后续复核否定任一子方法的原机制证据，必须保留G26复合节点缺口，不得仅凭LeNet部分或现代GTN库替整包销项。

本次仓库写入仅为本文件；不修改OCR总览、索引、覆盖矩阵、历史ID或卡片总数，不新增模型源码、权重、训练器或测试文件。原有总览／台账的“缺卡”文字保持原样，等待另行统一登记；本卡结论不是已实施的台账变更。
