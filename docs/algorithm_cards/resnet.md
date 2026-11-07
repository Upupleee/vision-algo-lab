# RN01 ResNet：原版残差分类网络、监督训练与梯度边界

[学习规范](../algorithm_learning_standard.md) · [分类装配 E04](../classification_assembly_evolution.md#34-e04resnet-扩大学习表示能力但闭集任务边界仍在) · [覆盖台账 G22](../evolution_coverage.md) · [跳连组件 C13](learning_components.md#c13-跳跃连接相加与拼接)

<a id="rn01"></a>
<a id="g22"></a>

## 任务、演进与覆盖状态

**任务与契约：**有标签图像的闭集分类。训练输入为图像与类别标签，推理输入为同预处理协议图像；输出 K 维 logits、可选 softmax 分数及类别。ImageNet 原实验 K=1000；整图分类不直接输出缺陷坐标、装配关系或未知类拒识。适用前提是类别标签可信、图像包含可判别信息、训练与部署分布具有可迁移性；预训练骨干也继承预训练数据假设。[cite:1](#source-1)

**演进定位：**更深 plain CNN 出现更高训练误差 → 将目标映射改写为相对输入的残差 → 恒等旁路与主支相加改善优化路径 → 增深仍增加卷积、激活显存与监督数据成本，投影另增参数 → 不消除过拟合、域偏移、采样信息损失或闭集限制。论文把退化与简单过拟合区分，并报告带 BN 的 plain 网络仍有健康梯度范数而出现退化，不能把它缩写为“仅解决梯度消失”。这是论文实验范围内的证据，不是深度越大必然更好的定理。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-05**。本文件只有 `resnet.md:RN01` 一张模型卡，承接 **G22／07-04／E04**；18／34／50／101／152 是同卡具体架构，不另计卡。主对象是 **2015-12-10 arXiv v1／CVPR 2016 原版 post-activation ResNet**；后文简称“v1”是架构俗称，勿与论文修订号或权重枚举混淆。

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：五种层级、两类块、A/B/C、训练与推理契约，限文献与静态结构 |
| L2 机制 | 已核验：退化、残差、投影、BN/ReLU 与监督作用，保留反例 |
| L3 工程权衡 | 部分：原训练配置、选型及诊断设计，无真实数据定参 |
| L4 数学推导 | 已核验：限定为链式法则、交叉熵、BN 局部导数、shape 与参数/MAC 代数；无全局收敛或泛化保证 |
| L5 实现 | 部分：固定外部源码静态对应、伪代码与测试设计；本仓库未实现 ResNet |
| 面试自测 | 答 5：贯穿 L1–L5，含答案、条件、易错点、证据 |
| 验证范围 | 本次完成文档检查、源码结构抽取和参数/层数算术检查；未执行网络前向、反向或训练 |

本地 [dl 目录](../../dl/README.md)中的 EdgeNet 相加跳连和 FocusNet 分类/回归组件不构成 ResNet；已检索该目录，未发现 ResNet 或 BatchNorm 定义。C13 是通用融合组件，其 `I+JF` 只描述相加映射；本卡补上原网络相加后的 ReLU、完整分层与训练，不修改或自动升级 C13。

## L1 步骤：具体架构与训练推理

### 输入、stem、stage 与分类头

布局使用 NCHW，N 为 batch，空间尺寸以像素计，卷积核布局为 `[Cout,Cin,Kh,Kw]`，groups=1、dilation=1。先以 **N×3×224×224** 走通固定分类入口；数值类型、RGB/BGR、量程、均值和插值方法必须与选定权重绑定，不把任意归一化称为原协议。

| 部分 | 共同操作 | 输出空间 | Basic 输出通道 | Bottleneck 输出通道 |
|---|---|---|---|---|
| conv1 / stem | 7×7 conv，64，stride=2，pad=3 → BN → ReLU | 112×112 | 64 | 64 |
| pool1 | 3×3 max pool，stride=2 | 56×56 | 64 | 64 |
| conv2_x | 基宽 b=64，首块 stride=1 | 56×56 | 64 | 256 |
| conv3_x | b=128，首块 stride=2 | 28×28 | 128 | 512 |
| conv4_x | b=256，首块 stride=2 | 14×14 | 256 | 1024 |
| conv5_x | b=512，首块 stride=2 | 7×7 | 512 | 2048 |
| head | 全局平均 → flatten 保留 N → FC(K) | 1×1 → N×K | FC 输入512 | FC 输入2048 |

结构来自原文 Table 1、§3.3；作者部署图在 7×7 特征上用固定 7×7 average pool，不能据此声称任意尺寸都已支持。**pool1 的具体坐标约定另见 L5**：作者 Caffe 是无 padding 的 ceil 池化，torchvision 是 pad=1 的 floor 池化；224 输入输出 shape 一样不代表值一样。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)

| 模型 | 残差主支 | conv2_x / conv3_x / conv4_x / conv5_x 块数 | 块数合计 | 命名层数计算 |
|---|---|---|---|---|
| ResNet-18 | Basic：3×3,b → 3×3,b | 2 / 2 / 2 / 2 | 8 | 1+2×8+1=18 |
| ResNet-34 | Basic：3×3,b → 3×3,b | 3 / 4 / 6 / 3 | 16 | 1+2×16+1=34 |
| ResNet-50 | Bottleneck：1×1,b → 3×3,b → 1×1,4b | 3 / 4 / 6 / 3 | 16 | 1+3×16+1=50 |
| ResNet-101 | 同上 | 3 / 4 / 23 / 3 | 33 | 1+3×33+1=101 |
| ResNet-152 | 同上 | 3 / 8 / 36 / 3 | 50 | 1+3×50+1=152 |

命名深度计 stem conv、残差主支 conv 和最后 FC；**不计 BN、ReLU、pool、相加，也不把 shortcut projection 另加到名称中**。表内数字为原文 Table 1；18/34 不要误写成 bottleneck，50 不要误写成50个块。CIFAR 的 6n+2、16/32/64 通道、32×32 输入是另一套原文实验设计，不替换本卡 ImageNet 五种结构。[cite:1](#source-1)

### 两类块的精确顺序

令块输入 x、主支 F、旁支 S，输出均为 `ReLU(F(x)+S(x))`。各卷积后均有 BN（含可学习 γ、β）；主支最后一个 BN 后、相加前**不单独 ReLU**，使 F 可表达正负修正。恒等旁路没有 BN 或 ReLU；投影旁路为 1×1 conv → BN，不单独激活。[cite:1](#source-1)[cite:4](#source-4)

| 块 | 主支顺序 | 首块降采样位置 | 输出通道 |
|---|---|---|---|
| Basic | 3×3(s,p=1) → BN → ReLU → 3×3(1,p=1) → BN | 第一个3×3，s=2 | b |
| 原版 Bottleneck | 1×1(s,p=0) → BN → ReLU → 3×3(1,p=1) → BN → ReLU → 1×1(1,p=0) → BN | **第一个1×1，s=2** | 4b |

各 stage 其余块 s=1。Bottleneck conv2_x 首块从64到256，即使空间不变也需要维度处理；Option B 使用 s=1 的投影。conv3_x/4_x/5_x 首块投影 s=2。Bottleneck 的“4倍”相对内部基宽 b，不是相对任意块输入通道；conv2_x 首个1×1为64→64，并非每个块都先缩小4倍。[cite:4](#source-4)

### Shortcut A、B、C 与原实验范围

| 选项 | 同 shape 块 | 维度增长/空间减半块 | 参数与原文定位 |
|---|---|---|---|
| A | S(x)=x | stride 采样原通道，增加通道补零 | 全部 shortcut 无学习参数；原18/34与plain的公平比较使用A |
| B | S(x)=x | 1×1投影到 Cout，空间变化用stride=2，再BN | 只在维度变化处投影；原50/101/152使用B |
| C | 每个块都1×1投影，再BN | 同样投影并按需stride=2 | 原34层消融比较；额外参数、计算，不是默认原版配置 |

A 在 stride=2 时不再是完整输入空间的恒等映射；零补的新通道没有对应的输入旁路，只能由主支产生。原文规定 stride 采样与补零，但未在已取材料中唯一确定补零分配到通道前后哪侧；L5 为可执行设计固定“原通道放前、尾部补零”，**不冒充原作者逐元素复刻**。A/B/C 是 shortcut 消融，不是 ResNet v1/v2/v1.5 的三个版本。原论文未证明 C 普遍最佳，也未要求所有18/34都只能用A。[cite:1](#source-1)

### 监督训练、停止与推理

1. 按对象/批次划分训练、验证与最终测试，定义 K 个互斥标签，y 为 `[N]` 整数且 `0≤y<K`；原文 ImageNet 有标签训练，不是让每个块单独拟合“残差真值”。
2. 原文从头训练：短边随机取 `[256,480]`，随机224×224裁剪及水平翻转、逐像素均值相减和所引标准颜色增强。工业任务若翻转会改变类别，应禁用并登记协议改变，不机械照搬。
3. 依层级表构造网络，初始化卷积（原文引用 He 初始化方法）；BN 用训练批统计，各参数通过同一最终分类损失反传。原版不使用 dropout。
4. 对每个 batch：清梯度 → logits 前向 → 稳定交叉熵与选定参数集合的权重衰减 → 反向 → momentum SGD 更新；记录 loss、训练误差、验证误差、梯度范数和有限值。
5. 原文配置：batch=256，学习率0.1，误差平台后除以10，momentum=0.9，weight decay=0.0001，最多600000次迭代；不是现代常见的固定90 epoch配方。作者另说明8 GPU、每卡32，不能将其自动理解为全局同步BN。
6. 达到冻结的训练预算后保存权重、类别表、预处理、BN统计、优化器与学习率状态；验证选模型与最终盲测分离。NaN/Inf、标签越界、shape不符立即拒绝该运行；达到预算只表示训练结束，不证明已收敛到全局最优。业务性能未达标不能登记为已验收。
7. 推理关闭参数梯度并使用冻结BN统计，图像经过同版本预处理，再前向至 logits；softmax/argmax只放在读出。原文比较采用10-crop，最佳配置另用多尺度全卷积评分；不能把中心单裁剪结果与多裁剪、模型集成的数字混用。[cite:1](#source-1)[cite:3](#source-3)

作者声明发布的 Caffe 模型用于测试/微调，**不是用该 Caffe 版本训练的**；训练后用足够大的训练批重新计算BN均值/方差的平均，而不是直接以 moving average 冒充同一结果。伪代码的通用训练流程不等于已经取得原内部训练程序；数据增强具体代码、最终统计重估细节和原优化器参数衰减分组未完整取得。[cite:3](#source-3)

## L2 机制：必要性、替换后果与反例

| 步骤/设计 | 因果机制 | 删除或替换的后果 |
|---|---|---|
| 残差重参数化 | 目标接近恒等时，只需学习小修正；同shape块存在F=0的候选 | 删除旁路变plain，函数类与优化路径改变；不能只说参数不变所以等价 |
| 相加而非拼接 | 同shape逐元素修正，输出通道不增加，反传从两支累加 | 拼接保留两套通道且后续权重不同，不是原残差块 |
| 1×1投影 | 匹配通道与空间，允许主/旁支同shape相加 | 广播、随意resize能掩盖模型错误；投影既耗参又不保证可逆 |
| 主支末端无ReLU | 允许负残差修正输入 | 加ReLU限制F≥0；与原块函数族不同 |
| 相加后的ReLU | 原版输出非负、提供非线性 | 删除它不是无损优化，而是改变架构；梯度门控也随之改变 |
| BN | 按通道规范化批激活并学习仿射，改变优化尺度；训练统计耦合样本 | 去BN需重新处理初始化/学习率；训练和推理统计错配会造成结果突变 |
| bottleneck | 将昂贵3×3放在窄通道空间，再投回宽输出 | 不是信息无损压缩，也不保证优于任何basic；优势依通道、硬件与任务 |
| 下采样与平均池化 | 减少空间计算并聚合全局类别信息 | 丢失小目标和精确位置；“stride变小”需重算算力、感受野与权重适配 |
| 最终标签交叉熵 | 为整个网络提供类别判别梯度 | 无标签、错误标签或不可见目标不会被shortcut自动修复 |

残差为零时，原版输出是 `ReLU(x)`。仅在 x 非负域（典型内部块输入来自前一ReLU）才等于 x；过渡块则是 `ReLU(S(x))`，不能称无损恒等。BN仿射偏置不为零时，将卷积权重设零也未必使 F=0，需同时检查末端BN输出。原文的构造性动机不保证 SGD 在有限时间内找到该构造。[cite:1](#source-1)[cite:4](#source-4)

## L3 工程权衡：参数、诊断与版本固定

### 参数尺度与选型顺序

| 参数/量 | 单位和依据 | 权衡及边界 |
|---|---|---|
| 深度/块数 | 无量纲；仅选五种已列结构作为对照 | 深度增加训练和显存；先验证18/34基线再比50以上，不保证单调增益 |
| 输入分辨率 | 像素；224是原分类裁剪协议 | 卷积计算近似随面积变化；小缺陷已丢失时增深无补救保证 |
| batch/BN统计域 | 图像数；原256总batch、每卡32 | 减batch同时改变优化与BN统计；梯度累积不等于一次大batch BN |
| 学习率/动量 | 无量纲更新系数；0.1/0.9为原配置 | 调整batch后不机械照用；核对动量是否额外乘(1−m)，避免有效学习率变动 |
| weight decay | 无量纲系数0.0001，需声明衰减参数集合 | BN、bias是否衰减不可由单个数字推断；AdamW不等于原SGD配置 |
| BN epsilon | 与激活方差同尺度；固定Caffe默认1e−5 | 防止除零不恢复可靠统计；跨库eps/方差估计/更新系数需一同固定 |
| shortcut/stride | A/B/C及算子位置，不是连续调参 | 为兼容权重而改stride会改变结果，即使参数shape可加载 |
| 类别头K | 类别数；原1000 | 微调换头后参数量及任务已变；没有未知类检测保证 |

数值起点分别来自原文、作者说明与固定 Caffe 默认值，选型顺序为工程建议：**标签/数据划分 → 预处理与权重协议 → 网络shape/算子顺序 → 小批过拟合测试 → 学习率与BN → 容量与正则 → 独立评测 → 性能剖析**，尚未用产线数据验证。[cite:1](#source-1)[cite:3](#source-3)[cite:5](#source-5)

### 失效症状与优先证据

| 症状 | 要采集的证据 | 处理顺序 |
|---|---|---|
| 更深模型训练误差也高 | plain/residual同配方曲线、梯度范数、各stage激活、预算 | 查实现/优化退化；不能直接归因为过拟合 |
| 训练好、验证差 | 按批次分组误差、标签泄漏、背景相关性 | 查过拟合及域偏移；不只继续加深 |
| train正常、eval突降 | BN running/reestimated stats、冻结策略、有效统计元素数 | 固定统计口径并用训练数据重估；不可拿验收集更新统计 |
| 权重能加载但结果差 | 首个1×1与3×3的stride、pool1采样窗口、RGB/BGR、均值 | 对齐逐层输出，不能仅比参数总数 |
| NaN或梯度异常 | 方差、混合精度溢出、loss/梯度有限值、学习率 | 先FP32基线与异常batch定位，再谈加速 |
| 小零件/细缺陷漏检 | 裁剪覆盖、输入像素尺寸、stride后可见性 | 改成像/ROI/任务头；整图分类概率不作定位证据 |
| 显存不足或延迟超预算 | 峰值显存、卷积/搬运耗时、batch与输入面积 | checkpointing/精度/算子融合分别验等价；不得删除旁路冒充优化 |

未提供通用分类误差界、现场误报/漏检界或时延保证。softmax高分不证明域内，真实验收应按产品、批次和类别代价报告混淆矩阵、置信区间及拒识协议；这里不虚构现场阈值或性能数字。

## L4 数学推导：监督、梯度、退化与复杂度

### 最终监督目标与参数更新

对训练样本 \( (X_i,y_i) \)，logits \(z_i=f_\theta(X_i)\in\mathbb R^K\)，定义一个明确的SGD教学目标：

\[
\mathcal L(\theta)=\frac1N\sum_{i=1}^N\left(\log\sum_{k=1}^K e^{z_{ik}}-z_{i,y_i}\right)
+\frac\lambda2\sum_{W\in\mathcal D}\|W\|_F^2.
\]

\(\mathcal D\) 是显式选定的衰减集合；原文给出 weight decay 系数，但已取得材料不足以完整复刻其内部参数分组。因此本式说明标准耦合 L2/SGD 机制，不伪称恢复了原训练源码的全部细节。

令 \(p_{ik}=\exp(z_{ik})/\sum_j\exp(z_{ij})\)，则：

\[
\frac{\partial\mathcal L_{\rm CE}}{\partial z_{ik}}=\frac1N(p_{ik}-\mathbf1[k=y_i]),\quad
v_{t+1}=m v_t+\eta_t\nabla_\theta\mathcal L_t,\quad
\theta_{t+1}=\theta_t-v_{t+1}.
\]

用 `a=max(z)` 写 `logsumexp(z)=a+log(sum(exp(z−a)))` 防溢出；融合交叉熵接收 logits，不先softmax再传。作者提示的动量实现正是将 lr 乘入速度增量的形式；学习率改变时，不能不核对速度定义就跨库拷贝优化器状态。[cite:3](#source-3) CE 导数与上述数值重写是本卡直接代数推导，不是残差分支的独立监督标签。

### 原版残差反传必须保留 ReLU 门

将整个batch特征向量化，记 \(u_l=S_l(x_l)+F_l(x_l;\theta_l)\)，\(x_{l+1}=\sigma(u_l)\)，\(D_l=\operatorname{diag}(\mathbf1[u_l>0])\)。远离ReLU零点，列梯度 \(g_l=\partial\mathcal L/\partial x_l\) 满足：

\[
J_l=\frac{\partial x_{l+1}}{\partial x_l}=D_l(J_{S_l}+J_{F_l}),\qquad
g_l=(J_{S_l}+J_{F_l})^TD_l g_{l+1}.
\]

恒等旁路的 \(J_S=I\)，故 \(g_l=D_lg_{l+1}+J_F^TD_lg_{l+1}\)，提供绕过主支卷积的直接项，但**仍经过相加后的激活门**。主支参数梯度是 \(J_{F,\theta}^TD_lg_{l+1}\)；有投影时其参数也收到梯度，不是旁路不训练。

多块反传是 \(g_l=J_l^T J_{l+1}^T\cdots J_{L-1}^T g_L\)。仅当同shape的恒等shortcut、且相加后映射为恒等（或所讨论局部区域所有门都打开）时，才可写：

\[
x_L=x_l+\sum_{i=l}^{L-1}F_i(x_i),\qquad
g_l=\left(I+\frac{\partial\sum_i F_i(x_i)}{\partial x_l}\right)^Tg_L.
\]

这是后续 identity-mapping/pre-activation 论文着重分析的条件形式，不是原版任意输入的无条件恒等式；过渡投影、下采样和关闭的ReLU门都破坏条件。[cite:1](#source-1)[cite:7](#source-7)

反例与条件界：

- 若某分量 \(u_l<0\)，该输出坐标的门导数为0；旁路不能越过这个门。
- 即使 \(D_l=I\)，取局部 \(F(x)=-x+c\) 且 c>0，则 \(I+J_F=0\)，梯度仍能抵消；这是数学反例，不是实测网络行为。
- 同shape且 \(D_l=I,\|J_F\|_2\le\epsilon<1\) 时，对任意 v 有 \((1-\epsilon)\|v\|\le\|(I+J_F)v\|\le(1+\epsilon)\|v\|\)。跨多块下界可乘为 \(\prod_l(1-\epsilon_l)\)，仍可能很小，且原网络不保证这些条件。
- 投影若不满秩或stride丢弃空间，\(J_S\) 无统一正下界；BN训练时还耦合batch，不能将其导数当固定逐样本缩放。

因此“存在恒等项”不等于梯度永不消失/爆炸，不等于优化凸，也不证明全局收敛；原文退化实验、构造性动机和本卡局部微分分析分别成立，不互相冒充证明。

### BN 训练与推理导数

对某通道统计域的 M=N×H×W 个标量 a，定义：

\[
\mu=\frac1M\sum_i a_i,\quad v=\frac1M\sum_i(a_i-\mu)^2,\quad
\hat a_i=\frac{a_i-\mu}{\sqrt{v+\varepsilon}},\quad b_i=\gamma\hat a_i+\beta.
\]

固定当前batch，若上游梯度为 \(q_i=\partial\mathcal L/\partial b_i\)，则：

\[
\frac{\partial\mathcal L}{\partial a_i}=\frac\gamma{\sqrt{v+\varepsilon}}
\left(q_i-\overline q-\hat a_i\,\overline{q\hat a}\right),\quad
\frac{\partial\mathcal L}{\partial\gamma}=\sum_iq_i\hat a_i,\quad
\frac{\partial\mathcal L}{\partial\beta}=\sum_iq_i.
\]

可见每个元素的梯度依赖统计域内其他元素；M=1时此训练定义退化为常量β，某些库直接拒绝。N=1但H×W>1不等于M=1，仍需考虑空间相关性与统计可靠性。推理用固定 \(\mu_*,v_*\) 时，导数才简化为 \(\gamma/\sqrt{v_*+\varepsilon}\)。此为所写BN定义的代数，不保证各库运行方差的无偏修正或累计方式一致；作者采用的训练后统计与一般running average须另行核验。[cite:3](#source-3)[cite:5](#source-5)

### Shape、可辨识性与参数检查

普通卷积每轴输出：

\[
H_o=\left\lfloor\frac{H_i+2p-d(k-1)-1}{s}\right\rfloor+1.
\]

原版1×1(s=2,p=0)与3×3(s=2,p=1,d=1)均给 \(\lceil H_i/2\rceil\)，但**算子内容不同**。加法前必须严格断言两路 N、C、H、W 全等；广播成功不是验证通过。分类头权重 `[K,C]`、bias `[K]`，平均池化特征 `[N,C]`；N=1禁止裸 `squeeze()`。

以 ResNet-50 的 conv3_x 首块为例：

| 操作 | 权重shape | 输出shape（省略N） | 卷积权重数 |
|---|---|---|---|
| 输入 | — | 256×56×56 | — |
| 主支1×1，s=2 | 128×256×1×1 | 128×28×28 | 32768 |
| 主支3×3，s=1 | 128×128×3×3 | 128×28×28 | 147456 |
| 主支1×1，s=1 | 512×128×1×1 | 512×28×28 | 65536 |
| 投影1×1，s=2 | 512×256×1×1 | 512×28×28 | 131072 |
| 相加→ReLU | — | 512×28×28 | 0 |

卷积权重合计376832；四处BN仿射共 `2×(128+128+512+512)=2560`，总379392个可学习标量（这些卷积均无bias）。BN均值/方差等缓冲另记。conv2_x首块的输入64×56×56、输出256×56×56也要投影，不能只以 stride≠1 判断投影需要。[cite:4](#source-4)

模型参数一般不可唯一辨识：例如内部通道置换同时补偿相邻层权重和BN参数可保持同一函数；窄通道与下采样通常不可逆。参数相同shape或参数量相同不意味着函数相同、学到同一表示，更不意味分类标签足以恢复原图几何。

### 复杂度及五种网络的可核算口径

标准密集卷积权重数 \(P_{\rm conv}=k^2 C_{in}C_{out}\)，可选bias再加 Cout；每图 MAC 为 \(H_oW_ok^2C_{in}C_{out}\)。BN仿射为2C参数；FC为CK+K参数、CK MAC。identity shortcut零参数，但相加仍需 \(O(NCHW)\) 运算与访存；“没有额外复杂度”是忽略相加的小量比较，不是零实际耗时。

忽略BN/bias、同分辨率时：Basic主支 \(9C_{in}b+9b^2\)，Bottleneck主支 \(C_{in}b+9b^2+4b^2\)。若两者都固定宽输入/输出 C，bottleneck取b=C/4，则其主支为 \(17C^2/16\)，而两层宽3×3为 \(18C^2\)。这是同外宽比较，**不代表标准ResNet-50一定比ResNet-34更省总算力**；两者stage外宽不同。

下表是按L1结构逐层求和、**Option B、所有conv无bias、含BN的γ/β与1000类FC bias、不含BN缓冲、只计conv/FC MAC**的确定性代数检查，不是训练/测速结果。原版bottleneck降采样在首个1×1。

| 模型 | 可学习参数 | 每张224×224图像MAC | 主支conv+FC深度 | B投影数 |
|---|---|---|---|---|
| 18 | 11689512 | 1814073344 | 18 | 3 |
| 34 | 21797672 | 3663761408 | 34 | 3 |
| 50 | 25557032 | 3857973248 | 50 | 4 |
| 101 | 44549160 | 7570194432 | 101 | 4 |
| 152 | 60192808 | 11282415616 | 152 | 4 |

参数量与 torchvision v0.20.1 元数据交叉对应，但其50/101/152的MAC不同：v1.5把首个1×1保留在高分辨率，移stride到3×3，例如其50为约4.089G MAC，而这里原版约3.858G。原论文Table 1的近似量为1.8/3.6/3.8/7.6/11.3 billion FLOPs，原文将multiply-add作为口径；若平台把乘和加各计1 FLOP，conv/FC约为2×MAC，再加其他算子，不能混比。[cite:1](#source-1)[cite:6](#source-6)

上述参数表**不是作者所有部署文件的字节级参数量**：固定提交的 `ResNet-50-deploy.prototxt` 的stem conv1没有设 `bias_term:false`，Caffe默认true，多64个bias，按图定义得到25557096；101/152的stem明确false。未下载权重检查blob，不声称该差异已被消除。A移除B的投影conv与BN参数，C增加同shape投影；不能给所有shortcut版本套同一总数。[cite:4](#source-4)[cite:5](#source-5)

训练通常还需保存逐层激活、梯度与优化器状态，激活存储量级为 \(O(N\sum_l H_lW_lC_l)\)，参数/梯度/动量量级为 O(P)，实现workspace另算。深度线性增加同宽块计算，checkpointing以重算换显存；实际延迟取决于带宽、kernel、精度与硬件，不从MAC直接推出毫秒。

## L5 实现：固定源码、伪代码与验证设计

### 源码入口与迁移边界

作者仓库固定提交 **`a7026cb6d478e131b765b898c312e25f9f6dc031`（2016-07-29）**，Caffe子模块固定 **`b590f1d27eb5cbd9bc7b9157d447706407c68682`**。它们是源码快照日期，不是方法首发年；作者发布的是从内部实现转换出的部署图。完整取得三份prototxt并抽取结构，未编译Caffe或加载模型。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

| 文件/符号 | 静态核对内容 | 不可外推的部分 |
|---|---|---|
| `prototxt/ResNet-{50,101,152}-deploy.prototxt` | conv数53/104/155，BN同数，相加数16/33/50；主支层数之外多4个投影 | 不含18/34部署图或原训练器；不能说五种都逐行核对作者实现 |
| `res2a_branch1` | 64→256，s=1投影 | 空间不变也可能需要投影 |
| `res3a_branch2a/2b/2c` | 1×1 s=2、3×3 s=1、1×1 s=1 | 参数shape不编码stride，加载成功不足以判原版 |
| `bn3a_branch1` / `scale3a_branch1` | BN使用global stats，Scale学习γ/β | Caffe BN与Scale共同对应现代带affine BN |
| `res3a` / `res3a_relu` | Eltwise默认SUM，之后ReLU | 不得改成concat、平均两支或pre-activation |
| `pool1` / Caffe `PoolingLayer::Reshape` | k=3,s=2,p=0，ceil取输出，末端窗口裁到有效区 | 不与torchvision p=1 floor池化逐点等价 |
| `pool5` / `fc1000` | 固定k=7平均池化、1000类FC | 泛化到任意尺寸需另写协议，非部署文件原样支持 |
| torchvision `BasicBlock/Bottleneck/ResNet._make_layer`，v0.20.1 | Basic expansion=1、Bottleneck=4；按stride或通道不符加projection+BN | Bottleneck是v1.5，现代初始化/权重/预处理不是原训练快照 |

版本必须分开：**2016 pre-activation（俗称v2）**把BN/ReLU放到权重层之前，改变相加后映射；**v1.5**仍是post-activation，但移动bottleneck降采样位置。torchvision `IMAGENET1K_V2` 是权重配方枚举，不等于pre-activation架构；`zero_init_residual`可选末BN置零也不是原2015配置的当然组成。[cite:6](#source-6)[cite:7](#source-7)

### 可实现伪代码：明确参考而非仓库模块

以下是控制流伪代码，卷积/BN等原语和参数创建未在仓库实现。构建时各层参数只创建一次、跨batch复用；`ConvBN`为conv→BN(含γ/β)，不含ReLU。主支conv不带bias是本卡统一教学约定；精确作者50部署移植须处理已列stem bias例外。18/34的stem池化沿用已核对作者50部署约定作为参考选择，不宣称取得了原18/34逐元素实现。

```text
Shortcut(x, Cout, stride, option):
    if option == C or (option == B and (channels(x) != Cout or stride != 1)):
        return ConvBN(x, Cout, kernel=1, stride=stride, pad=0)
    if option == A:
        require Cout >= channels(x)
        z = x[:, :, 0::stride, 0::stride]
        return PadChannelsAtEndWithZeros(z, Cout)
    require option == B and channels(x) == Cout and stride == 1
    return x

Block(x, b, stride, kind, option):
    require kind in {Basic, Bottleneck} and option in {A, B, C}
    if kind == Basic:
        r = ReLU(ConvBN(x, b, 3, stride, 1))
        r = ConvBN(r, b, 3, 1, 1)
        Cout = b
    else:
        r = ReLU(ConvBN(x, b, 1, stride, 0))
        r = ReLU(ConvBN(r, b, 3, 1, 1))
        Cout = 4 * b
        r = ConvBN(r, Cout, 1, 1, 0)
    skip = Shortcut(x, Cout, stride, option)
    require shape(r) == shape(skip)
    return ReLU(r + skip)

Forward224(X, depth, K, option):
    require shape(X) == [N, 3, 224, 224] and N > 0 and K > 1
    require Finite(X)
    kind, counts = ArchitectureTable[depth]
    x = ReLU(ConvBN(X, 64, 7, 2, 3))
    x = MaxPoolCaffeCeil(x, kernel=3, stride=2, pad=0)
    for stage in 0..3:
        for j in 0..counts[stage]-1:
            s = 2 if stage > 0 and j == 0 else 1
            x = Block(x, 64 * 2^stage, s, kind, option)
    require spatial_shape(x) == [7, 7]
    return Linear(MeanOverHW(x), K)

TrainStep(X, y, model, optimizer):
    require shape(y) == [batch(X)] and IntegerLabelsInRange(y, model.K)
    model.TrainMode()
    optimizer.ZeroGrad()
    logits = model.Forward224(X)
    loss = MeanLogSumExpMinusTarget(logits, y)
    require Finite(loss)
    loss.Backward()
    require AllGradientsFinite(model)
    optimizer.StepMomentumSGDWithDeclaredWeightDecay()

Infer(X, model):
    require ValidFrozenStatisticsAndProtocol(model)
    model.EvalMode()
    logits = NoGrad(model.Forward224(X))
    require shape(logits) == [batch(X), model.K] and Finite(logits)
    return logits, Softmax(logits, class_axis), Argmax(logits, class_axis)
```

权重衰减在此伪代码由优化器施加一次，不再向loss重复加入L2；保存/重估BN按已冻结协议另行执行。加载作者权重时，不凭部署prototxt猜RGB/BGR或均值常数，必须取得对应预处理说明/元数据；未齐全应报告协议缺失，而非悄悄套torchvision transforms。

### 测试、数值精度与优化等价性

| 检查 | 验收条件 | 本次状态 |
|---|---|---|
| 层数/参数算术 | 五组块数还原18/34/50/101/152，B投影3/3/4/4/4，匹配L4总数 | 已用PowerShell逐层代数求和；非模型实例化 |
| 作者源码结构 | 三份图conv/BN/add计数及过渡stride、pool、FC符合映射 | 已静态抽取；未运行Caffe |
| N=1/多batch前向 | 各stage shape和N×K输出正确，无广播或batch丢失 | 设计，未运行 |
| 恒等与投影块 | 构造F=0，非负同shape输入返回x；带负数返回ReLU(x)；过渡块严格对齐 | 设计，未运行 |
| 加法/ReLU梯度 | FP64有限差分避开零点，与含D门的链式法则一致；BN需声明train/eval | 设计，未运行 |
| 奇数/错误尺寸 | 固定入口拒绝非224；扩展实现用225、227、非方形测试采样相位/边界 | 设计，未运行；不靠resize隐藏差异 |
| train/eval统计 | 冻结后不随其他推理样本变化，统计更新只用训练数据 | 设计，未运行 |
| 预处理/权重迁移 | 对固定真实图逐层比对，允许误差由dtype/算子协议预先定义 | 未下载权重，无前向一致性证据 |
| 学习/泛化/性能 | 小批可过拟合，再分组盲测；报告top-1/top-5、延迟、峰值显存及配置 | 未训练、未测速、未做真实验收 |

仅冻结BN推理时可将 Conv(bias=b)→BN 融合：

\[
\alpha_c=\frac{\gamma_c}{\sqrt{v_{*,c}+\varepsilon}},\quad
W'_c=\alpha_cW_c,\quad b'_c=\alpha_c(b_c-\mu_{*,c})+\beta_c.
\]

这在实数算术和固定统计下等价；无bias时b=0。训练BN不可如此融合；ReLU跨越相加、stride搬移、pool padding改变均非等价变换。先用FP32/必要时FP64对照，再评估混合精度、量化、算子融合和checkpointing；量化误差及真实性能需单独测，不用“理论MAC减少”冒充优化验收。

## 面试自测：五问贯穿 L1–L5

| 层级问题 | 答案要点与成立条件 | 易错答案/反例 | 证据入口 |
|---|---|---|---|
| L1：224输入的ResNet-50为什么叫50层，conv3_x首块怎么走？ | 3/4/6/3个三层块，加stem和FC为50；256×56²→128×28²→128×28²→512×28²，投影到512×28²，相加后ReLU | 50个块；只在3×3处stride=2；conv2_x无需投影 | L1表、L4 shape表、原Table 1、作者branch2a：[cite:1](#source-1)[cite:4](#source-4) |
| L2：有shortcut为什么还会退化或梯度消失？ | 它改善优化参数化但不是定理；原版有ReLU门，JF可抵消I，投影非恒等；退化看训练误差，不只看测试 | “只要有+1就永不消失”；F=−x+c给局部零导数 | L2、L4链式法则；[cite:1](#source-1)[cite:7](#source-7) |
| L3：batch减小后eval变差，先加深还是调学习率？ | 先核预处理/标签、每卡BN统计与冻结策略、训练/验证曲线；再联合调batch和学习率，明确数据独立性 | 梯度累积等于大batch BN；用测试集重估均值 | L3诊断及作者BN说明：[cite:3](#source-3) |
| L4：推导原版块梯度并解释bottleneck省在哪？ | 从u=S+F、x'=ReLU(u)得(Js+JF)ᵀDg；窄3×3降低k²CinCout，首1×1降采样影响MAC；仅局部门全开时用I+JF | 省略D；把原34与50当同外宽；将MAC和双计FLOPs混用 | L4推导/参数表；[cite:1](#source-1)[cite:6](#source-6) |
| L5：权重shape全匹配就能复现作者输出吗？ | 不能；检查stride、pool相位、BN+Scale/eps/统计、stem bias、通道预处理与FC类别表，逐层比输出 | torchvision IMAGENET1K_V2就是原2016预激活；固定7×7自动支持任意尺寸 | L5源码表及测试矩阵；[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6) |

这些问题是知识库自测，不冒充真实面试题库；答5与文献核验不代表掌握已通过现场验证的训练系统。

## 来源、版本与获取深度

以下于 **2026-11-05** 联网获取。原文、作者仓库和其部署模型属于同一作者证据体系，按 **[SINGLE-SOURCE]** 限定历史实验结论；CVF与arXiv是版本核对，不算独立复现。torchvision为另一实现的交叉检查，但不能将其现代训练结果当作原模型验证。不存在三项独立性能复现证据，因此本卡不据此给模型性能排序。参数与shape计算为由已列结构直接推导的检查值，不是合成实验数据。

- <a id="source-1"></a>**[cite:1] He、Zhang、Ren、Sun，Deep Residual Learning for Image Recognition，arXiv:1512.03385v1，2015-12-10。** [摘要/提交历史](https://arxiv.org/abs/1512.03385)、[固定v1正文](https://arxiv.org/html/1512.03385v1)。摘要、历史、引言、§3.1–3.4、§4.1及Table 1–5相关内容已读。初次网页正文返回在§3.3截断，后续直接获取HTML并定向提取§3.3至§4.2之前，核对架构、A/B/C、BN、增强、SGD和消融；未逐页核验检测/定位附录，也未重新执行任何论文实验。
- <a id="source-2"></a>**[cite:2] 同题 CVPR 2016 正式版，770–778。** [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)。在线解析已读页首、引言、§3.1–3.3与式(1)/(2)附近；返回在Fig.3前后截断。未宣称正式版全部表格逐项对照；五架构完整表和训练细节以实际取得的固定2015 v1为依据，不用CVPR年份替换首发日期。
- <a id="source-3"></a>**[cite:3] 作者 KaimingHe/deep-residual-networks。** [仓库说明](https://github.com/KaimingHe/deep-residual-networks)、[固定README](https://github.com/KaimingHe/deep-residual-networks/blob/a7026cb6d478e131b765b898c312e25f9f6dc031/README.md)、[commit元数据](https://api.github.com/repos/KaimingHe/deep-residual-networks/commits/master)、[固定树](https://api.github.com/repos/KaimingHe/deep-residual-networks/git/trees/a7026cb6d478e131b765b898c312e25f9f6dc031?recursive=1)。说明正文与API已取，确定commit为`a7026cb6d478e131b765b898c312e25f9f6dc031`；核对转换模型、非原Caffe训练、8卡batch、BN训练后重估及动量口径。树只有50/101/152部署图，没有作者18/34训练源码。未下载README外链权重，不承诺下载链接当前可用。
- <a id="source-4"></a>**[cite:4] 作者固定部署源码。** [ResNet-50](https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/a7026cb6d478e131b765b898c312e25f9f6dc031/prototxt/ResNet-50-deploy.prototxt)、[ResNet-101](https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/a7026cb6d478e131b765b898c312e25f9f6dc031/prototxt/ResNet-101-deploy.prototxt)、[ResNet-152](https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/a7026cb6d478e131b765b898c312e25f9f6dc031/prototxt/ResNet-152-deploy.prototxt)。三份完整文本已获取，在内存统计conv/BN/Eltwise并抽取stem、stage首块、pool5、FC；核对stride=2在branch2a和branch1、BN/Scale、50 stem bias例外。不是逐层权重数值核验，未执行模型。
- <a id="source-5"></a>**[cite:5] BVLC/Caffe固定子模块b590f1d27eb5cbd9bc7b9157d447706407c68682。** [caffe.proto](https://raw.githubusercontent.com/BVLC/caffe/b590f1d27eb5cbd9bc7b9157d447706407c68682/src/caffe/proto/caffe.proto)、[pooling_layer.cpp](https://raw.githubusercontent.com/BVLC/caffe/b590f1d27eb5cbd9bc7b9157d447706407c68682/src/caffe/layers/pooling_layer.cpp)。已获取并定向读取Convolution/BN/Eltwise/Pooling默认值、ceil shape计算与窗口裁边；确认conv bias默认true、BN eps=1e−5、Eltwise默认SUM、pool pad默认0。未编译CPU/GPU后端或验证其跨库逐点等价。
- <a id="source-6"></a>**[cite:6] PyTorch/vision官方源码，固定tag v0.20.1。** [torchvision/models/resnet.py](https://raw.githubusercontent.com/pytorch/vision/v0.20.1/torchvision/models/resnet.py)。已读BasicBlock、Bottleneck、ResNet构造/_make_layer/_forward_impl及五种权重参数元数据；返回在152新配方说明附近截断，不宣称读完整个模块。核对v1.5 stride、现代pool、投影条件、zero_init_residual和权重V2区别；未安装或运行torchvision，此tag不是本机环境版本。
- <a id="source-7"></a>**[cite:7] He、Zhang、Ren、Sun，Identity Mappings in Deep Residual Networks，arXiv:1603.05027v3，所读修订2016-07-25。** [固定v3正文](https://arxiv.org/html/1603.05027v3)。已读引言、§2式(1)–(5)、恒等前提/过渡单元脚注、§3部分；用于界定后续pre-activation思想及梯度公式条件，不把修订日当首发日，不将该设计归入2015原版。未复现其1001层实验或完整核验后续全部消融。

## 本次验证与 G22 承接边界

本次只创建本文件。完成五种配置的主支深度、Option B投影数、参数量和conv/FC MAC的PowerShell逐层算术检查，以及三份作者prototxt静态结构核对；检查本卡L1–L5、自测五问、来源锚点、本地链接和写入范围。系统默认 `python` 不可用，检查使用已有PowerShell完成，未为本任务安装训练依赖。未新增源码、权重、训练器、测试脚本或修改任何台账。

**销项建议：**可将 **G22“缺少具体ResNet卡”／07-04“缺卡”转为“已有”**，承接键 `resnet.md:RN01`，保留G22历史ID；五层建议登记 **核／核／部／核（限定代数与静态结构）／部，答5**。理由是原残差块、projection、五种架构与监督训练已具体展开，不再仅有C13通用跳连。此次不实际写入索引/覆盖矩阵或调整总数，后续统一登记时再同步。

**不能销掉的事项：**本仓库未实现/未训练/未加载权重；原内部训练源码与全部预处理细节未取得；18/34作者逐元素实现和Option A通道补零位置未闭合；正式2016全文逐表比对未完成；无跨框架前向/梯度一致性、ImageNet复现或工业现场验收。原版模型卡已有不等于L5已核验，更不等于整个分类装配系统完成。
