# CR01 CRNN：七层卷积列序列、双向循环与 CTC 转录

[学习规范](../algorithm_learning_standard.md) · [OCR 演进](../ocr_codes_evolution.md) · [覆盖台账 G28](../evolution_coverage.md#去重缺口清单) · [CTC 专卡 CT01](ctc.md#ct01) · [切字基线 O01](remaining_domains.md#o01-切字加模板相关-ocr)

<a id="cr01"></a>
<a id="g28"></a>

## 任务、演进与覆盖状态

**任务契约：**输入已定位、阅读方向已校正的单行／单词灰度 ROI，训练另有整串标签；输出逐时间步类别分布、折叠后的原始字符串，以及另行标定的候选／拒识信息。CNN 从左向右抽列特征，两层 BiLSTM 建模上下文，CTC 在没有字符框监督时对齐并训练。CRNN 不是文本检测器、全页布局系统、条码解码器，也不自动支持未知字符。[cite:1](#source-1)

**演进定位：**显式切字的连笔／粘连错误、手工特征与序列模型分离训练 → 用 CNN 列特征替代固定整图表示，以双向循环利用左右上下文，用 CTC 联合训练 → 省去逐字边界标签，但增加整串训练数据、循环计算、宽度和字符集管理 → 仍受单调阅读顺序、图像可辨识性、域偏移和时间步预算限制。减少显式切字不是消除所有预处理；ROI 裁切错误仍会传递到识别结果。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-05**。仅有 `crnn.md:CR01` 一张模型卡，承接 **G28／08-03／O-H3**；CNN、BiLSTM、CTC 和版本对照不另计卡。预印本为 **2015-07-21 arXiv v1**，期刊为 **TPAMI 39(11), 2298–2304，2017-11**。作者代码固定到 `bgshih/crnn@f5d41e3355fc4447f77cd6d5e5777b3a160c93cb`，提交日期 2018-05-04；它是后续可审计快照，不冒充 2015 实验发布标签。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：原文三段式流程、七卷积骨架及固定代码形状；原文未唯一确定的尺寸细节保留冲突 |
| L2 机制 | 已核验：列序列、双向上下文、整串监督、折叠和长度约束，限静态资料与条件分析 |
| L3 工程权衡 | 部分：原配置与变宽／padding／拒识设计，无实际 OCR 定参 |
| L4 数学推导 | 部分：列映射、循环、CTC 梯度接口、复杂度与有限算术检查；非完整 CTC 专卡或全局理论证明 |
| L5 实现 | 部分：固定外部源码对应、参考伪代码、纯标准库检查；本仓库未实现 CRNN |
| 面试自测 | 答 5：L1–L5 各一问，含条件、易错点与证据入口 |
| 验证范围 | 文档链接、尺寸／布局／归一化与极小 CTC 数值检查；不包含网络前反向、训练、权重或真实图像评测 |

初版检索仓库 Python 源码未发现 CRNN／CTCLoss 专用实现；[DL 路线](../../dl/README.md)中的其他网络不能继承为本卡实现。**历史交接记录：**初版曾记录“未发现CTC独立卡”，因此当时只链接OCR总览；这不是当前文件状态。2026-11-05复核已读取现有 [CTC 专卡 CT01](ctc.md#ct01)，其前后向、数值稳定、重复字符与解码内容承接G27。专卡存在及其有限算法检查不等于CRNN／CTC生产实现或完整OCR接入，也不替G27宣告工程验收完成。本文件不修改索引、总览或台账；其中旧“缺卡”记录留待单独同步。

## L1 步骤：原骨架、明确尺寸与完整读出

### 输入协议与七个卷积

统一本文尺寸写法为 **H×W**，张量布局 **N×C×H×W**，卷积／池化参数也按 **(高,宽)**。原文表格按输出层在上、输入在下排列，应反向读取。输入高度 H=32、灰度 C=1；七个卷积的输出通道为 **64、128、256、256、512、512、512**，前六层核3×3、stride1、padding1，末层核2×2、stride1、padding0。CNN 后没有整图固定宽度全连接分类器；循环层中的逐帧线性读出仍存在，不能把“去掉 CNN 全连接层”理解成全网络没有线性权重。[cite:1](#source-1)[cite:4](#source-4)

为给出可检验的空间形状，下表明确采用**固定作者 Torch7 代码的算子参数**，不是替原文歧义补一个所谓唯一答案。设整数输入宽度 W≥4，q=⌊W/4⌋；所有尺寸按 floor 模式、dilation1 推导。Torch7 的 `SpatialMaxPooling(kW,kH,dW,dH,padW,padH)` 是宽在前，移植到高在前的 API 时必须换位。[cite:4](#source-4)

| 顺序 | 固定代码操作，参数为 H×W | 输出 C×H×W | W=100 时 |
|---|---|---|---|
| 输入 | byte 灰度转浮点并归一 | 1×32×W | 1×32×100 |
| Conv1 | 3×3，1→64，p=(1,1)，ReLU | 64×32×W | 64×32×100 |
| Pool1 | k=(2,2)，s=(2,2)，p=(0,0) | 64×16×⌊W/2⌋ | 64×16×50 |
| Conv2 | 3×3，64→128，p=(1,1)，ReLU | 128×16×⌊W/2⌋ | 128×16×50 |
| Pool2 | k=(2,2)，s=(2,2)，p=(0,0) | 128×8×q | 128×8×25 |
| Conv3 | 3×3，128→256，p=(1,1)，BN→ReLU | 256×8×q | 256×8×25 |
| Conv4 | 3×3，256→256，p=(1,1)，ReLU | 256×8×q | 256×8×25 |
| Pool3 | k=(2,2)，s=(2,1)，p=(0,1) | 256×4×(q+1) | 256×4×26 |
| Conv5 | 3×3，256→512，p=(1,1)，BN→ReLU | 512×4×(q+1) | 512×4×26 |
| Conv6 | 3×3，512→512，p=(1,1)，ReLU | 512×4×(q+1) | 512×4×26 |
| Pool4 | k=(2,2)，s=(2,1)，p=(0,1) | 512×2×(q+2) | 512×2×27 |
| Conv7 | 2×2，512→512，p=(0,0)，BN→ReLU | 512×1×(q+1) | 512×1×26 |

**最后时间步数是 T(W)=⌊W/4⌋+1，而不是无条件 W/4。**Pool4 的实际算术宽度是27，不能照抄源码行末的“26”注释；Conv7 再减1到26。W=101、102、103仍是T=26，W=104变为27。W≥4只保证这套局部算子尺寸非零，不保证识别可用，也不属于原论文推荐的测试最小宽度。

原文 §3.2／Table 1 则写后两次池化窗口 **1×2（按宽×高理解）**，并举100×32产生25帧；还写 **Conv5、Conv6 后两个 BN**。若机械把原文窗口移植为 k=(2,1)、s=(2,1)、p=0，并保留末层2×2无填充卷积，会得到 **T=q−1=24**，与25帧示例不一致；表格的标量 `s:2` 也不足以唯一确定两个轴。这里将“原文25帧陈述”“该字面移植24帧”“固定代码26帧”分别保留，不自行声称作者修正了哪一个版本。[cite:1](#source-1)[cite:4](#source-4)

### Map-to-Sequence、两层 BiLSTM 和输出

对固定代码得到的 F∈R^(N×512×1×T)，第 t 列构成 x_t∈R^(N×512)。现代 time-major 表达为 `F.squeeze(2).permute(2,0,1)`，得到 **[T,N,512]**；不能直接无转置 `reshape(T,N,512)`。一般高度 H′>1 时列维度为 D=C′H′，而不是把 H′并入时间；本架构通过 H′=1 固定 D=512。[cite:1](#source-1)

| 组件 | 输入 | 内部状态／原始输出 | 投影后输出 |
|---|---|---|---|
| BiLSTM1 | [T,N,512] | 每方向 hidden=256，h/c各[N,256]；两方向逻辑拼接[T,N,512] | 两个256→256线性结果相加：[T,N,256] |
| BiLSTM2 | [T,N,256] | 每方向 hidden=256，逻辑拼接[T,N,512] | 两个256→K线性结果相加：logits [T,N,K] |
| LogSoftmax | [T,N,K] | 在类别轴归一，时间不改变 | log_probs [T,N,K] |
| CTC／解码 | log_probs、有效长度 | 训练需要目标串；推理不需要目标 | 每样本loss／变长字符串 |

固定作者配置 K=36+1=37：数字、大小写合并的26个拉丁字母、blank。`BiRnnJoin` 不是把两方向256维向量直接相加；它先各自线性投影再相加。等价写法是拼接后乘 `[W_f W_b]`、偏置取 `b_f+b_b`；函数可等价，参数化和优化轨迹不必等价。第二层的输入因此是256，**不是**裸拼接的512；套用两层内置 BiLSTM 再接一个线性层属于另一结构。[cite:4](#source-4)[cite:5](#source-5)

Torch7 实际按时间组织 N×D 张量表，最后重新拼成 **[N,T,K]**，传到 C++ CTC；其变量名 `nFrame` 实际指 batch 维。现代 `torch.nn.CTCLoss` 接 **[T,N,K]**，这两个布局不能互传而只修改变量名。[cite:4](#source-4)[cite:8](#source-8)[cite:10](#source-10)

### 训练、停止条件与推理解码

1. 冻结 ROI、字符编码和数据划分。训练对为 `(image, target_string)`；目标中不含 blank，不以逐列伪标签替代整串监督。字符框不是必需监督，但图像仍必须覆盖完整目标。
2. 按选定版本灰度化、缩放和数值归一；计算每个样本真实 T_i、标签长度 U_i、相邻重复对数 R_i，先检查 `T_i≥U_i+R_i`。坏图、未知字符、越界长度或非有限值直接分流并保留计数。
3. 训练模式下执行 CNN → 列映射 → 两层 BiLSTM → log_softmax → CTC 负对数似然；清梯度、反传至 CNN、执行选定优化器更新。**不把 argmax／折叠放入可微训练图。**
4. 原预印本写 ADADELTA、ρ=0.9，训练缩放到 W×H=100×32；测试固定H=32、按比例缩放宽度但至少100。固定代码训练batch=64，`maxT=26`，`optim.adadelta` 的配置对象为空；不能据此断言它显式设置了论文ρ，也不能把训练加速用的固定宽误写成论文训练全程保长宽比。[cite:1](#source-1)[cite:4](#source-4)[cite:6](#source-6)
5. 工程训练按冻结的迭代预算／验证集规则停止，保存模型、BN统计、字表、宽度变换和优化器状态。固定代码给 `maxIterations=2000000`，循环条件是更新后 `iterations > maxIterations`；不把它描述成严格恰好2000000步。不宣称达到预算等于全局收敛。[cite:7](#source-7)
6. 推理切换 eval、冻结BN统计、不构建参数梯度；只读取每样本前T_i步。贪心 best-path 先逐步argmax，再**合并相邻重复，最后删blank**：`A,A,blank,A → AA`。blank 是无输出，不是空格，也不是终止符；CTC没有必须预测的EOS，遍历T_i步后停止。[cite:8](#source-8)[cite:9](#source-9)
7. 词典模式对候选串计算 CTC 总路径概率再排序。原文大词典用编辑距离邻域和BK-tree缩小候选，可能漏掉全局最佳词；固定快照 `decodingWithLexicon` 则复制输入并对传入词典穷举评分，不是已复现的BK-tree。有限宽 prefix beam 是可选现代扩展，也不保证全局最优；不能把“贪心后拼写纠正”称为原CTC精确解码。[cite:1](#source-1)[cite:11](#source-11)

## L2 机制：为什么需要这些步骤

| 设计 | 解决的困难／不变量 | 去掉或误用的后果 |
|---|---|---|
| 学习卷积特征 | 从局部笔画提取可联合优化的表示，参数沿宽共享 | 固定手工特征可作基线，但不再是同一端到端学习链 |
| 列序列保留宽轴 | 空间从左到右映射到时间，允许变长输出 | 全图flatten接固定分类头将宽度绑定；全局池化丢掉顺序 |
| 后两次非对称下采样 | 压低高度而少压宽度，保留足够对齐时间步 | 全部2×2 stride2使T更短，重复字符或长串可能无合法路径 |
| 两层双向循环 | 同时利用左右上下文；每列可见字符的一部分而非一个整字 | 单向版没有未来上下文；两层不保证单调提效，增加延迟与过拟合风险 |
| CTC对齐边缘化 | 一条标签串对应许多blank／重复路径，不必提供字符边界 | 每列CE需要额外对齐标签；对argmax回传没有CTC梯度 |
| BN与归一化 | 控制优化尺度，训练与推理必须用同协议 | BN统计掺入padding／测试图，或byte与[0,1]混用，会改变分布 |
| 正确有效长度 | 不把存储补齐区域当真实观察，重复字符需blank分隔 | 一律令T_i=T_max可能让模型利用补齐帧对齐，本不合法样本被掩盖 |

CTC的路径概率按帧因子相乘，但每帧分布可以依赖整行图像和双向状态；这**不等于**假设原始图像列相互独立。每个时间步可以覆盖重叠感受野，所以不能把帧峰位置当精确字符框。LSTM的记忆通路减轻长程梯度困难，不保证梯度永不消失或爆炸。[cite:1](#source-1)[cite:5](#source-5)[cite:9](#source-9)

## L3 工程权衡：归一、变宽与防泄漏

### 三种不同的宽度／数值协议

| 协议 | 实际约定 | 不可混淆之处 |
|---|---|---|
| 原预印本实验 | 训练100×32；测试高32、宽随比例缩放且至少100 | 这是训练／测试不同缩放策略，不是“始终不改变长宽比” |
| 固定作者公开代码 | `DatasetLmdb` 和 `loadAndResizeImage` 均缩放100×32；网络内 `(byte−128)/128` | 0→−1，128→0，255→127/128；不是255分母，也不是直接喂已归一张量 |
| 本卡变宽工程参考 | 灰度ROI按H=32等比缩放，W_i取事先固定取整规则；超宽分流；按有效宽运行 | 是另选的实现协议，需重新验证／训练，不能冒充原论文结果 |

上述原协议来自实读原文和代码。[cite:1](#source-1)[cite:4](#source-4)[cite:6](#source-6) 参考方案可固定 `W_i=max(4, floor(32·W_orig/H_orig+0.5))`；最小4仅为尺寸防错，字形过窄仍拒识。若保留论文“至少100”的工程习惯，要明确是横向拉伸到100，还是右侧加空白到100；二者改变的图像、有效长度和统计不同。

数值归一前先转浮点，禁止在uint8上减128造成回绕。若输入已经是p∈[0,1]，复刻固定代码需用 `(255p−128)/128`，不是再按byte路径二次归一；常见 `2p−1` 与此略有不同。padding值必须明确处于原像素域还是归一后域：归一后0对应原灰度128，并不代表白背景；白色255对应127/128。

### 变宽 batch 不能只给 CTC 一个长度

这是根据算子依赖关系提出的工程防错方案，**不声称作者已实现**：

- 每样本保存原尺寸、缩放后真实W_i、补齐宽W_pad及变换版本；用真实W_i逐层递推得到T_i，不能按标签长度推测T_i，更不能为了避开无穷loss把长度改成T_max。
- CTC `input_lengths` 只截断损失使用的时间范围；它不会阻止**反向LSTM从右侧补齐帧读回信息**。需按T_i进行packed／masked双向递归，或逐样本运行；第二层也必须遵守同一长度，变长排序后恢复原样本顺序。
- 即使循环层正确截断，**CNN仍可能在有效边界受padding污染**：卷积偏置、BN、激活及多层卷积会让补齐区域生成非零特征，再传播回有效列。最后把CNN特征裁到T_i，不保证与单独未补齐运行相等。
- 可靠基线是先按真实W_i分别跑CNN，再pad特征序列并pack两层RNN；或者按**完全相同宽度**分组跑CNN，再合并梯度。BN训练域应固定为这些组，不能声称与原大batch训练统计等价。近似宽度分桶只减轻浪费，不自动解决污染。
- 若必须图像级padding，需要逐层追踪各轴有效支持、重置无效激活并处理池化边界及masked BN；不能仅在输入乘一次mask。先建立“单独运行 vs padded batch”的有效logits、梯度对照，再接受优化。
- eval使用冻结BN统计；训练BN若把大量padding纳入N×H×W统计，即使所有padding像素是0仍改变均值／方差。不得用测试集重估统计或选择补齐策略。

另外的数据泄漏不同于padding污染：同一标签／工件连拍、同源裁块和增强版本不得跨训练／验证／测试；先按工件、批次、相机和字体分组，再做增强。字表、词典、拒识阈值只由训练／验证协议冻结，不能把测试真值词典当部署无约束识别能力。长度与背景对字符内容的偶然相关性也需做独立消融，不能仅凭整体准确率排除捷径学习。

### 参数、故障与调参顺序

| 参数／症状 | 要看什么 | 调整与代价 |
|---|---|---|
| H=32、W_i（像素） | 笔画是否欠采样、T_i与U_i+R_i | 先成像和缩放，后宽度预算；加宽增加CNN／RNN／CTC成本，不恢复原始丢失细节 |
| 两层、每方向h=256 | 序列长度分组CER、整串错率、梯度范数 | 是原配置起点，不是最优常数；改h改变参数和逐步计算 |
| K=字表+blank | 未知字、大小写、空格和Unicode分词 | 作者36类映射大小写合并，不能直接识别中文；扩大K需重建头与标签协议 |
| `11→1`或长串掉字 | T/U/R、blank索引、折叠顺序、裁切 | 先确认可对齐和输入完整，再比较解码；不能先删blank |
| 单张正常、混batch异常 | 反向RNN长度、CNN边界和BN统计 | 以逐样本eval为基线；padding多不等于只是速度慢 |
| loss=Inf／NaN | 无合法路径、标签越界、log_softmax、精度 | 先拒绝不合法样本，再FP32与log-domain检查；`zero_infinity`不是数据修复 |
| 高分读错随机编号 | 近似字、候选差值、词典覆盖、域偏移 | 记录原串与修正串；独立校准拒识，不能把常见词修正当正确性保证 |
| 延迟或显存超预算 | 按宽分组的CNN、RNN、搬运、解码耗时 | 先同宽分组和减少补齐，再量化／融合；双向RNN不能无代价改成严格流式 |

调参顺序为 **采集与ROI → 字表／标签和划分 → 宽度与T/U/R → 归一和padding → 网络／loss接口 → 优化器 → 解码与拒识 → 性能剖析**。验收分别统计CER、整串完全正确率、拒识率、误放行及端到端尾延迟；CER按参考字符总数作分母，空真值另列。没有本仓库的真实OCR精度、速度或现场误读界，不复用原论文机器上的时间数字作部署承诺。

## L4 数学推导：shape、梯度与复杂度

### 空间到时间的推导

对任一轴，floor卷积／池化输出为：

\[
n'=\left\lfloor\frac{n+2p-d(k-1)-1}{s}\right\rfloor+1.
\]

固定代码宽轴先两次整除2得到q；Pool3和Pool4各由 `(k,s,p)=(2,1,1)` 得 `n′=n+1`；Conv7由 `(2,1,0)` 得 `n′=n−1`，所以T=q+1。高轴为32→16→8→4→2→1。对给定目标y，U为字符数，R为相邻同字符对数，相同相邻字符必须被blank隔开，因此：

\[
T\ge U+R,\qquad \lfloor W/4\rfloor+1\ge U+R.
\]

固定代码下可得尺寸意义的下限 `W≥max(4,4(U+R−1))`。它是有足够步数容纳路径的条件，不是光学字宽建议；标签决定的最小训练宽度不能成为部署时偷看真值的策略。若所有logits有限、所有类别softmax概率为正，该长度条件允许构造合法路径；若输出被额外硬约束，仍需检查路径支持。

列映射是一种置换／重排。一般形式 `x[t,n,cH′+r]=F[n,c,r,t]`；上游梯度原位置回填 `dF[n,c,r,t]=dx[t,n,cH′+r]`，无可学习参数，不需插值。置换保持内积，可用 `<P(F),G>=<F,P⁻¹(G)>` 检查反传接口；只检查元素总数相同不足以发现N／T互换。

### 双向循环与投影梯度

每方向的常规无peephole LSTM写为：

\[
(i_t,f_t,o_t,g_t)=(\sigma,\sigma,\sigma,\tanh)(W_xx_t+W_hh_{t-1}+b),
\quad c_t=f_t\odot c_{t-1}+i_t\odot g_t,\quad h_t=o_t\odot\tanh(c_t).
\]

反向分支把前驱改为t+1，并从真实T_i处的零初态向左走；正向从0处零初态向右走。固定代码门顺序是 **i、f、o、g**，迁移权重到其他框架必须核对门块顺序及两个bias的合并，不只核对矩阵shape。[cite:5](#source-5)

对投影 `v_t=W_f h_t^f+W_b h_t^b+b_f+b_b`，上游g_t给出：

\[
\partial h_t^f=W_f^Tg_t,\quad \partial h_t^b=W_b^Tg_t,
\quad \partial W_f=\sum_{n,t}g_{tn}(h_{tn}^f)^T,
\quad \partial b_f=\partial b_b=\sum_{n,t}g_{tn}.
\]

两方向对同一输入的梯度相加，沿时间BPTT，再经列映射逆操作回CNN。`∂c_t/∂c_(t−1)=diag(f_t)` 只描述把其他门输入视作固定的直接记忆通路；完整Jacobian还有经h及门的间接项，不能据此证明梯度恒定。ReLU、max-pool在折点／并列最大值处需按实现选次梯度，有限差分测试要避开这些不可微点。

### CTC目标与logits梯度接口

设a_tk是logits，p_tk=softmax(a_t)_k，ℓ_tk=log p_tk；B先合并连续相同标签再删blank。单样本目标：

\[
Z=p(y\mid X)=\sum_{\pi:B(\pi)=y}\prod_{t=1}^{T}p_{t,\pi_t},\qquad L=-\log Z.
\]

把y扩展为 `y′=(blank,y1,blank,…,yU,blank)`，状态数S=2U+1。令α_t(s)包含t时刻发射，则它由s原地、s−1和合法的s−2到达；跨两步仅允许当前符号非blank且不同于s−2符号。实现用logsumexp相加，越界项为−∞；初始化仅首blank与首字符可达，末尾只接受最后blank和最后字符。这一接口摘要用于说明CRNN训练；完整递推、空目标边界与解码参考见现有 [CTC 专卡 CT01](ctc.md#ct01)，不代表已经接入CRNN。[cite:8](#source-8)[cite:9](#source-9)

定义β_t(s)为**不包含当前t发射**的后缀权重，终止接受状态β_T=1；则状态后验 `α_t(s)β_t(s)/Z` 不需要再除p_tk。若采用包含当前发射的另一β约定，必须除去一次重复发射，不能混用两种公式。令：

\[
\gamma_{tk}=\sum_{s:y'_s=k}\frac{\alpha_t(s)\beta_t(s)}Z,
\quad \frac{\partial L}{\partial\ell_{tk}}=-\gamma_{tk},
\quad \frac{\partial L}{\partial a_{tk}}=p_{tk}-\gamma_{tk}.
\]

最后一步由 `∂ℓ_j/∂a_k=1[j=k]−p_k` 和 `Σ_kγ_tk=1` 得到。成立条件是Z>0、非空有效时间轴和正确后验；不合法路径下Z=0不能硬算有限梯度。固定C++内核接log_probs并回传−γ，随后经过LogSoftMax变为p−γ。[cite:8](#source-8) 现代框架内部实现的log_probs梯度约定可能带归一流形项；验等价时优先比较**组合到logits的梯度**，不要把−γ手工重复接入库反传。

有效时间以外不直接贡献CTC的logits梯度，但若有效logits依赖补齐像素／反向状态，梯度仍可能间接流到补齐输入；这正是“loss mask不等于整个网络mask”的原因。标签、离散长度、argmax及折叠都不是可学习参数接口。

原代码按batch均值缩放总loss／梯度；现代CTCLoss默认`reduction='mean'`先按每样本目标长度除，再取batch平均，不是同一目标权重。本卡参考伪代码选`reduction='none'`后显式`sum/N`。现代接口还必须传非blank整数targets、input_lengths、target_lengths；不能把softmax概率或原始logits直接当作log_probs。[cite:7](#source-7)[cite:10](#source-10)

### 复杂度与保证范围

记第j卷积输出空间H_jW_j，输入／输出通道C_(j−1)、C_j，核面积k_j²；忽略常数并以一次乘加为一个MAC：

\[
\mathrm{CNN\ MACs}=N\sum_j H_jW_jC_{j-1}C_jk_j^2.
\]

BN、ReLU、pool另计线性遍历。高度和通道固定时CNN随输入宽度近似线性；同宽batch用NW，变宽独立计算用ΣW_i，图像补齐计算则用NW_max。

一个双向LSTM层、输入d、每方向h、读出o的门矩阵MAC约为 `8NTh(d+h)`，投影约`2NTho`；两层代入 **(d,h,o)=(512,256,256)、(256,256,K)** 后相加。参数量不随T增长，但训练激活保存约O(NTh)每层，顺序递归仍限制时间并行；减小h通常以二次项降低循环计算，但不是语义等价优化。

CTC单目标动态规划O(TU)，保存前后向表O(TU)，另有logits／梯度O(TK)；只求loss时可滚动状态降为O(U)。batch取ΣT_iU_i或补齐上界NT_maxU_max。贪心扫描O(TK)，词典穷举CTC评分O(Σ_(y∈D)T|y|)，复制整份logits还增加O(|D|TK)空间。beam宽B的常见扩展约O(TBK)，另加排序和前缀存储；BK-tree的性能依词典及半径，不将原文简写的O(log|D|)当任意数据上的最坏界。[cite:1](#source-1)[cite:11](#source-11)

未提供通用识别误差界、非凸优化全局收敛保证或部署时延界。宽度公式和CTC路径可行性是离散结构结论，不是OCR精度证明。

## L5 实现：源码对应、伪代码与有限验证

### 固定源码对应和移植边界

| 固定源码入口 | 本卡对应 | 尚未覆盖 |
|---|---|---|
| `model/crnn_demo/config.lua` | 七卷积、三个BN、池化、两层投影BiLSTM、LogSoftMax | 未在本机构造Torch7模型或加载权重 |
| `src/LstmLayer.lua`、`src/BiRnnJoin.lua` | 双向递归、门顺序、零初态、逐帧投影 | 原CUDA／nngraph逐元素前反向未运行 |
| `src/DatasetLmdb.lua`、`src/utilities.lua` | 固定缩放、byte路径、36类映射 | LMDB解码、真实字形、跨库插值未测试 |
| `src/training.lua` | 整串监督、batch均值、训练／eval、停止 | 原训练脚本及完整依赖未运行 |
| `src/cpp/ctc.cpp`、`src/inference.lua` | log-domain目标、贪心、词典穷举 | 未编译C++内核，未执行词典或beam解码器 |

这些路径均指**外部固定快照**，具体可点击入口见来源4–8、11，不是本仓库新增源码。作者README称测试环境为Ubuntu14.04、CUDA／Torch7；当前Windows环境不据此推断可直接运行。作者列出的PyTorch port是他人移植，不能把它当Shi原Torch7代码；本卡未用该port补证原结构。[cite:3](#source-3)

### 带shape的参考伪代码

以下是**未实现的工程接口伪代码**，不是可导入模块。算子按本卡固定代码结构定义；默认使用逐样本CNN避免图像padding污染，训练时BN统计域亦随之改变，不能称原训练复刻。生产实现若改为同宽组，需冻结BN与loss聚合协议。

```text
inputs: grayscale_rois[N], targets[N], alphabet, width_policy
for i in 0..N-1:
    image_i, W_i = resize_and_validate(grayscale_rois[i], H=32, width_policy)
    X_i = (float(image_i) - 128) / 128
    F_i = cnn_fixed_author_geometry(X_i[1,1,32,W_i])
    T_i = W_i // 4 + 1
    require shape(F_i) == [1,512,1,T_i]
    S_i = permute(squeeze(F_i, axis=2), [2,0,1])
    require shape(S_i) == [T_i,1,512]
    if training:
        y_i = encode_without_blank(targets[i], alphabet)
        U_i = length(y_i)
        R_i = count_adjacent_repeats(y_i)
        require T_i >= U_i + R_i
S, lengths, restore_order = pad_then_pack(S_i, T_i)
H1 = packed_bidirectional_lstm(S, input=512, hidden=256)
V1 = project_directions_and_add(H1, output=256)
H2 = packed_bidirectional_lstm(V1, input=256, hidden=256)
A_packed = project_directions_and_add(H2, output=K)
A = restore_order(unpack(A_packed))
require shape(A) == [max(T_i),N,K]
logp = log_softmax(A, axis=2)
require all_finite(logp)
if training:
    losses = ctc(logp, concat(y_i), input_lengths=T_i, target_lengths=U_i,
                 blank=0, reduction=none, zero_infinity=false)
    loss = sum(losses) / N
    require is_finite(loss)
    zero_parameter_gradients()
    backward(loss)
    require all_parameter_gradients_finite()
    optimizer_step()
else:
    for i in 0..N-1:
        path_i = argmax(logp[0:T_i,i,:], axis=1)
        raw_i = remove_blank(merge_adjacent(path_i))
        emit(raw_i, calibrated_rejection_policy(raw_i, logp[0:T_i,i,:]))
```

推理分支不需要targets；空目标U=0按全blank路径处理，且不按U做除法。标签padding不进入有效target长度；图像padding、序列padding、blank三者语义不同。packed双向层输出投影必须保留packed元数据，否则第二层可能重新读入补齐部分。部署GPU后端的dtype、CuDNN适用条件和确定性选项另锁版本检查，不为走快速路径把T_i强行全改成T_max。[cite:10](#source-10)

### 可执行的有限算术检查器

下面只用Python标准库，不依赖torch，不调用任何训练网络。检查固定尺寸公式、列置换、像素端点、重复折叠、极短路径的CTC前向与穷举一致性、组合logits梯度的有限差分。离散路径枚举和人为数学输入是**单元测试夹具**，不是采样数据、识别实验或性能证据。

```python
from itertools import product
from math import exp, log, floor, isfinite

def out_size(n, k, s=1, p=0):
    return floor((n + 2 * p - k) / s) + 1

def cnn_shape(w):
    h = 32
    trace = []
    for j, c in enumerate((64, 128, 256, 256, 512, 512, 512), 1):
        k, p = (2, 0) if j == 7 else (3, 1)
        h, w = out_size(h, k, p=p), out_size(w, k, p=p)
        trace.append((c, h, w))
        if j in (1, 2, 4, 6):
            sw, pw = (2, 0) if j in (1, 2) else (1, 1)
            h, w = out_size(h, 2, 2), out_size(w, 2, sw, pw)
            trace.append((c, h, w))
    return trace

def collapse(path):
    return tuple(v for i, v in enumerate(path)
                 if v != 0 and (i == 0 or v != path[i - 1]))

def logsumexp(xs):
    m = max(xs)
    return m if m == float('-inf') else m + log(sum(exp(x - m) for x in xs))

def log_softmax(rows):
    return [[v - logsumexp(row) for v in row] for row in rows]

def ctc_forward(lp, y):
    ext = [0]
    for k in y:
        ext.extend((k, 0))
    a = [float('-inf')] * len(ext)
    a[0] = lp[0][0]
    if y:
        a[1] = lp[0][y[0]]
    for row in lp[1:]:
        nxt = []
        for s, k in enumerate(ext):
            terms = [a[s]]
            if s > 0:
                terms.append(a[s - 1])
            if s > 1 and k != 0 and k != ext[s - 2]:
                terms.append(a[s - 2])
            nxt.append(row[k] + logsumexp(terms))
        a = nxt
    return logsumexp(a[-2:]) if y else a[0]

def enumerate_ctc(lp, y):
    mass = 0.0
    gamma = [[0.0] * len(lp[0]) for _ in lp]
    for path in product(range(len(lp[0])), repeat=len(lp)):
        if collapse(path) != y:
            continue
        p = exp(sum(lp[t][k] for t, k in enumerate(path)))
        mass += p
        for t, k in enumerate(path):
            gamma[t][k] += p
    if mass:
        gamma = [[v / mass for v in row] for row in gamma]
    return mass, gamma

for w in range(4, 205):
    assert cnn_shape(w)[-1] == (512, 1, w // 4 + 1)
assert cnn_shape(100) == [
    (64,32,100), (64,16,50), (128,16,50), (128,8,25),
    (256,8,25), (256,8,25), (256,4,26), (512,4,26),
    (512,4,26), (512,2,27), (512,1,26)]
assert [(v - 128) / 128 for v in (0,128,255)] == [-1,0,127/128]
assert collapse((1,1,0,1)) == (1,1)
assert collapse((0,0,0)) == ()
n, c, h, t = 2, 3, 2, 4
f = {(b,k,r,j): float(((b*c+k)*h+r)*t+j)
     for b,k,r,j in product(range(n),range(c),range(h),range(t))}
x = {(j,b,k*h+r): v for (b,k,r,j),v in f.items()}
back = {(b,k,r,j): x[j,b,k*h+r] for b,k,r,j in f}
assert back == f
assert sum(v*v for v in f.values()) == sum(v*v for v in x.values())
a = [[0.2,-0.1,0.7], [-0.3,0.5,0.1], [0.6,-0.2,0.4]]
lp = log_softmax(a)
for y in ((), (1,), (1,1), (1,2), (1,1,1)):
    mass, _ = enumerate_ctc(lp, y)
    z = ctc_forward(lp, y)
    if mass == 0:
        assert z == float('-inf')
    else:
        assert abs(z - log(mass)) < 1e-12
        assert isfinite(-z)
y = (1,1)
mass, gamma = enumerate_ctc(lp, y)
max_error = 0.0
for t, k in product(range(3), range(3)):
    plus, minus = [r[:] for r in a], [r[:] for r in a]
    plus[t][k] += 1e-6
    minus[t][k] -= 1e-6
    numeric = (-ctc_forward(log_softmax(plus), y)
               + ctc_forward(log_softmax(minus), y)) / 2e-6
    analytic = exp(lp[t][k]) - gamma[t][k]
    max_error = max(max_error, abs(numeric - analytic))
    assert abs(sum(gamma[t]) - 1) < 1e-12
assert max_error < 1e-7
assert ctc_forward(lp[:2], (1,1)) == float('-inf')
tail_a = lp + [[-0.01,-5.0,-6.0]]
tail_b = lp + [[-7.0,-0.02,-4.0]]
assert ctc_forward(tail_a[:3], y) == ctc_forward(tail_b[:3], y)
print('PASS: widths=201, trace=11, CTC_targets=5, gradient_coordinates=9')
print('max_gradient_error=', max_error)
```

最后一项只说明已经给定log_probs后，截取前T_i帧可隔离尾部值；**没有验证CNN／BiLSTM对图像padding不敏感**。该检查器未执行双向循环、BN、卷积数值核或自定义C++内核；shape算术正确不代表这些算子逐元素正确。

## 面试自测：贯穿 L1–L5 的五问

| 层级／问题 | 参考答案要点与成立条件 | 易错答案 | 证据入口 |
|---|---|---|---|
| L1：固定作者代码输入[N,1,32,100]，如何到字符串？ | 七卷积／四池化得到[N,512,1,26]；列序列[26,N,512]；第一双向投影到256，第二投影到K；log_probs后取有效帧、argmax、合并连续重复、删blank | “两次BiLSTM都输出512，CTC再训练逐列字符框” | L1形状表、来源4／5／8 |
| L2：为什么不把全部特征池化成一个向量？ | 全局向量丢失显式左右顺序；列时间轴支持可变长度单调对齐；上下文虽是双向，CTC路径仍按时间分解 | “CTC要求图像各列独立，一列必是一个字符” | L2、来源1／9 |
| L3：单张正常、补齐batch掉字，先改beam吗？ | 先查W_i→T_i、反向RNN真实终点、CNN边界污染及BN；用冻结BN单张基线对照；再查解码，不用T_max掩盖错误 | “只设input_lengths就完全防止padding泄漏” | L3防泄漏方案、来源10接口边界 |
| L4：为什么logits梯度为p−γ，`AA`至少几步？ | 由路径边缘后验得dL/dlogp=−γ，经log_softmax链式法则得p−γ，需Z>0；AA的U=2、R=1，至少3步，合法最短路径A,blank,A | “损失对logp和logits的梯度都相同；T≥U就够” | L4梯度、来源8／9、有限检查器 |
| L5：移植后shape全对为何仍不等价？ | 查N/T转置、门顺序i/f/o/g、方向投影及bias、BN位置、像素归一、池化padding、loss reduction；逐层数值／logits梯度与未补齐基线对照 | “权重能加载、尺寸相同就已复现原论文” | 固定源码表、版本表、来源4–10 |

## 版本区别、验证记录与未实现项

### 不能合并的版本事实

| 对象 | 已确认 | 保留差异／限制 |
|---|---|---|
| arXiv v1，2015-07-21 | 三段式、七卷积、两层256单元BiLSTM；BN在Conv5／6；训练100×32，测试宽可变 | Table1池化标记与25帧示例不能唯一推出可执行尺寸；未擅自修订原文 |
| TPAMI，2017-11 | Shi／Bai／Yao，39(11)，2298–2304，DOI 10.1109/TPAMI.2016.2646371 | 本次只读取期刊书目／摘要／参考文献，未逐式比对期刊全文；不能断言解决了v1歧义，DOI的2016不等于2016卷期 |
| 作者固定Torch7快照，2018-05-04 | BN在Conv3／5／7；后两pool为2×2核、非对称stride与pad；100宽T=26；两层都有方向投影 | train／示例loader固定100宽；训练clone上限maxT=26、验证输出也固定26；不提供通用变宽batch长度mask |
| 作者README所列PyTorch port | README明确归于meijieru的移植 | 本次未核其具体提交／源码，不以其默认结构或归一反推原作者版本 |
| 本卡现代工程参考 | 逐样本CNN＋packed双向RNN、显式长度、sum/N目标 | 是待实现设计；未训练／测试，不声称与作者数值或优化轨迹等价 |

作者LSTM的eval分支复用单元，训练分支依赖maxT clones；因此不能把“训练上限26”说成数学模型永远只能26帧，也不能仅改配置maxT便认定完整变宽训练已接通。数据loader、验证输出、目标数组、BN与长度处理都需同步核查。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)

### 有限验证记录

2026-11-05在Windows上使用仓库已有 `.venv/Scripts/python.exe`，从本文件提取Python代码块直接执行，未创建额外脚本或安装依赖。实际输出为 `PASS: widths=201, trace=11, CTC_targets=5, gradient_coordinates=9`；九个logits坐标的中心差分最大绝对误差为 **4.48546366715874e−10**，低于检查阈值1e−7。归一端点、列布局往返、重复／全blank折叠、不可能对齐及截断尾部检查均通过。

**初版历史核验记录：**当时本地文件与锚点链接共 **51处**（包含重复引用）核验通过；唯一CR01锚点、L1–L5五层、五问和两个代码围栏结构检查通过。排除本文件后的docs树按相对路径和原始内容计算SHA-256，初版写入前后均为 `ccd900eb64b1312f96a5fa0509db690190b74c7bc6b6f30a11878fe756431dc0`，当时未改其他文档。该计数与摘要不是本次三文档复核的结果；本次补上现有CTC专卡交接，不同步台账。

验证对象严格限于文档与算术检查器，不把输出写成神经网络训练通过，不把外部源码读取写成模型实现。网络shape的201个宽度覆盖4至204，11个中间shape覆盖W=100；CTC检查仅三时间步／三类别、五种目标和九个logits梯度坐标，不能推广为所有数值范围的CTC回归。

后续需要但本次**未执行**：真实灰度转换／插值对照、七卷积和两层BiLSTM前后向、GPU／混合精度稳定性、packed梯度与单样本对照、图像padding污染和BN统计消融、极长／空串／未知字符的数据集回归、权重迁移、词典／beam实现、按设备与批次隔离的CER和整串评测。

**G28承接边界：**本卡补齐具体CRNN模型的架构、宽度、循环读出及CTC训练接口说明，足以作为后续“具体方法专卡已存在”的登记依据；不自动把L3–L5全部升级，也不销G27或修改任何台账。**本仓库未实现**CRNN识别器、训练器、数据集、权重、CTC专用内核、文本检测／整形、拒识校准和现场追溯链。

## 来源与逐项读取范围

来源编号仅在本文件内有效，均于 **2026-11-05** 联网核验。原论文、期刊记录与作者实现是同一研究谱系，不充当三份独立效果验证。原机制默认 `[SINGLE-SOURCE]`；明确一致的版本信息可交叉核对，但BN／池化／帧数差异按 `[DISPUTED：版本口径不一致]` 原样保留；本卡shape、梯度和工程建议另标为直接推导或设计，不冒称原论文新增结论。

- <a id="source-1"></a>**[cite:1] Shi、Bai、Yao，原预印本。** [arXiv记录](https://arxiv.org/abs/1507.05717)、[v1正文](https://arxiv.org/html/1507.05717v1)。读取摘要／版本、§2.1–2.4、Table1、§3.2及部分实验讨论；通用抓取在转录节截断后，另直接提取HTML补读关键节和表格，不称逐页核验全部实验。支持2015-07-21、CNN列序列、BiLSTM、CTC、原训练与测试协议；不据论文benchmark承诺本仓库性能。
- <a id="source-2"></a>**[cite:2] TPAMI期刊记录。** [ACM收录页](https://dl.acm.org/doi/10.1109/TPAMI.2016.2646371)。书目、摘要和参考文献；页面Published为2017-11-01，卷期为2017-11。不是期刊全文逐式比对；与v1全部技术差异仍未核验。
- <a id="source-3"></a>**[cite:3] 作者仓库身份与固定版本。** [提交](https://github.com/bgshih/crnn/commit/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb)、[固定README](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/README.md)。读取GitHub提交API、完整树和README，确认作者、提交时间、Torch7／CUDA环境及他人PyTorch port归属；未下载权重、编译或执行安装脚本。
- <a id="source-4"></a>**[cite:4] 作者网络配置。** [config.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/model/crnn_demo/config.lua)。读取完整raw源码；支持七卷积参数、BN位置、池化、归一化、两层投影、K=37、maxT及训练配置。源码行末shape注释不替代尺寸推导。
- <a id="source-5"></a>**[cite:5] 作者循环与方向合并。** [LstmLayer.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/LstmLayer.lua)、[BiRnnJoin.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/BiRnnJoin.lua)。完整raw静态读取；核对门顺序、零初态、双向时间索引、clones及两线性投影相加，不是CUDA数值复现。
- <a id="source-6"></a>**[cite:6] 作者数据／字表／示例预处理。** [DatasetLmdb.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/DatasetLmdb.lua)、[utilities.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/utilities.lua)。完整raw静态读取；核对固定100×32、byte灰度、36类与大小写映射；未执行图像库或LMDB。
- <a id="source-7"></a>**[cite:7] 作者训练。** [training.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/training.lua)。完整raw静态读取；核对batch均值、验证输出固定maxT、eval切换和停止条件；不保证原脚本及旧依赖可直接运行。
- <a id="source-8"></a>**[cite:8] 作者CTC和贪心内核。** [ctc.cpp](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/cpp/ctc.cpp)。完整raw静态读取；核对[N,T,K]、blank=0、前后向约定、log_probs梯度和折叠顺序；未编译，未开展全边界正确性证明。
- <a id="source-9"></a>**[cite:9] Graves等，CTC，ICML2006。** [作者PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)。全文入口仅读到摘要、§2、§3、§4及4.1开头，返回截断；用于原CTC角色、路径概率及best-path近似，后验接口另对照作者代码，不冒称全文已通读或语音实验支持工业OCR。
- <a id="source-10"></a>**[cite:10] PyTorch CTCLoss官方文档。** [API页](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)。读取参数、shape、reduction、zero_infinity及后端说明；页面源码链接指向v2.11.0，这是本次网页状态，不是本地已安装版本。只支持现代接口对照，本卡没有安装／运行torch。
- <a id="source-11"></a>**[cite:11] 作者解码包装。** [inference.lua](https://github.com/bgshih/crnn/blob/f5d41e3355fc4447f77cd6d5e5777b3a160c93cb/src/inference.lua)。完整raw静态读取；词典模式限制单样本并对传入词典重复输入评分，目标数组宽度写死30；不等于原文BK-tree或已部署无限词典支持。
