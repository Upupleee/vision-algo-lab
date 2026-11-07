# SG01 迭代消息传递场景图：对象与有向关系联合推断

[学习规范](../algorithm_learning_standard.md) · [分类装配总览](../classification_assembly_evolution.md) · [覆盖任务 G25／07-07](../evolution_coverage.md) · [逐卡索引](README.md) · [A01 装配规则](remaining_domains.md#a01-数量与相对几何装配规则)

<a id="sg01"></a>
<a id="g25"></a>

## 任务、演进与覆盖状态

**任务：**给定图像与对象候选框，用对象区域和对象对联合区域的视觉特征初始化两类 GRU，迭代交换上下文，预测对象类别、类别相关框偏移以及有向的“主体—谓词—客体”。训练需要对象框／类别和关系标签；这不是仅凭类别标签学习关系，也不是装配合格性分类器。[cite:1](#source-1)[cite:2](#source-2)

**演进定位：**独立对象识别无法区分“人喂马”和“人站在马旁边”，独立联合框分类还可能混淆主客体 → Xu 等把对象和关系都设为有隐状态的预测实体 → 用两套共享 GRU 与四种角色相关门控聚合传递上下文 → 增加关系标注、成对区域计算及循环展开成本 → 候选漏检、稀疏／长尾标注、错误上下文和物理不可观测性仍在。该方法与显式几何规则可组合，不替代规则、检测前端或工艺传感器。[cite:1](#source-1)[cite:2](#source-2)

版本采用 **Danfei Xu、Yuke Zhu、Christopher B. Choy、Li Fei-Fei，CVPR 2017，Scene Graph Generation by Iterative Message Passing**；方法细节对照 arXiv **1701.02426v2，2017-04-12** 的 §3–4。作者实现是 **`danfeiX/scene-graph-TF-release`**，固定提交 **`4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4`**，提交时间 **2018-10-01 03:25:44 UTC**，消息 `Update download link`。这是读取到的发布仓库快照，不声称就是论文训练时的提交；2018 是代码提交年，不是方法年。核验日期 **2026-11-05**。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

本文件只计 **SG01 一张卡**，承接 **G25／07-07 E07**；对象 GRU、关系 GRU、门控、损失和评测均为卡内组件，不另计卡。

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：原文与固定作者源码静态对照，训练、推理、停止与三种评测设置已展开 |
| L2 机制 | 已核验：对象／关系双向交互、主客体角色、门控聚合及同步／顺序区别 |
| L3 工程权衡 | 部分：参数尺度、诊断顺序与失效方案齐备，无真实数据定参 |
| L4 数学推导 | 部分：消息、GRU、条件读出、损失梯度、复杂度与退化已展开；无完整概率推断等价性或收敛证明 |
| L5 实现 | 部分：固定源码映射、张量契约、参考伪代码、边界与测试设计；未实现 |
| 面试自测 | 答 5：逐层问题、答案、成立条件、易错点和证据 |
| 实现与验证 | **本仓库未实现、未训练、未加载权重、未运行 GRU／ROI pooling／场景图推理，无本仓库基准或装配验收结果** |

**证据边界：**原文、同文转换页及作者实现属于同一研究家族，核心机制标记 **[SINGLE-SOURCE，原文＋固定实现核对]**，不冒称三个独立复现。公式代数与工程反例是解释，不是实验；不转录论文成绩当本仓库成绩。阅读本地规范、总览、任务表不等于修改或同步这些台账。

## L1 步骤：视觉输入到有向三元组

### 输入、输出与监督契约

设单图候选集合为 \(B=\{b_i\}_{i=0}^{N-1}\)，有向边表 \(E\subseteq\{(i,j):i\ne j\}\)，边数 \(M\)。对象词表 \(C\) 含背景，谓词词表 \(R\) 含 none／背景。论文 VG 设置选 150 个对象类、50 个谓词；采用这套词表且加索引 0 时输出维数分别为 151、51，不能把这些数字写成任意数据集的固定结构。[cite:2](#source-2)[cite:4](#source-4)

| 输入／输出 | 内容与约束 | 不应混同 |
|---|---|---|
| 图像 \(I\) 与候选框 \(B\) | SGDet 用检测候选；PredCls／SGCls 用真值框 | 作者发布包不含 RPN 实现，使用预提取 proposal 数据库 |
| 对象视觉 \(f_i^v\) | 对 \(b_i\) 做 ROI pooling，再经对象 FC 分支 | 对象类别标签不是视觉特征；不是词向量输入 |
| 关系视觉 \(f_{ij}^e\) | 对两框最小轴对齐包围框 \(b_i\cup b_j\) 做 ROI pooling，再经关系 FC 分支 | 不是两框交集、掩膜并集或两个对象特征简单拼接 |
| 监督 \(y_i,t_i,y_{ij}\) | 对象类别、相对候选的框偏移、有向谓词 | 未标注对不保证物理上“无关系”，不等于工业未知状态 |
| 对象读出 | 每框类别分布与 \(4\lvert C\rvert\) 框偏移 | 关系网络不从无候选处生成全新节点 |
| 关系读出 | 每条有向边的 \(\lvert R\rvert\) 分布 | \((i,j)\) 和 \((j,i)\) 是两条边，不强制互为逆谓词 |

论文描述图像到图的整体模型；发布实现的实际入口是“图像＋候选图”。二者应分层记录，不把外置 RPN 说成已包含在作者包或本仓库里。[cite:1](#source-1)[cite:3](#source-3)

### 视觉初始化与循环

1. **准备候选和标签。**训练将候选与真值对象对应，构建带标签的有向对象对并采样未标注对；测试枚举所选对象间全部 \(N(N-1)\) 条边。ROI 与边索引必须在同一图像内。
2. **提取视觉。**MS COCO 预训练 VGG-16 卷积输出共享给两条区域分支；作者代码在 `conv_out` 截断梯度。对象框和联合框分别做 **7×7 ROI pooling、空间尺度 1/16**，各自经 4096、4096 的 FC／dropout。两支 FC 不是同一套变量。[cite:2](#source-2)[cite:4](#source-4)
3. **初始化两类状态。**对象 unary 为线性 4096→512，边 unary 为带 ReLU 的 4096→512。零隐状态分别接收一次 unary，得到 \(H^v_0,H^e_0\)。全部对象共享一个 GRU 的参数，全部边共享另一个 GRU 的参数；不是每个对象有独立参数，也不是对象与关系共用一套 GRU。[cite:4](#source-4)
4. **迭代消息。**分别计算主／客对象到边、出／入边到对象的 sigmoid 加权和，再送入相应 GRU；原文与代码的执行时序见 L2。后续输入是消息，不是每轮重新提 ROI 或拼入原始视觉特征。
5. **读出。**最终对象状态经类别 softmax 和框回归头，最终边状态经谓词 softmax。固定代码每轮建共享预测头，但训练 `losses()` 只取末轮无后缀输出；没有逐轮辅助损失相加。[cite:4](#source-4)[cite:5](#source-5)
6. **停止。**推理按固定 \(T\) 次 refinement 结束，不以状态变化小为收敛停止条件。原文 Fig.4 的实验在两次消息迭代处达到峰值，作者 README 的最终模型也用 \(T=2\)；不是“两次对一切图都最优”的定理。训练的优化步数与这个 \(T\) 是不同参数。[cite:2](#source-2)[cite:3](#source-3)

**走一遍具体语义输入：**同一图像中候选 0 为人、1 为马，联合框在交换顺序后相同，但边表同时保留 `(0,1)`、`(1,0)`。初始边视觉可能相同；进入有角色的对象→边聚合后，两条边可以得到不同状态和谓词。一张图最终可得到“人—喂—马”，而不是把联合区域分类结果无方向地贴给两条边。这里只是流程例子，没有声称运行该图或取得预测分数。

### 训练目标与采样

原文 §3.4 给出对象交叉熵、谓词交叉熵和框偏移 **L1**，并冻结全部卷积层、微调 FC 与 GRU。固定代码同样使用绝对值 L1，**不是因为“类似 Faster R-CNN”就自动写成 Smooth L1**。[cite:2](#source-2)[cite:5](#source-5)

固定实现先扩展真值关系到匹配的前景 proposal 对，对相同有向对象对去重，再按前景节点预算组织关系图，补采前景对象之间未标注的边，最后补背景框。`sparse_graph.yml` 的 `BATCH_SIZE=256` 配合 `IMS_PER_BATCH=2` 得每图 128 个 ROI 预算，`FG_FRACTION=0.5` 给每图 64 个前景 ROI 目标，`NUM_NEG_RELS=128` 是额外负／未标注边预算，**不是每图边总数固定为 128**。候选不足时不承诺凑足；同对多谓词在这里不是多标签学习。[cite:7](#source-7)[cite:9](#source-9)

关系损失调用 `ignore_bg=True`：\(y_{ij}=0\) 的整项 CE 乘零，随后仍对所有采样边取均值；有标签关系才有直接分类监督。未标注边仍可能参与消息和下游梯度，不能说其状态完全不学习。对象 CE 不采用这个背景屏蔽；框回归使用 `bbox_inside_weights` 选中有效类别坐标。[cite:4](#source-4)[cite:5](#source-5)

训练用末轮三项 loss 相加，经 Momentum SGD 反向传播穿过展开的全部 refinement 与视觉初始化。固定脚本预算 150000 优化步，配置初始学习率 0.001、momentum 0.9、step size 50000、gamma 0.1，训练 dropout keep probability 0.5、测试 1。脚本按优化步上限停止，无验证指标驱动的早停；多次学习率触发实际是赋值 `initial_lr * gamma`，并非每触发一次都乘当前学习率。修改为累乘调度是另一个训练协议。[cite:9](#source-9)

### 解码与 PredCls／SGCls／SGDet

| 设置 | 已给定的信息 | 需要预测的信息 | 固定实现行为 |
|---|---|---|---|
| PredCls／`pred_cls` | 真值框、评测时使用真值对象类 | 有向谓词 | 网络仍从图像和真值框前向；评测替换为真值对象类、对象分数置 1，不是向 GRU 输入真值 one-hot |
| SGCls／`sg_cls` | 真值框 | 对象类与谓词 | 不使用预测框偏移，保留真值框；对象类按预测分布 argmax |
| SGDet／`sg_det` | 仅图像，另需候选前端 | 对象框、对象类、谓词 | 从 proposal 中 NMS、截断，做类别相关框解码；对应论文名 **SGGen** |

固定测试对 SGDet proposal 使用 IoU 阈值 **0.3** 的 NMS，最多 **50** 框；这是 proposal 去重，不是三元组去重。随后最多 2450 条有向非自环边。PredCls／SGCls 使用真值框，不能机械施加同一个 50 框预算并仍称相同真值框协议。[cite:2](#source-2)[cite:8](#source-8)[cite:9](#source-9)

`sg_eval.py` 先排除谓词索引 0，为每个有向对只取一个最高分非背景谓词，再以 \(s_i s_j p_{ij}(r)\) 排序取 top-K；不枚举每对全部谓词进入 top-K。对象 argmax 没有同样显式去掉背景列；背景对象预测通常不能匹配真值三元组。SGDet 从该对象预测类别对应的四个回归数解码框，裁剪到原图。[cite:8](#source-8)[cite:10](#source-10)

论文使用 image-wise **R@50／R@100**：真值三元组需主体类别、谓词、客体类别一致，且主体框和客体框分别达到 **IoU≥0.5**。它不是联合大框 IoU，也不是只比三个词；两端位置不能交换。固定评测逐条真值查找是否至少命中一个预测，不执行全局一对一指派。原文选 Recall 是考虑 VG 关系标注稀疏，不能推出错误关系不重要、模型概率已校准或产线误放行很低。[cite:2](#source-2)[cite:8](#source-8)

**装配质量区别：**即使三个 SGG 设置的分数都好，也只说明指定对象／关系词表和标注协议上的表现；“在上面、靠近、支撑”不能证明插接深度、扭矩或锁止。质量系统还需 BOM、实例槽位绑定、物理测点／容差、可观测性、未知／复检策略和按整件统计的漏放／误拒。被遮挡部件无候选应保留 UNKNOWN，而不是让上下文“补出”合格事实。A01 是显式规则，SG01 是学习式上下文模型，两者都未在本仓库接成装配系统。

## L2 机制：方向、门控与更新时序

### 四种通道与两套参数

对象隐状态记 \(h_i\)，有向关系隐状态记 \(e_{ij}\)。消息通道组成二部结构：只在对象与其关联边之间传递，没有同轮对象→对象或边→边直接连接。边既是原场景图中的有向边，又是消息传递图中的一个有状态实体；不是只表示对象之间固定权重的邻接元素。[cite:1](#source-1)[cite:2](#source-2)

| 接收方 | 信息源与门参数 | 删除／替换后的后果 |
|---|---|---|
| 对象 \(i\) | 出边 \(e_{ij}\) 用 \(v_{out}\)，入边 \(e_{ji}\) 用 \(v_{in}\) | 混为无方向池化会丢失它作为主体／客体的角色 |
| 边 \(i\to j\) | 主体 \(h_i\) 用 \(w_s\)，客体 \(h_j\) 用 \(w_o\) | 对两端做完全对称运算，联合框又相同，容易混淆主客体 |
| 对象 GRU | 聚合消息＋该对象旧状态 | 删除记忆会失去先前视觉／上下文的持续表征 |
| 边 GRU | 聚合消息＋该有向边旧状态 | 删除关系状态，变成每轮由对象即时分类，不等于原模型 |

门是拼接后的 **一个标量 sigmoid**，再广播乘 512 维向量；不是逐通道门，也不是对邻居做 softmax。各门独立落在 `(0,1)`，总权重不必等于 1，可同时保留多个邻居。改用平均／max pooling 是论文的消融分支，不是等价优化；改为 softmax 或除以节点度数也改变模型。[cite:2](#source-2)[cite:4](#source-4)

### 同步论文解释与固定代码顺序

原文 §3.2 写后续输入来自 **previous step** 的状态，Fig.3 和式(3)(4)没有标出完整时间下标。按统一旧状态的同步解释：

\[
M^e_t=A_e(H^v_t,H^e_t),\quad M^v_t=A_v(H^v_t,H^e_t),
\qquad H^e_{t+1}=G_e(M^e_t,H^e_t),\quad H^v_{t+1}=G_v(M^v_t,H^v_t).
\]

固定 `models.py:_iterate` 的实际依赖是 **先边后对象的交替顺序更新**：

\[
H^e_{t+1}=G_e(A_e(H^v_t,H^e_t),H^e_t),\qquad
H^v_{t+1}=G_v(A_v(H^v_t,H^e_{t+1}),H^v_t).
\]

因此同一 refinement 内对象读到的是**新边**，边读到的是**旧对象**；不能因 TensorFlow 并行执行就把数据依赖说成同步。每个边阶段内全部边读同一份对象状态，每个对象阶段内全部对象读同一份新边状态，不是按对象编号逐个原地更新。卡内复现伪代码以固定源码顺序为准；同步版只作明确标识的对照，不宣称论文排除了所有其他实现调度。[cite:2](#source-2)[cite:4](#source-4)

从传播路径看，顺序版一轮允许 \(h_j^t\to e_{ij}^{t+1}\to h_i^{t+1}\)；完全同步版需要额外一步才能让相同新信息到对象。二者不能只用同一个“2 次”就比较相同感受范围。更多迭代也不保证更正确：对象误分类会影响关系，再反过来强化对象偏差；原文关于多轮下降是实验观察及噪声传播假设，不是收敛定理。[cite:2](#source-2)

## L3 工程权衡：参数、诊断与退化

| 参数／选择 | 尺度与依据 | 权衡／诊断 |
|---|---|---|
| 图像缩放 | 固定配置短边 600 像素、长边上限 1000 像素 | 细部接触可能在缩图／stride 16 后消失；先看可见性和 ROI 特征支持 |
| 候选数 \(N\) | SGDet 默认至多 50；并非物体数量先验 | 增大可补召回，但有向边数二次增长；先量候选召回再看关系分类 |
| NMS 阈值 | 0.3，无量纲 IoU | 密排件可能相互抑制；降低阈值更强抑制，不一定提高质量 |
| 隐维 \(d\) | 原文及代码为 512 | 改维数需重训、改变 GRU 及预测头权重形状；不是无损压缩 |
| refinement \(T\) | 作者最终设置 2，整数轮 | 耗时近似随 T 增长；必须同时记录同步／顺序调度和初始化是否计轮 |
| 图采样与正边比例 | 前景节点、未标注边各有预算 | 背景 CE 屏蔽后仍以 M 为分母，采样比例改变关系损失有效尺度 |
| 谓词词表与单标签 | 原 VG 50 谓词，不含背景的数量 | 相近／逆向词、同对多标签需明确定义；换成多标签 BCE 不再是原实现 |
| top-K 与质量阈值 | R@K 的 K 是列表容量，不是置信阈值 | top-K 只利于排序评测；产品接受阈值与 UNKNOWN 必须另行校准 |

数值来自固定配置、脚本和原文；以下工程诊断未实际运行。[cite:2](#source-2)[cite:7](#source-7)[cite:9](#source-9)

| 症状 | 优先取得的证据 | 调整顺序／回退 |
|---|---|---|
| PredCls 尚可、SGDet 很差 | 真值框与 proposal 召回，NMS 前后框，小对象分层 | 先修候选／尺度，再对象类，最后关系；不先堆 GRU 轮数 |
| 两方向预测几乎相同 | 边表主客顺序、联合 ROI、四门参数与 checkpoint 拼接顺序 | 排索引／权重错配，再检查角色不可辨识与训练偏置 |
| 稀疏图对象收不到关系 | 有边无反向边计数、`rel_pair_mask_inds`、dummy 边 | 先核作者“必须双向存在”过滤；明确采用原行为还是修正扩展 |
| 增加未标注对后 loss 变小 | 正边数／总边数，未归一化 CE 总和，梯度范数 | 先排分母缩放假象；不能据 loss 下降称关系性能提高 |
| 迭代越多越自信却越错 | 各轮对象／谓词混淆、门饱和、候选度数 | 固定数据比较 T 和调度；保留视觉-only 基线及较短迭代回退 |
| 验证异常优秀、换批失效 | 同图／同实物泄漏，标签先验，背景与频繁组合 | 按实物／批次／产品拓扑分组；稀有反装、缺件单独留出 |
| 图分数高但不到位漏放 | 插接深度／锁止直接真值，可见性与物理容差 | 加测量或工艺传感器；拒绝把自然谓词分数当放行证据 |

建议顺序：**任务／词表与数据分组 → 可观测性／候选召回 → ROI 与方向索引 → 时序及梯度 → 采样／损失尺度 → T 与容量 → 解码／阈值 → 性能优化**。无通用的召回、校准误差、收敛率或装配漏放上界。

## L4 数学：聚合、状态、监督与解码

### 条件读出不是精确概率推断保证

原文式(1)(2)以联合概率及 mean-field 形式动机化推断。工程上可把给定隐状态后的分类读出写成：

\[
q_\theta(Y\mid H^v_T,H^e_T)
=\prod_i q_v(y_i\mid h_i^T)\prod_{(i,j)\in E}q_e(y_{ij}\mid e_{ij}^T),
\quad q_v=\operatorname{softmax}(W_vh+b_v),\quad q_e=\operatorname{softmax}(W_ee+b_e).
\]

隐状态是图像和全图上下文的确定性函数，故这种**条件读出因子化不等于对象在图像上独立**。原文式(1)按对象对乘积书写联合项，会重复出现对象变量；此处不把那种高层写法重解释为已经证明归一化的全局 CRF。通用 GRU 更新也没有给出每一步降低明确变分自由能的证明。“mean-field inspired”不能推出精确 MAP、变分坐标下降或全局最优。[cite:1](#source-1)[cite:2](#source-2)

### 由角色邻接得到加权消息

对目标对象，把它作为主体的边与作为客体的边分组；对目标边，把主体和客体分开。原式(3)(4)可写为：

\[
\alpha^{out}_{ij}=\sigma(v_{out}^{\mathsf T}[h_i;e_{ij}]),\quad
\alpha^{in}_{ji}=\sigma(v_{in}^{\mathsf T}[h_i;e_{ji}]),
\]
\[
m_i=\sum_{j:(i,j)\in E}\alpha^{out}_{ij}e_{ij}
+\sum_{j:(j,i)\in E}\alpha^{in}_{ji}e_{ji},
\]
\[
\beta^s_{ij}=\sigma(w_s^{\mathsf T}[h_i;e_{ij}]),\quad
\beta^o_{ij}=\sigma(w_o^{\mathsf T}[h_j;e_{ij}]),\qquad
m_{ij}=\beta^s_{ij}h_i+\beta^o_{ij}h_j.
\]

四个向量属于 \(\mathbb R^{2d}\)，门是标量，消息属于 \(\mathbb R^d\)。这是“按边角色拆分 → 用发送／接收状态算兼容分数 → sigmoid → 加权求和”的构造，不是从某个物理能量唯一推出来的式子。代码 FC 另有可学习偏置；对象聚合拼接实际是 `[edge, vertex]`，与上式 `[vertex, edge]` 相反。若同时交换权重块，函数可等价；只交换输入不转换 checkpoint 不等价。[cite:2](#source-2)[cite:4](#source-4)[cite:10](#source-10)

对单项 \(m=\alpha e\)、\(\alpha=\sigma(a)\)、\(a=v^\mathsf T[h;e]\)，链式法则给出：

\[
\frac{\partial m}{\partial a}=\alpha(1-\alpha)e,\qquad
\frac{\partial m}{\partial e}=\alpha I+e\,[\alpha(1-\alpha)v_e]^\mathsf T.
\]

梯度既可直接经过被加权状态，也可经过门的兼容计算；门饱和时第二条路径变小，不代表整项梯度必为零。由于不归一化，\(\|m_i\|\le\sum_{e\text{ incident to }i}\|e\|\)；若每维隐状态在 `[-1,1]` 内，则每维消息绝对值至多关联有向边数。这只是条件性幅值界，说明高度节点的消息尺度可能更大，**不是预测误差界或迭代收敛界**。

### GRU 递推与时间共享

统一采用更新门 \(z\) 表示“保留旧状态”的一种 GRU 约定：

\[
r=\sigma(W_rx+U_rh+b_r),\quad z=\sigma(W_zx+U_zh+b_z),
\]
\[
\tilde h=\tanh(W_hx+U_h(r\odot h)+b_h),\qquad
G(x,h)=z\odot h+(1-z)\odot\tilde h.
\]

有些库把 z 定义为写入比例，或把 reset 放在矩阵乘法之后；移植前必须核旧 TensorFlow GRUCell 的门顺序、偏置和 reset 位置，不能仅按名称搬参数。作者源码只调用 TensorFlow r0.11 GRUCell，本卡未逐行取得该依赖内核，因此上式是明确约定的解释公式，**不是已完成旧 TF 与现代框架逐元素对齐**。[cite:3](#source-3)[cite:4](#source-4)

视觉初始化为 \(H^v_0=G_v(U_v,0),H^e_0=G_e(U_e,0)\)，之后沿 L2 的固定顺序迭代。每种 GRU 在时间上共享权重。若初态在 `[-1,1]`，tanh 候选和 sigmoid 凸混合使后续状态仍在此区间；但导数含权重矩阵及门导数，无法由状态有界推出映射收缩或梯度稳定。

### 末轮监督与梯度

对固定采样 batch 的 \(N>0,M>0\)，记末轮对象 logits 为 \(a_i\)、关系 logits 为 \(u_k\)，关系标签为 \(r_k\)，类别相关框输出为 \(D\in\mathbb R^{N\times4|C|}\)，目标为 \(T_b\)，有效权重为 \(W_b\)。固定源码的无额外系数目标为：

\[
L_{obj}=\frac1N\sum_i\left[\operatorname{LSE}(a_i)-a_{i,y_i}\right],
\quad L_{rel}=\frac1M\sum_{k=1}^{M}\mathbf1[r_k>0]
\left[\operatorname{LSE}(u_k)-u_{k,r_k}\right],
\]
\[
L_{box}=\frac1N\sum_{i,c}W_{b,ic}|D_{ic}-T_{b,ic}|,\qquad
L=L_{obj}+L_{rel}+L_{box}.
\]

这是 categorical 负对数似然与受掩膜绝对偏移之和；框项也可由独立 Laplace 残差假设解释，但不表示真实框误差已验证符合 Laplace。`LSE` 用减最大值的 log-sum-exp 实现，训练不先 argmax，也不把 softmax 概率当 logits 再喂 CE。[cite:5](#source-5)

\[
\frac{\partial L_{rel}}{\partial u_{kc}}
=\frac{\mathbf1[r_k>0]}M\left(p_{kc}-\mathbf1[c=r_k]\right),\qquad
\frac{\partial L_{box}}{\partial D_{ic}}
\in\frac{W_{b,ic}}N\,\partial|D_{ic}-T_{b,ic}|.
\]

非零残差的 L1 导数为符号；零残差处为次梯度集合 `[-1,1]`，不能写成二阶光滑损失。关系正标签仍通过 softmax 分母影响背景 logit；“屏蔽背景样本”不等于“删掉背景输出列”。若正边数为 \(M_+\)，则 \(L_{rel}=(M_+/M)\overline{CE}_+\)；增加被屏蔽的边会改变比例，因而不能与除以 \(M_+\) 的实现直接比较 loss。\(M>0,M_+=0\) 时关系项为零；\(M=0\) 的空均值没有定义，应显式防护。

反传穿过 `edge→vertex` 的顺序依赖、所有共享 GRU 和各门，最后到视觉 FC；卷积 `stop_gradient` 截断其上游。没有中途 detach，也没有把离散预测标签喂回去。末轮交叉熵下降不保证 R@K 上升，更不保证硬装配约束满足。

### 框解码、排序与复杂度

用候选中心 \((c_x,c_y)\)、宽高 \((w,h)\) 定义目标：

\[
t_x=(c_x^*-c_x)/w,\quad t_y=(c_y^*-c_y)/h,
\quad t_w=\log(w^*/w),\quad t_h=\log(h^*/h).
\]

解码为 \(\hat c_x=c_x+w\hat t_x,\hat w=w\exp(\hat t_w)\)，y／h 同理。有效宽高必须为正，指数可能溢出。固定作者旧实现宽高使用 `x2-x1+1`，右端点解码为 `cx+0.5*w` 而不减 1；因此零偏移也不能假定严格端点往返不变。该历史数值约定应保留或显式修正并重测，不悄悄混入半开区间框协议。[cite:10](#source-10)

对每对最高非背景谓词 \(\hat r_{ij}\)，排序分数为 \(S_{ij}=s_i s_jp_{ij}(\hat r_{ij})\)。正分数时 \(\log S_{ij}=\log s_i+\log s_j+\log p_{ij}\)，保持顺序，便于避免连乘下溢；零分需按 `-inf` 处理，不能随意加 epsilon 后声称严格等价。PredCls 的 \(s_i=s_j=1\)，但它不免除图像关系推断。[cite:8](#source-8)

不计视觉 CNN／FC，门计算与聚合约 \(O(Md)\)，两类 dense GRU 每轮约 \(O((N+M)d^2)\)，展开为 \(O(T(N+M)d^2)\)；状态存储推理约 \(O((N+M)d)\)，反传保存约随 T 增长。完全有向图 \(M=N(N-1)\)，故二部消息结构避免了边实体间再两两全连接，**没有消除对象候选的二次边数**。作者 `create_graph_data` 还建立 `N×M` 布尔矩阵，整套实现的峰值内存不能只报状态张量。[cite:4](#source-4)[cite:6](#source-6)

可辨识性退化包括：两端视觉／位置完全对称且所有角色参数也对称、真实状态在 RGB 中不可见、候选已漏掉关键对象、相同输入被赋冲突单标签。上下文不能凭空解决这些信息缺失；两条反向边联合框相同本身不是代码错误，丢失其角色才是。

## L5 实现：固定源码、张量与边界

### 张量布局与源码入口

| 张量／组件 | 布局与类型 | 固定源码入口 |
|---|---|---|
| 图像 batch | `[B,H,W,3]` float32，NHWC；BGR 减 `[102.9801,115.9465,122.7717]` | `fast_rcnn/test.py:_get_image_blob`，`config.py` |
| 对象 ROI／关系 ROI | `[N,5]`／`[M,5]` float32，首列图像索引，后四列缩放图 `x1,y1,x2,y2` | `compute_rel_rois`；ROI pooling 另用 `1/16` |
| 有向边表 | `[M,2]` int32，第一列主体、第二列客体，全 batch 节点索引 | `minibatch.py` 对后续图端点加 `box_idx_offset` |
| 对象／关系状态 | `[N,512]`／`[M,512]` float32 | `models.py:_cells/_iterate` |
| 两端 gather／门 | 两端各 `[M,512]`；拼接 `[M,1024]`；门 `[M,1]` | `_compute_edge_context_soft` |
| 对象聚合 | 出／入边各 `[Q,512]`，segment id `[Q]`，结果 `[N,512]` | `_compute_vert_context_soft`，`create_graph_data` |
| logits／框 | `[N,C_count]`、`[M,R_count]`、`[N,4*C_count]`，类别数含背景 | `_cls_pred/_rel_pred/_bbox_pred` |
| 标签／框权重 | 类别 `[N]`、谓词 `[M]` int32；框目标／权重 `[N,4*C_count]` | `_gather_samples`、`losses.py` |

布局、常数与入口均由固定文件静态读取；未执行这些算子。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)[cite:9](#source-9)

### 参考伪代码：保持作者的两阶段更新

下面是**不可直接运行的结构伪代码**，不是新增本地网络。进入 ROI pooling 前，`B` 已按图像预处理的同一比例缩放并附带 batch id；解码另保留原图框与缩放记录。`E` 要求无重复有向对、无自环、无跨图边；`reverse_mask` 表示该边的反向边也在 E 中，以匹配固定作者 soft 对象聚合的过滤。`scatter_sum(..., size=N)` 是本卡更显式的零填充表达，不声称作者原函数支持任意空图。

```text
require N >= 2 and M > 0 and T >= 1
require valid_boxes_and_same_image_edges(B, E)
F = stop_gradient(VGG16_conv(preprocess_bgr(I)))
V = object_fc(roi_pool(F, B, 7, 7, spatial_scale=1/16))
U = relation_fc(roi_pool(F, union_boxes(B, E), 7, 7, spatial_scale=1/16))
Hv = GRUv(linear(V, 512), zeros(N, 512))
He = GRUe(relu(linear(U, 512)), zeros(M, 512))
s, o = E[:, 0], E[:, 1]
reverse_mask = reverse_edge_exists(E)
for t in range(T):
    Hv_old, He_old = Hv, He
    bs = sigmoid(fc_sub(concat(Hv_old[s], He_old)))
    bo = sigmoid(fc_obj(concat(Hv_old[o], He_old)))
    Me = bs * Hv_old[s] + bo * Hv_old[o]
    He_new = GRUe(Me, He_old)
    ao = sigmoid(fc_out(concat(He_new, Hv_old[s])))
    ai = sigmoid(fc_in(concat(He_new, Hv_old[o])))
    Mv = scatter_sum(reverse_mask * ao * He_new, s, size=N)
    Mv = Mv + scatter_sum(reverse_mask * ai * He_new, o, size=N)
    Hv_new = GRUv(Mv, Hv_old)
    Hv, He = Hv_new, He_new
obj_logits, box_deltas = object_heads(Hv)
rel_logits = relation_head(He)
if training:
    loss = object_ce(obj_logits, y_obj)
    loss += mean((y_rel > 0) * ce_rows(rel_logits, y_rel))
    loss += mean(sum(box_weights * abs(box_deltas - box_targets), axis=1))
    backward_and_step(loss)
else:
    return decode_by_protocol(obj_logits, box_deltas, rel_logits, B, E, mode)
```

`reverse_mask` 在广播前应为 `[M,1]`。若要实现原式针对任意有向稀疏图的全部关联边求和，可去掉该过滤；那是明确的**修正／扩展版**，不是忠实执行固定代码。若要实现同步对照版，对象门和对象消息全部使用 `He_old`；只改门却保留新边 value 会生成第三种混合算法。

### 原文、作者代码和未实现差异

| 事项 | 原文／抽象机制 | 固定源码事实及影响 |
|---|---|---|
| 时序 | previous-step 文字及无时间下标的式(3)(4) | `_iterate` L208–216 先更新边再更新对象；不同于全部旧状态同步 |
| 对象关联边 | 式(3)分别求和所有出边／入边 | `data_utils.py` L31 用 `out_inds>=0 and in_inds>=0`；只有双向都存在才进入 soft 对象聚合。单向稀疏边仍可更新自身，但不回流对象 |
| 门与拼接 | 原式无显式偏置，对象门 `[h;e]` | FC 含偏置；对象门 `[e;h]`、边门 `[h;e]`；移植权重需要区分 |
| 初始视觉与计轮 | 首次视觉输入，后续消息；无消息 baseline 直接视觉分类 | 初始化单独调用 GRU，再做 `n_iter` 次 refinement；每类 GRU 共调用 `T+1` 次。`n_iter=0` 不创建最终头，不是可用 baseline；应选独立 `vrd` |
| 损失 | CE 对象／谓词＋L1 框；未在 §3.4 展开背景 mask | 谓词背景项屏蔽，均值分母仍为全部 M；仅末轮 loss，不是全轮深监督 |
| 图采样 | 原文训练采 128 框，先标注边、不足 128 补未标注边 | 发布采样器按前景节点预算截取关系，再额外采最多 128 未标注边；不能据原文概述推代码边数恒定 |
| 前端 | 全图模型以 RPN proposal 开始 | 发布包没有 RPN，只消费外部 proposal 数据库；没有本仓库前端接入 |
| 评测 | R@K，SGGen 对象与关系联合任务 | 发布名 `sg_det`；每对取一个非背景谓词，三分数连乘；不是任意全谓词 top-K 或质量阈值 |
| 旧数值与依赖 | 高层公式不规定所有离散细节 | 框 `+1` 及右端点约定、评测框转 int32、旧 TF ROI pooling 自定义 GPU op 均需保留版本或显式改版 |

这些差异有源码入口而非二手复述，不推测作者当年未发布训练分支的行为。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)[cite:10](#source-10)

### 边界、测试设计与优化约束

| 必测边界 | 可检验期望 | 当前状态 |
|---|---|---|
| 空框／单框／空边 | 返回明确空图或不可评估状态；不调用 `vecs[0]` 或空 CE | 未执行；作者测试跳过少于两框，`pad_and_gather` 不能因此视为已支持空边 |
| 无关系真值 | 独立记录不适用／跳过数量，不能悄悄计作 R@K=1 | 未执行；固定 `sg_eval.py` 返回 `(None,None)` |
| 单向边、双向边、孤立节点 | 原式收所有关联边；固定 soft 版只收双向对，孤立节点消息为零 | 未执行；零消息不等于 GRU 状态原样不变，偏置与旧状态仍影响更新 |
| dummy 边与 segment | dummy 索引为 M、追加全零行；id 按节点排序并覆盖每节点 | 未执行；不要让 -1 索引误取最后一条真实边 |
| 对象重编号与 batch 拼接 | 同步重排框、边、标签后输出仅置换；无跨图信息串扰 | 未执行；重复有向对需拒绝或定义合并，否则映射覆盖 |
| 时序对照 | 固定版对象只读新边，边只读旧对象；同步版双方只读旧状态 | 未执行 GRU 数值对照；按编号原地更新不能通过置换等变测试 |
| 全背景／混合谓词 | M>0 全背景直接关系 loss=0；混合时分母仍是 M | 未执行框架梯度；背景边可能经消息路径间接受监督 |
| 框与通道 | BGR／均值、缩放因子、ROI batch id、`+1` 宽高、原图裁剪一致 | 未执行图像测试；评测 `_triplet` 将框写入 int32，浮点现代版并非逐值一致 |
| 数值与训练模式 | 拒绝 NaN／Inf、非法类别／框；稳定 CE；训练／测试 dropout 分离 | 防护设计；不能称作者脚本已覆盖全部错误输入 |
| 评测语义 | 两端 IoU 各过阈值；主客反向不误配；K 大于 M 截断 | 未执行正式基准；零候选与空真值跳过需在新系统审计 |

优化先剖析视觉 FC／ROI、图构造、GRU 与排序占比。可缓存同图反向边共用的联合 ROI 视觉（测试无 dropout 随机性时更易等价），但**不能合并两条有向边状态**。训练直接共用 dropout 后特征会改变随机噪声相关性，需另验。用稀疏 edge-list／scatter 消除 `N×M` 布尔表可降内存，但必须保留双向过滤／dummy 语义，并考虑浮点求和次序；裁边、减 N、改归一化或同步调度都是模型／候选协议变化，不是无损加速。旧 GPU ROI pooling、TF r0.11 到现代框架的数值对齐仍未完成。

## 五层面试追问链

### Q1 · L1：给图像和对象框，怎样生成图，三种评测究竟给了什么？

**答题要点：**对象 ROI 与联合 ROI 分别提视觉，两类零状态 GRU 吃 unary，固定 T 轮消息后读出类别／框／谓词，再按任务解码。PredCls 给真值框且评测用真值类；SGCls 只给真值框；SGDet／论文 SGGen 还需候选定位和框回归。固定实现 PredCls 不把真值 one-hot 送入 GRU。**成立条件：**词表和候选／边索引合法，关系是有向单标签协议。**易错：**把 PredCls 当完整检测，或把高 R@100 当装配合格率。**证据：**L1；原文 §3、§4.1；`test.py` 和 `sg_eval.py`。[cite:2](#source-2)[cite:8](#source-8)

### Q2 · L2：联合框交换顺序不变，模型如何区分方向？所有状态同步吗？

**答题要点：**保留两条有向边，以不同主体／客体门和出／入边门编码角色；sigmoid 加权求和不归一化。原文 previous-step 可作同步解释，固定作者循环实际先边后对象，新边同轮回流对象。**成立条件：**角色参数、拼接次序、候选身份都保留；完全对称输入仍可能不可辨识。**易错：**说联合框自身带方向、门是邻居 softmax，或说顺序／同步同 T 一定等价。**证据：**式(3)(4)、`models.py` L208–299。[cite:2](#source-2)[cite:4](#source-4)

### Q3 · L3：增加候选和迭代为何反而更差，应先调什么？

**答题要点：**候选数使边数二次增长，额外框／边引入噪声，高度节点聚合尺度变大，更多轮可能放大错误上下文；先查前端召回、NMS、小物体可见性和关系标注，再查索引／损失分母，最后比较 T。**成立条件：**固定划分、权重来源、调度和解码协议，按产品／批次留出。**易错：**把两轮经验说成收敛定理，或无条件降低阈值解决缺件。**证据：**原文 Fig.4、L3、复杂度推导。[cite:2](#source-2)

### Q4 · L4：推导关系 CE 的梯度，未标注边是否完全不学习？

**答题要点：**从 `LSE(u)-u_y` 得 \(\mathbf1[y>0](p-onehot(y))/M\)。标签 0 的直接 CE 为零，但其隐状态可通过消息影响其他被监督输出，所以仍有间接梯度；加背景边还改变 M 和图结构。GRU 是学习更新，不能从“mean field”直接推出能量单调下降。**成立条件：**固定采样集合且 M>0，门／GRU 不 detach。**易错：**分母写成正边数，或把背景样本屏蔽解释成删掉 softmax 背景列。**证据：**L4，`losses.py` L5–23、`models.py` L96–112。[cite:4](#source-4)[cite:5](#source-5)

### Q5 · L5：公式对了，稀疏图复现为什么仍可能不一致？

**答题要点：**固定代码 `and` 条件只让同时存在反向边的对进入对象 soft 聚合；检查 dummy=M、segment 排序、跨图端点偏移、`[edge,vertex]` 拼接和先边后对象。用单向／双向／孤立节点、对象置换、全背景 CE、零偏移框等最小契约测试，再做逐层框架对齐。**成立条件：**先决定忠实快照还是修正原式的扩展，冻结版本及坐标。**易错：**改 `and` 为任意边后称完全等价；认为 `n_iter=0` 自动是基线；认为有伪代码等于已实现。**证据：**L5 差异表，`data_utils.py` L23–45，`models.py:_iterate`，`bbox_transform.py`。[cite:4](#source-4)[cite:6](#source-6)[cite:10](#source-10)

## 来源层级、获取深度与固定入口

**P0 原始证据优先。**论文解释方法，作者代码解释这个快照的执行；同文 arXiv／ar5iv 只补解析和版本，不算独立来源。P1 同行独立复现未用于性能结论；P2／P3 博客或第三方重写未用来代替作者代码。以下获取于 2026-11-05，属于静态阅读，不是运行或下载训练资产。

- <a id="source-1"></a>**[cite:1] P0，CVPR 2017 原文。** [会议 PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Scene_Graph_Generation_CVPR_2017_paper.pdf)。取得论文正文，解析输出在 §3.2 附近截断；已核作者、问题、图结构与式(1)，不标作全文逐页比对。后续方法／实验设置由指定 v2 补齐。
- <a id="source-2"></a>**[cite:2] P0 同文版本，arXiv 1701.02426v2。** [v2 HTML](https://arxiv.org/html/1701.02426v2)、[ar5iv 转换](https://ar5iv.labs.arxiv.org/html/1701.02426v2)。直接读取 HTML 的 §3.2–3.4、§4.1.1–4.1.3，保留公式 alttext，核式(2)–(4)、视觉／GRU 512 维、采样、三任务、损失及迭代消融；并读实验 setup。未将会议 PDF 截断后的全部内容逐页与 v2 做差异证明，NYU 完整专用实验流程未复现。
- <a id="source-3"></a>**[cite:3] P0 作者发布与版本。** [固定提交](https://github.com/danfeiX/scene-graph-TF-release/commit/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4)、[README](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/README.md)。GitHub API 核实完整 SHA／提交时间，并读取发布说明：Python 2.7、TensorFlow r0.11、CUDA 与自定义 ROI pooling，外部 proposals、T=2 示例、三评测模式。未下载数据／checkpoint。最初试探的无 `-release` 仓库入口返回 404，不计作有效来源；raw 内容抓取为空后改 GitHub contents API，未用空返回推断代码。
- <a id="source-4"></a>**[cite:4] P0 作者网络源码，固定 SHA。** [models.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/networks/models.py#L154-L323)。完整文件静态读取；L34–69 视觉分支，L96–112 损失，L184–220 初始化／顺序，L242–299 四门，L313–323 末轮损失，L371–379 final 模型。
- <a id="source-5"></a>**[cite:5] P0 作者损失／工具源码。** [losses.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/networks/losses.py#L5-L32)、[net_utils.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/networks/net_utils.py#L24-L63)。完整静态读取；背景 CE mask、分母、绝对值 L1、零行 gather 与 segment 操作，不含 TF 内核运行证明。
- <a id="source-6"></a>**[cite:6] P0 图结构／联合框。** [data_utils.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/roi_data_layer/data_utils.py#L3-L78)。完整静态读取；双向 `and`、dummy 索引、节点分组、同图约束和联合框 min/max。
- <a id="source-7"></a>**[cite:7] P0 作者采样。** [minibatch.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/roi_data_layer/minibatch.py#L17-L249)、[sparse_graph.yml](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/experiments/cfgs/sparse_graph.yml)。读取采样及 batch 组织关键段和完整配置；不扩大为 VG 全部预处理／别名清洗已核。
- <a id="source-8"></a>**[cite:8] P0 作者测试／评测。** [test.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/fast_rcnn/test.py#L110-L280)、[sg_eval.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/datasets/sg_eval.py#L3-L167)。完整静态读取；候选生成接口、真值替换、逐对非背景 argmax、三分数连乘、两端 IoU、空输入跳过及 int32 框。
- <a id="source-9"></a>**[cite:9] P0 作者训练及配置。** [train.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/fast_rcnn/train.py#L111-L206)、[train.sh](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/experiments/scripts/train.sh)、[config.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/fast_rcnn/config.py)。读取训练循环、脚本及指定配置字段，确认 optimizer、步数、学习率赋值、两图 batch、尺寸、NMS 和 BGR 均值；没有执行任何脚本。
- <a id="source-10"></a>**[cite:10] P0 作者依赖封装与框几何。** [network.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/networks/network.py#L179-L233)、[bbox_transform.py](https://github.com/danfeiX/scene-graph-TF-release/blob/4c9e3c6a5cb0e6a241a92dc9b786f74e69ca35c4/lib/fast_rcnn/bbox_transform.py#L10-L75)。前者读取 ROI／FC 段，后者完整读取；确认 affine 偏置、宽高 `+1`、解码右端点和裁剪。未读取旧 TF GRU 内核、未编译 GPU 自定义 op。

## 验证范围与 G25 销项依据

**此次完成的是单卡文档承接。**核对了五层正文、五问答、原文式(3)(4)与固定源码入口，保留了顺序更新、双向配对、关系背景屏蔽、采样、计轮及解码差异；文档结构／相对链接／引用锚点作静态检查。没有把“已读代码”升级为“代码已实现”。

| G25 原缺口／验收项 | 本卡依据 | 可销与不可销 |
|---|---|---|
| 07-07 对象／关系更新 | L1 双视觉输入；L2 两类 GRU、四通道与时序；L4 完整消息及状态式 | 可承接具体方法机制缺卡，不以 A01 规则替代 |
| 关系监督与迭代 | L1 采样与末轮损失；L4 mask／分母／梯度；固定 T 与训练步数分开 | 可承接训练目标说明，不表示完成训练 |
| 五层学习与实现入口 | L1–L5、张量表、伪代码、源码差异、答 5 | L1／L2 已核验仅限静态资料；L3／L4／L5 仍部分 |
| 基准与装配边界 | 三种任务、R@K 解码及工艺／UNKNOWN 区分 | 不销装配质量验收、候选检测实现或概率校准 |
| 固定作者证据 | 2017 原文／指定 v2＋2018 固定 SHA，来源获取深度逐项列明 | 不声称论文训练快照已还原或跨框架逐元素一致 |

**建议台账口径：**G25／07-07 的“缺少具体方法单卡”可转“已有”，后续深度列 **P2、未实现**，唯一键 `iterative_scene_graph.md:SG01`，五层记 **核／核／部／部／部、答5**。历史 G25 ID 保留；本次只创建本文件，**未修改覆盖矩阵、索引、领域总览或计数**，故这是可供后续同步的销项依据，不声称那些台账已更新。

未完成项包括：作者旧 TF／CUDA 环境构建、模型权重及数据来源校验、ROI／GRU 跨框架对齐、实际前反向与有限差分、空图／稀疏图回归、VG／NYU 训练评测、工业关系标注和整件质量验收。所有性能、精度、耗时与现场可靠性保持无本仓库实测状态。
