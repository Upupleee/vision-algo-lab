# SI01 SiamFC：孪生全卷积相似性、固定模板与尺度搜索

[学习规范](../algorithm_learning_standard.md) · [tracking 总览 E07](../tracking_guidance_evolution.md#37-e07siamfc离线相似性学习与在线相关互补) · [覆盖台账 G46／12-07](../evolution_coverage.md) · [T02 MOSSE 对照](remaining_domains.md#t02-mosse-型自适应相关滤波)

<a id="si01"></a>
<a id="g46"></a>

## 任务、演进与覆盖状态

**任务与契约：**输入首帧 RGB 图像、指定对象的轴对齐框以及有序后续帧；输出每帧该对象的二维框和可供诊断的相关响应。训练另需带同一对象轨迹标注的视频对；推理不输入后续真值，不更新网络或首帧模板。这里是任意指定对象的短期单目标跟踪，不是检测所有目标、MOT 身份管理、长期重识别、三维位姿或机器人控制。[cite:1](#source-1)

**演进定位：**仅依靠当前视频在线学习时，训练样本与实时预算限制模型表达力 → 离线学习共享孪生卷积嵌入 → 一次搜索图前向同时评价密集平移候选 → 首帧模板固定，在线只做有限尺度搜索及位置／尺寸更新。代价是视频标注、权重与计算环境、域适配和精确裁剪坐标契约；相似干扰物、出搜索区、遮挡及大形变仍会失效。它与运动模型可组合，但本卡没有组合实现。[cite:1](#source-1)

唯一键 **`siamfc.md:SI01`**，仅一张卡，承接 **G46／12-07／E07**。方法是 Bertinetto、Valmadre、Henriques、Vedaldi、Torr 的 **ECCV 2016 Workshops，LNCS 9914，850–865**，不是 ECCV 主会。核对日期 **2026-11-06**；作者 MatConvNet 仓库固定为 `bertinetto/siamese-fc` 提交 **`e86eb64d6f146b51135232c1d46a29f64c63678a`（2019-05-16 UTC）**。固定提交是本次阅读快照，不冒充 2016 camera-ready 的完整运行环境。[cite:1](#source-1)[cite:3](#source-3)

| 字段 | 状态与限定范围 |
|---|---|
| L1 步骤 | 已核验：论文训练／推理主链、固定作者代码调用与状态更新 |
| L2 机制 | 已核验：共享、valid 相关、类别平衡、固定模板与窗／尺度机制 |
| L3 工程权衡 | 部分：参数单位、版本差异、诊断与验收设计；无视频定参 |
| L4 数学推导 | 已核验：限定模型下的 shape、相关双支梯度、平衡 logistic 与坐标代数；不是精度保证 |
| L5 实现 | 部分：固定源码映射、教学伪代码、有限检查；本仓库未实现 SiamFC |
| 面试自测 | 答5：逐层问题、答案要点、条件、易错答案和证据入口 |
| 验证范围 | 联网正文／源码静态核验、PowerShell 双精度有限差分与尺寸／索引检查、文档链接和单文件检查；无模型前向、训练或视频评测 |

机制证据为 **[SINGLE-SOURCE]**：作者论文、出版页、预印本和作者源码是同一来源体系，不算三份独立复现实验。正文中“原文”“固定代码”“教学推导／工程建议”分别标出。仓库 Python 文件关键词检索没有发现 SiamFC 专用实现；本卡不把已有单帧 ZNCC 或学习匹配占位接口升级为跟踪器。

## L1 步骤：训练相似性，在线固定首帧特征

### 孪生结构与输入输出 shape

以便于移植的 `NCHW` 表示，模板 `z:[N,3,127,127]`，搜索图 `x:[N,3,255,255]`。两支使用同一组卷积参数，不是两套独立训练网络；“全卷积”是对搜索平移的有效区域等变性，不是“没有池化”。以下按固定 `vid_create_net.m` 的完整 AlexNet 型分支，所有卷积／池化内部 padding 为0；不采用 small 分支或 CFNet 后来的32通道版本。[cite:4](#source-4)[cite:5](#source-5)

| 层 | kernel／stride／groups | 总输出通道 | exemplar 空间尺寸 | search 空间尺寸 | 累积 stride |
|---|---|---:|---:|---:|---:|
| 输入 | — | 3 | 127×127 | 255×255 | 1 |
| conv1 | 11×11／2／1 | 96 | 59×59 | 123×123 | 2 |
| maxpool1 | 3×3／2 | 96 | 29×29 | 61×61 | 4 |
| conv2 | 5×5／1／2 | 256 | 25×25 | 57×57 | 4 |
| maxpool2 | 3×3／2 | 256 | 12×12 | 28×28 | 8 |
| conv3 | 3×3／1／1 | 384 | 10×10 | 26×26 | 8 |
| conv4 | 3×3／1／2 | 384 | 8×8 | 24×24 | 8 |
| conv5 | 3×3／1／2 | 256 | 6×6 | 22×22 | 8 |
| cross-correlation | 6×6 空间核，汇总256通道 | 1 | 模板作为动态核 | 17×17 | 对搜索输入为8 |

**出版表格冲突不能静默合并：**Springer Table 1 的 `Chan. map` 给出 conv3／4／5 为384×256、384×192、256×192，但 `Chans.` 列显示192／192／128；固定代码创建的总输出明确为384／384／256，README 也把该版描述为256输出通道。本卡 shape 采用固定代码，不把192／128误作完整输出通道；这些分歧不构成已读作者勘误。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

原文称每个线性层后训练时加入 BN；固定构造器实际对 conv1–4 使用 `Conv→BN→ReLU`，conv5 是仅卷积，无 BN／ReLU。`make_siameseFC` 把 BN 设为 true，不能因底层 `vid_create_net` 默认 false 就宣称主训练不使用 BN。推理 `load_pretrained` 设 `net.mode='test'`，使用累积统计；否则批次组成会改变特征，首帧缓存也失去一致性。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)[cite:12](#source-12)

### 离线训练链

1. 从有身份轨迹的训练视频中抽取同一对象两帧，裁出带上下文的 exemplar 和 search；对象类别标签不作为网络分类类别。不同视频／轨迹的训练与验证集合需隔离。
2. 按各自框尺度归一裁剪，保持图像几何比例，越界部分用 RGB 均值补齐；z、x 分别过共享嵌入，保留空间特征。
3. 对两支特征做 valid cross-correlation，得每个相对平移的单通道分数图；中心附近为正位置，其余为负位置，不是要求额外抽一半异对象图像对。
4. 以半径标签和正负类别平衡的 logistic 损失监督整个响应图；梯度经相关层流向模板、搜索两支，再累加到共享参数。
5. 按预定 epoch／学习率预算进行 SGD，以独立验证选择快照；损失下降不等于全局收敛或跟踪泛化。在线过程不再运行此训练步骤。[cite:1](#source-1)[cite:6](#source-6)[cite:7](#source-7)

**论文配置与固定脚本不同：**论文正文训练为127／255，50 epochs，每轮50,000 sampled pairs，batch8，学习率从0.01几何衰减至0.00001。固定 `experiment.m` 默认 `instanceSize=255-2*8=239`，对应搜索特征20×20、响应15×15；`numPairs=5.32e4` 再按0.1验证比例划分，不能记成论文每轮50,000训练对。脚本启用平移、stretch、颜色增强，默认灰度概率0；原文 OTB 特例为25%灰度对。本卡结构主表／在线伪代码仍为127／255，不声称直接执行默认训练能逐项复刻正文。[cite:1](#source-1)[cite:6](#source-6)

固定 `rPos=16` 是归一化输入像素，除以响应 stride8 得半径2格；`rNeg=0` 不产生中性环。`frameRange=100` 在采样器中按**同一轨迹有效帧列表的索引邻域**选另一帧，排除自身；有缺帧时不等同实际视频帧差≤100，更不是100秒。增强后标签仍按中心固定，属于该代码的训练约定，不能声称精确重算了任意大位移增强的真值图。[cite:6](#source-6)[cite:7](#source-7)

### 首帧初始化与在线链

给定原图框宽高 w、h，论文上下文边距为 \(p=(w+h)/4\)，固定代码 `contextAmount=0.5` 加到每个边长的量是 \(2p=0.5(w+h)\)：

\[
s_z=\sqrt{(w+2p)(h+2p)},\qquad a_z=127/s_z,\qquad
s_x=s_z+2\frac{(255-127)/2}{a_z}=s_z\frac{255}{127}.
\]

这里 s_z、s_x 单位均为**原图像素**，实际裁出边长约 s_z 的正方形再缩成127；不是裁成 `(w+2p,h+2p)` 长方形后强行拉成方图。`s_z` 的几何均值来自面积归一，极长细目标仍可能超出方形支持。[cite:1](#source-1)[cite:8](#source-8)

首帧只计算一次 \(Z=\varphi(z_1)\)，缓存并复制到尺度批次；后续每帧：

1. 以**上一帧中心**为所有尺度裁剪中心，令候选边长为 \(s_x a_j\)，\(a_j=q^{j-j_0}\)，统一 resize 到255×255，形成 S 张图。
2. 搜索分支前向，与固定 Z 相关并经过分数 adjustment，得到 S 张17×17图；逐图 bicubic 上采样16倍到272×272。
3. 给非中心尺度峰值乘 `scalePenalty`，选最高者；尺度选择发生在减最小值、归一化和加窗**之前**。固定代码最终取的是该尺度未乘惩罚的响应图。
4. 对选中响应减最小值并除以总和，将其与归一化二维 Hann 窗做凸组合；取峰得到位移。它不是对响应图做 softmax，也不是训练损失的概率输出。
5. 按选定版本的坐标回填公式更新中心，再阻尼更新框尺寸与搜索边长；Z、网络参数、BN 统计始终不更新。
6. 首帧输出输入框，此后每帧执行一次步骤1–5，视频结束停止。原版没有可靠的长期丢失／重检测状态机；生产中发现非法数值或歧义应拒绝／报告未知，属于另加工程防护，不是原模型保证。[cite:8](#source-8)[cite:9](#source-9)

## L2 机制：每一步解决什么问题

| 设计 | 因果机制／不变量 | 删除或替换的后果与反例 |
|---|---|---|
| 共享孪生嵌入 | 两图映入同一特征空间；同一 θ 接收两支梯度 | 两套独立权重改变度量假设与参数量，不能直接加载共享快照 |
| 无内部 padding、总 stride8 | 有效支持内，输入平移8格对应输出平移1格；密集计算共享重叠特征 | 任意1像素位移未必严格等变；内部零 padding 会暴露边界／绝对位置线索 |
| 多通道空间相关 | 模板的空间排列参与内积，输出一张平移响应图 | 展平为一个向量只得一个分数；depthwise相关保留通道而本版求通道和，不是同一头 |
| 正负位置损失各半 | 密集负位置多，类别平衡防止数量压倒正位置 | 全负预测可在未平衡均值中显得损失较小；权重不是样本的真实先验概率 |
| 首帧模板固定 | 避免把错误预测／遮挡物持续写入模板，且省去在线反传 | 不能适应所有形变或长期外观改变；新增模板 EMA 就是新变体 |
| 多尺度、非中心惩罚与阻尼 | 有限离散尺度估计，惩罚抖动、低通尺寸变化 | 增尺度数增加前向成本；大阻尼滞后快速缩放；只做等比缩放不改变宽高比 |
| bicubic＋Hann窗 | 插值缓解粗栅格峰定位，窗偏好近上一中心的运动 | 插值不创造新观测；大位移目标可能输给中心干扰，窗不是 Kalman 后验 |

前三行的等变性要求支持有效、采样一致、推理 BN 固定；不对旋转、遮挡和尺度天然等变。**边界裁剪补均值**与**卷积网络内部 padding=0**不是矛盾：前者构造图像输入，后者定义嵌入算子，图像外部的人造内容仍会影响边缘响应。[cite:1](#source-1)[cite:4](#source-4)[cite:8](#source-8)[cite:10](#source-10)

与 T02 的界线：MOSSE 在线解／更新最小输出平方误差滤波器，常涉及频域循环相关与共轭；SiamFC 离线训练嵌入，在线对固定首帧特征作**valid 空间交叉相关**，没有在线岭回归或 MOSSE 更新分子分母。与 SiamRPN 的界线：本卡没有 anchors、前景／背景分类分支、边框回归头或 proposal 解码；框宽高只由尺度金字塔的等比阻尼更新。上述是方法范围界定，不新增另外两张算法卡。[cite:1](#source-1)[cite:8](#source-8)

## L3 工程权衡：版本、量纲与失效诊断

### 参数不能拼成一个不存在的版本

| 项目 | 2016 Workshop 正文 | 固定作者代码 | 解释 |
|---|---|---|---|
| online z／x | 127／255 | 127／255 | 网络输入像素，不是原图框大小 |
| training x | 255 | 默认239 | 分别17×17／15×15监督图 |
| 尺度数 S | SiamFC 为5；另评估3s | `tracker.m` 默认3；另有5s文件 | 必须记录选用版本 |
| 尺度步长 q | 五尺度1.025 | 默认3s为1.0375；5s为1.0255 | 无量纲，相邻裁剪边长比 |
| scaleLR η | 五尺度0.35 | 默认3s为0.59；5s为0.34 | 尺寸阻尼，不是 SGD 学习率 |
| scalePenalty ρ | 正文说明惩罚，未列此数值 | 3s为0.9745；5s为0.962 | 只惩罚非中心尺度峰 |
| wInfluence λ | 正文说明余弦窗 | 3s为0.176；5s为0.168 | 归一化图与窗的凸组合系数 |
| responseUp U | 17→272 | 16 | 输出采样变密；与 stride8 不相等 |

上述是来源配置，不是现场最优参数范围。5s 文件需显式载入／传参，不会因其存在自动覆盖默认3s。代码搜索边长限幅为首帧 s_x 的 `[0.2,5]` 倍，作者自己称其为任意饱和值；**只有 s_x 被 clip，targetSize 没有同样 clip**，饱和后二者比例可脱钩。不可擅自把一致限幅写成已存在的作者功能。[cite:1](#source-1)[cite:6](#source-6)[cite:8](#source-8)[cite:11](#source-11)

| 症状 | 先保留的诊断证据与调整顺序 | 权衡／回退 |
|---|---|---|
| 始终偏一小段、方向反了 | 逐级保存 row／col 峰、中心基准、裁剪取整、s_x、resize坐标；先查 L4 映射 | 不先改网络或加经验偏移 |
| 静止时框大小振荡 | 各尺度原始峰及惩罚后峰、所选 j、η、窗前后峰 | 核对版本后再增加阻尼或尺度惩罚；滞后随之增加 |
| 快速运动丢失 | 原始响应是否在边缘、目标是否超出有效候选范围、帧间时间 | 降窗影响或扩大搜索需重算 response shape／中心并验收，不只是改255常量 |
| 同款目标高分误跟 | 第一／第二峰、窗前后位置、首帧框背景混入、视频身份真值 | 不能把峰值当身份认证；额外几何／重检测协议独立设计 |
| 常量图、NaN或异常跳回 | 是否全部响应相等、归一分母为0、BN模式、输入幅值 | 显式拒绝，不靠空峰兜底掩盖错误 |
| 换平台结果变了 | RGB／BGR、0–255／0–1、分组卷积、BN／adjust、插值后端和取整 | 先逐层特征与图对齐，再评测；不可任意套 ImageNet mean/std |
| 平均FPS够但偶发慢 | 解码／裁剪／GPU传输／同步／相关／上采样各阶段尾延迟 | 缓存模板、尺度批处理后再剖析，不能搬用论文硬件帧率 |

原代码加载 RGB single（通常0–255），灰度复制三通道，默认 `subMean=false`；首帧图像各通道均值固定用于后续越界 padding。`get_subwindow_tracking` 明确记录 camera-ready 曾用 `mexResize`，该提交用 `imresize`，可能产生差异；`make_scale_pyramid` 从最大区域先 resize 再二次裁剪，不是每尺度直接从原图裁剪一次，两次取整／插值不能假定逐像素等价。[cite:8](#source-8)[cite:10](#source-10)

验收应固定完整视频、初始化、是否允许真值重置、帧时间戳和推理环境，分层统计中心误差、IoU成功曲线、失锁／误重获及端到端延迟。原文 VOT 中失败后的真值重置是**评测协议**，不是跟踪器自己具备的找回能力。本卡无视频成绩，也不提供通用理论误差界或闭环安全保证。[cite:1](#source-1)

## L4 数学推导：相关、梯度与坐标回填

### 空间 shape、感受野与有效平移

无 padding 的单层空间尺寸为 \(n'=\lfloor(n-k)/s\rfloor+1\)，累积采样间距 J 与感受野 Rf 满足 \(J'=Js\)、\(Rf'=Rf+(k-1)J\)。从 `J=Rf=1` 展开主表，conv5 得 **J=8、Rf=87**；模板6格覆盖 \(87+5\cdot8=127\)，搜索22格覆盖 \(87+21\cdot8=255\)，相关输出 \(22-6+1=17\)。中心响应索引为零基8，未插值候选中心在搜索输入中覆盖中心两侧各 \(8\cdot8=64\) 像素，不应把整个255宽度都当可定位位移范围。

固定权重及有效支持下，\(h(L_{8\tau}x)=L_\tau h(x)\)；池化采样相位、边界与训练态 BN 都需单独考虑。它不保证任意连续平移严格等变，更不意味着跟踪坐标无偏。[cite:1](#source-1)[cite:4](#source-4)

### cross-correlation 与两支梯度

省略 batch，令 \(Z\in\mathbb R^{C\times m\times m}\)、\(X\in\mathbb R^{C\times n\times n}\)，\(u,v=0,\ldots,n-m\)：

\[
R_{u,v}=\sum_{c=0}^{C-1}\sum_{i,j=0}^{m-1}Z_{c,i,j}X_{c,u+i,v+j},\qquad
f_{u,v}=\alpha R_{u,v}+b.
\]

交叉相关**不翻转模板**。论文式(2)可取 α=1；固定代码加入1×1 adjustment，α初值0.001且学习率0，b初值0且学习率1。源码图和论文简式需分开，不能加载作者权重后把 α 丢掉或再额外乘一遍。[cite:1](#source-1)[cite:5](#source-5)[cite:13](#source-13)

设 \(G_{u,v}=\partial L/\partial f_{u,v}\)，越界 G 视为0，得到：

\[
\frac{\partial L}{\partial Z_{c,i,j}}=\alpha\sum_{u,v}G_{u,v}X_{c,u+i,v+j},\qquad
\frac{\partial L}{\partial X_{c,a,b'}}=\alpha\sum_{i,j}Z_{c,i,j}G_{a-i,b'-j},
\]

\[
\frac{\partial L}{\partial b}=\sum_{u,v}G_{u,v},\qquad
\frac{\partial L}{\partial\theta}
=J_{\varphi(z),\theta}^{\mathsf T}\nabla_ZL+J_{\varphi(x),\theta}^{\mathsf T}\nabla_XL.
\]

b′ 是搜索列索引，非分数偏置。每个搜索元素可参与多个窗口，反向必须累加；模板分支虽在**在线**缓存，**离线训练**不应 detach。`XCorr.backward` 调用卷积算子的输入梯度与滤波器梯度，再还原 batch；相关层自己没有独立可训练模板参数。[cite:9](#source-9)

### 平衡 logistic 目标、稳定计算与可辨识性

中心 c、stride k=8、输入像素半径 R=16 时，标签 \(y_u=+1\) 当 \(k\lVert u-c\rVert_2\le R\)，否则为−1。对一张图令正负计数 P、Q 均非零：

\[
w_u=\begin{cases}1/(2P)&y_u=+1\\1/(2Q)&y_u=-1,\end{cases}\qquad
L=\sum_u w_u\log(1+\exp(-y_uf_u)).
\]

正、负权重总和各0.5，全部和为1；batch 再求平均。不要在该加权和上不加说明地再除响应位置数。17×17中半径2格有13个正点、276个负点；固定默认15×15中同样13个正点、212个负点。它们不是13个不同对象，且负标签来自同一个搜索图的背景平移。[cite:1](#source-1)[cite:6](#source-6)[cite:7](#source-7)

令 \(t=-yf\)，稳定形式为 \(\max(t,0)+\log(1+\exp(-|t|))\)，由链式法则：

\[
\frac{\partial L}{\partial f_u}=-w_uy_u\sigma(-y_uf_u),\qquad
\frac{\partial^2L}{\partial f_u^2}=w_u\sigma(y_uf_u)\sigma(-y_uf_u)\ge0.
\]

单个 score 上凸不代表深网络 θ 上凸；有限数据下未提供全局最优／泛化保证。正负平衡改变训练权重先验，原始 logit、sigmoid 和在线减最小值归一图均不能不经独立校准当作身份正确概率。双线性相关还存在特征尺度耦合；损失低不唯一决定嵌入，也不能从峰值读出像素误差界。

### 窗、位置与尺寸：作者坐标和几何变体分开

设 \(\widetilde R_j\) 为上采样后的分数图，非中心尺度惩罚为 ρ：

\[
j^*=\arg\max_j \rho_j\max_{r,c}\widetilde R_j(r,c),\quad
A=\widetilde R_{j^*}-\min\widetilde R_{j^*},\quad
H=(1-\lambda)\frac{A}{\sum A}+\lambda\frac{W}{\sum W}.
\]

W 为二维 Hann 外积。若 A 全零，上式无定义，应显式处理；bicubic 可能过冲，不能预设插值后分数范围。若原始峰均为负，乘小于1的 ρ 会使非中心峰更接近0，未必起直觉中的惩罚作用；这是固定乘法规则的代数性质，不宣称本次视频已触发。

**固定作者代码：**坐标顺序为 `[row,col]=[y,x]`，MATLAB 峰为一基，M=17×16=272，零位移基准 \(c_M=\lceil M/2\rceil=136\)。`tracker` 传给 `tracker_eval` 的是 `round(s_x)`，因此：

\[
\Delta_{yx}=\bigl([r_{\max},c_{\max}]-[136,136]\bigr)
\frac8{16}\frac{\operatorname{round}(s_x)}{255},\qquad
p_{yx}^{new}=p_{yx}^{old}+\Delta_{yx}.
\]

移植为零基数组时，对应减去135，**不是**减136，也不是几何中心135.5。原代码此处**没有乘所选尺度 \(a_{j^*}\)**；峰并列时 `find(...,1)` 使用 MATLAB 列优先顺序。窗口／插值的几何中心与该整数基准可能存在半格差异，不能把数学上的对称中心偷偷替换进“原样复现”。[cite:8](#source-8)[cite:9](#source-9)

**另立的几何一致变体：**若每尺度独立从原图裁边长 \(s_j\)，采用 half-pixel resize，则输出零基 q 映回原响应 \((q+0.5)/U-0.5\)；减原响应中心8后等于 \((q-(M-1)/2)/U\)。相应原图位移为 \((q-135.5)\cdot8/16\cdot s_j/255\)。这包含所选尺度且用半格中心，**不是该作者提交的公式**；再加原代码两级裁剪取整，不能仅凭“更几何一致”声称逐像素复现。L5 伪代码选择作者公式，不混用该变体。

位置使用本帧更新前的 s_x。随后令 \(g=(1-\eta)+\eta a_{j^*}\)，更新：

\[
(h,w)^{new}=g(h,w)^{old},\qquad
s_x^{new}=\operatorname{clip}(g s_x^{old},0.2s_{x,1},5s_{x,1}).
\]

故宽高比保持不变；没有回归形变／旋转。输出 `[x,y,w,h]` 为 `[p_x-w/2,p_y-h/2,w,h]`。这采用中心±宽高一半的几何约定；若输入是整数像素的闭区间端点，需要在接口一次性统一，不随意增减1。[cite:8](#source-8)[cite:12](#source-12)

计算量：相关直接计算约 \(S(n-m+1)^2Cm^2\) 次乘加，当前形状为 \(S\cdot17^2\cdot256\cdot6^2\)；另有 S 次搜索嵌入及 \(O(S(17U)^2)\) 上采样。首帧特征缓存省去逐帧模板嵌入，不省搜索计算；训练还需两支激活、BN与优化器状态。不能仅用相关核复杂度推断整机 FPS。

## L5 实现：源码对应、伪代码与有限检查

### 数据布局、源码路径与移植契约

所有外部路径均相对固定提交；来源节提供可点击入口。本地没有这些 MATLAB 文件、权重或训练器。

| 外部源码 | 已读作用 | 实现时必须保持 |
|---|---|---|
| `training/vid_create_net.m`、`make_siameseFC.m` | 五层分组卷积、双支与adjust | conv2／4／5组数2，conv5无ReLU；共享参数不是复制后独立优化 |
| `training/experiment.m`、`vid_get_random_batch.m` | 配置、数据划分、配对与增强 | 默认训练239；有效帧索引范围；整图pair标签与响应位置标签分开 |
| `training/create_labels.m`、`util/create_logisticloss_label.m` | 半径与平衡 | 输入像素半径除stride；P／Q非空；中性环另配协议 |
| `util/XCorr.m`、`add_adjust_layer.m` | 相关前反向与分数缩放 | MatConvNet为H×W×C×N；每个pair只与自己的模板匹配 |
| `tracking/tracker.m`、`tracker_eval.m` | 首帧缓存、选尺度、窗、回填 | 3s默认、原始峰选尺度、一基中心136、基准s_x不乘选中尺度 |
| `tracking/get_subwindow_tracking.m`、`make_scale_pyramid.m` | padding、两级resize与取整 | MATLAB round与Python round的半值规则不同；颜色和插值后端固定 |
| `tracking/load_video_info.m`、`load_pretrained.m` | yx顺序、推理模式 | 灰度扩通道、test模式、输出bbox的xy转换 |

MatConvNet `XCorr` 把 `[H,W,C,N]` 搜索重排为 `[H,W,C*N,1]`，以 `[m,m,C,N]` 模板作为分组卷积核，输出再还原 `[17,17,1,N]`。NCHW 重实现可将搜索 reshape 为 `[1,N*C,H,W]`，核为 `[N,C,m,m]`，用 groups=N 得 `[1,N,17,17]`，再还原 `[N,1,17,17]`；这里的 batch 分组与骨干 groups=2 是两回事。训练若算出 `[N,N,17,17]` 就错配了所有模板；在线尺度批次则要有意复用同一首帧模板。[cite:9](#source-9)

### 对齐作者在线主链的教学伪代码

本段不是可运行实现；`crop_author`／`pyramid_author` 必须按已读两级裁剪源实现，不能用未指定插值的黑盒宣称等价。省略IO与设备细节，保留作者3s参数和坐标；`reject` 是显式新增的工程策略，原代码并无可靠拒绝接口。

```text
S, q, rho, eta, U, lam = 3, 1.0375, 0.9745, 0.59, 16, 0.176
pos_yx, size_hw = parse_box_with_declared_coordinates(box1)
avg_rgb = channel_mean(image1)
sz = sqrt((w + 0.5*(w+h)) * (h + 0.5*(w+h)))
sx = sz * 255 / 127
sx_min, sx_max = 0.2*sx, 5*sx
Z = phi_test(crop_author(image1, pos_yx, round_matlab(sz), 127, avg_rgb))
scales = [q**(-1), 1, q]
W = hann_outer(272, 272)
W = W / sum(W)
emit(box1)
for image in remaining_frames:
    old_sx = sx
    X = phi_test(pyramid_author(image, pos_yx, old_sx*scales, 255, avg_rgb))
    score = checkpoint_adjust(xcorr_valid(repeat(Z, S), X))
    require_shape(score, [S, 1, 17, 17])
    R = bicubic_author(score, factor=U)
    peaks = spatial_max(R) * [rho, 1, rho]
    j = argmax_first_in_scale_order(peaks)
    A = R[j] - min(R[j])
    if not all_finite(R) or sum(A) <= 0:
        reject("invalid_response")
        continue
    H = (1-lam)*A/sum(A) + lam*W
    peak_yx_one_based = argmax_matlab_column_major(H)
    delta_yx = (peak_yx_one_based - [136,136]) * 8/16 * round_matlab(old_sx)/255
    pos_yx = pos_yx + delta_yx
    gain = (1-eta) + eta*scales[j]
    sx = clip(old_sx*gain, sx_min, sx_max)
    size_hw = size_hw*gain
    emit(center_size_to_xywh(pos_yx, size_hw))
```

原 `tracker_eval` 还有两个静态边界风险：`numScale=1` 分支直接使用零初始化的 `responseMapsUP`，没有填入网络响应；常量响应除以0后空峰兜底用了 `ceil(scoreSize/2)=9`，而不是上采样图中心136。这不是可用的失锁处理，故伪代码限定S=3并显式拒绝非有限／平坦响应；本次未运行原 MATLAB，未修改其源码。[cite:9](#source-9)

优化前先建立逐层对照：RGB输入、各卷积输出、相关图、adjust图、尺度峰、H图、位移、框分别比较。向量化／批处理应保留pair对应、通道次序、并列峰规则；将两级裁剪换为一次裁剪、改变窗中心、AMP或融合BN都需单独验等价与误差。模板缓存只在初始化后保持，不在推理结束忘记清理而串到下一条视频。

### 可执行的有限代数检查

以下 PowerShell 5.1 检查器仅用系统双精度算术，验证无 padding shape、半径标签计数、平衡权重、logistic极值、**2×2模板／3×3搜索的单通道相关双支与偏置梯度**，以及作者一基／零基位移和尺寸不变量。小张量是明确的代数夹具，不是合成视频评测，也不代表256通道网络已经运行。它只检查求和／梯度结构，不覆盖 MatConvNet batch reshape、插值后端或实际裁剪。有限差分步长1e−5，绝对容差1e−8。

```powershell
$ErrorActionPreference = 'Stop'
function Assert-Check([bool]$ok, [string]$name) {
    if (-not $ok) { throw $name }
}
function Softplus([double]$t) {
    [Math]::Max($t, 0.0) + [Math]::Log(1.0 + [Math]::Exp(-[Math]::Abs($t)))
}
function Corr([double[]]$z, [double[]]$x) {
    $r = New-Object double[] 4
    for ($u=0; $u -lt 2; $u++) {
        for ($v=0; $v -lt 2; $v++) {
            for ($i=0; $i -lt 2; $i++) {
                for ($j=0; $j -lt 2; $j++) {
                    $r[2*$u+$v] += $z[2*$i+$j]*$x[3*($u+$i)+$v+$j]
                }
            }
        }
    }
    return ,$r
}
function Loss([double[]]$z, [double[]]$x, [double]$bias) {
    $r = Corr $z $x
    $y = @(1.0,-1.0,-1.0,1.0)
    $loss = 0.0
    for ($k=0; $k -lt 4; $k++) { $loss += 0.25*(Softplus (-$y[$k]*(0.001*$r[$k]+$bias))) }
    return $loss
}
$kernels = @(11,3,5,3,3,3,3)
$strides = @(2,2,1,2,1,1,1)
$ez = @(59,29,25,12,10,8,6)
$ex = @(123,61,57,28,26,24,22)
$zside=127; $xside=255; $train=239; $jump=1; $rf=1
for ($l=0; $l -lt 7; $l++) {
    $rf += ($kernels[$l]-1)*$jump
    $jump *= $strides[$l]
    $zside = [Math]::Floor(($zside-$kernels[$l])/$strides[$l])+1
    $xside = [Math]::Floor(($xside-$kernels[$l])/$strides[$l])+1
    $train = [Math]::Floor(($train-$kernels[$l])/$strides[$l])+1
    Assert-Check ($zside -eq $ez[$l] -and $xside -eq $ex[$l]) 'layer_shape'
}
Assert-Check ($rf -eq 87 -and $jump -eq 8 -and $xside-$zside+1 -eq 17 -and $train-$zside+1 -eq 15) 'final_shape'
foreach ($side in @(17,15)) {
    $p=0; $c=($side-1)/2
    for ($i=0; $i -lt $side; $i++) {
        for ($j=0; $j -lt $side; $j++) {
            if (($i-$c)*($i-$c)+($j-$c)*($j-$c) -le 4) { $p++ }
        }
    }
    $n=$side*$side-$p
    Assert-Check ($p -eq 13 -and [Math]::Abs($p*(0.5/$p)+$n*(0.5/$n)-1) -lt 1e-14) 'balanced_labels'
}
Assert-Check ((Softplus 1000) -eq 1000 -and (Softplus -1000) -eq 0) 'stable_logistic'
$z=[double[]]@(1,-2,3,0.5)
$x=[double[]]@(2,1,-1,0,4,2,-3,1,5)
$bias=0.2; $y=@(1.0,-1.0,-1.0,1.0); $r=Corr $z $x
Assert-Check (($r -join ',') -eq '2,16,-16.5,5.5') 'correlation_direction'
$gz=New-Object double[] 4; $gx=New-Object double[] 9; $gb=0.0
for ($u=0; $u -lt 2; $u++) {
    for ($v=0; $v -lt 2; $v++) {
        $k=2*$u+$v
        $g=-0.25*$y[$k]/(1+[Math]::Exp($y[$k]*(0.001*$r[$k]+$bias)))
        $gb += $g
        for ($i=0; $i -lt 2; $i++) {
            for ($j=0; $j -lt 2; $j++) {
                $a=2*$i+$j; $b=3*($u+$i)+$v+$j
                $gz[$a] += 0.001*$g*$x[$b]
                $gx[$b] += 0.001*$g*$z[$a]
            }
        }
    }
}
$eps=1e-5; $maxerr=0.0
foreach ($branch in @('z','x')) {
    $values=$x; $grad=$gx
    if ($branch -eq 'z') { $values=$z; $grad=$gz }
    for ($a=0; $a -lt $values.Length; $a++) {
        $old=$values[$a]; $values[$a]=$old+$eps; $lp=Loss $z $x $bias
        $values[$a]=$old-$eps; $lm=Loss $z $x $bias; $values[$a]=$old
        $maxerr=[Math]::Max($maxerr,[Math]::Abs(($lp-$lm)/(2*$eps)-$grad[$a]))
    }
}
$db=((Loss $z $x ($bias+$eps))-(Loss $z $x ($bias-$eps)))/(2*$eps)
$maxerr=[Math]::Max($maxerr,[Math]::Abs($db-$gb))
Assert-Check ($maxerr -lt 1e-8) 'finite_difference'
$delta=(152-136)*8.0/16*255/255
Assert-Check ($delta -eq 8 -and (151-135)*8.0/16 -eq 8 -and (136-136) -eq 0) 'coordinate_origin'
$gain=1-0.59+0.59*1.0375
Assert-Check ([Math]::Abs($gain-1.022125) -lt 1e-14 -and [Math]::Abs((40*$gain)/(80*$gain)-0.5) -lt 1e-14) 'scale_aspect'
Write-Output ('PASS shape/labels/logistic/xcorr/gradients/coordinates/scale; max_gradient_error={0:E3}' -f $maxerr)
```

**验证状态：**本次 PowerShell 5.1 执行返回 `PASS shape/labels/logistic/xcorr/gradients/coordinates/scale`，最大梯度绝对差 `5.911E-012`，低于1e−8；首次检查发现手写相关期望值末项误填3.5，按直接内积改为5.5后重跑通过。其通过仅支持有限代数断言。没有安装 MATLAB／MatConvNet、加载 `2016-08-17.net.mat`、跑网络前后向、训练、视频回放、OTB／VOT指标或现场闭环。Python命令在本次环境不可用，检查改用已有 PowerShell，不因此新增依赖或其他文件。

## 面试追问链：五问与答案要点

| 层级与问题 | 参考答案及成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：首帧框之后，一帧完整处理和输出是什么？ | 首帧上下文裁剪127、缓存Z；以后上一中心多尺度裁255、嵌入相关、上采样、先选尺度再归一加窗、峰回填、阻尼更新尺寸；输出框，视频结束停止 | 每帧用预测框更新模板；或把后续GT作为输入；原版无自动重检测 | L1、来源1／8／9 |
| L2：为什么共享、无padding和固定模板？删除会怎样？ | 共享保证同一特征空间；有效区域密集平移共享计算；固定模板避免在线污染。stride8只给采样相位一致的等变关系 | “对任意平移尺度旋转都不变”；全卷积仍含pool；固定模板仍会跟错同款物体 | L2、来源1／4／5 |
| L3：尺度振荡与快速运动丢失，分别先改什么？ | 先核对所选3s／5s版本、裁剪和原始峰；尺度问题看q／ρ／η，运动问题看窗λ、时间间隔与搜索支持；用留出视频做消融 | 把scaleLR当训练学习率；只降低分数阈值；扩大255却不改响应shape／中心 | L3、来源8／11 |
| L4：推导平衡logistic和相关层梯度，为什么两支都要反传？ | P／Q非空，各类总权0.5；G=−wyσ(−yf)；Z梯度收集搜索patch，X梯度按重叠支持散加，θ收到两条Jacobian路径之和 | 训练也detach模板；相关先翻核；损失在score上凸所以网络全局收敛 | L4、来源7／9／13 |
| L5：shape正确但偏半像素，怎么核验？ | 对齐HWC／NCHW、yx／xy、一基136与零基135、Matlab round及列优先并列峰、原代码用round(s_x)且不乘选中尺度；逐级保存响应再查插值／两级裁剪 | 把零基135.5或选中s_j公式混进作者复现；梯度微检查通过就宣称整网复现 | L4坐标、L5检查器、来源9／10／12 |

## 来源、获取深度与版本边界

均为 **2026-11-06** 联网核验。T表示实际读到方法正文／源码，B为书目或版本元数据；不表示运行成功。外部源码固定同一个40位提交，避免引用漂移的 master。未取得完整历史训练环境与权重哈希，不保证历史成绩可逐位重现；没有独立三方性能复核。

<a id="source-1"></a>
**[cite:1] 原始 Workshop 正文，T+B。** [Springer 正式章节](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56)，*Fully-Convolutional Siamese Networks for Object Tracking*，ECCV 2016 Workshops，DOI `10.1007/978-3-319-48881-3_56`，出版日2016-11-03。已读§2.1–2.4、式(1)–(7)、§4.1训练／跟踪配置及评测协议；性能数字未作为本仓库结果。原文五尺度1.025／阻尼0.35与固定源码分开。

<a id="source-2"></a>
**[cite:2] 出版 Table 1 与预印本版本，T+B。** [结构表](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56/tables/1)已读，通道列冲突见 L1；[arXiv书目](https://arxiv.org/abs/1606.09549)确认v1为2016-06-30、v2为2016-09-14、v3为2021-12-01。2021是修订时间，不是首次发表年；未逐版diff，也未将v3当2016逐字原稿。

<a id="source-3"></a>
**[cite:3] 作者仓库固定版本，T+B。** [提交元数据](https://api.github.com/repos/bertinetto/siamese-fc/commits/e86eb64d6f146b51135232c1d46a29f64c63678a)与[固定README](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/README.md)。确认2019-05-16提交、论文页码850–865、作者当时使用MATLAB2015b及MatConvNet v1.0-beta20；README另提2017 CFNet基线32通道／更高空间分辨率，本卡明确不采用它。

<a id="source-4"></a>
**[cite:4] 主干结构，T。** [vid_create_net.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/training/vid_create_net.m)。已读创建五层、池化、分组输入通道、BN／ReLU分支及空间合法性检查；主卡通道数以此实现为准。

<a id="source-5"></a>
**[cite:5] 孪生连接，T。** [make_siameseFC.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/training/make_siameseFC.m)。已读启用共享／BN、XCorr连接及adjust参数；未逐项展开 `add_pair_of_streams` 内部实现，不声称已运行共享参数图检查。

<a id="source-6"></a>
**[cite:6] 训练与配对，T。** [experiment.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/training/experiment.m)、[vid_get_random_batch.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/training/vid_get_random_batch.m)。已读239默认、验证分割、batch／LR／weightDecay、固定标签入口、有效轨迹帧采样及增强；没有训练数据下载或运行。

<a id="source-7"></a>
**[cite:7] 标签与权重，T。** [create_labels.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/training/create_labels.m)、[create_logisticloss_label.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/util/create_logisticloss_label.m)。核对径向阈值、balanced两类各半、中性标签配置；本卡采用默认无中性环，未审计MatConvNet整个loss底层。

<a id="source-8"></a>
**[cite:8] 在线状态与默认3s，T。** [tracker.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/tracker.m)。已读首帧特征缓存、均值padding、尺度批次、Hann归一、s_x单独clip与targetSize更新、bbox输出。

<a id="source-9"></a>
**[cite:9] 相关与后处理，T。** [XCorr.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/util/XCorr.m)、[tracker_eval.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/tracker_eval.m)。已读forward／backward／shape、先选尺度后加窗、一基中心、位移换算及单尺度／空峰静态风险。未运行原代码，不把静态风险称作实测故障。

<a id="source-10"></a>
**[cite:10] 裁剪与尺度金字塔，T。** [get_subwindow_tracking.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/get_subwindow_tracking.m)、[make_scale_pyramid.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/make_scale_pyramid.m)。已读均值边界、`round(pos-(sz+1)/2)`、两级resize、尺寸强制恢复及camera-ready插值差异说明。

<a id="source-11"></a>
**[cite:11] 5s配置，T。** [params-5s.txt](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/params-5s.txt)。实际值q=1.0255、ρ=0.962、η=0.34、λ=0.168；不能与论文或默认3s混写。

<a id="source-12"></a>
**[cite:12] 输入与推理模式，T。** [load_video_info.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/load_video_info.m)、[load_pretrained.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/tracking/load_pretrained.m)。已读yx顺序、首帧框入口、移除loss并设test；未核验完整VOT多边形转换与模型文件内部参数。

<a id="source-13"></a>
**[cite:13] 分数adjust，T。** [add_adjust_layer.m](https://github.com/bertinetto/siamese-fc/blob/e86eb64d6f146b51135232c1d46a29f64c63678a/util/add_adjust_layer.m)。已读1×1权重、偏置与学习率赋值；数值及是否学习由来源5调用确定。

## G46 销项范围与遗留事项

G46 的“缺具体 SiamFC 卡”已由 **SI01** 承接：离线平衡logistic训练、127／255与stride8结构、固定首帧模板、尺度金字塔、响应上采样／窗、位置及尺寸更新、梯度／shape／坐标、五问答和版本来源均有独立正文。按覆盖台账的“已有”判定，本卡提供缺卡转已有的依据，**不等于L3／L5全核验或模型实现完成**。

本次仅创建本文件，不修改索引、tracking总览、覆盖矩阵或统计数字；它们保留的“无专卡／G46缺卡”是尚待另次同步的旧账，不能声称已更新。遗留P2：锁定权重与完整运行依赖、逐层前向／反向／裁剪对齐、完整视频协议评测、丢失与重获机制、部署性能及独立安全验收。实现状态始终为**未实现**。
