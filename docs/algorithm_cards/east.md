# EA01 EAST：融合特征、RBOX／QUAD 与局部感知几何合并

[学习规范](../algorithm_learning_standard.md) · [覆盖台账 G29](../evolution_coverage.md) · [字符编码总览](../ocr_codes_evolution.md) · [逐卡索引](README.md)

<a id="ea01"></a>
<a id="g29"></a>

## 任务、演进与覆盖状态

**任务／输入输出：**有监督场景文本检测。训练输入为图像、词或文本行的四边形及忽略标签；推理输入只有图像、冻结权重和预处理／阈值协议，输出原图坐标四边形与检测分数，允许空集合。EAST 不识别字符，不给出阅读方向、字符串或业务合格结论。词级与行级由标注粒度决定，不能混标后把合并结果解释为自动理解行结构。[cite:1](#source-1)

**演进定位：**字符候选、聚合、分词等多级处理造成误差与计算叠加 → EAST 以共享 FCN 直接预测位置分数和完整词／行几何 → U 形融合兼顾语义与细节，局部感知 NMS 合并稠密预测 → 新增稠密几何监督、角度／顶点规范、后处理开销 → 超长文本受感受野限制，小文本受采样限制，强弯曲文本不是四边形模型的优势。它是文本检测回归分支，不是通用 OCR 或任意形状实例分割。[cite:1](#source-1)

**版本冻结：**核验日期 **2026-11-05**；仅 `east.md:EA01` 一张卡，承接 **G29／08-04／O-H4**。主理论为 Zhou 等 CVPR 2017，所读全文为 arXiv **1704.03155v2，2017-07-10**。源码主参照为作者账号 `zxytim/EAST` 的 **`4ace589043f2dcb29d5a2f07474c664be3b6fad8`，2017-07-31**。GitHub 元数据明确它是 `argman/EAST` 的 fork，README 自称 RBOX 重实现并感谢作者；**作者账号托管不等于原作者完整实验代码**，尤其不能由此宣称 QUAD 已公开复现。[cite:1](#source-1)[cite:2](#source-2)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：原文融合、标签、两种几何、训练与 LANMS；固定 RBOX 源码静态对照 |
| L2 机制 | 已核验：收缩正区、几何与分数分工、角度／顶点对应及合并假设 |
| L3 工程权衡 | 部分：单位、历史配置、失效诊断完整展开；没有现场定参数据 |
| L4 数学推导 | 部分：损失、梯度、尺度、几何反解和复杂度推导；无全模型收敛／泛化保证 |
| L5 实现 | 部分：固定源码、参考伪代码、有限代数核验；本仓库无 EAST 检测器 |
| 面试自测 | 答 5，逐层含答案、条件、易错点与证据入口 |
| 验证 | 文档／来源静态、有限代数及原始恢复函数／替换IoU的NMS控制流隔离核验；未运行 TensorFlow、Shapely、完整LANMS集成、网络训练或检测基准 |

已检索本地 `dl/`，没有 EAST／RBOX 检测定义。以下“论文”“固定 fork”“参考修正”分别记账；后续 `argman/EAST` 2019 C++ 实现仅用于辨认版本差异，不反向升级原版。原始完整作者实验实现的获取缺口保留。

## L1 步骤：输入到几何及训练闭环

### 1. 图像、融合特征与输出

论文骨干为 VGG16、PVANet、PVANet2x 三种实验路线；提取输入尺寸的 `1/32、1/16、1/8、1/4` 四层特征。从最深层上采样两倍，与对应浅层**按通道拼接**，经 `1×1` 降通道和 `3×3` 融合，再逐级上采样；最后 `3×3` 输出32通道，接独立 `1×1` 分数／几何头。它不是四层特征直接相加，也不是把四个尺度各自当独立检测头。[cite:1](#source-1)

固定 fork 改用 TF-Slim ResNet-v1-50，双线性上采样，融合通道依次128、64、32；NHWC 输入为 `[N,Hr,Wr,3]`，输出分数 `[N,Hr/4,Wr/4,1]`，RBOX `[N,Hr/4,Wr/4,5]`。读取 BGR 后反转成 RGB，再减 `(123.68,116.78,103.94)`；尺寸调整到32倍数，实际横纵比例分别保存。该预处理和骨干属于重实现，不是论文 PVANet 实验的完整复刻。[cite:3](#source-3)[cite:5](#source-5)

| 输出 | 每格含义 | 恢复目标 |
|---|---|---|
| 分数 S | `[0,1]` 的正区置信度 | 选哪些位置的几何值得恢复；不是字符识别概率 |
| RBOX | `(dt,dr,db,dl,θ)`，到矩形上／右／下／左边的距离及角度 | 以该像素为参照恢复旋转矩形四角 |
| QUAD | `(Δx1,Δy1,…,Δx4,Δy4)` | 每角为参照像素加偏移，表示一般四边形 |

RBOX／QUAD 是同一卡的两种替代几何头，不是必须同时输出的13维联合头。固定 fork 仅 RBOX；其距离 `512×sigmoid(z)`，角度 `(sigmoid(a)−0.5)π/2`，范围约 `(-π/4,π/4)`。`text_scale=512` 是输入图像像素的距离上限，不是输出步长或原图宽度归一化；超范围目标无法靠损失自行突破。[cite:1](#source-1)[cite:3](#source-3)

### 2. 正负区域、忽略区与几何标签

四角按环序排列。论文先求每个角相邻两边长度的最小值 `ri`，按对边平均长度决定长边组，先沿长边将两端分别内移 `0.3ri`，再处理短边，得到收缩四边形。**不是统一缩小包围框宽高到0.3，也不是 DB 的面积／周长偏移。**收缩内部为分数正样本，其他非忽略区域为负样本；原框与收缩框之间的环带在所选 fork 中没有自动变成 ignore。[cite:1](#source-1)[cite:4](#source-4)

RBOX 论文由 GT 四边形拟合最小面积覆盖矩形，只在正区位置记录到原矩形四边的距离与同一角度；QUAD 只在正区记录到原四角的偏移。**回归恢复原目标，不恢复收缩后的框。**背景几何没有监督语义，不能拿其零填充值训练距离趋零。[cite:1](#source-1)

固定 fork 的 `generate_rbox` 实际枚举候选平行四边形、选最小面积再转矩形并排序，不等同于直接调用 `minAreaRect`；小文本最短边 `<10 px` 或 `###` 等忽略标签将原多边形区域 `training_mask=0`。标签先在 resize 图生成，再用 `[::4,::4]` 采样到输出格，**距离不除4**。重叠标注的覆盖顺序及多边形退化需额外校验。[cite:4](#source-4)

### 3. 训练及停止条件

每个 batch 联合训练分数和有效正位置的几何；忽略位置不贡献两项梯度。论文分数用类平衡交叉熵，RBOX 用距离 IoU 加余弦角度损失，QUAD 用短边归一化 Smooth L1 和等价顶点排列最小值，详见 L4。论文使用 Adam、512×512随机裁剪、batch 24，学习率从 `1e−3` 每27300 minibatches降为十分之一至 `1e−5`，性能不再改善时停止；这些是历史训练配方，不是本站训练记录。[cite:1](#source-1)

固定 fork 实际用 `0.01×Dice + mean(mask×Y×(平滑IoU损失+20×角度损失))`，不是原论文 `balanced BCE + IoU + 10×angle`。训练时应保存各分项、有效正负数、几何量程和梯度；空监督或 NaN/Inf 必须有显式策略，不把低 loss 当正确训练证据。[cite:3](#source-3)

### 4. 推理与局部感知 NMS（LANMS）

1. 单次 FCN 前向后筛 `S>τs` 的格点，按源位置行优先顺序恢复候选；固定 fork 默认 `τs=0.8`。
2. 格点 `(v,u)` 的参照点是 resize 图 `(4u,4v)`；据 RBOX／QUAD 恢复四角，保留同一索引的分数。
3. 按行优先扫描，若当前框与最近累计框的多边形 IoU `>τn`，按分数加权合并对应四角、累加分数；否则提交累计框并另起一组。
4. 将**合并后的集合**送分数降序标准 NMS，保留高分框并删去 IoU `>τn` 的重叠框，直到候选用尽；等号不抑制。
5. 固定 fork 的附加步骤是在最终框覆盖的 score-map 区域求均值，再按 `box_thresh=0.1` 筛选；`eval.py` 调用 NMS 阈值0.2，不是函数默认0.3。均值重评分明确是重实现与原论文的差异。
6. 四角 x 除实际 `ratio_w`、y 除 `ratio_h` 回原图；如有 padding／ROI，另减 padding、加 ROI 原点。无候选时返回统一空列表。[cite:1](#source-1)[cite:5](#source-5)[cite:6](#source-6)

**固定 fork 不等于正确 LANMS 基准。**其 `nms_locality` 构建 `S` 后却返回 `standard_nms(np.array(polys), thres)`，而非 `S`；又因 `weighted_merge` 原地写当前候选，实际甚至不是“完全没合并的纯 NMS”。其 `restore_rectangle_rbox` 先输出非负角，再输出负角，但 `eval.py` 仍沿原格点顺序赋分；混合角度时可能破坏框—分数对应和扫描局部性。**本次重读固定源码并以隔离用例复核确认这两项已述风险**：恢复使用原始函数，NMS控制流使用原始函数但替换为简化轴对齐IoU；具体范围见核验记录，不是旧仓库完整运行或检测集成。L5 参考流程显式保留索引并对 `S` 做末级 NMS，不声称与这份 fork 逐值相同。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)

## L2 机制：必要性、不变量与替换后果

| 步骤 | 解决什么问题／保持什么 | 删去或替换的后果 |
|---|---|---|
| 逐级融合 | 深层语义支持大词，浅层空间细节帮助小词定位 | 只用最深层容易丢小文本；融合不能恢复输入已丢失的细节 |
| 收缩正区 | 减少边缘不确定性和邻近文本正区接触，多个内部点预测同一完整实例 | 用全框为正改变监督密度；缩过头则小词无正格 |
| 只在正位置回归 | 一个正位置有目标几何归属，背景没有确定框 | 对背景回归零会引入大量无意义梯度 |
| 几何尺度处理 | IoU减轻大框绝对残差支配，QUAD短边归一化关注窄文本 | 不能笼统声称所有项严格尺度不变，Smooth L1阈值和加1平滑均破坏精确不变性 |
| 角度与顶点规范 | 统一等价矩形／四边形的表示，保证平均的是同名角 | 循环错位平均可能收缩成中心，翻转顺序可能自交 |
| 局部合并再 NMS | 利用相邻点预测相关性，减少重复且起几何投票作用 | 只做 NMS 是选框而非平均；全局先按分数排序会改变局部合并轨迹 |
| 两类方向分开 | θ是几何坐标轴角，环序是多边形拓扑 | θ不能识别字是正立还是倒置，也不决定文字阅读顺序 |

几何分数累加可能大于1，反映投票质量／数量的混合，不是校准后概率；再用区域均值替换也是另一评分口径。LANMS 的局部相关假设对拥挤相邻词、不同词框高度重叠时可能失效，合并不是“同一实例”的证明。[cite:1](#source-1)[cite:5](#source-5)[cite:6](#source-6)

## L3 工程权衡：尺度、症状与调参顺序

| 参数／协议 | 单位与版本起点 | 诊断与代价 |
|---|---|---|
| 输入尺寸／stride | resize px；标签与推理 stride=4 | 先看收缩后每词保留几格；增大图增加骨干、候选及 NMS 开销 |
| 收缩系数 | 无量纲0.3，沿边移动 | 粘连先查标签和环序；缩小正区也会减少细小文本监督 |
| 距离上限 | fork `text_scale=512 px` | 统计目标距离超限和预测饱和率；不能只改推理输出缩放 |
| 角度 | 弧度，fork约±π/4 | ±45°处可能换边／换起点；检查距离通道与θ是否一起重排 |
| 分数门／框均值门 | 概率图数值0.8／0.1，fork限定 | 降门增加候选和误报；先排通道、预处理和错位赋分 |
| LANMS IoU门 | 无量纲，eval=0.2 | 降低门在当前比较上更易合并／抑制，不保证最终复杂链输出数单调 |
| 小框过滤 | fork训练最短边10 resize px；提交时相邻边5原图 px | 两个尺度不相同；过滤造成漏检无法靠降低score恢复 |
| 损失归约 | fork几何对整个batch网格 mean | 正区占比改变实际梯度强度；换正样本均值必须重新定权 |

以上参数来自固定文件，不是跨数据集最优值。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

建议诊断顺序：**标注与预处理 → resize／stride坐标 → 正负和ignore → 几何编码／反解往返 → 合并前后框及分数 → 阈值与损失权重**。具体保留每词正格数、角度分布、距离饱和比例、各过滤阶段实例召回、重复框、四边形 IoU、骨干／恢复／NMS 分项耗时。框旋转镜像优先查 y轴方向和角度符号；全部框偏四倍优先查“原点乘4、距离不乘4”；长行断裂先看感受野和标注粒度，不把提高 LANMS 门当万能修复。

数据应按场景／拍摄批次隔离；弯曲、低分辨率、近邻、超长文本分层评估。论文与 fork 的 F-score、硬件和计时范围不同，本卡不转抄为本站指标。未提供通用文本定位误差界，也不保证深网训练达到全局最优。

## L4 数学推导：几何、损失、梯度与退化

### 1. 坐标约定与可逆几何

图像坐标 x向右、y向下；距离通道是当前规范化矩形局部坐标的上右下左，不是原图四边界。以 fork 的旋转矩阵约定写

\[
R(\theta)=\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix},\quad
q_j=o+R(\theta)v_j,
\]

其中 \(o=(4u,4v)\)，四个局部顶点依次为 \((-d_l,-d_t),(d_r,-d_t),(d_r,d_b),(-d_l,d_b)\)。正θ使局部x轴在图像中指向右上方；不能不经转换套用 y向上的数学旋转角或 OpenCV `minAreaRect` 返回角。四边形顶点偏移采用 \(\Delta_j=q_j-o\)，QUAD解码 \(q_j=o+\Delta_j\)。源论文偏移文字方向不能代替编码／解码的明确符号契约。[cite:1](#source-1)[cite:4](#source-4)

若归一化时换了矩形起点／宽高，应同时置换四个距离与角度；几何集合相同不代表监督向量相同。矩形的180°等价性也不意味着 `1−cos(θ−θ*)` 自动处理了所有表示二义性。近正方形的最小面积矩形朝向可能不稳定，退化成线时几何不可辨识。

对固定局部顶点，\(\|\partial q/\partial\theta\|=\|v\|\)，因此同一小角误差在长框远端产生更大像素位移；有限角扰动产生 \(\|\delta q\|\le2\|v\||\sin(\delta\theta/2)|\)。这是固定表示、未经过裁剪／NMS的几何传播关系，不是检测误差上界。非等比 resize 后须先反变换四角；只将距离统一乘一个倍率一般不对。

### 2. 分数正负损失与无正样本

论文定义 \(\beta=1-N_+/N\)，单点类平衡 BCE 为

\[
\ell_s=-\beta y\log p-(1-\beta)(1-y)\log(1-p),\qquad
\partial_z\ell_s=[\beta y+(1-\beta)(1-y)](\sigma(z)-y).
\]

β由标签计算，训练中不对标签／β求导；本卡实现协议对非忽略集合统计并归约，论文没有给出任意ignore布局的通用处理。全部有效像素都为负时 β=1，严格套公式负项权重为0；全正时同理。不能声称原式自然学好任意纯背景batch。应通过采样确保有效组成，或明确加入改变目标的背景fallback；几何空正集取可回传零，而不是对空数组求均值。[cite:1](#source-1)

固定 fork 的 \(D=1-2I/U\)，\(I=\sum myp\)，\(U=\sum my+\sum mp+10^{-5}\)，故

\[
\frac{\partial D}{\partial p_j}=\frac{2m_j(I-y_jU)}{U^2}.
\]

当全batch无正标签时 I=0、y=0，Dice值为1但对 p 的梯度为0；再乘0.01不会恢复背景梯度。mask全零也不等于“损失必为0”。这里按源码代数分析，不把“loss非零”误读为“网络有学习信号”。[cite:3](#source-3)

### 3. RBOX 距离 IoU 与角度梯度

设预测／真实距离均非负且面积正，二者共用参照点：

\[
A=(d_t+d_b)(d_r+d_l),\quad
w_I=\min(d_r,d_r^*)+\min(d_l,d_l^*),\quad
h_I=\min(d_t,d_t^*)+\min(d_b,d_b^*),
\]
\[
I=w_Ih_I,\quad U=A+A^*-I,\quad
L_g=-\log(I/U)+\lambda_\theta[1-\cos(\theta-\theta^*)].
\]

原论文 \(\lambda_\theta=10\)。距离项忽略角度差，是局部轴对齐距离框 IoU，不是任意两个旋转框精确多边形 IoU；只有角度一致等条件下才可作该几何解释。[cite:1](#source-1)

在 min 不换支且 I,U>0处，\(\partial_dL=\partial_dU/U-\partial_dI/I\)。例如 d右小于真值时 \(\partial_{d_r}I=h_I\)，大于真值时为0；\(\partial_{d_r}A=d_t+d_b\)。min并列处不可唯一微分，框架选次梯度，差分测试应避开该点。角度导数为 \(\lambda_\theta\sin(\theta-\theta^*)\)，近零时约为 \(\lambda_\theta(\theta-\theta^*)\)。[cite:1](#source-1)

固定 fork 用 \(-\log[(I+1)/(U+1)]\)、角度权20；加1单位是面积坐标的平方像素量级，避免零除但不再严格尺度不变。所有距离共同放大c时未平滑 I/U不变，而加1版本一般改变。距离输出对logit的导数 `512σ(z)(1−σ(z))`，角度输出为 `(π/2)σ(a)(1−σ(a))`；饱和会抑制上游梯度，即便损失对距离或角度的导数不小。[cite:3](#source-3)

### 4. QUAD 短边归一化与排列

令 \(n_Q=\min_j\|q_j^*-q_{j+1}^*\|>0\)，\(\rho(r)=r^2/2\) 当 \(|r|<1\)，否则 \(|r|-1/2\)。论文 Eq.(13–14)为

\[
L_Q=\min_{\pi\in\mathcal P_Q}\frac1{8n_Q}\sum_{j=1}^{8}\rho(\hat\Delta_j-\Delta^*_{\pi,j}).
\]

**归一化在 Smooth L1 外面，不是先将残差除短边再送 Smooth L1。**固定可接受环序后，等价排列应保持边界邻接；本卡伪代码取4个循环起点并兼容反向环序的8种表示，不枚举包含交叉边的24种任意排列。论文只描述等价顶点顺序集合，所取8种枚举是明确的参考实现契约，不冒充已获取的 QUAD 作者源码。[cite:1](#source-1)

唯一最小排列且不在断点时 \(\partial_{\hat\Delta_j}L_Q=\rho'(r_j)/(8n_Q)\)，\(\rho'(r)=\operatorname{clip}(r,-1,1)\)。多排列并列时目标不可唯一微分，冻结tie-break并检查分支，不对argmin索引求导。短边趋零造成归一化放大，必须拒绝退化标注。线性残差区随尺度大致归一，但二次区及固定断点1并不保持严格尺度不变，不能由论文动机推出全域不变性。

### 5. LANMS 平均与复杂度

顶点已一一对应、分数正且有限时：

\[
\bar q_j=\frac{s_aq_{a,j}+s_bq_{b,j}}{s_a+s_b},\qquad \bar s=s_a+s_b.
\]

论文正文称“加权平均”，紧接的坐标展示式却未显式写除总权重；固定 Python `weighted_merge` 有分母。本卡按归一化坐标平均解释，并记录文字／显示式差异，不能把未归一化加权和当像素坐标。[cite:1](#source-1)[cite:6](#source-6)

给定 n 个四边形，局部扫描做 O(n) 次固定顶点几何比较，若缩成 m组，末级排序 O(m log m)、最坏 NMS O(m²)；总体最坏仍 O(n²)，最佳高度重叠场景接近线性，不是无条件 O(n)。即便单次固定对应的平均是凸组合，经过 IoU门、重排序、顶点重排，最终集合也不保证连续或顺序无关。

## L5 实现：源码映射、参考伪代码与检查

### 1. 固定源码映射及已知风险

| 入口 | 对应职责 | 边界 |
|---|---|---|
| `model.py:model/loss` | ResNet融合、受限RBOX头、Dice／几何目标 | 不是原PVANet、不是QUAD、不是论文损失权重 |
| `icdar.py:shrink_poly/generate_rbox` | 原图尺度标签、长短边收缩、距离／角度 | 退化线交点、重叠标签、零面积应额外守卫 |
| `icdar.py:sort_rectangle/restore_rectangle_rbox` | 角度规范与分支恢复 | 正负角重组需携带索引；不可丢框—分数对应 |
| `eval.py:resize_image/detect` | stride恢复、NMS与区域均值评分 | Python2整数除法；小尺寸对齐可能得到非正大小；现代移植需显式整数策略 |
| `locality_aware_nms.py` | Python多边形IoU与加权平均 | 无效polygon返回IoU=0不等于已过滤；末级传错集合风险见L1 |
| 后续 `argman/EAST:lanms/lanms.h` | C++多边形合并、Clipper和顶点匹配意图 | 独立固定2019提交，非主fork；整数化／排列实现须专项测试，未编译 |

上述证据分别来自固定 fork 和后续提交。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)

### 2. 参考训练与推理伪代码

以下为**设计伪代码**，不是可直接导入的本地模型。抽象几何库需实现并测试 `valid_simple_quad`、多边形裁剪／IoU、环序规范；不得用任意 `buffer(0)` 修复自交后仍冒称原标注。与固定 fork 不同的保护、索引保留和 `NMS(S)` 已显式列出。

```text
train_step(image, annotations, mode, loss_profile):
    require finite_image(image) and mode in {RBOX, QUAD}
    resized, transform = preprocess_with_recorded_transform(image)
    gt = transform_vertices(annotations, transform)
    require every_nonignored_quad_is_simple_finite_positive_area(gt)
    Y, M, G = build_shrunk_labels_and_original_geometry(gt, stride=4)
    S, prediction = network(resized, mode)
    require exact_shape_match(S, Y, M, prediction, G)
    require nonempty_valid_supervision(M)
    require declared_policy_for_single_class_batch(Y, M)
    positive = (Y == 1) and (M == 1)
    Ls = score_loss_with_frozen_profile(S, Y, M, loss_profile)
    if count(positive) == 0:
        Lg = differentiable_zero(prediction)
    else if mode == RBOX:
        Lg = masked_rbox_loss(prediction, G, positive, loss_profile)
    else:
        require all_target_short_edges_above_tolerance(G, positive)
        Lg = minimum_over_ring_preserving_permutations(prediction, G, positive)
    loss = combine_with_declared_reduction(Ls, Lg, loss_profile)
    require finite(loss) and finite_gradients(loss)
    optimizer_step(loss)

detect(image, mode, tau_score, tau_iou):
    resized, transform = preprocess_with_recorded_transform(image)
    S, G = network_eval(resized, mode)
    candidates = []
    for (v, u) in row_major_indices(S > tau_score):
        q = decode_one((4*u, 4*v), G[v,u], mode)
        if valid_simple_quad(q) and finite(S[v,u]) and S[v,u] > 0:
            candidates.append((canonical_ring(q), S[v,u], (v,u)))
    merged = []
    pending = none
    for q, score, source in candidates:
        if pending exists and polygon_iou(pending.q, q) > tau_iou:
            q = align_ring_vertices(pending.q, q)
            proposal = weighted_coordinate_mean(pending, q, score)
            if valid_simple_quad(proposal.q):
                pending = proposal
            else:
                merged.append(pending)
                pending = (q, score)
        else:
            if pending exists: merged.append(pending)
            pending = (q, score)
    if pending exists: merged.append(pending)
    kept = standard_polygon_nms(merged, tau_iou)
    kept = optional_fork_mean_rescore_with_declared_threshold(kept, S, stride=4)
    output = inverse_transform_all_vertices(kept, transform)
    return filter_valid_output_polygons(output)
```

`valid_simple_quad` 至少要求四个有限不同顶点、环序无交叉、面积／短边大于声明的容差；本卡按连续几何面积，不混inclusive框的 `+1`。出图范围应先做几何求交或按冻结输出协议处理，再检查顶点数和面积；逐点clip可能改变形状，不默认仍是四边形。无候选直接返回空列表；不以全零占位框代表检测成功。

### 3. 梯度、数值与优化验证设计

| 检查 | 预期不变量／应暴露的问题 | 当前范围 |
|---|---|---|
| 编码—解码 | θ=0及正负角，四角往返；混合角度框仍匹配原分数 | 有限公式核算；原始恢复函数隔离确认混合角重排，未跑编码／检测全链 |
| stride与resize | 只给原点乘4；横纵分别逆缩放；ROI与pad只补偿一次 | 代数协议，非整图运行 |
| 标签退化 | 重复点、共线、自交、负面积方向、全ignore、无正格 | 伪代码拒绝／显式策略，几何库未执行 |
| 梯度有限差分 | BCE、余弦角度、避开min并列的IoU、固定排列Smooth L1 | 有限标量核算，不是自动微分或网络反传 |
| LANMS | 相同框加权平均不漂移；不同起点先对齐；末级输入确为merged | 原始合并／NMS函数＋简化IoU确认末级传polys及原地修改；原Shapely多边形IoU／集成未执行 |
| 精度与吞吐 | 浮点均值、整数Clipper量化、并列分数、候选截断逐一对比 | 仅设计；无延迟／精度测量 |

BCE应使用稳定softplus／logits接口；几何面积与loss归约宜FP32或更高，不能先产生 Inf 再乘0掩膜期待消失。优化先剖析骨干、恢复和多边形交集，再考虑向量化／C++；必须保持候选顺序、顶点对应、等号规则和重评分区域一致。减少候选、换轴对齐IoU、先量化顶点都是可能改变算法输出的近似，不是默认等价优化。

## 面试自测：贯穿 L1–L5 的五问答

1. **L1：给一张倾斜文本图，EAST怎样得到原图框？** 答：记录resize，四尺度融合，分数筛格，原点乘stride、按几何恢复，行优先局部平均再NMS，按实际横纵比例逆变换；空集合法。条件：权重／预处理与几何头匹配。易错：把距离再乘4、把检测当识别。证据：L1、[cite:1](#source-1)[cite:5](#source-5)。
2. **L2：为何收缩正区但回归原框？LANMS是普通NMS吗？** 答：内部点监督更可靠、邻词较易分开，但目标仍是完整词／行；LANMS先对对应角投票平均再抑制。条件：正区可采样、局部预测确属同一实例。反例：密排邻词错合并；全背景回归零没有框语义。证据：L2、[cite:1](#source-1)[cite:6](#source-6)。
3. **L3：长行两端漏掉，先降低score吗？** 答：先看标注粒度、resize后长度、感受野、512距离饱和与NMS前候选；再做单因素对照。条件：使用固定fork的受限头；论文其他骨干不继承该上限。易错：把所有漏检归于阈值、无条件加大图。证据：L3、[cite:3](#source-3)。
4. **L4：IoU项真是旋转IoU吗？QUAD损失严格尺度不变吗？** 答：距离项忽略角度，是局部AABB代理，另加余弦；QUAD为Smooth L1之后除短边，固定断点导致不全域尺度不变。条件：正面积、短边非零、可微分支固定。易错：先归一化残差、宣称角度项解决阅读方向。证据：L4、[cite:1](#source-1)。
5. **L5：为什么照作者账号代码也可能错？** 答：这是RBOX fork，需检查恢复重排的索引、LANMS末级传入集合、Python2除法、空正Dice梯度、无效polygon与量化。条件：风险针对冻结提交，不能推断所有后续版本。易错：仅凭仓库名认定原版QUAD／算法已验证。证据：L5、[cite:2](#source-2)[cite:4](#source-4)[cite:6](#source-6)。

## 核验记录与 G29 销项边界

- 已读取本仓库学习规范、索引及G29行；已联网取得原论文§3全链与局限、固定fork的模型／标签／恢复／合并源码，确认来源身份与差异。
- 已执行两卡文档结构、引用锚点、本地链接及改动范围核验，并做有限标量几何／梯度／尺度恒等式检查；这是文档与代数验证，不是EAST实现测试。
- 2026-11-05补充复核确认：重读固定 `icdar.py`、`eval.py`、`locality_aware_nms.py`，在已有NumPy 2.5.4环境中提取原始函数隔离执行，不导入旧仓库模块。`restore_rectangle_rbox`输入原点 `[(10,10),(30,10),(50,10)]`、各框距离 `(2,3,2,3)`、角度 `[-0.2,0.2,-0.1]`；批量输出与逐项调用按索引 `[1,0,2]`重排后相同，而非原顺序。若原分数为 `[0.9,0.8,0.7]`，对应输出应配 `[0.8,0.9,0.7]`；`eval.py`仍按原顺序赋分这一连接关系仅作源码核对，未执行detect集成。
- NMS核验保留原始 `weighted_merge`、`nms_locality`、`standard_nms` 函数体，**仅将 `intersection` 替换为连续坐标轴对齐矩形IoU**，并包装末级入口记录实参。三个相同 `(0,0)–(2,2)` 方框、初分数均为1、门限0.2时，局部 `S`仅1项，末级实际仍收到3行polys，原数组分数变成 `[1,2,3]`，最终留下分数3的1框。断言确认传参和原地修改；此例最终框相同也不代表两条流程等价。简化IoU只用于控制流夹具，不验证原Shapely交集、旋转／无效多边形、顶点对齐或完整LANMS集成。
- 未下载权重、未训练、未加载模型、未跑任何OCR数据集；旧fork风险由静态发现补充为上述有限隔离确认，QUAD只有论文与参考伪代码，后续C++ LANMS也未编译或测试。
- **G29“缺少EAST具体方法专卡”可销项为已有EA01正文**；L3–L5仍部分、实现仍未实现，原始完整作者实验代码缺口仍保留。初版仅写两文件，本次复核仅编辑DB／EAST／CRNN三篇指定文档；索引和覆盖台账没有同步，不能称全库统计已经更新。

## 来源与证据边界

原文和作者关联fork属于同一方法证据链，不伪装成三个独立复现。`[CONFIRMED]`在本卡仅表示原文／静态源码的明确内容；单一源码实现行为标为`[SINGLE-SOURCE]`；论文与实现分歧保留为`[DISPUTED／版本不同]`，不强行统一。来源均于2026-11-05在线读取；arXiv HTML方法正文完整读取，不等于逐字比对CVPR排版PDF。

<a id="source-1"></a>
**[cite:1]** Zhou et al., *EAST: An Efficient and Accurate Scene Text Detector*：[arXiv v2全文](https://arxiv.org/html/1704.03155v2)，重点§3.2–3.6、Eq.(3–14)、Algorithm 1与§4.6。`[CONFIRMED：原文]`；加权平均展示式与文字的分母差异已记录。

<a id="source-2"></a>
**[cite:2]** [固定作者账号fork](https://github.com/zxytim/EAST/tree/4ace589043f2dcb29d5a2f07474c664be3b6fad8)、[README](https://github.com/zxytim/EAST/blob/4ace589043f2dcb29d5a2f07474c664be3b6fad8/readme.md)、[仓库元数据](https://api.github.com/repos/zxytim/EAST)。已核 `fork=true`、parent=`argman/EAST`；`[SINGLE-SOURCE：身份及RBOX重实现声明]`，不是原版全功能作者程序。

<a id="source-3"></a>
**[cite:3]** [model.py](https://github.com/zxytim/EAST/blob/4ace589043f2dcb29d5a2f07474c664be3b6fad8/model.py)：融合、受限头、Dice、角度20及全网格mean。raw首次访问失败后用GitHub内容API成功读取；`[SINGLE-SOURCE：固定实现]`，与论文配置不同。

<a id="source-4"></a>
**[cite:4]** [icdar.py](https://github.com/zxytim/EAST/blob/4ace589043f2dcb29d5a2f07474c664be3b6fad8/icdar.py)：收缩、矩形拟合、角度、stride标签与恢复分组。`[SINGLE-SOURCE：固定实现]`；补充隔离执行原始restore_rectangle_rbox确认正负角分组重排，无网络／几何库集成运行证据。

<a id="source-5"></a>
**[cite:5]** [eval.py](https://github.com/zxytim/EAST/blob/4ace589043f2dcb29d5a2f07474c664be3b6fad8/eval.py)：resize、RGB、坐标恢复、重评分及门限。`[SINGLE-SOURCE：固定实现]`。

<a id="source-6"></a>
**[cite:6]** [locality_aware_nms.py](https://github.com/zxytim/EAST/blob/4ace589043f2dcb29d5a2f07474c664be3b6fad8/locality_aware_nms.py)：加权平均、IoU规则与末级集合参数。`[DISPUTED／版本不同]`：具体实现不完全对应论文Algorithm 1；原始合并／NMS函数配简化轴对齐IoU的隔离核验已确认末级传polys及原地修改，原Shapely IoU与完整LANMS集成未运行。

<a id="source-7"></a>
**[cite:7]** 后续第三方上游 `argman/EAST`，固定 **`dca414de39a3a4915a019c9a02c1832a31cdd0ca`，2019-02-17**：[lanms.h](https://github.com/argman/EAST/blob/dca414de39a3a4915a019c9a02c1832a31cdd0ca/lanms/lanms.h)。只核C++入口、Clipper及顶点匹配代码意图；未证明所有排列等价或与Python逐值一致，不用它补充“原始作者实现已获得”。
