# DB01 DB：概率／阈值分支、可微二值化与文本区域恢复

[学习规范](../algorithm_learning_standard.md) · [覆盖台账 G30](../evolution_coverage.md) · [字符编码总览](../ocr_codes_evolution.md) · [逐卡索引](README.md)

<a id="db01"></a>
<a id="g30"></a>

## 任务、演进与覆盖状态

**任务／输入输出：**有监督场景文本检测。训练输入图像、文本实例多边形及ignore标签；网络训练输出概率图P、阈值图T和近似二值图B̂，推理通常只输出P，经固定阈值、轮廓／框构造与外扩，返回原图坐标文本多边形或四点框及分数。它不读取字符串，不给出阅读方向；阈值图不是类别图或独立边缘检测器。[cite:1](#source-1)

**演进定位：**分割式文本检测可描述弯曲边界，但固定二值化不参与训练、实例分组后处理可能复杂 → DB 将可微近似阶跃函数接到概率与可学习阈值后，联合监督 → 改进正负边界与近邻分离，使推理可继续使用简单阈值和几何恢复 → 付出阈值分支、shrink／border标签与多任务训练成本 → 缩核丢失细文本、阈值连通性、text-inside-text与外扩误并仍存在。它不是“训练出自适应阈值后推理必须逐像素比较P和T”。[cite:1](#source-1)

**版本冻结：**核验日期 **2026-11-05**；本文件仅 `db_text_detection.md:DB01` 一张卡，承接 **G30／08-05／O-H5**。主论文为 Liao et al., *Real-time Scene Text Detection with Differentiable Binarization*，arXiv **1911.08947v2，2019-12-03**／AAAI 2020。作者仓库 `MhLiao/DB` 固定 **`8ede1b676f87a1637a2ce91095beaeaea03188a6`，2020-04-09**，早于后续DB++；采用 `ic15_resnet50_deform_thre.yaml` 作为明确配置入口，不把类构造器默认值冒充实验有效值。[cite:1](#source-1)[cite:2](#source-2)[cite:7](#source-7)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：P／T／B̂、shrink／border标签、训练与推理区域恢复，限原文和源码 |
| L2 机制 | 已核验：可微阶跃、阈值监督、缩核分离、外扩与评分职责 |
| L3 工程权衡 | 部分：参数量纲、版本配置、诊断与定参顺序已展开；无现场数据 |
| L4 数学推导 | 部分：DB／BCE／Dice梯度、偏移尺度及退化条件；无全模型收敛或检测误差保证 |
| L5 实现 | 部分：固定作者源码、布局／数值／边界和参考伪代码；本仓库未实现DB |
| 面试自测 | 答 5，贯穿五层，包含答案、条件、易错项与证据 |
| 验证 | 文档／源码静态、有限标量核算及原始distance函数隔离核验；无PyTorch模型前反向、训练、权重或检测实测 |

已检索本地 `dl/`，没有DB／SegDetector检测器。**不混现代DB++、PP-OCR训练配方、蒸馏、额外增强或其他检测头。**下文记录的Dice、阈值0.3和多边形unclip=2.0已出现在固定2020作者代码中，不因它们与论文不同就误归为现代扩展。

## L1 步骤：监督生成、网络训练与推理

### 1. 预处理与共享融合特征

四层骨干特征 `C2,C3,C4,C5` 对应输入的 `1/4、1/8、1/16、1/32`。固定作者解码器先将各层 `1×1 conv` 投影为256通道，自顶向下最近邻上采样后相加；再将各层经 `3×3` 压到64通道，并上采样到1/4尺寸，按 `(P5,P4,P3,P2)` 通道拼接为 `[N,256,Hr/4,Wr/4]`。P和T各经 `3×3 conv → BN → ReLU → 两次stride=2转置卷积`，末端sigmoid，得到 `[N,1,Hr,Wr]`。二者是概率域，不是同名logits。[cite:3](#source-3)

论文采用ResNet-18／50并评估deformable convolution；所选IC15配置是 `deformable_resnet50`、`adaptive=True`、`k=50`，不是所有DB都必须含DCN。论文将调制可变形卷积用于conv3–conv5的3×3层；本文未编译其扩展内核。固定类默认 `k=10, adaptive=False` 被配置覆盖，不能只看构造器抄默认。[cite:1](#source-1)[cite:3](#source-3)[cite:7](#source-7)

固定数据集读取 `cv2.imread(..., IMREAD_COLOR).astype(float32)`，即BGR；`NormalizeImage`虽将常量命名为 `RGB_MEAN`，实际直接减 `(122.67891434,116.66876762,104.00698793)` 再除255、HWC转CHW，所读入口没有自动RGB重排。复刻需沿完整链保持通道一致，不能仅据变量名加一次换色。所有几何标签在空间增强／裁剪／resize之后生成，并记录有效图域、padding及原图高宽。[cite:8](#source-8)

### 2. Shrink 正核与有效掩膜

设合法简单文本多边形为G，连续面积A、周长L，以

\[
D=\frac{A(1-r^2)}{L},\qquad r=0.4
\]

作向内偏移得到 `Gs=offset(G,−D)`。Gs内部为GT概率标签1，其余有效图域为0；不可读／过小标注的原多边形区域mask=0。`r`是决定偏移量的无量纲参数，不是对每个顶点乘0.4，也不保证真实收缩面积恰为 `r²A`。[cite:1](#source-1)[cite:4](#source-4)

固定 `MakeSegDetectionData` 的最小尺寸为轴对齐包围范围的 `min(width,height)<8 resize px`，不是旋转框短边；ignore或过小区域被掩掉。正常标注用Pyclipper `JT_ROUND`／`ET_CLOSEDPOLYGON` 内缩；返回空时把原区域ignore，返回多路径时**只取第一条**。这不保证保留原实例所有分支，L5安全参考改为明确拒绝多路径而不是无声丢片段，属于声明过的参考策略。[cite:4](#source-4)

### 3. Border／阈值监督

用同一D外扩得到 `Gd=offset(G,+D)`；以到原多边形边界各线段的最小距离构造边界峰值图，再映射到 `[0.3,0.7]`。原边界处目标最高，向内外远离边界逐渐下降；不是二值“边缘=1、其他=0”，也不是训练P使用的缩核标签。[cite:1](#source-1)[cite:5](#source-5)

论文以Gs与Gd之间解释边带形状，阈值损失的监督支持明确是**整个Gd**；源码 `thresh_mask`也填整个外扩多边形，内部远离边界处目标为低值0.3，不是只在薄环带算L1。多实例的border值取逐像素max、支持域取并集。`MakeSegDetectionData`必须先于`MakeBorderMap`，使过小／空缩核产生的ignore能阻止非法阈值监督。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)[cite:7](#source-7)

### 4. 联合训练与停止

定义 `B̂=sigmoid(k(P−T))`。P和B̂共用缩核GT与有效mask；T用border目标及thresh_mask。固定代码命名 `binary=P`、`thresh=T`、`thresh_binary=B̂`，其中 `binary`仍是连续概率，不是布尔图。主配置 `serial=False`，P和T从融合特征分别预测，不把P当阈值分支的必需输入。[cite:3](#source-3)

| 对象 | 原论文v2 | 固定2020作者代码主配置 |
|---|---|---|
| P损失 | 带困难负样本挖掘的BCE | balanced BCE，loss权重5 |
| B̂损失 | 带困难负样本挖掘的BCE，α=1 | 全有效域Dice，权重1 |
| T损失 | Gd上L1求和，β=10 | Gd上masked L1均值，权重10 |
| 负例选择 | BCE正负比1:3 | P的负数最多正数3倍；B̂的Dice不沿用该top-k集合 |
| 目标总体 | `Ls+Lb+10Lt` | `5×BCE(P)+Dice(B̂)+10×L1(T)` |

**这两套目标不是同一个公式换名。**论文Eq.(8)展示的log似然缺少负号，正文称BCE且前文正负损失均为负对数；本卡按最小化负对数似然写正确定义，并保留展示式问题。源码损失组合见 `L1BalanceCELoss`，不能把论文BCE导数当作实际Dice分支梯度。[cite:1](#source-1)[cite:6](#source-6)

论文先SynthText预训练100k iterations，再目标数据finetune 1200 epochs；batch16，SGD初始学习率0.007、momentum0.9、weight decay0.0001、poly power0.9，训练图640×640。配置、增强支持及crop/pad协议需要一起冻结；本文没有训练。结束预算或按事先规定验证选模规则停止，非有限loss／梯度应中止并诊断，不能用NaN屏蔽掩盖几何标签错误。[cite:1](#source-1)[cite:7](#source-7)

### 5. 推理、评分与 unclip

推理 `model.eval()` 后源码直接返回P，T及DB近似二值分支不执行。固定阈值 `bitmap=P>τ` 仍在网络之外；轮廓提取、评分、外扩也仍是离散后处理。“DB可移除”指额外阈值／近似阶跃分支，不是网络不再需要二值化，也不意味着P>常数和P>T在所有输入上数学等价。[cite:1](#source-1)[cite:3](#source-3)

| 环节 | 论文／固定源码行为 | 必须分清 |
|---|---|---|
| 位图阈值 | 论文0.2；作者representer默认0.3，比较 `>` | 不是训练T的0.3–0.7目标范围，也不是框分数门 |
| 轮廓 | `findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)` | 按返回顺序截取；不是score top-k，不保证孔洞语义 |
| 四点框路径 | 先minAreaRect，短边至少3 map px，再框内均值评分 | 评分区域是外扩前矩形，含矩形内低概率像素 |
| 任意多边形路径 | `approxPolyDP`容差 `0.01×轮廓周长`，至少4点，再区域均值评分 | 不是四点框评分；多点轮廓可保弯曲形状 |
| 框评分门 | representer默认0.7，`score<box_thresh`才丢弃，等号保留 | README不同数据／骨干会覆盖；不是论文统一最优门 |
| unclip距离 | `D'=A'×u/L'`；四点框默认u=1.5，多边形分支显式u=2.0 | A'、L'来自实际传给unclip的外扩前形状 |
| 外扩后检查 | 两路径均检验最小外接矩形短边至少5 map px | 3和5是概率图像素，不是原图物理尺度 |
| 结果逆映射 | x按`原宽/map宽`、y按`原高/map高`，round并clip | 必须另处理真实pad／ROI；不能默认只乘一个scale |

固定配置将 `max_candidates` 从类默认100改为1000；截断发生在轮廓顺序上，不是按置信度排序。四点框路径先框化再扩张再框化；多边形路径先简化再扩张，扩张后仍返回多点形状，二者不可声称同一输出。[cite:7](#source-7)[cite:9](#source-9)

评分为原P在所选多边形栅格掩膜内的均值，**不是只平均P>τ的像素、不是面积、不是T的均值，也不是外扩后重新评分**。四点框路径的A'是拟合矩形面积，曲线多边形路径则是简化轮廓面积。固定polygon分支外扩多于一路会拒绝，但空返回没有完整守卫；box分支把多路径reshape拼接后拟合也需警惕。安全参考在调用几何拟合前统一检查空／多路径。[cite:9](#source-9)

## L2 机制：可微训练与离散恢复分工

| 机制 | 因果解释与不变量 | 删除／替换的后果 |
|---|---|---|
| 可微阶跃 | 围绕P−T提供可传播梯度，错误侧BCE梯度被k放大 | 训练改硬比较会切断该路径；不是整条几何处理全可微 |
| 阈值监督 | border目标对T提供空间约束，避免只靠P−T自由漂移 | 只监督B̂时P与T可共同平移而差值不变，二者不可单独辨识 |
| P独立监督 | 直接让P成为可推理的文本缩核概率图 | 仅优化P−T不能保证P单独与固定τ匹配 |
| shrink | 分开邻接实例的内部核，减轻边界粘连 | 过度内缩会消灭细小文本；放大r减少D，核更易相连 |
| border高阈值 | 原边界附近提高T，使不确定P不轻易成为近似正区 | 把border当P标签会把检测目标从内部核改成边缘 |
| unclip | 从预测核恢复较完整实例轮廓 | 它不是shrink的精确逆，偏移与简化、二值化均丢信息 |
| 分数门 | 对整候选区域的P支持度进行拒识 | 像素阈值和框门不可互换；已经连成一块的邻词不靠均值自动拆开 |

原论文明确提到text-inside-text的局限：一个实例正处于另一个实例中心区域时，单通道分割难以保持独立实例身份。DB没有像素实例ID；本卡不声称阈值分支解决任意重叠、孔洞、嵌套文字或阅读排序。[cite:1](#source-1)

## L3 工程权衡：参数量纲、诊断与调参顺序

| 参数 | 单位／历史起点 | 影响、证据与调节边界 |
|---|---|---|
| resize与输出分辨率 | 输入px；固定head输出全输入分辨率 | 图大更容易保留细核，但骨干、像素后处理及传输更贵；全尺寸图不等于新增细节 |
| shrink_ratio r | 无量纲，0.4 | r增大则D减小、核更大；先统计空核／多路径／相邻核连通 |
| D | 当前标签图px，A/L给长度单位 | 图缩放需重算D，不能复用原图的像素距离 |
| k | 无量纲，主配置50 | 阶跃更陡、正确远端更饱和；不是越大越好，不保证所有网络梯度放大50倍 |
| T目标范围 | 概率域0.3–0.7 | 属监督目标，不是输出层强制范围；源码T为sigmoid可落在0–1 |
| τ／box_thresh | P数值阈值0.3／均值门0.7，类默认限定 | τ改变连通性，框门只拒绝整个候选；有效值还受配置／命令覆盖 |
| unclip_ratio u | 无量纲，box1.5／polygon2.0 | 增大外扩更多，可能吞邻词或背景；优先核评分前后区域和图域 |
| 简化容差 | 0.01×轮廓周长，map px | 减少顶点降几何开销，可能抹去弯曲／细凹部 |
| max_candidates | 个数，所选配置1000 | 按轮廓顺序截断影响召回，不是无损加速；比较截断前后分布 |

参数起点来自原文或固定文件，不是本站调优结论。[cite:1](#source-1)[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)[cite:7](#source-7)[cite:9](#source-9)

排障顺序为 **标注合法性／ignore → 增强坐标与有效图域 → shrink／border可视化 → 张量维度与三项loss／梯度 → P位图与轮廓 → 候选均值／unclip → 阈值标定**。细文字漏检先查缩核是否为空或低于采样支持；粘连先看二值化之前P与GT核，不只缩小unclip；框偏大先区分核、拟合矩形和外扩三步；空图loss异常先查OHEM零正例与Dice梯度，而不是重设优化器。

验证应按词／行粒度和场景批次划分，分别报告polygon precision／recall／F、细文本召回、粘连／断裂、ignore处理、轮廓数量与端到端延迟。作者README说明内置评测协议并未覆盖所有原基准，不能混用不同ignore／匹配协议后的F分数。模型前向、GPU→CPU传输、找轮廓、评分、外扩均需计时；论文FPS不是本仓库速度。[cite:2](#source-2)

## L4 数学推导：梯度、标签几何与尺度

### 1. DB本体与BCE导数

令 \(x=P-T\)、\(b=\sigma(kx)\)，其中P、T都是概率域输出，k>0：

\[
\partial_P b=kb(1-b),\qquad \partial_T b=-kb(1-b),\qquad 0<\partial_Pb\le k/4.
\]

P=T时b=1/2，本体斜率最大为k/4；k=50时为12.5，而不是50。固定k时，远离阈值的本体梯度趋零。若B̂用BCE，负对数似然可写为

\[
\ell(b,y)=-y\log b-(1-y)\log(1-b)=\operatorname{softplus}(kx)-ykx,
\]
\[
\partial_P\ell=k(b-y),\qquad \partial_T\ell=-k(b-y).
\]

正标签在P<T的错误侧对P的导数趋近−k，负标签在P>T错误侧趋近+k；梯度下降分别推高／推低P，并给T相反作用。这里BCE与sigmoid的链式消去使错误侧梯度不随b本体一起消失。**不应先让概率下溢为0／1，再取log期待数值上自动消去。**论文Eq.(3–4)与该结果等价。[cite:1](#source-1)

若 \(P=\sigma(z_P),T=\sigma(z_T)\)，则

\[
\partial_{z_P}\ell=k(b-y)P(1-P),\qquad
\partial_{z_T}\ell=-k(b-y)T(1-T).
\]

概率头饱和仍可能削弱logit梯度；共享骨干梯度还要乘各自Jacobian并相加。故“k放大梯度”不能解释为所有参数梯度恒乘k或保证不消失。只监督b时 `(P+c,T+c)`在可行域内保持差值，P／T分解不可辨识；额外P与T监督限制这种自由度，但不证明网络参数唯一。

### 2. 固定作者Dice与masked L1

令m为有效mask，\(I=\sum_i m_i y_i b_i\)、\(U=\sum_i m_i b_i+\sum_i m_i y_i+\epsilon\)，固定代码

\[
L_b^{code}=1-2I/U,\qquad
\partial_{b_j}L_b^{code}=2m_j(I-y_jU)/U^2.
\]

因此

\[
\partial_{P_j}L_b^{code}=\frac{2m_j(I-y_jU)}{U^2}\,kb_j(1-b_j),\quad
\partial_{T_j}L_b^{code}=-\partial_{P_j}L_b^{code}.
\]

**Dice没有BCE的分母消去，b饱和时该分支梯度仍会变小**；不同像素通过I和U耦合，也不是逐点BCE。T还有L1直接监督：当 \(M_T=\sum m_T>0\)，`Lt=ΣmT|T−T*|/MT`，非零残差处导数为 `mT sign(T−T*)/MT`，再乘外部权10；零残差取次梯度，空支持直接返回零。[cite:6](#source-6)

OHEM选有效正例全部、困难负例最多 `K=min(N−,3N+)`，按选中数加ε归约；分数top-k选择视为离散索引，梯度只流入被选值。若整个归约集合无正例，K=0，固定BCE为0；Dice因I=0而等于1但对b梯度0，若同时没有thresh_mask则全部有效学习信号可能为0。不能用ε让数字有限就宣称空背景训练有效。混合batch是否还有学习信号还取决于归约范围与正确张量布局。[cite:6](#source-6)

### 3. Shrink偏移的来由、量纲与不精确性

理想小距离内缩的面积一阶近似为 `A(D)≈A−LD`，若期望核面积接近 `r²A`，解一阶关系得到

\[
D\approx A(1-r^2)/L.
\]

论文把该式作为偏移定义；这里的一阶面积解释是本卡推导，不是原文声称精确面积定理。顶点角度、凹部、自交、窄颈消失、Clipper join规则与离散量化都会造成高阶或拓扑改变，r=0.4也不总在“小偏移”范围。[cite:1](#source-1)[cite:4](#source-4)

在不改变形状的等比缩放c>0下，A变c²A、L变cL，故D变cD；`D'=uA'/L'`同样按c缩放。各向异性resize后周长不是一个统一倍率，须用当前坐标重算偏移。训练D与推理D'用的是**不同形状和不同公式**：D基于GT原多边形，D'基于预测核轮廓或其矩形；单次外扩不可能精确撤销阈值化、简化和GT内缩。

### 4. Border目标及点到线段距离

设边端点a、b，长度平方 \(h=\|b-a\|^2>0\)，对像素中心p：

\[
t=\operatorname{clip}\left(\frac{(p-a)\cdot(b-a)}h,0,1\right),\quad
\delta(p,[a,b])=\|p-[a+t(b-a)]\|.
\]

取所有边最小距离δ，定义 \(v(p)=1-\min(\delta(p)/D,1)\)，再 `T*=0.3+0.4v`，多实例取v的max。投影t必须clip到线段端点；到无限直线的垂距会在角外产生错误高值。D=0或重复边h=0时不能直接除；重复点先去重并检查至少3点、面积和周长，仍不合法则ignore／拒绝。[cite:5](#source-5)

**固定源码在正常非退化线段上也有错误，不只是零长度边的数值风险。**`make_border_map.py:distance` 令 `s1=||p−a||²`、`s2=||p−b||²`、`h=||b−a||²`，计算 `cosin=(h−s1−s2)/(2√(s1s2))`，再以 `cosin<0` 把垂距覆盖为最近端点距离。该条件不是“投影在线段外”的充要判据。取 `a=(0,0)、b=(2,0)、p=(1,2)`，有 `s1=s2=5、h=4、cosin=−0.6`：原公式先算出垂距2，却被该分支改成 `√5≈2.23606797749979`；真实投影参数t=0.5，点到线段距离应为2。`nan_to_num`不能修复这种有限值上的几何错误。[cite:5](#source-5)

**共线分段风险：**同一条边仅插入共线顶点也可能改变旧距离值。对 `p=(2,2)`，整段 `(0,0)→(4,0)` 的旧函数返回2；在 `(1,0)` 分为两段后，各段旧距离取min却为√5。真实边界没有变化，正确线段距离的min仍应为2；因此冗余共线顶点可能影响border目标，而不只是退化多边形问题。最终阈值图是否显现差异还取决于其他边的min及 `δ/D` 裁剪，不能由单边反例直接断言整图每个像素都不同。

上述clip投影公式是**修正几何语义的参考替代，不是原历史实现逐值复刻**；正常线段反例已证明二者不等价，并非仅“尚未证明等价”。仍须处理h=0、D=0及端点数值边界。按修正后的距离定义，阈值目标在原边界为0.7，离边界至少D处为0.3；监督域仍受Gd限制，并非全背景都强迫T=0.3。

### 5. 评分与复杂度、误差边界

给定外扩前候选Q的有效栅格像素集合Ω，\(s_Q=\sum_{p\in\Omega}P(p)/|\Omega|\)，要求Ω非空。若P在[0,1]，则分数也在[0,1]；它不是框属于某一文本实例的校准概率。栅格边界round、floor／ceil、孔洞填充改变Ω，可能在box_thresh附近改变接受结果。[cite:9](#source-9)

不计骨干，构图与扫描至少O(HW)；边距标签朴素计算随“边数×局部包围区域像素数”增长，轮廓简化和polygon offset复杂度取决于顶点数／交点及库实现，不能把整个后处理无条件写成O(HW)。截断候选与简化轮廓可省时但不是无损。网络误差、阈值处连通性突变与拓扑改变耦合，未提供通用检测误差上界；loss下降不保证实例正确或训练全局收敛。

## L5 实现：固定源码、边界与参考伪代码

### 1. 数据布局与源码风险

| 入口／张量 | 约定 | 必查风险 |
|---|---|---|
| `seg_detector.py` | P／T／B̂为N1HW；融合NCHW | `binary`并非hard bitmap；eval返回P张量而非三图dict |
| `MakeSegDetectionData` | 单图gt为1HW、mask为HW | 训练batch后N1HW与NHW不能直接无脑广播 |
| `MakeBorderMap` | thresh_map／thresh_mask为HW | 正常线段cosin分支错误、共线分段敏感；另有外扩空路径、零周长／零长度边、全出界包围域 |
| `BalanceCrossEntropyLoss` | loss取`[:,0]`，理论应全为NHW计算 | 固定源码positive/negative在squeeze前做`gt*mask`，见下文 |
| `DiceLoss／MaskL1Loss` | 显式取`pred[:,0]`，再与NHW匹配 | 批次归约与逐图归约不同；empty mask要保持设备／dtype |
| `SegDetectorRepresenter` | CPU contour、区域均值、Clipper外扩 | 空／多路径、全零占位框、int16坐标溢出、端点裁剪规则 |

固定 `BalanceCrossEntropyLoss.forward` 在 `gt[N,1,H,W]` 与 `mask[N,H,W]` 直接相乘时，按广播规则会得到 `[N,N,H,W]` 而不是 `[N,1,H,W]`（N>1）；随后NHW loss也可能继续广播。**这是按接口形状可证的源码风险，不代表已执行作者训练复现。**安全参考先统一P、GT、mask为NHW再做正负mask，并断言完全同shape；该修正与旧快照不保证数值一致。[cite:6](#source-6)

固定box路径预分配 `[num_contours,4,2]` 的int16数组，被跳过的条目仍为零框／零分；调用方需按有效分数处理，不能把数组长度当检测数。原图坐标若超int16范围会溢出，安全输出使用float32／int32。作者clip到 `[0,dest_width]`／`[0,dest_height]` 可作为连续边界，但不能拿上界直接当图像索引；取像素时必须限制到W−1／H−1。[cite:9](#source-9)

### 2. 标签与训练参考伪代码

以下为设计伪代码，不是本地模型或作者源码逐行复制。它显式拒绝非法多边形／多路径，修正布局、保留有效图域并给出空监督策略；`clipped_boundary_distance_peak`采用L4的clip投影距离，修正旧cosin分支而非逐值复刻。这些修正／安全策略不能伪称原实现已经具备。几何函数与网络调用仍需实际依赖和单元测试。

```text
make_targets(image_shape, polygons, ignored, valid_domain):
    Y = zeros(image_shape)
    M = copy(valid_domain)
    V = zeros(image_shape)
    MT = zeros(image_shape)
    accepted = []
    for G, flag in zip(polygons, ignored):
        G = remove_duplicate_adjacent_vertices(G)
        if flag or not finite_simple_positive_polygon(G) or too_small(G):
            mask_known_region_or_reject_annotation(M, G)
            continue
        D = area(G) * (1 - 0.4^2) / perimeter(G)
        require D > 0 and finite(D)
        shrunk = offset(G, -D)
        expanded = offset(G, D)
        if count_valid_paths(shrunk) != 1 or count_valid_paths(expanded) != 1:
            mask_known_region(M, G)
            continue
        if rasterized_support(shrunk[0], valid_domain) is empty:
            mask_known_region(M, G)
            continue
        accepted.append((G, D, shrunk[0], expanded[0]))
    for G, D, core, outer in accepted:
        raster_fill(Y, core, 1)
        raster_union(MT, outer)
        V = maximum(V, clipped_boundary_distance_peak(G, D, image_shape))
    MT = MT and valid_domain
    Tstar = 0.3 + 0.4 * V
    return Y, M, Tstar, MT

train_step(image_batch, targets, loss_profile):
    P, T = network_train(image_batch, adaptive=true)
    P, T, Y, M, Tstar, MT = explicitly_convert_all_to_NHW(P, T, targets)
    require all_shapes_equal(P, T, Y, M, Tstar, MT)
    require all_finite(P, T, Tstar)
    if no_valid_supervision(M, MT): return skip_with_reason
    if count(Y and M) == 0:
        return declared_background_policy_or_skip_with_reason
    Bhat = stable_sigmoid(50 * (float32(P) - float32(T)))
    if loss_profile == paper:
        L = mined_BCE(P, Y, M) + mined_BCE_from_logits(50*(P-T), Y, M)
        L = L + 10 * masked_L1_sum(T, Tstar, MT)
    else if loss_profile == fixed_author_intent:
        L = 5 * mined_BCE(P, Y, M) + masked_Dice(Bhat, Y, M)
        L = L + 10 * masked_L1_mean_or_zero(T, Tstar, MT)
    require finite(L) and finite_gradients(L)
    optimizer_step(L)
```

`mask_known_region_or_reject_annotation`不可对任意自交输入随便填充为背景；若无法确定原区域就拒绝该标注／样本并记录原因。参考流程支持忽略区域重叠后的有效mask约束，重叠实例冲突仍需数据协议。`fixed_author_intent`指修正shape后的目标意图，不代表原历史实现逐值复刻。

### 3. 推理与退化多边形参考伪代码

```text
detect(image, output_mode, tau, box_thresh, max_candidates):
    input, transform, valid_domain = preprocess_with_transform(image)
    P = network_eval(input)
    require P.shape == (1, 1, transform.map_h, transform.map_w)
    require finite_probability_map(P)
    bitmap = (P[0,0] > tau) and valid_domain
    contours = find_contours_with_frozen_order(bitmap)
    results = []
    for C in contours[:max_candidates]:
        if output_mode == polygon:
            Q = simplify(C, epsilon=0.01*closed_perimeter(C))
            if vertex_count(Q) < 4: continue
            u = 2.0
        else:
            Q, short_side = minimum_rectangle(C)
            if short_side < 3: continue
            u = 1.5
        if not finite_simple_positive_polygon(Q): continue
        support = rasterize(Q) and valid_domain
        if count(support) == 0: continue
        score = mean(P[0,0][support])
        if score < box_thresh: continue
        Dprime = area(Q) * u / perimeter(Q)
        paths = offset(Q, Dprime)
        if count_valid_paths(paths) != 1: continue
        expanded = paths[0]
        if not finite_simple_positive_polygon(expanded): continue
        rect, short_side = minimum_rectangle(expanded)
        if short_side < 5: continue
        output = expanded if output_mode == polygon else rect
        output = inverse_transform_vertices(output, transform)
        output = apply_declared_output_domain_policy(output)
        if not finite_simple_positive_polygon(output): continue
        results.append((float32_vertices(output), score))
    return results
```

无候选、空offset、多路径、零周长、自交、全出界评分掩膜都在拟合或除法之前拦截；不把NaN转0就继续当有效文本。固定作者推理没有显式通用LANMS步骤，本参考也不加EAST式NMS；需要额外抑制属于另一个部署协议。

### 4. 数值稳定与验证设计

在FP16中直接 `exp(−50(P−T))`可能溢出，即使P和T都合法；推荐DB差值、sigmoid、面积／loss归约至少FP32，并用稳定sigmoid。BCE用logits等价式避免log(0)，但T和P如果本身已有sigmoid，不能把 `P−T`当作未乘k的普通logit，也不能额外再给P做一次sigmoid。硬阈值、轮廓、offset不参与训练反传，不能误detach训练时P或T。

| 检查项 | 应满足的内容 | 当前状态 |
|---|---|---|
| DB与BCE梯度 | P=T时b=0.5、导数±k/4；BCE正负导数符号及有限差分 | 有限标量核算，不是网络autograd |
| Dice与空监督 | masked导数、全负Dice常量、OHEM的K=0、空L1支持 | 代数核算／策略，未执行原损失模块 |
| shape | N>1时所有loss操作均为NHW，拒绝NxNxHxW广播 | 形状规则静态核对，未跑PyTorch |
| 坐标与外扩 | 等比缩放A/L线性、x/y逆缩放分别处理、pad不被评分 | 有限代数；未跑图像／Clipper |
| Border距离 | 正常线段应取投影距离；插入共线顶点不改变边界距离 | 原始distance函数隔离执行，确认2→√5错误及分段反例；非完整标签管线 |
| 多边形 | 重复点、共线、自交、空缩核、窄颈多路径、外扩出界 | 伪代码覆盖，真实几何库测试待执行 |
| 评分与端点 | 只在外扩前Ω均值；空Ω拒绝；τ等号与box门等号分开 | 协议核对，无像素后处理执行 |
| 优化等价性 | FP32对照AMP、轮廓简化、CPU/C++和候选截断 | 未执行，不承诺速度或等价 |

## 面试自测：贯穿 L1–L5 的五问答

1. **L1：DB怎样从GT训练，再从图像生成文本多边形？** 答：增强后GT内缩得Y、外扩和边距得T*，P／T生成B̂并联合监督；推理通常仅P，固定阈值、轮廓、外扩前评分、unclip、逆映射。条件：词／行标注和shape协议一致。易错：推理必须有T、把binary变量当布尔图。证据：L1、[cite:1](#source-1)[cite:3](#source-3)。
2. **L2：为何既训练P又训练T？去掉T监督会怎样？** 答：B̂只看差值，P／T共同平移可能不变；P监督支撑单P推理，border监督约束阈值形状。条件：不意味着每项监督都证明参数唯一；论文也做无阈值监督消融。易错：T是文本边缘分类标签或保证分开所有嵌套文本。证据：L2、[cite:1](#source-1)。
3. **L3：小字漏掉，为什么不能只调unclip？** 答：先查resize后采样、空缩核／小文本ignore、P位图及候选截断；没有核就没有可扩张对象。条件：D以当前图像px计算，r增大使D减小。易错：把r当外扩倍数、把像素τ与区域均值门混用。证据：L3、[cite:4](#source-4)[cite:9](#source-9)。
4. **L4：k=50时梯度一定放大50倍吗？** 答：DB本体最大斜率12.5；BCE对P为50(b−y)，对T相反；真实logit还乘sigmoid导数。固定代码B̂用Dice，没有BCE式消去，须用Dice链式导数。条件：有限概率、正确loss版本及固定top-k分支。易错：拿论文BCE公式解释作者Dice全部行为。证据：L4、[cite:1](#source-1)[cite:6](#source-6)。
5. **L5：为什么batch1能运行却不代表实现正确？** 答：N1HW和NHW直接相乘在N>1可扩成NNHW；还需测零正OHEM、空／多路径offset、int16溢出和空评分区域。条件：这里是所选历史源码及接口形状风险，不冒充已复现训练故障。易错：用nan_to_num或ε掩盖标签／布局问题、把零占位框当检测。证据：L5、[cite:6](#source-6)[cite:9](#source-9)。

## 核验记录与 G30 销项边界

- 已读取学习规范及G30台账；联网取得原论文方法全文、早期作者模型、标签、损失、配置与后处理，锁定2020提交而非现代主分支。
- 已执行两卡结构、来源锚点、本地链接及改动范围检查，并做DB／BCE／Dice梯度、偏移尺度等有限标量核算；不等于框架反向、数据管线或几何库测试。
- 2026-11-05补充复核：重读固定 `make_border_map.py`，在已有NumPy 2.5.4环境中仅提取原始 `distance` 方法、去掉类缩进后隔离执行；正常线段返回√5而非2，共线分段前后返回2／√5，断言通过。未调用 `draw_border_map`、offset、栅格合成或训练，不能称完整border-map集成验证。
- 未下载权重、未编译DCN、未运行PyTorch／Clipper／OpenCV检测流程、未训练或测基准。论文与作者实现的损失、阈值、unclip及归约差异都保留，不报本站F／FPS。
- **G30“缺少DB具体方法专卡”可销项为已有DB01正文**；L3–L5仍部分，实现仍未实现。初版仅写两文件，本次复核仅编辑DB／EAST／CRNN三篇指定文档，没有修改逐卡索引、覆盖矩阵或全库计数；历史ID继续保留。

## 来源与证据边界

原论文与作者代码是相互对照的同源一手证据，不当成独立第三方实测。`[CONFIRMED]`只限定为已读原文／静态内容；代码独有细节为`[SINGLE-SOURCE]`；版本冲突为`[DISPUTED／版本不同]`。访问日期2026-11-05；已读arXiv v2 HTML方法全文，AAAI出版页用于书目核对，不声称逐字完成出版PDF校勘。

<a id="source-1"></a>
**[cite:1]** Liao et al., *Real-time Scene Text Detection with Differentiable Binarization*：[arXiv v2全文](https://arxiv.org/html/1911.08947v2)、[AAAI 2020出版记录](https://ojs.aaai.org/index.php/AAAI/article/view/6812)。重点Methodology Eq.(1–10)、训练细节及Limitation；`[CONFIRMED：原文]`。Eq.(8)展示负号缺失已按负对数定义解释，不隐瞒差异。

<a id="source-2"></a>
**[cite:2]** 作者[MhLiao/DB固定2020快照](https://github.com/MhLiao/DB/tree/8ede1b676f87a1637a2ce91095beaeaea03188a6)、[README](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/README.md)。作者身份由论文直接给出的代码链接交叉核对；评测协议、权重和速度说明只是来源陈述，权重未下载。

<a id="source-3"></a>
**[cite:3]** [decoders/seg_detector.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/decoders/seg_detector.py)：融合、双头、默认值、step_function及eval分支；`[SINGLE-SOURCE：固定实现]`。

<a id="source-4"></a>
**[cite:4]** [make_seg_detection_data.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/data/processes/make_seg_detection_data.py)：shrink、ignore、尺寸与多路径首项；`[SINGLE-SOURCE：固定实现]`，并与论文偏移式对照。

<a id="source-5"></a>
**[cite:5]** [make_border_map.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/data/processes/make_border_map.py)：0.3–0.7标签、外扩mask、线段距离和max合成；`[SINGLE-SOURCE：固定实现]`。重读并隔离执行原始distance，确认正常线段cosin分支错误及共线分段敏感；投影式是非逐值等价的修正，未做整图逐像素或标签管线对照。

<a id="source-6"></a>
**[cite:6]** [seg_detector_loss.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/decoders/seg_detector_loss.py)、[balance_cross_entropy_loss.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/decoders/balance_cross_entropy_loss.py)、[dice_loss.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/decoders/dice_loss.py)、[l1_loss.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/decoders/l1_loss.py)。`[DISPUTED／版本不同]`：与论文损失／归约不同；广播及零正样本结论限静态接口代数，未执行训练。

<a id="source-7"></a>
**[cite:7]** [IC15 ResNet50配置](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/experiments/seg_detector/ic15_resnet50_deform_thre.yaml)、[base.yaml](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/experiments/seg_detector/base.yaml)。前者已核loss_class、k=50、max_candidates=1000及优化器；后者只作显式处理顺序示例，不冒充前者所有数据集继承项。初次猜测`seg_detector_r50_deform_thre.yaml`为404，已改读真实IC15文件，不引用不存在文件。

<a id="source-8"></a>
**[cite:8]** [data/image_dataset.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/data/image_dataset.py)、[normalize_image.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/data/processes/normalize_image.py)：cv2读取和归一化入口；`[SINGLE-SOURCE：固定实现]`，不凭RGB_MEAN命名反推输入颜色。

<a id="source-9"></a>
**[cite:9]** [structure/representers/seg_detector_representer.py](https://github.com/MhLiao/DB/blob/8ede1b676f87a1637a2ce91095beaeaea03188a6/structure/representers/seg_detector_representer.py)：轮廓顺序、box／polygon路径、外扩前均值、unclip、尺寸和坐标。`[SINGLE-SOURCE：固定实现]`，与原论文的阈值／外扩配置差异已逐项记录；尚无运行证据。
