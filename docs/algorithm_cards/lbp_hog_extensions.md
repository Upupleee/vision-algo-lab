# LBP 分布分类、HOG 检测与旋转不变多尺度 LBP 算法卡

[学习规范](../algorithm_learning_standard.md) · [覆盖台账](../evolution_coverage.md) · [缺陷异常](../defect_evolution.md) · [分类装配](../classification_assembly_evolution.md) · [颜色外观](../color_appearance_evolution.md)

## 范围、版本与覆盖协议

核验日期：**2026-11-06**。本文件恰有三张独立卡：`lbp_hog_extensions.md:LB96`、`lbp_hog_extensions.md:HG05`、`lbp_hog_extensions.md:LB02`，分别承接 **G06／06-02、G07／07-03、G08／09-10**。顺序按任务编号，不表示2005早于2002。LB96的两个数据协议、HG05的提取／训练／检测、LB02的编码变体／分类分支均不另计卡。

| 卡号 | 锁定历史对象 | 本卡补齐的关键机制 | 排除的冒名替代 |
|---|---|---|---|
| [LB96](#lb96) | Ojala、Pietikäinen、Harwood，Pattern Recognition **29(1)，51–59，1996**，DOI `10.1016/0031-3203(95)00067-4` | 整数3×3基本LBP；LBP/C联合分布；Image Set I单向KL与Image Set II留一法双向G检验 | 不是只给256箱后接现代chi-square；不是1994会议论文的逐字版本 |
| [HG05](#hg05) | Dalal、Triggs，**CVPR 2005**，DOI `10.1109/CVPR.2005.177` | 默认R-HOG、线性软间隔SVM、一次困难负例增广重训、全图多尺度扫描与NMS接口 | 不是仓库64×64 HOG＋StandardScaler＋RBF SVC候选重排；不是默认OpenCV检测器逐数复现 |
| [LB02](#lb02) | Ojala、Pietikäinen、Mäenpää，IEEE TPAMI **24(7)，971–987，2002-07**，DOI `10.1109/TPAMI.2002.1017623` | 圆周采样、uniform／ri／riu2、VAR、逐尺度对数似然相加、Brodatz类原型与Outex 3-NN | 不是1996整数邻域；不是“uniform都59箱”；不是以RBF替换论文分类 |

历史定义依据分别见[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)。出版年、网页修订号、库tag和本次文档日期分别记录，不互相替代。

| 卡号 | L1步骤 | L2机制 | L3工程权衡 | L4数学推导 | L5实现 | 自测／实现状态 |
|---|---|---|---|---|---|---|
| LB96 | 已核验：两种原分类主链；原实验全部细节未复现 | 已核验：分布及互补对比度 | 部分：参数／排障已写，无真实数据调参 | 已核验：限定KL、列联表G及不变性代数 | 部分：参考伪代码及有限检查 | 答5；本仓库历史管线未实现 |
| HG05 | 已核验：原默认特征、训练、扫描和去重角色；NMS具体实现未核 | 已核验：梯度／块归一化／挖负例 | 部分：配置／失效证据已写，无训练实测 | 已核验：限定维数、投票、归一化、线性SVM模型代数 | 部分：源码差异、伪代码及有限检查 | 答5；仓库仅有不等价的局部教学实现 |
| LB02 | 已核验：编码、联合分布、两套分类协议 | 已核验：模式合并及多尺度近似 | 部分：参数／排障已写，无Outex实测 | 已核验：限定循环群、VAR及似然代数 | 部分：固定源码核对、伪代码及有限检查 | 答5；本仓库历史管线未实现 |

这里“已核验”只指实际读到的资料与明示条件下的推导，**不表示五层全完成、历史结果复现、训练完成或现场验收**。每卡均给实现接口、失败出口和测试设计；本次执行的只有文末确定性算术／穷举检查。论文原有指标不抄成仓库成绩。

### 旧卡承接边界

| 旧入口 | 可以复用 | 本卡补充／纠正 |
|---|---|---|
| [remaining_domains:D03](remaining_domains.md#d03-lbp-纹理直方图基线) | 基本二值比较、256箱、直方图丢空间信息 | D03明确是卡方型异常基线；不能代表1996 KL／G与LBP/C，也不能代表2002圆采样riu2、VAR、多尺度及3-NN |
| [learning_components:C02](learning_components.md#c02-支持向量机-svmrbf-svc) | 分类器和特征分离、训练划分原则 | RBF SVC是另一模型，不能覆盖HG05原线性训练链 |
| [hog_svm_matching.py](../../ml/hog_svm_matching.py) | `hog_features`、`make_training_set`、`verify_candidates`提供现存源码对照 | 当前是模板扰动正例、随机纹理负例、每次重训、概率排序；没有原历史训练集、困难负例增广、多尺度全图扫描或检测框NMS |

旧文件及台账不在本次写入范围；本文件建立新的完整阅读入口，不改变旧卡的合法教学用途，不把旧卡等级覆盖为较高值。[cite:6](#source-6)

<a id="lb96"></a>
## LB96 1996 基本 LBP、互补对比度与分布分类

### 任务契约与演进定位

**输入：**灰度纹理ROI、有效域、带标签模型图或样本库、冻结的邻域／位序／直方图配置，以及明确的`SET_I_KL`或`SET_II_G_LOO`协议。**输出：**LBP或LBP/C分布、到每个模型的距离、最近模型标签及无效原因。原任务是有标签闭集纹理分类，不是只有正常训练数据的异常检测；接入06-02时还需另定异常参考、局部窗口、阈值和整件聚合。

前序瓶颈是用均值、方差等少量标量压缩局部特征，丢掉分布形状；三值texture unit又有较多模式。1996方法使用二值局部模式及其分布，配合幅值互补量，通过非参数分布比较判类；代价是直方图稀疏、样本量和量化选择，遗留问题是尺度／旋转敏感、弱对比处易翻位、全局直方图不能唯一定位缺陷。论文§2.4以两级模式对比三值texture unit，§2.5和§3讨论联合分布及分类。[cite:1](#source-1)

### L1 步骤：两种分类协议分别闭合

1. **先选协议和模型单位。**Set I每类一个模型分布；Set II每个样本轮流作为待测，其余样本作为模型，按最近模型标签分类。不能先平均成每类一个中心再声称Set II留一法。
2. **使用整数3×3邻域。**对八邻居分别与中心比较，将高侧编码为1、低侧为0，按冻结位序累加二进制权重。这里不在圆上插值；对角像素距离为√2像素，不是2002半径1的圆邻域。正文说明中心阈值，等值细节与图中原始位权必须单独登记。
3. **计算互补C。**将八邻居按阈值分为高、低两组，C为两组灰度均值之差，不含中心。高侧／低侧任一为空时原文所读段落未明确实现约定；本卡参考实现记`degenerate_contrast=True`并取C=0，不能把该约定标为原作者代码。
4. **训练阶段确定量化边界。**基本LBP标签直接进入256箱，不能按数值大小压成32箱。连续特征按所有模型图汇总分布作等频分箱；原文大多数连续单特征用32箱，LBP/C中C用8箱，因此联合表为256×8。量化只沿连续轴进行，且所有模型、查询使用同一边界。[cite:1](#source-1)
5. **构建分布。**单特征计数或逐中心同时累加`H[lbp, c_bin]`。联合分布不是分别统计两个边缘后拼接或取乘积。原文把空箱置为1；本卡实现为**原始计数中的零替成1，再归一化**，不是对所有箱加1。
6. **执行Set I。**每类扫描经全图灰度校正的256×256模型图；按原实验对九类各抽1000个32×32或16×16子样本。评估某样本时，从其真类模型中扣除该样本分布，再处理空箱、归一化，以最小单向KL判类。这里用真标签扣除是实验留出操作，不能放到未知输入的实际推理里。
7. **执行Set II。**轮流移除查询本身，对其余每个模型样本计算双向列联表G统计量，选择G最小模型的类别。查询不能与自身比较；同源裁片仍可能有关联，留一不等于跨工件／跨批次外推。
8. **停止与异常。**遍历全部模型后输出闭集结果；无有效中心、空模型库、计数扣减成负数、版本不匹配时返回错误／UNKNOWN。等距时固定模型ID排序只属实现约定。原文没有经本卡核实的通用未知类拒识门。[cite:1](#source-1)

**历史实验与工程划分分开：**Set I原文把整幅256×256图均值校正到**256**、标准差40，然后取子窗；不能因8位习惯把256擅自“纠正”为128或截到255。论文称扣除查询取得无偏估计，但重叠子窗、同一原图统计和预先量化并不构成现代意义的独立跨域测试。Set II完整图像预处理、全部分组及原始清单未由本次已读正文闭合，不填造随机种子或全部准确率表。面向部署应另作工件／原图／批次隔离，这是一套新验证协议，不是历史实验的改写。[cite:1](#source-1)

### L2 机制：模式与幅值为何互补

LBP保留“哪里比中心亮”，主动舍弃差值大小。对于严格递增映射f，在**实际比较的采样值**上有`g>=gc ⇔ f(g)>=f(gc)`，因此码不变；量化合并、饱和、局部阴影和噪声会破坏这一前提。递减映射通常翻转不等号，不能称无条件单调不变。基本LBP也不旋转不变：邻点换位置使位权变动。

若两个邻域只有对比度倍率不同，其LBP可完全相同，C却不同；因此LBP/C保留“某种模式出现时对应多少局部对比度”的耦合。C不是2002的邻居方差VAR，也不是中心到所有邻居的平均绝对差。对比度有益不代表灰度不变性仍完整：对`I'=aI+b, a>0`，C变为aC。论文把LBP/SCOV也列为互补对照，但本卡不因此另计SCOV全算法覆盖。[cite:1](#source-1)[cite:4](#source-4)

| 删除／替换步骤 | 因果后果 |
|---|---|
| 用均值替代整个分布 | 相同均值的不同模式混合可能无法区分 |
| 将LBP码当连续数值等频分箱 | 相邻整数未必有相近局部结构，分箱会合并语义无关模式 |
| 用两个边缘直方图替代联合表 | 丢掉模式与C的相关关系；相同边缘不保证相同联合分布 |
| 一律用对称chi-square | 改变评分几何、平滑和最近模型排序，不是原KL／G的代数化简 |
| 用Set I的真类扣除做部署推理 | 泄露待预测标签，未知查询无法执行 |

### L3 工程权衡：量纲、支持与排障

| 参数／约束 | 原文依据或工程选择 | 诊断证据与调整顺序 |
|---|---|---|
| 邻域和窗口 | 原法整数3×3；Set I考察16／32像素边长子窗 | 先查纹理周期、像素物理尺度和ROI边缘，再改统计窗；改邻域就是新版本 |
| C量化 | 原文8箱；C单位为灰度，不是灰度² | 查看训练分位点、跨批饱和、重复分位点与空箱率；冻结边界再调分类 |
| 256或2048箱 | 256标签／256×8联合表 | 16×16有效中心只有196个，2048箱必大量为空；报告空箱替换总质量，不隐藏稀疏 |
| 灰度校正 | 历史Set I整图均值256、标准差40 | 浮点保存；常量图不能除以零；工程是否保留真实对比度由任务决定 |
| 平滑 | 历史零计数置1，不是统一Dirichlet加1 | 比较填补前后质量、近邻排名；不得只在分母塞epsilon后称等价 |
| 距离／拒识 | KL与G无统一可迁移阈值 | 先核分布／协议，再用独立已知与未知样本定拒识；阈值不由本卡算术例指定 |

若亮度变化导致异常：先看是否饱和、是否误用uint8减法、同一坐标／边界和C量化，再看KL与G是否混用，最后才调门限。联合特征并不保证总比单特征好：样本不足时，C提供的信息可被量化误差与填补质量淹没。应报告逐类混淆、批次误报、未知误接及缺陷尺寸分层召回；本次未执行这些验收。

### L4 数学推导：KL、双向G与不变性

设八邻居为g₀…g₇，中心为g_c，参考比较约定为`>=`，则

\[
b_j=\mathbf1[g_j\ge g_c],\qquad LBP=\sum_{j=0}^{7}2^jb_j.
\]

设A={j:b_j=1}、B={j:b_j=0}，两者非空时

\[
C=\frac1{|A|}\sum_{j\in A}g_j-\frac1{|B|}\sum_{j\in B}g_j.
\]

正仿射变化不改变A、B，故`C'=aC`，平移b消去。局部比较中若每个灰度噪声绝对值不超过δ，则差值扰动至多2δ；`|g_j-g_c|>2δ`是该位不翻转的充分条件。阈值附近和恰相等点不享有此保证，这不是分类误差界。

**Set I：**将平滑后计数归一化为s、m。若把给定模型概率m下的查询计数n视为多项计数，则

\[
\log P(n\mid m)=\log\frac{N!}{\prod_i n_i!}+\sum_i n_i\log m_i.
\]

对固定查询，第一项与模型无关；最大化`Σs_i log m_i`等价于最小化

\[
D(s\Vert m)=\sum_{i:s_i>0}s_i\log\frac{s_i}{m_i}
=\sum_i s_i\log s_i-\sum_i s_i\log m_i.
\]

这是论文式(1)的方向：**查询s在前，模型m在后**。在规范化概率与正模型支持上D≥0，s=m时为0；一般不对称、不满足三角不等式。若未平滑且s_i>0、m_i=0，D=∞；s_i=0的项取0。邻域重叠使像素计数非独立，以上多项模型解释评分，不赋予真实图像独立似然或无条件误差保证。[cite:1](#source-1)

**Set II：**令f₁ᵢ、f₂ᵢ为查询和候选模型的频数，行和N₁、N₂，列和cᵢ，总和T。共同分布假设下极大似然估计`p_i=c_i/T`，独立两行模型下`p_ri=f_ri/N_r`，所以似然比为

\[
G=2\sum_{r=1}^{2}\sum_i f_{ri}\log\frac{f_{ri}}{N_rc_i/T}
=2\left[\sum_{r,i}f_{ri}\log f_{ri}-\sum_rN_r\log N_r
-\sum_i c_i\log c_i+T\log T\right].
\]

这正是原文式(2)的两行列联表展开，不是`D(s||m)+D(m||s)`，也不是Pearson卡方。以`p=f₁/N₁, q=f₂/N₂, w=N₁/T, z=wp+(1-w)q`写成

\[
G=2T\{wD(p\Vert z)+(1-w)D(q\Vert z)\}.
\]

因此交换两行G不变；同倍放大**已确定频数**，G也同倍放大。归一化频率后直接当计数会丢失这个样本量信息；先对原始计数做零替换也会改变行和与w。G虽有渐近检验背景，这里按原文作近邻排序，不由稀疏、相关像素无条件导出显著性p值。LBP/C只需把二维格按固定顺序展平成i，不改判别公式。[cite:1](#source-1)

计数复杂度为O(8N)，C无需排序；256或2048箱存储，K个模型单次分类O(KB)。Set II对Q个样本完整留一为O(Q²B)，模型频数应缓存；特征空间改变或扣除查询后需重新执行规定的平滑，不复用错误的已归一化缓存。未提供通用分类精度或跨域泛化理论界。

### L5 实现：固定数据规范与参考伪代码

参考配置`LB96-reference-v1`：`(row,col)`行向下；邻居按行优先枚举`NW,N,NE,W,E,SW,S,SE`，权重1…128；`>=`；不插值、不补零，只遍历完整3×3支持。该位序是**显式教学配置**，没有把无法从提取文字读清的原图1位权矩阵冒充已逐像素核验。位序重排可一一置换直方图，但模型和查询必须同时重排；不能与库的圆周`P=8,R=1`混用。

频数使用64位整数，灰度及C用float64；先转有符号或浮点再相减。重复分位点不任意扰动：允许某些箱始终为空并登记；`searchsorted(..., side='right')`规定落在边界上的归属，两端开放尾箱覆盖训练范围外值。无有效中心时拒绝，而非生成全零“正常”特征。原始频数保留，用于扣除、G和质量诊断。

```text
EXTRACT96(image, mask, contrast_edges, mode):
    validate finite intensities, shape and frozen configuration
    H = zeros(256) if mode == LBP else zeros(256, 8)
    for center with complete valid integer 3x3 support:
        neighbors = row_major_eight_neighbors(center)
        bits = neighbors >= image[center]
        code = sum(bits[j] * 2**j for j in 0..7)
        if mode == LBP:
            H[code] += 1
        else:
            C, flag = two_group_contrast_or_flagged_zero(neighbors, bits)
            H[code, frozen_bin(C, contrast_edges)] += 1
    reject if no valid centers
    return raw H and degeneracy counts

ZERO_TO_ONE(H):
    return elementwise 1 where H == 0, otherwise H

CLASSIFY96(query_raw, models_raw, protocol):
    for model in models_raw:
        raw = model.raw
        if protocol == SET_I_EVALUATION and model.label == query.true_label:
            raw = subtract_exact_query_contribution(raw, query_raw)
        if protocol == SET_II_G_LOO and model.id == query.id:
            continue
        S = ZERO_TO_ONE(query_raw)
        M = ZERO_TO_ONE(raw)
        if protocol belongs to SET_I:
            score = KL(S / sum(S), M / sum(M))
        else:
            score = TWO_ROW_G(S, M)
        retain score, model.id, model.label
    return label of minimum (score, model.id)
```

`subtract_exact_query_contribution`必须使用确实属于模型统计域的局部中心集合；不能把从整图计算的边缘特征与独立裁片丢边界后的直方图盲减。本卡不声称原作者裁片边界实现已复现。实际部署`SET_I_INFERENCE`完全不使用`true_label`。

测试设计：位序一致性；整数邻域正仿射保码而C缩放；恒定与单侧邻域；零替换并非全箱加1；KL方向反例；G交换／同倍缩放；联合与边缘不可互换；自身排除及扣减非负；不同支持／量化版本拒绝。实际执行范围见[数值检查](#checks)。

### 五问五答：LB96

| 层级／问题 | 答题要点 | 条件、易错答案与证据 |
|---|---|---|
| L1：一张纹理图如何获得原法类别？ | 整数LBP或LBP/C计数→同一量化与零箱处理→Set I最小KL或Set II留一最小G→模型标签 | 不只答“提LBP接分类器”；Set II不是类均值模型。见L1、原文§3 |
| L2：LBP灰度不变，为何还加C？ | 模式丢差值幅度，C补局部对比，联合表保耦合 | C使正比例灰度不变性不再成立；不是方差。见L2、§2.4–2.5 |
| L3：小窗口LBP/C不稳定，先调阈值吗？ | 先查2048箱的支持和填补质量、C分位点、成像与窗口尺度 | 不能将256离散标签随意压32箱；零替换质量可能支配分布。见L3 |
| L4：Set II的G等于双向KL吗？ | 由共同分布与分别分布的似然比推导，等于样本量加权JS的2T倍 | 不是KL两方向直接相加，更不是Pearson卡方；相关像素不直接给p值。见L4 |
| L5：代码和公式都像原法，为什么排名不同？ | 核位序、整数／圆采样、C空组、箱边界、零替换时机、归一化与LOO | 查原始计数和逐模型分数；不能拿现代库默认距离证实历史复现。见L5及来源6 |

<a id="hg05"></a>
## HG05 2005 R-HOG、线性 SVM 与多尺度检测

### 任务契约与演进定位

**输入：**训练阶段为带行人标签／框的真实正图和不含行人的负图；推理为RGB图像及冻结的HOG、线性权重、金字塔、阈值和去重配置。**输出：**单窗线性分数或整图`(box, score)`列表，二者必须区分。适用前提是训练分布内大致直立、较可见的行人；不保证任意旋转、严重遮挡或装配关系正确。

前序瓶颈是仅靠稀疏关键点或边缘有无难以稳定捕捉人体轮廓。改进维度是稠密梯度方向分布、保留相对空间位置、重叠块内局部对比度归一化；线性分类让密集扫描可承受。代价是特征维数、全位置／尺度搜索和负例训练成本；遗留问题包括遮挡、姿态、背景域移与高分重复框。它不是“发明所有方向直方图”，也不是旋转对齐的SIFT。[cite:2](#source-2)

### L1 步骤：提取、训练、挖负例、检测

**默认特征链锁定原文§6开头及§6.1–6.6：**RGB、无gamma校正、无梯度前平滑、每通道`[-1,0,1]`中心差分，每像素取梯度模最大的**一个完整通道向量**；无符号0–180°的9箱，梯度模投票，位置和方向插值；16×16像素块包含2×2个8×8像素cell，块内Gaussian空间权重σ=8像素，块步幅8像素，L2-Hys，64×128检测窗，线性SVM。平方根gamma是消融选项，不是这个默认链的必选步骤。[cite:2](#source-2)

1. **训练组织。**INRIA论文选择1239张正例并左右翻转得到2478张；从1218张无人负图随机抽12180个初始负窗。原图划分先于翻转；不能把翻转后的同一人物散入训练与测试。
2. **初次训练。**将所有窗转为同一HOG向量，标签±1，使用软间隔**线性**SVM；原文默认`C=0.01`，训练器是为内存使用稍作修改的SVMlight。是否平均损失会改变C解释，不能把该数字无条件移植到别的库。
3. **困难负例挖掘。**用初次检测器穷举扫描1218张**训练负图**，收集误报窗，必要时下采样使最终描述子集合适配论文的1.7 GB内存预算。保持初始负例，并入困难负例，再训练一次最终模型。
4. **固定停止。**论文发现更多轮改善很小，采用一次增广重训，而非“直到没有负例”。每个检测器／参数组合重新挖负例，不能换HOG定义后复用旧特征。SVM求解器的收敛容差及迭代上限另存；本次未取得其修改版二进制与全部优化配置。
5. **多尺度推理。**对图像金字塔各层计算同样HOG，固定64×128窗在该层各位置扫描；计算`f=wᵀx+b`，保留超过操作阈值的候选并映射回原图坐标。尺度变化通过搜索处理，不意味着单个HOG向量尺度不变。
6. **去重。**将跨位置、跨尺度候选交给NMS，得到目标实例。CVPR论文图1明确写全位置／尺度扫描和常规NMS，但未给完整核函数、阈值和坐标离散规则；本卡L5提供**明确标注的greedy IoU参考NMS**，不称它就是原作者实现，也不把后来的mean-shift、`groupRectangles`或Soft-NMS默认值追认到2005。
7. **输出与停止。**金字塔层小到无法容纳窗口时停止；扫描有限候选，NMS候选耗尽后结束。输入过小、非有限特征或配置不匹配返回原因；阈值以上没有框是合法空结果，不强制输出最高框。[cite:2](#source-2)

**评测版本：**论文主要使用miss rate—FPPW的DET曲线，FPPW分母为受测窗口数；不能直接当NMS后每图误报数FPPI、检测AP或整件漏放率。窗口分类、全图扫描和NMS后实例评估必须分别保存。论文称`10^-4 FPPW`在其多尺度设置约对应640×480图的0.8个原始误报，不能在未知扫描步幅下套用这一换算。本卡不复述论文性能作为本地成绩。[cite:2](#source-2)

### L2 机制：为什么不是“梯度直方图加任意SVM”

方向直方图把很小的位置／朝向扰动平滑到相邻箱，同时保留较粗的人体布局；只统计全窗一个直方图会丢失头、肩、腿的位置关系。梯度模作为票权保留边缘强度，替成二值边缘有无会丢信息。最大通道梯度避免先灰度混合把某些颜色边界抵消，不等于颜色恒常。

块归一化降低局部光照、阴影和前背景对比变化的影响。一个cell参与多个重叠块，得到多种局部参照；这些条目并非可随意删除的重复副本。若先全窗统一归一化，不能模拟每块不同的局部适应。Gaussian窗口依块中心变化，同一个cell在不同块里连未归一化票权也可能不同；不能把全图每cell只存一个普通直方图视为原链的完全等价优化。[cite:2](#source-2)

困难负例让训练集覆盖滑窗真正易混淆的背景，而不是只增加容易负例的数量。它改变约束集合而非特征定义，不能用测试图挖例。线性SVM分数是间隔函数，不是概率；仓库`predict_proba`的RBF SVC另有核展开及概率校准，不是“同一个SVM更精确的版本”。多尺度扫描负责检索不同表观尺寸，NMS负责合并重复响应，两个步骤不能互相替代。[cite:2](#source-2)[cite:6](#source-6)

### L3 工程权衡：参数、失效与排障

| 参数／约束 | 来源与量纲 | 失效证据、代价与调节顺序 |
|---|---|---|
| cell／block／stride | 原默认8／16／8像素；9方向箱 | 小目标梯度不充分或肢体布局不匹配：先核目标缩放及上下文，再改cell并重训 |
| 输入窗口 | 原默认宽64、高128，包含背景上下文 | 强行拉伸非此比例目标改变几何；先明确裁剪、缩放和padding，不能借resize消除所有姿态变化 |
| 归一化 | 原L2-Hys裁剪0.2，epsilon为数值量 | 零块、局部高反光或分数漂移：查输入范围、Gaussian权重及第二次归一化 |
| SVM C | 原默认0.01；相对正则／损失权重 | 先查标签、特征顺序和训练难例，再调C；核对sum与mean损失约定 |
| 检测步幅／尺度比 | 本卡参考步幅8像素、金字塔比1.05，**不是已核实原论文扫描值** | 步幅过粗漏中心，尺度间距过粗漏尺寸；更密扫描增加时间和多重误报机会 |
| NMS IoU | 本卡参考0.5，无量纲，**不是原论文阈值** | 同人多框／相邻人误合并：先查坐标恢复及类内抑制，再改阈值；不能只看单框分数 |
| 硬负例预算 | 原按内存预算截取；参考实现固定样本ID、去重和种子 | 挖例数量、来源场景、轮次和分数分布；无预算会内存膨胀，过度重复会偏置背景 |

原论文另比较C-HOG、二阶梯度和Gaussian核SVM，不意味着默认R-HOG可以改成核SVM后仍标为同一基线。微小消融改善不用于证明未知场景最优。部署排障顺序：成像／几何→梯度方向及通道→逐块特征→线性分数→挖例与划分→金字塔→NMS→操作阈值。未训练、未跑INRIA／MIT，也未测端到端耗时。[cite:2](#source-2)

### L4 数学推导：投票、归一化与线性分类

对通道c分别计算`g_x=I(y,x+1,c)-I(y,x-1,c)`、`g_y=I(y+1,x,c)-I(y-1,x,c)`，以最大`g_x²+g_y²`选择通道，得到

\[
m=\sqrt{g_x^2+g_y^2},\qquad \theta=\operatorname{atan2}(g_y,g_x)\bmod\pi.
\]

本卡图像坐标y向下，正角朝向必须与权重训练一致。若先把x、y梯度分别取最大通道，会制造不存在的向量。设三角基`k(t)=max(0,1-|t|)`，cell中心为u、方向箱中心为θₖ，则一个块的投票可写成

\[
h_{u,k}=\sum_{p\in block}m_p\,G_{block}(p)\,
k((x_p-u_x)/a)k((y_p-u_y)/a)
k(d_\pi(\theta_p,\theta_k)/\Delta\theta),
\]

其中a=8像素，`Δθ=π/9`，方向距离以π循环。位置两轴各两邻箱、方向两邻箱，内点最多向8个组合投票；边界截去块外cell时需规定是否丢弃该部分票，不能未经说明重归一化成新权重。原文称位置与方向插值，本式是可实现的可分三角核表述。[cite:2](#source-2)

将2×2×9块向量v按固定顺序展平，L2-Hys为

\[
u=\frac{v}{\sqrt{\|v\|_2^2+\epsilon^2}},\quad
q_i=\min(u_i,0.2),\quad
z=\frac{q}{\sqrt{\|q\|_2^2+\epsilon^2}}.
\]

第二次归一化后z的分量**可以大于0.2**；0.2约束的是中间u。对正倍率v'=av，在epsilon=0且非零块时z不变；固定epsilon及量化下只是近似。全零块输出零，不能除以零后静默传播NaN。渐变、噪声和遮挡没有通用特征误差界。

宽64、高128、块16、步幅8产生

\[
n_x=1+(64-16)/8=7,\quad n_y=1+(128-16)/8=15,
\quad d=7\cdot15\cdot2\cdot2\cdot9=3780.
\]

**线性软间隔模型：**标签yᵢ∈{−1,+1}，原始形式为

\[
\min_{w,b,\xi\ge0}\frac12\|w\|^2+C\sum_i\xi_i,
\qquad y_i(w^Tx_i+b)\ge1-\xi_i.
\]

对固定w、b取最小合法松弛`ξ_i=max(0,1-y_i f_i)`，得hinge目标。忽略不可微边界`y_i f_i=1`时，活跃集合A={i:yᵢfᵢ<1}，有

\[
\nabla_wJ=w-C\sum_{i\in A}y_ix_i,\qquad
\partial_bJ=-C\sum_{i\in A}y_i.
\]

其对偶为

\[
\max_{0\le\alpha_i\le C,\;\sum_i\alpha_iy_i=0}
\sum_i\alpha_i-\frac12\sum_{ij}\alpha_i\alpha_jy_iy_jx_i^Tx_j,
\quad w=\sum_i\alpha_iy_ix_i.
\]

凸性属于固定特征下此模型，不保证数据可分或分类正确。困难负例加入新约束并重求解；对有限候选库而言可解释为逐步补充困难约束，但原文的一轮重训不保证已找到连续图像空间全部困难负例。线性推理可将支持向量合成w，单窗O(d)，而RBF通常需遍历支持向量，不可把后者称等价加速。

对参考金字塔层尺寸`W_l,H_l`，有效缩放`a_x=W_l/W_0,a_y=H_l/H_0`，层内边界框`[x,y,x+64,y+128)`反投影为`[x/a_x,y/a_y,(x+64)/a_x,(y+128)/a_y)`。记录实际整数resize后的比例，不能只累计理想1.05幂。greedy IoU抑制通过`intersection/(areaA+areaB-intersection)`比较重叠，不同尺度必须先回到同一坐标系。

设总扫描窗M、候选K、特征维数d，朴素评分O(Md)，特征可按尺度缓存；greedy NMS最坏O(K²)，排序O(K log K)。朴素梯度／块计算随各层像素和增长；提速只能在相同cell相位、块Gaussian参照、边界及归一化下声称等价。

### L5 实现：与仓库代码的差异及完整参考流程

本地`hog_features`输出64×64窗、16×16块、8步幅、9箱，按该尺寸推导为**1764维**，不是3780；它直接将输入转uint8后resize，浮点扰动若越界可能截断／回绕。`verify_candidates`使用StandardScaler与`SVC(kernel='rbf', C=10, probability=True)`，训练后给NCC候选排序，没有按SVM门限筛除，也没有原全图检测流程。[cite:6](#source-6)

外部固定`scikit-image v0.24.0`的`_hog.py`可用于核对最大通道梯度、块维数及L2-Hys，但默认3×3 cells/block、无原块Gaussian；源码文档的旧Notes说用L1，实际签名及执行默认是L2-Hys，**以固定代码为准**。未读取其全部Cython投票内核，不声明其插值与论文完全一致；也没有安装运行该库。[cite:5](#source-5)

参考配置`HG05-reference-v1`保留原默认特征与线性C=0.01；补充指定float64审核、HWC、展平次序`block_y,block_x,cell_y,cell_x,bin`、中心差分边界无支持时分量置0、方向中心从0°起每20°、epsilon=1e−5。后四项是可复现接口选择，不冒称修改版SVMlight／作者检测器内部默认。部署float32需与此基准差分。

```text
TRAIN_HG05(positive_train, negative_train_images, config):
    P = canonical_positive_windows_and_horizontal_mirrors(positive_train)
    N0 = sample_initial_negative_windows(negative_train_images, frozen_seed)
    w0, b0 = solve_linear_soft_margin_SVM(HOG(P + N0), labels, C=0.01)
    hard = []
    for image in negative_train_images:
        for window in finite_multiscale_scan(image, config):
            x = HOG_WITH_BLOCK_GAUSSIAN(window, config)
            if dot(w0, x) + b0 > 0:
                hard.append(window_id, x)
    hard = deterministic_deduplicate_and_budget_subsample(hard)
    w, b = solve_linear_soft_margin_SVM(HOG(P + N0) + hard, labels, C=0.01)
    return frozen(config, w, b, data_ids, solver_status)

DETECT_HG05(image, model):
    candidates = []
    for layer in pyramid_until_smaller_than_64_by_128(image, ratio=1.05):
        for x, y in valid_window_grid(layer, stride=8):
            feature = HOG_WITH_BLOCK_GAUSSIAN(layer_window(x,y), model.config)
            score = dot(model.w, feature) + model.b
            if score >= model.threshold:
                candidates.append(map_box_by_actual_layer_size(x,y,64,128), score)
    candidates = sort_by_descending_score_then_stable_id(candidates)
    kept = []
    while candidates is not empty:
        best = pop_first(candidates)
        kept.append(best)
        candidates = [c for c in candidates if IoU(best.box,c.box) <= 0.5]
    return kept
```

`hard score>0`、难例去重政策及扫描参数是参考协议；原文“false positives”未提供所有阈值／去重代码。阈值须在验证集确定后冻结，挖例不得扫描测试负图。SVM训练失败不能输出旧权重而标新版本；需保存求解状态、配置指纹和数据ID。

参考NMS用连续半开框、严格`IoU>0.5`抑制、同分按ID，先恢复原图坐标再抑制，零面积框拒绝。它覆盖节点要求的**去重机制**，不补造原作者NMS参数。若目标是逐数复现2005最终检测框，仍需取得当时扫描／NMS实现；这是保留的L5缺口。

测试设计：水平／垂直梯度与通道选择；无符号方向环绕；位置和方向投票质量；零块、强单分量及两次归一化；3780／1764维；线性分数合并等价；hinge梯度差分；训练／测试负图隔离；同框跨尺度恢复、相邻人NMS误合并；扫描空层和合法空检测。文末执行只核限定算术，不是完整HOG提取、SVM拟合或检测实测。

### 五问五答：HG05

| 层级／问题 | 答题要点 | 条件、易错答案与证据 |
|---|---|---|
| L1：从正负图到最终框还缺哪些步骤？ | 同一HOG→初训线性SVM→训练负图挖误报→一次增广重训→金字塔扫描→坐标恢复→NMS | 不能只返回窗口分类；RBF候选重排不能覆盖原法。见L1、原文§3–4 |
| L2：重叠块为何不是浪费？ | 同一局部轮廓在不同块参照下归一化；局部支持和Gaussian权重可能不同 | 不能用全局归一化或每cell仅一份普通直方图无损替换。见L2、§6.4 |
| L3：误报多，增加随机负窗还是改NMS？ | 先区分背景高分与同目标重复框；前者查训练难例／特征，后者查跨尺度坐标及NMS | 不能用测试集挖例；FPPW不等于FPPI。见L3 |
| L4：3780从哪来？Hys输出为何可大于0.2？ | 7×15块×36维；0.2限制第一次归一化后的中间量，第二次归一化可放大 | 64×64是1764维；不是每个最终分量都≤0.2。见L4 |
| L5：为何同名HOG库不能直接换？ | 核对梯度边界、bin中心、插值、Gaussian、展平和epsilon；权重必须与特征协议一致 | 源码Notes可能滞后；先逐块差分，再测分数和框。见L5及来源5／6 |

<a id="lb02"></a>
## LB02 2002 Uniform、旋转不变、多尺度与原始分类

### 任务契约与演进定位

**输入：**灰度ROI、有效域、`(P,R)`列表、编码类型、可选VAR量化器及带标签模型库；另指定`BRODATZ_CLASS_PROTOTYPE`或`OUTEX_3NN`。**输出：**各尺度分布、候选模型多尺度对数似然、类别与失败原因。多尺度是并列采样支持，不自动选择真实物理尺度；本卡不覆盖后来尺度自适应、LBP-HF、人脸分块或深度分类器。

1996基本LBP固定整数邻域且旋转后标签改变；2002通过圆对称采样、循环等价类、uniform压缩及更细角度采样改善旋转描述。多半径捕捉不同纹理尺度，VAR补幅值；代价是插值、更多有效边界、稀疏联合统计和不变性带来的信息损失。对光照和旋转的鲁棒不等于对投影、尺度、视角和反光全部不变。[cite:3](#source-3)

### L1 步骤：编码与版本完整分类协议

1. **冻结采样。**对P个等角邻点，`(row,col)=(r−R sin(2πp/P), c+R cos(2πp/P))`；非整数位置双线性插值；与中心作`>=`比较。固定首点、角度方向和坐标精度。
2. **明确映射。**basic保存2ᴾ个模式；ri取全部循环位移的最小码；uniform条件为环上翻转数U≤2；riu2对uniform取1的个数，非uniform统一映到P+1。不能把“uniform保留方向”的u2与riu2混称。
3. **可选VAR。**计算P个邻居相对**邻居均值**的总体方差，分母P，不包括中心、不用P−1。每个尺度独立从模型训练分布学习等频VAR箱；同尺度LBP/VAR按逐中心联合计数。
4. **每尺度独立归一化。**riu2有P+2箱；联合表为`(P+2)×B_VAR`，箱数和边界写入配置。有效支持边缘取`ceil(R)`；不同尺度各有自己的有效中心数，先各自归一化，不能直接合并原始计数让小半径自动获得更大权重。
5. **原对数似然。**对每个模型逐尺度算`L(S,M)=ΣS_b log M_b`，取大为好；多尺度分数是**各尺度L相加**，不是跨尺度联合大表，也不是特征拼接后接SVM。
6. **Brodatz实验#1。**每类将指定训练裁片的原始直方图相加后归一化，形成一个类模型；查询与全部16个类原型比较，取最大分数类。四角度版把四种训练角的裁片一起并为一个模型，不能默认保留四个模型后取最大。
7. **Outex实验#2。**保存每张训练样本的独立模型，按分数取最相似的三个模型，作**3-NN多数投票**。不是把20张样本合成一个类模型，也不是论文全程都1-NN。三个不同类时原文未详述破局，本卡按三者中最高相似模型的类别破局；同分按稳定ID，是明示参考选择。
8. **停止／拒绝。**遍历有限尺度和模型后决策；零有效支持或全候选分数−∞时返回UNKNOWN并报告原因。原协议是闭集识别，本卡未核到原通用拒识门，不能把L解释成已校准类别概率。[cite:3](#source-3)

#### 原实验数据与分类组织

| 协议 | 训练及模型单位 | 测试与预处理 | 决策 |
|---|---|---|---|
| Brodatz四训练角 | 16类，每类8张256×256源图中的第一张作训练；生成180×180旋转图；0°、30°、45°、60°各分121个不重叠16×16块，共7744训练块，合成16类原型 | 其余7张源图×6测试角20°、70°、90°、120°、135°、150°，共672张180×180图；数字旋转采用双线性，0°／90°补人工模糊模拟插值 | 最大逐尺度对数似然之和 |
| Brodatz单训练角 | 每次一个指定角度的121裁片／类，合成16类模型 | 其余9角×7测试源图×16类=1008样本；依次改变训练角并报告各次结果 | 同上；不混入四角度模型 |
| Outex TC_00010 | 24类，inca光源、0°，每类20张128×128样本，共480个独立模型 | 同inca其余8角，共3840样本 | 3-NN多数投票 |
| Outex TC_00012 | 同一480个inca／0°模型 | tl84问题000和horizon问题001，各9角、各4320样本，分别统计 | 3-NN多数投票 |

表中数据来自正式版§3.1–3.2，不是本仓库数据或运行结果。Outex该实验用100 dpi；先校正非方像素，再转灰度，提取居中的5×4个128×128非重叠块；**每块**均值归一到128、标准差20，这与LB96 Set I的**整图**256／40不同。原文式(15)文本层末项显示`0.114R`，与声称的标准RGB灰度转换不一致；本卡明确采用`0.299R+0.587G+0.114B`为实现修正，不声称已确认原实验究竟用了印式还是标准式。没有正式勘误证据。[cite:3](#source-3)

Outex前置参考可分性检查是480参考样本按每类10／10随机划分100次、3-NN；它不是TC_00010／00012的正式测试划分，不能混入正式评测。每个训练模型保存全部尺度的同一图像身份，不能每尺度各选最近模型后再把分数相加，那将制造不存在的“组合模型”。

**历史实现尚未闭合的细节：**已读§2.5给VAR等频量化和平均约10条目／箱的经验原则，但未可靠取得各表实际使用的完整B_VAR配置；正文未明确零模型箱如何平滑及全部平票规则。因此参考模式采用下述严格零支持／明确破局，或显式登记平滑变体；不自动把1996“零置1”移植为2002原实现，也不据本卡伪代码承诺重现论文每个数字。

### L2 机制：uniform、旋转和多尺度的不同角色

循环移位改变起始方向而不改变环上模式，ri把这些码视为同一等价类；riu2进一步保留只有一段连续1或常量的uniform模式，用“亮侧长度”表示，其他碎片模式汇入一个杂项。保留方向的u2则仍区分各段起点，判别力和旋转不变性不是同一个目标。

P增大是更细**角采样**，R增大是更大**空间支持**；在固定小R上盲增P，插值点会高度相关，并不生成更多独立像素。多尺度保留不同支持的模式统计，不使一根固定尺寸纹理在任意缩放后得到相同特征。合并稀有非uniform模式降低估计噪声，但也丢掉可能有用的细结构，不能保证所有材质都受益。[cite:3](#source-3)[cite:4](#source-4)

VAR对邻点循环置换不变，补充被符号编码丢掉的幅度；正仿射变换使VAR乘a²，所以LBP/VAR联合表一般不具严格灰度尺度不变性。这里“互补／orthogonal”是性质互补，不是证明两个随机变量独立。逐尺度L相加是以可估计性换取对跨尺度依赖的忽略；原文明确承认独立假设一般不成立。[cite:3](#source-3)

**数字图像上的重要限定：**对固定采样值的保序证明并不自动扩展为“先对整图作任意严格递增变换，再双线性插值仍保码”。非线性变换与插值一般不交换。端点0、2的中点插值为1，中心1.1，比较为0；先平方再插值，中点变2、中心变1.21，比较变1。平方在本例非负域严格递增，翻位仍发生。该例是确定性反例，不是实验数据；正仿射变换与线性插值可交换。任意角度旋转也会带来重采样和有效支持变化，循环位移的精确不变性不等于整图任意角度逐位相同。

### L3 工程权衡：尺度、维数与统计可靠性

| 参数／选择 | 依据、量纲 | 失效诊断与顺序 |
|---|---|---|
| (P,R) | 原实验(8,1)、(16,2)、(24,3)；P点数、R像素；原文称合理起点而非最优保证 | 先标定纹理尺寸／相机距离，再观察各半径分数；不只盲加P |
| basic／u2／ri／riu2 | 角度是否业务变量决定保留多少方向信息 | 正反装／方向检验若需要朝向，不应通过riu2主动删掉 |
| VAR箱数 | 原等频，按可用样本与箱数控制稀疏；本卡联合参考B_VAR=16是工程配置 | 查看重复分位点、每箱质量、离群VAR和空支持；不能把16称原各表固定值 |
| 单／多尺度 | 原最多三尺度，分数等权相加 | 劣尺度可能拖累；训练内消融，不能靠验收标签选尺度 |
| 模型单位 | Brodatz类原型／Outex样本3-NN | 数量或合并方式变化会变决策；先查模型ID与类数，再调k |
| 灰度／插值 | float64审核、冻结采样精度与边界 | 近等值位、旋转边缘、饱和；先看邻点差值再看整体相似度 |

P=8时basic256箱、ri36个等价类、u2为59箱、riu2为10箱，数值编码与连续箱索引仍需区分。P=16／24时riu2为18／26箱，三尺度分开统计共54箱；若错误地估计三尺度联合分布会有4680箱，还未乘VAR轴。维数降低不是无损压缩，应对同源与跨光源测试分别报告。无真实数据时，只能检查这些结构不变量，不能推断分类率。

### L4 数学推导：循环群、uniform计数与评分

**圆采样：**双线性插值对四个格点使用非负权重，其和为1。在完整支持上，常量与二维仿射强度场精确重建；非线性强度映射、非仿射图形或边界补值不享有该保证。记位串b，编码和循环最小值为

\[
LBP_{P,R}=\sum_{p=0}^{P-1}b_p2^p,\qquad
LBP^{ri}_{P,R}=\min_{0\le k<P}\operatorname{ROR}(LBP_{P,R},k).
\]

循环群作用把码分成轨道，取每轨道最小代表不受再循环移位影响；输出代表值并非连续0…35。P=8的轨道数由Burnside公式得到

\[
\frac18\sum_{k=0}^{7}2^{\gcd(8,k)}
=\frac{256+2+4+2+16+2+4+2}{8}=36.
\]

uniform翻转数及riu2定义为

\[
U(b)=|b_{P-1}-b_0|+\sum_{p=1}^{P-1}|b_p-b_{p-1}|,
\quad
LBP^{riu2}_{P,R}=\begin{cases}\sum_pb_p,&U\le2,\\P+1,&U>2.\end{cases}
\]

U总为偶数，因为环上0→1和1→0次数相等。除全0、全1外，每个uniform串恰有一段连续1；长度有P−1种、起点有P种，故保留方向的uniform串共有`P(P−1)+2`种，再加非uniform杂项，u2共有`P(P−1)+3`箱。旋转合并后只有长度0…P的P+1种，再加杂项，riu2为P+2箱。注意原文“P+1 uniform patterns”指旋转归并后的类型，不是全部有方向二进制串只有P+1个。[cite:3](#source-3)

**VAR：**

\[
\mu=\frac1P\sum_pg_p,\qquad VAR_{P,R}=\frac1P\sum_p(g_p-\mu)^2
=\frac1P\sum_pg_p^2-\mu^2.
\]

循环置换保持和与平方和，因此VAR不变；`g'=ag+b`给`VAR'=a²VAR`。这是确定邻域描述量，不是用来估计未知总体方差的无偏统计量，不能自行换P−1。计算建议两遍中心化，避免大基值下的二阶矩相消；常量VAR数学值为0，不是NaN。

**原分类式(12)–(14)：**令S、M为各箱概率，取自然对数，

\[
L(S,M)=\sum_{b:S_b>0}S_b\log M_b,\qquad
2D(S\Vert M)=2\sum_bS_b\log S_b-2L(S,M).
\]

固定S时前项与M无关，所以最大L与最小KL同序。2002文中式(13)的G是这里的单查询对模型拟合形式，**不是LB96 Set II由两行频数共同估计分布的G**；同一个字母不能消掉两个协议的差别。模型有零且查询同格正时L=−∞，双方零格贡献0；若采用平滑必须重新归一化并标版本。[cite:3](#source-3)

对N个尺度原文取`L_N=Σ_n L(S_n,M_n)`。若联合分布真能写成边缘乘积，则

\[
D(S_{XY}\Vert M_{XY})
=\sum_{x,y}S_X(x)S_Y(y)\log\frac{S_X(x)S_Y(y)}{M_X(x)M_Y(y)}
=D(S_X\Vert M_X)+D(S_Y\Vert M_Y).
\]

真实同图不同半径常有相关性，故按边缘评分相加只是近似，不能说它等价于充分建模跨尺度联合。每尺度归一化后相加等权；任意逐尺度样本量权重不保持原法。

**归一化拼接须区分查询与模型两边。**令\(\widetilde S=\operatorname{concat}(S_1,\ldots,S_N)/N\)、\(\widetilde M=\operatorname{concat}(M_1,\ldots,M_N)/N\)，各尺度概率和均为1，则

\[
L(\widetilde S,\widetilde M)
=\sum_{n,b}\frac{S_{n,b}}N\log\frac{M_{n,b}}N
=\frac{L_N}{N}-\log N.
\]

仅查询除以N、模型不除时才得到\(L_N/N\)，此时拼接模型总质量是N，不是一个归一化概率分布；两边均不除的拼接代数和为\(L_N\)。固定同一查询、相同N与尺度配置时，两边归一化带来共同正缩放和常数平移，故保持模型排序和3-NN邻居；但不能说“只乘1/N”。若工程拒识条件为\(L_N\ge\tau\)，新阈值必须为\(\widetilde\tau=\tau/N-\log N\)，**不可仅缩放为τ/N**。不同N之间也不能无条件比较原始分数；此阈值换算不是论文给出了通用拒识门。零支持时按−∞约定处理。[数值检查](#checks)用两尺度完整概率块检验该恒等式及仅缩放阈值的反例。

Outex 3-NN是在**模型图像层面**先按同一多尺度总分取前三，再投票。它不等于把各类前3个分数相加，亦不等于各尺度分别投票。模型数K、有效像素数A、尺度数N、维数Bₙ时，提取O(AΣPₙ)，评分O(KΣBₙ)；前三可线性维护，不必全排序。查表需要2ᴾ项；P=24、每项1字节约16 MiB，本卡直接计U及popcount可避免指数表，但需验证映射等价。

未提供对任意数字旋转、光源谱变化和尺度变化的分类误差界；有限循环群不变性、插值误差、统计估计误差和工业验收精度是四个不同层面。

### L5 实现：数据布局、固定库差异与参考伪代码

参考`LB02-reference-v1`明确：float64邻点插值，三尺度按(8,1)／(16,2)／(24,3)排列，环首点为右侧，row向下时负sin向上，丢弃不完整支持；riu2分量按0…P+1排列；严格零支持策略；3-NN平票取三者中最近模型标签。可选`LB02-reference-var16-v1`使用每尺度16个训练等频VAR箱，联合数组行是riu2、列是VAR；该16箱配置不是已核实原表参数。

固定`scikit-image v0.24.0/_texture.pyx`：`uniform`实际对应riu2，`nri_uniform`对应保留方向的u2，`ror`给循环最小代表，`default`是圆采样而非LB96整数邻域。其采样偏移四舍五入到5位小数，边界常量0外插；必须裁去边缘才能比较本卡有效支持。`var`用二阶矩差，零时返回NaN，与本卡数学0及稳定两遍计算不同。`uniform`内只累计相邻P−1条边的changes，看似少首尾项，但二值环U必为偶数：开链changes≤2恰与环U≤2等价，不能据此直接判该判定为bug；文末穷举核查这个等价性。[cite:5](#source-5)

```text
EXTRACT02(image, mask, scales, variant, variance_quantizers):
    result = []
    for P, R in scales:
        H = zeros(P+2) or zeros(P+2, frozen_VAR_bins(P,R))
        for center with complete valid bilinear support:
            g = sample_circle_bilinear(image, center, P, R)
            bits = g >= image[center]
            U = abs(bits[P-1]-bits[0]) + sum(abs(bits[p]-bits[p-1]))
            code = sum(bits) if U <= 2 else P+1
            if variant == RIU2:
                H[code] += 1
            else:
                mean = sum(g)/P
                VAR = sum((g-mean)**2)/P
                H[code, frozen_bin(VAR, variance_quantizers[P,R])] += 1
        reject if sum(H) == 0
        result.append(raw=H, probability=H/sum(H))
    return result

BUILD_MODELS02(training_samples, protocol):
    fit VAR edges only from designated model samples if needed
    extract raw per-scale histograms with the frozen edges
    if protocol == BRODATZ_CLASS_PROTOTYPE:
        sum raw histograms by class and scale, then normalize each
    else:
        preserve one normalized multiscale model per training sample
    return models with labels and stable IDs

CLASSIFY02(query, models, protocol):
    for model in models:
        score = 0
        for scale in fixed_scale_order:
            score += sum(S[b] * log(M[b]) for b where S[b] > 0)
        record score, model.id, model.label
    if every score == negative_infinity:
        return UNKNOWN_ZERO_SUPPORT
    rank by descending score then stable ID
    if protocol == BRODATZ_CLASS_PROTOTYPE:
        return first model label
    nearest = first three finite-score models
    reject if fewer than three
    return majority label, or nearest model label on three-way tie
```

伪代码中的`log(0)`要显式转−∞，不能真调用产生未处理异常；全负无穷的拒绝以及少于三个有限候选的出口是本卡工程政策。若选择平滑，单独保存方式、总质量与量化器，不改成现代chi-square或RBF后仍贴原法标签。

测试设计：256串／循环旋转映射；u2、ri、riu2箱数；首尾翻转与库判定等价；常量、斜坡与双线性；非线性变换先插值反例；VAR平移／缩放及中心排除；每尺度支持与概率和；同一模型跨尺度聚合；类原型与3-NN可产生不同决策；空箱、平票、训练图隔离。文末只执行可独立复核的有限测试，不运行Outex数据协议。

### 五问五答：LB02

| 层级／问题 | 答题要点 | 条件、易错答案与证据 |
|---|---|---|
| L1：原文是不是所有实验都最近类中心？ | Brodatz裁片按类合并后比较类原型；Outex保留480样本模型并3-NN；跨尺度先对同一模型加L | 不能改成全程1-NN、类均值或RBF。见L1、原文§3.1–3.2 |
| L2：uniform为什么有59箱也有10箱？ | P=8保留方向u2为58+1；riu2按亮段长度合并为9+1 | `uniform`库名称可能指riu2；uniform不是每个像素灰度一样。见L2／L4 |
| L3：把P加大能解决尺度变化吗？ | P控制角采样，R控制空间支持；要核像素尺度、纹理周期与有效统计量 | 固定小R上重复插值不增加独立证据；多尺度不是尺度不变。见L3 |
| L4：为什么加VAR后不再严格灰度不变？多尺度L为何相加？ | VAR在正倍率下乘a²；边缘L相加来自独立乘积分布的KL可加性，实际是近似 | 不把互补当独立；不把数字图像任意非线性变换保码当定理。见L4 |
| L5：库uniform少算首尾边一定错吗？ | 二值环翻转总为偶数，开链≤2与环≤2等价；用穷举而非凭外观断言 | 仍须核坐标舍入、边界零填及var零值NaN；见L5、来源5、检查器 |

<a id="checks"></a>
## 确定性数值检查与验证边界

以下完整输入都是**数学单元测试／反例**，不是合成研究数据、论文准确率或真实视觉实验。检查器只用Python标准库，可从本文件提取后在内存运行；不创建脚本、图像、模型或下载文件。给定两箱`s=(0.75,0.25), m=(0.5,0.5)`，KL约0.130812035941，反向约0.143841036226；给定频数`(3,1),(1,3)`，双向G约2.092992575058。原256／2048箱无需都非零才可检验评分代数。

```python
import math
from collections import Counter

checks = 0

def check(ok):
    global checks
    assert ok
    checks += 1

def close(a, b, tol=1e-10):
    check(abs(a-b) <= tol)

def probs(v):
    return [x/sum(v) for x in v]

def kl(s, m):
    return sum(x*math.log(x/y) for x, y in zip(s, m) if x)

def ll(s, m):
    if any(x > 0 and y == 0 for x, y in zip(s, m)):
        return -math.inf
    return sum(x*math.log(y) for x, y in zip(s, m) if x)

def gtest(a, b):
    na, nb = sum(a), sum(b)
    total = na+nb
    return 2*sum(v*math.log(v/(n*(x+y)/total))
                 for x, y in zip(a, b)
                 for v, n in ((x, na), (y, nb)) if v)

def bits(code, p):
    return [(code >> i) & 1 for i in range(p)]

def uniformity(b):
    return sum(abs(b[i]-b[i-1]) for i in range(len(b)))

def rotate(code, p):
    return (code >> 1) | ((code & 1) << (p-1))

def ri(code, p):
    values = []
    for _ in range(p):
        values.append(code)
        code = rotate(code, p)
    return min(values)

def riu2(code, p):
    b = bits(code, p)
    return sum(b) if uniformity(b) <= 2 else p+1

def var(v):
    mean = sum(v)/len(v)
    return sum((x-mean)**2 for x in v)/len(v)

s, m = [0.75, 0.25], [0.5, 0.5]
close(kl(s,m), 0.13081203594113697)
close(kl(m,s), 0.14384103622589042)
check(kl(s,m) != kl(m,s))
close(kl(s,m), sum(x*math.log(x) for x in s)-ll(s,m))
close(kl(s,s), 0)
close(gtest([3,1],[1,3]), 2.0929925750581915)
close(gtest([3,1],[1,3]), gtest([1,3],[3,1]))
close(gtest([6,2],[2,6]), 2*gtest([3,1],[1,3]))
close(gtest([3,1],[6,2]), 0)
a, b = [4,1], [1,2]
z = probs([x+y for x,y in zip(a,b)])
close(gtest(a,b), 2*(sum(a)*kl(probs(a),z)+sum(b)*kl(probs(b),z)))
check([1 if x == 0 else x for x in [0,2,0]] == [1,2,1])
check([1 if x == 0 else x for x in [0,2,0]] != [1,3,1])
neighbors = [12,8,11,7,6,5,4,3]
code = sum((x >= 10)*2**i for i,x in enumerate(neighbors))
check(code == 5)
check(sum((3*x+7 >= 37)*2**i for i,x in enumerate(neighbors)) == code)
high = [x for x in neighbors if x >= 10]
low = [x for x in neighbors if x < 10]
contrast = sum(high)/len(high)-sum(low)/len(low)
close(contrast, 6)
close(sum(3*x+7 for x in high)/len(high)-sum(3*x+7 for x in low)/len(low), 3*contrast)
check(256*8 == 2048 and 14*14 == 196)

check(sum(uniformity(bits(c,8)) <= 2 for c in range(256)) == 58)
check(len({ri(c,8) for c in range(256)}) == 36)
check(len({riu2(c,8) for c in range(256)}) == 10)
check(8*(8-1)+3 == 59)
check(all(riu2(c,8) == riu2(rotate(c,8),8) for c in range(256)))
check(all(ri(c,8) == ri(rotate(c,8),8) for c in range(256)))
for p in range(2,11):
    check(all((sum(abs(b[i]-b[i-1]) for i in range(1,p)) <= 2)
              == (uniformity(b) <= 2)
              for c in range(1 << p) for b in [bits(c,p)]))
check(riu2(0,8) == 0 and riu2(255,8) == 8 and riu2(85,8) == 9)
check(10+18+26 == 54 and 10*18*26 == 4680)
check((0+2)/2 < 1.1 and (0**2+2**2)/2 > 1.1**2)
close(var([1,3,5,7]), 5)
close(var([12,14,16,18]), 5)
close(var([3,9,15,21]), 45)
close(var([7,7,7,7]), 0)
close(ll([0,1],[0,1]), 0)
check(ll([1,0],[0,1]) == -math.inf)
query = [[0.8,0.2], [0.1,0.9]]
model = [[0.6,0.4], [0.3,0.7]]
total_ll = sum(ll(x,y) for x,y in zip(query,model))
close(total_ll, ll(query[0]+query[1], model[0]+model[1]))
n_scales = len(query)
query_concat = [v/n_scales for block in query for v in block]
model_concat = [v/n_scales for block in model for v in block]
close(sum(query_concat), 1)
close(sum(model_concat), 1)
normalized_ll = ll(query_concat, model_concat)
close(normalized_ll, total_ll/n_scales-math.log(n_scales))
check(not math.isclose(normalized_ll, total_ll/n_scales))
close(ll(query_concat, model[0]+model[1]), total_ll/n_scales)
threshold = -1.2
normalized_threshold = threshold/n_scales-math.log(n_scales)
check(total_ll >= threshold)
check((total_ll >= threshold) == (normalized_ll >= normalized_threshold))
check((total_ll >= threshold) != (normalized_ll >= threshold/n_scales))
print(f'two-scale identity: L_N={total_ll:.15f}, normalized={normalized_ll:.15f}, threshold={normalized_threshold:.15f}')
close(sum(kl(x,y) for x,y in zip(query,model)),
      sum(sum(v*math.log(v) for v in x) for x in query)-total_ll)
labels = ['B','A','A']
check(Counter(labels).most_common(1)[0][0] == 'A' and labels[0] == 'B')

check(((64-16)//8+1)*((128-16)//8+1)*36 == 3780)
check(((64-16)//8+1)**2*36 == 1764)
def hys(v, eps=1e-5):
    n = math.sqrt(sum(x*x for x in v)+eps*eps)
    clipped = [min(x/n,0.2) for x in v]
    n2 = math.sqrt(sum(x*x for x in clipped)+eps*eps)
    return [x/n2 for x in clipped]
check(hys([0,0]) == [0,0])
check(hys([1,0])[0] > 0.2)
close(sum(x*x for x in hys([3,4],0)), 1)
close(hys([3,4],0)[0], hys([30,40],0)[0])
close(sum(a*b*c for a in [0.25,0.75] for b in [0.4,0.6] for c in [0.3,0.7]), 1)
channels = [(3,4),(0,6),(5,0)]
check(max(channels,key=lambda g:g[0]**2+g[1]**2) == (0,6))
def objective(v):
    w, bias = v[:2], v[2]
    data = [([1,2],1), ([-1,1],-1)]
    return 0.5*sum(x*x for x in w)+0.01*sum(
        max(0,1-y*(sum(a*b for a,b in zip(w,x))+bias)) for x,y in data)
v = [0.2,-0.1,0.05]
analytic = [0.18,-0.11,0]
for i in range(3):
    vp, vm = v.copy(), v.copy()
    vp[i] += 1e-6
    vm[i] -= 1e-6
    close((objective(vp)-objective(vm))/2e-6, analytic[i], 1e-8)
def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])
    return inter/(area-inter)
close(iou([0,0,64,128],[0,0,64,128]), 1)
close(iou([0,0,64,128],[64,0,128,128]), 0)
check([v/0.5 for v in [8,16,72,144]] == [16,32,144,288])
close(iou([0,0,3,2],[1,0,4,2]), 0.5)
check(not iou([0,0,3,2],[1,0,4,2]) > 0.5)
print(f'PASS: {checks} deterministic checks; no image classification or detector training')
```

**实际执行记录：**本轮使用仓库已有`.venv\Scripts\python.exe -B`、Python 3.13.14，从本文件提取上面的唯一Python代码块在隔离字典中内存执行，输出`PASS: 68 deterministic checks; no image classification or detector training`。保留原60项并增加8项两尺度归一化／恒等式／阈值检查；实测`L_N=-1.033323375365077`，两边归一化拼接得`-1.209808868242484`，恰为`L_N/2-log(2)`。τ=−1.2时，正确新阈值为`-1.293147180559945`，原／新规则均接受；只除2得到−0.6却拒绝。结构检查通过：3张卡、15段L1–L5正文、15组问答、6个来源ID、50处本地链接（含锚点）及代码围栏／公式配对。执行不依赖NumPy、OpenCV、scikit-image或SVMlight。集合型断言包含P=8的256码穷举及P=2…10的开链／环判定核对，断言数不是真实图像样本数。本轮外围全树哈希比较因跨调用变量不保留而未完成，不记为通过；文档编辑调用仅写本文件及`edge_evolution_extensions.md`，未编辑源码、旧卡或台账。

**未执行：**整图LBP/C与圆采样提取、原始数据划分复现、库数值逐像素对齐、MIT／INRIA训练、困难负例挖掘实跑、原作者NMS复刻、Outex及Brodatz分类、吞吐量或现场验收。原位权／C退化／VAR量化参数／2002空箱／NMS细节保留为深度复现工作，不由有限算术消除。

## 来源、固定版本与获取层级

来源等级固定为**P0原始论文／官方源码、P1作者或同行评审二手说明、P2专业社区、P3一般参考**；与项目缺口优先级P0/P1/P2不是同一个字段。获取深度固定为**B书目、A摘要、T实际读到方法／源码、F获取失败**。T不等于全实验审计。以下均为2026-11-06访问，原实验具体协议主要是`[SINGLE-SOURCE]`：同文镜像、作者自述及引用它的库不伪装成三个独立复现实验。

<a id="source-1"></a>
**[cite:1] LB96，P0，B+A+T（部分正文）。** [1996原论文教学镜像](https://rendicahya.lecture.ub.ac.id/files/2018/03/A-Comparative-Study-of-Texture-Measures-With-Classification-Based-on-Feature-Distributions.pdf)，固定识别为Pattern Recognition 29(1)，51–59，DOI `10.1016/0031-3203(95)00067-4`。网页解析实际读到pp.51–53及p.54入口：§2.4基本LBP、§2.5互补特征／8箱、§3式(1)(2)、零箱置1、§4 Set I校正与扣除查询。没有读清图1原位权矩阵，也没有取得后半篇全部表格／Set II采集清单，因此不报告完整实验准确率或字节哈希。[出版入口](https://www.sciencedirect.com/science/article/pii/0031320395000674)出现验证页，记F；镜像本地HTTPS下载遇TLS故障，不能将其记为完整PDF已下载。网页正文足以核对两套分类主规则，不足以声明原二进制实现复现。

<a id="source-2"></a>
**[cite:2] HG05，P0，B+A+T。** [作者站2005论文](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)、[Berkeley同文镜像](https://people.eecs.berkeley.edu/~efros/courses/AP06/Papers/dalal-cvpr-05.pdf)。网页读§1–5；镜像PDF在内存解析8页，重点补读§6.1–6.6默认参数、方向投票、块Gaussian、L2-Hys及C=0.01。镜像字节SHA-256：`eb2bad4bc1e30c9a467a135100ba2be4d12fd8736675af0268e98ebd179c46f0`。图1／§4支持多尺度扫描、NMS角色、训练数据规模与一次增广重训；不支持将本卡greedy IoU=0.5或尺度比1.05说成原默认。未获得原修改版SVMlight和检测器源码。

<a id="source-3"></a>
**[cite:3] LB02，P0，B+A+T。** [MSU课程镜像的2002正式版](https://cse.msu.edu/~rossarun/BiometricsTextBook/Papers/Face/Ojala_LBP_PAMI02.pdf)，17页，首页971、卷24期7；在内存解析并重点读pp.973–983的式(7)–(15)、§2.3–2.6与§3.1–3.2。字节SHA-256：`f86b5f148689c833d95ccbd5679ddc525c70d6d7c9c5470b51030a73089b315f`。[ACM书目入口](https://dl.acm.org/doi/10.1109/tpami.2002.1017623)提供出版身份。[Stanford作者稿镜像](https://vision.stanford.edu/teaching/cs231b_spring1415/papers/lbp.pdf)首页为`Ref: TPAMI 112278`，网页仅取到前部；不把其稿页码当正式版页码。正式版优先于作者稿OCR误把中心g_c写成g_0的式(7)。式(15)RGB末项、VAR实际箱数、空箱与平票细节的限制见LB02，不编造勘误或作者参数。

<a id="source-4"></a>
**[cite:4] 作者回顾，P1，T。** Matti Pietikäinen，2010，[Scholarpedia: Local Binary Patterns](https://www.scholarpedia.org/article/Local_Binary_Patterns)，DOI `10.4249/scholarpedia.9775`，本次显示revision **203567**。核对3×3模式、圆周双线性、u2的58+1、直方图归一化与空间信息丢失。该页是持续修订综述，不把最新文字都当2010原版，也不以其参考列表里错误的1996卷期覆盖论文首页。

<a id="source-5"></a>
**[cite:5] 固定官方源码，P0（实现行为），T。** `scikit-image` **v0.24.0**：[LBP `_texture.pyx`](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/feature/_texture.pyx)、[HOG `_hog.py`](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/feature/_hog.py)。实际读采样、映射、VAR退化、梯度、块维数与归一化分支；未声称检查全部编译依赖或运行过该tag。它们是现代实现对照，不是1996／2002／2005原实验代码。LBP开链判定用数学穷举补证，HOG文档旧Notes与函数默认差异单列。

<a id="source-6"></a>
**[cite:6] 仓库源码与旧卡，T，工作树版本。** [hog_svm_matching.py](../../ml/hog_svm_matching.py)、[learning_components.md](learning_components.md)、[remaining_domains.md](remaining_domains.md)、[学习规范](../algorithm_learning_standard.md)、[覆盖口径](../evolution_coverage.md#判定与计数口径)。本次读取现有工作树，没有将未提交源码冒称某个固定commit。只静态核对HOG配置、RBF／概率排序及数据组织，没有运行它们或改变原文件。

获取限制补记（既有阅读记录）：Oulu旧下载入口返回空内容；部分出版／ResearchGate入口返回安全验证页，未绕过验证且未取其作为正文证据。Stanford本地下载发生连接失败，随后改读MSU公开正式版；此前MSU的Python证书链请求失败后使用正常验证证书的系统请求成功，未关闭TLS验证。论文只在内存读取，未写PDF、依赖或临时脚本。

**2026-11-06本轮链接复核：**本文件10个不同外部URL全部请求：HG05作者站、Berkeley镜像、LB02 MSU正式版及两份固定scikit-image源码共5个HTTP 200；ACM与ScienceDirect为HTTP 403；LB96教学镜像与Stanford作者稿镜像为TLS `UNEXPECTED_EOF_WHILE_READING`；Scholarpedia为证书过期的`CERTIFICATE_VERIFY_FAILED`。这些失败不改称404或内容为空；保留来源入口及失败原因，未关闭证书校验或绕过访问限制。既有正文阅读记录不冒称本轮重新取得，HTTP 200也不等于重审全部正文或复现实验。

## 覆盖判定与待续工作

依据覆盖台账“已有”要求的是**具体方法任务、核心步骤／机制、五层正文及问答**，不要求五层全部已核验或完成整套历史复现。本文件支持以下转换建议，但不直接修改台账、索引或领域总览：

| 任务／节点 | 本卡承接证据 | 建议 | 仍保留的深度工作 |
|---|---|---|---|
| G06／06-02 | LB96基本模式、互补C、联合表、Set I KL和Set II留一双向G均有原文依据及五层答5 | **可由部分转已有** | 原图位权／等值／C退化、Set II全部实验清单、逐数复现与工业异常验证 |
| G07／07-03 | HG05线性SVM、一次困难负例重训、全位置多尺度扫描、NMS机制及原／参考边界齐备 | **可由部分转已有** | 原扫描与NMS配置、修改版训练器、数据实跑和端到端检测验收 |
| G08／09-10 | LB02 uniform／ri／riu2、圆采样、VAR、多尺度L及类原型／Outex 3-NN分别闭合 | **可由部分转已有** | 原VAR箱数、零支持／平票／式(15)问题、正式数据划分与结果复现 |

三项均建议把后续工作作为**P2深度／复现任务**保留，而不是宣称“实现完成”。若验收目标改为严格逐数复现原论文，则三项都尚未达到；这与目录层可转“已有”不矛盾。
