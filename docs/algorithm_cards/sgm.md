# SGM 半全局匹配算法卡

[学习规范](../algorithm_learning_standard.md) · [三维总览](../3d_evolution.md) · [覆盖表 G38／10-05](../evolution_coverage.md) · [G01 局部 SAD 对照](remaining_domains.md#g01-双目-sad-块匹配与三角测量)

## 范围、版本与覆盖状态

核对日期：2026-11-05；仓库基准：HEAD `918a03e` 加当前工作树。唯一卡键 **`sgm.md:SM01`**，承接 **G38／10-05 E05**；MI 估计、路径聚合、左右检查和后处理属于同一算法卡，不拆计。2005 CVPR 方法与2008 TPAMI 扩展分别记账；固定实现对照为 **OpenCV 4.10.0 tag**，不是原作者 MI 实现。

| 项目 | 状态与证据边界 |
|---|---|
| 演进定位 | 被动立体对应：局部窗口跨深度边、单行 DP 条纹 → 像素 MI 代价＋多方向路径平滑近似 → 增加代价体、路径扫描与可见性处理；不是三角化或完整三维系统 |
| L1 | 已核验：核心 HMI-SGM 流程、迭代依赖、输出和停止协议；非全部2008可选扩展逐行复刻 |
| L2 | 已核验：MI 与匹配相互依赖、P1/P2、归一递推、重复数据项、后验一致性机制 |
| L3 | 部分：量纲、整数安全与排障方案明确；无目标场景调参、资源剖析或物理验收 |
| L4 | 已核验（限定代数）：固定代价下的链式 DP、归一不变量、聚合与复杂度、抛物线精化；不含 MI 迭代收敛或二维全局最优证明 |
| L5 | 部分：固定源码静态对照、文档内递推参考及有限穷举／边界检查；未实现完整 MI-SGM 图像流水线 |
| 面试自测 | L1–L5 五问，逐问含答案、条件、易错点和证据入口 |
| 实现 | **未实现**：仓库没有 MI-SGM／SGBM 立体模块；文档检查器不是产品代码、第三方接入或双目数据集复现 |
| 销项 | 本卡补齐 G38 具体方法缺卡所需机制；建议“缺卡→已有、P1→P2”，保留实现与验收工作。仅写本文件，不改总表／索引，不宣称总账已经更新 |

“已核验”限定实际读到的原文、源码和条件代数，不表示全文全部实验复核。2005取得 §3.2–3.3 正文片段，2008取得核心 §2.1–2.3 及 §2.5.3–2.6；不足处见来源记录。以下没有立体性能排行榜或自行生成的图像实验数据；有限整数集合只用于数学程序验证。

<a id="sm01"></a>
## SM01 原始 MI-SGM：像素代价与多路径近似

### 任务、输入输出与演进边界

输入是有已知极线几何的基准图像与匹配图像、合法视差范围和有效成像区域。原文允许一般极线映射，**并不要求所有输入均可校正成水平极线**；本卡实现讨论收窄为已校正双目，左图为基准，坐标零基，数组按 `[y,x,j]` 存储。

定义整数标签集合 \(\mathcal D=\{d_{\min},\ldots,d_{\max}\}\)，标签数 \(K=d_{\max}-d_{\min}+1\)，`j=d-dmin`。本文 `dmax` 包含端点，与 OpenCV `minDisparity+numDisparities` 的排他上界不同。左到右映射 \(q=(x-d,y)\)，右图视差也保存同一个物理量 \(d=x_L-x_R\)，故右到左为 \((x_R+d,y)\)。若交换图像后仍用“基准横坐标减匹配横坐标”的库接口，右视差会变号，须显式转换。

输出至少分开：整数视差、可选亚像素视差、有效掩膜、越界／不一致／歧义等拒绝理由、可选补全视差及补全掩膜。合法零视差不是缺失码；深度 \(Z=fB/d\) 还要求主点偏移已处理、尺度正确及非零有效视差，SGM 本身不输出可信毫米量。

| 方法版本 | 改变及保留边界 |
|---|---|
| Hirschmüller，CVPR 2005 | 像素 MI 的层次估计与多方向路径累积是主要贡献；论文也给出 BT 代价选项。不能把“原文允许 BT”误写为“原文 MI 就是 BT”。[cite:1](#source-1) |
| Hirschmüller，TPAMI 2008 | 系统讨论亚像素、左右检查、多基线、峰值过滤、弱纹理精化、补洞、大图分块与正射融合；在线发表2007-05-02，期刊年2008，DOI中的2007不是期刊年。[cite:2](#source-2) |
| OpenCV 4.10.0 StereoSGBM | BT 类像素度量＋块聚合、预处理和后处理；**未实现 MI**。默认5方向，`MODE_HH` 为8方向；`blockSize=1` 不会启用 MI，也不等于原文16路径设置。[cite:3](#source-3)[cite:4](#source-4) |

### L1 步骤：从未知对应到可拒绝的视差

#### MI 查表与层次迭代

1. 核验同步、极线、图像有效区域与视差区间；构建图像金字塔。每缩小一半，水平视差单位也缩小一半，不能只缩图不缩视差。
2. 最粗层用合法范围内的随机视差初始化；同时初始化同尺寸的 `valid_seed`：只有视差有限、在本层标签范围内、左右成像支持有效且投影不越界才为真，无合法标签的像素为假。这只是可用于估计的初始支持，不表示随机对应已正确。原文举例从1/16分辨率开始，最粗层计算3轮。随机种子、采样范围与无效处理属于复现配置，不是理论保证。也可用其他快速匹配初值，但必须记录为替代。
3. 由当前 \(D^{(t)}\) 将右图采样到左图，收集有效对应集合 \(\Omega_t\)。排除越界与已知无效点，避免遮挡导致的多对一重复映射；几何通常允许同一右像素仅保留可见的较近表面。最初的对应只是估计，不能假设已经全正确。
4. 统计对应灰度的联合直方图 \(P_{12}\)，除以有效对应数 \(n\)，从其行列和得到 \(P_1,P_2\)。**不能用两张完整图像各自的灰度直方图替代同一有效支持上的边缘分布。**
5. 联合概率先高斯平滑，再对零值设正下限取对数，再做一次高斯平滑，得到 \(h_{12}\)；边缘项同理。构造 \(C_{MI}=h_{12}-h_1-h_2\) 的灰度对查找表。高斯卷积在**灰度／直方图轴**上，不是对图像邻域做 SAD。原文2008提到7×7联合核和256×256查表的示例。[cite:2](#source-2)
6. 冻结本轮查表，为所有合法 \((p,d)\) 查询代价；统一缩放代价和惩罚，独立执行多路径聚合、WTA、可选亚像素和左右检查，更新视差与有效支持。
7. 最粗层完成预定轮数后进入细一层；视差与有效掩膜同步上采样，并恢复该层视差像素单位，仅用于估计该层 MI 查表。本卡采用同一最近邻父像素映射传播两者，掩膜不作数值缩放；再与细层有限性、标签范围、投影边界及左右成像支持相交，粗层无效不因上采样变有效。**重新计算全标签代价与路径状态**，核心 HMI 不把粗层视差当成只能局部微调的搜索中心。[cite:1](#source-1)[cite:2](#source-2)
8. 到原分辨率完成预定计算后停止。工程上可加最大轮数、支持不足退出和视差／查表变化监测，但没有“恰好3轮收敛”或 MI 单调改进定理；补洞结果不自动回流成真实对应。

2008大图扩展另外提出分块与视差范围缩减；这是可选近似，不可反过来将核心 HMI 描述成强制粗到细窄窗。无有效对应 `n=0` 时返回无法估计 MI，不能除零或伪造均匀置信视差。

#### 多路径聚合与输出顺序

对每个方向，列出不重不漏的扫描路径；每条路径从图像边界起步，起点 \(L_r=C\)，按 L4 递推并累加至 \(S\)。原文建议至少8个、宜16个方向以覆盖二维；16方向中的中间方向交替走一次水平／垂直和一次对角步，不等于直接用固定 `(2,1)` 跳过中间像素。少于8方向仍可定义递推，但不应叫原文推荐配置。[cite:2](#source-2)

每像素取 \(\arg\min_d S(p,d)\)，等价极小值只按确定规则选代表并另标歧义。原文2008 §2.3：亚像素抛物线近似；右视差可从同一 \(S(x_R+d,y,d)\) 选最小，或交换基准图重新做代价／聚合；小窗口3×3中值过滤后作左右检查。复用 \(S\) 的反查不是独立右图重算，因为聚合几何基准不对称；中值也可能抹掉细结构，是否启用需留配置与对照。[cite:2](#source-2)

核心流程伪代码如下；`estimate_mi_table` 的内容是前述统计与 L4 公式，`aggregate` 展开于 L5。此块是设计伪代码，不声称整条流水线已运行。

```text
for level in coarsest_to_finest:
    left, right, image_masks, label_range = inputs_at(level)
    if coarsest:
        Dseed = random_legal_disparity(label_range)
        valid_seed = seed_support(Dseed, image_masks, label_range)
    else:
        Dseed, valid_seed = rescale_seed_nearest(Dprev, valid_prev, level)
        valid_seed = valid_seed AND seed_support(Dseed, image_masks, label_range)
    for t in declared_iterations(level):
        pairs = visible_in_bounds_pairs(left, right, Dseed, valid_seed)
        if count(pairs) == 0: return NO_MI_SUPPORT
        table = estimate_mi_table(pairs, bins, gaussian, log_floor)
        Cleft, mask_left = lookup_all_legal_disparities(table, left, right)
        Cleft = prepare_costs(Cleft, mask_left)
        Sleft = aggregate(Cleft, directions, P1, P2)
        Dleft = winner_and_optional_quadratic(Sleft)
        Dright = reverse_lookup(Sleft) or independent_right_pass(table)
        Dleft, Dright = optional_mask_aware_median(Dleft, Dright)
        Dseed, valid_seed = left_right_check(Dleft, Dright, tolerance)
    Dprev, valid_prev = Dseed, valid_seed
return Dprev, valid_prev, optional_labeled_completion(Dprev, valid_prev)
```

`seed_support`与`rescale_seed_nearest`是显式声明、尚未实现的图像层依赖；层列表非空且每层迭代次数须为正。`valid_seed`是像素级统计支持，`mask_left`是全标签代价的 `[y,x,j]` 布尔掩膜，两者不可混用。`prepare_costs`先把无效标签转为`None`，再按每像素有效标签最小值平移负 MI；L5的`aggregate(c, directions, p1, p2)`不再额外接收mask。独立右图重算也须走同一适配与聚合契约，而不是直接将带负值的查表送入非负内核。

### L2 机制：不能删掉的因果关系

**MI 不是固定灰度距离。** 某一灰度值在另一相机中可能因响应差异映射为另一灰度，MI 从对应统计学习依赖，不要求亮度相等。但饱和、多种空间照明关系混杂、少量支持、透明反射或重复纹理仍会破坏估计；全局统计强不表示单个像素可唯一匹配。删去边缘熵项会变为只最小化联合熵：在遮挡改变支持时不是原 MI 目标。完全独立的强度分布使理想点互信息为0，不能凭 MI 产生几何证据。

**P1 与 P2 不只是“平滑强度”。** \(P_1\) 惩罚一步标签变化，允许斜面阶梯；\(P_2\) 对更大跳变施加较大但封顶的代价，避免惩罚随跳变无限增长而抹掉深度边。一般要求 \(0\le P_1\le P_2\)，常取严格不等式；将两者设相同变成 Potts 型跳变代价，失去对缓慢坡度的优待。全部设0则每步 \(L=C\)、\(S=RC\)，退回逐像素 WTA，不解决歧义。

**归一化减去前驱最小值，不是当前最小值。** 它去掉与当前标签无关的累积基线，使路径代价不随路径长度无界增长。改变每个标签不同的偏移会破坏路径选择；做完加法再截断也可能先溢出。归一不使每一行 \(\min L=0\)：当前数据项仍保留。

**多方向缓解条纹，但不是联合求解二维标签场。** 每个方向独立选择最优历史，几个方向对同一早先像素可能隐含不同标签，无法拼成一个共同全局解。逐像素最小化路径和并不保证最小化原始二维能量。原文也明确路径代价本身不强制可见性／顺序约束，不能删除后验一致性检查。[cite:2](#source-2)

**原文聚合重复数据项是定义，不应偷偷“修正”。** 写 \(L_r(p,d)=C(p,d)+M_r(p,d)\)，则

\[
S_{\mathrm{orig}}(p,d)=\sum_r L_r(p,d)=R C(p,d)+\sum_r M_r(p,d).
\]

若改为 \(\widetilde S=\sum_rL_r-(R-1)C=C+\sum_rM_r\)，只消除了当前像素的数据项重复，未解决其他像素或边的重叠。这是不同聚合规则，可能改变 WTA、代价曲线和阈值；**不是2005／2008式(14)**，也不是一个二维全局最优证书。完整将 \(S\) 除以R不改变 argmin，但仅除数据项会改变数据／平滑比例；惩罚和置信阈值不可原样照搬。

**左右一致失败不等于全是遮挡。** 越界、重复纹理错配、局部平滑错误与真实不可见都可能失败。即使通过，重复图案的互相一致错误仍可存在。补全只能输出推断标签，不能提升观测有效率。

### L3 工程权衡：代价尺度、失效和资源

#### 参数与调参顺序

| 参数／接口 | 单位、依据与诊断动作 |
|---|---|
| `dmin,dmax` | 当前层像素；依据标定、距离范围与校正偏移设置。先看峰值是否撞端点、越界比例和纵向极线残差，再扩大范围；不能用 P2 修复几何错误 |
| MI 分箱、高斯核、对数下限 | 分箱轴是强度，核宽单位是灰度 bin；对数下限单位是概率。少支持细分箱使统计不稳，强平滑又掩盖依赖；记录 n、空 bin 比例、核边界策略与查表范围 |
| MI 数值尺度 | 原式带1/n；图像层级或有效支持变化会改变幅值。若乘 n 或统一正尺度，P1/P2也须按相同尺度处理；否则看似相同惩罚实为不同正则化 |
| P1/P2 | 与每像素 C 同代价单位，不是视差像素。先固定 C 范围／量化再调两者；过大易边缘肥厚、细结构消失，过小易噪声与条纹；不存在跨 MI／BT／Census 通用数值 |
| 梯度自适应 P2 | 原文 \(P'_2/\lvert I(p)-I(q)\rvert\)，并要求不小于P1；零梯度须封顶／防除零。可用 \(\max(P_1,\min(P_{2,\max},P'_2/\max(g,g_0)))\)，其中 \(g_0>0\) 是本卡工程保护，不是原文固定超参 |
| 路径数 R | 无量纲；按实际方向集合、扫描相位和边界记录，不只记录“16”。多路径增加算术／带宽；更多路径不能恢复不存在的纹理 |
| 左右阈值 τ | 当前层视差像素；原文式(15)为1像素。亚像素采样方式、掩膜和定点换算要一致；增大τ会放过错配而非修复遮挡 |
| 唯一性／峰值过滤 | 代价差或比率与连通分量面积分开；代价加常数不影响 WTA，却可能改变比率阈值。峰值面积阈值须小于希望保留的真实结构尺度 |

建议顺序：同步与标定 → 符号／视差区间与掩膜 → MI 支持及尺度 → 代价曲线与 P1/P2 → 路径／后处理 → 资源优化。强辐射差时先核对 MI 分布是否有效，不能用增大块窗口冒充原 MI 复现。OpenCV 文档中的 `8*channels*blockSize²`、`32*channels*blockSize²` 是 SGBM 示例尺度，不是 MI 原版默认参数。[cite:4](#source-4)

#### 内存、整数溢出与优化约束

若完整保存 `C` 和 `S`，容量为 \(WHK(b_C+b_S)\) 字节，再加路径工作区、图像、视差和掩膜；不要误写成仅 \(O(WH)\)。串行逐路径只需上一像素K维状态，仍可能保留整个 S；融合多方向逐行扫描常需 \(O(RWK)\) 滚动工作区。保存全部 \(L_r\) 会额外达到 \(O(RWHK)\)，不是必要实现。

在合法有限代价已平移成 \(0\le C\le C_{\max}\)，且 \(0\le P_1\le P_2\le P_{2,\max}\) 时，有

\[
0\le L_r\le C_{\max}+P_{2,\max},\qquad
0\le S\le R(C_{\max}+P_{2,\max}).
\]

必须同时检查存储上界、前驱 `L+P1`／`minL+P2`、累加 S、抛物线分子分母、乘16及数组大小乘积。**“C量化到11 bit就永不溢出16 bit”不对任意P2成立**：2008原文给出了11 bit实现安排，但移植时仍须满足含P2的总界；有符号16位上限32767，无符号为65535，还须预留无效哨兵。下文检查器用参数代数否证不充分的预算，不视为图像性能实验。

代价含负值时先按每像素有限标签最小值作 \(C'=C-\min_d C\)，这是标签无关平移，在精确算术下对固定路径、固定合法集合的 WTA 不变；不能把负 MI 全截零。每个方向的当前路径值只减当前像素基线，因此聚合只减同像素的R倍基线；代价差保持，比率阈值须另校准。先应用mask，不把被屏蔽的NaN／Inf纳入最小值；有效标签非有限值必须拒绝，全无效像素保持全`None`。有限两端之差也可能溢出（如`1e308 - (-1e308)`），适配器必须检查平移结果。改变有效支持或每像素单独乘不同尺度不具相同不变性。浮点舍入与量化仍可使近同价翻转，先用宽整数／浮点参考比对。

`INF` 优先用掩膜而非最大整数参与加法；全无效前驱不允许 `INF-INF`。本卡定义遇到整像素无有效标签时截断该方向路径，后续有效点重启；这是明确的教学边界政策，不是已经验证的原作者遮罩实现。定长硬件实现应保留无效标志，合法候选用宽类型计算后验证可表示再存储。饱和会制造假并列，不能因为不会回绕就声称结果等价。

分块节省内存却截断路径上下文，有限重叠并不一般保证与整图相同；原文2008 §2.6采用重叠与加权融合，是工程近似。优化先比对每方向 L、聚合 S、WTA 和有效掩膜，再测代价构建、路径带宽与后处理；没有实测时不能据 \(O(WHK)\) 宣称实时。[cite:2](#source-2)

### L4 数学：MI 局部化、递推与保证范围

#### 从互信息到可查表的数据项

令 \(I_1=I_L\)，\(I_2=f_{D^{(t)}}(I_R)\)，有效支持大小 \(n=|\Omega_t|\)。原始目标为

\[
MI=H_1+H_2-H_{12},\quad H(P)=-\sum_z P(z)\log P(z),\quad
P_{12}(i,k)=\frac1n\sum_{p\in\Omega_t}\mathbf1[(I_1(p),I_2(p))=(i,k)].
\]

\(P_1(i)=\sum_kP_{12}(i,k)\)，\(P_2(k)=\sum_iP_{12}(i,k)\)。固定 bin 与归一化高斯核 g，论文通过引用 Kim 等的 Taylor 局部化写出

\[
h_{12}(i,k)=-\frac1n\bigl[\log(P_{12}*g_2)*g_2\bigr](i,k),\qquad
h_a(i)=-\frac1n\bigl[\log(P_a*g_1)*g_1\bigr](i),
\]
\[
mi(i,k)=h_1(i)+h_2(k)-h_{12}(i,k),\qquad
C_{MI}^{(t)}(p,d)=-mi(I_L(p),I_R(x-d,y)).
\]

负号来自“最大化 MI”转为“最小化代价”。第一次卷积估计密度、第二次卷积平滑对数项，不能写成只算 `-log(joint)` 或只平滑一次。若形式上忽略两次平滑，则 \(mi\) 近似 \(n^{-1}\log(P_{12}/(P_1P_2))\)，用于检查符号；它不是原文完整实现的替代。[cite:1](#source-1)[cite:2](#source-2)

局部化的动机可由熵的一阶变分看出：对于归一化密度扰动 \(\sum_z\delta P(z)=0\)，

\[
H(P+\delta P)\approx H(P)-\sum_z(\log P(z)+1)\delta P(z)
=H(P)-\sum_z\log P(z)\delta P(z).
\]

将新增对应的 Parzen 核代入产生对数密度的核加权项，常数基线不参与当前标签比较。这里展示的是**冻结统计、固定支持下的一阶解释**；未取得 Kim 等原推导全文，不把它写成对任意新 D 的精确 MI 分解。改变 D 会改变统计、遮挡与 n，重新估计后目标本身也变，不能从单轮 DP 最优推出联合 MI 迭代收敛。

直方图边缘补零、截断核重归一或反射会给边缘 bin 不同值；概率平滑和对数地板的执行顺序必须固定。地板令 log 有限，但引入估计偏差；只有 n>0、统计非病态时才有可用代价，不提供普适辐射不变性或误差界。

#### 二维目标与单链精确解

为避免邻域计数歧义，本卡把二维建模目标写为无向边集合 \(\mathcal E\) 中每边一次：

\[
E(D)=\sum_p C(p,D_p)+\sum_{\{p,q\}\in\mathcal E}V_{pq}(D_p,D_q),\qquad
V(a,b)=\begin{cases}0&a=b\\P_1&|a-b|=1\\P_2&|a-b|>1.\end{cases}
\]

原文式(11)用 \(q\in N_p\)；若复现选择对称邻域并双计边，就必须保留双计或相应换算惩罚，不能混着比较能量。多路径 `r` 的集合也不是这个无向边集的简单重命名。

对路径 \(p_0,\ldots,p_t\)，定义以标签d结束的最小前缀能量

\[
F_t(d)=\min_{d_0,\ldots,d_{t-1}}\left[\sum_{u=0}^t C(p_u,d_u)+\sum_{u=1}^t V(d_{u-1},d_u)\right],\quad d_t=d.
\]

拆出最后一个标签得 \(F_t(d)=C_t(d)+\min_k(F_{t-1}(k)+V(k,d))\)。分解为同标签、左右相邻和大跳变四种候选，可写为

\[
L_t(d)=C_t(d)+\min\{L_{t-1}(d),L_{t-1}(d-1)+P_1,
L_{t-1}(d+1)+P_1,m_{t-1}+P_2\}-m_{t-1},
\quad m_{t-1}=\min_kL_{t-1}(k).
\]

越出标签集的候选置无效。最后一项使用所有k的最小值而不是显式排除邻近k，是因为 \(P_2\ge P_1\ge0\)：若其最小值来自同标签／相邻标签，已有更便宜或同价候选压住它；不满足此条件，快捷式可能引入非法便宜跳变。计算一次m供所有d共用，把每像素 \(O(K^2)\) 降为 \(O(K)\)。起点 \(L_0=C_0\)。

归一化的严格不变量不是 \(L_t=F_t-\min F_t\)，而是

\[
L_t(d)=F_t(d)-\min_k F_{t-1}(k)\quad(t\ge1).
\]

归纳：若前一层 \(L_{t-1}=F_{t-1}-a\)，则其最小值也减a，代入递推后a抵消。因减项与d无关，各标签排序与固定终点最优前缀保持不变。下界由所有惩罚非负和最小前驱m给出；上界由总可选的 \(m+P_2\) 得 \(C_t\le L_t\le C_t+P_2\)。该界只约束代价值，不是视差误差上界。

对每条链穷举可验证F，但原文输出不沿一条共同二维路径回溯，而是逐像素最小化 \(\sum_rL_r\)。因此单链正确、归一正确、数据项去重，都不能证明二维E达到全局最小。

#### 亚像素、左右检查与遮挡

设整数最小值在内部标签 \(d_0\)，邻域代价 \(a=S(d_0-1),b=S(d_0),c=S(d_0+1)\)。用 \(q(u)=Au^2+Bu+b\) 过三点，有 \(A=(a+c-2b)/2\)、\(B=(c-a)/2\)，故

\[
\delta=\frac{a-c}{2(a-2b+c)},\qquad d_{sub}=d_0+\delta.
\]

三值须有限、中心为局部极小、曲率 \(a-2b+c>0\)。平坦、无效邻居、端点或非正曲率时不精化；真实离散极小的该插值在 \([-0.5,0.5]\) 内，超出通常意味着选点／数值问题，应回退而非任意剪出高精度结果。MI聚合曲线不天然是 SSD 抛物线，原文明确其近似性质，不承诺1/16像素物理准确度。[cite:2](#source-2)

在本文同符号约定下，\(q=(x-D_L(p),y)\)，通过条件是右图支持有效且 \(|D_L(p)-D_R(q)|\le\tau\)。若右图接口输出相反符号，则应检查和而非差。亚像素q可按事先声明的近邻／线性插值采样；线性插值需要两侧都有效，不能跨遮挡混值。原文的τ=1是像素单位，OpenCV原始定点输出须先处理 `/16` 和无效码。

2008 §2.5.3进一步对失败点沿其极线查找与右视差函数的交点：有交点倾向标为 mismatch，无交点标为 occlusion；这是依赖右视差正确性与离散交点容差的判别，不是天然真值。与遮挡区直接邻接的错配区在补洞时按遮挡处理。沿8方向传播有效值：遮挡取**第二小**值以偏向较远背景，错配取中位数；既不是所有无效点都取最小，也不是都取均值。越界／无候选或不足两个传播值必须保留无效或另定策略，不自行杜撰原文边界默认。补洞掩膜与原始观测掩膜始终分开。[cite:2](#source-2)

#### 复杂度与敏感性

固定灰度bin数B、固定核支持时，建 MI 统计及查表为 \(O(WH+B^2)\)；逐像素全视差代价 \(O(WHK)\)，R路径聚合 \(O(RWHK)\)，WTA \(O(WHK)\)。独立重算右图约增加另一遍对应工作；已知视差的 LR 检查是 \(O(WH)\)，若为右图从S搜索全部标签则 \(O(WHK)\)。区分这两个步骤，不把已知视差检查也错误写成必须全搜索。

HMI共有层 \(\ell\)、每层 \(T_\ell\) 轮时，总量为 \(\sum_\ell T_\ell[O(RW_\ell H_\ell K_\ell)+O(W_\ell H_\ell+B^2)]\)。原式(10)的约1.14来自宽、高、视差数同时减半，忽略查表／缩图开销，且最粗层3轮；不是实际平台“只慢14%”的保证。B或核宽可变时不能当常数隐藏。

没有通用视差误差或 MI 迭代收敛界。理想校正几何下 \(Z=fB_{base}/d\)，\(\partial Z/\partial d=-fB_{base}/d^2\)，仅说明小视差敏感性；标定误差、离群值和可见性不受该一阶导数控制。输出更细定点刻度不自动缩小真实深度偏差。

### L5 实现：固定源码、数据布局与有限验证

#### OpenCV 4.10.0 静态映射，不冒充原 MI

| 入口 | 核读到的实现及差异 |
|---|---|
| `modules/calib3d/include/opencv2/calib3d.hpp`，`StereoSGBM` 类说明 | 明确声明MI未实现、块而非单像素、默认5方向及HH的8方向；`numDisparities`须为16倍数，输入窗口契约属于该库，不是SGM数学要求。[cite:4](#source-4) |
| `modules/calib3d/src/stereosgbm.cpp`，`calcPixelCostBT` | 原像素与截断梯度的BT类成本，随后横纵窗口聚合；无联合直方图、熵项和HMI反馈链。[cite:3](#source-3) |
| `computeDisparitySGBM`，约L493–518、L696–813 | `CostType`／`DispType`为`short`；C预加P2，`delta=minL+P2`，随后用 `Cbuf+min(...)-delta`；预加与多减的P2抵消，不能只看delta误判为不同归一式 |
| 同函数约L793–813 | 标量分支用int中间值再写short，S有`saturate_cast`；SIMD另有16位向量加法。须按实际编译后端与参数预算验证，不能由单一分支断言任意参数无溢出或位级相同 |
| 同函数约L923–956 | 通过左赢家投影更新`disp2ptr`和最小代价，不是独立右图SGM；抛物线定点精化后用向下／向上取整的两个右坐标检查，区别于本卡严格双支持的参考协议 |
| `BufferSGBM`／`isFullDP` | 普通模式 C/S 用行缓冲，完整模式按高分配；两行Lr滚动，方向融合与对齐填充另占空间。不能把普通5路径内存口径套到HH全图模式 |

**已发现的文档／源码分歧：**该tag头文件说 `disp12MaxDiff<=0` 禁用检查，但核读的 `computeDisparitySGBM` 约L499实际回退为1。此处 `[DISPUTED]` 指同一版本接口文字与这个分支不一致；必须按入口、模式和源码行为处理，未本地编译运行，不能推广到所有版本／所有模式。类似地，`P1=0,P2=0` 在该分支有内部回退，不等于数学上的零惩罚。[cite:3](#source-3)[cite:4](#source-4)

#### 文档内可执行参考内核

下方仅用Python标准库；`prepare_costs`接收有限有符号代价及同形布尔mask，输出非负有限代价／`None`供`step`与`aggregate`使用，未实现 MI 查表、图像读取或原文16方向交替步相位。数值验证限Python普通小整数和binary64浮点，不承诺任意大整数、其他数值类型或无舍入误差。`aggregate`使用显式给出的普通整数方向，教学测试选8邻域；方向向量统一按 `(dx,dy)`。`d±1`针对连续单位标签，不适用于不等间隔标签。整像素全无效时重启后续路径为本卡声明政策。

`step`先算`delta = prev[k] - m`，再加入惩罚并按全局候选P2截断；用`delta >= p2 - penalty`预判截断，避免先计算可能溢出的`delta + penalty`。这利用的是候选最小化的等价界，不是对最终代价做饱和。最后才加当前C并检查有限性，`aggregate`每次累加也在写回前检查可表示性；不可表示时抛`OverflowError`，不把Inf交给WTA。固定宽度整数仍需另查目标类型上下界，有限性检查不能替代该预算。

```python
from itertools import product
from math import isfinite


def prepare_costs(c, mask):
    if not c or not c[0] or not c[0][0]:
        raise ValueError("empty cost volume")
    h, w, k = len(c), len(c[0]), len(c[0][0])
    if len(mask) != h or any(len(row) != w for row in c) or any(len(row) != w for row in mask):
        raise ValueError("volume or mask shape mismatch")
    out = []
    for row, flags in zip(c, mask):
        prepared = []
        for pixel, valid in zip(row, flags):
            if len(pixel) != k or len(valid) != k or any(type(ok) is not bool for ok in valid):
                raise ValueError("invalid label mask")
            values = [v for v, ok in zip(pixel, valid) if ok]
            if any(v is None or not isfinite(v) for v in values):
                raise ValueError("valid cost must be finite")
            baseline = min(values) if values else 0
            shifted = [v - baseline if ok else None for v, ok in zip(pixel, valid)]
            if any(v is not None and not isfinite(v) for v in shifted):
                raise OverflowError("shifted cost is not representable")
            prepared.append(shifted)
        out.append(prepared)
    return out


def step(c, prev, p1, p2):
    if not c or not (isfinite(p1) and isfinite(p2) and 0 <= p1 <= p2):
        raise ValueError("invalid dimensions or penalties")
    if any(v is not None and (not isfinite(v) or v < 0) for v in c):
        raise ValueError("cost must be nonnegative finite or None")
    if prev is not None and len(prev) != len(c):
        raise ValueError("label count mismatch")
    if prev is not None and any(v is not None and (not isfinite(v) or v < 0) for v in prev):
        raise ValueError("invalid predecessor")
    finite = [] if prev is None else [v for v in prev if v is not None]
    if not finite:
        return list(c)
    m = min(finite)
    out = []
    for d, cost in enumerate(c):
        if cost is None:
            out.append(None)
            continue
        choices = [p2]
        for k, penalty in ((d, 0), (d - 1, p1), (d + 1, p1)):
            if 0 <= k < len(c) and prev[k] is not None:
                delta = prev[k] - m
                candidate = p2 if delta >= p2 - penalty else min(p2, delta + penalty)
                choices.append(candidate)
        value = cost + min(choices)
        if not isfinite(value):
            raise OverflowError("path cost is not representable")
        out.append(value)
    return out


def aggregate(c, directions, p1, p2):
    if not c or not c[0] or not c[0][0]:
        raise ValueError("empty cost volume")
    h, w, k = len(c), len(c[0]), len(c[0][0])
    if any(len(row) != w or any(len(v) != k for v in row) for row in c):
        raise ValueError("ragged cost volume")
    if not directions or len(set(directions)) != len(directions):
        raise ValueError("empty or repeated directions")
    s = [[[None if v is None else 0 for v in pixel] for pixel in row] for row in c]
    inside = lambda x, y: 0 <= x < w and 0 <= y < h
    for dx, dy in directions:
        if (dx, dy) == (0, 0) or type(dx) is not int or type(dy) is not int:
            raise ValueError("invalid direction")
        visits = set()
        for y0 in range(h):
            for x0 in range(w):
                if inside(x0 - dx, y0 - dy):
                    continue
                x, y, prev = x0, y0, None
                while inside(x, y):
                    assert (x, y) not in visits
                    visits.add((x, y))
                    prev = step(c[y][x], prev, p1, p2)
                    for d, value in enumerate(prev):
                        if value is not None:
                            total = s[y][x][d] + value
                            if not isfinite(total):
                                raise OverflowError("aggregate cost is not representable")
                            s[y][x][d] = total
                    x, y = x + dx, y + dy
        assert len(visits) == h * w
    return s


def winner(s, dmin=0):
    valid = [j for j, v in enumerate(s) if v is not None]
    if not valid:
        return None
    return dmin + min(valid, key=lambda j: (s[j], j))


def refine(s, j):
    if not 0 < j < len(s) - 1:
        return float(j)
    a, b, c = s[j - 1:j + 2]
    if any(v is None or not isfinite(v) for v in (a, b, c)):
        return float(j)
    den = a - 2 * b + c
    if den <= 0 or b > min(a, c):
        return float(j)
    delta = (a - c) / (2 * den)
    return j + delta if abs(delta) <= 0.5 else float(j)


def lr_integer(x, dl, right, tau=1):
    if dl is None or type(dl) is not int or tau < 0:
        return False
    q = x - dl
    return 0 <= q < len(right) and right[q] is not None and abs(dl - right[q]) <= tau


def endpoints(rows, p1, p2):
    k = len(rows[0])
    best = [None] * k
    for labels in product(range(k), repeat=len(rows)):
        values = [rows[t][d] for t, d in enumerate(labels)]
        if any(v is None for v in values):
            continue
        cost = sum(values)
        for a, b in zip(labels, labels[1:]):
            cost += 0 if a == b else p1 if abs(a - b) == 1 else p2
        j = labels[-1]
        if best[j] is None or cost < best[j]:
            best[j] = cost
    return best
```

`endpoints`完全枚举所有路径标签，是独立于四候选捷径的检验器；不把其指数时间当成 SGM 复杂度。`lr_integer`只演示同符号整数检查，不冒充亚像素采样或遮挡分类完整实现。代码没有导入OpenCV，也没有给已有模块添加运行入口。

#### 有限穷举／边界检查器

将本文件两个Python块按出现顺序在同一命名空间执行即可复跑。每个枚举输入是明确定义的离散代数对象，不是图像数据、仿真效果或经验性能。固定宽度整数这里只检查可表示预算，不执行C++溢出行为。

```python
def must_raise(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError("expected rejection")


def verify():
    chains = masked = grids = 0
    for flat in product((0, 1), repeat=9):
        rows = [list(flat[i:i + 3]) for i in range(0, 9, 3)]
        for p1, p2 in ((0, 0), (1, 1), (1, 3)):
            prev = None
            for t, row in enumerate(rows):
                prev = step(row, prev, p1, p2)
                f = endpoints(rows[:t + 1], p1, p2)
                baseline = 0 if t == 0 else min(endpoints(rows[:t], p1, p2))
                assert prev == [v - baseline for v in f]
                assert all(row[d] <= prev[d] <= row[d] + p2 for d in range(3))
            chains += 1
    for flat in product((None, 0, 1), repeat=6):
        rows = [list(flat[:3]), list(flat[3:])]
        if any(all(v is None for v in row) for row in rows):
            continue
        prev = step(rows[1], rows[0], 1, 3)
        f = endpoints(rows, 1, 3)
        baseline = min(v for v in rows[0] if v is not None)
        assert prev == [None if v is None else v - baseline for v in f]
        masked += 1
    directions = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
    for flat in product((0, 1), repeat=8):
        c = [[[flat[(y * 2 + x) * 2 + d] for d in range(2)] for x in range(2)] for y in range(2)]
        s = aggregate(c, directions, 1, 3)
        for y in range(2):
            for x in range(2):
                expected = [0, 0]
                for dx, dy in directions:
                    coords, xx, yy = [], x, y
                    while 0 <= xx < 2 and 0 <= yy < 2:
                        coords.append((xx, yy))
                        xx, yy = xx - dx, yy - dy
                    rows = [c[b][a] for a, b in reversed(coords)]
                    f = endpoints(rows, 1, 3)
                    base = 0 if len(rows) == 1 else min(endpoints(rows[:-1], 1, 3))
                    expected = [expected[d] + f[d] - base for d in range(2)]
                assert s[y][x] == expected
        grids += 1
    assert step([2, None, 1], [None] * 3, 1, 3) == [2, None, 1]
    assert step([2], [9], 1, 3) == [2]
    assert step([None], [1], 1, 3) == [None]
    assert step([1, 2, 3], [100, 102, 101], 1, 3) == step([1, 2, 3], [0, 2, 1], 1, 3)
    assert winner([None, None]) is None
    assert winner([0, 0], -2) == -2
    for h, w in ((1, 1), (1, 3), (3, 1)):
        c = [[[0, 2] for x in range(w)] for y in range(h)]
        assert aggregate(c, directions, 0, 0) == [[[0, 16] for x in range(w)] for y in range(h)]
    assert refine([3, 1, 2], 1) == 1 + 1 / 6
    assert refine([1, 1, 1], 1) == 1
    assert refine([None, 1, 2], 1) == 1
    assert refine([0, 1], 0) == 0
    assert refine([0, 3, 0], 1) == 1
    assert lr_integer(2, 1, [None, 1, None], 0)
    assert not lr_integer(0, 1, [1, 1, 1])
    assert not lr_integer(2, 1, [None, None, None])
    assert not lr_integer(2, 1, [None, -1, None], 0)
    assert lr_integer(0, -1, [None, -1, None], 0)
    c = [[[0, 10], [3, 0], [0, 10]]]
    s = aggregate(c, ((1, 0), (-1, 0)), 2, 2)
    assert s[0][1] == [6, 4] and winner(s[0][1]) == 1
    deduplicated = [s[0][1][d] - c[0][1][d] for d in range(2)]
    assert deduplicated == [3, 4] and winner(deduplicated) == 0
    assert 16 * (2047 + 2048) > 32767
    assert 16 * (2047 + 2049) > 65535
    assert 16 * (1023 + 1000) < 32767
    rejected = 0
    for p1, p2 in ((2, 1), (-1, 2), (1, float("inf"))):
        try:
            step([0], [0], p1, p2)
        except ValueError:
            rejected += 1
    assert rejected == 3
    raw = [[[-3, -1, float("nan")], [-2, -5, 0]]]
    mask = [[[True, True, False], [True, True, True]]]
    shifted = prepare_costs(raw, mask)
    assert shifted == [[[0, 2, None], [3, 0, 5]]]
    assert prepare_costs([[[float("inf"), None]]], [[[False, False]]]) == [[[None, None]]]
    signed_rows = [[-3, -1, None], [-2, -5, 0]]
    s = aggregate(shifted, ((1, 0), (-1, 0)), 1, 3)
    for x in range(2):
        expected = [None if v is None else 0 for v in signed_rows[x]]
        for dx in (1, -1):
            rows = signed_rows[:x + 1] if dx == 1 else list(reversed(signed_rows[x:]))
            f = endpoints(rows, 1, 3)
            base = 0 if len(rows) == 1 else min(v for v in endpoints(rows[:-1], 1, 3) if v is not None)
            expected = [None if v is None else expected[d] + v - base for d, v in enumerate(f)]
        pixel_base = min(v for v in signed_rows[x] if v is not None)
        assert s[0][x] == [None if v is None else v - 2 * pixel_base for v in expected]
        assert winner(s[0][x]) == winner(expected)
    for call in (
        lambda: prepare_costs([[[float("nan")]]], [[[True]]]),
        lambda: prepare_costs([[[float("inf")]]], [[[True]]]),
        lambda: prepare_costs([[[0]]], [[[True, False]]]),
        lambda: step([-1], None, 1, 3),
        lambda: aggregate([], ((1, 0),), 1, 3),
    ):
        must_raise(ValueError, call)
    assert 1.0 + 1e16 - 1e16 == 0.0
    assert step([1.0], [1e16], 1, 3) == [1.0]
    assert step([None, 0.0, None], [1e16, None, None], 1, 3) == [None, 1.0, None]
    assert step([1.0], [1e308], 1e308, 1e308) == [1.0]
    assert step([None, 0.0, None], [1e308, 1e308, 0.0], 1e308, 1e308) == [None, 1e308, None]
    assert aggregate([[[1e308]]], ((1, 0),), 0, 0) == [[[1e308]]]
    overflow_calls = (
        lambda: prepare_costs([[[-1e308, 1e308]]], [[[True, True]]]),
        lambda: step([None, 1e308], [0.0, None], 1e308, 1e308),
        lambda: aggregate([[[1e308]]], ((1, 0), (-1, 0)), 0, 0),
    )
    for call in overflow_calls:
        must_raise(OverflowError, call)
    print({"chain_cases": chains, "masked_cases": masked, "grid_cases": grids,
           "overflow_rejections": len(overflow_calls), "boundary_checks": "PASS"})


verify()
```

重复数据项反例在中点的原聚合为`[6,4]`、去重为`[3,4]`，严格最优标签从1变为0，不依赖破同价规则；这说明两种定义不同，不是宣称去重更优。

新增浮点反例：旧顺序`1.0 + 1e16 - 1e16`返回0.0而不是当前代价1.0；即使先括起候选最小值再减基线，若候选仍计算`1e16 + P1`也会丢掉P1=1，所以必须先减前驱基线。检查器同时验证`1e308`下可避免的候选加法溢出不导致误拒绝，以及不可表示的平移、最终路径代价、多方向聚合分别抛出`OverflowError`。负MI回归用独立有符号链穷举对照平移后的双向S与赢家，不只检查输出非负。

### L1–L5 五问与参考答案

| 问题 | 答案要点、成立条件 | 易错答案／证据入口 |
|---|---|---|
| L1：没有视差，原 MI 代价从哪里来，细层搜索如何启动？ | 最粗层合法随机初值→对应联合统计→两次高斯及对数→MI查表→冻结代价做SGM→更新对应；细层上采样只用于统计，重新全范围匹配。无支持退出，预设轮数不等于收敛证明 | “直接对左右图灰度相减”“细层只能搜粗视差附近”。见L1、来源1／2的式(5)–(10) |
| L2：为什么分P1/P2、多方向、还减前驱最小值？ | 小跳变和大断层代价分开；多方向减少单行偏向；前驱最小值为标签无关基线，控制范围不改链排序。原文ΣL重复当前C，不可偷偷减去(R−1)C | “16路径就是二维全局最优”“归一化保证所有整数都安全”。见L2、L4与来源2式(11)–(14) |
| L3：换成MI后仍用SGBM的P1/P2，为什么边变厚或噪声增加？ | 核对代价归一尺度、1/n、分箱、有效支持和图像单位，再调同单位惩罚；先排同步、极线与视差区间，比较边界有效率和代价曲线，不只看平整度 | “增大窗口／P2就能修复遮挡”；库块窗口公式不是MI超参。见L3、来源3／4 |
| L4：推导四候选递推，能保证什么，亚像素又保证什么？ | 从链最小前缀能量拆最后一标签；P2≥P1≥0使全局min捷径成立；L=F−min前缀，C≤L≤C+P2。三点抛物线给δ但要求内部极小及正曲率，MI曲线仅近似 | “L=F−min当前F”“δ的小数位数即深度精度”。见L4、有限穷举、来源2 |
| L5：库输出稠密且LR通过，能否称原MI复现与无遮挡？ | 固定tag检查BT／块／方向、右图生成方法、short／定点缩放／哨兵；独立测路径、越界与溢出。LR失败有多因，通过也可能一致错配；补洞另标。当前仅文档参考通过有限检查 | “blockSize=1启用MI”“负视差都是无效”“填完洞全是观测值”。见L5、来源2／3／4 |

## 验证记录与 G38 销项边界

核验分层：原文／固定源码静态阅读；冻结成本下的代数证明；文档内Python有限穷举和边界断言；Markdown结构、引用与本地链接检查。检查器不调用StereoSGBM，不处理照片，不建立MI直方图，也不执行完整16路径相位、遮挡交点、补洞或物理深度测量。

以下只记录本文件内检查器的适用集合：3像素×3标签的二值代价全部512种，配3组惩罚共1536条链；2像素×3标签的`None/0/1`组合中两行都有合法标签的676种；2×2网格、2标签、二值代价的全部256种、8方向。每条链按终点标签独立枚举，对照前缀基线；网格按每方向各自穷举对照S，不拿二维全局最优当期望。另覆盖退化维度、负`dmin`、单标签、空有效集、标签端点、平坦曲率、LR符号与越界、重复数据项、整数预算和非法惩罚拒绝。

2026-11-05本轮修订在仓库已有`.venv/Scripts/python.exe`（Python **3.13.14**）中，以`-B`从本文件内存提取两个Python块、在同一命名空间按顺序执行；输出为`chain_cases=1536, masked_cases=676, grid_cases=256, overflow_rejections=3, boundary_checks=PASS`。新增mask→None、负MI逐像素平移及有符号穷举对照、`1e16`丢1、`1e308`候选安全截断和三类不可表示拒绝检查全部通过；本地链接／引用锚点、代码围栏与行尾空白检查通过。未写入独立测试脚本或运行实现模块。

验证边界：`valid_seed`首轮定义、视差与掩膜的跨层同步、当前层支持复核，以及L1/L5的四参数聚合签名仅作静态契约核对；图像依赖仍是伪代码，没有执行金字塔、随机初始化、MI估计或完整左右匹配。浮点检查仅覆盖所列代表值与异常分支，不证明所有binary64输入排序精确，也不覆盖`refine`的大幅值溢出／舍入、固定宽度C++累加或硬件行为。原文／外部源码的既有阅读记录保留，本轮未重新联网复核。此次仅修改`sgm.md`与`curless_levoy.md`，不改实现代码、台账、索引或数量；状态仍为**未实现**。

未完成项：原作者MI源码或与其输出等价的实现、MI统计／参数标定及完整图像集回归、固定编译后端整数行为和内存／耗时剖析、亚像素／遮挡真值验证、真实相机与深度尺寸验收。没有执行原论文速度／精度复现，未提供普适误差界。

**交接记录：**G38／10-05 → `sgm.md:SM01`；具体方法缺卡内容可销项，五层状态为 **核／核／部／核（限定代数）／部**，答5，原文T＋固定实现源码T，**未实现**；建议后续台账登记为已有并保留P2工作。总表、索引及其数量在本次单文件范围外，历史ID保留。

## 来源、获取深度与限制

以下均为2026-11-05联网核对。B=书目，T=实际读到的方法／源码正文；T不代表逐页精读、实验复测或本地实现。原论文两版与作者稿互有关联，不凑作三个独立实验证据。方法特有事实标记为 `[SINGLE-SOURCE]`（原作者论文族），OpenCV特有事实为 `[SINGLE-SOURCE]`（官方代码族）；没有三组独立性能复现，故不做跨方法性能结论。

<a id="source-1"></a>

**[cite:1] Hirschmüller 2005，原始会议稿，P0，B＋T片段。** *Accurate and Efficient Stereo Processing by Semi-Global Matching and Mutual Information*，CVPR 2005，卷2，807–814；作者机构PDF： https://www.robotic.dlr.de/fileadmin/robotic/hirschmu/cvpr05hh.pdf 。实际读到题名、会议日期、§3.1–3.2完整MI／HMI机制及§3.3目标与未归一递推、归一式开头；网页抽取随后截断。直接下载另因连接失败，未伪称核读该版§3.4–3.6全部内容；完整归一式、亚像素／LR和整数讨论以已取得的2008正文核实。另试的机构路径未取得有效PDF，不作为证据。

<a id="source-2"></a>

**[cite:2] Hirschmüller 2008，原始期刊版及作者接受稿，P0，B＋T。** *Stereo Processing by Semiglobal Matching and Mutual Information*，TPAMI **30(2):328–341，February 2008**，DOI `10.1109/TPAMI.2007.1166`，在线2007-05-02。期刊版镜像： https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Hirschmueller2008PAMI.pdf ；作者机构接受稿： https://elib.dlr.de/55367/1/Stereo_Processing-Hirschm%C3%BCller.pdf 。两份均取得14页PDF的文本；定向核读期刊§2.1–2.3、§2.5.3–2.6及接受稿II-B/C/E/F对应段落，含式(5)–(15)、式(18)，作者稿注明2007-04-16接受。为遵守单文件约束，PDF与抽取文本仅在内存处理，未落地附件。首次网页抽取截断且部分负号丢失，通过PDF文本与2005可读公式对照；不声称全部图表、实验、Kim等Taylor推导原文或正射融合已审完。机构旧路径 `pami08hh.pdf` 返回空内容，由上述有效机构稿补足。

<a id="source-3"></a>

**[cite:3] OpenCV固定实现，P0，源码T。** tag **4.10.0**， https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/stereosgbm.cpp ，原始内容 https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/src/stereosgbm.cpp 。核读`CostType`、`calcPixelCostBT`、`BufferSGBM`和`computeDisparitySGBM`的初始化／递推／聚合／视差精化与右图反查段。只固定tag，不声称commit哈希已经核验；未编译或运行该实现，未审所有SIMD及3WAY／HH4分支，不将内部短整数操作推广为通用安全保证。

<a id="source-4"></a>

**[cite:4] OpenCV同tag接口约定，P0，文档／头文件T。** https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/include/opencv2/calib3d.hpp ，原始内容 https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/include/opencv2/calib3d.hpp ；在线API https://docs.opencv.org/4.10.0/d2/d85/classcv_1_1StereoSGBM.html 。在线抓取可见类与定点枚举，但输出有截断、直接访问遇到验证页；用已读取的固定tag头文件核实“MI未实现”、5／8方向、块参数与阈值语义，没有绕过访问验证。`disp12MaxDiff`文字与所读源码分支不一致已在L5显式记录，不以文档覆盖代码事实。
