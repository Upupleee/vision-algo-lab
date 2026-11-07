# Gray-Edge：颜色导数统计与全局光源估计专卡

[学习规范](../algorithm_learning_standard.md) · [颜色双线总览](../color_appearance_evolution.md) · [覆盖与历史任务](../evolution_coverage.md) · [中性参考白平衡 P34](preprocessing.md#p34) · [颜色校正矩阵 P35](preprocessing.md#p35)

## 范围、版本与覆盖口径

核验日期：**2026-11-05**。本文件仅含独立卡 `gray_edge.md:GE01`，承接 **G33／09-11／A-H3 Gray-Edge，2007**。Gray／Grey 是拼写差异；本文按题名区分原始 Gray-Edge 统一框架、作者发布包的实际计算和安全移植建议，不把 P34 的已知中性参考增益当成自动估光机制。

| 对象 | 固定依据 | 边界 |
|---|---|---|
| 原始方法 | van de Weijer、Gevers、Gijsenij，*Edge-Based Color Constancy*，IEEE TIP **16(9), 2207–2214，2007**，DOI `10.1109/TIP.2007.901808` | 所读是署日 2007-06-14 的作者稿；HAL 入库 2010-12-20 不是方法年 [cite:1](#ge-source-1) |
| 作者源码 | 作者 LEAR 软件页直接链接的 `ColorConstancy.zip`，以本次原始字节 SHA-256 锁定 | 无 Git commit／正式版本号；包内时间戳不是签名发布证明，不称不可变下载 URL [cite:2](#ge-source-2)[cite:3](#ge-source-3) |
| 后续路线 | 作者资料站另列 Weighted Grey-Edge（2012） | 本卡不引入其权重图、镜面模型或训练；二值有效域 mask 不等于该扩展 [cite:4](#ge-source-4) |

**关键版本差异：**固定源码二阶响应为 `sqrt(xx²+4xy²+yy²)`，不能静默改成标准 Hessian Frobenius 的 `sqrt(xx²+2xy²+yy²)`；源码头部称 mask 中零应忽略，但执行逻辑恰好是 **0 保留、1 排除**。两项都在 L4／L5 明确处理，不以源码注释代替执行语义。[cite:3](#ge-source-3)

| 独立字段 | GE01 状态与限定 |
|---|---|
| 演进定位 | 低层统计恒常性；从 RGB 强度统计转向平滑后颜色导数统计，统一导数阶数 n、范数 p、尺度 σ |
| L1 步骤 | **已核验**：有效域、导数、空间聚合、估光归一、对角校正的原文／源码静态链 |
| L2 机制 | **已核验**：去符号、局部平滑与全局聚合分工、灰边假设、特例范围 |
| L3 工程权衡 | **部分**：参数单位、作者示例与失败诊断已写；无业务定参、相机标定和性能实测 |
| L4 数学推导 | **已核验（限定条件代数）**：对角模型下估计、齐次性、二阶定义、退化与敏感性；不声称原文光谱绝对值交换无条件成立 |
| L5 实现 | **部分**：固定源码核对、解释性伪代码、8 组有限算术检查；未运行 MATLAB 原程序或整图复现 |
| 面试自测 | **答5**：L1–L5 各一问，含答案、条件、易错点及证据入口 |
| 本仓库实现 | **未实现**：本次检索 Python 源码未见 Gray-Edge／`general_cc`／专用自动白平衡实现 |
| 验证范围 | 原文机制与七个 `.m` 文件静态阅读、发布包哈希、条件代数和文档检查；无真实光源估计成绩、工业样本或现场验收 |

原文与作者代码来自同一研究体系，不构成三项独立复现。方法有效性主张按 **[SINGLE-SOURCE]** 使用，有限检查是确定性算术用例，不是合成场景性能实验，更不是论文数据复测。未提供通用估光误差界。

<a id="ge01"></a>

## GE01 Gray-Edge／Grey-Edge

### 任务契约与演进定位

**输入**为 `I[H,W,3]`，固定 RGB 通道顺序、黑电平处理、动态范围和传递函数；主物理解释采用非负、线性相机 RGB、空间近似均匀的单光源。另输入导数阶 `n∈{0,1,2}`、空间高斯尺度 `σ`（当前图像像素）、Minkowski 阶 `p∈[1,∞]`，以及明确语义的排除 mask。非线性 JPEG 可以数值运行，作者演示也读 JPEG，但不能据此认定它满足线性成像假设。[cite:1](#ge-source-1)[cite:3](#ge-source-3)

**输出**为全局三维光源方向 `e_hat`，通常 L2 范数为 1；可附返回对角校正后的原图、有效样本数、状态和增益。它不是光源光谱、色温、绝对照度、XYZ 白点、空间光源场或材料反射率。估计向量是光源响应，校正增益取其倒数，二者不可互换。

**演进链**：Gray-World 的面积均值容易被大块主导色偏置 → Gray-Edge 改变统计对象，用颜色空间变化而非区域面积估光 → 颜色丰富的边缘集合有机会接近无彩色统计 → 增加导数噪声、尺度、边缘颜色偏置与有效域处理成本。纯色无纹理工件、同色重复纹理、混合光源仍可能失败；不是 Gray-World 的无条件升级，也不能替代受控测色和有参考的颜色校正。[cite:1](#ge-source-1)

### L1 步骤：从输入到校正输出

1. **冻结输入与参数。** 明确线性／编码域、RGB/BGR、float 类型与饱和阈值；n>0 时要求 σ>0，n=0、σ=0 表示不平滑。无训练、无迭代收敛；一次固定流程结束即停止。
2. **建立三通道共用的统计有效域 V。** 排除参考灰球／色板、无效区和饱和附近，去掉规定边带。mask 用来限制统计，不是在卷积前把图像乘成黑底；完整核支持版与作者原版应分别命名。
3. **计算每通道非负响应 A。** n=0 用原强度或高斯平滑强度；n=1 用 x／y 高斯导数的梯度幅值；n=2 用固定定义的二阶幅值。不能先转灰度再复制到三个通道，也不需要 Canny 二值边图。
4. **在空间上逐通道聚合。** 有限 p 计算 `q_c=(Σ_{u∈V} A_c(u)^p)^(1/p)`；p=∞ 单独取有效位置最大值。也可取幂均值，三通道共用同一非空 V 时只相差公共尺度。
5. **检查退化并归一估光。** `e_hat=q/||q||₂`。空 V、全零响应、非有限量或某通道响应不足，应拒判或显式走另一个已标定回退方法，不能用 epsilon 伪造有效白点。
6. **对原图而非导数图校正。** 作者返回 `J_c=I_c/(sqrt(3)·e_hat_c)`，即映向单位中性方向 `(1,1,1)/sqrt(3)`。工程版保留未裁剪浮点结果、增益和拒判状态；显示编码与裁剪另做。

步骤 3–6 的原始计算对应原文式(10)、(16)、(17)及作者 `general_cc.m`；拒判与完整支持 mask 是本卡额外工程契约，不冒充原程序已有保护。[cite:1](#ge-source-1)[cite:3](#ge-source-3)

### L2 机制：三个不同层次的范数与平均

**先局部导数幅值，后全局 Minkowski，最后 RGB 归一。** 第一层在每个颜色通道内部合并空间方向；第二层在整个有效图域合并响应；第三层才把三个颜色统计组成光源方向。这三层不能都叫“取范数”而省略轴。将 RGB 梯度先合成一个标量会丢掉估光所需的通道比例；跨通道相加再做导数同样不可逆。

**绝对值／幅值防止边缘符号抵消。** 同一颜色的上升、下降沿可能分别为 +a、−a，有符号均值可为零，但亮度变化强度不为零；Gray-Edge 要的是变化幅度的统计。先在空间求导数均值再取绝对值，不等价于先取幅值再平均。[cite:1](#ge-source-1)

**σ 不只是消噪旋钮。** 高斯导数把局部尺度与微分合并；小 σ 保留细变化但导数放大高频噪声，大 σ 抑制细纹理且可能合并不同颜色边界。删除平滑后使用差分是不同离散算子；先高斯再 Sobel 也不自动逐位等价于作者高斯导数。对 n=0，σ>0 是 General Gray-World 分支，不是 Gray-Edge。[cite:1](#ge-source-1)

**p 控制大响应的影响。** p=1 是幅值平均，较大 p 更看重强边／亮点，p=∞ 是逐通道最大响应；各通道最大值可以出现在不同位置。p=∞ 不等于先找到一个“最白像素”。较大的 p 不保证更鲁棒，饱和伪边或强彩色轮廓可能控制整个估计。[cite:1](#ge-source-1)

**归一只去掉公共尺度。** 未知反射统计尺度和曝光都可能乘在 q 上，L2 归一消除它们，但不会纠正通道不同的统计偏置。公共曝光不变性还要求没有剪裁、域变化或自适应阈值导致 V 改变；真实场景并不自动满足。

#### Gray-World／Shades of Gray／Max-RGB 特例的准确范围

| 名称 | n、p、σ | 数学与实现限制 |
|---|---|---|
| Gray-World | `0,1,0` | 非负原始 RGB 的通道均值；只有统计域相同时才与相应均值基线一致 |
| Shades of Gray／Grey | `0,p,0`，有限 p≥1 | 是强度的幂均值；不是 n=1 且 p 较大的另一个名字 |
| Max-RGB／White-Patch 型 | `0,∞,0` | 逐通道最大强度；不要求同一位置存在白块，不代表所有 White-Patch 实现 |
| General Gray-World | `0,p,σ`，σ>0 | 先平滑后统计；一般不是未平滑 Shades of Gray |
| 一阶 Gray-Edge | `1,p,σ`，σ>0 | 逐通道梯度幅值的统计；p=∞ 是 Max-Edge，不是 Max-RGB |
| 二阶 Gray-Edge | `2,p,σ`，σ>0 | 二阶幅值定义也必须锁定；不等于 Laplacian 的绝对值 |

依据为原文 Table I；作者演示 Shades of Grey 取 p=5，Gray-Edge 取 `(n,p,σ)=(1,5,2)`，这只是示例，不是通用最优值。作者函数缺省为 `(0,1,1)`，不是表中未平滑的 Gray-World。源码用 `mink_norm=-1` 表示 ∞，不能把数学 ∞ 直接传入有限幂分支。[cite:1](#ge-source-1)[cite:3](#ge-source-3)

这里将用户所述 shadeswhite 按“Shades of Gray／White-Patch 特例范围”澄清：2007 原文的正式方法名是 **Shades of Grey**，没有在本卡另造一个“Shades-White”算法。n=0、p=1 时，理想无边界、保积分卷积上的均值可不受平滑影响；有限图、replicate padding 和裁掉边带后不能据此保证 σ 任意等价。

### L3 工程权衡：调参、mask 与失效诊断

| 参数／契约 | 起点与单位 | 权衡及先查证据 |
|---|---|---|
| n | 0、1、2；无量纲，原文只研究至二阶 | 先把 n=0 作对照；二阶更强调曲率并更敏感于噪声／离散化，不保证优于一阶 |
| σ | 当前网格像素；作者示例 2 | 图像缩放后需对应物理支持；导数核半径为 `floor(3σ+0.5)`，不是 σ |
| p | 无量纲，≥1 或显式 ∞；作者示例 5 | 从 p=1 与示例对照，检查强响应贡献；不得拿测试集挑最优 p 后报告无偏成绩 |
| 饱和 | 作者固定 `max(R,G,B)>=255` | 适用其 double 码值契约；[0,1]、12／16 位和黑电平后的阈值必须另定 |
| mask | 三通道共享二值排除域 | 排除真值参考、坏点和不受本光源照明的区域；记录样本数、空间分布和支持损失 |
| 校正目标 | 作者目标单位中性方向；工程可另定目标响应 t | 公共亮度因子另管；增益上限、裁剪比例和拒判率均应记录 |

参数依据：[cite:1](#ge-source-1)[cite:3](#ge-source-3)。没有通用“推荐 σ=某值便足够”的结论。原文实验数据集的最优参数不同，不能将其中最优 p、σ 移植成工厂验收门限。

**作者 mask 的实际流程**为 `E=用户mask`（默认全零），`S=任一通道≥255`，先 3×3 膨胀 S，再 `V=(E+dilate33(S)==0)`，随后调用 `set_border(V,σ+1,0)`。因此 E=1 是排除；E 需先限定为二值，否则负值可能与饱和值抵消。作者只扩张饱和区，没有同样扩张用户排除区。[cite:3](#ge-source-3)

**mask 与卷积污染是两个问题。** 作者仍在完整原图上求导，3×3 饱和膨胀只有一像素半径，不覆盖通常约 3σ 的导数核；参考块／饱和点虽被排除，其附近响应仍可能混入它们。原文真实图实验排除了画面右下角灰球，但这不等于定义了所有核支持处理。安全版建议把全部排除域按离散核支持扩张，并只统计完整有效支持的中心；这改变样本集合，应作为“完整支持变体”而非作者逐位复现。[cite:1](#ge-source-1)[cite:3](#ge-source-3)

不要把 `I×V` 当滤波输入：这会在 ROI 轮廓人为制造黑白阶跃。mask 应选择卷积后响应中心，并排除其支持接触无效输入的情况。若输入存在 NaN，不能指望 `NaN×0=0`；应拒绝输入，或明确填充值并排除所有受影响支持，后者也是新增策略。

| 症状 | 诊断证据 | 优先行动／回退 |
|---|---|---|
| 纯色平面给出不稳定巨大增益 | 每通道响应能量、有效边数量、黑电平／噪声 | 判低可观测性；用固定采集／外部中性参考，不把导数噪声当光源证据 |
| 白平衡跟着工件花纹变 | 分区响应、强边占比、不同 p 的估计方向 | 核对灰边统计假设；换 ROI／参考方法比盲加阶数更有意义 |
| 反光附近强烈偏色 | 饱和图、膨胀域、核半径、未裁剪 J | 完整支持排除对照；不能恢复已饱和的真实通道 |
| 二阶随旋转显著改变 | xx／xy／yy 分量、混合项权重、边界 | 先区分作者系数 4 与标准系数 2，再讨论离散旋转误差 |
| 两实现整体亮度不同而色比相同 | q 的 sum／mean、L1／L2 归一、目标 t | 统一公共尺度和 `sqrt(3)`；不能将亮度差直接当估光角误差 |
| 多区域校正方向互相矛盾 | 各 ROI 的估光、阴影／反射光与灯位 | 全局单光源假设失败；分区模型是新增方法，不是调 p 即解决 |
| 表面“更白”但产品色差被抹平 | 原图旁路、固定参考和估计增益 | 区分自然图校正与计量；禁止未经批准改变待测产品本色 |

调参顺序为输入域／相机设置 → 真值与 mask → 核及边界 → n／σ → p → 增益与验收阈值。以同传感器域光源真值计算 `acos(clip(e_hat·e_true_hat,−1,1))`，说明角度单位、有效／拒判分母；报告中位数、尾部和分组偏差。原文式(18)是角误差，不是 ΔE、色温误差、概率或绝对照度误差。[cite:1](#ge-source-1)

### L4 数学推导：假设、导数与可辨识性

#### 成像模型与绝对值交换的条件

原文从 Lambertian 单光源谱积分出发：

\[
f_c(u)=\int_\omega E(\lambda)s(\lambda,u)c_c(\lambda)\,d\lambda,
\qquad e_c=\int_\omega E(\lambda)c_c(\lambda)\,d\lambda.
\]

若反射谱的空间平均对所有 λ 都等于 k，则交换可积函数的积分次序得到 `mean(f_c)=k e_c`，这说明强版 Gray-World 假设足够。但 Gray-Edge 还引入了幅值，原文式(12)–(14)将平均图像导数幅值与平均反射谱导数幅值联系起来，**不能仅用 Fubini 就证明绝对值可穿过光谱积分**。[cite:1](#ge-source-1)

对可积标量导数 h，且 `E·c_c≥0`，通常仅有

\[
\left|\int E c_c h\,d\lambda\right|\leq\int E c_c|h|\,d\lambda.
\]

等号需要相关波段导数无相消等额外条件；二维梯度范数的对应条件是向量同向而非任意方向。取两项 h=1、−1、权重均为1，左边0而右边2即为代数反例。因此本文不把“平均反射差无彩色”写成对任意宽带相机严格无偏估光的定理。原文假说是方法动机；下面采用明确的传感器对角近似给出可检验推导。

假设在固定线性传感器域与单光源下，可近似写成

\[
f_c(u)=e_c\rho_c(u),\qquad e_c>0,
\]

其中 ρ 是相应规范照明下的有效通道响应，而非已恢复的完整光谱反射率。窄带响应等条件可以支持此近似，但它不由“三通道图像”自动保证。空间不变 e 可提出卷积／导数；若 e=e(u)，则一阶已有 `∇(eρ)=e∇ρ+ρ∇e`，二阶还出现交叉项，单一全局估计便失去这一分解。

#### 高斯尺度、导数阶与二阶混合项

令 x 为列方向、y 为行方向；连续各向同性高斯

\[
G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right),
\quad f_c^\sigma=G_\sigma*f_c.
\]

可交换微分与卷积时，`f_x^σ=(∂xGσ)*f`，二阶同理。n=0 的 A 为 `|f^σ|`；n=1 为 `sqrt((f_x^σ)²+(f_y^σ)²)`。每个通道单独计算，不沿 RGB 轴求空间导数。[cite:1](#ge-source-1)[cite:3](#ge-source-3)

二阶必须先定义 `xx=∂xx f^σ`、`yy=∂yy f^σ`、**`xy=∂x∂y f^σ`**。作者 `gDer(f,σ,1,1)` 是普通混合偏导，并未除以 sqrt(2)，也不是将两个混合项相加后的另一个变量。对于足够光滑 f，Hessian 为

\[
H_c=\begin{pmatrix}xx&xy\\xy&yy\end{pmatrix},
\qquad A_{c,F}^{(2)}=\|H_c\|_F=\sqrt{xx^2+2xy^2+yy^2}.
\]

两个非对角元素各出现一次，故系数是2。标准 Frobenius 满足正交坐标变换下 `||QᵀHQ||F=||H||F`；Laplacian `|xx+yy|` 只取迹，鞍点可能相消，不能替换上述范数。

**固定作者包执行的是**

\[
A_{c,author}^{(2)}=\sqrt{xx^2+4xy^2+yy^2}.
\]

其行内虽称“frobenius”，但结合 `gDer` 的导数归一可确认并非标准 Hessian Frobenius。原文式(17)的简写没有展开 xx／xy／yy 权重，不能据原文替作者系数4补造定义。本文称它为“作者二阶响应”，不擅断作者意图，也不称改为2就是原版修复。[cite:3](#ge-source-3)

反例为 `H=diag(1,−1)`：旋转45°后对角为0、混合项绝对值为1。标准范数前后均为 sqrt(2)，作者响应从 sqrt(2) 变为2；因此不能宣称作者二阶定义严格旋转不变。它仍对单通道正比例缩放齐次，所以不妨碍下述**条件估光代数**，但会改变不同方向边缘的相对贡献。

#### Minkowski 聚合与估光归一

设共同有效域 V 的大小为 N>0，A 是固定阶数与范数定义的非负响应。用幂均值写为

\[
q_c=\left(\frac1N\sum_{u\in V}A_c(u)^p\right)^{1/p},\quad 1\leq p<\infty;
\qquad q_c=\max_{u\in V}A_c(u),\quad p=\infty.
\]

在 `f_c=e_cρ_c` 下，线性滤波和幅值齐次给 `A_c(f)=e_c A_c(ρ)`。令

\[
k_c=\left(\frac1N\sum_{u\in V}A_c(\rho)^p\right)^{1/p},
\quad q_c=e_ck_c.
\]

灰边／高阶灰边假设的明确传感器版本是 **三个 k_c 等于同一个 k>0**。于是 `q=ke`，再 `e_hat=q/||q||₂=e/||e||₂`。p=∞ 用最大值的齐次性同样成立；这不是从优化目标学出的 k，也没有隐藏的迭代求解器。[cite:1](#ge-source-1)

作者不用除 N，而计算空间 Lp 范数，结果只是乘上 `N^(1/p)`，最后归一方向相同。若各通道使用不同 mask／样本数，这个消去不再有保障；即使改用各自均值，也已破坏共同场景统计条件。n=0 的 p=1 和 ∞ 特例只在同一非负输入、V 和边界契约下成立。

对有限集合，令 M=max A，则 `M/N^(1/p)≤q_c≤M`，p→∞ 时幂均值趋于 M。对正 q、有限 p，有

\[
\frac{\partial q_c}{\partial A_c(u)}=\frac{A_c(u)^{p-1}}{Nq_c^{p-1}}.
\]

这解释高 p 为何强调强响应，而不是证明高 p 误差更小。作者 sum 版对应去掉 N 的导数。

#### 对角校正、尺度不确定性与误差

设目标传感器白响应 t 各分量正，通道校正为

\[
D=\operatorname{diag}\!\left(\alpha\frac{t_R}{\hat e_R},\alpha\frac{t_G}{\hat e_G},\alpha\frac{t_B}{\hat e_B}\right),\qquad J(u)=Df(u),
\]

α 是另定的公共亮度因子。作者采用 `t=(1,1,1)/sqrt(3)`、α=1，得到 L1 的公式。若估光正确且模型成立，则 `J_c=||e||₂ρ_c/sqrt(3)`，只恢复规范颜色比例，仍有公共尺度；不保证维持原曝光、亮度均值、动态范围或测色值。[cite:3](#ge-source-3)

给原图乘 a>0，在 V、σ、核不变且没有量化／饱和变化时，A、q 同乘 a，e_hat 不变，J 同乘 a。若为了比较输出各自自动拉伸／裁剪，这一齐次关系可能被破坏。

若实际 `k_c=k(1+δ_c)`，估计为 `normalize(e⊙(1+δ))`，δ 的公共部分可消去，但通道差异变成色偏。归一在非零 q 处的一阶扰动为

\[
\Delta\hat e\approx\frac{(I_3-\hat e\hat e^T)\Delta q}{\|q\|_2}.
\]

q 很小时方向对扰动敏感，校正又除以 e_hat_c，接近零的单通道可能放大噪声。此局部微分关系不是全局误差上界；无纹理常量图在理想 n>0 导数下 q=0，根本不可辨识。不能只加小分母并把输出登记为有效估计。

**成本**：核半径 r、单幅面积 HW、固定三通道；可分离滤波每个分量约 `O(HWr)`，一阶每通道两个、二阶三个分量，聚合 `O(HW)`；逐通道／分块累计可将响应存储降为工作缓冲，但要带 halo 并保持同一有效域。稠密二维卷积约 `O(HWr²)`。没有训练时间或收敛保证可报告，硬件速度须实测。

### L5 实现：固定源码、伪代码与有限核验

#### 作者入口及实际执行契约

| 文件 | 环节 | 已核验的实际行为 |
|---|---|---|
| `ColorConstancyDemo.m` | 调用与显示 | `double(imread(...))`；p=5、σ=2、n=1 示例；显示转 uint8，非函数内部裁剪 |
| `general_cc.m` | mask、聚合、估光与校正 | 默认 `(0,1,1)`；阈值255；mask=0有效；先保存原图；Lp sum 后 L2 归一；全原图除以 `white_c*sqrt(3)` |
| `norm_derivative.m` | 方向合并 | 一阶梯度幅值；二阶混合项系数4；文件主函数名为 `NormDerivative`，与调用／文件名不一致，移植需检查目标运行环境 |
| `gDer.m` | 可分离离散高斯导数 | `r=floor(3σ+0.5)`；离散矩归一；先横向 `filter2` 后纵向；补边后裁回原尺寸 |
| `fill_border.m` | 外边界 | 最近边界值复制，即 replicate／clamp，不是 reflect 或周期边界 |
| `dilation33.m` | 饱和扩张 | 3×3 最大值，边界复制；不是半径3或按σ扩张 |
| `set_border.m` | 统计边带 | 一基坐标保留 `width<x<W−width+1`、`width<y<H−width+1`，width=σ+1 |

上述均为外部作者源码，不是本仓库模块。[cite:3](#ge-source-3) 非整数 σ 时，`set_border` 依其不等式选择整数坐标；不能随意把 width 四舍五入。σ=2 时核半径6，但只去3像素边带，留下的部分响应仍依赖复制边界。n=0、σ=0 也去1像素边带并排饱和，故作者“Gray-World 示例”不必等于另一个库全图不筛选的 RGB 均值。

**离散导数归一**：一维采样 g 后，0阶令 `ΣK0=1`；1阶从 `−xg/σ²` 出发再除 `ΣxK1`，使一阶矩为1；2阶从 `(x²/σ⁴−1/σ²)g` 出发，先减其均值令 DC 为0，再除 `Σ(x²/2)K2` 令二阶矩为2。`filter2` 的相关约定与核符号归一需一起移植；不要直接换库卷积并假定所有中间导数符号一致。[cite:3](#ge-source-3)

上述矩约束使二维内部的 `xy` 对函数 `f(x,y)=xy` 返回普通混合导数1，而不是 `1/sqrt(2)`。本卡没有附加尺度归一因子 σⁿ；固定同阶、同σ下给所有通道同乘一个正因子会被最后归一消去，但混合不同尺度／阶数、设能量门限时不再可随意忽略。

**数值风险**：作者对整幅 `A^p` 后再乘 mask，巨大值可能先溢出，且 `Inf×0` 仍为 NaN；空域、全零和单通道零均无显式保护。安全移植应先选择有效有限样本，再用缩放或 log-sum-exp 累计。对每通道非零响应，`ℓ_c=(LSE(p log A_c)−log N)/p`，零响应取 log=-∞；随后以 `z_c=exp(ℓ_c−maxℓ)` 求 `e_hat=z/||z||₂`，不必还原巨大 q。p=∞ 独立分支。高增益或 exp 下溢仍需拒判，不能声称 log 域解决物理不可观测性。

#### 解释性伪代码

以下只描述一次前向计算，**未作为整图程序运行**。`mode=author` 保留固定包统计语义和系数4，但仍外加参数／退化检查；`mode=full_support` 是本卡工程变体，保守排除所有无效支持。若采用标准 Hessian 系数2，应另名 `hessian_frobenius` 并记录，不能与 mode 混成隐藏开关。

```text
EstimateGrayEdge(I, excluded, n, p, sigma, saturation, mode, second_norm):
    require RGB_float_HWC(I) and finite_nonnegative(I)
    require binary_HW(excluded) and n in {0,1,2}
    require sigma >= 0 and (n == 0 or sigma > 0)
    require (finite(p) and p >= 1) or p == infinity
    require finite(saturation) and saturation > 0
    require second_norm in {author_4, hessian_frobenius_2}
    r = 0 if sigma == 0 else floor(3*sigma + 0.5)
    S = any_channel(I >= saturation)
    if mode == author:
        require saturation == 255 and second_norm == author_4
        V = not (excluded or dilate_square_radius(S,1))
        V = keep_by_author_coordinate_inequalities(V,sigma+1)
    else if mode == full_support:
        V = not dilate_square_radius(excluded or S,r)
        V = keep_full_kernel_inside_image(V,r)
    else:
        return invalid_mode
    if count(V) == 0:
        return no_valid_support
    K = identity_or_required_author_kernels(sigma,n)
    if K has nonfinite_values or degenerate_moment_denominators:
        return invalid_discrete_scale
    for c in {R,G,B}:
        if n == 0:
            A = abs(I[c]) if sigma == 0 else abs(separable_filter(I[c],K0,K0,replicate))
        else if n == 1:
            dx = separable_filter(I[c],K1,K0,replicate)
            dy = separable_filter(I[c],K0,K1,replicate)
            A = hypot(dx,dy)
        else:
            xx = separable_filter(I[c],K2,K0,replicate)
            yy = separable_filter(I[c],K0,K2,replicate)
            xy = separable_filter(I[c],K1,K1,replicate)
            weight = 2 if second_norm == author_4 else sqrt(2)
            A = stable_norm(xx,weight*xy,yy)
        values = A[V]
        if nonfinite(values) or insufficient_channel_response(values):
            return unobservable_or_numeric_failure
        log_q[c] = log(max(values)) if p == infinity else stable_log_power_mean(values,p)
    e_hat = stable_L2_normalize_from_logs(log_q)
    gain = 1 / (sqrt(3)*e_hat)
    if nonfinite(gain) or violates_frozen_gain_limits(gain):
        return unsafe_correction
    J = I * broadcast_RGB(gain)
    if nonfinite(J):
        return numeric_failure
    return e_hat, J, gain, count(V), valid
```

`insufficient_channel_response`、增益门限需用相机噪声和任务约束预先冻结，本卡不编造数值。原包没有这些检查，安全版不声称与其异常输入行为一致。小σ可能令 r=0、1／2阶离散矩分母退化，即使 σ>0 也须核查；n=0、σ=0 必须走恒等分支，不能调用含 1/σ 的 `gDer`。

#### 已执行的有限数值检查

2026-11-05 使用现有 Python 3.12.14 环境的标准库 `math` 在内存执行 **8组断言，8/8通过**，未向仓库写测试脚本。下列数值为明确指定输入的代数检查，不是相机图像、数据集得分或统计性能结论。

| 检查 | 输入与运算 | 结果及证明范围 |
|---|---|---|
| 估光与校正链 | 两个有效响应向量 `(5,10,15)`、`(10,20,30)`，p=2；校正像素 `(10,20,30)` | 幂均值 q≈`(7.905694,15.811388,23.717082)`；e_hat≈`(0.267261,0.534522,0.801784)`；增益≈`(2.160247,1.080123,0.720082)`；输出三通道均≈21.602469，只验后半链 |
| sum／mean 等价 | 上例改用未除N的空间Lp范数 | e_hat 最大差<1e−14；仅共同域、共同p成立 |
| 离散核矩 | 按 `gDer` 公式计算 σ=2、r=6 的一维核 | `ΣK0=1`、`ΣK1=0`、`ΣxK1=1`、`ΣK2≈−1.73e−18`、`Σ(x²/2)K2=1`，误差<1e−12；未执行 MATLAB filter2 |
| 二阶旋转反例 | `diag(1,−1)` 与其45°旋转 | 标准范数均1.414214；作者响应1.414214→2；验证两种定义不可等同 |
| mask／边带支持 | 15×15、σ=2；原边带3；核半径6；中心单个饱和点 | 原边带留下81中心，饱和3×3排除后72；无饱和时完整核支持仅9中心；证明原边带不等于完整支持 |
| 大p稳定聚合 | 正数 `(1e200,2e200)`，p=100 | log域幂均值≈`1.986184990874202e200`，相对误差<1e−12且有限；直接幂将超double范围 |
| 符号抵消及光谱反例 | 空间导数 `(5,−5)`；谱积分两项 `(1,−1)` | 有符号和0而幅值和10；谱绝对和0而绝对值之和2，分别验证必要性及交换限制 |
| 退化拒判契约 | N=0、q=(0,0,0)、q=(1,0,2) | 检查器均拒绝；只验本卡前置条件，不声称原包已保护这些分支 |

响应向量检查没有覆盖图像到导数的全链，核矩检查也不是完整卷积等价测试。尚未执行：作者 MATLAB／Octave 调用、函数命名兼容测试、RGB/BGR 故障注入、整图 padding 与 mask 对照、标准二阶／作者二阶消融、真实数据角误差、计时与跨相机验证。

**后续验证设计**：先用常量／线性斜坡／`xy` 多项式的解析期望验证内点导数，再逐位置比较作者核、padding、V、A、q、e_hat、J；用同一图像受控曝光与RGB通道置换验证条件等变性，给饱和和排除区加扰动检查支持泄漏。优化前锁定核、xy权重、p和有效域；分块 halo 至少覆盖 r，不能在每块另做独立白平衡。浮点归约顺序变化允许给定容差，但不能只比较显示裁剪图。

### 五层面试追问与答题要点

| 层级／问题 | 参考答案与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| **L1：给一张RGB图，Gray-Edge从哪里估光，校正谁？** | 固定线性域与共用V；逐通道高斯导数幅值→空间Lp统计→三维L2归一；对原RGB按倒数增益校正；空域／无响应拒判 | 在灰度边图上估三个颜色；把光源向量直接乘原图；校正导数图而不是原图 | L1、估光算例；[cite:1](#ge-source-1)[cite:3](#ge-source-3) |
| **L2：为什么既平滑又取幅值？p=∞是不是找白点？** | 平滑规定局部尺度并抑制高频，幅值防正负抵消；p=∞逐通道取最大响应，n=1时是Max-Edge，不要求同位置 | 先求有符号平均后取绝对值；把最大边缘叫Max-RGB；认为σ只改速度 | L2、Table I、符号反例；[cite:1](#ge-source-1) |
| **L3：换成[0,1]图且边缘反光，先调p还是σ？** | 先核对饱和阈值尺度、线性域、mask极性和核支持；原255阈值失效，一像素膨胀不足；再固定n比较σ和p，并记录拒判 | 直接增大p；把mask乘图像制造黑边；认为原σ+1边带覆盖3σ核 | L3、mask算例；[cite:3](#ge-source-3) |
| **L4：推导q∝e，解释二阶xy系数及退化。** | 在空间恒定对角模型及三个反射响应Lp统计相等时，齐次性给q=ke；k>0才能归一。标准Hessian系数2，作者普通xy下系数4；广谱绝对值交换另需条件 | Fubini允许任意绝对值交换；源码称Frobenius所以必旋转不变；无纹理时加epsilon就可辨识 | L4、旋转与光谱反例；[cite:1](#ge-source-1)[cite:3](#ge-source-3) |
| **L5：公式一样却对不上作者输出，查什么？** | 查发布包哈希、n/p/σ默认、−1哨兵、mask反向注释、255、replicate、离散矩、系数4、sum与L2、sqrt(3)、输出未裁剪；再逐中间量比较 | 直接调用库Gaussian/Sobel/Laplacian便称等价；仅看uint8显示；忽略NaN×0和函数名差异 | L5源码表、8组检查；[cite:3](#ge-source-3) |

### 来源、固定快照与访问范围

各来源核验日期均为 **2026-11-05**；相同作者稿镜像不是独立研究证据。在线zip没有永久版本URL，本卡以“作者入口＋完整包与逐文件SHA-256”固定所读内容；以后重新下载须先核对哈希，不能以同一文件名假定内容不变。

- <a id="ge-source-1"></a>**[cite:1] 原始论文／作者稿。** [HAL入口](https://hal.inria.fr/inria-00548686/PDF/IP07_vandeweijer.pdf) 成功返回书目及正文至式(12)附近但截断；另从 [LaBRI教学托管的同题作者稿](https://dept-info.labri.fr/~achille/ti/Activites/Note-de-lecture/Articles/edge_based_color_constancy_2007.pdf) 取得17页PDF，在内存按公式／关键词提取。实读范围包含成像与Gray-World段、式(12)–(17)、参数解释、式(18)及Table I、真实图实验的灰球排除段与Table IV；未逐项复核全部实验数据和图像。稿内正文页码100–115／参考文献不等于期刊页2207–2214。PDF SHA-256：`b433b7f39128325b3981f210d4568c35000f4f9b33b243b553000263e173a25a`。光谱绝对值限制与对角模型推导为本卡明确条件的数学分析，不冒称论文已证明的普适结论。
- <a id="ge-source-2"></a>**[cite:2] 作者软件发布入口。** [Joost van de Weijer／LEAR Software](https://lear.inrialpes.fr/people/vandeweijer/software.html)：成功读取HTML及其ColorConstancy.zip链接，明确是作者edge-based实现并含Grey-World、max-RGB、Shades of Grey，页面称未针对速度优化。[后续UAB软件页](http://www.cat.uab.cat/~joost/software.html) 可取说明文字，但该站zip直连未成功；本卡实际锁定的是LEAR返回的文件，不声称两个站包字节相同。
- <a id="ge-source-3"></a>**[cite:3] 作者固定源码快照。** [ColorConstancy.zip](https://lear.inrialpes.fr/people/vandeweijer/code/ColorConstancy.zip)：成功下载至内存并读取七个`.m`文件；完整包SHA-256为 `e392da9d3c147db5cacc2cc3ea04acb02aa1fa1c54192aede9e27a5947de28a0`。包还含JPEG／`.asv`，本卡不执行备份脚本，不以JPEG演示构成性能核验；主脚本／导数文件zip内时间戳为2007-05-16，只作元数据。包中未见独立LICENSE文件，未核验商业复用许可，不将可下载等同明确开源授权。源码仅静态阅读，没有安装或运行MATLAB。
- <a id="ge-source-4"></a>**[cite:4] 作者体系研究资料站。** [ColorConstancy Source-code](https://colorconstancy.com/source-code/index.html)：成功读取公开页，分别列2007 Grey-Edge和2012 Weighted Grey-Edge，前者链接作者旧UAB包。该UAB链接本次超时，LEAR入口成功；此页用于来源链与版本区分，不提供独立性能复现证据。

| `ColorConstancy/` 下文件 | 原始字节 SHA-256 |
|---|---|
| `ColorConstancyDemo.m` | `fa2f049681bb71e89c004ed5fd533a7dbaff477064e80ebb2871ee8fee2e954e` |
| `general_cc.m` | `cfa453b9587f556e00a40b3f4b93c341cc0494ab8fcbacdd068c0cf9e356f23b` |
| `norm_derivative.m` | `94538a1278e966dc3cf761638732cfa2588005b70764c46f2fd6182fb7b5ad76` |
| `gDer.m` | `f0f3fe671a7bcc0eb721d0bd7b9e3f05cafb2145c06fbc6580c368d462bd8bbf` |
| `fill_border.m` | `f579e5a64c37a07bd40e37edc87bd279c10fc495277b43547c3ca2333134ece4` |
| `dilation33.m` | `00974f5b07f7128897ca590957ce747ccdcaba6b5149ee33ec71801585c2bd7b` |
| `set_border.m` | `a5cb42152c613213a09f92c00fcbc3e0e2814c6728b5dd27751f23f7cab67c2e` |

### 未实现与 G33 交接建议

本卡补齐了导数阶／高斯尺度／Minkowski范数、估光归一及对角校正的机制，也保留二阶混合项、mask、饱和、边界、原文假设条件及作者实现差异。**建议G33仅按“机制专卡缺失已补齐”销项**，可登记为已有GE01；不是标记原程序复现完成，更不是颜色域全面L5完成。

L3／L5继续为部分，本仓库继续未实现；作者代码全链对照、数据集复测、许可确认、现场测色与验收均未完成。本次仅新增本文件，未改覆盖矩阵、逐卡索引、颜色总览或其历史计数；任何后续登记须按本卡限定状态转录，不因答5或8组算术通过而升级实现等级。
