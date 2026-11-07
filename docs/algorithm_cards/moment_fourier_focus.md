# 矩定位与傅里叶单帧对焦专卡

[学习规范](../algorithm_learning_standard.md) · [测量总览](../measurement_evolution.md) · [对焦总览](../autofocus_evolution.md) · [缺口账](../evolution_coverage.md) · [原M05／F14卡](measurement_focus.md)

## 范围、覆盖与销项口径

核验日期：2026-11-05。本文只有两张独立卡，唯一键为 `moment_fourier_focus.md:MF01`、`moment_fourier_focus.md:MF02`，分别对应 G15／05-11、G43／11-08。它们不是旧 M05 的泛 Zernike／A31 教学方案，也不是旧 F14 的普通灰度 FocusNet。本文不修改总索引、206 卡计数或缺口账。

| 卡片 | 演进定位与新增内容 | L1／L2／L3／L4／L5 | 实现与验证范围 |
|---|---|---|---|
| [MF01](#mf01) Renshaw–Christian 2020 | 阶跃矩定位 → 有限宽斜坡／楔形；明确 A11、A20 约定、近似反解与宽度先验 | 已核验／已核验／部分／已核验／部分 | 核验指原文关键节和条件代数；仓库未实现，未运行图像定位 |
| [MF02](#mf02) Pinkard 等 2019 | 普通单帧方向歧义 → 离轴照明与傅里叶幅度特征共同设计，监督回归焦点修正 | 已核验／已核验／部分／部分／部分 | 作者存档正文片段、固定版本源码静态核对；仓库未实现 FCFNN，未训练或接设备 |

“已核验”不是全域精度保证。L3 的现场定参、MF02 的完整光学可辨识性证明，以及两卡 L5 的端到端复现均未完成。原文实验、作者源码行为、本文数学解释、本文已执行算术核验、尚未执行的测试设计分开标注。

**销项建议只针对“缺少对应原方法专卡”**：G15 可建议文档缺卡销项；G43 已具备光路、标签、傅里叶前端和作者训练实现的对应说明，可建议文档缺卡销项，但应保留“正式补充材料未取得、作者代码版本与发表快照未逐项对齐”的来源缺口。若销项标准是完整原版复现、全部 L1–L5 已核验或现场验证，则两项均不满足。该建议不表示已经修改缺口账。

<a id="mf01"></a>

## MF01 Renshaw–Christian 2020 斜坡／楔形矩求解

### 任务、演进与适用前提

Devin T. Renshaw、John A. Christian，*Subpixel Localization of Isolated Edges and Streaks in Digital Images*，Journal of Imaging 6(5):33，2020-05-18，DOI `10.3390/jimaging6050033`。方法属于已有像素候选的局部亚像素精化，不是整图候选检测、边链连接或尺寸标定。边缘使用线性斜坡 ESF，条纹使用三角楔形 LSF；这里只解释其有限宽模型、矩积分和非迭代近似解，不将 2020 年工作反归为 1993 年 A31 模糊补偿。[cite:1](#source-1)[cite:2](#source-2)

| 项目 | 契约 |
|---|---|
| 输入 | 灰度图、整数候选中心、类型 edge／streak、奇数窗边长 \(N_p\)、已知或独立标定的模糊宽度、条纹明暗极性 |
| 输出 | 亚像素 \((u,v)\)、法向角 \(\psi\)、归一化修正 \(\ell\)，以及有效／拒绝状态；不是 mm 尺寸或概率置信度 |
| 成立前提 | 窗内单条近似直边／条纹、近恒定背景、沿切向近似不变、宽度可信、非饱和且候选足够接近 |
| 改进与代价 | 用有限宽模型减少阶跃失配；只需三个实相关，但要确定 PSF→模型宽度映射，仍有近似和像素化偏差 |
| 当前实现 | 仓库未实现；[edge_subpixel.py](../../traditional/subpixel/edge_subpixel.py)仅提及 Zernike 待办，没有本卡的矩核／楔形求解入口 |

输入中心记为 \((\tilde u,\tilde v)\)。原文以 u 向右、v 向下，像素中心在整数坐标。以下先处理亮条纹；暗条纹须先反相为亮条纹。反相属于实现适配，不能在不处理极性的情况下直接复用“指向条纹中心”的方向解释。

### L1 步骤：从候选到非迭代修正

1. 由其他检测器提供候选，取完整 \(N_p\times N_p\) 奇数窗。边界缺窗、非有限值、明显饱和或多结构混入时拒绝，不补零制造新边缘。
2. 按原文 Eq.1 映射 \(x=2(u-\tilde u)/N_p\)、\(y=2(v-\tilde v)/N_p\)。单位圆半径在图像中是 \(R=N_p/2\)，**不是** \((N_p-1)/2\)。
3. 用像素单元与单位圆交集上的积分核计算 \(\Re A_{11},\Im A_{11},A_{20}\)，原文 Eq.18–23 是相关，不是翻核卷积。只在候选处算即可。
4. 算 \(\psi=\operatorname{atan2}(\Im A_{11},\Re A_{11})\)，以及 \(D=\Re A_{11}\cos\psi+\Im A_{11}\sin\psi=|A_{11}|\)。非退化亮结构下 \(Q=A_{20}/D\)。
5. edge 使用 Eq.41 的斜坡近似反解；streak 使用 Eq.57 的楔形经验修正式。**同一个 Q，不同的模型，不能用同一个位置公式。**
6. 检查根号、宽度、局部支持和最大允许修正，回填 \((u,v)=(\tilde u,\tilde v)+N_p\hat\ell(\cos\psi,\sin\psi)/2\)。每候选仅一次闭式修正，无训练和迭代收敛停止条件；无效候选返回原因。[cite:1](#source-1)

### L2 机制：消除背景与对比度，而非消除成像失配

连续单位圆上 \(\int_D x\,dA=\int_D y\,dA=\int_D[2(x^2+y^2)-1]dA=0\)，故常量背景 h 不进入这两个矩。法向对齐后关于切向对称，使 \(\Im A'_{11}=0\)；复矩相位给法向，不能将其丢掉再声称知道方向。两个非零矩都与对比度 k 成正比，取比值消去 k，不等于消除任意曝光变化：饱和、背景梯度和量化会破坏这些条件。

斜坡明确容纳模糊边的过渡宽度；楔形同时描述条纹的上升与下降两侧，输出峰脊位置而不是任一侧边。删除宽度修正会回到薄结构近似；拿 edge 解处理条纹甚至会把负 Q 解释为负向边缘位移。增大窗可增强部分噪声平均，却更容易包含曲率、邻边和条纹端点，不是单调提高精度。[cite:1](#source-1)

### L3 工程权衡：先冻结矩约定，再调窗口与宽度

| 参数／症状 | 单位与证据 | 调整顺序、代价和回退 |
|---|---|---|
| \(N_p\) | 原文举 5、7 像素奇数窗，数字图比较使用 5×5；不是所有工况最优值 | 先核候选和结构孤立性，再对比不同窗的残差、偏差及耗时 |
| \(\sigma_{px}\)、w | 原文 Eq.33 推荐 \(k_{edge}=1.80\)，Eq.43 用 \(k_{streak}=0.90\)；\(w=2k\sigma_{px}/N_p\)，全宽为 2w | k 是作者 Gaussian PSF 实验下的工程拟合，非 FWHM 恒等式；换光路／离焦后重新标定，不从两个矩自动估计未知宽度 |
| 系统性偏移 | 检查 R 是否误取 \((N_p-1)/2\)、共轭／归一化、宽度 px→圆域转换 | 先排符号尺度，再查模型；不能先加任意经验偏置 |
| \(D\) 弱、Q 爆炸 | 平坦区／低对比、居中对称条纹均可能 \(A_{11}\approx0\) | 不统一加 epsilon 伪造方向；报告退化，必要时回退原候选并标为未精化 |
| 根号负或 \(\lvert\hat\ell\rvert+w\ge1\) | 宽度／候选／多边／强噪声不符合所用完整支撑模型 | 拒绝或重检测；仅舍入级负数可按误差尺度处理，不能把所有负根号夹零 |
| 暗条纹反方向 | 明暗反转使 A11、A20 同时变号，但 atan2 方向翻转 | 先统一条纹极性再求解；边缘暗→亮法向可翻转，位置应在模型精度内一致 |
| 曲线边、条纹交叉 | 沿切向不变假设不成立 | 缩窗或换模型，统计有效率而非只统计成功返回样本 |

原文数字实验每种噪声／模糊组合使用 5000 张随机图，真实数据部分主要给定位叠图；这不构成真实图的通用亚像素误差上界。原文也指出极低噪声下 PAE 可更优，不能把摘要的“优于现有方法”无条件泛化。本文不重建这些实验、不填入未经读取的 RMSE 数字。[cite:1](#source-1)[cite:2](#source-2)

### L4 数学推导：原文矩、精确前向积分与近似反解

**1. 约定必须与旧 M05 分开。** 原文 Eq.7、8、15、17 采用

\[
A_{11}=\int_D(x+iy)I(x,y)\,dA,\qquad
A_{20}=\int_D[2(x^2+y^2)-1]I(x,y)\,dA,
\quad A'_{11}=e^{-i\psi}A_{11}.
\]

这里**不取共轭，也不乘 \((n+1)/\pi\)**。原文另定义 \(Z_{nm}=(n+1)A_{nm}/\pi\)。旧 [M05](measurement_focus.md#m05-zernike-矩边缘)把带共轭、带归一化的量称 A，因此若使用旧量 \(A^{old}\)，本卡 Q 应为 \(2A^{old}_{20}/(3|A^{old}_{11}|)\)，本卡角度为 \(-\arg A^{old}_{11}\)，不能把旧卡的 2/3 再乘到本卡原始矩比上。[cite:1](#source-1)

**2. 将二维积分约成一维。** 法向坐标记 \(t=x\cos\psi+y\sin\psi\)，切向为 s。令 \(B(t)=\sqrt{1-t^2}\)。积分 s 从 \(-B\) 到 B，得到

\[
A'_{11}=\int_{-1}^1 2t\sqrt{1-t^2}\,I(t)\,dt,
\qquad
A_{20}=\frac23\int_{-1}^1(4t^2-1)\sqrt{1-t^2}\,I(t)\,dt.
\]

第二式来自 \(\int_{-B}^B[2t^2+2s^2-1]ds=\frac23(4t^2-1)B\)。它解释了两类结构为什么共享相同矩核。下列前向积分假设 \(|\ell|+w<1\)，否则必须重新处理截断支撑，不能将超域的 arcsin 参数强行夹回区间。

**3. 斜坡模型与精确矩。** 原文 Eq.34 为

\[
I_E(t)=
\begin{cases}
h,&t\le\ell-w,\\
h+k\dfrac{t-\ell+w}{2w},&\ell-w<t\le\ell+w,\\
h+k,&t>\ell+w.
\end{cases}
\]

设 \(a=\ell-w,b=\ell+w\)，\(B_a=\sqrt{1-a^2},B_b=\sqrt{1-b^2}\)，以及

\[
F(t)=3\arcsin t+(5-2t^2)t\sqrt{1-t^2}.
\]

逐段积分得原文 Eq.35、36 的紧凑等价写法：

\[
A'_{11,E}=\frac{k}{24w}[F(b)-F(a)],\qquad
A_{20,E}=\frac{k}{15w}(B_a^5-B_b^5).
\]

这是该斜坡模型的**精确前向矩**，不是 Eq.41 的精确逆。原文 Eq.40 采用

\[
Q_E\approx\ell\left[1-\left(1+\frac\ell2\right)w^2\right].
\]

移项为 \(\tfrac12w^2\ell^2-(1-w^2)\ell+Q_E\approx0\)，取连续接近阶跃解的根：

\[
\hat\ell_E=\frac{1-w^2-\sqrt{(1-w^2)^2-2w^2Q_E}}{w^2}.
\]

当 w→0，精确阶跃矩满足 \(A'_{11}=2k(1-\ell^2)^{3/2}/3\)、\(A_{20}=\ell A'_{11}\)，因此 \(\hat\ell_E\to Q_E\)。为避免小 w 下相消，可使用有理化等价式

\[
\hat\ell_E=\frac{2Q_E}{1-w^2+\sqrt{(1-w^2)^2-2w^2Q_E}}.
\]

该数值改写是本文代数解释，非新增定位模型。还应注意：精确有限宽 Q 随 \(\ell\) 为奇函数，而 Eq.40 含 \(\ell^2\) 项，不严格保留该对称性；故正负位移／反极性误差必须分测，不能把此近似当无偏恒等式。[cite:1](#source-1)

**4. 楔形模型、薄线极限与有限宽修正。** 原文 Eq.44 可写为

\[
I_S(t)=h+k\max\left(1-\frac{|t-\ell|}{w},0\right).
\]

令 \(C=\sqrt{1-\ell^2}\)，分段积分即 Eq.45、46：

\[
A'_{11,S}=\frac{k}{12w}[2F(\ell)-F(a)-F(b)],\qquad
A_{20,S}=\frac{2k}{15w}(B_a^5+B_b^5-2C^5).
\]

薄楔形极限中面积约为 kw，一维核在峰处取值给出

\[
A'_{11,S}\sim2kw\ell\sqrt{1-\ell^2},\qquad
A_{20,S}\sim\frac23kw(4\ell^2-1)\sqrt{1-\ell^2},
\quad Q_{S0}=\frac{4\ell^2-1}{3\ell}.
\]

求解 \(4\ell^2-3Q\ell-1=0\)。亮条纹规定法向从候选指向峰脊，故 \(\ell\ge0\)；\(\ell\to0^+\) 时 Q→−∞，选择能趋近 0 的正根。原文 Eq.52 为

\[
\hat\ell_{S0}=\frac38Q+\sqrt{\left(\frac38Q\right)^2+\frac14}.
\]

原文 Eq.53–57 用满足薄线极限的参数模型作最小二乘拟合，保留系数 \(a_1=3/8,a_5=1/4\)，经验得到 \(a_3\approx-1/10\)：

\[
\hat\ell_S=\frac38Q_S+\sqrt{\left(\frac38Q_S\right)^2-\frac1{10}w^2+\frac14}.
\]

**根号内是 \((3Q/8)^2\)，不是 \(3Q^2/8\)**；已核对原文 MathML 括号。\(-w^2/10\) 是有限宽经验拟合，不是 Gaussian 积分精确结果，更不保证所有 w。令 \(t=3Q/8,c=1/4-w^2/10\)，t<0 时用 \(\hat\ell_S=c/(\sqrt{t^2+c}-t)\) 减少相消；t≥0 时直接相加。[cite:1](#source-1)

**5. 可辨识性、误差和复杂度。** Q 消去了 k，剩下 \(Q(\ell,w)\) 只有一个标量约束；w 未知时不能由这个比值唯一解出位置与宽度。\(A_{11}=0\) 时角度不可辨，尤以正居中的对称条纹为例；它与平坦背景不能只凭“小矩”区分。三次实相关在 K 个候选上为 \(O(KN_p^2)\)，矩核预存为 \(O(N_p^2)\)。若 \(\hat\ell=g(Q,w)\)，一阶扰动为 \(\delta\ell\approx g_Q\delta Q+g_w\delta w\)，其中 \(\delta Q\approx\delta A_{20}/D-A_{20}\delta D/D^2\)；小 D 和临界根号均可放大噪声。这是局部敏感性解释，不是统计置信区间或通用理论误差界。

### L5 实现：矩核、数值保护与可检验接口

**数据布局。** 图像按 `[row=v, col=u]` 存储。偏移为 `(s,k)` 的像素单元是 \([2(s-1/2)/N_p,2(s+1/2)/N_p]\times[2(k-1/2)/N_p,2(k+1/2)/N_p]\)。分别对其与圆域交集积分 x、y、\(2(x^2+y^2)-1\)，得到三个实核。不要仅以“像素中心落在圆内”二值取样充当原文像素面积积分。应检查 x 核沿列反对称、y 核为转置、三个核的常量响应近零；核和相关累加先用 float64。核积分精度、边界交面积和旋转误差须记录。[cite:1](#source-1)

以下为**未接入仓库的复现伪代码**。`tau_moment`、合法宽度范围和 `max_shift_px` 是外部校准质量门，不是作者统一默认阈值。

```text
refine(image, center_uv, kind, polarity, Np, sigma_px, masks, gates):
    require Np odd and Np > 1
    patch = complete_patch(image, center_uv, Np)
    if invalid_patch(patch): return REJECT_INPUT
    if kind == streak and polarity == dark: patch = -float64(patch)
    X = sum(float64(patch) * masks.x)
    Y = sum(float64(patch) * masks.y)
    Z = sum(float64(patch) * masks.radial2)
    D = hypot(X, Y)
    if D <= gates.tau_moment: return DEGENERATE_MOMENT
    psi = atan2(Y, X)
    Q = Z / D
    k_model = 1.80 if kind == edge else 0.90
    w = 2 * k_model * sigma_px / Np
    if not gates.valid_width(kind, w): return REJECT_WIDTH
    if kind == edge:
        rad = (1 - w*w)^2 - 2*w*w*Q
        if rad < 0: return REJECT_MODEL
        ell = 2*Q / (1 - w*w + sqrt(rad))
    else:
        t = 3*Q/8
        c = 1/4 - w*w/10
        if t*t + c < 0: return REJECT_MODEL
        r = sqrt(t*t + c)
        ell = c/(r-t) if t < 0 else t+r
    if not finite(ell) or abs(ell)+w >= 1: return REJECT_SUPPORT
    if Np*abs(ell)/2 > gates.max_shift_px: return REJECT_SHIFT
    return center_uv + (Np/2)*ell*(cos(psi), sin(psi)), psi, ell
```

**数值核验而非图像实测。** 使用原文图示的 \(N_p=5\)、图注所列 \(\sigma_{px}=0.3\) 和作者宽度系数，可算出 \(w_E=0.216,w_S=0.108\)。仅为核验公式，取解析测试点 \(k=1,\ell=0.2\)，比较分段一维积分与闭式前向矩；这些人为指定参数不是采集数据、论文样本或性能实验。

| 解析测试 | \(A'_{11}\) | \(A_{20}\) | Q | 近似反解 \(\hat\ell\) | 换算像素修正 |
|---|---|---|---|---|---|
| 斜坡 | 0.6125879485 | 0.1165134316 | 0.1901987002 | 0.2004904868 | 0.5012262171 px |
| 楔形 | 0.0421965443 | −0.0586941257 | −1.3909699629 | 0.2001303250 | 0.5003258124 px |

该点理想修正是 0.5 px，差异正好说明近似逆并非精确；不是声称精度达到表内小数位。已执行 SciPy 分段积分与等价式核验，前向矩最大绝对差小于 \(2\times10^{-15}\)。尚未执行像素积分核、候选检测或整图定位。

**测试与优化计划。** 必须覆盖常量核泄漏、矩约定换算、90°旋转、正负位移、明暗极性、w→0、居中条纹、靠圆边界、未知宽度、曲率／双边、饱和及候选误差。报告法向定位误差、拒绝率和宽度失配敏感性；真实图需独立位置真值。确认正确性后缓存同尺寸核、批量提取候选窗；全图相关与候选相关只能在相同边界和核定义下作等价比较。面积核生成、实际运行及物理尺度链均未实现。

### 五问五答：贯穿 L1–L5

| 层级与问题 | 答案要点、成立条件与易错项 | 证据入口 |
|---|---|---|
| L1：给一个像素候选怎样返回点？ | 完整窗→圆域积分矩→atan2→Q→选 edge／streak 解→乘 Np/2→有效性检查。不能把 l 当 px，不能先丢复矩方向。 | 原文 Eq.1、18–25、31、41、57 |
| L2：背景和对比度为什么消掉？ | 完整连续圆域中基函数对常量积分为零；同模型两矩含共同 k。背景梯度、饱和和离散核泄漏都是反例。 | 本卡 L2、原文 Eq.35–39、45–48 |
| L3：模糊增加时只增大窗口够吗？ | 先核 PSF 的 sigma 与归一化 w，再查曲率和邻边；更大窗有模型污染代价。1.80／0.90 是特定实验拟合，不是所有 PSF 常数。 | 原文 Eq.33、43；本卡参数表 |
| L4：楔形为何取正根，−w²/10 又从何来？ | 由薄线比值推二次式，Q→−∞ 时选择趋零根；有限宽项来自受薄线极限约束的拟合，不是严格积分逆。 | 原文 Eq.49–57 |
| L5：公式正确但结果翻转／放大 1.5 倍，先查什么？ | 共轭符号、按阶归一化、行列和相关翻核；检查 R=Np/2。居中条纹 A11=0 时应报退化而非加 epsilon 强行定位。 | 原文 Eq.14–23；旧M05；本卡实现协议 |

<a id="mf02"></a>

## MF02 Pinkard 2019 离轴照明与 FCFNN

### 任务、版本与仓库边界

Henry Pinkard、Zachary Phillips、Arman Babakhani、Daniel A. Fletcher、Laura Waller，*Deep learning for single-shot autofocus microscopy*，Optica 6(6):794–797，2019，DOI `10.1364/OPTICA.6.000794`。正式在线日期为 2019-06-05；Berkeley 的 UCB/EECS-2019-161 档案封面日期 2019-12-01、内稿日期 2019-05-13，不能当成方法首次提出日期。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

这是**照明设计＋特征选择＋监督回归**，不是数字重聚焦，也不是普通单帧 CNN 的同义词。训练需要焦栈，训练后单次预测只需要一张对应照明图；“single-shot”不表示训练免扫描、不表示运动稳定后无需复核。[cite:3](#source-3)

| 项目 | 契约 |
|---|---|
| 输入 | 与训练一致的离轴照明单张强度图、固定光学／采样参数、训练好的 FCFNN 与特征统计量 |
| 输出 | 以 µm 表示的焦点修正 \(\Delta z=z_*-z\)；不是清晰度分数、复原图或绝对平台坐标 |
| 光路前提 | 稳定透射成像通路、离轴 LED、物镜／相机／电动 z 轴；可编程 LED 阵列便于切换但不是必要硬件 |
| 训练前提 | 同一位置两种照明的配对焦栈、可靠 z 标签、可测峰的参考通道、覆盖部署样本类型 |
| 演进代价 | 用照明与频域先验压缩学习问题，但增加光路一致性、数据采集、训练、域外拒绝和闭环验收成本 |
| 仓库状态 | FCFNN 未实现；[autofocus_cnn.py](../../dl/autofocus_cnn.py#L23-L50)是三层 stride=2 卷积＋池化＋标量头，有 Adam/MSE 循环但不是本方法 |

特别注意：旧 F14 约定 \(d=z-z_*\)，作者数据生成代码实际用 \(\Delta z=z_*-z\)。本卡选择作者符号，控制量是 **z_new=z+Δz**。把本地 d 直接送入相加式会反向运动。[cite:9](#source-9)

### L1 步骤：光路、标签、训练和推理分开

**光路与采集。** 作者使用 Zeiss Axio Observer、20×／NA 0.5，quasi-dome LED 阵列替换照明。正文示范单颗 LED 相对光轴 24°；来自局限空间光源的照明在样品面近似空间相干，并不是“所有 LED 发光天然完全相干”。参考通道同时点亮多颗 LED，以降低空间相干性；透明样本采用半环形非对称照明获得相位对比，帮助参考焦点评分。两个通道经过同一样本、物镜和相机，不新增一个对焦检测相机。[cite:3](#source-3)[cite:5](#source-5)

**训练流程。**

1. 在多个 XY 位置采配对焦栈。正文称覆盖约 60 µm、步长 1 µm，并在实验说明写每栈 60 层；不要据此擅自构造含两个端点的 −30…+30 共61层。保存实际 z 位置与通道映射。
2. 从低相干参考通道计算各层径向平均 log 功率谱的高频总量，取焦点峰作为 \(z_*\)。每张离轴通道图成为一个输入，标签为 \(z_*-z_i\)。该“ground truth”是确定性评分代理，不是无误差的独立光学真值。
3. 按完整 XY 位置／样本分训练和验证，不能随机拆同焦栈的相邻层。作者 notebook 明确先分位置，再编译特征。
4. 对单帧分块，计算 FFT 幅度，按照明方向选频域区域，展平、逐例 L2 归一化；训练集逐特征均值／标准差用于标准化。
5. 训练全连接回归网络，按独立验证集 RMSE 监控并早停；保存网络权重、标准化统计、裁剪与光学配置。作者代码的具体参数、版本和导出限制见 L5。[cite:3](#source-3)[cite:7](#source-7)[cite:8](#source-8)[cite:9](#source-9)

**推理与停止。** 同照明采一帧→同样分块和频谱特征→关闭 dropout→回归每块修正→作者代码对同帧块预测取中位数。安全控制层先检查方向、行程、置信依据和最大步长，再执行 z+Δz、等待到位并独立复核；这些质量门是本文部署建议，不冒充作者已给出的完整安全控制。论文的单帧预测本身无迭代停止保证，复核失败应受控重扫或拒绝。

### L2 机制：为什么是离轴照明与傅里叶幅度

近空间相干离轴照明下，散射光与未散射光形成随离焦改变的强度纹理；离焦不只是普通非相干图像的对称模糊程度。作者据单次散射频谱支撑选取输入区域，避免把没有主要离焦信息的全部频率都输入大网络。已核读正文称该设计比所比较 CNN 少 2–3 个数量级参数及训练内存；这只是作者比较条件下的报告，不能由本地 FocusNet 参数量直接证实。[cite:3](#source-3)

幅度谱相较相位弱化了图像平移／精确结构位置的影响；作者的幅度与相位对照表明幅度输入在其数据上泛化更好。**用于网络的是相机强度图的 Fourier 幅度，不是光场的 Fourier 幅度**。后者在理想无截断传播时可能只变相位，不能用来否定前者所含的离焦纹理信息。另一方面，丢掉相位并非总是无损，某些对象、角度和频谱裁剪仍可能无法区分正负离焦。

单颗 24° LED 是正文的简化示范；作者后续采集教程说，同侧直线排列约10颗 LED 的图案可表现更好。它们是不同照明方案，不应把“10颗”混成正文单颗实验的配置。多颗不相干源同时开启通常是强度叠加，不可把不同 LED 当成固定相位的光场相加。[cite:3](#source-3)[cite:5](#source-5)

### L3 工程权衡：采集契约优先于加深网络

| 对象／症状 | 量纲与证据 | 处理顺序及代价 |
|---|---|---|
| 照明角度／方向 | 正文 24°；空气近似 \(NA_{illum}=\sin24°\approx0.4067\)，不等于物镜 NA 0.5 | 换角度、波长、物镜或相机采样会改变频谱尺度；先重标频域裁剪和数据，再考虑微调 |
| 参考焦点峰失真 | 1 µm 是采样步长，不是焦点误差上界；透明／厚样本可能多峰 | 检查低相干参考通道、相位对比和实际 z 轴；端点峰、立方插值过冲不能当可靠标签 |
| 特征裁剪 | 作者 notebook 使用 `led_width=0.6, non_led_width=0.1` 等配置，都是相对块宽无量纲比例 | 先核 LED 对应哪条数组轴；过窄会丢信息，过宽显著增大首层权重和缓存；不是通用最优值 |
| 平坦／黑帧 NaN | L2 范数或逐特征标准差可能为零；参考 log(0) 可为 −Inf | 拒绝无信息帧，固定常量特征政策；不可仅给任意 epsilon 就声称原实现已健壮 |
| 训练好、换组织失败 | 正文细胞训练在组织切片上失效；追加130个组织焦栈与原440个细胞焦栈后改善 | 增加部署域训练数据并按样本验证，不宣称天然跨域或零样本泛化 |
| 预测大致正确但运动反向 | 作者 Δz 与本地 d 相反；设备正方向还需实机标定 | 先做小幅受控方向测试和限位，不靠重新训练掩盖符号错误 |
| 耗时与内存 | 作者报告2048²图 CPU预测约50 ms；440栈训练 CPU约1.5 h／GTX1080Ti约30 min，另有每栈约2 min预计算 | 是论文报告、非本仓库复测；曝光、运动、稳定及预计算必须另计，不把推理延迟当闭环节拍 |

这些实验条件和数值来自作者存档正文，未重新测量。验证应按设备／日期／样本隔离，分报 µm RMSE、MAE、尾部误差、方向错误率、拒绝率及实际闭环失败率；少量同片相邻视野的好结果不能代替部署域验证。[cite:3](#source-3)[cite:7](#source-7)

### L4 数学推导：物理信息、训练目标与有限保证

**1. 光场传播到相机强度。** 以下是帮助理解作者设计的标量、薄样本、单色和近轴模型，**不是声称从未取得的 Note S2 抄得公式**。令样本复透射率 t(x)、频谱 T(f)，离轴平面波横向频率 \(\mathbf f_0\)，物镜瞳函数 P(f)，离焦量 d；可写

\[
U_d(\mathbf f)=P(\mathbf f)T(\mathbf f-\mathbf f_0)
\exp[-i\pi\lambda d\|\mathbf f\|^2],\qquad
I_d(\mathbf x)=\left|\mathcal F^{-1}U_d\right|^2.
\]

由乘积—相关关系，强度频谱为

\[
\widetilde I_d(\mathbf q)=\int U_d(\mathbf f+\mathbf q)U_d^*(\mathbf f)\,d\mathbf f.
\]

积分中的传播相位差是 \(-\pi\lambda d[2\mathbf f\cdot\mathbf q+\|\mathbf q\|^2]\)。因此即使 \(|U_d(\mathbf f)|\) 本身不随 d 改变，光场之间的干涉仍可改变 \(|\widetilde I_d|\)。瞳函数与照明偏移控制有效支撑；弱散射条件下，未散射参考与单次散射项之间的交叉项尤其有用。这解释频域选择的物理动机，但不是所有样本上 Δz 可唯一恢复的证明。[cite:3](#source-3)

**2. 幅度平移不变性及其边界。** 若 \(I'(x)=I(x-a)\)，Fourier 移位定理给 \(\widetilde I'(f)=e^{-i2\pi f\cdot a}\widetilde I(f)\)，故幅度相同；这减少特征对平移的敏感性。有限裁窗会改变内容、亚像素插值与边界也有影响，因此代码对任意平移并不严格不变。若正负离焦仅造成平移或被选区域幅度相同，则本特征仍无法辨别方向，不能以“离轴”二字担保唯一性。

**3. 代理标签。** 对参考通道 \(J_i\)，令

\[
L_i(r)=\operatorname{mean}_{\|q\|\approx r}\log|\mathcal F J_i(q)|^2,
\qquad S_i=\sum_{r\in\mathcal H}L_i(r),\quad
z_*=\arg\max_z S(z),\quad y_i=z_*-z_i.
\]

作者源码 \(\mathcal H\) 取径向谱数组从长度四分之一开始的 bins，对 S 做三次插值，在10000个等距索引位置上取峰，再乘 z 步长。**径向平均 log 功率**与“先平均功率再取 log”不是同一个量。插值精细网格只提高搜索分辨率，不提供真实焦点精度保证；非等距 z 轴须改用实际坐标，而不能继续乘一个统一步长。[cite:9](#source-9)[cite:10](#source-10)

**4. FCFNN 与损失。** 固定频域选择 C，单块幅度向量 \(v=\operatorname{vec}C(|FFT(I-\bar I)|)\)，先 \(q=v/\|v\|_2\)，再用训练集统计 \(x=(q-\mu)/s\)。MLP 的 ReLU 隐藏层输出标量 \(\hat y=f_\theta(x)\)，作者默认优化

\[
\mathcal L(\theta)=\frac1B\sum_i(f_\theta(x_i)-y_i)^2.
\]

代码支持 L2 正则但默认强度为0。若同一 x 的标签存在多义性，固定 x 的平方风险求导给 \(f^*(x)=\mathbb E[y\mid x]\)，不是任一实例必然正确的运动量。光路必须让有用信息可观测，数据还必须覆盖其统计关系；增加网络深度不能创造缺失信息。非凸训练无通用全局最优或“1–2帧收敛”保证。[cite:8](#source-8)

**5. 复杂度。** 对边长 n 的块，FFT 为 \(O(n^2\log n)\)，选取 d 维特征后，十个100单元隐藏层的主要矩阵乘为 \(O(100d+9\cdot100^2)\)。全连接参数数为 \(100d+91\,101\)，含偏置和最后的100→1输出。裁剪维数决定首层成本；不应仅比较层数。预计算训练特征约需 \(4N_{examples}d\) 字节的 float32 数据，不含标签、数据加载和优化器状态。[cite:8](#source-8)

### L5 实现：固定作者版本、配置与复现差异

本节外部源码固定到 `Waller-Lab/DeepAutofocus` commit **`09bb02aa082238991aa187ffaf0104c93ebc386c`**，不是把当前 master 当成永远不变的论文版本。作者 Figshare 页面将旧 `henrypinkard/DeepAF` 链接到此仓库；代码文件与 notebook 已静态读取，没有下载执行、安装 TensorFlow 或取得训练数据／权重。[cite:6](#source-6)[cite:7](#source-7)

| 子步骤 | 固定版本实际行为 | 复现时必须明确 |
|---|---|---|
| 分块与融合 | `get_patch_metadata`取不超过最短边／split_k的2幂边长；典型 `tile_split_k=2`；同帧预测中位数 | 2048²帧对应4个1024²块；块标签与帧标签不能被打乱；非标准尺寸不能只猜块数 |
| 去均值 | `_build_deterministic_graph`用无 axis 的 `reduce_mean`，即该输入张量整体均值 | 本文单例解释与多例实现并非逐位相同；逐帧去均值是显式改写，需比较尤其 DC 特征 |
| FFT特征 | 未 fftshift 的 `fft2d`；拼接 `[:a,:b]` 与 `[-a:,:b]` 的幅度并展平 | \(a=\lfloor\operatorname{int}(led\_width\,n)/2\rfloor\)，b同理；是沿 LED 轴的矩形选择，不是直接生成两个理想圆形瞳掩膜 |
| 归一化 | 每例 L2 归一化，再按训练特征均值／标准差标准化 | 预测复用训练统计；代码未为所有零范数／零标准差情况加保护 |
| 网络 | 10个 Dense(100, ReLU) 隐藏层，最后线性标量；输入dropout丢弃率0.6，隐藏dropout默认0 | 不把0.6误作保留率；不是卷积网络，也不是光学 Fourier 神经算子 |
| 优化 | Adam学习率 \(2\times10^{-5}\)，batch=25，MSE，正则默认0 | 是该代码默认，不声称每幅论文图均用全部默认值；notebook有参数覆盖 |
| 验证与停止 | 默认每25步遍历验证集算RMSE；距离最佳步超过2000步且无改善则停，另有total_steps | 部分notebook用5000步 patience；best checkpoint虽保存，最终恢复语句被注释，导出不自动等于最佳验证权重 |
| 标签 | `focal_plane - z_step*z_index`，预编译特征时忽略首层 | 忽略首层有作者设备到位问题背景，不应照抄为通用采集规律；符号与本地F14相反 |
| 参考峰实现 | `calc_focal_plane`为所有块共享同一峰；固定源码实际以 `crop(image,0,1)`取图，使用已算出的patch_size | 不冒称每块都有独立真值，也不保证用到整帧；应逐项冻结与说明 |
| 导出入口 | `_build_graph('predict')`返回确定性前端之后的特征张量作为输入；代码注释另提Java侧适配 | 不可把任意原图直接喂给导出签名，就宣称实现了论文全管线 |

来源为作者函数体，不依靠文件名或注释的“全部可复现”宣言。`feature_vector_generator_fn`还有 `trianing` 拼写分支，不能依赖它完成可靠位置划分；本卡采用 notebook 的显式 XY 分组后 `mode='all'` 路线。迁移到现代依赖时还需处理 TF1 Session／layers 和 NumPy旧类型别名。这里只记录复现障碍，不修改外部或本地源码。[cite:7](#source-7)[cite:8](#source-8)[cite:9](#source-9)

以下是**带明确质量门的未执行复现伪代码**；输入零特征剔除、恢复最佳权重、安全控制等是建议补强，与固定源码差异须在实际复现报告中保留。

```text
train(paired_stacks, optical_config):
    train_groups, validation_groups = split_by_specimen_and_xy(paired_stacks)
    for group in train_groups + validation_groups:
        require paired_channels_and_valid_actual_z(group)
        score = radial_mean_log_power_high_band(group.reference_stack)
        z_star = validated_peak(score, group.actual_z)
        for z, coherent_frame in group.coherent_stack:
            for patch in split_fixed_tiles(coherent_frame):
                F = fft2(patch - per_frame_mean(patch))
                v = concat(abs(F[:a,:b]).ravel(), abs(F[-a:,:b]).ravel())
                if norm(v) == 0: reject_example()
                else: store_feature_in_memory(v/norm(v), z_star-z, group.id)
    mu, sd, feature_policy = fit_training_only_statistics()
    model = dense_relu_width_100_depth_10_linear_scalar()
    repeat:
        x, y = next_training_batch(25)
        require shape(y) == shape(model_output) == (batch,)
        update_adam_mse(model, standardize(x, mu, sd), y, learning_rate=2e-5)
        every_25_steps_evaluate_validation_rmse()
        preserve_best_checkpoint()
    until validation_patience_exceeded_or_step_limit()
    restore_best_checkpoint()
    export(model, mu, sd, feature_policy, optical_config, delta_z_sign)

predict_and_control(frame, bundle, stage):
    require acquisition_matches(bundle.optical_config)
    features = bundle.identical_feature_pipeline(frame)
    predictions = bundle.model_inference_without_dropout(features)
    delta_z = median(predictions)
    if not safe_and_in_domain(delta_z, stage): return REJECT
    stage.move_to(stage.position + delta_z)
    await_stage_settling()
    return independent_focus_verification_or_controlled_fallback()
```

**数值与接口核验。** 从作者示例 \(n=1024,led\_width=0.6,non\_led\_width=0.1\) 按实际截断规则算出 a=307、b=51、d=2ab=31,314，而非直接使用非整数比例面积；上述网络有3,222,501个可训练参数。该数值是源码配置算术核验，不是所有论文实验的参数总数，也不能替代未取得的 Table S1。若改变裁剪宽度为1.0／1.0，首层规模会完全不同。由作者符号恒等式 \(z+(z_*-z)=z_*\) 可核对控制方向；本地 d 应采用减法。

**尚未执行的测试和验收。** 固定原图比较作者前端与重写前端的频率索引、幅度、归一化和特征顺序；验证图像／特征两类输入签名；检查零图、常量特征、同帧块次序、B与B×1标签广播、去均值的batch差异、dropout训练／预测切换、权重恢复一致性。数据侧检查成对通道与z坐标、参考峰重测、按样本划分和域外检测；设备侧验证限位、到位、回差、超时、重拍与方向。仅优化 FFT 缓存、批量推理或预计算，不在无对照时改变频谱区域／标签含义；没有训练和硬件验收证据就不报告“单帧闭环成功”。

### 五问五答：贯穿 L1–L5

| 层级与问题 | 答案要点、成立条件与易错项 | 证据入口 |
|---|---|---|
| L1：single-shot为何还采两套焦栈？ | 训练时低相干通道产生参考焦点，离轴通道每层作输入；训练后预测才是一帧。不能把训练扫描成本删掉。 | 作者正文Fig.1；教程；`calc_focal_plane` |
| L2：为什么FFT幅度还能含离焦，传播不是只变相位吗？ | 传播只变相位指光场频谱；相机测强度，强度频谱是光场频谱的相关，干涉可改变其幅度。平坦／对称对象仍可退化。 | 本卡L4模型；作者幅度／相位对照 |
| L3：细胞上准、组织上错，先改什么？ | 查样本域、参考标签和照明／采样；作者明确报告此失败，并补组织训练数据。不能先增加网络深度或许诺天然跨域。 | 作者正文Fig.3 |
| L4：MSE低就能确定运动方向吗？ | 最优回归是条件均值；同特征正负标签混合时可接近零。需可观测信息、正确标签和同域分布，无通用帧数保证。 | 本卡条件风险推导；`_build_graph` |
| L5：现有FocusNet可作为FCFNN复现吗？ | 不可：光路／成对标签／FFT幅度裁剪／全连接结构都不同；作者Δz符号与本地d相反，导出签名还可能要求特征而非原图。 | 本地`autofocus_cnn.py`；作者`generator_fn`、`_build_graph` |

## 来源层级与逐源获取记录

P0＝原论文、作者资料／源码、出版登记；P1＝独立权威综述；P2／P3＝社区／一般参考。另以“全文／全文片段／摘要／书目／源码／受限”记录实际读取深度，二者不可混为一维。本文核心机制仍为各自作者的 **[SINGLE-SOURCE]**，没有三个独立复现实验交叉验证；镜像、登记、教程和作者代码只用于版本／实现对照，不算独立性能证据。书目多入口一致可标 [CONFIRMED]；未取得的信息标 [INSUFFICIENT DATA]，不由检索摘要补造。

<a id="source-1"></a>

**[cite:1] Renshaw–Christian 原文，P0。** [MDPI HTML](https://www.mdpi.com/2313-433X/6/5/33/html)实际返回第1–4.1节开头，属**全文片段**；后续直接请求被403拒绝。[Europe PMC 全文XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8321028/fullTextXML)成功取得完整XML，**按节核读**4.1–4.3、5.1.1–5.2.2及关键MathML，确认Eq.34–58、参数和近似属性，并非声称每段附录／参考文献逐项审计。[Semantic Scholar PDF镜像](https://pdfs.semanticscholar.org/8ca0/17b29e3da3968176af409dfe46d715cf0852.pdf)只返回前部片段；[PMC网页](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321028/)为cookies提示，记受限。公式以XML为核验主源，不以损失括号的纯文本为准。[SINGLE-SOURCE]

<a id="source-2"></a>

**[cite:2] PubMed，P0书目与图注。** [PMID34460735](https://pubmed.ncbi.nlm.nih.gov/34460735/)已读摘要、书目与图注，交叉确认作者、2020-05-18、6(5):33、DOI，以及图5／8的5000次条件实验。它不是第二份独立算法实验。书目 [CONFIRMED]，性能仍为作者报告。

<a id="source-3"></a>

**[cite:3] Pinkard等作者存档稿，P0全文片段。** [UCB/EECS-2019-161 PDF](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/Archive/EECS-2019-161.pdf)已读封面、内稿日期、成像配置、Fig.1训练、Fig.2数据量和Fig.3泛化段，返回在后文截断；光路和论文实验数值只引用可见部分。另取得 [UCB/EECS-2019-138 PDF](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/Archive/EECS-2019-138.pdf)前部相同主题内容，封面／内稿日期为2019-10-16，不作独立实验支持。两份存档都不能证明正式Note S1–S4／Table S1已核读。[SINGLE-SOURCE]

<a id="source-4"></a>

**[cite:4] 正式出版登记，P0书目。** [Crossref JSON](https://api.crossref.org/works/10.1364/OPTICA.6.000794)已读，作者、题名、卷期与在线2019-06-05／印刷2019-06-20可核，页字段仅794；794–797由作者教程引文补充。[Optica正式正文](https://opg.optica.org/optica/fulltext.cfm?uri=optica-6-6-794&id=412060)返回验证码，**受限**，未绕过；补充入口请求未找到内容。正式补充材料的完整网络、光路推导和逐实验配置 [INSUFFICIENT DATA]；作者源码能补实现细节，但不把它改写为正式补充全文。

<a id="source-5"></a>

**[cite:5] 作者采集教程，P0后续教程全文。** [Pycro-Manager Single-shot autofocus](https://pycro-manager.readthedocs.io/en/latest/application_notebooks/Single_shot_autofocus_pycromanager.html)已读：离轴LED、电动XY/Z、配对通道、Micro-Magellan采集、Pycro-Manager控制hook、约10颗同侧直线LED方案；训练指向另一notebook。教程用后续软件接口，不冒充2019实验原始脚本或严密安全控制。[SINGLE-SOURCE]

<a id="source-6"></a>

**[cite:6] 作者软件存档，P0页面。** [Figshare v1](https://figshare.com/articles/Single-shot_autofocus_microscopy_using_deep_learning--code/7453436/1)，DOI `10.6084/m9.figshare.7453436`，页面标posted 2019-01-23，已读说明与GitHub关联；下载包未取得／核验。页面标MIT，GitHub页面显示BSD-3-Clause，不能将两个分发物许可证互换；正式复用应按实际下载版本核对。API尝试为空／403，不声称API内容已读。

<a id="source-7"></a>

**[cite:7] 作者仓库和notebook，P0源码静态读取。** [固定commit](https://github.com/Waller-Lab/DeepAutofocus/tree/09bb02aa082238991aa187ffaf0104c93ebc386c)、[deep_af_demo.ipynb](https://github.com/Waller-Lab/DeepAutofocus/blob/09bb02aa082238991aa187ffaf0104c93ebc386c/deep_af_demo.ipynb)。通过作者旧仓库链接重定向确认归属，API确认commit与文件树；已解析notebook代码单元核对XY划分、裁剪配置、图案实验与patience覆盖，没有执行单元。该commit首页更新到2021，未证明它与Figshare v1或2019每幅图的运行快照逐位相同。

<a id="source-8"></a>

**[cite:8] FCFNN网络，P0源码。** [固定版本defocusnetwork.py](https://github.com/Waller-Lab/DeepAutofocus/blob/09bb02aa082238991aa187ffaf0104c93ebc386c/util/defocusnetwork.py)已读构造器、归一化、确定性图、训练图、验证／预测、融合和导出函数体。支持本卡结构、超参数、MSE／Adam、零值风险及导出／最佳权重边界；未运行TensorFlow，源码存在不等于训练成功。[SINGLE-SOURCE]

<a id="source-9"></a>

**[cite:9] 标签与数据编译，P0源码。** [固定版本afutil.py](https://github.com/Waller-Lab/DeepAutofocus/blob/09bb02aa082238991aa187ffaf0104c93ebc386c/util/afutil.py)已读 `get_patch_metadata`、`calc_focal_plane`、`generator_fn`、`feature_vector_generator_fn`、`read_or_calc_design_mat`、`compile_deterministic_data`。据此核对代理真值、标签正负、首层剔除、缓存与分组约束；未取得作者数据文件。[SINGLE-SOURCE]

<a id="source-10"></a>

**[cite:10] 径向统计组件，P0源码。** [固定版本imageprocessing.py](https://github.com/Waller-Lab/DeepAutofocus/blob/09bb02aa082238991aa187ffaf0104c93ebc386c/util/imageprocessing.py)已读，尤其 `radialaverage` 的中心、半径取整分bin与径向均值。不能把该文件另一个“先功率径向平均”的函数误认成 `calc_focal_plane` 的实际log功率流程。[SINGLE-SOURCE]

## 本次核验与保留缺口

- 结构核验范围：两张独立卡、各五层实质正文、各五问五答、来源与实现字段、引用闭合、本地链接和显式锚点；只新增本文件，不更新账表或其他代码。
- 已执行数学核验：MF01两种分段积分对前向闭式矩、直接根式对有理化表达式；MF02裁剪维数、网络参数计数和标签方向恒等关系。采用仓库现有Python虚拟环境及NumPy／SciPy，不新增脚本或依赖。
- 未执行：矩核面积生成、图像级定位、原文Monte Carlo重建、外部TensorFlow源码运行、权重训练、真实数据误差比较、光路搭建和运动闭环。这里没有把条件算术成功扩展为算法性能或现场验收。
- MF01保留：PSF宽度标定、近零矩与居中条纹策略、离散采样与极性偏差、真实位置真值。没有A31联合宽度估计或通用精度保证。
- MF02保留：正式补充材料、发表时配置快照、原始数据／权重、现代依赖适配、频域索引与光学标定、代理标签可信度、跨域及安全闭环。仓库FocusNet仍不是FCFNN，两者都不因本文完成而获得训练／部署能力。
