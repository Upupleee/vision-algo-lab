# Haralick 灰度共现纹理与分类算法卡

[学习规范](../algorithm_learning_standard.md) · [06 缺陷异常总览](../defect_evolution.md) · [07 分类装配总览](../classification_assembly_evolution.md) · [覆盖台账](../evolution_coverage.md)

## 范围、状态与证据口径

核验日期：**2026-11-05**。本文件只有一张卡，唯一键为 `haralick.md:HA01`，承接历史任务 **G16**，同时对应 **06-01 与 07-01**；14 个特征、分类器分支和测试项不另计卡。原始版本指 Haralick、Shanmugam、Dinstein 的 IEEE TSMC **SMC-3(6)，1973-11，610–621**，不是现代库的默认配置，也不是所有 GLCM 变体的统称。[cite:1](#source-1)

| 项目 | 当前覆盖与限制 |
|---|---|
| 演进定位 | 一阶灰度直方图丢空间关系 → 给定距离／方向的灰度对联合统计 → 特征选择与有标签分类；增加量化、采样尺度、稀疏统计及谱计算成本 |
| L1 步骤 | **部分**：GLCM 共现—特征—方向汇总链已核验；原分类链的分段线性训练及 min-max 完整决策规则仍部分，教学替代不补齐原法 |
| L2 机制 | **已核验**：联合概率、方向处理与表示／决策分离；仅文献及条件代数意义 |
| L3 工程权衡 | **部分**：参数单位、失效证据与验收设计齐备；无真实材质／缺陷调参 |
| L4 数学推导 | **部分**：14 特征、sum variance 分歧、Q 与 SVD 等价、退化条件展开；正式勘误及原分类训练细节仍缺证 |
| L5 实现 | **部分**：明确伪代码、外部固定版本源码差异、小矩阵检查；**本仓库专用算法未实现** |
| 面试自测 | 五问五答，贯穿 L1–L5；不是企业真题 |
| 验证范围 | 终端内存执行本卡确定性小矩阵代数检查；不是真实图像分类、检出率、定位精度、节拍或现场验收 |

“已读原文”限于实际获取的正文和可辨识符号。作者站网页解析只返回前部，另从 Stanford 课程镜像的 12 页 PDF 内存提取，重点读 pp.616–619 分类／附录及相关量化段落；扫描层存在字符和公式丢失，不能声称逐字校勘全文。同文镜像不是独立证据。2019 论文与两个库提供定义／差异交叉核对，不是三项独立性能复现。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

<a id="ha01"></a>
## HA01 Haralick 1973：灰度共现特征与纹理分类

### 任务契约与两域交接

**输入：**灰度 ROI、有效像素掩膜、冻结的灰度量化规则、非零整数偏移集合、对称化／方向聚合方式、特征版本配置；分类训练另需真实类别标签及分组划分。多光谱逐通道或选波段必须明确，不能将 RGB 三通道直接当成三个相邻空间维。**输出：**各偏移的有效对数、GLCM、14 维特征及有效位、聚合向量；接入分类模型后才有类别／拒识，接入正常模型后才有异常分数。

| 节点 | 本卡承接 | 验收不能跨域替代 |
|---|---|---|
| 07-01 纹理分类 | 原文对砂岩显微图、航空图、ERTS 多光谱图作有标签类别识别；使用分段线性判别和 min-max 判别两支 | 材质／类别正确不证明数量、位置、插接、正反装合格；另接装配关系 |
| 06-01 缺陷异常 | 共享同一 GLCM 表示；用局部窗口特征与正常参考比较是**工程迁移** | 原文不是仅正常训练的工业异常检测；必须另定义正常分布、缺陷规格、定位／整件聚合及阈值 |

原文的任务与分类器来自原文 `[SINGLE-SOURCE]`；不引用原文准确率认证工业现场。GLCM 不是 LBP：前者统计指定偏移的灰度对，后者编码中心与邻居的大小比较；已有 D03 不能替代本卡。前序瓶颈、改变、机制、代价、遗留问题形成的是表示演进，不是“手工纹理全面替代一阶统计”或“深度网络全面替代 GLCM”。[cite:1](#source-1)

### L1 步骤：从量化到分类

1. **确认任务与分组。**先按工件／批次分训练、调参、验收集，再确定 ROI 和灰度通道。不能把同一纹理图相邻裁片分入训练与测试。
2. **量化。**本卡工程基线使用训练／标定阶段冻结的范围 \([a,b]\)，\(b>a\)，级数 \(G\ge2\)：\(q(x)=\min(G-1,\max(0,\lfloor G(x-a)/(b-a)\rfloor))\)。灰度标签为 \(0,\ldots,G-1\)，端点 \(b\) 进入最后一箱；非有限像素从有效域排除并报告，饱和比例另记。原文使用 \(1,\ldots,N_g\)，并在附录 III 讨论等概率量化；本卡固定宽度不是其所有实验的原设置。[cite:1](#source-1)
3. **定义偏移。**坐标为 `(row, col)`，行向下、列向右。四个无向方向可用 \((0,d),(-d,d),(-d,0),(-d,-d)\)，分别标为 0°、45°、90°、135°，其中 \(d\) 为正整数像素步长。轴向欧氏长度为 \(d\)，对角为 \(\sqrt2d\)，不能都叫“物理距离 d”。不插值、不 wrap、不补零造邻居。
4. **计数并对称化。**只统计两端都在 ROI／mask 中的像素对；先构造有向 \(C_\delta\)，历史对称基线取 \(C=C_\delta+C_\delta^T\)。对角元素也加倍；等价于正反两个方向都计数，不是只把上三角复制过去。
5. **逐偏移归一化。**\(T=\sum C\)，若 \(T=0\) 返回 `NO_PAIRS`；否则 \(P=C/T\)。记录独立扫描到的有向对数 \(N\) 和对称质量 \(T=2N\)，后者不是两倍独立样本。
6. **计算特征。**由联合分布、边缘分布、和／绝对差分布算 f1–f14；f3、f12、f14 的退化用无效位而非无条件填零。固定 sum variance、difference variance、熵底和 MCC 版本。
7. **方向／距离聚合。**每个距离可将四方向的每特征均值和极差连接为 28 维，原文附录 I 明确给出这一方案；多个距离再按固定顺序连接。保留方向的原向量便于排障。某方向缺失时本卡严格模式拒绝该 ROI，而非静默改用三个方向。[cite:1](#source-1)
8. **学习决策并冻结。**特征选择、缩放、模型拟合只用训练集，阈值只用调参集，随后一次性评价保留集。07-01 输出类别及未知状态；06-01 另拟合正常参考／单类模型并输出局部评分，不把向量本身称为缺陷概率。

**停止条件：**特征提取是有限遍历，无迭代收敛问题；遍历全部指定偏移后输出或带原因拒绝。分类训练的迭代上限、求解容差和失败回退取决于所选分类器，原论文未在本卡中被扩写成一个已核实的统一训练程序。

### L2 机制：步骤为何不可随意删除

GLCM 估计的是“从当前有效像素对集合随机抽取一对时，其量化灰度为 i、j 的概率”，而不是整幅图所有像素的独立同分布联合概率。相邻像素共用端点；重叠窗口更不独立。归一化去掉对数的直接倍率，却不消除小 ROI 的统计不确定性。灰度量化决定可区分的强度差，偏移决定观察的空间尺度，窗口决定被平均的材料范围。[cite:1](#source-1)[cite:2](#source-2)

| 去掉／替换 | 机制与具体后果 |
|---|---|
| 用灰度直方图代替共现 | 一阶计数相同的条纹与棋盘排列可有不同邻接；直方图不能表达该差别，共现也不能唯一恢复图像 |
| 每 ROI 独立拉伸 | 提高部分光变容忍，但统一污染造成的幅值偏移可能被消掉；同一 bin 不再代表同一标定强度 |
| 不对称化 | 可保留源到目标顺序，但行／列边缘一般不同；不能再无条件用 \(\mu_x=\mu_y\) 简式 |
| 先合计矩阵再求特征 | 合并了方向分布；非线性特征通常不同于逐方向特征均值，且按计数合并隐含对数权重 |
| 忽略灰度 0 | 0 可以是有效黑纹；只能由 mask 定义背景。库的 `ignore_zeros` 不是通用 ROI 掩膜替代品 |
| 只输出最高分类分数 | 闭集分类总会倾向某个已知类；没有域外校准便没有未知类保证，更没有装配合格保证 |

令 \(F(P)=\sum P_{ij}^2\) 为 ASM，则

\[
\frac{F(P_1)+F(P_2)}2-F\!\left(\frac{P_1+P_2}2\right)
=\frac14\lVert P_1-P_2\rVert_F^2\ge0.
\]

只有两矩阵相同才在这个二矩阵等权情形取等号。因此“方向池化”和“方向特征平均”不是等价优化。方向均值／极差对**方向集合的置换**不变，不保证任意角度旋转不变：旋转重采样、裁剪和不等长的轴／对角偏移均会改变有效对。

等概率量化的保序解释需限定：理想严格递增映射保持排序，用相应 CDF 分箱可获得同箱标签；离散 ties、饱和和有限样本使精确等概率未必可达。原文附录写 strictly monotonic，但所读证明使用保持不等号方向的步骤；对递减变换应另分析灰度标签反转，不能直接宣称标签完全相同。更不能由该论证推导固定宽度量化对任意光照不变。[cite:1](#source-1)[cite:2](#source-2)

### L3 工程权衡：尺度、稀疏与诊断

| 参数／约束 | 单位、选择依据与对照方案 | 失效证据及回退 |
|---|---|---|
| G、a、b／bin edges | G 无量纲，边界为相机灰度单位；可把 8／16／32 级作为小型工程搜索集，**不是论文最优值**；结合噪声幅度与最小纹理对比 | G 增大后有效非零格数、稀有格质量和跨批波动陡增：先核曝光与箱边，再降低 G／扩大统计支持；需重训模型 |
| 偏移 d、方向 | 像素或经标定的物理距离；先 d=1，再按纹理周期加入少量尺度；这是搜索起点而非通用推荐 | 纹理旋转后某方向突变：看各方向向量；业务需要朝向时保留方向，而非一律池化 |
| 窗口与步幅 | 像素／毫米；窗口要容纳若干纹理结构，同时不能吞没最小缺陷 | 大窗口漏细划痕、小窗口极不稳定：检查缺陷占比与有效对数，多尺度对照；不能只调阈值 |
| mask 与边界 | 两端均有效；报告 \(N_\delta\)、空方向、饱和／不可见比例 | ROI 边缘假纹理：检查是否补零、越界或只检查一端；无支持返回 UNKNOWN |
| 特征缩放、选择 | f2／f4／f7／f10 为 bin²，f6 为 bin，熵由 log 底决定；统计缩放只在训练拟合 | 某些特征量纲支配距离、强相关导致不稳定：先固定版本再做训练内选择，不偷看验收标签 |
| f14 开关 | 活跃支持数 r、c 决定谱成本；报告最小非零边缘概率和奇异值间距 | 稀有灰度造成数值放大／慢：先减少无效分辨率，必要时显式采用13特征协议并重训，不保留“完整14维”名义 |
| 聚合／阈值 | 固定 mean+range、拼接或计数池化；06-01 局部到整件最大值／分位数另定 | 窗口数增加使整件最大值变大；必须重新校准整件操作点，不把原窗口阈值无条件复用 |

**排障顺序：**成像与有效域 → 量化和坐标 → 有效对／概率不变量 → 逐方向特征版本 → 训练划分及分类器 → 阈值。稀疏不是“零格必须加 ε”：零概率合法，向所有格加伪计数会改变熵、相关与 MCC；若选择 Dirichlet 等平滑，须登记先验、总伪质量和模型版本。活跃支持很小并不必然无效，例如二值纹理完全可计算 MCC；单灰度则另属退化。

07-01 验收至少给逐类混淆、召回、已知误拒／未知误接收与批次分层。06-01 至少给真实缺陷尺寸分层召回、正常误报、窗口／区域定位、整件漏放和 UNKNOWN；纯正常数据只能校准其分布内误报，不能测真实缺陷召回。所有这些是**未执行的验收设计**，无通用检测率、分类误差或跨域泛化界。

### L4 数学：14 特征、版本与推导

#### 联合分布及辅助量

令 \(\Omega_\delta=\{u:u,u+\delta\text{ 均在有效域}\}\)，则

\[
C_\delta(i,j)=\sum_{u\in\Omega_\delta}\mathbf1[q(u)=i,q(u+\delta)=j],
\quad P=\frac{C_\delta+C_\delta^T}{2|\Omega_\delta|}.
\]

本卡用零基标签，定义 \(p_x(i)=\sum_jP_{ij}\)、\(p_y(j)=\sum_iP_{ij}\)，\(\mu_x=\sum_i i p_x(i)\)、\(\sigma_x^2=\sum_i(i-\mu_x)^2p_x(i)\)，y 同理。灰度标签保留原 bin 数值；压缩空行列只改变存储索引，不得将原标签 0、7 偷换成 0、1 来计算对比度。

\[
p_+(s)=\sum_{i+j=s}P_{ij},\quad s=0,\ldots,2G-2;
\qquad p_-(d)=\sum_{|i-j|=d}P_{ij},\quad d=0,\ldots,G-1.
\]

取 \(H(v)=-\sum_{v_k>0}v_k\ln v_k\)，约定 \(0\ln0=0\)。本卡默认**自然对数 nats**，原文写 log 未在已读附录明确底数；不能声称这就是原实验底数。设 \(H_X=H(p_x),H_Y=H(p_y),H_{XY}=H(P)\)，以及

\[
H_{XY1}=-\sum_{P_{ij}>0}P_{ij}\ln[p_x(i)p_y(j)],\qquad
H_{XY2}=-\sum_{p_x(i)p_y(j)>0}p_x(i)p_y(j)\ln[p_x(i)p_y(j)].
\]

若 \(P_{ij}>0\)，则两边缘必为正，故前式不应发生 `log(0)`；出现则是实现错误或非一致平滑。由边缘求和得 \(H_{XY1}=H_{XY2}=H_X+H_Y\)，互信息 \(I=H_X+H_Y-H_{XY}\ge0\)。这些恒等式要求概率来自**同一归一化矩阵**，不能混用原概率与加 ε 后的概率。

#### f1–f14 定义表

编号和名称依原文附录 I；清晰可读的原式与现代修正区分列出。f7 采用均值中心修正，f10 选择灰度绝对差的方差，f14 采用 Q／SVD 定义；这些选择构成 `HA01-corrected-nats` 教学配置，不是一个声称逐数复现 1973 的软件版本。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

| 编号 | 名称与本卡公式 | 边界／版本 |
|---|---|---|
| f1 | Angular Second Moment：\(\sum_{ij}P_{ij}^2\) | ASM，部分库的 energy 是 \(\sqrt{f1}\)，勿混名 |
| f2 | Contrast：\(\sum_{ij}(i-j)^2P_{ij}=\sum_d d^2p_-(d)\) | 依赖灰度标签间距 |
| f3 | Correlation：\(\frac{\sum_{ij}(i-\mu_x)(j-\mu_y)P_{ij}}{\sigma_x\sigma_y}\) | 任一方差为零未定义；有效时 \([-1,1]\) |
| f4 | Sum of Squares: Variance：\(\sum_{ij}(i-\mu_x)^2P_{ij}=\sigma_x^2\) | 原文写 μ，本卡明确为行边缘均值；不是矩阵所有元素的方差 |
| f5 | Inverse Difference Moment：\(\sum_{ij}\frac{P_{ij}}{1+(i-j)^2}\) | 不是 \(1+\lvert i-j\rvert\) 分母版本 |
| f6 | Sum Average：\(\mu_+=\sum_s s p_+(s)=\mu_x+\mu_y\) | 原文一基转零基后减2 |
| f7 | Sum Variance：\(\sum_s(s-\mu_+)^2p_+(s)\) | 原文可辨式以 f8 为中心；修正与旧版另述 |
| f8 | Sum Entropy：\(H(p_+)\) | 和分布熵，不是 f6 的 log |
| f9 | Entropy：\(H_{XY}\) | 联合熵，不是单幅灰度直方图熵 |
| f10 | Difference Variance：\(\sum_d(d-\mu_-)^2p_-(d),\ \mu_-=\sum_d d p_-(d)\) | 原文只写 variance of \(p_{x-y}\)，现代实现有歧义 |
| f11 | Difference Entropy：\(H(p_-)\) | 使用绝对差分布 |
| f12 | Information Measure of Correlation 1：\((H_{XY}-H_{XY1})/\max(H_X,H_Y)\) | 非退化时等于 \(-I/\max(H_X,H_Y)\)，范围 \([-1,0]\) |
| f13 | Information Measure of Correlation 2：\(\sqrt{1-\exp[-2(H_{XY2}-H_{XY})]}\) | 本卡用 nats；稳定形式 \(\sqrt{-\operatorname{expm1}(-2I)}\) |
| f14 | Maximal Correlation Coefficient：\(\sqrt{\lambda_2(Q)}\)，\(Q_{ij}=\sum_k\frac{P_{ik}P_{jk}}{p_x(i)p_y(k)}\) | 限正边缘支持，特征值按从大到小含重数；不是 Pearson f3，也不是 `corrcoef(P)` |

#### sum variance 与其他版本分歧

原附录 p.619 可辨 f7 为 \(\sum_s(s-f8)^2p_+(s)\)，其中 f8 是 sum entropy；2019 论文表2文字明确指出原定义有错误并已修正；mahotas v1.4.18 以 `preserve_haralick_bug` 控制保留该式或用均值中心，PyRadiomics v3.0.1 写均值中心并因其等于 cluster tendency 而弃用重复特征名。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

设 \(S=X+Y\)，展开任意中心 \(c\)：

\[
E[(S-c)^2]=E[(S-\mu_+)^2]+2(\mu_+-c)E[S-\mu_+]+(\mu_+-c)^2
=\operatorname{Var}(S)+(\mu_+-c)^2.
\]

所以旧式不等于方差，且会受 log 底及灰度标签原点影响；均值中心版对共同标签平移不变，并满足

\[
f7=\sigma_x^2+\sigma_y^2+2\operatorname{Cov}(X,Y).
\]

**未核实：**没有取得 IEEE／原作者发布的正式勘误单、勘误年份或可核对的“1973 修订实验代码”。只能称“原印式与后续数学修正”，不能捏造某年已官方纠正；也不能反推当年分类数值到底用了哪一式。该历史缺口保留。

f10 的另一版本是概率数组本身的总体方差：\(G^{-1}\sum_{d=0}^{G-1}[p_-(d)-1/G]^2\)，与 \(\operatorname{Var}(|X-Y|)\) 完全不同，且受补入零概率格数影响。mahotas v1.4.18 默认前者，`use_x_minus_y_variance=True` 才选后者；原文一句文字不足以消除原实验实现的歧义，故不宣称本卡 f10 已考证为唯一原始实现。[cite:1](#source-1)[cite:3](#source-3)

熵从 nats 换为 bits 会除以 \(\ln2\)。f12 的分子分母一起换底，数值不变；f13 若仍保留自然指数，则**数值改变**。若希望保持 nats 结果而内部用 bits，应写 \(\sqrt{1-\exp(-2\ln2\,I_{bits})}\)。mahotas 与所读 PyRadiomics 版本用 log2 再直接自然指数，这是一项应登记的兼容性配置，不应偷加 \(\ln2\) 后宣称数值与库一致。[cite:3](#source-3)[cite:4](#source-4)

#### MCC 的谱推导与边界

在正行／列边缘支持上令 \(D_x=\operatorname{diag}(p_x)\)、\(D_y=\operatorname{diag}(p_y)\)，并构造

\[
B=D_x^{-1/2}PD_y^{-1/2},\qquad
Q=D_x^{-1}PD_y^{-1}P^T,
\qquad D_x^{1/2}QD_x^{-1/2}=BB^T.
\]

因此 Q 虽通常不对称，却相似于实对称半正定矩阵；\(\lambda_k(Q)=\sigma_k(B)^2\)，从而 **f14 可直接取 B 的第二大奇异值**，不是奇异值再开平方。\(B\sqrt{p_y}=\sqrt{p_x}\)、\(B^T\sqrt{p_x}=\sqrt{p_y}\)，对应第一奇异值1。由加权 Cauchy–Schwarz，条件期望映射是二范数压缩，其余奇异值在 \([0,1]\) 内。

进一步，最大化 \(E[a(X)b(Y)]\)，约束 \(E[a]=E[b]=0\)、\(E[a^2]=E[b^2]=1\)；令 \(u=D_x^{1/2}a,v=D_y^{1/2}b\)，目标变为 \(u^TBv\)，且 \(u\perp\sqrt{p_x},v\perp\sqrt{p_y}\)。去掉常数模式后的最大值就是 \(\sigma_2(B)\)。这是本卡由 Q 式推导的有限离散概率结论，不冒充已读 Bell／Linfoot 原始证明。[cite:1](#source-1)[cite:4](#source-4)

- **空支持：**没有像素对，P 不存在，全部特征无效。
- **单一活跃灰度：**不存在满足单位方差的非恒定变换，MCC 与 Pearson 均按本卡返回无效；库返回0或1是约定，不是测到相关性。
- **独立分布：**两边都非退化且 \(P=p_xp_y^T\)，B 秩1，第二奇异值0，MCC=0，f3=0，f12=f13=0。
- **确定性关联：**多个灰度一一对应时，可有重复的第一奇异值1，第二个仍为1；不能“删除所有等于1的值再取最大”。负线性关系可使 f3=-1，但 MCC=1。
- **极小边缘：**先删严格零边缘，再运算；正但很小的边缘不任意删掉。截断／平滑改变经验分布。迭代谱算法须记录残差及谱间距，不能默认第二向量可稳定辨识。
- **数值越界：**只有在记录的浮点容差内才夹到 \([0,1]\)；明显负特征值、\(\sigma_1\) 显著偏离1或较大虚部应报错。不要对 Q 直接使用要求对称的 `eigvalsh`。

外部实现的关键反例：mahotas v1.4.18 的 `compute_14th_feature` 在活跃灰度数大于2时对裁剪后的 P 调 `np.corrcoef` 再取谱，至多2时直接返回0；这不是上述 Q 构造。PyRadiomics v3.0.1 写出 Q 并调用一般特征值求解，但加 ε 分母且常量返回1。本卡采用正支持 B/SVD 路线，不声称复刻任一库的退化约定。二值非独立矩阵即可检出 mahotas 该分支与 Q 式的不一致；这里只静态对照源码和代数，不声称安装运行过两个库。[cite:3](#source-3)[cite:4](#source-4)

#### 原始分类与可复现教学替代的界限

原文 pp.616–617 的分段线性支线，为 K 类建立 \(K(K-1)/2\) 个两两分隔超平面，增广向量 \(z=(f^T,1)^T\)，按 \(w_{ij}^Tz\) 的符号计票，最大票类获选；两类平票文字使用两者分隔平面破局。训练权重引用 Fukunaga 书第4章的 regression-type algorithm，本次未取得该书算法完整定义，不将普通 SVM／最小二乘默认求解器冒称其训练实现。多于两类平票及恰好零响应的全套实现政策未能从所读扫描式可靠闭合。[cite:1](#source-1)

min-max 支线由训练样本的各类逐特征最小／最大值建立轴对齐超矩形；航空图实验采用 leave-one-out。正文还包含多个矩形覆盖及全部不覆盖时的比较公式，但扫描文本的分母、乘积和不等号有缺损，**本次未可靠核实完整重叠／外部决策规则**；不能用“最近矩形”或“最近中心”补上再标原法。其可辨识假设是以各维范围构造 uniform distribution，零宽区间同样需实现约定。[cite:1](#source-1)

原文附录的通用 14×均值／极差＝28 维，不等于三组实验全用28维：显微图正文写 f1、f2、f3、f9 的方向 mean／variance；航空图写 f1–f11 的 mean、range、deviation 共33维；卫星又并入光谱统计，邻近段落对 variance／standard deviation 的文字不完全一致。因此本卡不补造完整历史特征配置或实验复现表。[cite:1](#source-1)

为使教学分类链可执行地定义，**另选而非冒充原法**的基线：训练内标准化 \(\tilde f_j=(f_j-m_j)/s_j\)，删除训练常量维，估计各类中心 \(c_k\)，预测最小平方欧氏距离类。该规则来自同协方差 \(\sigma^2I\)、同先验高斯模型下的最大似然：\(\log p(\tilde f\mid k)=\text{const}-\|\tilde f-c_k\|^2/(2\sigma^2)\)。非同协方差或类内多峰时不保证合适。距离／分差拒识门在调参集校准，无模型或无效特征就返回 UNKNOWN；该教学替代本身不是原文性能证据。

#### 复杂度与保证范围

若 ROI 有 N 个像素、D 个偏移，计数约 \(O(DN)\)；稠密 f1–f13 为 \(O(DG^2)\)，每偏移矩阵存储 \(O(G^2)\)，流式处理可不同时保存 D 个矩阵。活跃支持 r×c 的完整 SVD 约 \(O(rc\min(r,c))\)，方阵最坏 \(O(G^3)\)；不能把原文“快速计算”当成 G 任意增大时仍线性。稀疏计数减存储不自动免除谱开销。

概率扰动可影响特征，但未提供从相机噪声直接到工业误分类率的通用理论界。f14 的奇异值对 B 的算子范数扰动有 \(|\Delta\sigma_k|\le\|\Delta B\|_2\)；由于 B 还除以边缘平方根，小边缘会放大从 P 到 B 的扰动，这不是小样本 MCC 准确性保证。量化对强度是分段常数，跨箱处不连续；优化收敛亦不代表纹理类别可辨。

### L5 实现：接口、伪代码与小矩阵检查

#### 数据布局与实现契约

图像为 H×W 整数 bin，mask 为同形布尔数组；计数使用足够宽的整型（建议 int64 并检查上界），归一化／矩／SVD 使用 float64。距离差先转有符号宽类型，避免 uint8 的 0−255 回绕。方向由显式 `(dr,dc)` 列表管理，不依赖库对 angle 的隐含定义。数据对象保存 `quantizer_version, offsets, symmetric, aggregation, feature_variant, log_base, counts, valid_flags`。

以下是**未接入仓库的参考伪代码**，涉及的决策模型接口不存在于当前专用模块；固定特征选择不能随一张测试 ROI 缺值而改变维数。

```text
EXTRACT(image, mask, config):
    validate shapes, finite valid pixels, quantizer and nonzero offsets
    bins = quantize_with_frozen_edges(image, config)
    per_offset = []
    for (dr, dc) in config.offsets:
        C = int64_zeros(G, G)
        N = 0
        for (r, c) with mask[r, c]:
            v = (r + dr, c + dc)
            if v inside image and mask[v]:
                C[bins[r, c], bins[v]] += 1
                N += 1
        if N == 0: return NO_PAIRS(offset=(dr, dc))
        if config.symmetric: C = C + transpose(C)
        P = float64(C) / sum(C)
        assert P >= 0 and sum(P) approximately 1
        f, valid = FEATURES_14(P, original_bin_labels, config)
        if any selected feature invalid: return DEGENERATE_FEATURE
        per_offset.append(f)
    return aggregate_in_fixed_order(per_offset), counts, metadata

FIT_CLASSIFIER(train_groups, config):
    features = EXTRACT each labeled training ROI
    reject invalid samples with recorded reasons
    fit scaler and fixed feature selection on training groups only
    fit explicitly selected classifier on transformed training features
    return frozen feature_config, scaler, selector, classifier

PREDICT(roi, model, rejection_policy):
    if no fitted model: return UNKNOWN
    f = EXTRACT(roi, model.feature_config)
    if invalid: return UNKNOWN with reason
    scores = model.classifier(model.selector(model.scaler(f)))
    return decide_with_frozen_policy(scores, rejection_policy)
```

FEATURES_14 对应 L4 定义：只对正概率求 log；中心矩直接累加以避免 \(E[X^2]-E[X]^2\) 大数相消；f13 先核实互信息仅有舍入级负值，再用 `expm1`；f14 对正边缘子矩阵白化并 SVD。若改变对数底，须同时登记 f13 的指数约定。分类训练可以用已有组件思想，但当前 HOG-SVM 候选重排不等于 Haralick 分类器。

#### 可手算的小矩阵

以下为**人为指定的确定性代数题／单元检查输入**，不是合成性能数据，也不是纹理基准实验。已量化图像

\[
I=\begin{bmatrix}0&0&1\\0&0&1\end{bmatrix},\quad\delta=(0,1)
\Rightarrow C_\delta=\begin{bmatrix}2&2\\0&0\end{bmatrix},\quad N=4,
\quad P=\frac{C_\delta+C_\delta^T}{8}=\begin{bmatrix}1/2&1/4\\1/4&0\end{bmatrix}.
\]

故 \(p_x=p_y=(3/4,1/4)\)，\(p_+=(1/2,1/2,0)\)，\(p_-=(1/2,1/2)\)。\(\mu_x=1/4,\sigma_x^2=3/16\)，\(E[XY]=0\)，\(\operatorname{Cov}=-1/16\)。14 特征的核对值为：

| 特征 | 期望值（本卡 nats 配置） |
|---|---|
| f1／f2／f3／f4／f5 | \(3/8,\ 1/2,\ -1/3,\ 3/16,\ 3/4\) |
| f6／f7／f8／f9 | \(1/2,\ 1/4,\ \ln2,\ (3/2)\ln2\) |
| f10／f11 | \(1/4,\ \ln2\)；概率数组方差版本 f10 为0 |
| f12／f13 | 令 \(h=-(3/4)\ln(3/4)-(1/4)\ln(1/4)\)，\(I_m=2h-(3/2)\ln2\)，则为 \(-I_m/h,\sqrt{1-e^{-2I_m}}\) |
| f14 | \(1/3\)；\(Q=\begin{bmatrix}7/9&2/9\\2/3&1/3\end{bmatrix}\) 的特征值为1、1/9 |

旧 f7 的零基值为 \(1/4+(1/2-\ln2)^2\)，不等于1/4；整体标签加1后，修正 f7 不变、f6 加2、旧 f7 改变。\(P\) 已对称不意味着 Q 对称，该例直接否定误用对称特征值算法的理由。

#### 终端内存检查代码

该代码是**文档内的有限算术检查器**，不是新增算法模块。执行时从本 Markdown 提取代码块在内存运行，不写脚本／数据文件；使用当前环境已有 NumPy。它覆盖计数、全部14个有限定义、Q/SVD、零方差、方向聚合与标签原点；不覆盖真实图像量化／分类训练全流程。

```python
import numpy as np

def entropy(v):
    v = np.asarray(v, dtype=float)
    v = v[v > 0]
    return -np.sum(v * np.log(v))

def features(P, labels=None):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("shape")
    if not np.isfinite(P).all() or (P < 0).any() or not np.isclose(P.sum(), 1):
        raise ValueError("probability")
    g = len(P)
    x = np.arange(g, dtype=float) if labels is None else np.asarray(labels, dtype=float)
    i, j = np.meshgrid(x, x, indexing="ij")
    px, py = P.sum(axis=1), P.sum(axis=0)
    ux, uy = px @ x, py @ x
    vx, vy = px @ (x - ux)**2, py @ (x - uy)**2
    sums, diffs = i + j, np.abs(i - j)
    sv, dv = np.unique(sums), np.unique(diffs)
    ps = np.array([P[sums == k].sum() for k in sv])
    pd = np.array([P[diffs == k].sum() for k in dv])
    sa, da = sv @ ps, dv @ pd
    hx, hy, hxy = entropy(px), entropy(py), entropy(P)
    mutual = hx + hy - hxy
    if mutual < -1e-12:
        raise ValueError("mutual information")
    mutual = max(0., mutual)
    corr = np.nan if vx == 0 or vy == 0 else np.sum(P*(i-ux)*(j-uy))/np.sqrt(vx*vy)
    imc1 = np.nan if max(hx, hy) == 0 else -mutual/max(hx, hy)
    active_x, active_y = px > 0, py > 0
    mcc = np.nan
    if active_x.sum() > 1 and active_y.sum() > 1:
        B = P[np.ix_(active_x, active_y)] / np.sqrt(px[active_x, None]*py[None, active_y])
        singular = np.linalg.svd(B, compute_uv=False)
        if not np.isclose(singular[0], 1., atol=1e-12) or singular[1] > 1 + 1e-12:
            raise ValueError("spectrum")
        mcc = np.clip(singular[1], 0., 1.)
    return np.array([np.sum(P*P), np.sum(P*(i-j)**2), corr, vx,
                     np.sum(P/(1+(i-j)**2)), sa, ps @ (sv-sa)**2,
                     entropy(ps), hxy, pd @ (dv-da)**2, entropy(pd),
                     imc1, np.sqrt(-np.expm1(-2*mutual)), mcc])

image = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int64)
C = np.zeros((2, 2), dtype=np.int64)
for row in image:
    for a, b in zip(row[:-1], row[1:]):
        C[a, b] += 1
assert np.array_equal(C, [[2, 2], [0, 0]])
P = (C + C.T).astype(float)
P /= P.sum()
h = entropy([.75, .25])
mi = 2*h - 1.5*np.log(2)
expected = [.375, .5, -1/3, .1875, .75, .5, .25, np.log(2),
            1.5*np.log(2), .25, np.log(2), -mi/h,
            np.sqrt(-np.expm1(-2*mi)), 1/3]
np.testing.assert_allclose(features(P), expected, atol=1e-12)
Q = (P @ np.diag(1/P.sum(0)) @ P.T) / P.sum(1)[:, None]
np.testing.assert_allclose(np.sort(np.linalg.eigvals(Q)), [1/9, 1], atol=1e-12)
shifted = features(P, [1, 2])
np.testing.assert_allclose(shifted[np.arange(14) != 5], features(P)[np.arange(14) != 5], atol=1e-12)
assert np.isclose(shifted[5], 2.5)
legacy = .25 + (.5 - np.log(2))**2
assert not np.isclose(legacy, features(P)[6])
independent = np.outer([.75, .25], [.75, .25])
np.testing.assert_allclose(features(independent)[[2, 11, 12, 13]], 0., atol=1e-7)
anti = np.array([[0., .5], [.5, 0.]])
np.testing.assert_allclose(features(anti)[[2, 13]], [-1., 1.], atol=1e-12)
constant = features(np.array([[1., 0.], [0., 0.]]))
assert np.isnan(constant[[2, 11, 13]]).all()
assert constant[0] == 1 and constant[8] == 0 and constant[12] == 0
try:
    features(np.zeros((2, 2)))
except ValueError:
    pass
else:
    raise AssertionError("empty probability accepted")
A, B = np.diag([1., 0.]), np.diag([0., 1.])
assert np.isclose((features(A)[0]+features(B)[0])/2, 1)
assert np.isclose(features((A+B)/2)[0], .5)
np.testing.assert_allclose(features(np.pad(P, ((0, 1), (0, 1)))), features(P), atol=1e-12)
print("HA01: 14-feature and boundary checks passed")
```

#### 优化与尚未执行的测试

计数可将合法 `(i,j)` 编为 `i*G+j` 再 bincount，必须核整型溢出并与循环计数逐格相等。大 G 可用稀疏键值计数；求和／差分布不能因压缩空 bin 而改灰度距离。滑窗增量更新需正确移除旧边、加入新边并维护 mask；并行用局部计数后整数归并。先验证一致性再剖析计数、特征、SVD 各段，不凭向量化代码更短就称加速成功。

尚未运行的集成测试包括：四方向和多距离坐标、孔洞 mask 与边缘计数、全黑与合法零灰度、uint8 极值、固定箱边及饱和、非对称 P、不同活跃行列数、极小正边缘、方向缺失、与指定外部版本逐项对照、真实批次训练／拒识、滑窗热图与端到端延迟。外部库静态核读、检查器通过和专用算法实现是三种状态，**本卡仍为未实现**。

### 五问五答：L1–L5 贯穿自测

1. **L1：给定两行 `[0,0,1]`，水平偏移1怎样输出类别？**答：四个合法有向对得到 C；加转置、除8得本卡 P；算14特征与有效位，按冻结方向顺序聚合，再用训练好的变换／分类器判别。只有一个方向不能伪装成四方向28维；没有模型只输出特征或 UNKNOWN。易错：直接把高 contrast 叫缺陷。证据：L1、L5 小矩阵及原文共现定义。[cite:1](#source-1)
2. **L2：为何不能把四个 GLCM 合并后算特征来“等价提速”？**答：非线性 F 不与平均交换；ASM 差为矩阵差的平方范数项，合并还可能按各方向对数加权。若业务允许改变表示可另训，但不能冒称等价。易错：归一化过所以都一样。证据：L2 恒等式及 PyRadiomics 两种聚合说明。[cite:4](#source-4)
3. **L3：换批误报，先提高灰度级数还是降异常阈值？**答：都不先盲改；先看曝光／箱边、纹理尺度／方向、有效对与逐特征漂移，再分组对照。更大 G 可能更稀疏；固定异常分数下下降阈值扩大 NG 集合，可能加重正常误报。分类与异常阈值不是一回事。易错：G 越大信息必更可靠，或用正常集估缺陷召回。证据：L3 与量化依赖研究。[cite:2](#source-2)
4. **L4：sum variance 为何不能以 sum entropy 为中心？MCC 为何取第二谱值？**答：任意中心二阶矩等于方差加中心偏差平方，熵不是灰度和均值；MCC 白化后的第一奇异方向是常数模式，去均值约束排除它，第二奇异值才是非平凡最大相关。零方差时单位方差约束不可行。易错：f14 是 Pearson 或 Q 第二特征值不需开根。证据：L4 推导、原附录与后续修正。[cite:1](#source-1)[cite:2](#source-2)
5. **L5：两个“Haralick 14维”库不一致，如何定位？**答：逐步固定量化、标签原点、mask、偏移、对称化、归一化、聚合、f7／f10／log底／f13指数／f14谱定义与退化约定；先比较 C、P，再比逐特征。本卡二值 P 的 MCC 应为1/3，可检出 mahotas v1.4.18 小支持直接返回0的差异。易错：一律加 ε 或容差放大到一致，或把 corrcoef(P) 当 Q。证据：L4 外部源码差异和 L5 检查器。[cite:3](#source-3)[cite:4](#source-4)

## 来源、版本与获取级

获取级独立于可信度：**T关键正文／公式已读、T部分正文已读、C固定版本源码静态已读、M书目／摘要、S搜索摘录、F失败或未取得**。P0 为原论文／作者或项目官方材料，P1 为同行评议后续研究；项目源码对“该版本如何实现”是直接证据，对“1973 作者当年如何实现”不是直接证据。原文事实使用 `[SINGLE-SOURCE]`；sum variance 修正有后续论文及两个项目对照，MCC 定义与实现差异明确保留，不形成性能多数表决。

- <a id="source-1"></a>**[cite:1] Haralick、Shanmugam、Dinstein，Textural Features for Image Classification，1973。P0。** 作者托管 https://haralick.org/journals/TexturalFeatures.pdf ：T部分，页首、引言、共现定义已读，网页返回截断。Stanford 课程镜像 https://web.stanford.edu/class/biomedin260/lectures/8/Haralick%20-%20Texture%20Features.pdf ：同文镜像，T关键；12页 PDF 经内存文本提取，重点核 pp.616–617 两类判别和实验特征文字、pp.618–619 附录 I、pp.620–621 等概率量化相关段落。f7 的 f8 中心、f14 的 Q 分母、14编号和28维聚合可辨；min-max 完整比较式、部分训练符号及 variance／deviation 语义未完全核实。未取得出版商单独勘误、原代码或数据，不引用扫描表格准确率作本站结果。
- <a id="source-2"></a>**[cite:2] Löfstedt 等，Gray-level invariant Haralick texture features，PLOS ONE 14(2):e0212110，2019-02-22。P1／该新方法原始研究。** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212110 ：T部分，摘要、Theory 的量化—计数—归一化—特征链及表2说明已读；文字明确 sum variance 原定义有误、最大相关的谱计算存在不稳定性并未纳入该文示例。网页部分公式为图片，未据其不可读公式补符号；本卡不实施该文的灰度级不变新版本，不把它标作1973正式勘误。
- <a id="source-3"></a>**[cite:3] mahotas v1.4.18，features/texture.py。P0项目源码。** https://raw.githubusercontent.com/luispedro/mahotas/v1.4.18/mahotas/features/texture.py ：C，核读 `haralick_features`、标签表及接口，确认 f7 开关、f10 两解释、log2＋exp、零方差 f3=1、f14 的 corrcoef 和小支持返回0分支。版本标签固定，但未锁提交哈希；未安装运行、未核 C 扩展计数实现或整个包。
- <a id="source-4"></a>**[cite:4] PyRadiomics v3.0.1，radiomics/glcm.py。P0项目源码／文档。** https://pyradiomics.readthedocs.io/en/v3.0.1/_modules/radiomics/glcm.html ：T/C部分，GLCM、对称化、逐方向均值／矩阵合并说明已读；续读 https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/v3.0.1/radiomics/glcm.py 的 `getMCCFeatureValue`、`getSumVarianceFeatureValue`、`getSumSquaresFeatureValue`、`getImc2FeatureValue` 函数体，确认 Q、常量1、均值中心与弃用提示、log2 指数协议。未执行包或验证其每个边界。早先 latest/features 页面在非GLCM段截断，不作为这些精确公式的证据。

补充获取限制：泛化检索没有找到可可靠引用的正式勘误单；作者另一篇 statistical/structural texture 综述的网页只返回前部，未用它证明某年改正了14特征。PDF 的 Python 直接 HTTPS 获取曾因证书链失败，后改用正常验证证书的系统请求读取镜像，不关闭 TLS 校验；解析依赖以 PyPI pypdf 5.9.0 wheel 内存加载，无项目安装／依赖文件变更。没有把失败调用记成正文成功。

## 验证记录与 G16 承接边界

原卡编写阶段的验证分三层：文献／固定源码静态核对；L5 内存检查器的确定性矩阵断言；Markdown 的单卡编号、五层、五问答、来源锚点、相对链接、表格、围栏和 UTF-8 检查。**2026-11-05 原卡记录：已从本文件提取并运行 Python 检查块，14特征及边界断言通过；文档结构检查通过。**本轮状态同步不重跑该算法检查器，不借用仓库旧的匹配回归或冒烟测试作为 Haralick 验证。

原卡编写阶段记录的写入范围仅为本文件，未编辑其他卡片、源码、依赖或台账；当时尝试全工作区 SHA-256 基线比对发现其他文件同步新增／变化，比对未通过，故**不声称全工作区无变化**，未修改、删除或回退这些外部变化。当时没有执行依赖安装，不能把同期环境中出现的新包归为本卡安装结果。本轮另按四文档约束同步状态与索引，不把前次单文件范围或运行记录当成本轮结果。

按独立复核及现台账“**具体方法专卡与机制匹配，不等于完整复现**”口径，HA01 已承接 GLCM 表示、14特征版本边界与方向汇总：**06-01 纹理表示记已有，07-01 原分类链记部分，G16 保留部分任务，不整体销项**。分段线性具体训练及 min-max 重叠／外部决策规则仍缺证，不能以最近中心教学替代补齐。异常迁移仍须另行建模与验收；本轮仅同步本卡状态及三份台账，历史 G16 ID 保留，不修改总览或实现。

该承接不关闭以下缺口：正式勘误与原实验代码未取；f10 原实现歧义及原 log 底未确定；min-max 完整分歧决策／分段线性具体训练未可靠核实；未实现专用提取器／分类训练，未开展真实缺陷和材质数据验证。**当前机制承接口径下 G16 已不能整体销项**，不只是在要求“1973 全算法逐数复现／完整历史规则闭合”时才保留；最近中心教学替代不能满足原分类链缺口。
