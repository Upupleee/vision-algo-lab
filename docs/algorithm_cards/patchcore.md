# PC01 PatchCore：局部记忆库、coreset 与近邻异常评分

[学习规范](../algorithm_learning_standard.md) · [缺陷异常总览](../defect_evolution.md) · [覆盖台账 G21／06-08](../evolution_coverage.md) · [逐卡索引](README.md) · [PaDiM](padim_root_polynomial.md#g19) · [D04 统计基线](remaining_domains.md#d04-固定特征的收缩-mahalanobis-异常评分)

<a id="pc01"></a>
<a id="g21"></a>

## 任务、演进与覆盖状态

**任务：**用经确认的正常训练图建立局部特征 memory bank，给待检图输出整图异常分数和空间热图；数值越大越异常，但不是缺陷概率。目标域不需要异常训练标签，也不微调骨干；ImageNet 有监督预训练、正常样本审核、建库和阈值标定仍是成本，不能简称“没有训练或监督”。前提是所选特征能区分目标缺陷、正常库覆盖合法变化，且训练／推理的预处理和特征契约一致。[cite:1](#source-1)[cite:2](#source-2)

**演进分支：**全图深层嵌入易丢小缺陷，逐位置单高斯又限制正常多模态表达 → 冻结中层、多层局部聚合并共享正常 patch 库 → 用 minimax coreset 压缩覆盖冗余 → 以最近邻距离定位、最异常 patch 及其正常邻域作整图评分 → 付出建库、存储和检索成本，仍不保证逻辑错装、域漂移或污染下的可靠性。与 SPADE、PaDiM 是可比较的表示／建模分支，不是全面替代；D04 的收缩 Mahalanobis 不含这些机制。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-05**。本文件只有 **PC01 一张卡**，唯一键 `patchcore.md:PC01`，承接 **G21／06-08**。2021-06-15 arXiv v1、2022-05-05 v2、CVPR 2022 正式会议记录与 **2023-03-06 作者代码提交**分账。代码固定为 `amazon-science/patchcore-inspection@fcaa92f124fb1ad74a7acf56726decd4b27cbcad`，不是已证明的论文实验提交。[cite:1](#source-1)[cite:3](#source-3)[cite:4](#source-4)

| 字段 | 状态与边界 |
|---|---|
| L1 步骤 | 已核验：正常建库、跨层对齐、选集、检索、论文重加权与热图流程；含可实现伪代码 |
| L2 机制 | 已核验：局部上下文、共享库、覆盖目标、最大异常响应及替换反例 |
| L3 工程权衡 | 部分：给出参数尺度、诊断和验证顺序，无真实定参／召回／节拍数据 |
| L4 数学 | 部分：式 (1)–(7) 的方法核对、有限度量推导齐备；未重证随机投影全部概率理论，不给源码启发式套无条件近似比 |
| L5 实现 | 部分：固定外部源码静态核对、布局与数值设计；本仓库未实现 PatchCore，无专用权重、bank 或基准实测 |
| 面试自测 | 答 5：逐层问题、答案、条件、易错点与证据入口 |
| 验证范围 | 资料、静态源码、文档结构与有限代数检查；不是骨干前向、FAISS 运行、训练或算法复现 |

## L1 步骤：正常建库与两条评分契约

### 输入、输出和停止

输入为正常图集合 \(X_N\)、冻结的预训练骨干 \(\phi\)、层列表、预处理、局部宽度 \(p\)、patch 步长 \(t\)、最终维度 \(D\)、保留数 \(K\) 及选择用投影 \(\psi\)。记全部训练 patch 行数 \(N\)，单张测试图 patch 数 \(P\)，保留库 \(C\in\mathbb R^{K\times D}\)。本卡用 \(t\) 表示论文式 (3) 的步长 \(s\)，避免与异常分数 \(s\) 重名。

输出应包含特征／库版本、整图分数、patch 分数网格、上采样热图及原图有效域；阈值判定是额外业务层。非有限输入、空库、维度不匹配、非法 \(K\)、未知权重、超时或裁掉关键检测区应拒绝模型／返回 UNKNOWN，不得默认 OK。这些拒绝规则是工程设计，不是作者代码已具备的防护。

1. 按工件／批次划分训练、验证、最终验收；只把可信正常图加入 bank。确认 RGB、缩放、裁剪、ImageNet 归一化及 ROI 坐标，不在测试集重新拟合阈值。
2. 骨干 `eval`、关闭梯度，取中层特征，参考配置为 WideResNet50 的 `layer2`、`layer3`。对每层每个空间位置提取局部 \(p\times p\) 邻域，聚合成局部描述；将低分辨率层双线性对齐至高分辨率参考网格后融合，得到 \(P\times D\)。具体可执行布局见 L5。[cite:2](#source-2)[cite:5](#source-5)
3. 将所有训练图的所有位置描述按行汇入 \(M\in\mathbb R^{N\times D}\)，保留来源图／坐标便于审计；检索不要求同一空间坐标。论文集合并集不等于源码自动去重，作者使用行拼接。[cite:2](#source-2)[cite:5](#source-5)
4. 在选择空间 \(Z=\psi(M)\) 做 coreset，迭代选到 \(K\) 行停止，索引映射回**原始 D 维**行构建 \(C\)；投影不是默认推理描述。空初始集需定义首点，不能直接执行 `min(empty)`。作者的两种初始化另见 L5。[cite:2](#source-2)[cite:7](#source-7)
5. 对新图采用同一嵌入器。**论文路径**：每个 patch 对 \(C\) 求一阶欧氏 NN，取最大距离对应的测试 patch 与正常近邻，再按式 (7) 重加权整图分数。**固定代码路径**：FAISS 返回平方欧氏，对指定 \(k_{\rm nn}\) 个距离求均值，再取全图最大；没有论文重加权。[cite:2](#source-2)[cite:5](#source-5)[cite:6](#source-6)[cite:9](#source-9)
6. 未作整图权重乘法的 patch 距离按网格复位，双线性上采样，再高斯平滑。冻结图像／像素阈值，在独立数据上判定；全图流程在所有 patch 查询和后处理完成后停止，不以“分数收敛”作为退出条件。

### 尺寸走查与坐标契约

以作者 README 的 **224×224、两层、p=3、t=1、预聚合维度 1024、最终 D=1024** 为参数代入，不是本仓库前向实验。常规 ResNet 步幅下，两层空间尺寸为 28×28 和 14×14、通道为 512 和 1024；参考网格为 28×28，所以每图产生 \(P=784\) 行。先把较粗层的 **patch 中心网格**插值至 28×28，再按对应位置融合，绝不能把一层展平成 196 行后直接与另一层 784 行拼接。尺寸应在真实模型运行时断言，不依据骨干名字永久写死。[cite:5](#source-5)[cite:8](#source-8)

224×224 是网络输入／裁剪后的坐标，不自动是相机原始分辨率。若预处理先 resize 再 center crop，应保存缩放、裁剪偏移和 ROI 有效域；裁掉的原图区域没有异常证据，不能填零后宣称“全图正常”。共享 bank 在表示上可接不同网格尺寸，不代表固定 `input_shape` 的作者后处理器会自动恢复任意原图坐标。[cite:5](#source-5)[cite:11](#source-11)

## L2 机制：必要性、不变量与反例

| 步骤 | 为什么需要／保持什么 | 删除或替换的后果 |
|---|---|---|
| 冻结中层而非全图最终层 | 同时保留局部变化和足够语义上下文，减少最后分类层的任务偏置 | 最终全局平均会稀释细划痕；浅层也可能过度响应照明与噪声 |
| 局部聚合与跨层对齐 | 容纳小位移、融合尺度，参考网格的位置对应必须不变 | p=1 可能丢邻域上下文；过大 p 会抹小缺陷；错位拼接生成并不存在的描述 |
| 共享正常 memory bank | 保留非参数多模态支持，不逐位置拟合一个高斯 | 正常零件出现在错误位置仍可能找到低距离近邻；不能替代装配数量和关系检测 |
| k-center 型 coreset | 优先控制最坏覆盖半径，不按正常模式频次随机保留 | 随机采样可能漏罕见正常簇；覆盖采样也可能优先保留污染离群点，没有自动去污能力 |
| patch NN 与最大响应 | 单个局部异常即可提高整图响应，不用大量正常区域平均稀释 | 最大值也对边界伪影、单点坏成像敏感；均值／top-k 是新决策契约 |
| 正常近邻周围的重加权 | 用候选最近正常点的局部支持关系调整最大响应 | 不等于多取几个测试 patch 的近邻均值；不等于对全部热图乘同一权重 |
| 热图上采样／平滑 | 把稀疏响应映射为可用空间图，减弱块状离散性 | 不能恢复已经缩没的缺陷；平滑会降低尖峰并扩散边界，可能影响像素排序 |

方法机制来源于论文 §3；失效反例和业务限制为本卡工程分析。尤其不能把“Total Recall”的题名变成零漏检承诺，也不能把正常库覆盖半径当缺陷召回下界。[cite:1](#source-1)[cite:2](#source-2)

## L3 工程权衡：参数、证据与调参顺序

### 参数及尺度

| 参数 | 单位／有据起点 | 代价、限制和应记录的证据 |
|---|---|---|
| 输入尺寸、裁剪 | 输入像素；作者示例 resize=256、imagesize=224 | 先核最小缺陷实际占几个像素及裁剪漏区；加分辨率使 P 和库大小同步增加 |
| 层列表 | 网络层名；示例 layer2＋layer3 | 固定层顺序、权重标识、步幅和归一化；改变任一项通常需重建库 |
| p、t | 特征网格单元；示例 p=3，默认 t=1 | p 不是原图 3 像素；t 增大降库／查询成本但有漏采风险；优先用正奇数 p |
| 预聚合维度、D | 描述坐标数；README 示例均为 1024 | adaptive pooling 不是 PCA；维度变化会改变几何和距离尺度，不能沿用旧阈值 |
| 保留率 r、K | r 无量纲；示例 r=0.1，K=floor(rN) | 原文也讨论 1%、10%、25%；都不是通用最优值，至少需 K≥1；r=1 可直接保留全库 |
| 投影维度 q | 选择空间坐标数；代码默认 128 | 与 D、N、随机矩阵分布及容许畸变有关；128 不自动满足 JL 误差目标 |
| 起始参考数 a | 近似选择器默认 10 | 随机参考用于距离均值初始化，不等于先选十个 coreset 中心；记录种子及最终行号 |
| \(k_{\rm nn}\) 与 b | 两种不同的邻居数量；代码默认 \(k_{\rm nn}=1\) | b 是论文对 \(m^*\) 的正常邻域大小，已读方法未给统一默认 b；不得把 AN-1 或第三方 b=9 当论文定值 |
| 高斯 σ | 上采样后的像素标准差；原文／代码为 4 | 不是 4×4 核或 patch 网格的 4 格；核截断、边界模式随库版本冻结 |
| 异常阈值 | 欧氏／平方欧氏各自分数单位 | 只在同一模型版本和评分契约下解释；降低阈值扩大 NG 集合，通常同时影响误报与漏检 |

配置来源为论文、固定作者代码与 README，不是本仓库调参结论。[cite:2](#source-2)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)

### 症状与排查

| 症状 | 先收集的证据 | 调整顺序与回退 |
|---|---|---|
| 新批次正常大面积高分 | RGB／归一化、权重哈希、曝光、近邻来源和覆盖半径 | 先成像与契约，再补可信正常模式；不能只提高阈值 |
| 细缺陷消失 | 原图与裁剪图、各层网格响应、平滑前后峰值 | 先分辨率／有效 ROI，再层和 p，最后 σ；插值不能补回证据 |
| 压库后正常误报变多 | 全库对保留库的距离增量、被丢正常模式 | 增大 K／检查选择距离与随机种子；先精确查询隔离 ANN 误差 |
| bank 逐渐接受污点 | 采集来源、审核记录、新增行与近邻命中 | 停止未经审核的在线更新，回滚可信库；低分不是良品认证 |
| 复写分数不同但 NN 相同 | 欧氏还是平方欧氏、k 值、是否重加权／归一化 | 先锁评分契约，再比较后处理；不能仅用“同样 AUROC”证明数值一致 |
| 延迟／内存超预算 | 骨干、patch 展开、CPU↔GPU、选择器、查询的分段耗时和峰值 | 先分块／缓存，再 coreset，最后 ANN／量化并测损失；不引用作者时延作本机实测 |

推荐顺序：**分组划分与正常审计 → 预处理／可见域 → 局部表示 → 精确全库基线 → coreset 误差 → 评分与阈值 → 热图操作点 → ANN 加速**。仅正常验证可以测正常拒绝率和资源，不能证明真实异常召回。最终至少分别记录图像级 AUROC／操作点漏检、像素级 AUROC／区域指标、缺陷尺寸分层和批次分层；本卡未运行这些评测。

## L4 数学：聚合、覆盖目标与评分原式

### 局部表示与 memory bank

论文式 (1)–(4) 定义奇数 p 的邻域及自适应平均聚合：

\[
\mathcal N_p^{(h,w)}=\{(a,b):|a-h|\le\lfloor p/2\rfloor,\ |b-w|\le\lfloor p/2\rfloor\},
\qquad g_j(h,w)=f_{\rm agg}(\{\phi_j(x)(a,b):(a,b)\in\mathcal N_p^{(h,w)}\}).
\]

选用层的局部描述经双线性空间对齐和融合形成 \(z_x(h,w)\)，依步长采样形成 \(\mathcal P(x)\)，再建 \(M=\bigcup_{x\in X_N}\mathcal P(x)\)。普通逐通道 \(p^2\) 均值只是解释局部平滑的特例；作者代码实际上将 \(C_jp^2\) 展平后 adaptive average pooling，不能把 `AvgPool2d + concat` 宣称为逐元素相同实现。原文式 (3) 的索引不作为零基实现边界依据，源码的 Unfold 输出尺寸才是该版本可执行契约。[cite:2](#source-2)[cite:5](#source-5)[cite:6](#source-6)

### k-center 目标、贪心与条件近似保证

补明式 (5) 隐含的**固定预算**后，离散 k-center／minimax facility location 目标为：

\[
C^*\in\arg\min_{C\subseteq M,\ |C|=K}R_M(C),\qquad
R_M(C)=\max_{m\in M}\min_{c\in C}\|m-c\|_2.
\]

不约束 K 就可取全库使半径为零，失去压缩意义。它不是 k-means 的平方距离和，也不是选 K 个均值中心。论文 Algorithm 1 用最近已选点距离最大的候选逐次加入，在投影空间做选点但返回原始描述。[cite:2](#source-2)

**标准 farthest-first 的 2 近似推导：**有限度量空间、\(1\le K\le N\)、任意一个真实首点、之后每次精确选择距已选集最远且未选的点。设最终半径 R，加上第 K+1 个最远点得到 K+1 个两两距离至少 R 的点。最优 K 个半径 \(R^*\) 的簇中，必有两个落在同一簇；三角不等式给它们距离至多 \(2R^*\)，故 \(R\le2R^*\)。K=N 时 R=0。此处为本卡条件推导，不是对所有名为 greedy 的代码一概授予保证。

平方欧氏不是度量，不满足所需三角不等式；不过精确 farthest-first 的 min／argmax 在统一平方后选点相同，因此其**欧氏半径**仍有上述界，平方半径对应最多 \(4(R^*)^2\)，不能直接称“平方目标 2 近似”。

设规范化随机投影在**固定训练库全部点对**上满足：

\[
(1-\varepsilon)\|u-v\|_2^2\le\|\psi(u)-\psi(v)\|_2^2
\le(1+\varepsilon)\|u-v\|_2^2,\qquad 0<\varepsilon<1.
\]

若投影空间上确实执行标准精确 farthest-first，则把两侧界传回原空间：

\[
R_M(C_{\psi})\le2\sqrt{\frac{1+\varepsilon}{1-\varepsilon}}R_M(C^*).
\]

JL 是适当随机矩阵和维度下的高概率有限点集保距结果；典型维度尺度为 \(q=O(\varepsilon^{-2}\log(N/\delta))\)，分布、常数及失败概率 δ 需另按定理设定。可能需要的 q 大于 D，此时不存在所需的有效降维。固定 q=128、单次抽样或只检查少量点对都不能认证全点对条件；保证也不自动外推所有未来异常样本、NN 身份或召回率。[cite:2](#source-2)[cite:10](#source-10)

**不把此界贴给作者近似选择器：**其初始化距离均值持续参加 `min` 更新，未必等于距已选真实集合的最小距离；默认随机 `Linear` 也不是已证明满足指定 ε、δ 的校准矩阵。对该路径没有本卡可证的同一近似比，应实测保留库的原空间覆盖半径与分数误差。[cite:7](#source-7)

### coreset 如何影响一阶 NN 距离

同一欧氏空间，C 是 M 的真实子集，令 \(d_M(q)=\min_{m\in M}\|q-m\|_2\)，\(d_C(q)=\min_{c\in C}\|q-c\|_2\)。令 r 为实际原空间覆盖半径 \(R_M(C)\)，有：

\[
d_M(q)\le d_C(q)\le d_M(q)+r.
\]

第一侧来自候选减少；第二侧取 q 的全库最近点 m，再取覆盖 m 的 c，以三角不等式得出。对 patch 最大值同样得到至多 r 的增量；平方距离增量至多 \(2r\,d_M(q)+r^2\)，不是固定 r。该确定性界只约束未重加权的一阶距离；式 (7) 的近邻支持集也会变化，不能把同一加性界或分数单调性直接套给最终重加权分数。重建出更大但不嵌套的 coreset 也不保证逐样本分数单调下降。

### 最近邻与整图重加权原公式

为避免原式 (6) 连写 `argmax argmin` 的索引歧义，等价拆为：

\[
\delta(q)=\min_{m\in C}\|q-m\|_2,\quad
q^*\in\arg\max_{q\in\mathcal P(x)}\delta(q),\quad
m^*\in\arg\min_{m\in C}\|q^*-m\|_2,\quad s^*=\|q^*-m^*\|_2.
\]

式 (7) 在 v1 与 2022 v2 方法正文中均为：

\[
s=\left(1-\frac{\exp\!\left(\|q^*-m^*\|_2\right)}
{\displaystyle\sum_{m\in\mathcal N_b(m^*)}\exp\!\left(\|q^*-m\|_2\right)}\right)s^*.
\]

\(\mathcal N_b(m^*)\) 是以**正常库点 m*** 为查询、在正常库 C 中找出的 b 个近邻，不是以 q* 为查询另取 b 个最近点；分母随后计算的是 **q* 到这些支持点**的距离。指数为**正距离**，不是常见 `exp(-distance)`，也不是距离平方。[cite:2](#source-2)

库自查询通常包含 m* 自身，本卡参考实现显式包含该行；原文未单独规定 self-exclusion 和并列处理，固定作者路径又未实现此式，故这部分是透明的可执行约定，不声称已确认作者实验全部细节。要求 \(2\le b\le K\)，精确并列以行号稳定选择且保留 m*；K=1 或不启用重加权时明确返回未重加权 \(s^*\)，不能悄悄声称仍执行式 (7)。若强行含自身 b=1，则式 (7) 恒为零。

设所有支持距离为 \(r_i\)，其中 \(r_0=s^*\)。因为 m* 是全库 NN，\(r_i\ge r_0\)，故权重 \(w\in[1-1/b,1)\)，\(0\le s\le s^*\)。其他支持点更远时，分母变大、w 更接近 1。论文文字称“increase the anomaly score”，应理解为相对稠密支持情形增大权重，**不是把分数放大到原始最大距离以上**。该因子不是经校准的正常／异常后验。[cite:2](#source-2)，不等式为本卡直接推论。

稳定计算令 \(L=\operatorname{logsumexp}(r_i)\)，\(\ell=r_0-L\le0\)，则 \(w=1-e^\ell\)；用 `-expm1(ell)` 降低接近零时相消，用减最大值实现 logsumexp 防溢出。先检查有限值和非空支持，不能用极小分母掩盖缺失邻居。

### 热图后处理和平方欧氏差异

论文空间热图为：

\[
H=G_{\sigma=4}*\operatorname{BilinearResize}
\left(\operatorname{reshape}_{H_f,W_f}\{\delta(q)\}\right).
\]

整图重加权只作用于 s*，不对每个像素套同一 w；整图分数也不是平滑后 H 的最大值。固定代码的同一后处理链输入则是 FAISS 平方距离的 k 近邻均值。[cite:2](#source-2)[cite:5](#source-5)[cite:6](#source-6)

在**同一 C、同一表示、精确检索、k=1、无重加权／平滑／集成**时，\(s_{\rm code}=(s^*)^2\)，所以排序保持，非负阈值可按平方转换。但不能推出所有步骤等价：\(\operatorname{mean}(d^2)\ne\operatorname{mean}(d)^2\)，\(G*(d^2)\ne(G*d)^2\)，将平方距离塞进式 (7) 还会改变指数权重。已有加权分数不能只靠最后开平方复原。[cite:6](#source-6)[cite:9](#source-9)

### 复杂度与数值量纲

骨干计算量另记 \(F\)，以下均为算法量级，不是运行时间。令 q 为投影维度、a 为初始随机参考数量。

| 阶段 | 时间量级 | 主要空间 |
|---|---|---|
| 特征抽取／patch 展开 | 全训练图骨干 F，加 \(O(\sum_j N_j C_jp^2)\) 局部处理 | bank \(O(ND)\)；一批展开张量另占 \(O(B\sum_j P_j C_jp^2)\) |
| 稠密随机投影 | \(O(NDq)\) | \(O(Nq+Dq)\)，原 bank 仍需保留 |
| 标准流式 farthest-first | \(O(NKq)\) | 投影点 \(O(Nq)\)＋最近距离向量 \(O(N)\) |
| 作者 Greedy 全矩阵路径 | \(O(N^2q+NK)\) | \(O(N^2)\) 距离矩阵，不能仅报 O(N) |
| 作者 ApproximateGreedy 路径 | \(O(Naq+NKq)\) | 初始 \(O(Na)\)＋投影 \(O(Nq)\)＋O(N) 工作距离 |
| 精确测试查询 | 每图 \(O(PKD)\)，top-k 维护另计 | bank \(O(KD)\)；分块查询避免完整 \(P\times K\) 临时矩阵 |
| 论文重加权 | 每图再查一次 m* 邻域约 \(O(KD)\)，支持距离 \(O(bD)\) | O(bD)；可离线缓存正常近邻，增加建库成本 |
| 热图插值／高斯 | 插值 O(HW)；可分离有限高斯约 O(HWσ) | O(HW)；截断规则固定时解释此量级 |

FP32 的最终 bank 裸向量字节数为 \(4KD\)，不包括 FAISS 副本、模型、中间激活和 Python 对象。coreset **不自动消除先收集全库的内存峰值**，作者 `_fill_memory_bank` 先拼接全部描述才采样。ANN／半精度的代价和误差需按具体索引验证，不许用“FAISS”一词代替精确／近似方法名称。[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)

## L5 实现：固定源码映射、伪代码与检查

### 作者固定提交的具体路径

| 文件／函数 | 已核对的行为 | 论文／参考实现之间的边界 |
|---|---|---|
| `patchcore.py: PatchMaker.patchify` | Unfold，padding=floor((p−1)/2)，stride=t；输出 B×Pj×Cj×p×p | 零填充边界；偶数 p 不保持同样网格，参考契约限制奇数 p |
| `PatchCore._embed` | 每层先 patchify；把粗层每个通道／邻域偏移的中心网格双线性插值到首层网格，`align_corners=False` | 不是先把原始粗层图随意 resize 再 Unfold；边界、步长使替换不自动等价 |
| `common.py: MeanMapper` | 每行 Cj×p×p 展平，`adaptive_avg_pool1d` 到预聚合维度 | 不是单纯逐通道 p×p 平均；池化桶可能跨通道边界 |
| `Preprocessing / Aggregator` | 前者按层 stack；后者展平层及预聚合维，再 adaptive pooling 至 D | 聚合顺序、层顺序和 D 都进入特征定义；不是训练投影头 |
| `PatchCore._fill_memory_bank` | 所有图描述 concatenate，sampler 选子集，scorer 建索引 | “fit”主要是建库，不执行目标域梯度训练，也没有自动审核污染 |
| `sampler.py: _reduce_features` | 若 D=q 则不投影，否则临时随机 `Linear(D,q,bias=False)`，不训练 | 不是 sklearn 自动求 JL 最小维度；若 q>D 甚至变成升维；返回的是原始 D 维选中行 |
| `GreedyCoresetSampler` | 全部点对欧氏矩阵，初始 anchor 为各行距离向量的 L2 范数 | 此初始值不小于任一单点距离，第一轮后更新等于到首点的距离；除重复／并列防护外可解释成标准 farthest-first 的一种首点规则 |
| `ApproximateGreedyCoresetSampler` | 默认 a=10 个随机参考点，初值为各点到参考点的**欧氏距离均值**；迭代取最远再 min 更新 | 不是到已选集的精确距离，旧初值仍是上限夹持；不能直接宣称每一步均为标准贪心最远点 |
| `common.py: FaissNN` | CPU `IndexFlatL2`／GPU `GpuIndexFlatL2` | 此默认路径是穷尽式精确向量检索，不是因为用了 FAISS 就成为 ANN |
| `NearestNeighbourScorer.predict` | 返回距离沿邻居轴 `np.mean` | FAISS L2 为平方欧氏，未开平方；k>1 是均方距离，非论文支持域重加权 |
| `PatchCore._predict / PatchMaker.score` | 从 scorer 的 patch 分数直接取 max 为整图分数 | 这一整条固定路径没有式 (7) 的第二次库查询、exp 或权重；定义了其他 NN 接口不代表用到了重加权 |
| `RescaleSegmentor` | 到 `input_shape[-2:]` 双线性插值，`align_corners=False`，SciPy `gaussian_filter(sigma=4)` | 没有把整图权重乘热图；Gaussian 的边界及截断依赖实际 SciPy 版本 |

以上是指定提交的静态事实，不外推所有 PatchCore 实现，也不将差异直接判成 bug。源码入口及完整固定链接见来源 5–7；FAISS 距离语义由来源 9 交叉核对。作者依赖文件多为下限且 `faiss-cpu` 未锁版本，所以**锁 git SHA 不等于完整可复现环境**。[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:9](#source-9)[cite:11](#source-11)

### 可实现伪代码

以下是**论文评分＋明确首点的标准贪心参考设计**，不是固定作者近似选择器的逐行翻译，未接入本仓库。`EMBED` 对应前表布局；距离搜索可以内部用平方欧氏，但回到论文评分前必须开平方。以行号消解并列，掩蔽已选行以免重复采样；原始重复描述的保留／去重政策必须记录。

```text
BUILD(normal_images, config, K, projection, seed):
    validate_preprocessing_weights_and_normal_manifest(config)
    M = concatenate_rows(EMBED(image, config) for image in normal_images)
    require finite(M) and 1 <= K <= rows(M)
    Z = projection(M)
    require finite(Z)
    first = seeded_row_index(rows(M), seed)
    selected = [first]
    nearest = euclidean_distance_each_row(Z, Z[first])
    while len(selected) < K:
        i = stable_argmax_over_unselected(nearest, selected)
        selected.append(i)
        nearest = minimum(nearest, euclidean_distance_each_row(Z, Z[i]))
    C = contiguous_float32(M[selected])
    return exact_L2_index(C), C, selected, config

PREDICT_PAPER(image, index, C, config, b, reweight):
    Q, grid, valid_transform = EMBED_WITH_COORDINATES(image, config)
    d2, ids = index.search(Q, 1)
    require finite(d2) and valid_indices(ids, rows(C))
    d = sqrt(clamp_roundoff_negative_only(d2, 0))
    u = stable_argmax(d)
    j = ids[u, 0]
    raw = d[u, 0]
    score = raw
    reweight_applied = false
    if reweight and rows(C) >= 2:
        require 2 <= b <= rows(C)
        support = nearest_bank_rows_including_self(C, j, b)
        r = euclidean_distance_each_row(C[support], Q[u])
        ell = raw - logsumexp(r)
        require ell <= numerical_tolerance
        score = raw * (-expm1(min(ell, 0)))
        reweight_applied = true
    patch_map = reshape_row_major(d, grid)
    heatmap = gaussian(bilinear_resize(patch_map, config.input_hw), sigma=4)
    return score, patch_map, heatmap, valid_transform, reweight_applied

PREDICT_FIXED_CODE_SCORE(Q, index, k_nn):
    d2, ids = index.search(Q, k_nn)
    require 1 <= k_nn <= index.size and finite(d2) and valid_indices(ids, index.size)
    patch_scores = mean(d2, axis=neighbors)
    return max(patch_scores), patch_scores
```

伪代码中的验证比作者路径更严格。作者采样数量 `int(N*r)` 可得零，`BaseSampler` 又只接受 0<r<1（全库用 IdentitySampler），两个贪心循环没有显式掩蔽已选索引，重复描述／全零距离下可能再次选同一行；参考设计不能照抄后仍声称满足 K 个不同中心。若提前以覆盖半径阈值停止，应记录实际 K，不再称固定 K 复现。[cite:7](#source-7)

单点库 K=1 的工程回退与 L4 一致：即使请求重加权，也返回 raw，且 `reweight_applied=false`；关闭重加权时同样返回 raw。空库仍非法；K≥2 且请求重加权时，非法 b 必须拒绝，不能静默回退。此保护不是论文原算法性能保证。边界检查应分别覆盖 K=1 的开／关两种情况和 K≥2 的非法 b。

### 数值、坐标与优化等价性

- 特征一般用 FP32 连续 N×D 布局；查 NN 前检查 finite、维度和范围。批次展开为 `(batch,h,w)` 行序，网格复位必须用同一顺序，不能交换 h/w 或把整批最大值当单图最大值。
- \(\|a-b\|^2=\|a\|^2+\|b\|^2-2a^Tb\) 有大数相消风险。只对浮点舍入尺度内的小负值夹零；明显负值、溢出和 NaN 应报错，并用直接差分／更高精度小样核对。作者采样器使用 `clamp(0).sqrt()`，不表示所有病态输入已安全。
- 核查骨干 BatchNorm 的 eval 状态、通道和权重版本；不擅自 L2 归一化／白化特征。这些变换会改变原本的距离几何，需要重建库、重定阈值。
- 若还原论文式 (7)，sqrt 必须发生在指数之前；对已均值、已平滑或已重加权结果只做末端 sqrt 一般错误。邻居数量超过 K 时必须拒绝，不让索引返回的无效 ID 或无穷距离污染 softmax。
- 分块精确查询、缓存固定骨干输出、保留同一浮点精度／并列规则，有机会维持同一数学结果，但仍需逐行距离和 ID 对照；半精度、IVFPQ、改池化顺序、改 σ、改 coreset 是近似／模型变更，不是默认无损优化。
- 模型包至少记录 SHA、依赖版本、权重校验、输入变换、层序、p/t/D、库来源与选中行、随机种子、距离和评分契约及阈值。作者存在 pickle 读写入口，只能加载可信来源，不能把未知模型包当纯数据安全打开。

### 有限数值检查与待实施测试

下列是**显式数学单元例／恒等式代入**，不是采集数据、模拟性能或论文实验结果。它们只检查公式方向、尺度和次序：

| 检查 | 预期关系／值 | 可发现的问题 |
|---|---|---|
| 含自身 b=1 | w=0 | 把代码 k=1 错当论文 b=1 |
| b=2，支持距离分别 1、2 | \(w=1-e/(e+e^2)\approx0.73105858\)，s*=1 时 s 相同 | 正负指数、支持距离来源、权重是否超过 1 |
| 同一支持误用平方距离 | \(1-e/(e+e^4)\approx0.95257413\)，不等于前行 | NN 身份不变但重加权变了 |
| 两距离 1、3 等权平均 | 均方为 5，均值平方为 4 | 平均后开平方无法还原欧氏均值 |
| 两像素分数 0、2 的等权插值 | 平方后插值得 2，插值后平方得 1 | 热图平滑／插值与平方不交换 |
| q*=0，完整库 {1,2,4}，子集 {2,4} | 全库距离 1、子库 2、覆盖半径 r=1，达到 dC=dM+r | 子集 NN 增量界、保留率不等于误差比例 |
| ε=0.2 的条件投影界 | \(2\sqrt{1.2/0.8}=\sqrt6\approx2.44948974\) | 把平方保距条件误写成欧氏 2(1+ε) |
| 224 输入的上述 28×28 网格 | P=784，D=1024 时每图描述裸 FP32 为 3,211,264 字节 | 网格、维度与内存量纲；不是真实内存峰值 |

**尚未执行的算法测试设计：**固定权重的嵌入前向与逐步布局对照；边界脉冲／非方形网格的插值定位；空库、重复行、K=1／N、k>K、NaN、超大幅值；小规模全点对精确 NN 对 FAISS；标准贪心的覆盖半径／去重不变量；固定代码与论文评分双路径对照；不同块大小的结果与峰值内存；按工件分组的真实正常／缺陷评测。既有匹配冒烟和 RTImg 对焦数据不替代这些测试。

## 面试自测：贯穿 L1–L5 的五问

| 层级／问题 | 答案要点及成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：224 图经两层后怎样到异常图和整图分数？ | 局部展开、粗网格对齐、逐位融合成 784 行；正常建库／coreset；测试 NN；论文选 q*、m* 并式 (7)；原 patch 距离复位、插值、高斯；给出未知输入退出 | “全图平均特征直接 NN”漏掉局部流程；裁剪热图不自动是原图全域 | L1、L5 布局表，来源 2／5／6 |
| L2：为什么不用逐位置高斯或随机抽库？ | 共享库容纳多模态与一定位置变化；覆盖目标关注罕见模式；随机抽样是另一近似；正常污染也可能被覆盖选择保留 | “共享库能查出一切错装”“coreset 会自动删异常”均不成立 | L2、L4 覆盖界，来源 1／2 |
| L3：缩库后误报上升且节拍不足，先改什么？ | 先固定表示与精确检索，检查全库到子库原空间覆盖和罕见正常；再调 K／分块，最后 ANN；独立分组验证阈值和资源 | 直接增阈值可能掩盖丢失正常模式并漏缺陷；论文时延不是本机证据 | L3、复杂度表，来源 7／8 |
| L4：为什么说贪心 2 近似却不能直接保证作者近似版？ | 给出 K+1 点鸽巢和三角不等式证明；说明真实中心、精确最远点、度量条件；投影要全点对保距；距离均值初始化会破坏最近已选集不变量 | 平方距离直接用三角不等式；q=128 自动 JL；覆盖界等于召回界 | L4、L5 采样器表，来源 2／7／10 |
| L5：FAISS 最近邻索引相同，为何分数／热图不一致？ | L2 返回平方距离；固定代码均值后 max、无式 (7)；论文要欧氏、以 m* 取正常支持，正指数稳定求值；平方与均值／插值不交换 | 末端 sqrt 就能修复全部差异；把 k_nn 当 b；热图乘整图权重 | 式 (7)、有限数值例，来源 5／6／9 |

## 来源、版本与获取深度

**分层与诚实口径：**P0 原论文／会议记录支撑方法和年代；P0 作者固定源码支撑该实现行为；P0 库官方文档只支撑其 API／数学说明，不替代论文或作者实验。获取深度沿用台账 B=书目、A=摘要、T=方法正文／节选，另标“源码静态”。论文、预印本、补充材料和作者仓库来自同一研究线，不能充作三个独立复现实验。方法与论文效果为 `[SINGLE-SOURCE]`；平方欧氏语义经作者调用点与 FAISS 官方文档交叉确认。没有引入第三方教程冒充原始来源。

- <a id="source-1"></a>**[cite:1] Roth 等，CVPR 2022，Towards Total Recall in Industrial Anomaly Detection。** [会议页](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)：B/A 已读，2022 年 6 月、14318–14328；[CVF 正文 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)：T 部分，返回至 §3.1 及后续图示处截断，不能声称靠此响应逐页读完式 (7)。CVF 说明除水印外为接受版本，最终出版社版在 IEEE；本次未另取 IEEE 版。2022 摘要“up to 99.6%”仅为原作者结果，不是本仓库成绩。
- <a id="source-2"></a>**[cite:2] arXiv 2106.08265 的方法正文。** [2021 v1](https://arxiv.org/html/2106.08265v1)、[2022 v2](https://arxiv.org/html/2106.08265v2)：网页默认返回截断后，以只读 HTTP 按章节提取并核对 §3.2–3.3、式 (5)–(7)、Algorithm 1、附录 A；v2 另完整提取 §3.1，读取聚合、对齐及式 (1)–(4)，v1 读取前部及相关章节。两版重加权式一致；不声称全篇逐字／所有实验表版本 diff。2021 摘要为 99.1%，2022 为最高 99.6%，不是同一配置自动提升的本站复现结论。**完整评分公式的主要可读证据是这些版本正文，不掩盖 CVF PDF 截断。**
- <a id="source-3"></a>**[cite:3] arXiv 版本登记。** [摘要与提交历史](https://arxiv.org/abs/2106.08265)：B/A 已读，v1=2021-06-15，v2=2022-05-05，备注 Accepted to CVPR 2022；网页生成／抓取日期不替代方法年。
- <a id="source-4"></a>**[cite:4] 作者仓库固定版本。** [提交](https://github.com/amazon-science/patchcore-inspection/commit/fcaa92f124fb1ad74a7acf56726decd4b27cbcad)：通过 [GitHub API](https://api.github.com/repos/amazon-science/patchcore-inspection/commits/fcaa92f124fb1ad74a7acf56726decd4b27cbcad) 对应提交元数据，日期 2023-03-06，消息 Remove wrong symbolic link。先解析 main，再固定全部源码 URL；后续源码结论仅针对该 SHA，不把移动 main 当永久版本，也没有证明该提交生成论文表格。
- <a id="source-5"></a>**[cite:5] 固定作者 patchcore.py。** [源码](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/patchcore.py)：源码静态，读取 `_embed`、`_fill_memory_bank`、`_predict`、PatchMaker 和模型保存／载入链；确认网格插值、padding、max 及缺少式 (7)，未执行 PyTorch。
- <a id="source-6"></a>**[cite:6] 固定作者 common.py。** [源码](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py)：源码静态，读取 FaissNN／ApproximateFaissNN、MeanMapper／Aggregator、NearestNeighbourScorer、RescaleSegmentor；确认 IndexFlatL2、距离均值及 σ=4，未执行 FAISS／SciPy 热图。
- <a id="source-7"></a>**[cite:7] 固定作者 sampler.py。** [源码](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/sampler.py)：源码静态，完整读取两种 greedy、随机投影、Identity／RandomSampler；确认 q=128、a=10、全矩阵与近似均值初始化、索引回取原维度。未运行选择器，不将代码名视为理论证明。
- <a id="source-8"></a>**[cite:8] 作者 README 与 CVF 补充材料。** [固定 README](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/README.md)：首次抓取空返回，改用只读 HTTP 获取，正文／示例已读；说明仓库变更和硬件可能使成绩不同，且权重链接仍有未填项，未据此宣称取得模型。[CVPR 2022 supplemental PDF](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Roth_Towards_Total_Recall_CVPR_2022_supplemental.pdf)：T，读取 A 实现说明、C 分辨率／误分类分析及返回表格，未逐项复验表中数字。附录写 Python 3.7，固定 README 写 Python 3.8，版本差别保留。
- <a id="source-9"></a>**[cite:9] FAISS 官方 MetricType and distances。** [文档](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)：官方 API 语义已读，明确 METRIC_L2 返回平方欧氏，真实欧氏需 sqrt；这是移动文档于本次获取时的语义证据，不是作者运行环境版本锁。
- <a id="source-10"></a>**[cite:10] scikit-learn 官方 Random Projection。** [指南](https://scikit-learn.org/stable/modules/random_projection.html)：正文已读，说明有限点集保距、维度／畸变依赖、高斯矩阵 N(0,1/q) 和稀疏投影；JL 原始证明只作为其参考入口，未另逐页重证。它不是作者 sampler 的依赖调用证据，本卡条件近似推导单独标明。
- <a id="source-11"></a>**[cite:11] 作者输入与依赖。** [固定 mvtec.py](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/datasets/mvtec.py)、[固定 requirements.txt](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/requirements.txt)：只读 HTTP，核对 RGB、Resize／CenterCrop／ToTensor／Normalize 及依赖约束；未下载 MVTec、未安装作者环境。

## 有限验证记录与 G21 承接边界

本次只在此文件补 PC01，读取但不修改规范、领域总览、逐卡索引或覆盖台账。验证目标为独立编号、五层状态、答 5、公式引用、来源锚点、相对链接、围栏配对、有限代数及写入范围。Python 命令不可用，因此有限算术与文档检查采用 PowerShell／.NET；不为文档任务安装模型依赖，也不把这些检查描述成 PyTorch／FAISS 测试通过。

**已执行结果：**五层标题、单卡标题、围栏配对、11 个来源锚点、57 处本地文件／本页链接及其中 2 处跨文件锚点检查通过；有限数值表中的权重、平方差异、均值／插值不交换、覆盖半径、条件投影系数和字节数代入通过。未执行伪代码或任何图像检测流程。跨命令内存中的文件哈希基线未保留，故未完成全工作树前后哈希一致性证明；写入操作仅针对本文件，且观察到其他卡片存在并行变更，不将其归为本次修改或声称全工作树静止。

本卡承接 G21 要求的局部聚合／对齐、memory bank、coreset 与条件近似、NN、**原整图重加权公式**、热图、复杂度及实现差异，按台账“已有”的内容口径，**G21 的具体方法缺卡可销项（缺卡→已有）**。这只消除“没有该具体机制专卡”，不消除 L3／L4／L5 的实测与理论边界，不宣称本仓库已实现或复现 PatchCore。由于本次限定只写一文件，**台账行、索引计数和导航尚未同步，不能声称台账已经改成已有**；后续登记应保留 G21 历史 ID 及本卡五元状态，不机械升级到五层全核验。
