# FP01 FPFH 与 SAC-IA：局部描述和初始刚性对齐

[学习规范](../algorithm_learning_standard.md) · [三维总览 E06](../3d_evolution.md#36-e06fpfh-与-sac-ia为局部配准寻找初值) · [覆盖总账](../evolution_coverage.md) · [卡片索引](README.md) · [后续 ICP 基线](remaining_domains.md#g02-刚性-point-to-point-icp)

## 范围、版本与覆盖状态

核对日期：2026-11-05。唯一键 **`fpfh_sacia.md:FP01`**，承接 **G39／10-06／E06**；只有一张复合卡，含 FPFH 描述子、SAC-IA 初始对齐两个子算法，各自 L1–L5，共享五组逐层问答，不计成两卡或答10。本文不修改索引、总账、总览及其历史数量。

**任务边界。** FPFH 输入点云和逐点法向，输出局部直方图，不直接输出对应或位姿；SAC-IA 输入两份点云及各自逐点描述子，输出源到目标的候选刚体变换。FPFH 可以接别的对应后端，SAC-IA 也不在类型层强制只能用 FPFH。两者可串联，再经独立拒识和可选 ICP 精化；本仓库没有这条可运行点云管线。[cite:1](#source-1)[cite:3](#source-3)[cite:4](#source-4)

| 子算法 | L1 步骤 | L2 机制 | L3 工程权衡 | L4 数学推导 | L5 实现 | 实现／验证边界 |
|---|---|---|---|---|---|---|
| FP01 内的 FPFH | 部分（固定源码链已核） | 已核验（限定源码与官方教程） | 部分 | 部分（条件代数与有限检查） | 部分 | **未实现**；源码静态核对、参考伪代码、有限代数检查，不是 PCL 运行 |
| FP01 内的 SAC-IA | 部分（固定源码链已核） | 已核验（限定源码机制） | 部分 | 部分（刚体拟合与概率条件） | 部分 | **未实现**；无 C++ 编译、随机粗配准、真实点云或现场验收 |

**原文获取限制不可抹去。** 已联网读取 Rusu、Blodow、Beetz 的 ICRA 2009 原始出版页：题名、作者、DOI、摘要、引言开头和章节目录；会议为 2009-05-12 至 05-17，日本神户，Xplore 入库 2009-08-18 不是提出日。会议方法正文 PDF 本次未成功取得：作者旧站、大学镜像、CiteSeerX 等入口失败。因而原文深度是 **B＋A＋引言，不是 T**。2009 全文中的配对约定、权重具体度量、Huber 的残差语义、迭代／精化原句仍待逐式核验，不以读到 2016 课堂 review 或 2009 学位论文目录冒充会议全文。[cite:1](#source-1)[cite:10](#source-10)

**可复核实现基准。** PCL 固定 tag `pcl-1.14.1`，远端 tag 解析值 **`5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80`**；以下代码链接固定到该提交，不使用 master／latest 推断行为。默认单线程 `FPFHEstimation`、默认 L2 KD-tree、`SampleConsensusInitialAlignment` 的默认变换估计器是本文核对对象；不自动覆盖 OMP、GPU、自定义距离／损失或其他版本。代码和它自己的教程不是三个独立来源；代码特有行为标记 `[SINGLE-SOURCE：固定源码]`，论文与库的分歧不硬合并。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

**演进定位。** 先前 PFH 对每个查询邻域中点对作较完整连接，角特征统计昂贵；FPFH 改为中心星形 SPFH 与邻居 SPFH 聚合，减少重复计算，代价是丢掉部分点对和角度联合信息。初值未知时，ICP 的局部迭代会落到错误极小值；SAC-IA 在特征空间产生候选、在三维几何空间检验随机刚体假设。新增成本是法向／尺度选择、候选搜索和随机预算，遗留重复几何、低重叠、法向错误与拒识问题。这是“前端表示＋粗估计＋可选精化”的组合，不是 FPFH 取代 ICP。[cite:1](#source-1)[cite:8](#source-8)

<a id="fp01-fpfh"></a>
## 子算法一：FPFH 局部几何描述

### L1 步骤：法向、配对、SPFH、邻域聚合

**输入输出。** 输入搜索表面 `surface[M,3]`、同索引法向 `normals[M,3]`、查询点 `input[N,3]` 和可选查询索引，输出与所请求查询点顺序一致的 `features[N_query,33]` 及有效标记。最小复现先限定 `input=surface`、全部点参与，避免把关键点索引、搜索表面索引和缓存行号混用。法向估计是前置依赖，不是 FPFH 自动完成的内部步骤。[cite:3](#source-3)

1. 统一坐标单位、去除非有限坐标及无效法向，记录过滤前后点 ID。若尚无法向，可用半径 \(r_n\) 邻域的局部 PCA 平面拟合，取最小特征值对应单位向量；符号另定。单视图可向传感器视点翻转，多视融合不可简单把所有法向朝同一个原点翻转。[cite:9](#source-9)
2. 选描述邻域 \(r_f\) 或 kNN，冻结搜索模式和有效支持要求。官方教程建议描述尺度大于法向尺度，但不能从示例的厘米值推得跨物体通用参数。对于查询点邻居的 SPFH，也必须计算它们各自的邻域，不能只算法向或 SPFH 于查询关键点。[cite:3](#source-3)[cite:8](#source-8)
3. 对中心与每个非自身邻居计算点对特征。给定 \(\Delta=p_2-p_1\)、\(d=\|\Delta\|\)，先比较法向与连线的夹角再确定有序源点 s 和目标点 t，构造 Darboux 标架。此处“源／目标点”只是一个点云内的点对角色，不是 SAC-IA 两点云的源／目标角色。详细顺序及公式见 L4。[cite:2](#source-2)
4. 为每个点累计三段 SPFH，分别统计 \(\theta,\alpha,\phi\)。默认每段 11 bins，拼成 33 个 float；没有距离直方图、不是 11³ 联合 bins，也不是 33 字节。固定库对邻域列表大小 m 使用每对 `100/(m-1)`，跳过相同点索引，前提是列表含自身一次；不能直接对任意不含自身的列表照搬分母。[cite:3](#source-3)
5. 第二遍查询邻域，将搜索表面索引映射为 SPFH 缓存行，累计邻居的加权 SPFH，再逐段归一化。**固定 PCL 分支跳过所有零距离项，权重为 `1/dists[j]`，默认搜索返回 d²，因此是 1/d²；没有单独 `+SPFH(q)`，也没有显式 1/k。** 各非零段和为 100，三段总和 300，不是整体概率和为 1。[cite:3](#source-3)[cite:5](#source-5)
6. 全部查询点处理完即停止，没有迭代收敛。搜不到邻域时库写 NaN 并设置 `is_dense=false`；仅含自身时可能得到零段，而不自动视为有效几何。工程包装须把零支持、非有限输出等另行标无效，保留掩膜而非把零向量当可匹配描述。[cite:3](#source-3)

### L2 机制：不变性与信息损失

| 环节 | 为什么需要 | 删除／替换的后果与条件 |
|---|---|---|
| 单位法向与定向 | 用局部表面方向描述形状；法向符号影响角特征 | PCA 只确定一条轴；独立随机翻转通常改变 bins，刚体不变性不等于法向符号不变性 |
| 点对排序 | 减少输入顺序导致的同一无序点对编码差异 | 固定库用绝对内积对应的锐角比较；精确平局不交换，不能保证任意平局都与反向调用逐值一致 |
| Darboux 标架 | 在随点对一起旋转的局部坐标中测角 | 固定世界 xyz 分量会依赖姿态；漏掉叉积归一化会把长度或夹角幅度带进本应限定区间的量 |
| SPFH 中心星形连接 | 每点仅计算 O(k) 点对，避免邻域内部 O(k²) 全连接 | 完整 PFH 和 SPFH 不是等价缓存优化；丢掉邻居之间直接关系 |
| 三个边缘直方图 | 用线性维数保留各角度分布 | 不是统计意义上已经证明相互独立；相同边缘分布、不同联合分布可以不可区分 |
| 邻居 SPFH 重加权 | 引入二跳局部关系、补回部分邻点信息 | 不是重建全部 PFH；半径邻域的几何影响可延伸到 2r_f，重复点对也可能被重复计权 |

这些机制由实际代码和官方理论教程支持；“保留大部分判别能力”是文献场景下的经验表述，不是任意曲面都可唯一识别的定理。[cite:2](#source-2)[cite:3](#source-3)[cite:8](#source-8) 平面内部的多个点可能有几乎相同直方图；SAC-IA 增加预算也无法凭完全无区分度的局部统计恢复唯一对应。

### L3 工程权衡：先查支持与法向，再改直方图

| 参数／证据 | 量纲与选择依据 | 失效症状、代价和诊断 |
|---|---|---|
| 点间距、降采样尺度 h | 坐标长度；按真实密度统计，不假定点云均匀 | h 太大丢细结构，太小重复点和噪声主导；保留点间距分布和原始 ID |
| 法向半径 r_n | 长度；支持应够拟合平面，又不跨越要保留的边缘 | 小半径方向不稳，大半径跨面；记录邻点数、特征值间隙、法向分布与翻转率 |
| 特征半径 r_f／k | r_f 为长度，k 为数量，二者不同时混作一项 | 固定 r_f 保持物理尺度但点数变；固定 k 点数稳定但空间尺度变；二跳支持易跨遮挡边界 |
| bins B | 默认 11／角维，无量纲 | B 大更敏感且直方图稀疏；B 小丢区分度，改变 B 后不能继续写入仅容纳 33 floats 的输出类型 |
| 距离权重 | 论文式的 \(\omega\) 需定义；库默认 1/d² | 近重复点主导；`sqrt(dists)` 会改成另一模型，不能称严格等价优化 |
| 精度与单位 | PCL 此路径大量 float，归一化段和用 double 累计 | 大绝对坐标下相减精度损失；先合理平移原点并记录变换，转 double 不会恢复已量化信息 |

调参顺序：单位与 ID 对齐 → 非有限、重复点和法向定向 → 邻域点数／特征值间隙 → r_n、r_f 与密度差 → bins／权重变体 → SAC-IA 候选策略。先留存同一点的点对三元组、SPFH 和 FPFH 三段和，才能区分法向问题与后端随机性。原文实验参数尚未取得全文核对，本文不给冒充原配置的初值区间；官方 `r_f=0.05` 只是以米解释的 5 cm 示例。[cite:8](#source-8)[cite:9](#source-9)

### L4 数学推导：点对、分箱、权重与适用条件

**前置法向推导。** 邻域质心 \(\bar p\)，令 \(C=m^{-1}\sum_j(p_j-\bar p)(p_j-\bar p)^T\)。最小化 \(\sum_j[n^T(p_j-\bar p)]^2\)，约束 \(n^Tn=1\)，拉格朗日条件给出 \(Cn=\lambda n\)，最小目标对应最小特征值。近共线邻域的两个小特征值接近，法向轴本身就不稳定；相同点邻域不能估平面。朝视点规则为 \(n^T(v_p-p)>0\)，不满足时翻转，但它不解决多视点全局朝向一致性。[cite:9](#source-9)

**固定库的配对规则。** 先要求有限单位法向及 d>0，置 \(a_1=n_1^T\Delta/d\)、\(a_2=n_2^T\Delta/d\)。实际条件是 `acos(abs(a1)) > acos(abs(a2))`：成立则交换两法向并取 \(\Delta\leftarrow-\Delta\)，\(\phi=-a_2\)；否则保留并取 \(\phi=a_1\)。在有效域且不平局时，等价于把绝对内积较大的法向置为源法向。注意仅比较时取绝对值，\(\phi\) 自身保留符号。[cite:2](#source-2)

以下 **明确采用固定源码的叉积方向与单位化**，不能从另一教程的标架符号直接抄进实现：

\[
u=n_s,\qquad v=\frac{(p_t-p_s)\times u}{\|(p_t-p_s)\times u\|},\qquad w=u\times v,
\]
\[
\alpha=v^Tn_t\in[-1,1],\quad
\phi=u^T\frac{p_t-p_s}{d}\in[-1,1],\quad
\theta=\operatorname{atan2}(w^Tn_t,u^Tn_t)\in[-\pi,\pi].
\]

源码输出顺序是 `(f1,f2,f3,f4)=(theta,alpha,phi,d)`。d=0 或叉积范数=0 时底层 `pcl::computePairFeatures` 清零输出并返回 false；**FPFH 类的同名包装函数没有转发这个 bool，而是无条件返回 true**，所以其 SPFH 路径可能把失败零元组计入中央 bins。不能写成“该版本一定跳过所有退化点对”。生产防护必须在调用前检查或在独立实现中明确拒绝；这是有意偏离库异常路径的工程规则。[cite:2](#source-2)[cite:3](#source-3)

刚体不变性的条件证明：对 \(R\in SO(3)\)，点和平移一起变换、法向仅乘 R，则 \(\Delta'=R\Delta\)，点积与范数不变，且 \((Ra)\times(Rb)=R(a\times b)\)。排序不改变，标架随 R 旋转，三个标量不变。前提是对应邻域不变、法向跟随同一 R 且符号一致、无退化或分箱边界浮点跳变；反射的叉积多一个 det(R)，任意尺度／密度变化不在此证明内。

**SPFH 分箱。** 令 \(B=11\)、`clipB(z)=min(B-1,max(0,z))`：

\[
b_\theta=\operatorname{clipB}\left(\left\lfloor B\frac{\theta+\pi}{2\pi}\right\rfloor\right),\quad
b_\alpha=\operatorname{clipB}\left(\left\lfloor B\frac{\alpha+1}{2}\right\rfloor\right),\quad
b_\phi=\operatorname{clipB}\left(\left\lfloor B\frac{\phi+1}{2}\right\rfloor\right).
\]

对于含自身一次、其余点对全部有效的邻域 m，\(S_q^{(a)}[b]=100/(m-1)\sum_{j\ne q}\mathbf1[b_a(q,j)=b]\)，每段和为 100。上端点必须夹到 B−1；0 落在 bin 5。角度是弧度，\(\alpha,\phi\) 是内积，不是再取 acos 后以弧度分箱。atan2 的 ±π 两端在这份代码中分到不同端 bins，不额外做圆周软分箱。[cite:3](#source-3)

**理论聚合与固定库分开。** 官方理论教程及后续 review 转述的论文式为：

\[
F_q^{\rm theory}=S_q+\frac1k\sum_{j\in N(q)}\frac1{\omega_{qj}}S_j.
\]

教程把 \(\omega\) 说明为某度量空间中的距离，也允许另选权重。常见欧氏解释是 \(\omega=d\)，因而邻居项为 1/d；**2009 原文是否在本实验固定采用欧氏距离及如何最终归一化，本次未逐式核验**。不能把“教程允许一般权重”改写为“2009 必为 1/d²”。[cite:8](#source-8)[cite:10](#source-10)

默认 PCL 搜索令 \(\delta_{qj}=\|p_q-p_j\|^2\)，其实际计算为：

\[
A_q^{(a)}[b]=\sum_{j:\delta_{qj}>0}\frac{S_j^{(a)}[b]}{\delta_{qj}},\qquad
F_q^{(a)}[b]=\frac{100 A_q^{(a)}[b]}{\sum_c A_q^{(a)}[c]}.
\]

零和时保持该段零值。每个有效 SPFH 段均和为 100 时，还可化成 \(F_q^{(a)}=\sum_j\lambda_j S_j^{(a)}\)，\(\lambda_j=\delta_{qj}^{-1}/\sum_l\delta_{ql}^{-1}\)。因此只有邻居项中的公共 1/k 会被归一化消去，**缺失的中心项不能靠归一化补回**；中心点可能经邻居 SPFH 间接贡献，但不等于显式加中心 SPFH。[cite:3](#source-3)[cite:5](#source-5)

若所有几何长度同乘 s>0、邻域和法向不变，库中的全部邻居权重同乘 1/s²，段归一化抵消该因子；固定物理半径未同乘 s 时邻域已变，不再保证此性质。理论式若固定中心项为 1 而只让邻居 1/d 随单位变换，即使最终归一化也会改变中心／邻居相对贡献，所以“去掉距离直方图”不等于整条特征管线任意尺度不变。

**复杂度与误差。** 固定 B、平均邻居数 k、全点 SPFH 缓存下，点对与聚合算术 O(Nk)，相比 PFH 的 O(Nk²)；这不包括法向、建树、邻域搜索和子集收集的排序／集合成本。缓存约 O(NB)，缓存所有邻接表另需 O(Nk)。半径支持二跳至多 2r_f，法向自己的支持可再向外扩展；kNN 无统一 2r_f 上界。未提供通用描述误差或配准成功率界：硬分箱在边界不连续，近零叉积会放大扰动，边缘直方图还存在不可辨识性。

### L5 复现设计：明确分支和失败策略

下面是**未执行的整算法伪代码**；有效数据路径对照固定库，退化路径采用显式拒绝而非复制“失败零元组入箱”。`theory_euclidean` 是明示的教学解释，不声称已逐字复现 2009 原实现。

```text
FPFH_REFERENCE(P, normals, cfg):
    require finite(P) and finite(normals) and unit_normals(normals)
    require cfg.B == 11 and cfg.input_equals_surface
    neighborhoods = search_all(P, cfg.radius_or_k, return_squared_distances=True)
    SPFH, valid = allocate_zero(N, 3, 11), allocate_false(N)
    for q in point_order:
        J = neighborhoods[q].indices
        require q occurs exactly once in J and len(J) >= 2
        tuples = []
        for j in J excluding q:
            ok, theta, alpha, phi = ordered_darboux_pair(P[q], normals[q], P[j], normals[j])
            if not ok:
                mark q invalid
                break
            tuples.append(theta, alpha, phi)
        if len(tuples) != len(J) - 1:
            continue
        for tuple in tuples:
            add 100 / (len(J) - 1) to each of its three clamped bins
        valid[q] = True
    for q in point_order:
        J, delta = neighborhoods[q]
        if not valid[q] or any required SPFH invalid:
            output[q] = invalid
            continue
        if cfg.mode == pcl_1_14_1:
            A = sum(SPFH[j] / delta[j] for j with delta[j] > 0)
        else if cfg.mode == theory_euclidean:
            neighbors = J excluding q
            A = SPFH[q] + sum(SPFH[j] / sqrt(delta[j]) for j in neighbors) / len(neighbors)
        output[q] = normalize_each_nonzero_block_to_100_or_invalid(A)
    return output, validity_mask, point_id_map
```

**实现审查点。** 存储 `[point,theta/alpha/phi,bin]`，拼接顺序固定；法向与搜索表面等长并同 ID；稀疏查询需保存 `surface_index→spfh_row` 查表，不能拿查表后的行号回查坐标。先检查 NaN、零法向和单位长度，再做 acos／floor；只对小浮点越界夹紧到合法域，不能把严重无效输入全部夹成合法特征。零距离自点、不同 ID 的重复坐标和零叉积是三种不同边界。[cite:2](#source-2)[cite:3](#source-3)

优化前分测邻域搜索、点对计算、SPFH 缓存、加权和数据搬运；复用邻域／缓存、批量处理和并行归约需对照相同点集及排序。改变距离指数、删除中心项、改 bins、换邻域或删无效点均是契约变化；OpenMP 的浮点归约也不能预设逐 bit 等价。所有等价性测试都要看逐段误差、候选近邻排序及 SAC-IA 下游结果，不只看总和 300。

<a id="fp01-sacia"></a>
## 子算法二：SAC-IA 初始刚性对齐

### L1 步骤：间距采样、随机对应、拟合、全云评分

**输入输出与前提。** 源点 `P[N,3]` 与 `FP[N,D]`、目标点 `Q[M,3]` 与 `FQ[M,D]` 一一对应、有限且同一特征配置。输出 \(T_{Q\leftarrow P}\)，满足列向量 \(q\approx Rp+t\)，另有评分和独立验收状态。两点云要有足够重叠与可区分几何、相同物理尺度、刚性关系；没有这些条件，返回一个 4×4 矩阵不是成功证据。[cite:1](#source-1)[cite:4](#source-4)

1. 检查点／特征数量、有效性、样本数 s、候选数 K 和预算 I；建目标特征 KD-tree 与目标几何 KD-tree。源码检查点和特征数相等，但不完整检查全部有限性、几何退化和 K 的安全范围，包装层必须补上。[cite:4](#source-4)
2. 从源点云随机抽 s 个不同索引，逐个要求与已选点的**欧氏距离**不小于 `min_sample_distance`。距离相等可接受，因为实际拒绝条件是 `<`。连续失败达到 `3*N` 后，该版会把成员 `min_sample_distance_` 减半并继续；有效接收点后失败计数归零。这个放宽会影响本次及对象后续调用，不是原始间距的硬保证，也不是总抽样次数上限。[cite:4](#source-4)
3. 对每个被抽源点，在**特征空间**找目标 K 个最近描述子，再通过 `getRandomIndex(K)` 从列表中随机取一个。不是永远最近的一个、不是按相似度 softmax 抽样，也不是枚举 K^s 组合。不同源点可以抽到相同目标索引；默认路径没有互检、唯一匹配或边长预剔除。[cite:4](#source-4)
4. 对 s 对点估计源到目标的刚性 R、t。默认 `TransformationEstimationSVD` 构造参数 `use_umeyama=true`，实际走 `pcl::umeyama(src,tgt,false)`，最后 false 表示不估尺度；另一路显式 SVD 有反射修正。不能称此步为直接最小化全云 Huber 损失。[cite:6](#source-6)
5. 每个假设都作用于**全部输入源点**，在三维目标云找每个变换后点的最近邻，再累计误差。这里重新找的是几何最近邻，已经不是生成假设时的 s 个特征候选。默认误差详见 L4，为平方几何距离上的截断归一化和。[cite:4](#source-4)[cite:5](#source-5)
6. 首个无初值假设或严格低于当前最好误差的假设更新 `final_transformation_`。运行到 `max_iterations_` 预算结束后返回最好变换，不在此循环中自动调用 ICP，也不对所有内点再拟合。外部可对最佳候选另作覆盖／刚性／拒识检查，再决定是否精化。[cite:4](#source-4)

**停止与状态的源码细节。** 恒等 guess 在 `isApprox(Identity,0.01f)` 容差内时不预评分；非恒等 guess 先评分，循环计数从 1 开始，因此占用一轮预算。源码每轮是直接从原始源点估绝对变换，不是 `T_new*T_best` 的增量复合。该函数不使用继承的 `transformation_epsilon_`／`euclidean_fitness_epsilon_` 或概率置信提前停止。`converged_` 表示发生过上述更新：首个假设即使全云误差饱和也可置 true；非恒等 guess 很好而没有改善时反而可保持 false。I=0 不产生随机候选。包装层应拒绝无效预算，而不是把这些标志当业务成功率。[cite:4](#source-4)[cite:7](#source-7)

### L2 机制：特征提议与几何验证不可混同

| 环节 | 因果作用 | 删除／混用的反例 |
|---|---|---|
| 样本空间间距 | 降低样本扎堆，提高几何杠杆长度 | 很远的三点仍能共线；仅有 d_min 不保证刚体可观测 |
| K 个特征候选中随机选 | 允许最近特征错误而次近特征正确时仍有机会 | K=1 固定错误对应；K 增大又稀释每次选中真对应概率，不能只增 K 不看预算 |
| s 对点刚性拟合 | 将离散对应假设变成 SE(3) 候选 | 无尺度拟合不能修复米／毫米混用；重复目标点可造成秩亏 |
| 全云几何评分 | 检验假设是否解释更广的几何，而非只拟合被抽样点 | 三点恰好拟合不代表其余点重合；特征距离低也不是毫米误差低 |
| 鲁棒损失与最佳保留 | 控制远外点影响，从有限提议中选较好候选 | 低重叠时大量截断项一样为 1；重复物体可能产生低误差错误实例 |
| 独立拒识与局部精化 | 区分可用初值、错误解和未观测区域 | `hasConverged()` 不能代替重叠率、分布覆盖或姿态真值；ICP 不能保证救回错误对称解 |

以上是固定库机制的因果解释，不把 `SampleConsensusPrerejective` 的边长／内点比例检查倒填进 SAC-IA；同样不把 G02 的局部优化当作本子算法已实现。[cite:4](#source-4)[cite:6](#source-6)

### L3 工程权衡：量纲、随机预算与诊断次序

| 项目 | 固定库默认／量纲 | 调参依据与排障证据 |
|---|---|---|
| 样本数 s | 3，点数 | 至少三个非共线且对应正确的点才支持一般刚体；增大 s 加约束也降低整组正确率 |
| d_min | 0.0，坐标长度 | 相对点间距、物体尺度与可见区域设；记录实际最小边距及所有减半事件，不接受静默放宽 |
| K | 10，目标候选数量 | 必须 `1<=K<=目标有效特征数`；保存真对应候选召回、描述距离间隔和重复目标比例 |
| I | 1000，假设预算 | 是默认值非成功保证；记录随机种子、运行库、实际 s 点与 K 列表、最好分数轨迹及耗时分位 |
| 截断参数 τ | 构造默认 100.0；实际分支与平方距离比较 | 若物理距离门为 d_c，则此路径设 τ=d_c²；单位改变 s 倍，τ 应同乘 s² |
| 验收门 | 业务另定，不能用默认 100 代替 | 几何内点率用明确 d_c，报告重叠支持空间分布、旋转／平移误差和错误接受率 |

默认数值全部来自固定构造函数，非 2009 原文配置，也非推荐参数。[cite:4](#source-4) 调参先后为：源／目标变换方向及尺度 → 点特征 ID、FPFH 有效性和候选覆盖 → 样本共线／目标重复 → 评分平方距离与损失 → K、s、d_min、I → 验收门。无真值时能报告几何自洽，不可直接报告真实位姿精度；只对成功样本报 RMSE 会隐去粗配准失败。

**对象复用隐患。** `computeTransformation` 只在 `error_functor_` 为空时创建默认 `TruncatedError(corr_dist_threshold_)`，`computeErrorMetric(cloud,float)` 的第二参数实际上未使用。第一次运行后仅改 `setMaxCorrespondenceDistance`，不一定更新已构造损失对象中的阈值；应明确重建算法对象或设置新的损失对象并回归验证。这与 d_min 被自动减半一样，是状态污染而非论文数学变化。[cite:4](#source-4)[cite:7](#source-7)

### L4 数学推导：刚体最小二乘、评分与有限搜索

**固定对应的刚体拟合。** 给定正确或假设的对应 \((p_i,q_i)\)，求：

\[
\min_{R,t}\sum_{i=1}^{s}\|Rp_i+t-q_i\|^2,\qquad R^TR=I,\quad\det R=1.
\]

对 t 求导得 \(t=\bar q-R\bar p\)。令 \(x_i=p_i-\bar p,y_i=q_i-\bar q\)、\(H=\sum_i x_i y_i^T\)，目标化为常数减去 \(2\operatorname{tr}(RH)\)。设 \(H=U\Sigma V^T\)，则：

\[
R=V\operatorname{diag}(1,1,\det(VU^T))U^T,\qquad t=\bar q-R\bar p.
\]

反射修正确保 SO(3)，但不修复错误对应；别把另一种 \(H=YX^T\) 定义下的 U、V 顺序混进来。该式对应源码显式 SVD 支路的代数，默认 Umeyama 无尺度路径解决相同固定对应问题，不声称两个浮点实现逐 bit 相等。[cite:6](#source-6)

**可辨识与退化。** s=1 不能定旋转，s=2 或全部共线留下绕连线旋转自由度。三个非共线对应的中心化矩阵通常秩为 2，仍可唯一确定 proper 刚体变换；因此不能以 `sigma3=0` 一票否决所有三点／平面解，应检查第二个奇异值相对第一个及空间跨度。近共线时姿态噪声会放大；目标索引重复、几何几乎重合、对称重复结构和不真实刚性关系分别是不同失败来源。SVD 返回矩阵不等于可观测性检查通过。

**论文报告的 Huber 与库默认评分。** 2016 review 把 2009 §IV 的损失转述为 Huber：

\[
\rho_{t_e}(e)=\begin{cases}\frac12e^2,&|e|\le t_e,\\
t_e(|e|-\frac12t_e),&|e|>t_e.\end{cases}
\]

这是“原文机制的二手转述”，不是本次读到的会议式号。会议原文中 e 的具体定义、支持集和停止原句尚未取得，保留 `[INSUFFICIENT DATA：原文逐式核验]`；若采用通常的几何残差 \(e=\|Rp+t-q\|\)，Huber 小残差段为 d²／2、远处线性增长。这是明确写出的解释条件，不能拿来覆盖库事实。[cite:10](#source-10)

固定库中先取 \(j^*(i)=\arg\min_j\|Rp_i+t-q_j\|^2\)，再把 \(\delta_i=\|Rp_i+t-q_{j^*(i)}\|^2\) **不经开方**传入损失。默认：

\[
E_{\rm PCL}(T)=\sum_{i=1}^{N}\min(\delta_i/\tau,1),\qquad \tau>0.
\]

这是 N 个源点的有向累计量，非均值、非内点计数、非描述距离、非双向 Chamfer，也非 `getFitnessScore()`。后者在固定基类实现中对 `nn_dists[0] <= max_range` 的点取平均平方距离，`max_range` 也直接与平方距离比较；零入选返回 double 最大值。调用者务必区分内部候选排名与外部报告指标。[cite:4](#source-4)[cite:5](#source-5)[cite:7](#source-7)

即便显式安装该类 `HuberPenalty`，它仍收到 \(e=\delta=d^2\)，小残差段变成 \(d^4/2\)，阈值是平方长度，损失单位是长度四次方；不等于把原本以长度残差定义的 Huber 简单替换进去。要 Huber(d)，需自定义函数先 `sqrt(delta)`，这属于明确的实现改动。[cite:4](#source-4)

**抽样概率的限定推导。** 令 q 表示“一整轮得到足够正确、非退化、可用的对应”的概率。独立同分布轮次下 I 次全失败概率 \((1-q)^I\)，希望至少一次成功概率不低于 p，则：

\[
I\ge\left\lceil\frac{\log(1-p)}{\log(1-q)}\right\rceil,\quad 0<p,q<1.
\]

若额外假设每个源样本处在重叠可用区的概率为 o，K 列表包含真对应的概率为 c、仅一个正确目标且近似均匀抽取，再粗略令 \(q\approx(oc/K)^s\)。这不是可直接代入的实测量：间距约束令采样相关，候选互相相关，重复几何和退化降低 q，自动放宽还改变分布。只有“抽到正确假设”概率，不保证评分最终选中它；固定 PCL 循环没有使用此公式更新停止次数。[cite:4](#source-4)

忽略预处理，令特征搜索代价 \(C_f(M,D,K)\)、几何近邻代价 \(C_g(M)\)，外循环成本约 \(O(I[C_{sample}+sC_f+s+NC_g])\)，3×3 SVD 为常数阶。典型树查询可较快，但不能为 33 维搜索无条件承诺 O(log M)，最坏可接近遍历。源码采样 while 无硬总次数界；工程参考必须加入预算才能有确定终止契约。未提供全局收敛、普遍位姿误差或现场成功率保证。

### L5 复现设计：有限预算、退化拒绝与可追溯日志

下面是**有界工程参考伪代码，未实现**，不是固定源码逐行抄写：显式拒绝共线／重复目标、检查实际候选数量、设采样尝试上限、初值总是作为可选候选评分。若要逐行为复刻库，必须另开兼容模式保留 L1 的 guess、标志及自动减半语义，不能悄悄混在一起。

```text
SACIA_REFERENCE(P, FP, Q, FQ, cfg):
    require finite_and_aligned(P, FP) and finite_and_aligned(Q, FQ)
    require 3 <= cfg.s <= len(P) and 1 <= cfg.K <= len(FQ)
    require cfg.I >= 1 and cfg.tau > 0 and cfg.max_sample_attempts >= cfg.s
    feature_tree, geometry_tree = build_target_trees(FQ, Q)
    best = score_optional_valid_guess_or_none(cfg.guess)
    for trial in range(cfg.I):
        S = draw_unique_spaced_source_samples(P, cfg.s, cfg.d_min, cfg.max_sample_attempts)
        if S unavailable:
            record sample_budget_failure
            continue
        C = []
        for i in S:
            J = feature_tree.nearestK(FP[i], cfg.K)
            if len(J) != cfg.K:
                fail invalid_feature_search
            C.append(J[draw_index(len(J), cfg.rng)])
        if duplicate(C) or near_collinear(P[S]) or near_collinear(Q[C]):
            record degenerate_hypothesis
            continue
        T = rigid_fit_no_scale(P[S], Q[C])
        if not finite_SE3(T):
            continue
        delta = geometry_tree.nearest_squared_distances(transform(T, P))
        E = sum(min(value / cfg.tau, 1) for value in delta)
        if best absent or E < best.E:
            best = save(T, E, S, C, trial)
    if best absent:
        return failure_no_valid_hypothesis
    accepted = independent_overlap_coverage_and_pose_checks(best, P, Q, cfg)
    return best.T, best.E, accepted, stop_reason_budget, diagnostics
```

固定库的 `findSimilarFeatures` 不使用 `nearestKSearch` 返回数量，而仍从 K 抽索引；KD-tree 却会把 K 截到实际有效点数，故 K 大于有效目标数有越界风险。若滤除无效描述子，应同时滤掉相应坐标并保留 ID 映射，不可只缩短特征数组。随机数为 C `rand()` 路径，同 seed 不保证跨运行库序列一致，最好记录实际样本与候选选择。[cite:4](#source-4)[cite:5](#source-5)

坐标统一用列向量 `T[4,4]`，齐次最后行为 `[0,0,0,1]`；旋转验证 `R^T R≈I, det(R)≈1`，normal 只乘 R 不加 t。失败应返回显式状态而非恒等矩阵加“成功”。自定义 loss 接口名写明 `squared_distance`；阈值、种子、PCL／Eigen／FLANN、降采样、法向定向、特征 bins／权重、输入哈希和变换方向都应序列化。

性能优化优先剖析：特征可预计算／缓存候选列表，几何评分可批量查询；只在原始非负累计评分超过当前最优且无需完整诊断时可提前拒绝本假设，这不等于提前停止整个算法。并行 trial 会改变随机抽取顺序／并列解选择，必须预先固定每轮输入。更换评分点子集、近似近邻、预剔除或内点再拟合均可能改变解，不能标成纯等价加速。

## 原文、理论教程与固定库差异清单

| 主题 | 原文／理论侧及证据深度 | PCL 1.14.1 已核对行为 | 不得混写的结论 |
|---|---|---|---|
| 描述与初始对齐 | 2009 摘要、引言明确两个职责，B/A＋引言 | 两个独立模板类 | FPFH 直接输出位姿 |
| 标架与点对排序 | 2009 方法正文未取得；官方 PFH 教程写 u×单位连线，与源码不同 | 绝对夹角比较；连线×u 后再次归一化 | 教程符号与库逐值等价；绝对值比较即法向符号不变 |
| FPFH 聚合 | 官方理论式显式 S_q＋(1/k)ΣS_j/ω；原文度量配置待核 | 没有独立中心项，默认 1/d²，分段归一至100 | 只差常数因子；把 d² 开方属等价优化 |
| 描述布局 | 教程有“四个特征／33-byte”的不一致文字 | 三角维×11 bins＝33 floats；d 不入直方图 | 跟抄文字写成44维或33字节 |
| SAC-IA 损失 | review 转述 Huber；原文 e 语义未核 | 默认 TruncatedError(delta)，可选 HuberPenalty(delta) | 默认 Huber(d)；Huber 与截断误差相同 |
| 采样及候选 | review 转述间距与相似特征列表；原文具体停止待核 | 间距失败自动减半，K 候选随机选一，目标可重复 | 硬间距、互为近邻、唯一对应或边长预剔除都自动存在 |
| 停止与精化 | 原始摘要说明为局部优化提供初值 | 固定预算、guess 特例、无 ICP／全内点重拟合 | 收敛 flag 等于全局正确或物理验收 |

差异来源：[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)[cite:8](#source-8)[cite:10](#source-10)。明确存在的“教程／库差异”不直接标成已完成的“2009 全文／库差异考证”。

## 有限测试与验证范围

以下检查使用精确给定的数学输入作为**单元测试夹具**，不是合成实验数据、论文结果或点云成功率。本文件内嵌检查器只依赖 JavaScript 标准运算；读取该代码块后可直接在 Node.js 中执行，不需要新建脚本文件。它覆盖两个子算法的关键代数／分支，但没有邻域 KD-tree、PCA、真实 SVD 求解、随机抽样器或 PCL 二进制调用。

| 组 | 有限判据 | 不覆盖 |
|---|---|---|
| F1 配对与标架 | 非平局正反配对、指定旋转平移不变、零距离／平行拒绝 | 全体平局规则、真实法向估计及库的失败零元组入箱复现 |
| F2 分箱 | 三维上下端点与零点落箱、33维长度、SPFH段和100 | 所有浮点边界、全部邻域与NaN路径 |
| F3 权重 | 长度1与2给出1/d权重比2:1、1/d²权重比4:1；中心项使结果改变 | 真实数据的描述判别能力 |
| S1 几何方向 | 非共线固定对应的已知R、t残差零、正交且det=1；共线面积零 | 数值SVD求解、噪声下可观测性 |
| S2 评分量纲 | 截断边界、Huber(d)与Huber(d²)不同、阈值同平方尺度变换 | PCL对象生命周期与真实近邻 |
| S3 候选与终止 | 固定候选索引非总取top1；间距边界、预算耗尽和饱和评分仍更新首假设 | rand分布、真实采样while、PCL运行 |
| S4 概率公式 | 独立同分布假设下最小整数轮数达到给定置信度 | SAC-IA真实独立性、实际成功率 |

```javascript
const assert = (x, name) => { if (!x) throw new Error(name); };
const close = (a, b) => Math.abs(a - b) <= 1e-10;
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = a => Math.sqrt(dot(a, a));
const eq = (a, b) => a.length === b.length && a.every((v, i) => close(v, b[i]));
const rot = p => [-p[1], p[0], p[2]];
function pair(p1, n1, p2, n2) {
    let dp = sub(p2, p1), d = norm(dp);
    if (d === 0) return null;
    const a1 = dot(n1, dp) / d, a2 = dot(n2, dp) / d;
    let phi = a1;
    if (Math.acos(Math.abs(a1)) > Math.acos(Math.abs(a2))) {
        [n1, n2] = [n2, n1];
        dp = mul(dp, -1);
        phi = -a2;
    }
    let v = cross(dp, n1);
    if (norm(v) === 0) return null;
    v = mul(v, 1 / norm(v));
    const w = cross(n1, v);
    return [Math.atan2(dot(w, n2), dot(n1, n2)), dot(v, n2), phi];
}
const p = [0,0,0], q = [1,0,0], n = [0,0,1], m = [0.6,0,0.8];
const f = pair(p,n,q,m), t = [2,-3,4];
assert(eq(f, [Math.atan2(0.6,0.8),0,-0.6]), 'F1 expected');
assert(eq(f,pair(q,m,p,n)), 'F1 order');
assert(eq(f,pair(add(rot(p),t),rot(n),add(rot(q),t),rot(m))), 'F1 rigid');
assert(pair(p,n,p,m) === null && pair(p,[1,0,0],q,[1,0,0]) === null, 'F1 degenerate');
const bin = (x, lo, hi) => Math.max(0,Math.min(10,Math.floor(11*(x-lo)/(hi-lo))));
assert(eq([-Math.PI,0,Math.PI].map(x => bin(x,-Math.PI,Math.PI)),[0,5,10]), 'F2 theta');
assert(eq([-1,0,1].map(x => bin(x,-1,1)),[0,5,10]), 'F2 dot');
const spfh = Array(33).fill(0);
for (const v of [f,f]) for (let a=0;a<3;a++) spfh[a*11+bin(v[a],a===0?-Math.PI:-1,a===0?Math.PI:1)] += 50;
assert(spfh.length===33 && [0,1,2].every(a => close(spfh.slice(a*11,a*11+11).reduce((x,y)=>x+y,0),100)), 'F2 sum');
const normalize = a => mul(a,100/a.reduce((s,v)=>s+v,0));
const inverseD = normalize([100/1,100/2,0]);
const inverseD2 = normalize([100/1,100/4,0]);
const theory = normalize(add([0,0,100],mul([100/1,100/2,0],0.5)));
assert(eq(inverseD,[200/3,100/3,0]) && eq(inverseD2,[80,20,0]), 'F3 weights');
assert(theory[2]>0 && !eq(theory,inverseD2), 'F3 center');
assert(eq(inverseD2,normalize([100/9,100/36,0])), 'F3 scale');
const P = [[0,0,0],[1,0,0],[0,1,0]], Q = P.map(x=>add(rot(x),t));
const cp = mul(P.reduce(add,[0,0,0]),1/3), cq = mul(Q.reduce(add,[0,0,0]),1/3);
assert(eq(sub(cq,rot(cp)),t) && P.every((x,i)=>eq(add(rot(x),t),Q[i])), 'S1 direction');
const axes = [[1,0,0],[0,1,0],[0,0,1]].map(rot);
assert(close(dot(axes[0],cross(axes[1],axes[2])),1) && close(dot(axes[0],axes[1]),0), 'S1 SO3');
assert(norm(cross(sub(P[1],P[0]),sub(P[2],P[0])))>0 && norm(cross([1,0,0],[2,0,0]))===0, 'S1 rank');
const trunc = (delta,tau) => Math.min(delta/tau,1);
const huber = (e,h) => e<=h ? 0.5*e*e : h*(Math.abs(e)-0.5*h);
assert(eq([0,1,4,9].map(x=>trunc(x,4)),[0,0.25,1,1]), 'S2 trunc');
assert(close(huber(0.5,1),0.125) && close(huber(0.25,1),0.03125), 'S2 squared');
assert(close(trunc(9,4),trunc(81,36)) && close(trunc(1,4),trunc(9,36)), 'S2 scale');
assert([7,4,2][1]===4 && !(2<2) && 1.9<2, 'S3 candidate spacing');
let best = Infinity, updates = 0, trials = 0;
for (const E of [3,3,3]) { trials++; if (E<best) { best=E; updates++; } }
assert(trials===3 && updates===1 && best===3, 'S3 saturated first');
const probability=0.9, qGood=0.25, I=Math.ceil(Math.log(1-probability)/Math.log(1-qGood));
assert(I===9 && 1-(1-qGood)**I>=probability && 1-(1-qGood)**(I-1)<probability, 'S4 probability');
console.log('FP01 finite checks: 7 groups passed; no PCL or end-to-end registration');
```

**本次执行结果。** 2026-11-05，Node.js v24.14.0，从本 Markdown 提取并执行上述代码块，七组有限检查通过；同时检查十段层级正文、共享五问、来源锚点和本地文件链接。该结果只验证表内有限代数与简化分支，不包含 PCL 编译、完整 FPFH／SAC-IA 或真实配准；两个子算法的实现状态继续为未实现。

**待实施的算法／真实数据测试。** 固定输入与法向逐点对照 PCL SPFH／FPFH、搜索返回 d²、稀疏 surface 映射；测试重复点、近零叉积、NaN、空云、仅自身邻域及候选不足。SAC-IA 应实际验证同种子候选轨迹、非恒等 guess、I=0／1、共线和重复目标、d_min 减半及损失阈值对象复用；加入无重叠、平面、对称／重复物体、密度差、遮挡和真实姿态真值。指标分开报有效特征率、候选召回、粗配准接受／拒绝及误接受、旋转角和平移误差、独立内点 RMSE、覆盖、峰值内存和延迟分位。当前无上述实测。

## 共享五问：每问同时追问两个子算法

| 层级／问题 | 参考答案与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：从两云到初值如何走完，何时停止？ | FPFH：单位定向法向→有序点对→三段SPFH→指定权重聚合，遍历完停止；SAC-IA：源点间距采样→K特征候选随机取一→无尺度刚体拟合→全源云几何评分→预算内取最好，后续独立拒识 | “FPFH直接给R/t”“算出三个对应就停止”“默认自动ICP” | 两子算法 L1；[cite:3](#source-3)[cite:4](#source-4) |
| L2：为什么既要法向和SPFH加权，又要随机候选与几何验证？ | FPFH用局部标架保姿态不变，聚合二跳关系但失去部分联合信息；SAC-IA允许top1不正确而候选中仍有真对应，再由几何解释全云 | “33维可唯一识别所有点”“点间距大就不共线”“特征距离即配准误差”；平面与重复件同时破坏两步 | 两子算法 L2／L4；[cite:2](#source-2)[cite:8](#source-8)[cite:4](#source-4) |
| L3：换点密度／单位后失败，先增迭代或放宽门吗？ | 先核FPFH法向方向、邻域支持、r_n/r_f与ID，再核SAC-IA候选覆盖和退化；d_min按长度变，库τ按长度平方变；最后调K/s/I并记录自动放宽 | “所有参数乘1000”“默认100一定是100米”“加I修复无真对应”；对象残留阈值和d_min也会导致不一致 | 两子算法 L3；[cite:4](#source-4)[cite:5](#source-5)[cite:9](#source-9) |
| L4：推导权重和刚体拟合，再解释停止概率的保证？ | 写有中心的理论式与无显式中心的库1/d²段归一式，解释1/k能消去而中心不能；由t=均值差、H的SVD及det修正求R；独立轮次仅有1−(1−q)^I抽中概率，不保证选对 | “归一化消除1/d与1/d²差异”“三点秩2必不可解”“Huber(delta)就是Huber(d)”“达到I必全局最优” | 两子算法 L4；[cite:3](#source-3)[cite:4](#source-4)[cite:6](#source-6) |
| L5：数学正确但库对不上，如何定位并声明复现状态？ | FPFH查叉积方向、角顺序、端点、m−1、自点、平方距离及包装bool；SAC-IA查K实际数量、源目标方向、默认Umeyama、全云损失、guess／flag和状态复用；有限检查不等于PCL或真实点云测试 | “读过源码就是已实现”“七组小检查证明工程成功”“二手review等同读到2009全文”；保留双方L1部分及未实现 | 两子算法 L5／有限测试／来源状态；[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)[cite:10](#source-10) |

## G39 复合承接与销项边界

| 核验维度 | 本卡依据 | 可登记结论 |
|---|---|---|
| 身份与范围 | FP01 一卡、两个子算法各五层、共享答5，映射10-06 | 不膨胀卡数／节点数，不以ICP前后置链接代替两子机制 |
| FPFH机制 | 法向、排序、Darboux、三段分箱、权重与d²差异、退化、伪代码 | 固定PCL实现路径和理论解释已具体展开 |
| SAC-IA机制 | 间距／放宽、随机候选、刚体拟合、全云评分、停止／flag、退化 | 具体初始对齐已展开，不混成泛称RANSAC或ICP |
| 学习与验证 | 十段L1–L5、五问覆盖两者、七组有限检查及待实施测试 | 文档机制承接成立；两子算法仍未实现，L3/L4/L5不升满 |
| 2009原文要求 | 仅B/A＋引言；方法全文获取失败，Huber与部分理论是分层引用 | **不支持“已读2009全文并完成原文逐式对照”的无条件销项**；严格G39验收仍保留原文证据补核项 |

若总账“已有”仅要求具体方法卡、五层正文与答5，则本卡提供复合机制承接依据；若本轮销项包含用户要求的2009原文方法核验，则只能作**有保留承接**，不能伪报完整关闭。本文不代改任何台账状态。

## 来源与获取状态

<a id="source-1"></a>
**[cite:1] 原始会议出版记录，P0，B＋A＋引言。** Radu Bogdan Rusu、Nico Blodow、Michael Beetz，*Fast Point Feature Histograms (FPFH) for 3D Registration*，ICRA 2009，DOI `10.1109/ROBOT.2009.5152473`。[IEEE出版页](https://xplorestaging.ieee.org/document/5152473)。本次可见§II PFH、§III FPFH、§IV SAC-IA目录，但未见其方法正文；不把目录当正文。全文入口 [作者旧地址](https://ias.in.tum.de/people/rusu/publications/own/Rusu09ICRA.pdf)、[大学镜像](https://lars.mec.ua.pt/public/LAR%20Projects/BinPicking/2016_RodrigoSalgueiro/LIB/FPFH/05152473.pdf)、[CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3710&rep=rep1&type=pdf) 均未取得可读会议全文。摘要的优点描述不外推为实测性能。

<a id="source-2"></a>
**[cite:2] PCL点对内核，P0，T（源码）。** [features/src/pfh.cpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/features/src/pfh.cpp)，`pcl::computePairFeatures`：绝对夹角排序、Delta×normal、单位化、theta/alpha/phi/d顺序、零距离／零叉积返回false。只讨论此函数，非同文件RGB扩展。

<a id="source-3"></a>
**[cite:3] PCL FPFH，P0，T（源码）。** [fpfh.h](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/features/include/pcl/features/fpfh.h)、[impl/fpfh.hpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/features/include/pcl/features/impl/fpfh.hpp)。核对 `computePairFeatures` 包装、`computePointSPFHSignature`、`weightPointSPFHSignature`、`computeSPFHSignatures`、`computeFeature`；默认11/11/11、无显式中心项、1/dists、分段100与索引重映射。

<a id="source-4"></a>
**[cite:4] PCL SAC-IA，P0，T（源码）。** [ia_ransac.h](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/ia_ransac.h)、[impl/ia_ransac.hpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/impl/ia_ransac.hpp)。核对构造参数、`getRandomIndex`、`selectSamples`、`findSimilarFeatures`、两损失类、`computeErrorMetric`、`computeTransformation`；库自称实现论文§IV不表示与原文逐式相同。

<a id="source-5"></a>
**[cite:5] PCL搜索距离契约，P0，T（源码与接口）。** [kdtree_flann.h](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/kdtree/include/pcl/kdtree/kdtree_flann.h)、[impl/kdtree_flann.hpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/kdtree/include/pcl/kdtree/impl/kdtree_flann.hpp)。默认 `L2_Simple<float>`，接口注明输出平方距离；radiusSearch内部radius²，nearestKSearch按有效点数缩短结果。仅限默认距离配置。

<a id="source-6"></a>
**[cite:6] PCL刚体求解，P0，T（源码）。** [transformation_estimation_svd.h](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/transformation_estimation_svd.h)、[impl/transformation_estimation_svd.hpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/impl/transformation_estimation_svd.hpp)。默认Umeyama且无尺度、备用SVD的H方向和反射修正；没有验证Eigen各版本二进制等价。

<a id="source-7"></a>
**[cite:7] PCL配准基类，P0，T（相关源码）。** [registration.h](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/registration.h)、[impl/registration.hpp](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/registration/include/pcl/registration/impl/registration.hpp)。核对阈值setter、align和getFitnessScore；基类文档中通用ICP术语不能覆盖SAC-IA实际循环。

<a id="source-8"></a>
**[cite:8] PCL固定理论教程，P0，T（教程，不是会议原文）。** [fpfh_estimation.rst](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/doc/tutorials/content/fpfh_estimation.rst)、[pfh_estimation.rst](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/doc/tutorials/content/pfh_estimation.rst)。支持理论聚合式、二跳支持、边缘直方图和复杂度；保留与源码的叉积／单位化和布局文字差异，不引用教程速度倍数作仓库测量。

<a id="source-9"></a>
**[cite:9] PCL法向教程，P0，T（教程）。** [normal_estimation.rst](https://github.com/PointCloudLibrary/pcl/blob/5f608cfb5397fe848c7d61c4ae5f5b1ab760ba80/doc/tutorials/content/normal_estimation.rst)。局部协方差、法向符号、视点定向、多视点限制和邻域尺度；本文未把法向估计算作第三张卡。

<a id="source-10"></a>
**[cite:10] 后续课堂review，P2，T（二手转述）。** Daniel Adler，2016-04-21，[Review: Fast Point Feature Histograms (FPFH) for 3D Registration](https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses%3A446%3A2016%3A446-2016-06%3Adadler6_fpfh-review.pdf)。读取其§4及§4.1的SPFH聚合、间距、相似候选与Huber转述；不是2009作者原文、不是独立性能验证，残差语义与原文参数不据此升级。原文证据缺口是本卡明确保留项。
