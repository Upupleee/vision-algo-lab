# Structured Forests：结构化边界预测专卡

[学习规范](../algorithm_learning_standard.md) · [测量双线总览](../measurement_evolution.md) · [覆盖与历史任务](../evolution_coverage.md) · [普通 RF 对照](learning_components.md#c03-随机森林-rf)

## 范围、版本与覆盖口径

核验日期：**2026-11-05**。本文件只有一张独立卡 `structured_forests.md:SF01`，承接 **G14／05-08 Structured Forests，2013**。目标是解释具体的结构化标签训练与边界输出，不把普通随机森林分类器、像素独立投票、HED 或本地 EdgeNet 改名充当本算法。唯一卡片锚点为 [SF01](#sf01)。

| 版本对象 | 本卡锁定内容 | 不可混用的边界 |
|---|---|---|
| 原始方法 | Dollár、Zitnick，*Structured Forests for Fast Edge Detection*，ICCV 2013，1841–1848；CVF 开放版 8 页 | 以第 3–4 节与第 4 页脚注为机制依据，不拿后续论文替代原文 [cite:1](#sf-source-1) |
| 后续扩展 | *Fast Edge Detection Using Structured Forests*，arXiv:1406.5549 **v1，2014-06-20**，第 4.2 节 | **Edge Sharpening 是后续新增**；2013 有多尺度和评测前 NMS，但无该 sharpening 步骤 [cite:2](#sf-source-2) |
| 作者 edges 代码 | `pdollar/edges@94260b5d0fc068202598312e01a37604972dcc9e`；核心文件头标 Version 3.01 | 是后续代码快照，不是 2013 发布包。关闭 sharpen 也不能宣称逐位恢复 V1.0 [cite:3](#sf-source-3) |
| 作者基础工具箱 | `pdollar/toolbox@e87332637bbe8e8b92dd487c87567d9628404523`；所读文件头标 Version 3.24 | 锁定 `forestTrain`、`forestFindThr`、`pca` 静态解释入口，不代表完整依赖已经安装／验证 [cite:7](#sf-source-7) |

作者发布历史列 V1.0（2013-11-12）对应 ICCV；V2.0（2014-06-20）对应 arXiv，并明确 added sharpening；V3.0 加入其他功能但 edge detection code unchanged，V3.01（2014-09-08）有小修正。仓库当前 README 标题、文件头、发布历史和不可变提交是四种信息，不能合并成“2013 V3.01”。**未取得 V1.0 历史发布包及其权重哈希**。[cite:3](#sf-source-3)

| 独立字段 | SF01 状态与证据边界 |
|---|---|
| 演进定位 | 传统 ML／结构化输出森林；从中心像素分类转向局部分割结构预测 |
| L1 步骤 | **已核验**：2013 训练—推理主链及后续扩展的静态对照 |
| L2 机制 | **已核验**：置换不变标签、节点局部离散化、叶代表与重叠融合 |
| L3 工程权衡 | **部分**：参数有原文／固定源码依据，未作业务定参或性能剖析 |
| L4 数学推导 | **已核验（限定代数）**：编码、代理损失、代表元与聚合推导；不是泛化或定位精度证明 |
| L5 实现 | **部分**：固定源码对应、伪代码及有限数值检查；未编译／调用作者 MATLAB/MEX |
| 面试自测 | **答5**，逐层问题、答案、条件、易错点和证据入口见后文 |
| 本仓库实现 | **未实现**：无 SF 训练器、模型权重、结构化推理接口或评测链；检索 Python 源码未见 `edgesTrain/edgesDetect/StructuredEdge` 实现 |
| 验证范围 | 文档引用与链接、源码静态核对、参数算术和条件恒等式；无整图、BSDS500、工业图、现场计量实测 |

“已核验”仅表示指定资料／代数层面的核验。论文、代码均来自同一作者体系，不算独立复现；方法主张标 **[SINGLE-SOURCE]**。不提供跨论文速度排名，不将论文 F-score 当作本仓库或物理尺寸精度。

<a id="sf01"></a>

## SF01 Structured Forests for Fast Edge Detection

### 任务契约与演进定位

**训练输入**为图像及对应人工分割标注；每个样本是较大的图像 patch 与同中心、较小的**区域编号 mask**。区域编号只在该 mask 内有意义，不是“前景／背景”语义类别。多位标注者的分割应作为各自有效的结构样本，不对区域编号作算术平均。只有细边二值图时，不能无条件反推出原区域划分，必须另行解决监督格式问题。[cite:1](#sf-source-1)[cite:4](#sf-source-4)

**推理输入**以 RGB 图像为本卡主契约，输出同尺寸连续边缘响应 `E[H,W]`；可选 NMS 变细，仍不是毫米坐标、闭合实例分割或已校准概率。2013 亦研究 RGBD；不能把 RGB 权重直接当作深度模型。作者接口可额外返回粗法向 `O`、每位置每树叶索引和局部分割，后两项不是全图一致实例 ID。[cite:1](#sf-source-1)[cite:5](#sf-source-5)

前序瓶颈是中心像素分类丢失直线、平行线、T／Y 结等邻域相关性；SF 改变的是**监督结构与输出粒度**，而不是仅增加树数。相对预先固定 sketch-token 类别，SF 在每个节点重新构造适合该节点标签分布的离散代理类别。代价是分割标注、标签编码、PCA、训练存储与域偏移；局部结构先验不保证全图拓扑或亚像素准确性。[cite:1](#sf-source-1)

### L1 步骤：训练与推理的完整链

**训练链**：[cite:1](#sf-source-1)[cite:4](#sf-source-4)[cite:7](#sf-source-7)

1. 按原图／工件／批次划分训练、验证、测试；再采样 patch。原文 RGB 基准使用 `32×32` 输入支持和 `16×16` 分割输出。后续代码分别从靠近标注边界及远离边界处取正负 patch；这里的“正负”是采样策略，不是替代结构 mask 的最终监督。
2. 提取颜色和梯度通道，构造单通道查值与平滑通道的两位置差值特征。每棵树选取特征子集，节点再采样候选特征。
3. 对到达节点的全部分割 mask，**共用该节点随机选中的像素对**，编码每对是否属于同一区域；中心化、去常量列、PCA，再量化成少量代理类别。原文用 `m=256` 个像素对、至多 5 个 PCA 维度、`k=2` 类。
4. 固定本节点代理类别后，比较候选输入特征的阈值 split，以最大信息增益选择分支。子节点用自己的结构样本重新编码／离散化，不继承全局固定 token 类别。
5. 在纯代理类、样本不足、深度或增益门限触发时停止。选节点中最具代表性的原始分割作为叶输出，并生成其边界图；这不是把代理类编号存成边缘概率。
6. 独立训练多棵树，保存特征／阈值／孩子数组、代表分割、稀疏边界及完整参数。后续代码有“小区域清理”步骤，不能悄悄当作原始代表元定义。

**推理链**：一次全图计算通道；按 stride 取位置并走树，每次只做输入特征比较；叶给出整个局部分割／边图；将不同树和不同位置的边图放回同一坐标累加；归一化、裁剪和按实现规定平滑。可选多尺度为 `1/2、1、2` 倍各自检测，响应重采样回原尺寸后平均。2013 评测前使用 NMS；**推理不运行标签 PCA，也不需要测试图分割真值**。[cite:1](#sf-source-1)[cite:5](#sf-source-5)

**2014 扩展插入点**是“叶分割 → 根据该图像 patch 修正分割 → 生成边图 → 重叠融合”，而不是在已平均的边图上做反差增强。所有位置和指定尺度结束后推理停止；sharpen 是有限的局部修正，不是必须收敛的全图优化。[cite:2](#sf-source-2)[cite:6](#sf-source-6)

### L2 机制：不能用普通 RF 替换的环节

**结构标签编码**：令 `y(a)` 是局部像素 a 的区域 ID，编码 `z_ab=1[y(a)=y(b)]`。对任意区域编号的一一重命名，所有同区关系不变；直接对 ID 做欧氏距离却会因编号改变。完整像素对编码可描述分区的等价关系；随机只选部分像素对会丢信息，两种不同分区可能碰撞。用二值边图做距离也容易因一像素偏移导致大量不重叠，因此作者采用同区关系作为训练相似性。[cite:1](#sf-source-1)

**节点局部离散化**：PCA 是对本节点的**输出标签编码**去噪／降维，不是输入颜色特征的 PCA。二分类时按第一主成分投影的正负分组；特征空间中的最佳阈值只需把这些代理类分开。删掉这一步而改成中心像素边／非边 Gini，学到的是普通分类森林，不是 SF；改用全训练集固定聚类则改变了节点自适应机制。[cite:1](#sf-source-1)[cite:4](#sf-source-4)

**输入特征与 split**：RGB 使用 3 个 LUV 通道、2 尺度归一化梯度幅值、每尺度 4 个方向通道，共 13 通道。通道查值提供局部颜色／梯度证据，大尺度平滑后的自相似差值提供区域内部与两侧的对比。二者都是输入 X；像素对同区编码属于监督 Y，不能把两种“像素对”混成同一特征。去掉差值特征不等于完全失去结构监督，但改变原特征契约，必须作为消融重新训练。[cite:1](#sf-source-1)[cite:4](#sf-source-4)

**叶代表与边图**：选择编码空间离均值最近的训练分割，保证输出仍是一个合法局部分区，而非逐像素平均后不存在的 ID。原文脚注用的是**平方欧氏 medoid**，不是一般未平方距离 medoid；固定代码在 PCA 投影前选择代表。单叶默认不能创造训练中没有的分割结构；多个边图叠加能产生新软响应，但不是合成出一致的全图区域标号。[cite:1](#sf-source-1)[cite:4](#sf-source-4)

**重叠融合**：一次叶预测对整个 `16×16` 邻域投票，省去每像素大量走树。原文用棋盘式交替树组降低相邻输出相关性，不能据此称投票独立；移位叶边界平均仍可能变宽。后续 sharpening 用局部图像颜色对齐各分割，NMS 则在最终响应的法向上抑制非峰，两者作用位置与机制不同，不能互称别名。[cite:1](#sf-source-1)[cite:2](#sf-source-2)[cite:6](#sf-source-6)

### L3 工程权衡：参数、尺度与排障

| 参数／对象 | 可追溯起点及单位 | 权衡与诊断 |
|---|---|---|
| 输入／标签宽度 | 原文 `32／16`，原图像素 | 大支持增加上下文但混入邻边；换尺寸需训练及检查 MEX 容量，不是任意运行时旋钮 |
| 通道与平滑 | 13 通道；原文 triangle 半径 2／8 像素；`shrink=2` | 固定代码把平滑半径换算到通道网格；不可把原图半径原样套到缩小网格 [cite:4](#sf-source-4) |
| 标签编码 | `nSamples=256` 像素对、`nClasses=2`、PCA 至多 5 维 | 更多对更细但耗时／内存上升；增加 k 不保证提升真实边界质量 |
| 特征抽样 | 后续代码每树 `fracFtrs=1/4`，节点约 `sqrt(F)` | 这是两级抽样；不是 RF 教材中每树必须 bootstrap 的保证 [cite:7](#sf-source-7) |
| 停止与样本 | 后续代码 `nPos=nNeg=5e5`、`minChild=8`、`maxDepth=64` | 属于后续默认值；原文给每树一百万 patch，不证明每棵树实际能采满或无限深更好 |
| 森林与步长 | 原文每位置 T=1–4，训练 2T 树；stride=2 像素；后续默认训练8、评估4 | 减少 stride 增加重叠与成本；后续 wrapper 仅令 stride≥shrink，安全复现还需验证网格整除关系 |
| sharpen | 2013 不含；后续默认2，0关闭 | 是预存边带支持深度及修正强度，不能直接等同“两轮同步 Lloyd”；只能在模型支持范围内降低 [cite:6](#sf-source-6) |
| multiscale／NMS | 后续默认均关闭 | 报告参数时必须显式记录；双倍分辨率有面积成本，不是简单三倍运行时间 |

表中起点来自原文第 3–4 节和固定 `edgesTrain.m`，不是已在本地挑选的最优值。[cite:1](#sf-source-1)[cite:4](#sf-source-4)

**调参顺序**：先验证 RGB／浮点范围与通道布局、模型版本、标注和裁剪坐标；再检查训练／验证是否按原图分组及正负 patch 覆盖；随后对比未 NMS 响应、叶分割和边图；最后才调森林容量、步长、多尺度及 sharpening。阈值只在验证集冻结，不能用测试图挑阈值后报成 ODS。

| 症状 | 应保留的证据 | 优先处理及回退 |
|---|---|---|
| 输出整图错位／固定半窗偏移 | 原图、padding、patch 中心、叶支持与裁剪坐标 | 先统一 MATLAB 列主序和零／一基索引，不能用整体平移补偿掩盖原因 |
| 平坦图大量边缘 | 输入动态范围、颜色转换、叶索引、背景标签 | 查 RGB/BGR 和训练域；结构先验也可能在域外产生假边 |
| 双边粘连或边带太厚 | 融合前各叶边图、sharpen=0 与开启后的响应 | 判断预测错位还是不可分近邻边；检查是否仅 NMS 后“看起来更细” |
| sharpen 把边吸到纹理上 | 边带内颜色、区域均值、细结构与反光 | 关闭 sharpen 作对照；颜色贴合不是语义边界正确的充分条件 |
| 分数异常放大／边缘暗圈 | 累加值、覆盖计数、归一化因子、NMS 边缘衰减 | 区分理想平均和作者经验倍率；保留 ROI 有效支持，不盲裁分数到[0,1] |
| 边界 F-score 好、尺寸偏 | 原灰度剖面、标注厚度、法向偏差、标定链 | SF 仅作候选；接显式连续定位、几何拟合和不确定度验收，不能宣称亚像素能力 |

### L4 数学推导：代理损失、代表元与融合

#### 结构空间与近似距离

令输出宽度为 g，像素数 `q=g²`，全像素对集合为 P，大小 `D=q(q−1)/2`。编码

\[
\Pi(y)_{ab}=\mathbf1[y(a)=y(b)],\qquad
\|\Pi(y_i)-\Pi(y_j)\|_2^2
=\sum_{(a,b)\in P}\mathbf1[\text{两分区对此像素对的同区判断不同}].
\]

因此平方距离等于配对关系的 Hamming 不一致数，不是像素 ID 差或边缘定位距离。给完整标签 ID 施加任意双射 π，有 `Π(π∘y)=Π(y)`。完整关系矩阵确定分区到编号置换；只观察 m 个关系时不保证可辨识。[cite:1](#sf-source-1)

从 D 对中均匀无放回抽 m 对，本卡可推出 `E[(D/m)·||Πφ(y_i)−Πφ(y_j)||²]=||Π(y_i)−Π(y_j)||²`，因为每对入样概率为 `m/D`。这仅是**抽样距离的期望恒等式**；作者没有按它保证每节点距离准确。再用数据相关 PCA、符号量化后，不保证相似性排序或真实结构损失不变。[cite:1](#sf-source-1)

对节点中心化矩阵 Z 的前 r 个主方向 U，`a_i=(z_i−z̄)U`。PCA 最小化给定秩的平方重建残差；二类量化是 `c_i=1+1[a_i1<0]`，多类可由前 `floor(log2 k)` 维的正交象限编码。主方向整体反号通常只交换类别名字，不改变 Gini；重特征值、零投影和浮点近零时仍需明确规则。当前代码零投影归非负侧、删除空类别；全常量编码直接返回一个类和第一个代表。[cite:4](#sf-source-4)[cite:7](#sf-source-7)

#### split 损失的来源与边界

给定**本节点固定**的代理类别 c，节点分支 `bθ(x)=1[fθ(x)<τ]`，其中 f 是单点通道值或两位置差值。以经验类比例 p 为常数分类分布，最小平均负对数似然为熵：

\[
\min_{p\in\Delta^{k-1}}-\frac1n\sum_i\log p_{c_i}
=-\sum_c\hat p_c\log\hat p_c=H(S).
\]

拉格朗日约束 `Σp=1` 得 `p_c=n_c/n`，零频项按 `0 log 0=0`。允许左右分支各有分布时，最小损失变为 `nL/n·H(SL)+nR/n·H(SR)`；故最大化信息增益等于在候选 split 集内最小化此**代理类条件损失**，不是端到端像素 BCE、IoU 或结构化 SVM 间隔损失。[cite:1](#sf-source-1)

固定作者默认用 Gini。对代理 one-hot 向量 e_c，以概率向量 p 做平方损失：

\[
\min_p\frac1n\sum_i\|e_{c_i}-p\|_2^2
=1-\sum_c\hat p_c^2=G(S),\qquad
\Delta=G(S)-\frac{n_L}{n}G(S_L)-\frac{n_R}{n}G(S_R).
\]

均值最小化平方损失给 `p=p̂`，展开平方项即得 Gini。这是本卡对代理准则的代数解释，不表示作者在训练另一套 Brier 网络。阈值只用输入特征，标签编码只用来评价训练 split；子节点重建类别后，整个树不再是对一套固定全局类别目标的优化。贪心只在抽样候选中求优，不保证全树最优或自然图泛化。[cite:4](#sf-source-4)[cite:7](#sf-source-7)

#### 叶代表为什么能在线性时间选择

对节点的 n 个采样编码，限制代表必须来自训练集合，原文脚注目标是

\[
k^*=\arg\min_k\sum_i\|z_k-z_i\|_2^2,
\quad
\sum_i\|z_k-z_i\|_2^2
=n\|z_k-\bar z\|_2^2+\sum_i\|z_i-\bar z\|_2^2.
\]

第二项与 k 无关，因此找离均值最近的编码即可，耗时 `O(nm)` 而非成对遍历的 `O(n²m)`。输出是对应的**原分割 y_k**，不是均值编码 z̄，也不是 PCA 类编号。平方号不可省；对 `Σ||z_k−z_i||` 的一般 medoid，上式不成立。[cite:1](#sf-source-1)

#### 分割到边图及重叠融合

教学上明确一种局部四邻域边界算子：`B(y)(u)=1[存在有效四邻居 v，使 y(u)≠y(v)]`；越过 patch 外界不额外造边。它标记分界两侧像素，天然可能是双像素宽。**这不是作者所有分支逐位相同的算子**：后续未 sharpen 的预存边图由 `gradientMag(single(seg))>.01` 构造；sharpen 后则用四邻域不同标签检测，并只检查扩展边带。必须冻结边图厚度／稀疏位置规则再谈等价性。[cite:4](#sf-source-4)[cite:6](#sf-source-6)

对位置 p、树 t 的局部预测 B_pt，局部到全图的平移为 T_p，理想覆盖平均为

\[
E(u)=\frac{\sum_{p,t}w_{pt}(u)B_{pt}(T_p^{-1}u)}{\sum_{p,t}w_{pt}(u)},
\]

其中 w 只在有效支持内为非负；分母为零时应标无覆盖而非除零。若 B 为二值且分母有效，可推出 `0≤E≤1`；不能反推概率已经校准。稠密内部、g 能被步长 s 整除时，每像素覆盖约 `(g/s)²T` 次，边缘／ROI／非整除位置另计。大量重叠通常相关，不可按独立 Bernoulli 样本量计算置信区间。[cite:1](#sf-source-1)

**固定作者代码并非显式除逐像素覆盖图**：先累加稀疏边，裁剪后乘 `s²/(g²T)`，再按 sharpen=0／1／≥2 分别乘 **2／1.8／1.66**，最后 `convTri(E,1)`。这些经验归一化倍率不能从理想平均式推出，也不能视作概率界或“无影响的常数”；NMS 还有自身倍率和图像边缘衰减。[cite:5](#sf-source-5)[cite:6](#sf-source-6)

#### sharpening 的局部目标而非全局保证

2014 论文先由预测分割估计区域颜色均值 `μ_s`，再将像素 u 分给四邻域允许的区域中颜色距离最小者：

\[
y_{new}(u)=\arg\min_{s\in A(u)}\|I(u)-\mu_s\|_2^2.
\]

固定均值与候选集时，这是离散局部最小化。若候选含原标签，可对该点保证颜色项不增；实际邻域约束、更新顺序及候选集合变化，不允许直接宣称整个过程全局收敛、拓扑保持或语义误差下降。2014 原文明示 sharpening 不保证提升结果。[cite:2](#sf-source-2)

固定 MEX 用平滑 RGB、每隔一行一列采样计算均值、稀疏边带内**原地顺序更新**，均值不会在每个像素修改后重算；`sharpen` 决定使用多少预存边带，不是直接执行该次数的全图同步迭代。颜色均值采样可能遇到某小区域未被采中的退化，应在移植时明确拒绝／回退策略并单独标为与作者不同，不能以任意 epsilon 冒充已定义均值。[cite:5](#sf-source-5)[cite:6](#sf-source-6)

#### 复杂度与保证范围

节点 n 个样本、m 个像素对、F 个候选输入特征：编码 `O(nm)`；稠密精确 PCA 可估为 `O(nm·min(n,m))`；逐特征排序阈值扫描 `O(Fn log n)` 加累计类统计；代表选择 `O(nm)`。不要因只保留 5 维就断言 PCA 求解成本必为 `O(5nm)`。全图 A 个采样位置、T 棵评估树、平均深度 d：走树约 `O(ATd)`；密集边图回填上界 `O(ATg²)`，稀疏边回填随实际边数减少。另计通道提取、padding、多尺度、sharpen 和 NMS，节点总存储与每叶结构存储分别核算。

没有通用真实边界误差界、尺寸误差界或训练收敛到全局最优的保证；取最优代理 split 不保证结构损失下降。输出格点边缘分数，不提供原始 SF 自带亚像素定位的承诺。

### L5 实现：源码对应、伪代码与测试

#### 不可变源码对应与数据布局

以下路径均为外部作者源码，不是本仓库新增实现；完整不可变链接集中在来源表。

| 算法环节 | 作者入口 | 移植必须保留的契约 |
|---|---|---|
| 图像与监督采样 | `edgesTrain.m/trainTree` | 特征 `single[N,F]`；标签先 `uint8[g,g,N]` 后 cell；每 mask 局部连续编号；按标注者采样 |
| 通道 | `edgesChns.m` | RGB 转 LUV；两尺度梯度；普通／自相似通道平滑不同；输出通道网格不是原图网格 |
| 编码／代表／PCA | `edgesTrain.m/discretize` | 无放回抽取共享像素对；去常量列；代表在 PCA 前选；全常量分支；kmeans 是可选而非默认 |
| 节点训练 | `forestTrain.m/treeTrain`、`forestFindThr.cpp` | 先节点离散化，再采候选；排序后扫描相邻不同值，中点阈值；差值需≥`1e-6`，gain>`1e-10`，然后检查孩子人数 |
| 森林合并 | `edgesTrain.m/mergeTrees` | 代表分割转零基 ID；清理≤5像素区域后构边；`eBins:uint16`、`eBnds:uint32` 为稀疏位置与累积边界 |
| 走树与投票 | `private/edgesDetectMex.cpp` | `fids` 零基，模型 `child` 转换后 `<thr` 走 `child−1`、否则 `child`；`child=0` 为叶；棋盘式树组 |
| 最终输出 | `edgesDetect.m`、`private/edgesNmsMex.cpp` | 对称 padding、四倍数对齐、半输出窗裁剪、经验倍率、平滑、法向、可选 NMS；分项验收 |

源码依据：[cite:4](#sf-source-4)[cite:5](#sf-source-5)[cite:6](#sf-source-6)[cite:7](#sf-source-7)。当前工具箱是**先选总体最佳阈值，再检查 minChild**，不一定在所有满足 minChild 的阈值中重新寻优；提前过滤阈值的重写会改变树。深度检查实际是 `depth>maxDepth`，不要在逐位比较中擅自替换为 `>=`。

MATLAB/MEX 为列主序，零基像素地址 `row+col*height+channel*height*width`；Python 默认 C-order 展平不同。`segs[g,g,node,tree]` 和每预测输出的 `segs[g,g,h/s,w/s,T]` 不等于 BCHW。阈值相等走另一支，float32 的特征舍入可改变叶；双精度累计微检查不能证明 MEX 逐位一致。固定 MEX 还有 `4096` 像素缓存及区域／通道固定数组，不得把 g、区域数或通道数无限扩展。[cite:6](#sf-source-6)

#### 机制伪代码

以下为本卡独立整理的**解释性伪代码，未作为 SF 程序运行**。它覆盖核心依赖；`FrozenBoundary`、`Finalize` 必须选择论文概念版或固定作者代码版，不允许默默切换。

```text
TrainNode(samples, depth, frozen_options):
    pairs = sample_shared_distinct_pixel_pairs(g, m)
    Z = same_region_codes(samples.segmentations, pairs)
    Zc = center_and_remove_constant_columns(Z)
    representative = nearest_observed_segmentation_to_mean(Zc)
    classes = one_class_if_empty_else_pca_sign_quantization(Zc, k)
    if pure(classes) or too_few(samples) or depth_stop(depth):
        return Leaf(representative, FrozenBoundary(representative))
    feature_ids = sample_input_feature_candidates()
    theta, gain = best_sorted_threshold(samples.features, classes, feature_ids)
    left, right = partition_by_feature_less_than_threshold(samples, theta)
    if insufficient_gain(gain) or child_too_small(left, right):
        return Leaf(representative, FrozenBoundary(representative))
    return Node(theta, TrainNode(left, depth+1), TrainNode(right, depth+1))

Detect(image, forest, frozen_options):
    padded, transform = pad_and_record_coordinates(image)
    channels = extract_regular_and_similarity_channels(padded)
    accumulator = empty_edge_accumulator()
    for position in valid_scan_grid:
        for tree in checkerboard_selected_trees(position):
            leaf = descend_using_input_features_only(tree, channels, position)
            segmentation = leaf.representative
            if version_is_post_2013 and sharpen_enabled:
                segmentation = local_color_snap(segmentation, image_patch(position))
            boundary = FrozenBoundary(segmentation)
            scatter_add_with_coordinate_transform(accumulator, boundary, position)
    response = Finalize(accumulator, transform, frozen_options)
    return optional_nms(response)
```

多尺度在 `Detect` 外对三个比例分别调用，重采样到原尺寸再平均，最后统一 NMS。移植作者代码时还要实现模型合并的小区域清理、稀疏边带、确切归一化和裁剪；这些不是本伪代码已经交付的功能。

#### 数值核对与可执行微检查

先核对**来自原文参数的算术**，不生成／冒充实验数据：`g=16 → q=256 → D=32640`；`256/32640≈0.0078431373`，说明每节点只抽约0.7843%的像素对，并非保留同百分比的结构信息。RGB 查值特征 `32²×13/2²=3328`，自相似 `C(25,2)×13=3900`，合计 **7228**；后续每树四分之一为1807维，节点 `round(sqrt(1807))=43` 候选。原文 stride=2、T=4 的理想内部覆盖 **256** 次，平均系数 `1/256=0.00390625`；后续代码 sharpen=0／1／2 的裁剪前后缩放系数分别为 **0.0078125／0.00703125／0.006484375**，尚未计最终平滑。这些只是公式与源码常数核算。[cite:1](#sf-source-1)[cite:4](#sf-source-4)[cite:5](#sf-source-5)

下面的数组只是**数学单元测试夹具，不是图像样本、训练数据、模拟评测或论文实验结果**。检验编号置换、PCA 类别交换、Gini、代表元恒等式、覆盖与空分母；不实现训练器、特征提取、sharpen 或 NMS。要求 Python＋NumPy，可直接提取本 Markdown 中唯一 Python 代码块执行，无须另存文件。

```python
import math
import numpy as np

g, shrink, trees, channels = 16, 2, 4, 13
assert math.comb(g * g, 2) == 32640
assert 32 * 32 * channels // shrink**2 == 3328
assert math.comb(25, 2) * channels == 3900
assert 3328 + 3900 == 7228
assert round(math.sqrt(7228 / 4)) == 43
coverage = (g // shrink)**2 * trees
assert coverage == 256
np.testing.assert_allclose(np.array([2, 1.8, 1.66]) / coverage,
                           [0.0078125, 0.00703125, 0.006484375])

def encode(labels, pairs):
    values = np.asarray(labels).reshape(-1, order="F")
    return np.array([values[a] == values[b] for a, b in pairs], dtype=float)

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
a = np.array([[0, 0], [1, 1]])
b = np.array([[0, 1], [0, 1]])
np.testing.assert_array_equal(encode(a, pairs), encode(7 + 5 * a, pairs))
assert not np.array_equal(encode(a, pairs), encode(b, pairs))
assert encode(a, [(0, 3)]).tolist() == encode(b, [(0, 3)]).tolist()
Z = np.stack([encode(a, pairs), encode(a, pairs), encode(b, pairs)])
centered = Z - Z.mean(axis=0)
direct = ((Z[:, None, :] - Z[None, :, :])**2).sum(axis=(1, 2))
identity = len(Z) * (centered**2).sum(axis=1) + (centered**2).sum()
np.testing.assert_allclose(direct, identity)
assert int(np.argmin(direct)) == int(np.argmin((centered**2).sum(axis=1)))
_, _, vt = np.linalg.svd(centered, full_matrices=False)
scores = centered @ vt[0]
classes = scores < 0
assert np.all(scores != 0)
np.testing.assert_array_equal((-scores < 0), ~classes)

def gini(ids):
    _, counts = np.unique(ids, return_counts=True)
    return 1 - np.sum((counts / len(ids))**2)

parent = gini(classes)
gain = parent - classes.mean() * gini(classes[classes])
gain -= (~classes).mean() * gini(classes[~classes])
assert math.isclose(parent, 4 / 9) and math.isclose(gain, 4 / 9)
np.testing.assert_allclose(gini(classes), gini(~classes))
assert not np.any(np.ones((3, 6)) - np.ones((3, 6)).mean(axis=0))

def covered_average(total, count):
    if np.any(count <= 0):
        raise ValueError("uncovered output")
    return total / count

count = np.full((2, 2), coverage)
np.testing.assert_allclose(covered_average(count, count), 1)
np.testing.assert_allclose(covered_average(np.zeros_like(count), count), 0)
try:
    covered_average(np.array([0]), np.array([0]))
except ValueError:
    pass
else:
    raise AssertionError("zero coverage must be rejected")
print("SF01 finite algebra checks passed")
```

**本次结果**：该微检查代码已执行通过；只是有限算术／代数检查，不是作者代码测试通过，不证明训练／整图推理正确。没有 BSDS 指标或现场结果可报告。

#### 待实施测试与等价性门槛

| 测试层 | 输入与断言／指标 | 当前状态 |
|---|---|---|
| 标签／离散化 | ID 置换不改编码；常量编码停叶；共享对而非每样本独立抽对；近零 PCA、重特征值、重复样本 | 前两类有限恒等式已检查；完整节点实现未测 |
| 阈值／树 | 用冻结真实训练节点数据比较候选排序、中点、相等值、gain、minChild、深度和叶索引 | 设计，未取得训练节点快照，未运行 |
| 代表分割／边图 | 同一叶比对代表、≤5像素区域清理、梯度边图和四邻域图差异、稀疏编码解码 | 设计；不能拿教学四邻域算子替代作者分支 |
| 坐标／融合 | 非方形与奇数尺寸实图、靠边 ROI、不同 stride、棋盘树组；逐像素比较累加、裁剪、倍率和最终图 | 设计，未运行整图；排除未覆盖位置 |
| sharpen | 固定真实叶与图像 patch，比对采样均值、空均值组、边带、更新顺序、边图变化 | 设计，未运行；不能以颜色损失下降代替边界真值评价 |
| 数值与性能 | MATLAB/MEX 标量基线对比移植版；先叶索引一致再比响应容差，分别记录通道、走树、scatter、sharpen、NMS 时间／内存 | 未编译、无优化或加速数字；并行浮点累加须固定容差 |
| 任务验收 | BSDS 分组与标准容差匹配，ODS/OIS/AP；工业域另记漏检、法向定位偏差、近邻边混淆、尺寸和拒识率 | 未训练／评测；自然图指标与计量指标分报 |

优化优先复用全图通道、预计算查表、存稀疏边、走树与聚合分阶段；任何改变通道计算、PCA 近似、阈值、边图厚度、模型清理或更新顺序的优化都先按以上层级比对。论文原始分割标签和权重均未下载，本次不声称可复现论文训练数字。

### 五问五答：贯穿 L1–L5

| 层级／问题 | 参考答案与成立条件 | 易错答案／证据入口 |
|---|---|---|
| L1：给一幅 RGB 图，SF 怎么得到整图边缘，PCA 在哪执行？ | 通道→按位置走树→叶局部分割／边图→坐标回填重叠→归一化→可选多尺度与 NMS。PCA 在训练节点处理标签；2014 sharpen 在融合前改叶分割 | “测试每个像素做 PCA 分类”“2013 默认 sharpen=2”；第 L1 节、来源1／2／5 |
| L2：为什么不是把16×16的区域ID展平直接训练普通RF？ | ID 没有跨 mask 数值意义；同区关系对编号置换不变，节点离散化只生成局部代理类，叶仍返回整个分区 | “ID 差越小结构越近”“预测中心边／非边就等价”；第 L2 节、来源1／4 |
| L3：边缘变厚，先加树还是开sharpen？ | 先看融合前叶边错位、坐标／倍率、近邻边和域偏移；再对照sharpen=0及开启，分别看原响应与NMS结果。更多相关树不解决系统错位 | “树越多定位越准”“sharpen 保证贴真边”；第 L3 节、来源2／5／6 |
| L4：为何代理Gini可算，代表元为何等于离均值最近？ | one-hot平方损失最小值为 `1−Σp²`；固定代理类用加权子节点Gini选split。展开平方距离得代表元恒等式；只对平方欧氏目标成立 | “SF 最小化边缘BCE”“任意medoid都等于最近均值”“局部代理增益保证结构全局最优”；第 L4 节、来源1脚注／7 |
| L5：公式相同而移植结果不同，查哪些地方？ | 先比通道与列主序、ID／child基数、float32阈值和叶，再比清理后边图、scatter、裁剪倍率、sharpen更新顺序、NMS。空均值和零覆盖独立处理；未运行就不能报精度 | “统一加epsilon”“复制伪代码即复现”“普通sklearn RF权重可替代”；第 L5 节、来源4–7 |

## 原文获取层级与来源记录

获取层级按“书目／摘要／全文片段／全文／源码”分别记录；取得 PDF 二进制不自动等于逐页公式复核。以下均为本次实际联网获取，PDF 在内存解析，不在仓库新增下载文件。独立来源不足不借镜像数量补足；**具体机制为 [SINGLE-SOURCE] 作者体系**，代数推导另有明确假设，未作独立性能确认。

<a id="sf-source-1"></a>

**[cite:1] ICCV 2013 原始论文。** [CVF 书目与摘要](https://openaccess.thecvf.com/content_iccv_2013/html/Dollar_Structured_Forests_for_2013_ICCV_paper.html)已读；[CVF 原 PDF](https://openaccess.thecvf.com/content_iccv_2013/papers/Dollar_Structured_Forests_for_2013_ICCV_paper.pdf) **8页全文文本取得并阅读**，重点核对1843页训练、1844页§3.1–3.3及平方medoid脚注、1845页§4特征／映射／融合、1846页评测。最初网页解析在§3截断，随后内存PDF解析补齐，不以截断结果冒称全文。SHA-256：`b9dc0608a60ea4f2697fa72435f0259eae4667d2a9a9a9e35ff9a637306e4b2c`。[作者PDF镜像](https://pdollar.github.io/files/papers/DollarICCV13edges.pdf)仅取得开头片段，不另算独立证据。未取2013补充材料，未访问IEEE权威排版全文作逐式差异比较。

<a id="sf-source-2"></a>

**[cite:2] 2014 后续版本。** [arXiv v1 HTML](https://arxiv.org/html/1406.5549v1)确认题名、版本日期与目录，返回正文前段；[v1 PDF](https://arxiv.org/pdf/1406.5549v1)取得13页文件，**正文重点片段：第5页§4.1–4.2已读**，不是13页逐页审读声明。SHA-256：`9542251048fbd5294369517a81fd55d24ec751dc34c39a6b4423bde0f9f04c08`。只用于sharpening扩展的归属和机制，不把其结果灌入2013配置。

<a id="sf-source-3"></a>

**[cite:3] 作者代码与历史。** [作者仓库主页](https://github.com/pdollar/edges) **README／安装／历史全文已读**；`git ls-remote`取得HEAD为`94260b5d0fc068202598312e01a37604972dcc9e`，后续源码请求均固定此提交：[不可变树](https://github.com/pdollar/edges/tree/94260b5d0fc068202598312e01a37604972dcc9e)。GitHub commits API本次限流，未从该接口取得提交日期，因此不编造快照发布日期；未取得tag，文件头V3.01不是已验证发布tag。README要求MATLAB、Image Processing Toolbox和Piotr工具箱≥3.26；未安装／编译这些依赖。

<a id="sf-source-4"></a>

**[cite:4] 作者训练与特征源码，完整文件文本已读。** [edgesTrain.m](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/edgesTrain.m)，含`trainTree`、`discretize`、`stComputeSimFtrs`、`mergeTrees`；[edgesChns.m](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/edgesChns.m)。读取入口为同提交raw文件。早期尝试`private/edgesTrainTree.m`返回空，实际训练函数嵌于`edgesTrain.m`，不引用不存在的文件。

<a id="sf-source-5"></a>

**[cite:5] 作者推理wrapper，源码全文已读。** [edgesDetect.m](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/edgesDetect.m)。核对输入输出、padding、multiscale、sharpen前颜色平滑、经验归一化、裁剪、粗法向及NMS调用。接口注释叫probability map，本卡不据此声称已校准概率。

<a id="sf-source-6"></a>

**[cite:6] 作者推理MEX，源码全文已读。** [edgesDetectMex.cpp](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/private/edgesDetectMex.cpp)、[edgesNmsMex.cpp](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/private/edgesNmsMex.cpp)。前者网页抓取空后用HTTPS正文请求读全；已核对查表、棋盘树组、稀疏回填、均值子采样、原地修正和四邻边界，后者核对双线性法向比较及图像边缘衰减。**未编译、未运行、未做并行正确性或安全审计**。

<a id="sf-source-7"></a>

**[cite:7] 作者基础工具箱固定源码，所列文件全文已读。** `git ls-remote`锁定`e87332637bbe8e8b92dd487c87567d9628404523`：[forestTrain.m](https://github.com/pdollar/toolbox/blob/e87332637bbe8e8b92dd487c87567d9628404523/classify/forestTrain.m)、[forestFindThr.cpp](https://github.com/pdollar/toolbox/blob/e87332637bbe8e8b92dd487c87567d9628404523/classify/private/forestFindThr.cpp)、[pca.m](https://github.com/pdollar/toolbox/blob/e87332637bbe8e8b92dd487c87567d9628404523/classify/pca.m)。这是节点离散化回调、阈值扫描和PCA布局证据，不是工具箱所有依赖的审读；文件头局部版本不能替代依赖包版本清单。

<a id="sf-source-8"></a>

**[cite:8] 作者许可，完整文本已读。** [license.txt](https://github.com/pdollar/edges/blob/94260b5d0fc068202598312e01a37604972dcc9e/license.txt)为MSR-LA，许可授予明确限定**非商业用途**并含分发／衍生作品约束；不能因代码公开或README写Full Rights就认定可直接商用。基础工具箱部分文件另标Simplified BSD，不能据此覆盖edges许可。接入工业产品前需单独确认许可，本文不复制作者实现作为本仓库代码。

## 验证边界与销项建议

建议将 **G14 的“缺少具体算法专卡”转为已有具体卡**，目标链接为`structured_forests.md#sf01`，保留历史ID和05-08节点；建议登记 **L1／L2／L3／L4／L5＝核／核／部／核（限定代数）／部，答5，未实现**。满足销项的是2013结构编码／节点离散化／split／叶代表／边界融合机制闭合，不是“普通RF已有一张卡”。

**不建议销项**：2013 V1.0发布包与历史权重恢复、作者MATLAB/MEX运行与逐位对齐、训练数据和分组协议复现、真实工业域定参、sharpen完整测试、性能剖析、物理计量与部署许可。2014扩展已说明来源边界，不能反标成2013能力；作者edges代码的非商业许可限制需单独处理。[cite:8](#sf-source-8)

本次只新增本文件；未改卡片索引、覆盖矩阵、测量总览或源码，故它们仍可能显示旧“缺卡”和旧计数。后续维护者可据本卡核对后同步台账，但不得顺带把实现／验证状态升级。文档核验与有限代数通过不等于SF已实现、已训练、已复现论文或已通过现场验收。
