# FG01 FCGF：稀疏三维几何特征与 hardest-contrastive 学习

[学习规范](../algorithm_learning_standard.md) · [三维总览](../3d_evolution.md) · [覆盖台账](../evolution_coverage.md) · [卡片索引](README.md) · [FPFH／SAC-IA](fpfh_sacia.md) · [ICP](remaining_domains.md#g02-刚性-point-to-point-icp)

<a id="fg01"></a>
## 范围、版本与覆盖状态

核对日期：**2026-11-06**。唯一键 **`fcgf.md:FG01`**；承接 **G42／10-09／E09**，只有一张卡，网络、损失、采样和下游接口不拆卡，共享五组逐层问答。本文件提供 G42 具体方法缺卡的销项依据；不改总览、索引、台账或其历史计数。

**任务与边界。** FCGF（Fully Convolutional Geometric Features）输入已有三维点云，以稀疏体素网络输出占用位置的几何描述子；不是从 RGB 图像预测深度，不是二维 SuperPoint，也不是直接回归 SE(3) 的配准器。训练需要有重叠的点云对和可信的几何对应监督；推理提特征不需要真值变换。输出描述子后，最近邻对应、鲁棒刚体估计、拒识、可选 ICP 和融合都是可组合但独立的阶段。[cite:1](#source-1)[cite:2](#source-2)[cite:8](#source-8)

| 项目 | 状态与限定 |
|---|---|
| 演进定位 | 2019 ICCV：从逐 patch 重复计算／有限上下文，转向整云稀疏全卷积共享激活与稠密于占用点的几何特征；新增训练、稀疏坐标管理和域迁移成本 |
| L1 步骤 | 已核验，限原文方法段与固定作者源码的数据—网络—损失链 |
| L2 机制 | 已核验，限稀疏支持、残差／跳连、度量监督与负排除因果 |
| L3 工程权衡 | 部分；参数量纲、诊断和协议齐备，无真实点云调参结果 |
| L4 数学推导 | 已核验，限本文显式模型、固定索引分支的局部梯度及有限代数检查；不含泛化／全局收敛证明 |
| L5 实现 | 部分；固定外部源码、参考伪代码和有限 shape／索引／梯度检查，不是完整复现 |
| 实现／验证 | **本仓库未实现 FCGF**；未安装或运行 MinkowskiEngine，未加载权重、训练、做点云配准或物理验收 |
| 面试自测 | 答5，逐层包含条件、易错答案和证据入口 |

**版本冻结。** 方法依据 Choy、Park、Koltun 的 ICCV 2019 正文；代码依据作者仓库 `chrischoy/FCGF` 的 **`0612340ead256adb5449da8088f506e947e44b4c`（2021-02-18）**，默认 `ResUNetBN2C`／`HardestContrastiveLossTrainer`／室内 occupancy 输入路径。它是已迁移到 MinkowskiEngine v0.5 系列接口的后续快照，不是“2019 实验提交”。本次发现远端当前提交为 `499813ea35f57d885cbc3458cd74fc57dae860c7`，不拿其新后端说明覆盖已固定路径。权重文件名含 2019 也不等于已经核验权重内容。[cite:2](#source-2)

证据主要来自同一作者体系：论文、作者镜像和作者实现不能凑成三次独立验证。论文机制与代码对照标为 `[CONFIRMED：限定一手资料交叉]`；代码特有行为标为 `[SINGLE-SOURCE：固定源码]`。没有独立复现实验支撑性能结论，本文不转用论文速度倍数或模型库精度作本仓库结果。

## L1 步骤：点云、稀疏网络、训练与推理分流

### 输入输出与索引契约

设点云对为 A、B；原始有效点数分别为 R、S，量化后占用体素数 N、M。坐标长度单位必须一致，通常把室内数据解释为米；下标是从 0 开始的整数，不是物理坐标。列向量约定为 \(q\approx Rp+t\)，行存储写成 `P @ R.T + t`。

| 对象 | shape／类型 | 语义与不能混用的索引 |
|---|---|---|
| 原始点／有效 ID | `P_raw[R,3]`，浮点；`raw_id[R]` | 删除 NaN／Inf 后仍可追溯输入；不用图像像素行列充当 xyz |
| 体素键 | `V[N,3]`，整数 | \(v=\lfloor p/h\rfloor\)，h 是长度；负数必须 floor 而非向零截断 |
| 代表点与映射 | `P[N,3]`、`sel[N]`、可选 `inv[R]` | `P=P_raw[sel]`；`V[inv]` 对应逐原始点量化键。代表点不是体素中心或邻点均值 |
| 稀疏输入 | `C[N_total,4] int32`、`X[N_total,1] float32` | 此现代接口使用 `[batch,x,y,z]`；occupancy 输入为全 1，几何由支持坐标表达 |
| 输出特征 | `F_A[N,D]`、`F_B[M,D]` | 默认 D=32；必须与代表点坐标按键对齐，不能只凭长度相等假定行序一致 |
| 真值 | `T_B_from_A[4,4]`、`P_all[K,2] int64` | 正对每行 `(i,j)` 指向量化后的 A、B 特征；多对多，不是默认一对一 |
| 挖掘子集 | `selA[H_A]`、`selB[H_B]`、`P_s[K_s,2]` | 子集局部 argmin 位置必须经 sel 映回整批特征行 |

论文 §3 写 `[x,y,z,b]`，现代 ME 文档明确旧版曾把 batch 放末列，当前置于首列；应通过 `sparse_collate`／`batched_coordinates` 构造，不照论文排版直接传入新 API。batch 只是隔离场景的标签，不沿它做三维空间卷积。`coordinate_map_key`、`coordinate_manager`、tensor stride 一起决定支持与映射，不只是一个 xyz 数组。[cite:1](#source-1)[cite:5](#source-5)[cite:7](#source-7)[cite:9](#source-9)

### 一次推理与一次训练更新

1. **检查与量化。** 统一长度单位，拒绝非有限点、空云、非正 h；每个点云独立量化、去重并保存代表点。固定室内数据路径用 `sparse_quantize(..., return_index=True)` 取一个原始代表，再对其坐标 floor；不做体素均值，也不先求 FPFH／法向。过滤与 inverse map 是本文复现防护，作者 helper 并未完整实现这些检查。[cite:5](#source-5)[cite:7](#source-7)
2. **构造稀疏输入。** 每个占用体素一行常数 1，按 batch 拼接 C、X。A、B 两侧各自累计行偏移：`offsetA[b]=sum(N_a,a<b)`，`offsetB[b]=sum(M_a,a<b)`。局部正对 `(i,j)` 加两列各自偏移，不能两列都加 A 偏移。[cite:5](#source-5)
3. **共享权重提特征。** A、B 分别通过同一个 3D ResUNet，输出恢复到输入占用分辨率的 D 维向量。按输出坐标键映回输入代表点行号；若缺坐标、重复键或 stride 不符，拒绝继续。不能把转置卷积理解为原卷积的数学逆，也不能把输出写成原始 R 个独立新特征。[cite:3](#source-3)
4. **逐行归一化。** 默认归一为单位 L2 特征；零向量在固定代码中会除零。防护版本把零／非有限向量判无效，或明示采用带 epsilon 的不同算子，不能默默称其与作者实现等价。[cite:3](#source-3)
5. **训练分支。** 在正确真值变换下，对量化代表点做几何半径搜索生成完整正对 `P_all`，抽取正对及两侧候选特征，双向挖 hardest，排除假负，计算损失、反传、更新共享网络。输入 xyz、量化键、真值匹配与 argmin 选择本身不是可学习量。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)
6. **推理分支。** 只返回 `(代表点,描述子,有效标记,映射,单位/版本)`；要对应则另做特征最近邻／可选互检，要位姿则再做三维几何估计。单云前向处理完即停止，无“特征收敛”循环；训练按 epoch／验证选模及预算停止，源码默认 100 epochs 不是全局最优判据。[cite:2](#source-2)[cite:4](#source-4)[cite:8](#source-8)

### 残差 UNet 的固定 shape 链

下面是 `ResUNetBN2C`，不是全部 B／C／D／E、BN／IN 变体的共同通道表。`N_s` 是 stride s 下实际占用数，不可写成 `N/s³`；转置层及 skip 需用稀疏坐标图对齐。[cite:3](#source-3)

| 阶段 | 空间 stride／通道 | 运算及 skip |
|---|---|---|
| 输入／E1 | 1：1→32 | 5³ conv，BN，残差块，ReLU；保存 `out_s1[N1,32]` |
| E2 | 2：32→64 | 3³ stride-2 conv，BN，残差块，ReLU；保存 `out_s2[N2,64]` |
| E3 | 4：64→128 | 同类下采样；保存 `out_s4[N4,128]` |
| E4 | 8：128→256 | 同类下采样；瓶颈 `out_s8[N8,256]` |
| D4 | 4：256→128 | 3³ stride-2 transpose，BN，残差块，ReLU；与 E3 拼通道得 256 |
| D3 | 2：256→64 | transpose，BN，残差块，ReLU；与 E2 拼通道得 128 |
| D2 | 1：128→64 | transpose，BN，残差块，ReLU；与 E1 拼通道得 96 |
| 输出头 | 1：96→64→D | 1³ conv＋ReLU，再 1³ final（bias=True）；默认 D=32，最后逐行 L2 归一 |

一个 `BasicBlockBN` 是 `3³ conv → BN → ReLU → 3³ conv → BN → 加 residual → ReLU`。块内相加要求同支持、同通道；UNet skip 是 `ME.cat` 的通道拼接，不是 residual 加法。论文图2图注概述“除末层外卷积带 BN/ReLU”，但固定代码的 `conv1_tr` 后只有 ReLU、没有额外 BN；复现固定代码须保留这项差异。图2图形的全部通道标签未由本次文本提取独立核清，详细数值以固定源码为证，不冒称逐像素核读原图。[cite:1](#source-1)[cite:3](#source-3)

## L2 机制：为什么这些环节不能省略

| 环节 | 因果机制 | 删除／替换的后果 |
|---|---|---|
| 三维稀疏支持 | 只在选定占用坐标及核连接上计算，共享相邻区域中间激活 | 把 xyz 投成二维图再用图像特征改变邻接、遮挡与几何契约；dense 3D 网格则浪费空域存储 |
| 常数 occupancy 特征 | 即使每行都是 1，邻接支持不同也产生不同卷积响应 | 只把 `[N,1]` 送 MLP 而丢 C，将无法从常数行区分局部几何 |
| 下采样＋上采样＋skip | 扩大上下文后恢复高分辨率局部细节 | 单独瓶颈特征不能直接逐行对应原始分辨率；去 skip 与保留不是等价优化 |
| residual | 特征与梯度有恒等路径，便于学习增量 | 不是 SE(3) 不变性证明；共享卷积不等于任意旋转／尺度不变 |
| 输出归一化 | 固定特征范数，让欧氏距离与夹角单调对应，限制靠放大范数分离负对 | BN 不是逐点 L2 归一；去归一需重新核对权重、距离范围与 margin |
| 几何正监督 | 真值变换后邻近的代表点被拉近，容忍采样不一 | 按两个数组相同行号配正对、或用当前预测近邻当真值会污染监督 |
| hardest＋假负排除 | 关注最易混淆特征，避免把邻近同表面点推远 | 局部特征相关，随机负大多容易；仅排一个正 partner 会漏掉半径内其他正对 |
| 双向与分别平均 | 正对两端都获得困难负约束，每方向按有效负数归一 | 把 mask 后损失除以抽样前点数，或把两方向混成按总数量平均，改变权重 |

这里“全卷积”的优势是空间共享与可变点数处理，不意味着填满未观测空间。栅格量化、stride、边界和旋转会改变占用模式；任意刚体不变性不是架构的严格性质。局部相关也使“批内有大量特征”不等于大量独立训练样本；场景／对象级切分比随机点级拆分更能避免泄漏。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

## L3 工程权衡：参数单位、采样单位与排障

### 固定配置与可变协议

| 参数 | 固定室内默认／单位 | 调整依据、症状与成本 |
|---|---|---|
| `voxel_size=h` | 0.025，点坐标长度；米制时 2.5 cm | 小 h 保细节但占用点、核图和挖掘成本上升；大 h 合并细结构。换米／毫米应同时换 h、GT 半径与配准门 |
| 正对搜索半径 \(r_+\) | `1.5*h`，长度；未增广时 0.0375 m | 大则跨面假正多，小则真值噪声／密度差使正对稀少；检查每对扫描的正对数、端点覆盖、几何残差 |
| `pos_thresh=τp` | 0.1，**平方特征距离** | 固定代码在 d² 上 hinge；正项零区是 d≤√τp，不是 d≤0.1 |
| `neg_thresh=τn` | 1.4，特征距离 | 与几何米制半径无关；单位归一特征 d≤2，不合理的大 margin 可能不可满足 |
| `neg_weight=wn` | 1，无量纲；两方向再除 2 | 对应各方向权重 0.5；不要在训练外层重复乘一次 0.5 |
| 正对与候选预算 | 每批单位配置 1024／256，数量 | 实际入口各乘配置 batch_size，正对／候选不足时截到可用数量；预算不是原始点数或独立场景数 |
| batch／梯度累积 | batch_size=4，iter_size=1 | batch_size 是扫描对数，累计更新可跨多个 batch；不等于每次每点独立采样 |
| 网络／输入 | C 型、D=32、核5、常数1 | 换成 RGB／normal、IN、其他通道或不归一属于模型变体；必须匹配 checkpoint |

默认数值来自固定 `config.py`，不是跨传感器推荐值。模型库还列有 KITTI 不归一、不同 h／D 的权重；不能说“所有 FCGF 模型都单位归一”。随机尺度开关默认 false；室内增广启用后两云同乘 s，正对半径也乘 s，但体素 h 不跟着乘，因而占用模式有意改变。[cite:2](#source-2)[cite:5](#source-5)[cite:8](#source-8)

**采样单位必须说清。** 论文 §4.2 先说按 scene 抽 anchor／挖掘集合，再用 minibatch 特征描述候选集合；不能据此虚构严格逐场景配额。固定训练入口明确在拼接后的 `F0`／`F1` 全部行中均匀无放回抽候选，并在拼接后的全部正对行中抽 `P_s`。默认 B=4 时，上限为 **4096 正对、每侧 1024 候选**；不是每云保证 1024／256。点多的扫描进入候选的机会更多，半径邻居多的 anchor 可在正对表多次出现，等权的是“正对行”，不是“唯一物理点”。候选可来自不同 batch 项；其与 anchor 的组合不在正表时会被当负例，代码没有显式逐 scene 限制。重复或关联扫描跨项出现会带来假负风险。[cite:1](#source-1)[cite:4](#source-4)

**诊断顺序。** 单位／T 方向／增广复合 → 原始 ID、量化代表、稀疏输出坐标映射 → 每批正对数量与覆盖 → 两方向被排除比例／空集合 → 描述子范数与正负距离分布 → h、预算、margin、模型／数据域。损失 NaN 先查零范数、空正对和空有效负均值，不能先调学习率。训练损失低而配准失败，应查低重叠、重复几何、场景泄漏、对应覆盖及刚体退化，不只是多训 epoch。

**资源与优化。** 稀疏卷积算术按实际 kernel-map 连接数 E 计，约 \(O(E C_{in}C_{out})\)，另有哈希／建图成本，不应硬写成 O(N) 且与核无关。挖掘距离为 \(O(K_s(H_A+H_B)D)\)；固定 `pdist` 用广播差产生 `[K_s,H,D]` 中间张量，不只是 `[K_s,H]` 的存储。分块寻找 argmin、停止保留挖掘中间图后用原特征重算获选距离，能减少损失部分显存，但需保持同一候选集合、平方／开方、平局策略、后排除和归一分母，才能在非切换点保持等价。ANN 或先排除再搜索改变目标，不是严格等价加速。[cite:3](#source-3)[cite:4](#source-4)[cite:6](#source-6)

## L4 数学推导：映射、损失与梯度边界

### 稀疏卷积与代表点误差的条件界

对同一 batch、同一 stride 的示意层，令偏移集合 \(\mathcal K\)，核连接 \(\mathcal E_\delta\) 保存输入行 r 与输出行 t：

\[
y_t=\sum_{\delta\in\mathcal K}\sum_{(r,t)\in\mathcal E_\delta}W_\delta x_r,
\quad W_\delta\in\mathbb R^{C_{out}\times C_{in}}.
\]

它是论文式(3)按坐标映射展开的行索引形式：没有对应占用输入就没有那一项。stride 改变时连接由坐标管理器产生，不能直接把上一层行号除 2。固定支持时，设上游梯度 g，则

\[
\frac{\partial L}{\partial x_r}=\sum_{\delta,t:(r,t)\in\mathcal E_\delta}W_\delta^Tg_t,
\qquad
\frac{\partial L}{\partial W_\delta}=\sum_{(r,t)\in\mathcal E_\delta}g_tx_r^T.
\]

坐标整数／哈希连接不由这些式子获得可用的连续梯度；对点坐标的可微优化需另外定义模型。残差块的局部 Jacobian 有恒等加项，skip 拼接反传则把上游按通道拆回两支；不能混写成同一规则。[cite:1](#source-1)[cite:3](#source-3)

若两点被分到同一个边长 h 的半开立方体，各坐标差小于 h，故原点到所选代表点距离 **小于 √3h**；若用中心替代才有 √3h/2 界，而本代码不是中心法。两份云各自替换代表、刚体 R 保范数时，固定对应的几何残差变化小于 2√3h。这只是条件量化扰动界：不保证对应仍被选中、描述误差小、位姿精度或测量误差受同样界约束；本方法未提供通用配准／泛化误差上界。

### groundtruth 正匹配不是特征最近邻

定义完整正对集合：

\[
\mathcal P_{all}=\{(i,j):\|R_{gt}p_i+t_{gt}-q_j\|_2<r_+\}.
\]

固定实现用 Open3D 半径查询，`K=None` 保留全部邻居，不是只选最近的一个；严格边界点是否包含由所用 Open3D 版本验证，本文公式以开球作为明确参考契约。搜索是在变换后的**代表点物理坐标**上，不是整数体素索引或描述子上。室内 loader 期待预处理的两云本来在共同坐标系，独立刚体增广后 `T_gt=T1 @ inv(T0)`；一般原始关系为 T 时，应为 `T1 @ T @ inv(T0)`，不能给任意未对齐两云套 identity。[cite:5](#source-5)[cite:6](#source-6)

这是由已有 GT 派生的监督，不把 GT 本身当网络预测；真值误差、非刚性运动和动态物体可产生错标。KITTI 的里程计／ICP 缓存路径另有标注生成流程，不等于纯室内 groundtruth；本卡不宣称已验收其坐标复合或缓存质量。

### 论文 hardest-contrastive 与固定实现并列

令 \([z]_+=\max(0,z)\)，\(d(f,g)=\|f-g\|_2\)。对抽样正对 \((i,j)\in\mathcal P_s\)，两侧候选为 \(S_A,S_B\)，先选

\[
k_i=\arg\min_{k\in S_B}d(f_i,g_k),\qquad
\ell_j=\arg\min_{\ell\in S_A}d(g_j,f_\ell).
\]

然后以几何真值确定 \(I_i,J_j\)，排除已选中的假负。论文先挖再按空间球过滤，正文将 \(d_t\) 称为球的 **diameter**；若采用此原文字面定义，排除半径是 \(d_t/2\)，不能无说明当半径。固定代码不用显式 d_t，也不在 loss 内重新查 xyz，而是检查获选对是否属于完整 \(\mathcal P_{all}\)，同批真实配对且正表完整时等价于该正表的几何半径排除。[cite:1](#source-1)[cite:4](#source-4)

令 \(K=|\mathcal P_s|\)，\(Z_A=\sum I_i\)，\(Z_B=\sum J_j\)。在 K、Z_A、Z_B 都非零时，论文形式为：

\[
L_{paper}=\frac1K\sum_{(i,j)}[d(f_i,g_j)-m_p]_+^2
+\frac{\lambda_n}{Z_A}\sum_{(i,j)}I_i[m_n-d(f_i,g_{k_i})]_+^2
+\frac{\lambda_n}{Z_B}\sum_{(i,j)}J_j[m_n-d(g_j,f_{\ell_j})]_+^2.
\]

论文给 \(\lambda_n=0.5\)。这是按方向的有效负平均，不是按所有 pair 的单一总平均。原文还提出 hardest-triplet；本文不把它和 contrastive 混成一个 loss，也不另计一张卡。[cite:1](#source-1)

固定源码实际为 \(\epsilon_d=10^{-7}\)、\(d_\epsilon(f,g)=\sqrt{\|f-g\|^2+\epsilon_d}\)：

\[
L_{code}=\frac1K\sum_{(i,j)}[\|f_i-g_j\|^2-\tau_p]_+
+\frac{w_n}{2}\left(
\frac1{Z_A}\sum_{(i,j)}I_i[\tau_n-d_\epsilon(f_i,g_{k_i})]_+^2
+\frac1{Z_B}\sum_{(i,j)}J_j[\tau_n-d_\epsilon(g_j,f_{\ell_j})]_+^2\right).
\]

**正项差异是真正目标变化。** \([d-m_p]_+^2\) 不等于 \([d^2-\tau_p]_+\)，即使令 \(m_p=\sqrt{\tau_p}\) 只对齐零区，非零损失与梯度仍不同。负项默认 `neg_weight=1` 再除 2 与论文各方向 0.5 的结构对应，但数值 epsilon 及采样／排除协议仍须分别记录。普通 `ContrastiveLossTrainer` 的随机负路径又使用无 margin 正项及不同 epsilon，不能拿它代替本卡固定 hardest 路径。[cite:4](#source-4)[cite:6](#source-6)[cite:8](#source-8)

**先 min 后排除，无补采。** `D01ind` 是 `selB` 的局部行，映射为 `selB[D01ind]` 后，与正对第一列一起查 hash；反向查的是 `(selA[D10ind],j)`，不是 `(j,selA[D10ind])`。hash 为 `i+j*H`，`H=max(N_total,M_total)`；有效索引低于 H 且 int64 不溢出时无碰撞。查表使用全部正对，不是仅 `P_s`。若最近候选为真对应，即使第二近候选是真负且有非零损失，该 anchor 的此方向也整个被丢弃；预先把正对置 Inf 再 min 属于另一个算法。[cite:4](#source-4)[cite:7](#source-7)

### 梯度、归一化与非光滑边界

设 \(\Delta=f-g\)，在 hinge 激活且 d>0 的固定分支上：

\[
\nabla_f[d-m_p]_+^2=2(d-m_p)\Delta/d,
\qquad
\nabla_f[d^2-\tau_p]_+=2\Delta,
\]
\[
\nabla_f[\tau_n-d_\epsilon]_+^2
=-2(\tau_n-d_\epsilon)\Delta/d_\epsilon,
\qquad \nabla_g=-\nabla_f.
\]

hinge 未激活时为零；还需乘各项有效均值和权重。负梯度下降把两特征推开，正梯度下降拉近；一个点被多个正对或两个挖掘方向选中时，梯度要累加，不能覆盖。

归一化前向量 z 非零，\(f=z/\|z\|\)，则

\[
\frac{\partial f}{\partial z}=\frac{I-ff^T}{\|z\|},\qquad
\nabla_zL=\frac{(I-ff^T)\nabla_fL}{\|z\|}.
\]

该投影梯度与 z 正交，不能把除以范数仅当推理后处理而漏掉反传。单位特征满足 \(d^2=2-2f^Tg\in[0,4]\)。零范数无定义；若换成 `z/max(norm,eps)`，小范数段的 Jacobian 不再是此式。[cite:3](#source-3)

argmin、几何 mask、正对表和整数坐标不反传；但**获选距离的数值必须保留梯度**，才会流到 anchor、获选负特征及共享网络。固定实现的 `min` 值仍在 autograd 图里，转 NumPy 的是获选索引／排除操作，不是损失距离。局部唯一最小值、mask 固定时，停止挖掘图后重算同一距离具有相同梯度；最小值平局、hinge 边界、量化边界不保证普通导数唯一，应记录子梯度／tie 规则。被排除的**该负项**无梯度，不代表这个特征因正项或其他 anchor 也无梯度。[cite:4](#source-4)

当 `K=0`、`Z_A=0` 或 `Z_B=0`，固定代码的空 `.mean()` 可产生 NaN，不存在“自动安全跳过”。本文防护伪代码规定拒绝该批并上报原因，不以 `max(Z,1)` 偷改目标；若工程采用单方向可用就继续或回传连接图的零值，必须命名成另一种空集策略并单测。

## L5 复现设计、有限检查与下游接口

### 参考伪代码：固定有效路径，显式拒绝异常

这是**未运行的整算法伪代码**。下列 `ALIGN_BY_COORDINATE_KEY` 和空集／零范数拒绝为工程防护，不宣称作者 helper 已提供；稀疏网络实现见 L1 通道表。形状相同不足以证明坐标行一致。

```text
VOXELIZE(P_raw, h, batch_ids):
    require finite(P_raw) and rows(P_raw) > 0 and h > 0
    V_raw = floor(P_raw / h)
    V, sel, inv = unique_voxels_with_maps(V_raw, batch_ids)
    P = P_raw[sel]
    C = batched_coordinates(V, batch_ids[sel])
    X = ones(rows(V), 1)
    return P, C, X, sel, inv

ENCODE(P, C, X, model):
    S = SparseTensor(X, coordinates=C)
    Y = model.forward_without_final_normalization(S)
    Z = ALIGN_BY_COORDINATE_KEY(Y, C, required_stride=1)
    require shape(Z) == (rows(C), model.output_dimension) and finite(Z)
    require all(row_norm(Z) > 0)
    F = Z / row_norm(Z)
    return F

TRAIN_BATCH(pairs, cfg):
    zero_grad_at_start_of_accumulation()
    for each scan pair b:
        augment points and update T_gt consistently
        PA, CA, XA, selA, invA = VOXELIZE(A, cfg.h, per_point_batch_id(b))
        PB, CB, XB, selB, invB = VOXELIZE(B, cfg.h, per_point_batch_id(b))
        Pall_b = radius_pairs(transform(PA, T_gt), PB, r_positive)
        append voxelized arrays and maps to side-specific batch lists
    PA, CA, XA = concatenate_A_lists_in_pair_order()
    PB, CB, XB = concatenate_B_lists_in_pair_order()
    Pall = concatenate_positive_pairs_with_separate_A_B_offsets()
    FA = ENCODE(PA, CA, XA, shared_model)
    FB = ENCODE(PB, CB, XB, shared_model)
    require len(Pall) > 0 and all pair indices in range
    Ps = sample_without_replacement(Pall, min(len(Pall), cfg.npos * cfg.B))
    SA = sample_without_replacement(rows(FA), min(rows(FA), cfg.nhn * cfg.B))
    SB = sample_without_replacement(rows(FB), min(rows(FB), cfg.nhn * cfg.B))
    DAB = sqrt(pairwise_squared(FA[Ps[:,0]], FB[SB]) + 1e-7)
    DBA = sqrt(pairwise_squared(FB[Ps[:,1]], FA[SA]) + 1e-7)
    dA, localA = row_min(DAB)
    dB, localB = row_min(DBA)
    globalB, globalA = SB[localA], SA[localB]
    maskA = pair_not_in_full_positive_set(Ps[:,0], globalB, Pall)
    maskB = pair_not_in_full_positive_set(globalA, Ps[:,1], Pall)
    require sum(maskA) > 0 and sum(maskB) > 0
    Lpos = mean(relu(row_squared_norm(FA[Ps[:,0]] - FB[Ps[:,1]]) - cfg.taup))
    Lneg = (mean(relu(cfg.taun - dA[maskA])^2) + mean(relu(cfg.taun - dB[maskB])^2)) / 2
    L = Lpos + cfg.wn * Lneg
    require finite(L)
    backward(L / cfg.iter_size)
    optimizer_step_after_accumulation()
```

伪代码逐对量化和建正表，再把同侧多扫描先 collate 后一次前向，以保留固定程序的 BN 批统计；逐对单独前向并累计损失**不等价于原批量 BN**。推理为 `VOXELIZE → ENCODE`，另返回代表点和两份映射。固定 collate 的 `correspondences` 最初转换为 `.int()`，loss 索引再 `.long()`；大规模实现应在中间偏移、hash 和索引上全程保证整数范围，不能把 int32 溢出后再转 int64 当补救。[cite:4](#source-4)[cite:5](#source-5)

### 有限 shape／索引／梯度检查器

以下 PowerShell 检查只使用明确给出的**代数边界常量与索引夹具**，不是模拟训练数据、实测点云或算法性能实验。检查残差 UNet 通道拼接、负坐标 floor、双侧偏移、子集回映、正表全量排除、后排除与先排除的非等价、正项两版本差异，以及归一化负损失／固定正项的局部有限差分。使用三维向量只是数学梯度夹具，不替代三维稀疏网络前向。

```powershell
$ErrorActionPreference = 'Stop'
$script:checks = 0
function Require([bool]$condition, [string]$name) {
    if (-not $condition) { throw $name }
    $script:checks++
}
function Norm([double[]]$v) {
    $s = 0.0
    foreach ($x in $v) { $s += $x * $x }
    return [Math]::Sqrt($s)
}
function NegativeLoss([double[]]$z, [double[]]$w) {
    $nz = Norm $z
    $nw = Norm $w
    $d2 = 0.0
    for ($i = 0; $i -lt 3; $i++) { $d2 += [Math]::Pow($z[$i] / $nz - $w[$i] / $nw, 2) }
    $a = [Math]::Max(0.0, 1.4 - [Math]::Sqrt($d2 + 1e-7))
    return $a * $a
}
Require ((128 + 128 -eq 256) -and (64 + 64 -eq 128) -and (64 + 32 -eq 96)) 'skip channels'
Require (([Math]::Floor(-0.2) -eq -1) -and ([int][Math]::Truncate(-0.2) -eq 0)) 'negative floor'
$localPair = @(1, 2)
$offset = @(3, 5)
$globalPair = @(($localPair[0] + $offset[0]), ($localPair[1] + $offset[1]))
Require (($globalPair[0] -eq 4) -and ($globalPair[1] -eq 7)) 'separate offsets'
$keys = @('0,-1,0,0', '0,0,0,0', '1,-1,0,0')
Require ((@($keys | Select-Object -Unique).Count -eq 3)) 'batch key isolation'
$outputKeys = @($keys[2], $keys[0], $keys[1])
$rowMap = @{}
for ($r = 0; $r -lt $outputKeys.Count; $r++) { $rowMap[$outputKeys[$r]] = $r }
$aligned = @($keys | ForEach-Object { $rowMap[$_] })
Require (($aligned[0] -eq 1) -and ($aligned[1] -eq 2) -and ($aligned[2] -eq 0)) 'coordinate permutation mapping'
$selB = @(7, 2, 5)
$localMin = 1
$j = $selB[$localMin]
Require ($j -eq 2) 'local to global'
$H = 11
$Pall = @('4,7', '4,2')
$Ps = @('4,7')
Require (($Pall -contains "4,$j") -and ($Ps -notcontains "4,$j")) 'full positive exclusion'
$hashes = @(for ($i = 0; $i -lt 8; $i++) { for ($k = 0; $k -lt 11; $k++) { $i + $H * $k } })
Require ((@($hashes | Select-Object -Unique).Count -eq 88)) 'hash injectivity on range'
$distances = @(0.2, 0.8)
$isPositive = @($true, $false)
$argmin = 0
for ($r = 1; $r -lt $distances.Count; $r++) { if ($distances[$r] -lt $distances[$argmin]) { $argmin = $r } }
$postExclude = -not $isPositive[$argmin]
$allowedDistances = @(for ($r = 0; $r -lt $distances.Count; $r++) { if (-not $isPositive[$r]) { $distances[$r] } })
$preExcludeMin = ($allowedDistances | Measure-Object -Minimum).Minimum
$preExcludeLoss = [Math]::Pow(1.4 - $preExcludeMin, 2)
Require ((-not $postExclude) -and ($preExcludeLoss -gt 0)) 'no re-mining after exclusion'
$d = 0.8
$mp = [Math]::Sqrt(0.1)
$paper = [Math]::Pow($d - $mp, 2)
$code = $d * $d - 0.1
Require ([Math]::Abs($paper - $code) -gt 0.1) 'positive variants differ'
$z = [double[]]@(1.0, 2.0, 0.5)
$w = [double[]]@(2.0, 1.0, 0.2)
$nz = Norm $z
$nw = Norm $w
$f = [double[]]@($z | ForEach-Object { $_ / $nz })
$g = [double[]]@($w | ForEach-Object { $_ / $nw })
$delta = [double[]]@(0, 0, 0)
$dist2 = 0.0
for ($i = 0; $i -lt 3; $i++) { $delta[$i] = $f[$i] - $g[$i]; $dist2 += $delta[$i] * $delta[$i] }
$de = [Math]::Sqrt($dist2 + 1e-7)
Require (($de -gt 0) -and ($de -lt 1.4)) 'active smooth negative branch'
$gf = [double[]]@($delta | ForEach-Object { -2 * (1.4 - $de) * $_ / $de })
$dot = 0.0
for ($i = 0; $i -lt 3; $i++) { $dot += $f[$i] * $gf[$i] }
$grad = [double[]]@(0, 0, 0)
for ($i = 0; $i -lt 3; $i++) { $grad[$i] = ($gf[$i] - $f[$i] * $dot) / $nz }
$step = 1e-6
$maxError = 0.0
for ($i = 0; $i -lt 3; $i++) {
    $zp = [double[]]$z.Clone()
    $zm = [double[]]$z.Clone()
    $zp[$i] += $step
    $zm[$i] -= $step
    $fd = ((NegativeLoss $zp $w) - (NegativeLoss $zm $w)) / (2 * $step)
    $err = [Math]::Abs($fd - $grad[$i])
    $maxError = [Math]::Max($maxError, $err)
    Require ($err -lt 1e-7) "normalized negative gradient $i"
}
$orthogonal = 0.0
for ($i = 0; $i -lt 3; $i++) { $orthogonal += $z[$i] * $grad[$i] }
Require ([Math]::Abs($orthogonal) -lt 1e-12) 'normalization tangent gradient'
$fdPositive = (([Math]::Pow($d + $step, 2) - 0.1) - ([Math]::Pow($d - $step, 2) - 0.1)) / (2 * $step)
Require ([Math]::Abs($fdPositive - 2 * $d) -lt 1e-7) 'code positive derivative'
$maskedGradient = 0.0 * $grad[0]
Require ($maskedGradient -eq 0) 'excluded negative contribution only'
Require ((Norm ([double[]]@(0, 0, 0))) -eq 0) 'zero norm must reject'
Require (@($Pall | Where-Object { $false }).Count -eq 0) 'empty valid set must reject'
"PASS checks=$script:checks max_gradient_error=$maxError"
```

**检查边界。** 它不运行 PyTorch autograd／ME 稀疏算子，不验证实际 kernel-map、转置卷积、BN、CUDA 平局行为或所有参数梯度；空集测试只确认拒绝条件，不证明外部代码已有守卫。需要后续固定依赖环境做：输入行打乱后按键对齐、重复体素、两个 batch 同 xyz 隔离、非对称两云点数、GPU 索引设备兼容、无正对／无负例、完整网络梯度、分块挖掘与原路径等价、真实对应与 SE(3) 验收。当前环境 `python` 指向不可用的商店入口，因此用系统 PowerShell 执行有限检查，不安装完整训练栈。

### 匹配、配准和深度获取的交接边界

| 阶段 | 输入→输出 | 监督／验收职责 |
|---|---|---|
| 深度获取前置 | 双目／结构光／ToF／已有扫描→三维点 | 标定、尺度、无效观测与深度误差；FCGF 不提供这一步 |
| FCGF | 单云 xyz＋固定模型→代表点描述子 | 点—体素—输出行映射，范数／有效性、训练域及版本 |
| 特征匹配 | 两云特征→候选 `(i,j)` | 最近邻是特征空间；互检／ratio／唯一性为可选额外策略，不默认自带 |
| 鲁棒初始配准 | 三维候选点对→`T_B_from_A`＋质量 | RANSAC 或其他估计器；空间覆盖、外点、退化和错误接受率；不是二维单应 H |
| 局部精化／融合 | 合格初值＋几何→精化位姿／融合表面 | ICP 局部域、重叠、误差与补全标签另验；不是 FCGF loss 的组成 |

固定 `trainer.py` 的验证分支以特征最近邻调用 `est_quad_linear_robust`，不应笼统说“作者验证一定用 RANSAC”；本文所列 RANSAC＋ICP 只是可选工程后端。Open3D 特征布局常需 `[D,N]`，而网络是 `[N,D]`，转置之后仍须保持点 ID。同一匹配集的 hit ratio（GT 几何距离门下的点对比例）、feature-match recall（扫描对层面 hit ratio 超门的比例）和 registration recall（位姿验收通过比例）不是一个指标；固定验证 `hit_ratio>0.05` 也不是 5% 的位姿成功率。[cite:4](#source-4)[cite:6](#source-6)

本地 [learned_matching.py](../../dl/learned_matching.py) 是图像接口且抛 `NotImplementedError`；本次在仓库 Python 文件检索 FCGF／Minkowski／ResUNet／HardestContrastive 无命中。二维 ZNCC、SuperPoint 说明、3×3 单应矩阵或已有通用测试都不能升级 FG01 的三维实现状态。

## 五组 L1–L5 面试追问与答案

| 层级／问题 | 答题要点与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：给两云和已知 T，怎样走完训练；推理少哪步？ | 分别量化保代表与映射，batch 双侧偏移，共享 ResUNet、行归一；GT 半径建正表，抽样、双向挖掘、后排除、loss 反传。推理无需 GT／loss，只提特征；要位姿另接几何后端 | 按两云同序号配正对；直接输出深度或单应 H | L1；来源3–7 |
| L2：常数 1 为什么有特征？为何 hardest 后还要排除？ | 信息在三维支持及邻接，卷积共享上下文；同表面相邻特征相关，几何邻近候选可能是假负，须排完整正表 | 用只有常数输入的逐点 MLP 替代；仅排正对唯一 partner 即足够 | L2；来源1、3–5 |
| L3：换成毫米输入或密度更高，先改什么？一个 batch 抽多少？ | 先同比换长度单位、h／r+／几何门，核对权重采样尺度；特征 margin 不随毫米换算。固定 B=4 抽至4096正对、每侧1024候选，来自拼接行而非每云配额 | τn 也乘1000；密度增加只调学习率；把点数当独立场景数 | L3；来源4、5、8 |
| L4：写出正项两版本梯度和单位化 Jacobian；mask 有梯度吗？ | 论文激活正项为2(d−mp)Δ/d，代码为2Δ；单位化为(I−ffᵀ)/norm。固定唯一 argmin／mask 下获选距离可微，离散选择不反传，排除仅清该负项 | 把平方距离 hinge 再平方；detach 所有距离；零范数仍套公式 | L4；来源1、3、4、6 |
| L5：为什么 loss NaN、shape 正确却对应错？能先排正再 min 吗？ | 查零范数／空均值、坐标键行序、sel回映、反向pair列序、双侧batch偏移、hash范围；先排再min会选下一负，与作者丢弃该anchor不同 | 只断言N×32就认为对齐；用epsilon掩盖所有空集；把ANN称等价优化 | L5检查器；来源3–7、9 |

<a id="verification"></a>
## 验证记录与 G42 销项依据

本卡承接的是 **G42“FCGF 三维网络、度量学习和对应输出缺少具体卡”**，不是宣布三维训练／配准系统完成。销项范围：稀疏体素与坐标映射、固定残差 UNet、输出归一、GT 正匹配、hardest-contrastive、负排除与梯度、采样单位、对应／刚体估计／深度前置分界均已展开，有公式、伪代码、有限检查及答5。文档层可由“P1 缺卡”转“已有机制／P2 深化”；外部台账未修改，历史 ID、数量和旧“无独立卡”文字不会由本次自动同步。

**本次执行结果：** 本文 PowerShell 检查器 **19 项通过**，归一化负项三分量中心差分的最大绝对误差为 **9.08×10⁻¹¹ 以下**，差分步长 10⁻⁶、断言门 10⁻⁷。UTF-8、表格列数、代码围栏、11 个唯一显式锚点、58 处 Markdown 本地链接、五层正文与答5检查通过；远端裸 URL 的获取深度分别记录于来源，不混入本地链接统计。首次执行发现 PowerShell 数组加法括号问题，修正后重跑通过，不隐去失败记录。

**改动与验证边界：** 本次写入操作只有新增和完善本文件；总览、索引、台账未写入，既有工作树修改未触碰。尝试采集全工作树哈希基线，但命令间内存变量未保留，故不宣称完成编辑前后全树哈希一致性证明；改动范围依据写入记录与 Git 状态核对。没有下载权重或数据、没有作者工程运行、没有真实特征匹配／配准／测速或现场验收，未执行伪代码不记为实现。

**P2 保留项：** 2019 原实验资产及权重内容／许可证核验；固定 ME／PyTorch／Open3D 兼容环境与算子坐标回归；论文排除尺度和图2全部细节的原图复核；KITTI 标注／变换缓存、跨批假负、空集政策与采样偏置；真实训练收敛、独立场景划分、旋转／密度／单位域偏移；下游 SE(3)、拒识和计量验收。L3／L5 仍部分，FCGF 仍未实现。

## 来源、获取深度与代码变体

均为 2026-11-06 本次联网记录。B=书目，A=摘要／说明，T=读到的方法正文，C=固定源码静态阅读；T 不等于逐页全文、全部实验或补充材料已审。代码默认值不是论文年代，作者镜像不是独立证据。以下链接除明确标动态 API／文档外都冻结到指定提交。

<a id="source-1"></a>
**[cite:1] 2019 原论文。** Christopher Choy、Jaesik Park、Vladlen Koltun，*Fully Convolutional Geometric Features*，ICCV 2019，8958–8966。出版页 https://openaccess.thecvf.com/content_ICCV_2019/html/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.html ：B+A。会议 PDF https://openaccess.thecvf.com/content_ICCV_2019/papers/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.pdf ：T，读到引言、§3 稀疏式(1)–(3)、图2图注、§4.1–4.2 hardest-contrastive 及有效负归一。返回在后续 triplet 段截断；图形内通道标签不完整，未宣称全部实验／附录精读。作者镜像 https://jaesik.info/publications/data/19_iccv.pdf ：T，同样读到 §4.2，核对 indicator 的 anchor i 与 sphere diameter 措辞；参考文献编号有别，属于同作者版本，不另当独立复现。

<a id="source-2"></a>
**[cite:2] 作者仓库与版本。** 历史提交解析 API https://api.github.com/repos/chrischoy/FCGF/commits?until=2021-12-31T23:59:59Z&per_page=1 ：核得 SHA `0612340ead256adb5449da8088f506e947e44b4c`，日期 2021-02-18，消息 `minor updates in readme and requirements`。固定说明 https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/README.md ：A+C，读到 ME v0.5 迁移、模型库归一／不归一、训练／benchmark 和下游 DGR 入口。动态当前提交 API https://api.github.com/repos/chrischoy/FCGF/commits?per_page=1 只用于确认当前版本不同，不用于解释固定算法。README 速度与论文表述及后端时期不同，不混合引用为统一性能结论。

<a id="source-3"></a>
**[cite:3] 网络与残差块。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/model/resunet.py 与 https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/model/residual_block.py ：C，已读 raw 源码，核对 C 型编码／解码通道、skip、残差两卷积、输出头、归一无 epsilon 与坐标图复用；BN／IN 及 B–E 变体存在，不宣称都已运行。

<a id="source-4"></a>
**[cite:4] 训练、hardest 与验证。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/lib/trainer.py ：C，读取 `AlignmentTrainer`、随机 contrastive 对照、`HardestContrastiveLossTrainer.contrastive_hardest_negative_loss`、训练调用及验证；首次返回在训练段截断，另以 raw HTTP 行窗口核到 411–520 行。核心 loss 为 397 行起的该方法，427–450 行明确两方向 min、全正表 hash、后排除、正项无外平方与分别平均，485–497 行明确预算乘 batch 与 loss 反传。无源码执行，不把 README 的索引重算优化自动当此方法已实现。

<a id="source-5"></a>
**[cite:5] 点云与 batch 数据路径。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/lib/data_loaders.py ：C，读到 `collate_pair_fn`、`PairDataset`、`IndoorPairDataset.__getitem__` 以及部分 KITTI 标注路径；已核对左右偏移、增广 T、代表点、常数 occupancy、floor 与半径倍率。文件后部获取不全，不宣称全部数据类和随机变换已审。

<a id="source-6"></a>
**[cite:6] 几何正对与特征距离。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/util/pointcloud.py 与 https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/lib/metrics.py ：C，核对 `get_matching_indices` 变换源云后全半径邻居、Open3D 特征转置、评价距离，以及 `pdist` 的广播差和 `sqrt(D2+1e-7)`。底层 Open3D 边界行为未运行核验。

<a id="source-7"></a>
**[cite:7] hash 与推理量化 helper。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/util/misc.py ：C，核对 `_hash` 的 `i+j*M`、`extract_features` 的 floor、return_index、代表坐标、int32 batch 坐标及 float32 特征。helper 可选 RGB／normal 不意味着默认 occupancy 模型能直接接收这些通道；有限断言不替代本文守卫。

<a id="source-8"></a>
**[cite:8] 固定配置。** https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/config.py ：C，核对默认模型、特征维度／归一、体素／半径、正负阈值、batch 与采样预算、epoch 及增广开关。loss 方法签名默认预算与训练入口实际传入值不同，以调用处而不是孤立签名作训练配置。

<a id="source-9"></a>
**[cite:9] ME 现代坐标接口说明。** https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html ：T/API，读到 batch 首列、0.4 之前末列警告、coordinate manager／map key／tensor stride、unique／inverse／slice 接口。它是动态文档，仅解释版本迁移和映射契约，未冻结其底层实现、不充当 2019 引擎源码或本次实际运行证据。
