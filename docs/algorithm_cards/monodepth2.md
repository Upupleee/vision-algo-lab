# MD01 Monodepth2：自监督深度、最小重投影与尺度边界

[学习规范](../algorithm_learning_standard.md) · [三维总览 E08](../3d_evolution.md#38-e08monodepth2训练监督变化不等于尺度问题消失) · [覆盖台账 G41／10-08](../evolution_coverage.md) · [三角化前置 C09](calibration.md#c09-三角化) · [双目对应 PM01](psmnet.md#pm01)

<a id="md01"></a>
<a id="g41"></a>

## 任务、演进与覆盖状态

**任务与输入输出：**训练用目标 RGB 图像、相邻时刻图像或同步双目另一视图、相机内参，以及已知双目外参或可学习的时序位姿；不以稠密深度 GT 构造主训练损失。推理只输入一张 RGB 图，通过深度网络得到稠密预测，不需要 pose 网络、相邻帧或重投影损失。这里的深度是针孔相机光轴方向的 Z，不是沿射线的欧氏距离；输出张量叫 `disp` 不代表它已经是双目水平像素视差。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)

**演进定位：**稠密深度标注昂贵 → 用图像重建提供自监督 → 原有多源平均误差惩罚遮挡，静止／共速物体破坏运动假设，低分辨率重建产生孔洞与纹理复制 → 以逐像素最小重投影、identity auto-mask、全分辨率多尺度采样改变训练目标。代价是多视训练数据、相机模型、位姿学习和训练显存；遗留问题是动态场景、辐射变化、弱纹理与域外泛化。它与 SGM／PSMNet 的推理时双目对应、ICP 的已有几何配准不同，不是三维测量全链替代品。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-06**。唯一键 `monodepth2.md:MD01`，仅一张卡，承接 **G41／10-08／E08**。采用 Godard、Mac Aodha、Firman、Brostow 的 **ICCV 2019** 论文 *Digging Into Self-Supervised Monocular Depth Estimation*，不将最终机制反推为全部属于2018预印本初版。作者代码锁定 `nianticlabs/monodepth2` 提交 **`19e16e1c29e103a5a7c6e2f20dc9ac33210aa396`，2026-11-02**；该次提交改 README 题名，是本次阅读快照，**不是2019实验现场代码或已验证运行环境**。[cite:1](#source-1)[cite:3](#source-3)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：论文方法主链、固定作者网络、投影、损失与训练／推理分支 |
| L2 机制 | 已核验：min reprojection、identity automask、tie 噪声、全分辨率采样及平滑因果 |
| L3 工程权衡 | 部分：原配置、单位、数据／内参契约与诊断顺序；无真实数据定参 |
| L4 数学推导 | 已核验：限定针孔／局部可微域的投影、采样、门控、归一化梯度与尺度不辨识；无通用误差界 |
| L5 实现 | 部分：固定源码对应、教学伪代码、有限代数／坐标检查；本仓库未实现 Monodepth2 |
| 面试自测 | 答5：贯穿 L1–L5，含答案、成立条件、易错点和证据入口 |
| 验证范围 | 既有联网方法／代码阅读、Python 标准库双精度与60位Decimal有限差分及静态 shape／分支检查、文档与单文件边界检查；本次修订未重新联网、未训练、未加载权重、未运行网络或数据集评测 |

机制依据属于同一作者论文／代码体系，标记 **[SINGLE-SOURCE]**；CVF PDF、同文 HTML 与作者仓库不是三份独立性能证据。下文严格区分原文、固定代码、条件推导与工程建议；不提供本仓库 KITTI 成绩或米／毫米测量承诺。此次检索仓库 Python 源码的 `monodepth`、`Monodepth`、`BackprojectDepth`、`Project3D` 未发现专用实现，既有二维模块不升级为深度模型。

## L1 步骤：训练重建链与单图推理

### 三种训练模式与依赖

| 模式 | 训练源集合，目标为时刻 t | 位姿来源 | 推理与尺度边界 |
|---|---|---|---|
| M：单目视频 | `frame_ids=[0,-1,1]`，源为前／后帧 | 两次共享 pose 网络前向，预测各对相机变换 | 单图深度网络；深度与平移共同缩放的歧义仍在 |
| S：仅双目 | `frame_ids=[0]` 加 `use_stereo`，源为 `s` | 已知 `stereo_T`；不建立 pose 网络 | 仍是单图推理；训练基线可锚定尺度，但不是任意新相机的度量保证 |
| MS：单目＋双目 | `[0,-1,1]` 加 `s` | 时序变换由 pose 预测，双目用已知外参 | 单图推理；须区分双目尺度约定与跨域有效性 |

固定 README 的 S 示例使用 `eigen_full`，M／MS 默认使用 `eigen_zhou`；数据划分不同，不应只按训练时图像数量解释全部效果。S 仅有一个重投影源，源间 `min` 此时退化为该源本身。固定代码的 S 默认仍启用 identity automask；它不是“所有双目训练必关掩膜”的实现。[cite:3](#source-3)[cite:4](#source-4)[cite:9](#source-9)

1. 冻结训练模式、权重、数据 split、内外参、图像尺寸和库版本；目标与源按真实时序／同步关系配对，检查有限值和足够的视差／运动激励。
2. 生成各尺度 `color` 与 `color_aug`。同一样本各帧使用相同颜色增强与水平翻转；网络输入用增强图，光度误差用未做颜色增强、但经过同样几何处理的图。
3. 深度网络只对目标图前向，产生尺度 `s=0,1,2,3` 的 sigmoid 输出 `q_s:[N,1,H/2^s,W/2^s]`。
4. M／MS 对每个时序源估计相机变换。负帧输入顺序为 `[I_(t-1),I_t]`，正帧为 `[I_t,I_(t+1)]`；对负帧的预测矩阵取逆，统一得到目标到源 `T_(t→r)`。
5. 默认对每个 `q_s` 双线性插值到 H×W，再映射为深度；用目标像素射线和深度反投影，变换到源相机，投影为源像素坐标，反向采样源图以重建目标。
6. 逐源计算 SSIM＋L1 光度误差；与未 warp 源图的 identity 误差比较，逐像素选最优分支。对原尺度 `q_s` 计算均值归一化 edge-aware smoothness；聚合各尺度后反传，同时更新深度与时序 pose 参数。
7. 按固定训练预算停止，以独立验证集检查泛化与选模。推理则只执行目标图深度网络、读取 `q_0`、按指定协议转深度／恢复空间尺寸，一次前向结束；可选翻转后处理需要额外一次前向。

步骤1中的有限值、运动激励与拒绝门是工程建议，作者代码没有完整实现这些守卫。步骤2–7的实际调用对应 `process_batch → predict_poses → generate_images_pred → compute_losses`；不是在线优化每张测试图的深度和 pose。[cite:4](#source-4)[cite:10](#source-10)[cite:11](#source-11)

### 深度网络与 pose 网络

以下展开默认 ResNet18；N 为 batch，H、W 为网络输入尺寸，作者训练器要求均为32倍数。使用论文典型 H=192、W=640，特征形状均不省略通道；没有实际分配或运行这些张量。[cite:2](#source-2)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)

| 路径 | 结构与数据布局 | 192×640 时的空间尺度 |
|---|---|---|
| 深度编码器 | RGB `[N,3,H,W]`；内部统一 `(image−0.45)/0.225`；ResNet18 五级特征通道64、64、128、256、512 | 96×320、48×160、24×80、12×40、6×20 |
| 深度解码器 | i=4→0；3×3卷积＋ELU，nearest 上采样2倍，与相应 encoder skip **按通道拼接**，再3×3＋ELU | 解码通道依次256、128、64、32、16；最终192×640 |
| 深度输出头 | 在 i=3、2、1、0 各用反射 padding 的3×3卷积到1通道，再 sigmoid | `q_3`24×80，`q_2`48×160，`q_1`96×320，`q_0`192×640 |
| pose 编码器 | 独立于深度编码器；两张 RGB 按通道拼成 `[N,6,H,W]`；同一 pose 网络复用于两对时序帧 | 最深特征 `[N,512,6,20]`；ImageNet 首层权重复制两份后除2 |
| pose 解码器 | 最深特征1×1压到256＋ReLU；两层3×3、256通道＋ReLU；最终1×1输出，空间均值，再乘0.01 | 分成 axis-angle 与 translation，各3维；轴角模长以弧度计，平移单位随训练尺度 |

**论文／代码细节：**论文描述每对输入预测一个6-DoF pose；固定默认 `PoseDecoder(..., num_frames_to_predict_for=2)` 实际输出 `[N,2,1,6]`，`predict_poses` 的 pairs 分支仅取 `[:,0]` 构造该对 T，不能误把另一个槽当另一相邻帧的估计。`shared`、`posecnn`、`all` 是可选分支，本卡主伪代码限定默认 `separate_resnet+pairs`；尤其 `posecnn` 还会用均值逆深度缩放平移，不能混入默认公式。[cite:4](#source-4)[cite:8](#source-8)

skip 拼接不是残差相加；decoder 的 nearest 上采样与损失前的 bilinear 上采样不是同一算子。深度和 pose 都可接 ImageNet 初始化，但其任务输出不受深度／pose GT 监督；预训练有监督不等于本任务使用深度标签。默认 sigmoid 有界参数化也不等于模型知道真实单位。

## L2 机制：遮挡、identity 掩膜与多尺度

### 最小重投影不是显式可见性求解

对每个目标像素，先得到各源的完整 SSIM＋L1 误差，再沿**源视图维**取最小值。其动机是：目标像素可能在前帧被挡住、在后帧可见，平均误差会迫使正确深度也解释不可见观测；最小误差允许选择较好源。不能先把重建 RGB 平均再算误差，也不能每幅图统一选一帧，更不能分别对 RGB 通道选不同源。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)

它没有 z-buffer 或遮挡真值，不保证误差最小的源一定可见。所有源都遮挡、反射、高光、重复纹理、相机曝光变化时，错误对应也可能误差小。只有一个源时，min 无法从别的视角补救；它也不是左右深度一致性损失、光流估计或运动分割。

### identity automask 与 tie 噪声

identity 误差比较的是目标图与**未变形的源图同一像素位置**，不是目标图与自己比较，更不是在预测深度图上执行恒等变换。论文用严格条件“最小 warp 误差 < 最小 identity 误差”决定保留光度监督，去除没有重建改善的像素。相机静止、物体与相机近似共速及低纹理区域常被筛掉；并非所有动态物体都能筛掉，也可能移除合法但信息弱的像素。[cite:2](#source-2)

固定代码将 `identity_errors` 放在前、`reprojection_errors` 放在后，拼接后执行一次 `torch.min(dim=1)`；先给每个 identity 候选加 `randn * 0.00001`。这是均值0、标准差 `1e−5` 的高斯微扰，只加在 identity **损失候选**，不加在图像、深度或 warp 误差上；作用是打破精确／近似相等的竞争，不是物体运动阈值或概率校准。它有正有负，所以相等时可选到 warp，也可选到 identity，不能说静止像素在代码中必定100%屏蔽。[cite:4](#source-4)

`identity_selection/s` 实际标记的是 **warp 被选中为1**，不要被变量名误导。该输出用于记录，固定代码并没有再拿它乘一次光度损失；梯度通过最小值选中的分支传播。原文乘掩膜与代码联合 min 的数值／梯度差别在 L4 明列。

### 全分辨率多尺度与 edge-aware smoothness

| 设计 | 机制与删除／替换后果 | 不能推出的结论 |
|---|---|---|
| 多尺度预测 | 深／浅 decoder 头都获监督；粗头可约束较大空间支持 | 多个头不是多张真实深度标签，也不是多尺度结果平均后推理 |
| 全分辨率 photometric sampling | 每个粗头先恢复到输入 H×W，再使用原分辨率纹理重投影；避免在低分辨率模糊图上“钻空子” | 不增加粗头真实分辨率，不保证没有局部极小值或纹理复制 |
| 边缘感知平滑 | 图像梯度大处降低逆深度平滑权重，允许深度边；低纹理处提供正则 | 纹理边未必是深度边；无纹理物体边也可能被抹平 |
| 每图均值归一化 | 抑制仅靠整体缩小被平滑量来降低正则的自由度 | 不等于绝对深度归一到米，也不消除光度模型的尺度歧义 |
| 反射 padding／border sampling | 前者用于 decoder 和 SSIM 局部窗口，后者在 warp 越界时复制最近边界 | 边界复制不是可见性检测；不能拿 padding 的数值当观测真值 |

默认实现上采样的是 `q_s`，之后才 `disp_to_depth`；不是先取倒数得到 D 再插值。论文正文概括为上采样 depth maps，重实现必须保留这个具体区别，因为倒数和插值不交换。平滑项则用**原尺度**的 `q_s/mean(q_s)` 与该尺度未增强颜色图，而非所有平滑项都在全分辨率计算。`v1_multiscale` 将光度采样改回各尺度；`avg_reprojection` 改变源聚合；它们是消融开关，不是无损加速。[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)

## L3 工程权衡：坐标、训练与尺度协议

### 内参、颜色和数据边界

固定 loader 用 PIL RGB、`ToTensor` 到[0,1]；编码器内部才做标量归一化，不能再叠加一套 ImageNet 三通道 mean／std。颜色增强亮度／对比度／饱和度范围为0.8–1.2，色相−0.1–0.1，以50%概率启用；各帧同参数。水平翻转独立以50%概率启用，并调整双目平移符号。光度比较不使用颜色增强结果，否则会让人为辐射扰动进入监督。[cite:2](#source-2)[cite:6](#source-6)[cite:10](#source-10)

KITTIDataset 的归一化内参固定为首两行 `[0.58,0,0.5,0]`、`[0,1.92,0.5,0]`；MonoDataset 再分别乘当前尺度宽／高，保存4×4 K 和伪逆。它不是逐图读取完整标定矩阵的通用相机支持。相机主点被假设居中，换数据不能原封不动照抄这些数值；非中心主点尤其要核对翻转或禁用该增强。一般 crop／resize 应按实际像素映射更新焦距与主点，所有尺度 K 与图像坐标必须一致。[cite:10](#source-10)

默认 `source_scale=0`，所以四个光度重建都使用 H×W 图像、K0、inv_K0，不能用原始磁盘尺寸的 K，也不能误用 K_s。作者 `BackprojectDepth` 固定 batch、高宽构造栅格，训练／验证 loader 都 `drop_last=True`；最后小 batch 若改为保留，必须重建网格或按 batch 广播，不可直接复用旧 shape。[cite:4](#source-4)[cite:5](#source-5)

### 参数与诊断顺序

| 参数／症状 | 单位、证据与优先动作 | 权衡与边界 |
|---|---|---|
| 默认训练配置 | H×W=192×640、batch12、20 epochs；Adam，前15轮学习率1e−4，余下1e−5；scales=0,1,2,3 | 原文／作者默认，不是现场最优值；旧 StepLR 在 epoch 起始调用的行为须随库版本核对 |
| 深度映射上下限 | `min_depth=0.1,max_depth=100`；这是网络深度坐标的参数化范围 | 纯单目不因此获得米；评估截断范围另为1e−3到80，不能混成训练输出范围 |
| 光度与平滑权重 | α=0.85；λ=1e−3，各尺度平滑再除2^s | [0,1]颜色输入下的约定；改变动态范围、损失归约或有效分母均改变相对强度 |
| 全图 warp 错位 | 先查目标→源方向、负帧矩阵求逆、K缩放、栅格x/y顺序、align_corners | 不先增大平滑或训练轮数掩盖几何错误 |
| 大量 identity 胜出 | 看每尺度选中比例、静止帧、时间间隔、光度残差、pose及采样可见性 | 禁用 automask 会恢复错误监督；先确认是否真的缺运动信息 |
| 动态车深度过远／孔洞 | 对照共速运动、重复纹理和反射；按运动／边界分组评价 | min／mask不能建模所有物体运动；必要时换监督或显式运动模型，记为变体 |
| 细节复制到深度、薄物体消失 | 核对是否误开v1、q与D插值顺序、各尺度smooth与λ；再查域差 | 增分辨率增训练采样成本，减平滑也可能增噪声 |
| 新相机尺度偏差 | 检查训练M/S/MS、K、resize、权重元数据、基线约定及是否用了GT中位数缩放 | 缺独立尺度锚点时输出相对深度，不拟合一常数就宣称全域可测 |

配置依据为论文§3.3、固定 options、trainer 与 evaluate_depth；诊断动作是待执行工程设计。[cite:2](#source-2)[cite:4](#source-4)[cite:9](#source-9)[cite:11](#source-11)

### 单目、双目和推理缩放必须分开

- **M 的尺度：**任意正数 k 同时放大深度和时序平移，理想投影不变。学习到数据集大小先验或限制0.1–100数值范围，并不能把不可观测尺度变成可追溯单位。
- **S／MS 的尺度：**已知固定双目基线阻止这种共同自由缩放；但作者 loader 实际用名义基线0.1，README／评估器按 KITTI 基线0.54 m给深度乘5.4，并禁用 median scaling。5.4是该权重／训练协议的换算，不是任何单目模型、任何双目相机的通用常数。[cite:3](#source-3)[cite:10](#source-10)[cite:11](#source-11)
- **M 的评估：**默认逐图以有效GT中位数除预测中位数作缩放，再计算误差。这是有GT的离线尺度对齐，不是无需GT的在线米制深度能力；指标须注明 split、crop、深度范围、是否 median scaling、是否翻转后处理。[cite:11](#source-11)
- **空间恢复：**`evaluate_depth` 对尺度0的 scaled inverse depth 用 `cv2.resize` 恢复GT尺寸后取倒数；不是视差resize后乘宽度比例。这里预测的是逆深度参数，不是像素位移；若另要双目像素视差，才需结合相应像素焦距和基线推导。[cite:5](#source-5)[cite:11](#source-11)
- **单图脚本：**`test_simple` 使用权重存储的 feed 高宽，LANCZOS缩图、网络前向；`.npy` 保存在feed尺寸，只有着色预览恢复原图尺寸。`--pred_metric_depth` 对纯M模型仅打印警告，仍可能写乘5.4的数组，文件名／开关不构成米制认证；本卡建议缺尺度时拒绝度量标签。[cite:12](#source-12)

推理不显式输入K并不表示焦距变化无影响，网络仍依赖训练成像分布。若用于尺寸测量，必须另有标定、独立尺度约束、物理真值及域外拒绝验证；当前没有这些验收证据。

## L4 数学推导：投影、采样、门控和梯度

### 有界逆深度与相机投影

用 q 表示 sigmoid 输出，避免与像素视差 d 混淆。给定正上下限 Dmin、Dmax，固定代码定义：

\[
b=1/D_{max},\quad a=1/D_{min}-1/D_{max},\quad
\rho=b+aq,\quad D=\rho^{-1},\quad
\frac{\partial D}{\partial q}=-aD^2.
\]

若 q=σ(l)，则 \(\partial D/\partial l=-aD^2q(1-q)\)。上下限默认得到 b=0.01、a=9.99；sigmoid饱和会改变梯度幅度，有限正输出不等于有效几何。若 U 为固定双线性插值算子，训练使用 \(D_s=1/(b+aUq_s)\)，局部梯度为 \(U^T[-aD_s^2\odot\nabla_{D_s}L]\)。这一般不等于插值已转换的深度 \(U[1/(b+aq_s)]\)。[cite:4](#source-4)[cite:5](#source-5)

**理想针孔式（不含 ε）：**目标像素齐次坐标 \(\tilde p=(x,y,1)^T\)，内参Kt、Kr，目标到源变换 \(T=[R\mid t]\)。忽略畸变，采用列向量及零skew内参：

\[
v=K_t^{-1}\tilde p,\quad X_t=D(p)v,\quad
X_r=RX_t+t=(X,Y,Z)^T,\quad
x_r=f_xX/Z+c_x,\quad y_r=f_yY/Z+c_y.
\]

这一步将**目标像素**映到源图，再读取 \(I_r(x_r,y_r)\) 填入目标重建 \(\hat I_t(p)\)，即 inverse warp／反向采样；不是将源像素前向散射到目标。矩阵逆必须为 \(T^{-1}=[R^T\mid -R^Tt]\)，不能只把t取负。固定函数 `invert=True` 对R转置后采用 `R @ Translation(-t)`，而正向为 `Translation(t) @ R`。[cite:5](#source-5)

令 \(w=Rv=(w_x,w_y,w_z)^T\)，在 Z>0、没有越界／离散切换处：

\[
\frac{\partial(x_r,y_r)}{\partial D}
=\left(\frac{f_x(w_xZ-Xw_z)}{Z^2},
\frac{f_y(w_yZ-Yw_z)}{Z^2}\right)^T,\quad
J_\pi=\begin{bmatrix}f_x/Z&0&-f_xX/Z^2\\0&f_y/Z&-f_yY/Z^2\end{bmatrix}.
\]

有 \(\partial p_r/\partial t=J_\pi\)。若采用源相机坐标下左乘小旋转扰动，\(\delta X_r=-[RX_t]_\times\delta\omega\)，旋转雅可比为 \(-J_\pi[RX_t]_\times\)。这是指定扰动约定的局部关系，不是无需变换就能直接代替作者 axis-angle 坐标的全局导数；完整实现需经 Rodrigues 与参数化的链式法则。

**固定代码式（ε=1e−7）：**`Project3D` 先计算 `P=(K @ T)[:, :3, :]`，再以 `P @ points` 得到 \((f_xX+c_xZ,f_yY+c_yZ,Z)^T\)，最后把前两项除以第三项加 ε。因而在栅格归一化之前，实际像素坐标为：

\[
x_r^\epsilon=\frac{f_xX+c_xZ}{Z+\epsilon},\qquad
y_r^\epsilon=\frac{f_yY+c_yZ}{Z+\epsilon}.
\]

固定K、R、t与 ε，对D求导时 \(\partial(X,Y,Z)/\partial D=w\)，商法则给出：

\[
\frac{\partial x_r^\epsilon}{\partial D}
=\frac{(f_xw_x+c_xw_z)(Z+\epsilon)-(f_xX+c_xZ)w_z}{(Z+\epsilon)^2}
=\frac{f_x[w_x(Z+\epsilon)-Xw_z]+c_x\epsilon w_z}{(Z+\epsilon)^2},
\]
\[
\frac{\partial y_r^\epsilon}{\partial D}
=\frac{f_y[w_y(Z+\epsilon)-Yw_z]+c_y\epsilon w_z}{(Z+\epsilon)^2},\qquad
J_\pi^\epsilon=\begin{bmatrix}
f_x/(Z+\epsilon)&0&(c_x\epsilon-f_xX)/(Z+\epsilon)^2\\
0&f_y/(Z+\epsilon)&(c_y\epsilon-f_yY)/(Z+\epsilon)^2
\end{bmatrix}.
\]

因此固定代码的平移／上述旋转局部雅可比须分别使用 \(J_\pi^\epsilon\) 与 \(-J_\pi^\epsilon[RX_t]_\times\)。它不是 \(f_xX/(Z+\epsilon)+c_x\)，也不能只替换理想导数的分母或漏掉 \(c_x\epsilon w_z\)、\(c_y\epsilon w_z\)。ε趋于0才恢复理想式。这个补丁不拒绝负Z、零深度或NaN，Z=−ε仍奇异；也使理想尺度不变量在数值上不再绝对精确，不能用它证明相机后方点合法。[cite:5](#source-5)

### 双线性采样与坐标约定

设 \(x_r=i+u,y_r=j+v\)，i、j是floor索引，0<u,v<1。每个RGB通道：

\[
\hat I=(1-u)(1-v)I_{00}+u(1-v)I_{10}+(1-u)vI_{01}+uvI_{11},
\]
\[
\partial_u\hat I=(1-v)(I_{10}-I_{00})+v(I_{11}-I_{01}),\qquad
\partial_v\hat I=(1-u)(I_{01}-I_{00})+u(I_{11}-I_{10}).
\]

因此 \(\partial\hat I/\partial D=\nabla_{p_r}\hat I\cdot\partial p_r/\partial D\)，再连到深度参数；pose也通过同一采样链收到梯度。平坦邻域图像梯度接近0，即使几何公式可导，也缺乏有效光度训练信号。整数边界、padding切换处为分段可导，不能在这些点用普通光滑中心差分强求同一导数。

固定 `Project3D` 输出 `[N,H,W,2]`，最后轴顺序是x、y，归一化为 \(g_x=2x_r/(W-1)-1\)、\(g_y=2y_r/(H-1)-1\)。这与 `align_corners=True` 的像素中心映射一致；`grid_sample(...,padding_mode="border")` **未显式传 align_corners**，旧版依赖环境与现代默认可能不同。迁移时应显式冻结与该栅格相容的约定并对照旧运行时，而不是把损失前 `interpolate(...,align_corners=False)` 的参数复制到 grid_sample。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

若把上述gx喂给False约定，内部解释为 \(x'=(g_x+1)W/2-1/2\)，偏差 \(x'-x=x/(W-1)-1/2\)，从左端−0.5到右端+0.5像素变化。H或W=1又会使源码归一化除零。border复制越界值时，对继续向外的采样坐标可能没有有用梯度，不是“边界不会影响训练”。

### 光度损失、min 与 identity 分支

令 \(E_r(p)=pe(I_t,\hat I_{r\to t})(p)\)、\(A_r(p)=pe(I_t,I_r)(p)\)。固定代码先按通道平均：

\[
pe(I,J)=0.85\,\operatorname{mean}_c\!\left[\operatorname{clip}\left(\frac{1-SSIM_c(I,J)}2,0,1\right)\right]
+0.15\,\operatorname{mean}_c|I-J|.
\]

SSIM用3×3均值窗口、stride1、reflection padding，C1=0.01²、C2=0.03²，方差按 \(E[x^2]-E[x]^2\) 而非无偏样本估计。均值μ、方差σ²、协方差σIJ满足：

\[
SSIM=\frac{(2\mu_I\mu_J+C_1)(2\sigma_{IJ}+C_2)}
{(\mu_I^2+\mu_J^2+C_1)(\sigma_I^2+\sigma_J^2+C_2)}.
\]

其梯度通过局部均值／二阶矩的乘除链传播；对m=9的窗口样本Jk，有 \(\partial\mu_J/\partial J_k=1/m\)、\(\partial\sigma_J^2/\partial J_k=2(J_k-\mu_J)/m\)、\(\partial\sigma_{IJ}/\partial J_k=(I_k-\mu_I)/m\)，再对分式求导并累积重叠窗口。clamp饱和区梯度为0，折点依具体子梯度约定。L1非零残差的导数是sign，再除通道数；这不是单点RGB与深度一一独立的梯度。[cite:5](#source-5)

原文的掩膜写法与固定实现分开列：

\[
E=\min_r E_r,\quad A=\min_r A_r,\quad
\mu=\mathbf1[E<A],\quad \ell_{paper}=\mu E;
\]
\[
\eta_r\sim\mathcal N(0,10^{-10}),\quad
\ell_{code}=\min(A_1+\eta_1,\ldots,A_R+\eta_R,E_1,\ldots,E_R).
\]

这里 `1e−5` 是标准差，方差为1e−10。在远离切换边界、固定噪声且无并列时，令j为唯一获胜分支：\(\partial\ell_{code}/\partial E_r=\mathbf1[j=R+r]\)。若identity胜出，它不依赖深度／pose参数，该像素的**直接光度项**对这些参数的梯度为0；若warp胜出，仅获胜源的该项反传。比较／argmin索引不作为可微运动估计；并列点使用框架选支／子梯度约定，噪声会影响可复现性。[cite:2](#source-2)[cite:4](#source-4)

**数值不等价：**不加噪声时，identity胜出的代码损失为A而论文掩膜损失为0；局部对参数的梯度可以相同，损失值并不相同。加入噪声后近似并列区域还会改变选择；不能用两者日志数值逐项相等验收。作者用全batch／全图像素的 `.mean()`，不是除以 \(\sum\mu\)；若改成仅有效点平均，梯度尺度及λ相对权重都会变化。被掩掉的像素仍可经相邻SSIM窗口、smooth项和共享网络参数受影响，不能声称其预测永远零梯度。

### 原尺度平滑、均值归一化与总目标

固定代码对每张图、每个尺度的**sigmoid输出q_s**计算均值，不是对整个batch取均值，也不是对已映射ρ或深度D平滑。令n_s为该尺度像素数，\(m_s=\frac1{n_s}\sum_iq_{s,i}+10^{-7}\)，\(\bar q_{s,i}=q_{s,i}/m_s\)：

\[
S_s=\operatorname{mean}_{x\text{邻边}}\big[|\Delta_x\bar q_s|e^{-\operatorname{mean}_c|\Delta_x I_t^s|}\big]
+\operatorname{mean}_{y\text{邻边}}\big[|\Delta_y\bar q_s|e^{-\operatorname{mean}_c|\Delta_y I_t^s|}\big],
\]
\[
L_{code}=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}
\left[\operatorname{mean}_{N,H,W}\ell_{code,s}+\frac{10^{-3}}{2^s}S_s\right].
\]

横／纵差分各取其实际邻边集合均值，最右列／最下行没有绕回邻居。论文式(3)写均值归一化 inverse depth；代码采用q而非带非零偏置b的ρ，故 \(q/\operatorname{mean}q\) 与 \(\rho/\operatorname{mean}\rho\) 不完全一样，应按固定代码描述。[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)

若 \(h_i=\partial S/\partial\bar q_i\) 是加权绝对差分反传得到的梯度，则：

\[
\frac{\partial\bar q_i}{\partial q_j}=\frac{\delta_{ij}}m-\frac{q_i}{n m^2},\qquad
\frac{\partial S}{\partial q_j}=\frac{h_j}m-\frac{\sum_i h_iq_i}{n m^2}.
\]

均值分母**不detach**，否则第二项丢失。绝对差非零时，邻边 \(w|\bar q_j-\bar q_i|\) 对两端贡献分别为 \(-w\operatorname{sign}(\bar q_j-\bar q_i)\) 和其相反数，随后带上各邻边均值分母。忽略epsilon时q整体正比例缩放不改变该正则；有epsilon只是近似不变，且这不是度量尺度监督。

### 可辨识性、复杂度与保证边界

在理想针孔投影中，把 \(D'=kD,t'=kt\)，保持R不变，则 \(RX_t'+t'=k(RX_t+t)\)，齐次除法消去k。M光度目标无法辨识这个自由度；S的外部固定平移不允许任意同乘k，提供不同约束。若只有纯旋转t=0，\(X/Z\) 与D无关，理想深度投影导数直接为0；固定 ε 式则残留 \(\partial x_r^\epsilon/\partial D=\epsilon(f_xw_x+c_xw_z)/(Dw_z+\epsilon)^2\)，y同理，这只是数值正则引入的依赖，不是物理尺度信息。静止、纹理平坦和非常小的视差同样削弱监督。平滑、深度范围、网络先验可偏好某个输出，但不能据此声称恢复了真实单位。

设源数R、尺度数S，默认训练重建／误差计算量约为 O(NSRHW)，且四个尺度都在H×W上采样，不是各尺度面积等比缩小之和；网络卷积成本和反传激活另计。推理仅深度网络及转换，不包含pose与多源重投影。分辨率、batch、源数增加会增加成本；实际显存峰值、速度与精度必须实测。

训练目标非凸，min分支可切换，光度／静态场景假设会失效；**未提供本场景通用理论误差界、全局最优保证或可追溯测量精度**。有限差分仅核对局部数学，不能证明真实训练收敛、泛化或动态遮挡正确。

## L5 实现：源码对应、伪代码与有限检查

### 默认分支教学伪代码

以下是重实现设计，不是可导入模块或已接入训练器。限定 `separate_resnet+pairs`、`scales=[0,1,2,3]`、automask启用、非v1、非avg、无predictive mask；SSIM、decoder与K处理遵循本卡前文。`REQUIRE`为额外工程守卫，`GRID_SAMPLE`显式True是与作者端点栅格相容的迁移选择，仍需旧运行时对照验证。

```text
INPUT target_rgb, source_rgbs, K0, inv_K0, mode, stereo_T, training
IF NOT training:
    image = RESIZE_RGB_TO_WEIGHT_FEED_SIZE(target_rgb)
    q0 = DEPTH_DECODER(DEPTH_ENCODER(TO_TENSOR(image)))[0]
    rho0 = 0.01 + 9.99 * q0
    REQUIRE ALL_FINITE(rho0) AND ALL(rho0 > 0)
    RETURN rho0, 1 / rho0, scale_status_from_training_contract
REQUIRE H mod 32 == 0 AND W mod 32 == 0 AND NONEMPTY(sources)
color, color_aug, K, inv_K = PAIRED_PYRAMIDS_AND_INTRINSICS(inputs)
REQUIRE FINITE_IMAGES_AND_CAMERA_MATRICES(color, K, inv_K)
q = DEPTH_DECODER(DEPTH_ENCODER(color_aug[target, 0]))
FOR r IN temporal_sources:
    pair = [r, target] IF r < 0 ELSE [target, r]
    aa, tr = POSE_DECODER(POSE_ENCODER(CONCAT_CHANNEL(color_aug[pair, 0])))
    T[r] = TRANSFORM(aa[:, 0], tr[:, 0], invert=(r < 0))
IF mode IN [S, MS]:
    T[stereo_source] = stereo_T
total = 0
FOR s IN [0, 1, 2, 3]:
    q_full = BILINEAR(q[s], size=[H,W], align_corners=False)
    D = 1 / (0.01 + 9.99 * q_full)
    points = BACKPROJECT(D, inv_K[0])
    E, A = [], []
    FOR r IN sources:
        grid = PROJECT(points, K[0], T[r], z_epsilon=1e-7)
        warped = GRID_SAMPLE(color[r,0], grid, mode=bilinear,
                             padding=border, align_corners=True)
        E.APPEND(PHOTO_ERROR(warped, color[target,0]))
        A.APPEND(PHOTO_ERROR(color[r,0], color[target,0]))
    E = CONCAT_SOURCE(E)
    A = CONCAT_SOURCE(A)
    noise = RANDN_LIKE(A) * 1e-5
    candidates = CONCAT_SOURCE([A + noise, E])
    REQUIRE ALL_FINITE(candidates)
    best, index = MIN_WITH_INDEX(candidates, axis=source)
    warp_selected = index >= NUMBER_OF_SOURCES
    q_norm = q[s] / (MEAN(q[s], axes=[height,width], keepdim=True) + 1e-7)
    smooth = EDGE_SMOOTH(q_norm, color[target,s])
    total += MEAN(best) + 1e-3 * smooth / (2 ** s)
loss = total / 4
REQUIRE ISFINITE(loss)
ZERO_GRAD()
BACKWARD(loss)
STEP_OPTIMIZER()
```

`warp_selected`只记录、不再乘best；identity损失独立于待优化参数。纯S没有时序源，pose循环为空；MS中的stereo_T不由pose网络覆盖。几何前后必须保留计算图，不能把D／T／grid转成NumPy、整数坐标或detach后再采样。若新增Z>0／越界有效掩膜、只对有效点归一或低有效率跳过batch，这些均改变作者原目标，应单独命名／对照，不偷偷写成既有功能。

### 固定源码与移植风险

| 入口 | 已核对内容 | 仍需真实运行验证 |
|---|---|---|
| `networks/resnet_encoder.py`、`depth_decoder.py` | 标量输入归一、ResNet多图首层、skip通道拼接、nearest、四个sigmoid头 | 权重版本、每层shape、现代torchvision预训练API与参数装载 |
| `networks/pose_decoder.py`、`trainer.predict_poses` | 空间均值、0.01缩放、pairs取第0槽、负帧求逆 | 轴角零邻域、旋转次序、两帧顺序、完整pose参数梯度 |
| `layers.BackprojectDepth/Project3D` | `[N,4,HW]`点、K@T、Z+epsilon、xy栅格与固定batch | H/W端点、正Z与负Z、dtype/device、非有限值、真正grid_sample反传 |
| `trainer.compute_losses` | concat源轴、identity噪声、min、全像素均值、原尺度smooth／2^s | 固定随机种子后的分支／梯度、AMP、并列、SSIM邻域耦合 |
| `datasets/mono_dataset.py`、`kitti_dataset.py` | 同步增强、K缩放、左右／flip下0.1平移符号 | 自定义相机主点／畸变、同步与失配帧、现代Pillow／ColorJitter API |
| `evaluate_depth.py`、`test_simple.py` | GT crop、median scaling、5.4、feed尺寸npy与原尺寸预览 | 权重／数据完整性、相对／绝对尺度分开报告、禁止误标单位 |

来源全部锁定相同SHA，不将主分支后续变动隐含纳入。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)[cite:10](#source-10)[cite:11](#source-11)[cite:12](#source-12) README声明原实验环境PyTorch0.4.1、CUDA9.1、Python3.6.6、Ubuntu18.04；它不是当前Windows环境可直接运行的证明。代码声明非商业使用许可，实际集成前须另核完整LICENSE及使用授权，本卡不复制作者完整实现或给出商业授权结论。[cite:3](#source-3)

### 本次已执行的有限检查

2026-11-06本次修订在现有 `.venv/Scripts/python.exe`（Python 3.13.14）中以内存命令复跑下表全部检查，仅使用标准库math、decimal等和列表；无新增脚本／依赖，无torch网络运行。下面数值是**人为选取的解析测试点**，只用于检验代数恒等式、索引和局部差分，不是合成训练数据、论文实验或精度测量。原有检查使用53位有效二进制位的float，中心差分步长h=1e−5，导数绝对误差阈值1e−7；避开min、abs、采样格点等折点。

新增固定 ε 投影检查保持源码 ε=1e−7与非零主点cx/cy=320/96，沿用理想投影行其余输入。为可靠区分小量，不使用float32，也不以过小float步长对约445／119的坐标相减；所有输入由十进制字符串构造 `Decimal`，精度60位，中心差分h=1e−12，绝对误差阈值1e−22。前向独立按 `X_r=Dw+t → KX_r → 前两项/(第三项+ε)` 计算，两轴分别与解析导数比较；另将解析式中的主点 ε 项删除作负对照，断言错误式与差分的偏差均大于1e−7。高精度检查验证数学表达式，不代表源码float32／AMP实际数值误差。

| 检查及精确输入约定 | 实际结果 | 能证明／不能证明 |
|---|---|---|
| D(q)，q=0.4、a=9.99、b=0.01 | D=0.2496255617，dD/dq=−0.6225060811；差分绝对差3.89415610869e−10 | 映射导数正确，不说明这是真实距离 |
| 理想投影，D=4，Rv=(0.2,0.1,1)，t=(0.3,−0.2,0.4)，fx/fy=500/510，cx/cy=320/96 | 投影(445,119.1818182)，对D导数(−5.6818182,6.3223140)；最大差1.41591893765e−9 | 针孔局部导数；未含Z+epsilon、畸变或网络pose |
| 固定 ε 投影，沿用上一行输入，ε=1e−7；Decimal设置见前文 | 投影约(444.999989886363866,119.181815473140557)，对D导数约(−5.681815754132339,6.322314521506361)；x/y差分绝对差2.9348220537e−25／3.2656581788e−25 | 两轴断言通过，验证先乘K再除Z+ε；不是理想式换分母或真实tensor反传 |
| 同一固定式的漏主点项负对照 | 漏项大小cxεwz/(Z+ε)²≈1.6528924869e−6，cy项≈4.9586774606e−7；错误导数与差分的偏差两轴均大于1e−7，拒绝断言通过 | 非零主点用例能检出旧公式；没有人为增大ε掩盖精度问题 |
| 同一理想投影，D和t同时乘7 | 本次float坐标最大差0（阈值1e−10），断言通过 | 理想尺度不变量；不认证浮点epsilon下严格不变 |
| 双线性4角值0.1/0.4/0.2/0.8，u=0.3、v=0.4 | 值0.266，du/dv梯度0.42/0.19；最大差9.75108882528e−13 | 内部单格采样；未测边界、真实grid_sample或SSIM |
| identity／warp分支四例 | A=[0.2,0.3]、E=[0.1,0.4]选warp；E=[0.4,0.5]选identity；单源A=E=0.2时，identity固定扰动+1e−5选warp、−1e−5选identity | 四例全部通过；固定正负扰动是选支检查，不冒充高斯分布统计试验 |
| `min(0.2,E)`局部导数 | E=0.1时0.9999999999996122，E=0.4时0；最大差3.87800902502e−13 | 门控梯度；没有在tie处声称可导 |
| 三点归一smooth，q=[0.2,0.4,0.8]，邻边有效权重[0.7,0.6]，epsilon=1e−7 | 加权邻边和0.8142855398；梯度约[−2.0816320824,−0.3673467354,0.7040816064]；最大差1.00705666028e−10 | 验证含均值分母的链式项；权重已视为有效系数，未执行整图x/y均值或SSIM反传 |
| W=640、x=100，作者端点gx | True还原100；False还原99.6564945227，偏差−0.3435054773像素 | 静态坐标换算；不是已加载旧PyTorch对照 |
| q端点0.2和0.8的中点 | 先插值q再转D=0.1998001998；先转D再取均值=0.3114883630，不相等 | 倒数与插值不交换；不是深度插值效果实测 |
| 默认四头shape与基线比例 | `[1,1,192,640]`、`[1,1,96,320]`、`[1,1,48,160]`、`[1,1,24,80]`；0.54/0.1=5.4 | 尺寸／比值算术，未执行网络，不授予米制输出资格 |

各项断言通过，但未执行完整伪代码、SSIM有限差分、Rodrigues反传、真实tensor多尺度梯度、随机噪声复现、训练、权重推理或KITTI指标。因此L5保持部分；L4“已核验”限文中模型推导及这些局部检查，不宣称全框架自动微分验证。

### 后续正确性与优化验收

先固定旧环境或显式迁移语义，逐层对齐q、D、T、grid、warp、每源误差、min索引、smooth和梯度，再比较优化版本。必须补测单位变换／纯旋转、目标→源往返、正负时间帧、左右与flip符号、零旋转、负Z／NaN／空源、最后小batch、H/W非法尺寸、并列与噪声种子、border及SSIM反射支持。有限值检查不能替代“投影在物理上有效”。

缓存与广播固定像素网格、分尺度顺序归约、推理仅保留深度路径是候选优化；必须保证与参数无关的缓存不误缓存T或D。AMP在SSIM方差、Z除法、归一化分母和1e−5 tie竞争处尤其需要对照；源维分块若改变最小值并列次序，也可能改变梯度。不能用降低采样分辨率冒充等价全分辨率训练。

真实验收须按场景／序列隔离训练、验证和测试，记录相机与权重哈希、K／基线、split／crop／mask、median scaling、后处理、深度单位状态，报告相对误差、绝对尺度偏差、动态／遮挡／边界／距离分层及失败率；物理尺寸另需可追溯真值与不确定度。当前没有数据、权重、速度、显存峰值或尺寸达标结论。

## 五问自测：贯穿 L1–L5

| 层级／问题 | 参考答案要点 | 成立条件与易错答案 | 证据入口 |
|---|---|---|---|
| L1：三帧192×640如何训练，测试为何只要一张？ | 目标图出四尺度q；时序pairs预测T，负帧求逆；每尺度q上采样后转D，以K反投影／投影／反采样；多源光度与identity竞争，加原尺度smooth，联合反传；测试仅深度网q0 | M训练有相机内参与时序源，测试没有在线重建；错答是pose网也必须推理、四个深度求平均、disp直接是米 | L1；[cite:4](#source-4)[cite:7](#source-7)[cite:8](#source-8) |
| L2：min和automask分别解决什么，tie噪声加在哪里？ | min选误差较低源，减少遮挡源惩罚；identity检验warp是否优于不动；高斯标准差1e−5只加identity损失，打破并列；没有独立mask网 | 至少一个源可解释该点才有帮助；错答是显式遮挡真值、所有运动全剔除、identity为目标对自身、噪声必使warp获胜 | L2／L4；[cite:2](#source-2)[cite:4](#source-4) |
| L3：换相机后整体尺度错，能否给所有输出乘5.4？ | 先查M/S/MS、相机K与resize、feed尺寸／权重、基线约定；5.4只为0.1名义基线对应KITTI0.54m；M逐图GT中位数缩放是离线评价 | 有独立尺度证据才谈度量；错答是min_depth单位写米便可测、双目训练意味着任意场景米制、用GT对齐成绩证明在线尺度 | L3；[cite:3](#source-3)[cite:10](#source-10)[cite:11](#source-11)[cite:12](#source-12) |
| L4：梯度如何到深度／pose，纯旋转为何不能恢复深度？ | 选中源损失→SSIM/L1→双线性图像梯度→投影雅可比→D与T→q／轴角；smooth含均值分母链式项；理想式t=0投影与D无关，D/t共缩放不改变理想投影；固定ε残留不提供物理尺度信息 | 局部可微、正Z、非并列、纹理有效；错答是对argmin索引求连续梯度、identity胜出后所有路径都零梯度、优化收敛即尺度可辨识 | L4；[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5) |
| L5：新版框架重投影偏半像素、loss与论文公式不一致，如何检查？ | 对照Project3D端点栅格与grid_sample的align_corners；区分q插值False与采样True；检查identity联合min数值非掩膜零值、全像素分母、原尺度smooth、随机种子及负帧逆矩阵 | 固定SHA／环境后逐项对照再优化；错答是统一所有插值参数、只看最终loss、把边界复制当valid mask、读源码即复现 | L5；[cite:4](#source-4)[cite:5](#source-5)[cite:11](#source-11) |

## 来源、版本与获取深度

以下为2026-11-06实际联网记录。B=书目／提交元数据，T=方法正文，C=固定源码。CVF PDF读取到§3.1及式(3)附近后返回截断；随后尝试内存完整下载出现 `IncompleteRead`，不声称完整PDF逐页审阅。改从同文ar5iv HTML以内存解析补读§3.1–3.3及§4.1开头，已覆盖式(4)、式(5)、最终训练损失和实现配置。未独立审核全部实验、补充材料、历版预印本或许可全文。

<a id="source-1"></a>

**[cite:1] ICCV 2019 原文，B+T。** Clément Godard、Oisin Mac Aodha、Michael Firman、Gabriel Brostow，*Digging Into Self-Supervised Monocular Depth Estimation*。https://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf 。已读摘要、引言、相关工作、§3.1与重投影／光度／平滑开头；PDF返回范围与完整下载失败分开记录。

<a id="source-2"></a>

**[cite:2] 同文 HTML 转写，T，非独立实验。** https://ar5iv.labs.arxiv.org/html/1806.01260 。首轮页面返回在方法前截断，后用标准库HTTP与HTML解析在内存定位§3 Method至§4 Experiments，补齐§3.2的min、auto-mask、全分辨率采样、§3.3网络／训练配置；未以转写代替PDF图表视觉核对，不声称固定到了2018 v1。

<a id="source-3"></a>

**[cite:3] 固定作者提交与README，B+C。** 首先通过 https://api.github.com/repos/nianticlabs/monodepth2/commits/master 获取SHA与日期，再锁定 https://github.com/nianticlabs/monodepth2/commit/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396 。README：https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/README.md 。实际读其raw全文，核对2019论文归属、旧依赖、三模式命令、数据split、基线0.1／0.54与5.4、禁用median scaling及非商业使用声明；未下载模型／数据或执行其中命令。

<a id="source-4"></a>

**[cite:4] 训练主链与损失，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/trainer.py 。raw读取初始化、网络分支、process_batch、predict_poses与generate_images_pred；返回在compute_reprojection_loss附近截断，后以内存HTTP补读compute_reprojection_loss和compute_losses全文，核对identity噪声、联合min、分母、归一化和2^s权重。未执行作者训练器。

<a id="source-5"></a>

**[cite:5] 数学算子，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/layers.py 。raw全文已读，定位disp_to_depth、transformation_from_parameters、rot_from_axisangle、BackprojectDepth、Project3D、upsample、get_smooth_loss及SSIM；数学推导按正文声明的理想／数值模型区分。

<a id="source-6"></a>

**[cite:6] 编码器，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/networks/resnet_encoder.py 。raw全文已读，核对五级特征、统一标量归一、两图6通道与首层权重复制／平均；未下载ImageNet资产。

<a id="source-7"></a>

**[cite:7] 深度解码器，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/networks/depth_decoder.py 。raw全文已读，核对五级解码通道、skip拼接、四输出头及sigmoid；nearest的具体实现见来源5。

<a id="source-8"></a>

**[cite:8] pose解码器，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/networks/pose_decoder.py 。raw全文已读，核对1×1 squeeze、两层3×3、最终6维槽、空间均值和0.01缩放；与来源4核对pairs第0槽使用，不把两个槽解读为已预测前后帧。

<a id="source-9"></a>

**[cite:9] 默认配置与消融，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/options.py 。raw全文已读，核对H/W、depth范围、scales、训练预算、pose分支和eval选项；可选参数存在不表示全部分支已复现。

<a id="source-10"></a>

**[cite:10] 数据、增强与相机约定，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/datasets/mono_dataset.py 和 https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/datasets/kitti_dataset.py 。两份raw全文已读，核对RGB／颜色增强、尺度K、中心主点、左右与flip的平移符号、训练／评价GT读取角色；实际KITTI文件、标定链与深度生成函数内部未运行。

<a id="source-11"></a>

**[cite:11] 深度评价，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/evaluate_depth.py 。raw全文已读，核对单图输入、逆深度resize、crop、median scaling、STEREO_SCALE_FACTOR=5.4、1e−3／80截断与可选翻转后处理；没有复算论文指标。

<a id="source-12"></a>

**[cite:12] 单图预测入口，C。** https://github.com/nianticlabs/monodepth2/blob/19e16e1c29e103a5a7c6e2f20dc9ac33210aa396/test_simple.py 。raw全文已读，核对权重feed尺寸、RGB、LANCZOS、npy与可视化尺寸差异，以及纯M请求metric时仅警告的行为；未加载或执行权重。

## G41销项建议与未实现边界

本卡补齐 **G41／10-08／E08 的独立 Monodepth2 机制卡入口**。建议将“P1缺卡”子项销为“已有MD01；核／核／部／核（限定模型）／部；答5；2019原文正文＋固定作者代码；未实现”，后续实测与移植工作保留P2。若销项标准是五层教学机制与可追溯来源，则缺卡子项已具备销项依据；若要求网络运行、训练复现或物理尺度验收，不能整体标完成。

保留缺口：旧／新框架采样等价性、完整SSIM／pose／网络梯度、权重资产与实际数据协议、许可全文核对、KITTI复现、跨相机泛化及独立度量尺度验收。**单目输出不可宣称米制；双目训练的尺度锚点也不能自动升级为任意场景可追溯测量。**

导航可用 `monodepth2.md#md01`，G41历史入口可用 `monodepth2.md#g41`。**本次仅修订本文件的投影推导、相关限定与有限检查记录，不修改总览、台账、索引、源码、依赖或测试，不提交。**三维总览与现有台账的“缺卡／无专卡”属于待另行同步的旧状态；本文报告销项建议不冒充这些文件已修改。外部源码阅读、伪代码和有限检查均不等于本仓库已实现、网络测试通过或现场验证。
