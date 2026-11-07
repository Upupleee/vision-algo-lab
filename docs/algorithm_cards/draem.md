# DR01 DRAEM：合成异常监督、重建与判别联合学习

[学习规范](../algorithm_learning_standard.md) · [缺陷异常总览](../defect_evolution.md#37-e07draem-用可生成的监督弥补单纯重建目标) · [覆盖台账 G20／06-07](../evolution_coverage.md) · [逐卡索引](README.md) · [Focal 组件 C24](learning_components.md#c24-focal-loss)

<a id="dr01"></a>
<a id="g20"></a>

## 任务、演进与覆盖状态

**任务：**训练时给定目标类别的正常图像及外部纹理图像，在正常图上生成异常外观和已知掩膜，联合学习正常重建与像素异常判别；推理时仅输入待检图，输出重建图、像素异常分数图和整图异常分数。不需要目标域真实缺陷训练标签，但明确使用**合成像素监督**，不能称为没有监督信号、无需训练或自带真实缺陷概率校准。[cite:1](#source-1)[cite:4](#source-4)[cite:7](#source-7)

**演进定位：**正常自编码器可能把异常也重建好，固定残差度量又难以兼顾纹理与边界 → DRAEM 以可生成的异常监督连接正常重建与学习式比较 → 判别网联合观察输入及重建，分割损失还反传到重建网 → 增加两网训练、外部纹理、合成分布和损失配置成本 → 仍有合成到真实的域差、缺料与逻辑异常盲区。这是与正常特征建模并行的分支，不是 PaDiM、PatchCore 或 SSIM-AE 的改名，也不是 GAN 的对抗训练。[cite:1](#source-1)[cite:2](#source-2)

版本：Zavrtanik、Kristan、Skočaj，**ICCV 2021，8330–8339**；本文以会议开放接受稿及 arXiv **2108.07610v2，2021-09-27** 对照。作者仓库固定到 **`2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`**，提交时间 **2021-10-12 14:48:07 UTC**，消息 `Typos in README`。它是确实读取的 2021 作者源码快照，不宣称就是论文全部实验使用的提交；核验日期 **2026-11-05**。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

本文件只有 **DR01 一张卡**，承接 **G20／06-07 E07**。异常模拟、SSIM、focal 和整图聚合是卡内机制，不另计卡。

| 字段 | 状态与边界 |
|---|---|
| L1 步骤 | 已核验：原文及固定源码静态对照，含模拟、训练、停止、推理及评分 |
| L2 机制 | 已核验：掩膜监督、两网角色、损失耦合及删除／替换反例 |
| L3 工程权衡 | 部分：尺度、调参、症状与验证方案齐备，无真实工况定参 |
| L4 数学推导 | 部分：混合恒等式、三项损失、链式梯度、聚合及退化已展开；未做自动微分数值比对或完整统计泛化证明 |
| L5 实现 | 部分：固定外部源码对应、数据布局、伪代码、边界和测试设计；本仓库未实现 |
| 面试自测 | 答 5：L1–L5 各一问，附答案、成立条件、易错点和证据 |
| 实现与验证 | **未实现、未训练、未加载权重、未执行异常生成或端到端推理、无本仓库缺陷实验结果** |

**诚实边界：**下文的合成异常是被研究方法实际采用的训练机制，不是本次伪造数据、模拟实验成绩或真实缺陷验收。论文、作者 README 的公开成绩和本仓库实测是三本账；此卡不转录性能排名，不以文档检查、源码阅读或代数推导充当训练复现。原文及作者代码属同一研究来源家族，核心机制标记为 **[SINGLE-SOURCE，原文＋作者实现相互核对]**，不是三个独立复现实验。

## L1 步骤：异常三元组到整图判断

### 输入、输出与数据契约

记正常目标图为 \(x\)，增广后的外部纹理为 \(a\)，二值模拟掩膜为 \(m\)，受损图为 \(\tilde x\)，重建为 \(r=R_\theta(\tilde x)\)。训练三元组是 \((x,\tilde x,m)\)，不是把真实测试缺陷掩膜拿来训练。原文 §3.2 将判别网观察的输入统称为 \(I\)，不能据此把训练时不可用的干净目标送进判别网；固定训练代码明确拼接 **`(gray_rec, aug_gray_batch)`**。[cite:1](#source-1)[cite:7](#source-7)

| 对象 | 固定源码契约 | 注意事项 |
|---|---|---|
| \(x,a,\tilde x,r\) | 网络布局 `B×3×256×256`；输入 float32、除以 255 | OpenCV 读入后未做 BGR→RGB，因此源码实际是 **BGR**；变量名 `gray_batch` 不表示单通道 |
| \(m\) | `B×1×256×256`，float32 存储的 0／1 | 1 是模拟异常支持，0 是正常；focal 内部转 long 类别索引 |
| 判别输入 \(c\) | `cat([r, x_tilde], dim=1)`，`B×6×H×W` | 顺序是重建在前、观测在后；换序不与已有权重等价 |
| 判别输出 \(z\) | `B×2×H×W` logits | 沿通道 softmax，取索引 1 得异常图 \(p\)，索引 0 是正常 |
| 推理输出 | \(r,p\) 及每图一个分数 \(s\) | 重建仅是诊断输出，不是真实良品参考；分数越大越异常 |

尺寸与布局由训练／测试调用及模型共同确定；源码未做 ImageNet 标准化，也没有固定预训练特征骨干。[cite:4](#source-4)[cite:5](#source-5)[cite:7](#source-7)[cite:8](#source-8)

### 异常模拟：Perlin 形状、纹理与混合

**论文口径：**Perlin 噪声形成连续形状场，以均匀随机阈值二值化；从与目标图分布无关的纹理集合取图，再从 posterize、sharpness、solarize、equalize、brightness change、color change、auto-contrast 中取 3 种增强。将增广纹理限制在掩膜内，与正常图混合；纹理不透明度 \(\beta_p\in[0.1,1.0]\)。原文已读段落没有给出随机阈值的数值区间，不补造它。MVTec 实验使用 DTD 作为纹理源。[cite:1](#source-1)[cite:2](#source-2)

**固定代码口径：**以下流程有具体值，不能把它悄悄称为逐参数复刻论文。

1. 从 `train/good/*.png` 随机取正常图并缩放至 256×256；以约 0.3 概率进行 `(-90,90)` 度旋转。旋转后图像才是重建目标 \(x\)。
2. 从纹理源 `*/*.jpg` 随机取一张，缩放到同尺寸；从代码的 **10 种**增强中无放回选择 3 种顺序执行。
3. 两个方向独立采样整数 \(k_h,k_w\in\{0,\ldots,5\}\)，设 Perlin 网格分辨率 \((2^{k_h},2^{k_w})\)。生成单尺度噪声场后再作 `(-90,90)` 度旋转，用固定 **`noise > 0.5`** 得到 \(m\)；并非采样每个像素独立的白噪声，也没有调用多 octave 分形函数。
4. 采样代码变量 \(b\sim U[0,0.8)\)，使用 \(\tilde x=(1-m)x+(1-b)ma+bmx\)。这里 \(b\) 是掩膜内的**原图权重**，纹理不透明度是 \(1-b\)，与论文 β 的命名方向相反。
5. 以约 0.5 概率返回未受损 \(x\) 和全零掩膜；另一路若 Perlin 掩膜为空，也返回 `has_anomaly=0`。因此实际异常比例不保证恰好一半。代码不对对象前景再取交集，也不保证异常面积下限。
6. 转为 CHW float32，返回干净目标、受损图和掩膜；`has_anomaly` 是样本元数据，训练损失使用像素掩膜而不是这个整图标志。[cite:4](#source-4)[cite:9](#source-9)

代码的 10 种增强为 `GammaContrast`、`MultiplyAndAddToBrightness`、`EnhanceSharpness`、`AddToHueAndSaturation`、`Solarize`、`Posterize`、`Invert`、`Autocontrast`、`Equalize`、`Affine(rotate=(-45,45))`。后三通道操作仍接收源码读出的 BGR 数据；移植到 RGB 时需记录协议变化，不假定每种色彩增强与旧结果相同。[cite:4](#source-4)

### 两个子网络的具体结构

论文给出重建 encoder–decoder 和判别 U-Net-like 的角色；下表的精确通道来自固定 **`model_unet.py` 默认结构＋训练调用**，不是从泛 U-Net 名称推断。[cite:1](#source-1)[cite:5](#source-5)

| 网络／阶段 | 256×256 输入下的形状与运算 | 关键区别 |
|---|---|---|
| 重建编码器 | 通道 128、256、512、1024、1024；空间 256、128、64、32、16 | 5 个双卷积块，块间 4 次 2×2 max-pool；仅把最深 `b5` 传给解码器 |
| 重建解码器 | 4 次 ×2 上采样；每阶段双卷积块输出通道 512、256、128、128；最后 3×3 卷积到 3 通道 | **没有编码器到解码器的横向跳连**；输出无 sigmoid，不保证落在 [0,1] |
| 判别编码器 | 输入 6 通道；块输出 64、128、256、512、512、512；空间 256、128、64、32、16、8 | 6 个双卷积块，5 次池化，保留 `b1…b6` |
| 判别解码器 | 5 次 ×2 上采样；依次拼接 `b5,b4,b3,b2,b1`；融合块输出 512、256、128、64、64 | 拼接后通道依次 1024、768、384、192、128；最后 3×3 卷积输出 2 logits |

两网普通卷积块均用 **3×3、padding=1、BatchNorm、ReLU**，每块两个卷积；上采样是 `bilinear, align_corners=True` 后再接卷积／BN／ReLU，不是转置卷积。判别跳连保留空间细节，而重建瓶颈避免直接通过横向通道复制输入；这是结构作用解释，不是“无跳连必定清除所有异常”的定理。[cite:5](#source-5)

### 联合训练、停止与推理

1. 每类别重新初始化两网；正常样本和纹理源按固定协议抽样，不使用测试缺陷选增强或模型。
2. 前向 \(r=R_\theta(\tilde x)\)，\(z=D_\phi([r,\tilde x])\)，\(p=\operatorname{softmax}(z)_1\)。
3. 全图计算重建 MSE、SSIM 损失及掩膜 focal，三项相加；对同一 loss 执行一次 `zero_grad → backward → step`，一个 Adam 管理两网参数组。**训练拼接没有 `detach`**。[cite:7](#source-7)
4. 固定训练脚本按 `args.epochs` 结束，每 epoch 调度学习率并分别覆盖保存两份权重；README 示例为 700 epochs、batch 8、初始学习率 \(10^{-4}\)。没有以验证集最优指标自动早停，也没有保存完整优化器／调度器／随机数状态用于精确续训。[cite:7](#source-7)[cite:11](#source-11)
5. 推理切换两网 `eval`，仅用真实待检图 \(x_q\)：\(r=R_\theta(x_q)\)，\(p=\operatorname{softmax}(D_\phi([r,x_q]))_1\)。不生成异常、不用干净配对目标、不计算训练损失。
6. 像素定位保留原始 \(p\)；整图分数用 **21×21 均值池化，stride=1、padding=10 后取空间最大值**。固定测试代码的像素 AUROC／AP 使用未平滑图，整图指标使用平滑后的最大值；不能交换两条支路。[cite:8](#source-8)

训练轮数耗尽不代表已收敛或达到业务漏检要求。工程接入时，遇到空数据源、无法解码、非有限损失／梯度、尺寸不支持或权重不配套，应中止并标为失败；推理遇到未知输入协议、失效成像、非有限输出或超时，应返回 UNKNOWN／复检。这些退出防护是本卡设计，**不是声称作者脚本已实现**。

## L2 机制：为何不能只保留一个网络或一项损失

| 步骤／机制 | 解决的问题与不变量 | 删除、替换或退化后的后果 |
|---|---|---|
| Perlin 二值形状 | 用空间相关形状提供已知异常支持；掩膜外观测等于正常目标 | 独立像素噪声更像去噪任务；固定矩形可能使分割学会边框捷径，不等于原模拟器 |
| 外部纹理＋随机混合 | 让掩膜内有多种局部异常外观，并保留程度不同的原背景 | 单一强纹理或始终完全覆盖容易形成合成外观捷径；太弱或与原图相同则产生不可见正标签 |
| 干净正常重建目标 | 受损输入对应原正常图，鼓励恢复而非复制受损内容；全图损失同时约束正常区域 | 若以受损图为重建目标，恒等复制就能低损失；若只监督掩膜内部，正常区域失真约束减弱 |
| 重建网与判别网分工 | 重建提供正常性参照；判别结合输入、参照和上下文学习差异 | 仅用残差会把正常纹理重建偏差也当异常；仅判别受损图容易记住合成外观 |
| 六通道联合观察 | 保留原外观与重建的完整信息，允许学习外观条件化比较 | 只输入绝对差会丢失差异方向及双方外观；“拼接”本身不强制网络一定使用两半输入 |
| MSE＋SSIM | MSE 约束逐像素强度；SSIM 用局部统计补充结构关系 | MSE 的逐像素误差不显式比较邻域结构；仅 SSIM 又不等同精确像素保真 |
| focal＋联合反传 | 难像素获得相对更高权重，并让判别目标影响重建表征 | 换普通 CE 是消融而非必然失败；训练 `detach(r)` 切断分割→重建路径，改变联合目标 |
| 未受损样本分支 | 直接监督正常输入输出全零异常图，抑制把一切纹理变化都当异常 | 掩膜占比／正常比例失衡会改变训练先验；focal 并不能自动修复错标签或模拟分布偏差 |

这些机制由原文 §3、§4.2 的设计动机及固定实现支持；“删除后可能怎样”是条件性因果分析，不是本仓库消融结果。论文自身报告了仅重建、仅判别及联合方案的消融，不能将其优劣扩展为任意产品、纹理源或架构的通用保证。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)[cite:7](#source-7)

**反例：**缺掉一根引脚后露出原本常见背景，像素外观可以完全正常，但装配关系不正确；DRAEM 不必把“本应有引脚的位置”整片标高。原文 §4.1 正面讨论了此类缺料困难。再如 \(m=1\) 但 \(a=x\)，合成监督仍说异常，观测却不变；网络无法仅凭相同输入可靠区分这两个标签。这不是降低阈值就能消除的可辨识性问题。[cite:2](#source-2)

## L3 工程权衡：参数尺度、诊断与调参

### 参数与尺度

| 参数／协议 | 数值、单位及依据 | 代价与边界 |
|---|---|---|
| 输入尺寸 | 作者代码 256×256 像素 | 不保持任意原图纵横比；细缺陷可能在缩图时消失。切片是新协议，需验证上下文、接缝与多片整件聚合 |
| Perlin `res` | 每轴 1、2、4、8、16、32 个网格单元，独立抽样 | 不是异常直径；相应基础网格间距为图像边长除以 `res`，二值后区域还受阈值与旋转影响 |
| 掩膜阈值 | 代码 0.5，无量纲；论文均匀随机阈值但区间未给 | 先记录空掩膜比例、面积分布和形状，再调阈值／分辨率；不得声称固定 0.5 是论文唯一值 |
| 混合强度 | 论文纹理权重 [0.1,1.0]；代码实际纹理权重 \((0.2,1]\) | 强度接近正常时任务难但标签可噪；强度太大可能只学容易的合成偏差 |
| SSIM | 代码窗口 11 像素、高斯 σ=1.5 像素，系数 λ=1 隐含于直接相加 | 窗口变大扩大结构耦合、增加计算；改变归一化会改变稳定常数含义 |
| focal | 代码 γ=2、α 为每类 1、`smooth=1e-5` | 不是习惯性照搬的 α=0.25；γ 增大不保证真实召回提升，极难错标签仍有影响 |
| 优化配置 | Adam；README 示例 batch 8、700 epochs、学习率 \(10^{-4}\) | 原文与代码学习率里程碑不同，见差异表；batch 改动影响 BN、梯度统计及资源，不只是速度 |
| 整图窗口 | 代码 21×21 像素，非毫米 | 太大会稀释微缺陷；改尺寸／核／ROI 后必须重校整图阈值 |
| 业务阈值 | 像素与整图阈值分开，无通用默认值 | 只用正常验证集能估当前正常分布的误报，不能测真实缺陷召回，也不能把 softmax 当校准概率 |

代码数值来自固定源码，论文数值来自原文；现场调参和失效处理是未执行的工程方案。[cite:2](#source-2)[cite:4](#source-4)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)[cite:11](#source-11)

### 症状、证据与调整顺序

| 症状 | 先看什么证据 | 调整及回退 |
|---|---|---|
| 合成样本定位好、真实缺陷差 | 真实留出集按缺陷类型／尺寸分层；合成掩膜面积、纹理频率、透明度及正常合法变化 | 先排数据泄漏和成像域差，再比较模拟器与表示；不要把测试缺陷反复用于调模拟器后仍称盲验 |
| 正常区域整片高分 | 原图／重建／热图同屏、通道顺序、归一化、BN 模式、训练正常覆盖 | 先修协议，再审计正常集；不先放宽阈值掩盖系统漂移 |
| 训练 mask 总是零或区域很碎 | 源图读取、Perlin 幅值、阈值、空掩膜率、网格和缩放 | 排生成错误再调参数；空掩膜不能强标 `has_anomaly=1` |
| 微裂纹整图漏检 | 原图宽度、256 缩放后支持、未平滑 \(p\) 与池化图比较 | 先保可见分辨率，再比较窗口；直接改成 raw max 会更敏感于孤立误报，属于新评分协议 |
| focal 不降或出现 NaN | 标签唯一值、logits／概率接口、softmax 维度、三项 loss 与梯度范数 | 先修接口及有限值，再调损失权重；不要把降低学习率当一切错误的修复 |
| 接上判别后重建方向改变 | 分别观察重建损失梯度与分割反传梯度、是否意外 detach | 两种目标可相互冲突；有意 detach 或分阶段训练需单独命名与验证，不能仍称原联合训练 |
| 缺件但背景热图低 | 本应出现对象的位置、身份、关系与原文缺料反例 | 补装配关系／几何规则，不承诺用纹理模拟覆盖逻辑异常 |

建议顺序为 **数据分组／正常定义 → 输入及合成三元组可视检查 → 两网布局与梯度 → 损失数值／优化 → 分辨率与模拟分布 → 评分／阈值 → 性能优化**。本卡未执行可视检查或真实数据试验，只规定之后需要采集的证据。

## L4 数学：合成监督、联合目标与梯度

### Perlin 场与混合不变量

固定 `rand_perlin_2d_np` 在二维网格顶点采随机单位梯度，对格内位置与四个顶点的位移求点积，再按五次平滑函数插值：

\[
f(t)=6t^5-15t^4+10t^3,\qquad
P(u,v)=\sqrt2\,\operatorname{lerp}_{f(v)}
\left(\operatorname{lerp}_{f(u)}(n_{00},n_{10}),
\operatorname{lerp}_{f(u)}(n_{01},n_{11})\right).
\]

\(n_{ij}\) 是顶点梯度与相对位移的内积，\(u,v\in[0,1)\) 为格内坐标。端点处 \(f'\) 与 \(f''\) 为零，插值提供平滑空间相关结构；随后旋转、阈值 \(m=\mathbf1[P>t]\) 才形成不规则二值区域。**Perlin 场不是缺陷概率，也不能从 `res` 直接推得最终缺陷面积。** 此生成过程在数据加载器中完成，不参与两网参数反传。[cite:4](#source-4)[cite:9](#source-9)

用统一的纹理权重 \(\alpha\) 重写原文式 (4)：

\[
\tilde x=(1-m)x+m[(1-\alpha)x+\alpha a]
=x+\alpha m(a-x).
\]

论文 \(\alpha=\beta_p\)，代码 \(\alpha=1-b\)。故 \(m=0\Rightarrow\tilde x=x\)，掩膜内是凸混合；若 \(x,a\in[0,1]\) 且 \(\alpha\in[0,1]\)，则 \(\tilde x\in[0,1]\)。这项范围性质只对混合输入成立，不对无输出激活的 \(r\) 成立。[cite:1](#source-1)[cite:4](#source-4)

退化包括空掩膜、\(\alpha=0\)、\(a=x\) 或对比度低于成像噪声；生成掩膜不自动等于可见缺陷支持。\(\alpha=0\) 虽不在上述默认取值内，仍应作为边界测试。掩膜标的是注入位置，不是 \(\lvert\tilde x-x\rvert\) 每个像素必然非零的位置。

### 重建 MSE 与 SSIM

设每 batch 标量元素数 \(N=3BHW\)，代码所谓 L2 是 **均方误差**，不是未平方的欧氏范数：

\[
L_{\mathrm{MSE}}=\frac1N\sum_j(r_j-x_j)^2,
\qquad \frac{\partial L_{\mathrm{MSE}}}{\partial r_j}=\frac{2(r_j-x_j)}N.
\]

独立等方差高斯观测模型的负对数似然去掉常数可产生平方误差，这是解释该损失的一种假设，不代表真实缺陷或成像噪声满足此模型。代码对整图求均值，不只对异常掩膜求均值。[cite:6](#source-6)[cite:7](#source-7)

每个通道的局部归一化高斯权重为 \(w_j\)，局部均值 \(\mu_r=\sum_jw_jr_j\)、\(\mu_x=\sum_jw_jx_j\)，方差 \(v_r=\sum_jw_jr_j^2-\mu_r^2\)、\(v_x\) 同理，协方差 \(c=\sum_jw_jr_jx_j-\mu_r\mu_x\)。单窗口：

\[
S=\frac{(2\mu_r\mu_x+C_1)(2c+C_2)}
{(\mu_r^2+\mu_x^2+C_1)(v_r+v_x+C_2)},\qquad
L_{\mathrm{SSIM}}=1-\operatorname{mean}(S).
\]

固定实现 \(C_1=(0.01L)^2,C_2=(0.03L)^2\)，11×11 高斯核、σ=1.5，各通道独立卷积，padding=5 的**零填充**后跨 batch／通道／位置取均值。它不是 MS-SSIM，不是先转灰度，也不是论文推理时必须输出的异常图。[cite:1](#source-1)[cite:6](#source-6)

为何能传结构梯度：在固定动态范围 \(L\) 和权重下，

\[
\frac{\partial\mu_r}{\partial r_j}=w_j,\quad
\frac{\partial v_r}{\partial r_j}=2w_j(r_j-\mu_r),\quad
\frac{\partial c}{\partial r_j}=w_j(x_j-\mu_x).
\]

令 \(A=2\mu_r\mu_x+C_1,B_s=2c+C_2,C=\mu_r^2+\mu_x^2+C_1,D=v_r+v_x+C_2\)，则

\[
\frac{\partial S}{\partial r_j}
=\frac{(2\mu_xw_jB_s+2Aw_j(x_j-\mu_x))CD
-AB_s(2\mu_rw_jD+2Cw_j(r_j-\mu_r))}{(CD)^2}.
\]

一个像素属于多个重叠窗口，最终 SSIM 梯度是各窗口导数的负平均，因此它不只是逐像素强度修正。这里使用乘积／商法则，避免对可能非正的 \(S\) 取对数来推导。稳定常数在有效范围下保护分母，但浮点方差的大数相消、溢出或非有限输入仍需检查；没有通用收敛保证。

**源码数值限定：**`ssim()` 默认由第一个参数的最大／最小值猜 \(L\)：max>128 取上界 255，否则 1；min<-0.5 取下界 -1，否则 0。训练第一个参数是未限幅重建 \(r\)。`SSIM` 类虽存 `val_range`，其 `forward` 未把该值传给 `ssim()`，所以只改构造参数不会固定范围。上述导数限定在分支选定后 \(L\) 不变的区域；猜测分支可能跳变。移植时显式固定范围属于应记录并验证的修正，不悄悄归给原代码。[cite:6](#source-6)[cite:7](#source-7)

### Focal 与两网梯度路径

先给无平滑的二类 focal 便于推导。\(y=m\in\{0,1\}\)，\(p=\operatorname{softmax}(z)_1\)，\(p_t=yp+(1-y)(1-p)\)：

\[
\ell_f=-a_t(1-p_t)^\gamma\log p_t,\qquad
L_f=\frac1{BHW}\sum\ell_f.
\]

\(a_t\) 是类别权重，与前述纹理权重 \(\alpha\) 无关。\(\gamma=0\) 时退化为加权交叉熵；\(p_t\) 接近 1 时调制项抑制容易样本。固定默认为 \(a_t=1,\gamma=2\)，不是论文给出了 α=0.25 的配置。[cite:6](#source-6)

令 \(u=z_1-z_0\)、\(q=2y-1\)，则 \(p_t=\sigma(qu)\)。链式法则得到：

\[
\frac{\partial\ell_f}{\partial p_t}
=a_t\gamma(1-p_t)^{\gamma-1}\log p_t
-a_t\frac{(1-p_t)^\gamma}{p_t},
\]

\[
\frac{\partial\ell_f}{\partial u}
=a_tq(1-p_t)^\gamma
\left[\gamma p_t\log p_t-(1-p_t)\right].
\]

源码实际先将 one-hot 裁到 \([\epsilon,1-\epsilon]\)，再算与概率的内积并加 \(\epsilon\)，\(\epsilon=10^{-5}\)。二类时送入 log 的量是 \(\hat p_t=(1-2\epsilon)p_t+2\epsilon\)；实际导数需用 \(\partial\ell_f/\partial\hat p_t\) 乘 \((1-2\epsilon)\partial p_t/\partial u\)。这**不是**常规“对每类 log 概率加权求和”的 label-smoothed CE。无平滑推导不能直接当固定源码的逐位数值结果。[cite:6](#source-6)

总目标与梯度为：

\[
L=L_{\mathrm{MSE}}(r,x)+\lambda L_{\mathrm{SSIM}}(r,x)
+L_f(D_\phi([r,\tilde x]),m),\qquad r=R_\theta(\tilde x).
\]

\[
\nabla_\phi L=\nabla_\phi L_f,\qquad
\nabla_\theta L=
J_{R,\theta}^{T}\left[
\nabla_r L_{\mathrm{MSE}}+\lambda\nabla_rL_{\mathrm{SSIM}}
+J_{D,r}^{T}\nabla_zL_f\right].
\]

\(D\) 的 focal 表达式隐含一次 softmax；固定训练 λ=1。第三项是 **判别→重建** 的监督，原文 §3.1 明说有下游训练信号，源码没有 `detach` 与之相符。训练时 `detach(r)` 会删去这一项；只更新判别参数或分开优化也需明确协议。推理脚本出现 `gray_rec.detach()` 只截断推理计算图，不证明训练采用冻结重建。[cite:1](#source-1)[cite:7](#source-7)[cite:8](#source-8)

网络目标非凸，三个梯度方向可以冲突；正常重建并不是对每一个观测唯一可辨识的“真实无缺陷版本”，判别也可能忽略拼接中的一部分。联合训练下降不意味着全局最优、真实缺陷零漏检或跨域误差界；原文未提供这类通用保证。

### 整图聚合、边界与复杂度

原文式 (5) 为均值滤波后最大值。固定实现可写为

\[
\bar p_{ij}=\frac1{441}\sum_{u=-10}^{10}\sum_{v=-10}^{10}p^{\mathrm{zero}}_{i+u,j+v},
\qquad s=\max_{i,j}\bar p_{ij}.
\]

零补边仍计入平均分母，对应 `avg_pool2d` 的默认 `count_include_pad=True`。边缘窗口不是除以实际有效像素数；有效区归一化或反射填充都会改变边缘分数。单个孤立高像素被局部平均稀释，区域响应更稳定；这一解释不等于已测得更好召回。[cite:2](#source-2)[cite:8](#source-8)

每层卷积近似计算量 \(O(BH_\ell W_\ell k_\ell^2 C_{in,\ell}C_{out,\ell})\)，两网相加；训练还存反向激活，判别跳连需保留多尺度特征。Perlin 与混合为固定网格下 \(O(BHW)\)，普通窗口 SSIM／均值池化还乘窗口面积，算法优化可改变常数或阶数。本方法没有随正常库样本数增长的近邻检索，但离线训练与模型显存成本仍在；此卡不报未实测参数量、显存、FPS 或延迟。

## L5 实现：固定源码、伪代码与边界验证

### 源码对应与原文差异

所有源码引用均绑定同一完整 SHA，不以可移动的 `main` 作为复现版本。

| 原文／方法描述 | 固定提交的具体行为 | 复现解释 |
|---|---|---|
| §3.3 随机均匀阈值 | `data_loader.py:augment_image` 固定 threshold=0.5 | **实现差异**；论文随机阈值区间未知，不编造 |
| 纹理不透明度 β∈[0.1,1.0] | 变量 beta∈[0,0.8)，原图权重；纹理权重 1-beta∈(0.2,1] | 既有符号方向差异，也有分布区间差异，不能只改变量名就称完全一致 |
| 7 种候选增强选 3 | 10 种候选增强选 3，含 Invert、GammaContrast、几何旋转等 | 同为随机增强思想，但候选集合／参数不逐项相同 |
| §4.1 正常图旋转 (-45,45) 度 | 正常图约 30% 概率旋转 (-90,90)；Perlin 场也用该旋转范围 | 纹理自身的 (-45,45) 旋转不能拿来抵消这一差异 |
| §4.1 初始 lr=1e-4，400／600 epoch 各 ×0.1，训练 700 epochs | `MultiStepLR` 里程碑为 0.8E／0.9E，各 ×0.2；E=700 时为 560／630 | **调度差异**；README 的 700 轮示例并非原文所有参数的证明 |
| 重建 encoder–decoder／判别 U-Net-like | `model_unet.py` 精确通道、无重建跳连、两类 logits、BN 和 bilinear 上采样 | 原文概述由源码细化；不把两者都称为同一 U-Net |
| 论文 Lrec=λ LSSIM+l2，Lseg 为 focal | MSE 均值＋SSIM 均值＋focal 均值直接相加，γ=2、α=None、smooth=1e-5 | 代码 λ=1；不能反推原文未详列的全部数值设定 |
| §3.4 均值滤波后最大值 | `test_DRAEM.py` 用 21×21、零补边；像素指标另用 raw p | 具体窗口／边界来自代码，非声称论文正文给出了这两个数值 |
| 训练联合学习 | `train_DRAEM.py` 拼接无 detach；测试拼接有 detach | 与联合反传一致；测试 detach 不可倒推成训练机制 |

差异证据：[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)。补充材料主要是定性图与监督方法对照，不能从其图注推导遗漏的架构或超参数。[cite:10](#source-10)

### 参考伪代码

以下是**固定代码口径＋明确防护的教学伪代码，未执行、不是本仓库 API**。`R_5block_no_skip`、`D_6block_with_skip` 对应 L1 结构，`focal_code`／`ssim_code` 对应 L4 数值契约；不是用泛函数名省略算法机制。防护及 `no_grad` 属接入建议，非作者脚本逐行翻译。

```text
train(normal_paths, texture_paths, epochs, batch_size):
    require nonempty(normal_paths) and nonempty(texture_paths)
    require epochs > 0 and batch_size > 0
    R = R_5block_no_skip(in=3, out=3, width=128)
    D = D_6block_with_skip(in=6, out=2, width=64)
    initialize_conv_and_bn_as_author(R, D)
    optimizer = Adam(parameters(R) + parameters(D), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[0.8*epochs, 0.9*epochs], gamma=0.2)
    R.train(); D.train()
    for epoch in range(epochs):
        for batch in random_draw_batches(normal_paths, batch_size):
            triples = []
            for selected_normal in batch:
                x_u8 = decode_BGR_resize(selected_normal, height=256, width=256)
                if uniform01() > 0.7:
                    x_u8 = affine_rotate(x_u8, uniform(-90, 90))
                x = float32(x_u8) / 255
                a_u8 = decode_BGR_resize(random_choice(texture_paths), 256, 256)
                a = float32(apply_three_distinct_author_augmenters(a_u8)) / 255
                res = (2**randint(0, 6), 2**randint(0, 6))
                P = perlin_gradient_dot_fade_lerp((256, 256), res)
                P = affine_rotate(P, uniform(-90, 90))
                m = float32(P > 0.5)[..., None]
                b = uniform01() * 0.8
                x_tilde = x*(1-m) + a*m*(1-b) + x*m*b
                if uniform01() > 0.5:
                    x_tilde = x
                    m = zeros_like(m)
                require finite(x, x_tilde, m) and binary(m)
                triples.append(to_CHW(x, x_tilde, m))
            x, x_tilde, m = stack(triples)
            r = R(x_tilde)
            z = D(concat_channels(r, x_tilde))
            probabilities = softmax(z, dim=1)
            loss = mean_square(r-x) + ssim_code(r, x) + focal_code(probabilities, m)
            require finite(loss)
            optimizer.zero_grad()
            loss.backward()
            require finite_gradients(R, D)
            optimizer.step()
        scheduler.step()
        save_paired_weights(R, D, epoch)
    return R, D

infer(query, paired_weights):
    require valid_image(query) and compatible(paired_weights)
    x, coordinate_record = preprocess_same_BGR_256_protocol(query)
    R, D = load_paired_models(paired_weights)
    R.eval(); D.eval()
    with no_grad():
        r = R(x)
        p = softmax(D(concat_channels(r, x)), dim=1)[:, 1:2]
        smoothed = avg_pool2d(p, kernel=21, stride=1, padding=10, count_include_pad=True)
        s = spatial_max_per_image(smoothed)
    require finite(r, p, s)
    return r, p, s, coordinate_record
```

`randint(0,6)` 上界不含 6；原作者 `__getitem__` 忽略传入索引重新随机抽正常图，所以一个 epoch 是数据集长度所规定的抽样次数，**不是保证每张正常图恰看一次**。重写为无放回遍历会改变采样过程；若保存中途续训状态还需包括 Python／NumPy／torch／imgaug 和多 worker 随机性。[cite:4](#source-4)[cite:7](#source-7)

### 实现易错点与优化边界

- **坐标与尺寸：**OpenCV `dsize=(W,H)`，Perlin `shape=(H,W)`；二者不能对调后只用正方形样例验证。原判别网下采样 5 次，以 32 的倍数为安全形状约束；原 Perlin 实现还依赖每轴形状能被对应 `res` 整除。任意奇数尺寸不能不经处理直接拼接跳连。[cite:4](#source-4)[cite:5](#source-5)[cite:9](#source-9)
- **双网与概率：**`DiscriminativeSubNetwork` 类的默认输入／输出参数都是 3，训练入口显式改成 6／2；不能仅无参数实例化。focal 接收的是已经 softmax 的概率，传 raw logits 或做两次 softmax 均改变目标。重建输出不要为显示方便在训练图中直接 clip。[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)
- **空值与标签：**空纹理路径不能静默改用随机噪声后仍称固定作者版。训练掩膜在生成阶段二值化；测试真实掩膜代码却使用默认双线性 resize 后除以 255，最终转 uint8，会把中间值截到 0、改变边缘标签。改用最近邻是更明确的标签协议，但须单列与原结果的差异。[cite:4](#source-4)[cite:8](#source-8)
- **训练／推理状态：**`eval()` 不等于关闭梯度；原测试脚本未包完整 `no_grad`，卡内伪代码添加它节省推理图开销，不改变 eval 下的前向意图。不要把推理中的 detach 复制进训练。两份权重必须来自同一轮／同一协议。[cite:7](#source-7)[cite:8](#source-8)
- **设备与统计：**原 SSIM 构造中直接 `.cuda()`，窗口缓存不是通用设备迁移接口；focal 在 CPU 上构造 one-hot 后搬设备，可能有同步／分配成本。CPU、AMP、多 GPU 或现代 torch 移植均需专项测试；先以 FP32 对照 SSIM／focal 数值，再考虑融合。[cite:6](#source-6)
- **代码外观与真实行为：**两个编码器都出现 `mp2 = self.mp3(b2)`，但其池化模块均为相同、无参数的 `MaxPool2d(2)`；不能只凭名字把它断言为当前数值错误。若未来池化配置不同，则需重新核对实际调用。[cite:5](#source-5)

先剖析纹理解码／imgaug、两网卷积、SSIM、CPU one-hot 和数据搬运，再优化瓶颈。缓存纹理原图可以研究，但把受损图一次性预生成并固定会减少随机组合；替换 SSIM 库、修改池化补边、AMP、缩通道、移除 BN 或换预训练骨干均不能默认等价。模型权重只从可信源加载，旧式 pickle 权重不得在未核验来源时反序列化。此卡没有下载／执行外部权重或安装历史环境。

### 待执行的最小验证集

| 验证对象 | 输入与检查设计 | 能证明与不能证明的范围 |
|---|---|---|
| 混合不变量 | 固定数组检查 m=0、m=1、α=0／1、a=x、空掩膜；核对掩膜外相等与凸混合范围 | 单元性质，不是合成缺陷训练成绩 |
| 形状和标签 | 非方形且满足整除约束的输入、奇数尺寸拒绝、HWC→CHW 往返、0／1 类别和拼接顺序 | 接口契约，不证明定位准确 |
| 梯度耦合 | 仅 focal 反传，检查重建参数梯度；对照 detach 后此路径消失，并检查判别参数梯度 | 应在非饱和非退化输入上验证，不要求所有权重每次都非零 |
| 三项损失 | FP64 小张量对 MSE／固定范围 SSIM／带平滑 focal 做有限差分；与固定代码 FP32 对照 | 局部梯度数值，不是非凸全局收敛证明 |
| 整图评分 | 常量图、孤立响应、角点与中心相同响应；核对 441 分母、raw p 与平滑 p 分支 | 聚合定义及边界，不是业务阈值有效性 |
| 推理一致性 | eval＋no_grad、重复推理、成对保存加载、迁移设备、BN 状态、旧／新损失接口对照 | 固定协议一致性，不覆盖所有软硬件 |
| 真实验收 | 工件／批次分组隔离训练、调参、验收；记录像素 AUROC／AP、整图 AUROC、固定阈值下误报／漏检，细缺陷分层及 UNKNOWN 比例 | 需真实正常与缺陷留出集，单凭正常样本不能获得缺陷召回 |

**该表全部为未执行设计。** 未生成测试图片、未下载 MVTec／DTD、未跑本表单元测试，未以既有匹配／对焦测试冒充 DR01 验证。后续测试可使用明确标为测试夹具的确定性数组，但其数值结果不能冒充缺陷实验成绩。

## 面试自测：贯穿 L1–L5 的五问

| 层级／追问 | 参考答案要点 | 成立条件、易错答案与证据 |
|---|---|---|
| L1：给一张正常图和一张纹理图，如何训练？测试时还要纹理吗？ | 生成 Perlin 二值 m、增广纹理并混合得到受损图；R 受损→正常，D 观察重建＋受损并监督 m；三损失联合反传。测试只用待检图，经两网得 p，均值池化后最大值得整图分数 | 正常训练目标不能泄漏到 D；测试不用纹理、掩膜或正常配对图。误答“用重建 MSE 直接当 DRAEM 整图分数”。[cite:4](#source-4)[cite:7](#source-7)[cite:8](#source-8) |
| L2：为什么不用纯重建或只对合成异常训练分割网？ | 纯重建可能重建异常、固定残差会放大正常重建误差；纯分割可能记住合成外观。联合观察提供正常参照与上下文，分割目标还能调整重建子空间 | 这是动机和作者实验支持，不是通用保证；缺引脚后正常背景仍可漏。误答“两网必定学到真实异常距离”或“这是 GAN”。[cite:1](#source-1)[cite:2](#source-2) |
| L3：微划痕漏检，先把 focal γ 调大还是阈值调低？ | 先查原图可见性、256 缩图后宽度、raw p 与 21×21 池化对比、训练合法纹理和模拟面积／强度；再用独立验证比较分辨率、模拟器、窗口及阈值 | 不拿测试集反复调参；降低阈值扩大 NG 集合也可能增加误报，γ 不补回已丢像素。无毫米／召回默认保证。[cite:4](#source-4)[cite:6](#source-6)[cite:8](#source-8) |
| L4：写出联合梯度；训练 detach 重建会怎样？ | θ 的梯度由 MSE、SSIM 和经过 D 输入雅可比的 focal 三部分组成；φ 只由 focal 更新。detach 删除第三部分。混合式 x+αm(a-x) 说明 m=0 不变、a=x 时标签可能不可辨 | focal 数值还含代码平滑，不能把无平滑式当逐位原实现；非凸目标不保全局最优。误答“两网只是分别训练后串起来”。[cite:1](#source-1)[cite:6](#source-6)[cite:7](#source-7) |
| L5：损失公式对了但复现不同，如何定位？ | 核 SHA、BGR／RGB、6→2 参数、拼接顺序、softmax 一次、掩膜类型、BN eval、SSIM 范围猜测、β方向、阈值、调度器与池化补边；用单元性质／有限差分／成对权重回归分层验证 | 源码读过不等于跑通；最近邻 mask、固定 SSIM 范围和现代库替换都要标变体。误答“调 seed 就能修复所有差异”。[cite:3](#source-3)[cite:4](#source-4)[cite:6](#source-6)[cite:8](#source-8) |

## 来源、版本与获取深度

以下均于 **2026-11-05** 在线获取。`T` 表示读到正文或源码，不等于逐页复现；`B` 为书目或版本元数据。同一论文的 PDF、arXiv HTML、补充及作者代码不计作独立性能证据。各项事实按原文与固定提交限定，工程推论、条件代数及未执行设计已分别标明。

- <a id="source-1"></a>**[cite:1] ICCV 2021 原始接受稿与书目。** [CVF 会议页](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html)（B：作者、题名、2021、8330–8339）及 [CVF PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf)（T：正文提取读到 §3.3、式 (1)–(4)、β范围，后续返回截断）。CVF 说明为除水印外同接受稿，不冒充 IEEE 排版终版逐页核对。
- <a id="source-2"></a>**[cite:2] arXiv 2108.07610v2。** [固定 v2 正文](https://arxiv.org/html/2108.07610v2)（T：页首日期 2021-09-27、§3.1–3.3；初次文本在混合段截断），另直接读取原 HTML 的 [§3.4](https://arxiv.org/html/2108.07610v2#S3.SS4)、[§4.1](https://arxiv.org/html/2108.07610v2#S4.SS1) 与 [§4.2](https://arxiv.org/html/2108.07610v2#S4.SS2) 相关段，补齐式 (5)、700 轮／400与600衰减、正常图旋转、DTD、缺料边界及消融动机。未逐项重算实验表，未对 v1／v2／终版作全文 diff。
- <a id="source-3"></a>**[cite:3] 作者仓库固定提交。** [提交页](https://github.com/VitjanZ/DRAEM/commit/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef)；实际由 GitHub `commits/main` API 解析出完整 SHA、2021-10-12 UTC 时间与提交消息，再按 SHA 读取文件；[固定 tree API](https://api.github.com/repos/VitjanZ/DRAEM/git/trees/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef?recursive=1) 核文件清单（B／元数据）。此 SHA 只标本卡所核代码，不证明论文所有训练 run 的源码溯源。
- <a id="source-4"></a>**[cite:4] 异常生成与数据加载。** [data_loader.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/data_loader.py)（T：完整文件已读）。初次抓取空内容、GitHub blob 页只给最近提交信息，后以直接 HTTP 获取 raw 正文成功；核对 `MVTecDRAEMTrainDataset` 的增强、Perlin、beta、无异常分支和随机索引，以及测试 mask resize。未执行 imgaug／OpenCV 数据加载。
- <a id="source-5"></a>**[cite:5] 两个子网。** [model_unet.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/model_unet.py)（T：完整文件已读），含 `ReconstructiveSubNetwork`、`DiscriminativeSubNetwork` 与各编码／解码器。结构和形状为静态跟踪，未实例化网络、计参数或测显存。
- <a id="source-6"></a>**[cite:6] 损失实现。** [loss.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/loss.py)（T：完整文件已读），核 focal 默认参数、one-hot 平滑与概率接口、SSIM Gaussian／padding／范围猜测和 CUDA 窗口。focal 文件注明第三方来源，不将此代码当 DRAEM 作者原创损失；本卡未另核所转引 focal 原论文全文、未执行梯度检查。
- <a id="source-7"></a>**[cite:7] 联合训练。** [train_DRAEM.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/train_DRAEM.py)（T：完整文件已读），核初始化、6→2 判别调用、MSE＋SSIM＋focal、无 detach、Adam 两参数组、调度器、正常目录及保存方式。未运行训练。
- <a id="source-8"></a>**[cite:8] 推理与指标。** [test_DRAEM.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/test_DRAEM.py)（T：完整文件已读），核 eval、测试 detach、异常通道、21×21 池化与 raw 像素评分两支路、mask 转 uint8。库的池化默认行为按该调用契约解释；未跑权重、评测或独立 torch 边界探针。
- <a id="source-9"></a>**[cite:9] Perlin 内核。** [perlin.py，固定源码](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/perlin.py)（T：完整文件已读），训练调用 `rand_perlin_2d_np`，核梯度点积、五次 fade、lerp 和网格整除限制；同文件有 octave 版本不代表训练使用。未生成噪声或异常图。
- <a id="source-10"></a>**[cite:10] ICCV 补充材料。** [Supplement PDF](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Zavrtanik_DRAEM_-_A_ICCV_2021_supplemental.pdf)（T：返回的各节文字、图注及 Table 1 已读），主要为 MVTec／DAGM 定性例及模拟监督对照；未逐幅放大检查图像，不引用其图片作为本地输出。
- <a id="source-11"></a>**[cite:11] 使用说明与历史环境。** [README.md](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/README.md)（T：完整正文已读），核 DTD、训练命令及独立的发布权重说明；[requirements.txt](https://raw.githubusercontent.com/VitjanZ/DRAEM/2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef/requirements.txt)（T：读取并核关键条目），是历史 conda 环境规格，含 Python 3.8.11、PyTorch 1.8.0、CUDA toolkit 10.2.89、NumPy 1.20.3、imgaug 0.4.0、OpenCV 4.5.3.56，不是现代 pip 一键兼容性承诺。试探 `scripts/train.sh` 返回 404，随后 tree 确认不存在，故以 README 命令和实际 Python 入口为准。没有下载数据／权重、安装依赖或核验商业部署许可链。

## 验证范围与 G20 销项建议

本次交付只检查文档结构、L1–L5 与答 5 覆盖、公式及源代码静态对应、引用锚点、本地路径、固定 SHA 一致性和单文件写入范围。未执行网络训练、推理、异常生成、自动微分或真实缺陷评估；上述设计测试仍未执行，所有性能与现场验收状态保持空缺。

按台账“已有”的定义，**建议 G20／06-07 从缺卡转为已有具体方法卡**：`draem.md:DR01`，五层 **核／核／部／部／部，答5，未实现**。销的是“DRAEM 异常模拟、双子网与联合损失缺少具体卡”这一文档任务，保留 G20 历史 ID；并非销掉复现、论文／代码参数差异、真实缺陷泛化与业务阈值验收。后续仍需配套版本运行、梯度及边界单元验证和真实留出集评估，可列 P2。

本次**不修改索引、总览、覆盖矩阵或统计**。若后续集中入账且期间无其他变更，可据当前台账将卡数 219→220、正文文件 15→16；总节点仍 113，已有 48→49、缺卡 31→30，部分 10／非算法 24 不变，剩余任务 41→40。06 缺陷异常域为已有 3→4、缺卡 3→2；这些只是待复核的销项增量建议，不表示相关文件已经同步。
