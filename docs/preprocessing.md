# 预处理增强：理论、选型与验收

[返回文档导航](README.md) · [标定坐标](calibration_coordinates.md) · [分割区域](segmentation_regions.md)

[算法卡片总索引](algorithm_cards/README.md) · [预处理46卡索引（含2指标）](algorithm_cards/README.md#预处理与指标) · [L1–L5正文](algorithm_cards/preprocessing.md) · [保留旧稿](algorithms/preprocessing_l1_l5.md)

卡片按文件＋卡号定位，族级教学基线不是所有算法全史；正文覆盖、实现与验证状态分开，本次导航更新不升级验证状态。

## 双线阅读导航

- **工程实践线（按任务选型）**：[任务定义](#任务定义) → [成像与数学原理](#成像与数学原理) → [方法选型表](#方法选型表) → [失败模式](#失败模式) → [现场排障](#现场排障) → [评测与验收](#评测与验收) → [当前仓库状态](#当前仓库状态)。先确认退化与下游公差，再决定是否增强。
- **方法演进线（按问题读论文）**：[代表性演进时间线](#代表性演进时间线) → [并行分支与选型回接](#并行分支与选型回接) → [指标与算法边界](#指标与算法边界) → [论文年份核验记录](#论文年份核验记录)；需要实现背景时回读[传统路线](#传统路线)、[ML 路线](#ml-路线)、[DL 路线](#dl-路线)。年代是阅读顺序，不是部署升级顺序。

两线在“问题、代价和验收”处汇合。后文保留原有工程内容，文末增补的论文方法不代表仓库已经实现。

## 任务定义

预处理把相机原始图像变换为更适合下游定位、测量、分割或识别的图像。输入、输出、位深、颜色空间、ROI、几何尺寸和允许延迟必须明确；目标是降低与任务无关的变化，而不是让图像“看起来更漂亮”。任何增强都可能改变噪声分布、边缘位置、纹理频谱或灰度可比性，因此验收必须同时观察下游指标。

典型子任务包括暗场／平场校正、坏点处理、去噪、背景与照明补偿、动态范围和对比度调整、颜色变换、去模糊及重采样。曝光、焦距、照明和机械稳定性属于采集端；能在采集端消除的问题，不应优先用算法掩盖。

## 成像与数学原理

简化的灰度成像模型可写为

$$I(x,y)=Q\{g(x,y)R(x,y)+d(x,y)+n(x,y)\},$$

其中 $R$ 是场景反射或透射响应，$g$ 是光照与像素增益，$d$ 是暗电流／偏置，$n$ 是读出、散粒和量化等噪声，$Q$ 表示饱和与量化。若有同曝光条件下的暗场 $D$ 和均匀平场 $F$，常见校正为

$$I_c=\frac{I-D}{F-D+\epsilon}\,m,$$

其中 $m$ 用于恢复期望量纲，$\epsilon$ 只防止数值除零，不能修复失效像素。平场、曝光、增益、焦距或光路改变后，校正数据通常不能无条件复用。

线性平移不变滤波是卷积 $I'=I*k$。低通可压制高频噪声，也会削弱窄缺陷和边缘；中值滤波对脉冲噪声有效，但会改变细线和角点。直方图均衡改变全局灰度映射，CLAHE 在局部窗口内限制对比度放大；二者都可能放大噪声。几何重采样通过目标像素反投影并插值，最近邻保标签但锯齿明显，双线性较平滑，双三次可能过冲。对测量任务，必须评估边缘偏移而非只看 PSNR。

颜色图像还涉及传感器响应、去马赛克、白平衡、颜色矩阵、gamma 和颜色空间。非线性 gamma 后的数值不再与光子数线性对应；BGR/RGB 通道顺序、整型范围和浮点范围混淆会直接导致错误。

## 传统、ML 与 DL 路线

### 传统路线

- 采集校正：暗场／平场、坏点替换、固定图样噪声校正、白平衡。
- 空间处理：高斯、中值、双边、非局部均值、形态学背景估计。
- 灰度处理：线性拉伸、gamma、直方图均衡、CLAHE、同态或背景除法。
- 几何处理：裁剪、缩放、旋转、畸变校正；标签图应采用适合离散类别的插值。
- 恢复处理：维纳、盲／非盲反卷积等；需要可信的点扩散函数和噪声模型。

传统方法参数可解释、部署简单，适合成像条件受控且退化模型明确的场景。其主要风险是多个“看似温和”的步骤串联后产生不可忽略的相位、边缘和动态范围偏差。

### ML 路线

ML 路线通常先提取噪声、对比度、频谱、局部纹理和清晰度特征，再用回归、分类或策略模型选择滤波器、参数或是否旁路。它适合有限种、可枚举的成像状态，但需要覆盖现场状态的独立数据；模型选出的参数仍须以最终任务指标验收。

### DL 路线

卷积网络、残差网络和生成模型可用于去噪、去模糊、超分辨率、低照度增强和颜色恢复。其优势是表达复杂退化，风险包括训练域外失真、生成不存在的纹理、边缘幻觉、时延和权重不可追溯。计量和缺陷场景应优先约束几何一致性，保留原图，并建立“增强不可改变判定真值”的独立验证集。

## 方法选型表

| 现场症状 | 首选起点 | 适用前提 | 主要风险 | 验收重点 |
|---|---|---|---|---|
| 固定亮度不均、暗角 | 暗场／平场校正 | 光路与曝光固定 | 分母过小、校正帧失配 | 均匀区残差、边缘位置 |
| 椒盐或坏点 | 中值／坏点表 | 异常稀疏 | 细线被抹除 | 坏点残留、最小缺陷召回 |
| 近似高斯噪声 | 小尺度高斯或保边去噪 | 目标尺度大于核宽 | 模糊边缘 | SNR 与定位偏差双指标 |
| 缓慢照明梯度 | 背景估计后减除／相除 | 背景尺度远大于目标 | 大目标被当背景 | 目标幅度保持、边界伪影 |
| 局部低对比度 | CLAHE 或局部归一化 | 噪声受控 | 噪声和光晕被放大 | 下游召回、假阳性 |
| 已知运动／散焦模糊 | 先修采集，必要时反卷积 | PSF 可估 | 振铃、噪声放大 | 边缘偏差、伪缺陷 |
| 多种复杂退化 | ML 参数选择或 DL 恢复 | 有代表性成对／任务标注 | 域偏移、幻觉 | 原图对照、跨批次下游指标 |

## 失败模式

- 饱和、欠曝或量化后信息已经丢失，拉伸和网络都不能可靠恢复真值。
- 去噪核大于缺陷宽度，导致缺陷消失；锐化和反卷积则可能制造双边缘与振铃。
- CLAHE、局部归一化或背景相除在分块边界、遮挡边缘和低信号区产生光晕。
- 先缩放再测量引入半像素、插值和坐标回填错误；多次重采样叠加模糊。
- 使用不同曝光／温度／光路的平场，校正后反而出现固定纹理。
- 用 PSNR/SSIM 单独选型：整体图像相似不等于小缺陷召回或亚像素位置稳定。
- 训练集只含合成噪声，真实相机的信号相关噪声、压缩、坏点和批次漂移未覆盖。

## 现场排障

1. 冻结原始帧、相机参数、光源状态、镜头、距离、温度和软件版本，禁止只保存增强结果。
2. 查看原图直方图、饱和比例、暗区均值／方差、均匀区空间残差和重复帧差分。
3. 逐步旁路处理链，从最接近采集端的步骤开始定位；每次只改变一个参数。
4. 对边缘、细线、小缺陷和 ROI 边界做局部放大，比较处理前后坐标与幅值。
5. 核对 dtype、数值范围、通道顺序、颜色空间、边界填充和插值方式。
6. 若仅某批次失败，优先查曝光、增益、光源衰减、焦点、污染和校正帧版本，而非立即重调阈值。
7. 用下游定位／分割／测量结果复判；视觉观感只能作为辅助证据。

## 评测与验收

评测集应按设备、批次、曝光、温度、焦点、缺陷尺度和困难条件分层，并保留未经处理原图。成对真值可报告 MAE/MSE、PSNR、SSIM；均匀性可报告校正后均值的空间极差或变异系数；噪声可报告均匀 ROI 的标准差和功率谱。没有无噪声真值时，不应伪造 PSNR。

工业验收还应包含：边缘／角点位移，最小缺陷的召回与假阳性，定位或尺寸误差，重复采集离散度，最坏帧时延、内存及失败时旁路策略。阈值必须来自冻结的验证集，最终在独立验收集报告，并与“无预处理”基线配对比较。

## 当前仓库状态

- 当前没有独立预处理模块、统一配置、专用测试或预处理评测报告。
- [pose_ncc.py](../traditional/template_matching/pose_ncc.py) 内含灰度输入检查、旋转／尺度插值和金字塔面积重采样；这些是匹配实现细节，不是通用增强管线。
- [edge_subpixel.py](../traditional/subpixel/edge_subpixel.py)、[caliper.py](../traditional/caliper_measurement/caliper.py) 内含任务局部的高斯平滑和双线性采样；[focus_measures.py](../traditional/autofocus/focus_measures.py) 含灰度转换。它们没有组成统一预处理接口。
- [synthetic.py](../data/synthetic.py) 可生成噪声、模糊和几何扰动数据，只是测试数据基础，不是增强实现。
- ML 目录目前只有匹配候选重排序示例；DL 目录没有去噪、去模糊或超分辨率实现。不得把研究路线视为仓库能力。

## 权威官方与原始参考

- OpenCV 官方：[Image Filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)、[Histogram Equalization](https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html)、[CLAHE](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html)、[Color Space Conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)、[Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)。
- OpenCV 官方：[Non-Local Means Denoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html) 与 [Image Denoising API](https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html)。
- Wang et al., [Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861)，SSIM 原始论文。
- scikit-image 官方：[Image Metrics API](https://scikit-image.org/docs/stable/api/skimage.metrics.html)，用于核对 PSNR、SSIM 等定义；本仓库未依赖 scikit-image。

## 代表性演进时间线

核验日期：2026-11-04。以下选择 6 个方法里程碑，主要覆盖保边滤波、非局部去噪与学习式恢复，并非预处理全史；采集校正、对比度增强、反卷积和几何重采样仍是独立分支。年份按所列原论文的会议／期刊发表年排列，预印本另记；“改进”指机制变化，不承诺任何数据集或工位上必然更优。代价与适用场景是据方法假设作出的工程归纳，不是本仓库实测结果。

| 年份／方法与分支 | 所解决的问题 | 代表性改进 | 代价与限制 | 适用场景 | 原始来源 | 本仓库状态 |
|---|---|---|---|---|---|---|
| 1998 · 双边滤波（局部保边） | 普通空间平均跨越强边缘，去噪与保边冲突 | 联合空间距离和灰度／颜色相似度加权，减弱跨边缘混合 | 空间与值域尺度需调节；强噪声影响权重，细纹理仍可能消失；直接计算随邻域增大而变贵 | 边界明显、噪声适中、无需训练的平滑 | [cite:1](#pre-source-1) | 未实现或封装；现有局部高斯平滑不是双边滤波 |
| 2005 · NLM（非局部自相似） | 只利用近邻像素，难利用远处重复结构 | 以图像块相似度寻找并平均非局部样本 | 搜索窗与块比较开销；重复结构不足或噪声过强时匹配不稳 | 重复纹理较多、允许更大计算量的单幅去噪 | [cite:2](#pre-source-2) | 未实现或接入 NLM；测试数据加噪不等于去噪实现 |
| 2007 · BM3D（分组变换域） | 单纯相似块平均未充分利用组内稀疏结构 | 相似二维块堆成三维组，变换域收缩、协同维纳滤波与重叠聚合 | 分组、变换和聚合成本；依赖噪声模型／强度；实现的商业许可须另核对 | 噪声模型可估、希望无训练去噪的质量基线 | [cite:3](#pre-source-3) | 未实现，无专用依赖或基准；作者代码可获得不等于可直接商用 |
| 2010 · 引导滤波（局部线性保边） | 大窗口保边处理成本与部分边缘伪影问题 | 用引导图与输出的局部线性关系建模，可用均值统计实现随像素数线性增长的计算 | 引导结构可能被转移；引导错误、配准偏差和正则参数影响细节 | 保边平滑、细节分解、可信引导下的上采样 | [cite:4](#pre-source-4) | 未实现；没有引导图输入协议或一致性测试 |
| 2017 · DnCNN（监督残差学习；预印本 2016） | 手工先验对复杂图像分布表达有限 | CNN 学习噪声残差，结合批归一化；包含未知高斯噪声水平的训练设定 | 需要训练数据和权重管理；真实噪声域偏移、算力成本、细缺陷被抹除风险 | 训练退化覆盖现场、能做原图配对验收的恢复 | [cite:5](#pre-source-5) | 未实现，无 DnCNN 权重、训练或验收记录 |
| 2018 · Noise2Noise（噪声监督训练） | 获取干净监督目标困难 | 在相应统计条件下，用同一潜在信号的噪声观测训练恢复器，不必用干净目标 | 仍需合适的配对／对齐数据；平方损失下目标噪声的条件均值须正确，相关噪声或运动可能造成偏差 | 可重复观测静态对象、干净参考难采集的去噪 | [cite:6](#pre-source-6) | 未实现，无噪声对采集协议、训练或验证集 |

## 并行分支与选型回接

- **保边分支**：双边滤波与引导滤波都可用于局部平滑；后者改变建模和计算方式，不保证所有纹理上更好。回到[方法选型表](#方法选型表)的噪声行，并用边缘位移验收。[cite:1](#pre-source-1) [cite:4](#pre-source-4)
- **自相似分支**：NLM 与 BM3D 都利用块重复性，BM3D 增加组内变换域协同处理；它们不是 CLAHE 或平场校正的替代品。重复性不足时，不应只因年代较新选择 BM3D。[cite:2](#pre-source-2) [cite:3](#pre-source-3)
- **学习分支**：DnCNN 是恢复网络方案，Noise2Noise 主要改变监督数据条件，二者处于不同设计维度；不能写成“Noise2Noise 淘汰 DnCNN”。ML 参数选择也可与传统滤波共存。[cite:5](#pre-source-5) [cite:6](#pre-source-6)
- **采集与增强分支**：先处理曝光、饱和与固定图样，再按需要去噪或增强；CLAHE 改变局部对比度，并不估计干净图像。每个可选步骤都应支持旁路，不要求串齐时间线全部算法。

## 指标与算法边界

SSIM 原论文发表于 **IEEE TIP 2004**，不是 DOI 中出现的 2003；它是全参考图像质量指标，不是增强／去噪算法，因此不计入上述 6 个里程碑。[cite:7](#pre-source-7) PSNR 同样是评价指标；以它们构造损失，也不自动得到可部署的恢复算法。SSIM／PSNR 的提高不能替代[评测与验收](#评测与验收)中的定位偏差、最小缺陷召回和时延检查。

## 论文年份核验记录

年份核验优先采用原论文、正式论文集和作者机构记录；下面逐项保留标题、版本与可追溯链接。标记 `[CONFIRMED]` 表示列出的出版信息已由原始记录确认，不表示经过三方独立复现实验；作者页与作者预印本不算独立性能证据。仅由单一作者记录明确区分的版本标为 `[SINGLE-SOURCE]`，不据此宣称方法全面优越。没有列出早期稿的条目，只确认指定论文的正式年份，不声称不存在更早工作。

- <a id="pre-source-1"></a>**[cite:1] [CONFIRMED]** Tomasi and Manduchi, *Bilateral Filtering for Gray and Color Images*，**ICCV 1998**；[作者机构书目](https://fds.duke.edu/db/aas/ml/faculty/tomasi/publications/237327/printer)与[原论文镜像](https://home.cis.rit.edu/~cnspci/references/dip/filtering/tomasi1998.pdf)一致。这里确认的是这篇代表论文，不把 1998 声称为所有保边滤波的起点。
- <a id="pre-source-2"></a>**[cite:2] [CONFIRMED]** Buades, Coll and Morel, *A Non-Local Algorithm for Image Denoising*，**CVPR 2005**；[原论文](https://stanford.edu/class/ee367/reading/NLM.pdf)、[书目交叉核对](https://scikit-image.org/docs/0.25.x/api/skimage.restoration.html)。作者的 [*Non-Local Means Denoising*（IPOL 2011）](https://www.ipol.im/pub/art/2011/bcm_nlm/)是后续实现说明，不能将 2011 当作 NLM 首篇会议论文年，也不能把网页后来的 PDF 转换日期当发表年。
- <a id="pre-source-3"></a>**[cite:3] [CONFIRMED]** Dabov et al., *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering*，**IEEE TIP 2007，16(8):2080–2095**；[作者机构记录](https://researchportal.tuni.fi/fi/publications/image-denoising-by-sparse-3-d-transform-domain-collaborative-filt/)与[原论文镜像](https://sites.units.it/ramponi/teaching/DIP/DIPmaterials/z08_BM3D_Foi07.pdf)。[作者项目页](https://webpages.tuni.fi/foi/GCF-BM3D/index.html)还列有早期版本和后续变体，本条不把所有 BM3D 家族论文合并为同一年；该页下载软件明确限制非营利教学／科研用途，工业使用须另核许可。
- <a id="pre-source-4"></a>**[cite:4] [SINGLE-SOURCE]** He, Sun and Tang, *Guided Image Filtering*；[作者项目及原论文入口](https://people.csail.mit.edu/kaiming/eccv10/index.html)分别标注 **ECCV 2010** 与 **TPAMI 2013**，时间线采用会议版年份；不是 2015 年的 *Fast Guided Filter*。
- <a id="pre-source-5"></a>**[cite:5] [CONFIRMED]** Zhang et al., *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*；[arXiv v1](https://arxiv.org/abs/1608.03981)为 **2016-08-13**，[作者机构期刊记录](https://research.polyu.edu.hk/en/publications/beyond-a-gaussian-denoiser-residual-learning-of-deep-cnn-for-imag)为 **IEEE TIP 2017，26(7):3142–3155**。不是“2016 年会议论文”。
- <a id="pre-source-6"></a>**[cite:6] [CONFIRMED]** Lehtinen et al., *Noise2Noise: Learning Image Restoration without Clean Data*；[arXiv v1](https://arxiv.org/abs/1803.04189)为 **2018-03-12**，[正式论文集](https://proceedings.mlr.press/v80/lehtinen18a.html)为 **ICML 2018，PMLR 80:2965–2974**；统计条件见该页原文及补充材料，不能简化成“任意两张噪声图都能训练”。
- <a id="pre-source-7"></a>**[cite:7] [CONFIRMED]** Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*；[作者机构记录](https://nyuscholars.nyu.edu/en/publications/image-quality-assessment-from-error-visibility-to-structural-simi)确认 **IEEE TIP 2004，13(4):600–612**，[出版记录](https://dl.acm.org/doi/10.1109/tip.2003.819861)对应同一 DOI；2003 是收稿／修订相关年份，不是该期刊卷期年。

仓库状态依据本次对 `traditional/`、`ml/`、`dl/`、`data/`、`tests/` 的源码和文件清单核对；“未实现”指未发现专用实现、接入及验证链路，不是说外部库没有该能力。已有局部处理的证据入口仍见[当前仓库状态](#当前仓库状态)。本次没有运行论文复现、下载权重或新增依赖。
