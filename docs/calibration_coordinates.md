# 标定与坐标：理论、链路与验收

[返回文档导航](README.md) · [预处理增强](preprocessing.md) · [分割区域](segmentation_regions.md) · [现有匹配坐标协议](pose_ncc.md#返回协议与坐标)

[算法卡片总索引](algorithm_cards/README.md) · [标定10卡索引](algorithm_cards/README.md#标定坐标) · [L1–L5正文](algorithm_cards/calibration.md)

2026-11-06覆盖回链：[KB01全模型与标定](algorithm_cards/kb_zernike_extensions.md#kb01)补P23／P9／P6、四系数换算、球面H四步标定及圆点质心，[G01／02-04](evolution_coverage.md#02-标定坐标)部分转已有／P2。KB01仍部／核／部／部／部、答5；C03子模型不升级。原坐标次序／全零因子初始化及规范仍待核，无标定接口或实拍验收；原来源记录与主表六节点不变。

卡片含成像模型、估计方法与观测前端，不是全部标定算法全史。确定性代数和外部API冒烟不等于真实标定或物理验收，本次导航更新不升级验证状态。

## 双线阅读导航

- **工程实践线（按坐标链落地）**：[任务定义](#任务定义) → [成像与数学原理](#成像与数学原理) → [方法选型表](#方法选型表) → [失败模式](#失败模式) → [现场排障](#现场排障) → [评测与验收](#评测与验收) → [当前仓库状态](#当前仓库状态)。先明确矩阵方向、单位与可观测量，再选择求解器。
- **方法演进线（按问题读论文）**：[代表性演进时间线](#代表性演进时间线) → [并行分支与选型回接](#并行分支与选型回接) → [指标与算法边界](#指标与算法边界) → [论文年份核验记录](#论文年份核验记录)；对照[传统路线](#传统路线)、[ML 路线](#ml-路线)、[DL 路线](#dl-路线)，区分成像模型、鲁棒估计、手眼关系与场景重定位。

两线共同回到独立物理控制点验收；时间线是代表性研究脉络，不是从旧到新的单一替代链。原有章节保留，研究方法的本地实现状态逐项另记。

## 任务定义

标定用于估计成像模型和坐标系之间的变换，使像素观测能以明确约束映射到相机、工件、机器人或世界坐标。应区分：相机内参与畸变、相机相对标定板的外参、平面像素到工件坐标的单应、双目外参、手眼关系，以及最终测量尺度。它们不是同一个矩阵，也不能由一次模板匹配自动替代。

任务必须定义坐标系名称、轴向、手性、单位、变换方向、时间戳和参考点。例如 $^AT_B$ 表示把 B 系点表达为 A 系；若团队采用相反约定，应在接口中固定，禁止只写含糊的 `transform`。

## 成像与数学原理

针孔模型在齐次坐标下为

$$s\begin{bmatrix}u\\v\\1\end{bmatrix}=K\begin{bmatrix}R&t\end{bmatrix}\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix},\qquad
K=\begin{bmatrix}f_x&\gamma&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.$$

$K$ 是内参，$R,t$ 是外参，$s$ 与深度相关。单个像素通常只确定一条空间射线，不能在未知深度时唯一恢复三维点。镜头畸变先作用于归一化成像坐标；常见模型含径向系数和切向系数。去畸变改变像素坐标，后续 ROI、模板和标注必须说明处于原始图还是校正图。

同一平面 $Z=0$ 上可写成单应关系

$$s\,p=H\,P_{plane},$$

但单应只适用于同一平面或纯旋转等特定条件。把固定的 `mm/px` 用于有透视、深度变化或畸变的视场，会产生位置相关误差。

刚体变换用齐次矩阵表示：

$$^AT_C=\,^AT_B\,^BT_C,\qquad (^AT_B)^{-1}=\begin{bmatrix}R^T&-R^Tt\\0&1\end{bmatrix}.$$

变换链顺序不可交换。手眼标定经典形式为 $AX=XB$；眼在手上和眼在手外的坐标定义、采样运动和输出方向不同，必须对照具体 API，而不能仅凭算法名判断。

## 传统、ML 与 DL 路线

### 传统路线

- 平面靶标多视角标定：检测棋盘格、圆点或 ChArUco 特征，初始化内外参，再最小化重投影误差。
- PnP：已知三维点与二维像点时估计相机姿态；退化构型、共面性和初值会影响解。
- 平面映射：用对应点估计单应，配合 RANSAC 抑制误匹配；只覆盖被验证的工作平面。
- 双目／多相机：估计各自内参和相对外参，校正后通过视差与基线三角化。
- 手眼：采集多组机器人和相机相对运动，求解坐标链中的固定刚体关系。

### ML 路线

ML 可用温度、焦距、位置、残差模式等手工特征回归系统误差，或识别失效标定帧。补偿模型必须建立在可追溯标定链之上，不能替代坐标定义；训练与测试要按设备、工位或时间分组，避免同一标定序列泄漏。

### DL 路线

DL 可用于靶标点检测、无靶自标定、单目深度、位姿或畸变参数估计。它适合特征难检测或模型复杂的情况，但尺度可观测性、域偏移和几何一致性仍需显式约束。输出小数位数不是精度证明；未经独立量具或三维真值验证，不能宣称达到物理测量精度。

## 方法选型表

| 需求 | 首选方法 | 必要条件 | 不适用或风险 | 核心验收 |
|---|---|---|---|---|
| 去畸变、统一射线模型 | 多视角内参标定 | 靶标尺寸可信、覆盖视场与姿态 | 焦距／光路改变后失配 | 留出帧重投影误差、直线残差 |
| 平面工件像素转毫米 | 去畸变后平面单应 | 工件共面、工作距离固定 | 高度变化造成系统偏差 | 全视场控制点物理误差 |
| 已知 3D 点求相机姿态 | PnP + 鲁棒估计 | 对应正确、构型可观 | 共线／近退化、镜像解 | 位姿误差、重投影与稳定性 |
| 双目三维 | 双目标定与校正 | 同步、纹理、基线合适 | 遮挡、反光、远距视差过小 | 深度偏差、完整率、重复性 |
| 相机到机器人坐标 | 手眼标定 | 多样且准确的机器人运动 | 运动轴单一、时间不同步 | 独立点位 TCP／靶点误差 |
| 温漂或装调残差补偿 | 物理标定 + ML 残差模型 | 有跨状态物理真值 | 外推、掩盖松动 | 按设备／温度留出验证 |

## 失败模式

- 标定板打印缩放、翘曲或方格尺寸填写错误，重投影误差仍可能较小但物理尺度错误。
- 标定图姿态单一、只在视场中央或全部近正视，参数相关性强，边缘外推失真。
- 混用 `(x,y)` 与数组 `[y,x]`、像素中心与像素外边界、度与弧度、毫米与米。
- 把 ROI 坐标直接当整图坐标，或只加中心偏移而未组合完整变换。
- 先裁剪／缩放图像却未同步更新内参；对已去畸变图再次使用原始畸变参数。
- 左乘／右乘、主动／被动旋转、行向量／列向量约定混用，变换链在单点上偶然正确。
- 用训练残差、同一批标定图或平均重投影误差替代独立空间控制点验收。
- 机械松动、变焦、重新对焦、温漂或相机触发与机器人位姿不同步，使静态标定失效。

## 现场排障

1. 画出所有坐标系及轴向，逐项写明每个矩阵“输入系 → 输出系”、单位和时间戳。
2. 用原点和三个轴上的已知点做手算单元测试，再检查逆变换和往返闭环。
3. 将检测角点和重投影点覆盖到每张原图，按图、半径和视场位置查看残差，而非只看均值。
4. 核对图像是否裁剪、缩放、翻转、去畸变；同步修正主点、焦距和 ROI 偏移。
5. 平面任务在视场四角和中心放独立控制点，并加入允许的最大高度变化。
6. 手眼任务检查运动多样性、机器人位姿单位、旋转表达、采集同步和眼在手上／手外配置。
7. 若误差随温度、焦距、位置或时间单调变化，先查机械和光学状态，再考虑补偿模型。

## 评测与验收

内参标定至少报告：有效帧数、靶标覆盖、每帧及全局重投影 RMSE、留出图误差、视场位置分布和重复标定参数离散度。平均重投影误差只能说明所选模型对图像点的拟合，不能单独证明毫米精度。

物理验收应使用未参与求解、可追溯的控制点或量块，报告 x/y/z 或距离的偏差、绝对误差分位数、最大误差、重复性和全视场误差图。手眼验收应执行独立位姿的闭环落点或抓取测试；双目应按距离和表面类型报告深度误差与无效比例。接受阈值来自工艺公差和测量不确定度预算，不应直接照搬库函数默认值。

## 当前仓库状态

- 当前没有相机内参／畸变、双目、PnP、平面毫米映射、手眼标定的专用实现、标定数据或物理精度测试。
- [pose_ncc.py](../traditional/template_matching/pose_ncc.py) 输出模板到图像的二维相似变换，包含原点、中心、包围框和角点；它不是相机标定或手眼标定。
- [pose_ncc.md](pose_ncc.md#第一章坐标与变换) 已说明数组 `[y,x]`、像素中心、旋转方向、ROI 回填和 3×3 变换。该协议只覆盖现有匹配接口。
- [caliper.py](../traditional/caliper_measurement/caliper.py) 输出像素坐标和像素 RMS；没有像素到毫米链路。合成实验中的亚像素误差不能转换为真实尺寸承诺。
- ML/DL 当前没有标定或坐标补偿实现，也没有已训练权重或现场评测。

## 权威官方与原始参考

- OpenCV 官方：[Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)、[Camera Calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)、[Perspective-n-Point](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。
- OpenCV 官方：[Hand-Eye Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga887960ea1bde84784e7f1710a922b93c) 与 [ChArUco calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)。
- Zhengyou Zhang, [A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/), IEEE TPAMI, 2000。
- Tsai and Lenz, [A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration](https://doi.org/10.1109/70.34770), IEEE Transactions on Robotics and Automation, 1989。
- Hartley and Zisserman, [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)，多视几何原始教材站点。

## 代表性演进时间线

核验日期：2026-11-04。以下 6 个里程碑覆盖几何链路中的不同子问题，不是相机标定全史，也不把 RANSAC 或重定位网络统称为“内参标定算法”。表按所选正式论文年份排序，技术报告、预印本和扩展期刊版分开记。代价与适用场景是依据方法假设作出的工程归纳，不能当作本仓库精度实测。

| 年份／方法与分支 | 所解决的问题 | 代表性改进 | 代价与限制 | 适用场景 | 原始来源 | 本仓库状态 |
|---|---|---|---|---|---|---|
| 1981 · RANSAC（鲁棒模型估计） | 错误对应点使直接拟合偏离真实模型 | 最小样本生成假设，以一致集区分内点与外点 | 阈值和模型必须匹配；外点多时试验次数增加；退化样本仍会失败 | 与单应／PnP 求解器组合，剔除错误对应 | [cite:1](#cal-source-1) | 未接入标定链；`learned_matching.py` 仅在说明中计划 RANSAC，函数仍抛出 `NotImplementedError` |
| 1989 · Tsai–Lenz（手眼标定） | 机器人末端与相机之间的固定刚体关系未知 | 用多组相对运动约束求解手眼关系，分步估计旋转与平移 | 需要可靠机器人运动与视觉位姿；运动轴单一导致退化，旋转误差会影响平移 | 眼在手上系统的相机—末端关系求解 | [cite:2](#cal-source-2) | 未实现，无机器人运动数据、手眼求解或落点测试 |
| 1999 · Zhang 平面靶标标定（内参与畸变；期刊 2000） | 传统三维标定装置昂贵且采集受限 | 多个未知姿态的平面靶标构成约束，闭式初始化后非线性优化 | 靶标尺寸／平整度与视场覆盖关键；近正视、姿态单一会病态 | 常规相机多视角内参标定，作为后续坐标链基础 | [cite:3](#cal-source-3) | 未实现；模板相似变换不能估计相机内参或畸变 |
| 2006 · Kannala–Brandt（广角／鱼眼模型） | 透视投影加常规畸变项难覆盖极大视角 | 以入射角相关投影模型描述广角和鱼眼成像，并进行参数标定 | 模型阶数和可观测性要控制；外围采样不足导致外推风险，反投影需配套模型 | 大视场镜头、鱼眼相机的射线建模 | [cite:4](#cal-source-4) | 未实现，无鱼眼数据、投影／反投影或外围误差测试 |
| 2007 · EPnP（已标定相机位姿；期刊 2009） | 大量 3D–2D 对应的 PnP 求解成本 | 用控制点表达几何关系，给出随对应点数线性增长的非迭代方案 | 内参及对应应可信；噪声、退化构型仍需检查，常结合鲁棒估计与重投影精化 | 已有内参与三维参照点时的外参／物体位姿估计 | [cite:5](#cal-source-5) | 未实现，无 `solvePnP` 接入或独立位姿测试 |
| 2015 · PoseNet（学习式场景重定位） | 显式局部特征对应在部分困难图像中不稳 | 用 CNN 从 RGB 图像直接回归场景坐标系中的六自由度相机位姿 | 需要场景位姿监督和训练；域偏移、精度及尺度依赖训练参考系，不建立可追溯内参／手眼链 | 已建图场景重定位、粗位姿候选研究 | [cite:6](#cal-source-6) | 未实现，无 PoseNet 模型、场景位姿数据或物理精度验收 |

## 并行分支与选型回接

- **模型与标定**：Zhang 的多视角平面约束与 Kannala–Brandt 的鱼眼投影处理不同问题；可借用类似靶标采集流程，但须使用匹配的投影模型、初始化和残差，不能认为鱼眼模型“替代所有张氏标定”。[cite:3](#cal-source-3) [cite:4](#cal-source-4)
- **鲁棒估计与位姿**：RANSAC 是外层模型估计框架，EPnP 是 PnP 求解器，可组合而非互相替代。PnP 通常以前置内参为条件，不能把其外参输出当成完整相机标定。[cite:1](#cal-source-1) [cite:5](#cal-source-5)
- **手眼关系**：Tsai–Lenz 连接机器人运动链与相机链；即使相机内参已知，固定手眼变换仍需独立求解。眼在手外配置必须重新核对输入／输出方向，不应照搬眼在手上样例。[cite:2](#cal-source-2)
- **学习与补偿**：PoseNet 是场景重定位分支，不是温漂补偿、内参标定或毫米级计量的替代品；ML 残差补偿也需要先建立物理链。[cite:6](#cal-source-6) 按[方法选型表](#方法选型表)的具体输出选分支，再执行[现场排障](#现场排障)中的闭环和控制点检查。

## 指标与算法边界

重投影 RMSE、旋转／平移误差、闭环落点误差和重复性是评价量；把重投影误差用于优化，仍需配套明确的成像模型、参数与求解步骤。RANSAC 是算法框架，不是“内点率指标”；内点率高也不证明物理尺度正确。现有 [metrics.py](../tests/metrics.py) 的像素 RMSE 和 [caliper.py](../traditional/caliper_measurement/caliper.py) 的拟合 RMS，均不能直接解释成毫米标定精度。

PoseNet 的会议版与后续修订预印本包含不同的朝向误差数值；后者明确说明修正数值错误。本时间线只引用机制与年份，不混合两个版本的性能数字，更不将其结果换算成本仓库测量能力。[cite:6](#cal-source-6)

## 论文年份核验记录

正式会议／卷期年、在线发表年、预印本首投年是不同字段；不从 DOI、PDF 文件名或网页更新时间反推发表年。`[CONFIRMED]` 表示出版信息得到原始记录确认，不代表三方独立复现；只有一个原始书目记录支持的条目标 `[SINGLE-SOURCE]`。作者机构页和其原稿不能算独立性能证据，下面未作算法优劣排名。

- <a id="cal-source-1"></a>**[cite:1] [CONFIRMED]** Fischler and Bolles, *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*，**Communications of the ACM 1981，24(6):381–395**；[出版方页面](https://cacm.acm.org/research/random-sample-consensus/)与[原论文镜像](https://cecas.clemson.edu/~stb/ece847/internal/classic_vision_papers/fischler_ransac_1981.pdf)。不是后来 OpenCV API 的发布年份。
- <a id="cal-source-2"></a>**[cite:2] [SINGLE-SOURCE]** Tsai and Lenz, *A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration*，**IEEE Transactions on Robotics and Automation 1989，5(3):345–358**；[DOI](https://doi.org/10.1109/70.34770)对应的 [IEEE 书目索引](https://xplorestaging.ieee.org/document/34770/metrics)明确列出 June 1989。此处是手眼论文，不能与 Tsai 的其他相机标定论文混名混年。
- <a id="cal-source-3"></a>**[cite:3] [CONFIRMED]** Zhang 的 [*Flexible Camera Calibration by Viewing a Plane from Unknown Orientations*](https://www.microsoft.com/en-us/research/publication/flexible-camera-calibration-viewing-plane-unknown-orientations/)发表于 **ICCV 1999**；[*A Flexible New Technique for Camera Calibration*](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)发表于 **IEEE TPAMI 2000，22:1330–1334**。更早 [MSR-TR-98-71 技术报告原稿](https://www.johnloomis.org/ece564/notes/CameraCalibration/readings/TR98-71.pdf)署 **1998**，另列 1999 等更新日期；三者不能只写成一个含糊的“1998 论文”。
- <a id="cal-source-4"></a>**[cite:4] [CONFIRMED]** Kannala and Brandt, *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*，**IEEE TPAMI 2006，28(8):1335–1340**；[作者书目](https://users.aalto.fi/~kannalj1/publications/publications.html)及[原论文镜像](http://www.close-range.com/docs/A_GENERIC_CAMERA_MODEL_AND_CALIBRATION_METHOD_Kannala-Brandt_pdf697.pdf)。本条针对该期刊论文，不声称鱼眼标定始于 2006。
- <a id="cal-source-5"></a>**[cite:5] [CONFIRMED]** Moreno-Noguer, Lepetit and Fua, *Accurate Non-Iterative O(n) Solution to the PnP Problem*，**ICCV 2007**；Lepetit, Moreno-Noguer and Fua, *EPnP: An Accurate O(n) Solution to the PnP Problem*，**IJCV 2009，81:155–166**。[作者项目页](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/)分别列出两版，[会议原稿](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/Moreno-NoguerFL07.pdf)提供机制说明；期刊 DOI 中的 2008 不应改写为会议年。
- <a id="cal-source-6"></a>**[cite:6] [CONFIRMED]** Kendall, Grimes and Cipolla, *PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization*；[arXiv](https://arxiv.org/abs/1505.07427)首投 **2015-05-27**，[正式会议页](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html)为 **ICCV 2015**。arXiv v4 是 **2016-02-18** 修订，注明朝向误差数值纠正；修订年不是新会议年。

仓库状态经 `traditional/`、`ml/`、`dl/` 与 `tests/` 清单和源码核对，未发现以上标定／重定位的专用实现与验收链。尤其 [learned_matching.py](../dl/learned_matching.py) 中的 `findHomography(RANSAC)` 是计划说明而非执行代码；[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 是二维相似匹配，不是 EPnP 或 Zhang 标定。外部算法存在不等于本仓库已集成，本次未进行标定实验或物理精度测量。
