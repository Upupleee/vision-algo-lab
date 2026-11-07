# 边缘几何测量：方法演进与工程实践双线总览

[文档导航](README.md) · [编写与学习规范](algorithm_learning_standard.md) · [测量与对焦卡片索引](algorithm_cards/README.md#测量与对焦) · [独立对焦总览](autofocus_evolution.md) · [标定坐标](calibration_coordinates.md)

2026-11-06覆盖回链：[DE95](algorithm_cards/edge_evolution_extensions.md#de95)、[DE17](algorithm_cards/edge_evolution_extensions.md#de17)、[HE01](algorithm_cards/edge_evolution_extensions.md#he01)分别承接G03／G05／G04，按[覆盖矩阵](evolution_coverage.md#05-边缘几何测量)由部分转已有／P2；均核／核／部／核（各自限定）／部、答5，原版未实现，L4不提供一般精度界。HE01与旧[S21](algorithm_cards/segmentation.md#s21-u-net-与-edgenet-边界)合计仍是05-09一节点，不新增U-Net卡。DE17容差NMS下η=5/6反例、构链顺序／AGPL、本地M04简化与EdgeNet风险保留。[ZM01](algorithm_cards/kb_zernike_extensions.md#zm01)全部分、答5，G02仍部分／P0，1993原算子与A31缺证；不升级M05或回填2020原式。主表11节点及本章历史来源／未实现记录不改，新增卡证据另账。

## 1. 范围与证据口径

本章讨论二维图像中的边缘、角点、亚像素定位与几何拟合，并把物理量标定、不确定度和拒识作为测量链的必要边界。**方法演进线**回答“先前瓶颈、改变的维度、改进机制、代价与遗留问题”；**工程实践线**回答“当前选什么、源码做到哪里、失败看什么证据、怎样验收”。它不是完整算法史，也不是传统、ML、DL 逐代替换的排名。

核验日期：**2026-11-04**。年份采用指定原始文献或规范版本，不把收稿、入库、软件发布日期当作方法提出日期。来源编号为本章本地索引；逐 URL 的实际访问深度见第 7 节。全文入口已取得但返回被截断时，只记“全文片段”，不声称逐页读完。原文摘要中的性能措辞不作为通用保证，不给跨论文精度／速度排名。

“已有实现”“第三方封装／示例”“网络与损失骨架”“未实现”“历史测试记录”“现场验证”分别判断。卡片的 L1–L5 覆盖不因总览建立而升级；本章不修复源码或旧文献表。已有 [研究笔记](research_notes.md) 和 [实验记录](experiments.md) 是历史记录，不是本次重新测得的数据。

## 2. 核心演进维度与双线连接

| 核心维度 | 持续解决的问题 | 机制与代价 | 工程应保留的证据 |
|---|---|---|---|
| 成像与边缘模型 | 物体边界、灰度过渡中心和采样像素不是同一对象 | 阶跃、斜坡、erf／PSF、像素面积积分逐步明确假设；模型越细，参数越多 | 原灰度剖面、曝光／饱和、模糊宽度、像素中心和采样定义 |
| 亚像素估计 | 整数检测不足以表达连续位置 | 梯度峰插值、矩比、非线性拟合是并行估计器；提高精度也引入曲率／归一化／初始化敏感性 | 法向误差、角度与亚像素相位扫描、失败率，而非输出小数位数 |
| 二维可观测性 | 直边沿切向不可定位，角点可提供两个方向约束 | 结构张量／局部正交约束；窗口过大可混入其他结构 | 最小特征值、条件数、初值、窗口支持、拒绝或保持原点的比例 |
| 几何拟合 | 局部点需转成尺寸所依赖的几何要素 | PCA 正交直线、加权估计、鲁棒共识；鲁棒性需要阈值和计算预算 | 点集覆盖、正交残差、异常点、几何模型是否正确 |
| 标定与不确定度 | 像素重复性不等于毫米准确度 | 坐标链、畸变／尺度、误差传播和相关性建模；需要可追溯参考量 | 标定版本、测量定义、环境、重复采集、不确定度预算 |
| 数据先验与算力 | 手工规则难区分杂乱边缘与目标边界 | 浅层结构化学习或多尺度网络；付出标注、训练、域偏移和推理资源成本 | 独立工件划分、权重版本、边界标签精度、端到端延迟 |

上述维度为工程分析框架。成像／插值误差由 Devernay 与后续 IPOL 分析支撑，矩模型由 Ghosal–Mehrotra 和后续斜坡模型支撑，参数拟合与不确定度另有独立路线；不存在一个方法同时消除所有误差的结论。[cite:3](#me-source-3)[cite:4](#me-source-4)[cite:5](#me-source-5)[cite:6](#me-source-6)[cite:10](#me-source-10)[cite:11](#me-source-11)

| 工程环节 | 可组合的方法角色 | 当前仓库边界 |
|---|---|---|
| 成像与 ROI | 控曝光／焦点、定位工件、定义测量法向 | 有局部处理与匹配基础；无完整标定测量系统 |
| 候选检测 | Sobel 幅值、Canny、Harris、ML／DL 边界预测 | Sobel 已调用；完整 Canny 未接入；Harris 是示例 |
| 连续定位 | 一维抛物线／erf、简化 Devernay、矩模型、角点精化 | 前三类和两种角点精化有函数；Zernike 无函数 |
| 几何估计与拒识 | 点集覆盖、PCA／鲁棒拟合、模型质量门 | 仅有无权 PCA 直线；无 RANSAC、圆／椭圆测量链 |
| 物理量输出 | 标定映射、尺寸定义、不确定度及验收规则 | 未实现；不能把 `CaliperResult.points` 称世界坐标 |

## 3. 方法演进线：代表性里程碑

下表列 **11 个代表节点**，包含方法、学习分支、规范及可复现实现四种类型。年份用于定位文献，不宣称相关数学思想均由该文首创；每行工程用途为条件性选型分析。

| 年份／版本与分支 | 先前瓶颈 → 核心改进 | 新增代价与遗留问题 | 适用场景与卡片 | 原始来源／仓库状态 |
|---|---|---|---|---|
| 1981，RANSAC／鲁棒估计 | 全点最小二乘受粗差拖偏 → 最小样本建模、残差共识、选支持模型 | 抽样次数随内点率与模型复杂度变化；重复结构可形成错误共识 | 杂边点集的直线／几何估计；[M12](algorithm_cards/measurement_focus.md#m12-pca-二维正交直线拟合)仅为非鲁棒对照，无 RANSAC 独立卡 | [cite:1](#me-source-1)；**未实现** |
| 1986，Canny／候选检测 | 检出、定位和多重响应之间缺少统一目标 → 条件模型下优化滤波、梯度 NMS 与滞后判边 | 平滑尺度折中；噪声／阶跃假设不覆盖全部场景；二值边不是亚像素坐标 | 一般边缘候选；[M06](algorithm_cards/measurement_focus.md#m06-canny)、[M07](algorithm_cards/measurement_focus.md#m07-sobel) | [cite:2](#me-source-2)；**完整流程未实现**，有 Sobel 调用 |
| 1988，Harris–Stephens／角点 | 直边切向歧义、离散方向判断不足 → 局部自相关二次型区分平坦、直边与角点 | 尺度、阈值和窗口依赖；不自动完成亚像素精化 | 棋盘／交点初检；[M08](algorithm_cards/measurement_focus.md#m08-harris-角点检测)至[M10](algorithm_cards/measurement_focus.md#m10-opencv-cornersubpix-精化) | [cite:3](#me-source-3)；**库调用示例＋两种精化**，示例用真值挑初值 |
| 1993，Ghosal–Mehrotra／正交矩 | 局部边缘参数需要紧凑估计 → Zernike 矩映射理想二维阶跃的方向、位置、背景和对比度 | 离散圆域积分、矩归一化、弱矩及真实模糊模型仍有问题 | 可建模的孤立边缘；[M05](algorithm_cards/measurement_focus.md#m05-zernike-矩边缘) | [cite:4](#me-source-4)；**未实现**；不能认领 A31 模糊联合估计 |
| 1995，Devernay RR-2724／插值 | NMS 仅保留整数点 → 沿梯度方向对幅值作局部插值并定位峰 | 梯度离散化、离网插值、峰形近似；非任意模糊无偏 | 二维孤立边缘连续定位；[M04](algorithm_cards/measurement_focus.md#m04-devernay-二维边缘精化)、[M02](algorithm_cards/measurement_focus.md#m02-三点抛物线梯度峰) | [cite:5](#me-source-5)；**简化版已实现**，非报告完整复现 |
| 2008，JCGM 100／计量规范 | 只报残差或重复性无法说明测量可信范围 → 测量模型、标准／合成／扩展不确定度 | 需识别输入分量、相关性和覆盖条件；不是新边缘算法 | 可追溯尺寸报告；[标定总览](calibration_coordinates.md)，无独立 GUM 算法卡 | [cite:6](#me-source-6)；**物理测量链未实现**；当前 PDF 含 2010 勘误 |
| 2011，Hagara–Kulla／erf 拟合 | 理想阶跃不反映模糊过渡 → 用四参数 erf 模型直接拟合一维灰度 | 非线性求解、平台支持、噪声与模型错配；并非所有 erf 思想的起源 | 卡尺剖面；[M03](algorithm_cards/measurement_focus.md#m03-erf-积分高斯剖面拟合)、[M01](algorithm_cards/measurement_focus.md#m01-卡尺-caliper) | [cite:7](#me-source-7)；**同类模型已实现，非逐项复现**；论文用 `fminsearch`，本地用 `curve_fit` |
| 2013，Structured Forests／传统 ML | 逐像素局部规则忽略边界结构 → 随机森林学习结构化边缘 mask | 需特征、边缘标签和训练；边界检测分数不等于尺寸误差 | 杂背景中的候选边界；[学习组件 C03](algorithm_cards/learning_components.md#c03-随机森林-rf)为通用组件，非该模型专卡 | [cite:8](#me-source-8)；**未实现** |
| 2015，HED／DL 边缘；U-Net／相关架构 | 局部描述上下文不足 → HED 多尺度特征与侧输出监督；U-Net 编解码定位提供另一架构参照 | 标签不确定度、下采样、显存与域迁移；两文任务及网络不同 | 学习边缘候选或连续偏移研究；[M11](algorithm_cards/measurement_focus.md#m11-edgenet-边缘与偏移网络) | [cite:9](#me-source-9)；**HED 未实现**；本地 EdgeNet 是自定义骨架，不是 HED／原版 U-Net |
| 2017，IPOL Canny/Devernay／可复现实现 | 原算法实现细节与振荡伪差难比较 → 分析离散误差、修改实现并输出边链 | 连通、阈值与采样规则影响拓扑／位置；代码许可需单独检查 | 需要有序轮廓和实现对照；[M04](algorithm_cards/measurement_focus.md#m04-devernay-二维边缘精化) | [cite:10](#me-source-10)；**外部完整实现未接入**，本地只输出无序点 |
| 2020，Renshaw–Christian／矩模型继续演进 | 理想阶跃不覆盖扩展过渡和线状条纹 → 斜坡边缘、楔形条纹的统一矩定位框架 | 模型、宽度和局部孤立性仍重要；不是对任意 PSF 的通用补偿 | 已有候选后的边缘／条纹精化；[M05](algorithm_cards/measurement_focus.md#m05-zernike-矩边缘)为相关基础，无此文专卡 | [cite:11](#me-source-11)；**未实现**；可读正文明确最终使用 A11、A20，不支持“必用 A31” |

RANSAC 改变的是点集估计，不替代检测；Harris 与边缘法解决的可观测方向不同；矩法、erf 与 Devernay 是不同成像假设和求解方式，不是统一精度阶梯。2017、2020 的传统方法文献与 2013、2015 的学习方法并存，本身就不支持“有 DL 后传统方法失效”的叙事。[cite:1](#me-source-1)[cite:3](#me-source-3)[cite:8](#me-source-8)[cite:9](#me-source-9)[cite:10](#me-source-10)[cite:11](#me-source-11)

## 4. Devernay、Zernike 与 A31 的核验结论

### 4.1 Devernay 的报告、入库与后续实现

Frédéric Devernay 的正确条目是 *A Non-Maxima Suppression Method for Edge Detection with Sub-Pixel Accuracy*，**INRIA Research Report RR-2724，1995 年 11 月**。HAL 封面的 **2006-05-24** 是入库日期，不是提出年份。报告正文第 2 节说明在 NMS 后插值梯度模长峰；本地命名虽沿用 Devernay，但实际是高斯平滑＋Sobel、单 `high` 阈值、近似方向 NMS、沿单位梯度正负一像素插值，无双阈值滞后、无边链。[cite:5](#me-source-5)

Grompone von Gioi 与 Randall 的 IPOL 文为 **2017**，作者不是 Devernay；正式发表日期为 2017-11-28，页面软件条目另列 **1.0／2017-10-10**，2022 的页面说明是编辑信息修订。IPOL 分析峰形近似、有限差分、离网插值与数值误差，不能将“对模糊不敏感”“固定亚像素精度”作为无条件结论。本地源码头的速度／误差参考不升级为该论文或现场保证。[cite:10](#me-source-10)

### 4.2 Zernike 书目纠偏及不能确认的归属

可核验书目为 **Sugata Ghosal、Rajiv Mehrotra，*Orthogonal moment operators for subpixel edge detection*，Pattern Recognition 26(2)，295–306，1993 年 2 月，DOI `10.1016/0031-3203(93)90038-X`**。Crossref、NASA/ADS及后续引文相符；旧笔记的 *Detection of Edges via Zernike Moments, IEEE Trans. IP, 1993* 已在[撤回记录](research_notes.md#zernike-retraction-20261106)纠偏，保留旧说法用于追踪，不再作为有效书目。[cite:4](#me-source-4)

2026-11-06同步[ZM01原页证据](algorithm_cards/kb_zernike_extensions.md#source-10)：已读VI1992 pp.247–252，明确A00/A11/A20、主动省略归一化、积分制核及像素判边；相位／方向表述有冲突。该早期论文不是已确认的1993作者稿，不回填1993原式；G02仍部分／P0，ZM01全部分、无实现。笔记的A31确定归属、无据性能排名与拟合无偏承诺已撤回，源码头旧待办未改且不作证据；原核验非本轮重跑。

NASA/ADS 摘要明确模型是**理想二维阶跃**，四参数为背景、阶跃高度、边缘到窗口中心距离与方向；摘要没有把模糊宽度列为该四参数之一。出版正文入口被验证页阻挡，**未取得 1993 全文**，因此不能确认旧笔记“论文用 A31 联合估计位置与模糊宽度”的说法，也不能反向断言原文任何位置绝无 A31。[cite:4](#me-source-4)

2020 原文第 3.4 节明确其边缘／条纹方法最终只用 **A11、A20**，由三个实相关响应计算；检索到的第 4 节片段还说明边缘宽度在许多情况下可由 PSF 已知。它至少说明“模糊边缘的 Zernike 定位必用 A31”不是可通用的结论；该文也不是“1993 已有 A31 联合估计”的证明。[cite:11](#me-source-11)

| 争议项 | 本章采用的结论 | 后续补证条件 |
|---|---|---|
| 正确题名、期刊、年卷页 | 采用 1993 Pattern Recognition 条目；书目已交叉核验 | 取得出版全文后继续逐式核对，不改写历史为 IEEE TIP |
| A31 是否属于原文模糊补偿 | **[INSUFFICIENT DATA]**，不确认归属 | 给出原文页码／公式、基函数定义、PSF 类型、宽度与位置解的条件 |
| A31 作为新增研究变量 | 可以提出为参考建模方案，不可称已核验闭式算法 | 检验联合参数雅可比秩、弱对比退化、像素积分与残差 |
| 仓库是否有 Zernike 算法 | **无可调用函数**；只有笔记、源码头与教学卡 | 实现、回归、真实图评测须分别完成 |
| 理想矩比能否照搬 | 必须先统一归一化与共轭约定 | 常量泄漏、旋转相位、反极性、半径尺度和积分精度测试 |

### 4.3 归一化问题比“多加一个矩”更先发生

[M05](algorithm_cards/measurement_focus.md#m05-zernike-矩边缘)使用带 `(n+1)/π` 的标准归一化和共轭基函数；在其明确的正对比阶跃模型下，位置是 `l = 2 A20 / (3 A11')`，不是不加说明的 `A20 / abs(A11)`。2020 文则把 **未归一化积分称 A、归一化矩称 Z**，并使用自己的复数相位约定；同名 A11 不代表数值可直接互换。[cite:11](#me-source-11)

这部分是对卡片定义与可读原文定义的对照，不冒充 1993 原文逐式复现。即使归一化正确，离散点采样、像素面积积分和连续圆域积分仍不同；纯理想阶跃在整数点采样下可能随小位移保持完全相同，不能从不变观测恢复唯一连续位置。A31 的新增研究不能跳过这些可辨识性条件。

## 5. 传统、ML、DL 并行关系及源码状态

传统路线用显式成像／几何模型，适合结构稳定、误差需要解释的 ROI；ML 用手工特征和浅层学习器区分候选，适合有代表性标签但不需要端到端网络的任务；DL 用数据学习上下文与边界表示，仍需要连续坐标、标定和验收。可评估“学习候选 → 原灰度剖面精化 → 鲁棒几何 → 标定”的混合链，但当前仓库未接成此系统。HOG＋SVM 仅用于模板候选重排，不是测量 ML 实现。

| 卡片／源码入口 | 实际实现状态 | 源码核对后的重要边界 |
|---|---|---|
| [M01 卡尺](algorithm_cards/measurement_focus.md#m01-卡尺-caliper)／[caliper.py](../traditional/caliper_measurement/caliper.py) | 简化线探针、erf 优先与异常回退已实现 | 每探针单线采样，无切向厚度平均；`polarity=0` 是全体梯度均值统一翻转；`points` 为图像像素坐标 |
| [M02](algorithm_cards/measurement_focus.md#m02-三点抛物线梯度峰)、[M03](algorithm_cards/measurement_focus.md#m03-erf-积分高斯剖面拟合)／[edge_subpixel.py](../traditional/subpixel/edge_subpixel.py) | 梯度三点插值、四参数 erf 无约束拟合已实现 | 偏移裁半像素不是精度界；erf 成功返回未必中心有效，sigma 事后取绝对值不替代优化约束 |
| [M04](algorithm_cards/measurement_focus.md#m04-devernay-二维边缘精化)、[M06](algorithm_cards/measurement_focus.md#m06-canny)、[M07](algorithm_cards/measurement_focus.md#m07-sobel) | 简化 Devernay＋Sobel；完整 Canny 未实现 | 无滞后、无边链；NMS 邻居向量未统一长度；`np.where` 两分支先求值可能产生除零警告 |
| [M05 Zernike](algorithm_cards/measurement_focus.md#m05-zernike-矩边缘) | 理论参考，**未实现** | 不存在可用于报告精度的 Zernike 回归；A31 归属与联合估计仍缺证据 |
| [M08](algorithm_cards/measurement_focus.md#m08-harris-角点检测)至[M10](algorithm_cards/measurement_focus.md#m10-opencv-cornersubpix-精化)／[corner_subpixel.py](../traditional/subpixel/corner_subpixel.py) | Harris 示例；手写固定窗单次解；OpenCV 移窗迭代封装 | 示例及测试按真值挑候选，不能证明初检召回；OpenCV 封装强转 uint8，归一化浮点／16 位输入会改变信息 |
| [M11 EdgeNet](algorithm_cards/measurement_focus.md#m11-edgenet-边缘与偏移网络)／[subpixel_edge_net.py](../dl/subpixel_edge_net.py) | 网络＋损失骨架，无训练／验证权重 | 实际三通道，不是注释两通道；相加 skip 非原版 U-Net 拼接；BHW 掩膜直接索引 B2HW 偏移有布局问题，奇数尺寸亦需检查 |
| [M12 PCA](algorithm_cards/measurement_focus.md#m12-pca-二维正交直线拟合)／[caliper.py](../traditional/caliper_measurement/caliper.py) | SVD 正交直线拟合已实现 | 无强度加权、无外点剔除；两点或整体平移都可能低 RMS；不是圆／椭圆拟合 |
| [M13 Laplacian](algorithm_cards/measurement_focus.md#m13-laplacian-二阶算子)／[focus_measures.py](../traditional/autofocus/focus_measures.py) | 在对焦评分内调用二阶算子 | 无零交叉测量管线；能量评分不等于边缘提取 |
| [ML 状态](../ml/README.md)、[DL 状态](../dl/README.md)、[标定](calibration_coordinates.md) | 测量专用 ML、完整学习测量链和毫米输出均未实现 | 通用 RF 卡、网络层和路线文字不构成该域交付能力 |

## 6. 工程实践线：选型、排障与验收

### 6.1 按任务选择起点

| 任务／约束 | 优先评估路线 | 必要前提与代价 | 失败后的检查／回退 |
|---|---|---|---|
| 已定位、单边缘的窄 ROI | 卡尺＋抛物线，与 erf 作对照 | 法向、极性和窗口平台正确；erf 增加迭代成本 | 保存剖面和残差；先修 ROI／成像，不盲目扩大拟合迭代 |
| 二维未知走向轮廓 | 简化 Devernay 作基线；需有序曲线时另评估 IPOL 完整实现 | 有梯度信息、孤立局部峰；拓扑与连续坐标分开验收 | 查角度／相位误差与断边；不把无序点直接当闭合轮廓 |
| 棋盘或局部交点 | Harris 初值＋角点精化 | 两个有效梯度方向、独立于真值的初检 | 查特征值和初值距离；不可观测时拒绝，不靠极小 epsilon |
| 毛刺、遮挡或多条干扰边 | 先改善候选，再评估 RANSAC／稳健损失 | 需模型、内点阈值和最低覆盖；当前未实现 | 保留全部点、残差和拒绝比例，不只看最终 RMS |
| 模糊边缘的模型比较 | erf、Devernay、正确离散矩模型并列研究 | 统一 PSF、采样、位置和误差定义；Zernike 尚需实现 | A31 未补证时不编闭式补偿，不给虚构速度／精度 |
| 杂背景、语义边界难区分 | ML／DL 候选＋显式几何精化 | 标签精度、工件级划分、推理资源和域覆盖 | 回到原灰度核验边界，低置信／域外拒识；当前仅骨架 |
| 输出公差相关的毫米尺寸 | 先定义测量量、标定和不确定度预算 | 对焦位置可能影响倍率／边界；需可追溯参考 | 像素算法通过也不放行物理尺寸；补完整测量链 |

这些建议来自机制与源码约束，不是统一基准的优胜结论。对焦影响测量剖面和可能的倍率，相关诊断见 [对焦总览](autofocus_evolution.md)；对焦最清晰不自动等于尺寸偏差最小，须共同验证。

### 6.2 症状到证据的排障顺序

| 症状 | 首先采集／检查 | 处置顺序及不能混淆的结论 |
|---|---|---|
| 卡尺靠边崩溃或结果异常 | ROI 四邻采样坐标、图像边界、有效点 | `_sample_points` 先索引后屏蔽；调用前保证完整支持，修复另立任务，不能把 NaN 填补当越界保护 |
| 反极性／混合边误选 | 每条原剖面、带符号梯度与探针方向 | 先定法向与极性，再强度门；当前自动极性不是逐探针任意选边 |
| erf 低 RMS 但中心飘 | 左右平台、中心／宽度／幅值、残差结构 | 排除双边和饱和，检查参数耦合；返回无异常不等于物理可信 |
| Devernay 随旋转周期漂移 | 角度、亚像素相位、NMS 保留点、离网样点 | 分离导数核和插值误差；先比较同采样规则，不用降阈值掩盖系统偏差 |
| 角点有固定半像素偏差 | 像素中心／格点定义、ROI 回填、插值及量化链 | 先统一真值；棋盘格交界可在 `cell*n-0.5`，不直接给坐标统一补偿 |
| PCA 残差很小、尺寸仍错 | 点集整体位置、覆盖、外点、标定参考量 | 残差是内部拟合一致性；整体平移偏差不由 RMS 暴露 |
| Zernike 数值随窗径跳变 | 单位圆映射、面积权、归一化、相位和反极性 | 先验常量与解析矩，再测模糊；没有函数时先建受控实现，不调不存在的 A31 参数 |
| EdgeNet loss 索引失败／形状错 | B3HW、BHW、B2HW 布局及奇数 H/W | 先修数据契约和掩膜布局，再谈训练；本次仅记录不改源码 |
| 学习边缘看起来准、尺寸偏 | 原图到网络缩放、偏移单位、标注粗细、域变化 | 边缘 F-score 与法向／尺寸误差分报；分割漂亮不代表计量正确 |

### 6.3 最小验收协议与已知覆盖

- **观测与真值**：保存未增强原图、ROI、曝光／增益、镜头与采样规则；边缘法向定位、二维角点和尺寸真值分别定义。测量参考量须记录可追溯性及不确定度。
- **数据矩阵**：覆盖角度、亚像素相位、模糊、噪声、正反极性、饱和、近邻边、曲率与遮挡；标清合成／实采。合成生成器与拟合模型同源的好结果不当成独立证据。
- **误差与拒识**：同时报告位置偏差、重复采集离散度、有效率／失败率、点集覆盖、拟合残差；像素到物理量还需标定参数及相关性传播，不给未证通用上界。
- **学习路线**：按工件／场景／批次划分，冻结阈值和权重，分别验收候选漏检、连续定位、几何量和域外拒识；未训练骨架不参与性能排名。
- **成本**：统一图像、ROI、候选数、拟合次数和软硬件，分项记录采集、预处理、定位、拟合与传输；算法调用时间不替代设备节拍。

[test_smoke.py](../tests/test_smoke.py)现有测量测试为一类卡尺斜边、无噪 erf 剖面、单斜边 Devernay 和真值辅助初值角点；没有 Zernike、鲁棒拟合、完整 Canny、学习模型训练或毫米测量验收。历史通过记录见[实验记录](experiments.md)与[卡片验证边界](algorithm_cards/measurement_focus.md#本次验证未覆盖项与来源状态)，本章不将其改写成本次实测。此次只检查文档结构、链接、引用闭合、源码对应与两文件写入范围。

## 7. 来源与逐源访问记录

**访问层级**：“全文”要求正文实际可读；“全文片段”指原始 PDF／HTML 返回了部分正文；“摘要”只支持摘要可见主张；“书目”只支持作者、题名、年份版本等；“受限”不算读到正文。**证据标签**：[CONFIRMED] 仅用于已交叉确认的书目事实；[SINGLE-SOURCE] 表示方法说明仍主要依赖一篇原文；[INSUFFICIENT DATA] 表示缺证据。同文的 DOI、登记记录和镜像不是三次独立实验。

### 7.1 原始方法与规范

<a id="me-source-1"></a>

**[cite:1] Fischler、Bolles，RANSAC，1981。** *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*，Communications of the ACM 24(6)，381–395，DOI `10.1145/358669.358692`。[ACM 出版页](https://cacm.acm.org/research/random-sample-consensus/)：**摘要＋书目已读**，支持粗差下模型估计及正式年卷页；未取得原文全文，不搬用抽样成功率为仓库性能。[SINGLE-SOURCE]

<a id="me-source-2"></a>

**[cite:2] John Canny，1986 期刊版。** *A Computational Approach to Edge Detection*，IEEE TPAMI PAMI-8(6)，679–698，DOI `10.1109/TPAMI.1986.4767851`。[IEEE 条目](https://xplorestaging.ieee.org/document/4767851/authors)：**检索书目**；[Utah 大学原文 PDF 镜像](https://www.sci.utah.edu/~gerig/CS6640-F2012/Materials/canny86.pdf)：**全文片段已读**，含首页、三目标、白高斯噪声假设与定位分析，返回在后文截断；[ACM 当期目录](https://dl.acm.org/toc/itpm/1986/8/6)：**检索书目**交叉年卷页。期刊为 1986 年 11 月，收稿 1984／修订 1985 不是出版年份；未将更早技术报告等同此版。[CONFIRMED] 书目；方法 [SINGLE-SOURCE]。

<a id="me-source-3"></a>

**[cite:3] Chris Harris、Mike Stephens，1988。** *A Combined Corner and Edge Detector*，Alvey Vision Conference，147–151，DOI `10.5244/C.2.23`。[BMVA 原文 PDF](https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf)：**全文已读**，包括局部自相关、特征值、响应及连接讨论；[Cornell 课程引文](https://www.cs.cornell.edu/courses/cs4670/2016sp/lectures/lec10_features2_web.pdf)：**检索书目**交叉年份会名页码。它不是 `cornerSubPix` 当前实现源码，也不证明本地初检已部署。[CONFIRMED] 书目。

<a id="me-source-4"></a>

**[cite:4] Sugata Ghosal、Rajiv Mehrotra，1993。** 正确书目见第 4.2 节。[Crossref 出版登记](https://api.crossref.org/works/10.1016/0031-3203(93)90038-X)：**书目 JSON 已读**，`published-print` 为 1993 年 2 月，26(2)，295–306；[NASA/ADS](https://ui.adsabs.harvard.edu/abs/1993PatRe..26..295G)：**摘要＋书目已读**，四参数理想阶跃；[Cambridge 后续独立论文](https://www.cambridge.org/core/journals/journal-of-navigation/article/highaccuracy-extraction-algorithm-of-planet-centroid-image-in-deepspace-autonomous-optical-navigation/7DB66F1C19E099AEC7AD2E1234906B96)：**检索书目**，同题名／年卷页；[出版 DOI 入口](https://doi.org/10.1016/0031-3203(93)90038-X)：**受限，返回验证码页**。书目 [CONFIRMED]；原始公式、A31 模糊补偿归属 [INSUFFICIENT DATA]，不引用摘要抗噪措辞为普遍结论。

<a id="me-source-5"></a>

**[cite:5] Frédéric Devernay，RR-2724，1995。** [HAL 原报告 PDF](https://inria.hal.science/inria-00073970/file/RR-2724.pdf)：**全文片段已读**，封面、摘要、目录及第 1–2.2 节，确认 1995 年 11 月报告与 2006 入库不同；[作者出版目录](http://devernay.free.fr/publis/)：**检索书目**确认 RR-2724／1995；独立作者 IPOL 文章见来源 10。报告日期 [CONFIRMED]；没有逐页复核全部实验或审计外部源码。

<a id="me-source-6"></a>

**[cite:6] JCGM 100:2008。** *Evaluation of measurement data — Guide to the expression of uncertainty in measurement*。[BIPM 官方 PDF](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)：**全文片段／版本与目录已读**，封面注明 GUM 1995 with minor corrections、2008 第一版及 2010 corrected version，目录列测量模型、相关输入、合成／扩展不确定度。此处引用规范角色与版本，不声称逐条审核全文或完成计量认证。[SINGLE-SOURCE]

<a id="me-source-7"></a>

**[cite:7] Miroslav Hagara、Peter Kulla，2011。** *Edge Detection with Sub-pixel Accuracy Based on Approximation of Edge with Erf Function*，Radioengineering 20(2)，516–524。[期刊原文 PDF](https://www.radioeng.cz/fulltexts/2011/11_02_516_524.pdf)：**全文片段已读**，到第 6 节部分，核对 2011 年 6 月、阶跃／斜坡／erf 模型、四参数拟合、`fminsearch` 与像素积分；后文截断。只作为该拟合方案代表，不认定 erf 模型首创，不外推实验精度。[SINGLE-SOURCE]

<a id="me-source-8"></a>

**[cite:8] Piotr Dollár、C. Lawrence Zitnick，ICCV 2013。** *Structured Forests for Fast Edge Detection*，1841–1848。[CVF 原论文页面](https://openaccess.thecvf.com/content_iccv_2013/html/Dollar_Structured_Forests_for_2013_ICCV_paper.html)：**摘要＋书目已读**，结构化标签、随机决策森林；有 PDF 入口但本次未读全文。自然图边界评价不等于工业亚像素尺寸评价。[SINGLE-SOURCE]

<a id="me-source-9"></a>

**[cite:9] 2015 学习分支，两篇不同论文。** Saining Xie、Zhuowen Tu，*Holistically-Nested Edge Detection*，ICCV 2015，1395–1403：[CVF 页面](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)，**摘要＋书目已读**，多尺度、全卷积和深监督。Olaf Ronneberger、Philipp Fischer、Thomas Brox，*U-Net: Convolutional Networks for Biomedical Image Segmentation*：[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)，**摘要＋提交历史已读**，v1 为 2015-05-18，页面写 MICCAI 2015 conditional acceptance；未据此拼造正式会议页码。两者均未读全文，不能把本地 EdgeNet 当任一原论文的复现。[SINGLE-SOURCE] 各自的方法说明。

<a id="me-source-10"></a>

**[cite:10] Rafael Grompone von Gioi、Gregory Randall，2017。** *A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm*，IPOL 7，347–372，DOI `10.5201/ipol.2017.216`。[IPOL 页面](https://www.ipol.im/pub/art/2017/216/)：**摘要＋完整页面书目／版本记录已读**；[低分辨率原文 PDF](https://www.ipol.im/pub/art/2017/216/article_lr.pdf)：**全文片段已读**，第 1–4 节可见，后文截断。页面确认边链及轻微修改，PDF 解释四类误差；软件版本、论文发表、2022 编辑信息修订分别记录。未下载／审计代码，开放入口不意味着已接入或商用许可无约束。

<a id="me-source-11"></a>

**[cite:11] Devin T. Renshaw、John A. Christian，2020。** *Subpixel Localization of Isolated Edges and Streaks in Digital Images*，Journal of Imaging 6(5)，33，DOI `10.3390/jimaging6050033`。[MDPI 正文](https://www.mdpi.com/2313-433X/6/5/33)：**全文片段已读**，到第 4 节开头，含第 3.2–3.4 节归一化定义和“最终只用 A11、A20”；[PubMed](https://pubmed.ncbi.nlm.nih.gov/34460735/)：**摘要＋书目／图注已读**，确认 2020-05-18；[学术索引 PDF 镜像](https://pdfs.semanticscholar.org/8ca0/17b29e3da3968176af409dfe46d715cf0852.pdf)：**仅检索正文片段**，可见宽度可由 PSF 已知的限定，未读其全文。[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321028/)返回 cookies 提示；[MDPI XML](https://www.mdpi.com/2313-433X/6/5/33/xml)返回机器人检查；[MDPI HTML 独立请求](https://www.mdpi.com/2313-433X/6/5/33/html)返回 403，均记**受限**，未绕过验证。不能把后续斜坡模型反归于 1993 的 A31，亦未确认所有 PSF 的宽度联合估计。

### 7.2 本地证据与剩余缺口

本地证据以第 5 节函数体、[测量卡 M01–M13](algorithm_cards/measurement_focus.md#m01-卡尺-caliper)、[学习规范](algorithm_learning_standard.md)、[测试](../tests/test_smoke.py)和[历史记录](experiments.md)为准。源码注释与函数体冲突时按函数体描述，但不自动修复。工程建议和明确假设下的数学解释不是外部论文实测。

剩余缺口为 1993 全文及 A31 原始归属、各方法统一真实数据比较、完整学习训练、鲁棒几何、标定与不确定度实施、现场尺寸验收。上述缺口不妨碍使用已有传统函数作研究基线，但必须阻止“Zernike 已实现”“IPOL 完整复现”“有亚像素坐标即达到同等物理精度”等误读。
