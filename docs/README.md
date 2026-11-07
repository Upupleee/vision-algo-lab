# 工业视觉文档导航

[返回仓库首页](../README.md) · [快速开始](../README.md#快速开始) · [算法卡片逐卡索引](algorithm_cards/README.md) · [演进覆盖与去重缺口](evolution_coverage.md) · [六项原文获取与验收](source_requests.md) · [横向基础与评测目录](#横向基础与评测目录) · [阅读路径](#阅读路径)

本页按任务领域组织现有资料，不改变 `traditional/`、`ml/`、`dl/` 的代码分层。传统路线不引入学习框架；ML 指手工特征与浅层学习器，DL 指神经网络路线。

**状态口径**：有实现不等于已完成现场验证；“骨架”不等于可用系统；“实现待建设”表示当前没有该领域的专用实现，理论文档已建立也不改变这一状态。各域所列待建设方法只是后续研究方向，相关基础链接不代表该域已经实现。已验证结果引用既有记录（主要为 2026-10-30），不是本次重新运行的结果；依赖可用性以实际环境为准。

## 双线阅读与编写规范

所有领域统一采用 [领域演进核心主线＋算法 L1–L5 学习规范](algorithm_learning_standard.md)。该规范兼顾面试追问与现场排障：领域层解释演进维度，算法层依次展开步骤、机制、工程权衡、数学推导与实现，并分别记录覆盖状态；不表示现有章节已经全部达到 L5。

知识库保留两条互相链接的主线：**方法演进线**回答“当时解决什么问题、改进了什么、付出什么代价”；**工程实践线**回答“当前该选什么、失败查哪里、如何验收”。按年份排列不表示新方法全面替代旧方法，任务不同的分支应并列说明。

| 已建立双线章节 | 方法演进与原始来源 | 工程实践 |
|---|---|---|
| 定位匹配配准（样章） | [代表性时间线与分支](matching_evolution.md) | [工程选型](matching_evolution.md)、[ZNCC 实现与现场排障](pose_ncc.md) |
| 预处理增强 | [方法演进与版本说明](preprocessing.md) | 同文的选型、失败模式、排障与验收 |
| 标定坐标 | [方法演进与版本说明](calibration_coordinates.md) | 同文的坐标链、选型、排障与验收 |
| 分割区域 | [方法演进与版本说明](segmentation_regions.md) | 同文的区域定义、选型、排障与验收 |
| 边缘几何测量 | [边缘、亚像素与几何估计里程碑](measurement_evolution.md#3-方法演进线代表性里程碑) | [选型、排障与验收](measurement_evolution.md#6-工程实践线选型排障与验收)、[Devernay／Zernike／A31 核验边界](measurement_evolution.md#4-devernayzernike-与-a31-的核验结论) |
| 缺陷异常 | [纹理、正常建模与异常定位节点](defect_evolution.md#3-方法演进线有据代表节点) | [选型、评分与验收](defect_evolution.md#4-工程实践线选型评分与验收) |
| 分类装配 | [分类、检测、关系与过程节点](classification_assembly_evolution.md#3-方法演进线有据代表节点) | [对象证据到装配决策](classification_assembly_evolution.md#4-工程实践线从对象证据到装配决策) |
| 字符编码 | [OCR 识别与检测](ocr_codes_evolution.md#3-历史演进线-aocr-识别与检测)、[编码、质量标准与业务语法](ocr_codes_evolution.md#4-历史演进线-b编码质量标准与业务语法) | [选型、交接、排障与验收](ocr_codes_evolution.md#5-工程实践线选型与交接) |
| 颜色外观 | [表示、公式与模型规范](color_appearance_evolution.md#3-历史演进线-a表示公式指标与模型规范)、[校正、恒常性与空间外观算法](color_appearance_evolution.md#4-历史演进线-b校正恒常性与空间外观算法) | [任务定义、排障与验收](color_appearance_evolution.md#5-工程实践线从任务定义到验收) |
| 三维重建测量 | [深度获取、对应、配准与融合节点](3d_evolution.md#3-方法演进线九个有据节点) | [选型、误差与排障](3d_evolution.md#4-工程实践线选型误差与排障) |
| 对焦成像质量 | [评分、峰模型与学习对焦里程碑](autofocus_evolution.md#3-方法演进线代表性里程碑) | [选型、排障与验收](autofocus_evolution.md#6-工程实践线选型排障与验收)、[七种评分的实际定义](autofocus_evolution.md#51-七种评分的实际定义) |
| 跟踪运动视觉引导 | [估计、关联、跟踪与控制节点](tracking_guidance_evolution.md#3-方法演进线九个有据节点) | [选型与排障](tracking_guidance_evolution.md#4-工程实践线选型与排障)、[坐标同步、延迟与闭环安全](tracking_guidance_evolution.md#5-坐标同步延迟与闭环安全) |

12 个领域均已有双线总览入口；总览建立不表示算法实现、L5 或现场验收完成。

[主时间线逐节点覆盖矩阵](evolution_coverage.md)汇总 **113个节点：已有83、部分2、缺卡4、非算法24**，方法分母89。剩余 **6项覆盖任务＝4项P0缺卡＋2部分**：G09／G10／G11／G36缺卡、G02部分／P0、G16部分／P1；G16仅07-01计剩余，06-01已有。G01–G48历史ID保留，42项已有；**G44机制已有，但证明一致性与采样安全[独立P0保留、不降级](evolution_coverage.md#g44-independent-p0)**，不计为第7项覆盖任务。原守卫／初值／变增益辨识不等于证明或实测通过。卡数仍 **255＝245＋10＝206＋9＋4＋4＋8＋4＋4＋4＋2＋10**；ZM01、HC73全部分，HA01及RE01状态不升级，LG01／FP01各仍一卡，HED／U-Net复合节点不拆分。原实现、P2及非算法资料边界保留，详见[最小缺页清单](evolution_coverage.md#minimum-source-pages)。

**2026-11-06原文请求清单轮次：**依据focus、ZM01、FTP、HC73四份卡片最新续查整理[六项中文获取／验收清单](source_requests.md)，列明已有书目证据、印刷页与PDF页序、最小方法内容、合法渠道及作者稿版本门槛。六项全部未关闭；255卡、113节点及83／2／4／24不变，G44两项独立P0保留。仅新增清单并更新本页与覆盖账，不重复检索日志，不重开已闭合子项。

各领域后续维护沿用以下规范：

1. 每个里程碑记录年份／版本、任务分支、代表方法、问题、核心改进、代价与局限、适用场景、原始来源和仓库状态。表过宽时拆成时间线与方法卡片。
2. 区分预印本、会议、期刊扩展及软件实现日期；不从 DOI 数字推断年份，不将项目更新日期作为算法提出日期。
3. 工业用途属于适用性分析，不等于论文已验证该产线效果。没有证据时不写通用精度、速度排名或“最新最佳”。
4. 算法、评价指标、搜索策略、硬件优化分开标注。FFT、金字塔和并行化并不自动代表新任务算法。
5. 已有实现、示例、骨架、未实现、理论已建及现场已验证分别记录；不可访问的原始来源明确标注获取限制。
6. 时间线是代表性里程碑，不是完整算法史。12份总览与45份算法卡正文互补，255个编号不增，不自动计作L5完成。前次四文件10卡纳入保留为历史；本轮更新原状态摘要，不提升层级。HA01及RE01的L1仍部分，LG01与FP01各只计一卡；源码风险、P2深度缺口及G44独立P0不清除。

## 算法卡片入口

2026-11-06原文请求清单轮次：45文件仍 **255张卡**，不新增编号。详见[逐卡类型、五层覆盖与验证边界](algorithm_cards/README.md)；唯一键为“文件名＋卡号”。前次四文件10卡纳入、G44机制／风险同步、[研究笔记撤回记录](research_notes.md#zernike-retraction-20261106)及focus／HC73来源归因均保留为历史。本次只整理原文请求并检查本地链接和计数，不执行卡内算法检查器；原公式目视、有限算术、冒烟、训练／实物结果均不冒称本轮重跑，五层和实现状态不升级。

| 当前文件 | 复核数量 | 范围 |
|---|---|---|
| [预处理与指标](algorithm_cards/README.md#预处理与指标) | 46 | P01–P44＋Q01–Q02，含PSNR／SSIM |
| [分割区域](algorithm_cards/README.md#分割区域) | 31 | S01–S31，含区域算子与族级基线 |
| [定位匹配配准](algorithm_cards/README.md#定位匹配配准) | 51 | 01–51，含实现、搜索与评测组件 |
| [标定坐标](algorithm_cards/README.md#标定坐标) | 10 | C01–C10，含模型与观测前端 |
| [学习组件](algorithm_cards/README.md#学习组件) | 27 | C01–C27，分类器、算子、优化、损失与指标 |
| [测量与对焦](algorithm_cards/README.md#测量与对焦) | 27 | M01–M13＋F01–F14，七评分仅F01–F07 |
| [其余领域](algorithm_cards/README.md#其余领域) | 14 | 缺陷、装配、OCR／编码、颜色、三维与跟踪 |
| [对焦起源](algorithm_cards/README.md#对焦起源) | 3 | focus_origins:G09／G10／G11；五层与答5已建，原机制仍缺证 |
| [FTP与混合切换伺服](algorithm_cards/README.md#ftp与混合切换伺服) | 2 | G36缺卡／P0；G44机制已有、证明／采样安全独立P0，均未实现 |
| [矩定位与傅里叶对焦](algorithm_cards/README.md#矩定位与傅里叶对焦) | 2 | moment_fourier_focus:MF01／MF02，对应G15／G43 |
| [PaDiM与根多项式校正](algorithm_cards/README.md#padim与根多项式校正) | 2 | padim_root_polynomial:G19／G34；具体方法已有，未实现 |
| [Hungarian指派](algorithm_cards/README.md#hungarian指派) | 1 | hungarian:H01；G13具体方法已有，无产品内核 |
| [结构化森林边缘](algorithm_cards/README.md#结构化森林边缘) | 1 | structured_forests:SF01；G14具体方法已有，未实现 |
| [Haralick共现纹理](algorithm_cards/README.md#haralick共现纹理) | 1 | haralick:HA01；G16仍部分，06-01已有／07-01部分 |
| [One-Class SVM](algorithm_cards/README.md#one-class-svm) | 1 | one_class_svm:OC01；G17具体方法已有，未实现 |
| [SSIM Autoencoder](algorithm_cards/README.md#ssim-autoencoder) | 1 | ssim_autoencoder:SA01；G18具体方法已有，作者AE锁定未闭合，未实现 |
| [DRAEM](algorithm_cards/README.md#draem) | 1 | draem:DR01；G20具体方法已有，双网联合训练未实现 |
| [PatchCore](algorithm_cards/README.md#patchcore) | 1 | patchcore:PC01；G21具体方法已有，论文／固定源码评分分开，未实现 |
| [ResNet](algorithm_cards/README.md#resnet) | 1 | resnet:RN01；G22具体方法已有，原版及现代变体分开，未实现 |
| [Faster R-CNN](algorithm_cards/README.md#faster-r-cnn) | 1 | faster_rcnn:FR01；G23已有，RPN／RoIPool与版本训练已展开，未实现 |
| [OpenMax](algorithm_cards/README.md#openmax) | 1 | openmax:OM01；G24已有，尾部／重分配与源码风险分开，未实现 |
| [迭代消息传递场景图](algorithm_cards/README.md#迭代消息传递场景图) | 1 | iterative_scene_graph:SG01；G25已有，先边后对象／双向过滤差异保留，未实现 |
| [LeNet-5与GTN](algorithm_cards/README.md#lenet-5与gtn) | 1 | lenet_gtn:LG01；G26两子方法各五层、共享答5，复合仍一卡，未实现 |
| [CTC](algorithm_cards/README.md#ctc) | 1 | ctc:CT01；G27已有，L4限定模型，生产实现／OCR接入未实现 |
| [CRNN](algorithm_cards/README.md#crnn) | 1 | crnn:CR01；G28已有，尺寸／BN冲突及padding风险保留，未实现 |
| [EAST](algorithm_cards/README.md#east) | 1 | east:EA01；G29已有，RBOX／QUAD与fork风险分开，未实现 |
| [DB文本检测](algorithm_cards/README.md#db文本检测) | 1 | db_text_detection:DB01；G30已有，distance／广播风险保留，未实现 |
| [TrOCR](algorithm_cards/README.md#trocr) | 1 | trocr:TR01；G31已有，预训练／词元协议与拒识边界保留，未实现 |
| [1971 Retinex](algorithm_cards/README.md#1971-retinex) | 1 | retinex_1971:RE01；G32已有，部／核／部／部／部不升级，阈值／路径配置及二维复现仍缺 |
| [Gray-Edge](algorithm_cards/README.md#gray-edge) | 1 | gray_edge:GE01；G33已有，L4核限条件代数，二阶系数／mask差异保留，未实现 |
| [FC4](algorithm_cards/README.md#fc4) | 1 | fc4:FC01；G35已有，头语义／测试选模／零向量风险保留，未实现 |
| [Curless–Levoy体积融合](algorithm_cards/README.md#curlesslevoy体积融合) | 1 | curless_levoy:CL01；G37已有，L4仍部分，未实现 |
| [SGM](algorithm_cards/README.md#sgm) | 1 | sgm:SM01；G38已有，L4核限代数，SGBM非MI，未实现 |
| [FPFH与SAC-IA](algorithm_cards/README.md#fpfh与sac-ia) | 1 | fpfh_sacia:FP01；G39已有，两子算法仍一卡，2009原文未得保留P2，未实现 |
| [PSMNet](algorithm_cards/README.md#psmnet) | 1 | psmnet:PM01；G40已有，L4核限模型，GT／padding及旧框架风险保留，未实现 |
| [Monodepth2](algorithm_cards/README.md#monodepth2) | 1 | monodepth2:MD01；G41已有，L4限针孔／局部可微域，尺度及采样风险保留，未实现 |
| [FCGF](algorithm_cards/README.md#fcgf) | 1 | fcgf:FG01；G42已有，L4限模型／固定索引局部梯度，稀疏映射及假负风险保留，未实现 |
| [SORT](algorithm_cards/README.md#sort) | 1 | sort:SO01；G45已有，L4限模型代数，门控／生命周期与异常索引风险保留，未实现 |
| [SiamFC](algorithm_cards/README.md#siamfc) | 1 | siamfc:SI01；G46已有，L4限shape／梯度／坐标代数，版本及插值风险保留，未实现 |
| [Deep SORT](algorithm_cards/README.md#deep-sort) | 1 | deep_sort:DS01；G47已有，L4限模型代数，训练／权重血缘、gallery／绕门及视频缺口保留P2，未实现 |
| [BYTE与ByteTrack](algorithm_cards/README.md#byte与bytetrack) | 1 | bytetrack:BT01；G48已有，L4限模型代数，依赖／Removed残留／去重及视频缺口保留P2，未实现 |
| [KB与Zernike扩展](algorithm_cards/README.md#kb与zernike扩展) | 2 | kb_zernike_extensions:KB01／ZM01；G01已有／P2、G02部分／P0，五元不升级，未实现 |
| [边缘演进扩展](algorithm_cards/README.md#边缘演进扩展) | 3 | edge_evolution_extensions:DE95／DE17／HE01；G03／G05／G04已有／P2，原版未实现，U-Net不增卡或节点 |
| [LBP与HOG扩展](algorithm_cards/README.md#lbp与hog扩展) | 3 | lbp_hog_extensions:LB96／HG05／LB02；G06／G07／G08已有／P2，原分类／检测管线未实现 |
| [SFF与Haralick扩展](algorithm_cards/README.md#sff与haralick扩展) | 2 | sff_haralick_extensions:SF94／HC73；G12已有／P2、G16部分／P1，SFF及原训练器未实现 |
| 合计 | **255** | 206＋9＋4＋4＋8＋4＋4＋4＋2＋10＝245＋10；45文件，卡数不等于已复现算法数 |

[五份旧稿及迁移说明](algorithm_cards/README.md#旧稿入口与迁移边界)保留原文与历史记录；新入口不代表旧稿来源或测试已重新核验。

## 12 领域目录

| 序号 | 领域 | 当前覆盖摘要 |
|---|---|---|
| 01 | [预处理增强](#01-预处理增强) | 理论文档已建立；已有模块内局部处理，专用管线待建设 |
| 02 | [标定坐标](#02-标定坐标) | 理论文档已建立；已有匹配坐标约定，标定实现待建设 |
| 03 | [分割区域](#03-分割区域) | 理论文档已建立；专用实现待建设，边缘网络不等于区域分割 |
| 04 | [定位匹配配准](#04-定位匹配配准) | 传统基线、ML 示例、DL 未实现接口 |
| 05 | [边缘几何测量](#05-边缘几何测量) | 双线总览已建立；传统基线、DL 骨架；Zernike 未实现，A31 归属缺证 |
| 06 | [缺陷异常](#06-缺陷异常) | 双线总览已建立；专用实现与检出率评测待建设 |
| 07 | [分类装配](#07-分类装配) | 双线总览已建立；专用系统待建设，候选验证仅为相关基础 |
| 08 | [字符编码](#08-字符编码) | 双线总览已建立；OCR／读码实现与标准质量评测待建设 |
| 09 | [颜色外观](#09-颜色外观) | 双线总览已建立；测色、校准与外观实现待建设 |
| 10 | [三维重建测量](#10-三维重建测量) | 双线总览已建立；深度、点云与物理测量实现待建设 |
| 11 | [对焦成像质量](#11-对焦成像质量) | 双线总览已建立；传统基线与历史真实数据记录、DL 骨架；无实机闭环 |
| 12 | [跟踪运动视觉引导](#12-跟踪运动视觉引导) | 双线总览已建立；跟踪／控制实现待建设，单帧定位仅为相关基础 |

### 01 预处理增强

- **算法卡片**：[46卡逐项索引](algorithm_cards/README.md#预处理与指标) · [L1–L5正文](algorithm_cards/preprocessing.md)。
- **知识库**：[任务、原理、选型、排障与验收](preprocessing.md)。
- **传统**：已有算法中的滤波、梯度及重采样处理；独立的去噪、平场校正、对比度增强管线待建设。
- **ML**：手工噪声／纹理特征驱动的增强参数选择待建设。
- **DL**：学习式去噪、去模糊、超分辨率待建设。
- **应用排障**：用于稳定后续定位与测量；先查曝光饱和、噪声、插值和量化。增强可能移动边缘或损失高频信息，不应只按视觉观感验收。
- **已知状态及现有链接**：[研究笔记](research_notes.md) 记录双三次插值过冲、截断造成的角点偏差；[合成数据](../data/synthetic.py) 可用于受控扰动。暂无独立预处理评测，不把合成扰动当作已实现的增强方案。

### 02 标定坐标

- **算法卡片**：[10卡逐项索引](algorithm_cards/README.md#标定坐标) · [L1–L5正文](algorithm_cards/calibration.md)。
- **知识库**：[成像模型、坐标链、选型、排障与验收](calibration_coordinates.md)。
- **传统**：已有模板到图像的相似变换、像素中心与 ROI 回填说明；相机内外参、畸变校正、手眼标定及像素到毫米换算待建设。
- **ML**：特征驱动的标定误差补偿待建设。
- **DL**：学习式标定、姿态估计待建设。
- **应用排障**：检查 `[y,x]` 与 `(x,y)`、原点与中心、ROI 偏移、角度正方向、单位和变换链；亚像素坐标不能直接当作物理精度。
- **已知状态及现有链接**：[返回协议与坐标](pose_ncc.md#返回协议与坐标)、[坐标与变换](pose_ncc.md#第一章坐标与变换)、[半像素陷阱](research_notes.md)。当前匹配变换不是相机标定，也没有毫米测量链路。

### 03 分割区域

- **算法卡片**：[31卡逐项索引](algorithm_cards/README.md#分割区域) · [L1–L5正文](algorithm_cards/segmentation.md) · [学习组件27卡](algorithm_cards/README.md#学习组件)。
- **知识库**：[区域定义、阈值／形态学／学习路线与验收](segmentation_regions.md)。
- **传统**：阈值、形态学、连通域、区域统计待建设。
- **ML**：像素／区域手工特征分类待建设。
- **DL**：语义分割、实例分割待建设；已有边缘网络仅作相关方向参考。
- **应用排障**：面向目标区域、面积和粘连分离；检查光照漂移、阈值、孔洞、连通性及标注边界，分别记录过分割与欠分割。
- **已知状态及现有链接**：暂无专用分割实现和 IoU/Dice 评测；[DL 路线说明](../dl/README.md)、[边缘网络骨架](../dl/subpixel_edge_net.py) 输出边缘及偏移，不能视为完整区域分割系统。

### 04 定位匹配配准

- **算法卡片**：[51卡逐项索引](algorithm_cards/README.md#定位匹配配准) · [L1–L5正文](algorithm_cards/matching.md)。
- **知识库**：[算法演进与工程选型样章](matching_evolution.md)；[现有 ZNCC 原理与排障](pose_ncc.md)。
- **传统**：已有 [平移 NCC](../traditional/template_matching/ncc_matching.py)、[梯度形状匹配](../traditional/template_matching/shape_based.py) 和 [旋转／尺度 ZNCC](../traditional/template_matching/pose_ncc.py)；通用透视与非刚性配准待建设。
- **ML**：已有 [HOG + SVM 候选重排序示例](../ml/hog_svm_matching.py)；不重新估计位置，也未按分类阈值剔除候选。
- **DL**：[学习式匹配](../dl/learned_matching.py) 仅有未实现接口，当前调用抛出 `NotImplementedError`；SuperPoint/LightGlue、LoFTR 是研究路线而非已接入能力。
- **应用排障**：先确认坐标、模板纹理、掩膜和姿态范围，再看分数、分差、NMS 与候选上限；勿用降低阈值掩盖粗姿态网格导致的漏检。
- **已知状态及现有链接**：[接口与理论](pose_ncc.md)、[现场排障](pose_ncc.md#第六章现场排障与验证方法)、[实验记录](experiments.md)、[ML 边界](../ml/README.md)、[DL 边界](../dl/README.md)。传统姿态基线有合成回归与拒识记录，默认全分辨率复核不代表实时加速；不支持出画目标、透视或非刚性变形，未验证产线节拍。

### 05 边缘几何测量

- **算法卡片**：[测量与对焦索引](algorithm_cards/README.md#测量与对焦) · [测量M01–M13正文](algorithm_cards/measurement_focus.md#m01-卡尺-caliper) · [MF01斜坡／楔形矩定位](algorithm_cards/moment_fourier_focus.md#mf01) · [SF01结构化森林](algorithm_cards/structured_forests.md#sf01)；两张独立新卡均未实现，不补成A31归属证明；SF的2014 sharpening不回填2013。
- **知识库**：[边缘、亚像素、几何拟合的演进与工程实践](measurement_evolution.md)；含 Devernay 版本、Zernike 书目及 A31 证据边界，不等于毫米测量链已建。
- **传统**：已有 [卡尺与直线拟合](../traditional/caliper_measurement/caliper.py)、[剖面拟合与 Devernay 边缘](../traditional/subpixel/edge_subpixel.py)、[亚像素角点](../traditional/subpixel/corner_subpixel.py)。
- **ML**：手工边缘特征与真假边缘分类器待建设。
- **DL**：已有 [U-Net 风格边缘网络与损失骨架](../dl/subpixel_edge_net.py)，没有完整训练循环或已验证权重。
- **应用排障**：检查边缘极性、剖面方向、模糊宽度、噪声和异常点；区分像素定位误差、重复性与经标定后的尺寸误差。
- **已知状态及现有链接**：[合成精度记录](experiments.md)、[Zernike 与插值历史笔记](research_notes.md)、[DL 完成度](../dl/README.md)。Zernike 无可调用实现；A31 模糊补偿的原文归属尚缺证据，见[总览核验结论](measurement_evolution.md#4-devernayzernike-与-a31-的核验结论)。现有 Devernay 仅为简化基线；合成数据的低误差不构成真实尺寸精度承诺。

### 06 缺陷异常

- **算法卡片**：[其余领域索引](algorithm_cards/README.md#其余领域) · [D01–D04教学基线](algorithm_cards/remaining_domains.md#d01-配准后绝对差分) · [G19 PaDiM](algorithm_cards/padim_root_polynomial.md#g19) · [HA01共现纹理](algorithm_cards/haralick.md#ha01) · [OC01单类SVM](algorithm_cards/one_class_svm.md#oc01) · [SA01 SSIM-AE](algorithm_cards/ssim_autoencoder.md#sa01) · [DR01 DRAEM](algorithm_cards/draem.md#dr01) · [PC01 PatchCore](algorithm_cards/patchcore.md#pc01)；均未实现，指标不替代AE、focal不替代双网、D04不替代记忆库；06域具体卡已有不等于异常系统已建。
- **知识库**：[纹理、正常建模、异常定位的演进与工程实践](defect_evolution.md)；分开异常评分、空间热图与规格判定，基准节点不算检测算法。
- **传统**：对齐差分、背景建模、纹理统计待建设。
- **ML**：手工特征加单类分类器／异常评分待建设。
- **DL**：重建式、特征嵌入式异常检测及缺陷分割待建设。
- **应用排障**：面向划伤、污点、缺料等；先排除错位、照明和批次差异，再按缺陷尺寸及严重度统计漏检、误报，避免训练／测试样本泄漏。
- **已知状态及现有链接**：暂无缺陷数据集、专用实现或检出率记录；[定位基础](pose_ncc.md) 可辅助前置对齐，[实验记录](experiments.md) 可参考记录方式。匹配拒识不能直接解释为缺陷判定。

### 07 分类装配

- **算法卡片**：[A01装配规则基线](algorithm_cards/remaining_domains.md#a01-数量与相对几何装配规则) · [分类器与学习组件](algorithm_cards/README.md#学习组件) · [HA01原分类边界](algorithm_cards/haralick.md#ha01) · [RN01原版ResNet](algorithm_cards/resnet.md#rn01) · [FR01 Faster R-CNN](algorithm_cards/faster_rcnn.md#fr01) · [OM01 OpenMax](algorithm_cards/openmax.md#om01) · [SG01场景图](algorithm_cards/iterative_scene_graph.md#sg01)；07-01原分类链仍部分，G16不整体销项；具体卡均未实现，组件不替代模型、检测／开放集／关系推断不等于装配系统，源码差异和P2缺口保留。
- **知识库**：[分类、检测、关系与过程理解的演进和工程实践](classification_assembly_evolution.md)；对象类别、装配关系与工序证据分别验收，分类正确不等于装配合格。
- **传统**：几何规则、数量与相对位置约束、装配防错待建设。
- **ML**：通用 HOG／纹理特征分类与装配判定待建设；已有 SVM 模板候选验证示例。
- **DL**：零件分类、目标检测及装配关系识别待建设。
- **应用排障**：面向有无、错件、反装和漏装；先定义 OK/NG 与未知类，区分候选漏检和分类错误，检查类别不平衡及困难负样本。
- **已知状态及现有链接**：[ML 路线与限制](../ml/README.md)、[候选验证代码](../ml/hog_svm_matching.py)。现有示例每次调用生成合成样本并训练，概率未做现场校准，不是装配检测系统，也没有通用分类混淆矩阵记录。

### 08 字符编码

- **算法卡片**：[O01固定字库OCR](algorithm_cards/remaining_domains.md#o01-切字加模板相关-ocr) · [B01 Code128-B子集](algorithm_cards/remaining_domains.md#b01-code-128-b-读码与校验子集) · [LG01 LeNet-5／GTN](algorithm_cards/lenet_gtn.md#lg01) · [CT01 CTC](algorithm_cards/ctc.md#ct01) · [CR01 CRNN](algorithm_cards/crnn.md#cr01) · [EA01 EAST](algorithm_cards/east.md#ea01) · [DB01 DB](algorithm_cards/db_text_detection.md#db01) · [TR01 TrOCR](algorithm_cards/trocr.md#tr01)；均未实现生产接入，LG01复合仍一卡；检测不等于识别，CTC历史数值检查不升级CRNN，EAST／DB源码风险、TrOCR版本／词元资产与随机编号拒识缺口保留，不覆盖全部OCR／码制或标准符合性。
- **知识库**：[OCR 与编码双分支演进、标准版本及工程实践](ocr_codes_evolution.md)；文本检测、识别、解码、质量分级与业务语法分开，读到载荷不等于标准合规。
- **传统**：字符切分、模板字符识别、一维码／二维码解码待建设。
- **ML**：字符手工特征分类待建设。
- **DL**：文本检测、序列识别与端到端 OCR 待建设。
- **应用排障**：面向喷码、序列号和追溯码；检查字符尺寸、反光、运动模糊、静区及校验规则，区分“不可读”与“读错”。
- **已知状态及现有链接**：暂无 OCR／条码实现、字典或字符准确率评测；[对焦评分代码](../traditional/autofocus/focus_measures.py) 仅为成像检查基础，[评测目录](#横向基础与评测目录) 提供记录入口，不代表已有解码能力。

### 09 颜色外观

- **算法卡片**：[C01 Lab／Delta E76](algorithm_cards/remaining_domains.md#c01-cielab-与-delta-e-76) · [P35线性颜色校正](algorithm_cards/preprocessing.md#p35) · [G34根多项式RPCC](algorithm_cards/padim_root_polynomial.md#g34) · [RE01 1971 Retinex](algorithm_cards/retinex_1971.md#re01) · [GE01 Gray-Edge](algorithm_cards/gray_edge.md#ge01) · [FC01 FC4](algorithm_cards/fc4.md#fc01)；具体机制已有，均未实现／实拍标定。RE01的L1仍部分，原阈值／二维路径缺口保留；GE01二阶系数／mask及FC01头语义／fold风险不清除，不等于色度校准或仪器认证。
- **知识库**：[颜色表示、色差、外貌模型与校正算法的双线总览](color_appearance_evolution.md)；区分公式、模型、规范与算法，保留观察条件、空间纹理和批次验收边界。
- **传统**：颜色空间、白平衡、色差、光泽和纹理外观分析待建设。
- **ML**：颜色／纹理特征分类与外观分级待建设。
- **DL**：外观分类、质量回归待建设。
- **应用排障**：面向色偏、批次一致性和表面外观；检查光源光谱、曝光、白平衡、相机响应和背景，避免把照明差异当作产品色差。
- **已知状态及现有链接**：暂无颜色标定、色差指标或专用数据集；[真实对焦数据说明](../README.md#真实数据集rtimg)、[数据源接口](../traditional/autofocus/source.py) 仅供采集与组织方式参考，彩色图像的存在不等于已完成颜色评测。

### 10 三维重建测量

- **算法卡片**：[G01双目基线](algorithm_cards/remaining_domains.md#g01-双目-sad-块匹配与三角测量) · [G02点到点ICP](algorithm_cards/remaining_domains.md#g02-刚性-point-to-point-icp) · [G36 FTP教学卡](algorithm_cards/ftp_hybrid_servo.md#g36) · [CL01融合](algorithm_cards/curless_levoy.md#cl01) · [SM01 SGM](algorithm_cards/sgm.md#sm01) · [FP01 FPFH／SAC-IA](algorithm_cards/fpfh_sacia.md#范围版本与覆盖状态) · [PM01 PSMNet](algorithm_cards/psmnet.md#pm01) · [MD01 Monodepth2](algorithm_cards/monodepth2.md#md01) · [FG01 FCGF](algorithm_cards/fcgf.md#fg01) · [族级排除项](algorithm_cards/remaining_domains.md#泛称各列的族级边界)；G37–G42机制已有但均未实现，FP01两子算法仍一卡、2009方法原文未得保留P2，MD01尺度／采样与FG01稀疏映射／假负缺口保留P2，读卡状态不升级；FTP原机制仍缺证，G36保留P0缺卡。
- **知识库**：[深度获取、对应、配准、融合的演进与工程实践](3d_evolution.md)；主动测距、双目与学习先验并行，几何完整性、尺度和可追溯误差不能互相替代。
- **传统**：双目、结构光、三角测量、点云配准与几何拟合待建设。
- **ML**：点云手工特征分类、测量误差回归待建设。
- **DL**：学习式深度、立体匹配与点云网络待建设。
- **应用排障**：检查内外参、基线、同步、遮挡、反光和坐标系；用可追溯尺寸真值评估深度误差与完整率，二维匹配分数不能替代三维精度。
- **已知状态及现有链接**：暂无深度图、点云、三维重建或毫米测量实现；[二维坐标基础](pose_ncc.md#第一章坐标与变换) 仅可参考变换约定，[通用指标](../tests/metrics.py) 尚不是三维专用评测套件。

### 11 对焦成像质量

- **算法卡片**：[测量与对焦索引](algorithm_cards/README.md#测量与对焦) · [对焦F01–F14正文](algorithm_cards/measurement_focus.md#f01-brenner) · [G09／G10／G11起源卡](algorithm_cards/README.md#对焦起源) · [MF02离轴照明FCFNN](algorithm_cards/moment_fourier_focus.md#mf02)；三张起源卡有正文／答5但原机制未闭合，FCFNN有具体方法卡但未实现；评分、变换、诊断与搜索分开计数。
- **知识库**：[清晰度评分、峰模型、搜索与学习对焦的双线总览](autofocus_evolution.md)；含七评分实际定义、RTImg 代理真值及相机／电机闭环边界，SFF 深度不等于自动对焦。
- **传统**：已有 [清晰度函数](../traditional/autofocus/focus_measures.py)、[对焦搜索](../traditional/autofocus/focus_search.py)、[真实数据评估](../traditional/autofocus/real_dataset.py) 和 [数据源抽象](../traditional/autofocus/source.py)。
- **ML**：手工清晰度特征加 RF 等回归器待建设。
- **DL**：[CNN 对焦回归](../dl/autofocus_cnn.py) 包含网络与训练循环，仍缺统一随机状态、权重保存和现场精度验证。
- **应用排障**：检查 ROI、纹理、噪声、曲线单峰性、真值定义和采样间距；区分图像解码与评分成本。现有大图实验中降采样损伤对焦灵敏度，不能直接当作无损加速。
- **已知状态及现有链接**：[RTImg 评估与已知问题](../README.md#真实数据集rtimg)、[合成实验](experiments.md)、[对焦曲线模型](research_notes.md)、[DL 状态](../dl/README.md)。既有记录中 DCT 大图评分受平坦背景稀释，方差对内容敏感；没有通用成像质量认证或真实相机／电机闭环验证。

### 12 跟踪运动视觉引导

- **算法卡片**：[T01 Kalman](algorithm_cards/remaining_domains.md#t01-线性-kalman-常速状态估计) · [T02 MOSSE](algorithm_cards/remaining_domains.md#t02-mosse-型自适应相关滤波) · [T03光流](algorithm_cards/remaining_domains.md#t03-lucaskanade-局部光流) · [T04关联](algorithm_cards/remaining_domains.md#t04-门控后一对一全局指派) · [H01 Hungarian](algorithm_cards/hungarian.md#h01) · [G44混合切换伺服](algorithm_cards/ftp_hybrid_servo.md#g44) · [SO01 SORT](algorithm_cards/sort.md#so01) · [SI01 SiamFC](algorithm_cards/siamfc.md#si01) · [DS01 Deep SORT](algorithm_cards/deep_sort.md#ds01) · [BT01 BYTE／ByteTrack](algorithm_cards/bytetrack.md#bt01)；均无专用实现，G45–G48机制已有、深度保留P2，原五层与训练／权重／依赖／视频缺口不变。G44机制已有、原式已辨识，证明一致性与采样安全独立P0保留，无控制仿真、硬件在环或实机，非闭环系统。
- **知识库**：[状态估计、关联、跟踪与视觉控制的双线总览](tracking_guidance_evolution.md)；分开感知与控制，坐标同步、观测年龄、延迟和独立安全闭环须另行验收。
- **传统**：光流、卡尔曼滤波、数据关联、轨迹估计和视觉伺服待建设。
- **ML**：手工特征的目标关联或运动状态判别待建设。
- **DL**：学习式跟踪、多目标关联、视觉引导待建设。
- **应用排障**：检查时间戳、帧间位移、遮挡、目标身份切换、相机与机器人坐标链、端到端延迟和失锁策略；单帧定位成功不等于闭环稳定。
- **已知状态及现有链接**：暂无视频跟踪、运动控制接口或闭环测试；[单帧定位基础](pose_ncc.md)、[定位场景评测](../tests/benchmark_pose_ncc.py) 仅是潜在前置能力，没有轨迹和运动引导指标。

## 横向基础与评测目录

以下为跨领域共用入口；本次只建立文档索引，不新增空目录、算法或测试脚本。

| 主题 | 现有入口 | 阅读重点与边界 |
|---|---|---|
| 算法学习与组件 | [255卡逐项索引](algorithm_cards/README.md)、[学习组件27卡](algorithm_cards/learning_components.md) | 文件＋卡号唯一；五层讲解覆盖与实现／验证状态独立 |
| 环境与路线隔离 | [快速开始](../README.md#快速开始)、[传统依赖](../requirements.txt)、[ML 依赖](../requirements-ml.txt)、[DL 依赖](../requirements-dl.txt) | 从仓库根用 `-m` 运行；环境历史记录不等于当前依赖已安装 |
| 数学与坐标基础 | [坐标变换](pose_ncc.md#第一章坐标与变换)、[ZNCC 与数值边界](pose_ncc.md#第二章zncc掩膜与数值边界)、[研究笔记](research_notes.md) | 像素中心、ROI、单位、插值、退化输入与数值稳定性 |
| 搜索与决策 | [姿态网格与金字塔](pose_ncc.md#第三章姿态网格与空间金字塔)、[亚像素精化](pose_ncc.md#第四章平移亚像素精化)、[拒识与多目标](pose_ncc.md#第五章拒识与多目标) | 采样误差、候选截断、阈值、分差和 NMS 的适用条件 |
| 数据与真值 | [合成数据](../data/synthetic.py)、[数据源](../traditional/autofocus/source.py)、[RTImg 说明](../README.md#真实数据集rtimg) | 固定采样条件、标注坐标及单位；真实对焦真值采用距离最近帧，不假设存在 `-0000` 参考帧 |
| 通用指标 | [metrics.py](../tests/metrics.py) | 已有 RMSE、逐点定位误差、曲线单峰性、预热后的中位耗时；不能代替各领域专用指标 |
| 兼容与边界回归 | [冒烟测试](../tests/test_smoke.py)、[姿态回归测试](../tests/test_pose_ncc.py) | 既有记录为 8/8 与 37/37；测试断言容差不是现场精度承诺 |
| 场景评测 | [姿态匹配基准](../tests/benchmark_pose_ncc.py)、[实验记录](experiments.md) | 合成真值、一对一目标统计、漏检／误检与耗时；保留粗网格失败场景，不能只看退出码 |
| 精度与重复性 | [现场排障与验证](pose_ncc.md#第六章现场排障与验证方法)、[ML 研究约定](../ml/README.md)、[DL 研究约定](../dl/README.md) | 分开报告真值误差与重复测量离散度；记录数据划分、随机状态、权重及软硬件环境 |
| 可视化与实验记录 | [绘图工具](../traditional/viz.py)、[实验记录与待做对比](experiments.md)、[笔记模板](research_notes.md) | 同一真值与指标比较三条路线，明确失败样本、已知边界和未验证项 |

待补评测包括分割 IoU/Dice、分类混淆矩阵、缺陷漏检／误报、字符错误率、色差、三维误差和跟踪指标；当前不得把这些列为已具备的统一评测能力。

## 阅读路径

1. **首次运行**：[仓库快速开始](../README.md#快速开始) → [合成数据](../data/synthetic.py) → [冒烟测试](../tests/test_smoke.py) → [基线记录](experiments.md)，先理解数据、运行入口与历史结果。
2. **按任务选型**：[12 领域目录](#12-领域目录) → 对应领域双线总览 → 传统 / ML / DL 与状态 → 现有实现和边界；实现待建设领域先定义任务、真值与验收指标，不把总览或相关基础当作交付能力。
3. **深入匹配**：[可运行示例](pose_ncc.md#可运行示例) → [坐标与变换](pose_ncc.md#第一章坐标与变换) → [理论与搜索边界](pose_ncc.md) → [回归测试](../tests/test_pose_ncc.py) → [场景评测](../tests/benchmark_pose_ncc.py)。
4. **测量与对焦**：[测量双线总览](measurement_evolution.md) 或 [对焦双线总览](autofocus_evolution.md) → [测量与对焦卡片](algorithm_cards/README.md#测量与对焦) → [研究笔记](research_notes.md)／[合成实验](experiments.md) → [真实对焦记录](../README.md#真实数据集rtimg)，保留历史来源限制，区分合成精度、代理真值与现场证据。
5. **现场排障与扩展**：[现场排障](pose_ncc.md#第六章现场排障与验证方法) → [横向基础与评测](#横向基础与评测目录) → [ML 完成度](../ml/README.md) / [DL 完成度](../dl/README.md) → [待做实验](experiments.md#待做对比实验)，先留证据与复现条件，再调参或引入新路线。
6. **逐卡系统学习**：[学习规范](algorithm_learning_standard.md) → [逐卡索引](algorithm_cards/README.md) → 对应正文及来源 → 原验证记录；按文件＋卡号记录缺口，不把族级基线或L5设计当作全部算法已复现。
