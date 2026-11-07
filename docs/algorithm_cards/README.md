# 算法卡片逐卡索引

[文档总目录](../README.md) · [学习规范](../algorithm_learning_standard.md) · [演进覆盖与历史任务](../evolution_coverage.md) · [旧稿入口与迁移边界](#旧稿入口与迁移边界)

## 范围与计数

复核日期：2026-11-06。按45个文件中的独立编号正文逐卡计数，共 **255张卡＝245＋10＝206＋9＋4＋4＋8＋4＋4＋4＋2＋10**。这是当前知识库的具体方法、实现组件、历史目标卡与**族级教学基线**，不是所有算法全史，不递归覆盖论文的全部参考文献或方法族的所有变体。算法、算子、模型、策略、损失、指标与工程组合分开标注，卡片数不等于独立算法数或已实现数量。

唯一键为 `文件名:卡号`，文件名相对本目录；例如 `calibration.md:C01`、`learning_components.md:C01`、`remaining_domains.md:C01` 是三个不同条目。同名方法跨领域保留各自契约，不跨文件合并卡号；旧稿另有命名空间，不凭编号认定对应。无独立编号的颜色补充、族级边界表、来源和检查器不另计卡。

**领域总览与卡片分账**：[12领域双线总览](../README.md#双线阅读与编写规范)不自动另计卡或L5完成。[覆盖矩阵](../evolution_coverage.md)共 **113节点：已有83／部分2／缺卡4／非算法24**；48个历史ID为42已有、2部分、4缺卡。剩余 **4项P0缺卡＋2部分＝6项覆盖任务**：G09／G10／G11／G36缺卡、G02部分P0、G16部分P1；G16仅07-01计剩余。**G44机制已有，证明一致性与采样安全[独立P0保留、不降级](../evolution_coverage.md#g44-independent-p0)**，不另增覆盖任务。255卡不增，HED／U-Net仍一节点，LG01／FP01各仍一卡；ZM01及HC73全部分，HA01、RE01及其余卡原等级、实现与P2边界不变。缺原材料见[最小缺页清单](../evolution_coverage.md#minimum-source-pages)。

| 文件 | 实际数量 | 编号范围与计数边界 | 逐卡索引 |
|---|---|---|---|
| [preprocessing.md](preprocessing.md) | 46 | P01–P44 共44张＋Q01–Q02 两张指标卡；颜色补充不另计 | [预处理与指标](#预处理与指标) |
| [segmentation.md](segmentation.md) | 31 | S01–S31；含区域算子、RoIAlign 与形状先验基线 | [分割区域](#分割区域) |
| [matching.md](matching.md) | 51 | 01–51；含两套实现、搜索拒识、计算与评测组件 | [定位匹配配准](#定位匹配配准) |
| [calibration.md](calibration.md) | 10 | C01–C10；含成像模型、几何估计与观测前端 | [标定坐标](#标定坐标) |
| [learning_components.md](learning_components.md) | 27 | C01–C27；分类器、网络算子、优化、损失与指标 | [学习组件](#学习组件) |
| [measurement_focus.md](measurement_focus.md) | 27 | M01–M13 测量13张＋F01–F14 对焦14张；评分仅F01–F07 | [测量与对焦](#测量与对焦) |
| [remaining_domains.md](remaining_domains.md) | 14 | D01–D04、A01、O01、B01、C01、G01–G02、T01–T04 | [其余领域](#其余领域) |
| [focus_origins.md](focus_origins.md) | 3 | G09／G10／G11；历史目标未闭合，不与F02／F01／F06混号 | [对焦起源](#对焦起源) |
| [ftp_hybrid_servo.md](ftp_hybrid_servo.md) | 2 | G36／G44；教学参考与原机制证据分开 | [FTP与混合切换伺服](#ftp与混合切换伺服) |
| [moment_fourier_focus.md](moment_fourier_focus.md) | 2 | MF01／MF02，分别承接G15／G43 | [矩定位与傅里叶对焦](#矩定位与傅里叶对焦) |
| [padim_root_polynomial.md](padim_root_polynomial.md) | 2 | G19／G34，分别承接异常与颜色回归 | [PaDiM与根多项式校正](#padim与根多项式校正) |
| [hungarian.md](hungarian.md) | 1 | H01，承接G13；现代内核与1955程序分开 | [Hungarian指派](#hungarian指派) |
| [structured_forests.md](structured_forests.md) | 1 | SF01，承接G14；2013与后续sharpening分开 | [结构化森林边缘](#结构化森林边缘) |
| [haralick.md](haralick.md) | 1 | HA01，承接G16；06-01已有／07-01部分 | [Haralick共现纹理](#haralick共现纹理) |
| [one_class_svm.md](one_class_svm.md) | 1 | OC01，承接G17；不是两类SVC或SVDD | [One-Class SVM](#one-class-svm) |
| [ssim_autoencoder.md](ssim_autoencoder.md) | 1 | SA01，承接G18；SSIM指标不替代AE | [SSIM Autoencoder](#ssim-autoencoder) |
| [draem.md](draem.md) | 1 | DR01，承接G20；合成监督、重建与判别联合学习 | [DRAEM](#draem) |
| [patchcore.md](patchcore.md) | 1 | PC01，承接G21；论文重加权与固定代码分开 | [PatchCore](#patchcore) |
| [resnet.md](resnet.md) | 1 | RN01，承接G22；五种架构不另计卡 | [ResNet](#resnet) |
| [faster_rcnn.md](faster_rcnn.md) | 1 | FR01，承接G23；RPN／RoIPool及训练方案不另计卡 | [Faster R-CNN](#faster-r-cnn) |
| [openmax.md](openmax.md) | 1 | OM01，承接G24；尾部与重分配不拆卡 | [OpenMax](#openmax) |
| [iterative_scene_graph.md](iterative_scene_graph.md) | 1 | SG01，承接G25；对象／关系两类状态不拆卡 | [迭代消息传递场景图](#迭代消息传递场景图) |
| [lenet_gtn.md](lenet_gtn.md) | 1 | LG01，承接G26；两子方法各五层、共享答5，仍一卡 | [LeNet-5与GTN](#lenet-5与gtn) |
| [ctc.md](ctc.md) | 1 | CT01，承接G27；前后向／解码不拆卡 | [CTC](#ctc) |
| [crnn.md](crnn.md) | 1 | CR01，承接G28；引用CT01不重复计卡 | [CRNN](#crnn) |
| [east.md](east.md) | 1 | EA01，承接G29；RBOX／QUAD不拆卡 | [EAST](#east) |
| [db_text_detection.md](db_text_detection.md) | 1 | DB01，承接G30；训练分支及恢复不拆卡 | [DB文本检测](#db文本检测) |
| [trocr.md](trocr.md) | 1 | TR01，承接G31；预训练／生成／拒识不拆卡 | [TrOCR](#trocr) |
| [retinex_1971.md](retinex_1971.md) | 1 | RE01，承接G32；1971原机制与参考配置分开 | [1971 Retinex](#1971-retinex) |
| [gray_edge.md](gray_edge.md) | 1 | GE01，承接G33；阶数／范数特例不拆卡 | [Gray-Edge](#gray-edge) |
| [fc4.md](fc4.md) | 1 | FC01，承接G35；三种输出语义不拆卡 | [FC4](#fc4) |
| [curless_levoy.md](curless_levoy.md) | 1 | CL01，承接G37；融合／可见性／提取不拆卡 | [Curless–Levoy体积融合](#curlesslevoy体积融合) |
| [sgm.md](sgm.md) | 1 | SM01，承接G38；MI与路径聚合不拆卡 | [SGM](#sgm) |
| [fpfh_sacia.md](fpfh_sacia.md) | 1 | FP01，承接G39；两子算法各五层、共享答5，仍一卡 | [FPFH与SAC-IA](#fpfh与sac-ia) |
| [psmnet.md](psmnet.md) | 1 | PM01，承接G40；basic／stacked不拆卡 | [PSMNet](#psmnet) |
| [monodepth2.md](monodepth2.md) | 1 | MD01，承接G41；M／S／MS模式不拆卡 | [Monodepth2](#monodepth2) |
| [fcgf.md](fcgf.md) | 1 | FG01，承接G42；网络／损失／采样不拆卡 | [FCGF](#fcgf) |
| [sort.md](sort.md) | 1 | SO01，承接G45；框状态／指派／生命周期不拆卡 | [SORT](#sort) |
| [siamfc.md](siamfc.md) | 1 | SI01，承接G46；离线训练／在线搜索不拆卡 | [SiamFC](#siamfc) |
| [deep_sort.md](deep_sort.md) | 1 | DS01，承接G47；训练来源／运动门／级联不拆卡 | [Deep SORT](#deep-sort) |
| [bytetrack.md](bytetrack.md) | 1 | BT01，承接G48；BYTE策略与ByteTrack系统不拆卡 | [BYTE与ByteTrack](#byte与bytetrack) |
| [kb_zernike_extensions.md](kb_zernike_extensions.md) | 2 | KB01／ZM01；G01已有、G02仍部分，不与三维G01／G02混号 | [KB与Zernike扩展](#kb与zernike扩展) |
| [edge_evolution_extensions.md](edge_evolution_extensions.md) | 3 | DE95／DE17／HE01；G03／G05／G04，U-Net复用S21 | [边缘演进扩展](#边缘演进扩展) |
| [lbp_hog_extensions.md](lbp_hog_extensions.md) | 3 | LB96／HG05／LB02；G06／G07／G08，版本与分类分支不拆卡 | [LBP与HOG扩展](#lbp与hog扩展) |
| [sff_haralick_extensions.md](sff_haralick_extensions.md) | 2 | SF94／HC73；G12已有、G16仍部分，06-01不重复计数 | [SFF与Haralick扩展](#sff与haralick扩展) |
| 合计 | **255** | 46＋31＋51＋10＋27＋27＋14＝206；再加9＋4＋4＋8＋4＋4＋4＋2＋10＝255＝245＋10 | 45文件，不计本索引及旧稿 |

## 覆盖与验证口径

下段汇集前批卡片的状态限定，其中“本轮DS01／BT01”沿指245卡轮次；四文件10卡状态属于前次255卡纳入轮次，不把历史记录当作本次新验证。

五层列固定按 **L1步骤／L2机制／L3工程权衡／L4数学推导／L5实现** 排列。为便于逐行阅读，`核`＝原文“已核验”，`部`＝原文“部分”；匹配稿的 `核（源码）`、`部（复现设计）`、`部（混合）` 保留限定。SF01的L4保留 `核（限定代数）`，HA01的L1为 `部（GLCM链核、原分类链部分）`，不把子链已核提升为整卡已核。SA01的L4为 `核（限定SSIM代数）`，RN01的L4为 `核（限定代数与静态结构）`；DR01／PC01的L4仍为部，四卡L5均部。前批FR01／OM01／SG01／CR01／EA01／DB01的L4仍为部；CT01的L4核限既定离散模型，LG01两子方法的L4核分别限定结构代数／局部梯度与有限DAG路径／组合梯度，八卡L3／L5均部。前批TR01为核／核／部／部／部，RE01为部／核／部／部／部；GE01的L4为核（限定条件代数），FC01的L4核限非零及可微域。前批CL01的L4仍部，SM01的L4核限代数，PM01的L4核限模型；FP01两子算法均部／核／部／部／部，保留固定源码链与各层限定。前批MD01／FG01／SO01／SI01均核／核／部／核（各自限定）／部，L4分别限针孔与局部可微域、显式模型与固定索引局部梯度、模型代数、shape／梯度／坐标代数。本轮DS01／BT01均核／核／部／核（限定模型代数）／部、各答5；DS01限状态／门控／余弦与训练梯度，BT01限状态／IoU／分数融合与指定依赖拒配模型，均不保证身份正确性，不把阅读状态升级为实现。上述卡L3／L5均部。其余合法状态为“未写”“不适用并说明原因”，本次未新增这些状态。

本索引只转录各文件覆盖表或五元协议，**不升级验证状态**。“已核验”仍只具有原文所列的源码、静态资料或条件代数含义，不自动表示算法运行、训练完成、全史考证或现场验收。相同算法在不同文件的状态可以不同，不取较高状态覆盖另一张卡。

本轮扩展卡状态逐项转录：KB01为部／核／部／部／部，ZM01与HC73均部／部／部／部／部；DE95／DE17／HE01／LB96／HG05／LB02／SF94均核／核／部／核（各自限定）／部。L4依次限定三点与一维模型、峰位／链评分／固定支持复杂度、二值损失／链式梯度、KL／列联表G／不变性、维数／投票／归一化／线性SVM、循环群／VAR／似然、等距重写／尺度还原／条件敏感性；不授予一般误差界或收敛保证。

验证边界列是定位摘要，详细输入、来源与未覆盖项以卡片为准。前次四文件10卡纳入及其50层／50组问答检查属于历史；本轮同步G44、研究笔记、focus／HC73归因、台账及必要总览，核对本地链接、计数及原状态文字。原稿中的下载、公式目视、代数／源码检查、37项姿态回归、8/8冒烟及其他算法成功记录均不作为本轮执行。FP01与LG01各两子方法仍一卡；不重跑算法检查器、不训练、不复测RTImg或作物理／闭环验收，外链可用性与原文证据等级不因本地链接检查升级。

## 预处理与指标

[领域总览](../preprocessing.md) · [原卡与覆盖协议](preprocessing.md#范围与证据约定)

P01–P22 的原检查器和历史验算不扩展到新增P23–P44或Q01–Q02；新增卡的例题、伪代码与指标参考函数不是已执行整图恢复实验。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| preprocessing.md:P01 | [暗场校正](preprocessing.md#p01-暗场校正) | 校正算法 | 部／部／部／部／部 | 未实现；参考均值与减法验算，非采集验收 |
| preprocessing.md:P02 | [平场校正](preprocessing.md#p02-平场校正) | 校正算法 | 部／部／部／部／部 | 未实现；增益公式验算，不含真实响应标定 |
| preprocessing.md:P03 | [高斯滤波](preprocessing.md#p03-高斯滤波) | 滤波算子 | 部／部／部／部／部 | 任务局部实现；离散核验算，非独立增强管线 |
| preprocessing.md:P04 | [中值滤波](preprocessing.md#p04-中值滤波) | 滤波算子 | 部／部／部／部／部 | 未实现；秩统计验算 |
| preprocessing.md:P05 | [双边滤波](preprocessing.md#p05-双边滤波) | 滤波算法 | 部／部／部／部／部 | 未实现；两点权重验算 |
| preprocessing.md:P06 | [非局部均值 NLM](preprocessing.md#p06-非局部均值-nlm) | 去噪算法 | 部／部／部／部／部 | 未实现；块权重验算 |
| preprocessing.md:P07 | [BM3D](preprocessing.md#p07-bm3d) | 去噪算法 | 部／部／部／部／部 | 未实现；仅收缩与聚合子步骤验算 |
| preprocessing.md:P08 | [引导滤波](preprocessing.md#p08-引导滤波-guided-filter) | 滤波算法 | 部／部／部／部／部 | 未实现；局部回归验算 |
| preprocessing.md:P09 | [Fast Guided Filter](preprocessing.md#p09-fast-guided-filter) | 近似加速算法 | 部／部／部／部／部 | 未实现；全尺寸重建验算，不保证与原法等价 |
| preprocessing.md:P10 | [DnCNN](preprocessing.md#p10-dncnn) | 学习恢复模型 | 部／部／部／部／部 | 无权重、训练或推理；仅残差接口验算 |
| preprocessing.md:P11 | [Noise2Noise](preprocessing.md#p11-noise2noise) | 监督策略 | 部／部／部／部／部 | 无配对采集或训练；仅期望目标验算 |
| preprocessing.md:P12 | [线性拉伸](preprocessing.md#p12-线性拉伸) | 强度映射 | 部／部／部／部／部 | 无统一增强接口；映射验算 |
| preprocessing.md:P13 | [gamma 变换](preprocessing.md#p13-gamma-变换) | 强度映射 | 部／部／部／部／部 | 未实现；幂映射验算 |
| preprocessing.md:P14 | [直方图均衡](preprocessing.md#p14-直方图均衡) | 增强算法 | 部／部／部／部／部 | 未实现；CDF映射验算 |
| preprocessing.md:P15 | [CLAHE](preprocessing.md#p15-clahe) | 局部增强算法 | 部／部／部／部／部 | 未实现；裁剪再分配验算 |
| preprocessing.md:P16 | [同态滤波](preprocessing.md#p16-同态滤波) | 频域增强算法 | 部／部／部／部／部 | 未实现；对数频域恒等验算 |
| preprocessing.md:P17 | [维纳恢复](preprocessing.md#p17-维纳恢复) | 非盲恢复算法 | 部／部／部／部／部 | 未实现；单频增益验算 |
| preprocessing.md:P18 | [最近邻插值](preprocessing.md#p18-最近邻插值) | 重采样算子 | 部／部／部／部／部 | 数据生成调用；非标签处理接口 |
| preprocessing.md:P19 | [双线性插值](preprocessing.md#p19-双线性插值) | 重采样算子 | 部／部／部／部／部 | 匹配／卡尺局部实现；内点采样验算 |
| preprocessing.md:P20 | [双三次插值](preprocessing.md#p20-双三次插值) | 重采样算子 | 部／部／部／部／部 | 数据生成调用；非通用重采样接口 |
| preprocessing.md:P21 | [面积重采样](preprocessing.md#p21-面积重采样) | 重采样算子 | 部／部／部／部／部 | 匹配金字塔调用；面积权重验算 |
| preprocessing.md:P22 | [灰度转换](preprocessing.md#p22-灰度转换) | 颜色表示转换 | 部／部／部／部／部 | 对焦局部实现；颜色补充不另计卡或视作已接入 |
| preprocessing.md:P23 | [坏点表与替换](preprocessing.md#p23) | 校正教学基线 | 部／部／部／部／部 | 未实现；固定表＋同色中值，不恢复整簇／列真值 |
| preprocessing.md:P24 | [DSNU／PRNU](preprocessing.md#p24) | 响应模型与估计 | 部／部／部／部／部 | 未实现；多照度线性拟合，非全噪声模型 |
| preprocessing.md:P25 | [均值与线性卷积](preprocessing.md#p25) | 滤波算子 | 部／部／部／部／部 | 仅卡尺filter2D局部调用；无统一算子 |
| preprocessing.md:P26 | [形态学背景估计](preprocessing.md#p26) | 背景估计基线 | 部／部／部／部／部 | 未实现；灰度开闭背景，非二值分割替代 |
| preprocessing.md:P27 | [背景减除](preprocessing.md#p27) | 加性校正基线 | 部／部／部／部／部 | 未实现；静态背景，不覆盖视频前景模型族 |
| preprocessing.md:P28 | [背景除法](preprocessing.md#p28) | 乘性校正基线 | 部／部／部／部／部 | 未实现；不等于独立平场标定 |
| preprocessing.md:P29 | [局部归一化](preprocessing.md#p29) | 统计变换算子 | 部／部／部／部／部 | 未实现；局部均值／方差，ZNCC分数不算此算子 |
| preprocessing.md:P30 | [Richardson–Lucy](preprocessing.md#p30) | 非盲反卷积基线 | 部／部／部／部／部 | 未实现；与维纳并列，不覆盖全部非盲方法 |
| preprocessing.md:P31 | [盲反卷积](preprocessing.md#p31) | 族级交替优化设计 | 部／部／部／部／部 | 未实现；受约束基线，非完整盲恢复家族 |
| preprocessing.md:P32 | [反差遮罩锐化](preprocessing.md#p32) | 锐化教学基线 | 部／部／部／部／部 | 未实现；不是反卷积或全部锐化器 |
| preprocessing.md:P33 | [Bayer去马赛克](preprocessing.md#p33) | 颜色重建基线 | 部／部／部／部／部 | 无RAW管线；同色线性基线，非学习式去马赛克 |
| preprocessing.md:P34 | [白平衡](preprocessing.md#p34) | 校正教学基线 | 部／部／部／部／部 | 未实现；中性参考对角增益，非全部自动白平衡 |
| preprocessing.md:P35 | [颜色校正矩阵](preprocessing.md#p35) | 颜色模型估计 | 部／部／部／部／部 | 未实现；线性3×3拟合，非光谱恢复／完整颜色管理 |
| preprocessing.md:P36 | [裁剪与ROI](preprocessing.md#p36) | 坐标交接组件 | 部／部／部／部／部 | 匹配局部协议；无统一预处理对象 |
| preprocessing.md:P37 | [缩放与采样坐标](preprocessing.md#p37) | 几何采样组件 | 部／部／部／部／部 | 匹配局部resize；非全采样规范 |
| preprocessing.md:P38 | [旋转与重采样](preprocessing.md#p38) | 几何采样组件 | 部／部／部／部／部 | 匹配局部warp；非通用几何管线 |
| preprocessing.md:P39 | [畸变校正重采样](preprocessing.md#p39) | 标定交接组件 | 部／部／部／部／部 | 未实现标定／remap管线；模型另接标定卡 |
| preprocessing.md:P40 | [ML选择与旁路](preprocessing.md#p40) | 族级决策策略 | 部／部／部／部／部 | 无训练数据／策略；有限动作风险设计，非恢复器 |
| preprocessing.md:P41 | [DL去模糊](preprocessing.md#p41) | 族级网络设计 | 部／部／部／部／部 | 无网络／权重／训练；配对残差CNN，非全族复现 |
| preprocessing.md:P42 | [DL超分辨率](preprocessing.md#p42) | 族级网络设计 | 部／部／部／部／部 | 无网络／权重／训练；固定倍率，不支持任意倍率承诺 |
| preprocessing.md:P43 | [DL低照度](preprocessing.md#p43) | 族级网络设计 | 部／部／部／部／部 | 无网络／权重／训练；受限增益＋残差基线 |
| preprocessing.md:P44 | [DL颜色与生成边界](preprocessing.md#p44) | 族级网络设计 | 部／部／部／部／部 | 无网络／权重；确定性基线，GAN／扩散求解未展开 |
| preprocessing.md:Q01 | [PSNR](preprocessing.md#q01-psnr-全参考指标) | 全参考指标 | 部／部／部／部／部 | 参考函数未执行；不是缺陷召回或下游精度 |
| preprocessing.md:Q02 | [SSIM](preprocessing.md#q02-ssim-单尺度结构相似度) | 全参考指标 | 部／部／部／部／部 | 单尺度参考函数未执行；未与库数值交叉验证 |

## 分割区域

[领域总览](../segmentation_regions.md) · [原卡覆盖协议](segmentation.md#阅读协议与证据边界)

31卡均无完整分割管线实测；L1／L2的“核”仅限原稿静态对照。外部来源转引未重访的限制保留，所有未明确执行的测试仍为设计。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| segmentation.md:S01 | [固定阈值](segmentation.md#s01-固定阈值) | 阈值算法 | 核／核／部／部／部 | 未接入；强度可分前提，不含业务定参 |
| segmentation.md:S02 | [自适应均值阈值](segmentation.md#s02-自适应均值阈值) | 阈值算法 | 核／核／部／部／部 | 未接入；窗口及背景尺度待验 |
| segmentation.md:S03 | [自适应高斯阈值](segmentation.md#s03-自适应高斯阈值) | 阈值算法 | 核／核／部／部／部 | 未接入；局部加权，不是Otsu |
| segmentation.md:S04 | [Otsu](segmentation.md#s04-otsu-自动全局阈值) | 阈值算法 | 核／核／部／部／部 | 未接入；直方图目标不保证业务区域正确 |
| segmentation.md:S05 | [腐蚀](segmentation.md#s05-腐蚀-erosion) | 形态学算子 | 核／核／部／部／部 | 未接入；核与边界测试待执行 |
| segmentation.md:S06 | [膨胀](segmentation.md#s06-膨胀-dilation) | 形态学算子 | 核／核／部／部／部 | 未接入；连接区域可能合并真实实例 |
| segmentation.md:S07 | [开运算](segmentation.md#s07-开运算-opening) | 形态学算子 | 核／核／部／部／部 | 未接入；形状筛选不等于面积过滤 |
| segmentation.md:S08 | [闭运算](segmentation.md#s08-闭运算-closing) | 形态学算子 | 核／核／部／部／部 | 未接入；尺度补全不等于拓扑填孔 |
| segmentation.md:S09 | [孔洞填充](segmentation.md#s09-孔洞填充) | 拓扑算法 | 核／核／部／部／部 | 未接入；邻接与ROI外界需冻结 |
| segmentation.md:S10 | [连通域](segmentation.md#s10-连通域-connected-components) | 区域标记算法 | 核／核／部／部／部 | 未接入；标签编号不是类别／实例真值 |
| segmentation.md:S11 | [区域生长](segmentation.md#s11-区域生长-region-growing) | 种子分割基线 | 核／核／部／部／部 | 未接入；动态规则有顺序依赖 |
| segmentation.md:S12 | [洪水填充](segmentation.md#s12-洪水填充-flood-fill) | 可达性算法 | 核／核／部／部／部 | 未接入；固定／浮动范围及API未运行 |
| segmentation.md:S13 | [距离变换](segmentation.md#s13-距离变换-distance-transform) | 几何算子 | 核／核／部／部／部 | 未接入；距离图不直接给实例 |
| segmentation.md:S14 | [VS浸水分水岭](segmentation.md#s14-vincentsoillevs浸水分水岭) | 分水岭算法 | 核／核／部／部／部 | 未实现；不等于全部分水岭变体 |
| segmentation.md:S15 | [标记控制分水岭](segmentation.md#s15-标记控制分水岭) | 受控分割算法 | 核／核／部／部／部 | 未接入；种子生成及平台协议另验 |
| segmentation.md:S16 | [负距离标记分水岭](segmentation.md#s16-负距离标记分水岭) | 工程组合 | 核／核／部／部／部 | 未接入；依赖中心与颈部形状假设 |
| segmentation.md:S17 | [二值图割](segmentation.md#s17-二值图割-graph-cut) | 能量优化算法 | 核／核／部／部／部 | 未接图割库；可图表示条件与小图穷举待验 |
| segmentation.md:S18 | [GrabCut](segmentation.md#s18-grabcut) | 交互分割算法 | 核／核／部／部／部 | 未接入；非凸交替不保证全局分割正确 |
| segmentation.md:S19 | [随机游走](segmentation.md#s19-随机游走-random-walker) | 图扩散算法 | 核／核／部／部／部 | 未接入；未知分量需连接有效种子 |
| segmentation.md:S20 | [FCN](segmentation.md#s20-fcn) | 语义分割模型 | 核／核／部／部／部 | 未实现；无区域训练与验收 |
| segmentation.md:S21 | [U-Net与EdgeNet边界](segmentation.md#s21-u-net-与-edgenet-边界) | 模型与骨架对照 | 核／核／部／部／部 | 原始U-Net未复现；边缘骨架不是区域分割系统 |
| segmentation.md:S22 | [RoIAlign](segmentation.md#s22-roialign) | 特征采样算子 | 核／核／部／部／部 | 未接入；不是独立分割模型 |
| segmentation.md:S23 | [Mask R-CNN](segmentation.md#s23-mask-r-cnn) | 实例分割模型 | 核／核／部／部／部 | 未实现；无实例训练、权重与指标记录 |
| segmentation.md:S24 | [原始SAM](segmentation.md#s24-原始-sam) | 提示分割模型 | 核／核／部／部／部 | 未接入、无权重；不自带业务分类 |
| segmentation.md:S25 | [颜色／多通道阈值](segmentation.md#seg-card-25) | 阈值族基线 | 核／核／部／部／部 | 未接入；轴对齐范围规则，非概率分类 |
| segmentation.md:S26 | [形态学梯度](segmentation.md#seg-card-26) | 复合算子 | 核／核／部／部／部 | 未接入；输出边带／地形而非实例 |
| segmentation.md:S27 | [顶帽](segmentation.md#seg-card-27) | 残差算子 | 核／核／部／部／部 | 未接入；小亮结构响应不是缺陷概率 |
| segmentation.md:S28 | [黑帽](segmentation.md#seg-card-28) | 残差算子 | 核／核／部／部／部 | 未接入；小暗结构响应不是拓扑填孔 |
| segmentation.md:S29 | [轮廓与孔洞层级](segmentation.md#seg-card-29) | 表示与统计组件 | 核／核／部／部／部 | 未接入；保留孔洞层级不等于填孔 |
| segmentation.md:S30 | [极值与形态学重建](segmentation.md#seg-card-30) | 算子族基线 | 核／核／部／部／部 | 未接入；极值筛选不等于完整分水岭 |
| segmentation.md:S31 | [凹点／形状先验拆分](segmentation.md#seg-card-31) | 族级工程规则 | 核／核／部／部／部 | 未实现；冻结规则，非唯一标准算法或论文复现 |

## 定位匹配配准

[领域总览](../matching_evolution.md) · [原稿K／R／M协议](matching.md#11-覆盖状态) · [原验证记录](matching.md#7-核验记录与证据边界)

K的L5“核（源码）”不保证本卡已运行；25–27的ML路径尤其只有源码证据。R为未接入复现设计，M为既有组件与未实现扩展混合。历史37项回归及8/8冒烟不覆盖全部51卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| matching.md:01 | [匹配滤波](matching.md#01-匹配滤波) | 检测算法 | 核／核／核／核／部（复现设计） | 无独立检测器；未运行，噪声模型有前提 |
| matching.md:02 | [SAD](matching.md#02-sad绝对差和) | 相似度量 | 核／核／核／核／部（复现设计） | 本地NCC接口无SAD模式；未运行 |
| matching.md:03 | [SSD](matching.md#03-ssd平方差和) | 相似度量 | 核／核／核／核／部（复现设计） | 无独立匹配入口；未运行 |
| matching.md:04 | [原始互相关](matching.md#04-原始互相关) | 相似度量与组件 | 核／核／核／核／部（混合） | ZNCC分子使用相关；独立相关检测未运行 |
| matching.md:05 | [未去均值NCC](matching.md#05-ncc未去均值的归一化相关) | 相似度量 | 核／核／核／核／部（复现设计） | 本地所谓NCC主要为ZNCC；本卡未接入 |
| matching.md:06 | [ZNCC数学契约](matching.md#06-zncc共同数学契约) | 相似度量 | 核／核／核／核／核（源码） | 两实现及有限回归；反相等代数不冒充专项运行 |
| matching.md:07 | [平移ZNCC](matching.md#07-平移-znccfft-分子矩形积分图) | 具体实现 | 核／核／核／核／核（源码） | 已实现；旧接口兼容与正常样例，不保证极端域 |
| matching.md:08 | [姿态ZNCC](matching.md#08-姿态-zncc掩膜-fft-统计实现) | 具体实现 | 核／核／核／核／核（源码） | 已实现；历史37项有限回归，非真实工况验收 |
| matching.md:09 | [SSDA](matching.md#09-ssda逐次相似性检测) | 统计早停策略 | 核／核／核／核／部（复现设计） | 未实现／运行；1972全文获取缺口保留 |
| matching.md:10 | [相位相关](matching.md#10-相位相关) | 频域配准算法 | 核／核／核／核／部（复现设计） | 未接入；不自动支持任意旋转尺度 |
| matching.md:11 | [Chamfer](matching.md#11-chamfer-与距离变换匹配) | 轮廓匹配算法 | 核／核／核／核／部（复现设计） | 未实现／运行；距离与支持协议待验 |
| matching.md:12 | [稀疏梯度形状匹配](matching.md#12-稀疏梯度方向形状匹配) | 具体实现 | 核／核／核／核／核（源码） | 已实现、有限冒烟；启发式不保全局最优／低分拒识 |
| matching.md:13 | [前向LK](matching.md#13-lucaskanadelk前向局部配准) | 局部配准算法 | 核／核／核／核／部（复现设计） | 未接入；小位移、初值与纹理约束 |
| matching.md:14 | [ECC](matching.md#14-ecc相关系数连续参数配准) | 局部配准算法 | 核／核／核／核／部（复现设计） | 未实现／运行；无全局最优保证 |
| matching.md:15 | [RANSAC](matching.md#15-ransac) | 鲁棒估计策略 | 核／核／核／核／部（复现设计） | 未运行；骨架min_inliers不是已验证门限 |
| matching.md:16 | [SIFT](matching.md#16-sift) | 特征前端 | 核／核／核／核／部（复现设计） | 无本地运行记录；库版本及特征协议待验 |
| matching.md:17 | [FAST](matching.md#17-fast) | 特征检测器 | 核／核／核／核／部（复现设计） | 未运行；不包含描述子 |
| matching.md:18 | [BRIEF](matching.md#18-brief) | 特征描述子 | 核／核／核／核／部（复现设计） | 未运行；不能假定contrib已安装 |
| matching.md:19 | [ORB](matching.md#19-orb) | 特征前端 | 核／核／核／核／部（复现设计） | 未接入；WTA与距离定义随实现核验 |
| matching.md:20 | [SURF](matching.md#20-surf) | 特征前端 | 核／核／核／核／部（复现设计） | 书目／机制范围；未取得全文或运行 |
| matching.md:21 | [SuperPoint](matching.md#21-superpoint) | 学习特征前端 | 核／核／核／核／部（复现设计） | 仅路线／接口骨架；无训练或权重运行 |
| matching.md:22 | [SuperGlue](matching.md#22-superglue) | 学习匹配器 | 核／核／核／核／部（复现设计） | 未实现；未加载权重或运行 |
| matching.md:23 | [LoFTR](matching.md#23-loftr) | 学习匹配器 | 核／核／核／核／部（复现设计） | 仅路线；无本地粗细匹配链 |
| matching.md:24 | [LightGlue](matching.md#24-lightglue) | 学习匹配器 | 核／核／核／核／部（复现设计） | 未接入；当前作者默认值及动态延迟未实测 |
| matching.md:25 | [HOG](matching.md#25-hog-特征) | 特征描述组件 | 核／核／核／核／核（源码） | 本地调用OpenCV；缺sklearn阻碍模块导入，ML未运行 |
| matching.md:26 | [SVM／RBF SVC](matching.md#26-svmrbf-svc-与概率) | 分类器 | 核／核／核／核／核（源码） | 有源码；未运行sklearn、训练或现场概率校准 |
| matching.md:27 | [NCC＋HOG＋SVM](matching.md#27-ncchogsvm-候选重排序组合) | 教学组合 | 核／核／核／核／核（源码） | 只重排不按分类阈值拒识；未运行ML路径 |
| matching.md:28 | [角度／尺度网格](matching.md#28-角度尺度闭区间网格) | 搜索策略 | 核／核／核／核／核（源码） | 有实现与有限网格回归；off_grid基准未重跑 |
| matching.md:29 | [空间金字塔](matching.md#29-空间金字塔与候选传播) | 搜索加速策略 | 核／核／核／核／核（源码） | 有有限回归；近似可运行不等于普遍等价 |
| matching.md:30 | [全分辨率复核](matching.md#30-全分辨率完整复核) | 搜索复核策略 | 核／核／核／核／核（源码） | 默认完整评分；不是实时加速承诺 |
| matching.md:31 | [Top-K](matching.md#31-top-k候选容量) | 容量截断策略 | 核／核／核／核／核（源码） | 有实现；可能截断真目标和竞争峰 |
| matching.md:32 | [中心距离NMS](matching.md#32-中心距离-nms) | 候选抑制策略 | 核／核／核／核／核（源码） | 有实现及有限回归；不是IoU NMS或几何共识 |
| matching.md:33 | [阈值／分差／拒识](matching.md#33-分数阈值空间分差与多目标拒识) | 决策策略 | 核／核／核／核／核（源码） | 只限新接口；有限歧义测试，非产线阈值校准 |
| matching.md:34 | [PSR](matching.md#34-psr峰旁瓣比) | 响应诊断指标 | 核／核／核／核／部（复现设计） | 未实现／专项运行；空间分差不是PSR |
| matching.md:35 | [最近邻与互近邻](matching.md#35-最近邻与互近邻) | 对应搜索策略 | 核／核／核／核／部（复现设计） | 骨架仅提名；未执行对应链 |
| matching.md:36 | [ratio test](matching.md#36-ratio-test) | 歧义过滤策略 | 核／核／核／核／部（复现设计） | 未运行；距离及第二近邻契约待验 |
| matching.md:37 | [噪声白化](matching.md#37-噪声白化) | 统计变换组件 | 核／核／核／核／部（复现设计） | 无协方差估计或白化链 |
| matching.md:38 | [局部上采样DFT](matching.md#38-局部上采样-dft-亚像素配准) | 频域精化算法 | 核／核／核／核／部（复现设计） | 未实现；已有抛物线不等于局部DFT |
| matching.md:39 | [幅度谱＋对数极坐标](matching.md#39-傅里叶幅度对数极坐标旋转尺度估计) | 配准工程组合 | 核／核／核／核／部（复现设计） | 未实现／运行；姿态枚举不等于该扩展 |
| matching.md:40 | [逆组合](matching.md#40-逆组合inverse-compositional) | 局部求解策略 | 核／核／核／核／部（复现设计） | 未实现；固定支持／权重及可组合warp有前提 |
| matching.md:41 | [Homographic Adaptation](matching.md#41-homographic-adaptation) | 训练适应策略 | 核／核／核／核／部（复现设计） | 无训练／权重；不是RANSAC或推理必需步骤 |
| matching.md:42 | [掩膜与归一化重采样](matching.md#42-二值掩膜与归一化重采样) | 支持域组件 | 核／核／核／核／核（源码） | 有实现及有限掩膜回归；非软权重NCC |
| matching.md:43 | [FFT相关与缓存](matching.md#43-fft-相关补边和缓存) | 计算组件 | 核／核／核／核／核（源码） | 使用SciPy；未自写FFT、GPU或跨姿态缓存 |
| matching.md:44 | [积分图与局部矩](matching.md#44-积分图与局部矩统计) | 统计计算组件 | 核／核／核／核／核（源码） | 旧ZNCC使用；新mask版本不用矩形积分图 |
| matching.md:45 | [稳定ZNCC与直接兜底](matching.md#45-稳定-zncc-与分块直接兜底) | 数值稳定组件 | 核／核／核／核／核（源码） | 有极端输入有限回归；不恢复量化前纹理 |
| matching.md:46 | [相似变换与ROI坐标链](matching.md#46-相似变换包围画布与-roi-坐标链) | 坐标模型组件 | 核／核／核／核／核（源码） | 坐标有限回归；ROI组合仅代数，非标定／毫米链 |
| matching.md:47 | [三点抛物线精化](matching.md#47-三点抛物线平移精化) | 亚像素估计算法 | 核／核／核／核／核（源码） | 两NCC共用；只精化平移，不精化角度尺度 |
| matching.md:48 | [严格界与启发式早停](matching.md#48-非负部分和严格界与启发式早停) | 剪枝策略对照 | 核／核／核／核／部（混合） | 只有形状行级启发式；严格评分剪枝未实现 |
| matching.md:49 | [几何模型与DLT](matching.md#49-几何模型dlt-与重投影验证) | 模型估计与验证 | 核／核／核／核／部（混合） | 只构造相似H；点对DLT及估计链未运行 |
| matching.md:50 | [特征＋匹配＋几何＋精化](matching.md#50-特征前端匹配器ransac局部精化组合) | 族级工程组合 | 核／核／核／核／部（复现设计） | 接口抛NotImplementedError；无端到端链 |
| matching.md:51 | [一对一评测与计时](matching.md#51-一对一评测误差重复性与计时) | 指标与评测组件 | 核／核／核／核／核（源码） | 有评测源码；完整基准未重跑，非现场性能 |

## 标定坐标

[领域总览](../calibration_coordinates.md) · [原覆盖表](calibration.md#范围坐标与诚实边界) · [原验证缺口](calibration.md#验证范围与验收缺口)

十项均无专用实现、标定数据或物理验收链。历史确定性矩阵／坐标测试、外部EPnP和ChArUco板对象API冒烟，不是图像检测、真实标定或训练。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| calibration.md:C01 | [Zhang](calibration.md#c01-zhang-平面靶标标定) | 标定算法 | 部／部／部／部／部 | 未实现；已知单应约束与K恢复单元测试，非联合标定 |
| calibration.md:C02 | [Tsai–Lenz](calibration.md#c02-tsailenz-手眼标定) | 手眼估计算法 | 部／部／部／部／部 | 未实现；非交换AX=XB代数测试，非机器人验收 |
| calibration.md:C03 | [Kannala–Brandt](calibration.md#c03-kannalabrandtkb鱼眼模型) | 成像模型 | 部／部／部／部／部 | 未实现；角度多项式／反解测试，不覆盖全视场与全模型 |
| calibration.md:C04 | [EPnP](calibration.md#c04-epnp) | 位姿估计算法 | 部／部／部／部／部 | 未接入；外部求解器精确点冒烟，非图像匹配或外点测试 |
| calibration.md:C05 | [PoseNet](calibration.md#c05-posenet-场景重定位) | 学习重定位模型 | 部／部／部／部／部 | 无权重／训练；仅四元数指标与损失测试 |
| calibration.md:C06 | [针孔成像](calibration.md#c06-针孔成像) | 成像模型 | 部／部／部／部／部 | 无专用接口；前投影、射线、相机中心单元测试 |
| calibration.md:C07 | [径向／切向畸变](calibration.md#c07-径向与切向畸变) | 畸变模型 | 部／部／部／部／部 | 未接入；前投影与Newton往返，非图像remap验收 |
| calibration.md:C08 | [单应变换与估计](calibration.md#c08-单应变换与估计) | 几何模型与估计 | 部／部／部／部／部 | 估计链未实现；归一化DLT代数测试 |
| calibration.md:C09 | [三角化](calibration.md#c09-三角化) | 几何重建算法 | 部／部／部／部／部 | 未实现；双视图代数与深度传播，非真实双目精度 |
| calibration.md:C10 | [ChArUco](calibration.md#c10-charuco-观测与标定) | 观测前端与组合 | 部／部／部／部／部 | 未接入检测／标定；仅板角点ID布局API测试 |

## 学习组件

[原范围与五元协议](learning_components.md#范围状态与符号) · [分类装配双线总览](../classification_assembly_evolution.md) · [ML路线](../../ml/README.md) · [DL路线](../../dl/README.md)

逐卡保持“核／核／部／部／部”。原稿未执行torch或sklearn训练，源码存在不代表骨架可训练；参考算子不因外部API存在而视作已接入。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| learning_components.md:C01 | [StandardScaler](learning_components.md#c01-standardscaler) | 特征变换 | 核／核／部／部／部 | pipeline已调用；仅静态核对，未运行 |
| learning_components.md:C02 | [SVM／RBF SVC](learning_components.md#c02-支持向量机-svmrbf-svc) | 分类器 | 核／核／部／部／部 | 候选重排序示例，非像素分类／现场校准 |
| learning_components.md:C03 | [随机森林](learning_components.md#c03-随机森林-rf) | 分类器 | 核／核／部／部／部 | 未实现；参考路线，无空间一致性保证 |
| learning_components.md:C04 | [GBDT](learning_components.md#c04-梯度提升树-gbdt) | 分类器族基线 | 核／核／部／部／部 | 未实现；不等同全部XGBoost／LightGBM实现 |
| learning_components.md:C05 | [Conv2d](learning_components.md#c05-二维卷积-conv2d) | 网络算子 | 核／核／部／部／部 | EdgeNet／FocusNet已定义；未执行 |
| learning_components.md:C06 | [ReLU](learning_components.md#c06-relu) | 激活算子 | 核／核／部／部／部 | 本地已定义；未执行网络 |
| learning_components.md:C07 | [Sigmoid](learning_components.md#c07-sigmoid) | 概率链接算子 | 核／核／部／部／部 | fused BCE隐含链接，非独立前向已验证 |
| learning_components.md:C08 | [Softmax](learning_components.md#c08-softmax) | 概率链接算子 | 核／核／部／部／部 | 本地未显式实现；参考模型 |
| learning_components.md:C09 | [MaxPool2d](learning_components.md#c09-最大池化-maxpool2d) | 池化算子 | 核／核／部／部／部 | EdgeNet已定义；未执行 |
| learning_components.md:C10 | [平均／全局池化](learning_components.md#c10-平均池化与全局自适应平均池化) | 池化算子 | 核／核／部／部／部 | FocusNet用AdaptiveAvgPool2d(1)；非区域输出 |
| learning_components.md:C11 | [ConvTranspose2d](learning_components.md#c11-转置卷积-convtranspose2d) | 上采样算子 | 核／核／部／部／部 | EdgeNet已定义；不是卷积逆运算保证 |
| learning_components.md:C12 | [插值上采样](learning_components.md#c12-插值上采样-interpolation) | 重采样算子 | 核／核／部／部／部 | 分割未接入；标签与连续值协议待验 |
| learning_components.md:C13 | [相加与拼接跳连](learning_components.md#c13-跳跃连接相加与拼接) | 网络结构组件 | 核／核／部／部／部 | EdgeNet用相加；未用原始U-Net拼接 |
| learning_components.md:C14 | [Flatten与Linear](learning_components.md#c14-flatten-与-linear) | 网络读出组件 | 核／核／部／部／部 | FocusNet池化后使用；未执行 |
| learning_components.md:C15 | [BatchNorm](learning_components.md#c15-batchnorm) | 归一化算子 | 核／核／部／部／部 | 本地没有；原始资料未完整核验 |
| learning_components.md:C16 | [LayerNorm](learning_components.md#c16-layernorm) | 归一化算子 | 核／核／部／部／部 | 未实现；参考模型，原始资料未完整核验 |
| learning_components.md:C17 | [Dropout](learning_components.md#c17-dropout) | 随机正则算子 | 核／核／部／部／部 | 未实现；训练／推理切换待验 |
| learning_components.md:C18 | [多头注意力](learning_components.md#c18-缩放点积与多头注意力-attention) | 网络算子 | 核／核／部／部／部 | 未实现；论文检索片段不是全文／模型复现 |
| learning_components.md:C19 | [反向传播与自动微分](learning_components.md#c19-反向传播与自动微分) | 梯度计算组件 | 核／核／部／部／部 | FocusNet有backward；未执行反传验收 |
| learning_components.md:C20 | [Adam](learning_components.md#c20-adam) | 优化器 | 核／核／部／部／部 | 训练循环创建；未训练，不保证非凸全局最优 |
| learning_components.md:C21 | [多类交叉熵](learning_components.md#c21-多类交叉熵-cross-entropy) | 训练损失 | 核／核／部／部／部 | 未实现区域CE；输入契约与参考推导 |
| learning_components.md:C22 | [BCEWithLogits](learning_components.md#c22-bce-与-bcewithlogits) | 训练损失 | 核／核／部／部／部 | EdgeNet heat分支已写；不代表完整损失链可用 |
| learning_components.md:C23 | [Dice损失与指标](learning_components.md#c23-dice-损失与-dice-指标) | 损失与指标 | 核／核／部／部／部 | 无实现；soft训练目标与hard指标协议分开 |
| learning_components.md:C24 | [Focal Loss](learning_components.md#c24-focal-loss) | 训练损失 | 核／核／部／部／部 | 未实现；无优于CE的本站实验 |
| learning_components.md:C25 | [L1偏移监督](learning_components.md#c25-l1-绝对误差与边缘偏移监督) | 训练损失 | 核／核／部／部／部 | 有监督意图但布尔索引布局错误；未运行 |
| learning_components.md:C26 | [MSE](learning_components.md#c26-均方误差-mse) | 回归损失 | 核／核／部／部／部 | FocusNet有调用；形状、训练与部署未验证 |
| learning_components.md:C27 | [IoU／Jaccard](learning_components.md#c27-ioujaccard) | 区域指标 | 核／核／部／部／部 | tests/metrics无实现；非现有区域评测能力 |

## 测量与对焦

[测量领域](../README.md#05-边缘几何测量) · [对焦领域](../README.md#11-对焦成像质量) · [原覆盖表](measurement_focus.md#口径覆盖索引与演进) · [原验证记录](measurement_focus.md#本次验证未覆盖项与来源状态)

[测量双线总览](../measurement_evolution.md)连接 M01–M13 的边缘模型、亚像素估计与几何拟合，并说明 Devernay／Zernike／A31 的来源边界；[对焦双线总览](../autofocus_evolution.md)连接 F01–F14 的评分、峰模型、搜索及学习分支。七种清晰度评分仍仅 F01–F07；原始论文、比较研究及计量规范节点不另计卡，不因总览补证升级 L5。

历史8/8冒烟及明确微检查不覆盖所有故障输入或27卡全域。像素误差不等于毫米精度；RTImg全量重测、真实相机／电机闭环和网络训练仍无本次证据。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| measurement_focus.md:M01 | [卡尺](measurement_focus.md#m01-卡尺-caliper) | 测量组合实现 | 核／核／部／核／部 | 简化线探针；有限合成测试，边界索引风险保留 |
| measurement_focus.md:M02 | [梯度抛物线](measurement_focus.md#m02-三点抛物线梯度峰) | 亚像素估计 | 核／核／部／核／部 | 已实现；有限微检查，不保证全输入域 |
| measurement_focus.md:M03 | [erf剖面拟合](measurement_focus.md#m03-erf-积分高斯剖面拟合) | 成像模型拟合 | 核／核／部／核／部 | 已实现无约束版；中心／残差质量门不完整 |
| measurement_focus.md:M04 | [Devernay](measurement_focus.md#m04-devernay-二维边缘精化) | 亚像素估计 | 核／核／部／核／部 | 简化实现、有限合成测试；非外部完整管线复现 |
| measurement_focus.md:M05 | [Zernike矩边缘](measurement_focus.md#m05-zernike-矩边缘) | 矩模型算法 | 部／部／部／部／部 | 无实现；A31补偿与原文证据仍缺 |
| measurement_focus.md:M06 | [Canny](measurement_focus.md#m06-canny) | 边缘检测算法 | 核／核／部／部／部 | 完整管线未接入；不由名称推断可运行 |
| measurement_focus.md:M07 | [Sobel](measurement_focus.md#m07-sobel) | 微分算子 | 核／核／部／核／部 | 已调用库；不是完整边缘或测量系统 |
| measurement_focus.md:M08 | [Harris](measurement_focus.md#m08-harris-角点检测) | 角点检测器 | 核／核／部／核／部 | 库调用示例；角点测试有真值辅助初值限制 |
| measurement_focus.md:M09 | [手写角点精化](measurement_focus.md#m09-手写梯度正交角点精化) | 亚像素估计实现 | 核／核／部／核／部 | 已实现单次解；不是完整移窗迭代 |
| measurement_focus.md:M10 | [cornerSubPix](measurement_focus.md#m10-opencv-cornersubpix-精化) | 第三方封装 | 核／核／部／部／部 | 已封装；未完整核验库内部及全场景 |
| measurement_focus.md:M11 | [EdgeNet](measurement_focus.md#m11-edgenet-边缘与偏移网络) | 网络与损失骨架 | 核／部／部／核／部 | 偏移索引布局错误；前向／反传／训练未验证 |
| measurement_focus.md:M12 | [PCA二维直线](measurement_focus.md#m12-pca-二维正交直线拟合) | 几何拟合算法 | 核／核／部／核／部 | 已实现及RMS微检查；不覆盖圆／椭圆／RANSAC |
| measurement_focus.md:M13 | [Laplacian](measurement_focus.md#m13-laplacian-二阶算子) | 微分算子 | 核／核／部／核／部 | 评分内已调用；零交叉管线未实现 |
| measurement_focus.md:F01 | [Brenner](measurement_focus.md#f01-brenner) | 清晰度评分 | 核／核／部／核／部 | 已实现横向跨距差平方；有限微检查 |
| measurement_focus.md:F02 | [Tenengrad](measurement_focus.md#f02-tenengrad) | 清晰度评分 | 核／核／部／核／部 | 已实现无阈值版；非任意成像条件保证 |
| measurement_focus.md:F03 | [SMD](measurement_focus.md#f03-smd) | 清晰度评分 | 核／核／部／核／部 | 已实现双向邻差绝对值；有限微检查 |
| measurement_focus.md:F04 | [归一化方差](measurement_focus.md#f04-归一化方差-variance) | 清晰度评分 | 核／核／部／核／部 | 当前方差除均值；内容敏感性保留 |
| measurement_focus.md:F05 | [Laplacian energy](measurement_focus.md#f05-laplacian-energy) | 清晰度评分 | 核／核／部／核／部 | 当前二阶响应平方均值；不同于算子卡M13 |
| measurement_focus.md:F06 | [Vollath F4](measurement_focus.md#f06-vollath-f4-自相关差) | 清晰度评分变体 | 核／核／部／核／部 | 两非零滞后相关均值之差；不混同其他变体 |
| measurement_focus.md:F07 | [dct_sharpness](measurement_focus.md#f07-dct_sharpness-当前评分) | 清晰度评分实现 | 核／核／部／核／部 | 系数top-k L1；不是按频率选高频，名称偏差保留 |
| measurement_focus.md:F08 | [DCT-II](measurement_focus.md#f08-dct-ii-变换) | 基础变换 | 核／核／部／核／部 | 已调用库及常量块微检查；不是第八个评分 |
| measurement_focus.md:F09 | [单峰性比例](measurement_focus.md#f09-单峰性比例) | 曲线诊断指标 | 核／核／部／核／部 | 两份函数已实现；不是清晰度评分器 |
| measurement_focus.md:F10 | [三点高斯峰](measurement_focus.md#f10-三点高斯峰位) | 峰位精化算法 | 核／核／部／核／部 | 等距简式；不等距微检查揭示模型边界 |
| measurement_focus.md:F11 | [粗扫加缩步爬山](measurement_focus.md#f11-粗扫加缩步爬山) | 搜索策略 | 核／核／部／核／部 | 已实现；有限函数检查，未接电机 |
| measurement_focus.md:F12 | [黄金分割](measurement_focus.md#f12-黄金分割) | 搜索策略 | 核／核／部／核／部 | 已实现；依赖单峰区间，未接电机 |
| measurement_focus.md:F13 | [全扫描与数据源](measurement_focus.md#f13-全扫描与数据源) | 离散搜索基线 | 核／核／部／核／部 | 评估／数据源已实现；非真实闭环验收 |
| measurement_focus.md:F14 | [FocusNet](measurement_focus.md#f14-focusnet-有符号离焦回归) | 网络与训练骨架 | 核／部／部／核／部 | 未训练；标签广播、方向可辨识性及保存验证缺口 |

## 其余领域

[12领域目录](../README.md#12-领域目录) · [原覆盖表](remaining_domains.md#逐项覆盖索引) · [族级排除项](remaining_domains.md#泛称各列的族级边界) · [原验证记录](remaining_domains.md#验证范围与来源)

| 领域双线总览 | 已有卡片交接 | 新里程碑不能自动继承的覆盖 |
|---|---|---|
| [缺陷异常](../defect_evolution.md) | D01–D04；另有[G19 PaDiM](padim_root_polynomial.md#g19)、[HA01 GLCM](haralick.md#ha01)、[OC01单类SVM](one_class_svm.md#oc01)、[SA01 SSIM-AE](ssim_autoencoder.md#sa01)、[DR01 DRAEM](draem.md#dr01)、[PC01 PatchCore](patchcore.md#pc01) | D03不是GLCM，D04不是PaDiM／PatchCore；指标不替代AE，focal不替代双网；新卡均未实现，基准节点不转算法 |
| [分类装配](../classification_assembly_evolution.md) | A01＋[学习组件](#学习组件)；另有[HA01](haralick.md#ha01)、[RN01 ResNet](resnet.md#rn01)、[FR01](faster_rcnn.md#fr01)、[OM01](openmax.md#om01)、[SG01](iterative_scene_graph.md#sg01) | 07-01原分类链仍部分，不用教学分类器整体销G16；新卡均未实现，源码风险保留，分类／检测／开放集／关系推断不等于装配系统 |
| [字符编码](../ocr_codes_evolution.md) | O01、B01；另有[LG01](lenet_gtn.md#lg01)、[CT01](ctc.md#ct01)、[CR01](crnn.md#cr01)、[EA01](east.md#ea01)、[DB01](db_text_detection.md#db01) | LG01复合仍一卡；具体OCR卡均未实现生产接入，检测不等于识别，CT01不升级CR01；码制、质量、GS1规范分开，固定B集不等于完整标准实现 |
| [颜色外观](../color_appearance_evolution.md) | C01、D03／D04＋[P35颜色校正](preprocessing.md#p35)；另有[G34 RPCC](padim_root_polynomial.md#g34) | RPCC不由P35替代；Lab／Delta E76不覆盖CIEDE2000／CAM16；基本LBP不等于旋转不变多尺度版 |
| [三维重建测量](../3d_evolution.md) | G01–G02＋[标定坐标](#标定坐标)；另有[G36 FTP教学卡](ftp_hybrid_servo.md#g36) | FTP1983原机制仍缺证；SAD／ICP不覆盖SGM、学习对应或融合，不自动获得物理验收 |
| [跟踪运动视觉引导](../tracking_guidance_evolution.md) | T01–T04；另有[H01 Hungarian](hungarian.md#h01)、[G44伺服卡](ftp_hybrid_servo.md#g44) | H01无MOT接入；G44机制已有，证明／采样安全独立P0，组件不等于跟踪系统或控制闭环 |

本节原14卡全部未实现；算术微检查不等于端到端算法测试。原文件族级边界只作历史范围说明，当前具体补卡以本索引和覆盖矩阵为准，TrOCR及学习跟踪已有独立正文。Haralick原分类链仍部分、FTP仍缺原机制；G44机制已有但证明／采样安全独立P0。不能从原组件卡继承更高层级或实现状态。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| remaining_domains.md:D01 | [配准后绝对差分](remaining_domains.md#d01-配准后绝对差分) | 缺陷教学组合 | 部／部／部／核／部 | 未实现；亮度校正＋差分基线，非缺陷语义系统 |
| remaining_domains.md:D02 | [单高斯指数背景](remaining_domains.md#d02-单高斯指数背景模型) | 背景教学基线 | 部／部／部／核／部 | 未实现；方差算术检查，不是MOG2 |
| remaining_domains.md:D03 | [LBP纹理直方图](remaining_domains.md#d03-lbp-纹理直方图基线) | 纹理教学基线 | 部／部／部／核／部 | 未实现；基本8邻域LBP，不涵盖全部纹理方法 |
| remaining_domains.md:D04 | [收缩Mahalanobis](remaining_domains.md#d04-固定特征的收缩-mahalanobis-异常评分) | 异常评分基线 | 部／部／部／核／部 | 未实现；固定特征与收缩系数，非PatchCore／PaDiM |
| remaining_domains.md:A01 | [数量与相对几何](remaining_domains.md#a01-数量与相对几何装配规则) | 装配规则基线 | 部／部／部／核／部 | 未实现；UNKNOWN三态，定位／分割是前置 |
| remaining_domains.md:O01 | [切字加模板OCR](remaining_domains.md#o01-切字加模板相关-ocr) | OCR教学组合 | 部／部／部／核／部 | 未实现；固定字库，不覆盖多行／连笔／任意字体 |
| remaining_domains.md:B01 | [Code 128-B子集](remaining_domains.md#b01-code-128-b-读码与校验子集) | 解码教学基线 | 部／部／部／部／部 | 未实现；校验算术不等于完整码表／规范或GS1系统 |
| remaining_domains.md:C01 | [CIELAB与Delta E 76](remaining_domains.md#c01-cielab-与-delta-e-76) | 颜色表示与指标 | 部／部／部／核／部 | 未实现；同白点色差基线，不等于色度计／Delta E 2000 |
| remaining_domains.md:G01 | [双目SAD与三角测量](remaining_domains.md#g01-双目-sad-块匹配与三角测量) | 三维教学组合 | 部／部／部／核／部 | 未实现；深度算术检查，非真实双目毫米验收 |
| remaining_domains.md:G02 | [point-to-point ICP](remaining_domains.md#g02-刚性-point-to-point-icp) | 刚性配准算法 | 核／核／部／核／部 | 未实现；非全局配准、点到面或非刚性 |
| remaining_domains.md:T01 | [线性Kalman](remaining_domains.md#t01-线性-kalman-常速状态估计) | 状态估计算法 | 核／核／部／核／部 | 未实现；常速＋门控，不含EKF／UKF或Q/R自适应 |
| remaining_domains.md:T02 | [MOSSE型相关滤波](remaining_domains.md#t02-mosse-型自适应相关滤波) | 跟踪教学基线 | 核／核／部／核／部 | 未实现；频域比值算术检查，非视频跟踪验收 |
| remaining_domains.md:T03 | [局部LK光流](remaining_domains.md#t03-lucaskanade-局部光流) | 光流教学基线 | 部／部／部／核／部 | 未实现；局部小位移，不含稠密／深度光流 |
| remaining_domains.md:T04 | [门控后一对一指派](remaining_domains.md#t04-门控后一对一全局指派) | 数据关联策略 | 部／部／部／核／部 | 未实现；目标与例题，不是完整匈牙利推导或MOT |

## 对焦起源

[原覆盖与证据口径](focus_origins.md#范围证据与升级口径) · [对焦总览](../autofocus_evolution.md) · [历史缺口](../evolution_coverage.md#去重缺口清单)

三张卡各有五层正文及答5，层级针对历史目标全部为“部分”。现代算子、二手公式与原方法分开；原评分／系统证据仍未闭合，G09／G10／G11继续保留缺卡，不把现代F02／F01／F06的状态继承给历史目标。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| focus_origins.md:G09 | [Tenenbaum 1970](focus_origins.md#g09) | 历史梯度对焦／系统目标 | 部／部／部／部／部 | 答5；原评分范数／阈值／支持及主动流程未取；现代Sobel已有，原系统未复现 |
| focus_origins.md:G10 | [Brenner 1976](focus_origins.md#g10) | 历史显微评分／系统目标 | 部／部／部／部／部 | 答5；Santos p.272错年对应已闭合，最早成因未明；1976原评分／搜索仍缺，现代跨距差已有，SCANIT未复现 |
| focus_origins.md:G11 | [Vollath 1987](focus_origins.md#g11) | 历史相关评分目标 | 部／部／部／部／部 | 答5；二手1988求和式与F06均值差已区分；1987原F4/F5及二值化未复现 |

## FTP与混合切换伺服

[原覆盖与证据口径](ftp_hybrid_servo.md#范围证据与覆盖状态) · [三维总览](../3d_evolution.md) · [跟踪引导总览](../tracking_guidance_evolution.md)

两卡各有五层正文及答5。G36仍缺卡／P0：后续链和自推几何不替代1983原版。G44原守卫、初值、首次缩放及变增益原式已辨识，机制转已有；证明一致性与采样安全独立P0保留、不转P2，原文与离线R分开，无控制仿真、硬件在环或实机。G44的L1／L2采用原状态文字，不擅压成整层“核”。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| ftp_hybrid_servo.md:G36 | [FTP 1983](ftp_hybrid_servo.md#g36) | 主动重建教学参考 | 部／部／部／部／部 | 答5；频谱—相位—高度链有后续补证，1983原式未取；未实现／采集条纹或标准件 |
| ftp_hybrid_servo.md:G44 | [混合切换视觉伺服2007](ftp_hybrid_servo.md#g44) | 控制算法与离线参考 | 原守卫／初值已辨识、离散仍参考；原局部结论及限制已定位；部／部／部 | 答5；机制已有，pp.531–535页图及式(21)–(32)已有核读记录，非全篇证明背书；证明／采样安全独立P0，未实现，无仿真、硬件在环或实机 |

## 矩定位与傅里叶对焦

[原覆盖与销项口径](moment_fourier_focus.md#范围覆盖与销项口径) · [测量总览](../measurement_evolution.md) · [对焦总览](../autofocus_evolution.md)

G15／G43仅“缺少具体方法专卡”转已有，历史任务ID保留。原文关键节、条件代数或作者源码静态核对不等于实现；不把M05泛矩／A31或F14普通灰度CNN改名充当两卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| moment_fourier_focus.md:MF01 | [Renshaw–Christian 2020](moment_fourier_focus.md#mf01) | 斜坡／楔形矩定位算法 | 核／核／部／核／部 | 答5；对应G15，原矩约定及近似逆已核；未实现，未跑矩核／图像定位；条件积分核算非精度验收 |
| moment_fourier_focus.md:MF02 | [Pinkard 2019 FCFNN](moment_fourier_focus.md#mf02) | 光路与学习对焦组合 | 核／核／部／部／部 | 答5；对应G43，作者存档片段与固定源码静态核对；正式补充及发表快照未齐；未实现／训练／接设备 |

## PaDiM与根多项式校正

[原覆盖与证据口径](padim_root_polynomial.md#范围状态与证据口径) · [缺陷总览](../defect_evolution.md) · [颜色总览](../color_appearance_evolution.md)

G19／G34仅具体方法缺卡转已有。PaDiM章节日期采用出版社2021-03-05，ACM差异及正式版全文比对仍保留；RPCC已取得UEA接受稿，旧空入口不再代表完全未读。两卡均有答5、未实现，参数／测试设计不作运行结果。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| padim_root_polynomial.md:G19 | [PaDiM](padim_root_polynomial.md#g19) | 逐位深度高斯异常算法 | 核／核／部／部／部 | 答5；v1方法静态核对，共享选维／逐位完整协方差；无权重或基准实测；降低异常阈值扩大NG集合 |
| padim_root_polynomial.md:G34 | [Root-Polynomial RPCC](padim_root_polynomial.md#g34) | 齐次非线性颜色回归 | 核／核／部／部／部 | 答5；接受稿6／13／22基项及非负域；无实拍标定；伪代码检查sqrt(W)A有效秩并拒绝全零权重 |

## Hungarian指派

[原覆盖与销项口径](hungarian.md#范围状态与销项口径) · [跟踪总览](../tracking_guidance_evolution.md)

G13仅具体步骤与正确性缺卡转已有。现代势能增广内核、矩形固定基数与惩罚型拒配分别说明；1955第4–5节未全取，文档参考不等于独立产品内核或MOT接入。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| hungarian.md:H01 | [Hungarian势能增广](hungarian.md#h01) | 指派算法内核 | 核／核／部／核／部 | 答5；现代流程、原对偶与有限参考验证；非1955程序逐行复刻，无产品模块／真实视频验收 |

## 结构化森林边缘

[原版本与覆盖口径](structured_forests.md#范围版本与覆盖口径) · [测量总览](../measurement_evolution.md)

G14具体方法缺卡转已有；2013结构化标签训练、叶代表及重叠融合已展开，2014 sharpening及后续作者源码另列，普通RF不替代本卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| structured_forests.md:SF01 | [Structured Forests 2013](structured_forests.md#sf01) | 结构化学习边缘算法 | 核／核／部／核（限定代数）／部 | 答5；原文与固定源码静态核对，无训练器／权重，未编译调用MATLAB/MEX；V1.0包及整图验收仍缺 |

## Haralick共现纹理

[原状态与证据口径](haralick.md#范围状态与证据口径) · [G16承接边界](haralick.md#验证记录与-g16-承接边界) · [缺陷总览](../defect_evolution.md) · [分类总览](../classification_assembly_evolution.md)

HA01一张卡跨两域登记：06-01纹理表示已有，07-01原分类链部分，G16保留部分、不整体销项。L1按独立复核更正为部分；GLCM链已核不补齐分段线性训练及min-max完整决策，最近中心教学替代不是原法。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| haralick.md:HA01 | [Haralick 1973](haralick.md#ha01) | 共现纹理与分类方法 | 部（GLCM链核、原分类链部分）／核／部／部／部 | 答5；14特征版本与小矩阵记录非分类验收；原训练／min-max规则仍缺证，无专用提取器或分类实现，异常迁移另验 |

## One-Class SVM

[原任务与覆盖状态](one_class_svm.md#任务演进与覆盖状态) · [缺陷总览](../defect_evolution.md)

G17具体方法缺卡转已有。原点分离、原对偶／KKT、ρ恢复、等权PSD模型ν界及两坐标求解已展开；两类SVC及SVDD不替代本卡，统计域外不等于质量缺陷。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| one_class_svm.md:OC01 | [One-Class SVM](one_class_svm.md#oc01) | 正常支持域算法 | 核／核／部／核／部 | 答5；L4限等权PSD核模型，源码静态及数学核对非完整统计理论重证；无专用实现／训练模型或缺陷实测 |

## SSIM Autoencoder

[原任务与覆盖状态](ssim_autoencoder.md#任务演进与覆盖状态) · [G18承接边界](ssim_autoencoder.md#有限验证记录与-g18-销项依据) · [缺陷总览](../defect_evolution.md)

G18仅具体方法缺卡转已有、保留P2。128×128 AE、正常训练、局部结构目标、重叠重建平均和残差已展开；Q02指标不替代模型。VISAPP 2019书目与旧错误会议入口分开，论文literal常数与Wang平方常数冲突保留，第三方AE不冒充作者实现。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| ssim_autoencoder.md:SA01 | [SSIM Autoencoder](ssim_autoencoder.md#sa01) | 正常重建异常模型 | 核／核／部／核（限定SSIM代数）／部 | 答5；L4限固定非负归一化窗、正稳定常数的局部SSIM及梯度，不含网络收敛／检测保证；未实现，作者AE未锁，无权重／训练／框架前向；147断言为原卡历史算术 |

## DRAEM

[原任务与覆盖状态](draem.md#任务演进与覆盖状态) · [G20承接边界](draem.md#验证范围与-g20-销项建议) · [缺陷总览](../defect_evolution.md)

G20仅具体方法缺卡转已有、保留P2。Perlin合成监督、无横向跳连重建网、判别网、三项损失及无detach联合训练已展开；原文与固定源码的阈值、β方向、增强和学习率调度差异不抹平。raw像素异常图与21×21均值池化后max的整图评分分开。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| draem.md:DR01 | [DRAEM](draem.md#dr01) | 合成监督联合学习模型 | 核／核／部／部／部 | 答5；原文与固定源码静态核对，混合／损失／梯度推导不升级L4；未实现、未训练／加载权重／生成异常／推理，自动微分与真实泛化测试仍未执行 |

## PatchCore

[原任务与覆盖状态](patchcore.md#任务演进与覆盖状态) · [G21承接边界](patchcore.md#有限验证记录与-g21-承接边界) · [缺陷总览](../defect_evolution.md)

G21仅具体方法缺卡转已有、保留P2。局部对齐、共享bank、coreset、NN及原式(7)已展开；论文欧氏正指数重加权与固定作者代码的平方距离均值后max分开，后者无原式(7)。标准贪心条件近似界不直接授予作者近似选择器。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| patchcore.md:PC01 | [PatchCore](patchcore.md#pc01) | 局部记忆库异常算法 | 核／核／部／部／部 | 答5；有限度量推导不等于全JL理论／无条件近似保证；未实现，无权重／bank／骨干前向／FAISS或基准实测；历史仅核K=1开／关回退和K≥2非法b分支，不升级L5 |

## ResNet

[原任务与覆盖状态](resnet.md#任务演进与覆盖状态) · [G22承接边界](resnet.md#本次验证与-g22-承接边界) · [分类总览](../classification_assembly_evolution.md)

G22仅具体方法缺卡转已有、保留P2。原post-activation两类块、A/B/C旁路、18／34／50／101／152架构及监督训练已展开；原版、v1.5、预激活和权重V2不混同，五种架构只计RN01一卡。C13跳连不反向升级，闭集分类不等于装配系统。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| resnet.md:RN01 | [原版ResNet](resnet.md#rn01) | 残差分类模型 | 核／核／部／核（限定代数与静态结构）／部 | 答5；L4限链式法则、CE、BN局部导数、shape及参数/MAC，无全局收敛／泛化保证；未实现、未训练／加载权重／执行前反向；原训练和18/34作者逐元素实现仍缺 |

## Faster R-CNN

[原状态与版本](faster_rcnn.md#任务演进与覆盖状态) · [G23承接边界](faster_rcnn.md#本次验证与-g23-销项建议) · [分类总览](../classification_assembly_evolution.md)

G23具体方法缺卡转已有、保留P2。anchor标签／采样、RPN、RoIPool检测头与四步交替／近似联合齐备；2015原版、2016v3及2018作者Python重实现分开，不用S23或FPN／RoIAlign替代原版。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| faster_rcnn.md:FR01 | [Faster R-CNN](faster_rcnn.md#fr01) | 两级目标检测模型 | 核／核／部／部／部 | 答5；文献／固定源码静态与历史有限算术非网络运行；未实现，无前反向／权重／训练／AP；空GT／anchor、inclusive右端解码及原资产缺口保留 |

## OpenMax

[原状态](openmax.md#任务演进与覆盖状态) · [G24承接边界](openmax.md#验证记录与-g24-承接边界) · [分类总览](../classification_assembly_evolution.md)

G24具体方法缺卡转已有、保留P2。正确分类筛选、逐crop MAV、距离右尾、rank、带符号未知激活和拒识齐备；激活守恒不等于概率搬运，CDF单调不证明所有未知必拒。单作者证据链不作独立复现。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| openmax.md:OM01 | [OpenMax](openmax.md#om01) | 开放集校准与拒识算法 | 核／核／部／部／部 | 答5；未实现，未调用libMR／骨干／真实尾部拟合；纯欧氏训练推理尺度、CLI未传参及直接exp风险保留；序列化、许可、完整风险保证和未知类验收仍缺 |

## 迭代消息传递场景图

[原状态](iterative_scene_graph.md#任务演进与覆盖状态) · [G25承接边界](iterative_scene_graph.md#验证范围与-g25-销项依据) · [分类总览](../classification_assembly_evolution.md)

G25具体方法缺卡转已有、保留P2。对象／关系视觉、两类GRU、四门消息、末轮监督和三任务解码已展开；原式同步解释与固定代码先边后对象、双向边过滤、谓词背景mask及采样差异不抹平。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| iterative_scene_graph.md:SG01 | [迭代消息传递场景图](iterative_scene_graph.md#sg01) | 对象与有向关系联合模型 | 核／核／部／部／部 | 答5；未实现，未运行ROI／GRU／训练或权重；发布包依赖外部proposals，n_iter=0非基线，空图／单向稀疏边及旧TF对齐未验，不保证精确推断或收敛 |

## LeNet-5与GTN

[原状态与双子方法](lenet_gtn.md#任务演进与覆盖状态) · [G26承接边界](lenet_gtn.md#g26-承接结论与未完成项) · [OCR总览](../ocr_codes_evolution.md)

G26具体方法缺卡转已有、保留P2。原LeNet-5的C3部分连接、可训练下采样、缩放tanh／RBF及GTN图组合、正确／全路径目标与梯度分别有正文；两子方法各五层、共享答5，LG01仍只一卡。现代GTN不是1998原程序，连接数矛盾与正式版后段未比对保留。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| lenet_gtn.md:LG01 | [LeNet-5／GTN](lenet_gtn.md#lg01) | 字符网络与图全局训练复合卡 | LeNet-5：核／核／部／核（限定结构代数、损失及局部梯度）／部；GTN：核／核／部／核（限定有限DAG路径、占用率及组合梯度）／部 | 共享答5；未实现，原源码／码表／权重未齐，无完整网络、图组合／CNN梯度桥接或OCR实测；现代库DAG检查不全、remove梯度未实现，局部历史差分不补齐工程 |

## CTC

[原状态](ctc.md#范围状态与销项口径) · [原卡与实现边界](ctc.md#ct01) · [OCR总览](../ocr_codes_evolution.md)

G27具体方法缺卡转已有、保留P2。blank／折叠、前后向与后验、logits梯度、最小帧长、数值稳定及解码近似齐备；给定目标求概率不等于解码，有限宽prefix beam不冒称原prefix search或精确MAP。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| ctc.md:CT01 | [CTC](ctc.md#ct01) | 单调对齐训练与解码算法 | 核／核／部／核（限定模型）／部 | 答5；L4限前后向／后验／梯度、最小帧长及复杂度，无通用识别误差界；生产实现／OCR接入未实现；文档参考及历史穷举／差分／框架交叉非本次重跑，不升级CR01 |

## CRNN

[原状态](crnn.md#任务演进与覆盖状态) · [原卡与版本差异](crnn.md#cr01) · [CTC专卡](ctc.md#ct01) · [OCR总览](../ocr_codes_evolution.md)

G28具体方法缺卡转已有、保留P2。七卷积列映射、两层双向循环投影、CTC训练／转录与宽度协议明确；原文25帧、字面移植24帧、固定代码26帧及BN位置差异均保留。CT01是独立卡，其L4核验不继承给CR01。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| crnn.md:CR01 | [CRNN](crnn.md#cr01) | 卷积循环序列识别模型 | 核／核／部／部／部 | 答5；未实现，无网络前反向／训练／权重或OCR实测；原尺寸冲突、byte归一、loss归约与变宽协议分开；CTC长度不消除CNN边界／BN或反向RNN的padding污染 |

## EAST

[原状态](east.md#任务演进与覆盖状态) · [G29承接边界](east.md#核验记录与-g29-销项边界) · [OCR总览](../ocr_codes_evolution.md)

G29具体方法缺卡转已有、保留P2。融合、沿边收缩标签、RBOX／QUAD、几何损失、恢复及局部合并后NMS均已展开；作者账号的RBOX fork不等于原完整实验代码，QUAD资产缺口保留，论文与fork的骨干／损失／重评分分开。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| east.md:EA01 | [EAST](east.md#ea01) | 多方向文本几何检测模型 | 核／核／部／部／部 | 答5；未实现，未训练／加载权重／执行完整LANMS或检测基准；正负角恢复重排错配分数、末级NMS传polys非S及原地修改风险保留；原函数隔离核验为历史，非本次重跑 |

## DB文本检测

[原状态](db_text_detection.md#任务演进与覆盖状态) · [G30承接边界](db_text_detection.md#核验记录与-g30-销项边界) · [OCR总览](../ocr_codes_evolution.md)

G30具体方法缺卡转已有、保留P2。P／T／近似二值、shrink／border监督、联合损失、推理P的固定阈值与轮廓／评分／unclip齐备；论文BCE与2020代码Dice、阈值和外扩配置不混同，不归作现代DB++。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| db_text_detection.md:DB01 | [DB](db_text_detection.md#db01) | 可微二值化文本检测模型 | 核／核／部／部／部 | 答5；未实现，无模型前反向／训练／权重／检测实测；distance正常线段及共线分段错误、N1HW×NHW跨batch广播、零正OHEM／Dice、空多路径／int16风险保留；隔离检查仅历史证据 |

## TrOCR

[原状态与版本](trocr.md#任务演进与覆盖状态) · [G31承接边界](trocr.md#验证记录与-g31-销项依据) · [OCR总览](../ocr_codes_evolution.md)

G31具体方法缺卡转已有、保留P2。patch编码、因果解码／交叉注意力、两阶段OCR预训练／微调、teacher forcing及生成停止齐备；随机编号拒识是工程设计，不是论文已校准模块。v1的EOS起点与会议BOS、v5后期配置及固定库默认值分别保留，官方库不冒充作者原实验代码。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| trocr.md:TR01 | [TrOCR](trocr.md#tr01) | 预训练生成识别模型 | 核／核／部／部／部 | 答5；未实现，无网络／权重／训练／OCR实测；会议后半比对、原recipe、权重／tokenizer／processor、视觉padding与cache／beam、随机编号拒识验收仍缺；有限NumPy检查为历史，不升级L4／L5 |

## 1971 Retinex

[原状态与范围](retinex_1971.md#范围状态与证据口径) · [G32承接边界](retinex_1971.md#验证记录与-g32-销项依据p2-保留项) · [颜色总览](../color_appearance_evolution.md)

G32具体方法缺卡转已有、保留P2。独立复核卡内pp.6–10核读记录及机制链：后点／前点近邻比、近1阈值抑制、顺序累计且候选乘积超过1才reset、同一夹后状态读出并传播、独立路径同点算术平均；不以同态、SSR／MSR或后续版本替代。原数值阈值、路径分布／数目／长度及停止规格仍缺，参考log死区和访问平均不替作者补参；不因机制承接升级L1。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| retinex_1971.md:RE01 | [1971 Retinex](retinex_1971.md#re01) | 空间比值与路径明度方法 | 部／核／部／部／部 | 答5；L2核限原文关键段落及条件代数；未实现，二维路径／阈值／停止及重复访问政策、原程序／观察者数据、颜色输出与真实图像验收仍缺；pp.6–10成功核读和图8算术为历史，后续HTTP 465／p.4截断记录不抹去 |

## Gray-Edge

[原状态与版本](gray_edge.md#范围版本与覆盖口径) · [G33承接边界](gray_edge.md#未实现与-g33-交接建议) · [颜色总览](../color_appearance_evolution.md)

G33具体方法缺卡转已有、保留P2。逐通道高斯导数幅值、空间Minkowski聚合、RGB方向归一与原图对角校正齐备。作者二阶xy系数4不静默换为标准Hessian的2，mask实际0保留／1排除；原边带和饱和膨胀不等于完整核支持。Shades of Gray／Max-RGB为限定特例，不另计卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| gray_edge.md:GE01 | [Gray-Edge](gray_edge.md#ge01) | 颜色导数统计光源估计 | 核／核／部／核（限定条件代数）／部 | 答5；L4限空间恒定对角模型及共同统计条件，不保证光谱绝对值无条件交换；未实现，作者包仅固定哈希／静态阅读，8组算术为历史；MATLAB全链、mask／核支持、命名兼容、许可及真实估光未验 |

## FC4

[原状态](fc4.md#任务演进与覆盖状态) · [G35承接边界](fc4.md#g35-销项建议与保留边界) · [颜色总览](../color_appearance_evolution.md)

G35具体方法缺卡转已有、保留P2。共享卷积、局部估光／置信池化及全局角度监督已展开；原文显式4通道、代码默认有符号3通道及可选4通道softmax分开。局部归一不可从显式头删除，也不可加到默认隐式头冒充等价；置信不是校准概率。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| fc4.md:FC01 | [FC4](fc4.md#fc01) | 置信加权全卷积光源模型 | 核／核／部／核／部 | 答5；L4核限非零及可微域内归一化、角损失和头梯度；未实现，无前向／权重／训练或真实估光；pool8的3×3／2×2、通道协议、测试集选模、色板／fold、零向量与supp获取缺口保留；有限差分仅历史代数 |

## Curless–Levoy体积融合

[原状态与卡片](curless_levoy.md#cl01) · [三维总览](../3d_evolution.md#33-e03体积融合整合多视观测而非替代配准)

G37具体方法缺卡转已有、保留P2。视线距离、权重累积、空／未知与等值面已展开；VripPack 0.31是后续实现对照，补洞不作实测真值，一般三维变分证明未闭合。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| curless_levoy.md:CL01 | [Curless–Levoy体积融合](curless_levoy.md#cl01) | 带权视线距离融合算法 | 核／核／部／部／部 | 答5；未实现，原文／固定源码静态链不等于实扫融合；L4仍部分，标量检查仅历史；量化、扫描可见性、整网格提取及补洞标签验收仍缺 |

## SGM

[原状态与卡片](sgm.md#sm01) · [三维总览](../3d_evolution.md#35-e05sgm在局部窗口与二维全局优化之间取舍)

G38具体方法缺卡转已有、保留P2。HMI代价反馈、多路径归一递推、聚合与左右检查已展开；2005／2008与固定OpenCV SGBM分开，后者不实现MI，单链DP不证明二维全局最优。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| sgm.md:SM01 | [MI-SGM](sgm.md#sm01) | 立体代价与路径聚合算法 | 核／核／部／核（限定代数）／部 | 答5；未实现完整MI-SGM，未调用StereoSGBM或处理照片；有限穷举／边界检查仅历史；MI迭代、整数预算、左右检查分支差异与真实深度验收仍缺 |

## FPFH与SAC-IA

[原状态](fpfh_sacia.md#范围版本与覆盖状态) · [FPFH](fpfh_sacia.md#fp01-fpfh) · [SAC-IA](fpfh_sacia.md#fp01-sacia) · [三维总览](../3d_evolution.md#36-e06fpfh-与-sac-ia为局部配准寻找初值)

G39具体方法缺卡转已有、保留P2。SPFH／邻域聚合与间距采样／随机特征候选／刚体拟合／全云评分分别有五层正文；固定PCL变体明确，2009会议方法原文未取得，仅B＋A＋引言，不升级为T。FP01两子算法共享答5，仍只一卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| fpfh_sacia.md:FP01 | [FPFH／SAC-IA](fpfh_sacia.md#范围版本与覆盖状态) | 描述子与初始刚性对齐复合卡 | FPFH：部（固定源码链已核）／核（限定源码与官方教程）／部／部（条件代数与有限检查）／部；SAC-IA：部（固定源码链已核）／核（限定源码机制）／部／部（刚体拟合与概率条件）／部 | 共享答5；未实现，无PCL编译／随机粗配准或点云实测；2009原式、权重／Huber语义、固定库退化点对与状态污染、拒识验收仍缺；有限检查仅历史 |

## PSMNet

[原状态与卡片](psmnet.md#pm01) · [三维总览](../3d_evolution.md#37-e07psmnet学习双目对应与代价聚合)

G40具体方法缺卡转已有、保留P2。四尺度SPP、shift拼接体、basic／stacked hourglass、中间监督与期望视差齐备；2021作者快照非2018实验内部代码，几何零填充、GT mask和遮挡不混同。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| psmnet.md:PM01 | [PSMNet](psmnet.md#pm01) | 学习式双目立体模型 | 核／核／部／核（限定模型）／部 | 答5；未实现，无前反向／训练／权重或立体实测；GT／maxdisp、padding空宽、basic旧分支及align_corners风险保留；shape／索引／数值检查仅历史，不升级工程状态 |

## Monodepth2

[原状态与卡片](monodepth2.md#md01) · [三维总览 E08](../3d_evolution.md#38-e08monodepth2训练监督变化不等于尺度问题消失)

G41具体方法缺卡转已有、保留P2。训练／单图推理、最小重投影、identity掩膜、全分辨率多尺度与M／S／MS尺度协议已展开；GT中位数对齐不等于在线米制能力。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| monodepth2.md:MD01 | [Monodepth2](monodepth2.md#md01) | 自监督单图深度模型 | 核／核／部／核（限定针孔与局部可微域）／部 | 答5；未实现，无网络／训练／权重或深度实测；2026阅读快照非2019实验代码，尺度／内参／采样边界与域外验收仍缺；有限差分仅历史 |

## FCGF

[原状态与卡片](fcgf.md#fg01) · [三维总览 E09](../3d_evolution.md#39-e09fcgf学习点云特征而非取消刚体估计)

G42具体方法缺卡转已有、保留P2。量化代表点、稀疏ResUNet、hardest-contrastive与假负排除已展开；描述子不直接输出SE(3)，论文与2021后续源码差异分开。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| fcgf.md:FG01 | [FCGF](fcgf.md#fg01) | 稀疏三维几何特征模型 | 核／核／部／核（限定显式模型与固定索引局部梯度）／部 | 答5；未实现，未运行MinkowskiEngine／权重／训练或配准；坐标映射、批采样／假负、零范数／空集及真实点云验收仍缺；有限检查仅历史 |

## SORT

[原状态与卡片](sort.md#so01) · [跟踪总览 E06](../tracking_guidance_evolution.md#36-e06sort检测质量与简单时序组件的组合)

G45具体方法缺卡转已有、保留P2。七维框状态、预测、IoU指派与出生／确认／删除均有系统说明；warmup／输出谓词不等于永久确认，组件卡不替代该链。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| sort.md:SO01 | [SORT](sort.md#so01) | 检测驱动多目标系统 | 核／核／部／核（限定模型代数）／部 | 答5；未实现，无完整作者依赖栈／检测器／视频或MOT评测；IoU门控分支、NaN／Inf索引、生命周期及身份验收仍缺；有限分支检查仅历史 |

## SiamFC

[原状态与卡片](siamfc.md#si01) · [跟踪总览 E07](../tracking_guidance_evolution.md#37-e07siamfc离线相似性学习与在线相关互补)

G46具体方法缺卡转已有、保留P2。共享离线训练、平衡logistic、首帧固定模板、valid相关与有限尺度搜索已展开；2016 Workshop与2019阅读快照分开，不包含MOT或长期重识别。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| siamfc.md:SI01 | [SiamFC](siamfc.md#si01) | 学习式短期单目标模型 | 核／核／部／核（限定模型shape、梯度与坐标代数）／部 | 答5；未实现，无前向／权重／训练或视频评测；通道／BN、训练239与255、尺度／插值／零响应及视频验收仍缺；有限差分仅历史 |

## Deep SORT

[原状态与卡片](deep_sort.md#ds01) · [训练来源](deep_sort.md#source-8) · [跟踪总览 E08](../tracking_guidance_evolution.md#38-e08deep-sort给关联增加外观证据)

G47具体方法缺卡转已有、保留P2。八维xyah、Re-ID离线训练来源、运动门／gallery、年龄级联、受限IoU回退及生命周期已展开；2017方法、2018训练与2025跟踪快照分开，后续训练细节不追认为2017实验全复现。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| deep_sort.md:DS01 | [Deep SORT](deep_sort.md#ds01) | 学习外观辅助多目标关联系统 | 核／核／部／核（限定模型代数）／部 | 答5；未实现，无CNN／训练／冻结图或完整MOT评测；2017快照／权重血缘、2018训练论文后部、空特征／gallery、IoU绕门污染、截顶后拒配及视频身份验收仍缺；34项有限检查仅历史 |

## BYTE与ByteTrack

[原状态与卡片](bytetrack.md#bt01) · [依赖版本边界](bytetrack.md#source-9) · [跟踪总览 E09](../tracking_guidance_evolution.md#39-e09bytebytetrack低分候选的条件使用)

G48具体方法缺卡转已有、保留P2。BYTE高低分两阶段、ByteTrack的unconfirmed第三次指派、出生／失踪／重激活／移除／去重及分数据集配置已展开；策略与系统仍一卡，不混入BoT-SORT，低分不出生、不恢复旧Lost。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| bytetrack.md:BT01 | [BYTE／ByteTrack](bytetrack.md#bt01) | 低分条件关联策略与多目标系统 | 核／核／部／核（限定模型代数）／部 | 答5；未实现，无完整Python／LAP／Cython／检测器或MOT评测；原文截断／算法图片、历史依赖／tie-break、inclusive IoU、Removed残留／去重、坐标／空帧时钟及视频验收仍缺；26项有限检查仅历史 |

## KB与Zernike扩展

[原状态与证据](kb_zernike_extensions.md#范围证据与覆盖状态) · [标定总览](../calibration_coordinates.md) · [测量总览](../measurement_evolution.md)

KB01承接P23／P9／P6、四系数换算、球面单应标定与圆点质心观测，G01部分转已有／P2；坐标与全零因子初始化疑点不清除。ZM01仅补书目、摘要四参数与归一化对照，1993原算子及A31仍缺证，G02部分／P0保留。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| kb_zernike_extensions.md:KB01 | [KB全模型与标定](kb_zernike_extensions.md#kb01) | 相机模型与标定算法 | 部／核／部／部／部 | 答5；未实现，无P23标定或实拍验收；原坐标、因子初始化与参数规范未闭合，后续工具箱不替代2006原版，有限代数仅历史 |
| kb_zernike_extensions.md:ZM01 | [1993 Zernike原算子边界](kb_zernike_extensions.md#zm01) | 矩边缘历史目标卡 | 部／部／部／部／部 | 答5；未实现；VI1992六页三矩、归一化省略／积分制核已核，非1993作者稿；1993原核／方向／判边及A31归属仍缺，旧笔记错误归属已撤回 |

## 边缘演进扩展

[原状态与版本](edge_evolution_extensions.md#范围与核验口径) · [测量总览](../measurement_evolution.md)

G03／G05／G04部分转已有／P2；1995整数像素挂接与2017连续边链分开。HE01只补HED，旧S21继续承接U-Net，05-09不拆节点、不增U-Net卡。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| edge_evolution_extensions.md:DE95 | [Devernay 1995](edge_evolution_extensions.md#de95) | 亚像素NMS算法 | 核／核／部／核（限定三点代数与一维模型）／部 | 答5；原版未实现，本地仅简化版；NMS／采样／单阈值差异与平台风险保留，原程序未得，单项冒烟仅历史 |
| edge_evolution_extensions.md:DE17 | [IPOL Canny／Devernay 2017](edge_evolution_extensions.md#de17) | 修改版边缘与边链算法 | 核／核／部／核（限定峰位、链评分与固定支持复杂度）／部 | 答5；未接入／编译v1.0，附包回归未执行；容差NMS下η=5/6反例、构链顺序及AGPL许可边界保留，不保证一般精度 |
| edge_evolution_extensions.md:HE01 | [HED 2015](edge_evolution_extensions.md#he01) | 深监督边缘模型 | 核／核／部／核（限定二值损失与链式梯度）／部 | 答5；未实现，无Caffe前反向／权重／BSDS评测；2015架构与2023源码快照分开，不输出亚像素偏移，不升级EdgeNet |

## LBP与HOG扩展

[原状态与版本](lbp_hog_extensions.md#范围版本与覆盖协议) · [缺陷总览](../defect_evolution.md) · [分类总览](../classification_assembly_evolution.md) · [颜色总览](../color_appearance_evolution.md)

G06／G07／G08部分转已有／P2。1996整数邻域分布分类、2005线性滑窗检测、2002圆周多尺度编码分别登记；D03卡方异常基线和RBF候选重排均不替代原方法。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| lbp_hog_extensions.md:LB96 | [1996 LBP／C与分布分类](lbp_hog_extensions.md#lb96) | 纹理特征与分类算法 | 核／核／部／核（限定KL、列联表G与不变性代数）／部 | 答5；未实现历史管线；Set I单向KL与Set II留一G分开，位序／空组约定及异常迁移另验，有限算术仅历史 |
| lbp_hog_extensions.md:HG05 | [2005 HOG＋线性SVM检测](lbp_hog_extensions.md#hg05) | 特征、训练与检测组合 | 核／核／部／核（限定维数、投票、归一化与线性SVM模型代数）／部 | 答5；原检测器未实现，仅有不等价RBF候选重排；一次挖负例重训、扫描及NMS角色已写，NMS具体实现未核，无训练或实图验收 |
| lbp_hog_extensions.md:LB02 | [2002旋转不变多尺度LBP](lbp_hog_extensions.md#lb02) | 纹理编码与分类算法 | 核／核／部／核（限定循环群、VAR与似然代数）／部 | 答5；未实现历史管线；uniform／ri／riu2、VAR及两套分类协议分开，旋转／插值与零箱边界保留，无Outex实测 |

## SFF与Haralick扩展

[原状态与交接](sff_haralick_extensions.md#范围状态与交接意见) · [对焦总览](../autofocus_evolution.md) · [分类总览](../classification_assembly_evolution.md)

SF94承接1994局部SML—逐像素焦栈—三点峰—物理高度链，G12部分转已有／P2，不回填1989全部细节。HC73补原投票、重叠密度与外部式字面，但训练及外部原意仍缺，G16保留部分／P1，06-01不重复登记。

| 唯一键 | 卡片 | 类型 | L1／L2／L3／L4／L5 | 实现与验证边界 |
|---|---|---|---|---|
| sff_haralick_extensions.md:SF94 | [1994 SML与逐像素SFF](sff_haralick_extensions.md#sf94) | 局部焦栈深度算法 | 核／核／部／核（限定等距重写、尺度还原与条件敏感性）／部 | 答5；未实现SFF系统，查表离散／设备标定及真实高度未验；F10非等距／零分风险保留，1994不替1989，有限检查仅历史 |
| sff_haralick_extensions.md:HC73 | [1973 Haralick原分类链](sff_haralick_extensions.md#hc73) | 纹理分类历史目标卡 | 部／部／部／部／部 | 答5；未实现原训练器；1972 pp.100–110及具体选法缺证，≥字面／第一版位置已核；零宽及集合边界已声明，原意另待证，不强求勘误或唯一化才核字面 |

## 旧稿入口与迁移边界

`docs/algorithms/` 五份旧稿全部保留，只增补导航与链接修复说明，不把其旧编号、旧来源或历史运行记录直接复制为新卡状态。预处理及分割的承接以新稿中的映射表为准，其余文件按主题定位，不承诺编号一一相同。

| 保留旧稿 | 当前阅读入口 | 迁移边界 |
|---|---|---|
| [预处理旧稿](../algorithms/preprocessing_l1_l5.md) | [预处理与指标](#预处理与指标) | P编号不同；参照新稿承接表与P22颜色补充 |
| [分割旧稿](../algorithms/segmentation_l1_l5.md) | [分割区域](#分割区域)、[学习组件](#学习组件) | 原独有七项承接S25–S31；组件另篇 |
| [匹配旧稿](../algorithms/matching_l1_l5.md) | [定位匹配配准](#定位匹配配准) | 旧37卡与新51卡分拆不同；来源编号不共用 |
| [测量旧稿](../algorithms/measurement_l1_l5.md) | [测量与对焦](#测量与对焦) | 测量正文定位M01–M13，保留旧记录 |
| [对焦旧稿](../algorithms/autofocus_l1_l5.md) | [测量与对焦](#测量与对焦)、[其余领域](#其余领域) | 对焦定位F01–F14，跟踪另在T01–T04 |

三种缺失文件目标修为实际存在的新文件：`calibration_l1_l5.md` → [标定卡](calibration.md)，`learning_components_l1_l5.md` → [组件卡](learning_components.md)，`remaining_domains_l1_l5.md` → [其余领域卡](remaining_domains.md)（测量、对焦各引用一次，共四处）。旧匹配稿额外缺失的来源锚点与源码引用另以具名转接说明修复；不虚构旧来源表，也不升级为本次全文核验。

维护时先核对独立编号正文，再同步计数、唯一键、五层状态和边界；最后检查所有仓库Markdown中的本地文件、标题／显式锚点、源码行号和引用式链接。只改导航不能改变测试、实现、依赖或验证等级。
