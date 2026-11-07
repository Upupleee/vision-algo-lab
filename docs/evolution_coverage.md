# 12 领域主时间线与算法卡覆盖核对

[文档导航](README.md) · [仓库首页](../README.md) · [255 卡逐卡索引](algorithm_cards/README.md) · [学习规范](algorithm_learning_standard.md) · [六项原文获取与验收](source_requests.md)

核对日期：2026-11-06。对象为当前工作树的 **12份领域总览主时间线＋45份 algorithm_cards 正文**，不是仅核对目录或同名标题。本地复核计 **255卡＝245＋10＝206＋9＋4＋4＋8＋4＋4＋4＋2＋10**；本轮纳入四份扩展文件的KB01／ZM01、DE95／DE17／HE01、LB96／HG05／LB02、SF94／HC73。G01／G03／G04／G05／G06／G07／G08／G12八项部分转已有／P2，原版或完整系统仍未实现；G02部分／P0、G16部分／P1保留。KB01为部/核/部/部/部，ZM01及HC73全部分，其余七卡核/核/部/核（各自限定）/部、各答5；不升级原卡等级。HED补HE01，U-Net复用S21，05-09仍一节点。FP01两子算法各五层、共享答5且仍一卡，2009原文缺口保留P2；LG01仍一卡。来源访问深度沿用原卡记录，未重新联网、执行卡内算法检查器、训练或验收设备；历史微检查和源码隔离核验不是本次重跑，HA01的L1仍部分，RE01保持部/核/部/部/部。

本轮增量同步不新增卡：G44由缺卡转**机制已有**，守卫／初值／变增益原式已辨识；证明一致性与采样安全仍为[独立P0](#g44-independent-p0)，不转P2。G02采用ZM01的VI1992补证但不替代1993；G10仅关闭Santos错年对应；HC73仍缺1972训练正文。原式核读与有限检查均沿用原记录，不冒称本轮实测。前段“四文件10卡”指前次纳入轮次，当前255卡不变。

**2026-11-06原文请求清单轮次：**前两段的扩展卡纳入及G44机制同步均为历史轮次；本次依据四份卡片最新续查建立[六项原文获取／验收清单](source_requests.md)，不新增原方法证据或检索日志。G02／G09／G10／G11／G16／G36六项全部未关闭，255卡、113节点、83／2／4／24及方法分母89不变；G44证明一致性与采样安全独立P0保留，不计第七项。

## 判定与计数口径

- **节点单位**：主时间线一行计一个节点；复合行不拆行，预印本／正式版不重复计数。匹配含新增的 North、SSDA，共14行；OCR 两分支合计14行；颜色两分支合计13行。旁注、工程选型表、卡片内历史线、参考文献及未展开方向不再增节点。
- **已有**：卡片正文已解释该节点所选具体方法的任务、核心步骤与机制，并有五层正文和问答；不要求完成论文全复现，不表示五层全为“已核验”。可接受明确标出的求解器／工程变体，但不能丢掉节点的关键机制。
- **部分**：只有实质子模型、局部步骤或复合节点中的一个方法被卡片承接；逐行指出未承接内容。仅共享算子、前置、名称或泛族不够判“部分”。
- **缺卡**：没有该节点具体机制的独立卡，或现有卡与历史节点具体机制的承接关系尚未核实；即便存在相关组件、现代变体或未闭合教学专卡，仍保守判缺卡，不据此断言无正文。G09／G10／G11／G36已有五层教学正文及答5，但原机制证据未闭合，不转“部分”或“已有”；G44原机制已辨识，改列已有，证明／采样安全另账。对应卡列中的“对照／前置／组件”链接只供阅读，不能继承其五层或问答。
- **非算法**：本页按主角色排除标准、表示／色差公式／外貌模型规范、数据基准、比较综述和传感硬件节点；仍保留一行和缺口。不表示这些对象没有计算步骤。可执行的优化、训练策略、状态估计、特征提取及几何求解方法留在方法分母中；KB节点同时含投影与参数标定，KB01承接全模型与标定，C03仍仅四系数子模型，不能据扩展卡升级C03。ToF节点以锁相像素传感硬件为主，单列非算法。
- **唯一键**：节点用本页“领域号-序号”；卡用“文件名:卡号”。跨域节点保留，统计是领域节点覆盖，不是去重算法数。已有／部分／缺卡／非算法四态互斥；非算法有关联卡也不转入已有。

**五层与问答记法**：五元组固定为 L1步骤／L2机制／L3工程／L4推导／L5实现。`核`、`部`分别转录原卡“已核验”“部分”，不重新授予等级；`核源`为原卡“已核验（源码）”，`部设`为“部分（复现设计）”。`答5`表示逐层问题及答案均有正文，不是五层已验收。`局部`限定为所列卡的子方法；`—／答0`表示目标无独立卡及目标五层问答，非总览没有文字。非算法为“不适用”，有关联卡时另列其原状态。各卡均可由表中锚点直接检查。

**证据列**分开“总览原始来源记录”和“卡片实际边界”。`T`=曾读方法正文或节选（不保证全文）；`A`=摘要／作者公开说明；`B`=书目；`限`=原文未取得；`转引`=原卡沿用既有资料。证据链接直达总览或新增卡的来源记录；后续补证不抹去旧记录的单源、冲突与获取限制，同文多入口不算独立复现实验。实现列为原卡与总览记录的当前接入边界，不将历史微检查写成本次运行。

**缺口优先级**：P0=先消除版本、公式、标准或控制安全误读；P1=补关键方法／系统专卡；P2=完善已有卡的深度、复现与专项验证。优先级是本次编辑建议，不是原论文结论。缺卡与部分的去重任务见文末，非算法的资料任务单列。

## 覆盖汇总

| 领域 | 主节点数 | 已有 | 部分 | 缺卡 | 非算法 | 方法分母 |
|---|---:|---:|---:|---:|---:|---:|
| 01 预处理增强 | 6 | 6 | 0 | 0 | 0 | 6 |
| 02 标定坐标 | 6 | 6 | 0 | 0 | 0 | 6 |
| 03 分割区域 | 7 | 7 | 0 | 0 | 0 | 7 |
| 04 定位匹配配准 | 14 | 14 | 0 | 0 | 0 | 14 |
| 05 边缘几何测量 | 11 | 9 | 1 | 0 | 1 | 10 |
| 06 缺陷异常 | 8 | 7 | 0 | 0 | 1 | 7 |
| 07 分类装配 | 8 | 5 | 1 | 0 | 2 | 6 |
| 08 字符编码 | 14 | 6 | 0 | 0 | 8 | 6 |
| 09 颜色外观 | 13 | 5 | 0 | 0 | 8 | 5 |
| 10 三维重建测量 | 9 | 7 | 0 | 1 | 1 | 8 |
| 11 对焦成像质量 | 8 | 2 | 0 | 3 | 3 | 5 |
| 12 跟踪运动视觉引导 | 9 | 9 | 0 | 0 | 0 | 9 |
| 合计 | **113** | **83** | **2** | **4** | **24** | **89** |

核算：113=83＋2＋4＋24；89=113−24=83＋2＋4。方法节点已有卡匹配率为 **83/89=93.26%**；已有或部分承接为 **85/89=95.51%**，后者不是完整覆盖率。部分与缺卡合计6个节点，对应 **6个剩余覆盖任务：4个P0缺卡任务＋2个部分承接任务**。G01–G48历史ID全部保留，前41项已有加本轮G44共42项已有；前41项的P2边界保留，G44证明／采样安全仍独立P0。4项缺卡为G09／G10／G11／G36；**P1缺卡为0，部分仅G02的P0与G16的P1各一项**。G16仍为06-01已有、07-01部分，仅后者计剩余节点，不重复扣跨域去重；24个非算法节点不进方法分母。覆盖已有不等于实现、证明或现场验收完成。

## 01 预处理增强

源表：[代表性演进时间线](preprocessing.md#代表性演进时间线)，严格采用6行；SSIM／PSNR 在旁注中，不追加为主节点。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 01-01 双边滤波，1998 | 算法 | 空间与值域联合加权，减少跨边混合 | **已有**：[preprocessing:P05](algorithm_cards/preprocessing.md#p05-双边滤波)，双权重及归一化明确 | 部/部/部/部/部；答5 | [T/B](preprocessing.md#pre-source-1)；卡仅两点权重验算 | 未实现 | P2：整图边缘／噪声验证 |
| 01-02 NLM，2005 | 算法 | 按块相似性聚合远处重复样本 | **已有**：[preprocessing:P06](algorithm_cards/preprocessing.md#p06-非局部均值-nlm)，明确像素版、块距与搜索 | 部/部/部/部/部；答5 | [T/B](preprocessing.md#pre-source-2)；卡为块权重例 | 未实现 | P2：边界、距离与整图对照 |
| 01-03 BM3D，2007 | 算法 | 分组变换、硬阈值、协同维纳与聚合 | **已有**：[preprocessing:P07](algorithm_cards/preprocessing.md#p07-bm3d)，两阶段及 basic 的角色明确 | 部/部/部/部/部；答5 | [T/B](preprocessing.md#pre-source-3)；只验子步骤，许可另核 | 未实现 | P2：分组与整图复现 |
| 01-04 Guided Filter，2010／2013 | 算法 | 局部线性回归与均值统计保边 | **已有**：[preprocessing:P08](algorithm_cards/preprocessing.md#p08-引导滤波-guided-filter)，回归及系数平均齐备；不拿 P09 快速版替代 | 部/部/部/部/部；答5 | [单源版本记录](preprocessing.md#pre-source-4)；卡局部回归 | 未实现 | P2：全图与引导错位验证 |
| 01-05 DnCNN，2016预印本／2017 | 模型 | CNN 学噪声残差，配合 BN | **已有**：[preprocessing:P10](algorithm_cards/preprocessing.md#p10-dncnn)，残差目标、17层方案及推理相减 | 部/部/部/部/部；答5 | [版本确认](preprocessing.md#pre-source-5)；卡仅残差接口例 | 无模型权重、训练或推理 | P2：锁变体及训练复现 |
| 01-06 Noise2Noise，2018 | 训练策略 | 条件统计成立时以噪声目标替代干净目标 | **已有**：[preprocessing:P11](algorithm_cards/preprocessing.md#p11-noise2noise)，配对契约与风险展开明确 | 部/部/部/部/部；答5 | [原文及版本记录](preprocessing.md#pre-source-6)；卡为期望目标例 | 无配对采集或训练 | P2：相关噪声与配对验收 |

## 02 标定坐标

源表：[代表性演进时间线](calibration_coordinates.md#代表性演进时间线)，6行。总览“无独立测试”与卡内后续外部 EPnP 单元冒烟不是同一范围；后者仍不构成仓库接入。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 02-01 RANSAC，1981 | 鲁棒估计 | 最小样本假设与残差共识抗外点 | **已有**：[matching:15](algorithm_cards/matching.md#15-ransac)，跨文件有独立算法卡，不止标定中的提名 | 核/核/核/核/部设；答5 | [原始记录](calibration_coordinates.md#cal-source-1)；卡未跑估计链 | 未接入标定链 | P2：PnP／单应残差协议 |
| 02-02 Tsai–Lenz，1989 | 手眼算法 | AX=XB 分离旋转、平移 | **已有**：[calibration:C02](algorithm_cards/calibration.md#c02-tsailenz-手眼标定)，四元数求解与运动退化明确 | 部/部/部/部/部；答5 | [B，单源](calibration_coordinates.md#cal-source-2)；卡有非交换代数例 | 未实现，无机器人数据 | P2：同步及实物闭环 |
| 02-03 Zhang，1999／2000 | 标定算法 | 平面多姿态约束初始化，再联合优化 | **已有**：[calibration:C01](algorithm_cards/calibration.md#c01-zhang-平面靶标标定)，H→V→B→K 及优化流程 | 部/部/部/部/部；答5 | [版本分离](calibration_coordinates.md#cal-source-3)；卡仅给定 H 的 K 恢复 | 未实现完整标定 | P2：检测、畸变与留出图 |
| 02-04 Kannala–Brandt，2006 | 模型与标定 | 入射角投影覆盖广角／鱼眼 | **已有**：[kb_zernike_extensions:KB01](algorithm_cards/kb_zernike_extensions.md#kb01)补P23／P9／P6、四系数换算、球面H标定及圆点质心；[C03](algorithm_cards/calibration.md#c03-kannalabrandtkb鱼眼模型)仍仅前半空间子模型 | KB01 部/核/部/部/部；答5；C03全为部，不升级 | [原始记录](calibration_coordinates.md#cal-source-4)；[扩展原文关键节T](algorithm_cards/kb_zernike_extensions.md#source-1)、[作者稿版本](algorithm_cards/kb_zernike_extensions.md#source-2)；同文非独立实验 | 未实现标定／反投影接口，无实拍验收 | P2：G01部分转已有；坐标次序、全零因子初始化、参数规范及原代码仍缺 |
| 02-05 EPnP，2007／2009 | 位姿算法 | 控制点表达，固定维度核心求解 | **已有**：[calibration:C04](algorithm_cards/calibration.md#c04-epnp)，重心、距离约束、β候选与位姿选择 | 部/部/部/部/部；答5 | [T/B](calibration_coordinates.md#cal-source-5)；卡仅外部精确点冒烟 | 未接入，无图像位姿链 | P2：退化、外点与 β 初始化 |
| 02-06 PoseNet，2015 | 学习重定位 | 从 RGB 直接回归场景位姿 | **已有**：[calibration:C05](algorithm_cards/calibration.md#c05-posenet-场景重定位)，原始损失与改造分开，场景监督明确 | 部/部/部/部/部；答5 | [修订纠错记录](calibration_coordinates.md#cal-source-6)；卡仅损失／四元数例 | 无训练、权重 | P2：骨干与场景留出复现 |

## 03 分割区域

源表：[代表性演进时间线](segmentation_regions.md#代表性演进时间线)，7行；IoU／Dice 旁注不计节点。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 03-01 Otsu，1979 | 算法 | 最大类间方差自动定阈值 | **已有**：[segmentation:S04](algorithm_cards/segmentation.md#s04-otsu-自动全局阈值)，准则、遍历和并列规则明确 | 核/核/部/部/部；答5 | [B，单源](segmentation_regions.md#seg-source-1)；卡为静态对照 | 未接入 | P2：直方图与 ROI 验证 |
| 03-02 Vincent–Soille，1991 | 算法 | 排序与分层队列浸水 | **已有**：[segmentation:S14](algorithm_cards/segmentation.md#s14-vincentsoillevs浸水分水岭)，平台、新盆地、冲突明确；非泛分水岭 | 核/核/部/部/部；答5 | [原论文记录](segmentation_regions.md#seg-source-2)；卡原文转引未重访 | 未实现 | P2：平台与队列语义 |
| 03-03 GrabCut，2004 | 算法 | GMM 与图割交替减少交互 | **已有**：[segmentation:S18](algorithm_cards/segmentation.md#s18-grabcut)，四态提示、联合非凸与二值子问题 | 核/核/部/部/部；答5 | [原始记录](segmentation_regions.md#seg-source-3)；卡转引 | 未接入 | P2：提示及空成分测试 |
| 03-04 FCN，2014预印本／2015 | 模型 | 卷积化、上采样、跳层融合稠密预测 | **已有**：[segmentation:S20](algorithm_cards/segmentation.md#s20-fcn)，32s／16s／8s 与现代 FCN-ResNet 区分 | 核/核/部/部/部；答5 | [版本记录](segmentation_regions.md#seg-source-4)；卡转引 | 无网络、训练、权重 | P2：原架构与像素验收 |
| 03-05 U-Net，2015 | 模型 | 编解码与裁剪拼接兼顾上下文和定位 | **已有**：[segmentation:S21](algorithm_cards/segmentation.md#s21-u-net-与-edgenet-边界)，正文解释原型与边界权重，不只提 EdgeNet 名字 | 核/核/部/部/部；答5 | [版本记录](segmentation_regions.md#seg-source-5)；卡有原型／本地骨架对照 | 原版未实现；EdgeNet 非区域系统 | P2：原版训练与结构复现 |
| 03-06 Mask R-CNN，2017 | 模型 | 并行 mask 分支与 RoIAlign | **已有**：[segmentation:S23](algorithm_cards/segmentation.md#s23-mask-r-cnn)，候选、分类／框／mask 与损失契约 | 核/核/部/部/部；答5 | [版本记录](segmentation_regions.md#seg-source-6)；卡转引 | 未实现 | P2：实例训练与 AP |
| 03-07 原始 SAM，2023 | 模型 | 提示条件化与多候选掩膜 | **已有**：[segmentation:S24](algorithm_cards/segmentation.md#s24-原始-sam)，提示、编码缓存、解码和坐标恢复；不含后续 SAM | 核/核/部/部/部；答5 | [版本记录](segmentation_regions.md#seg-source-7)；卡源码／论文转引 | 无权重或接入 | P2：提示／候选与工业域验证 |

## 04 定位匹配配准

源表：[方法演进线：并行分支与连接点](matching_evolution.md#21-方法演进线并行分支与连接点)，14行，含 North 与 SSDA；正文分节顺序不同不影响计数。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 04-01 匹配滤波，1943／1963重印 | 检测算法 | 指定噪声下最大线性输出 SNR | **已有**：[matching:01](algorithm_cards/matching.md#01-匹配滤波)，核、延迟及条件证明明确 | 核/核/核/核/部设；答5 | [B/A，North全文限](matching_evolution.md#source-14)；现代代数非原式转录 | 无独立检测器 | P0：全文与最优性边界 |
| 04-02 SSDA，1972 | 评分策略 | 候选内部逐次累积，提前淘汰 | **已有**：[matching:09](algorithm_cards/matching.md#09-ssda逐次相似性检测)，统计早停与严格界分开 | 核/核/核/核/部设；答5 | [B/A＋Lewis转述，原文限](matching_evolution.md#source-15) | 未实现；行级 break 不算 | P0：固定版本与停止保证 |
| 04-03 相位相关，1975 | 配准算法 | 互功率谱相位恢复平移 | **已有**：[matching:10](algorithm_cards/matching.md#10-相位相关)，谱归一、回绕、符号明确 | 核/核/核/核/部设；答5 | [引文支持，原文限](matching_evolution.md#source-1) | 未接入 | P2：原文与频谱测试 |
| 04-04 Chamfer，1977 | 轮廓算法 | 边缘距离场查表替代严格重合 | **已有**：[matching:11](algorithm_cards/matching.md#11-chamfer-与距离变换匹配)，区分离散近似与精确 EDT | 核/核/核/核/部设；答5 | [T，官方原文](matching_evolution.md#source-2) | 未实现；方向点积不是此法 | P2：支持域与距离单位 |
| 04-05 Lucas–Kanade，1981 | 局部配准 | 梯度线性化迭代代替枚举 | **已有**：[matching:13](algorithm_cards/matching.md#13-lucaskanadelk前向局部配准)，J、更新和停止明确 | 核/核/核/核/部设；答5 | [B/A，版本区分](matching_evolution.md#source-3) | 未实现 | P2：初值与雅可比验证 |
| 04-06 RANSAC，1981 | 鲁棒估计 | 小样本共识剔除错误对应 | **已有**：[matching:15](algorithm_cards/matching.md#15-ransac)，与02-01同卡但任务不同 | 核/核/核/核/部设；答5 | [B/A，未逐页原文](matching_evolution.md#source-4) | 未实现 | P2：点对几何链 |
| 04-07 FFT-NCC，1995 | 具体计算方案 | FFT 分子与局部矩形统计联合加速 | **已有**：[matching:07](algorithm_cards/matching.md#07-平移-znccfft-分子矩形积分图)，确为中心化模板＋积分图；不是泛 FFT | 核/核/核/核/核源；答5 | [T，原版／扩展分离](matching_evolution.md#source-5)；本地源码与有限历史回归 | 已实现相关路线，非逐项论文复现 | P2：极端数值与性能 |
| 04-08 SIFT，1999／2004 | 特征前端 | 尺度、方向归一与局部描述 | **已有**：[matching:16](algorithm_cards/matching.md#16-sift)，DoG、定位、方向、描述齐备 | 核/核/核/核/部设；答5 | [作者原文／书目记录](matching_evolution.md#source-6) | 未接入 | P2：原始参数与运行 |
| 04-09 ECC，2008 | 配准算法 | 连续最大化归一化相关 | **已有**：[matching:14](algorithm_cards/matching.md#14-ecc相关系数连续参数配准)，含局部更新条件 | 核/核/核/核/部设；答5 | [T，HAL作者稿](matching_evolution.md#source-7) | 未实现 | P2：方向、支持与收敛 |
| 04-10 ORB，2011 | 特征前端 | 定向 FAST＋筛选旋转二值测试 | **已有**：[matching:19](algorithm_cards/matching.md#19-orb)，方向、rBRIEF、距离类型明确 | 核/核/核/核/部设；答5 | [B及独立引文](matching_evolution.md#source-8) | 未实现 | P2：WTA与多尺度协议 |
| 04-11 SuperPoint，2017／2018 Workshop | 学习前端 | 联合点／描述学习与单应适应 | **已有**：[matching:21](algorithm_cards/matching.md#21-superpoint)，训练流程、65通道和描述采样 | 核/核/核/核/部设；答5 | [版本记录](matching_evolution.md#source-9)；卡为模型设计 | 仅接口路线，无前向 | P2：权重与坐标契约 |
| 04-12 SuperGlue，2019／2020 | 学习匹配器 | 上下文注意力＋OT＋拒配 | **已有**：[matching:22](algorithm_cards/matching.md#22-superglue)，dustbin、Sinkhorn 与索引输出 | 核/核/核/核/部设；答5 | [原始记录](matching_evolution.md#source-10)；未跑权重 | 未实现 | P2：分配与空集测试 |
| 04-13 LoFTR，2021 | 学习匹配器 | 无独立点检测的粗细对应 | **已有**：[matching:23](algorithm_cards/matching.md#23-loftr)，粗分配与细窗口期望均展开 | 核/核/核/核/部设；答5 | [原始记录](matching_evolution.md#source-11)；无本地复现 | 只有研究路线 | P2：粗细尺度与域验证 |
| 04-14 LightGlue，2023 | 学习匹配器 | 学习置信驱动动态深度／剪点 | **已有**：[matching:24](algorithm_cards/matching.md#24-lightglue)，matchability与停止置信分开 | 核/核/核/核/部设；答5 | [原始记录](matching_evolution.md#source-12)；当前作者默认值未核 | 未接入 | P2：索引回填及尾延迟 |

## 05 边缘几何测量

源表：[代表性里程碑](measurement_evolution.md#3-方法演进线代表性里程碑)，11行。原总览的“无 RANSAC 独立卡”只适用于测量文件；跨文件核对可定位 matching:15，不能据域内未列就判全库缺卡。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 05-01 RANSAC，1981 | 鲁棒估计 | 对含粗差点集采样求共识 | **已有**：[matching:15](algorithm_cards/matching.md#15-ransac)；[M12](algorithm_cards/measurement_focus.md#m12-pca-二维正交直线拟合)仅非鲁棒对照 | 核/核/核/核/部设；答5；测量应用另验 | [A/B](measurement_evolution.md#me-source-1)；无直线鲁棒运行 | 未实现 | P2：法向残差与覆盖门 |
| 05-02 Canny，1986 | 边缘算法 | 平滑、法向 NMS 与滞后连接 | **已有**：[measurement_focus:M06](algorithm_cards/measurement_focus.md#m06-canny)，步骤齐备；原变分最优滤波推导仍缺 | 核/核/部/部/部；答5 | [T片段](measurement_evolution.md#me-source-2)；卡 API／可达模型 | 完整管线未接入 | P2：变分推导与完整流程 |
| 05-03 Harris–Stephens，1988 | 角点检测 | 自相关二次型识别双向可定位结构 | **已有**：[measurement_focus:M08](algorithm_cards/measurement_focus.md#m08-harris-角点检测)，结构张量及响应；精化不并入原算法 | 核/核/部/核/部；答5 | [T全文](measurement_evolution.md#me-source-3)；库调用与历史例 | 库调用示例；初值有真值辅助 | P2：独立初检召回 |
| 05-04 Ghosal–Mehrotra，1993 | 矩边缘算法 | 理想二维阶跃的矩参数估计 | **部分**：[ZM01](algorithm_cards/kb_zernike_extensions.md#zm01)补VI1992三矩、归一化省略、积分制核及判边；[M05](algorithm_cards/measurement_focus.md#m05-zernike-矩边缘)仍教学模型，1992不冒充1993 | ZM01 部/部/部/部/部；答5；M05亦全部分，仅局部 | [旧A/B及1993全文限](measurement_evolution.md#me-source-4)；[VI1992六页原文／页图](algorithm_cards/kb_zernike_extensions.md#source-10)、[1993获取限制](algorithm_cards/kb_zernike_extensions.md#source-11)；不记1993方法T | 无可调用函数，无原样本或实拍复跑 | P0：G02仍部分；1993版本、原核／方向／检测规则未闭合；[笔记撤回错误书目／A31口吻](research_notes.md#zernike-retraction-20261106)，不反证1993全文绝无A31 |
| 05-05 Devernay RR-2724，1995 | 亚像素算法 | 沿梯度方向插值 NMS 幅值峰 | **已有**：[edge_evolution_extensions:DE95](algorithm_cards/edge_evolution_extensions.md#de95)承接方向插值NMS、三点幅值峰及整数像素挂接滞后／连接；[M04](algorithm_cards/measurement_focus.md#m04-devernay-二维边缘精化)仍简化对照 | 核/核/部/核（限定三点代数与一维模型）/部；答5 | [原T片段](measurement_evolution.md#me-source-5)；[扩展原报告T](algorithm_cards/edge_evolution_extensions.md#source-2)；镜像非独立复现 | 本地简化版已实现，原版未实现；单项冒烟仅历史 | P2：G03部分转已有；原程序／许可、NMS取样、阈值／连接及平台边界仍待验 |
| 05-06 JCGM 100:2008／2010勘误 | 非算法·计量规范 | 模型与相关性下的不确定度预算 | **非算法**：无规范专卡；标定卡不是 GUM 复现 | 不适用；无规范问答 | [T封面／目录片段](measurement_evolution.md#me-source-6)；未逐条审核 | 无物理测量链 | P0：N01，规范与验收链 |
| 05-07 Hagara–Kulla erf，2011 | 模型拟合算法 | 四参数灰度过渡拟合代替理想阶跃 | **已有**：[measurement_focus:M03](algorithm_cards/measurement_focus.md#m03-erf-积分高斯剖面拟合)，四参数、目标、Jacobian与风险明确；求解器不同已声明 | 核/核/部/核/部；答5 | [T片段](measurement_evolution.md#me-source-7)；卡为同类模型非论文全复现 | curve_fit 无约束版已实现，非 fminsearch 原配置 | P2：参数质量门与论文对照 |
| 05-08 Structured Forests，2013 | 学习边缘算法 | 森林学习结构化边界标签 | **已有**：[structured_forests:SF01](algorithm_cards/structured_forests.md#sf01)，同区编码、节点局部离散化、叶代表及重叠融合齐备；普通RF不替代 | 核/核/部/核（限定代数）/部；答5 | [原A/B](measurement_evolution.md#me-source-8)；[新增T原文关键节](algorithm_cards/structured_forests.md#sf-source-1)、[后续固定源码](algorithm_cards/structured_forests.md#sf-source-3)；2014 sharpening不回填2013 | 未实现，无训练器／权重；未编译或调用MATLAB/MEX | P2：G14缺卡转已有；V1.0发布包、权重与整图验收仍缺 |
| 05-09 HED／U-Net，2015 | 复合学习节点 | HED 侧输出深监督；U-Net 编解码定位 | **已有**：[edge_evolution_extensions:HE01](algorithm_cards/edge_evolution_extensions.md#he01)补五侧输出、深监督、logit融合及训练／推理；[S21](algorithm_cards/segmentation.md#s21-u-net-与-edgenet-边界)继续承接U-Net；[M11](algorithm_cards/measurement_focus.md#m11-edgenet-边缘与偏移网络)仅骨架对照 | HE01 核/核/部/核（限定二值损失与链式梯度）/部；S21 核/核/部/部/部；各答5；M11原核/部/部/核/部不升级 | [原A/B，两篇分列](measurement_evolution.md#me-source-9)；[HED原文T](algorithm_cards/edge_evolution_extensions.md#source-6)、[后续作者源码](algorithm_cards/edge_evolution_extensions.md#source-7)；2023快照非2015实验 | HED／原U-Net未实现，无训练或BSDS评测；EdgeNet索引风险保留 | P2：G04部分转已有；资产、训练／评测及域验证仍缺，复合行仍一节点、不增U-Net卡 |
| 05-10 IPOL Canny/Devernay，2017 | 可复现算法实现 | 离散误差分析、修改实现与边链 | **已有**：[edge_evolution_extensions:DE17](algorithm_cards/edge_evolution_extensions.md#de17)承接轴向NMS／精化、先构链后链上滞后、有序输出；[M04](algorithm_cards/measurement_focus.md#m04-devernay-二维边缘精化)／[M06](algorithm_cards/measurement_focus.md#m06-canny)仍仅局部对照 | 核/核/部/核（限定峰位、链评分与固定支持复杂度）/部；答5 | [旧T片段，代码未审计](measurement_evolution.md#me-source-10)；[扩展论文T](algorithm_cards/edge_evolution_extensions.md#source-4)、[v1.0源码静态核对](algorithm_cards/edge_evolution_extensions.md#source-5)，不抹旧记录 | 外部实现未接入／编译；附包回归未执行，本地仍无序点 | P2：G05部分转已有；容差NMS下η=5/6、构链顺序、整图回归与AGPL许可边界保留 |
| 05-11 Renshaw–Christian，2020 | 矩定位算法 | 斜坡边缘／楔形条纹统一模型 | **已有**：[moment_fourier_focus:MF01](algorithm_cards/moment_fourier_focus.md#mf01)，斜坡／楔形、原始矩约定、宽度先验与近似反解明确；不沿用M05的归一化或A31归属 | 核/核/部/核/部；答5 | [原记录](measurement_evolution.md#me-source-11)；[新增T关键节／MathML](algorithm_cards/moment_fourier_focus.md#source-1)，仅条件代数，非整图实测 | 未实现；未运行矩核或图像定位 | P2：G15缺卡已转已有；宽度标定、极性／离散误差及复现仍缺 |

## 06 缺陷异常

源表：[有据代表节点](defect_evolution.md#3-方法演进线有据代表节点)，E01–E08，共8行。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 06-01 E01 Haralick，1973 | 纹理方法 | 灰度对共现表达空间统计 | **已有**：[haralick:HA01](algorithm_cards/haralick.md#ha01)，GLCM计数、14特征版本及方向汇总已承接；异常检测是另行工程迁移 | 部（GLCM链核／原分类链部分）/核/部/部/部；答5，不升级整卡L1 | [原T部分](defect_evolution.md#source-1)；[新增T关键／扫描限制](algorithm_cards/haralick.md#source-1)，勘误及原实现歧义保留 | 未实现专用提取器或异常系统 | P2：纹理表示已有；G16仍因07-01原分类链部分保留，异常迁移另验 |
| 06-02 E02 基本 LBP与分布分类，1996 | 纹理分类方法 | 3×3二值模式及特征分布判别 | **已有**：[lbp_hog_extensions:LB96](algorithm_cards/lbp_hog_extensions.md#lb96)补LBP/C联合分布、Set I单向KL、Set II留一G；[D03](algorithm_cards/remaining_domains.md#d03-lbp-纹理直方图基线)卡方异常基线不替代 | 核/核/部/核（限定KL、列联表G与不变性代数）/部；答5 | [原T部分](defect_evolution.md#source-2)；[扩展1996原文T](algorithm_cards/lbp_hog_extensions.md#source-1)，单源机制，1994版不混同 | 原分类管线与异常迁移均未实现 | P2：G06部分转已有；位序／空组约定、划分／零箱及真实异常阈值另验 |
| 06-03 E03 One-Class SVM，1999／2001 | 正常支持域算法 | 核空间与原点分离，估正常接受域 | **已有**：[one_class_svm:OC01](algorithm_cards/one_class_svm.md#oc01)，原对偶、KKT、ρ恢复、ν界及两坐标更新明确；两类SVC不替代 | 核/核/部/核/部；答5，L4限等权PSD核模型 | [原T部分](defect_evolution.md#source-3)；[新增原文](algorithm_cards/one_class_svm.md#source-1)、[固定源码静态核对](algorithm_cards/one_class_svm.md#source-4)，版本及边界分开 | 未实现，无已训练模型或缺陷实测 | P2：G17缺卡转已有；求解器边界、阈值校准与真实数据验收仍缺 |
| 06-04 E04 SSIM Autoencoder，2018／2019v3 | 重建模型 | 以局部结构目标训练与评估重建 | **已有**：[ssim_autoencoder:SA01](algorithm_cards/ssim_autoencoder.md#sa01)，具体AE、正常训练、重叠重建平均及局部残差明确；Q02指标不替代模型 | 核/核/部/核（限定SSIM代数）/部；答5 | [旧会议入口失败](defect_evolution.md#source-4)保留；[v3方法T](algorithm_cards/ssim_autoencoder.md#source-2)、[VISAPP书目](algorithm_cards/ssim_autoencoder.md#source-3)；常数口径冲突、作者AE未取得 | 未实现，无AE权重／训练或框架前向；第三方AE不冒充作者实现 | P2：G18缺卡转已有；作者实现锁定、常数消歧与真实缺陷验收仍缺 |
| 06-05 E05 MVTec AD，2019 | 非算法·基准 | 正常训练与真实局部缺陷测试分层 | **非算法**：无数据协议专卡，不以 D01–D04 算覆盖 | 不适用；无基准问答 | [T部分](defect_evolution.md#source-5)；未下载数据 | 无本地基准接入 | P1：N02，数据划分与指标 |
| 06-06 E06 PaDiM，2020／2021 | 异常算法 | 多层预训练嵌入的逐位置高斯 | **已有**：[padim_root_polynomial:G19](algorithm_cards/padim_root_polynomial.md#g19)，固定维抽样、多层嵌入、逐位完整协方差与评分明确；D04不替代本卡 | 核/核/部/部/部；答5 | [原日期冲突记录](defect_evolution.md#source-6)；[新增T v1方法](algorithm_cards/padim_root_polynomial.md#source-1)、[出版字段](algorithm_cards/padim_root_polynomial.md#source-2)，章节采用2021-03-05，ACM差异成因仍未解 | 未实现，无权重运行或基准实测；阈值须独立验证校准 | P2：G19缺卡已转已有；正式版比对、日期差异与部署验证仍缺 |
| 06-07 E07 DRAEM，2021 | 联合学习模型 | 合成异常监督连接重建与判别 | **已有**：[draem:DR01](algorithm_cards/draem.md#dr01)，Perlin模拟、双子网、MSE＋SSIM＋focal及联合反传齐备；整图池化与raw像素图分开 | 核/核/部/部/部；答5 | [原T至§3.3](defect_evolution.md#source-7)；[v2后续方法T](algorithm_cards/draem.md#source-2)、[固定训练源码](algorithm_cards/draem.md#source-7)；阈值／混合权重／调度差异保留 | 未实现，未训练／加载权重／生成异常／推理；梯度测试仍为设计 | P2：G20缺卡转已有；参数差异、梯度／边界及真实泛化验收仍缺 |
| 06-08 E08 PatchCore，2021／2022 | 记忆库算法 | 局部嵌入、coreset压缩与近邻评分 | **已有**：[patchcore:PC01](algorithm_cards/patchcore.md#pc01)，局部聚合／对齐、共享库、coreset、NN与原式(7)重加权齐备；D04不替代 | 核/核/部/部/部；答5 | [旧重加权未核记录](defect_evolution.md#source-8)；[v1/v2方法T](algorithm_cards/patchcore.md#source-2)、[固定评分源码](algorithm_cards/patchcore.md#source-6)；源码平方距离均值后max，无原式(7) | 未实现，无骨干前向、FAISS、bank或基准实测；指定分支有限检查非复现 | P2：G21缺卡转已有；评分双路径、采样／数值与真实缺陷验收仍缺 |

## 07 分类装配

源表：[有据代表节点](classification_assembly_evolution.md#3-方法演进线有据代表节点)，E01–E08，共8行；保留源表任务顺序。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 07-01 E01 Haralick，1973 | 纹理分类方法 | 共现统计连接有标签类别决策 | **部分**：[HC73](algorithm_cards/sff_haralick_extensions.md#hc73)补两两投票、min-max闭区间／密度及式(8)字面；[HA01](algorithm_cards/haralick.md#ha01)仍承接GLCM，原训练未闭合 | HC73 部/部/部/部/部；HA01 部（GLCM链核／原分类链部分）/核/部/部/部；各答5，不升级 | [原T部分](classification_assembly_evolution.md#source-1)；[原扫描T/I](algorithm_cards/sff_haralick_extensions.md#source-2)、[1972目录／导读](algorithm_cards/sff_haralick_extensions.md#source-5)；临时图片不归因用户、不增独立来源 | 未实现提取器／原训练器，无真实材质或装配验收 | P1：G16保留部分；缺1972 pp.100–110训练正文与具体选法；≥字面已核，作者原意另待证，零宽／集合边界已声明，不强求勘误与唯一化才承认字面；06-01不重复登记 |
| 07-02 E02 AND/OR装配计划图，1986 | 非算法·计划表示 | 子装配合取与可选分解路径 | **非算法**：对照 [remaining_domains:A01](algorithm_cards/remaining_domains.md#a01-数量与相对几何装配规则)是静态规则，非计划图 | 不适用；无该表示问答 | [T部分，1990版未核](classification_assembly_evolution.md#source-2) | 无规划／执行 | P1：N03，图表示与视觉证据分层 |
| 07-03 E03 HOG＋SVM，2005 | 检测组合 | 局部梯度、块归一、线性判别与困难负例 | **已有**：[lbp_hog_extensions:HG05](algorithm_cards/lbp_hog_extensions.md#hg05)补默认R-HOG、线性SVM、一次挖负例重训、全图多尺度扫描与NMS角色；[25](algorithm_cards/matching.md#25-hog-特征)／[26](algorithm_cards/matching.md#26-svmrbf-svc-与概率)／[27](algorithm_cards/matching.md#27-ncchogsvm-候选重排序组合)仍组件／RBF重排 | 核/核/部/核（限定维数、投票、归一化与线性SVM模型代数）/部；答5；旧三卡等级不变 | [原T部分](classification_assembly_evolution.md#source-3)；[扩展原文T](algorithm_cards/lbp_hog_extensions.md#source-2)；NMS具体实现未核 | 有候选重排示例，原检测器未实现；无训练或实图验收 | P2：G07部分转已有；原资产、NMS具体实现、负例／扫描配置及检测验收仍缺 |
| 07-04 E04 ResNet，2015／2016 | 分类模型 | 残差映射缓解深层优化退化 | **已有**：[resnet:RN01](algorithm_cards/resnet.md#rn01)，原post-activation块、A/B/C投影、五种架构与监督训练齐备；C13通用跳连不替代 | 核/核/部/核（限定代数与静态结构）/部；答5 | [原T部分](classification_assembly_evolution.md#source-4)；[v1正文T](algorithm_cards/resnet.md#source-1)、[固定部署源码](algorithm_cards/resnet.md#source-4)；原版／v1.5／预激活及权重V2分开 | 未实现，未训练／加载权重／执行前反向；参数算术与部署图静态核对非模型实测 | P2：G22缺卡转已有；原训练／预处理、18/34作者实现与跨框架验收仍缺 |
| 07-05 E05 Faster R-CNN，2015／2016v3 | 检测模型 | RPN与检测器共享卷积特征 | **已有**：[faster_rcnn:FR01](algorithm_cards/faster_rcnn.md#fr01)，anchor监督、RPN、RoIPool两级检测、四步交替／近似联合已展开；S23不替代 | 核/核/部/部/部；答5 | [原T部分](classification_assembly_evolution.md#source-5)；[v1方法T](algorithm_cards/faster_rcnn.md#source-1)、[v3方法T](algorithm_cards/faster_rcnn.md#source-2)、[固定框源码](algorithm_cards/faster_rcnn.md#source-9)；2018重实现非2015快照 | 未实现，无网络前反向／权重／训练或AP实测；空GT、空anchor及inclusive右端解码风险保留 | P2：G23缺卡转已有；原资产、坐标／采样边界及完整检测验收仍缺 |
| 07-06 E06 OpenMax，2016 | 开放集算法 | 激活距离尾部分布重分配未知质量 | **已有**：[openmax:OM01](algorithm_cards/openmax.md#om01)，正确分类筛选、逐crop MAV、距离右尾、rank、带符号未知激活与拒识齐备；C08不替代 | 核/核/部/部/部；答5 | [原算法1片段T](classification_assembly_evolution.md#source-6)；[方法／风险T](algorithm_cards/openmax.md#source-1)、[固定源码差异](algorithm_cards/openmax.md#source-7)；单作者链非独立复现 | 未实现，未调用libMR／骨干；纯欧氏训练推理尺度、CLI传参及直接exp风险保留 | P2：G24缺卡转已有；尾部拟合、序列化、许可与开放集风险验收仍缺 |
| 07-07 E07 迭代消息传递场景图，2017 | 关系模型 | 对象节点与有向关系边联合推断 | **已有**：[iterative_scene_graph:SG01](algorithm_cards/iterative_scene_graph.md#sg01)，双视觉、对象／关系GRU、四门消息、末轮监督及三任务解码明确；A01不替代 | 核/核/部/部/部；答5 | [原T部分](classification_assembly_evolution.md#source-7)；[v2方法T](algorithm_cards/iterative_scene_graph.md#source-2)、[固定图源码](algorithm_cards/iterative_scene_graph.md#source-6)；会议PDF截断与版本限制保留 | 未实现，未跑ROI／GRU或场景图；先边后对象、双向过滤、背景mask和外部proposal依赖保留 | P2：G25缺卡转已有；空／稀疏图、旧框架对齐、关系与装配验收仍缺 |
| 07-08 E08 IKEA ASM，2020／2021／2023修订 | 非算法·数据基准 | 多视角对象、姿态与动作评测 | **非算法**：无专卡；跟踪组件不覆盖基准 | 不适用；无基准问答 | [T部分，WACV非CVPR](classification_assembly_evolution.md#source-8) | 未接数据或流程评测 | P1：N04，时序／跨视角划分 |

## 08 字符编码

源表：[OCR主线](ocr_codes_evolution.md#3-历史演进线-aocr-识别与检测)6行＋[编码规范主线](ocr_codes_evolution.md#4-历史演进线-b编码质量标准与业务语法)8行。标准节点全部保留，不用 Code128-B 教学子集认领规范符合性。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 08-01 O-H1 LeNet-5／GTN，1998 | 识别与训练模型 | 字符卷积表示及多模块全局训练 | **已有**：[lenet_gtn:LG01](algorithm_cards/lenet_gtn.md#lg01)，LeNet部分连接／下采样／RBF及GTN图组合／全局路径目标均有原机制正文；复合仍一卡 | LeNet-5 核/核/部/核（限定结构代数、损失及局部梯度）/部；GTN 核/核/部/核（限定有限DAG路径、占用率及组合梯度）/部；共享答5 | [原T部分](ocr_codes_evolution.md#source-1)；[作者稿指定页T](algorithm_cards/lenet_gtn.md#source-1)、[现代GTN静态源码](algorithm_cards/lenet_gtn.md#source-6)；正式版后段未逐式比对、连接数矛盾保留 | 未实现，原1998资产未齐；现代库非原训练程序，DAG检查不全／remove梯度未实现风险保留 | P2：G26缺卡转已有；原网络、图组合／CNN梯度桥接及OCR验收仍缺 |
| 08-02 O-H2 CTC，2006 | 对齐训练算法 | blank、路径求和及动态规划 | **已有**：[ctc:CT01](algorithm_cards/ctc.md#ct01)，折叠、前后向边界／后验／梯度、最小帧长与解码近似齐备；O01不替代 | 核/核/部/核（限定模型）/部；答5 | [原T部分](ocr_codes_evolution.md#source-2)；[原论文T部分](algorithm_cards/ctc.md#source-1)、[固定官方源码](algorithm_cards/ctc.md#source-2)；有限检查为原卡历史记录 | 生产实现／OCR接入未实现；文档可执行参考非生产模块，未训练或测CER | P2：G27缺卡转已有；后端、批量／零长／数值协议与真实识别验收仍缺 |
| 08-03 O-H3 CRNN，2015／2017 | 序列模型 | CNN列特征＋双向RNN＋CTC | **已有**：[crnn:CR01](algorithm_cards/crnn.md#cr01)，七卷积列序列、两层BiLSTM投影、CTC与宽度协议明确；CT01独立卡不反向升级CR01 | 核/核/部/部/部；答5 | [原v1 T／期刊B/A](ocr_codes_evolution.md#source-3)；[原文](algorithm_cards/crnn.md#source-1)、[固定结构源码](algorithm_cards/crnn.md#source-4)；原文25帧／字面24帧／代码26帧及BN差异保留 | 未实现，无网络前反向／训练／权重；CNN与反向RNN的padding污染不由CTC长度自动消除 | P2：G28缺卡转已有；尺寸冲突、归一化、变宽与完整OCR验收仍缺 |
| 08-04 O-H4 EAST，2017 | 文本检测 | 直接预测多方向词／行几何 | **已有**：[east:EA01](algorithm_cards/east.md#ea01)，融合、收缩标签、RBOX／QUAD、损失、恢复与LANMS齐备；P36不替代 | 核/核/部/部/部；答5 | [旧A/B摘要记录](ocr_codes_evolution.md#source-4)；[v2方法T](algorithm_cards/east.md#source-1)、[固定fork风险](algorithm_cards/east.md#source-6)；作者账号fork非原完整实验代码 | 未实现；正负角恢复重排错配分数、末级NMS传polys非S及原地修改风险保留；隔离核验为历史，非本次运行 | P2：G29缺卡转已有；QUAD资产、索引／合并、空正损失与检测验收仍缺 |
| 08-05 O-H5 DB，2019／2020 | 文本检测 | 可微二值化进入训练 | **已有**：[db_text_detection:DB01](algorithm_cards/db_text_detection.md#db01)，P／T／近似二值、shrink／border监督、训练及固定阈值／轮廓／unclip恢复齐备；S01不替代 | 核/核/部/部/部；答5 | [旧A/B未读记录](ocr_codes_evolution.md#source-5)；[v2方法T](algorithm_cards/db_text_detection.md#source-1)、[固定border源码](algorithm_cards/db_text_detection.md#source-5)、[损失源码](algorithm_cards/db_text_detection.md#source-6)；论文／代码损失、阈值与unclip差异保留 | 未实现；distance正常线段及共线分段错误、N1HW×NHW跨batch广播风险保留；原函数隔离核验为历史 | P2：G30缺卡转已有；几何标签、布局／空监督、空多路径与检测验收仍缺 |
| 08-06 O-H6 TrOCR，2021／2023 | 生成识别模型 | 预训练图像／文本Transformer生成词元 | **已有**：[trocr:TR01](algorithm_cards/trocr.md#tr01)，patch编码、因果解码／cross-attention、预训练／微调、teacher forcing及生成停止齐备；C18不替代 | 核/核/部/部/部；答5 | [原A/B](ocr_codes_evolution.md#source-6)、[会议记录](ocr_codes_evolution.md#source-7)；[v1方法T](algorithm_cards/trocr.md#source-1)、[会议部分T及获取限制](algorithm_cards/trocr.md#source-2)、[固定库接口](algorithm_cards/trocr.md#source-8)；EOS／BOS及v5配置分开，非作者原代码 | 未实现，无网络／权重／训练或OCR实测；视觉padding、cache／beam与词元资产未验；有限NumPy检查为历史 | P2：G31缺卡转已有；会议后半、原recipe、权重／tokenizer／processor及随机编号拒识验收仍缺 |
| 08-07 B-H1 ISO/IEC15417:2007 | 非算法·Code128规范 | 编码、尺寸、参考解码契约 | **非算法**：关联 [remaining_domains:B01](algorithm_cards/remaining_domains.md#b01-code-128-b-读码与校验子集)仅固定B集，缺完整A/B/C控制 | 不适用；B01 部/部/部/部/部、答5仅子集 | [A/B，正文限](ocr_codes_evolution.md#source-8) | 无完整码表或解码器 | P0：N05，完整规范与子集声明 |
| 08-08 B-H2 ISO/IEC16022:2024 | 非算法·Data Matrix规范 | 二维符号、编码纠错与质量契约 | **非算法**：无具体卡，B01不是二维网格解码 | 不适用；无规范问答 | [A/B，正文限](ocr_codes_evolution.md#source-10) | 未实现 | P0：N06，正文与网格协议 |
| 08-09 B-H3 ISO/IEC18004:2024 | 非算法·QR规范 | 格式、纠错与参考解码契约 | **非算法**：无具体卡，不借Data Matrix参数 | 不适用；无规范问答 | [A/B，正文限；后续草案阶段未消歧](ocr_codes_evolution.md#source-11) | 未实现 | P0：N07，锁已发布版本 |
| 08-10 B-H4 ISO/IEC15415:2024 | 非算法·二维质量规范 | 符号属性测量与质量分级 | **非算法**：无卡；解码与分级不是同任务 | 不适用；无规范问答 | [A/B，正文限](ocr_codes_evolution.md#source-12) | 无质量评测 | P0：N08，测量条件／分级 |
| 08-11 B-H5 ISO/IEC15416:2025 | 非算法·线性码质量规范 | 可比属性与整体质量评定 | **非算法**：无卡；读码成功不算质量验收 | 不适用；无规范问答 | [A/B，正文限](ocr_codes_evolution.md#source-13) | 无质量评测 | P0：N09，替代版与设备 |
| 08-12 B-H6 ISO/IEC29158:2025 | 非算法·DPM质量规范 | 直接标记的照明与参数调整 | **非算法**：无卡，不是新纠错算法 | 不适用；无规范问答 | [A/B，正文限](ocr_codes_evolution.md#source-14) | 无DPM评测 | P0：N10，照明及测量协议 |
| 08-13 B-H7 15417/Amd1:2026 | 非算法·修订件 | 配置管理显式记录适用修订 | **非算法**：B01不覆盖修订内容，无专卡 | 不适用；无修订问答 | [B，仅存在／适用关系](ocr_codes_evolution.md#source-9)；技术变化未知 | 未实现修订件符合性 | P0：N11，不猜校验／码表变化 |
| 08-14 B-H8 GS1 Release26.0，2026 | 非算法·应用规范 | 载体、标识符、业务与验证规则分层 | **非算法**：B01校验不是GS1业务解析 | 不适用；无规范问答 | [T前置／目录，条款未读](ocr_codes_evolution.md#source-15) | 无GS1解析／追溯系统 | P0：N12，原载荷与业务语法 |

## 09 颜色外观

源表：[表示／公式／模型规范](color_appearance_evolution.md#3-历史演进线-a表示公式指标与模型规范)8行＋[算法分支](color_appearance_evolution.md#4-历史演进线-b校正恒常性与空间外观算法)5行。前8行主角色均非算法；Lab 有卡也不混入算法计数。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 09-01 C-H1 CIE1931观察者 | 非算法·色度表示 | 标准匹配函数建立三刺激基准 | **非算法**：前置关系 [remaining_domains:C01](algorithm_cards/remaining_domains.md#c01-cielab-与-delta-e-76)不含原观察者表格 | 不适用；无该表示问答 | [A，现代官方追溯，1931原文限](color_appearance_evolution.md#source-1) | 无光谱测色链 | P0：N13，观察者与参考数据 |
| 09-02 C-H2 Lab／Delta E76；2019规范 | 非算法·表示与指标 | 白点归一、非线性坐标与欧氏差异 | **非算法**：关联 [remaining_domains:C01](algorithm_cards/remaining_domains.md#c01-cielab-与-delta-e-76)已有具体公式，非标准全文 | 不适用；C01 部/部/部/核/部、答5仅教学定义 | [A/B，标准限](color_appearance_evolution.md#source-2)；卡代数例 | 未实现测色／校准 | P0：N14，域与标准条件 |
| 09-03 C-H3 CIEDE2000，2001 | 非算法·色差公式 | 明度／彩度／色相及交互修正 | **非算法**：C01不含完整Delta E00，无专卡 | 不适用；无公式五层问答 | [A/B](color_appearance_evolution.md#source-3)、[报告说明](color_appearance_evolution.md#source-4)，正文限 | 未实现 | P0：N15，完整辅助量与分支 |
| 09-04 C-H4 CIECAM02，2004报告 | 非算法·外貌模型规范 | XYZ与观察条件映射感知属性 | **非算法**：无专卡；Lab不能替代CAM | 不适用；无模型问答 | [A/B，正文限](color_appearance_evolution.md#source-6) | 未实现 | P1：N16，条件与内部方程 |
| 09-05 C-H5 Sharma等，2005 | 非算法·公式验证 | 补充色对揭示角度／符号分支错误 | **非算法**：无测试资料卡，不计新色差算法 | 不适用；无专项问答 | [T部分；测试数据未下载](color_appearance_evolution.md#source-5) | 未执行色对回归 | P0：N17，与N15共用测试而不合并节点 |
| 09-06 C-H6 CAM16／CAT16／UCS，2017 | 非算法·外貌模型 | 调整适应结构与均匀空间 | **非算法**：无专卡，三个对象不合称一个万能指标 | 不适用；无模型问答 | [A/B，全文限](color_appearance_evolution.md#source-7) | 未实现 | P1：N18，模型／变换／空间分层 |
| 09-07 C-H7 ISO/CIE11664-6:2022 | 非算法·公式标准维护 | 版本锁定、小修订与编辑调整 | **非算法**：无专卡，不是2022新色差算法 | 不适用；无规范问答 | [A与修订说明，正文限](color_appearance_evolution.md#source-4) | 无标准符合性 | P0：N19，条款与版本回归 |
| 09-08 C-H8 CIECAM16，2022报告 | 非算法·外貌模型报告 | CAM16形成CIE报告，替代159:2004 | **非算法**：无专卡，不由2017题名自动覆盖 | 不适用；无规范问答 | [A/B，正文限](color_appearance_evolution.md#source-8) | 未实现 | P1：N20，报告与论文差异 |
| 09-09 A-H1 Retinex，1971 | 空间明度方法 | 通过区域与边缘关系区分照明和反射 | **已有**：[retinex_1971:RE01](algorithm_cards/retinex_1971.md#re01)，近邻比值、近1阈值抑制、顺序累计且乘积超过1才reset、夹后状态传播及独立路径同点平均贯通；同态／SSR／MSR不替代 | 部/核/部/部/部；答5，L1不升级；L2核限原文关键段落及条件代数 | [原T扫描部分](color_appearance_evolution.md#source-9)；[卡内pp.6–10既有核读T](algorithm_cards/retinex_1971.md#source-1)、[独立复核与后续获取限制](algorithm_cards/retinex_1971.md#验证记录与-g32-销项依据p2-保留项)；HTTP 465／p.4截断不抹去，不冒称本次重读原件 | 未实现，无原程序或二维图像复现；对称log死区、有限路径与按访问平均为参考设计，图8算术仅历史 | P2：G32缺卡转已有；数值阈值／响应、二维路径分布／数量／长度、停止／重复访问、颜色输出及实图验收仍缺 |
| 09-10 A-H2 旋转不变多尺度LBP，2002 | 纹理算法 | uniform／旋转不变编码及多尺度分布 | **已有**：[lbp_hog_extensions:LB02](algorithm_cards/lbp_hog_extensions.md#lb02)补圆周采样、uniform／ri／riu2、VAR、多尺度似然及Brodatz／Outex两分类协议；[D03](algorithm_cards/remaining_domains.md#d03-lbp-纹理直方图基线)仍基本8邻域 | 核/核/部/核（限定循环群、VAR与似然代数）/部；答5 | [原T部分](color_appearance_evolution.md#source-10)；[扩展2002原文T](algorithm_cards/lbp_hog_extensions.md#source-3)，固定源码对照不等于原实验 | 原管线未实现，无Outex实测 | P2：G08部分转已有；与1996分开，插值／旋转、零箱／VAR量化及真实分类验收仍缺 |
| 09-11 A-H3 Gray-Edge，2007 | 光源估计 | 颜色导数统计替代均值统计 | **已有**：[gray_edge:GE01](algorithm_cards/gray_edge.md#ge01)，逐通道高斯导数幅值、空间Minkowski聚合、RGB归一及原图对角校正齐备；P34不替代 | 核/核/部/核（限定条件代数）/部；答5 | [原T部分](color_appearance_evolution.md#source-11)；[作者稿方法T](algorithm_cards/gray_edge.md#ge-source-1)、[固定哈希源码包](algorithm_cards/gray_edge.md#ge-source-3)；作者同源非独立复现，光谱绝对值交换有条件 | 未实现，未跑MATLAB／整图；二阶xy系数4非标准2、mask实际0保留／1排除、边带非完整核支持；8组算术仅历史 | P2：G33缺卡转已有；作者全链、核／mask／命名兼容、许可与真实角误差验收仍缺 |
| 09-12 A-H4 Root-Polynomial，2015 | 颜色回归 | 齐次根多项式保持曝光缩放结构 | **已有**：[padim_root_polynomial:G34](algorithm_cards/padim_root_polynomial.md#g34)，6／13／22基项、拟合及曝光等变明确；区分非负原域与signed-power现代延拓 | 核/核/部/部/部；答5 | [旧入口空返回](color_appearance_evolution.md#source-12)保留；[新增T UEA接受稿](algorithm_cards/padim_root_polynomial.md#source-4)已取得方法，不再列为完全未读 | 未实现，无实拍标定；伪代码检查加权有效矩阵秩，全零权重拒绝 | P2：G34缺卡已转已有；噪声、负值变体及留出颜色验收仍缺 |
| 09-13 A-H5 FC4，2017 | 学习光源估计 | 局部估计按学习置信加权聚合 | **已有**：[fc4:FC01](algorithm_cards/fc4.md#fc01)，共享卷积、局部方向／置信、全局归一与角度监督反传明确；三种头语义分开，C10不替代 | 核/核/部/核/部；答5，L4核限非零及可微域 | [旧A/B未读记录](color_appearance_evolution.md#source-13)；[方法T+B](algorithm_cards/fc4.md#source-1)、[固定作者源码](algorithm_cards/fc4.md#source-3)、[supp限](algorithm_cards/fc4.md#source-10)；同作者体系非独立实验 | 未实现，无前向／训练／权重或真实估光；默认有符号3通道、可选softmax、pool8尺寸、测试选模与零向量风险保留；差分仅历史代数 | P2：G35缺卡转已有；固定资产、通道／mask／fold、独立验证、跨相机与部署验收仍缺 |

## 10 三维重建测量

源表：[九个有据节点](3d_evolution.md#3-方法演进线九个有据节点)，E01–E09，共9行。硬件主导的固态调制式ToF单列非算法，其相位测距子方法仍可另建卡，但不擅自拆当前主节点。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 10-01 E01 FTP，1983 | 主动重建算法 | 投影条纹频带分离与相位形貌恢复 | **缺卡**（原机制待核）：已有教学专卡 [ftp_hybrid_servo:G36](algorithm_cards/ftp_hybrid_servo.md#g36)，后续文献支持频谱—相位—高度参考链，但1983具体原式／流程未对照 | G36 部/部/部/部/部；答5；教学正文不等于原版承接闭合 | [原记录B／限](3d_evolution.md#source-1)；[新增获取限制](algorithm_cards/ftp_hybrid_servo.md#source-2)、[后续T](algorithm_cards/ftp_hybrid_servo.md#source-3)；几何为自推 | 未实现，未采集条纹／标准件；只有伪代码与验收设计 | P0：G36保留，原版频带／相位／标定与参考方案逐项核对 |
| 10-02 E02 ICP，1992 | 刚性配准算法 | 近邻对应与刚体拟合交替 | **已有**：[remaining_domains:G02](algorithm_cards/remaining_domains.md#g02-刚性-point-to-point-icp)，point-to-point步骤与SVD明确；不含点到面／非刚性 | 核/核/部/核/部；答5 | [A/B，原文全文限](3d_evolution.md#source-2)；卡现代SVD教学非逐行原版 | 未实现 | P2：初值、重叠与SE(3)回归 |
| 10-03 E03 Curless–Levoy，1996 | 融合算法 | 带权有符号距离累积与表面提取 | **已有**：[curless_levoy:CL01](algorithm_cards/curless_levoy.md#cl01)，视线距离、截断／置信、加权累积、空／未知与等值面链齐备；ICP不替代融合 | 核/核/部/部/部；答5 | [原T部分及公式提取限制](3d_evolution.md#source-3)；[卡内原文T](algorithm_cards/curless_levoy.md#source-1)、[后续VripPack对照](algorithm_cards/curless_levoy.md#source-2)；非1996提交源码 | 未实现，无实扫融合／整网格提取；标量检查仅历史 | P2：G37缺卡转已有；一般三维变分证明、可见性／补洞标签、量化与实扫验收仍缺 |
| 10-04 E04 Lange–Seitz ToF，2001 | 非算法·传感硬件 | 锁相像素并行测幅值、偏置和相位 | **非算法**：无硬件／解调专卡；G01三角化不是ToF | 不适用；无硬件问答 | [A/B＋引言](3d_evolution.md#source-4) | 无驱动、采集或测距验收 | P0：N21，距离定义与解调链 |
| 10-05 E05 SGM，2005／2008 | 立体算法 | 多路径聚合近似二维平滑 | **已有**：[sgm:SM01](algorithm_cards/sgm.md#sm01)，HMI迭代代价、归一路径递推、聚合、亚像素与左右检查齐备；SAD／SGBM不替代MI | 核/核/部/核（限定代数）/部；答5 | [原T两版片段](3d_evolution.md#source-5)；[2005方法T](algorithm_cards/sgm.md#source-1)、[2008方法T](algorithm_cards/sgm.md#source-2)、[固定OpenCV源码](algorithm_cards/sgm.md#source-3)；SGBM未实现MI | 未实现完整MI-SGM图像流水线；有限穷举／边界检查仅历史 | P2：G38缺卡转已有；MI全链、整数预算、左右检查分支差异及真实深度验收仍缺 |
| 10-06 E06 FPFH与SAC-IA，2009 | 特征及粗配准 | 简化几何描述，建立局部对齐初值 | **已有**：[fpfh_sacia:FP01](algorithm_cards/fpfh_sacia.md#范围版本与覆盖状态)，[FPFH](algorithm_cards/fpfh_sacia.md#fp01-fpfh)的SPFH／邻域聚合与[SAC-IA](algorithm_cards/fpfh_sacia.md#fp01-sacia)的间距采样／随机候选／刚体拟合／全云评分分别展开；明确固定PCL变体，复合仍一卡 | FPFH 部（固定源码链已核）/核（限定源码与官方教程）/部/部（条件代数与有限检查）/部；SAC-IA 部（固定源码链已核）/核（限定源码机制）/部/部（刚体拟合与概率条件）/部；共享答5 | [原A/B＋引言](3d_evolution.md#source-6)；[2009原文获取限制](algorithm_cards/fpfh_sacia.md#source-1)、[固定FPFH源码](algorithm_cards/fpfh_sacia.md#source-3)、[固定SAC-IA源码](algorithm_cards/fpfh_sacia.md#source-4)；会议方法正文未取得，不记T | 未实现，无PCL编译／随机粗配准或真实点云验收；有限检查仅历史 | P2：G39缺卡转已有；2009原式、权重／Huber语义、源码变体与状态污染、退化及真实配准验收仍缺 |
| 10-07 E07 PSMNet，2018 | 学习立体模型 | 金字塔上下文与3D代价体正则 | **已有**：[psmnet:PM01](algorithm_cards/psmnet.md#pm01)，四尺度SPP、shift拼接体、basic／stacked hourglass、监督及期望视差齐备；G01不替代网络 | 核/核/部/核（限定模型）/部；答5 | [原A/B未读全文记录](3d_evolution.md#source-7)；[卡内原文T](algorithm_cards/psmnet.md#source-1)、[固定作者代码](algorithm_cards/psmnet.md#source-3)；2021快照非2018实验代码 | 未实现，无网络／权重／训练或立体实测；shape／索引／数值检查仅历史 | P2：G40缺卡转已有；GT／padding／maxdisp、旧框架分支、插值与域外深度验收仍缺 |
| 10-08 E08 Monodepth2，2018／2019 | 学习深度模型 | 最小重投影、自动掩膜与多尺度采样 | **已有**：[monodepth2:MD01](algorithm_cards/monodepth2.md#md01)，训练／推理分流、min重投影、identity掩膜、全分辨率采样及尺度协议齐备；三角化前置不替代 | 核/核/部/核（限定针孔与局部可微域）/部；答5 | [原T部分](3d_evolution.md#source-8)；[卡内会议方法T](algorithm_cards/monodepth2.md#source-1)、[固定作者代码](algorithm_cards/monodepth2.md#source-4)；沿用原卡记录，未重访 | 未实现，无网络／权重／训练或深度实测；有限差分仅历史 | P2：G41缺卡转已有；尺度／内参、采样边界、分支差异与域外验收仍缺 |
| 10-09 E09 FCGF，2019 | 学习几何特征 | 全卷积点云特征与度量学习 | **已有**：[fcgf:FG01](algorithm_cards/fcgf.md#fg01)，量化代表点、稀疏ResUNet、hardest负排除与对应接口齐备；二维前端不替代 | 核/核/部/核（限定显式模型与固定索引局部梯度）/部；答5 | [原A/B未读全文记录](3d_evolution.md#source-9)；[卡内方法T](algorithm_cards/fcgf.md#source-1)、[固定网络](algorithm_cards/fcgf.md#source-3)、[训练源码](algorithm_cards/fcgf.md#source-4)；后续快照非2019实验提交 | 未实现，未运行MinkowskiEngine／权重／训练或点云配准；有限检查仅历史 | P2：G42缺卡转已有；稀疏坐标、采样／假负、零范数／空集及真实配准验收仍缺 |

## 11 对焦成像质量

源表：[代表性里程碑](autofocus_evolution.md#3-方法演进线代表性里程碑)，8行。1985、1988、2013三行按比较／分析研究主角色列非算法；1988摘要虽提新准则，未取得可识别的完整准则，不擅自拆出算法。1970与1976为方法／系统复合节点；这两行及1987相关评分与现有现代变体的具体承接关系未核实，保守判缺卡，保留对照而不认定无关联。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 11-01 Tenenbaum，1970 | 梯度对焦／系统方法 | 梯度清晰度连接任务驱动相机调节 | **缺卡**（原机制待核）：已有 [focus_origins:G09](algorithm_cards/focus_origins.md#g09)正文；[F02](algorithm_cards/measurement_focus.md#f02-tenengrad)仍仅现代无阈值Sobel对照，原范数／阈值／支持与主动流程未闭合 | G09 部/部/部/部/部；答5；F02 核/核/部/核/部、答5仅对照，不继承 | [原记录](autofocus_evolution.md#af-source-1)；[新增T封面／目录／图题](algorithm_cards/focus_origins.md#source-1)，未取得原评分方法页 | 现代评分已实现；原主动系统未复现 | P0：G09保留，VI.3原评分及VI.6搜索／停止 |
| 11-02 Brenner等，1976 | 显微评分／系统方法 | 自动采集质量控制与跨距差评分关联 | **缺卡**（原机制待核）：已有 [G10](algorithm_cards/focus_origins.md#g10)；[F01](algorithm_cards/measurement_focus.md#f01-brenner)仅现代跨2像素差，1976评分／搜索未闭合 | G10 部/部/部/部/部；答5；F01 核/核/部/核/部、答5仅对照，不继承 | [原历史记录](autofocus_evolution.md#af-source-2)；[Santos p.272完整引文及错年对应已闭合](algorithm_cards/focus_origins.md#g10-citation-1971-closed)；最早误引成因未明，不再索取已得参考表 | 现代评分已实现，无论文系统 | P0：G10保留；缺1976评分、方向／跨度／门控／聚合与搜索原页，不以书目销机制项 |
| 11-03 Groen等，1985 | 非算法·比较研究 | 明确条件下比较既有评分 | **非算法**：关联 [F02](algorithm_cards/measurement_focus.md#f02-tenengrad)、[F04](algorithm_cards/measurement_focus.md#f04-归一化方差-variance)、[F05](algorithm_cards/measurement_focus.md#f05-laplacian-energy)仅本地变体 | 不适用；各卡 核/核/部/核/部、答5，不覆盖研究 | [A/B，实验正文未取](autofocus_evolution.md#af-source-3) | 有评分，未复现实验 | P1：N22，条件／函数定义对照 |
| 11-04 Vollath，1987 | 相关评分方法 | 非零滞后自相关构造清晰度准则 | **缺卡**（原机制待核）：已有 [focus_origins:G11](algorithm_cards/focus_origins.md#g11)正文；[F06](algorithm_cards/measurement_focus.md#f06-vollath-f4-自相关差)不同支持均值差与二手求和式已区分，但二手式标1988，不能回填1987原式 | G11 部/部/部/部/部；答5；F06 核/核/部/核/部、答5仅对照，不继承 | [原记录](autofocus_evolution.md#af-source-4)；[新增原摘要A／B](algorithm_cards/focus_origins.md#source-5)、[二手T](algorithm_cards/focus_origins.md#source-4)，1987方法仍限 | 本地变体已实现，原F4/F5及二值化未复现 | P0：G11保留，1987定义／支持／归一化／二值化 |
| 11-05 Vollath，1988 | 非算法·性质分析 | 场景、工作范围、噪声与唯一焦点分析 | **非算法**：关联 [measurement_focus:F09](algorithm_cards/measurement_focus.md#f09-单峰性比例)不是该研究或新准则 | 不适用；F09 核/核/部/核/部、答5仅诊断 | [B＋检索A，全文限](autofocus_evolution.md#af-source-5) | 未复现比较研究 | P1：N23，准则辨识与条件 |
| 11-06 Nayar／Nayar–Nakagawa SFF，1989／1994 | 深度估计算法 | SML局部评分＋高斯峰插值 | **已有**：[sff_haralick_extensions:SF94](algorithm_cards/sff_haralick_extensions.md#sf94)承接1994门控SML、逐像素焦栈、相邻三点／查表与台位移到高度；[F10](algorithm_cards/measurement_focus.md#f10-三点高斯峰位)／[F13](algorithm_cards/measurement_focus.md#f13-全扫描与数据源)仍子步骤 | 核/核/部/核（限定等距重写、尺度还原与条件敏感性）/部；答5 | [原1989封面B；1994 T部分](autofocus_evolution.md#af-source-6)；[扩展1994原文T/I](algorithm_cards/sff_haralick_extensions.md#source-1)，不回填1989全部细节 | 同类峰拟合已实现；SFF系统未实现，无真实高度验收 | P2：G12部分转已有；查表离散、逐点有效域、F10非等距／零分风险与设备标定仍缺 |
| 11-07 Pertuz等，2012在线／2013 | 非算法·综述比较 | 分类评分机理并比较噪声等因素 | **非算法**：关联 [F01](algorithm_cards/measurement_focus.md#f01-brenner)、[F02](algorithm_cards/measurement_focus.md#f02-tenengrad)、[F03](algorithm_cards/measurement_focus.md#f03-smd)、[F04](algorithm_cards/measurement_focus.md#f04-归一化方差-variance)、[F05](algorithm_cards/measurement_focus.md#f05-laplacian-energy)、[F06](algorithm_cards/measurement_focus.md#f06-vollath-f4-自相关差)、[F07](algorithm_cards/measurement_focus.md#f07-dct_sharpness-当前评分)为七个具体变体，非全方法集 | 不适用；七卡均 核/核/部/核/部、各答5 | [T部分，附录未全取](autofocus_evolution.md#af-source-7)；F07非高频平方能量 | 有七评分，无统一论文复现 | P0：N24，变体映射与DCT语义 |
| 11-08 Pinkard等 FCFNN，2019 | 光学＋学习对焦 | 离轴照明与傅里叶网络消除部分方向歧义 | **已有**：[moment_fourier_focus:MF02](algorithm_cards/moment_fourier_focus.md#mf02)，光路、配对标签、FFT幅度裁剪及全连接回归明确；F14普通FocusNet不替代FCFNN | 核/核/部/部/部；答5 | [原记录](autofocus_evolution.md#af-source-8)；[新增作者存档T](algorithm_cards/moment_fourier_focus.md#source-3)、[固定版本源码静态核对](algorithm_cards/moment_fourier_focus.md#source-8)；正式补充与发表快照仍缺 | FCFNN未实现，未训练／接设备；FocusNet仍仅骨架 | P2：G43缺卡已转已有；补充材料、配置快照与安全闭环仍缺 |

## 12 跟踪运动视觉引导

源表：[九个有据节点](tracking_guidance_evolution.md#3-方法演进线九个有据节点)，9行。控制律是算法，缺卡不能因“涉及硬件”而移入非算法。

| 领域节点／版本 | 类型 | 核心改进 | 对应卡／覆盖判定依据 | 五层及问答覆盖 | 证据状态 | 实现状态 | 缺口优先级 |
|---|---|---|---|---|---|---|---|
| 12-01 E01 Kuhn Hungarian，1955 | 指派算法 | 求全局一对一最优而非贪心 | **已有**：[hungarian:H01](algorithm_cards/hungarian.md#h01)，现代势能增广步骤、紧边与最优证书明确；矩形／拒配区分，不以T04黑盒替代 | 核/核/部/核/部；答5，现代内核非1955程序逐行复刻 | [原A/B](tracking_guidance_evolution.md#source-1)；[新增T原文部分](algorithm_cards/hungarian.md#source-2)、[现代讲义](algorithm_cards/hungarian.md#source-3)；原第4–5节仍未全取 | 无独立产品内核或MOT接入；文档参考验证非真实视频验收 | P2：G13缺卡转已有；历史程序、业务代价及视频验收仍缺 |
| 12-02 E02 Kalman，1960 | 状态估计 | 递归状态预测与协方差校正 | **已有**：[remaining_domains:T01](algorithm_cards/remaining_domains.md#t01-线性-kalman-常速状态估计)，线性递推、增益、Joseph与dt模型明确 | 核/核/部/核/部；答5 | [T原文片段](tracking_guidance_evolution.md#source-2)；卡依据现代讲义，非全原文 | 未实现；不含自适应Q/R | P2：噪声模型与缺测回归 |
| 12-03 E03 Lucas–Kanade，1981 | 局部光流 | 梯度线性化并在窗口估位移 | **已有**：[remaining_domains:T03](algorithm_cards/remaining_domains.md#t03-lucaskanade-局部光流)局部平移；[matching:13](algorithm_cards/matching.md#13-lucaskanadelk前向局部配准)补完整迭代契约，非靠角点同名 | T03 部/部/部/核/部；13 核/核/核/核/部设；各答5，不取高合并 | [A/B](tracking_guidance_evolution.md#source-3)；卡为限定模型推导 | 未实现 | P2：局部迭代与失锁协议 |
| 12-04 E04 混合切换视觉伺服，2007 | 控制算法 | 状态依赖切换协调图像与位姿误差 | **已有**：[G44](algorithm_cards/ftp_hybrid_servo.md#g44)已承接两条≥守卫、严格初始能量域、首次α及变增益式(21)–(32)；原文与离线R分开 | L1原守卫／初值已辨识、工程离散仍参考；L2原局部结论及限制已定位；L3/L4/L5仍部分；答5，不转全核 | [原A／B记录](tracking_guidance_evolution.md#source-9)保留；[作者PDF T+V，pp.531–535](algorithm_cards/ftp_hybrid_servo.md#source-5)已核读，非全篇证明复核 | 未实现，无控制仿真／硬件在环／实机；参考伪代码不下发命令 | P0：机制缺卡关闭，证明一致性与采样安全[独立保留](#g44-independent-p0)，不降为P2、不认证式(32) |
| 12-05 E05 MOSSE，2010 | 跟踪算法 | 输出最小二乘滤波与在线统计更新 | **已有**：[remaining_domains:T02](algorithm_cards/remaining_domains.md#t02-mosse-型自适应相关滤波)，目标、闭式解、更新及PSR明确；岭版本显式分开 | 核/核/部/核/部；答5 | [T作者扩展片段](tracking_guidance_evolution.md#source-4)；未审作者代码 | 未实现；灰度平移基线非全相关滤波族 | P2：频域方向、冻结与视频 |
| 12-06 E06 SORT，2016 | 多目标系统算法 | 检测＋框状态Kalman＋IoU指派＋生命周期 | **已有**：[sort:SO01](algorithm_cards/sort.md#so01)，七维框状态、预测／IoU指派、出生、warmup／确认及删除链齐备；组件卡不替代系统 | 核/核/部/核（限定模型代数）/部；答5 | [原A/B记录](tracking_guidance_evolution.md#source-5)；[卡内v1方法T](algorithm_cards/sort.md#source-2)、[固定作者代码](algorithm_cards/sort.md#source-3)；非2016发表快照 | 未实现，无检测器／完整依赖栈／视频或MOT评测；局部分支检查仅历史 | P2：G45缺卡转已有；IoU门控分支、NaN／Inf索引、生命周期与身份验收仍缺 |
| 12-07 E07 SiamFC，2016 Workshop | 学习单目标 | 离线孪生嵌入与在线相关搜索 | **已有**：[siamfc:SI01](algorithm_cards/siamfc.md#si01)，共享嵌入、平衡logistic、固定exemplar、valid相关与尺度／坐标回填齐备；非MOSSE在线更新 | 核/核/部/核（限定模型shape、梯度与坐标代数）/部；答5 | [原T指定章节](tracking_guidance_evolution.md#source-6)；[卡内方法T](algorithm_cards/siamfc.md#source-1)、[固定作者快照](algorithm_cards/siamfc.md#source-3)；2019快照非2016运行环境 | 未实现，无前向／权重／训练或视频评测；有限差分仅历史 | P2：G46缺卡转已有；通道／BN、训练239与255、尺度／插值、零响应与视频验收仍缺 |
| 12-08 E08 Deep SORT，2017 | 学习关联系统 | 运动关联加入外观度量与级联 | **已有**：[deep_sort:DS01](algorithm_cards/deep_sort.md#ds01)，八维xyah、Re-ID离线训练来源、运动门／gallery、年龄级联、受限IoU回退及生命周期齐备；[T04](algorithm_cards/remaining_domains.md#t04-门控后一对一全局指派)仍仅组件 | 核/核/部/核（限定模型代数）/部；答5 | [原A/B，会议由后文引文交叉](tracking_guidance_evolution.md#source-7)；[卡内2017方法T限定](algorithm_cards/deep_sort.md#source-1)、[后续训练来源](algorithm_cards/deep_sort.md#source-8)；2018训练／2025跟踪快照非2017实验，训练论文后部与权重血缘仍缺 | 未实现，无CNN／训练／完整应用或MOT评测；34项有限检查仅历史 | P2：G47缺卡转已有；权重／版本、空特征／gallery、IoU绕门污染、截顶后拒配与视频身份验收仍缺 |
| 12-09 E09 BYTE／ByteTrack，2021／2022 | 关联策略及系统 | 高分先配、低分补剩余轨迹 | **已有**：[bytetrack:BT01](algorithm_cards/bytetrack.md#bt01)，高低分两阶段、unconfirmed第三次指派、失踪／重激活／移除／去重及分数据集配置齐备；[T04](algorithm_cards/remaining_domains.md#t04-门控后一对一全局指派)不替代系统，策略与系统仍一卡 | 核/核/部/核（限定模型代数）/部；答5 | [原T部分，配置分数据集](tracking_guidance_evolution.md#source-8)；[2021初版T部分](algorithm_cards/bytetrack.md#source-2)、[2022正式正文T](algorithm_cards/bytetrack.md#source-3)、[固定作者代码](algorithm_cards/bytetrack.md#source-4)；原文截断、算法图片未逐字核，后续依赖非历史锁包 | 未实现，无完整Python／检测器／LAP／Cython或MOT评测；26项有限检查仅历史 | P2：G48缺卡转已有；依赖／tie-break、inclusive IoU、Removed残留／去重、坐标／空帧时钟与视频验收仍缺 |

## 跨域重复与去重规则

| 重复／关联对象 | 保留节点 | 为什么不能删行或高估覆盖 | 去重处理 |
|---|---|---|---|
| RANSAC，1981 | 02-01、04-06、05-01 | 标定鲁棒外层、匹配几何验证、测量点集拟合是不同领域契约；均可接 matching:15，但不证明三域实现 | 方法阅读共用一张卡；三域验证分别维护，不产生缺卡任务 |
| Lucas–Kanade，1981 | 04-05、12-03 | 配准变换与局部帧间运动输出不同；T03和matching:13原五层状态不同，不覆盖为较高值 | 共用方法理论；保留两卡边界与两领域节点 |
| Haralick，1973 | 06-01、07-01 | HA01承接GLCM；HC73补原分类字面规则但训练／外部原意未闭合，06-01已有、07-01仍部分 | G16保留部分，仅07-01计剩余节点；历史关联仍列两域，不从剩余6节点重复扣1，不增历史任务 |
| U-Net，2015 | 03-05、05-09的子项 | 分割已有S21；测量复合行由S21与HE01分别承接U-Net与HED，现整体已有 | U-Net不新建重复卡；G04部分转已有／P2，05-09仍只一节点 |
| LBP两个年代 | 06-02、09-10 | LB96承接1996整数编码／分布分类，LB02承接2002圆周旋转不变多尺度，两版本不合并 | G06与G08分别已有／P2，D03仍只复用基础定义 |
| Devernay原报告／IPOL实现 | 05-05、05-10 | DE95与DE17分别承接1995原报告和2017误差分析／边链目标 | G03与G05分别已有／P2，本地M04简化版不升级为任一原版 |
| Vollath两篇 | 11-04、11-05 | 1987相关准则与1988性质分析不是同文修订 | 前者G11补方法，后者N23补研究资料，不混入同一算法缺口 |
| 色差／CAM与标准维护 | 09-03／05／07、09-06／08 | 公式、测试注记、标准与模型报告各有来源契约 | 非算法资料可协作补证，但主节点不合并、不变成算法卡覆盖 |

“同一来源”只在明确同文同机制时去重；年份相近、共享数学工具或家族名称不能自动去重。复合节点也不膨胀卡数：LeNet-5／GTN由LG01一卡承接两子方法，08-01仍只一节点；FPFH／SAC-IA由FP01一卡承接两子算法，各五层、共享答5，10-06仍只一节点；BYTE／ByteTrack由BT01一卡承接策略与系统，12-09仍只一节点；HED与旧U-Net合计仍是05-09一节点；**6个剩余覆盖任务不是预测新增卡数量**，G44风险不另增节点或卡。

## 去重缺口清单

以下保留 **G01–G48共48个历史任务ID**，不删除或重排：前41项已有加本轮G44，共42项；其余 **6项覆盖6个“部分／缺卡”节点：4项缺卡、2项部分**。G16保留部分，06-01已有而07-01仍部分，只后者计剩余节点。**G09／G10／G11／G36四项P0缺卡、G02部分／P0及G16部分／P1保留**。前41项P2边界不变，G44仅机制覆盖销项，证明／采样安全仍P0、不降级。其余节点／资料优先级不变，原版或完整系统未实现不随覆盖变化而关闭。

| 缺口键 | 节点 | 状态 | 优先级 | 去重后需补的具体内容 |
|---|---|---|---|---|
| G01 | 02-04 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[KB01](algorithm_cards/kb_zernike_extensions.md#kb01)补全模型／子模型换算、视场、球面标定与圆点观测；部/核/部/部/部不升级，未实现；坐标／因子初始化、规范与原代码仍缺 |
| G02 | 05-04 | 部分 | P0 | [ZM01](algorithm_cards/kb_zernike_extensions.md#zm01)已有VI1992六页三矩／归一化／制核补证，非1993作者稿；全部分、未实现。缺1993原核／方向／检测规则；旧书目与A31确定归属已撤回，不等于证实1993绝无A31 |
| G03 | 05-05 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[DE95](algorithm_cards/edge_evolution_extensions.md#de95)补原NMS、幅值峰及整数像素挂接；原版未实现，本地仍简化版；原程序／许可、取样／单阈值／平台与真实验证缺口保留 |
| G04 | 05-09 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[HE01](algorithm_cards/edge_evolution_extensions.md#he01)补HED侧输出、深监督、logit融合和训练；U-Net复用S21，复合仍一节点；HED／原U-Net未实现，资产／训练／评测与骨架风险保留 |
| G05 | 05-10 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[DE17](algorithm_cards/edge_evolution_extensions.md#de17)补轴向精化、先链后滞后、输出与v1.0许可；未接入／编译，η=5/6容差反例、构链顺序及附包／整图回归保留 |
| G06 | 06-02 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[LB96](algorithm_cards/lbp_hog_extensions.md#lb96)补LBP/C、Set I KL及Set II留一G；未实现，位序／空组、零箱／划分与异常迁移另验，D03不升级 |
| G07 | 07-03 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[HG05](algorithm_cards/lbp_hog_extensions.md#hg05)补R-HOG、线性SVM、一次挖负例重训、滑窗与去重角色；原检测器未实现，NMS具体实现／资产与实测仍缺，RBF候选重排不冒充原法 |
| G08 | 09-10 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[LB02](algorithm_cards/lbp_hog_extensions.md#lb02)补圆周uniform／ri／riu2、VAR、多尺度似然与两分类协议；未实现，旋转／插值、零箱／VAR及Outex验收仍缺，与1996分开 |
| G09 | 11-01 | 缺卡 | P0 | 已有[教学卡／答5](algorithm_cards/focus_origins.md#g09)；仍需VI.3原评分、阈值／支持及VI.6主动调焦流程，不以F02替代 |
| G10 | 11-02 | 缺卡 | P0 | 已有[教学卡／答5](algorithm_cards/focus_origins.md#g10)；[Santos错年对应已闭合](algorithm_cards/focus_origins.md#g10-citation-1971-closed)，不再缺完整引文；仍需1976评分／搜索原页，不以F01或书目销项 |
| G11 | 11-04 | 缺卡 | P0 | 已有[教学卡／答5](algorithm_cards/focus_origins.md#g11)；二手1988求和式与F06差异已说明，仍需1987原定义／支持／二值化 |
| G12 | 11-06 | 已有 | P2 | 原部分／P1于2026-11-06转已有：[SF94](algorithm_cards/sff_haralick_extensions.md#sf94)补1994 SML、逐点焦栈、相邻三点／查表与高度链；SFF未实现，1989差异、查表／有效域、F10非等距／零分及设备验收仍缺 |
| G13 | 12-01 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[H01](algorithm_cards/hungarian.md#h01)补势能增广、正确性及矩形／拒配区分；现代内核非1955程序逐行复刻，无产品内核或MOT接入 |
| G14 | 05-08 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[SF01](algorithm_cards/structured_forests.md#sf01)补结构化标签训练、叶代表及边界融合；2014 sharpening另列，未实现／运行作者训练推理 |
| G15 | 05-11 | 已有 | P2 | 原P0缺卡于2026-11-05转已有：[MF01](algorithm_cards/moment_fourier_focus.md#mf01)补斜坡／楔形与矩求解；宽度、离散误差及图像复现仍缺 |
| G16 | 06-01、07-01 | 部分 | P1 | [HA01](algorithm_cards/haralick.md#ha01)承接06-01；[HC73](algorithm_cards/sff_haralick_extensions.md#hc73)全部分、未实现，1972 pp.100–110训练正文及选法仍缺。≥字面、第一版位置及集合／退化边界子项已闭合；作者原意另待证，不强求修式或唯一标签才核字面；仅07-01计剩余 |
| G17 | 06-03 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[OC01](algorithm_cards/one_class_svm.md#oc01)补原对偶、KKT、ρ／ν及核求解；等权PSD模型边界明确，无专用实现或已训练模型 |
| G18 | 06-04 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[SA01](algorithm_cards/ssim_autoencoder.md#sa01)补具体AE、正常训练、局部结构损失、重建融合与残差；未实现，作者AE锁定、常数口径及真实验收仍缺 |
| G19 | 06-06 | 已有 | P2 | 原P0缺卡于2026-11-05转已有：[PaDiM](algorithm_cards/padim_root_polynomial.md#g19)补嵌入、选维、逐位高斯与出版字段分离；ACM差异成因、正式版比对及实现仍缺 |
| G20 | 06-07 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[DR01](algorithm_cards/draem.md#dr01)补异常模拟、双子网、联合损失／反传及整图聚合；未实现，论文／代码参数差异、梯度与真实泛化仍待验 |
| G21 | 06-08 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[PC01](algorithm_cards/patchcore.md#pc01)补聚合、coreset、NN及原式(7)；固定源码平方距离评分无重加权，未实现，库／权重／FAISS与真实验收仍缺 |
| G22 | 07-04 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[RN01](algorithm_cards/resnet.md#rn01)补原残差块、投影、五架构和监督训练；未实现，原训练／预处理、18/34作者实现及跨框架验证仍缺 |
| G23 | 07-05 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[FR01](algorithm_cards/faster_rcnn.md#fr01)补anchor／RPN、两级检测及版本训练；未实现，空GT／anchor、inclusive解码、原资产与AP验收仍缺 |
| G24 | 07-06 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[OM01](algorithm_cards/openmax.md#om01)补MAV／右尾／rank／未知激活／拒识；未实现，距离尺度／传参／exp风险、libMR拟合／许可与完整风险保证仍缺 |
| G25 | 07-07 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[SG01](algorithm_cards/iterative_scene_graph.md#sg01)补对象／关系更新、监督和三任务；未实现，顺序更新／双向过滤差异、空稀疏图与装配验收仍缺 |
| G26 | 08-01 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[LG01](algorithm_cards/lenet_gtn.md#lg01)分别承接原LeNet-5及GTN全局训练，复合仍一卡、共享答5；未实现，连接数冲突／原资产、DAG守卫／remove梯度及CNN桥接仍缺 |
| G27 | 08-02 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[CT01](algorithm_cards/ctc.md#ct01)补前后向／后验／梯度、重复字符、数值与解码近似；生产实现未实现，参考检查不等于OCR接入，批量／后端和CER验收仍缺 |
| G28 | 08-03 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[CR01](algorithm_cards/crnn.md#cr01)补列序列、双向循环投影、CTC接口与宽度；未实现，24／25／26帧冲突、BN／归一化／padding污染及OCR验收仍缺 |
| G29 | 08-04 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[EA01](algorithm_cards/east.md#ea01)补RBOX／QUAD、损失／恢复／LANMS；未实现，fork非完整原实验代码，恢复索引和末级NMS集合风险、QUAD资产与检测验收仍缺 |
| G30 | 08-05 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[DB01](algorithm_cards/db_text_detection.md#db01)补P／T／近似二值训练及区域恢复；未实现，distance／共线分段错误、跨batch广播、空监督／多路径及检测验收仍缺 |
| G31 | 08-06 | 已有 | P2 | 原P1缺卡于2026-11-05独立复核转已有：[TR01](algorithm_cards/trocr.md#tr01)补patch编码／因果生成、预训练／微调、词元及拒识设计；未实现，原recipe／固定词元资产、cache／视觉padding及真实随机编号验收仍缺 |
| G32 | 09-09 | 已有 | P2 | 原P1缺卡于2026-11-05独立复核转已有：[RE01](algorithm_cards/retinex_1971.md#re01)以既有pp.6–10记录承接近邻比／阈值／累计reset／独立路径平均；部/核/部/部/部不升级，未实现；原数值阈值、二维路径分布／数量／长度、停止／重复访问及真实颜色验收仍缺 |
| G33 | 09-11 | 已有 | P2 | 原P1缺卡于2026-11-05独立复核转已有：[GE01](algorithm_cards/gray_edge.md#ge01)补导数阶／尺度／范数、归一及校正；未实现，作者二阶系数4与标准2、mask反向注释／支持污染、许可及全链实图验收保留 |
| G34 | 09-12 | 已有 | P2 | 原P0缺卡于2026-11-05转已有：[RPCC](algorithm_cards/padim_root_polynomial.md#g34)已取得接受稿，明确基项／拟合／有效域；加权秩防护为设计，无实拍验收 |
| G35 | 09-13 | 已有 | P2 | 原P1缺卡于2026-11-05独立复核转已有：[FC01](algorithm_cards/fc4.md#fc01)补局部方向／置信池化、角损失与监督；未实现，三种头、pool8／通道差异、测试选模／色板／fold、零向量及跨相机验收仍缺 |
| G36 | 10-01 | 缺卡 | P0 | 已有[教学卡／答5](algorithm_cards/ftp_hybrid_servo.md#g36)的后续参考链与几何自推；仍需FTP1983原频带／相位／标定逐项对照 |
| G37 | 10-03 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[CL01](algorithm_cards/curless_levoy.md#cl01)补视线距离、权重累积、可见性与等值面；未实现，L4仍部分，一般三维证明、补洞标签与实扫验收仍缺 |
| G38 | 10-05 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[SM01](algorithm_cards/sgm.md#sm01)补HMI代价、路径递推、聚合与左右检查；未实现，L4核限代数，SGBM非MI，完整图像链与整数／后处理验收仍缺 |
| G39 | 10-06 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[FP01](algorithm_cards/fpfh_sacia.md#范围版本与覆盖状态)两子算法各五层、共享答5且仍一卡；均部/核/部/部/部并保留各层限定，未实现；2009方法原文未取得，原式／权重／Huber与固定PCL变体、状态污染及点云验收仍缺 |
| G40 | 10-07 | 已有 | P2 | 原P1缺卡于2026-11-05转已有：[PM01](algorithm_cards/psmnet.md#pm01)补SPP、拼接代价体、hourglass与视差监督；未实现，L4核限模型，GT／padding／插值、旧框架分支与立体实测仍缺 |
| G41 | 10-08 | 已有 | P2 | 原P1缺卡于2026-11-06转已有：[MD01](algorithm_cards/monodepth2.md#md01)补最小重投影、identity掩膜、多尺度与训练／推理尺度协议；未实现，L4限针孔／局部可微域，内参／采样／尺度及域外验收仍缺 |
| G42 | 10-09 | 已有 | P2 | 原P1缺卡于2026-11-06转已有：[FG01](algorithm_cards/fcgf.md#fg01)补量化索引、稀疏网络、hardest负排除与特征输出；未实现，L4限显式模型及固定索引局部梯度，采样／零范数／空集和真实配准仍缺 |
| G43 | 11-08 | 已有 | P2 | 原P0缺卡于2026-11-05转已有：[MF02](algorithm_cards/moment_fourier_focus.md#mf02)补光路、标签、FFT前端与FCFNN；正式补充、发表快照、训练／设备验证仍缺 |
| G44 | 12-04 | 已有 | P0 | [原机制及答5](algorithm_cards/ftp_hybrid_servo.md#g44)已有，原守卫／初值／首次α／变增益已辨识；仅机制缺卡销项。[证明一致性、采样安全独立P0](#g44-independent-p0)保留，不转P2；无仿真、硬件在环或实机 |
| G45 | 12-06 | 已有 | P2 | 原P1缺卡于2026-11-06转已有：[SO01](algorithm_cards/sort.md#so01)补七维框状态、IoU指派及出生／确认／删除；未实现，L4限模型代数，门控分支、NaN／Inf索引与视频身份验收仍缺 |
| G46 | 12-07 | 已有 | P2 | 原P1缺卡于2026-11-06转已有：[SI01](algorithm_cards/siamfc.md#si01)补离线共享训练、首帧模板、相关及尺度搜索；未实现，L4限shape／梯度／坐标代数，通道／BN／训练尺寸、插值与视频验收仍缺 |
| G47 | 12-08 | 已有 | P2 | 原P1缺卡于2026-11-06独立复核转已有：[DS01](algorithm_cards/deep_sort.md#ds01)补外观训练来源、八维状态、运动门／gallery、年龄级联、IoU回退及生命周期；核/核/部/核（限定模型代数）/部、答5，未实现；2017快照／权重血缘、2018训练论文后部、空特征／gallery、绕门污染与完整视频验收仍缺 |
| G48 | 12-09 | 已有 | P2 | 原P1缺卡于2026-11-06独立复核转已有：[BT01](algorithm_cards/bytetrack.md#bt01)补BYTE两阶段、ByteTrack三次指派、分数据集配置、失踪／重激活／移除／去重；核/核/部/核（限定模型代数）/部、答5，未实现；原文截断／算法图片、历史依赖／tie-break、inclusive IoU、Removed残留、坐标／空帧时钟与视频验收仍缺 |

<a id="g44-independent-p0"></a>

### G44独立P0风险（不计覆盖缺口）

| 风险 | 当前状态 | 闭合门槛 |
|---|---|---|
| 证明一致性 | **P0保留、不降级**；p.534邻域记号、p.535式(23)的H／范数、式(25)、(27)–(29)推导疑点 | 可归属勘误／作者澄清或独立严谨分析；原印与修正式分开，不静默修补，不把原式可读当定理已证 |
| 采样安全 | **P0保留、不降级**；式(32)仅为论文条件，无最小驻留时间证明；越界、交点停滞及设备时序未验 | 有效区间增长界及采样／时延／噪声／执行器模型，独立安全保护与分阶段验证；当前无控制仿真、硬件在环或实机 |

两风险共用G44历史ID，不产生新卡或第7项覆盖任务；“已有”只表示具体机制已承接。依据为[G44原式、疑点和复现门槛](algorithm_cards/ftp_hybrid_servo.md#g44)。

<a id="minimum-source-pages"></a>

### 本轮原材料最小缺页清单

只索取承载规则的原页及必要上下文，题名／版次／印刷页须可核；未定位的具体页不猜测，合法副本或可确认作者稿均须核版本。以下保留缺页摘要；按四卡最新续查整理的[六项原文获取／验收清单](source_requests.md)统一记录准确书目、印刷页与PDF页序、最小方法内容及合法作者稿门槛，六项全部未关闭。G44另列风险材料，非全库P2／N类资料全集。

| 对应项 | 最小尚缺原材料 | 已得材料不重复索取 |
|---|---|---|
| G02／ZM01，部分P0 | 1993期刊pp.295–306中的矩定义、原核／系数、归一化、方向、判边及必要上下文；具体方法页未定位。若主张A31联合宽度，须有对应式／PSF／单位与条件 | VI1992 pp.247–252已得，不冒充1993；[版本门与原核缺口](algorithm_cards/kb_zernike_extensions.md#zm01) |
| G09，缺卡P0 | Tenenbaum1970 VI.3 pp.311–317的准则及被引算子定义；VI.6 pp.336–351中初始化、更新、细化／停止，优先p.347流程图及解释；PDF序号未核 | 1970封面／目录、1971 IJCAI pp.3、7已得；[页码限制](algorithm_cards/focus_origins.md#g09-original-mapping-20261106) |
| G10，缺卡P0 | Brenner1976 pp.100–111中评分、方向／跨度／门控／聚合及搜索段、图式、跨页续文；确切方法页未定位，若仅转引则补直接来源 | Santos pp.265、266、272及1971→1976对应已闭合；[方法缺页](algorithm_cards/focus_origins.md#g10-original-mapping-20261106) |
| G11，缺卡P0 | Vollath1987 pp.279–288的准则定义／式号、支持、均值／归一化、二值化；1988 pp.133–146相应定义和假设用于版本对照，具体方法页未定位 | Santos二手求和式已得；无图题曲线裁图不替代定义；1988仍为N23，不新增任务；[版本限制](algorithm_cards/focus_origins.md#g11-original-mapping-20261106) |
| G36，缺卡P0 | FTP1983 pp.3977–3982中的频带／谐波、相位与展开、光路／高度、标定原页及上下文；具体页／式号未定位 | 2005 pp.378–380转换文本与后续几何同形不替代原版；[原版对照](algorithm_cards/ftp_hybrid_servo.md#g36) |
| G16／HC73，部分P1 | Fukunaga1972第一版§4.3 p.100、§4.4 p.106，优先pp.100–110及被引定义；需目标值、权重、样本估计／求解并映射W_ij。若仍不能唯一选法，再补同期说明 | 1973 pp.616–617的≥、1972版权／p.viii／p.5已核；勘误或原程序仅在声称更正原意／历史实现时补，不强求全部；[分层清单](algorithm_cards/sff_haralick_extensions.md#g16-专项复核与最小销项证据清单) |
| G44，独立P0风险 | 不是缺原公式页：需针对p.534–535疑点的勘误／作者澄清或独立证明材料；采样时序与安全验证记录另补 | pp.531–535页图已核；不再索取同一守卫／增益截图，不将缺页补齐等同安全验收 |

### 非算法资料待办

N01–N24 与24个非算法节点一一对应，不算缺算法卡；具体优先级及证据限制已列矩阵。N01为计量规范；N02–N04为异常基准、装配表示和活动基准；N05–N12为8个编码规范；N13–N20为8个颜色表示／公式／模型／验证规范；N21为ToF硬件；N22–N24为3个对焦比较／分析研究。补齐方式可以是规范阅读页、数据协议或硬件说明，不必强行生成“算法卡”。

### 已有卡仍需补的深度

83个已有节点仅表示具体卡与机制匹配。预处理、标定多卡五层仍全部为部分；分割L3–L5仍部分；匹配R卡无实现；FFT-NCC的L5核源也只限源码与有限历史测试。G09／G10／G11／G36五层仍部分；G44按原表保留L1原守卫／初值辨识、L2原局部结论定位及L3–L5部分，不将机制已有翻译成全层“核”。其他卡五元及限定以本页逐节点与逐卡索引为准，HA01的L1、RE01的部/核/部/部/部及FP01的固定源码限定不变；LG01与FP01两子方法仍各计一卡。历史代数、源码隔离、训练版本／权重、依赖与视频缺口分别保留，不因有卡清除P2深度工作，更不清除G44独立P0。不能把83、85或255称作“已完整复现算法数”。

本轮KB01为部/核/部/部/部，ZM01与HC73均部/部/部/部/部；其余七卡核/核/部/核（各自限定）/部、各答5。DE95／DE17／HE01的L4分别限三点与一维模型、峰位／链评分／固定支持复杂度、二值损失／链式梯度；LB96／HG05／LB02分别限KL／列联表G／不变性、维数／投票／归一化／线性SVM模型、循环群／VAR／似然；SF94限等距重写／尺度还原／条件敏感性。KB初始化疑点、1993原算子／A31、DE17容差与顺序、HED资产、LBP／HOG版本、SFF设备链及Haralick原训练／式(8)原意均不因机制覆盖而关闭。

## 核验与维护

本次仅创建source_requests.md并更新本页、docs/README.md，依据四卡既有续查证据整理请求与验收，不修改四卡正文。核对新清单及入口的本地链接、113个节点、G01–G48历史ID与六项未关闭任务、255个逐卡唯一键；远端URL不重新访问。12份源表不增删节点；历史公式目视、代数检查与算法运行均不冒称本轮执行。

本轮口径：12域节点数6／6／7／14／11／8／8／14／13／9／8／9，四态83／2／4／24，方法分母89；已有93.26%，已有或部分95.51%。48个历史任务为42已有、2部分、4缺卡，剩余6项，G44两项风险独立P0；45份正文仍255卡。所有卡等级保持原状，ZM01／HC73全部分；本次不改G44机制、证明或安全结论。

### 历史同步检查结果（G44机制与风险分账）

本节保留此前G44同步轮次的检查与14文件变更记录，不作为本次三文件原文请求清单的验证结果。

2026-11-06实际本地检查：73份非忽略仓库Markdown（显式纳入research_notes）共4917处本地链接通过，去重目标1258个，含104处源码行号、13处引用式链接；文件、标题／显式锚点及行号范围均有效。845处显式Markdown外链仅计数、未访问，裸URL不纳入链接统计，不认证远端可用性或阅读深度。

255个唯一键无重复，均定位45份正文编号标题，索引／导航两份文件数量表均45文件、255卡。12份总览14个主分支表实数为6／6／7／14／11／8／8／6／8／8／5／9／8／9，合计113；逐节点、12域汇总、方法分母89与48个历史任务闭合。剩余键恰为G02／G09／G10／G11／G16／G36，G44为已有／P0，未混入6项覆盖任务。

本轮14份修改文档的UTF-8、169张表格列数、围栏／显示公式配对、显式锚点唯一性与行尾空白检查通过；focus／HC73未残留图片归因用户的旧措辞。检查初轮排除了5处数学／代码误识链接，并修正将领域号误计汇总列的检查规则；发现G44表内绝对值竖线会拆列，改为等价的lvert／rvert记法，不改数学含义。git diff --check通过，仅有既存LF／CRLF提示；未跟踪文档亦直接检查，不能只靠Git差异验收。

变更限根README、三台账、research_notes、ftp_hybrid_servo／focus_origins／sff_haralick_extensions／kb_zernike_extensions及三维／测量／对焦／分类／跟踪五份总览。检查脚本仅存系统临时目录，复用现有虚拟环境，无源码、依赖、测试或卡编号变更，未提交。原始PDF目视、36项HC73算术及其他历史算法检查未重跑；无控制仿真、硬件在环、原实验、RTImg、训练或实机验收，不把文档校验升级为实测。

<a id="本轮本地复核结果255卡仅文档检查"></a>

### 历史本地复核结果（255卡，G44同步前）

以下保存前次255卡轮次原统计与文件摘要；82／2／5／24、7项任务及“前245条逐行保持”只指当时状态，不是本轮结果。旧锚点保留，不能用历史链接计数代替当前检查。

2026-11-06实际复算：45份正文独立编号与255个索引唯一键一一对应，无缺失或重复；两份文件数量表均为45文件、255卡。前245条逐卡索引按原顺序以LF连接取UTF-8 SHA-256，保持 `86900e17dceec5ec31f78287648b46fb32252fb674176d643bfe70cb5dec157e`。四份扩展正文为2＋3＋3＋2卡，共50段L1–L5正文、50组逐层问答；各卡五元与限定按原状态表转录，KB01的四个部分层级、ZM01／HC73的全部分及旧卡等级不升级。

12份总览的14个主分支表逐表实数为6／6／7／14／11／8／8／6／8／8／5／9／8／9，合并12域为6／6／7／14／11／8／8／14／13／9／8／9，共113行；113个节点键唯一，12域汇总与四态82／2／5／24闭合。方法分母89，已有82/89=92.13%，已有或部分84/89=94.38%。48个历史任务按原序保留：41已有、2部分、5缺卡；剩余7项为G02部分／P0、G16部分／P1和G09／G10／G11／G36／G44五项P0缺卡。24个非算法ID仍一一对应，HED／旧U-Net复合节点及Haralick跨域计数不增加。

Git列出的73份非忽略仓库Markdown共4843处本地链接全部通过，去重目标1236个，含104处源码行号与13处引用式链接；核验本地文件、标题／显式锚点及源码行号范围。九份目标文档共1574处：本页408、docs/README 268、algorithm_cards/README 526、标定45、测量87、缺陷44、分类48、颜色63、对焦85。四份扩展卡分别32／99／50／37处；811处显式Markdown外链只计数、未访问，裸URL不混入统计，远端可用性与原文访问深度不升级。

九份修改文档的UTF-8、表格列数、围栏、显式锚点唯一性及行尾空白检查通过。扩展正文初轮检查曾发现一处既有格式例外：edge_evolution_extensions.md的DE17问答表，原第202行表头／数据为4列，第203行分隔为5列；当时按授权范围未改卡片。该格式例外后续已修，2026-11-06只读复核确认表头、分隔行及五组问答均为4列，四份扩展文件共48张表的列数与分隔结构检查全部通过；此次仅更正本处历史措辞，不改卡片、算法或状态。初轮检查将行内数学误认链接、G39复合承接标题误计卡及领域号误计汇总列；核对原文后修正规则，未为消除误报改旧卡。

变更仅三台账与标定／测量／缺陷／分类／颜色／对焦六总览的必要回链。Git列出的103个非目标、非忽略文件，按路径排序，将“路径:内容SHA-256小写十六进制”以LF连接再取UTF-8 SHA-256，前后一致为 `414364674ce0c0d1e21791207654e9538fddd08323d177b3b4cb68ebf3398606`。旧历史区仅将245卡标题改为历史、保留旧锚点并加轮次说明；还原该包装后的历史正文摘要保持 `dc31b3661c241223b5bf0e6378dfeb649e5c8cda3f14087a6f09d7e76a37569f`。边界不含忽略目录或Git元数据，不声称全工作区无并发变化。使用既有虚拟环境以内存命令检查，未新增脚本／依赖；git diff --check通过，仅有既存LF／CRLF提示，未据此改文件；未跟踪文档另作直接检查。原用户修改未回退、未提交。四扩展卡、DE95冒烟及此前所有算法检查均为历史，未重跑、训练、复测RTImg或验收设备；证据限制、未实现及P2深度缺口继续保留。

### 历史核验记录（非本轮重跑）

以下保留前批223卡台账的链接统计、PatchCore有限分支检查及文件摘要，均为当时工作树的历史证据，不代表本轮239卡统计或检查结果；段内“本轮／本次”仅指该历史轮次。

前批专项记录：[PaDiM L3问答](algorithm_cards/padim_root_polynomial.md#g19)说明固定分数下降低异常阈值扩大NG集合，不能靠降阈值修复域差；[RPCC L5](algorithm_cards/padim_root_polynomial.md#g34)检查sqrt(W)A有效秩并拒绝全零权重，岭项可解不等于数据可辨识。这些历史检查不作为本轮算法实测。本轮仅对[PC01参考伪代码](algorithm_cards/patchcore.md#pc01)的评分分支作有限执行检查：K=1且reweight=true／false均回退raw、reweight_applied=false；K≥2且请求重加权时非法b拒绝，不静默回退。检查不执行EMBED、FAISS、coreset或热图，不授予L4／L5升级；其他卡内算法检查器不重跑。

本轮实际检查通过：三份台账共878处本地链接（本页297、docs/README 220、algorithm_cards/README 361）；加四张新卡（SA01 35、DR01 96、PC01 57、RN01 50）为1116处、跨文件去重目标567个，另67处外链只计数。Git列出的47份仓库Markdown合计3495处链接：2995处本地链接全部通过、500处外链未重访；本地去重目标875个，含96处源码行号与13处引用式链接。223个唯一键与19份独立编号正文一一对应，新增20层正文／20组问答及三处五元状态转录一致；113节点、12域汇总、48个历史任务与37项剩余任务闭合，五项P0缺卡及G02的P0部分保留。三台账与四新卡的UTF-8、表格列数、围栏、显式锚点及行尾空白检查通过。

PatchCore有限检查从PC01提取评分条件分支，只将布尔字面量和require转为Python可执行语法，以确定性一维库与直接距离辅助函数检查，不运行完整PREDICT。15例通过：K=1下reweight真／假各测b=0／1／2，共6例均返回raw且applied=false；K=2／3且请求重加权时分别拒绝b=0／1／K+1，共6例且均未进入支持查询；关闭重加权另2例返回raw；合法K=2、b=2、支持距离1／2的1例得0.731058578630005且applied=true。只证明这些有限分支和代数输入，不认证其他输入域、作者实现或检测性能。

检查器初轮漏识别matching编号后的点号，另将代码索引、相邻引用标记和LaTeX矩阵误认作链接；核对实际正文、补齐编号规则并排除代码／数学区后重跑通过，未为消除误报修改旧稿。跨命令PowerShell变量未保留，随后重新以内存命令执行；未新增脚本或依赖。Git列出的83个非忽略、非目标文件按路径排序拼接路径与内容的SHA-256前后一致：`2fd013ea2030fb53e8ade351b3117cac088114e301ae38838d77dfe68bbddf2c`。此边界不包括忽略目录或Git元数据，不称全工作区证明；未回退用户原有变更。`git diff --check`通过，但三台账尚未跟踪，另以直接文件检查覆盖，不能仅依赖该命令。

<a id="本轮本地复核结果231卡"></a>
### 历史本地复核结果（231卡，非本轮重跑）

以下统计、状态变化和文件摘要属于上一轮231卡台账，段内“本轮／本次”仅指该历史轮次；保留旧标题锚点供已有链接使用，不作为当前计数。

本轮实际检查通过：三台账共977处本地链接（本页323、docs/README 236、algorithm_cards/README 418）；八卡分别为FR01 99、OM01 47、SG01 83、LG01 46、CT01 39、CR01 53、EA01 55、DB01 60处。Git列出的55份仓库Markdown共3576处本地链接全部通过，去重目标989个，含96处源码行号与13处引用式链接；620处外链只计数、未重访。检查覆盖本地文件、标题／显式锚点与源码行号范围，不验证远端可用性。

27份正文独立编号与231条索引唯一键一一对应，无缺失或重复；45段层级正文与40组问答齐备，原卡覆盖表、逐卡索引及八个节点的状态逐层一致。LG01两子方法各五层、共享答5且只计一卡；CT01的L4核保留模型限定，其他六卡L4仍部分，八卡L3／L5均部分。前223条索引除PC01的“本轮仅核”改为“历史仅核”外逐行一致，等级不变。113节点及12域汇总、48个历史任务、29项剩余任务闭合，五项P0缺卡、G02的P0部分、G16部分及24个非算法节点保留。

三台账与八卡的UTF-8、表格列数、围栏、显式锚点唯一性及行尾空白检查通过。检查器曾将行内LaTeX矩阵乘法误识为链接，并将L1下的子标题误作正文结束；核对原文后修正检查规则通过，未修改卡片或旧稿以消除误报。使用已有虚拟环境以内存命令执行文档检查，未新增脚本／依赖，也未运行卡内算法检查器。`git diff --check`通过；三台账尚未跟踪，另以直接文件检查覆盖，不能仅依赖Git差异检查。

变更范围复核：Git列出的91个非忽略、非目标文件，按路径排序，以每行“路径:内容SHA-256”、换行连接后再取UTF-8 SHA-256，前后一致为 `C532E191F04FFC7C8CFDC3B8AC2940A4394DCF492F33F563D7D580ECA00FD745`。不含忽略目录或Git元数据，不称全工作区证明；未回退用户原有变更。历史PatchCore分支、CTC数值交叉、EAST／DB源码隔离等检查未重跑，不冒称本轮证据。

<a id="本轮本地复核结果235卡仅文档检查"></a>
### 历史本地复核结果（235卡，非本轮重跑）

以下为上一轮235卡工作树的记录，段内“本轮／本次”及三文件变更范围仅指该历史轮次，不作为当前239卡统计或验证。

2026-11-05实际检查通过：三台账共1023处本地链接（本页337、docs/README 244、algorithm_cards/README 442）；四卡分别为TR01 62、RE01 44、GE01 41、FC01 66处，合计七文件1236处。Git列出的59份仓库Markdown共3835处本地链接全部通过，去重目标1038个，含96处源码行号与13处引用式链接；661处外链只计数、未重访。检查覆盖本地文件、标题／显式锚点与源码行号范围，不认证远端可用性或来源深度。

31份正文独立编号与235条逐卡唯一键一一对应，无缺失或重复；无编号补充、G26／G35交接标题和检查器不计新卡。原231条索引逐行保持，按原顺序以换行连接后取UTF-8 SHA-256，前后一致为 `60baa2dc0af5df85c02e67f00fabe57808f8b4d600f2ad5926bc323e27f973df`。235条五层状态格式合法，四卡20段层级正文／20组问答齐备，覆盖表、索引与四个节点逐层一致：TR01核/核/部/部/部，RE01部/核/部/部/部，GE01核/核/部/核（限定条件代数）/部，FC01核/核/部/核/部且L4限非零及可微域。四项均已有、P2且未实现，不把“缺卡销项”写作“实现完成”。

12份总览的14个分支源表实际合计113行，与12域节点及四态64／10／15／24闭合；48个唯一历史任务为23已有、10部分、15缺卡，25个剩余任务与25个部分／缺卡节点一一对应。五项P0缺卡、G02的P0部分、G16部分及N01–N24保留。当前摘要、汇总表、导航与任务清单无过期计数；223／231卡旧统计仅保留在标明轮次的历史记录或累计算式中，旧231卡标题锚点继续有效。

三台账及四卡的UTF-8、表格列数、围栏配对、显式锚点唯一性和行尾空白检查通过。初次编号扫描将G26／G35交接标题列为候选，过期计数扫描将G19后的“缺卡”误识为19个缺卡；按正文编号和任务ID语义修正规则后检查通过，未为消除误报修改卡片。使用已有虚拟环境在内存执行文档检查，未新增脚本或依赖；`git diff --check`通过，三台账尚未跟踪，另以直接文件检查覆盖。

变更范围复核：Git列出的95个非忽略、非目标文件，按路径排序，以每行“路径:内容SHA-256”、换行连接后再取UTF-8 SHA-256，前后一致为 `c4d5e2bc841ad5b6929198b24ca0f41692042742c563c785ade01afceb389fd3`。不含忽略目录或Git元数据，不称全工作区证明；未回退用户原有变更。本次仅文档检查，未运行任何卡内算法检查器；四卡算术／差分、PatchCore分支、CTC数值交叉和EAST／DB源码隔离记录均未重跑。

链接核验只确认当前本地文件及锚点存在，不证明远端论文仍可访问，也不提升来源深度。本次变更仅限 docs/evolution_coverage.md、docs/README.md、docs/algorithm_cards/README.md；不修改根README、任何卡片正文、总览、旧稿、源码、依赖或测试，不把用户原有未提交变更当成本次修改。完整算法运行、训练、RTImg重测、标准符合性及设备闭环均不在本次验证范围。

<a id="本轮本地复核结果239卡仅文档检查"></a>
### 历史本地复核结果（239卡，非本轮重跑）

以下保留239卡轮次的原检查记录；本节及此前历史节中的当前／本轮统计仅指各自当时工作树，不作为本轮245卡结果。旧标题锚点保留，原记录正文不改。

2026-11-05实际检查通过：四份目标文档共1143处本地链接（本页353、docs/README 252、algorithm_cards/README 464、3d_evolution 74）；四卡分别为CL01 38、SM01 24、FP01 86、PM01 84处，八文件合计1375处。Git列出的63份仓库Markdown共4125处本地链接全部通过，去重目标1088个，含96处源码行号与13处引用式链接；680处外链只计数、未重访。检查覆盖本地文件、标题／显式锚点和源码行号范围，不认证远端可用性、原文获取深度或算法性能。

35份正文独立编号与239条逐卡唯一键一一对应，无缺失或重复；G26／G35／G39交接标题、无编号补充与检查器不另计卡。原235条索引按原顺序以换行连接后取UTF-8 SHA-256，前后一致为 `8b9460eff250b0be561fcbc7efc21e4c4b816d7f27b0da5c370fe8561b3c8bab`。四卡25段层级正文及20组问答齐备；FP01两子算法各五层、共享答5且只计一卡。CL01核/核/部/部/部，SM01核/核/部/核（限定代数）/部，PM01核/核/部/核（限定模型）/部；FP01两子算法均部/核/部/部/部，L1固定源码链、L2与L4各自限定原样保留。2009会议方法正文未取得，原式／权重／Huber与固定PCL差异仍列P2，不升级读卡或来源状态。

12份总览的14个分支源表实际合计113行，12域节点数为6／6／7／14／11／8／8／14／13／9／8／9；逐节点与汇总表四态68／10／11／24闭合，方法分母89。48个历史任务为27已有、10部分、11缺卡；剩余21任务与21个部分／缺卡节点对应。G37–G40机制缺卡转已有、均P2且未实现；五项P0缺卡、G02的P0部分、G16部分及N01–N24全部保留。旧223／231／235卡统计仅作历史轮次记录，旧标题锚点继续有效。

四份目标文档与四卡的UTF-8、表格列数、围栏配对、显式锚点唯一性及行尾空白检查通过。首轮检查将行内代码索引和相邻引文标签误识为引用式链接，并把PM01层内子标题误作层级正文结束；核对正文后修正检查规则通过，未修改卡片或旧稿消除误报。检查在现有虚拟环境以内存命令执行，未新增脚本或依赖；`git diff --check`通过，目标文档尚未跟踪，另以直接文件检查覆盖，不仅依赖Git差异。

变更范围复核：Git列出的98个非忽略、非目标文件，按路径排序，以每行“路径:内容SHA-256”、换行连接后再取UTF-8 SHA-256，前后一致为 `bf2c08e06c927f636f60cbd470ea634add9d5ef6f1280d5695a01c9abeabeb68`。范围不含忽略目录或Git元数据，不称全工作区证明。此次仅更新 docs/evolution_coverage.md、docs/README.md、docs/algorithm_cards/README.md 和 docs/3d_evolution.md 的已有覆盖描述、索引及核验记录，未回退用户原有变更；卡片正文、其余总览、源码、依赖与测试未改，未提交。四卡算术／穷举／shape、前批差分、PatchCore分支、CTC数值交叉及EAST／DB源码隔离均为历史证据，本轮未重跑；训练、RTImg、物理测量和设备闭环不在本次验证范围。

<a id="本轮本地复核结果243卡仅文档检查"></a>
### 历史本地复核结果（243卡，非本轮重跑）

以下保留243卡轮次的原检查记录，段内“本轮／本次”仅指当时工作树；不作为本轮245卡统计或算法重跑证据，旧标题锚点继续有效。

2026-11-06实际检查：Git列出的67份非忽略仓库Markdown共4408处本地链接，去重目标1145个，含96处源码行号与13处引用式链接；核验本地文件、标题／显式锚点和源码行号范围。五份目标文档共1270处（本页365、docs/README 260、algorithm_cards/README 484、3d_evolution 81、tracking_guidance_evolution 80）；四卡MD01／FG01／SO01／SI01分别86／58／28／56处，九文件共1498处。713处显式Markdown外链只计数、未访问，裸URL不混入本地计数；不认证远端可用性、阅读深度或算法性能。

39份正文独立编号与243条索引唯一键一一对应，文件计数表和导航合计一致，无缺失或重复。原239条索引按原顺序以换行连接后取UTF-8 SHA-256，保持 `cf739fb96fbdf4dadb921188032c2378c6e13a25c4efa3a49bf45c1275f9ee76`。四卡20段L1–L5正文及20组问答齐备；覆盖表、索引和四节点均转录核／核／部／核（各自限定）／部，L4限定与未实现边界不变。首轮编号规则误排含“交接”的P36，并误纳G26承接标题；核对标题后收窄规则，未改卡片消除误报。首轮链接检查的两处待补目标为本节锚点，随实际检查记录补齐后再核验。

12份总览的14个主分支表合计113行，与逐节点及12域汇总的72／10／7／24闭合；方法分母89，已有80.90%，已有或部分92.13%，不是完整复现率。48个历史任务为31已有、10部分、7缺卡；17个剩余任务与17个部分／缺卡节点对应。G41／G42／G45／G46转已有、均P2且未实现；缺卡保留G09／G10／G11／G36／G44五项P0及G47 Deep SORT／G48 ByteTrack。G02的P0部分、G16部分及N01–N24保留。当前摘要／表格／任务清单无过期统计；223／231／235／239卡旧账保留在标明轮次的历史节或累计算式，SiamFC训练尺寸239不误当卡数。

五份目标文档与四卡的UTF-8、表格列数、围栏配对、显式锚点唯一性及行尾空白检查通过。现有虚拟环境以内存命令执行文档检查，未新增脚本或依赖；`git diff --check`通过，提示现有已跟踪文件的LF／CRLF转换警告，未据此修改文件。五份目标文档尚未跟踪，直接文件检查另行覆盖，不仅依赖Git差异。

变更范围：仅三份台账及3d／tracking总览必要回链。Git列出的101个非目标、非忽略文件，以排序的“路径:内容SHA-256”逐行拼接再取UTF-8 SHA-256，前后一致为 `e87a9882890bf6bb20e9669cfca5b5f5357c6961b2ac25e7d238af20b9e96ed6`；卡片正文、其余总览、源码、依赖与测试未改。历史区除239卡标题改为历史、保留旧锚点并加轮次说明外，正文摘要还原后一致为 `4736b59470fcdb0d96f144858d0d570d3d9fee574b545967624b97e63f57d8f1`。边界不含忽略目录或Git元数据；用户原有修改未回退、未提交。四卡及前批算法微检查均未重跑，未训练、未运行网络／数据集、未复测RTImg或验收物理／设备闭环。

<a id="本轮本地复核结果245卡仅文档检查"></a>
### 历史本地复核结果（245卡，非本轮重跑）

以下保留245卡轮次原检查记录，段内“本轮／本次”仅指当时工作树，不作为当前255卡统计或算法重跑证据；此前历史节中的轮次措辞也仅适用于各自记录，旧锚点继续有效。

2026-11-06实际检查通过：Git列出的69份非忽略仓库Markdown共4540处本地链接，去重目标1172个，含96处源码行号与13处引用式链接；四份目标文档共1222处（本页374、docs/README 264、algorithm_cards/README 496、tracking_guidance_evolution 88），DS01／BT01分别58／41处，六文件合计1321处。核验本地文件、标题／显式锚点及源码行号范围；764处显式Markdown外链只计数、未访问，裸URL不混入本地计数，不升级远端可用性或来源阅读深度。

41份正文的独立编号与245个逐卡唯一键一一对应，无缺失或重复；文件计数表和导航合计一致。前243条索引逐行保持，按原顺序以LF连接并取UTF-8 SHA-256，前后一致为 `e2dea26d35b36a35241eeacefc9bb47eac30e65d9f4ea84e8dfd985f39c09a17`。DS01／BT01共10段L1–L5正文、10组问答齐备；卡内覆盖表、逐卡索引、矩阵及跟踪回链均为核／核／部／核（限定模型代数）／部、各答5，P2且未实现。首轮编号检查漏掉不同级别标题、误纳历史交接ID，并误将GE／LG前缀当作G／L编号排除；核对正文后修正规则通过，未为消除误报修改卡片。

12份总览的14个主分支表实际为6／6／7／14／11／8／8／14／13／9／8／9，共113行；逐节点与12域汇总四态74／10／5／24闭合。方法分母89，已有74/89=83.15%，已有或部分84/89=94.38%，后者不是完整覆盖率。G01–G48原顺序保留，共33已有、10部分、5缺卡；15项剩余任务对应15个部分／缺卡节点。G47／G48由原P1缺卡转已有／P2，不表示实现完成。5项缺卡全为P0（G09／G10／G11／G36／G44）；P1缺卡0不等于P1部分清零，9项P1部分及G02的P0部分保留。G16仍仅07-01计剩余节点，N01–N24一一对应非算法节点且不变；历次计数与旧标题锚点继续保留。

四份目标文档及两卡的UTF-8、表格列数、围栏配对、显式锚点唯一性与行尾空白检查通过。系统python仍为商店入口，改用既有虚拟环境以内存命令执行文档检查，未新增脚本或依赖；`git diff --check`通过，仅有既存文件的LF／CRLF提示，未据此改文件。目标文档尚未跟踪，另作直接文件检查，不仅依赖Git差异检查。

变更范围仅 docs/evolution_coverage.md、docs/README.md、docs/algorithm_cards/README.md 及 docs/tracking_guidance_evolution.md 必要回链。Git列出的104个非目标、非忽略文件，按路径排序，将每行“路径:内容SHA-256小写十六进制”以LF连接后取UTF-8 SHA-256，前后一致为 `61d18fa9dd19071f87bd10bbf5b10fae0ddab908750adb4fda2959745867fb39`。包括两卡正文、三维及其他总览、源码、依赖与测试均未改；不覆盖忽略目录或Git元数据，不声称整个工作区无并发变化。用户原有变更未回退，未提交。DS01的34项、BT01的26项及前批有限算法检查均为历史证据，本轮未重跑；训练、模型／数据集、RTImg和物理／设备闭环均未执行。版本／权重血缘、原文截断、依赖／tie-break、空特征／gallery、IoU绕门及Removed残留／去重等证据与工程缺口继续保留。
