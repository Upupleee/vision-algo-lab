# 分割与区域：理论、选型与验收

[返回文档导航](README.md) · [预处理增强](preprocessing.md) · [标定坐标](calibration_coordinates.md)

[算法卡片总索引](algorithm_cards/README.md) · [分割31卡索引](algorithm_cards/README.md#分割区域) · [L1–L5正文](algorithm_cards/segmentation.md) · [学习组件27卡](algorithm_cards/README.md#学习组件) · [保留旧稿](algorithms/segmentation_l1_l5.md)

区域算法、算子、分类器、损失与指标分开；族级教学基线不是所有算法全史，正文覆盖不等于已实现或现场验证，本次导航更新不升级验证状态。

## 双线阅读导航

- **工程实践线（按区域输出落地）**：[任务定义](#任务定义) → [成像与数学原理](#成像与数学原理) → [方法选型表](#方法选型表) → [失败模式](#失败模式) → [现场排障](#现场排障) → [评测与验收](#评测与验收) → [当前仓库状态](#当前仓库状态)。先定义类别、实例、孔洞和连通性，再选生成区域的方法。
- **方法演进线（按问题读论文）**：[代表性演进时间线](#代表性演进时间线) → [并行分支与选型回接](#并行分支与选型回接) → [指标与算法边界](#指标与算法边界) → [论文年份核验记录](#论文年份核验记录)；可回读[传统路线](#传统路线)、[ML 路线](#ml-路线)、[DL 路线](#dl-路线)，比较不同输出协议而非只比较发表年代。

两线在标注协议和验收指标处汇合。语义、实例、交互式和提示式分割是并行问题；本页保留原文，新增研究脉络不能视为已完成实现或复现。

## 任务定义

分割为像素或区域赋予类别／实例归属，用于提取目标、缺陷、材料区域及其面积、轮廓和拓扑。必须先确定输出语义：二值前景、互斥多类、可重叠多标签、每个目标独立的实例，还是全景分割。边缘检测只输出边界证据，不自动形成闭合、带类别且可统计的区域。

任务协议应规定标签集合、忽略区、孔洞、接触目标、边界模糊区、最小目标尺寸、连通性、ROI 外行为及空前景处理。标注规则不一致造成的上限损失，通常不能靠换模型消除。

## 成像与数学原理

二值阈值可写为 $M(x,y)=1[I(x,y)\ge t]$。Otsu 方法在灰度直方图上选择阈值，使类间方差最大；它依赖直方图可分性，不理解目标语义。局部阈值让 $t=t(x,y)$，可适应缓慢光照变化，也可能把纹理和噪声分成前景。

形态学以结构元素 $B$ 操作集合 $A$：腐蚀 $A\ominus B$ 收缩前景，膨胀 $A\oplus B$ 扩张前景；开运算去除小前景，闭运算填补小间隙。结构元素把“多大、多近、什么方向算同一区域”编码进算法，因此尺寸必须与物理目标尺度对应。

连通域在选定的 4 或 8 邻域下给前景分配标签。二者会对角接触产生不同拓扑。分水岭把梯度或距离图视作地形，从标记扩张区域，适合分离粘连目标，但种子过多会过分割，种子不足会欠分割。

对二值预测 $P$ 与真值 $G$：

$$IoU=\frac{|P\cap G|}{|P\cup G|},\qquad Dice=\frac{2|P\cap G|}{|P|+|G|}.$$

二者衡量重叠但不完整表达边界、拓扑、实例拆分或极小缺陷。空集时分母处理必须预先约定，不能在实现间默默采用不同规则。

## 传统、ML 与 DL 路线

### 传统路线

- 灰度／颜色阈值：固定、Otsu、自适应或多通道规则。
- 区域生成：形态学、孔洞填充、连通域、轮廓和区域统计。
- 区域生长／洪水填充：从可靠种子按灰度或颜色容差扩张。
- 粘连分离：距离变换、标记控制分水岭、凹点或形状先验。
- 图模型：图割、GrabCut、随机游走等，以数据项和空间平滑项联合优化。

传统方法适合照明稳定、外观可分、尺度明确的工位。规则链应保留中间掩膜和每步参数，避免无法定位过分割来源。

### ML 路线

ML 可对像素、超像素或候选区域提取颜色、纹理、梯度、位置和形状特征，再用 SVM、随机森林或 GBDT 分类，之后进行空间平滑与连通域过滤。它比固定阈值能表达多维边界；在特征有效、任务受限且与从头训练的 DL 比较时，可能需要较少标注，但不能外推到预训练模型或所有任务。数据需求应以相同划分和验收协议比较；特征窗口、类别不平衡和后处理也会影响最终边界质量。

### DL 路线

语义分割为每像素分类，U-Net 等编码器—解码器融合上下文与细节；实例分割还需区分同类个体，Mask R-CNN 是检测加掩膜的代表路线。小目标、边界和类别不平衡可使用加权交叉熵、Dice 类损失、焦点损失或边界辅助监督。训练必须记录划分、增强、标签版本、权重和推理阈值；网络输出概率图不等于已经得到可验收区域。

## 方法选型表

| 场景 | 首选起点 | 升级条件 | 主要风险 | 核心指标 |
|---|---|---|---|---|
| 稳定背景、前景灰度可分 | 阈值 + 形态学 + 连通域 | 光照漂移或类别重叠 | 阈值漂移、小目标被滤除 | 前景 IoU、目标级召回 |
| 缓慢不均匀光照 | 背景补偿 + 局部阈值 | 纹理被误分 | 分块边界、噪声前景 | 假阳性面积、边界误差 |
| 接触的近圆／凸目标 | 距离变换 + 标记分水岭 | 形状差异大、种子不稳 | 过分割／欠分割 | 实例计数、拆分／合并率 |
| 多通道外观可分、数据较少 | 手工特征 + 浅层分类器 | 特征无法覆盖域变化 | 邻域不一致、泄漏 | 宏平均 F1、分层验证 |
| 复杂纹理与背景、像素标签充足 | 语义分割网络 | 同类实例必须分开 | 域偏移、小目标漏检 | 类别 IoU、边界与目标指标 |
| 同类粘连且需逐件统计 | 实例分割 | 标注成本可承受 | 掩膜重叠、检测漏召回 | AP、计数误差、合并率 |

## 失败模式

- 光照、曝光、白平衡或批次改变使颜色／灰度分布漂移，固定阈值整体失效。
- 开闭运算结构元素过大，删除小缺陷、连接相邻目标或改变面积和轮廓。
- 4/8 连通性、孔洞规则或轮廓取法不一致，区域数与面积统计不可比。
- 分水岭种子来自噪声峰导致过分割，或粘连中心只有一个种子导致欠分割。
- 像素准确率被大背景主导，模型全预测背景仍看似较高。
- 随机切 patch 让同一原图进入训练与测试，产生空间或批次泄漏。
- 标注边界厚度、遮挡、反光和“不确定区”规则不统一，误差主要反映标注差异。
- 缩放标签时使用线性插值生成非法类别；推理恢复原尺寸后边界与坐标未同步。
- 只看 IoU，忽略一个目标被拆成多个或多个目标被合并的实例错误。

## 现场排障

1. 同时保存原图、预处理图、概率／得分图、阈值掩膜、形态学中间结果和最终标签。
2. 按过分割、欠分割、漏目标、伪目标、边界偏移和类别混淆分类失败样本。
3. 核对相机与预处理版本、阈值、颜色空间、ROI、缩放、连通性和结构元素物理尺寸。
4. 观察类别分布和得分直方图；阈值变化只能在冻结验证集上选择，不能逐张追真值调参。
5. 粘连问题覆盖显示距离图与种子；检查错误来自前景掩膜、种子还是分水岭扩张。
6. DL 失败时先确认权重、归一化、通道顺序、输入尺寸和标签映射，再判断是否域偏移。
7. 按设备、批次、缺陷尺寸、位置和成像条件聚合指标，避免总体均值掩盖局部失效。

## 评测与验收

语义分割报告每类 IoU、Dice、精确率、召回率和宏平均，必要时加入边界 F 分数或容差带内边界误差。实例任务报告目标级召回／精确率、AP、计数误差，以及拆分率和合并率。面积、周长或尺寸是业务输出时，还须直接报告这些量相对可追溯真值的误差。

数据集应以原始采集单元、设备、工件批次或时间分组，禁止相邻帧和同图 patch 跨集合。验收集冻结后不得用于调阈值。报告应包含类别频数、忽略区规则、空掩膜约定、后处理参数、置信区间或重复运行离散度、最坏帧时延和内存。像素重叠达标但关键小缺陷漏检时，仍应判为业务失败。

## 当前仓库状态

- 当前没有专用区域分割实现、数据标注协议、IoU/Dice 指标、训练流程或分割验收记录。
- [subpixel_edge_net.py](../dl/subpixel_edge_net.py) 是 U-Net 风格的边缘 heat 与偏移输出骨架，含损失函数但没有完整训练循环或已验证权重；边缘输出不能视为语义或实例区域。
- [edge_subpixel.py](../traditional/subpixel/edge_subpixel.py) 和 [caliper.py](../traditional/caliper_measurement/caliper.py) 提取边缘或拟合几何，不生成带类别的闭合区域。
- [pose_ncc.py](../traditional/template_matching/pose_ncc.py) 接受二值模板掩膜来限定相关计算支持域；它不从图像预测分割掩膜。
- ML 目录目前只有 HOG + SVM 匹配候选重排序示例，没有像素／区域分类实现。现有 [metrics.py](../tests/metrics.py) 也没有分割专用指标。

## 权威官方与原始参考

- OpenCV 官方：[Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)、[Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)、[Connected Components API](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)、[Watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)。
- Nobuyuki Otsu, [A Threshold Selection Method from Gray-Level Histograms](https://doi.org/10.1109/TSMC.1979.4310076), IEEE Transactions on Systems, Man, and Cybernetics, 1979。
- Ronneberger, Fischer and Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), 2015。
- He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870), 2017。
- Dice, [Measures of the Amount of Ecologic Association Between Species](https://doi.org/10.2307/1932409), 1945；Jaccard, [Étude comparative de la distribution florale](https://www.biodiversitylibrary.org/page/43951746), 1901。
- scikit-image 官方：[Evaluating segmentation metrics](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_metrics.html)，说明过分割与欠分割需要区别评估；本仓库未依赖 scikit-image。

## 代表性演进时间线

核验日期：2026-11-04。以下 7 个算法／模型里程碑是代表性取样，不是分割全史，不用 IoU、Dice 等指标凑里程碑。按正式会议／期刊年排列，预印本首投年另列；同年的 FCN 与 U-Net 作为不同设计侧重点并列，不作先后替代断言。表内改进描述机制；代价和场景是工程归纳，未在本仓库进行性能对比。

| 年份／方法与分支 | 所解决的问题 | 代表性改进 | 代价与限制 | 适用场景 | 原始来源 | 本仓库状态 |
|---|---|---|---|---|---|---|
| 1979 · Otsu（全局阈值） | 手工阈值缺少统一的选择准则 | 根据灰度直方图最大化类间方差，自动选阈值 | 不理解类别与空间结构；光照梯度、直方图重叠或极小前景会影响结果 | 成像受控、前景背景灰度可分的二值基线 | [cite:1](#seg-source-1) | 未接入 Otsu 区域管线，无分割协议或测试 |
| 1991 · Vincent–Soille 分水岭（地形区域） | 数字图像分水岭计算需高效实现 | 用浸水模拟和队列组织地形泛洪计算 | 局部极小值导致过分割；工程标记控制依赖种子，区域不自带语义类别 | 与距离变换和可靠标记组合分离粘连对象 | [cite:2](#seg-source-2) | 未实现分水岭、距离图种子或拆分／合并率测试 |
| 2004 · GrabCut（交互图模型） | 仅颜色或仅边缘信息难得到可靠前背景 | 迭代估计颜色模型并作图割，减少所需人工交互 | 需要框选／修正等先验；颜色相近和复杂边界困难，迭代有成本 | 辅助标注、交互式前景提取，不直接等同无人值守多类分割 | [cite:3](#seg-source-3) | 未实现交互工具或 GrabCut 接入 |
| 2015 · FCN（稠密语义预测；预印本 2014） | 图像级分类网络不能直接给出对应尺寸的像素类别 | 全卷积转换、上采样和跳跃融合，端到端像素监督 | 需要像素标签和训练资源；下采样导致的边界、小目标损失仍需处理 | 有代表性像素标签的多类语义分割基线 | [cite:4](#seg-source-4) | 未实现 FCN，无语义分割训练和权重 |
| 2015 · U-Net（编码器—解码器） | 有限标注下同时获取上下文与精细定位困难 | 对称扩张路径结合跳跃连接，并强调数据增强和局部定位 | 多尺度特征占用内存；少量图像成功不等于无需代表性标注；域偏移仍在 | 生物医学及局部结构明确的稠密区域分割 | [cite:5](#seg-source-5) | 仅有 U-Net 风格边缘网络骨架；不是原论文区域分割复现，无分割权重／验收 |
| 2017 · Mask R-CNN（检测式实例分割） | 同一类别的多个个体需要分别输出掩膜 | 在检测框架中增加并行掩膜分支，用 RoIAlign 改善特征对齐 | 实例标注和检测计算成本；漏检会导致实例掩膜缺失，分辨率限制细节 | 逐件计数、遮挡目标和需要实例身份的区域统计 | [cite:6](#seg-source-6) | 未实现检测—掩膜管线，无实例数据或 AP 测试 |
| 2023 · SAM（提示式分割） | 每个新任务单独训练分割器成本高 | 以大规模掩膜数据和提示式接口支持可迁移的掩膜生成 | 需要提示或候选筛选；图像编码和多候选开销，领域偏移与微小缺陷边界须验证 | 交互标注、掩膜提议与领域适配起点 | [cite:7](#seg-source-7) | 未接入 SAM，无权重、提示协议、候选筛选或工业验收 |

## 并行分支与选型回接

- **灰度／几何区域**：Otsu 提供前景阈值，分水岭处理地形分区；可组合成前景掩膜、距离图、种子和分水岭链，而不是“1991 算法淘汰 1979 算法”。1991 条目是高效数字实现的代表论文，不是分水岭思想的首次提出。[cite:1](#seg-source-1) [cite:2](#seg-source-2)
- **交互与提示**：GrabCut 的颜色／图优化与 SAM 的预训练提示模型在标注工具中可并存；两者都需要明确输入先验与输出筛选。SAM 的掩膜不会自动变成工厂缺陷类别或尺寸真值。[cite:3](#seg-source-3) [cite:7](#seg-source-7)
- **语义分支**：FCN 与 U-Net 都将上下文和局部信息用于稠密预测，网络结构和训练侧重点不同；U-Net 不是“FCN 之后所有任务的必选替代”。[cite:4](#seg-source-4) [cite:5](#seg-source-5)
- **实例分支**：Mask R-CNN 解决个体身份和掩膜联合预测，不是单纯把语义分割年份往后延。传统分水岭也可在合适形状约束下分离实例；选择取决于标签、遮挡和计数需求。[cite:2](#seg-source-2) [cite:6](#seg-source-6)
- **手工特征 ML 分支**：像素／区域分类器可输出得分，再接空间平滑和连通域；它与 DL、传统后处理可组合。未把某一通用分类器的发表年冒充“区域分割首创年”。回到[方法选型表](#方法选型表)，用[评测与验收](#评测与验收)统一输出协议后才比较。

## 指标与算法边界

Jaccard／IoU、Dice 是集合重叠指标；用其可微近似构造损失，也不能把指标原始年代当作神经分割算法的年代。Dice 的生态学论文为 **1945**；原文列出的 Jaccard **1901** 年书目信息得到后续学术文献支持，但本次 BHL 原文链接遇到安全验证，未完成原文页级核验，因此不将其列为已核验算法里程碑。[cite:8](#seg-source-8)

IoU／Dice 对语义掩膜重叠、AP 对实例匹配、拆分／合并率对拓扑错误各有侧重；不能跨协议直接排名。分水岭区域号不等于类别，SAM 掩膜不自带业务标签，边缘 heat／偏移也不等于闭合区域。应按[任务定义](#任务定义)固定协议，且不得把“输出了掩膜”直接等同“缺陷检测已验收”。

## 论文年份核验记录

`[CONFIRMED]` 表示原始出版记录支持所列年份与版本，不表示三方独立复现或性能结论；只有单一原始记录支持的年份标 `[SINGLE-SOURCE]`。预印本与会议原稿是同一研究的不同版本，不构成独立实验验证。只确认所列代表论文，不用会议年推断整个领域的发明时间；搜索摘要中混入的修订、入库日期以原始记录区分。

- <a id="seg-source-1"></a>**[cite:1] [SINGLE-SOURCE]** Otsu, *A Threshold Selection Method from Gray-Level Histograms*，**IEEE Transactions on Systems, Man, and Cybernetics 1979，9(1):62–66**；[原文 DOI](https://doi.org/10.1109/TSMC.1979.4310076)及对应 [IEEE 书目索引](https://xplorestaging.ieee.org/document/4310076/keywords)确认 January 1979。该记录针对所列英文期刊论文，不推断作者所有早期工作的首发时间。
- <a id="seg-source-2"></a>**[cite:2] [CONFIRMED]** Vincent and Soille, *Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations*，**IEEE TPAMI 1991，13(6):583–598**；[原论文](https://people.cmm.minesparis.psl.eu/users/marcoteg/cv/publi_pdf/MM_refs/vincent/1991_PAMI_VincentSoille_watershed.pdf)与[出版记录](https://dl.acm.org/doi/10.1109/34.87344)。不能把这篇数字实现论文视作所有标记控制变体或 OpenCV 实现的同义词。
- <a id="seg-source-3"></a>**[cite:3] [CONFIRMED]** Rother, Kolmogorov and Blake, *GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts*，**ACM TOG／SIGGRAPH 2004**；[作者机构记录](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/)与[原稿镜像](http://www.cad.zju.edu.cn/home/gfzhang/training/Matting/GrabCut%20(sig04).pdf)。TOG 与 SIGGRAPH 是这里对应的发表形式，不是相隔两年的两代算法。
- <a id="seg-source-4"></a>**[cite:4] [CONFIRMED]** Long, Shelhamer and Darrell, *Fully Convolutional Networks for Semantic Segmentation*；[arXiv v1](https://arxiv.org/abs/1411.4038)首投 **2014-11-14**，[正式论文集](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)为 **CVPR 2015**。因此不是“CVPR 2014”，也不把后续期刊版本混入本条。
- <a id="seg-source-5"></a>**[cite:5] [CONFIRMED]** Ronneberger, Fischer and Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*；[arXiv v1](https://arxiv.org/abs/1505.04597)为 **2015-05-18**，[作者项目记录](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)明确 **MICCAI 2015，LNCS 9351:234–241**。不是仅因 arXiv 编号而认定会议年。
- <a id="seg-source-6"></a>**[cite:6] [CONFIRMED]** He et al., *Mask R-CNN*；[arXiv v1](https://arxiv.org/abs/1703.06870)为 **2017-03-20**，[正式会议页](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)为 **ICCV 2017**。arXiv v3 的 **2018** 是修订年，不改写会议年。
- <a id="seg-source-7"></a>**[cite:7] [CONFIRMED]** Kirillov et al., *Segment Anything*；[arXiv v1](https://arxiv.org/abs/2304.02643)为 **2023-04-05**，[正式论文集](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)为 **ICCV 2023**。出版数据库后续入库日期不是会议发表年份；本条只讨论原始 SAM，不混入后续版本。
- <a id="seg-source-8"></a>**[cite:8] 指标旁注，不计里程碑。** Dice, *Measures of the Amount of Ecologic Association Between Species*，**Ecology 1945，26(3):297–302**，由[出版方记录](https://esajournals.onlinelibrary.wiley.com/doi/10.2307/1932409)确认 `[SINGLE-SOURCE]`。Jaccard 的 1901 年书目可由[后续学术论文的参考文献](https://academic.oup.com/ismej/article/11/3/791/7537961)核对；[原文链接](https://www.biodiversitylibrary.org/page/43951746)本次仅返回安全验证页面，标记 `[INSUFFICIENT DATA：原文页级核验未完成]`，保留原引用而不宣称原文已读。

仓库状态经 `traditional/`、`ml/`、`dl/`、`tests/` 文件清单和源码核对。[subpixel_edge_net.py](../dl/subpixel_edge_net.py) 的 `EdgeNet` 输出边缘 heat 与偏移，而非类别／实例掩膜；[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 只消费匹配支持掩膜；[metrics.py](../tests/metrics.py) 没有区域 IoU／Dice／AP。上述“未实现”指未发现专用方法接入与验证链，不是外部算法不可获得；本次未训练、下载权重或声称复现论文精度。
