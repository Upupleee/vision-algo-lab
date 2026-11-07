# FR01 Faster R-CNN：RPN、RoIPool 检测头与版本训练边界

[学习规范](../algorithm_learning_standard.md) · [覆盖台账 G23](../evolution_coverage.md) · [分类装配总览 E05](../classification_assembly_evolution.md) · [Mask R-CNN S23](segmentation.md#s23-mask-r-cnn) · [RoIAlign S22](segmentation.md#s22-roialign)

<a id="fr01"></a>
<a id="g23"></a>

## 任务、演进与覆盖状态

**任务／输入输出：**有监督的二维轴对齐目标检测。训练输入是图像、每个目标的框和类别；推理输入只有图像及冻结的预处理／权重／阈值协议，输出原图坐标的框、类别和分数，允许空集合。RPN 的 objectness 是前景／背景监督，不是最终类别概率，也不是无需框标签的无监督发现。检测框不直接给出实例掩膜、朝向、物理尺寸、槽位关系或装配合格性。[cite:1](#source-1)[cite:2](#source-2)

**适用前提：**类别定义与框标注一致，漏标、difficult／crowd 的处理有明确协议；预训练和部署图像之间具有可迁移信息，目标在采样分辨率上仍可辨。部署若需要未知类别拒识、旋转框或工业计量，必须另建任务与验收，不能由 softmax、四个回归量或小数坐标推断能力。

**演进定位：**R-CNN 逐候选卷积成本高 → Fast R-CNN 共享整图特征但仍依赖外部候选 → Faster R-CNN 用 RPN 学习候选并共享卷积 → 额外付出 anchor 设计、匹配采样、框监督和多任务训练成本 → 小目标、拥挤遮挡、NMS 删真框、域迁移仍可能失效。它改进候选瓶颈，不把整条检测链变成免费，也不意味着全面淘汰传统定位方法。[cite:1](#source-1)[cite:3](#source-3)

核验日期：**2026-11-05**。本文件只有 `faster_rcnn.md:FR01` 一张模型卡，承接 **G23／07-05／E05**；RPN、RoIPool、两种训练方案是该卡组成，不另计卡。主对象为 **2015-06-04 arXiv v1／NIPS 2015** 与 **2016-01-06 arXiv v3 扩展技术稿**。作者 Python 重实现固定为 `781a917b378dbfdedb45b6a56189a31982da1b43`（2018-01-23），不是2015发布快照；依赖的 Caffe 子模块固定为 `0dcd397b29507b8314e252e850518c5695efbb83`。原 MATLAB 实现未运行、未逐行复核；作者明确提示两者有实现差异，模型不能直接互换。[cite:2](#source-2)[cite:4](#source-4)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：原版候选—检测、anchor 标签与采样、四步交替／近似联合链；限文献和源码静态对应 |
| L2 机制 | 已核验：共享特征、两级监督、RoIPool、筛选与梯度边界，附删除步骤的后果 |
| L3 工程权衡 | 部分：有单位、历史配置、排障及调参顺序；无真实检测数据定参 |
| L4 数学推导 | 部分：回归参数化、两级目标、链式法则和有限边界推导已展开；不构成全模型收敛／泛化证明 |
| L5 实现 | 部分：固定源码、布局、伪代码与有限标量／集合核算；本仓库未实现 Faster R-CNN |
| 面试自测 | 答 5：逐层问题、参考答案、条件、易错点与证据入口 |
| 验证 | 文档、来源和有限数学／分支验证；无网络前反向、权重加载、训练、AP 或速度实测 |

已检查本地 `dl/`，未发现 Faster R-CNN／RPN 检测器定义；已有学习组件、S22 和 S23 不能替代本卡，也不因本卡新增而升级状态。下文参数中的“默认”均带版本范围；论文报告、本卡核算与未来验收三者分账。

## L1 步骤：从图像到框及训练闭环

### 1. 输入、共享特征和 anchor 网格

以原版 VGG16 路线为具体主例：图像转换为浮点、按权重协议减均值，单尺度 resize；作者 Python 配置使用 BGR 均值 `(102.9801,115.9465,122.7717)`、短边600 px、长边不超过1000 px，缩放因子与实际高宽放入 `im_info=[H,W,scale]`。特征布局为 `[1,512,Hf,Wf]`，VGG 到 `conv5_3`，不先做原分类器的 pool5；总步长16 px。ZF 的共享层数与 RPN 隐通道不同，不把 VGG 的512通道泛化给所有骨干。[cite:2](#source-2)[cite:7](#source-7)[cite:10](#source-10)[cite:12](#source-12)

RPN 在共享特征上做 `3×3 conv → ReLU`，VGG 隐通道512、ZF为256；两个并行 `1×1 conv` 输出每位置 `2A` 个 objectness logits 与 `4A` 个偏移量。原文 `A=9`，由3种尺度、3种比例组成；概念尺度面积为 `128²、256²、512² px²`，比例为1:1、1:2、2:1。每种 anchor 类型有自己的输出参数，在空间位置间共享，不是九套独立 backbone。[cite:1](#source-1)[cite:2](#source-2)

作者代码从 inclusive 基框 `[0,0,15,15]` 生成 anchors：中心7.5，先枚举 `ratios=[0.5,1,2]`（这里是 `h/w`），按 `round(sqrt(256/r))` 求宽、再求高，乘无量纲 `scales=[8,16,32]`；顺序为比例优先、尺度次之。非方形 anchor 经取整后面积不必严格等于概念面积。位置 `(y,x)` 加平移 `(16x,16y,16x,16y)`，共有 `Hf×Wf×A` 个 anchor；这个数量不等于论文损失的 `Nreg`。[cite:5](#source-5)

### 2. RPN 匹配、忽略与正负采样

训练标签针对 **原 anchor** 与 GT 的 IoU，而非网络刚解码的 proposal；为每个 anchor 保存匹配 GT 索引。流程如下。[cite:1](#source-1)[cite:6](#source-6)

1. 在真实 resize 图像范围内筛出完整落入图内的 anchors；默认 `allowed_border=0`，跨界 anchor 标签为 `-1`，不贡献损失。不能先裁 anchor 再匹配，那会改变参考框和监督。
2. 对有效 anchors 与 GT 求 IoU 矩阵 `M[i,j]`；每 anchor 取最大值及对应 GT，每 GT 求最大 IoU。
3. 每 GT 的所有并列最高 IoU anchors 标正；另外高重叠 anchors 标正。**论文措辞为 `IoU>0.7`，固定 Python 源码用 `IoU>=0.7`**；复刻代码时等号必须遵从源码。对非正且与所有 GT 的 IoU 均 `<0.3` 的 anchor 标负，其余忽略。
4. 固定配置 `RPN_CLOBBER_POSITIVES=False`：先标低 IoU 背景，再让最高匹配／高阈值正样本覆盖背景。这样低于0.3但属于某 GT 最佳匹配的 anchor 仍可为正；并非将低 IoU 一律标负。
5. 在正样本中无放回随机取最多128个，其余重置 `-1`；从负样本取最多 `256−实际正样本数` 个。正负“至多1:1”指正样本不超过预算一半，不要求每图恰好128正、128负；候选不足时源码不会凭空补齐256个。
6. 只对采样后正样本启用4维回归权重；负样本只做 objectness，忽略样本两项均不计。若某 anchor 同时是多个 GT 的最佳，其回归目标仍取该 anchor 行上的最大 IoU GT；并列取值顺序需冻结。

**最高匹配不是每 GT 永久保送一个正样本。**并列最高可能给一个 GT 多个正 anchors；之后随机限额又可能删掉某 GT 的全部正 anchors。若某 GT 与所有有效 anchors 都零重叠，固定源码的并列比较会把全体列最高标正，再限额；这不证明其监督质量可用。空 GT 或没有任何有效 anchor 时，源码 `argmax` 路径不能直接照搬，应采用 L5 显式守卫，不能宣称原实现在这些输入上已可靠。[cite:6](#source-6)

### 3. Proposal 解码、裁剪、筛选和 NMS

推理在全部位置的 anchors 上预测，包括跨图边界的 anchors；训练时 ProposalLayer 也可生成全网格候选，别把“RPN loss 忽略跨界 anchor”误写成“候选层完全不看这些 anchor”。每个 anchor 用预测偏移解码成框，再按以下次序处理。[cite:2](#source-2)[cite:7](#source-7)

| 顺序 | 操作 | 固定 Python 配置／边界 |
|---|---|---|
| 1 | 解码偏移 | 对同序 anchor 应用中心平移和指数宽高；旧版端点细节见 L5 |
| 2 | clip proposals | x 限制到 `[0,W−1]`，y 到 `[0,H−1]`，不是限制到 padded batch 画布 |
| 3 | 最小尺寸过滤 | inclusive 宽高均 `>=16×scale`；16是原图 px，比较在 resize 坐标 |
| 4 | objectness 降序并截 pre-NMS | 近似联合训练12000、普通测试6000 |
| 5 | 类别无关 greedy IoU NMS | 阈值0.7；纯 Python 基线抑制 `IoU>阈值`，等号保留 |
| 6 | 截 post-NMS | 近似联合训练2000、普通测试300，候选不足不补框 |

NMS 每次保留当前最高分，删除与它过度重叠的其余框，直到候选为空；Top-N 正数为上限，所选源码非正数表示不截断，不是返回空。NMS 阈值与得分阈值单位不同；此处不按最终类别做 NMS。纯 Python 基线的等号规则不能只由 `config.py` 注释猜测，GPU／Cython 后端仍需专项等价测试。[cite:7](#source-7)

上述12000／6000不是全部训练方式的共同值：作者四步交替脚本生成离线候选时显式设置 `pre=-1`、`post=2000`，再供下一阶段检测训练使用。原文“训练检测器用约2000 proposals”也不是指2000个 RoI 全部进入每次损失。[cite:12](#source-12)

### 4. RoIPool 与第二阶段检测头

训练检测头时对 proposals **重新匹配** GT，不沿用 RPN anchor 标签。Fast R-CNN 原文使用两图共128个 RoI、每图64、最多25%前景：最大 IoU `>=0.5` 为真实前景类，`[0.1,0.5)` 为背景。固定 Python 近似联合配置改为单图，ProposalTargetLayer 把 GT 框追加到候选以供训练采样，取至多128个 RoI、至多32正，并把背景下界覆盖成 **`0.0`**，即 `[0,0.5)`；不要只读全局默认0.1。追加 GT 只在训练，推理不能用真值补候选。[cite:3](#source-3)[cite:8](#source-8)[cite:10](#source-10)

| 检测头步骤 | 输入 → 输出 | 约束 |
|---|---|---|
| RoIPool | 共享 `conv5_3` 与 `[R,5]` 的 `(batch,x1,y1,x2,y2)` → `[R,512,7,7]` | 坐标乘1/16；框端点 round，bin 边界 floor／ceil，逐通道 max |
| fc6／fc7 | flatten → 4096 → 4096 | 各有 ReLU，训练 dropout=0.5；推理关闭 dropout |
| 分类分支 | `[R,4096]` → `[R,K+1]` | 0为背景，其他为业务类别，softmax 跨类别归一化 |
| 框分支 | 同一特征 → 类别相关4维回归 | 原文表述4K；固定 VOC 图输出84，即4×21含背景槽，背景槽不监督、不输出 |

空 bin 输出0、argmax=-1；一个输入特征若被多个 bin／RoI 选为最大值，梯度累加，不是最后一次覆盖。RoIPool 内部把畸形尺寸强制为至少1格是旧内核行为，不等于输入非法框可不校验。[cite:10](#source-10)[cite:11](#source-11)

推理时 RPN scores 只用于候选筛选；检测头按每个前景类别输出分数和该类框回归。所选 `test.py` **先将 proposal 除以 scale 还原原图，再按该版本解码检测偏移并裁剪**；每类得分筛选后分别 NMS，最后可限全图输出量。不能简单取每 RoI 的 argmax 一类替代原每类处理，也不默认将 RPN 分数乘到最终类别分数。测试函数的0.05分数、0.3最终 NMS、100个全图检测是该脚本默认而非理论常数；并列分数通过阈值限量可能超过100，不应视为严格容量证明。[cite:12](#source-12)

### 5. 四步交替训练、近似联合与终止

**2015方案：**先分别适应两任务，再冻结共享层对齐专属层。阶段名中的“第二阶段检测器”和“四步训练的第二步”不是同一概念。[cite:1](#source-1)[cite:2](#source-2)

| 四步 | 初始化及候选来源 | 更新参数 | 不更新／不回传 |
|---|---|---|---|
| 1：训练 RPN | ImageNet预训练共享骨干，RPN新增层初始化 | 骨干允许微调的层＋RPN专属层 | 尚无检测损失 |
| 2：训练独立 Fast R-CNN | **另一个 ImageNet 初始化**检测网，候选来自步骤1 | 检测骨干允许微调的层＋检测头 | 不通过固定候选更新步骤1 RPN；此时两网未共享权重 |
| 3：对齐 RPN | 用步骤2检测网初始化，固定其共享卷积 | 仅RPN专属层 | 共享卷积冻结；随后重新生成候选 |
| 4：对齐检测头 | 使用步骤3候选，保留共享骨干 | 仅Fast R-CNN专属层／FC | 共享卷积冻结，RPN不受检测梯度更新 |

原文 RPN 的 PASCAL 配方为学习率0.001跑60000次、0.0001再20000次，momentum=0.9、weight decay=0.0005；ZF微调全部卷积，VGG从conv3_1起微调以节省内存。不能把“共享骨干训练”解释成所有层均未冻结。固定交替脚本四段预算为80000／40000／80000／40000次，额外候选生成是阶段间操作，不是优化迭代。[cite:1](#source-1)[cite:2](#source-2)[cite:12](#source-12)

**2016 v3近似联合：**单图一次共享 backbone 前向，RPN 产生 proposal；把 proposal／离散匹配采样视为常量交给检测头。一次 backward 合并 RPN 与检测损失对共享特征的梯度。**停止的是检测损失经过 proposal 坐标回到 RPN 的路径，不是检测损失对共享特征的直接路径。**固定作者代码的 `ProposalLayer.backward` 和 `ProposalTargetLayer.backward` 不传播梯度，RoIPool GPU backward 只写特征 bottom。v1正文把联合优化列为后续问题；v3明确讨论近似联合与非近似联合，不能倒写成2015原实验全部使用近似联合。[cite:1](#source-1)[cite:2](#source-2)[cite:7](#source-7)[cite:8](#source-8)[cite:11](#source-11)

**非近似联合边界：**v3指出还需要对框坐标可微的 RoI 运算，提及 RoI warping 并留到本文范围外。仅换 RoIAlign 并不能自动使 argmax匹配、随机采样、Top-K和NMS全可微，也不保证特定库返回框坐标梯度。它不是本卡已经实现的第三套训练器。[cite:2](#source-2)

训练停止由事先冻结的阶段预算／验证选模规则定义；NaN/Inf、坏框、无有效监督应中止或按显式策略跳过并记录。结束四步或损失下降不证明非凸模型达到全局最优。推理在有限候选处理完后结束；空结果是合法输出，不代表已证明现场“无零件”。

## L2 机制：每一步为何存在

| 机制 | 解决的困难／保持的契约 | 删除、替换或过度解读的后果 |
|---|---|---|
| 共享整图卷积 | 避免候选与检测重复提取大部分特征 | 分开两个完整骨干失去共享收益；共享不使RPN／FC零成本 |
| 多尺度／比例 anchors | 给固定局部特征提供多个回归参考 | 单一方框对细长件可能匹配差；anchor尺度金字塔不等于FPN |
| 最高匹配＋忽略区 | 阈值无正例时仍提供回归入口，避免中间重叠标签过于武断 | 删最高匹配可能无正监督；全把中间区当负会改变目标 |
| 限额随机采样 | 限制大量背景对分类损失的支配 | 全部anchors直接均值会改变类先验和梯度尺度；采样分数不是校准概率 |
| 两次框回归 | RPN做类别无关粗候选，检测头结合RoI类别特征再定位 | 删除第二次回归不再等于原检测器；两阶段的参考框／标签不同 |
| clip／最小尺寸／NMS | 合法图域、可用候选尺寸、重复抑制和算力控制 | 顺序改变可改变IoU和保留集合；拥挤真目标也可能被误抑制 |
| RoIPool＋分类／回归并行头 | 可变大小RoI变成固定特征，同时优化识别和定位 | 改成整图GAP失去实例定位；改成RoIAlign属于明确的后续变体 |
| 近似联合的特征梯度相加 | 两任务共同更新可训练共享表示 | `detach(F)`会误切检测训练；不切proposal则不能声称复刻原近似边界 |

这些机制由原文与作者固定源码共同限定，不代表三项独立实验验证。特别是“RPN漏召回”应先看候选数量与定位质量；若完全没有对应实例的可用RoI，第二阶段没有额外独立全图实例搜索来补齐。但检测回归可以把一个低IoU proposal修好，因此**最终AP或检测召回并不被同阈值的原始proposal recall简单数值上界约束**。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

## L3 工程权衡：参数、证据与调参顺序

### 参数尺度与版本差异

| 参数 | 单位／历史起点 | 调整代价与先决条件 |
|---|---|---|
| resize短边／长边 | px；Python为600／1000 | 放大小件同时增加骨干计算与anchors数量；先查原图信息是否足够 |
| feature stride | resize图 px／特征格；原ZF/VGG为16 | 减小步长增加空间激活和候选数，不是无成本提高物理精度 |
| anchor scales／ratios | 倍率／无量纲；代码8/16/32与0.5/1/2 | 用缩放后GT宽高和匹配IoU分布诊断；改变A需同步2A／4A输出及布局 |
| RPN正负阈值与预算 | IoU无量纲；0.7／0.3、256、正上限128 | 低阈值扩大正集同时降低定位监督质量；先核标注与坐标 |
| proposal最小尺寸 | 原图 px；16再乘scale | 小零件可能被提前删去；不能拿检测分数阈值补救 |
| pre／post NMS预算 | 框数；训练与测试及交替脚本不同 | 更多候选增加NMS与RoI头耗时；记录每一筛选步骤的召回变化 |
| RPN／检测NMS | IoU；分别0.7／脚本0.3 | 前者类别无关，后者每类；提高阈值通常少删重叠框，但多重复框 |
| RoI采样／bbox归一化 | 128预算、正比例0.25；偏移无量纲 | 切换训练配置要同步背景下界、损失reduction和反归一化 |

表中数值是论文或固定文件内的历史起点，不是本仓库调优所得，不构成新数据集的推荐最优值。[cite:2](#source-2)[cite:7](#source-7)[cite:10](#source-10)[cite:12](#source-12)

### 按症状定位失效环节

| 症状 | 应保留的诊断证据 | 调整顺序／回退 |
|---|---|---|
| 小螺钉漏检 | resize后GT宽高、每GT最大anchor IoU、各筛选阶段recall | 标注／分辨率 → stride与anchors → min-size／预算 → 分类；必要时另评估切片或FPN |
| 密排同类件少框 | NMS前后框与被抑制对象、每类结果 | 先区分RPN和检测NMS，再对照阈值与预算，不只降低score |
| 正样本为零／突然激增 | 图内anchor数、IoU矩阵、并列最高数量、采样前后计数 | 坐标与漏标 → anchor支持 → clobber／阈值 →采样；非法输入拒绝 |
| 框持续偏一像素或缩放后偏移 | resize因子、inclusive／连续端点、解码前后框、RoIPool量化 | 固定版本坐标链 → 黄金向量 → 再看学习问题，不先重训掩盖 |
| RPN loss正常而检测头不学习 | 四项loss、共享层／RPN头／检测头各自梯度 | 查误detach特征、标签通道、bbox归一化、冻结列表，再调学习率 |
| 框质量正常但线上慢 | backbone／proposal排序与NMS／RoIPool与FC分项耗时及尾延迟 | 先找瓶颈，再减少候选或换后端；截断改变结果须重新评估 |
| 换批次高分误检 | 困难负例、漏标、类别词表、预处理与实物分组划分 | 排泄漏和域差 → 补监督与校准 → 重新验收；高objectness不等于未知类拒识 |

验证集需按实物／批次隔离；proposal recall@N、不同IoU阈值的定位质量、逐类AP、小件／遮挡分层、漏检／重复框、端到端延迟分别报告。VOC AP与COCO多IoU AP不是同一个数；GPU型号、输入尺寸、候选数、预热和包含的阶段不一致时，不复用论文速度作本站承诺。

### 原版、FPN、RoIAlign 与 Mask 分开

| 对象 | 与本卡主版的关系 | 不能混同的内容 |
|---|---|---|
| 原ZF／VGG Faster R-CNN | 单尺度图像、单层共享特征、多尺度anchors、RoIPool、两级框预测 | 原版不要求ResNet、FPN或RoIAlign |
| Faster R-CNN＋FPN | 后续特征提取扩展；自顶向下与横向连接生成多尺度语义特征 | 不是原版3种anchor尺度；跨层anchor／RoI分配须另核，不能直接沿用单层布局 |
| RoIAlign | 2017 Mask R-CNN提出的去量化对齐，用连续位置双线性采样 | 不等于FPN，也不保证所有筛选可微或恢复丢失的图像细节 |
| Mask R-CNN | 在检测框架增加并行mask分支，配RoIAlign | 本卡不输出mask、不训练mask损失，不升级S23实现状态 |

FPN证据限其CVPR 2017官方摘要与所读Mask R-CNN结构说明；本卡不宣称展开了FPN全文训练配方。Mask R-CNN原文§3明确对照RoIPool的框／bin量化与RoIAlign的连续采样。[cite:13](#source-13)[cite:14](#source-14)

## L4 数学推导：目标、导数与有限保证

### 1. IoU、回归参数化与反解

设合法框面积为正，同一坐标约定下

\[
\operatorname{IoU}(a,g)=\frac{|a\cap g|}{|a|+|g|-|a\cap g|}.
\]

inclusive像素端点的宽为 `x2−x1+1`，连续边界的宽为 `x2−x1`；交集、面积、编码、解码与测试必须成套，不能只在IoU里随意加1。若参考框宽高为正，GT宽高也为正，令中心宽高为 \((x_a,y_a,w_a,h_a)\) 与 \((x^*,y^*,w^*,h^*)\)，用参考尺寸归一化平移、对尺寸比取log：[cite:1](#source-1)[cite:9](#source-9)

\[
t_x^*=\frac{x^*-x_a}{w_a},\quad t_y^*=\frac{y^*-y_a}{h_a},\quad
t_w^*=\log\frac{w^*}{w_a},\quad t_h^*=\log\frac{h^*}{h_a}.
\]

消去归一化及log得

\[
\hat x=x_a+w_a t_x,\quad \hat y=y_a+h_a t_y,\quad
\hat w=w_a e^{t_w},\quad \hat h=h_a e^{t_h}.
\]

在一致的中心宽高模型中，将图像和框同时平移，编码不变；连续模型下共同正比例缩放也不变。固定inclusive端点直接按scale缩放时额外的 `+1` 不按比例缩放，不可据抽象公式断言旧代码严格尺度等变。宽高非正时log非法；任意实数偏移虽数学上产生正宽高，有限精度下指数会溢出或下溢。

对固定anchor求导：\(\partial\hat x/\partial t_x=w_a\)、\(\partial\hat w/\partial t_w=\hat w\)。因此相同中心偏移误差在大anchor上产生更大像素误差；相同log宽度误差表现为相同相对尺度误差。若仅偏移受扰动 \(|\delta t_x|\le\epsilon_x, |\delta t_w|\le\epsilon_w\)，未裁剪的连续模型有

\[
|\delta\hat x|\le w_a\epsilon_x,\qquad
|\delta\hat w|\le\hat w(e^{\epsilon_w}-1),\qquad
|\delta x_{edge}|\le w_a\epsilon_x+\tfrac12\hat w(e^{\epsilon_w}-1).
\]

这是**固定参考框、精确指数和同一几何分支的确定性传播界**，不是框相对GT的误差保证，更不覆盖NMS换序、错分类、浮点溢出或标注误差。单坐标clip到固定区间是1-Lipschitz，不放大该坐标差；但clip可能改变IoU与排序后的抑制集合，所以输出集合不因此连续。

### 2. RPN 分类与回归损失

令 \(S\) 为采样后的非忽略anchor集合，\(y_i\in\{0,1\}\)，\(P=\{i\in S:y_i=1\}\)。两类softmax可用logit差 \(z_i=z_{fg,i}-z_{bg,i}\) 写成 \(p_i=\sigma(z_i)\)。Bernoulli负对数似然为

\[
\ell_{obj}(z,y)=\log(1+e^z)-yz,
\quad \partial_z\ell_{obj}=\sigma(z)-y.
\]

实现用稳定softplus或log-sum-exp，不能先算溢出的exp再取log。忽略标签不是第三个类别；只在S内归约。论文 Eq.(1) 为

\[
L_{RPN}=\frac1{N_{cls}}\sum_{i\in S}\ell_{obj}(z_i,y_i)
+\frac\lambda{N_{reg}}\sum_{i\in P}\sum_{d\in\{x,y,w,h\}}\rho(t_{id}-t^*_{id}).
\]

论文描述 `Ncls=256`、`Nreg≈2400`为特征网格anchor locations数、`λ=10`；`Nreg`不是正anchor数、也不是把位置再乘9的总anchor数。标量回归项是对4坐标求和；负／忽略不回归。[cite:1](#source-1)[cite:2](#source-2)

原文引用Fast R-CNN的Smooth L1，广义实现记为

\[
\rho_\beta(r)=
\begin{cases}r^2/(2\beta),&|r|<\beta\\|r|-\beta/2,&|r|\ge\beta,\end{cases}
\qquad
\rho_\beta'(r)=\begin{cases}r/\beta,&|r|<\beta\\\operatorname{sign}(r),&|r|\ge\beta.\end{cases}
\]

\(\beta>0\) 时函数及一阶导数在两断点连续，导数幅值不超过1；这只限制对**回归输出残差**的导数，不约束乘网络Jacobian后的全部参数梯度。Fast R-CNN原式为 \(\beta=1\)；固定Caffe用 \(\beta=1/\sigma^2\)，RPN图设置 `sigma=3`即 \(\beta=1/9\)，检测头默认sigma=1。[cite:3](#source-3)[cite:10](#source-10)[cite:11](#source-11)

**论文目标与固定Python实现不能写成相同常数。**该实现RPN inside权重只在正样本为1，outside权重在有效采样样本为 `1/|S|`；损失层再除 `bottom[0].num()`，此RPN张量batch为1，外部loss_weight=1。因此其有效回归项为

\[
L^{py}_{RPN,reg}=\frac1{|S|}\sum_{i\in P,d}\rho_{1/9}(t_{id}-t^*_{id}),
\]

不是直接再乘10／除2400，也不是除 `4|P|`。分类图忽略-1并按有效项归一。两种配方都应单列，不能用论文“对λ不敏感”宣称逐值或梯度等价。零有效样本时上述分母没有定义，必须在求loss之前守卫。[cite:6](#source-6)[cite:10](#source-10)[cite:11](#source-11)

### 3. 检测头目标与bbox归一化

RoI的真实类别为 \(u\in\{0,\ldots,K\}\)，softmax为 \(q\)，框预测分支为 \(v^k\)，目标 \(v^*\) 相对 **proposal** 编码。原Fast R-CNN每RoI目标为

\[
\ell_{det}=-\log q_u+\lambda_{det}[u\ge1]\sum_d\rho_1(v_d^u-v_d^*),\quad \lambda_{det}=1.
\]

背景不回归；前景只训练真实类的4个坐标，不把全类别回归通道都拉向同一个框。CE logits导数为 \(q_c-[c=u]\)。采样RoI批量平均与“只对正RoI平均回归”不是同一reduction；固定图的bbox损失按RoI数量归约，未激活槽位权重为0。[cite:3](#source-3)[cite:8](#source-8)[cite:11](#source-11)

固定end2end配置使用 \(\tilde v^*=(v^*-\mu)/s\)，\(\mu=(0,0,0,0)\)、\(s=(0.1,0.1,0.2,0.2)\)。**这些是检测RoI回归目标的归一化，不用于此版RPN目标。**推理若网络输出标准化偏移，需要 \(v=s\odot\tilde v+\mu\)。作者 `snapshot()` 将此变换折叠进 `bbox_pred` 权重和偏置再保存、随后恢复训练参数；使用该导出权重再手工反归一化会重复应用。若原FC为 \(Wh+b\)，导出参数是 \(\operatorname{diag}(s)W\) 和 \(s\odot b+\mu\)，逐元素代入即可验证等价。[cite:9](#source-9)[cite:10](#source-10)[cite:12](#source-12)

### 4. RoIPool和近似联合的梯度边界

对固定RoI与bin支持 \(B_{rj}\)，\(Y_{rj}=\max_{i\in B_{rj}}F_i=F_{a(r,j)}\)，故

\[
\frac{\partial L}{\partial F_i}=\sum_{r,j}[i=a(r,j)]\frac{\partial L}{\partial Y_{rj}}.
\]

这是固定最大值选择分支下的链式法则；并列最大采用内核的确定argmax子梯度，非所有最大项均分。量化后的RoI支持随框坐标分段常量，在切换边界不光滑，不能借此声称对候选几何已有有用的完整训练导数。[cite:3](#source-3)[cite:11](#source-11)

令共享特征 \(F=f_\theta(I)\)，RPN输出 \(r=g_\phi(F)\)，筛选后框 \(b=S(r)\)，检测损失 \(D(F,b,\psi)\)。若假设存在可微的候选与坐标采样替代，完整链形式应含

\[
\nabla_\theta L=J_f^T\left(\nabla_F L_{RPN}+\partial_FD
+J_g^T J_S^T\partial_bD\right).
\]

该式的可微假设**不适用于原离散S全域**。原近似联合实际优化的局部图使用 `stop_gradient(b)`，所以只保留前两项；RPN专属参数 \(\phi\) 只收到RPN损失梯度，检测专属参数 \(\psi\) 只收到检测损失梯度，允许训练的共享参数收到两者相加。四步训练的第3／4步还冻结 \(\theta\)，不把“冻结”混成“共享层梯度一定为零的数学定理”。[cite:2](#source-2)[cite:7](#source-7)[cite:11](#source-11)

### 5. 复杂度、终止与不可保证项

设网格位置数 \(M=H_fW_f\)、anchor数 \(A\)、GT数 \(G\)、pre-NMS保留 \(Q\)、RoI数 \(R\)。显式IoU匹配需 \(O(MAG)\) 时间和矩阵存储，分块可降峰值但必须维护每行及每列最大与并列。RPN小卷积开销约为 \(O(M(9CD+6AD))\)，C是输入通道、D是RPN隐通道；骨干另计。排序 \(O(MA\log(MA))\)，朴素NMS最坏 \(O(Q^2)\)。VGG两层FC每RoI主要为 `512×7×7×4096+4096×4096` 乘加，故增加proposal预算会直接放大检测头开销。

NMS每轮至少移除被选中的一个候选，所以有限Q下最多Q轮终止；保留集合在所选 `IoU>τ` 抑制规则下两两IoU不超过τ，但这是排斥性质，不证明目标召回最优、最大权独立集或拥挤场景正确。固定候选集和分数时，增加post-NMS前缀长度使保留前缀包含旧集合；改变pre-NMS范围、阈值、回归或分数并无同样整体性能保证。

本卡**未提供全检测器通用理论误差界**；没有全局收敛、工业漏检率或泛化保证。宽高指数敏感性、clip的局部不放大、NMS有限终止与Smooth L1输出导数有界，均不能替代真实数据验收。

## L5 实现：固定源码、参考伪代码与边界验证

### 1. 固定源码对应和坐标契约

| 责任 | 固定源码入口 | 必须检查的具体行为 |
|---|---|---|
| anchor枚举／匹配 | `lib/rpn/generate_anchors.py`、`anchor_target_layer.py` | ratio-major顺序、图内筛选、并列最高、正覆盖负、随机限额与unmap [cite:5](#source-5)[cite:6](#source-6) |
| 候选筛选 | `lib/rpn/proposal_layer.py`、`lib/nms/py_cpu_nms.py` | 前景取后A通道，解码→clip→min-size→Top-K→NMS，backward为空 [cite:7](#source-7) |
| RoI目标 | `lib/rpn/proposal_target_layer.py` | 训练追加GT、重新匹配、真实类4槽权重、单图采样 [cite:8](#source-8) |
| 编码解码 | `lib/fast_rcnn/bbox_transform.py` | inclusive宽高、旧端点反解、clip [cite:9](#source-9) |
| 网络与配置 | VGG16 `faster_rcnn_end2end/train.prototxt`、end2end.yml、config.py | 18／36通道、7×7池化、4096FC、84回归槽、配置覆盖而非只看默认 [cite:10](#source-10) |
| 算子内核 | Caffe `roi_pooling_layer.cpp/.cu`、`smooth_L1_loss_layer.cpp/.cu` | round／floor／ceil、空bin、特征梯度累加、sigma与reduction；CPU部分为NOT_IMPLEMENTED [cite:11](#source-11) |
| 阶段与导出／推理 | `tools/train_faster_rcnn_alt_opt.py`、`lib/fast_rcnn/train.py`、`test.py` | 阶段间候选、冻结策略承接、折叠归一化、还原坐标、每类NMS [cite:12](#source-12) |

预测布局要与anchor顺序逐元素对应：RPN bbox `[1,4A,Hf,Wf]` 转 `[1,Hf,Wf,4A]` 再reshape成 `[HfWfA,4]`，行顺序为 `(y,x,a)`，a最快；objectness `[1,2A,Hf,Wf]` 是前A背景、后A前景，不是每anchor两通道简单交错。标签unmap回全网格时跨界／未采样项填-1，回归权重填0；把NHWC直接当NCHW可能shape完全合法而匹配全错。[cite:6](#source-6)[cite:7](#source-7)

**旧框解码的非往返恒等必须保留。**固定 `bbox_transform` 用 `w=x2−x1+1`、`cx=x1+0.5w`；其 `bbox_transform_inv` 返回 `x1=cx−0.5w`、`x2=cx+0.5w`，右端没有减1。因此未clip时 `decode(a,0)` 的右／下端比a大1，`decode(a,encode(a,g))` 得到的右／下端也比g大1。这是该固定实现的可核算事实，不是中心宽高公式本身失效。`generate_anchors` 的中心又使用 `(w−1)/2`，两处不能不加区别地共用同一helper。[cite:5](#source-5)[cite:9](#source-9)

若复刻历史权重，就保留并标记 `legacy_py` 解码及其先还原再检测回归的顺序；若编写现代一致连续边界版本，则明确采用 `w=x2−x1`、一致中心／端点、IoU与min-size，重新验证和训练。不能在卡片里悄悄“修复”减1后仍声称与原作者逐元素一致，也不能用现代连续版零偏移恒等测试去宣布旧权重实现错误。

### 2. 参考伪代码：作者近似联合链与显式守卫

以下是**文档参考，不是本仓库网络实现**；函数名表示上文已经定义的数学／源码责任，不可直接当成安装库API。`legacy_py` 固定旧坐标，配置采用已读end2end覆盖后的值。无GT、无图内anchor、非有限值等守卫是本卡设计，区别于旧代码未防护的argmax路径；默认拒绝空GT训练批，若业务要纯负图必须另写并测试标签策略。

```text
TRAIN_STEP(image, gt, cfg, model, optimizer, rng):
    require finite(image), valid_class_ids(gt), finite_and_valid_inclusive_boxes(gt)
    if count(gt) == 0:
        return REJECT_EMPTY_GT
    I, info, G = preprocess_image_and_boxes_together(image, gt, cfg)
    F = model.backbone(I)
    Z, T = model.rpn(F)
    A = anchors_in_y_x_ratio_scale_order(shape(F), stride=16)
    require shape_and_order_agree(A, Z, T), finite(Z), finite(T)
    inside = fully_inside(A, info.height, info.width)
    if count(inside) == 0:
        return REJECT_NO_VALID_ANCHOR
    M = inclusive_iou(A[inside], G.boxes)
    match = row_argmax_first(M)
    y = fill(count(inside), -1)
    y[row_max(M) < 0.3] = 0
    y[any_column_max_tie(M)] = 1
    y[row_max(M) >= 0.7] = 1
    disable_random_excess(y == 1, limit=128, set_to=-1, rng=rng)
    disable_random_excess(y == 0, limit=256-count(y == 1), set_to=-1, rng=rng)
    S = indices(y >= 0)
    P = indices(y == 1)
    if count(S) == 0:
        return REJECT_NO_VALID_SUPERVISION
    targets = encode_legacy(A[inside[P]], G.boxes[match[P]])
    LRcls = mean_valid_two_class_CE(Z[inside[S]], y[S])
    LRbox = sum_smooth_l1(T[inside[P]]-targets, beta=1/9) / count(S)
    B = PROPOSE(A, foreground_softmax(Z), T, info, cfg.TRAIN)
    B = stop_gradient(B)
    candidates = concatenate(B, G.boxes)
    rois, u, target = match_and_sample_rois(candidates, G, fg_iou=0.5,
                                         bg_interval=[0,0.5), budget=128,
                                         max_fg=32, bbox_stds=(0.1,0.1,0.2,0.2), rng=rng)
    if count(rois) == 0:
        return REJECT_NO_VALID_ROI
    H = roi_pool(F, stop_gradient(rois), size=(7,7), spatial_scale=1/16)
    logits, V = model.detector_head(H, training=True)
    LDcls = mean_CE(logits, u)
    LDbox = sum_true_foreground_class_smooth_l1(V, target, u, beta=1) / count(rois)
    loss = LRcls + LRbox + LDcls + LDbox
    require finite(loss)
    optimizer.zero_grad()
    backward(loss)
    require finite_gradients(model.trainable_parameters)
    optimizer.step()
    return loss, sampling_counts, filtering_counts

PROPOSE(A, scores, T, info, mode):
    B = decode_legacy_py(A, T)
    require finite(B), finite(scores)
    B = clip_inclusive(B, info.height, info.width)
    keep = inclusive_width(B) >= mode.min_size * info.scale
    keep = keep AND inclusive_height(B) >= mode.min_size * info.scale
    B, scores = select_same_indices(B, scores, keep)
    order = descending_score_order_with_frozen_tie_policy(scores)
    order = prefix_if_positive_limit(order, mode.pre_nms_top_n)
    keep = greedy_nms(B[order], scores[order], threshold=0.7, suppress_if=IoU>threshold)
    keep = prefix_if_positive_limit(keep, mode.post_nms_top_n)
    return B[order[keep]]

INFER(image, exported_model, cfg):
    I, info = preprocess(image, cfg)
    F = exported_model.backbone(I)
    Z, T = exported_model.rpn(F)
    A = anchors_in_y_x_ratio_scale_order(shape(F), stride=16)
    B = PROPOSE(A, foreground_softmax(Z), T, info, cfg.TEST)
    if count(B) == 0:
        return EMPTY_DETECTIONS
    H = roi_pool(F, B, size=(7,7), spatial_scale=1/16)
    logits, V = exported_model.detector_head(H, training=False)
    require finite(logits), finite(V)
    V = unnormalize_only_if_not_folded_into_exported_weights(V)
    C = decode_legacy_py(B / info.scale, V)
    require finite(C)
    C = clip_inclusive(C, original_height(image), original_width(image))
    return score_filter_and_per_class_nms(C, softmax(logits), skip_class=0,
                                         nms_threshold=cfg.TEST.NMS)
```

伪代码只在候选坐标使用stop-gradient，保留F的检测梯度；正样本为空时回归项用与模型相连的零张量／空和，不对空集合求mean。固定tie策略属于可复现设计；旧NumPy降序排序未承诺跨后端稳定并列顺序，比较时必须单独记录该差异。推理必须同时检查解码结果有限，不能认为后clip自然修复NaN／Inf；要限制log尺度时应另设显式clamp策略并登记改变，而非冒充原实现已有防溢出。

### 3. 有限边界验证与未执行测试

本次使用PowerShell/.NET内存核算下列确定性数学与集合用例，不安装／调用Caffe或PyTorch、不落地额外测试文件。输入仅为人为指定的**单元边界向量**，不是合成检测数据集或性能实验。它们检验公式与分支，不证明作者网络已运行。

执行记录：独立重跑后 **47项数学／分支断言通过**；连同五层、五问答、14组来源、代码围栏、本地链接／锚点及空白检查共 **248项断言通过**。最初检查命令遇到PowerShell数组运算括号和跨调用函数不保留问题，修正为单次自包含命令后完成；这些是检查器问题，不是网络失败或网络通过的证据。

| 检查组 | 可复查输入／预期 | 验证边界 |
|---|---|---|
| 标签等号与优先级 | 非列最高IoU=0.7在源码规则为正、论文严格措辞不触发；0.3不为负；0.2若列最高仍为正 | 标量布尔规则，不是整张IoU矩阵匹配运行 |
| 采样预算 | 200正/300负→128/128；10/300→10/246；0/10→0/10 | 数量上限，不模拟随机数或保证每GT保送 |
| 旧版编码／解码 | anchor与GT相同，target=0；未clip解码右／下端+1 | 特意确认旧版不是严格往返；不测权重输出 |
| bbox反归一化 | 直接 `s*(Wh+b)+μ` 与折叠参数一致 | 有限标量代数，不加载导出权重 |
| Smooth L1 | β=1与1/9的0、正负断点值／导数及中心差分 | 只检标量输出导数；不替代网络自动微分 |
| proposal边界 | inclusive宽16恰好通过，宽15拒绝；clip幂等；空候选NMS返回空 | 不包含完整ProposalLayer／CUDA调用 |
| NMS | 两框IoU=阈值保留，重复框在τ<1抑制；τ=1时相同框保留 | 纯集合规则；分数固定、无并列排序争议 |
| RoIPool量化／梯度 | C++式round(0.5)=1；同一argmax被两输出选择时上游2、3累加为5；空bin零梯度 | 不编译池化内核，不声称CPU与GPU等价 |
| 近似梯度路径 | 标量 `F=θ,r=φF,D=ψF+r`：停止r后∂θD=ψ、∂φD=0；完整可微替代为ψ+φ、θ | 验证链式法则区别，不伪称实际S可微 |

**仍待执行：**逐位置anchor／logit／delta对齐黄金向量；完整匹配的GT并列与全零列；空GT／无图内anchor／空proposal守卫；低精度指数溢出；RoIPool全部bin的前反向及并列argmax；四项loss实际reduction；只启用检测loss时RPN专属参数无梯度、共享层仍有梯度的自动微分检查；四步第3／4步共享权重不变；CPU／GPU NMS等号与排序一致性；真实图像训练、proposal recall、AP与尾延迟。没有这些证据，不标记“本仓库已实现”“网络测试通过”或“现场验收”。

优化时先冻结参考输出、坐标、采样种子、并列策略与分母，再分析瓶颈。匹配分块需跨块恢复每GT并列最高；NMS换GPU／位掩码内核需核等号和排序；减少候选、改变min-size、换FPN／RoIAlign属于结果会变的算法配置，不作为无损加速。缓存只适用于同图同权重同预处理且没有训练更新的共享特征。

## 面试追问链：五问与参考答案

| 层级 | 问题 | 参考答案与成立条件 | 易错答案／反例 | 证据 |
|---|---|---|---|---|
| L1 | 一张图如何得到最终类别框？训练中哪些数据不能进推理？ | 预处理→共享特征→anchor objectness／偏移→解码clip、尺寸、Top-K、RPN NMS→RoIPool→类别／框头→原图解码→每类NMS；GT只用于训练匹配／追加，推理允许空结果 | 把2000 proposals当2000 loss样本；推理加GT；把RPN分数当最终类别 | [cite:2](#source-2)[cite:7](#source-7)[cite:12](#source-12) |
| L2 | 为什么有每GT最高IoU正例和中间忽略区？能保证每GT有训练正例吗？ | 最高匹配弥补阈值找不到正例，中间区避免模糊负监督；但受图内anchor支持和后续随机限额影响，不保证每GT保留；全零IoU并列是退化反例 | “所有IoU<0.3必为负”“每GT一定只有一个正anchor”“负anchor也回归最近GT” | [cite:1](#source-1)[cite:6](#source-6) |
| L3 | 小件和密排漏检都直接降低检测score吗？ | 先核可观测性与标签，追踪resize后尺度、anchor覆盖、min-size、各阶段候选recall及两处NMS，再对照预算／阈值；FPN是另一个明确版本，不是免费调参 | 候选已删仍调分类score；0.7同时用于标签、两处NMS；将FPN称2015必需结构 | [cite:7](#source-7)[cite:10](#source-10)[cite:13](#source-13) |
| L4 | 推导RPN回归与loss，近似联合少了哪个梯度项？ | 归一化平移／log尺寸反解，正样本Smooth L1；说明论文256／约2400／λ10与固定代码有效样本分母／β1/9不同；删的是检测损失经proposal坐标回RPN的项，保留对共享F直接梯度 | “除正例数即可”“Smooth L1防止整个网络梯度爆炸”“联合训练全部路径可微” | [cite:2](#source-2)[cite:6](#source-6)[cite:11](#source-11) |
| L5 | 零偏移框右边多1像素且训练不学习，如何查？ | 先锁inclusive端点与legacy解码，再查scale顺序／折叠归一化、y-x-a布局、fg后A通道、-1屏蔽及F误detach；用边界向量与真实自动微分分开验证 | 静默减1后宣称历史权重等价；框shape对就判正确；把CPU未实现路径说成已测试 | [cite:7](#source-7)[cite:9](#source-9)[cite:11](#source-11)[cite:12](#source-12) |

## 来源层级、版本与获取深度

本次来源按 **P0原论文／作者代码、P1权威二手、P2专业社区、P3一般资料**分级；正文依据均为P0，本地规范／台账是项目状态依据，不冒充原论文证据。论文、其arXiv修订、作者重实现属于同一研究链，**不凑成三项独立验证**。方法细节按 `[SINGLE-SOURCE]` 原始研究或固定实现契约使用；年份与路径身份可交叉核验，性能／泛化没有独立证据，未据此作排名。不同版本冲突保留并归因，不取一个数覆盖全部。

- <a id="source-1"></a>**[cite:1] P0，原版Faster R-CNN，[SINGLE-SOURCE]。** [NIPS 2015官方PDF](https://papers.nips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)已读摘要、§1–3及RPN标签／损失／优化；返回正文截断后，另取[arXiv v1正文](https://arxiv.org/html/1506.01497v1)的共享卷积与实现细节段，明确四步交替和“联合优化为后续问题”。不是逐页全部实验复核。
- <a id="source-2"></a>**[cite:2] P0，2016 v3扩展稿，[SINGLE-SOURCE]。** [版本历史](https://arxiv.org/abs/1506.01497v3)核对v1 2015-06-04、v3 2016-01-06；[v3正文](https://arxiv.org/html/1506.01497v3)首轮返回截断在anchor前部，随后直接获取HTML并提取§III-A2至III-C，读到loss、normalization、三种训练讨论、四步法及边界处理。不把工具截断说成全文已读，也不将v3修订年写为方法首次年份。
- <a id="source-3"></a>**[cite:3] P0，Girshick，Fast R-CNN，ICCV 2015，[SINGLE-SOURCE]。** [官方PDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)已读§2、RoIPool、式(1)–(4)、检测框归一化和两图128 RoI采样；用来补第二阶段原始定义，不把Fast与Faster视为同篇。
- <a id="source-4"></a>**[cite:4] P0，作者Python重实现及身份。** [提交记录](https://github.com/rbgirshick/py-faster-rcnn/commit/781a917b378dbfdedb45b6a56189a31982da1b43)、[固定README](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/README.md)、[固定Git tree](https://api.github.com/repos/rbgirshick/py-faster-rcnn/git/trees/781a917b378dbfdedb45b6a56189a31982da1b43)均取得；tree确认Caffe子模块hash。README说明这是MATLAB重实现、权重不兼容，并含近似联合。2018提交不是2015原始发行；没有克隆、编译或下载权重。
- <a id="source-5"></a>**[cite:5] P0，anchor生成，[SINGLE-SOURCE：固定源码]。** [generate_anchors.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/generate_anchors.py)函数正文已读；核base=16、ratio／scale顺序、取整及inclusive中心。
- <a id="source-6"></a>**[cite:6] P0，anchor监督，[SINGLE-SOURCE：固定源码]。** [anchor_target_layer.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/anchor_target_layer.py)正文已读；核inside、并列最高、0.7等号、clobber、无放回采样、inside／outside权重、unmap及空backward。未运行NumPy/Caffe该层。
- <a id="source-7"></a>**[cite:7] P0，proposal与默认值，[SINGLE-SOURCE：固定源码]。** [proposal_layer.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/proposal_layer.py)、[config.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/fast_rcnn/config.py)、[py_cpu_nms.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/nms/py_cpu_nms.py)正文已读；NMS等号依据可执行比较而非config注释；GPU／Cython实现未核成等价。
- <a id="source-8"></a>**[cite:8] P0，检测监督，[SINGLE-SOURCE：固定源码]。** [proposal_target_layer.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/rpn/proposal_target_layer.py)正文已读；训练追加GT、class-specific4槽、单图采样、反向边界。
- <a id="source-9"></a>**[cite:9] P0，框变换，[SINGLE-SOURCE：固定源码]。** [bbox_transform.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/fast_rcnn/bbox_transform.py)正文已读；`+1`宽高与未减1的右端解码均为实际函数行为，非推测现代API。
- <a id="source-10"></a>**[cite:10] P0，具体网络与覆盖配置，[SINGLE-SOURCE：固定源码]。** [VGG16 train.prototxt](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt)、[faster_rcnn_end2end.yml](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/experiments/cfgs/faster_rcnn_end2end.yml)正文已读；没有只凭模型名推断loss sigma、背景下界或batch。
- <a id="source-11"></a>**[cite:11] P0，固定Caffe子模块，[SINGLE-SOURCE：固定源码]。** [RoIPool CPU](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp)、[RoIPool CUDA](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cu)、[Smooth L1 CPU](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/smooth_L1_loss_layer.cpp)、[Smooth L1 CUDA](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/smooth_L1_loss_layer.cu)函数体已读；[caffe.proto](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto)另提取SmoothL1LossParameter核sigma默认1。CPU RoIPool backward、Smooth L1 CPU前反向标NOT_IMPLEMENTED，未声称编译可运行。
- <a id="source-12"></a>**[cite:12] P0，阶段、导出和测试，[SINGLE-SOURCE：固定源码]。** [train_faster_rcnn_alt_opt.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/tools/train_faster_rcnn_alt_opt.py)、[test.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/fast_rcnn/test.py)正文已读；[train.py](https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/fast_rcnn/train.py)初次抓取为空，改直接读取raw后提取snapshot函数，核反归一化折叠。未逐个solver与所有stage图全量核验，四步冻结语义以论文为准。
- <a id="source-13"></a>**[cite:13] P0，FPN版本边界，[SINGLE-SOURCE]。** [CVPR 2017官方页](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)书目与摘要已读：自顶向下和横向连接生成多尺度语义特征；未读取其全部实验和实现，不引用性能作比较结论。
- <a id="source-14"></a>**[cite:14] P0，Mask R-CNN／RoIAlign，[SINGLE-SOURCE]。** [ICCV 2017官方PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)已读§1–3的并行mask、RoIPool量化、RoIAlign双线性采样和FPN／C4区别；初用小写content_iccv_2017路径返回404，改大写ICCV路径成功。未逐表核全部实验，不升级现有S22／S23验证等级。

## 本次验证与 G23 销项建议

**本次边界：**仅新增本文件；规范、总览、S22／S23和台账只读。完成固定原始资料的关键机制核对、五层正文／答5、数学与分支有限检查及本地链接／锚点检查；没有实现或训练模型、下载权重、运行CUDA内核、测量真实AP／recall／速度。已有工作区改动不属于本次成果。

文件写入记录只涉及 `faster_rcnn.md`。最终范围检查另观察到并行新增的 `openmax.md`，未读取或改动，不计入本卡成果；因此不声称整个工作区只有一个变化，汇总计数也必须在后续统一核对。

**建议销项口径：**G23／07-05可从“缺少具体Faster R-CNN卡”转为“已有 FR01”，原P1缺卡任务保留历史ID，后续深度列 **P2、未实现**；五层记 **核／核／部／部／部，答5**。销的是RPN目标、anchor分配与版本训练链的文档缺口，不是整个算法实现、收敛理论或产线验收。后续P2包括原MATLAB／Python差异、实际梯度与坐标黄金向量、训练复现及小件／拥挤真实验收；不以FPN／RoIAlign新版本替代2015主版。

台账更新应在另行授权时同步逐卡索引、07-05、G23及汇总计数；本次不改这些文件。仅在基线计数仍为本次所读值、且没有其他并行补卡时，候选同步增量是独立卡223→224、文件19→20，113节点中的已有52→53／缺卡27→26，部分10与非算法24不变，剩余任务37→36；实际维护必须重新核对，不能机械覆盖并行成果。
