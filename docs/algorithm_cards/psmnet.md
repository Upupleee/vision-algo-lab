# PM01 PSMNet：金字塔上下文、拼接代价体与视差监督

[学习规范](../algorithm_learning_standard.md) · [三维总览 E07](../3d_evolution.md#37-e07psmnet学习双目对应与代价聚合) · [覆盖台账 G40／10-07](../evolution_coverage.md) · [G01 双目几何基础](remaining_domains.md#g01-双目-sad-块匹配与三角测量)

<a id="pm01"></a>
<a id="g40"></a>

## 任务、演进与覆盖状态

**任务与契约：**输入同步、已极线校正、同尺寸的左右 RGB 图像，输出以左图为参考的稠密水平视差。约定左像素 \((x,y)\) 对应右像素 \((x-d,y)\)，视差以输入图像的水平像素计；训练另需左视差 GT 及其有效性协议。网络输出不是点云、物理深度或遮挡真值；只有已知像素焦距 \(f\)、物理基线 \(B\) 且主点差已处理时，才能另用 \(Z=fB/d\) 三角化。立体代价体的“3D”是视差、高、宽，不是世界坐标 XYZ。[cite:1](#source-1)

**演进定位：**局部 patch 特征在弱纹理、重复纹理和反射区域存在歧义 → 用残差／空洞卷积与 SPP 汇集区域上下文 → 拼接左右特征，保留可学习匹配信息 → 以 3D CNN 在视差和空间方向共同正则 → 可微视差回归和中间监督训练。新增代价是带标签的数据、权重与域适配、显存和计算；不能消除遮挡、标定误差或域外失效。它是学习式双目对应前端，不是 SAD、SGM、单目深度或全套三维测量系统的别名。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-05**。唯一键 `psmnet.md:PM01`，仅此一张卡，承接 **G40／10-07／E07**。论文为 Chang、Chen 的 **CVPR 2018，5410–5418**；代码固定为作者仓库 `JiaRenChang/PSMNet` 的 **`87ac9093afbf6545c093bd9d26c5ffd66e49a7b8`，2021-09-22**。这是本次静态核对版本，不冒充 2018 实验内部代码；README 的依赖更新、后发权重和经验建议不反写成原论文机制。[cite:1](#source-1)[cite:3](#source-3)

| 字段 | 状态与范围 |
|---|---|
| L1 步骤 | 已核验：论文主链、固定代码张量布局、basic／stacked hourglass、训练推理分支 |
| L2 机制 | 已核验：四尺度上下文、拼接与 shift、三类 mask、正则与期望回归反例 |
| L3 工程权衡 | 部分：原文配置、预处理／padding／maxdisp 契约及排障设计；没有现场定参 |
| L4 数学推导 | 已核验：限定模型下的拼接、softmax 符号、Smooth L1、梯度与尺度敏感性；不是误差上界 |
| L5 实现 | 部分：外部源码对应、教学伪代码、索引／shape／代数检查；本仓库未实现 PSMNet |
| 面试自测 | 答5：贯穿 L1–L5，含参考答案、成立条件、易错点及证据入口 |
| 验证范围 | 在线阅读、Python 标准库双精度有限差分及静态尺寸检查、文档链接和单文件检查；无网络前向／反传、训练、权重加载、数据集或硬件评测 |

证据属于同一作者论文／代码体系，机制证据标记 **[SINGLE-SOURCE]**；CVF PDF、ar5iv 转写与作者仓库不能算三份独立实验。下文分别标记原文、固定代码和推导／工程建议；没有独立性能复核，不复述排行榜作为本仓库成绩。仓库 Python 源码检索未发现 `PSMNet`、`hourglass` 或 `disparityregression` 的专用实现。

## L1 步骤：共享特征到全分辨率视差

### 输入、坐标和总体流程

令 batch 为 \(N\)，实际送入网络的高宽为 \(H,W\)，`D=maxdisp` 为全分辨率视差候选数；固定代码使用 \(d=0,\ldots,D-1\)，不是包含 D 的 D+1 个候选。为便于描述固定 stacked 网络，取 H、W、D 为16的正倍数，\(h=H/4,w=W/4,q=D/4\)，同时要求 h、w 至少64以容纳最大 SPP 窗口。16倍数是避免 hourglass 跳接尺寸错配的保守契约，不是所有立体网络的定理。[cite:4](#source-4)[cite:5](#source-5)

1. 核对图像配对、校正、颜色、视差单位和权重版本；左右图同步裁剪或同步 padding。原图尺寸单独保存，GT 只在训练／评价时进入损失或指标。
2. 同一个 `feature_extraction` 分别处理左右图，得到 `F_L,F_R:[N,32,h,w]`；共享参数，不是把两张图先混成6通道输入。
3. 在每个粗视差 \(\delta=0,\ldots,q-1\) 处，将 `F_L(y,x)` 与 `F_R(y,x−δ)` 按通道拼接，填入零初始化体 `V:[N,64,q,h,w]`。
4. basic 或 stacked hourglass 对 V 正则化，得到每个粗视差的单通道分数。这里的“代价体”在拼接时仍是64通道特征，尚不是一个标量 SAD 距离。
5. **先**将单通道分数体三线性上采样至 `[N,1,D,H,W]`，去掉通道维后沿 D 做 softmax，再以 `0..D−1` 求期望；不是先做 argmin，也不是先回归粗视差再统一乘4。
6. stacked 训练返回三个 `[N,1,H,W]` 预测，分别与同一全分辨率 GT 计算损失；推理只返回第三个预测，但第三个分数仍依赖前两个分类头的累加。basic 只有一个输出。
7. 训练反传、更新直至预先冻结的预算，使用独立验证集选模型；推理一次前向结束，按保存的原图尺寸裁回。不保证损失全局最优或自动收敛。空有效 GT、非有限分数、非法尺寸应显式跳过／报错并留记录，此为工程防护建议。

步骤2–6对应固定模型调用链；步骤1、7中的完整输入校验与拒绝策略不是作者脚本已经实现的能力。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)

### 特征提取与 SPP 四尺度

以下 shape 不列 N，以论文和代码均使用的训练尺寸 `H=256,W=512` 为静态展开。`convbn` 是卷积加 BN，ReLU 是否存在按调用处确定；固定 `BasicBlock` 包含两次3×3卷积，残差相加后**没有**额外 ReLU。[cite:4](#source-4)

| 固定代码层段 | 配置 | 输出 C×H×W |
|---|---|---|
| firstconv | 三次3×3，32通道；第一层 stride2，其余 stride1；各带 BN／ReLU | 32×128×256 |
| layer1 | 3个残差块，32通道，stride1 | 32×128×256 |
| layer2 → output_raw | 16个残差块，64通道，首块 stride2 | 64×64×128 |
| layer3 | 3个残差块，128通道，dilation1 | 128×64×128 |
| layer4 → output_skip | 3个残差块，128通道，dilation2 | 128×64×128 |
| branch1 | AvgPool kernel=stride=64；1×1，128→32，BN／ReLU | 32×1×2 |
| branch2 | AvgPool kernel=stride=32；1×1，128→32，BN／ReLU | 32×2×4 |
| branch3 | AvgPool kernel=stride=16；1×1，128→32，BN／ReLU | 32×4×8 |
| branch4 | AvgPool kernel=stride=8；1×1，128→32，BN／ReLU | 32×8×16 |
| 分支恢复 | 每支 bilinear 到64×128；不是四支都全局池化成1×1 | 每支32×64×128 |
| 拼接／融合 | `[output_raw,output_skip,branch4,branch3,branch2,branch1]`；64+128+4×32=320；3×3，320→128，BN／ReLU；1×1，128→32 | 32×64×128 |

**64、32、16、8 是特征图上的池化窗口和步长，不是输出 bin 数，不是输入原图像素尺度。**本例最大分支仍为1×2。固定平均池化没有 padding，非整倍尺寸按 floor 得到输出，末端未覆盖条带不能自动视为全局汇聚。池化后1×1降通道，再双线性插回原特征尺寸；不展平、不接全连接，保留左右几何布局。[cite:2](#source-2)[cite:4](#source-4)

**原文差异：**Table 1 的 conv3／conv4 dilation 为2／4，固定代码为1／2；论文 Table 2 明列半 dilation 消融，因此不能把固定代码说成表1逐层原样。§3.2 明确分支使用1×1卷积，Table 1 分支行却标3×3；代码支持正文的1×1。本卡保留这个正文／表格差异，不额外杜撰作者的更正说明。分支通道拼接顺序也按代码记录，加载既有权重时不可随意重排。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)

### 拼接体的 shift、边界与三种 mask

对左图参考坐标 x，粗视差 δ 对应输入分辨率约4δ像素位移。固定代码仅写入 `x≥δ` 的位置，两半通道的 `x<δ` 都保持初始化零；**不是左特征处处填满、只把右侧越界补零**。δ=0要单独处理，因为 `:-0` 是空切片。[cite:5](#source-5)[cite:6](#source-6)

| 名称 | 含义 | 固定代码实际行为／不能混用之处 |
|---|---|---|
| 几何支持 M(δ,x)=1[x≥δ] | 该右坐标在裁剪后特征图内 | 隐含在拼接切片和零初始化里；没有额外 mask 通道，没有向最终 logits 写 −∞ |
| GT 有效掩膜 m(y,x) | 数据集中该像素有可信、满足选定范围的视差标签 | 只控制监督／评价分母；不由网络置信或 M 直接生成 |
| 遮挡／可靠性掩膜 | 真实匹配是否可见、结果是否允许交付 | 原模型没有显式左右一致性或遮挡拒绝输出；稠密预测不等于全部像素实测有效 |

零填充经过3D卷积和邻域信息传播后可以变为非零预测，softmax也可给几何越界候选正概率。若在最终全分辨率候选上额外施加 `d≤x` 硬屏蔽，这是**新变体**；同位置裁剪可能让原本存在的右对应落到裁剪框外，故不得把这些像素一律宣称“GT错误”。输入 padding 区与物理遮挡也不等同。换成 `roll` 而不清除绕回部分，会制造右图尾部到左边界的假对应。

### basic 与 stacked hourglass 的准确连接

| 层段 | basic | stacked hourglass |
|---|---|---|
| 入口 dres0 | 两层3×3×3，64→32→32，BN／ReLU | 相同 |
| 四分之一尺度残差 | dres1–4各两层3D卷积，逐块加回输入；总计入口2+残差8+分类2=12层3D卷积 | dres1两层加回，得共享 `cost0` |
| 上下采样 | 不使用 hourglass 编解码 | dres2、3、4是三个独立 hourglass，每个两次 stride2下采样和两次 stride2转置卷积恢复 |
| 分类头 | classify：32→32→1，一个分数体 | classif1、2、3各32→32→1，分数逐级相加 |
| 训练／推理 | 都是一个视差预测 | 训练三个全分辨率视差，推理只回归最后一个 |

每个 hourglass 从 `[N,32,q,h,w]` 到 `[N,64,q/2,h/2,w/2]`，再到 `[N,64,q/4,h/4,w/4]`，经 `ConvTranspose3d(k=3,s=2,p=1,output_padding=1)` 两次恢复32通道。全分辨率坐标下依次是1/4→1/8→1/16→1/8→1/4；视差维也下采样，不能只缩 H、W。[cite:5](#source-5)

固定调用关系为：

```text
cost0 = dres0(V)
cost0 = dres1(cost0) + cost0
u1, pre1, post1 = HG1(cost0, None, None)
out1 = u1 + cost0
u2, pre2, post2 = HG2(out1, pre1, post1)
out2 = u2 + cost0
u3, pre3, post3 = HG3(out2, pre1, post2)
out3 = u3 + cost0
a1 = classif1(out1)
a2 = classif2(out2) + a1
a3 = classif3(out3) + a2
```

HG 内 `pre` 是第一次下采样后第二次卷积的结果，可加前一栈 `postsqu` 再 ReLU；第一次上采样加 `presqu`，若无则加本栈 pre，再 ReLU，最后上采样。**第三栈传入的是 pre1 而非 pre2；三个 out 都加 cost0，而非各自输入。**三个损失不是对三个空间尺度监督，也不是三张预测视差的算术平均；累加的是尚未 softmax 的分数。[cite:5](#source-5)

## L2 机制：上下文、可学习代价与中间监督

| 设计 | 因果机制与保留的信息 | 删除或替换后的后果／反例 |
|---|---|---|
| 残差与空洞卷积 | 保留细节路径，扩大空间上下文而不继续降低1/4特征密度 | 只扩大普通卷积步长可能抹去薄结构；空洞扩大支持不保证镜面可匹配 |
| 四尺度 SPP | 聚合不同范围与区域的上下文，同时保留 raw／skip局部特征 | 只用全局均值可能混合前景背景；删 SPP 后网络仍可训练，但失去这一上下文入口 |
| 拼接而非预先求绝对差／相关 | 后续网络可同时访问左右特征各分量，学习非固定匹配函数 | 压成单标量改变表达能力；但拼接增加通道与显存，不保证所有域都优于相关 |
| 3D正则 | 邻域视差和空间联合聚合，使局部歧义可受周围结构约束 | 逐像素softmax不能利用相邻代价形态；过度平滑会跨深度边抹去薄片 |
| hourglass与跳接 | 编解码扩大支持，跳接保留较高分辨率信息；跨栈继续修正表示 | 不再只是堆同尺寸卷积；删跳接破坏尺寸和特征通路，不能沿用旧权重作等价替换 |
| 中间监督 | 较早分类头直接收到GT梯度，并经分数累加参与后续预测 | 只监督最后一栈仍有梯度但优化路径改变；0.5／0.7／1不是概率或跨域最优证明 |
| soft argmin／期望 | 将离散候选分数映成可微连续视差 | 双峰的均值可能落在两个表面之间；亚像素小数不是额外几何证据 |

结构机制依据为论文§3和固定代码，删除后的解释为建模分析，不是本次消融实验。[cite:1](#source-1)[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)

对于两个分离表面的相同概率峰，期望落在中间，即使中间候选几乎无概率。故应分别检查代价分布、边界、遮挡和域外样本；softmax和为1并不使其成为经过校准的正确率。监督只约束期望时，多种分布可产生同一视差，概率形状本身不可由单一视差标签唯一辨识。

## L3 工程权衡：GT、预处理与尺度

### 损失范围与有效 GT 不能统一猜测

| 依据 | 实际规则 | 边界与建议 |
|---|---|---|
| 原文§3.6／§4.1 | 对有标签像素求Smooth L1均值；Scene Flow超搜索范围像素排除 | 原文“有标签”不意味着所有像素都有效；不能以图像H×W当分母 |
| 固定 main.py 训练 | `disp_true < args.maxdisp` | 未同时检查正值／finite；负哨兵、−∞可能被选中，不能把这个单条件当通用数据清洗 |
| 固定 finetune.py 训练 | `disp_true > 0` | KITTI零值无效；未排除≥D或+∞，不能声称与Scene Flow是同一mask |
| 工程重实现建议 | 数据集valid AND finite AND 非padding AND 合同允许的视差范围 | 对KITTI可用`0<GT<D`；若数据规定零视差有效则不机械删除零；显式记录相对作者脚本的变化 |

固定代码三个头各取有效元素的均值，权重为0.5、0.7、1.0；basic权重1。mask由GT／数据协议决定，不随预测是否“容易”变化。空有效集必须单独处理，不能靠 `max(count,1)` 隐藏全部标签失效；对 NaN 标签应先布尔选择有效元素，而非只乘零，因为 `0×NaN` 仍为 NaN。[cite:2](#source-2)[cite:7](#source-7)[cite:8](#source-8)

评价协议另外冻结：main.py 的测试mask硬编码 `<192`，未跟随 `args.maxdisp`，不可复制到可配置评测。EPE、3px或KITTI相对误差条件、遮挡范围和无效分母分开写；README明确其报告的3px验证误差使用官方Matlab代码，而不是仓库脚本。本卡不从旧脚本推定新的统一指标定义。[cite:3](#source-3)[cite:7](#source-7)

### 图像、GT编码、裁剪和 padding

- **颜色：**固定loader用PIL `convert('RGB')`，`ToTensor()`将常见uint8图像映射到[0,1]，再按mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]`逐通道归一。实际调用 `get_transform(augment=False)`；文件里存在 ColorJitter 并不表示训练使用它。不能换成BGR、每图去均值或左右独立随机几何变换而声称相同协议。[cite:9](#source-9)[cite:10](#source-10)
- **GT编码：**KITTI loader 将PNG数值转float32后除256，得到像素视差；Scene Flow loader读取PFM路径，不套KITTI的1/256。编码系数不是网络1/4特征步长。PFM reader的内部解析细节本次未读取，不声称已完成文件端序／翻转验证。[cite:10](#source-10)
- **裁剪：**训练左右和左GT用同一个256×512框，不做resize，故视差数值不变。KITTI loader验证路径裁右下368×1232，和整图padding推理不是同一评价区域；两者成绩不可直接混比。[cite:10](#source-10)
- **padding：**`Test_img.py`在归一化后给左右图同样补顶部和右侧零，`top=(-H0)%16`、`right=(-W0)%16`。归一化域的0对应均值颜色，不是RGB黑。补齐不是缩放，视差单位不变；最终用 `pred[...,top:top+H0,:W0]` 裁回可避免`:-0`陷阱。[cite:11](#source-11)
- **尺寸边界：**仅补到16倍数仍未保证SPP可用；小图还要满足特征高宽≥64。很小的池化网格在训练时还可能触发BN每通道样本数不足，按每设备batch检查，不靠`eval()`冒充训练解决方案。
- **源代码分歧：**`Test_img.py`对top／right四种情况分别裁回；`submission.py`在任一padding非零时统一使用 `[:, :-right_pad]`，遇到top>0而right=0会得到空宽度。main.py测试路径只裁顶部，不处理右padding。它们都是静态可见风险，不是本次已经运行并修好的功能。[cite:7](#source-7)[cite:11](#source-11)[cite:12](#source-12)

若另行按水平比例 \(s_x\) resize，需同时令 \(d'=s_xd\)、\(f'_x=s_xf_x\)，并在恢复原图时将视差除以 \(s_x\)；图像插值和数值缩放缺一不可。两图若不同水平裁剪偏移 \(a_L,a_R\)，视差变成 \(d'=d-a_L+a_R\)，不能继续原GT。以上是几何推论，不是固定脚本的resize功能。

### maxdisp、配置与诊断次序

原文训练裁剪256×512，D=192，Adam的β为0.9／0.999，batch12；Scene Flow以0.001训练10轮，KITTI微调300轮，前200轮0.001、后100轮0.0001；正文另记录用于最终KITTI提交的1000轮训练。0.5／0.7／1权重来自原文消融，不是损失必须满足的数学常数。固定脚本默认10／300轮，不能把其默认执行称为最终提交权重复刻。[cite:2](#source-2)[cite:7](#source-7)[cite:8](#source-8)

| 症状 | 优先证据与调整顺序 | 代价／不能承诺 |
|---|---|---|
| 大视差接近上限、近物体截断 | 检查GT分布、输出候选0..D−1、resize比例；按目标距离和fB选择覆盖范围，再选合法D | 增D扩大搜索但显存／耗时近线性上升；不是消除遮挡 |
| 全图尺度偏差 | 先查KITTI除256、RGB、权重／插值版本、是否重复乘4，再查标定与resize | 不先乘README提到的1.17经验因子；它不是理论尺度校准 |
| 左边界异常 | 检查shift方向、两半通道清零、crop后可见性和padding裁回 | 加硬mask属于新变体，必须独立验收 |
| hourglass相加报shape错 | 检查D、H、W的16倍数与每次下采样／转置卷积尺寸 | 不能只把最后输出resize到GT大小掩盖中间错位 |
| 换材质／相机失败 | 先查曝光、校正、纹理、输入归一与域差；做距离／材质／边界分层评测 | 再考虑微调、模型容量或回退SGM；不靠训练集损失验收泛化 |
| 显存不足 | 先核对N、D、H、W及是否同时保留多头全分辨率体；剖析3D激活 | 缩图、减D、分块改变问题或上下文；AMP需重新验证数值稳定和精度 |

拼接体有 \(2CNqhw\) 个元素；FP32仅这一体所需字节为 \(4\cdot2CNqhw\)。3D卷积计算量约为各层 \(Nq_\ell h_\ell w_\ell k^3C_{in}C_{out}\) 之和。分数上采样另占 \(NDHW\) 个元素，训练还有多栈激活、梯度与优化器状态；单个体的大小绝不是显存峰值。没有跨GPU的统一实时性结论。

## L4 数学推导：符号、梯度与尺度边界

### 拼接体和反向索引

省略batch和y，以 \(M(\delta,x)=\mathbf1[x\ge\delta]\) 表示合法粗对应，越界右特征按0延拓：

\[
V_{c,\delta,x}=M(\delta,x)F^L_{c,x},\qquad
V_{C+c,\delta,x}=M(\delta,x)F^R_{c,x-\delta}.
\]

若上游梯度为 G，则固定布尔支持下：

\[
\frac{\partial L}{\partial F^L_{c,x}}=
\sum_{\delta=0}^{q-1}M(\delta,x)G_{c,\delta,x},\qquad
\frac{\partial L}{\partial F^R_{c,u}}=
\sum_{\substack{0\le\delta<q\\u+\delta<w}}G_{C+c,\delta,u+\delta}.
\]

左特征为多个候选复用，右特征梯度沿逆shift收集；不能只取一个视差平面。共享特征网络参数收到左右两条路径梯度之和。这里对特征可导，整数shift索引本身不是可学习连续位移，mask不对模型求导。[cite:5](#source-5)

### soft argmin 的负号、候选终点与不变量

用 \(c_d\) 表示“越小越好”的代价，用 \(z_d=-c_d\) 表示“越大越好”的分数，温度固定为1：

\[
p_d=\frac{\exp(z_d)}{\sum_{j=0}^{D-1}\exp(z_j)},\qquad
\hat d=\sum_{d=0}^{D-1}d\,p_d.
\]

原文式(1)写 \(\mathrm{softmax}(-c)\)，且求和上端写 \(D_{max}\)；固定实现用 `range(maxdisp)` 与 D 个输出bin，即0..D−1。应明确自己的候选集合，而不是凭符号多加一个bin。固定 stacked 代码直接 `softmax(cost,dim=1)`，其变量名cost实为可学习score；不能在已训练权重后任意插负号，作者所说的两种约定可学习，不表示固定logits取反后输出不变。[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5)

softmax对全体logit加同一常数不变，故使用 \(z-\max z\) 防指数溢出。沿D归一后 \(\sum p_d=1\)，输出属于候选凸包 \([0,D-1]\)，有限非退化分数通常在内部；不需要输出再乘4。所有分数相同则 \(\hat d=(D-1)/2\)，不是“零视差”或“无效”标记。

从 \(\partial p_d/\partial z_j=p_d(\mathbf1[d=j]-p_j)\) 得：

\[
\frac{\partial\hat d}{\partial z_j}
=\sum_d d\,p_d(\mathbf1[d=j]-p_j)
=p_j(j-\hat d),\qquad
\frac{\partial\hat d}{\partial c_j}=-p_j(j-\hat d).
\]

导数之和为0，与共同平移不变量一致。对远离均值的候选提高score会把预测向该候选移动；分布高度饱和时，大多数梯度变小。这个可微性不保证找到正确表面，更不提供泛化误差界。

### Smooth L1、有效分母及多头反传

令 \(r_{k,i}=\hat d_{k,i}-d_i^*\)，有效集 \(\Omega\) 只含协议允许的有限GT，\(n=|\Omega|>0\)。用输入像素为单位，原文阈值为1像素：

\[
\rho(r)=\begin{cases}\tfrac12 r^2,&|r|<1\\|r|-\tfrac12,&|r|\ge1\end{cases},\qquad
L=\sum_{k=1}^{K}\frac{\lambda_k}{n}\sum_{i\in\Omega}\rho(r_{k,i}).
\]

stacked的 \(K=3,\lambda=(0.5,0.7,1)\)，basic的K=1、λ=1。原文使用GT减预测，因ρ为偶函数，损失相同；本卡固定预测减GT以避免导数符号混淆。令 \(\psi(r)=r\)（|r|<1），否则为sign(r)，则有效像素：

\[
\frac{\partial L}{\partial z_{k,j,i}}
=\frac{\lambda_k}{n}\psi(r_{k,i})p_{k,j,i}(j-\hat d_{k,i}).
\]

无效像素对该监督项的直接梯度为0，但共享卷积参数和邻域特征仍可收到有效像素传来的梯度。原文称Smooth L1比L2对异常值不敏感，是尾部导数有界的含义；并不使错误标签无害。[cite:2](#source-2)[cite:7](#source-7)[cite:8](#source-8)

设 \(U\) 为固定坐标约定下的三线性上采样算子，\(z_k=Ua_k\)。有 \(\nabla_{a_k}L=U^T\nabla_{z_k}L\)，每个低分辨率格点收集其插值权重覆盖的高分辨率梯度。若记累加前分类输出为 \(b_1,b_2,b_3\)，则 \(a_1=b_1,a_2=b_2+a_1,a_3=b_3+a_2\)，所以仅就这条累加路径：

\[
\nabla_{b_1}L=g_1+g_2+g_3,\quad
\nabla_{b_2}L=g_2+g_3,\quad
\nabla_{b_3}L=g_3,\qquad g_k=U^T\nabla_{z_k}L.
\]

完整参数梯度还含hourglass之间的特征依赖，不能把此式说成三个栈互不影响。这也解释为什么推理可不计算前两头的高分辨率概率，却不能删除前两头的低分辨率分数。

### 上采样不是简单的视差乘4

特征恢复用bilinear，仅处理H、W；最终单通道体用trilinear，同时处理D、H、W。原文§3.4对体上采样称bilinear，固定代码明确 `mode='trilinear'`；应以所实现的五维张量算子注明差异。[cite:2](#source-2)[cite:5](#source-5)[cite:6](#source-6)

以粗视差长度q到D为例，`align_corners=True`的连续源坐标是 \(u=d(q-1)/(D-1)\)；False的半像素坐标为 \(u=(d+1/2)q/D-1/2\)，边缘另依库规则处理。固定源码没有显式传此参数，README建议给upsample加True；不可不记录PyTorch版本就宣称行为相同。H、W也受此选择影响。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

即使q=D/4，True的端点映射比例仍不是严格4；且softmax是非线性的，通常 \(\mathrm{softmax}(Ua)\ne U\mathrm{softmax}(a)\)。因此“先上采样logits再期望”不能随意替换成“粗期望插值再乘4”。README的输出乘1.17是issue经验提示，不是上述坐标公式推导出的通用修正，也未在本卡验证。

深度误差的一阶敏感性为 \(\partial Z/\partial d=-fB/d^2\)，小视差时放大；还须合并f、B、校正和同步误差。非凸训练不保证全局最优，亚像素回归不保证同等毫米精度；论文未提供本场景的通用理论误差界。

## L5 实现：伪代码、静态对应与有限验证

### 教学伪代码与实现边界

以下是**重实现设计伪代码**，不是本仓库可导入模块。`SHARED_FEATURES`和`REGULARIZE`分别严格对应L1的固定结构；`ALIGN`必须由目标权重／运行时协议显式冻结，不能在这里猜默认值。零初始化、指定softmax轴、GT防护与裁回都写明；尚未进行PyTorch自动微分验证。

```text
INPUT left_rgb, right_rgb, maxdisp, mode, training
REQUIRE paired_rectified_same_shape(left_rgb, right_rgb)
N, H0, W0 = batch_and_spatial_shape(left_rgb)
top, right = (-H0) mod 16, (-W0) mod 16
L = PAD_TOP_RIGHT(NORMALIZE_RGB(left_rgb), top, right, value=0)
R = PAD_TOP_RIGHT(NORMALIZE_RGB(right_rgb), top, right, value=0)
H, W = spatial_shape(L)
REQUIRE maxdisp > 0 AND maxdisp mod 16 == 0
REQUIRE H / 4 >= 64 AND W / 4 >= 64
FL = SHARED_FEATURES(L)
FR = SHARED_FEATURES(R)
N, C, h, w = shape(FL)
q = maxdisp / 4
V = ZEROS_LIKE_DEVICE_DTYPE(FL, [N, 2*C, q, h, w])
FOR delta IN 0 .. q-1:
    IF delta < w:
        V[:, 0:C, delta, :, delta:w] = FL[:, :, :, delta:w]
        V[:, C:2*C, delta, :, delta:w] = FR[:, :, :, 0:w-delta]
scores = REGULARIZE(V, mode)
selected = scores IF training ELSE [scores[-1]]
predictions = []
FOR a IN selected:
    z = TRILINEAR(a, size=[maxdisp,H,W], align_corners=ALIGN)
    z = SQUEEZE_ONLY_CHANNEL(z)
    REQUIRE ALL_FINITE(z)
    p = SOFTMAX(z - MAX(z, axis=1, keepdim=True), axis=1)
    d = ARANGE(maxdisp, device=z.device, dtype=z.dtype)
    pred = SUM(p * RESHAPE(d, [1,maxdisp,1,1]), axis=1, keepdim=True)
    predictions.APPEND(pred[..., top:top+H0, 0:W0])
IF training:
    valid = dataset_valid AND ISFINITE(gt) AND allowed_gt_range(gt,maxdisp)
    REQUIRE COUNT(valid) > 0
    lambda = [0.5,0.7,1.0] IF mode == stacked ELSE [1.0]
    loss = SUM_k(lambda[k] * MEAN(SMOOTH_L1(predictions[k][:,0][valid] - gt[valid])))
    OPTIMIZER_ZERO_GRAD()
    BACKWARD(loss)
    OPTIMIZER_STEP()
ELSE:
    RETURN predictions[-1]
```

此处用区间`0:w−δ`避免δ=0的`:-0`，并对δ≥w跳过。它保留作者零支持语义，没有偷偷加概率hard mask；普通实现若用不可微的数组拷贝／detach构造体，会切断特征学习，须用真实框架验证切片赋值的梯度。若训练输入已经是256×512，padding恰为零，回到作者的典型尺寸契约。

### 固定源码移植清单

| 入口 | 已核对的对应与潜在陷阱 | 重实现需验证 |
|---|---|---|
| submodule.py | 共享特征、SPP与期望；disparityregression硬编码`.cuda()`，并`keepdim=True` | device／dtype一致，候选向量注册buffer或按输入生成；输出为N×1×H×W，不误删batch |
| stackhourglass.py | `maxdisp//4`、两半零填充、跨栈pre1、分数累加、显式softmax dim1 | 跳接逐元素shape、训练三个输出／推理一个输出、每个头有限梯度 |
| basic.py | `/4`与`range(self.maxdisp/4)`保留旧Python整数除法假设；`volatile`旧接口；softmax未指定dim | 不能据README宣称所有分支直接兼容现代Python；应显式`//4`、dim=1并做等价性测试 |
| main.py／finetune.py | GT mask不同；finetune basic分支生成output却引用output3；部分标量仍用旧`loss.data[0]` | 按分支独立走通小批训练；读取源码不等于这些路径已可运行 |
| preprocess／loaders | RGB、ImageNet归一、KITTI除256、同步裁剪；KITTI loader仍有旧式import | 数据端到端通道哨兵与标签坐标核对；依赖升级不能只改模型 |
| Test_img／submission | 顶／右padding；不同裁回实现；PNG输出乘256再转uint16 | 零padding边界、原尺寸还原、单位回读、非有限／负值处理 |

入口均固定到本文来源中的同一提交。[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)[cite:8](#source-8)[cite:9](#source-9)[cite:10](#source-10)[cite:11](#source-11)[cite:12](#source-12)

### 本次已执行的 shape／索引／数值检查

用现有Python环境的标准库 `math` 和列表在内存执行，未安装torch、未生成额外测试文件。以下是解析公式与索引规则的确定性检查，不是模拟立体数据、网络推理或精度测量。shape来自算子规则，不声称分配过这些大型张量。

| 检查 | 实际结果 | 能证明／不能证明 |
|---|---|---|
| 论文训练尺寸N=1,H=256,W=512,D=192 | 特征 `[1,32,64,128]`；SPP网格1×2、2×4、4×8、8×16；体 `[1,64,48,64,128]` | 通道与尺度算术一致；不是网络前向 |
| hourglass三个被卷积轴 | 48×64×128 → 24×32×64 → 12×16×32 → 24×32×64 → 48×64×128 | 静态上下采样及跳接尺寸合法 |
| FP32单体内存 | 拼接体96 MiB；一个全分辨率单通道分数体也为96 MiB | 精确元素计数；不是总显存／训练峰值 |
| shift索引哨兵 | L=[0,1,2,3,4]、R=[10,11,12,13,14]，δ=2；得到[(0,0),(0,0),(2,10),(3,11),(4,12)] | 两半通道前两列均零，未绕回；这些整数是索引标签，不是图像数据 |
| D=192全零logits解析点 | p=1/192，期望95.5；j=0、95、191的导数分别−0.4973958333、−0.0026041667、+0.4973958333 | 验证score符号、候选终点和均匀分布；不是已训练分布 |
| softmax中心有限差分 | 步长1e−4，上述三点最大绝对差 `8.41e−10`以内，阈值1e−7通过 | Python双精度局部代数成立；未验证框架autograd／FP16 |
| Smooth L1解析检查 | r=0.5时ρ=0.125；n=1、λ=1时最后logit梯度0.2486979167；三个相同残差的总损失0.275 | 验证链式法则和加权，不表示实际训练loss |
| padding分支 | 376×1240→384×1248，top/right=8/8；375×1232→384×1232，9/0；368×1232→原尺寸，0/0 | 覆盖仅top非零与无padding；暴露`:-0`不合法裁回 |
| 视差插值比例 | True源坐标斜率47/191≈0.2460733，反比≈4.0638298；False内部斜率48/192=0.25 | 证明两种坐标约定不同；未以实际库插值复现输出 |

进一步待实现的测试包括：真实tensor切片梯度与左右共享参数梯度、每栈低分辨率score和全分辨率输出逐层比对、NaN／空GT／超范围标签、batch1与多batch、仅右padding、maxdisp／SPP非法尺寸拒绝、AMP与FP32误差、权重版本对齐、真实双目EPE及有效率。单元代数检查不升级这些待验证项。

### 优化与验收边界

先做逐元素正确性基线，再剖析cost-volume构造、3D卷积和全分辨率体占用。向量化shift、避免重复生成候选向量、推理跳过前两头的上采样／回归可作为候选优化；不得删除仍被a3依赖的低分辨率分类头。AMP、分块、减D、换相关体或换SPP均需重新校验精度／边界／上下文，不标成无损。

真实评测应固定图像与GT版本、训练／验证／测试分组、范围和有效分母，记录视差误差、遮挡／边界／材质分层、拒绝率、显存、预热及尾延迟；物理深度还需标定与可追溯真值。当前无数据、无权重执行、无性能数字、无尺寸验收结论。

## 五问自测：贯穿 L1–L5

| 层级／问题 | 参考答案要点 | 成立条件与易错答案 | 证据入口 |
|---|---|---|---|
| L1：256×512双目、D=192怎样得到左视差？ | 共享特征32×64×128；四个SPP分支拼成320再融合32；64×48×64×128拼接体；3D正则；score上采样到192×256×512，沿D softmax求期望；stacked训三头、推理第三头 | 已校正、RGB、合法尺寸；错答是先argmin、直接输出深度、最后视差还乘4 | L1；[cite:4](#source-4)[cite:5](#source-5) |
| L2：SPP的64是否代表64×64输出？零填充是否排除了遮挡？ | 64是固定窗口／步长；本例输出1×2；x<δ两半全零只标记粗几何支持，不是−∞概率屏蔽，更不是遮挡检测 | 固定代码而非PSPNet自适应池化；错答是四支全局1×1、零体即零概率、GT mask即遮挡mask | L1／L2；[cite:2](#source-2)[cite:4](#source-4)[cite:5](#source-5) |
| L3：换相机后视差整体偏小、边界失败，先改什么？ | 核对校正、RGB、GT除256、resize／f尺度、D上限、权重与align_corners，再查crop／padding和域偏移；之后才微调或增D | 有独立验证与单位协议；错答是无条件乘1.17、降低损失阈值、仅增加网络深度 | L3；[cite:3](#source-3)[cite:9](#source-9)[cite:10](#source-10)[cite:11](#source-11) |
| L4：推导soft argmin梯度并解释负号与多峰？ | score导数p_j(j−期望)，cost导数取负；再乘λψ(r)/n，经U转置和分数累加反传；双峰均值可能不对应真实表面 | 有限分数、n>0、固定GT支持及插值规则；错答是固定权重任意取反仍等价、softmax就是校准置信度 | L4；[cite:2](#source-2)[cite:5](#source-5)[cite:7](#source-7) |
| L5：只补顶部时结果空宽、basic不能跑，怎么验证？ | 原尺寸切片避免`:-0`；核对旧`/4`、volatile、softmax轴、finetune变量；测试两半清零、左右梯度、每栈shape、空GT与squeeze维度，再谈AMP | 静态定位不是动态修复；错答是无参数squeeze、把示例改成固定batch、文档检查当模型复现 | L5；[cite:6](#source-6)[cite:8](#source-8)[cite:11](#source-11)[cite:12](#source-12) |

## 来源、版本与获取深度

以下为2026-11-05联网记录。B=书目／提交元数据，T=实际读取的方法正文，C=实际读取的固定代码；没有下载或执行权重。CVF PDF文本返回在Table 1后截断，因此以同文ar5iv转写补读§3.2–3.6、§4.1–4.2、Table 2–3；不声称逐页视觉核对图1或审计全部实验。没有为满足来源数量而引入未经核实的二手讲解。

<a id="source-1"></a>

**[cite:1] 2018原始会议论文，B+T。** Jia-Ren Chang、Yong-Sheng Chen，*Pyramid Stereo Matching Network*，CVPR 2018，5410–5418。https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf 。已读摘要、引言、§3.1–3.2及Table 1的提取内容；表格提取存在重复行，未据此补造网络层。

<a id="source-2"></a>

**[cite:2] 同一原文的HTML转写，T，非独立研究。** https://ar5iv.labs.arxiv.org/html/1803.08669 。已补读固定池化、拼接体、basic十二层、stacked中间监督、式(1)–(2)、数据预处理／训练配置及loss权重；为解决PDF返回截断而用，不改变2018论文归属。

<a id="source-3"></a>

**[cite:3] 作者仓库固定提交与README，B+C。** 提交 https://github.com/JiaRenChang/PSMNet/commit/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8 ，首次通过 https://api.github.com/repos/JiaRenChang/PSMNet/commits/master 解析出此SHA和2021-09-22日期，再固定所有源码URL。README：https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/README.md 。已读依赖、align_corners提示、1.17经验提示、后发权重与官方评测声明；未验证提示的准确性或所有分支可运行性。

<a id="source-4"></a>

**[cite:4] 共享特征、SPP与视差期望，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/models/submodule.py 。通过对应raw URL读取全文件，定位 `BasicBlock`、`feature_extraction`、`disparityregression`。

<a id="source-5"></a>

**[cite:5] stacked hourglass，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/models/stackhourglass.py 。raw全文已读，定位hourglass跳接、cost-volume切片、三个累加头、三线性插值与softmax符号说明。

<a id="source-6"></a>

**[cite:6] basic，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/models/basic.py 。raw全文已读，确认十二层3D卷积及旧除法／volatile／softmax接口残留；未运行该分支。

<a id="source-7"></a>

**[cite:7] Scene Flow训练／测试入口，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/main.py 。raw全文已读，核对训练mask、三损失、Adam、默认预算及测试192常量／裁回范围。

<a id="source-8"></a>

**[cite:8] KITTI微调，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/finetune.py 。raw全文已读，核对正GT mask、loss权重、学习率切换和basic分支变量；不声称该旧入口已成功执行。

<a id="source-9"></a>

**[cite:9] 预处理，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/dataloader/preprocess.py 。raw全文已读，确认实际`augment=False`路径仅ToTensor／Normalize；存在而未调用的增强不计入默认训练。

<a id="source-10"></a>

**[cite:10] 图像与GT loader，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/dataloader/SecenFlowLoader.py 与 https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/dataloader/KITTILoader.py 。两份raw全文已读；前者文件名确为`SecenFlowLoader.py`。已核对RGB、同步裁剪、PFM调用、KITTI除256及验证裁剪；未取得真实数据运行。

<a id="source-11"></a>

**[cite:11] 单对图像推理入口，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/Test_img.py 。raw全文已读，确认归一化后top／right padding、四分支裁回、视差乘256输出；没有接入本仓库。

<a id="source-12"></a>

**[cite:12] 提交推理入口，C。** https://github.com/JiaRenChang/PSMNet/blob/87ac9093afbf6545c093bd9d26c5ffd66e49a7b8/submission.py 。raw全文已读，对照仅top padding时`:-right_pad`问题；没有实际提交或外部写入。

## G40销项建议与未实现边界

本卡补齐 **10-07／E07 的独立PSMNet教学入口**，建议覆盖台账将G40从“缺卡”调整为“已补PM01；L1／L2／L4已核验，L3／L5部分；答5；原文正文＋固定作者代码；未实现”。若G40的销项仅要求有可追溯五层单卡，可关闭“缺卡”子项；若要求全部层级实测，不应整体标已完成。

后续导航可引用 `psmnet.md#pm01`，三维总览“无专卡”可另行同步为“有专卡、未实现”。**本次只创建本文件，不修改台账、总览、索引、依赖、源码或测试。**既有G01仅是双目几何／SAD基础，不升级为PSMNet实现；本文伪代码、外部源码阅读和有限代数检查均不代表模型已复现、权重已验证、数据集达标或物理尺寸已验收。
