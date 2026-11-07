# FC01 FC4：全卷积局部估光、置信加权池化与监督边界

[学习规范](../algorithm_learning_standard.md) · [颜色总览 A-H5](../color_appearance_evolution.md#4-历史演进线-b校正恒常性与空间外观算法) · [覆盖台账 G35／09-13](../evolution_coverage.md) · [平均池化组件 C10](learning_components.md#c10-平均池化与全局自适应平均池化)

<a id="fc01"></a>
<a id="g35"></a>

## 任务、演进与覆盖状态

**任务与契约：**输入一幅具有明确相机响应、黑电平和编码协议的三通道图像，训练时另有同一传感器 RGB 空间中的全局光源真值；输出单位长度的三维光源颜色方向，以及用于解释聚合的局部估计／置信图。预测的是光源色度的两个自由度，不是绝对照度、物体反射光谱、产品 Delta E 或置信区间。原文集中解决单一全局光源估计；存在局部图不等于完成空间多光源分解。[cite:1](#source-1)

**演进定位：**局部 patch 对光源的可辨识性不同 → 等权训练和汇聚容易受纯色墙面等歧义区域影响 → FCN 共享重叠区域的卷积计算，同时学习局部光源方向及其贡献权重 → 全局角度监督经池化反传，决定哪些局部估计有帮助 → 代价是带光源标签的数据、预训练／微调和域适配；不消除反射与照明的根本歧义，也不能自动替代测色标定。C10 只有平均池化算子，不能承接这一训练与聚合机制。[cite:1](#source-1)

核验日期：**2026-11-05**。唯一键为 `fc4.md:FC01`，仅一张卡，承接 **G35／09-13／A-H5**。原对象为 Hu、Wang、Lin 的 **CVPR 2017** 论文；作者代码 README 记载网络／训练脚本于 **2018-04-25** 发布，不能把公开代码日期当算法提出年。固定代码提交为 **`fc86df1ce82308b7dda28d71d705c25d6edd8251`**，不是浮动 master，也不声称它就是论文实验当时的内部版本。[cite:1](#source-1)[cite:2](#source-2)

| 字段 | 状态与范围 |
|---|---|
| L1 步骤 | 已核验：论文网络、三种输出语义、固定代码训练／推理调用链；不代表网络已运行 |
| L2 机制 | 已核验：语义歧义、共享卷积、归一顺序、置信梯度分配及反例 |
| L3 工程权衡 | 部分：原文／代码配置、fold、色板泄漏与诊断设计；没有现场定参 |
| L4 数学推导 | 已核验：非零及可微域内的归一化、角损失、两类头梯度；有限差分仅验代数 |
| L5 实现 | 部分：固定源码静态对应、教学伪代码及有限验证；本仓库未实现 FC4 |
| 面试自测 | 答5：L1–L5 各一问，含答案、成立条件、易错点和来源 |
| 验证范围 | 原文／源码阅读，NumPy FP64 的局部公式检查，文档结构和变更范围检查；无 FC4 前向、自动微分、训练、权重或真实数据评测 |

证据仅来自原论文和同一作者代码体系，机制标记 **[SINGLE-SOURCE]**；不同 URL、论文与同团队实现不是三份独立实验。下文“原文”“固定代码”“推导／工程建议”分栏处理，不把差异抹平。仓库 Python 源码检索未发现 FC4 专用定义；已有深度学习组件不能据此升级为已实现。

## L1 步骤：全卷积网络、三种头与端到端流程

### 论文结构与局部输出契约

原文 Fig.2、§3.1 用 ImageNet 预训练 AlexNet 的 conv1–conv5，再接池化、**6×6 卷积输出64通道**的 conv6，以及 **1×1 卷积输出4通道**的 conv7。四通道经 ReLU；前三通道按局部位置做 L2 归一化成为 RGB 方向，第四通道为非负置信。局部位置对应有重叠的感受野，不是独立 RGB 像素；不存在解码器、上采样或像素级光源监督。[cite:1](#source-1)

| 层段 | 原文定义／作用 | 本卡边界 |
|---|---|---|
| AlexNet conv1–conv5 | 预训练中层语义特征；共享重叠窗口计算 | 已取论文没有逐层展开所有核、group、LRN和边界，固定仓库也未提供可直接核验的 AlexNet 实现，不补造完整原配置 |
| 末端池化 → conv6 | 6×6，64输出，聚合较大语义上下文；conv6 使用 dropout | 尺寸图中的约 H/32×W/32 是示意，不能替代逐算子的 floor／padding |
| conv7 | 1×1，4输出 → ReLU；RGB逐位置归一，保留置信幅值 | 这是一层共享输出，不要求两个独立网络分支 |
| 置信池化 | 局部单位方向乘权 → 空间求和 → 全局L2归一 | 输出一个全局三维方向；不是平均池化特征后再分类 |
| SqueezeNet替代 | v1.1 conv1–fire8；Fig.2文字写额外2×2池化 | 固定公开代码的pool8实际为3×3，见下一节；二者不能视为逐元素相同 |

§4.1 还明确给出更简洁实现：直接输出加权三向量，长度充当置信、方向充当局部光源。故“四个语义量”不意味着所有 FC4 必须有四个物理输出通道。[cite:1](#source-1)[cite:2](#source-2)

### 固定作者代码：具体 SqueezeNet backbone

`fcn.py` 实际导入 `squeeze_net.create_convnet`，虽变量／scope 仍叫 `AlexNet`，并不执行 AlexNet。数据布局是 **NHWC**，卷积核是 **HWIO**；不得套用 NCHW 的通道归一轴。以下以代码训练输入 `[N,512,512,3]` 展开；shape 是按静态算子规则计算，不是本次执行网络所得。[cite:3](#source-3)[cite:4](#source-4)

Fire(s,e1,e3) 为 `1×1 squeeze(s) → ReLU → 并行1×1 expand(e1)、3×3 expand(e3) → 通道拼接 → ReLU`；expand 的3×3使用 SAME。默认不带 residual shortcut。这里 s/e1/e3 是通道数，不是步长。[cite:4](#source-4)

| 固定代码层段 | 核／步长／边界及通道 | 512输入的输出shape，不列N |
|---|---|---|
| conv1 → ReLU | 3×3，stride2，VALID，64输出 | 255×255×64 |
| pool1 | 3×3 max，stride2，VALID | 127×127×64 |
| fire2 → fire3 | 各为Fire(16,64,64) | 127×127×128 |
| pool3 | 3×3 max，stride2，SAME | 64×64×128 |
| fire4 → fire5 | 各为Fire(32,128,128) | 64×64×256 |
| pool5 | 3×3 max，stride2，SAME | 32×32×256 |
| fire6 → fire7 | 各为Fire(48,192,192) | 32×32×384 |
| fire8 | Fire(64,256,256) | 32×32×512 |
| pool8 | **3×3 max，stride2，VALID** | 15×15×512 |
| fcn.fc1 | 6×6，64输出，沿用Slim默认SAME／ReLU，再dropout | 15×15×64 |
| fcn.fc2 | 1×1，3或4输出；具体激活见三种语义 | 15×15×3或4 |
| 聚合 | 仅沿H、W求和，再沿RGB归一 | 3 |

`get_features_out()` 返回 `pool8`；文件虽还构造 fire9、conv10 和分类 softmax，这些不是光源估计前向的依赖路径。换骨干时不能把分类 logits 当局部 RGB。预训练 pickle 的转换细节未执行核验；源码还存在自定义 bias 初始化，移植时须比对权重和逐层输出，不能只加载同shape参数。[cite:4](#source-4)

### 归一化顺序：原文、默认代码与可选代码分开

令位置索引 i 遍历 M=H′W′，`N(v)=v/‖v‖₂` 仅在非零向量上定义。

| 版本 | 局部输出与权重 | 空间汇聚及关键差异 |
|---|---|---|
| **原文显式4通道** | `uᵢ=ReLU(aᵢ[:3])`，`vᵢ=N(uᵢ)`，`cᵢ=ReLU(aᵢ[3])` | `s=Σcᵢvᵢ`，`y=N(s)`；没有要求先softmax置信；不能把RGB与置信一起做四维L2归一 |
| **固定代码默认3通道** | `SEPARATE_CONFIDENCE=False`；fc2的`activation_fn=None`，直接得`pᵢ∈R³` | `s=Σpᵢ`，`y=N(s)`；非零时可解释`cᵢ=‖pᵢ‖₂`、`vᵢ=pᵢ/cᵢ`。**此分支没有末端ReLU，允许负分量和相消**，不是原文非负域的逐字实现 |
| **固定代码可选4通道** | fc2→ReLU，前三通道L2归一；第四通道展平后在每张图M个位置上softmax为`αᵢ` | `s=Σαᵢvᵢ`，`y=N(s)`；softmax不是原文`cᵢ/Σcᵢ`，且ReLU后0分数仍有正softmax权重 |
| **固定代码等权消融** | `WEIGHTED_POOLING=False`，对每个局部fc2三向量先L2归一 | 长度差异被消除，再空间求和；`SEPARATE_CONFIDENCE=False`本身并未关闭加权 |

这些分支对应 `FCN.build_branches()` 的116–139行；全局聚合在178–186行。默认 `WEIGHTED_POOLING=True`、`FC_POOLING=False`、`USE_SHORTCUT=False`。三通道默认分支不应在聚合前再归一，否则偷偷变成等权消融；显式分支不应保留RGB原幅值，否则幅值和置信会双重加权。[cite:3](#source-3)[cite:5](#source-5)

若所有权重非负、总和 C>0 且 s≠0，则 `N(Σcᵢvᵢ/C)=N(Σcᵢvᵢ)`；所以最终方向无需显式除以权重总和。这个等价只允许删除一个共同的正标量，**不能删除局部归一，不能把softmax改成线性归一**，也不能用于零向量。

### 训练与推理的依赖、停止条件

1. 固定数据版本、图像ID、相机、标签通道顺序及fold；先遮去ColorChecker，再做裁剪／旋转／缩放。训练标签是整图光源向量，不是置信标签；原单光源假设下裁剪继承同一标签。
2. 线性图像先进行与数据版本匹配的黑电平等预处理。颜色增强在**线性域**同时作用图像和真值；输入骨干前再应用论文的`x^(1/2.2)`。这只是幂律近似，不是严格分段sRGB转换。[cite:1](#source-1)
3. 训练批前向：骨干 → conv6／fc1和dropout → 选定输出头 → 空间聚合 → 全局方向。真值也按RGB轴L2归一；对每张图求角误差，再取batch均值。
4. 全局角损失连同选定正则反向传播至局部RGB、置信和骨干；默认代码`PER_PATCH_WEIGHT=0`，虽计算局部损失，但不能写成每个patch都有独立监督。置信分支没有额外人工标签。[cite:3](#source-3)[cite:5](#source-5)
5. Adam更新；到预先冻结的预算停止，在训练子集内部验证选模型。非有限值、空有效图、无效真值应中止或登记拒判；不以加epsilon掩盖缺失方向。此独立验证策略为工程建议，**不是固定作者脚本原行为**。
6. 推理关闭dropout、冻结参数，遵循相同颜色协议。论文／代码基准测试将整图按两轴0.5倍缩小、保留纵横比，再全卷积计算；不是统一拉伸至512正方形。任意大小仍受最小合法池化尺寸、显存和静态图创建约束。[cite:1](#source-1)[cite:3](#source-3)
7. 有效输出归一成RGB光源方向；若要白平衡，用线性图像按通道除以光源并另选共同亮度尺度。光源分量近零时增益发散，应限增益／拒判且记录。显示gamma、裁剪和输出量化在校正后进行；这不等于完成颜色计量校准。

## L2 机制：为什么需要局部估光与置信

| 设计 | 机制与不变量 | 删除／替换的后果及反例 |
|---|---|---|
| 全卷积共享 | 同图重叠感受野共享卷积，可在一个全局损失中比较局部贡献 | 独立patch反复卷积增加冗余；逐patch损失迫使歧义墙面也拟合全局真值，目标已变 |
| 中层语义与颜色敏感性 | 可借熟悉物体颜色降低反射／照明歧义，同时必须保留光源色偏信息 | 只追求分类不变性可能丢光源信息；更深分类backbone不是必然更优，原文特意讨论此冲突 |
| 局部RGB方向归一 | 将“预测什么方向”和“信多少”解耦 | 显式头不归一会让RGB长度成为另一套隐式权重；置信可视化失去原解释 |
| 学习置信 | 用最终损失判断一个方向对当前全局合成的帮助 | 等权使大量歧义patch主导；但模型也可能偏爱训练背景，不能保证高纹理永远高置信 |
| 全局角损失 | 只惩罚光源方向，正的共同尺度不改变目标 | RGB欧氏损失混入幅值目标；不能把输出幅值直接解释为照度或概率 |
| 非负与数值域 | 原文非负贡献不会相互抵消；至少一个正贡献时s非零 | 默认代码允许有符号输出，可出现非零局部向量相消成零，必须区别处理 |
| 全局汇聚 | 所有局部贡献产生一个方向，适合单光源任务 | 两盏不同色灯照两个区域时，输出可落在二者之间，却不正确解释任一区域 |

机制依据为原文§3和代码分支；反例是建模分析，不是本仓库实验。[cite:1](#source-1)[cite:3](#source-3)

**置信不等于校准概率：**原文显式 c 可整体乘任意正数而不改全局方向；默认三通道的长度也可共同缩放。softmax版本虽和为1，但只表达同图位置的相对贡献，不能把最大值当“预测正确概率”或跨图可比不确定度。全局标签通常无法唯一确定每个局部方向／权重，故漂亮热图不是局部估光真值。[cite:1](#source-1)[cite:2](#source-2)

## L3 工程权衡：原文配置、固定代码与泄漏

### 原文实验与公开配置不可拼接

| 项目 | 原文§4.1 | 固定代码事实／工程界限 |
|---|---|---|
| 优化器与batch | Adam，batch16 | 默认Adam，batch16；`train_step_sgd`名字下实际仍构造Adam，不可凭变量名写成SGD |
| 学习率 | AlexNet 1e−4，SqueezeNet 3e−4；预训练和新增层同学习率 | BASE_LEARNING_RATE=3e−4，LR_DECAY=1；默认统一Adam；FINE_TUNE_LR_RATIO=0.1属于另一条非默认更新路径 |
| 权重衰减 | 所有层5e−5 | GLOBAL_WEIGHT_DECAY=5.7e−5；arg_scope给Slim卷积登记L2，SqueezeNet用原生get_variable／conv2d，所读路径未为这些backbone变量登记同项，不能声称覆盖所有层 |
| dropout | conv6概率0.5 | fc1后keep_prob=0.5，推理1.0；概率刚好相同不代表drop和keep概念可混用 |
| 几何增强 | 随机边长为短边0.1–1倍，旋转−30°至30°，水平翻转概率0.5，裁剪最终512² | scale以log-uniform方式抽样，最小边长夹到10像素；AUGMENTATION_ANGLE=60实际乘`random−0.5`，不是±60° |
| 颜色增强 | SqueezeNet训练reprocessed ColorChecker时，图像／标签RGB同步乘[0.6,1.4]随机因子 | AUGMENTATION_COLOR=0.8得到同范围；offdiag=0、额外gamma增强=0、curve关闭；默认配置不能自动外推为论文所有数据集统一增强 |
| MCC遮挡 | 训练与测试均把色板多边形填RGB(0,0,0) | BOARD_FILL_COLOR=1e−5，但目标图像是uint16；微小浮点填充值不代表保留有效色板信息；具体OpenCV转换未动态验证 |
| 训练预算 | 本次所读正文未给6000 epochs | EPOCHS=6000、TEST_PERIOD=20是公开代码配置；不是原文已声明的通用收敛预算 |
| 输入／测试尺度 | 线性图像幂律1/2.2；测试缩小至50% | FCN输入clip到0–65535、除65535、幂律、乘255、通道反转，再SqueezeNet减[104,117,123]；必须保留实际调用链而非照搬注释 |

依据：[cite:1](#source-1)[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:7](#source-7)。上述配置是复现起点，不是经现场调优的推荐常数。

**RGB/BGR特别记录：**`datasets.py`用OpenCV读取BGR；增强时把RGB标签反转到图像通道顺序，变换后再反转回RGB。`fcn.build_branches`又对图像最后一轴反转，`squeeze_net.py`仍按固定数组减均值。源内注释有“RGB／BGR”混杂；本卡按操作记录，不擅自补一个反转“修正”。新实现若统一改RGB输入，必须用纯通道哨兵和已知权重逐层对照，不能宣称直接等价。[cite:3](#source-3)[cite:4](#source-4)[cite:6](#source-6)[cite:7](#source-7)

**黑电平与gamma不能叠加猜测：**Gehler读取路径按文件名前缀`IMG`减129，否则减1，截负，再用全图最大值缩放至65535；这仅是固定数据版本的脚本，不是所有相机通用黑电平。`raw.max()==0`在该路径没有保护。Cheng／NUS路径读取darkness_level和saturation_level；不能照搬Gehler常数。外部JPEG入口先用`(img/255)^2.2*65536`近似反gamma，无法撤销相机已有的白平衡、色调映射或压缩，不能把JPEG演示当RAW协议评测。[cite:6](#source-6)[cite:8](#source-8)

### Fold、ColorChecker与模型选择泄漏

原文使用 **reprocessed ColorChecker 568张**和 **NUS 8-Camera 1736张、按相机子集**，均三折交叉验证；不是把NUS所有相机随意混为一个IID数据集。论文结果仅在这些数据协议下解释，本卡不转录性能表为复现成绩。[cite:1](#source-1)

| 风险 | 固定代码证据 | 复现实验应如何记录／改进 |
|---|---|---|
| 原图跨折、标签错配 | Gehler用`folds.mat`的te_split，MATLAB索引减1，核对Xfiles与文件名，检查fold互斥 | 原图ID先划分，再生成增强；不得把同图裁剪随机分到训练／测试；新数据另按场景／批次分组 |
| fold编号误读 | FOLD=T训练gT、g((T+1)%3)，测试g((T+2)%3) | T=0测g2、T=1测g0、T=2测g1；README的fold1and2权重用于g0，不是用于所有图 |
| 把训练图也算测试 | FOLD=−1选择g0/g1/g2全体为TEST_FOLDS | 只适合显式诊断等用途；不能用一个已训练两折的模型给全数据报盲测分数 |
| **测试集选checkpoint** | `FCN.train`每20轮调用test，再`LowestTrigger.push(mean(errors))`保存最好；README直言历史协议无独立验证集 | 这是选择层面的泄漏风险，即使test不反传也存在；严格评测需训练两折内部再划验证，或预先固定预算，最终留出折只评一次 |
| NUS重新生成分折 | ChengDataSet随机shuffle后均分三份，所读路径未固定seed；主入口默认只处理Gehler | 固定并发布相机与图像ID清单，重复预处理不得悄悄换折；不能声称已复刻论文NUS折表 |
| 色板直接泄露光源 | 论文与数据脚本均先遮MCC；色板正是生成光源真值的参考 | 训练／验证／测试全部遮挡；按原始图坐标遮挡再变换，检查角点顺序、缩放和旋转，不允许只遮训练图 |
| 边缘与掩膜捷径 | 图像填黑不等于在网络特征图上强制置信为零 | 检查色块残边、插值混合和感受野穿越色板；需要mask-aware策略时作为新变体单独评测，不写成原算法 |

依据：[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:5](#source-5)[cite:6](#source-6)。这里不推断论文所有报告都存在同一程度的偏差，只指出已取得公开脚本的选择行为及其统计后果。

### 诊断顺序与成本

先核对数据版本／黑电平／RGB顺序／标签 → 遮挡和fold → 头语义及归一轴 → 小批训练可下降与梯度有限 → 学习率／正则 → 骨干容量／图像尺度 → 独立相机和批次评测。跨相机失败先查传感器响应与域移，不先增大网络；热图全零先查ReLU、输入范围和全黑图，不把epsilon当可观测性修复。

卷积成本近似为各层`H′W′k²CinCout`之和；局部聚合只需O(3M)，显式头额外一个通道，softmax再需O(M)。减小测试尺度减少像素面积，也改变感受野对应的场景尺度；不能声称对输出无损。报告均值、中位数、trimean、最好／最坏25%均值及95分位时必须保留同一测试分母，并单列拒判数量，不丢弃NaN样本美化成绩。没有通用角误差界、泛化保证或现场延迟承诺。

## L4 数学推导：方向、梯度与退化

### 成像假设与尺度不可辨识性

简化对角模型为`I_c(x)=e_c ρ_c(x)`；正向量 e 是全局光源，ρ为未知反射响应。对任意正向量 d，`e′=e⊙d`与`ρ′=ρ⊘d`可生成相同I；单幅三通道图像不能无条件区分两者。FC4借训练数据语义先验缩小可能性，不给唯一物理解。只用角误差时，e与ae（a>0）也完全等价，单位方向有两个自由度。

记单位真值 t、显式头局部方向 vᵢ、非负置信 cᵢ，则原文式(1)、(3)、(4)为：

\[
s=\sum_{i=1}^{M}c_i v_i,\quad y=\frac{s}{\|s\|_2},\quad
\ell=K\arccos(y^Tt),\qquad K=180/\pi.
\]

要求`s≠0`、`‖t‖₂=1`；训练目标是batch均值加实际选定参数集合的正则。原文的全局角度监督不定义每个置信应取的绝对值。[cite:1](#source-1)

### 全局归一化必须保留投影Jacobian

令 r=‖s‖₂、z=yᵀt，在r>0且−1<z<1的可微域内：

\[
dy=\frac{1}{r}(I-yy^T)ds,\qquad
\nabla_y\ell=-\frac{K}{\sqrt{1-z^2}}t,
\]

\[
q\equiv\nabla_s\ell
=-\frac{K}{r\sqrt{1-z^2}}(I-yy^T)t
=-\frac{K}{r\sqrt{1-z^2}}(t-zy).
\]

`I−yyᵀ`删除径向梯度，故`sᵀq=0`，吻合正尺度不变性。直接用`1/r`替代完整Jacobian会产生错误的径向分量。对本式角损失，若把梯度解释为球面切空间梯度，可以把投影吸收到上游梯度记号；若上游指普通欧氏梯度，则不能省略。

**与原文式(5)、(6)的关系：**正文印出的简式写`cᵢ/r · ∂L/∂ŷ`及相应内积，未显式展示上述投影。此处给出对本卡明确函数的完整链式法则，不将正文简写逐字当通用归一化反传公式。补充材料未成功获取，不能声称已核验作者补充推导是否采用相应梯度约定。[cite:1](#source-1)[cite:10](#source-10)

### 显式头：RGB、置信和共享骨干怎样收到监督

设`uᵢ=ReLU(aᵢ)`、`vᵢ=uᵢ/‖uᵢ‖₂`、`cᵢ=ReLU(bᵢ)`。在非零、远离ReLU拐点的区域：

\[
\nabla_{v_i}\ell=c_iq,\qquad
\frac{\partial\ell}{\partial c_i}=v_i^Tq,
\]

\[
\nabla_{u_i}\ell=\frac{c_i}{\|u_i\|_2}(I-v_iv_i^T)q,\quad
\nabla_{a_i}\ell=\mathbf1[a_i>0]\odot\nabla_{u_i}\ell,\quad
\frac{\partial\ell}{\partial b_i}=\mathbf1[b_i>0]v_i^Tq.
\]

因此只有对**已归一局部方向**vᵢ的梯度共享q方向并按cᵢ缩放；到了归一前RGB／logits，还经过各自投影、范数和ReLU，不能泛称所有patch全部参数梯度同向。若`vᵢᵀq<0`且置信ReLU开启，梯度下降会增大cᵢ；反之会减小。cᵢ=0会屏蔽直接局部RGB梯度，但负置信logit也可能因ReLU死区收不到梯度；“必然自动恢复有用区域”没有保证。

对于共享特征Fᵢ，若头是`aᵢ=W_rgb Fᵢ+d_rgb`、`bᵢ=w_cᵀFᵢ+d_c`，则：

\[
\nabla_{F_i}\ell=W_{rgb}^T\nabla_{a_i}\ell+w_c\frac{\partial\ell}{\partial b_i}.
\]

它再经conv6和骨干反传；重叠感受野对同一卷积权重的梯度相加。置信学习无需单独标签，并不意味着没有监督。[cite:1](#source-1)[cite:3](#source-3)

### 默认隐式头与softmax变体

默认直接输出pᵢ，`s=Σpᵢ`，所以`∇pᵢℓ=q`。将pᵢ解释成cᵢvᵢ时，可在局部球坐标下解释方向和长度的作用；但实现不会额外反传一个独立的置信输出头。`pᵢ=0`时其方向分解未定义，即使总和非零、对pᵢ的线性梯度仍存在。默认无ReLU，局部有符号贡献可相消。[cite:3](#source-3)

可选代码令`dᵢ=ReLU(bᵢ)`、`αᵢ=exp(dᵢ)/Σexp(dⱼ)`、`s=Σαᵢvᵢ`，则：

\[
\frac{\partial\ell}{\partial d_i}=\alpha_i(v_i-s)^Tq,\qquad
\nabla_{v_i}\ell=\alpha_iq.
\]

还需乘ReLU导数，RGB侧仍需局部归一Jacobian。由于sᵀq=0，上式在精确可微条件下可简为`αᵢvᵢᵀq`；一般上游损失不能未经检查就丢掉softmax耦合项。显式原文c整体正缩放不变、softmax logits共同平移不变，是不同参数化的不变量。

### 数值角损失与敏感性

固定代码先L2归一预测／真值，再将点积夹至 **[−0.999999,0.999999]**，取acos乘180/π并平均。因此同方向输入的训练损失也有约 **0.0810284752°** 下限；点积进入夹断平台后梯度可能为零。这是代码稳定化后的目标，不是原文精确角误差。`utils.angular_error`却夹到[−1,1]，返回弧度，由调用者转换度；两个口径不可混用。[cite:3](#source-3)[cite:9](#source-9)

投影能解释角损失的几何结构，但不使s≈0处变得可靠：`‖dy‖≤‖ds‖/r`是一阶局部敏感性关系，r趋零时放大误差。z=±1的acos表达和ReLU零点应单列非光滑／数值边界；有限差分不在这些点冒称通过。没有通用优化收敛、角误差或跨相机泛化理论界。

## L5 实现：边界明确的伪代码与有限验证

### 源码对应与推理零向量

| 源码入口，均固定同一提交 | 核对内容 | 本次状态 |
|---|---|---|
| `squeeze_net.py:build_model/fire_module/get_features_out` | NHWC、pool8、fire通道、预训练核转置 | 静态读取；未加载pickle权重 |
| `fcn.py:build_branches/build` | gamma／反转、两种头、等权消融、空间求和与损失 | 静态读取；未执行TensorFlow图 |
| `fcn.py:get_angular_loss/train/test/test_external` | 角度夹断、Adam、测试选模、不同零向量处理 | 静态读取，不声称通过原作者测试 |
| `config.py:initialize_dataset_config` | 默认开关、训练预算、fold轮换 | 配置与命令参数须随权重存档 |
| `datasets.py`／`data_provider.py:augment` | 黑电平、MCC遮挡、分折、线性颜色增强与标签同步 | 未下载真实数据、未执行预处理 |
| `fc4.py:test`／`utils.py:angular_error` | 外部JPEG入口、弧度／度转换 | 演示入口不代替RAW评测 |

固定代码的零向量并未统一安全处理：`test()`用`all_pixels/(norm+1e−7)`，零向量仍是零；随后`utils.angular_error`除零仍可产生NaN。`test_external()`直接`est/=norm(est)`；近零分量还使白平衡除法发散。训练使用TensorFlow的L2归一化epsilon机制并不能证明零向量是有效方向，更不能替代数据有效性判断。[cite:3](#source-3)[cite:9](#source-9)

**工程建议而非原代码复刻：**在输入全黑／全遮挡、真值零向量、非有限输出、所有有效贡献为零或有符号相消时，返回结构化`invalid`及原因。若业务需回退，可另返固定白光或历史估计并标记`fallback`，绝不能把回退伪装成FC4有效预测。阈值按模型数值尺度和精度单独标定，不将epsilon解释为物理置信门限。

### 教学伪代码：训练目标与拒判分离

下例是语言无关的复现设计，不是新增可运行模块；`unit_strict`先检查有限值和非零阈值，再沿末维归一。它不使用epsilon把零向量变成方向。`source_preprocess`必须绑定上文某一确定协议，不能同时混搭原文头、默认代码头和第三方权重。

```text
function pool(head, mode, tau):
    require all_finite(head)
    if mode == PAPER_EXPLICIT:
        u = relu(head[..., 0:3])
        c = relu(head[..., 3:4])
        active = (c > 0)
        require norm(u[active], RGB) > tau
        v = zeros_like(u)
        v[active] = unit_strict(u[active], tau)
        p = c * v
    elif mode == CODE_IMPLICIT:
        p = head
    elif mode == CODE_SOFTMAX_EXPLICIT:
        a = relu(head)
        v = unit_strict(a[..., 0:3], tau)
        alpha = softmax_spatial_per_image(a[..., 3:4])
        p = alpha * v
    else:
        fail("unknown mode")
    s = sum(p, axes=(H, W))
    return unit_strict(s, tau)

freeze outer_test_ids, inner_validation_ids, preprocessing, mode, budget
for step in budget:
    image, target = sample_training_original_ids()
    image = mask_colorchecker_before_geometry(image)
    image, target = synchronized_linear_augmentation(image, target)
    x = source_preprocess(image)
    head = network(x, dropout_keep=0.5)
    y = pool(head, mode, tau)
    t = unit_strict(target, tau)
    z = sum(y * t, axis=RGB)
    loss = mean(acos(clip(z, -0.999999, 0.999999)) * 180 / pi)
    total_loss = loss + regularizer(explicit_parameter_set)
    require all_finite(total_loss, gradients(total_loss))
    adam_update(total_loss)
select_checkpoint_using_inner_validation_only()

function infer(image):
    if input_invalid(image): return invalid("input")
    head = network(source_preprocess(image), dropout_keep=1.0)
    try: y = pool(head, mode, tau)
    catch numerical_invalid: return invalid("illuminant direction")
    return valid(illuminant_rgb=y)
```

显式代码在零RGB位置原本可能由库返回零，本伪代码选择更严格的拒判，因此**不是逐输入等价实现**。原文训练目标若要求精确acos，应另选相应数值策略；不能把本例的夹断当原文定义。训练遇非法样本应登记并修复／中止，不能静默跳过改变测试分母。

### 本次已执行的有限验证

仅使用仓库已有`.venv`内的NumPy，以FP64在内存执行，没有安装TensorFlow、创建测试文件或落地外部源码。以下固定小向量是**公式单元检查输入，不是图像数据、模拟基准或模型性能证据**。

设`u=[[2,1,1],[1,3,2]]`、`c=[0.7,1.3]`、`t=N([1,2,3])`，采用L4的精确角损失，中央差分步长`h=1e−6`，远离ReLU／acos夹断边界：

| 检查 | 结果 | 证明范围 |
|---|---|---|
| 对6个局部RGB分量与2个置信分量，完整解析梯度对中央差分 | 最大绝对差 **1.86e−8**，低于检查容差1e−6 | 仅此可微输入的链式法则；不是网络自动微分／训练验证 |
| `N(s)`与`N(9s)` | FP64本次最大差0 | 此输入共同正尺度不变；不保证任意浮点范围均严格相等 |
| `sᵀq`与`cᵀ∇cℓ` | 绝对值分别约1.58e−14、1.74e−14 | 投影和权重共同尺度不变的局部一致性 |
| `acos(0.999999)×180/π` | **0.08102847520767846°** | 固定夹断的算术下限，不是实测误差 |
| 512输入的pool8路径尺寸 | **255→127→64→32→15** | 静态卷积／池化shape规则，不是运行TensorFlow |

**尚未执行的验证：**真实图像／权重的逐层对齐、RGB通道哨兵、色板mask可视检查、三折训练、独立验证集选择、NUS跨相机实验、ReLU死区／softmax自动微分、近零与有符号相消的端到端拒判、最小图像尺寸、FP16溢出、权重转换和延迟剖析。后续应优先做这些验证，而不是用公式检查将L5改为已实现。

优化前先锁正确性：动态shape与原静态图缓存、NHWC↔NCHW、偶数6×6核的SAME非对称padding、pool8的3×3对2×2差异、均值通道顺序都须逐层对照。混合精度时至少对归一、空间累加和角损失保留适当累加精度；对有符号输出检查相消。分块推理若不保留感受野halo、边界padding与位置集合，不能声称等价于整图FCN。

## 面试自测：贯穿 L1–L5 的五问

| 层级与问题 | 答案要点 | 成立条件／易错答案 | 证据入口 |
|---|---|---|---|
| L1：给一幅512²图，FC4如何输出全局RGB？ | 绑定预处理，SqueezeNet至pool8得到15²×512，6×6/64及1×1头，按所选头形成加权局部RGB，沿空间求和后L2归一；无效方向拒判 | 15²是固定代码尺寸，不是论文额外2×2池化的复刻；不能把RGB均值当完整FC4，也不能把3通道默认分支叫无置信 | L1；[cite:1](#source-1)[cite:3](#source-3)[cite:4](#source-4) |
| L2：为何不能先归一默认三通道输出再平均？ | 输出幅值就是隐式置信；先归一抹掉长度差异，变等权消融。显式头则必须先归一RGB以免幅值双重加权 | 要区分两种参数化；最终归一能消去共同正因子，不能消去不同位置的因子；置信不是概率 | L2；[cite:2](#source-2)[cite:3](#source-3) |
| L3：三折误差很低是否可以销掉数据风险？ | 核对MCC遮挡、原图ID、FOLD轮换、相机与黑电平；作者脚本用测试误差选checkpoint，需独立内层验证或冻结预算后盲测 | test无反传不等于无选择泄漏；只遮训练色板、增强后分折、全数据测试均不合格 | L3；[cite:2](#source-2)[cite:3](#source-3)[cite:6](#source-6) |
| L4：置信怎样由全局角损失学到？ | 推出`q=−K(t−zy)/(r√(1−z²))`，`∂ℓ/∂cᵢ=vᵢᵀq`，`∇vᵢℓ=cᵢq`；再经局部归一和ReLU到logits及骨干 | s非零且在可微域；不能遗漏全局投影，也不能把cᵢq当归一前RGB梯度；隐式头无独立c输出 | L4；[cite:1](#source-1)[cite:3](#source-3) |
| L5：全黑图或局部向量相消时，加epsilon足够吗？ | 不足；方向不可辨识，检查输入／真值／局部及聚合有限性，返回invalid，回退另标；测试分母保留拒判 | 默认代码允许负输出，可能相消；训练夹断有非零损失下限，外部推理直接除范数存在零向量问题；有限差分通过不等于FC4复现 | L5；[cite:3](#source-3)[cite:9](#source-9) |

## 来源与获取层级

所有访问日期为 **2026-11-05**。优先级P0表示原论文／作者材料；获取层级另记`T`方法正文／源码、`A`作者说明／摘要、`B`书目、`限`未取得。单源事实不冒充独立复现实验，静态代码不冒充运行记录。

- <a id="source-1"></a>**[cite:1] P0／T+B，Hu、Wang、Lin，FC4: Fully Convolutional Color Constancy With Confidence-Weighted Pooling。** [CVF书目与摘要](https://openaccess.thecvf.com/content_cvpr_2017/html/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.html)，CVPR 2017，4085–4094；[CVF原论文PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf)。网页提取止于§3.1开头，随后在内存取得10页PDF，用已有pypdf读取第4–7页，核对Fig.2、§3.1–3.2、式(1)–(6)、§4.1–4.2及Table3/4；前3页由网页正文读取。不是仅凭摘要扩写；未通读全部后续讨论／参考文献，部分字体有解码警告，图内拓扑以可读正文及源码交叉核对。
- <a id="source-2"></a>**[cite:2] P0／T+A，作者固定README及提交身份。** [固定README](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/README.md)，[提交记录](https://github.com/yuanming-hu/fc4/commit/fc86df1ce82308b7dda28d71d705c25d6edd8251)。GitHub主页抓取空；改用GitHub API确认SHA与仓库树，再读取该SHA下raw正文。已读发布记录、数据FAQ、训练／复现指引和SEPARATE_CONFIDENCE说明；作者报告的代码结果不作为本仓库实测。
- <a id="source-3"></a>**[cite:3] P0／T，固定作者fcn.py。** [网络、训练和推理源码](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/fcn.py)。取得raw源码；重点逐段读取44–242、285–386、测试聚合及675–740、851–879行。核对默认有符号三通道、显式softmax、空间归一、正则登记、测试选模和acos夹断。没有安装／运行其Python2和TensorFlow1依赖，库默认值及硬件数值等价仍需环境锁定后验证。
- <a id="source-4"></a>**[cite:4] P0／T，固定作者squeeze_net.py。** [backbone源码](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/squeeze_net.py)。已读全文，尤其conv1–fire8、pool_layer、fire_module、get_features_out；确认pool8使用3×3而非论文图注2×2。未反序列化模型pickle、未检查预训练权重数值，不把自定义实现等同任意标准库SqueezeNet v1.1。
- <a id="source-5"></a>**[cite:5] P0／T，固定作者config.py。** [配置源码](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/config.py)。已读全文，核对FCN_INPUT_SIZE、augmentation、Adam、5.7e−5、6000 epochs、三个fold和两种置信开关；配置默认值不证明某个下载checkpoint恰用此全部配置。
- <a id="source-6"></a>**[cite:6] P0／T，固定作者datasets.py。** [预处理／fold源码](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/datasets.py)。已读Gehler和Cheng类、MCC坐标、黑电平、分折与主入口；未下载568／1736张图、MAT标签／fold数组或实际生成缓存。原始数据内容、mask正确性及NUS标签对齐没有动态验证。
- <a id="source-7"></a>**[cite:7] P0／T，固定作者data_provider.py。** [增强及数据供给源码](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/data_provider.py)。已读全文，核对log-uniform裁剪、旋转／翻转、图像标签通道变换、颜色缩放与夹断；未执行OpenCV增强。几何函数辅助实现只作静态参考，不声称无坐标误差。
- <a id="source-8"></a>**[cite:8] P0／T，固定作者fc4.py。** [命令入口](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/fc4.py)。读取test、外部JPEG处理与train入口，区分数据包0.5倍测试与外部图像演示；未加载预训练checkpoint、未生成校正图。
- <a id="source-9"></a>**[cite:9] P0／T，固定作者utils.py。** [角误差与辅助函数](https://github.com/yuanming-hu/fc4/blob/fc86df1ce82308b7dda28d71d705c25d6edd8251/utils.py)。已读angular_error、summary_angular_errors、get_WB_image及辅助正文；角误差函数返回弧度且无零范数保护，区别于训练损失的度和内缩clip。
- <a id="source-10"></a>**[cite:10] P0入口／限，补充材料。** [CVF页面列出的supp链接](https://openaccess.thecvf.com/content_cvpr_2017/supplemental/Hu_FC4_Fully_Convolutional_2017_CVPR_supplemental.pdf)实际返回404；README还指向[作者托管supp](https://github.com/yuanming-hu/public_files/raw/master/fc4-supp.pdf)，对应raw请求长时间未返回并已停止，本次未取得正文。不能声称补充证明、全部网络边界或补充实验已核验。

## G35 销项建议与保留边界

按台账“已有”要求具体方法机制、五层正文及问答，而不要求完成训练工程，**建议将G35／09-13从“缺卡”转“已有卡：fc4.md:FC01”**；五层记为 **核／核／部／核／部，答5**，证据记 **T+B（原文方法＋固定作者源码，单源；supp限）**。这只销掉“FC4具体机制缺卡”，不销掉实现、数据、独立评测与现场验收缺口。

后续优先级建议保留 **P2：固定环境及权重、预处理／mask／fold复核、独立验证选模、三折和跨相机复现、零向量拒判与部署剖析**。**实现状态仍为“未实现”**。本次仅创建本文件，不更新颜色总览、索引、覆盖台账或全局卡数；合并时应重新核算当前工作树，不能用本卡推算所有并行新增卡的总数。
