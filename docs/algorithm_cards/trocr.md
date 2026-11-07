# TR01 TrOCR：预训练图像 patch 编码与自回归词元识别

[学习规范](../algorithm_learning_standard.md) · [OCR 总览](../ocr_codes_evolution.md) · [覆盖台账 G31](../evolution_coverage.md#去重缺口清单) · [CRNN CR01](crnn.md#cr01) · [CTC CT01](ctc.md#ct01) · [注意力组件](learning_components.md#c18-缩放点积与多头注意力-attention)

<a id="tr01"></a>
<a id="g31"></a>

## 任务、演进与覆盖状态

**任务契约：**输入已定位、方向正确且覆盖完整文字的 RGB 单词／单行 ROI；训练另需整串真值。输出词元 ID 序列、终止状态、未修饰字符串及另行校准的候选／拒识记录。图像 Transformer 编码二维 patch，文本 Transformer 通过因果自注意力和图像交叉注意力生成文字。TrOCR 是识别器，不是文本检测器、整页阅读顺序分析器、条码解码器或自动纠错验真系统。[cite:1](#source-1)[cite:2](#source-2)

**演进定位：**切字错误与 CNN／RNN 表示、外接语言模型分离 → 联合使用预训练视觉和文本 Transformer，以整串监督学习图像到词元的条件生成 → 减少逐字框和外接语言模型依赖，但引入预训练资产、词元协议、序列搜索与显存成本 → 仍受裁切、分辨率、语言先验和跨域分布限制。与 CRNN＋CTC 是并行识别路线，不是“新版 CTC”；EAST／DB 仍可作为独立定位前端。[cite:1](#source-1)[cite:2](#source-2)

核验日期：**2026-11-05**。唯一键为 **`trocr.md:TR01`**，承接 **G31／08-06／O-H6**；模型变体、训练阶段、注意力与解码不拆卡。年份分别为 **arXiv v1：2021-09-21**、**v5：2022-09-06**、**AAAI-23：2023**。出版页记录 *37(11), 13094–13102*、Published **2023-06-26**；总览中的机构会议月份不等于该网页出版日。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

固定实现参照为 **Hugging Face Transformers 官方仓库 v4.30.2**，标签对象 `da9daa4f5e0a0c73acab09b4906d59e88bc43be1` 指向提交 **`66fd3a8d626a32989f4569260db32785c6cbf42a`**，标签日期 2023-06-13。它是库维护方正式版本，不冒充论文作者 Fairseq 原实验代码或权重快照；标签未签名，不把 Git 对象可寻址误写成签名认证。[cite:4](#source-4)

| 字段 | 状态与限定 |
|---|---|
| L1 步骤 | 已核验：原文编码／解码／训练阶段与固定库接口；版本差异独立保留 |
| L2 机制 | 已核验：patch、位置、因果条件化、交叉注意力、teacher forcing 与 EOS，限原文和静态源码 |
| L3 工程权衡 | 部分：原设置与随机编号拒识设计；无目标域定参或风险校准 |
| L4 数学推导 | 部分：条件似然、局部梯度、形状和复杂度；无全模型梯度复现或通用识别误差界 |
| L5 实现 | 部分：固定源码、参考伪代码和有限 NumPy 契约检查；未实现 TrOCR |
| 面试自测 | 答5：L1–L5 各一问，含答案、条件、易错点与证据 |
| 验证范围 | 仅文档检查与确定性代数夹具；不是网络运行、真实 OCR 评测或现场验收 |

本次仓库 Python 源码关键词检索未发现 `TrOCR`、`trocr`、`VisionEncoderDecoder` 或 `transformers` 接入；当前虚拟环境没有 PyTorch。**本仓库未实现 TrOCR，没有加载权重、训练、图像推理或专用端到端测试。**本卡内检查器不改变此状态。只创建此文档；索引、总览、覆盖台账的旧“缺卡”字段不在本次修改范围。

## L1 步骤：训练与推理分开走通

### 图像 patch、位置与编码形状

统一图像布局为 **[B,C,H,W]**，尺寸按高×宽，序列为 **[B,S,D]**。原文输入 C=3，直接 resize 到 **384×384**，patch 为 **16×16**，因此网格为 **24×24**、图像 patch 数 **N=576**，每块展开维度 **3×16×16=768**。这不是 576 个字符，也不是保持长宽比的任意宽输入。[cite:1](#source-1)[cite:2](#source-2)

| 位置 | 操作 | 形状与边界 |
|---|---|---|
| ROI | RGB 转浮点，按锁定处理器缩放与归一 | [B,3,384,384]；不是 OpenCV 的默认 BGR |
| Patchify | 行优先遍历网格，块内按通道／行／列展开 | [B,576,768]；不把 batch 轴混入网格 |
| Patch embedding | 每块同一个线性映射 768→D_e | [B,576,D_e] |
| 特殊视觉 token | CLS；DeiT 蒸馏变体另有 distillation token | S=577 或 578，不能总把 S 写成576 |
| 位置与 encoder | 加可学习位置编码，视觉双向 self-attention＋FFN | E：[B,S,D_e]；跨 patch 全局交互 |
| Decoder memory | 接整个视觉序列，必要时投影宽度 | [B,S,D_m]；不只取 CLS 作整图分类 |
| Teacher-forced decoder | 右移后的 token 输入、因果 mask、cross-attention | 隐状态 [B,T,D_d] |
| 词表读出 | 线性映射至 V 个词元，再 softmax | logits：[B,T,V]；T 与 S 不必相等 |

固定 ViT 实现用 `Conv2d(kernel_size=P,stride=P)` 实现共享 patch 线性投影，之后 `flatten(2).transpose(1,2)`；这在算子上使用卷积，不等于使用 CNN 骨干。论文“convolution free”描述骨干路线，不能解释为实现里绝不出现 Conv2d。该 ViT 路径只加 CLS；论文 DeiT 蒸馏路线的第二视觉 token 不能凭模型名字擅自添加。[cite:5](#source-5)

### 预训练文本 decoder 与词元协议

文本初始化不是“直接拿双向 RoBERTa 做推理”：迁移兼容的嵌入、self-attention、FFN 和归一化参数，改为因果注意力，插入原文本编码器没有的 encoder-decoder attention；缺失参数随机初始化。2021 v1 写 RoBERTa；AAAI 正文还写 MiniLM。不能宣称交叉注意力已经从 RoBERTa 学会看图，也不能声称加载 checkpoint 等于完成跨模态训练。[cite:1](#source-1)[cite:2](#source-2)

每层 decoder 的主路径是 **masked self-attention → cross-attention → FFN**，各子层有残差／归一化。cross-attention 的 **Q 来自 decoder，K/V 来自 encoder memory**；固定 TrOCR decoder 为子层输出后残差相加再 LayerNorm，不应凭通用 Transformer 印象改成 Pre-LN 并直接加载原权重。[cite:6](#source-6)

| 版本／对象 | 起始与标签约定 | 不能混写的内容 |
|---|---|---|
| 2021 v1 §2.3 | 目标末尾 EOS；将 EOS 旋转至 decoder 输入开头；推理从 EOS 开始 | 原文确实如此，不改写成该版本明示 BOS |
| AAAI 2023 Task Pipeline | 目标末尾 EOS；右移后在开头加 BOS；推理从 BOS 开始 | 这是符号级描述，不给跨库通用 token ID |
| 固定库 `TrOCRConfig` 默认值 | `bos_token_id=0`、`pad_token_id=1`、`eos_token_id=2`、`decoder_start_token_id=2`，V=50265 | 生成起点由 start 控制，不必等于 bos；类默认值不是已下载模型配置 |
| 本卡参考契约 | `target=[y1,…,yU,EOS]`，`input=[START,y1,…,yU]`，另补 PAD | START 必须由选定权重／训练协议冻结，不擅自改为0或2 |

以上差异分别来自原文与固定代码。[cite:1](#source-1)[cite:2](#source-2)[cite:7](#source-7) 若 START 与 EOS 共用 ID，**输入前缀中的起始 EOS 不触发完成**；只有新生成 EOS 才结束。不要用“序列中出现过 EOS”检查停止，否则第一步就停。

论文的 wordpiece 是子词生成单位统称：v1 使用 Fairseq BPE，后期正文写 BPE／SentencePiece；不是默认 BERT WordPiece，更不是一词元等于一字符。部署必须同时锁定 tokenizer、词表、merges／SentencePiece 模型、特殊符号、Unicode／空格规范及 decode 的清理选项。普通 RoBERTa tokenizer 自动加的 BOS/EOS 与训练标签格式是否一致需查权重协议；本卡不把 `add_special_tokens=True` 当普遍正确捷径。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

### 训练阶段与完整流程

1. **基础模型初始化**：从视觉预训练模型与文本预训练模型迁移兼容参数，记录新建层、层映射、位置编码与词表。视觉 masked image modeling／文本 masked language modeling 是基础模型来源，不是后续 OCR loss。
2. **OCR 预训练第一阶段**：按图像—文本配对做条件生成训练。v1 与 v5 记载从公开 digital-born PDF 页面取得 **684M textlines**，不是684M人工标注真实手写行，也不是本仓库持有的数据。
3. **OCR 预训练第二阶段**：从第一阶段分别初始化任务分支；印刷约 **3.3M**、手写约 **17.9M** 行。印刷数据含真实收据经商业 OCR 生成的伪标签及合成行，手写含字体渲染与 IIIT-HWS；不能统称全部人工真值。v5 加入场景文字 MJ＋ST 约 **16M** 图像的分支。
4. **下游微调**：印刷 SROIE、手写 IAM 等使用各自下游标签继续训练；AAAI 正文明确场景文字实验是上述微调描述的例外，不能编成所有任务都再做同一轮微调。
5. **每一步训练**：图像预处理 → encoder → token 右移 → teacher forcing decoder → 与原目标逐位置对齐的交叉熵 → 反传 decoder、新交叉层和未冻结 encoder → 优化器更新。PAD 不计 loss，EOS 计 loss；训练不需要对 argmax 或 beam 反传。
6. **训练停止**：实际复现须预先固定预算、验证集早停／选模准则及随机种子，保存模型、优化器、tokenizer 和处理器。本文未核到足以完整重放所有阶段的学习率日程、步数与训练资产，不补造这些参数。

阶段事实与规模来源分别为 v1 §2–3、AAAI Pre-training／Fine-tuning 及 v5 Data；规模仅作原文训练来源记录，不作独立实验交叉验证或效果保证。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

推理使用 eval、不构建梯度，先编码一次图像；从 START 开始，根据图像与已生成前缀求下一词元分布，以 greedy 或 beam 搜索。新生成 EOS、最大新词元数、位置上限、时间／资源预算均可能停止；**没有自然生成 EOS 的长度截断必须标为 truncated，而不是正常完成**。保留原始 ID 与候选分数，最后去特殊符号并反词元化，不做 CTC 的相邻重复合并。文本里两个相同 token 可以是两个真实字符或片段。

## L2 机制：每一步的必要性与反例

| 设计 | 为什么这样做 | 删除／误用的后果与边界 |
|---|---|---|
| 图像 patch 与共享投影 | 将二维像素转成可被 Transformer 处理的局部视觉序列 | 一个 patch 可覆盖多个笔画或多个字；不能把 patch attention 峰当字符框 |
| 位置编码 | 给自注意力提供网格位置／顺序信息 | 无位置的注意力对输入置换等变，难以区分字序；不意味着可自动适应任意新网格 |
| 文本预训练 | 为 token 表示和上下文提供初始化 | 不是已学会视觉对齐；编号域中常见串先验可能压过弱字形证据 |
| Cross-attention | 当前文字状态从整张 ROI 取视觉信息 | 去掉后主要成为条件不足的语言生成器；输出流畅不代表看对图 |
| 因果 mask＋右移 | 第 t 个输出只见 START 与 y<t，不能偷看当前答案 | 不右移会学复制；不加因果 mask 可从后续位置读到真值，训练低loss而推理崩溃 |
| Teacher forcing | 用真值前缀并行计算所有训练位置的条件似然 | 推理前缀来自模型，存在 exposure bias；不能把训练CE换算成部署整串准确率 |
| EOS 与 PAD 分离 | EOS 是受监督的终止事件，PAD 是存储补齐 | 把 EOS mask 掉会破坏终止学习；把 PAD 计入loss会使短串补齐主导训练 |
| Beam search | 保留多个前缀，缓解早期局部最优 | 有限beam仍会剪掉真串；更宽beam可能更偏语言先验，不能用它解决无法辨识的O/0 |

机制由原文与固定 attention／训练接口支持；反例和部署后果是条件分析，不冒充原论文工业验证。[cite:1](#source-1)[cite:2](#source-2)[cite:6](#source-6)[cite:8](#source-8)

TrOCR 建模的是 `p(y_t | y_<t, image)`，不是 CTC 的所有 blank 路径求和；没有 CTC 的 `T≥U+相邻重复数` 对齐约束，也没有 blank 折叠。它仍有图像分辨率、token 预算和位置嵌入容量限制。注意力可以跨不同区域读信息，但不提供正确二维阅读顺序或单调字符定位的理论保证。

## L3 工程权衡：版本、预处理与随机编号拒识

### 架构与设置分账

| 对象 | 原始资料中可核验的配置 | 不能据此推出 |
|---|---|---|
| v1 最终 TrOCR BASE | BEiT BASE：12层／768维／12头；decoder 为 RoBERTa LARGE 的后12层／1024维／16头 | BASE 不表示两端都是 base，也不表示decoder只有512维 |
| v1／v5 TrOCR LARGE | BEiT LARGE：24层／1024维／16头；large decoder：12层／1024维／16头 | 不能把文本原始24层全部照搬 |
| v5 SMALL | DeiT SMALL：12层／384维／6头；MiniLM small decoder：6层／256维／8头 | 是后期新增分支，不回填为v1完整配置 |
| 消融中的 base decoder | v1写6层／512维／8头，又写迁移RoBERTa BASE后6层 | 后者常规宽度与512的直接映射没有在已读正文解释；不得声称能无适配直接加载 |
| 固定库 TrOCRConfig | 默认decoder 12层／1024维／16头／FFN4096，位置容量512，V=50265 | 这是类默认值；不能替代任意small/base/large权重配置 |

v1 架构与后期变体／beam设置来源不同，明确区分。[cite:1](#source-1)[cite:3](#source-3)[cite:7](#source-7) AAAI 已读核心正文确认 MiniLM 和场景分支，但本次下载全文受403／反机器人页面影响；上表后期细节明确据 **2022 v5**，不伪称逐项读到会议设置页，也不宣称二者逐字相同。

v1／v5 的 Settings 写 batch=2048、learning rate=5e-5；v5 写 beam=10。这些是原实验设置，不是本地推荐常数，更不涵盖缺失的全部训练日程。固定库 `CrossEntropyLoss()` 接口本身没有 label smoothing；原文写交叉熵，本文不猜原 Fairseq recipe 是否配置了 smoothing。若后续启用，需记录其定义并用 L4 的软标签梯度，不能仍声称与无平滑目标相同。[cite:1](#source-1)[cite:3](#source-3)[cite:8](#source-8)

### 预处理与 padding 的实际边界

原文384×384直接缩放可能拉伸长行。若改为等比缩放＋补边、切片或更大宽度，需重新处理位置编码、有效支持、token预算及训练分布；仅改变输入尺寸不是无损优化。ROI→原图坐标变换与未增强原图需保存，文本被裁掉或饱和后没有搜索参数能恢复证据。

固定库 `ViTImageProcessor` 的**类默认**尺寸是224×224，插值 BILINEAR、rescale=1/255，再按 mean/std 归一；原模型处理器配置会覆盖默认值。不能裸建默认处理器再声称得到论文384输入。数值协议必须来自所选权重资产；本卡的明确参考约定为RGB uint8转float、384×384、`x=(p/255−0.5)/0.5`，属于待验证方案，不冒称已取得目标checkpoint的processor JSON。[cite:9](#source-9)

- 已是[0,1]的浮点图不能再次除255；uint8先转浮点再做减法。灰度需明确复制为三通道还是重新训练单通道输入，不能隐式广播蒙混。
- 图像padding不因为像素为0就“不参加注意力”。固定 `VisionEncoderDecoderModel.forward` 将 `encoder_attention_mask=None`；即使decoder接受mask，该包装路径也不是变宽有效patch屏蔽方案。更早的encoder self-attention已可传播补边信息。
- 文本采用右padding，传入正确的 decoder attention mask；loss 的 `-100` 只屏蔽该位置的监督，不等于屏蔽整个网络。左padding、位置编号或cache偏移改变后，必须建立独立对照。
- tokenizer和权重必须一起锁定；未知字符、空白、组合字符、大小写不能先“修好”再算CER。空真值在数学上可训练为EOS，但工业非空字段的空输出应按业务拒识。

padding与loss接口依据固定库源代码；其余为工程防错条件。[cite:6](#source-6)[cite:8](#source-8)

### 随机编号的拒识与验收

随机序列号没有自然语言的词频规律。应先隔离“看不清”与“偏向常见串”：保存原ROI、token级分数、top候选、greedy／beam差异及变换前后结果；在新编号与常见词两组中比较错误。**beam分数、平均token概率、EOS概率都不是已校准的整串正确概率。**以下是本卡风险设计，不是论文内置拒识模块。

| 风险／症状 | 优先诊断与处理 | 验收证据 |
|---|---|---|
| O/0、I/1、B/8高分误读 | 先看原始笔画与成像；再查编号域训练覆盖、候选差值与语义先验 | 新编号、近似字符、罕见组合单独分层；不以常用词集代替 |
| 合法但错误的编号 | 同时保存无约束原串与格式约束候选；校验码／外部工单核对独立记录 | 格式合法只证明语法；错误工件的合法编号仍可能误放行 |
| 缺字、提前EOS、超长 | 查完整ROI、tokenization、START／EOS／max_new_tokens与位置容量 | EOS完成和截断分开；超时、无有限候选、空串分别给原因 |
| beam变宽后串更“像词” | 冻结模型，比较搜索宽度对随机串整串正确率与误放行的影响 | 不只报告更高模型得分，也不将选中真值的beam当无条件成功 |
| 置信度阈值迁移失效 | 按字体、批次、设备、编号长度隔离校准和测试 | 绘制风险—覆盖率；冻结门限后独立测试，报告样本量与分母 |

建议输出契约为 `{raw_ids, raw_text, candidates, score_definition, ended_by_eos, truncated, reject_reason, roi_transform, model_revision, tokenizer_revision, processor_revision, business_check}`。模型阈值与第二候选差值只在固定解码／归一协议下比较；可联合图像质量、完成状态、格式与独立业务核对拒识，不能给通用数值门限。拒识后进入复拍／人工复核，而非自动改成最常见合法串。

验收至少分别记录：全部应读样本的整串正确率、CER=(替换＋删除＋插入)/参考字符数、拒识率、错误接受数/全部应读数，以及错误接受数/接受数。空参考和零接受另列，拒识不能从分母消失；CER可能大于1。数据按工件／连拍／批次／字体来源分组后再增强，并专门留出未见编号，避免同标签跨集合。没有真实编号试验，故本文不给误放行上界或产线精度承诺。

参数调节顺序为 **采集与ROI → 权重／tokenizer／处理器一致性 → 标签与mask → 长度预算 → 微调数据和优化 → beam／拒识 → 性能**。beam宽K无单位，输入分辨率单位为像素，最大长度单位为token而不是字符；扩大K和T都增加尾延迟，不能以单步decoder时间代表整链耗时。

## L4 数学推导：条件似然、交叉梯度与复杂度

### Patch 投影与注意力形状

令行优先patch矩阵 X_p∈R^(B×N×3P²)，投影 W_p∈R^(3P²×D_e)。加s个特殊token与位置矩阵后：

\[
Z_0=\operatorname{Concat}(Z_{special},X_pW_p+b_p)+P_{pos},\qquad
N=(H/P)(W/P),\quad S=N+s.
\]

H、W均可被P整除是本文patch契约；一般卷积遇非整除会floor丢边，不能默认为图像完整覆盖。块重排的反传是逆置换，投影梯度为 `dW_p=Σ X_pᵀdZ_p`、`dX_p=dZ_p W_pᵀ`。特殊视觉token不是文本BOS/EOS；s的变化同时改变位置编码和memory长度。

对一个cross-attention头，令decoder输入D∈R^(T×D_d)、memory E∈R^(S×D_m)：

\[
Q=DW_Q,\quad K=EW_K,\quad V=EW_V,\qquad
A=\operatorname{softmax}_{S}(QK^T/\sqrt d+M),\quad O=AV.
\]

Q为T×d，K/V为S×d，A为T×S；多头得到[B,h,T,S]，按head拼接后输出投影回D_d。self-attention则Q/K/V来自同一序列；decoder self mask在右移输入的 j≤t 处为0、j>t处为−∞。它允许位置t看输入t，因为该输入是y_(t−1)，不是监督y_t；混用“相对目标”与“相对已右移输入”的索引会错屏蔽对角线。[cite:1](#source-1)[cite:6](#source-6)

若D_e≠D_d，固定包装器在 `cross_attention_hidden_size is None` 时用线性层D_e→D_d；若显式设置cross维度，K/V可直接投影不同宽度memory。不能把[B,S,768]无投影当作[B,S,1024]，也不能在已有适配层外再随意加一次。[cite:8](#source-8)

### Teacher forcing、交叉熵与梯度

设真值为 U 个内容token，y_(U+1)=EOS，T=U+1。按概率链式法则：

\[
p_\theta(y\mid I)=\prod_{t=1}^{U+1}p_\theta(y_t\mid y_{<t},I),\qquad
L=-\frac1M\sum_{b,t}m_{bt}\log p_{bt,y_{bt}},\quad M=\sum_{b,t}m_{bt}>0.
\]

这里m为非PAD目标mask，EOS的m=1；M是整batch有效token数，故长行贡献更多token项。与“先每样本均值再batch均值”不同。固定包装器展平logits与labels后调用CE，标签中 `-100` 作为忽略值；传入的目标PAD若仍是词表ID=1，就会作为真正类别被训练。全忽略目标M=0要主动拒绝，不把NaN替换成零冒充成功。[cite:8](#source-8)

令z为logits，p=softmax(z)。由 `log p_y=z_y−logΣexp(z_k)` 得：

\[
\frac{\partial L}{\partial z_{btk}}=\frac{m_{bt}}M\left(p_{btk}-\mathbf1[k=y_{bt}]\right).
\]

若明确采用均匀label smoothing，`q_k=(1−ε)1[k=y]+ε/V`，梯度改为 `m(p−q)/M`；另一种仅向非目标类分配ε的约定不同，不能混用。稳定实现用log_softmax／logsumexp，不先取极小概率再log。PAD目标位置的**直接logits梯度**为零，不保证所有补齐像素／参数的梯度为零，因为其他有效输出可依赖它们。

对单头注意力，令G=∂L/∂O，上游矩阵链式求导为：

\[
\bar V=A^TG,\quad \bar A=GV^T,\quad
\bar R=A\odot\left(\bar A-\operatorname{rowsum}(A\odot\bar A)\right),
\]

\[
\bar Q=\bar RK/\sqrt d,\quad \bar K=\bar R^TQ/\sqrt d,\quad
\bar E=\bar K W_K^T+\bar V W_V^T.
\]

行和按列广播；在有效mask内做softmax微分，禁止所有key被mask的行。`dW_K=EᵀdK`、`dW_V=EᵀdV`；梯度从词表头经过decoder，再通过每层cross-attention的K/V两路回到encoder，encoder另有自身残差路径。teacher forcing中的离散真值ID没有对ID的梯度，但查表得到的embedding权重可学习；argmax／beam不在这条训练反传链。此为局部可微推导，不是整套预训练网络梯度正确性的证明。

### 解码目标、近似与资源

原始候选得分为 `s(y)=Σ_t log p(y_t|y_<t,I)`，应包括新生成EOS，不包括给定START的概率。greedy只最大化当前条件项；beam每步保留有限K条前缀仍是近似。若用长度归一 `s(y)/l(y)^α`，必须冻结l是否包括START／EOS；它改变排序目标，不是严格MAP。固定库 `BeamHypotheses.add` 用传入hyp长度的幂作除数，因此不能把库的 `sequences_scores` 直接解释为“每字符平均log概率”。[cite:10](#source-10)

**有限反例的结构：**第一步较高概率的A，其最优后续条件概率可以小于另一前缀B的后续优势，因此 `p(A)>p(B)` 不推出 `max_z p(A,z)>max_z p(B,z)`。L5用明示的概率树检查此代数事实；这些不是模拟OCR成绩。对原始累计log分，后续log概率≤0，当前前缀分是后续完整串分的上界；长度归一时该上界不能照搬。固定库区分 `early_stopping=True/False/"never"`，不能把收到一个EOS就停称为正确beam终止。[cite:10](#source-10)

忽略batch并以层数L_e、L_d表示，标准密集注意力的主要量级为：

\[
\text{encoder}:O(L_e(SD_e^2+S^2D_e)),\qquad
\text{teacher-forced decoder}:O(L_d(TD_d^2+T^2D_d+STD_d)+TD_dV).
\]

交叉K/V投影还含每层O(SD_mD_d)，FFN宽度非固定比例时另列O(TD_dD_ff)。注意力矩阵存储约O(hS²)与O(hT²+hTS)每层；开启权重输出会扩大内存。推理有K条beam时decoder主要项乘K；KV cache避免每步重算旧前缀投影，累计self-attention仍约O(KT²D_d)，cross部分约O(KTSD_d)，不是“缓存后整个解码O(T)”。未缓存而反复全前缀前向会重复做更高阶工作。

E及cross K/V在逻辑上可复用，但实现可能为beam展开存储；缓存空间约O(KL_d(T+S)D_d)。beam重排必须同步所有层缓存与样本索引；量化、不同attention内核、剪枝及长度限制不保证输出等价，需比较有效logits和序列。未提供通用识别误差界、全局优化收敛保证、任意beam精确解码或部署时延上界。

## L5 实现：固定源码、参考伪代码与有限检查

### 源码对应

全部代码路径相对 `transformers` 仓库，版本固定在前述提交；源码只读取相关段落，未在本仓库安装或运行该库。

| 逻辑 | 固定源码位置 | 核验重点 |
|---|---|---|
| Patchify／CLS | `models/vit/modeling_vit.py`，ViTPatchEmbeddings、ViTEmbeddings | Conv2d投影、flatten／transpose、CLS及位置插值分支 [cite:5](#source-5) |
| 因果mask／cross／cache | `models/trocr/modeling_trocr.py`，TrOCRAttention、TrOCRDecoderLayer、TrOCRDecoder | K/V来源、post-LN、缓存拼接、位置偏移 [cite:6](#source-6) |
| 词表／特殊ID | `models/trocr/configuration_trocr.py`，TrOCRConfig | START与BOS分离、默认维度与位置容量 [cite:7](#source-7) |
| 训练桥接 | `models/vision_encoder_decoder/modeling_vision_encoder_decoder.py`，shift_tokens_right、forward | 宽度投影、右移一次、−100转PAD、同位置CE、视觉mask为None [cite:8](#source-8) |
| 预处理 | `models/vit/image_processing_vit.py`，ViTImageProcessor | 默认224不能代替384资产；resize→rescale→normalize [cite:9](#source-9) |
| 搜索完成与长度分 | `generation/beam_search.py`，BeamSearchScorer、BeamHypotheses | EOS完成集合、长度分母、不同早停策略 [cite:10](#source-10) |

### 训练与解码参考伪代码

以下大写操作为**待实现接口**，不是仓库API。选择通用 `START` 契约、右padding、无label smoothing、按有效token均值；显式decoder输入时不再让包装器移位一次。使用真实checkpoint时先复核其完整标签格式，不能用这段规范替换未知的训练协议。

```text
TRAIN_STEP(images, texts, assets):
    REQUIRE assets.tokenizer_revision and assets.model_revision and assets.processor_revision
    pixels = PREPROCESS_RGB(images, assets.processor)
    content_ids = TOKENIZE_CONTENT(texts, add_special_tokens=false)
    targets, lengths = APPEND_EOS_AND_RIGHT_PAD(content_ids, assets.eos, assets.pad)
    REQUIRE ALL(lengths <= assets.decoder_position_budget)
    labels = MASK_TARGET_PADDING_WITH_MINUS_100(targets, lengths)
    decoder_ids = SHIFT_RIGHT_ONCE(labels, assets.start, assets.pad)
    decoder_mask = POSITIONS_LESS_THAN(lengths)
    REQUIRE COUNT(labels != -100) > 0
    ZERO_GRAD()
    E = ENCODER(pixels)
    logits = DECODER(decoder_ids, E, decoder_mask, causal=true, use_cache=false)
    loss = MASKED_CROSS_ENTROPY_MEAN(logits, labels)
    REQUIRE IS_FINITE(loss)
    BACKWARD(loss)
    REQUIRE ALL_TRAINABLE_GRADIENTS_FINITE()
    OPTIMIZER_STEP()
    RETURN loss

BEAM_REFERENCE(image, K, max_new_tokens, assets):
    REQUIRE K >= 1 and max_new_tokens >= 1
    E = ENCODER_EVAL(PREPROCESS_RGB(image, assets.processor))
    active = [(ids=[assets.start], log_score=0)]
    finished = []
    FOR step IN 1..max_new_tokens:
        candidates = []
        FOR prefix IN active:
            logp = LOG_SOFTMAX(NEXT_LOGITS_EVAL(prefix.ids, E))
            REQUIRE ALL_FINITE(logp)
            FOR token IN ALLOWED_TOKENS(assets):
                candidate = APPEND_WITH_SCORE(prefix, token, logp[token])
                IF token == assets.eos:
                    finished.ADD(candidate)
                ELSE:
                    candidates.ADD(candidate)
        active = TOP_K_BY_RAW_LOG_SCORE(candidates, K, stable_tie_order=true)
        IF active.IS_EMPTY(): BREAK
    IF finished.IS_EMPTY(): RETURN REJECT(reason=truncated_or_no_eos, partial=active)
    best = MAX_BY_RAW_LOG_SCORE(finished)
    unresolved = ANY(prefix.log_score > best.log_score FOR prefix IN active)
    IF unresolved: RETURN REJECT(reason=search_budget_unresolved, best=best, partial=active)
    raw = DETOKENIZE(best.ids, remove_only_declared_special_tokens=true, cleanup=false)
    RETURN CALIBRATED_ACCEPT_OR_REJECT(raw, best, EVIDENCE(image, assets))
```

这里beam参考为易审计的**无缓存、原始log分、分开保存已完成候选**版本，不逐行复制官方beam。K=1是单活跃前缀搜索，不等于只取一个token的greedy（EOS候选另存）；真正greedy需每步只保留argmax且遇生成EOS停止。由于剪枝，`unresolved=false`只对保留下来的搜索路径有效，不能升级为全词表树全局最优证明。若启用长度惩罚，应替换上界与停止规则，不能沿用原log分的比较。

### 可执行的有限代数检查器

以下Python代码只依赖已存在的NumPy；使用确定性的整数网格、向量和小概率树作为数学夹具，**不是合成OCR数据、模型预测或性能模拟**。没有encoder／decoder完整网络，也没有加载tokenizer；有限差分检查局部光滑子图，不能继承为全模型反传通过。

```python
import math
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def decoder_attention(scores, values, valid):
    causal = np.tril(np.ones(scores.shape[-2:], dtype=bool))
    mask = causal[None] & valid[:, None, :]
    if not mask.any(axis=-1).all():
        raise ValueError("all keys masked")
    attn = softmax(np.where(mask, scores, -np.inf))
    return attn, attn @ values

def patchify(x, p):
    b, c, h, w = x.shape
    if h % p or w % p:
        raise ValueError("nondivisible image")
    return x.reshape(b, c, h // p, p, w // p, p).transpose(0, 2, 4, 1, 3, 5).reshape(b, -1, c * p * p)

def finite_difference(f, x):
    h = 1e-6
    out = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        a, b = x.copy(), x.copy()
        a[idx] += h
        b[idx] -= h
        out[idx] = (f(a) - f(b)) / (2 * h)
    return out

def masked_ce(z, labels):
    valid = labels != -100
    if not valid.any():
        raise ValueError("empty supervision")
    z = z[valid]
    y = labels[valid]
    m = z.max(axis=-1)
    return np.mean(m + np.log(np.exp(z - m[:, None]).sum(axis=-1)) - z[np.arange(len(y)), y])

def run_checks():
    x = np.arange(2 * 3 * 384 * 384).reshape(2, 3, 384, 384)
    patches = patchify(x, 16)
    assert patches.shape == (2, 576, 768)
    assert 576 + 1 == 577 and 576 + 2 == 578
    np.testing.assert_array_equal(patches[1, 25], x[1, :, 16:32, 16:32].reshape(-1))
    try:
        patchify(np.zeros((1, 3, 17, 16)), 16)
    except ValueError:
        pass
    else:
        raise AssertionError("missing divisibility guard")
    xp = np.arange(24, dtype=float).reshape(2, 12) / 24
    w = np.arange(36, dtype=float).reshape(12, 3) / 36
    g = np.arange(6, dtype=float).reshape(2, 3) / 6
    np.testing.assert_allclose(finite_difference(lambda a: np.sum((xp @ a) * g), w), xp.T @ g, atol=1e-8)
    labels = np.array([[3, 4, 2, -100], [4, 2, -100, -100]])
    inputs = np.full_like(labels, 1)
    inputs[:, 0] = 2
    inputs[:, 1:] = np.where(labels[:, :-1] == -100, 1, labels[:, :-1])
    np.testing.assert_array_equal(inputs, [[2, 3, 4, 2], [2, 4, 2, 1]])
    valid = labels != -100
    causal = np.tril(np.ones((4, 4), dtype=bool))
    scores = np.arange(32, dtype=float).reshape(2, 4, 4) / 10
    pad_values = np.arange(8, dtype=float).reshape(2, 4, 1)
    attn, output = decoder_attention(scores, pad_values, valid)
    pad_keys = np.broadcast_to(~valid[:, None, :], attn.shape)
    causal_visible_pad = pad_keys & causal[None]
    assert causal_visible_pad.any()
    assert np.all(attn[pad_keys] == 0), "padding key weights must be zero"
    assert np.all(attn[:, ~causal] == 0)
    np.testing.assert_allclose(attn.sum(axis=-1), 1.0, rtol=0, atol=1e-15)
    changed_scores = scores + 100 * pad_keys
    changed_values = pad_values + 100 * (~valid)[..., None]
    changed_attn, changed_output = decoder_attention(changed_scores, changed_values, valid)
    np.testing.assert_array_equal(attn, changed_attn)
    np.testing.assert_array_equal(output, changed_output)
    for b in range(len(valid)):
        length = int(valid[b].sum())
        short_attn = softmax(np.where(causal[:length, :length], scores[b, :length, :length], -np.inf))
        np.testing.assert_allclose(output[b, :length], short_attn @ pad_values[b, :length], rtol=0, atol=1e-15)
    causal_only = softmax(np.where(causal[None], scores, -np.inf))
    causal_changed = softmax(np.where(causal[None], changed_scores, -np.inf))
    causal_output = causal_only @ pad_values
    causal_changed_output = causal_changed @ changed_values
    np.testing.assert_array_equal(output[valid], causal_output[valid])
    np.testing.assert_array_equal(causal_output[valid], causal_changed_output[valid])
    assert np.all(causal_only[causal_visible_pad] > 0)
    assert np.max(np.abs(causal_output[~valid] - causal_changed_output[~valid])) > 1
    try:
        decoder_attention(scores, pad_values, np.zeros_like(valid))
    except ValueError:
        pass
    else:
        raise AssertionError("missing all-masked guard")
    values = np.arange(4, dtype=float)[:, None]
    future_attn = softmax(np.where(causal, 0.0, -np.inf))
    changed = values.copy()
    changed[-1] += 100
    np.testing.assert_array_equal((future_attn @ values)[:3], (future_attn @ changed)[:3])
    z = np.arange(40, dtype=float).reshape(2, 4, 5) / 20
    analytic = softmax(z)
    for b, t in zip(*np.where(valid)):
        analytic[b, t, labels[b, t]] -= 1
    analytic *= valid[..., None] / valid.sum()
    numeric = finite_difference(lambda a: masked_ce(a, labels), z)
    np.testing.assert_allclose(numeric, analytic, atol=1e-8)
    assert np.all(numeric[~valid] == 0)
    assert math.isfinite(masked_ce(z + 1000, labels))
    try:
        masked_ce(z, np.full_like(labels, -100))
    except ValueError:
        pass
    else:
        raise AssertionError("missing empty-loss guard")
    q = np.array([[0.2, -0.1], [0.3, 0.4]])
    k = np.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.2]])
    v = np.array([[0.2, 0.7], [-0.4, 0.1], [0.3, 0.5]])
    upstream = np.array([[0.3, -0.2], [0.1, 0.4]])
    a = softmax(q @ k.T / math.sqrt(2))
    da = upstream @ v.T
    dr = a * (da - (a * da).sum(axis=1, keepdims=True))
    grads = (dr @ k / math.sqrt(2), dr.T @ q / math.sqrt(2), a.T @ upstream)
    def objective(q0, k0, v0):
        return np.sum((softmax(q0 @ k0.T / math.sqrt(2)) @ v0) * upstream)
    numeric_grads = (
        finite_difference(lambda t: objective(t, k, v), q),
        finite_difference(lambda t: objective(q, t, v), k),
        finite_difference(lambda t: objective(q, k, t), v),
    )
    for actual, expected in zip(numeric_grads, grads):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    def probs(prefix):
        if not prefix:
            return {"A": 0.6, "B": 0.4}
        if prefix == ("A",):
            return {"EOS": 0.5, "X": 0.5}
        return {"EOS": 1.0}
    active, finished = [((), 0.0)], []
    for step in range(2):
        children = []
        for prefix, score in active:
            for token, p in probs(prefix).items():
                item = (prefix + (token,), score + math.log(p))
                (finished if token == "EOS" else children).append(item)
        active = sorted(children, key=lambda item: item[1], reverse=True)[:2]
    best = max(finished, key=lambda item: item[1])
    assert best[0] == ("B", "EOS")
    np.testing.assert_allclose(math.exp(best[1]), 0.4)
    assert 0.6 * 0.5 < 0.4
    def finish_status(start, generated, eos, budget):
        ended = eos in generated
        return ended, not ended and len(generated) >= budget
    assert finish_status(2, [], 2, 2) == (False, False)
    assert finish_status(2, [3, 2], 2, 2) == (True, False)
    assert finish_status(2, [3, 3], 2, 2) == (False, True)
    return {"groups": 5, "padding_key_max_abs": float(np.max(np.abs(attn[pad_keys]))), "padding_perturbation_max_abs": float(np.max(np.abs(output - changed_output))), "causal_only_padding_key_max": float(np.max(causal_only[causal_visible_pad])), "cross_gradient_max_abs": max(float(np.max(np.abs(n - g))) for n, g in zip(numeric_grads, grads)), "ce_gradient_max_abs": float(np.max(np.abs(numeric - analytic)))}

print(run_checks())
```

检查分五组：①patch形状／布局／不可整除守卫与投影梯度；②右移、组合因果／padding key mask、右padding对照与全key屏蔽守卫；③PAD目标的直接梯度屏蔽、稳定CE及空监督守卫；④单头Q/K/V梯度；⑤有限beam概率树、START=EOS与截断。判据是明确assert及梯度绝对容差1e-8（另保留NumPy默认相对容差），不是声称所有输入域通过。

**两个不同命题：**固定位置与有效前缀、仅在右侧补齐时，有效query本来就不能经因果注意力访问未来pad key；因此截短／补齐的有效输出一致、甚至去掉padding key mask后有效输出仍一致，都不足以证明key mask生效。本检查另外检查**所有query对被屏蔽key的权重精确为零**，并要求存在因果上可见的pad key（位于补齐query的过去或对角线），扰动这些key对应的scores／V后所有query输出不变。补齐query仅用作诊断探针，不增加监督；key屏蔽也不意味着补齐query的输出本身为零。`valid` 是本卡右移／有效长度契约，不是通过 `inputs != PAD` 猜出的mask：补齐槽位中可能仍有右移过来的EOS。

以下变异检查在**仓库根目录**运行，只读本文第一段Python并在内存中删除组合mask里的 `valid` 限制，不写文件。基线必须通过；变异必须因同一个“pad key权重为零”断言而失败，若未失败或错误原因不同则检查器报错。上面的causal-only对照说明为何不能只用有效右padding输出不变来杀死此变异。

```python
from pathlib import Path
import re

card = Path("docs/algorithm_cards/trocr.md")
code = re.findall(r"^```python\n(.*?)^```", card.read_text(encoding="utf-8"), re.M | re.S)[0]
original = "mask = causal[None] & valid[:, None, :]"
assert code.count(original) == 1
exec(compile(code, str(card), "exec"), {})
mutant = code.replace(original, "mask = np.broadcast_to(causal[None], scores.shape)", 1)
try:
    exec(compile(mutant, str(card) + ":without_valid", "exec"), {})
except AssertionError as error:
    assert str(error) == "padding key weights must be zero", str(error)
    print("TR01 mutation: removing valid failed as required: " + str(error))
else:
    raise AssertionError("removing valid survived")
```

尚未执行的模型级测试必须保留：实际tokenizer往返与特殊符号；全部层前反向；冻结／解冻参数；真实图像单张与batch一致性；空／全mask视觉memory守卫；eval下cache开关logits一致性；beam重排不串样本；FP32与混合精度；量化后整串等价；EOS被强制加入与自然预测的区分；新编号CER、整串正确率、拒识风险与端到端尾延迟。固定库视觉padding无mask的限制不能靠本检查器消除。

## 面试自测：五问与答题要点

| 层级／问题 | 参考答案与成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：一张384×384单行图怎样变成字符串，哪里结束？ | 16×16 patch得576块，加视觉特殊token／位置，经encoder；decoder从配置START按前缀cross-attend生成；新EOS结束，超预算为截断；最后反词元化 | 576块就是576字；把输入START=EOS判为已完成；整页直接当已校正单行 | L1形状与词元表；[cite:1](#source-1)[cite:2](#source-2)[cite:7](#source-7) |
| L2：RoBERTa已预训练，为什么还要cross-attention和因果mask？ | 文本模型没有视觉K/V通道；新增层需学习；右移＋因果mask限制只见已知前缀，避免teacher forcing泄漏 | 双向RoBERTa不改mask也能用；只要语言流畅就是看懂图；teacher forcing等于推理喂真值 | L2；[cite:6](#source-6) |
| L3：随机编号高分读成常见串，先加beam吗？ | 先检查原图、裁切、token协议、近似字与编号域数据；隔离新编号，比较greedy/beam与误放行；验证集校准拒识后独立验收 | beam更大必然准确；格式合法就放行；用测试集调门限并去掉拒识样本 | L3随机编号与验收；属于工程设计，非论文已实现拒识 |
| L4：推导PAD屏蔽CE梯度，并说明视觉encoder如何得到梯度？ | M>0且logits有限时dL/dz=m(p−onehot)/M；经词表头与cross的K、V两路回encoder；EOS不屏蔽，ID不求导 | 对argmax／beam求导；loss为零说明图像完全没梯度；p−onehot漏掉归一或mask | L4似然与局部链式；[cite:8](#source-8) |
| L5：训练loss正常但生成短串或batch串样本，如何验证？ | 核对是否重复右移、−100是否进embedding、START/BOS/EOS是否匹配；比较单张／batch、cache开关和beam重排；无EOS截断必须暴露 | 看shape相同就通过；类默认224等于模型384；有限NumPy梯度通过就是全模型复现 | L5源码／检查器与未测清单；[cite:5](#source-5)[cite:6](#source-6)[cite:8](#source-8) |

## 来源层级与逐源核验记录

来源编号仅在本卡内有效。P0为原论文／官方源码；记录页只能支持书目字段，正文部分只支持实读范围。v1、v5、会议版是同一研究的不同版本，官方库与其模型介绍有继承关系，**不计三份独立性能证据**。机制归属默认 `[SINGLE-SOURCE]`；版本差异 `[DISPUTED：协议不同，分别保留]`，不指某一版本必然错误。本卡不提供跨方法排名或独立复现实验成绩。

- <a id="source-1"></a>**[cite:1] P0，2021预印本。** [arXiv版本记录](https://arxiv.org/abs/2109.10282)、[固定v1 HTML](https://arxiv.org/html/2109.10282v1)、[固定v1 PDF](https://arxiv.org/pdf/2109.10282v1)。HTML读到§2.4，后续截断；PDF通过内存下载和pypdf解析10页，针对性读取§2–3的架构、EOS旋转、两阶段训练、增强、配置及最终BASE/LARGE组合，未逐项复核全部实验。v1的RoBERTa base 512宽度映射缺口保留。
- <a id="source-2"></a>**[cite:2] P0，AAAI 2023原文及书目。** [出版页](https://ojs.aaai.org/index.php/AAAI/article/view/26538)、[会议PDF入口](https://ojs.aaai.org/index.php/AAAI/article/download/26538/26310)，DOI [10.1609/aaai.v37i11.26538](https://doi.org/10.1609/aaai.v37i11.26538)。正文解析实际覆盖13094–13096的架构、初始化、BOS/EOS、训练／微调及后续表1片段；不是仅摘要，但也不是全文通读。补充直接下载返回403，view入口返回反机器人HTML，未尝试绕过，会议后半设置与实验未据此核验。
- <a id="source-3"></a>**[cite:3] P0，2022修订预印本，补充而不替代会议全文。** [v5 HTML](https://arxiv.org/html/2109.10282v5)、[v5 PDF](https://arxiv.org/pdf/2109.10282v5)。内存解析10页，重点读取Data Augmentation、Data、Settings、Architecture Comparison：SMALL/MiniLM、MJ/ST分支、beam=10、BPE/SentencePiece、模型组合与阶段规模。不能把v5日期写成2023，未进行会议全文逐字diff。
- <a id="source-4"></a>**[cite:4] P0，官方固定版本对象。** [v4.30.2标签引用API](https://api.github.com/repos/huggingface/transformers/git/ref/tags/v4.30.2)、[标签对象API](https://api.github.com/repos/huggingface/transformers/git/tags/da9daa4f5e0a0c73acab09b4906d59e88bc43be1)、[固定提交](https://github.com/huggingface/transformers/tree/66fd3a8d626a32989f4569260db32785c6cbf42a)。实读标签对象、日期、commit指向及unsigned字段；下述源码经GitHub Contents API以v4.30.2读取base64正文，不依赖漂移main。
- <a id="source-5"></a>**[cite:5] P0，官方ViT固定源码。** [modeling_vit.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/models/vit/modeling_vit.py#L120-L182)。实读ViTEmbeddings和ViTPatchEmbeddings相关行；blob `bfd440caae2b01a48d29a1a46e018dff236f0446`。只证明该ViT实现路径，不声称所有论文变体均使用它。
- <a id="source-6"></a>**[cite:6] P0，官方TrOCR decoder固定源码。** [modeling_trocr.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/models/trocr/modeling_trocr.py)。实读47–76、164–271、313–440、480–541、624–650及输出头／loss相关段；blob `7e2e11c00e22e4f5b63643a340e7055acfd7271b`。核验mask、Q/K/V、post-LN、cache与条件维度；未运行attention内核。
- <a id="source-7"></a>**[cite:7] P0，官方decoder默认配置。** [configuration_trocr.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/models/trocr/configuration_trocr.py#L102-L149)。实读构造器；blob `b3f03373618484971e8d10c144eedfaa434367a3`。默认值不等于具体权重文件；Microsoft托管模型元数据请求超时，未取得权重revision／processor／tokenizer快照。
- <a id="source-8"></a>**[cite:8] P0，官方视觉—文本包装固定源码。** [modeling_vision_encoder_decoder.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py)。实读37–50、216–221、579–629与prepare方法；blob `d3e464cbfffa0892f51ab4aa261c28335a16b9c4`。右移、−100替换、维度桥接、CE及encoder_attention_mask=None均为静态证据。
- <a id="source-9"></a>**[cite:9] P0，官方处理器固定源码。** [image_processing_vit.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/models/vit/image_processing_vit.py#L72-L94)。通过raw v4.30.2入口取得正文，读构造器、resize及261–270处理顺序；未把类默认mean/std常量当作已读取的模型处理器JSON。RGB／0.5参考协议明确是本卡选择。
- <a id="source-10"></a>**[cite:10] P0，官方beam固定源码。** [beam_search.py](https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/src/transformers/generation/beam_search.py#L861-L929)。实读参数语义、Scorer接口及BeamHypotheses的add／is_done；blob `d8bd8010dce3cc6a3c70bd38c7fe5817ffbfe2b4`。未逐行审计所有generation处理器，不将参考beam当库等价实现。

## 验证记录与 G31 销项依据

2026-11-05本轮从本文件提取两段Python围栏，以现有 `.venv/Scripts/python.exe` 在仓库根目录内存执行：**5组基线检查通过，删除valid限制的变异按预期失败**，失败原因为 `padding key weights must be zero`。被屏蔽key权重最大绝对值 `0.0`，其scores／V扰动后所有query输出最大绝对差 `0.0`；仅causal对照的可见pad key最大权重为 `0.36716540111092555`，但其有效右padding输出对照仍通过，明确说明后者不能证明key屏蔽。CE梯度最大绝对差 `1.7016266173897066e-10`，单头cross-attention Q/K/V梯度最大绝对差 `2.4381881930102267e-11`。环境为Python **3.13.14**、NumPy **2.5.4**；未安装PyTorch／Transformers，未新增脚本、依赖、图片或权重，未训练。这些值只是确定性夹具的核验，不是识别误差或官方attention内核运行结果。

文档静态检查通过：5段L1–L5正文、5组问答、单一TR01锚点、10个来源锚点、56处来源引用和62处本地文件／锚点链接；UTF-8、围栏配对及行尾空白检查通过。`git diff --check`通过但不能覆盖未跟踪文件，故另外直接检查两卡。本轮只编辑本文与 `retinex_1971.md`。Git枚举96个非目标文件，最终修改时间均早于本轮起点2026-11-05 14:08:26（Asia/Shanghai）；按排序后的UTF-8路径、NUL及内容SHA-256原始摘要聚合，最终SHA-256为 `42d931cc169451b430ce587665157abcf7c2f29c000fef62f687eec517ecb050`。初始内存摘要未跨命令保留，故不冒称全程前后哈希比对成功；修改时间与编辑操作记录支持仅两卡被本轮修改，不覆盖Git元数据或忽略目录，也不是防篡改证明。

G31原缺口是“TrOCR预训练／微调、词元生成与随机编号拒识”。本卡用原文和固定库承接**图像patch→视觉编码→文本因果解码／cross-attention→teacher forcing／CE→beam／EOS→拒识**完整机制，并提供L1–L5、答5、版本差异与可执行有限检查器，因此具备将 **08-06从缺卡转为已有具体卡** 的文档依据。仅补注意力C18或泛称Transformer不足以完成该项，本卡未以组件代替原模型链。

销项范围仅为**具体方法缺卡**：L1／L2已核验，L3／L4／L5部分，实现仍为**未实现**。正式会议后半全文比对、原作者训练recipe、固定权重／tokenizer／processor、模型级反传／cache测试、真实随机编号拒识和性能验收均保留为后续深度工作。本文不修改台账字段或汇总数量，也不宣称台账已经同步。
