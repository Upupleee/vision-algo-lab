# 边缘演进扩展卡：Devernay 1995、IPOL 2017 与 HED 2015

[学习规范](../algorithm_learning_standard.md) · [覆盖表](../evolution_coverage.md#05-边缘几何测量) · [测量演进](../measurement_evolution.md)

## 范围与核验口径

核验日期：2026-11-06。本文件仅新增三张独立卡：[DE95](#de95) 对应 G03／05-05，[DE17](#de17) 对应 G05／05-10，[HE01](#he01) 对应 G04／05-09 的 HED 子项。沿用 [M04](measurement_focus.md#m04-devernay-二维边缘精化) 的本地简化实现分析；U-Net 复用 [S21](segmentation.md#s21-u-net-与-edgenet-边界)，不再造一张 U-Net 卡。05-09 仍是一个复合节点，不因补 HED 增加节点数。

“已核验”只表示注明范围内的原文、静态源码或条件代数已核对；不代表训练、整图复现或现场验收。三卡均有 L1–L5 正文及逐层一问一答，共十五问。每问给出条件、易错点和证据入口。

| 卡片 | L1 | L2 | L3 | L4 | L5 | 实现与验证状态 |
|---|---|---|---|---|---|---|
| DE95 | 已核验：原文核心流程与挂接方式 | 已核验：法向峰和幅值插值 | 部分：无系统参数实验 | 已核验：三点代数及限定一维模型；非一般误差界 | 部分：本地差异静态核对，无1995程序 | 本地仅简化版；本次运行单项冒烟与边界检查 |
| DE17 | 已核验：论文算法1–3、5与源码全链 | 已核验：改方向、连链和滞后 | 部分：无完整程序实测 | 已核验：峰位、链评分及固定支持复杂度；非一般精度界 | 部分：v1.0核心函数静态核对，未编译 | 外部实现未接入；未执行附包回归 |
| HE01 | 已核验：2015架构、训练／推理分流 | 已核验：侧监督和logit融合 | 部分：无训练或域验证 | 已核验：限定二值损失、链式梯度；非优化收敛证明 | 部分：固定作者代码静态核对 | HED未实现；无Caffe前反向、权重或BSDS评测 |

来源等级：原报告、会议原文、作者源码为 P0；教学站镜像承载原报告，不是独立复现实验。以下算法事实标为 `[SINGLE-SOURCE]` 或“同一作者体系交叉核对”，不把论文、其镜像、作者代码算作三个独立来源。有限检查是本次实际运行的局部检查，不是论文实验重现。

## 版本与输出契约总表

| 对象 | NMS与亚像素位置 | 滞后／链构造 | 版本与许可边界 |
|---|---|---|---|
| DE95 | 梯度方向的邻域插值NMS；同方向三点幅值二次峰，保留整数像素与二维偏移 | 原报告§3：用原整数像素位置、幅值做通常滞后和像素链接；偏移另存供后续使用 | RR-2724，1995年11月；HAL入库2006年不是方法年份；未取得原程序及软件许可 |
| DE17 | 按绝对梯度分量选水平／垂直轴；NMS和精化均沿该轴；每点至少一个坐标保持整数 | 先按连续点位置和梯度方向构链，再沿链双向滞后；输出有序点及链分界 | v1.0，2017-10-10；文章2017-11-28；代码AGPL-3.0-or-later，文章另列CC-BY-NC-SA |
| 本地M04 | 八邻近似方向NMS；随后沿单位梯度±1 px双线性采样、裁偏移 | 只有high单阈值，无滞后、无链；输出无序N×2点 | 当前工作树函数，不是1995或2017逐行复现；不继承外部代码许可声明 |
| HE01 | 网络输出五张侧边缘图和融合概率图；评测另做NMS／细化，不输出dx/dy | HED本身不规定Canny双阈值滞后，也不生成DE17式链 | ICCV 2015；作者代码锁定2023提交，不能倒称2015实验快照；代码根许可为BSD-2-Clause条款，模型／数据另核 |

依据：[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)[cite:6](#source-6)[cite:13](#source-13)；本地依据 [edge_subpixel.py](../../traditional/subpixel/edge_subpixel.py#L57-L105)。

<a id="de95"></a>

## DE95：Devernay 原报告的亚像素 NMS

### 任务、演进与状态

输入为灰度图及选定的梯度／平滑算子，输出是与整数候选像素关联的连续边缘位置；通常再进入阈值、链接和几何处理。前序瓶颈是像素级定位误差及为求曲率而进行较强正则化；1995方法复用NMS已计算的幅值，增加廉价二次峰估计。代价是离散插值、导数算子和局部边缘模型带来的偏差；它不全面替代模型拟合，更不保证物理尺寸精度。[cite:1](#source-1)[cite:2](#source-2)

适用前提是非零、可稳定定向的局部梯度和孤立幅值峰；交叉点、窄双边缘、纹理、饱和和曲率均可能破坏前提。原报告§4.1实验使用Deriche四阶高斯递归梯度滤波，也讨论其他滤波器；不能把本地GaussianBlur＋3×3 Sobel写成原报告的固定配置。[cite:2](#source-2)

### L1：步骤与挂接位置

1. 按所选算子计算带符号梯度及其幅值，保留整数像素索引。
2. 沿梯度直线取邻域两侧比较位置；在网格外位置插值得到幅值。报告§2.1列线性插值，并比较二次插值版本。
3. 对局部最大值，用中心和两侧三个**幅值**拟合二次曲线，求同方向峰位；NMS和精化应共用明确的采样几何。
4. 按整数像素存储连续修正量。原报告§3明确：阈值滞后、像素链接仍可使用原像素位置及梯度幅值，不必因连续坐标重写整条处理链。
5. 完成有限像素扫描及阈值／链接遍历后，输出被保留像素对应的连续位置；无有效局部峰则输出空集。具体平局规则、阈值数值和链接器不是本卡擅自补出的1995源码细节。[cite:2](#source-2)

```text
gradient, magnitude = chosen_gradient_operator(image)
for pixel p in valid_support:
    d = gradient_line_step_to_neighborhood_boundary(p)
    a, b, c = interpolate(magnitude,p-d), magnitude[p], interpolate(magnitude,p+d)
    if selected_local_maximum(a,b,c):
        offset[p] = ((a-c)/(2*(a-2*b+c))) * d
        retain_integer_pixel_and_strength(p,b)
accepted = pixel_hysteresis_and_linking(integer_candidates,strength)
return integer_positions[accepted] + offset[accepted]
```

该伪代码是原文流程的解释性表达，不是本地新增函数；`selected_local_maximum` 的平局和边界必须在复现配置中显式定义。

### L2：两次插值解决不同问题

第一次插值用来获得网格外的方向幅值并判断NMS；第二次二次插值用来定位幅值峰。前者的“线性／二次”与后者的“二次峰”不能混为一谈。仅取最大像素会丢掉峰的左右不对称；仅对所有像素求峰而无NMS，会把非极大值、近线性段也解释为边缘。[cite:2](#source-2)

沿法向精化是因为梯度幅值跨边缘存在峰，而沿切向对直边近似不变，切向位置不可由这个一维模型单独辨识。整数像素与偏移分开存储，则连接拓扑不随浮点取整反复改变；连续位置用于后续几何，但不意味着所有拓扑决定都在连续空间进行。

幅值平方虽在同一离散样本集合上保持大小次序，却不保持二次插值的峰位；而“先平方再进行网格外线性插值”连比较值也可能不同。原报告明确警告平方幅值引入定位偏差，不能把去掉开方当作等价优化。[cite:2](#source-2)

### L3：参数、失败证据与调参顺序

| 项目 | 单位／证据 | 调整次序与代价 |
|---|---|---|
| 梯度与平滑 | 平滑尺度为px，幅值随灰度范围及导数增益变化 | 先固定成像、灰度尺度、导数核和边界，再选平滑；更平滑会混合邻近边缘 |
| NMS支持与方向插值 | 方向相关偏差、交替锯齿、候选重复 | 先对照方向与相位，不先降低阈值；变更采样位置不是纯加速 |
| 双阈值 | 梯度幅值单位；弱段是否与强边相连 | 用独立有边／无边样本标定，不直接复制本地high=30或其他库数值 |
| 峰曲率 | 分母接近零、偏移不稳定 | 检查局部三点峰和噪声；拒绝／回退是工程策略，不冒称原报告统一实现 |
| 几何测量 | 法向误差、完整率、重复率和最终拟合残差 | 先评估定位与拓扑，再拟合及标定；小拟合残差不排除一致偏置 |

本地默认`sigma_grad=1.0`、`high=30.0`只是现函数初值，缺跨场景最优依据。原报告的条件实验不能推出“任意模糊均0.05 px”。2017后续分析表明高斯边缘存在随相位、角度和模糊变化的误差，故不沿用原报告一维推论去承诺一般二维无偏。[cite:2](#source-2)[cite:4](#source-4)

### L4：三点峰、坐标步长与保证边界

令三点横坐标为−1、0、1，幅值为a、b、c，设\(q(t)=At^2+Bt+C\)。插值条件给出：

\[
A=\frac{a-2b+c}{2},\quad B=\frac{c-a}{2},\quad C=b,
\qquad \eta=-\frac{B}{2A}=\frac{a-c}{2(a-2b+c)}.
\]

需要负曲率\(a-2b+c<0\)。设\(u=b-a\ge0,v=b-c\ge0,u+v>0\)，则\(\eta=(u-v)/(2(u+v))\)，所以\(|\eta|\le1/2\)。平台a=b=c时峰不可辨识，不能用这个界掩盖零分母。对三个幅值共同施加\(ka+t\)、\(kb+t\)、\(kc+t\)，k>0时峰位不变；平方不是这种仿射变换。[cite:2](#source-2)[cite:4](#source-4)

坐标回填是\(\hat p=p+\eta d\)，**d是采样步向量，不默认是单位梯度**。例如\(|g_x|\ge|g_y|\)、\(g_x\ne0\)时，可按正x方向写\(d=(1,g_y/g_x)\)，两侧落在相邻列上；\(\eta\)是相对此步长的参数，不是欧氏距离。若单位法向为n，\(d=n/\|n\|_\infty\)只是在选定正负方向后的等价参数化；同时换向会交换a、c，最终位移不变。将原来的d直接替换成n却不重采样幅值，会改变算法。[cite:2](#source-2)[cite:4](#source-4)

原报告一维例：像素积分阶跃在中心像素的覆盖比例为\(\alpha\in[0,1]\)，导数样本在报告的符号／幅值条件下归约为\((a,b,c)=(1-\alpha,1,\alpha)\)，代入得\(\eta=\alpha-1/2\)，与该例真实位置相同。反对称导数、积分像素、局部符号及归约成立是前提；只凭“反对称”不能忽略绝对值的符号变化。报告称可把对称模糊并入导数算子；2017又对高斯剖面、有限差分、离网格插值及数值误差分别分析。两者模型和采样假设需区分，不把前者扩成任意连续PSF采样图像的零误差定理。[cite:2](#source-2)[cite:4](#source-4)

固定支持导数、NMS和峰估计为O(HW)，候选附加成本O(N)；其他链接器成本需按实际算法算。未提供可用于任意真实图像的通用定位误差上界，也不保证毫米精度。

### L5：本地简化差异与复现边界

| 步骤 | 原文／原型 | 本地`devernay_edges`实际行为 |
|---|---|---|
| 梯度 | 可选梯度算子；报告实验为Deriche递归滤波 | float64，GaussianBlur后3×3 Sobel；阈值尺度不等同 |
| NMS | 沿梯度方向的插值幅值比较 | 遍历八邻，`abs(ux*sx+uy*sy)>0.5`选比较邻居，再用严格大于抑制 |
| 方向角 | 由采样几何决定 | 对角邻居未除以长度√2；注释“±45°”不是统一角门 |
| 精化 | 共用NMS方向样本；偏移另存 | NMS后重新沿单位梯度±1采样，`map_coordinates`双线性，和候选判定不是同组三点 |
| 阈值／连接 | 原整数像素上可继续通常滞后和链接 | NMS前`mag>=high`；无low、无强弱可达、无边链 |
| 边界／稳定性 | 原程序细节未取得 | 邻居clip、采样nearest；不排除边带；`np.where`除法仍先求值，随后裁偏移 |
| 输出 | 整数索引与连续偏移可关联保留 | 仅N×2 `[x,y]`，不能从扫描顺序当作轮廓拓扑 |

来源为本地 [函数体](../../traditional/subpixel/edge_subpixel.py#L57-L105)，不是用注释代替源码。其`np.where`、平台和平局问题在[有限检查](#validation)中单列；本次不修代码。1995源码未取得，许可未知；2017 AGPL不能反向当作1995软件许可。

### 五级追问与答题要点

| 层级 | 问题 | 答案、成立条件与易错点 | 证据 |
|---|---|---|---|
| L1 | 连续点出现后，原报告如何继续滞后和连接？ | 保存整数候选／幅值及另存偏移，在像素域继续通常处理；不是丢弃整数索引后重新四舍五入，也不是2017连续距离链 | §3，[cite:2](#source-2) |
| L2 | 幅值平方最大值相同，为何不能去开方？ | 同点排序可相同，但平方改变三点曲率、峰位；离网格插值还不与平方交换 | §2.2，[cite:2](#source-2) |
| L3 | 斜边出现交替偏移，先降high吗？ | 先检查角度／相位、离网格插值及梯度核；降阈值只改候选，不能消除定位模型偏差 | DE95 L3、[cite:4](#source-4) |
| L4 | 半像素界怎么来，二维如何回填？ | 用u、v非负推导界；平台除外；回填p+ηd，d是实际采样步而非任意单位法向 | DE95 L4、[cite:2](#source-2) |
| L5 | 本地有浮点点集，算原报告完整复现吗？ | 不算；NMS、取样、阈值和链接均有差异，单斜边冒烟只支持该输入，不能证明拓扑或原版等价 | DE95差异表、[有限检查](#validation) |

<a id="de17"></a>

## DE17：IPOL 2017 修改版 Canny/Devernay 与边链

### 任务、演进与固定版本

输入是行主序double灰度图`image[x+y*X]`，参数为高斯尺度sigma与高低幅值阈值H、L；输出为`x[N]`、`y[N]`、`curve_limits[M+1]`和N、M。链k使用半开区间`[curve_limits[k],curve_limits[k+1])`；闭链末点重复首点，TXT接口用`-1 -1`分链，不是API坐标数组中的边缘点。[cite:5](#source-5)

1995的离网格幅值插值可产生方向／相位相关振荡；2017把NMS和精化共同改为坐标轴方向，去掉这一误差来源，并给出有序边链实现。新增代价是轴向选择的不连续性、局部连链歧义及拓扑约束；并未消除抛物线模型和有限差分误差。[cite:4](#source-4)

锁定`devernay_1.0.zip`：版本**1.0（2017-10-10）**，本次读取字节的SHA-256为`4e47141e7edbebdfe89f3d94c3d90df7954929aff27783dfca0a1b60ed1f9301`。官方页还给出SWH目录标识`60b02dbe634dd2f67fb851d363de3bc4096d508b`。文章发表2017-11-28，2022-01-01仅注明编辑信息修订，不能据此称软件v2022。[cite:3](#source-3)[cite:5](#source-5)

### L1：实际调用顺序与停止条件

```text
if sigma > 0: image = separable_gaussian(image,sigma)
Gx,Gy,mag = centered_difference_without_half_factor(image)
Ex,Ey = axis_NMS_and_axis_parabolic_refinement(mag,Gx,Gy)
next,prev = local_chaining(Ex,Ey,Gx,Gy)
next,prev = hysteresis_on_chains(next,prev,mag,H,L)
x,y,curve_limits = enumerate_and_unlink_chains(next,prev,Ex,Ey)
```

这是固定源码`devernay()`的调用序，不是“先普通Canny二值图，再添亚像素”。`compute_edge_points()`不使用强度阈值；在已精化点上先构链，之后才执行滞后。[cite:5](#source-5)

NMS优先检验水平条件：`greater(b,Lx) && !greater(Rx,b) && abs(Gx)>=abs(Gy)`，通过后取d=(1,0)；否则检验竖直条件并取d=(0,1)。按`greater`的容差语义，平局保留规则为左严格、右非严格／较小y严格、较大y非严格；这不等价于实数严格比较，右侧略高也可被视为平局。梯度分量相等时若两个方向均满足，水平分支先选。两像素外边带不参与候选。[cite:4](#source-4)[cite:5](#source-5)

构链对每个候选检查索引偏移−2…2的5×5邻域，区分前向／后向并优先近邻；处理已有连接冲突，保持每点最多一个前驱、一个后继。滞后从**已有链关系**且幅值≥H的点开始双向访问，只有幅值≥L才能继续；遇到低于L的点断链，访问过则停止，最终删除无效链点。完全孤立的高响应点因无`next/prev`不进入输出。枚举链在端点停止，闭链重复首点后结束并拆掉已遍历关系，避免再次输出。[cite:5](#source-5)

### L2：连接不是任意八连通可达

轴向NMS与精化共用同轴三个幅值，消除先用离网格方向判峰、再沿另一支持插值的混配；但源码容差接受不保证中心在真实大小关系下满足b≥a、b≥c。2017§5正文用严格大小作解释，算法3处理平局，源码还通过`greater`引入绝对容差；复现须分别记录这些语义，不能由“通过NMS”直接推出半像素界。[cite:4](#source-4)[cite:5](#source-5)

链评分使用连续点之间的距离，因此连续偏移发生在链之前。方向兼容性只是连接向量相对于两端梯度具有一致切向符号，并非源码额外要求“夹角小于某固定角”。5×5也只是整数候选索引窗口，不是“欧氏半径≤2 px”。连续距离只负责评分。[cite:5](#source-5)

滞后沿已选的单前驱／单后继关系传播，和在所有弱边像素的八连通图上做广度搜索不同；分叉不能全部保留。局部替换连接还可能依赖探索顺序，论文§6及源码都给出反例，故不能声称全局最短、扫描顺序无关或所有真实轮廓均完整。[cite:4](#source-4)[cite:5](#source-5)

### L3：参数与故障证据

sigma单位为px；v1.0允许sigma=0时不平滑。离散高斯截断半径为`ceil(sigma*sqrt(6*ln(10)))`，采用可分离FIR和对称边界。梯度为`I[x+1]-I[x-1]`、`I[y+1]-I[y-1]`，未除2；H、L属于这个导数尺度，不能直接复制Sobel或OpenCV Canny阈值。[cite:5](#source-5)

工程上先校验有限灰度、合法尺寸、`0<=L<=H`及sigma，再定平滑／噪声尺度，再调阈值。此处校验是调用方建议，不能当作源码已经完整实现；源码部分非法输入通过`exit()`退出，并非可恢复异常。固定绝对比较容差`1000*DBL_EPSILON`不是按图像动态范围缩放的相对容差，幅值极小也可能改变候选选择。

断链需分别观察候选密度、兼容方向、5×5搜索限制、冲突替换和L截断；只降H可能加入噪声链。窄条／交叉／角点须记录漏段、错连、重复点、方向及闭环情况。大型sigma增加卷积成本，扩大邻域或并行重排扫描会改变拓扑，不能不经回归就称等价优化。

### L4：轴向峰、链评分及复杂度

峰公式与DE95相同，但\(d\in\{(1,0),(0,1)\}\)。\(\hat p=p+\eta d\)意味着每点最多只改一个坐标；“亚像素”不要求x、y同时有小数。**半像素界仅在真实大小关系b≥a、b≥c且非平台时成立**：令u=b−a≥0、v=b−c≥0、u+v>0，才有\(\eta=(u-v)/(2(u+v))\)及\(|\eta|\le1/2\)。这是条件代数界，不是源码每个候选的无条件保证。[cite:4](#source-4)[cite:5](#source-5)

源码`greater(x,y)`在x≤y或x−y<1000ε时返回假，ε=`DBL_EPSILON`。取\((a,b,c)=(1-2000\varepsilon,1,1+500\varepsilon)\)，并令轴选择满足\(|G_x|\ge|G_y|\)，则`greater(b,a)`为真、`!greater(c,b)`也为真，水平候选被接受；但c>b，v=−500ε，已经不满足半像素界前提。按源码次序`0.5*(a-c)/(a-b-b+c)`得到\(\eta=5/6\approx0.8333333333333334\)，即使曲率仍为负也超过1/2。源码没有末端裁剪；其范围注释不能替代该前提。下方保留可运行反例，不默改比较函数、峰公式或添加裁剪；本次未编译原C程序，检查不代表整图可达性或原版回归通过。

对于连续点p、q，令\(\Delta=q-p\)、\(s_p=g_y(p)\Delta_x-g_x(p)\Delta_y\)、\(s_q=g_y(q)\Delta_x-g_x(q)\Delta_y\)。源码的兼容条件与评分为：

\[
s_ps_q>0,\qquad \operatorname{score}(p,q)=\frac{\operatorname{sign}(s_p)}{\|q-p\|_2}.
\]

非候选、同一像素或不兼容时记0。最大正分为最近前向，最小负分为最近后向；建立连接时还比较目标已有连接的分数并拆除冲突。这不是最小化全图总长度的全局优化。完全重合的连续点需关注零距离及符号乘积退化；源码先做兼容判断，不应删掉这些分支只保留倒距离。[cite:5](#source-5)

固定链图上，滞后相当于保留包含强点的≥L链段，单个有效节点不会无限重复传播。固定高斯支持和5×5窗口时，各扫描与链遍历均为O(HW)，存储O(HW)；显式考虑FIR半径r时平滑为O(HWr)。链算法没有迭代收敛证明需求，但有有限遍历终止与链接不变量要求。2017四类误差为抛物线假设、有限差分、离网格幅值插值、数值误差；修改仅去掉第三类，未提供通用真实场景精度界。[cite:4](#source-4)

### L5：固定源码映射、许可及验证设计

| 源码位置（v1.0） | 已核对内容 | 尚未验证 |
|---|---|---|
| `devernay.c:73–79,125–203,268–285` | 绝对比较容差、高斯边界、无1/2系数中心差分 | 大sigma、极小图、溢出和不同浮点平台 |
| `devernay.c:344–420` | 两像素边带、轴向平局、峰位回填 | 角度／相位系统误差和候选完整率 |
| `devernay.c:218–262,434–555` | 方向符号、连续倒距离、冲突拆链和扫描顺序 | 交叉／近邻链及顺序敏感的整图回归 |
| `devernay.c:567–618` | 先链后滞后、H/L边界包含、低段截断、孤点不留 | 强弱段与闭环的原C程序运行 |
| `devernay.c:638–720,761–798`及`devernay.h` | 分界数组、闭环重复首点、输出分配、调用序 | 编译ABI、内存释放、包内`make test` |

源码包内`devernay.c/h`被README标为同行评审核心；这不是所有IO与封装都经过同等核验的证据。本次读取核心与README／头文件，未逐行审计整个包。[cite:5](#source-5)

代码头、README和官方元数据一致为**AGPL-3.0-or-later**；文章PDF首页标**CC-BY-NC-SA**，不要混成一套许可。AGPL允许使用／修改，但分发及经网络提供修改程序等情形有相应源码义务，不能写“免费即可闭源商用”；部署前按实际组合方式审查，本文不作法律结论。未复制代码进入本仓库。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

后续应先编译并比对附包`image.pgm`与`image_out.txt`，再测试空图、平台、精确H/L、弱链无强种子、低于L的断点、孤立强点、闭环、交叉与反极性；比较坐标、链分界和首尾重复，不能只比较点数。本地M04无链且核／NMS／采样全部不同，单项冒烟不转移为DE17验证通过。

### 五级追问与答题要点

| 层级 | 问题 | 答案、成立条件与易错点 | 证据 |
|---|---|---|---|
| L1 | 2017源码先滞后还是先构链？ | 梯度→轴向NMS及精化→链→链上滞后→输出；不是普通二值Canny后补小数 | `devernay()`，[cite:5](#source-5) |
| L2 | 5×5链等于八连通或2 px圆邻域吗？ | 均不是；按整数窗口挑候选，再用连续距离和两端切向符号，最多一前一后 | `chain`／`chain_edge_points`，[cite:5](#source-5) |
| L3 | 从本地Sobel迁移为何不能照搬H/L？ | 中心差分增益、平滑和动态范围不同；先对幅值／噪声尺度，再验链断点 | DE17 L3 |
| L4 | 只修x或y为何仍可称亚像素？ | 沿选定轴拟合连续峰；半像素界要求真实b≥a、b≥c且非平台，不能仅凭容差NMS通过推出；反例η=5/6，亦非二维各向同性精度保证 | DE17 L4、[cite:4](#source-4)[cite:5](#source-5) |
| L5 | 并行构链是无损加速吗，能直接闭源分发吗？ | 构链顺序影响冲突解，先验等价性；代码AGPL许可与文章许可分开审查，不能因开放下载就认定无限制 | 源码冲突说明及许可，[cite:5](#source-5) |

<a id="he01"></a>

## HE01：HED 的侧输出、深监督与融合训练

### 任务、演进与状态

HED指Xie、Tu的*Holistically-Nested Edge Detection*，ICCV 2015，1395–1403。输入自然图像，输出多尺度／多层级的稠密边缘响应；训练需要图像与边缘标签，不是物体内部区域标签或亚像素偏移真值。前序瓶颈是逐patch CNN计算冗余、手工特征限制以及各层输出缺乏直接监督；HED用共享全卷积主干和多侧监督，把逐图训练、多层表示与融合联合起来。新增代价是监督数据、GPU训练和域偏移；不替代局部物理边缘模型。[cite:6](#source-6)

原论文的五侧头分别接`conv1_2,conv2_2,conv3_3,conv4_3,conv5_3`，步长1、2、4、8、16，感受野5、14、40、92、196；去掉pool5及全连接层，保留conv5组。这里是裁剪VGG-16，不是五个独立VGG，也不是U-Net的对称编解码器。[cite:6](#source-6)

### L1：训练与推理分开

训练流程：

1. 固定训练／验证／测试划分与边缘标注版本；按原论文BSDS方案，用至少三位标注者一致的像素为正，其余为负，并做旋转／翻转增强。
2. 加载预训练VGG-16对应主干；每个阶段末尾以1×1分类头生成单通道侧logit。
3. 将侧logit经固定双线性上采样并裁剪对齐原图。五个侧输出都与同一目标边缘图建立独立监督。
4. 拼接对齐的五张logit图，学习1×1加权融合，另加融合监督；总目标为五侧损失之和加融合损失，联合反传到共享主干及各头。
5. 按选定版本的SGD配置和验证方案训练到预定迭代预算，保存检查点；有限步SGD不保证全局收敛。[cite:6](#source-6)[cite:8](#source-8)

推理时不需要标签，一次前向生成五侧概率及融合概率。原文区分融合输出、侧概率平均及late merging；式(6)给出对融合与侧输出的后续聚合，表3分别报告不同组合，不能把所有组合统称一张相同结果。作者后续README推荐的fusion-output也不是强制复现原文late merging。评测前执行NMS／细化，按选定阈值取得边图；这不是网络内置的Canny双阈值或亚像素点链。[cite:6](#source-6)[cite:7](#source-7)

### L2：为何不是 U-Net，去掉侧监督会怎样

侧监督给中浅层直接的边缘分类梯度，融合监督则协调多层证据；只有融合损失时，侧输出可以成为服务融合的任意中间特征，不必各自形成可解释的边缘图。深监督不是“可视化五层特征”，也不是每层各自独立训练再平均。论文§3.2给出有／无深监督的对照，但其结果限于实验设置，不保证所有数据都受益。[cite:6](#source-6)

HED以共享主干的五个输出分支和独立损失实现多尺度，而U-Net通过解码路径、同尺度特征拼接恢复稠密定位。HED侧输出的上采样不等于U-Net完整解码器。当前 [EdgeNet](../../dl/subpixel_edge_net.py#L22-L60)只有两级池化、两级转置卷积和相加skip，末端输出heat、dx、dy三通道；没有五侧头、五侧监督和可学习融合，所以它既不复现HED，也不是原始U-Net。

融合应发生在logit上，再sigmoid；“概率平均”和“logit加权再sigmoid”一般不同。边缘概率没有经过标定时也不应视为真实后验置信。NMS后的薄边图仍是像素栅格表达，不直接产生物理亚像素定位。[cite:6](#source-6)[cite:8](#source-8)

### L3：论文与固定作者配置差异

固定作者仓库为`s9xie/hed`，提交**`912632b986acc6dd6cc33b95603b2f279d7bd9f2`**，提交日期2023-03-27。锁提交只固定代码；外部下载模型和数据仍需各自哈希。README明确说已更新Caffe、训练接口及预训练模型，故它不是2015快照。[cite:7](#source-7)

| 项目 | ICCV 2015原文 | 固定提交实际配置 |
|---|---|---|
| 数据形状／批次 | 训练400×400，mini-batch 10；测试原尺寸 | `new_height/new_width=0`，batch_size=1，`iter_size=10`梯度累积；不自动等同固定尺寸10图batch |
| 基础优化 | lr=1e-6，momentum=0.9，weight decay=0.0002 | solver保持这些值，但网络有各层`lr_mult`，conv5组权重100、偏置200 |
| 迭代／降率 | 10,000次，5,000次后lr除10 | solver写stepsize=10,000、max_iter=30,001；`solve.py`却调用`solver.step(100000)`，不能只按max_iter报告实际停止 |
| 侧头／融合 | 侧损失权重1；融合权重初始化1/5 | 五侧损失及融合损失均1；融合卷积权重初值0.2，卷积还可含偏置，不能假设softmax非负归一约束 |
| 上采样 | 固定双线性插值 | `solve.py`先做bilinear surgery，四个deconv层lr_mult=0；单独载prototxt不等于正确初始化完整训练链 |
| 标注共识 | 至少三人标注为正，其余负 | `cv_gt/255`借OpenCV整数舍入编码；代码说明针对五人聚合图，依赖资产编码，不能套任意0–255标签 |
| 在线验证 | 原文以验证集调超参数 | `test_iter=0`，TEST数据源仍为train_pair，注释说明只搭图；不是有效留出验证 |

依据：[cite:6](#source-6)[cite:8](#source-8)[cite:10](#source-10)[cite:11](#source-11)。表中冲突保留为复现配置风险，不擅自替作者选一套“正确历史配置”。

调参先看标签编码／共识与对齐，再看每个侧输出的正负分布、梯度范数及loss尺度，最后才改学习率、各头权重、阈值或模型容量。高层宽边不应仅靠提高阈值掩盖；全负训练图可能使原平衡损失失去有效梯度。迁移到工业材质要独立验证弱划痕、纹理误报、反光和跨相机；ODS/OIS/AP不等于边缘法向px误差。

### L4：目标函数与多路径梯度

设二值标签\(y_j\in\{0,1\}\)，正负集合分别为\(Y_+,Y_-\)，\(\beta=|Y_-|/|Y|\)。第m个对齐侧logit为\(a_j^{(m)}\)，\(p_j^{(m)}=\sigma(a_j^{(m)})\)。

\[
\ell_m=-\beta\sum_{j\in Y_+}\log p_j^{(m)}-(1-\beta)\sum_{j\in Y_-}\log(1-p_j^{(m)}),
\quad \mathcal L_{side}=\sum_{m=1}^{5}\alpha_m\ell_m.
\]

原文式(2)之后一处正负集合的文字描述与式子语义相反；这里按二值标签、式子及源码计数统一为Y+边缘、Y−非边缘，不传播排版／文字歧义。[cite:6](#source-6)[cite:9](#source-9)

原文融合为\(z_j=\sum_m h_m a_j^{(m)}\)、\(p_j^f=\sigma(z_j)\)，总目标\(\mathcal L=\mathcal L_{side}+\mathcal L_{fuse}\)，共同优化W、侧头w和融合h。原文对融合Dist写cross-entropy，未在式(3)重新展开平衡权重；固定代码明确给融合调用同一个**修改过的平衡SigmoidCrossEntropyLoss**，不能把这项实现细化当作原文已逐字规定。[cite:6](#source-6)[cite:8](#source-8)[cite:9](#source-9)

固定二值标签时设\(\omega_j=\beta\)（正类）或\(1-\beta\)（负类），则：

\[
\frac{\partial\ell_m}{\partial a_j^{(m)}}=\omega_j(p_j^{(m)}-y_j),\qquad
\frac{\partial\mathcal L}{\partial a_j^{(m)}}=\alpha_m\omega_j(p_j^{(m)}-y_j)+h_m\omega_j^f(p_j^f-y_j).
\]

\(\omega^f\)应与选择的融合损失一致。通过上采样／裁剪的Jacobian回传，再对共享主干汇总所有侧分支和融合路径；不能把各头对W的梯度彼此覆盖。融合梯度为\(\partial\mathcal L_f/\partial h_m=\sum_j\omega_j^f(p_j^f-y_j)a_j^{(m)}\)。侧logit是h和W的共同训练输入，不应在融合前无意detach。

数值实现宜用logit形式\(\omega_j[\operatorname{softplus}(a_j)-y_ja_j]\)。固定源码按图内正负数定权，像素损失求和后除batch数，不除像素数；不同图尺寸会改变梯度规模。只有一个类别时，实际出现类别的权重为零；若输入没有任何严格0／1标签，计数分母为零。忽略标签需显式设计，不能把非0／1直接传入当作ignore：该CPU前向跳过它们，反向却先算sigmoid−target且未将其清零。[cite:9](#source-9)

条件代数不证明深网非凸优化到全局最优，也没有通用泛化或定位误差界。卷积主要成本可写为\(O(\sum_l H_lW_lk_l^2C_{in,l}C_{out,l})\)，侧输出／融合及损失为O(5HW)，训练还需保留激活；这些量级不是本机耗时测量。

### L5：源码、坐标、资产与本地差异

| 证据入口 | 必须保持的契约 | 边界 |
|---|---|---|
| `examples/hed/train_val.prototxt` | 五侧头、四个上采样、五次crop、logit concat及融合；六个loss | 输入按Caffe通道／均值协议处理，不能直接拿灰度EdgeNet输入替换 |
| `src/caffe/layers/crop_layer.cpp` | 追溯共同祖先的坐标映射，要求同尺度及整数偏移，再crop至data尺寸 | 不是只凭尺寸差做center crop；首卷积pad=35和奇数尺寸需整链测试 |
| `examples/hed/solve.py`及solver | bilinear初始化、加载5stage-vgg、GPU及步数配置 | Python 2风格脚本，现代环境不能据静态阅读断言可运行 |
| `sigmoid_cross_entropy_loss_layer.cpp` | 平衡权重、按图计数、batch归一、稳定logit损失 | 已核CPU实现；未取得同名`.cu`：固定路径HTTP 404、该提交目录无此文件。GPU调用链未核验，不推断GPU等价或不计算 |
| `image_labelmap_data_layer.cpp` | 图像和标签尺寸、同步几何变换、共识编码 | 未下载真实标签资产；resize分支是线性插值，标签量化行为须实测 |

依据：[cite:8](#source-8)[cite:9](#source-9)[cite:10](#source-10)[cite:11](#source-11)[cite:12](#source-12)。本地 [EdgeNet](../../dl/subpixel_edge_net.py#L22-L60)只有一个末端三通道头，损失是heat BCE加边缘位置L1意图，无训练循环和HED权重。`pred[:,1:]`为B×2×H×W，`m=heat>0.5`为B×H×W，非空布尔索引存在布局不匹配；旧注释“两通道”也不符`out=3`。这些为静态事实，本次未运行Torch、未修此骨架。

固定提交根LICENSE是两条再分发保留声明及免责声明的BSD条款，可概括为BSD-2-Clause；不是“无版权”。论文开放可读不等于同一代码许可，模型下载、VGG初始权重、BSDS数据和外部NMS工具均要分别核许可与哈希，当前未完成这些资产的法律／血缘核验。[cite:13](#source-13)

下一阶段需测试五侧形状、奇偶／非方图对齐、loss数值与梯度、全正／全负／非法标签、不同图尺寸和梯度累积归一，再执行固定权重前向、训练和独立留出评测。若要尺寸测量，还需另接图像边缘模型、几何拟合和标定，不直接将HED概率当偏移。

### 五级追问与答题要点

| 层级 | 问题 | 答案、成立条件与易错点 | 证据 |
|---|---|---|---|
| L1 | 训练及测试各输出什么，何时用标签？ | 训练五侧加融合共六损失；推理一次前向出五侧及融合概率，不需标签；NMS是评测后处理 | §2.2、§4，[cite:6](#source-6) |
| L2 | 去掉五侧loss但保留融合，还是原HED训练吗？ | 不是；只有融合监督不能约束每个侧图独立预测边缘，梯度路径与目标均变；有上采样也不因此成为U-Net | §3.2，[cite:6](#source-6) |
| L3 | 全负图loss很小表示学会背景吗？ | 未必；原按图平衡使唯一类别权重为零，先检查类别计数／标签；改损失需记录为变体 | CPU loss，[cite:9](#source-9) |
| L4 | 平均概率等于融合logit再sigmoid吗？ | 一般不等；给出非线性反例并推导侧梯度的自身项及融合项；没有softmax约束不推定h非负或和为1 | HE01 L4、[有限检查](#validation) |
| L5 | 本地heat+dx/dy能否冒充HED？ | 不能：头数、目标、skip和输出语义不同；先验侧图／crop／标签及权重，且固定2023源码非2015实验快照 | 本地源码、[cite:7](#source-7)[cite:8](#source-8) |

<a id="validation"></a>

## 本次有限检查与未完成验证

仅使用现有`.venv\Scripts\python.exe -B`，Python 3.13.14、NumPy 2.5.4；未安装依赖，未生成测试文件、下载资产文件或修改源码。网络PDF和ZIP在内存中读取；以下局部代数输入只是明确的检查夹具，不作为研究数据或算法精度统计。

| 检查 | 本次结果 | 能支持／不能支持 |
|---|---|---|
| 峰位六组检查 | 通过：三点(6,9,8)得0.25；仿射幅值变换保持；平方不保持；五个α的一维例；真实非平台峰的半像素界示例；轴向与单位法向回填不同 | 仅代数和所列输入，不是角度／模糊图像扫描 |
| DE17容差反例 | `greater(b,a)=True`、`greater(c,b)=False`、c>b；η=`0.8333333333333334`>0.5 | 复写固定源码的标量比较及计算次序；不是C程序或整图运行，源码未改 |
| 平衡BCE梯度 | 本轮明示四logit输入的中心差分，步长1e-6，最大绝对差`8.878643653620344e-11` | 条件二值标量目标正确；未调用Caffe loss或网络，不沿用旧检查的误差数值 |
| logit融合／单类权重 | 非线性反例通过；单一类别权重为零检查通过 | 不证明完整HED训练／CUDA梯度 |
| 现有`test_devernay_edges()` | 通过：157点，中位法向误差0.0129 px | 仓库生成的单个斜边夹具；不是DE95、DE17或真实尺寸测量精度 |
| 本地常量8×9、high=30 | 返回(0,2)空点集 | 仅当前函数正常阈值空图契约 |
| 本地常量8×9、high=0 | 返回72点，并出现`invalid value encountered in divide`警告 | 暴露平台／无严格峰及`np.where`预求值问题；不是“全部边界测试通过” |
| IPOL代码包 | 读取v1.0、记录SHA-256、核对核心函数与许可 | 未编译、未跑包内样例、未比较闭链／弱段 |
| HED作者代码 | 固定提交的图／CPU损失／训练入口／crop静态核对 | 未运行网络、未加载权重、未训练、未验证GPU实现 |

本轮实际复跑通过：峰位六组、DE17容差反例、二值BCE差分、融合／单类权重及现有单项冒烟；两个常量输入分别返回(0,2)、(72,2)，high=0的除法警告如实保留。结构检查通过：三卡锚点、15段L1–L5正文、15组问答、13个来源ID、99处本地链接（含锚点／行范围）及代码围栏／公式配对；这类检查不升级算法实现状态。

**2026-11-06本轮链接复核：**18个不同外部URL逐一请求，17个HTTP 200，唯一404为[cite:9](#source-9)明示保留的缺失`.cu`路径；固定目录API确认无此文件。IPOL ZIP在内存重取后的SHA-256与DE17记录一致，重新核对`greater`与峰位计算分支。HTTP 200只证明本轮请求可达，不表示重新逐页读完论文、验证模型或运行GPU；来源中的既有正文阅读记录与本轮状态检查分开。未安装依赖或写入下载文件，本轮文档写入范围仅本文件及`lbp_hog_extensions.md`。

本轮复跑使用以下唯一Python检查块，从文档提取并在隔离字典中内存执行；它明确给出峰位、BCE及融合检查输入，并调用现有单项冒烟，不启动Caffe或CUDA。

```python
import math
import runpy
import sys
import warnings
import numpy as np
from traditional.subpixel.edge_subpixel import devernay_edges

def peak(a, b, c):
    return 0.5*(a-c)/(a-b-b+c)

assert peak(6,9,8) == 0.25
assert peak(3*6+7,3*9+7,3*8+7) == peak(6,9,8)
assert not math.isclose(peak(6**2,9**2,8**2), peak(6,9,8))
for alpha in (0,0.25,0.5,0.75,1):
    assert math.isclose(peak(1-alpha,1,alpha), alpha-0.5, abs_tol=1e-15)
assert all(abs(peak(a,b,c)) <= 0.5 for a,b,c in ((0,1,1),(1,1,0),(6,9,8),(0,1,0)))
assert not np.allclose(0.25*np.array([1,0]), 0.25*np.array([0.8,0.6]))
print('PASS: six peak groups')

eps = sys.float_info.epsilon
def greater(x, y):
    if x <= y:
        return False
    if x-y < 1000*eps:
        return False
    return True

a,b,c = 1-2000*eps,1.0,1+500*eps
gx,gy = 1.0,0.0
assert greater(b,a) and not greater(c,b) and abs(gx) >= abs(gy)
assert c > b and a-b-b+c < 0
eta = peak(a,b,c)
assert math.isclose(eta,5/6) and eta > 0.5
print(f'PASS: DE17 tolerance counterexample eta={eta!r}')

logits = np.array([-2.0,-0.3,0.7,2.0])
labels = np.array([0.0,1.0,0.0,1.0])
beta = np.mean(labels == 0)
weights = np.where(labels == 1,beta,1-beta)
def loss(z):
    return np.sum(weights*(np.logaddexp(0,z)-labels*z))

analytic = weights*(1/(1+np.exp(-logits))-labels)
numeric = []
for i in range(4):
    delta = np.zeros(4)
    delta[i] = 1e-6
    numeric.append((loss(logits+delta)-loss(logits-delta))/2e-6)
error = float(np.max(np.abs(analytic-numeric)))
assert error < 1e-8
print(f'PASS: balanced BCE max_abs_error={error!r}')
sigmoid = lambda x: 1/(1+math.exp(-x))
assert not math.isclose((sigmoid(0)+sigmoid(2))/2, sigmoid(1))
for y in (np.zeros(4),np.ones(4)):
    beta = np.mean(y == 0)
    assert np.all(np.where(y == 1,beta,1-beta) == 0)
print('PASS: logit fusion and single-class weights')
print(runpy.run_path('tests/test_smoke.py')['test_devernay_edges']())
for high,expected in ((30,(0,2)),(0,(72,2))):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        points = devernay_edges(np.zeros((8,9)), high=high)
    messages = [str(w.message) for w in caught]
    assert points.shape == expected
    assert any('invalid value encountered in divide' in m for m in messages) == (high == 0)
    print(f'constant high={high}: shape={points.shape}, warnings={messages}')
```

## 覆盖销项依据与保留事项

| 缺口 | 建议登记 | 本卡已补齐的具体内容 | 必须继续保留 |
|---|---|---|---|
| G03／05-05 | 建议由“部分／P1缺机制”转“已有／P2深度验证” | DE95原文NMS、三点峰、整数像素挂接滞后／链接、幅值平方警告及本地逐项差异 | 原1995程序／许可、全实验条件与二维系统误差未复现；本地仍简化版，L3/L5部分 |
| G04／05-09 | 建议HED子项转已有，仍与S21共同承接一个复合节点 | HE01五侧输出、六损失联合训练、logit融合、推理聚合、非U-Net及本地差异 | HED及原U-Net未实现；历史训练资产／参数、GPU、标签共识、评测协议与域验证未闭合；不得提升S21既有层级 |
| G05／05-10 | 建议由“部分／P1缺机制”转“已有／P2深度验证” | DE17轴向离散修改、先构链后滞后、连续距离评分、输出拓扑、固定版本／许可 | 未编译或端到端运行；链顺序依赖、弱段／闭环／边界和精度回归、部署许可审查继续保留 |

以上为可登记建议，本文件不修改覆盖表、卡目录或总计。只有在维护者采纳且其余状态无变化时，原15个剩余具体方法任务才相应减为12；不声称已经更新索引或全部L5完成。

## 来源与获取限制

<a id="source-1"></a>

- **[cite:1] P0／原报告入口。** Frédéric Devernay，*A Non-Maxima Suppression Method for Edge Detection with Sub-Pixel Accuracy*，RR-2724，1995年11月。[HAL PDF](https://inria.hal.science/inria-00073970/file/RR-2724.pdf)。本次网页文本到正文p.3，独立字节请求可解析为8页（含前置页），未取得其后正文；不得记“官方完整20页已读”。HAL封面入库日2006-05-24与报告年份分开。

<a id="source-2"></a>

- **[cite:2] 原报告镜像／同文非独立来源。** [IPOL教学目录中的RR-2724](https://dev.ipol.im/~morel/Dossier_MVA_2011_Cours_Transparents_Documents/2011_Cours1_Document1_1995-devernay--a-non-maxima-suppression-method-for-edge-detection-with-sub-pixel-accuracy.pdf)。本次可解析25个PDF页面，重点核对印刷pp.3–9的NMS、三点峰、一维例、平方幅值警告、§3整数像素挂接及实验梯度说明；公式抽取有旧字体乱码，以可辨认项、2017式(5)及条件代数交叉，不声称逐图、逐公式审核全报告。作者出版页可读但截断，独立HTTPS请求TLS失败；未因此伪造原程序或许可。[SINGLE-SOURCE]

<a id="source-3"></a>

- **[cite:3] P0／IPOL官方元数据。** [文章与代码页](https://www.ipol.im/pub/art/2017/216/)，DOI `10.5201/ipol.2017.216`。已核作者、页码347–372、发表时间、软件v1.0日期、AGPL元数据、SWH标识和2022编辑修订说明；SWH未独立取回比对目录内容。

<a id="source-4"></a>

- **[cite:4] P0／IPOL方法原文。** [2017低分辨率PDF](https://www.ipol.im/pub/art/2017/216/article_lr.pdf)。网页抽取前部截断，内存读取取得26页；定向核对§2–5、§6相关说明、§7算法1–3、5及§8复杂度，未逐项重现后部图像实验。§5严格比较说明与算法3平局规则的差异按算法和源码解释；文章首页CC-BY-NC-SA与代码许可区分。[SINGLE-SOURCE，论文／代码同体系交叉]

<a id="source-5"></a>

- **[cite:5] P0／固定软件包。** [devernay_1.0.zip](https://www.ipol.im/pub/art/2017/216/devernay_1.0.zip)，v1.0，2017-10-10，SHA-256见DE17。实际读取`README.txt`、`devernay.h`、`devernay.c`关键函数与包内目录，许可由代码头／README和官方页相互核对；未执行包内程序。后续复现可用该哈希核包，不能只固定下载URL。

<a id="source-6"></a>

- **[cite:6] P0／HED会议原文。** [ICCV 2015 CVF PDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf)。网页抽取在目标函数附近截断；内存PDF为9页，定向补读pp.1398–1402的式(1)–(6)、架构、参数、共识、评测及聚合。另查[2015 arXiv v1](https://arxiv.org/html/1504.06375v1)，其章节／文字不是正式版替身。公式局部特殊字体乱码不影响本文重新书写已辨认关系；未逐图审核或复现实验。[SINGLE-SOURCE]

<a id="source-7"></a>

- **[cite:7] P0／作者仓库及版本。** [s9xie/hed](https://github.com/s9xie/hed)，已读README的更新声明、模型／数据／代码入口及外部NMS说明；GitHub提交API返回[固定提交912632b986acc6dd6cc33b95603b2f279d7bd9f2](https://github.com/s9xie/hed/tree/912632b986acc6dd6cc33b95603b2f279d7bd9f2)，日期2023-03-27。README当前模型指标不转写为2015原论文指标，更不是本仓库实测。

<a id="source-8"></a>

- **[cite:8] P0／固定训练图。** [train_val.prototxt](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/train_val.prototxt)。已读五侧头、冻结上采样、crop、logit融合、六损失、输入均值和学习率倍率。

<a id="source-9"></a>

- **[cite:9] P0／固定CPU损失。** [sigmoid_cross_entropy_loss_layer.cpp](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp)。已读前反向计数与归一化；本轮未取得同提交[同名`.cu`路径（404证据）](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cu)，HTTP明确返回404；[固定目录API](https://api.github.com/repos/s9xie/hed/contents/src/caffe/layers?ref=912632b986acc6dd6cc33b95603b2f279d7bd9f2)返回200、100个条目，包含该`.cpp`但无同名`.cu`。这是路径不存在／源码未取得，不是读到空函数或GPU入口空返回；GPU调用、回退与实际计算均未核验，不推断GPU不计算、零损失或CPU/GPU等价。非二值标签前反向不一致仅限已读CPU实现，不声称已发生于作者二值训练资产。

<a id="source-10"></a>

- **[cite:10] P0／固定训练入口。** [solver.prototxt](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/solver.prototxt)、[solve.py](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/solve.py)。已核优化、bilinear surgery及`step(100000)`与max_iter的配置差异，未启动训练核对实际调度。

<a id="source-11"></a>

- **[cite:11] P0／固定标签读取。** [image_labelmap_data_layer.cpp](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/src/caffe/layers/image_labelmap_data_layer.cpp)。已核同步尺寸、resize、`cv_gt/255`与五人共识注释；未下载真实标签，不能声称数据已满足该编码。

<a id="source-12"></a>

- **[cite:12] P0／固定坐标裁剪。** [crop_layer.cpp](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/src/caffe/layers/crop_layer.cpp)。已读坐标映射追溯、同尺度／整数位移检查及前反向拷贝；未执行所有依赖层的坐标组合测试。

<a id="source-13"></a>

- **[cite:13] P0／固定代码许可。** [LICENSE](https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/LICENSE)。已读共享版权、源码／二进制再分发两项保留声明及免责声明；仅用于该代码根许可说明，不扩展为模型、数据或第三方工具全部许可已清。
