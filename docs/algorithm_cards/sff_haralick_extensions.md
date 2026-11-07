# SFF 局部深度与 Haralick 原分类扩展卡

[学习规范](../algorithm_learning_standard.md) · [覆盖台账](../evolution_coverage.md#去重缺口清单) · [对焦总览](../autofocus_evolution.md) · [分类总览](../classification_assembly_evolution.md) · [HA01 原卡](haralick.md#ha01)

## 范围、状态与交接意见

核验日期：**2026-11-06**。本文件只建立两张独立扩展卡：`sff_haralick_extensions.md:SF94` 承接 **G12／11-06**；`sff_haralick_extensions.md:HC73` 承接 **G16 中 07-01 原分类链**。HC73 复用 HA01 的 GLCM 表示，不重复登记 06-01，也不另造一个历史任务。未修改原卡、台账、导航、源码或依赖清单。

| 卡／任务 | 本次核清的机制 | L1／L2／L3／L4／L5 | 逐项交接意见 |
|---|---|---|---|
| [SF94](#sf94)／G12 | 原 SML 的双向二阶差绝对值、门控局部和；逐像素焦栈、峰的相邻三点、等距高斯参数、台位移到表面高度、归一化查表 | 已核验／已核验／部分／已核验／部分 | 按“具体方法机制匹配，不等于完整复现”的台账口径，**建议 G12／11-06 由部分转已有**；仅针对 1994 机制。系统未实现，真实深度精度未验；本文件不实际销项或改计数 |
| [HC73](#hc73)／G16 | 复核 pp.616–617 原投票、闭区间／密度式(7)、外部式(8)原印 ≥；新增1972第一版目录 p.viii、导读 p.5，定位 §4.3 p.100／§4.4 p.106 | 部分／部分／部分／部分／部分 | **G16／07-01 保留部分，不整体销项**。训练正文未得；外部字面已核、原意未核。零宽须定义域／退化声明，集合输出不强求唯一化；不以最近盒、SVM 或默认最小二乘替代。见文末最小证据清单 |

“已核验”仅指下文明确条件内的原文机制和代数，不指工程全域正确、论文实验复现或现场验收。两卡均有 L1–L5 正文、五问答和有限检查；层级状态不因有代码块而自动升级。两篇原论文的具体机制均标 **[SINGLE-SOURCE]**，作者版与课程镜像是同文，不能算三份独立验证。`[INSUFFICIENT DATA]` 表示实际缺证；“原印可读”与“作者原意确定”分开。

## 证据定位与原卡差分

| 对象 | 原卡只覆盖什么 | 本卡新增原页证据 | 仍不能宣称什么 |
|---|---|---|---|
| F10／F13 → SF94 | 全局标量曲线的等距三点式／离线全扫；无局部 SML | Nayar–Nakagawa 1994，p.825 §III 高度链，p.826 式(12)，p.827 式(13)–(15)，p.828 式(16)–(22) | 不能把 `gaussian_fit` 或文件数据源当整套 SFF；不能把 1994 内容无差别回填 1989 报告 |
| HA01 → HC73 分段线性 | 原卡知道两两平面及外引训练，但符号和边界未闭合 | Haralick 等 1973，p.616 式(5)、p.617 式(6)及两类平票文字，p.621 参考[19]；Fukunaga 1972 目录 p.viii、导读 p.5 | 目录定位 §4.3 起p.100、§4.4 起p.106，不等于已读算法；训练目标、目标值与求解步骤仍缺 |
| HA01 → HC73 min-max | 原卡只可靠核清逐类逐维极值，未核完整比较式 | p.617 式(7)重叠密度乘积；式(8)扫描放大可辨 **≥** 及全部 K 类比较 | 不默改为 ≤；不以“最近矩形”解释全部项；无正式勘误、无原实验代码 |

原文页码一律为**印刷页码**；PDF 序号另列在来源节。Nayar 作者 PDF 的关键后页无可提取文字层，已直接读取 pp.826–831 扫描图，另放大 p.828 式(19)。Haralick 已读取 pp.616–617 扫描图并放大式(7)(8)，避免将文本提取结果当完整公式。[cite:1](#source-1)[cite:2](#source-2)

<a id="sf94"></a>
## SF94：Nayar–Nakagawa 1994 SML 与逐像素 SFF 深度

### 任务、演进与契约

**版本：**Shree K. Nayar、Yasuo Nakagawa，*Shape from Focus*，IEEE TPAMI 16(8)，824–831，August 1994，DOI `10.1109/34.308479`。1989 CMU 报告与该期刊文不是同一作者表／版本，本卡不核定其全部差异。[cite:1](#source-1)

**前序瓶颈 → 改变 → 机制 → 代价 → 遗留问题：**一般场景全图自动对焦的一个最佳位置无法表示粗糙多深度表面；SFF 改用局部纹理评分和每像素一条焦曲线，令采样最大值精化为连续台位移，再通过已标定焦平面恢复高度。它增加多帧采集、局部窗口、逐像素估计和定位标定成本；仍受弱纹理、遮挡、窗口跨深度、倍率漂移和峰形近似限制，并非所有场景的被动深度通解。原文 p.825 明确选择移动物体、固定光学系统与传感器；**只在焦峰附近小位移近似倍率不变**，不是整栈严格配准保证。[cite:1](#source-1)

输入为 `I[k,y,x]` 浮点灰度焦栈、实测台位移 `d[k]`、焦平面相对初始参考面的距离 `d_f`、固定的像素步距 `s`、窗口半径 `N`、门限 `T1`、有效域。位置与 `d_f` 同长度单位；`s,N` 为像素单位，`T1` 为灰度差响应单位。原离散模型使用等距 `d[k]=d0+k*h`、`h>0`；不是 RTImg 镜头对焦距离标签。

输出为每个有效像素的离散峰索引 `m`、连续最佳台位移 `d_hat`、表面高度 `h_surface=d_f-d_hat`、峰宽 `sigma_F`、峰幅 `F_p`、有效位与失败原因。高度图不是相机光心坐标的 Z；转成点云还需横向倍率／像素尺寸及外参，原式不替代该坐标链。可由每点最清晰帧另做图像拼接，但这不是深度估计本身。

| 独立字段 | 状态与范围 |
|---|---|
| L1／L2 | 已核验：1994 文明确局部评分—逐点曲线—相邻三点—高度链；防护分支是本卡工程约定 |
| L3 | 部分：参数量纲、论文场景参数和排障顺序已分开；未做本地真实纹理／设备选参 |
| L4 | 已核验：原式的等距重写、尺度还原和条件敏感性；不认可无条件高斯性或误差界 |
| L5 | 部分：有参考伪代码、文档内有限检查与本地组件差分；无专用 SFF 模块／硬件闭环 |
| 验证 | 确定性算术／边界断言；不生成仿真焦栈，不报告恢复精度或速度 |

### L1 步骤：局部焦栈不能压成全图一条曲线

1. **建立采集轴。**固定镜头／传感器、照明、曝光和坐标约定，测 `d_f`，在覆盖所需高度范围的合法台位移上采图。原模型以物体向焦平面移动为 `d` 增大，故高度是 `d_f-d_hat`；反向设备编码须先转换符号。离线估计以有限焦栈遍历结束为停止条件，不包含自动扩扫／电机控制算法。
2. **计算 modified Laplacian。**对每帧、每个完整支持中心计算横纵二阶差，分别取绝对值再相加：

\[
D_x I=2I[y,x]-I[y,x-s]-I[y,x+s],\quad
D_y I=2I[y,x]-I[y-s,x]-I[y+s,x],\quad ML=|D_xI|+|D_yI|.
\]

3. **门控并局部求和。**原式(14)的实际比较符是 `ML ≥ T1`：

\[
F_k[y,x]=\sum_{v=y-N}^{y+N}\sum_{u=x-N}^{x+N}
ML_k[v,u]\,\mathbf1[ML_k[v,u]\ge T_1].
\]

   原文是和，不是均值，不是“超过阈值只计1”，也不是截断成 `max(ML-T1,0)`。式(13)没有除以 `s²`。本卡完整支持模式要求窗口内所有 ML 所需的灰度样本有效，外边界至少留 `N+s` 像素；不补零制造纹理。这是显式工程边界，原文未规定全部边界策略。[cite:1](#source-1)，p.827 式(13)(14)。
4. **逐像素形成焦曲线。**固定 `(y,x)`，读取 `F[:,y,x]`，找离散全栈最大 `m`。每个像素独立取峰，不先全图求和再给所有像素同一个 `m`。原文说最大模态上的 `F[m-1],F[m],F[m+1]`，不是将全栈三个最高分排序后拟合；其脚注承认弱次峰与多模态，不能承诺始终选到真实表面。[cite:1](#source-1)，p.827 §VI。
5. **以三个相邻正分拟合。**限定内部峰、同一支持、等距台位移、可辨负曲率，按 L4 计算 `d_hat,sigma_F,F_p`。本卡对端点、非正邻分、并列最大、近乎平坦或非有限结果返回带原因的无效连续估计，可保留离散峰作为诊断但不当成功深度。该保守拒绝策略不是论文完整异常处理的复述。
6. **回填高度与质量字段。**`h_surface[y,x]=d_f-d_hat[y,x]`；记录峰邻域分数、曲率和有效支持。原文 p.831 的实例另对深度图做 5×5 中值滤波以去除弱纹理错误点；本卡不默认滤波，更不以补洞后的稠密图掩盖原无效率。[cite:1](#source-1)

### L2 机制：非线性 SML、局部性与模型假设

普通 Laplacian 是 `D_x+D_y`，横纵符号相反会抵消；原 modified Laplacian 改为 `|D_x|+|D_y|`。三角不等式给出 `ML ≥ |D_x+D_y|`。如两向响应等幅反号，普通响应为零而 ML 非零，这不是“另一个同名能量版本”。窗口汇聚把局部纹理响应变成更稳定的统计，却同时混入邻近深度；增大窗口不能无代价提升精度。[cite:1](#source-1)，pp.826–827。

门限抑制弱响应参与，不能保证只删噪声；二阶差的绝对值也有噪声偏置。SML 含绝对值、比较和求和，**不是线性卷积**。p.826 式(7)(8)关于线性移不变算子与模糊卷积交换的说明，不能越过非线性环节直接证明整个 SML 可写成单个卷积。

原文用离焦低通和主导频率的衰减说明峰附近近似高斯；它在 p.826 借 central limit theorem 作解释性论述，p.827 实测又明确曲线两侧并不对称。因此“高斯 PSF → 任意纹理的 SML 对台位移严格高斯”**不是已证明定理**。本卡采用的是 p.827 式(15)的局部经验模型，不从中央极限定理推导无偏深度或全局单峰。[cite:1](#source-1)

局部三点方法消去每点不同的峰幅与峰宽，适应纹理强弱不同；但并不创造无纹理处的深度信息。对完整支持上的常量或仿射灰度平面，两向二阶差均为零，任何焦位都可能同分；必须报告不可辨识而不是输出首帧。未知加性**评分底**破坏 log 二次性；它不同于可被二阶差消去的加性**图像灰度偏置**。

### L3 工程权衡：量纲、原实验与排障顺序

| 参数／环节 | 原文信息及选择依据 | 失效证据与本卡处理 |
|---|---|---|
| `s=step` | 像素跨度；p.827 为适应纹理尺寸而引入。式(13)未做连续导数的 `s²` 归一化 | 换跨度会改频响和幅值，不能只重用旧 `T1`；先看纹理周期和像素尺度 |
| `N`／窗口 | p.827 典型 3×3 或 5×5，即 N=1 或2；同页 Fig.5 评价实验却用10×10，不能硬写成同一个整数 N | 大窗口平滑深度边，小窗口弱纹理波动大；边界／薄结构分开评价；偶数窗口须显式锚定，非本卡对称式默认 |
| `T1` | 灰度响应单位；p.827 Fig.5 和 p.830 钢球实验使用7，`s=1` | 7是原采集条件参数，不是任意归一化图像通用阈值；检查阈上比例、曝光、位深及噪声 |
| `h=Δd` | 长度单位；p.827 脚注说峰宽未知、步距按经验选择，三点须处在峰区 | 粗步漏窄峰／邻分为零；过细则分差被噪声淹没并增加采集成本。先用代表场景高密度扫描观察峰宽，再评估候选步距，非无条件固定帧数 |
| 峰与倍率 | p.825、p.827 仅峰附近小位移近似倍率恒定 | 查看图像特征随 d 的位移与多峰，不把空间漂移当局部焦差；额外配准属于扩展，会改变重采样纹理 |
| `d_f` 与台位移 | p.825 §III 两者由测量获得；输出高度是差值 | 零点、轴符号、单位、回差错误会整体污染深度；先校轴再调插值 |
| 曲率／拒绝门 | 本卡工程策略；无原文通用阈值 | 记录 log 曲率、邻分、重复采集波动和端点率；有限检查用的容差仅为浮点断言阈值 |
| 后处理 | p.831 个别实验深度图用5×5中值；非原 SML 必要步骤 | 原深度与滤波深度分开报告，不能用平滑后外观替代真实表面误差 |

排障顺序：位置与曝光同步 → 单位、方向、参考面 → 有效域与 ML／门控 → 各像素完整焦曲线 → 峰区间与曲率 → 深度映射 → 可选后处理。验收至少需要独立高度参考、绝对误差／重复性、无效率、深度阶跃保真、弱纹理分层、运动及采集尾延迟；**本次均未执行**。

原实验参数用于防止错误泛化，不作本站结果：p.830 钢球实验为 `Δd=100 μm`、13幅、5×5 SML；p.831 通孔填充实例为18幅、8 μm 步距。原文不同实验选择不同步距，不存在“1994 法统一只需三帧”结论。[cite:1](#source-1)

### L4 数学推导：由局部高斯到物理高度

#### 等距三点消元

令三个相邻位移为 `d_m-h,d_m,d_m+h`，正分的对数为 `ell_-,ell_0,ell_+`。原 p.827 式(15)、p.828 式(16)为

\[
F(d)=F_p\exp\left[-\frac{(d-\bar d)^2}{2\sigma_F^2}\right],\qquad
\ell(d)=\ln F_p-\frac{(d-\bar d)^2}{2\sigma_F^2}.
\]

以中心化变量 `t=d-d_m` 写成 `ell(t)=a t²+b t+c`，则

\[
D=\ell_- -2\ell_0+\ell_+,\qquad
a=\frac{D}{2h^2},\quad b=\frac{\ell_+-\ell_-}{2h},\quad c=\ell_0.
\]

顶点满足 `2at+b=0`，因此原式(17)(18)可稳定重写为

\[
\delta=\frac{\ell_- -\ell_+}{2D},\quad
\bar d=d_m+h\delta,\qquad
\sigma_F^2=-\frac{h^2}{D},\qquad
\ln F_p=\ell_0+\frac{(d_m-\bar d)^2}{2\sigma_F^2}.
\]

最后一式等价于原 p.828 式(19)的 **`F_p = F_m / exp[-(d_m-d_bar)²/(2 sigma_F²)]`**，不是乘以该负指数。直接查看放大扫描可确认斜线；亦可回代式(15)校验。应有 `F_p ≥ F_m`。等距、非平台局部最大使 `D<0`；若两侧都不高于中心，则令 `A=ell_0-ell_-≥0,B=ell_0-ell_+≥0`，有 `δ=(A-B)/(2(A+B))`，故 `|δ|≤1/2`。越过半步不是靠裁剪就能修复的正常无噪精确解。[cite:1](#source-1)

#### 归一化查表与不等距边界

p.828 式(20)–(22)令 `r_-=F_-/F_m,r_+=F_+/F_m`，中心分数变为1，位置变为 `(-1,0,1)`。两邻比确定归一化顶点 `d_tilde`、宽度 `sigma_tilde` 和幅度 `F_tilde_p`，再恢复

\[
\bar d=d_m+h\tilde d,\quad \sigma_F=h\tilde\sigma_F,\quad
F_p=F_m\tilde F_p.
\]

这说明可建立以 `(r_-,r_+)` 为索引的二维查表，消除每点重复 log／参数求解。近平台 `r_-=r_+=1` 退化，零邻分不能取 log。原文未给完整查表离散精度、插值和越界协议；本卡直接计算，不声称复现了原表或验证其加速／误差。三点不是三个任意峰值，也不是查表后可不记录 `h,d_m`。

若真实位置不等距，设 `t_i=(d_i-d_m)/s_d`，其中 `s_d>0` 是显式尺度，解三行 `[t_i²,t_i,1]` 系统再取顶点，是数学上可定义的**扩展**；它不是原等距式的“平均步距修正”。本卡严格模式拒绝不等距。本地 F10 对 `(0,1,3)` 和 `log F=(-1,0,-4)` 返回0.55，而真实坐标二次顶点为1；不能将该函数原样用于一般台位移。[cite:3](#source-3)

#### 可辨识性、敏感性与复杂度

`K<3`、重复位置、端点最大、无纹理、非正分、`D≈0`、混合深度和显著横向漂移均破坏相应条件。固定峰索引且小扰动下，令 `U=ell_- -ell_+`，有

\[
\Delta\delta\approx\frac{\Delta U}{2D}-\frac{U\,\Delta D}{2D^2},\qquad
\Delta\ell\approx\Delta F/F.
\]

因此低邻分的相对噪声与小曲率放大误差；这个一阶式不覆盖峰索引跳变。已标定 `h_surface=d_f-d_hat` 给出 `Δh_surface=Δd_f-Δd_hat`；若要写方差，须保留协方差项，不能默认标定与估计独立。峰宽不是高度估计标准差，也不是置信区间。**未提供通用理论深度误差界**。

K幅、H×W像素、窗口边长 w=2N+1：直接 ML 为 `O(KHW)`，直接窗口和为 `O(KHWw²)`，积分图／分离滑动和可降为 `O(KHW)`；逐点峰检索 `O(KHW)`，插值 `O(HW)`。保存全部 F 需 `O(KHW)`；流式维护候选和两邻帧可减少存储，但必须保留最终全局峰的正确前后邻分及并列信息，不是只记三个最高值。采集耗时另计。

### L5 实现：明确数据布局、伪代码与等价优化

参考数据布局 `float64 I[K,H,W]`、`float64 d[K]`、`bool valid[K,H,W]`；位置严格递增、同单位。二阶差先转浮点再运算，避免无符号溢出。窗口在像素域、峰拟合在第0轴，两者不可混。原文未规定存储和浮点类型，以下为**未接入仓库的参考伪代码**；安全拒绝策略显式强于原文。

```text
SFF94(stack, d, d_f, step, N, T1, tolerances):
    require finite grayscale K-by-H-by-W, K >= 3, step >= 1, N >= 0
    require finite d_f, finite T1 >= 0, matching strictly increasing d
    require equal spacing within recorded acquisition tolerance
    h = d[1] - d[0]
    for k in 0..K-1:
        dx = 2*I[k,y,x] - I[k,y,x-step] - I[k,y,x+step]
        dy = 2*I[k,y,x] - I[k,y-step,x] - I[k,y+step,x]
        ML = abs(dx) + abs(dy)
        Q = where(ML >= T1, ML, 0)
        F[k] = complete_support_box_sum(Q, radius=N)
        support[k] = all_required_source_pixels_valid
    for each (y,x) valid at every k with identical support:
        curve = F[:,y,x]
        m = unique_argmax_or_invalid(curve)
        if m is invalid or endpoint: emit INVALID_PEAK; continue
        triple = curve[m-1:m+2]
        if any triple <= 0 or not finite: emit INVALID_LOG; continue
        ell = log(triple)
        D = ell[0] - 2*ell[1] + ell[2]
        if D >= -curvature_tolerance: emit FLAT_OR_NONCONCAVE; continue
        delta = (ell[0]-ell[2])/(2*D)
        if abs(delta) > 0.5 + position_tolerance: emit INVALID_VERTEX; continue
        d_hat = d[m] + h*delta
        sigma2 = -h*h/D
        log_Fp = ell[1] + (d[m]-d_hat)^2/(2*sigma2)
        emit d_f-d_hat, d_hat, m, sqrt(sigma2), log_Fp, VALID
    return height_map, validity_map, diagnostics
```

`log_Fp` 可先保留在 log 域，只有需要幅度且范围可表示时才 `exp`。把全零邻分夹到 ε 后取 log 会产生伪峰宽；端点回退若保留，应标 `DISCRETE_ONLY`，不能混入有效连续深度。针对 mask，本卡宁可拒绝支持改变的点，不把不同窗口像素数的和直接跨焦位比较。

| 源码／原卡 | 实际角色 | SF94 所缺环节 |
|---|---|---|
| [focus_measures.py](../../traditional/autofocus/focus_measures.py#L45-L64) 的 `smd`／`laplacian_energy` | 一阶绝对差均值／整图 Laplacian 平方均值 | 均不是 ML 后阈值局部和；没有 `F[K,H,W]` |
| [focus_search.py](../../traditional/autofocus/focus_search.py#L13-L29) 的 `gaussian_fit` | 一条曲线的 argmax 和三点等距简式 | 无逐像素支持、物理高度、有效位；非正分夹值、不等距平均间隔不能继承为原法 |
| [F13](measurement_focus.md#f13-全扫描与数据源) | 离线全局曲线与文件帧 | 不代表台位置实测／采图同步，不是 SFF 采集系统 |

优化首先逐元素对照：阈值在求和前执行，积分图四角只汇聚 `Q`；mask 的完整支持计数另算；分块 halo 至少 `N+step`；求和顺序变化仅容许明确浮点误差，改变窗口／阈值／灰度预处理不是等价优化。查表须增加量化边界、平台附近误差、缩放还原的回归，尚未实施。

### 五问五答：SF94 贯穿 L1–L5

| 层 | 问题 | 答案要点及条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|---|
| L1 | 有一个倾斜面的焦栈，如何得到高度而非一个最佳焦位？ | 每点 ML→门控窗口和→沿K取峰→相邻三点→`d_f-d_hat`；位置单位和符号先统一 | 先全图均值再把单一峰复制成深度图 | p.825 §III、p.827 §VI；L1 |
| L2 | 为何 `abs(Dx+Dy)` 不能替换 `abs(Dx)+abs(Dy)`？ | 横纵二阶响应可反号；相消使前者为0而后者非零，且绝对值／门控破坏线性 | SML 是一个等效5点卷积核 | pp.826–827 式(12)–(14)；L2 |
| L3 | 步距更细是否一定降低深度误差？ | 需峰被覆盖、三点位于有效峰区、运动及评分分差可辨；过细可使曲率差淹没于噪声，采集成本增加 | 用电机分辨率当深度精度，固定取论文的100 μm | p.827 脚注、p.830实验；L3 |
| L4 | 如何由三个正分求峰宽、峰位、峰幅？ | log二次消元，`D<0`，`sigma²=-h²/D`，顶点偏移和 `log Fp`；原式(19)是除指数 | 把峰宽当估计标准差，或令 `Fp<Fm` | p.828 式(16)–(22)；L4 |
| L5 | 现有 F10 能直接循环调用完成复现吗？ | 仅等距正常三点局部可对照；另建SML、mask、逐像素数据、拒绝、台位移标定和高度链 | 非等距平均步距正确，零分夹ε即有深度 | F10函数体、有限检查；L5 |

<a id="hc73"></a>
## HC73：Haralick 1973 分段线性与 min-max 原分类链

### 任务、演进与诚实边界

**版本：**Robert M. Haralick、K. Shanmugam、Its'hak Dinstein，*Textural Features for Image Classification*，IEEE TSMC SMC-3(6)，610–621，November 1973。本卡聚焦 pp.616–617 的两类决策方法；HA01 已写 GLCM、14特征和版本分歧，不重复用“GLCM能接任意分类器”代替这次原法缺口。[cite:2](#source-2)

演进位置是“空间灰度对统计 → 可计算纹理表示 → 有标签类别判别”。两两超平面支线用于样本相对较多的显微和卫星数据；min-max 支线用于每类样本较少的航空图。前者需估计两两权重并解决投票冲突，后者仅估逐维范围却丢弃特征相关性、易受极值和小样本影响。它们不是装配关系推理、异常检测，也不是现代开放集识别。

输入为已冻结特征协议的有标签矩阵 `X[M,p]` 和类标 `y[M]`，以及训练／测试分组；推理输入同维度向量。输出是原规则所允许的候选类别集合、投票／密度／外部比较诊断与有效性。集合不唯一时不得暗中选最小类号。原论文主要做闭集类别分配；本卡的 `UNRESOLVED` 用于表达原规则未定义／未核清，不是经验证的未知类检测器。

| 项目 | 已知与缺口 |
|---|---|
| L1 | 部分：重核 pp.616–617 两支流程；1972 §4.3 起p.100、§4.4 起p.106已定位，训练正文及具体选法未得 |
| L2 | 部分：外部 ≥ 字面与均匀模型适用域已分离；目录／导读不能确定训练机制或作者更正意图 |
| L3 | 部分：区分零宽的除法定义域、可选修复与单标签接口的平票要求；原实验是否触发退化未验 |
| L4 | 部分：式(7)(8)非严格比较对应候选集合，零响应／零宽的未定义性可证；1972目标值与求解公式未取 |
| L5 | 部分：保留原印决策核与停止训练接口，补有限分支检查；未实现原训练器，不以保护性拒绝冒充原历史行为 |

### L1 步骤：分开两支训练与推理

#### 共同表示与实验协议

先依据 HA01 固定量化、偏移、共现归一化、特征版本和方向聚合，然后按所选实验配置形成向量；不能默认三组原实验都是14×均值／极差的28维。p.617 显微实验为 f1、f2、f3、f9 的方向 mean／variance 共8变量；航空图为 f1–f11 的四方向44分量再取 mean、range、deviation 为33维。卫星实验另合光谱统计，p.616 写 mean／standard deviation，p.617 又出现 mean variance 等文字；本卡保留该历史配置措辞差异，不凭现代惯例补成唯一可复现配置。[cite:2](#source-2)

HA01 的 f7 原印与修正、f10 方差解释、log 底及 f14 实现差异仍保留。分类机制核清不抹平这些表示版本；下文分类代数以**已经给定同一版本的有限特征向量**为前提。

#### 分段线性支线

1. 原文将数据划分训练与测试；对 K 类的每对 `(i,j)` 建一个分隔超平面，共 `K(K-1)/2` 个，法向量包含偏置项。
2. p.616 明说以参考[19]第4章的 **regression type algorithm** 得到权重；p.621 [19]确定为 **Fukunaga，1972，Introduction to Statistical Pattern Recognition**。新增第一版版权页、目录 p.viii 与导读 p.5 原页：第4章为 *Linear Classifiers*，§4.3 *Linear Discriminant Function for Minimum Mean-Square Error* 起p.100，§4.4 *Desired Output and Mean-Square Error* 起p.106。它们是优先取证位置，**不是 HC73 已确定采用的节号或算法号**。未取得这些节的正文，不能确定目标值编码、类／样本权重、经验目标、求解步骤及秩亏约束；只有原法确实迭代时才进一步要求初值／更新／终止，不能先假定它必须迭代。训练仍缺，不写成 `fit_SVM`／默认 `lstsq` 后宣称原训练完整。[cite:2](#source-2)[cite:5](#source-5)
3. 原 p.617 式(6)是 `Z=[1;X]`，**常数在前**。可将常数移到最后，但须同步置换权重；HA01 的 `(f,1)` 仅是重排表示，不可直接配原 `W`。
4. 推理计算所有非自对响应；非零正响应给该类一票。选最大票类；若仅两个类 `m,n` 平票，原文字用两者超平面，`W_mn^T Z ≥ 0` 判 m，否则判 n。多类循环平票、投票式遇零的统一方案未在该段提供，本卡返回未决，不补造训练／推理约定。

#### Min-max 支线

1. 对每类 k、每维 n，从**当前训练样本**取下界 `b_nk=min X_n`、上界 `a_nk=max X_n`。原记号 b 是最小、a 是最大，不能按英文字母惯例倒置。
2. 航空图按 leave-one-out：留出一张，其余169张计算范围，再判留出张；重复170次。不是先用170张确定极值再“留一”预测。现代分组验证需另设计，不能把其历史协议直接当工业跨批验收。[cite:2](#source-2)，p.617。
3. 找覆盖集合 `C(X)={k: 对所有n，b_nk≤X_n≤a_nk}`。在参与比较的宽度均为正时，至少一个盒覆盖就仅在 C 中按原式(7)比较均匀密度 `Π_n 1/(a_nk-b_nk)`，最大者获选，即同等先验解释下体积最小的覆盖盒；只一个覆盖就选它。这个快捷分支不授权将零宽密度也称为已定义。
4. 全部盒均不覆盖时，原式(8)对所有 K 类比较逐维**最近端点绝对距离／本类该维宽度**之和，扫描原印是 **≥**，即取最大和。这个字面规则不是最近盒策略，见 L4 的原式及反例。若要求“作者实际意图”的预测，本卡在外部区返回 `UNRESOLVED_ORIGINAL_INTENT`；只有明确选择 `PRINTED_1973` 模式才执行字面最大值。
5. 零宽度、无类样本、NaN／Inf或未决并列均有显式状态。有限极值扫描与有限类别遍历结束即停止；这支无迭代收敛过程。原文未给零宽度修复或唯一化并列的全套政策，不能默认加ε即历史复现。

### L2 机制：超平面投票、盒内密度与外部不确定性

多个两两平面划分空间；在不落于平面上的每个符号单元内投票不变，边界为分段平面。全胜类的区域是相应半空间交，具有凸性；但一般最大票类还可能拥有不全胜的单元，**不能从两两投票直接推导每类整体区域必为单一凸多面体**。需区分论文摘要的几何描述与含冲突／平票实现的严格集合性质。

min-max 的每类盒 `B_k=∏[b_nk,a_nk]` 包住该类所有训练点，但也包进未观察过的特征组合。例如两个相关特征沿一条斜线分布，轴对齐盒仍允许离线很远的组合；不能把“盒内”解释为有充分样本支持。极值对离群样本敏感；大盒虽然覆盖更多，却在重叠时因体积大而密度低。

在正宽度条件下，假设每类在盒内联合均匀：`p(X|k)=1/Vol(B_k)`，盒外为0；同等先验下的最大似然／MAP 正好得到式(7)。这个推导**只解释盒内**；若 X 在全部盒外，所有似然均为0，均匀模型无法推出式(8)。非等先验应乘 `π_k`，但原式(7)没有该项，本卡不悄悄增加它。[cite:2](#source-2)

外部式(8)已确认不是文字提取误把 ≤ 变 ≥；临时证据来源中的 p.617 图与课程扫描一致，不归因用户，仍是同篇论文的转载，不增加独立研究数量。但“确认原印 ≥”不证明当年程序确实取最大，或排除排印错误。作者文章目录及限定检索未取得针对本式的正式勘误／原代码，故记 **[INSUFFICIENT DATA：原意与实现]**，不写成“勘误不存在”。忠实讲解原印式不以取得勘误为先决条件；若要称为“更正后的原法”或“原实验实际判决”，才必须补相应一手证据。改成最小值是另一个版本，且仍不等于最近盒。[cite:2](#source-2)[cite:7](#source-7)

### L3 工程权衡：先冻结特征，再诊断分类规则

| 项目 | 量纲／代价及应保存的证据 | 失效与处理 |
|---|---|---|
| 特征协议 | p及每维单位由量化、特征、方向汇总决定 | 混用原版／修正 f7、f10 或熵底会改变输入；先比向量再查分类器 |
| 两两平面 | 数量 K(K−1)/2；推理 `O(K²p)`，权重存储同阶 | 标签方向／偏置位置错误会系统性反转票；记录每对响应与训练算法出处 |
| 训练秩／类别样本 | 每对可用样本数与 p+1 自由度需比较 | 原训练目标未核，不能引用现代最小二乘的秩条件当历史实现已完成；样本不足须显式失败 |
| min-max 宽度 | `a-b` 与该维特征同单位；宽度可为0 | 一类常量维、留一后仅一个样本都会退化；不用无记录的ε扩盒 |
| 盒覆盖与重叠 | 记录空覆盖率、多覆盖率、体积／log密度和极值样本ID | 高维小样本可大量落盒外；外部有输出不代表可靠，先保留原印／未决模式 |
| 并列／零响应 | 原投票式除以响应绝对值；min-max用非严格比较可多解 | 不能用 `argmax` 首项暗定唯一类；本卡输出候选集合或未决 |
| 划分 | 原航空协议LOO；现场应按工件、图块来源和批次分组 | 相邻裁块跨训练／测试会泄漏；表示选择和缩放也须限训练域 |

对于 min-max，统一的正仿射逐维变换若同时施于训练和测试，会使所有盒密度乘相同常数、端点距离比不变；它不是“哪怕每类分别缩放也不变”。对两两分类器，特征缩放还改变权重数值和训练条件，必须保存变换并对应权重。调参顺序：表示版本 → 划分 → 类别／维度／宽度与秩 → 响应符号和偏置 → 盒分支 → 决策政策；不先换分类器掩盖原机制缺口。

未执行的验收包括原论文配置逐项复现、真实类别混淆、跨批次泛化、外部样本风险、未知类拒识及装配合格率。原论文分类准确率不是本仓库数据，不用于证明修复或性能。06-01 异常迁移仍需单独正常模型与真实缺陷验收。

#### 零宽／平票规则是否必需：按所声称的契约判定

| 情形 | 必须交代什么 | 不必补造什么／升级边界 |
|---|---|---|
| 零宽 `a_nk=b_nk` | 式(7)所比较的类、式(8)全部类的分母须非零；限定正宽有效域，或报告退化。闭区间覆盖测试自身仍有定义 | 不必为了机制核验发明ε、删维或低维密度；全局拒绝任意零宽是本卡更保守的工程策略，不是原文命令 |
| 原密度／外部得分并列 | `≥`允许所有满足比较的k；返回完整候选集合即可忠实表示判决关系 | 集合接口不需要强行破局。要宣称对任意输入输出唯一标签，才需额外策略；不能将库的首个`argmax`当原法 |
| 两类最大票并列 | p.617 已给两者平面 `≥0`／`<0` 的破局文字，适用时应保留 | 不另换随机、最小类号等规则；不将局部文字扩成全部零响应约定 |
| 式(5)零响应／三类以上最大票并列 | 零响应是`0/0`；多类最大票集合可记录，但原段未提供完整单标签破局。可排除该域或显式未决 | 不必以未经原文支持的半票、循环淘汰等方式填满全部输入域；量化特征／有限样本下也不能凭“概率零”跳过检查 |
| 声称复现1973实验标签／准确率 | 需取得实际训练／测试输入并检查各分支是否触发；触发才须取得对应历史政策，或明确无法逐样本复现 | 不要求凭空寻找一份覆盖从未触发情形的原作者政策；原程序是可用证据之一，不是机制销项必须拿到的唯一物件 |

本表是由原式定义域和输出契约得到的复核判断，不是新找到的历史约定。**必要的是边界声明与可追溯性，不是任意“修好”所有边界。**因此旧交接中笼统要求“补齐退化政策才可核机制”的说法在本轮收窄；训练正文缺失仍是独立、实质的阻塞。[cite:2](#source-2)[cite:3](#source-3)

### L4 数学推导：可核公式与不可补造的训练

#### 两两投票与训练的证据边界

令 `W_ji=-W_ij` 表示同一平面的反向响应，这是把原文只存 i<j 的权重扩展成全部有序对的记号约定。`s_ij=W_ij^T Z`。原 p.616 式(5)可写为

\[
V_i=\sum_{j\ne i}\frac{|s_{ij}|+s_{ij}}{2|s_{ij}|}.
\]

当 `s_ij>0` 时分子为2s，项为1；负时分子为0，项为0；零时是 `0/0`，不能直接当0、半票或正票。非零响应且反对称时每对恰贡献一票，故 `∑V_i=K(K−1)/2`，这是可实现的不变量。若恰有两最大票类，使用原文另给的两类平票比较；若三类形成 `1胜2、2胜3、3胜1`，三者各一票，原两类破局文字不自动覆盖该循环。[cite:2](#source-2)，pp.616–617。

**关于“regression type”的有限推导。**为了说明为何不能仅凭这两个词补训练器，可在抽象回归假设下写设计矩阵 A 的行为 `Z_l^T`、目标 t、样本权重 Λ：

\[
J(w)=(Aw-t)^T\Lambda(Aw-t),\quad
\nabla J=2A^T\Lambda(Aw-t),\quad
A^T\Lambda A\,w=A^T\Lambda t.
\]

只有在**预先指定这个目标**、Λ正定且 A 满列秩时才可唯一求得正规方程解；实际计算通常用分解而非显式逆。不同 t、Λ、正则化、迭代更新会给不同平面。此式是本卡的条件代数说明，**不是已核 Fukunaga 1972 算法，不是 HC73 可替代训练器**；本卡不训练它、不引用其收敛去补原法。目录已核 §4.3 起p.100、§4.4 起p.106、§4.5 起p.111，故优先索取 **pp.100–110 连续原页及其引用的定义**；这是由相邻节起页界定的取证范围，不是已核算法页／式号。p.5 明说本章包含多种设计准则，更不能由章名倒推出唯一算法。Cover 1973 书评只确认1972书目；其 p.828 “八种下降法”的讨论对象为 Duda–Hart，不能移植成 Fukunaga 或 HC73 的训练步骤。[cite:4](#source-4)[cite:5](#source-5)

#### 重叠：原式(7)是最大密度，不是最近中心

令 `w_nk=a_nk-b_nk>0`、`V_k=∏w_nk`。原式(7)为

\[
k\in C(X),\qquad \prod_{n=1}^{p}\frac1{w_{nk}}
\ge\prod_{n=1}^{p}\frac1{w_{nj}}\quad\forall j\in C(X).
\]

因 log 严格递增，可最大化 `L_k=-∑log w_nk`，避免高维连乘上溢／下溢。并列体积时不严格不等式允许多个 k；“if and only if”没有保证唯一。边界 `X_n=b_nk` 或 `a_nk` 仍算覆盖。若包含一个零宽维，密度式本身不定义，不能在 log域加ε掩盖。

正仿射变换 `X'_n=α_n X_n+β_n, α_n>0` 后，`w'_nk=α_n w_nk`，所以 `L'_k=L_k-∑log α_n`；所有类共享偏移，重叠排序不变。类先验、每类独立尺度和按查询点调整盒宽都改变规则。

#### 外部：式(8)原印与几何反例

**仅在 `C(X)=空集` 时**，p.617 式(8)印为

\[
E_k(X)=\sum_{n=1}^{p}
\frac{\min\{|X_n-a_{nk}|,|X_n-b_{nk}|\}}{a_{nk}-b_{nk}},
\qquad E_k(X)\ge E_j(X),\quad j=1,\ldots,K.
\]

故严格字面模式选 `argmax E_k`。它不是归一化的点到盒距离：如果某坐标已在该类区间内部，最近端点距离通常仍大于0；真正的逐维区间距离是 `max(b-X,0,X-a)`。而“全部盒不覆盖”只意味着每盒至少一个维度在外，**不意味着每维都在外**。即使将 ≥ 改为 ≤，也不能自动把 E 叫标准点到盒距离。

一维取两个区间 `B1=[0,1]、B2=[3,4]`、查询 `X=2.25`，无覆盖。原印有 `E1=1.25、E2=0.75`，选 B1；最近区间选 B2。这是公式反例／单元检查，不是实验分类数据。二维若区间在某维为 `[0,2]`、该维查询为1，则该维 E贡献1/2而真正区间距离为0；另外一维在外即可使整个盒外部规则启用。

外部 E 对共享正仿射逐维变换不变，但宽度越小，给定绝对偏离的归一化比可能越大；原印最大化可偏爱窄而远的盒。因此无法由盒内均匀模型为其给出最大似然解释。**目前能核清字面，不能核清原意**；不宣布原作者“肯定写错”，也不静默修正。G16 因训练和此项未决保留部分。

#### 复杂度与保证范围

min-max 训练为 `O(Mp)`、模型存储 `O(Kp)`、单样本预测 `O(Kp)`；直接每轮重训的LOO为 `O(M²p)`。可按类／维维护带计数的第一、第二极值减少重训，但必须处理重复极值和留一后空类，尚未实现。两两推理成本已给，原分段线性训练复杂度和终止保证因算法未取得不填写。训练覆盖不保证泛化，也没有通用分类误差界；任何原测试结果都不能推出工业异常召回。

### L5 实现：原印决策核与明确停止的训练接口

使用 float64 的 `X[M,p]`、`bounds[K,p]`；元数据保存 `feature_version,class_order,bias_first,pair_orientation,external_mode,tie_policy,training_source`。对每个样本和边界检查有限性，缺值不能直接参与 `min/max`。模型与查询的特征顺序必须完全一致。

以下为**部分原法的参考伪代码，不是完整原法训练实现**。`REGRESSION_1972` 是尚未取得的算法，不提供假返回值；严格调用立即报告缺证。min-max 的 `PRINTED_1973` 明确只执行扫描原式，不声称作者原意已定。

```text
TRAIN_PIECEWISE_1973(X, y, feature_protocol):
    validate labeled finite vectors and frozen feature_protocol
    require classes and pair orientation explicitly defined
    return UNRESOLVED_TRAINING_SOURCE(Fukunaga_1972_chapter_4)

PREDICT_PIECEWISE_WITH_SUPPLIED_WEIGHTS(X, W):
    Z = concatenate(1, X)
    votes = zeros(K)
    for i < j:
        s = dot(W[i,j], Z)
        if not finite(s) or s == 0: return UNRESOLVED_ZERO_RESPONSE
        if s > 0: votes[i] += 1
        else: votes[j] += 1
    T = all_classes_with_maximum_vote(votes)
    if size(T) == 1: return T
    if size(T) == 2:
        m,n = ordered(T)
        return {m} if dot(W[m,n],Z) >= 0 else {n}
    return UNRESOLVED_MULTICLASS_TIE(T, votes)

FIT_MINMAX(X_train, y_train):
    for class k:
        if class empty: return INVALID_EMPTY_CLASS
        b[k] = featurewise_min(X_train[y_train==k])
        a[k] = featurewise_max(X_train[y_train==k])
    if any nonfinite bound or any a <= b: return DEGENERATE_WIDTH
    return b,a

PREDICT_MINMAX(X, b, a, external_mode):
    require finite matching dimensions and a > b
    C = {k: every b[k,n] <= X[n] <= a[k,n]}
    if C not empty:
        L[k] = -sum(log(a[k]-b[k])) for k in C
        return all_argmax(L, restricted_to=C), INSIDE_OR_OVERLAP
    E[k] = sum(min(abs(X-a[k]),abs(X-b[k]))/(a[k]-b[k]))
    if external_mode != PRINTED_1973:
        return UNRESOLVED_ORIGINAL_INTENT(E)
    return all_argmax(E), OUTSIDE_PRINTED_GE

LOO_MINMAX(X, y, external_mode):
    for q in all_sample_indices:
        bounds = FIT_MINMAX(X excluding q, y excluding q)
        if invalid bounds: record invalid fold; continue
        record PREDICT_MINMAX(X[q], bounds, external_mode)
    return per_sample_results_with_unresolved_and_invalid_counts
```

两类平票文字虽写 `≥0`，投票式本身遇任何零响应已不定义；伪代码提前返回未决，不以该局部破局文字扩张成所有平面的零响应约定。若现代实现采用数值“近零”带，应记录宽度并重新验证候选稳定性，不假称原文阈值。

`FIT_MINMAX` 在任一类任一维零宽时全局拒绝，强于只检查当前分支所需分母；这是保守接口，不是证明原法连覆盖测试都不能执行。`all_argmax` 保留集合，不默认每次成功结果只有一个标签。log密度与倒数乘积在精确实数域排序等价；浮点舍入可能改变近乎相等的排序或平票，现有小例检查不保证全域等价，真实实现须记录比较精度／容差政策。

优化只能在定义固定后做：非零投票项用布尔比较而非先算除法，保持方向与票数守恒；重叠比较用 log密度；批量 minmax 的广播轴需为 `[batch,K,p]`；极值缓存必须与逐折删除重训逐元素相等。**未执行原书训练、全LOO分类、外部版本修正、真实特征提取或速度剖析**。

### 五问五答：HC73 贯穿 L1–L5

| 层 | 问题 | 答案要点及条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|---|
| L1 | 取得33维航空纹理向量后，原法如何训练与评估？ | 每次留一，剩余样本按类逐维取极值；先看盒覆盖，再重叠密度／外部原印分支；记录退化与并列 | 全数据建盒后再留一，或一律最近中心 | p.617 Min-Max、航空实验段；L1 |
| L2 | 重叠时为何较小盒可胜出？ | 正宽均匀盒模型密度为体积倒数，只有覆盖类参与；同等先验解释 | 距离哪个中心近就选哪个；所有外部样本仍有非零均匀似然 | p.617 式(7)；L2、L4 |
| L3 | 零宽或平票是否必须补成唯一标签？ | 分母涉及零宽须限定域／报告退化；非严格比较可返回集合，无需强行破局；全域单标签才须另定政策 | 加ε／首个argmax后宣称原作者如此处理 | p.617 式(7)(8)；L3必要性表、L5 |
| L4 | 式(8)等于最近矩形吗？ | 不等；扫描为≥取大，且区间内部维仍算到端点距离；一维反例选更远盒 | 把OCR修成≤就算历史规则闭合 | p.617式(8)扫描放大；L4 |
| L5 | 有1972目录、默认最小二乘与两两投票能销项吗？ | 不能；§4.3 p.100／§4.4 p.106只定位取证范围，目标值、样本估计与求解仍缺；迭代要求须以正文为准 | 将1990第二版p.124–180或书评的Duda–Hart算法冒成1972原页 | p.616、p.621[19]；1972 p.viii、p.5；L4、L5 |

## 有限检查：只验证公式与明确分支

以下代码为文档内**确定性单元检查器**，不是性能数据，不生成仿真图像焦栈。小数组和代数输入只用于检查式(13)(14)、原峰参数关系、既有F10反例和分类原印分支。使用当前环境已有 NumPy；运行时从此文件提取代码块到内存，`python -B` 不写脚本或字节码。

```python
import numpy as np
from traditional.autofocus.focus_search import gaussian_fit

checks = []

def checked(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def sml_valid(image, step=1, radius=0, threshold=0.):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("image")
    if step < 1 or radius < 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("config")
    H,W = image.shape
    out = np.full((H,W), np.nan)
    ml = np.full((H,W), np.nan)
    for y in range(step,H-step):
        for x in range(step,W-step):
            dx = 2*image[y,x]-image[y,x-step]-image[y,x+step]
            dy = 2*image[y,x]-image[y-step,x]-image[y+step,x]
            ml[y,x] = abs(dx)+abs(dy)
    for y in range(step+radius,H-step-radius):
        for x in range(step+radius,W-step-radius):
            v = ml[y-radius:y+radius+1,x-radius:x+radius+1]
            out[y,x] = np.where(v >= threshold,v,0).sum()
    return out

def peak3(d,F,d_f):
    d,F = np.asarray(d,dtype=float),np.asarray(F,dtype=float)
    if d.shape != (3,) or F.shape != (3,):
        return "INVALID",None
    if not np.isfinite(d).all() or not np.isfinite(F).all() or not np.isfinite(d_f):
        return "INVALID",None
    gaps = np.diff(d)
    if (gaps <= 0).any() or not np.isclose(gaps[0],gaps[1],rtol=1e-12,atol=0):
        return "POSITION",None
    if (F <= 0).any():
        return "LOG",None
    ell = np.log(F)
    D = ell[0]-2*ell[1]+ell[2]
    if D >= -1e-12:
        return "CURVATURE",None
    if not (F[1] > F[0] and F[1] > F[2]):
        return "PEAK",None
    delta = (ell[0]-ell[2])/(2*D)
    if abs(delta) > .5+1e-12:
        return "VERTEX",None
    dh = d[1]+gaps[0]*delta
    var = -gaps[0]**2/D
    lp = ell[1]+(d[1]-dh)**2/(2*var)
    return "OK",np.array([dh,var,lp,d_f-dh])

yy,xx = np.mgrid[-2:3,-2:3]
saddle = xx**2-yy**2
checked("ML prevents cancellation",sml_valid(saddle)[2,2] == 4)
checked("threshold includes equality",sml_valid(saddle,threshold=4)[2,2] == 4)
checked("threshold is not subtraction",sml_valid(saddle,threshold=3)[2,2] == 4)
checked("threshold suppresses",sml_valid(saddle,threshold=5)[2,2] == 0)
checked("SML is local sum",sml_valid(saddle,radius=1)[2,2] == 36)
checked("constant unidentifiable",sml_valid(np.ones((5,5)),radius=1)[2,2] == 0)
checked("affine unidentifiable",sml_valid(xx+yy,radius=1)[2,2] == 0)
checked("full support boundary",np.isnan(sml_valid(saddle,radius=1)[1,1]))
checked("step has no normalization",sml_valid(saddle,step=2)[2,2] == 16)
status,params = peak3([0,1,2],np.exp([-1,0,-1]),10)
checked("peak width amplitude height",status == "OK" and np.allclose(params,[1,.5,0,9]))
_,asym = peak3([-1,0,1],np.exp([-2,0,-1]),10)
checked("asymmetric peak amplitude sign",np.allclose(asym,[1/6,1/3,1/24,10-1/6]))
_,scaled = peak3([8,10,12],7*np.exp([-2,0,-1]),20)
checked("normalized lookup restoration",np.allclose(scaled,[10+1/3,4/3,np.log(7)+1/24,10-1/3]))
checked("flat rejected",peak3([0,1,2],[1,1,1],10)[0] == "CURVATURE")
checked("nonpositive rejected",peak3([0,1,2],[0,1,.5],10)[0] == "LOG")
checked("nonuniform rejected",peak3([0,1,3],np.exp([-1,0,-4]),10)[0] == "POSITION")
checked("endpoint rejected",peak3([0,1,2],[3,2,1],10)[0] != "OK")
checked("F10 nonuniform counterexample",np.isclose(gaussian_fit(np.array([0.,1.,3.]),np.exp([-1.,0.,-4.])),.55))

def minmax(x,lo,hi,mode="PRINTED_1973"):
    x,lo,hi = np.asarray(x,float),np.asarray(lo,float),np.asarray(hi,float)
    if lo.ndim != 2 or hi.shape != lo.shape or x.shape != (lo.shape[1],) or len(lo) == 0:
        raise ValueError("shape")
    if not all(np.isfinite(v).all() for v in [x,lo,hi]) or (hi <= lo).any():
        raise ValueError("bounds")
    inside = ((x >= lo)&(x <= hi)).all(axis=1)
    if inside.any():
        ids = np.flatnonzero(inside)
        score = -np.log(hi[ids]-lo[ids]).sum(axis=1)
        return "INSIDE",set(ids[score == score.max()].tolist())
    score = (np.minimum(abs(x-hi),abs(x-lo))/(hi-lo)).sum(axis=1)
    if mode != "PRINTED_1973":
        return "UNRESOLVED",set()
    return "OUTSIDE_GE",set(np.flatnonzero(score == score.max()).tolist())

checked("overlap inverse volume",minmax([1],[[0],[.5]],[[2],[1.5]]) == ("INSIDE",{1}))
checked("closed boundary",minmax([0],[[0],[3]],[[1],[4]]) == ("INSIDE",{0}))
checked("outside literal GE",minmax([2.25],[[0],[3]],[[1],[4]]) == ("OUTSIDE_GE",{0}))
checked("outside intent unresolved",minmax([2.25],[[0],[3]],[[1],[4]],"INTENT") == ("UNRESOLVED",set()))
checked("equal volume set",minmax([1],[[0],[0]],[[2],[2]]) == ("INSIDE",{0,1}))
checked("outside tie set",minmax([2],[[0],[3]],[[1],[4]]) == ("OUTSIDE_GE",{0,1}))
checked("shared affine invariance",minmax([9.5],[[5],[11]],[[7],[13]]) == ("OUTSIDE_GE",{0}))
endpoint = np.minimum(abs(np.array([1.,3.])-2),abs(np.array([1.,3.])))/2
interval = np.maximum.reduce([np.zeros(2),-np.array([1.,3.]),np.array([1.,3.])-2])/2
checked("endpoint is not box distance",np.allclose(endpoint,[.5,.5]) and np.allclose(interval,[0,.5]))
try:
    minmax([1],[[1]],[[1]])
except ValueError:
    checked("zero width rejected",True)
else:
    checked("zero width rejected",False)

def pair_votes(K,scores):
    v = np.zeros(K,dtype=int)
    for (i,j),s in scores.items():
        if not np.isfinite(s) or s == 0:
            return None
        v[i if s > 0 else j] += 1
    return v

v = pair_votes(3,{(0,1):1.,(0,2):1.,(1,2):-1.})
checked("pair vote conservation",np.array_equal(v,[2,0,1]) and v.sum() == 3)
cycle = pair_votes(3,{(0,1):1.,(0,2):-1.,(1,2):1.})
checked("three way tie unresolved",np.array_equal(cycle,[1,1,1]))
checked("zero response undefined",pair_votes(2,{(0,1):0.}) is None)
responses = np.array([-3.,2.])
checked("equation five sign identity",np.array_equal((abs(responses)+responses)/(2*abs(responses)),[0,1]))
X = np.array([2.,3.])
W = np.array([5.,7.,11.])
checked("bias permutation must match",W@np.r_[1.,X] == np.r_[W[1:],W[0]]@np.r_[X,1.])
checked("noncovering density excluded",minmax([1],[[0],[3]],[[2],[3.25]]) == ("INSIDE",{0}))
checked("upper boundary included",minmax([1],[[0],[3]],[[1],[4]]) == ("INSIDE",{0}))
checked("degenerate coverage still defined",bool(((np.array([1.]) >= np.array([1.])) & (np.array([1.]) <= np.array([1.]))).all()))
two_scores = {(0,1):1.,(0,2):1.,(0,3):-1.,(1,2):1.,(1,3):1.,(2,3):1.}
two_votes = pair_votes(4,two_scores)
checked("two way maximum vote tie",np.array_equal(two_votes,[2,2,1,1]) and two_votes.sum() == 6)
two_candidates = np.flatnonzero(two_votes == two_votes.max())
m,n = two_candidates
checked("two way tie original separating plane",(int(m) if two_scores[(m,n)] >= 0 else int(n)) == 0)
print(f"SF94/HC73: {len(checks)} finite checks passed")
```

检查范围不包含：真实焦栈从采集到高度的集成、mask全组合、查表离散误差、硬件运动、全部LOO、完整 GLCM 原配置、Fukunaga 原训练、原实验精度或通用性能。上面的标量高斯值是代数题，不是生成的仿真测量；通过它们仅证明这些输入下的公式／分支一致。

## 来源、访问级与精确页码

访问级：**T** 已读正文／关键公式；**I** 已读扫描图；**B** 仅书目；**F** 未取得。页码是可复核定位，不把未取得正文的书目页号猜成方法页。原文单源事实均为 `[SINGLE-SOURCE]`，条件代数与代码检查是独立推导，不是第三方实验复现。

<a id="source-1"></a>

**[cite:1] Nayar、Nakagawa，Shape from Focus，1994，P0，T＋I。** [Columbia 作者 PDF](https://www1.cs.columbia.edu/CAVE/publications/pdfs/Nayar_PAMI94.pdf)，[DOI](https://doi.org/10.1109/34.308479)。8页PDF，序号1–8对应印刷824–831；首页上方和末页后部有其他论文，未混入本卡。正文提取只返回前段，后续通过扫描图核读：

| 精确定位 | 本卡采用内容 | 访问边界 |
|---|---|---|
| p.824，PDF1，摘要、§I、式(1) | 作者／卷期、粗糙纹理表面与多图局部深度任务 | T；不使用相邻上一篇参考文献 |
| p.825，PDF2，§II–III、Fig.2 | 固定光学／传感器移动物体，近峰倍率近似；`d_s=d_f-d_bar`，台位移采样 | T；轴符号依据该几何，不套用镜头距离标签 |
| p.826，PDF3，式(7)–(12)、Fig.4 | 线性卷积论述、二阶导、相消与 modified Laplacian | I；高斯解释是原文启发，不扩成普遍定理 |
| p.827，PDF4，式(13)(14)、§V、Fig.5 | step、ML绝对值、`≥T1`局部和、N、典型窗口；评价实验10×10、T1=7 | I；不将10×10强配为奇数窗口的N |
| p.827 §VI与脚注、式(15)；p.828 PDF5 式(16)–(19) | 最大模态相邻三点、局部高斯、峰位／宽度／幅度 | I；式(19)另放大确认除指数，非乘负指数 |
| p.828，式(20)–(22) | 邻分比、归一化坐标、二维查表与参数还原 | I；查表网格／边界／误差规格未给，未实施 |
| pp.829–830，PDF6–7，§VII、Fig.8 | 钢球采集与SML设置；系统和原实验边界 | I；未将图表误差／速度当本站结果 |
| p.831，PDF8，Fig.9续文与§VIII | 18幅／8μm实例、5×5中值、弱纹理和有限窗口误差讨论 | I；后部另一篇论文不采用 |

<a id="source-2"></a>

**[cite:2] Haralick、Shanmugam、Dinstein，Textural Features for Image Classification，1973，P0，T＋I。** [作者托管 PDF](https://haralick.org/journals/TexturalFeatures.pdf) 网页取得前部，开头标明“Reprinted by permission from IEEE”；本轮直接HTTPS取文件返回406后停止该路径，未更换身份／请求头绕过。[Stanford 同文扫描镜像](https://web.stanford.edu/class/biomedin260/lectures/8/Haralick%20-%20Texture%20Features.pdf) 为12页，序号1–12对应印刷610–621，原取证轮次正常HTTPS获取并渲染 pp.616–617，与临时证据来源中的两页扫描对读，不归因用户。文件645185字节，SHA-256 `3ccacd21d1fc39000bc7ea6d1a2b023b0d9831d55f7903ba57b083dfb9717dec`。作者版、课程转载及临时同文图片不是三份独立研究；本卡不再分发原扫描，本次文档同步未重新下载或执行卡内检查。

| 精确定位 | 本卡采用内容 | 已核／未核 |
|---|---|---|
| pp.610–612，PDF1–3 | 原文身份、表示到分类任务、两类决策摘要 | T；GLCM详细定义沿用HA01，未声称本次重核其全部附录 |
| p.616，PDF7右栏 Classification Algorithms、式(5) | 两两平面数、外引regression type、正响应计票 | T＋I；所引算法本体未取得 |
| p.617，PDF8左栏式(6)及前段 | 常数在前的Z、最大票、两类平票 `≥0`／`<0` | I；零响应全局协议和多类平票未见明确规则 |
| p.617，式(7) | 闭区间覆盖、仅覆盖类比较、正宽度倒数乘积的 `≥` | I并放大；等密度可多解，零宽未给约定 |
| p.617，式(8) | 全部盒外时，最近端点距离／宽度之和、比较全部K类、原印 `≥` | I并放大；不能确认作者原意、勘误或当年程序方向 |
| p.617，三组实验段 | 航空图LOO、33维，显微8维，卫星附加光谱与措辞差异 | T＋I；不复刻配置歧义／实验数值 |
| p.621，PDF12参考[19] | Fukunaga 1972书目，Academic Press | T；只支撑引用对象，不提供书第4章的算法页码 |

<a id="source-3"></a>

**[cite:3] 本仓库规范、台账、原卡及组件，源码静态证据。** 已读[学习规范](../algorithm_learning_standard.md)、[G12/G16台账](../evolution_coverage.md#去重缺口清单)、[对焦总览](../autofocus_evolution.md)、[F10](measurement_focus.md#f10-三点高斯峰位)、[F13](measurement_focus.md#f13-全扫描与数据源)、[HA01](haralick.md#ha01)、[focus_measures.py](../../traditional/autofocus/focus_measures.py#L30-L72)、[gaussian_fit](../../traditional/autofocus/focus_search.py#L13-L29)。本次实际调用现有 `gaussian_fit` 验证非等距反例，其余为本卡有限检查器，不复用原卡历史测试成功记录。

<a id="source-4"></a>

**[cite:4] Thomas M. Cover，Book Reviews，IEEE TIT 19(6)，1973，P1，T部分／B用途。** [Stanford 作者存档](https://www-isl.stanford.edu/people/cover/papers/transIT/0827cove.pdf)，已读 p.827 书目与后续书评正文，确认 Fukunaga 1972、Academic Press、369页的条目身份；**书评不是Fukunaga第4章，不支持具体回归训练步骤**。搜索到的[1990第二版镜像](https://lira.epac.to/DOCS-TECH/AI/Introduction%20to%20Statistical%20Pattern%20Recognition%202nd%20Edition.pdf)直接读取为空，仅有搜索片段，记F，不作1972原算法证据。未取得1972第4章、作者原实验代码或针对式(8)的正式勘误；这些缺口保留。

<a id="source-5"></a>

**[cite:5] Fukunaga，Introduction to Statistical Pattern Recognition，1972第一版，P0，T＋I仅限公开试读。** [PagePlace公开试读PDF](https://api.pageplace.de/preview/DT0400.9780323162784_A23652694/preview-9780323162784_A23652694.pdf) 为20页，正常HTTPS取得1041797字节，SHA-256 `1b68d82d2705ffdd33f059292d085cc9d8b02cdeb78dfbdd64b251712fee6a79`；[Elsevier第一版商品／目录页](https://shop.elsevier.com/books/instruction-to-statistical-pattern-recognition/fukunaga/978-0-12-269850-7) 核对出版年、版次与章节名，页面题名写成 *Instruction*，正文与原书题名为 *Introduction*。印本ISBN `9780122698507`，电子本ISBN `9780323162784`。商品页标386页，Cover书评标369页，未核总页数统计差别原因，不用它推算章节页。

| 原书定位／试读PDF序号 | 本轮实际读取 | 支撑范围／不得推断 |
|---|---|---|
| 无印刷页号的版权页，PDF5 | 1972 Academic Press版权、馆藏卡号72-75627；原图已读 | 第一版身份；不是1990版。版权标记不等于整书可再分发 |
| p.viii，PDF9，Contents | 第4章 *Linear Classifiers*；§4.1 p.90，§4.2 p.94，§4.3 p.100，§4.4 p.106，§4.5 p.111，Computer Projects p.119，Problems p.120 | 精确节起页已核；§4.3／§4.4正文、算法式号未读 |
| p.5，PDF20，§1.2 Chapter Outlines | 线性／分段线性为简单常见选择；第4章含Bayes、最小错误概率、最小均方误差等不同设计 | 不能凭“regression type”选择唯一目标／目标值或声称已取得分段线性训练 |
| pp.100–110及其引用定义 | **F；不在20页公开试读内** | 由目录划定的优先索取范围，不冒称算法原页已读；如依赖前节，再按正文交叉引用补页 |

临时证据来源中的版权页、p.viii、p.5图片与试读关键页一致，不归因用户；来源定位以上述公开URL为准，不将同一书页重复算独立佐证。只读取公开preview，没有修改URL猜取全文、扩展试读或访问受保护页。

<a id="source-6"></a>

**[cite:6] Elsevier／ScienceDirect第二版记录，P0，B，版本排除证据。** [DOI 10.1016/C2009-0-27872-X](https://doi.org/10.1016/C2009-0-27872-X) 实际解析为 **Second Edition，1990**，第4章为 *Parametric Classifiers*，pp.124–180，正文标“No access”。这些页不是1972第4章，不能回填 HC73。部分聚合检索结果把1972年份与“completely revised second edition”摘要混在一起，已排除；不从第二版目录借算法。1972第一版的ScienceDirect书目地址及Google Books ISBN入口本轮抓取为空，记F，不能据此断言原书不存在或已读全文。

<a id="source-7"></a>

**[cite:7] Haralick作者文章目录与本轮勘误检索记录，P0目录／非完备检索。** [作者Journal Articles](https://haralick.org/journals/journals.shtml) 中有1973文章与作者PDF链接；已读目录范围未见针对该文式(8)的更正条目。于2026-11-06检索标题配 `errata`、`erratum`、`corrections`，`Haralick min-max correction`，`4309314 correction`，以及 `site:haralick.org` 限定；未取得可归属于式(8)的正式勘误。`Fukunaga 1972 errata` 也未返回可采用的第4章勘误。IEEE论文DOI入口抓取为空，未核其完整更正关联库。检索结果中其他文章勘误、GLCM特征修正、min-max归一化均不充当本分类式勘误；未访问搜索返回的非授权下载站。结论仅为**本轮未取得**，不是“没有勘误”的全称判断。

## 本次验证记录与后续验收

扫描获取使用正常证书校验的系统HTTPS请求；原PDF在内存解析。pypdf 提取 Nayar 后页为空、提取扫描图因缺 `jbig2dec` 失败后，改用安装在系统临时目录的 `pypdfium2 5.14.0` 渲染关键页；PNG也只在系统临时目录，未作为仓库交付附件。未修改项目依赖。浏览器自动化接口当前不可调用，故没有声称浏览器截图或人工接管成功；实际证据为已读取的本地原页渲染图。

本次验证分为：原文逐式与扫描符号核对；文档检查器内存执行；Markdown两卡、十层、十问答、来源锚点、相对链接、围栏与UTF-8检查。**已执行：31项有限检查通过，文档结构与本地相对链接检查通过。**初次结构检查的命令在 PowerShell 管道中丢失非 ASCII 匹配字符，且未限定问答表范围；修正检查表达式后重跑，不将这次检查器失败计为算法失败或成功。该记录不意味着完整训练、真实SFF、原论文性能或现场系统通过。

后续 SF94 应实现完整支持SML、逐像素焦轴及设备标定，验证查表／流式优化等价，再以真实带高度参考的焦栈评价精度、边界和无效率；这不阻止按机制口径提出G12升级意见，但实现仍未完成。HC73 本轮交接以以下分层清单为准：训练正文仍缺，**G16保留部分**；不再把原作者全套退化修复、原程序或正式勘误一概列为“讲清原印机制”的必要条件。意见只在本文件登记，不代改台账、卡数或历史任务状态。

### G16 专项复核与最小销项证据清单

| 核验项 | 当前结果／可关闭的子问题 | 尚需的最小证据与销项边界 |
|---|---|---|
| 1973式(7)(8)印刷内容 | **字面复核完成**：p.617两式均≥，闭区间覆盖；式(7)只比较覆盖类，式(8)全部盒外时比较所有K类 | 不再重复索取同一符号截图；原印模式允许保留候选集合，不称最近盒 |
| 1972版本与章内位置 | **身份／取证位置完成**：版权页、p.viii、p.5；§4.3 p.100／§4.4 p.106 | 缺的是连续算法正文，不是更多书目链接。优先由合法持有人／馆藏服务提供pp.100–110及被引定义 |
| 分段线性训练 L1／L2／L4／L5 | **未闭合，实质阻塞**：章内多法，HC73只指向整章 | 原页须能核目标值、类／样本权重、由样本估计参数、权重求解及适用条件，并写清如何映射到`W_ij`、`Z=[1;X]`。若原章仍不能唯一选法，需作者说明／同期方法报告等消歧；不凭相似度默认lstsq。迭代法才补初值／更新／终止 |
| 外部式(8)作者意图 | **未闭合但与字面分开** | 声称更正原法须有指向p.617式(8)的正式勘误、可归属的作者说明或可追溯原程序证据之一；不要求三者齐备。若仅核原印机制，保留不确定性即可，不以无勘误阻止字面子项关闭 |
| 零宽／平票 L3／L5 | **必要性复核完成**：定义域／异常声明必须，ε与唯一化不是集合契约必需 | 本卡限定正宽、零响应未决、min-max全候选、多类票未决已足以表达边界；只有扩成全域单标签或逐样本历史复现时，才要求对应扩展／历史政策及测试 |
| 原实验与本地工程 | **未实现／未复现** | 取得实际向量、标签、划分、表示版本及分支触发记录后，才可核原实验输出；现场泛化另验。这些不由文档单元检查替代，也不与机制学习销项混为一项 |

**最终交接：HC73 L1–L5仍全部为“部分”；G16／07-01不销项，06-01不重新登记。**本次可关闭的是“≥是否OCR误读”“第一版取证位置未知”“是否必须发明零宽／唯一平票政策”三个子问题，不是整个原分类链。SF94正文、来源和原检查不变；既有共同验证记录保留为上一轮记录，本轮执行结果另列，避免将历史成功当新证据。

**本轮实际验证（2026-11-06，G16专项）：**使用仓库现有 `.venv/Scripts/python.exe -B` 将文档Python块提取到内存执行，**36项有限检查通过**，其中新增5项只涉及HC73：排除不覆盖类的密度竞争、上界覆盖、零宽覆盖测试仍有定义、两类最大票平局、按原两类平面破局。检查器输入是确定性代数例，不是实验性能或训练数据。UTF-8、围栏配对、来源锚点、相对文件链接、两卡L1–L5结构检查通过。与本轮修改前临时副本逐段比较：SF94正文、[cite:1]来源段及SFF函数／断言均一致；本轮只编辑本Markdown的HC73相关内容与交接证据，没有改源码、依赖、台账或导航。临时图片／试读提取仅在系统临时目录；复用已有NumPy与临时目录中的PDF渲染器，未安装依赖。系统`python`命令不可用后改用已有虚拟环境；作者PDF直接请求406、部分书目抓取为空均按失败记录，没有关闭证书校验、绕过访问控制或扩大试读权限。

### G16 续查：训练正文获取与实证状态（2026-11-06）

**结论：在当前已取得证据下，HC73 原分段线性训练不可唯一对应；Fukunaga 1972 第一版 pp.100–110 仍为 F／[INSUFFICIENT DATA]。**这不是断言原书没有具体训练方法，而是尚无这些正文可供核定。本轮只追加 HC73 取证与核验记录，不补写五层训练内容、不提供默认 `lstsq`，L1–L5 仍全部为“部分”；G16／07-01 不销项。前文31项、36项及“本轮”均保留为各自历史轮次记录，本续查的实际执行范围单列如下。

#### 公开来源与 TEMP 实查

| 路径／证据 | 本续查实际结果 | 能支持什么／不能支持什么 |
|---|---|---|
| [cite:5] PagePlace 第一版公开试读、Elsevier 第一版商品页 | 正常HTTPS返回200，PDF共20页、1041797字节，SHA-256与[cite:5]完全一致；重新渲染并读取版权页、p.viii、末页p.5；商品页可读 | 1972身份、§4.3 p.100／§4.4 p.106／§4.5 p.111再次确认；试读止于p.5，不含pp.100–110；未修改preview地址或扩展试读范围 |
| [cite:2] Haralick 作者公开PDF、Stanford课程同文PDF | 作者入口的网页提取只取得pp.610–612前部；Stanford正常HTTPS返回200，12页、645185字节，SHA-256与[cite:2]完全一致；本地逐页提取并重读pp.616–617图及p.621参考文献文字 | 本轮全文证据归于Stanford同文扫描，不把作者入口前部提取称为全文；同文重复获取不增加独立研究数 |
| 系统 TEMP 既有证据 | 获取前递归列出可访问的`.pdf`文件，列出的7份PDF均非Fukunaga／Haralick；另找到并读取既有`hc73-page-7.png`与`hc73-page-8-formula.png`，`sff-haralick-pdfium`为渲染依赖目录 | 没有找到可复用的1972训练正文PDF；“已有渲染器／旧原论文截图”不等于“已有原书全文”。结论仅限本次可访问的TEMP枚举范围 |
| 作者机构及公开检索 | 使用书名＋1972＋PDF、`site:purdue.edu`、第一版ISBN、两节标题及pp.100–110检索；未取得合法公开的训练原页 | 无命中或只有书目不是训练证据；搜索返回的第二版片段、非授权下载站未用于补法，未访问非授权下载站 |
| [cite:8] UConn机构书目 | 实际读到1972、ISBN、馆藏号和页数，同时读到第二版摘要 | 机构域名不消除记录内部版本冲突；仅作书目与污染警示，不作为训练正文 |
| [cite:9] 第一版ScienceDirect及Google Books入口 | 前者返回“页面不存在或已移动”正文；后者提取为空 | 均未取得算法页；不把网页失败解释成原书不存在，也不以1990 DOI替换第一版 |

获取使用默认证书校验，未登录、借阅、购买、处理验证码或更换身份绕过限制。PDF在内存解析；逐页文本和必要原页渲染仅写入系统TEMP的`hc73-g16-followup`，不交付扫描副本。浏览器自动化接口未能调用，故没有浏览器交互／截图验证声明；证据来自正常HTTPS、网页提取及本地原页渲染。

#### Haralick 对应规则与训练证据门槛

本续查再次确认：p.616仅写用[19]第4章的“regression type algorithm”求两两`W_ij`，未在该段给节号、目标编码或求解式；p.621将[19]明确列为Fukunaga 1972。p.617式(6)为常数在前的`Z=[1;X]`，最大票及两类平票平面比较仍按前文保留；式(7)只在闭区间覆盖类中比密度，式(8)全部盒外才比较所有K类，原印仍为**≥**。这些是已重读的一手字面事实 **[SINGLE-SOURCE]**，不是新取得的训练法或作者勘误。[cite:2](#source-2)

**最小补证不是再找一份目录。**需合法取得第一版pp.100–110连续正文及其中实际引用的定义，核出目标／目标值、类或样本权重、分布量如何由训练样本估计、参数求解与适用条件，再说明如何对应两类样本、`W_ij`方向及`Z=[1;X]`。若正文仍不能唯一选出HC73采用的变体，才另需可归属的作者说明或同期方法报告消歧。秩亏、正则化和伪逆不得默认补入；初值、更新与终止仅在核得迭代法时补。拿到书页不自动意味着已经证明HC73选法。

只有上述具体训练与对应关系取得后，才补L1步骤、L2机制、L3参数／适用域、L4推导、L5实现及针对性检查。式(8)作者原意仍与原印规则分开；本轮未新取得勘误／原程序，也没有重新开展完整勘误检索，不将历史检索记录冒充新增发现。

<a id="source-8"></a>

**[cite:8] University of Connecticut Library，Source record，机构书目，B；内部版本冲突。**[公开MARC记录](https://search.lib.uconn.edu/discovery/sourceRecord?docId=alma994186103502432&recordOwner=01UCT_STORRS&vid=01UCT_STORRS%3A01UCT)。本次读取字段020为`0122698509`／`9780122698507`，260为Academic Press、1972，300为`xiii, 369 pages`，010为`72075627`；但520摘要开头写“This revised second edition”。按字段保留冲突，不能把摘要当1972原文，不能由馆藏页推定已获全文。版本依据仍以[cite:5]原版权页为准。

<a id="source-9"></a>

**[cite:9] 第一版公开入口续查，F。**[ScienceDirect第一版ISBN入口](https://www.sciencedirect.com/book/9780122698507/introduction-to-statistical-pattern-recognition)本次返回“The page you were looking for does not exist or was moved”；[Google Books第一版ISBN入口](https://books.google.com/books?vid=ISBN9780122698507)本次提取为空。这里只记录访问结果，不支持任何训练公式或“无合法馆藏”的断言。

**本续查实际核验：**现有`.venv/Scripts/python.exe -B`重新内存执行未修改的文档Python块，36项有限检查通过；无新增算法测试，未执行Fukunaga训练、真实分类实验或性能验收。UTF-8、围栏、来源锚点、相对文件链接与两卡五层结构检查通过。相对获取前的临时基线，原文件63878字节全部保留，只在末尾追加本HC73续查节，因而SF94正文、来源、函数及断言均不变。本轮编辑操作仅针对本文件，前后`git status --short`清单一致；但全工作区112文件的初始哈希表未跨命令保留，该项核验报错，**不宣称完成全工作区哈希一致性验证**，也不以Git清单替代未跟踪文件的内容比较。训练原页未得／原法不可唯一对应、五层全部分、原训练器未实现的状态不因文档检查通过而升级。
