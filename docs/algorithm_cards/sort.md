# SORT 算法卡

[学习规范](../algorithm_learning_standard.md) · [跟踪总览 E06](../tracking_guidance_evolution.md#36-e06sort检测质量与简单时序组件的组合) · [覆盖台账 G45／12-06](../evolution_coverage.md#12-跟踪运动视觉引导) · [T01 Kalman 组件](remaining_domains.md#t01-线性-kalman-常速状态估计) · [H01 指派内核](hungarian.md#h01)

## 范围、版本与覆盖状态

核对日期：**2026-11-06**；本仓库基准：HEAD `918a03e` 加当前工作树。唯一卡键 **`sort.md:SO01`**，承接 G45／12-06；只讲具体 SORT 检测驱动多目标跟踪系统，不以通用 Kalman、单次 Hungarian 或四维点状态卡替代。

方法来源为 Alex Bewley、Zongyuan Ge、Lionel Ott、Fabio Ramos、Ben Upcroft 的 *Simple Online and Realtime Tracking*，ICIP **2016**，DOI `10.1109/ICIP.2016.7533003`。本次实际读取 arXiv **v1，2016-02-02** 的方法 §3.1–3.4；v2 修订于 **2017-07-07**，没有逐版比对。实现基线固定为作者 `abewley/sort` 提交 **`2236dff5019565958b84df7d871d41cc1db58ac7`**（提交日期 2023-11-28）的 `sort.py`，不是宣称找到了 2016 发表时的软件快照。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

| 字段 | 覆盖与限定 |
|---|---|
| 演进定位 | 在线 tracking-by-detection：每帧框检测 → 七维运动估计 → 预测框 IoU 指派 → 生命周期；不使用 Re-ID 外观网络 |
| L1 | 已核验：指定代码版本的调用顺序、输入输出与计数契约；有完整系统伪代码，不表示运行了作者程序 |
| L2 | 已核验：框变换、速度不确定度、预测关联、门控和生命周期的作用及反例 |
| L3 | 部分：量纲、调参顺序、失效诊断明确；没有真实视频的噪声标定、身份指标或性能剖析 |
| L4 | 已核验（限定模型代数）：F/H、Q/R/P、变换与误差传播、Kalman 更新、IoU 指派目标及复杂度；不是身份正确性或闭环误差保证 |
| L5 | 部分：固定源码映射、数值风险、有限算术／分支检查；未运行完整作者依赖栈、检测器、视频或 MOT 评测 |
| 面试自测 | 贯穿五层的 5 问，含答案要点、条件、易错点与证据入口 |
| 实现状态 | **未实现**：本仓库没有 SORT 跟踪模块；外部源码阅读、本文伪代码与局部检查不等于产品实现 |
| G45 销项口径 | “缺少 SORT 专卡及框状态／IoU／出生确认删除说明”的内容缺口由 SO01 承接，可供后续台账转“已有”；仅新增本文件，台账、索引与旧总览未同步，不能声称它们已销账 |

证据按论文、作者代码、依赖源码分别标注；同作者论文与仓库不算独立复现。SORT 特有细节按 `[SINGLE-SOURCE]` 的版本事实解读；依赖初始化可交叉核对，但不存在三套独立实验支持通用性能结论。不搬用摘要的 Hz 或精度增益作为本机测量。

<a id="so01"></a>
## SO01 SORT：框状态、预测关联与轨迹生命周期

### 任务、契约与演进分支

输入是每次调用对应一帧的浮点数组 `dets.shape=(N,5)`，列为 `[x1,y1,x2,y2,score]`；调用者负责检测、类别筛选、分数过滤和去重，坐标必须属于同一原图像素系。有效几何域要求四坐标有限且 `x2>x1, y2>y1`。这里是连续边界约定，宽高为差值，**没有 inclusive 像素坐标的 `+1`**。分数列不是状态、不是测量方差，也不是输出的最后一列。[cite:3](#source-3)

输出是 `shape=(K,5)` 的 `[x1,y1,x2,y2,track_id]`，或明确的 `(0,5)` 空数组。框由当前滤波状态恢复，通常不是原始检测框；只返回本帧已获检测支持且满足输出规则的轨迹，不输出仅预测的失配框。内部轨迹数、检测数和输出行数可以不同。输出没有类别、协方差、速度或可信度字段，ID 与浮点框拼接后通常也处于浮点数组中。

前序瓶颈 → 改变维度 → 新代价 → 遗留问题：多目标逐帧检测缺乏身份连续性，复杂多假设／重识别处理又带来额外成本；SORT 用简单常速框状态与全局一对一框重叠指派，突出检测质量和在线效率；代价是检测器依赖、噪声与生命周期参数，以及候选矩阵；仍不处理显式长遮挡重识别、复杂相机运动、同形目标交叉与机器人闭环。论文的短时遮挡讨论只是 IoU 对尺度的偏好，不是遮挡后找回身份的保证。[cite:2](#source-2)

### L1 步骤：逐帧完整链

#### 状态和框的接口

检测框先变换为观测列向量，轨迹内部采用七维列向量：

\[
z=[u,v,s,r]^T\in\mathbb R^{4\times1},\qquad
x=[u,v,s,r,\dot u,\dot v,\dot s]^T\in\mathbb R^{7\times1}.
\]

`u,v` 是框中心，`s` 是面积而非宽度或边长，`r=w/h` 是宽高比。不存在 `dot_r`，也不是 `[cx,cy,a,h,vx,vy,va,vh]` 八维状态。作者模型的每步时间为 **1 次帧调用**；速度单位分别为 px/frame、px/frame、px²/frame，不是 m/s。[cite:2](#source-2)[cite:3](#source-3)

每帧必须调用一次 `Sort.update`，包括无检测帧 `np.empty((0,5))`。跳过空帧等于让预测时钟、warmup 和删除计数一起停走；把空帧伪装成全零框则会错误出生。流结束时停止调用；算法没有离线回溯、跨全视频优化或自动补齐最后一帧之后的时间。

#### 固定代码的系统伪代码

下列为语义展开，不是新增可运行跟踪模块。`KF_PREDICT/KF_UPDATE` 的矩阵在 L4 给出；`LAP_MIN` 是矩形固定基数线性指派，机制见 H01，不偷换成允许任意拒配的另一目标。除明确标注异常行为外，以下以有限、正宽高输入及有限预测为工作域。[cite:3](#source-3)

```text
BIRTH(box):
    x = [BOX_TO_Z(box), 0, 0, 0]
    F, H, P, Q, R = L4 中的固定矩阵
    id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    age = hits = hit_streak = time_since_update = 0
    history = []
    return track

PREDICT(track):
    if x[s] + x[dot_s] <= 0:
        x[dot_s] *= 0.0
    KF_PREDICT(x, P, F, Q)
    age += 1
    if time_since_update > 0:
        hit_streak = 0
    time_since_update += 1
    history.append(STATE_TO_BOX(x))
    return history[-1]

ASSOCIATE(detections, predicted_tracks, tau):
    if track_count == 0:
        return empty_pairs, all_detection_indices, empty_track_indices
    W = pairwise_IoU(detections, predicted_tracks)
    if min(W.shape) == 0:
        candidates = empty_pairs
    else:
        A = (W > tau)
        if max(row_sums(A)) == 1 and max(column_sums(A)) == 1:
            candidates = indices_where(A)
        else:
            candidates = LAP_MIN(-W)
    unmatched_dets = detection_indices_not_in(candidates)
    unmatched_tracks = track_indices_not_in(candidates)
    matches = []
    for (d, t) in candidates:
        if W[d,t] < tau:
            unmatched_dets.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((d,t))
    return matches, unmatched_dets, unmatched_tracks

UPDATE_FRAME(dets):
    frame_count += 1
    predict every existing track exactly once
    compress NaN/Inf predicted rows; remove NaN tracks in reverse order
    matches, unmatched_dets, unmatched_tracks = ASSOCIATE(dets, predictions, tau)
    for (d,t) in matches:
        track[t].time_since_update = 0
        track[t].history = []
        track[t].hits += 1
        track[t].hit_streak += 1
        KF_UPDATE(track[t], BOX_TO_Z(dets[d]))
    for d in unmatched_dets:
        tracks.append(BIRTH(dets[d]))
    outputs = []
    for track in reverse(tracks):
        box = STATE_TO_BOX(track.x)
        if time_since_update < 1 and
           (hit_streak >= min_hits or frame_count <= min_hits):
            outputs.append([box, id + 1])
        if time_since_update > max_age:
            remove track
    return outputs as shape (K,5), including (0,5)
```

`UPDATE_FRAME` 特意保留了源码的 NaN/Inf 不一致，不把它描述成可靠防护，风险见 L5。作者无轨迹早退分支的 `unmatched_track_indices` 实际形状是 `(0,5)`，不是通常期望的 `(0,)`；主流程不使用这个数组，伪代码仅表达它为空。

#### 出生、确认、删除与 ID 的准确口径

| 字段 | 出生值及改变时点 | 常见错误解释 |
|---|---|---|
| `frame_count` | 每个 `Sort` 实例从0开始，每次帧调用先加1 | 不是每条轨迹年龄，也不由图像时间戳计算 |
| `age` | 出生0，每次 `predict` 加1 | 出生帧没有对新轨迹再预测，因此不从1开始 |
| `hits` | 出生0，每次成功 `update` 加1；漏检不清零 | 出生检测不计一次 hit；累计 hits 不参与输出判断 |
| `hit_streak` | 出生0，匹配更新加1；预测开始时若旧 `time_since_update>0` 才清零 | 不是终身确认标志；第一次漏检帧末可能仍保留旧值 |
| `time_since_update` | 出生／匹配更新为0；预测加1 | 不匹配不需额外再加1，否则删除提前 |
| `id`／类变量 `count` | 出生取当前 count，再递增；输出 `id+1` | 创建新的 `Sort` 实例不会重置类变量；不是行号或真实身份认证 |

源码没有显式 `Tentative/Confirmed/Deleted` 枚举，也没有一旦确认永远保留的布尔状态。所谓“确认”在此只能指当帧输出谓词。默认 `min_hits=3`，全局 `frame_count<=3` 的 **warmup** 允许出生轨迹直接输出；warmup 后出生的轨迹要经历出生之后连续3次匹配更新，通常在第4次连续检测时首次输出。第一次漏检不输出，下一帧预测清零 streak；即使找回原 ID，也可能重新等到连续命中门槛才输出。[cite:3](#source-3)

默认 `max_age=1`：第一次空帧使 `time_since_update=1`，暂留但不输出；第二次连续空帧使其为2，在关联之后删除。若第二步预测时计数虽已超过 max_age，却在**同帧关联成功**，更新会归零并保留原 ID；不能在关联前先删过龄轨迹。删除后再出现只能新生，SORT 不进行旧 ID 的 Re-ID 恢复。对非负整数 A，最多有 A 个连续未匹配帧结束时仍保留，在第 A+1 个未匹配帧末删除。

一个默认参数的有限调用推演（假定所有“匹配”均通过关联，ID 从内部0开始）如下。它是计数分支算例，不是实拍轨迹：

| 帧 | 输入事件 | age / hits / hit_streak / time_since_update（帧末） | 输出 |
|---|---|---|---|
| 1 | 出生 | 0 / 0 / 0 / 0 | ID 1，warmup |
| 2 | 匹配 | 1 / 1 / 1 / 0 | ID 1，warmup |
| 3 | 匹配 | 2 / 2 / 2 / 0 | ID 1，warmup |
| 4 | 匹配 | 3 / 3 / 3 / 0 | ID 1，达到门槛 |
| 5 | 空帧 | 4 / 3 / 3 / 1 | 空；内部尚未删除 |
| 6 | 匹配 | 5 / 4 / 1 / 0 | 空；同一 ID 留存但 streak 重建 |
| 7 | 匹配 | 6 / 5 / 2 / 0 | 空 |
| 8 | 匹配 | 7 / 6 / 3 / 0 | ID 1 |

### L2 机制：为何不能只拼两个组件

**框表示。** 中心用来平移，面积允许物体接近或远离，比例在预测模型中不变；七维比点位置模型多了明确的尺寸动态。比例没有速度不等于比例永远锁死，测量更新仍会改变 r，且 Q 的 r 项非零。检测是 `[x1,y1,x2,y2]` 而观测是 `[u,v,s,r]`，不变换就用 H 相乘会产生量纲错误。

**零速度与大初始速度方差。** 一张检测框只能观测位置和尺寸，不能观测速度；零速度是初始化均值，不是“知道目标静止”。大 P 使接下来的观测可快速校正速度。若把速度方差设为零且又不给相关过程不确定性，模型可能长时间不能适应运动；反过来巨大噪声也可能放大抖动。[cite:2](#source-2)[cite:3](#source-3)

**先预测，再算 IoU，再更新。** 检测发生在当前帧，必须与当前时刻预测框比较；用上一帧框关联会惩罚真实位移，用先验配对前就更新则存在循环依赖。预测结果只作关联和状态先验，不能当成新的检测回灌，否则会伪造命中证据。P 参与滤波增益，但指定源码的关联只看预测框 IoU，**没有 Mahalanobis 门、置信椭圆或类别约束**。

**一对一不是全时序最优。** IoU 矩阵行是检测、列是轨迹；线性指派避免两检测同时更新一轨迹，但目标只覆盖当前帧。交叉时若几何不可区分，即使当前代价严格最优也可能换 ID。未匹配检测全部新生，包括因竞争输掉候选、或后门控被拒的检测，不只包括“与所有轨迹都完全不重叠”的框。重复检测可以制造重复 ID，源码不做 NMS。

**出生确认与过期。** 输出筛选减少孤立假阳性，但有启动等待和重获等待；过期防止无检测的模型无限外推。增加 max_age 只增加可匹配的旧预测，不会让丢失目标自动出现在返回值里；增加 min_hits 还会延长该实现的全局 warmup，不能简单理解成“调大永远更严格”。[cite:3](#source-3)

### L3 工程权衡：参数、诊断与论文差异

#### 参数单位与调参顺序

| 项目 | 固定代码值／单位 | 调整依据与风险 |
|---|---|---|
| `iou_threshold` | 默认0.3，无量纲 | 提高通常减少弱重叠候选、增加碎轨；降低增加误配；源码含分支切换，完整最终匹配集合未必单调 |
| `min_hits` | 默认3，连续匹配更新次数；同时是全局 warmup 帧数 | 看假轨与首次输出延迟；不是总检测数，不是秒 |
| `max_age` | 默认1，未更新的帧调用计数 | 看漏检间隔和位移；增大增加幽灵候选、漂移和错接机会，没有 Re-ID 保证 |
| F 的时间步 | 固定1 frame | 不规则采样须另实现真实 dt 的 F/Q；只把 max_age 改成秒但不改预测不成立 |
| Q/R/P | L4 所列经验数值，混合坐标量纲 | 不是可跨分辨率、目标大小照搬的标定值；先修输入尺度，再以残差与真值验证 |
| 检测分数／类别 | `Sort.update` 不过滤、不加权 | 调用者先完成过滤；不能认为传 score=0 就会被拒绝 |

建议顺序：核对时间顺序与帧率 → 图像／ROI 坐标和合法框 → 固定检测器、类别与去重策略 → 分析漏检、交叉和 IoU 矩阵 → 校准噪声与关联阈值 → 调生命周期 → 最后剖析求解器。必须固定检测输入做消融，否则不能把检测器进步归因于 SORT。参数不提供通用最优范围；接口防护建议限制 `max_age,min_hits` 为非负整数、tau 为有限 `[0,1]`，这些检查不是作者已实现功能。

| 症状 | 先保留的证据 | 对照与退出条件 |
|---|---|---|
| 目标运动越快越断轨 | 原帧号、dt、预测框、实际位移、IoU | 查跳帧／相机运动，先别仅降门限；大位移无重叠需另评估运动／外观关联 |
| 重获后内部 ID 在但输出消失 | frame_count、hit_streak、time_since_update | 区分状态保留和输出门槛，不误判为滤波器丢了状态 |
| 交叉换 ID 或重复 ID | 原始检测、重复框、候选矩阵、被后门控拒绝的对 | 固定检测评估身份指标；几何歧义须加信息而非换更快求解器 |
| 框面积爆炸／变成 NaN | s、r、dot_s、Q/R/P、输入宽高 | 回溯首个非法观测／状态，隔离轨迹；不能以零填补继续驱动后续系统 |
| 长期保留旧候选 | 是否每帧调用、实际漏检数、删除时点 | 查空帧被跳过及严格大于条件，不能只调小 max_age 掩盖时钟错误 |

#### 2016 论文与固定作者代码的差异

| 主题 | 论文 v1 §3 | 指定 `sort.py` 的实际语义 |
|---|---|---|
| 检测输入 | Faster R-CNN，person 类且概率大于50%的实验设置 | 独立接收检测数组，不执行检测网络、类别／分数筛选；demo 将 MOT xywh 转 xyxy |
| 常速框估计 | 七维状态、比例常量、未观测速度的大方差 | F 的步长1；Q/R/P 有具体经验系数及面积速度防护，论文未逐项列这些矩阵数值 |
| 最优关联与阈值 | Hungarian，加最小 IoU 拒绝低重叠配对 | 无歧义 `W>tau` 快路；否则全矩阵 `-W` 指派后用 `<tau` 拒配，不重求解 |
| 求解器名称 | Hungarian | 优先 `lap.lapjv(extend_cost=True)`；ImportError 时 SciPy `linear_sum_assignment`，不是手写 Kuhn1955；当前 SciPy 文档说明 modified Jonker–Volgenant |
| 确认 | probationary period，积累检测证据 | 没有显式永久 confirmed；用 streak＋全局 warmup 的输出谓词，出生 hits=0 |
| 删除 | 未检测达 `T_Lost` 帧终止，实验设1 | `time_since_update > max_age` 且在关联后执行；默认1保留第1次未匹配帧，不能无差别把边界写成 `>=1` |
| 出生与 ID | 未跟踪对象新建唯一身份；删除后再现新身份 | 任意 unmatched detection 均出生；内部类变量 count，输出 `id+1`，逆序遍历返回 |

差异不意味着论文结论被证伪；2016 方法、2017 文稿修订、2023 仓库快照、当下依赖实现是不同证据层。不能用 CLI 文案“minimum associated detections”替代真实 streak 条件，也不能把论文简述擅自补成 Deep SORT 的确认状态机。[cite:2](#source-2)[cite:3](#source-3)[cite:6](#source-6)

### L4 数学推导：SORT 的具体模型

#### 框变换、逆变换与噪声坐标

记左上、右下边界为 \((a,b,c,d)\)，有

\[
w=c-a>0,\quad h=d-b>0,\quad
u=(a+c)/2,\quad v=(b+d)/2,\quad s=wh,\quad r=w/h.
\]

由 \(sr=w^2\)，正宽高域内唯一取正根，恢复

\[
w=\sqrt{sr},\quad h=s/w,\qquad
(a,b,c,d)=(u-w/2,v-h/2,u+w/2,v+h/2).
\]

这解释了为什么不能直接用 s 充当宽度，以及为什么 s≤0 或 r≤0 没有合法框。模型仅在变换后的观测空间线性；角点到面积／比例是非线性函数，即使角点噪声高斯，也不意味着四维 z 噪声严格高斯或各分量独立。

对角点观测协方差 \(\Sigma_b\) 作局部一阶传播：

\[
J=\frac{\partial(u,v,s,r)}{\partial(a,b,c,d)}=
\begin{bmatrix}
1/2&0&1/2&0\\
0&1/2&0&1/2\\
-h&-w&h&w\\
-1/h&w/h^2&1/h&-w/h^2
\end{bmatrix},\qquad R_z\approx J\Sigma_bJ^T.
\]

这是本文的局部误差传播推导，**不是作者代码自适应 R**。高度趋零时比例的导数发散；对同样角点噪声，大框与小框面积噪声不同，因此常量对角 R 只是简化。若图像等比缩放 k，状态转换矩阵为 \(D=\operatorname{diag}(k,k,k^2,1,k,k,k^2)\)，则 P/Q 应按 \(DPD^T,DQD^T\) 变换，观测 R 按前四维对应矩阵变换；面积方差随 k⁴ 而非 k² 缩放。

#### F、H 与可观测性

固定步长的模型为 \(x_t=Fx_{t-1}+q_t\)，\(z_t=Hx_t+e_t\)，其中

\[
F=\begin{bmatrix}
1&0&0&0&1&0&0\\
0&1&0&0&0&1&0\\
0&0&1&0&0&0&1\\
0&0&0&1&0&0&0\\
0&0&0&0&1&0&0\\
0&0&0&0&0&1&0\\
0&0&0&0&0&0&1
\end{bmatrix},\qquad H=[I_4\;0_{4\times3}].
\]

一次观测只给前四维，速度不可直接观测；两时刻的 \([H;HF]\) 包含三个位置差分方向，在非零已知时间间隔下秩为7，因此理想模型可以从连续测量辨识三速度。它不证明存在错误身份关联时还能辨识真实物体运动。推广到 dt 时须把 F 的三个位置—速度系数改成 dt，并按运动噪声模型重新离散化 Q；本文不把一般白加速度 Q 冒充原代码的对角 Q。

#### Q、R、P 的具体数值

FilterPy **1.4.5** 初始化 `x=0, P=I7, Q=I7, R=I4`；SORT 随后做乘法，得到：[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

\[
P_0=\operatorname{diag}(10,10,10,10,10000,10000,10000),
\]
\[
R=\operatorname{diag}(1,1,10,10),\qquad
Q=\operatorname{diag}(1,1,1,1,0.01,0.01,0.0001).
\]

`Q[-1,-1] *= 0.01` 后又对整个速度块 `Q[4:,4:] *= 0.01`，因此 **dot_s 的 Q 是 0.0001，不是0.01**。P 速度块先乘1000，再对全矩阵乘10，故为10000，不是1000。Q、P 的对角元单位是各状态量单位的平方，R 是观测单位的平方；“R 的后两项为10”不意味着尺寸误差统一为10像素。r 的预测均值不变，但 Q 的 r 项为1，协方差增长且观测可校正比例。

#### 条件滤波及统计边界

在零均值、白过程／观测噪声、互不相关，且给定正确关联的模型下，\(\operatorname{Cov}(q)=Q,\operatorname{Cov}(e)=R\)，预测为

\[
x_t^-=Fx_{t-1}^+,\quad P_t^-=FP_{t-1}^+F^T+Q.
\]

设创新 \(y=z-Hx^-\)，\(S=HP^-H^T+R\)，令线性校正 \(x^+=x^-+Ky\)。其误差协方差为

\[
P^+(K)=(I-KH)P^-(I-KH)^T+KRK^T.
\]

对 \(\operatorname{tr}P^+\) 关于 K 求导得 \(2KS-2P^-H^T=0\)，当 S 可逆时

\[
K=P^-H^TS^{-1},\qquad x^+=x^-+Ky.
\]

FilterPy 1.4.5 用上述 Joseph 形式更新 P；默认计算 `inv(S)`，并非默认 Cholesky 求解。无匹配时只预测，不校正、不累加 hits。线性高斯且关联已知时可作条件后验解释；一般情况下不能把 IoU 配对的不确定性当成被 P 完整建模。非负面积速度防护是分段启发式，既不投影协方差也非约束 Kalman 的推导结果。本文**未提供通用理论误差上界**，P 不是实际框误差或身份错误的确定性上界。[cite:5](#source-5)

有限算术检查使用明确给定的输入框 `[0,0,10,20]`：z=`[5,10,200,0.5]`，逆变换回原框。零速度第一次预测框不动，但中心—速度子块从 `diag(10,10000)` 变为 `[[10011,10000],[10000,10000.01]]`。若下一次同尺寸框水平移动1，中心创新为1，得到

\[
S_u=10012,\quad K_u=10011/10012,\quad
K_{\dot u}=10000/10012,\quad
u^+=5+10011/10012,\quad\dot u^+=10000/10012.
\]

大初始速度方差使一次位移迅速形成速度估计；这只是代数输入，不是视频精度实验。

#### IoU、指派及后门控的非等价性

对有效框 D、T，交集宽高分别取 \(\max(0,\min(x_2)-\max(x_1))\) 与对应 y 式，交面积为 I；\(W_{dt}=I/(A_D+A_T-I)\)。固定基数 \(k=\min(N,M)\) 的候选目标是

\[
\max_X\sum_{d,t}W_{dt}X_{dt},\quad
X_{dt}\in\{0,1\},\quad \sum_tX_{dt}\le1,\quad
\sum_dX_{dt}\le1,\quad\sum_{d,t}X_{dt}=k.
\]

代码最小化 `-W`；在固定 k 下与最小化 `1-W` 等价。若允许可变匹配数，这个常数项随匹配数改变，不能直接套等价性。代码快路取 `W>tau` 图上无行列竞争的边，不是完整固定基数目标的普适求解器。后门控只删 `W<tau` 的候选，不做重新分配。[cite:3](#source-3)

一个由合法框直接构成的反例：T1=`[0,0,10,10]`，T2=`[5,0,15,10]`；D1=T1，D2=`[-5,0,5,10]`，共同图像原点平移不改变结论。tau=0.3 时

\[
W=\begin{bmatrix}1&1/3\\1/3&0\end{bmatrix}.
\]

第一行、第一列均有竞争，进入 LAP；对角总收益1高于交叉2/3，故先选对角。后门控删掉0，只剩 D1→T1；若先禁掉低于门限的边，再求最大合法基数，则能选两条1/3的交叉边。二者的优化目标不同，**不能用 H01 推荐的先门控拒配方案偷偷替换 SORT 后门控而仍声称逐版等价**。原主流程还会让被拒的 D2 新出生、T2 留作未匹配候选。

等于阈值还存在分支依赖：tau=0.5，仅一个 IoU=0.5 候选时，快路图全0，走 LAP 后 `<0.5` 为假，接受；若加入一个互不交叉的 IoU=1 配对，矩阵 `diag(0.5,1)` 的快路图只有第二条边且行列最大度均为1，直接匹配第二条，第一条0.5不会被补配。它不是统一的 `>=tau` 门图。tau=0 甚至可使 LAP 的0重叠候选通过后门控，故需专门测边界。

复杂度：M 条轨迹的固定7维滤波每帧 O(M)，N×M IoU 时间／内存 O(NM)；稠密方阵指派常用三次量级描述，矩形和具体 LAP/JV 后端需分别测量，不把该界归为本机耗时。当前源码寻找 unmatched 时逐对象做数组成员查询，也有 O((N+M)k) 扫描；检测网络耗时不在这些跟踪组件计数中。无检测时不构造有元素的矩阵、不求解 LAP，只有旧轨迹预测和过期处理。

### L5 实现：固定源码、边界与有限检查

#### 源码对应与数据布局

以下行号全部对应固定提交，不指向漂移的 master：[cite:3](#source-3)

| 源码入口 | 本卡核对内容 |
|---|---|
| [`sort.py:L36-L45`](https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py#L36-L45) | 优先 lap，导入失败才回退 SciPy；非 ImportError 的求解错误不自动回退 |
| [`sort.py:L47-L91`](https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py#L47-L91) | IoU 广播轴、无 `+1`、框／四维观测变换 |
| [`sort.py:L94-L152`](https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py#L94-L152) | F/H、Q/R/P、计数、面积速度防护与 history |
| [`sort.py:L154-L196`](https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py#L154-L196) | 空轨迹早退、快路、LAP、unmatched 与后门控 |
| [`sort.py:L199-L253`](https://github.com/abewley/sort/blob/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py#L199-L253) | 逐帧调用、非法预测压缩、出生、逆序输出、删除及输出 ID |

作者 requirements 固定 `filterpy==1.4.5`、`scikit-image==0.17.2`、`lap==0.4.0`，没有锁全体传递依赖或 SciPy 回退版本。本仓库 requirements 中虽列有 NumPy/SciPy，也不证明当前终端安装了这些包，更不证明已接入 SORT。外部文件含 GPL-3.0-or-later 声明，部署／拷贝前需按其许可证处理；本文是文档分析，没有把作者源码移入产品模块。[cite:3](#source-3)[cite:4](#source-4)

#### 数值与语义边界

| 边界 | 固定源码行为／风险 | 复现或加固时的要求 |
|---|---|---|
| 零高、负宽高、NaN/Inf 检测 | 无统一输入守卫；r 除零、无效面积、IoU 分母非正可能传播 | 在调用边界拒绝并记录原因；不把零框当空帧 |
| s 或 r 非正 | `sqrt(s*r)` 与 `s/w` 可能 NaN、零除；双负即使根为实数也非合法框 | 校验 s>0、r>0、恢复宽高>0且全部有限；简单 epsilon 不是合法性证明 |
| `s+dot_s<=0` | 只做 `dot_s *= 0.0`，不修复本来非正的 s/r，不改 P；Inf乘0可变NaN | 防护只针对有限有效状态的下一步面积；不能宣称完整正值约束 |
| NaN 与 Inf 预测 | `masked_invalid` 同时压缩 NaN/Inf 行；`to_del` 却只按 `isnan` 删除 tracker | 仅Inf行可能被压缩而对象未删，后续列索引错指轨迹；加固须同一掩膜同步对象和预测，属于行为修正 |
| 更新后非法状态 | 预测时检查不覆盖随后 update／新生造成的非法框；返回前无完整有限性检查 | 对输出及状态再次验证，不能认为入口一次检查解决所有数值问题 |
| 空集合 | `(0,5)` 进入正常逐帧时钟；无轨迹时全部检测出生；双方空返回 `(0,5)` | 区分空二维数组、`[]`、`None`；主流程不能因 N=0 提前跳过预测／删除 |
| 计数与边界 | `>` 删除、`>=` streak、`<=` warmup；未校验负参数 | 测0、1、临界等号与过界；不要提前删除／提前归零 streak |
| 输出顺序与 ID | 逆序遍历当前轨迹；类 count 跨实例累计，输出加1；浮点数组存 ID | 不依赖行顺序做身份；跨序列显式命名空间，不随意重置活跃系统 count；极大整数 ID 有浮点精确性限制 |
| 同价／近同价关联 | 后端或输入排列可选不同最优解 | 测目标值和一对一约束；若要求确定性需另订 tie-break，不能声称原算法保证身份稳定 |

优化只能在固定输入、匹配集合、streak 和输出逐帧对照之后做。可考虑向量化 IoU、集合式 unmatched 查询和分离无头运行入口；不能缓存旧 IoU 代替预测后的 IoU，不能把仅预测轨迹加入输出，也不能以先门控／虚节点拒配替换后门控而不说明模型改变。`history` 每次匹配清空、预测追加；max_age 很大时连续失配 history 也增长。作者文件导入时含绘图／图像依赖和 `TkAgg` 设置，完整部署不等于复制两个类即可。

#### 有限检查与复跑入口

当前环境没有可用的 Python 解释器，因此未导入／执行作者 Python、NumPy、FilterPy 或 LAP。下列 **PowerShell 5、双精度、无第三方依赖** 检查器仅执行本卡的坐标算术、2×2候选收益枚举、计数规则和掩膜谓词。生命周期输入 `B/M/X/E` 分别代表出生／已关联／未关联／无轨迹空帧，是分支测试输入，**不是检测数据或模拟性能评测**；关联是否正确不由它验证。循环各断言组失败会终止；运行结果应与文末验证记录对应。

```powershell
$ErrorActionPreference = 'Stop'
$script:checks = 0
function Check($ok, $name) {
    if (-not $ok) { throw $name }
    $script:checks += 1
}
function Near([double]$a, [double]$b) {
    return [Math]::Abs($a - $b) -lt 1e-10
}
function BoxZ($b) {
    $w = $b[2] - $b[0]
    $h = $b[3] - $b[1]
    return @(($b[0] + $w/2), ($b[1] + $h/2), ($w*$h), ($w/$h))
}
function ZBox($z) {
    $w = [Math]::Sqrt($z[2]*$z[3])
    $h = $z[2]/$w
    return @(($z[0]-$w/2), ($z[1]-$h/2), ($z[0]+$w/2), ($z[1]+$h/2))
}
function IoU($a, $b) {
    $w = [Math]::Max(0, [Math]::Min($a[2],$b[2])-[Math]::Max($a[0],$b[0]))
    $h = [Math]::Max(0, [Math]::Min($a[3],$b[3])-[Math]::Max($a[1],$b[1]))
    $i = $w*$h
    return $i/(($a[2]-$a[0])*($a[3]-$a[1])+($b[2]-$b[0])*($b[3]-$b[1])-$i)
}
function Life([string[]]$events, [int]$maxAge=1, [int]$minHits=3) {
    $frame=0; $nextId=0; $t=$null
    foreach ($event in $events) {
        $frame += 1
        if ($null -ne $t) {
            $t.age += 1
            if ($t.tsu -gt 0) { $t.streak=0 }
            $t.tsu += 1
        }
        if ($event -eq 'B') {
            if ($null -ne $t) { throw 'unexpected live track' }
            $t=@{age=0; hits=0; streak=0; tsu=0; id=$nextId}
            $nextId += 1
        } elseif ($event -eq 'M') {
            if ($null -eq $t) { throw 'match without track' }
            $t.tsu=0; $t.hits+=1; $t.streak+=1
        }
        $out=0
        if ($null -ne $t) {
            if ($t.tsu -lt 1 -and ($t.streak -ge $minHits -or $frame -le $minHits)) {
                $out=$t.id+1
            }
            $snapshot=$t.Clone()
            if ($t.tsu -gt $maxAge) { $t=$null }
        } else { $snapshot=$null }
        [pscustomobject]@{frame=$frame; output=$out; alive=($null -ne $t); state=$snapshot}
    }
}
$z=BoxZ @(0,0,10,20)
Check (($z -join ',') -eq '5,10,200,0.5') 'box to observation'
Check (((ZBox $z) -join ',') -eq '0,0,10,20') 'round trip'
Check (Near (IoU @(0,0,10,20) @(0,0,10,20)) 1) 'same box'
Check (Near (IoU @(0,0,10,20) @(10,0,20,20)) 0) 'touching boxes'
$qArea=1*0.01*0.01; $pVelocity=1*1000*10
Check ((Near $qArea 0.0001) -and $pVelocity -eq 10000) 'noise multipliers'
$p00=10+10000+1; $p01=10000; $s=$p00+1
$ku=$p00/$s; $kv=$p01/$s
Check ($s -eq 10012 -and (Near $ku (10011.0/10012)) -and (Near $kv (10000.0/10012))) 'first gain'
$joseph=(1-$ku)*(1-$ku)*$p00+$ku*$ku
Check (Near $joseph $ku) 'scalar Joseph'
$area=200.0; $vs=-200.0
if ($area+$vs -le 0) { $vs*=0.0 }
Check (($area+$vs) -eq 200) 'area equality guard'
$w00=IoU @(0,0,10,10) @(0,0,10,10)
$w01=IoU @(0,0,10,10) @(5,0,15,10)
$w10=IoU @(-5,0,5,10) @(0,0,10,10)
$w11=IoU @(-5,0,5,10) @(5,0,15,10)
Check ((Near $w01 (1.0/3)) -and (Near $w10 (1.0/3)) -and $w11 -eq 0) 'counterexample IoU'
Check (($w00+$w11) -gt ($w01+$w10) -and $w11 -lt 0.3 -and $w01 -gt 0.3 -and $w10 -gt 0.3) 'post gate differs'
$equal=IoU @(0,0,10,10) @(0,0,20,10)
Check ($equal -eq 0.5 -and -not ($equal -gt 0.5) -and -not ($equal -lt 0.5)) 'threshold equality'
$a=0,0,0,1
Check (($a[0]+$a[1]) -eq 0 -and ($a[2]+$a[3]) -eq 1 -and ($a[0]+$a[2]) -eq 0 -and ($a[1]+$a[3]) -eq 1) 'equality omitted by fast branch'
$r=@(Life @('B','M','M','M','X','M','M','M'))
Check (($r.output -join ',') -eq '1,1,1,1,0,0,0,1') 'warmup and reacquisition'
Check ($r[0].state.hits -eq 0 -and $r[4].state.streak -eq 3 -and $r[5].state.streak -eq 1 -and $r[7].state.hits -eq 6) 'counter timing'
$r=@(Life @('E','E','E','B','M','M','M'))
Check (($r.output -join ',') -eq '0,0,0,0,0,0,1') 'late birth'
$r=@(Life @('B','X','X','B'))
Check ($r[1].alive -and -not $r[2].alive -and $r[2].state.tsu -eq 2 -and $r[3].state.id -eq 1) 'strict expiry and new id'
$r=@(Life @('B','X','M'))
Check ($r[2].alive -and $r[2].state.id -eq 0 -and $r[2].state.tsu -eq 0) 'associate before delete'
$r=@(Life @('B','X') 0 0)
Check ($r[0].output -eq 1 -and -not $r[1].alive) 'zero parameter boundary'
$r=@(Life @('E','E','E'))
Check (($r.output -join ',') -eq '0,0,0' -and -not $r[2].alive) 'empty frames'
Check ([double]::IsInfinity([double]::PositiveInfinity) -and -not [double]::IsNaN([double]::PositiveInfinity)) 'inf mask mismatch predicate'
"SO01 finite checks: $script:checks passed"
```

这些检查覆盖20组断言，不等于所有输入穷尽证明；其中计数器是单轨分支抽取、不是多目标容器，2×2反例直接比较两个排列，不是运行 LAP。NaN/Inf 项只验证掩膜不一致的谓词，不声称执行了 NumPy 压缩／错更新链。

后续真实实现仍需：作者与复现逐帧差分（全部计数、匹配及未匹配集合、输出 ID）→ N/M 为0、1及矩形多目标、重复框、同价与阈值等号 → 非有限值／负面积与索引对齐 → 长序列类别、丢帧、乱序、相机运动和交叉分层 → 固定检测下的 ID 切换、碎轨、IDF1／HOTA 等协议评估 → 分开统计检测与跟踪尾延迟。记录参数／依赖／随机性与所有失败序列，不用本文20项检查宣称达到真实跟踪精度。

### 贯穿 L1–L5 的五问自测

| 层级／问题 | 答题要点 | 条件、易错答案与证据入口 |
|---|---|---|
| L1：默认参数下，前3帧空，第4帧新目标且之后连续检测，何时首次输出？中间空帧如何传？ | 传 `(0,5)` 并每帧调用；第4帧出生 hits/streak=0，已过全局 warmup；第5–7帧三次匹配，首次输出在第7帧 | 假定关联均成功；“第6帧，因为出生算一次”错误。见 L1计数表、L5 late birth 检查 |
| L2：已有 Kalman＋Hungarian，为什么还不是 SORT？预测后门控为什么不能替换为先门控？ | 还需框观测变换、七维模型、IoU、未匹配出生与输出／删除状态；L4两框反例后门控只留1对，先禁边最大合法基数能留2对 | 不把“全局最优”扩展为真实身份最优；不把替换模型说成等价优化。见论文§3、源码关联分支 |
| L3：漏检后 ID 未删但连续两帧没有返回，应该增大 max_age 吗？ | 查 tsu/streak/frame_count；默认 min_hits=3，重获后 streak 从1重建，暂不输出不一定是删除；max_age 控保留、不控预测输出 | 先确认内部状态与检测是否有效；“增大 max_age 就连续返回预测框”错误。见 L1帧5–8与L3诊断 |
| L4：为什么速度 P 是10000、面积速度 Q 是0.0001？能从框噪声推出固定 R 吗？ | 分别有1000再10、0.01再0.01的重复作用；给出 z 变换和 J，局部 R≈JΣJᵀ，非线性和框尺度使常量对角 R 只是近似 | s 为面积，方差缩放不同于像素；P 不是误差上界。见 L4 Q/R/P、增益推导与雅可比 |
| L5：NaN 清理后为何仍可能错更新另一条轨迹？如何验证加固不改变正常语义？ | 压缩 mask 含Inf，删除列表仅NaN，预测行与对象索引可能错位；用统一有限掩膜同步，随后逐帧比较合法输入的匹配、计数和输出 | 这是需修正的异常域，不声称原码防护可靠；还测阈值等号、逆序删除和跨实例 ID。见 L5数值表及固定代码L220以后 |

## 来源、获取深度与未完成范围

以下均为 **2026-11-06** 联网核对：B=书目，A=摘要，T=实际读到方法／代码，F=获取失败。只有明确列出的部分算已读，URL 存在不等于实验复现。

<a id="source-1"></a>

**[cite:1] SORT 书目／版本，P0，B+A。** [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)：核对五位作者、ICIP 2016、相关 DOI、v1 2016-02-02／v2 2017-07-07；页面指向作者 `abewley/sort`。未取得 IEEE 正式版逐页比对，不把 v2 年份改写为方法提出年。

<a id="source-2"></a>

**[cite:2] SORT 2016 v1 方法，P0，T。** [固定 v1 HTML](https://arxiv.org/html/1602.00763v1)：实际读取引言、§3.1 检测过滤、§3.2 七维模型、§3.3 IoU／Hungarian、§3.4 出生与删除及实验开头；后部返回截断，没有审核全部实验／参考文献。全文方法是同作者原始证据，不当作独立复现。HTML 数学有重复转写，本卡状态式与作者数组交叉核对。

<a id="source-3"></a>

**[cite:3] 作者固定代码，P0，T。** [固定 `sort.py` 原文](https://raw.githubusercontent.com/abewley/sort/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py) 与 [提交页面](https://github.com/abewley/sort/commit/2236dff5019565958b84df7d871d41cc1db58ac7)。先从 [GitHub commits API](https://api.github.com/repos/abewley/sort/commits/master) 解析提交，再固定 SHA 读取整份源码及函数行号；该提交时间2023-11-28、改动 README，不冒称每段代码都在该日写成。所取 `sort.py` 文本按 UTF-8 编码的 SHA-256：`f5b10481ff9055b70f9e3f5d0126d5269ee91b16764a267a01f78ec3a2b931de`。读取不等于运行。

<a id="source-4"></a>

**[cite:4] 作者依赖，P0，T。** [同提交 requirements.txt](https://raw.githubusercontent.com/abewley/sort/2236dff5019565958b84df7d871d41cc1db58ac7/requirements.txt)：确认 FilterPy 1.4.5、scikit-image 0.17.2、lap 0.4.0；不是全环境 lockfile，不证明历史2016运行环境。

<a id="source-5"></a>

**[cite:5] FilterPy 依赖实现，P0，T。** [1.4.5 `kalman_filter.py`](https://raw.githubusercontent.com/rlabbe/filterpy/1.4.5/filterpy/kalman/kalman_filter.py)：读取初始化 `x/P/Q/R`、预测与更新，包括 `inv(S)` 与 Joseph 协方差式；首次长响应截断后，按指定源码行段取得 predict/update 正文。仅作为版本依赖事实与代数核对，不算 SORT 独立性能证据。

<a id="source-6"></a>

**[cite:6] SciPy 求解器说明，P0，T。** [linear_sum_assignment 官方 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)：当前文档说明矩形线性指派、返回索引和 modified Jonker–Volgenant；未固定本地 SciPy 环境，不追认2016代码必用此后端。固定 SORT 主路径仍先尝试 lap，未运行 lap/SciPy 两分支一致性测试。

## 验证记录与销项边界

本次仅新增本文件；未编辑总览、台账、导航、源码或依赖。仓库 Python 源码检索未发现 `KalmanBoxTracker`、`class Sort` 或 FilterPy 接入；当前终端 Python 命令指向不可用的商店入口。没有为文档核验安装依赖，也未新增测试脚本或跟踪模块。

本次从 Markdown 提取 PowerShell 代码块到内存执行，实际结果为 **`SO01 finite checks: 20 passed`**。UTF-8、五层／五问结构、9张表列数、4个围栏边界、28处本地链接／锚点和尾空白检查通过；13处外部链接另行计数，不把链接计数等同于所有 URL 再访问。`git diff --check` 通过；本文件未跟踪，故另作直接文件检查，不能只靠 Git 检查。

Git 列出的102个非目标、非忽略文件按排序路径及各文件 SHA-256 拼接后计算的指纹，前后均为 `DD918758F5B79EAEBC25AC6917744F2C3045CF3B499F333406D0A67FD7D3077C`。这只证明所列文件在两次取样之间内容一致，不包括忽略文件与 Git 元数据；用户已有改动未回退。

有限检查不覆盖检测器训练、完整 Kalman 矩阵运行、LAP 后端、真实视频、MOT 指标、长时间性能或现场安全。视觉引导还需要独立标定、同步、控制和安全层，不能把输出框 ID 直接当执行器命令。

**销项结论：SO01 补齐 G45 的具体方法卡内容，可供后续将“缺卡”转“已有”；实现仍未实现，L3/L5 仍部分。** 保留的缺口是2016正式发表软件快照／文稿差异复核、作者完整依赖运行、数值加固及其逐帧等价性验证、真实检测与身份指标验收。台账当前“缺卡”和总览“无专卡”为未同步旧状态，不在本次单文件范围内悄然更改，也不重算全库覆盖率。
