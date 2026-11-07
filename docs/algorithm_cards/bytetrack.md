# BYTE／ByteTrack 算法卡

[学习规范](../algorithm_learning_standard.md) · [tracking 总览 E09](../tracking_guidance_evolution.md#39-e09bytebytetrack低分候选的条件使用) · [覆盖台账 G48／12-09](../evolution_coverage.md#12-跟踪运动视觉引导) · [SORT SO01](sort.md#so01) · [指派 H01](hungarian.md#h01)

## 范围、版本与覆盖状态

核对日期：**2026-11-06**；本仓库基准：HEAD `918a03e` 加当前工作树。唯一卡键 **`bytetrack.md:BT01`**，承接 **G48／12-09**。BYTE 是高低分检测的通用关联策略，ByteTrack 是把它与检测器、运动模型和生命周期组合的具体系统；二者共享本卡，不拆成两张，不把通用 Kalman＋LAP 组件当成系统实现。

方法原文为 *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*：arXiv **v1，2021-10-13**；v2 为2021-10-14，v3 为2022-04-07；正式发表为 **ECCV 2022**，DOI `10.1007/978-3-031-20047-2_1`。2021 v1 列 Yifu Zhang、Peize Sun、Yi Jiang、Dongdong Yu、Zehuan Yuan、Ping Luo、Wenyu Liu、Xinggang Wang；当前 v3 书目另列 Fucheng Weng，不能把当前作者栏直接当成 v1 作者栏。本次读取 v1 引言／Algorithm 1 及2022正式正文 §3、§4.1–4.2，不声称逐版全文校勘。[cite:1](#source-1)[cite:2](#source-2)[cite:3](#source-3)

作者实现固定为论文所链 `ifzhang/ByteTrack` 提交 **`d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`**，日期 **2022-12-11**；GitHub API 当前解析到 `FoundationVision/ByteTrack`，是仓库迁移入口，不是改用 BoT-SORT。这个提交是教程链接合并，**不等于2021首发或ECCV实验环境快照**。以下“固定代码”专指该提交的 `yolox/tracker/byte_tracker.py` 与配套文件，不指 Ultralytics 重写、BoT-SORT、部署端其他语言变体或整个方法族。[cite:4](#source-4)

| 字段 | 覆盖与限定 |
|---|---|
| 演进定位 | tracking-by-detection；从只收高分检测，改为高分优先、低分条件续接；增加候选、第二次关联及生命周期成本 |
| L1 | 已核验：输入、三次实际指派调用、预测、出生／确认／重激活／移除／去重／输出顺序；不是运行作者程序 |
| L2 | 已核验：BYTE 的低分证据机制、分阶段必要性和状态不变量／例外 |
| L3 | 部分：参数量纲、分数据集配置与诊断明确；无真实视频标定、消融或耗时剖析 |
| L4 | 已核验（限定模型代数）：xyah、F/H、噪声、Kalman、IoU、分数融合与带拒配代价；无通用身份正确性保证 |
| L5 | 部分：固定源码映射、依赖／坐标／容器边界及有限检查；未运行完整Python、检测器或MOT评测 |
| 面试自测 | 贯穿 L1–L5 的5问，含答案、前提、易错点和证据入口 |
| 实现状态 | **未实现**：本仓库无 BYTETracker／STrack 模块；文档伪代码、源码阅读与算术检查不等于落地跟踪器 |
| 验证范围 | 文档静态核验与本文 PowerShell 有限分支／代数检查；没有本仓库 MOTA、IDF1、HOTA 或 FPS 实测 |
| G48口径 | 可建议“具体方法缺卡”转“已有、P2深度待验”；本次只新增本文件，不改台账、索引或总览，不宣称已同步销账 |

证据按论文、作者系统代码、依赖源码区分。同作者的两版论文与仓库不是三份独立复现；版本细节按 **[SINGLE-SOURCE]** 事实解读，数学推导另标成立条件。没有统一工况下三套独立实验支持性能排名，故不搬用摘要数字充当工业效果。

<a id="bt01"></a>
## BT01 BYTE策略与ByteTrack系统

### 任务、输入输出与适用前提

输入是每帧检测框及分数、原图尺寸 `img_info=(height,width,...)`、网络输入尺寸 `img_size=(height,width)`。固定 `update` 的五列分支接收浮点 NumPy `(N,5)`：`[x1,y1,x2,y2,score]`；非五列分支调用 `.cpu().numpy()`，按YOLOX常见七列输出读取前四坐标、`objectness * class_confidence`，第七列类别不参与关联。因此不能把任意六／七列 NumPy 数组直接传进该分支，也不能认为它自动做多类别隔离。[cite:4](#source-4)

框通过 `scale=min(net_h/orig_h,net_w/orig_w)` 后原地执行 `bboxes /= scale`。这只逆等比缩放，**不减居中letterbox偏移、不补ROI原点、不做畸变校正**；调用者须匹配该坐标契约。五列分支的切片是输入视图，重复使用同一数组可能重复缩放；输入整数数组也不满足浮点原地除法要求。有效输入应满足有限分数／坐标、正宽高、正图像尺寸，工程上应先约束分数在 `[0,1]`，这些完整守卫并非原码已有。

输出是 **`list[STrack]`**，不是 SORT 的 `(K,5)` 数组：通过 `tlwh`／`tlbr`、`track_id`、`score` 读取结果；空结果是 `[]`。正常流程只输出 `tracked_stracks` 中 `is_activated=True` 的对象，框为当前滤波状态，score 是最近一次匹配检测分数；不返回旧 Lost 的纯预测框。返回的是对象引用，不是不可变历史快照。调用层的面积／长宽比过滤与文件保存不是 `BYTETracker.update` 自身逻辑。[cite:4](#source-4)[cite:7](#source-7)

前序瓶颈 → 改进 → 代价 → 遗留问题：固定高分过滤丢掉遮挡、模糊时的真检测，造成漏轨和碎轨；BYTE 让低分框在已有轨迹几何证据支持下参与续接，而不普遍放宽出生；代价是更多候选和关联计算、误接背景风险、检测器分数校准与状态管理；完全漏检、长时无重叠、同形目标交叉和强相机运动仍不能凭空恢复身份。适用于确有“低分但位置仍可信”的检测，不是低分框普遍正确的假设。[cite:2](#source-2)[cite:3](#source-3)

### L1 步骤：从分数分层到完整生命周期

#### 严格区间与三类阈值

令检测分数为 s，`track_thresh=τ`，正常配置 `0.1<τ<1`。固定代码为：

\[
D_H=\{d:s_d>\tau\},\qquad D_L=\{d:0.1<s_d<\tau\},\qquad
\tau_{birth}=\tau+0.1.
\]

**s=0.1 或 s=τ 都被排除**，不是低分闭区间；高分集合与低分集合不重叠，也不覆盖所有检测。只有经过高分关联及unconfirmed确认后仍剩余、且 `s>=τ_birth` 的高分框才能出生：代码是 `< det_thresh: continue`，等号可出生。`τ<s<τ+0.1` 可以续接／确认，但不能独立出生。`τ+0.1>1` 时合法概率分数不可能新生，参数应先检查，不能把出生阈值和高分阈值混为一个。[cite:4](#source-4)

| 阶段 | 轨迹候选 | 检测候选 | 代价与 `cost_limit` |
|---|---|---|---|
| 第一次关联 | 已确认Tracked，与旧 `lost_stracks` 按ID并集 | `D_H` | IoU距离；`mot20=False`再 `fuse_score`；`match_thresh`，demo默认0.8 |
| 第二次关联 | 第一阶段未匹配且 **state仍为Tracked** 的候选 | `D_L` | 纯IoU距离，固定0.5，**不融合score** |
| unconfirmed确认 | 开帧时 `is_activated=False` 的tracked对象 | 第一阶段剩余高分框 | IoU距离；非MOT20再融合score；固定0.7 |

表中后两阈值是代价上限，不是检测分数，也不是把“0.8”直接当最小IoU。旧 Lost 不参加第二阶段；unconfirmed不参加前两阶段，也不在主池 `multi_predict` 中预测。通常所谓“两阶段”指 BYTE 的高／低分关联；系统另有unconfirmed的第三次LAP调用。[cite:4](#source-4)[cite:5](#source-5)

#### 固定系统伪代码

下面是语义展开，不是新增可运行模块。`JOIN/SUB` 按 `track_id` 并／差；`LAP(C,λ)` 调用 `lap.lapjv(C,extend_cost=True,cost_limit=λ)`，返回配对、未匹配行／列。`U` 为临时unconfirmed，`A/R/Lnew/Dnew` 分别是本帧更新、新找回、新丢失、新移除列表；保留原码容器先后次序，而非偷偷修正。[cite:4](#source-4)[cite:5](#source-5)

```text
UPDATE_FRAME(detections, original_size, network_size):
    frame_id += 1
    A, R, Lnew, Dnew = [], [], [], []
    decode scores; map boxes back by dividing scale
    DH = boxes with score > track_thresh
    DL = boxes with 0.1 < score < track_thresh
    U = objects in tracked_stracks with is_activated == False
    T = objects in tracked_stracks with is_activated == True
    P = JOIN(T, lost_stracks)
    for t in P:
        if t.state != Tracked: set mean[7] = 0
        Kalman predict once
    C1 = 1 - IoU(P, DH)
    if not mot20: C1 = FUSE_SCORE(C1, DH)
    M1, UP, UH = LAP(C1, match_thresh)
    for (i,j) in M1:
        if P[i].state == Tracked:
            UPDATE(P[i], DH[j]); append to A
        else:
            RE_ACTIVATE(P[i], DH[j], new_id=False); append to R
    T2 = [P[i] for i in UP if P[i].state == Tracked]
    M2, UT2, UL = LAP(1 - IoU(T2, DL), 0.5)
    for (i,j) in M2:
        UPDATE(T2[i], DL[j]); append to A
    for i in UT2:
        mark T2[i] Lost; append to Lnew
    Hremain = DH[UH]
    CU = 1 - IoU(U, Hremain)
    if not mot20: CU = FUSE_SCORE(CU, Hremain)
    MU, UU, UHU = LAP(CU, 0.7)
    for (i,j) in MU:
        UPDATE(U[i], Hremain[j]); append to A
    for i in UU:
        mark U[i] Removed; append to Dnew
    for j in UHU:
        if Hremain[j].score < track_thresh + 0.1: continue
        ACTIVATE(Hremain[j], frame_id); append to A
    for t in lost_stracks:
        if frame_id - t.end_frame > max_time_lost:
            mark t Removed; append to Dnew
    tracked_stracks = old tracked_stracks whose state == Tracked
    tracked_stracks = JOIN(JOIN(tracked_stracks, A), R)
    lost_stracks = SUB(lost_stracks, tracked_stracks)
    lost_stracks.extend(Lnew)
    lost_stracks = SUB(lost_stracks, removed_stracks)
    removed_stracks.extend(Dnew)
    tracked_stracks, lost_stracks = REMOVE_DUPLICATES(tracked_stracks, lost_stracks)
    return [t for t in tracked_stracks if t.is_activated]
```

第二阶段源码仍有一个 `else: re_activate`，但入口已筛选 `state==Tracked`，正常单线程流程中该else不可达，不能据此声称低分框会重激活旧Lost。第二阶段未匹配低分框全部丢弃，不进入unconfirmed确认或出生。

每个真实帧都应推进一次跟踪时钟，包括无检测帧（可用浮点 `np.empty((0,5))` 进入核心）。**不要把这个接入要求说成作者demo已经满足**：固定 `demo_track.py` 在 `outputs[0] is None` 时跳过 `update`，真实视频帧号继续、核心时钟不动。核心不能接收 `None` 或一维空列表。流结束即停止，没有离线回填／插值和自动继续预测。[cite:7](#source-7)

#### 状态、ID与输出生命周期

`TrackState` 枚举为 `New=0,Tracked=1,Lost=2,Removed=3`；**unconfirmed不是第五个枚举**，而是 `Tracked + is_activated=False`。`frame_id` 在轨迹对象中表示最后出生／匹配时刻，`end_frame` 就返回它；预测不更新它。`tracklet_len` 出生0、普通update加1、重激活归0，不是确认阈值，也不是去重所用年龄。[cite:4](#source-4)[cite:6](#source-6)

| 事件 | 状态／字段变化 | 输出与身份 |
|---|---|---|
| 高分剩余框达到出生阈值 | 新ID，xyah初始化、速度0、Tracked、start_frame=end_frame=当前帧 | 仅核心第1帧出生直接 `is_activated=True`；以后出生为unconfirmed，当帧不输出 |
| 已确认Tracked获高分或低分匹配 | KF校正、Tracked、is_activated=True、end_frame更新、score替换 | 保留ID并输出，低分续接的score可低于高分阈值 |
| unconfirmed在下一次调用匹配剩余高分 | 普通update，is_activated=True | 首次输出原ID，不要求该框再达到出生阈值 |
| unconfirmed没匹配 | 直接Removed | 不进入Lost buffer；仅有低分框也不能确认 |
| 已确认Tracked两阶段都失败 | mark_lost，只改state；is_activated通常仍True | 加入Lost，不输出；不能只看布尔标志判断可信 |
| 旧Lost在第一阶段获高分 | `re_activate(new_id=False)`，KF校正、Tracked、is_activated=True、tracklet_len=0 | 原ID立即输出，不走一次新的unconfirmed；start_frame不重置 |
| 旧Lost未恢复且严格超时 | mark_removed后汇总 | 无输出；列表清理滞后例外见下一节，不把Removed无条件当吸收态 |

ID由 `BaseTrack._count` 类计数器递增，首ID为1；新建tracker实例不自动清空全局计数。删除后普通新生不会主动搜索旧ID；不能把几何重激活称为身份认证或外观Re-ID。

生命周期的正常分支例：第1帧高分出生A并输出 → 第2帧仅低分且几何通过二阶段，继续输出A → 第3帧空，A转Lost、无输出 → 第4帧仅低分，旧Lost不参加二阶段、仍无输出 → 第5帧高分且第一阶段配上A，重激活同ID并输出。另一个第6帧出生的新目标B当帧不输出；第7帧剩余高分可确认B，只有低分则B被移除。这是以“匹配已成立”为输入的分支推演，不是视频实验。

#### 移除、去重的顺序与原码例外

失踪时间阈值为

\[
B=\operatorname{int}\!\left(\frac{f_{configured}}{30}\,track\_buffer\right),\qquad
max\_time\_lost=B,
\]

删除判据是 `current_frame - end_frame > B`，不是 `>=B`。它按最后一次有效匹配时间计数，不是按进入Lost列表那帧计数。旧Lost先预测／关联，再做过期检查；如果本帧高分成功找回，end_frame已更新为本帧，不会因开帧时过龄而删除。新转Lost的 `Lnew` 尚未加入旧Lost迭代，即使B=0，也不会在该迭代立刻处理。[cite:4](#source-4)

更隐蔽的次序是：先从Lost减去 **旧** `self.removed_stracks`，然后才把本帧 `Dnew` 追加到Removed。因此本帧过期对象通常会以 `state=Removed` 在Lost列表残留到下一帧；去重也可能提前去掉它。下一帧构造主池并未按Lost枚举过滤此列表，残留Removed仍能参与第一阶段，非Tracked分支又可调用 `re_activate`。若此时高分匹配成功，甚至能恢复原ID；Removed历史列表持有相同对象引用，也会看到其状态变回Tracked。**这是固定代码的容器行为，不是BYTE理论要求，也不是可靠的“永不复活”删除实现。** 修成先汇总Removed再差集、并限制主池合法state，是合理加固但会改变该边界行为，需单列差分验证。[cite:4](#source-4)[cite:6](#source-6)

去重在容器并／差和移除追加之后、输出之前执行，仅比较Tracked与Lost两组。取 `1-IoU < 0.15`（即IoU严格大于0.85）的所有跨组对；比较每条 `track.frame_id - track.start_frame`，较短的一方从列表删掉，**相等时删Tracked侧**。Lost的frame_id是最后命中时刻，不是当前帧，年龄不含后续失踪期；它也不是命中次数、分数或`tracklet_len`。去重不调用 `mark_removed`，不加入Removed历史，不做同组NMS，不保证多对重叠时全局保留最长轨迹。按ID的JOIN和基于IoU的去重是两件事。

### L2 机制：低分证据为何必须受条件约束

**先高后低，而非一起降检测门限。** 高分匹配先占定身份，低分只能竞争未匹配且仍Tracked的对象；背景低分不能抢走已由高分更新的轨迹，也不能直接出生。一次把两组拼接做LAP，会改变优先级和出生集合，不是等价向量化。必要前提是高分整体更可信、低分真框位置尚有用；高分假阳性同样会误导第一阶段。[cite:2](#source-2)[cite:3](#source-3)

**预测先于关联，校正后于配对。** 主池每帧只预测一次，以当前时刻预测框关联当前检测；第二次关联复用同一时刻的未匹配预测，不能再预测一次。低分检测是新观测，不是把预测框伪造为观测回灌。Lost预测仅把高度速度 `mean[7]` 置0，**不清中心速度、不清宽高比速度、不清协方差**；其作用是抑制失踪期间高度外推，不是冻结整框或完整约束滤波。

**低分不用score融合。** `fuse_score`将几何相似度乘检测分数；低分本就弱，若把它强行用于二阶段，会额外压制要挽救的候选。2022原文还说明严重遮挡／模糊下低分外观特征不可靠，第二阶段选择IoU而非Re-ID。但这不能推出“BYTE禁止任何Re-ID”：它可与外观关联方法组合，第一阶段可用外观证据。[cite:3](#source-3)[cite:5](#source-5)

**Lost与unconfirmed不等价。** Lost曾获确认但暂失，主要依赖高分重新接回；unconfirmed只有出生证据，下次未获高分确认就删，以减少孤立假轨。允许旧Lost吃低分、让unconfirmed共享buffer，都会扩大幽灵身份／背景续接空间，不是原码规则。首帧直接确认只看核心 `frame_id==1`，不是SORT的多帧warmup。

**系统边界。** 固定路径没有调用 `embedding_distance`、`gate_cost_matrix`、`fuse_motion`，即使这些函数和卡方表存在于配套文件，也不等于本路径使用Mahalanobis门或外观网络。没有GMC／相机运动补偿、Re-ID特征融合、BoT-SORT专用状态模型；不得以同样有两阶段关联就把BoT-SORT规则嫁接进来。BDD100K论文设置及其他BYTE组合是另一配置，不反推本路径具备其能力。[cite:3](#source-3)[cite:4](#source-4)[cite:5](#source-5)

### L3 工程权衡：参数、版本与排障

| 参数／接口 | 固定代码或明确来源 | 诊断、尺度与代价 |
|---|---|---|
| `track_thresh` | demo默认0.5；2022论文§4.1默认0.6，除非另说 | 无量纲分数；降低同时改变高／低集合和出生阈值，不是单独改善召回 |
| 低分下界 | 核心硬编码0.1，严格大于 | 上游检测阈值过高会先删掉候选，再改跟踪器无效；需共同保存检测器conf/NMS设置 |
| `det_thresh` | `track_thresh+0.1` | 出生更严；检查等号与浮点表示，不挪用于确认或重激活 |
| `match_thresh` | demo默认0.8；二阶段0.5、确认0.7硬编码 | 都是cost limit，越大通常越宽；分数融合时需同时解释s与IoU；存在全局竞争 |
| `mot20` | demo默认False，True关闭第一阶段及确认阶段的score融合 | 不是启用Re-ID或改二阶段阈值；固定core中也不代做裁剪或类别筛选 |
| `track_buffer` | demo默认30，按30FPS基准缩放为B | 增大候选存活期会增加计算、漂移和错接风险，不会输出Lost框 |
| 时间步 | KF固定dt=1个调用 | 构造参数frame_rate只改变B，不改变F/Q；低帧率／跳帧不是仅增buffer就解决 |
| 输出过滤 | demo另有 `min_box_area=10`、`aspect_ratio_thresh=1.6` | 面积px²、长宽比无量纲；不要把调用层过滤归为核心出生条件 |

若每个真实帧都调用且传入正确帧率f，B/f约为 `track_buffer/30` 秒，整数截断及严格超时边界仍需计入。15／30／60FPS、buffer=30分别给B=15／30／60，不是始终“保留30帧”。固定image demo传 `args.fps`，video demo却硬编码 `frame_rate=30`，即使已经读到视频fps；再叠加None帧跳过，物理保留时间会偏离预期。丢帧或不规则采样应显式设计时间戳与F/Q离散化，而非宣称已有真实dt支持。[cite:4](#source-4)[cite:7](#source-7)

| 症状 | 先采集证据 | 调整／对照顺序 |
|---|---|---|
| 二阶段完全没有恢复 | 检测器输出分数直方图、进入DH/DL数量、原始conf/NMS | 先确认低分候选未在上游丢失，再看IoU和二阶段资格 |
| 高分时断轨，低分反而接上 | 原始IoU、融合后代价、mot20、各阶段limit | 分开查 `1-IoU*s` 与 `1-IoU`，不能把阈值当同一IoU门 |
| Lost有低分重叠却找不回 | state、is_activated、主池／T2索引、最后命中帧 | 核对“旧Lost不入二阶段”；先保留该失败原因，不直接放宽生命周期 |
| 孤立新目标不输出或很快消失 | 出生帧、分数与det_thresh、确认候选 | 区分未出生、unconfirmed、确认失败；不要照搬SORT min_hits |
| 过期后原ID偶尔又出现 | 三个持久列表ID及对象state、Dnew、处理顺序 | 复查Removed滞留主池；加固前后单独测临界帧，不当成Re-ID成功 |
| 小框相贴还算有重叠 | bbox依赖版本、原始坐标与变换、IoU矩阵 | 检查inclusive `+1`；更换连续IoU须重标门限并比较匹配差异 |
| 平均FPS够但拥挤帧慢 | 检测／IoU／LAP／去重各耗时、池大小、尾延迟 | 固定检测输入先剖析；不能用网络FPS代替端到端延迟 |

2022正式论文§4.1中MOT17／MOT20／HiEve用IoU，描述IoU低于0.2拒配与Lost保留30帧；这不是指定代码“三次关联统一IoU≥0.2”的逐行规范。BDD100K配置引入UniTrack Re-ID；§4.2还说明该低帧率／强相机运动设置不用Kalman。本文KF＋IoU路径是明确的一种系统实现，**不把MOT配置推广为所有数据集通用**。训练属于检测器／可选外观前端，不是BYTE另有一个端到端可微训练损失。[cite:3](#source-3)

### L4 数学推导：状态、代价与保证的边界

#### xyah变换与八维常速模型

由角点框得 `w=x2-x1>0,h=y2-y1>0`，观测与状态为

\[
z=(c_x,c_y,a,h)^T,\quad c_x=x_1+w/2,\ c_y=y_1+h/2,\ a=w/h,
\qquad x=(c_x,c_y,a,h,v_x,v_y,v_a,v_h)^T.
\]

反变换 `w=a*h,x1=cx-w/2,y1=cy-h/2,x2=x1+w,y2=y1+h`。这里 **a是宽高比，不是SORT的面积s**；有比例速度va和高度速度vh，不能复制SO01的七维F、面积速度防护或Q/R。中心／高度单位px，a无量纲；速度单位分别px/调用与1/调用。

\[
F=\begin{bmatrix}I_4&I_4\\0&I_4\end{bmatrix},\qquad H=[I_4\quad0],\qquad
x_t^-=F\widetilde x_{t-1}^+,\quad P_t^-=FP_{t-1}^+F^T+Q(h_{t-1}).
\]

对于非Tracked，\(\widetilde x\)仅将vh置0；P没有按该置零操作投影。因此这一步是启发式均值修正，不是对“vh已知为0”的严格条件化。正高度和正比例也未由线性模型自动维持。

固定KF用 `p=1/20,v=1/160`，以下向量是**标准差**，代码平方后才作对角协方差：[cite:6](#source-6)

\[
P_0=\operatorname{diag}\big((2ph,2ph,10^{-2},2ph,10vh,10vh,10^{-5},10vh)^2\big),
\]
\[
Q(h)=\operatorname{diag}\big((ph,ph,10^{-2},ph,vh,vh,10^{-5},vh)^2\big),
\]
\[
R(h^-)=\operatorname{diag}\big((ph^-,ph^-,10^{-1},ph^-)^2\big).
\]

式中 `vh` 连写在噪声向量里表示权重v乘高度h，不是状态分量 \(v_h\)。Q用预测前高度，R在project时用预测高度；不是按检测score设R。`fuse_score`改变的是关联代价，**没有把低分检测自动变成高测量方差**。

令创新 \(r=z-Hx^-\)，\(S=HP^-H^T+R\)。最小化给定关联下线性校正误差协方差的迹得

\[
K=P^-H^TS^{-1},\qquad x^+=x^-+Kr,\qquad P^+=P^--KSK^T.
\]

代码用SciPy Cholesky分解及 `cho_solve` 求K，不显式求逆；P用减法形式，不是SO01依赖的Joseph实现。Cholesky要求S正定；`check_finite=False`不等于自动处理NaN。线性高斯、噪声与正确关联等条件下可作条件估计解释，经验高度噪声和非Tracked重置并不给出全系统最优性。[cite:6](#source-6)

对理想常速系统，\([H;HF]\)的秩为8，所以两次已关联且已知非零间隔的观测可辨识四个速度；单帧只能初始化速度先验。宽高比变换在h趋0时病态；交叉同形框时身份不可辨识，满秩运动模型不能补足身份信息。本文**未提供通用理论误差上界**，P既不是身份错误概率，也不是机械安全误差界。

#### IoU及依赖的像素约定

数学上连续矩形IoU为 `intersection/(areaA+areaB-intersection)`；但固定matching调用外部 **`cython_bbox.bbox_overlaps`**，不能照搬SO01的无`+1`约定。核对到的依赖快照采用inclusive形式：

\[
w_I=\max(0,\min(x_{2A},x_{2B})-\max(x_{1A},x_{1B})+1),
\quad h_I=\max(0,\min(y_{2A},y_{2B})-\max(y_{1A},y_{1B})+1),
\]
\[
A=(x_2-x_1+1)(y_2-y_1+1),\quad J=\frac{w_Ih_I}{A_A+A_B-w_Ih_I},\qquad C_{ij}=1-J_{ij}.
\]

STrack的宽高／xyah变换仍是坐标差值，没有`+1`；这两个约定在该依赖组合下并存，不能为了形式统一偷偷都改掉。以合法框 `[0,0,10,10]` 与 `[10,0,20,10]` 为算术输入，连续IoU为0，inclusive为 `11/(121+121-11)=1/21`。小框、边界及缩放时差异更显著；inclusive项也破坏严格的连续尺度不变性。ByteTrack依赖未全锁，本文将该bbox快照作为明确的实现核对对象，不声称每个安装包都已经验证。[cite:5](#source-5)[cite:8](#source-8)

#### 分数融合与带拒配的LAP

`fuse_score`按检测列广播分数，不用轨迹score，也不是两个分数求平均：

\[
C'_{ij}=1-(1-C_{ij})s_j=1-J_{ij}s_j.
\]

在 `s∈[0,1]` 时，\(C'\ge C\)。要落在代价上限λ以内，原始IoU需要 \(J\ge1-\lambda\)，融合后需要 \(Js\ge1-\lambda\)；当s>0时即 \(J\ge(1-\lambda)/s\)。若右端大于1，没有几何框可满足。它只是候选代价条件，**不保证在一对一竞争中获得配对，等号还可能与拒配同价**。例如IoU=0.5、s=0.6，原始代价0.5，融合后0.7；不能把二阶段0.5当第一阶段0.5使用。[cite:5](#source-5)

三次指派各自构造矩形矩阵（行轨迹、列检测），不是一次全视频优化。`matching.linear_assignment`实际调用LAP/Jonker–Volgenant后端，将`thresh`作为`cost_limit`传入；**不是SORT的“先配满再按IoU删边”**，也不是遍历greedy。空矩阵直接返回空配对及全体未匹配索引；非空但无配对时`np.asarray([])`的形状为`(0,)`，不总是`(0,2)`，复现包装应统一契约。

为把“允许拒配”写清，取独立核对的 **lap v0.5.12** 有限cost_limit扩展：n条轨迹、m个检测时增加虚行列，真实—虚拟边代价λ/2、虚拟—虚拟边0。其目标等价于

\[
\min_X\left[\sum_{ij} C_{ij}X_{ij}
+\frac\lambda2\left(n+m-2\sum_{ij}X_{ij}\right)\right],
\quad X_{ij}\in\{0,1\},\ \sum_jX_{ij}\le1,\ \sum_iX_{ij}\le1.
\]

去掉常数即最大化 \(\sum_{ij}(\lambda-C_{ij})X_{ij}\)，**不等于先最大化合法匹配数再最小化代价**。抽象代价算术例 `C=[[0.1,0.7],[0.7,1.0]],λ=0.8`，只配0.1那条的目标为0.9，两条交叉边为1.4，后者虽有两条低于阈值的边也不更优。等号C=λ的单边匹配与两端拒配同价，不能承诺后端必接纳或必拒绝。[cite:9](#source-9)

这段虚节点推导明确针对所读v0.5.12，不冒称2022作者锁定该版本；另外读取的v0.4.0标签源码使用不同扩展布局，不能复用矩阵尺寸／虚边数值描述。作者requirements只写 `lap`，**历史实际安装包、tie-break和精确最优配对仍需锁包运行验证**。可复现接口事实是传入cost_limit并读取x/y中的-1，而不是从另一个未被调用的 `_indices_to_matches` 函数推断等号规则。

复杂度：固定8维Kalman每帧O(T)；三次IoU矩阵代价约为O((T+L)H+T₂D+UHᵣ)，内存为各阶段矩阵规模；T为已确认数、L为Lost数、H/D为高／低分数、U为unconfirmed数、Hᵣ为剩余高分数。另有Tracked×Lost去重。稠密LAP常用扩展方阵三次量级估计，实际取决于后端及矩形扩展；Removed历史不裁剪，还会增加长序列列表差集成本。两阶段优先级、出生和去重是启发式，没有整体MOTA／身份误差收敛定理。

### L5 实现：源码映射、边界与有限测试

#### 固定源码对应

以下链接全部固定作者SHA；函数名而非漂移行号作为核对入口：

| 源码 | 对应正文与核查点 |
|---|---|
| [`byte_tracker.py`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py) | STrack的predict/multi_predict、activate/update/re_activate；BYTETracker构造与update；joint/sub/remove_duplicate_stracks |
| [`matching.py`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/matching.py) | ious→cython_bbox、iou_distance、fuse_score、linear_assignment；存在但未调用的外观与运动门函数 |
| [`kalman_filter.py`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/kalman_filter.py) | 八维F/H、initiate的P、predict的Q、project的R、Cholesky校正与减法P |
| [`basetrack.py`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/basetrack.py) | 枚举、类计数ID、end_frame和仅修改state的mark_lost/mark_removed |
| [`demo_track.py`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/tools/demo_track.py) | 默认阈值、图像／视频frame_rate差异、None帧跳过update、调用层输出过滤 |
| [`requirements.txt`](https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/requirements.txt) | NumPy／lap等未锁；列出filterpy不代表本路径调用它，自带KF实际依赖SciPy |

#### 数值、容器和等价性边界

| 边界 | 风险与验证要求 |
|---|---|
| 数据布局 | mean `(8,)`、P `(8,8)`；批量 `(T,8)`／`(T,8,8)`；IoU行列保持轨迹／检测方向，筛选后索引只属于对应阶段 |
| 非有限值与非法框 | 原码无全面守卫；h=0使a除零，非正a/h和NaN预测会污染bbox／Cholesky；应在接入及输出检查并拒绝，而非填零框继续 |
| 浮点阈值 | `s==τ`两边都不收；`τ+0.1`可能非精确十进制；应测试同dtype的相邻可表示值，不无说明加epsilon改算法 |
| 兼容性 | 原码有 `np.float`，现代NumPy兼容性需要明确修订；cython_bbox需扩展包，requirements没有完整锁定，不能宣称即装即跑 |
| 状态与列表 | Lost布尔可仍True、Removed可残留Lost、去重只删容器不改state；输出筛选本身仅查is_activated，不附加统一state守卫 |
| 引用与ID | 返回对象会继续被更新；Removed历史也持同对象；多流需实例隔离与全局ID命名空间，不以返回行号识别目标 |
| 求解边界 | N/M为0、无可配边、矩形、同价、score融合、cost_limit等号分别测；不要擅自改用强制配满的SciPy调用 |
| 优化等价 | 向量化预测／IoU后逐帧比对匹配、各容器、state、is_activated、start/end、ID与输出；勿将两阶段合一、再预测、提前删Lost或把unconfirmed纳入主池 |

去重和移除修复属于行为改动，应对正常域与异常／临界域分别给出差分，不把“修复后更合理”写成逐版本完全等价。真实工程还需按场景评估交叉、遮挡、低分背景、出画后重入、类别混合、相机运动和乱序帧；固定检测做BYTE消融，再评估检测器更换，避免把前端改进归给关联策略。

#### 有限检查器与复跑入口

下面是 **PowerShell 5双精度、无第三方依赖** 的分支／代数检查。输入框、代价和事件只是明确的单元算术夹具，不是合成视频、性能数据或论文复现。`Confirm/Recover`以已经通过关联为前提；容器测试抽取单ID路径，**不运行Kalman全矩阵、LAP或完整多目标循环**。代码可从本Markdown提取到内存执行，无需生成脚本文件。

```powershell
$ErrorActionPreference = 'Stop'
$script:checks = 0
function Check($ok, $name) {
    if (-not $ok) { throw $name }
    $script:checks += 1
}
function Near([double]$a, [double]$b) { return [Math]::Abs($a-$b) -lt 1e-10 }
function ScoreBand([double]$s, [double]$tau=0.5) {
    if ($s -gt $tau) { return 'H' }
    if ($s -gt 0.1 -and $s -lt $tau) { return 'L' }
    return 'X'
}
function Birth([int]$frame, [int]$id=1) {
    return @{state='Tracked'; active=($frame -eq 1); start=$frame; last=$frame; len=0; id=$id}
}
function Confirm($t, [int]$frame) {
    $t.state='Tracked'; $t.active=$true; $t.last=$frame; $t.len+=1
}
function Recover($t, [int]$frame) {
    $t.state='Tracked'; $t.active=$true; $t.last=$frame; $t.len=0
}
function SubId($a, $b) { return @($a | Where-Object { $_.id -notin @($b | ForEach-Object { $_.id }) }) }
Check (((@(0.1,0.2,0.5,0.51) | ForEach-Object { ScoreBand $_ }) -join ',') -eq 'X,L,X,H') 'strict partition'
$tau=0.5; $birth=$tau+0.1
Check ((ScoreBand $birth) -eq 'H' -and -not ($birth -lt $birth)) 'birth equality'
Check ((ScoreBand 0.55) -eq 'H' -and 0.55 -lt $birth) 'continue but no birth'
$w=10.0; $h=20.0; $z=@(5.0,10.0,($w/$h),$h)
Check (($z -join ',') -eq '5,10,0.5,20') 'xyah'
$rw=$z[2]*$z[3]; $x1=$z[0]-$rw/2; $y1=$z[1]-$z[3]/2
Check ($x1 -eq 0 -and $y1 -eq 0 -and $rw -eq 10 -and $z[3] -eq 20) 'box round trip'
$m=@(5.0,10.0,0.5,20.0,2.0,3.0,0.01,4.0); $m[7]=0
$next=0..3 | ForEach-Object { $m[$_]+$m[$_+4] }
Check ((Near $next[0] 7) -and (Near $next[2] 0.51) -and $next[3] -eq 20 -and $m[4] -eq 2) 'lost only zeros vh'
$p=1.0/20; $v=1.0/160
Check ((Near ([Math]::Pow(2*$p*$h,2)) 4) -and (Near ([Math]::Pow(10*$v*$h,2)) 1.5625)) 'initial covariance scales'
Check ((Near ([Math]::Pow($p*$h,2)) 1) -and (Near ([Math]::Pow($v*$h,2)) 0.015625)) 'process covariance scales'
Check ((Near (1-0.5*0.6) 0.7) -and (Near (1-0.5) 0.5)) 'fuse score versus raw'
Check (Near (11.0/(121+121-11)) (1.0/21)) 'inclusive touching IoU'
$single=0.1+0.8; $cross=0.7+0.7
Check ($single -lt $cross) 'rejection objective not max cardinality'
Check (Near 0.8 (0.8/2+0.8/2)) 'cost limit equality tie'
Check (((@(15,30,60) | ForEach-Object { [Math]::Truncate($_/30.0*30) }) -join ',') -eq '15,30,60') 'fps buffer'
Check (-not ((31-1) -gt 30) -and ((32-1) -gt 30)) 'strict expiration'
$t=Birth 1
Check ($t.active -and $t.state -eq 'Tracked' -and $t.len -eq 0) 'first frame birth'
$u=Birth 6 2
Check (-not $u.active -and $u.state -eq 'Tracked') 'late birth unconfirmed'
Confirm $u 7
Check ($u.active -and $u.last -eq 7 -and $u.len -eq 1 -and $u.id -eq 2) 'confirmation original id'
$t.state='Lost'; $pool=@($t,$u); $second=@($pool | Where-Object { $_.state -eq 'Tracked' })
Check ($second.Count -eq 1 -and $second[0].id -eq 2 -and $t.active) 'old lost excluded despite active flag'
Recover $t 32
Check ($t.id -eq 1 -and $t.start -eq 1 -and $t.last -eq 32 -and $t.len -eq 0 -and -not ((32-$t.last) -gt 30)) 'reactivate before expire'
$u=Birth 6 2; $u.state='Removed'
Check (-not $u.active -and $u.state -eq 'Removed') 'unconfirmed failure has no buffer'
$t=Birth 1; $t.state='Removed'; $lost=@($t); $removed=@(); $newRemoved=@($t)
$lost=@(SubId $lost $removed); $removed+= $newRemoved
Check ($lost.Count -eq 1 -and $removed.Count -eq 1) 'new removed lingers in lost'
Recover $lost[0] 33
Check ($lost[0].state -eq 'Tracked' -and $removed[0].state -eq 'Tracked' -and $lost[0].id -eq 1) 'residual can reactivate and aliases history'
$lost=@(SubId $lost @($t))
Check ($lost.Count -eq 0) 'tracked subtracted by id'
$lost=@($t); $lost=@(SubId $lost $removed)
Check ($lost.Count -eq 0) 'old removed subtracts next time'
Check (-not (0.15 -lt 0.15) -and (0.149 -lt 0.15)) 'strict duplicate cost'
$trackedDuration=7-3; $lostDuration=5-1
Check (-not ($trackedDuration -gt $lostDuration)) 'duplicate tie drops tracked side'
"BT01 finite checks: $script:checks passed"
```

复现完整系统仍需锁定Python／NumPy／SciPy／lap／cython_bbox／PyTorch版本，逐帧保存第一／二／确认阶段矩阵、匹配与未匹配索引、全部生命周期字段。测试至少覆盖 `(0,0)/(0,M)/(N,0)`、纯低分无轨迹、阈值临界、Lost被高分找回、Lost仅低分、unconfirmed失败、超时前后与Removed残留、去重年龄相等及一对多重叠、非法框、空帧时钟、跨序列ID和输入数组复用。最终再用真实固定检测评估ID切换、碎轨、IDF1/HOTA和尾延迟；本文检查不能替代这些验收。

### 贯穿 L1–L5 的五问自测

| 层级／问题 | 答题要点 | 前提、易错点与证据入口 |
|---|---|---|
| L1：τ=0.5时，s=0.1／0.5／0.55／0.6分别能做什么？晚出生怎样输出？ | 前两者都不入池；0.55仅高分关联／确认不能出生；0.6达到计算得到的出生门槛可出生；非首帧出生需下次剩余高分确认 | s与阈值按同dtype判断；“0.5进低分”“高分就新生”“低分可确认”均错。见L1区间、三阶段表与源码update |
| L2：为何不是把全部检测一起做LAP？旧Lost能否用低分重激活？ | 高分有优先权，二阶段只含仍Tracked的未匹配项；低分不出生、unconfirmed独立；旧Lost只在高分阶段重激活，二阶段else通常不可达 | 以固定路径为准，不把论文抽象remaining直接当全体Lost；没有BoT-SORT的GMC／Re-ID。见L1伪代码、L2机制 |
| L3：把buffer从30改60就能适配低帧率吗？为什么视频实际保留时间不符？ | B按传入frame_rate/30缩放，KF仍dt=1；查demo硬编码30、None帧跳过、真实时钟、关联后过期与严格大于 | buffer控制保留不是运动模型或Lost输出；先核帧时钟再调参数，固定检测验证误接。见L3参数与demo |
| L4：IoU=0.5、分数0.6为何代价0.7？LAP是不是总挑最多合法边？ | C'=1-IoU*s；给出虚节点拒配目标，合法边也有竞争；0.1单边加拒配0.8优于两条0.7；C=λ可同价 | 虚节点矩阵推导限lap v0.5.12；非历史包保证，IoU还须核inclusive约定。见L4及来源8／9 |
| L5：Removed为何还能短暂进主池？去重为什么可能删当前输出？ | 先减旧Removed再追加本帧Removed，Lost留引用；主池不按state过滤，可走re_activate；跨Tracked/Lost去重同持续期删Tracked且不mark_removed | 区分枚举、is_activated与容器成员；先修语义再逐帧差分，不称原码吸收态可靠。见L1次序、有限检查和源码末尾 |

## 来源、版本与获取深度

以下均为 **2026-11-06** 联网核对；B=书目、A=摘要、T=实际读到方法／代码、F=获取失败。T不表示完整依赖运行或全部实验审核。

<a id="source-1"></a>

**[cite:1] 论文书目，P0，B+A。** [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)：题名、当前作者表、v1/v2/v3日期及论文作者仓库链接。当前摘要成绩不用于本仓库实测。

<a id="source-2"></a>

**[cite:2] 2021初版原文，P0，T（部分）。** [固定v1 HTML](https://arxiv.org/html/2110.06864v1)：读到作者栏、引言、相关工作、Algorithm 1和§3开头；长响应在§3截断，带章节锚点再次获取仍未取得后部，不声称已读v1完整§3–5。Algorithm 1可见高／低阈值、IoU两阶段及出生阈值ε；其 `if >τ_high / else if >τ_low` 和 `score>ε` 是论文抽象条件，不能覆盖成固定代码的双严格区间与出生 `>=τ+0.1`。HTML有数学重复转写，未逐页对照v1 PDF。

<a id="source-3"></a>

**[cite:3] ECCV 2022正式正文，P0，T。** [Springer正式章节](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_1)，DOI如前；读取§3 BYTE、§4.1配置、§4.2相似性和数据集讨论。确认第一阶段可用IoU或Re-ID、二阶段IoU、Lost保留／不输出及YOLOX系统；BDD100K使用外观且相关配置不用Kalman。Algorithm 1在出版页是图片，本卡不声称逐字符读出该图片；文字段落没有固定代码的unconfirmed／容器清理全部细节。后部响应截断，未核全部表格／附录／训练资产。

<a id="source-4"></a>

**[cite:4] 固定作者系统代码，P0，T。** 先用 [commits API](https://api.github.com/repos/ifzhang/ByteTrack/commits/main)取得SHA、作者与2022-12-11提交时间，再读取[固定byte_tracker.py原文](https://raw.githubusercontent.com/ifzhang/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py)。完整读取STrack／BYTETracker及三个列表辅助函数；提交元数据不证明每行写于同日，不是运行证据。

<a id="source-5"></a>

**[cite:5] 关联实现，P0，T。** [固定matching.py原文](https://raw.githubusercontent.com/ifzhang/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/matching.py)：实际读取IoU依赖调用、linear_assignment、fuse_score及未走入主流程的外观／运动门函数。没有本地LAP求解器执行。

<a id="source-6"></a>

**[cite:6] 状态与Kalman，P0，T。** [固定basetrack.py](https://raw.githubusercontent.com/ifzhang/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/basetrack.py)；[迁移入口同SHA的kalman_filter.py](https://raw.githubusercontent.com/FoundationVision/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/kalman_filter.py)。读取枚举／ID／end_frame及完整KF初始化、预测、project、update。原ifzhang路径首次抓取返回空内容（F），终端直连尝试无法连接（F）；改用迁移入口成功，未把失败当作原码缺失。

<a id="source-7"></a>

**[cite:7] 调用层与依赖清单，P0，T。** [固定demo_track.py](https://raw.githubusercontent.com/ifzhang/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/tools/demo_track.py)和[固定requirements.txt](https://raw.githubusercontent.com/ifzhang/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/requirements.txt)。核对默认参数、输入／输出路径、None帧与frame_rate差异；清单不是全环境lockfile，未安装依赖或加载权重。

<a id="source-8"></a>

**[cite:8] IoU依赖源码，P0，T。** [cython_bbox固定提交4874834e6de5c3a07c68d027edcaa62012d76ef0](https://raw.githubusercontent.com/samson-wang/cython_bbox/4874834e6de5c3a07c68d027edcaa62012d76ef0/src/cython_bbox.pyx)，由[仓库API](https://api.github.com/repos/samson-wang/cython_bbox/commits/master)解析，日期2023-11-23，源码内版本0.1.5。核对inclusive `+1`及float64，不把2023依赖快照说成2022实验已锁版本；未运行Cython扩展。

<a id="source-9"></a>

**[cite:9] LAP依赖源码，P0，T。** [lap v0.5.12扩展实现](https://raw.githubusercontent.com/gatagat/lap/v0.5.12/_lapjv_cpp/_lapjv.pyx)核对有限cost_limit的 `(n+m)` 虚节点布局、λ/2代价与-1未匹配映射；另读 [v0.4.0标签源码](https://raw.githubusercontent.com/gatagat/lap/v0.4.0/lap/_lapjv.pyx)可见不同扩展布局。只作显式依赖版本的代数说明，未核作者历史安装wheel、未执行后端、未证明tie-break跨版本一致。

## 验证记录与销项建议

本次只新增本文件。检索仓库Python中的 `BYTETracker|STrack|ByteTrack|bytetrack` 未发现实现；没有修改模块、测试脚本、requirements、导航或台账。当前可发现的Python命令为WindowsApps入口，本文不依赖Python可运行性，也未为文档安装算法依赖。

实际从Markdown提取PowerShell代码到内存执行，结果为 **`BT01 finite checks: 26 passed`**。首次检查因函数名Group与PowerShell别名冲突而失败，改名ScoreBand后全部通过；不是算法分支故障。UTF-8解码、五层标题、8张表列数、围栏配对、41处本地链接／锚点和尾空白检查通过，另计20处远端链接，不把计数当成所有URL再次访问。`git diff --check`通过；本文件未跟踪，已另作直接文件检查，不仅依赖Git差异检查。

开始时枚举106个既有非忽略文件，但终端调用未保留内存哈希表，未取得有效的前后内容指纹对比，故不声称已完成全工作树哈希不变证明。写入操作仅针对本文件，未回退用户已有改动。有限测试没有执行完整KF、LAP、Cython、YOLOX、MOT视频或现场控制；文档通过不升级这些验证状态。

**建议G48／12-09的“具体方法缺卡”转为“已有”，保留P2深度缺口；BT01五层为核／核／部／核（限定模型代数）／部，答5，未实现。** 仍需锁定作者实际依赖与检测器资产、完整多目标逐帧差分、Removed残留／去重与坐标边界加固、真实视频身份及性能验收。此次未写台账／总览／索引，旧“缺卡”尚未同步，不能宣称已正式销账或重算全库覆盖率。跟踪输出也不等于机器人控制输入：坐标标定、时间同步、控制与独立安全层须另行实现和验收。
