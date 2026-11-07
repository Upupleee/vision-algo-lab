# Deep SORT 算法卡

[学习规范](../algorithm_learning_standard.md) · [tracking 总览 E08](../tracking_guidance_evolution.md#38-e08deep-sort给关联增加外观证据) · [覆盖台账 G47／12-08](../evolution_coverage.md#12-跟踪运动视觉引导) · [SORT SO01](sort.md#so01) · [指派 H01](hungarian.md#h01)

## 范围、版本与覆盖状态

核对日期：**2026-11-06**；本地基准：HEAD `918a03e` 加当前工作树。唯一卡键 **`deep_sort.md:DS01`**，承接 G47／12-08“Deep SORT 外观训练、运动门与级联关联”。本文只补具体方法卡，不以 Kalman、Hungarian 或普通外观相似度组件代替系统。

方法为 Nicolai Wojke、Alex Bewley、Dietrich Paulus 的 *Simple Online and Realtime Tracking with a Deep Association Metric*，**ICIP 2017，3645–3649**，DOI `10.1109/ICIP.2017.8296962`；实际读取 arXiv **1703.07402v1，2017-03-21**。作者跟踪代码固定为 `nwojke/deep_sort` **`f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab`**，提交时间 **2025-03-02**；不是2017发表时的软件快照。[cite:1](#source-1)[cite:2](#source-2)

训练来源另固定为 `nwojke/cosine_metric_learning` **`eda0daaa5462c61ac44553f2151070fd7e316cc8`**，提交时间 **2018-10-28**，对应 Wojke、Bewley 的 *Deep Cosine Metric Learning for Person Re-identification*，**WACV 2018**，DOI `10.1109/WACV.2018.00087`。2017跟踪论文、2018训练论文、2018训练代码、2025跟踪代码、公开权重文件是不同版本证据，不能合写为“2017原文完整训练实现”。[cite:8](#source-8)[cite:9](#source-9)

| 字段 | 状态与限定 |
|---|---|
| 演进定位 | 在线 tracking-by-detection；在 SORT 基础上改变框状态、关联证据、关联顺序与生命周期，不是简单给七维 SORT 加一个 embedding |
| L1 | 已核验：接口、八维状态、预测／两阶段关联／更新顺序及伪代码，对应固定源码 |
| L2 | 已核验：运动门、gallery 最近邻、年龄级联、IoU 回退和状态机的因果作用 |
| L3 | 部分：参数量纲、调参顺序与失效诊断明确；没有真实视频阈值标定和部署剖析 |
| L4 | 已核验（限定模型代数）：xyah、F/H、噪声尺度、白化门控、余弦／训练梯度及指派差异；不保证身份正确性 |
| L5 | 部分：固定核心源码映射、边界与有限测试；没有运行 CNN、训练、完整应用或 MOT 评测 |
| 面试自测 | 贯穿 L1–L5 的5问，含答案、成立条件、易错点与证据入口 |
| 实现状态 | **未实现**：本仓库没有 Deep SORT／ReID 模块；本文伪代码和内存执行外部核心测试不算本地产品实现 |
| G47 销项口径 | DS01 补齐“缺专卡／训练来源／运动门／级联”的内容缺口，可供后续将机制覆盖转“已有”、保留工程验收P2；本次不改台账、总览或索引，不声称已完成它们的状态同步 |

## DS01 Deep SORT：学习外观辅助的在线关联

<a id="ds01"></a>

### 任务、输入输出与演进分支

先前瓶颈 → 改变维度 → 机制 → 新代价 → 遗留问题：SORT 仅靠框重叠，在交叉、漏检和相机运动后身份证据不足；Deep SORT 增加离线学习的行人外观表示，以运动门排除不可信位置、gallery 最近邻衡量外观，并优先匹配最近见过的轨迹；新增网络推理、特征内存、训练域及阈值校准成本；仍依赖检测、正确时间和可区分外观，不保证任意遮挡后恢复，不负责机器人控制。[cite:1](#source-1)

核心输入是每帧 `List[Detection]`，每个对象包含 `tlwh=[left,top,width,height]`、`confidence`、`feature`。固定 `Detection` 将框转 `float64`、特征转 `float32`；使用同一原图像素系、连续宽高差、无 `+1`。正常工作域要求有限坐标、w>0、h>0、同维且有限非零特征；这些是调用契约，**不是原码已有的完整校验**。官方网络的特征维度为128，但匹配类本身不写死128；不能让一部分检测128维、一部分空维。[cite:3](#source-3)[cite:7](#source-7)

`Tracker.predict()` 和 `Tracker.update(detections)` 分开，必须每帧依次各调用一次，包括 `update([])`；预测时钟固定 dt=1次调用，不读取真实时间戳。`update` 修改 `tracker.tracks`，**不返回 SORT 式 K×5数组**。应用再选择 confirmed 且 `time_since_update<=1` 的轨迹，转 `tlwh` 保存 `[frame,id,left,top,w,h]`，写 MOT 文件时补 `1,-1,-1,-1`。因此应用会输出已确认轨迹的**第一次漏检预测框**；这与 SO01 仅输出当前帧匹配轨迹不同。[cite:3](#source-3)[cite:6](#source-6)

官方预生成文件为 N×138：前10列保留 MOT 检测字段，后128列为特征；检测框来自 POI 工作，外观替换为作者网络。框检测、ReID 推理、跟踪和结果展示是不同阶段；`Tracker` 不做检测、NMS、类别过滤或置信度加权。应用先按置信度／高度过滤并 NMS；没有内建跨类别禁止关联。流结束时停止帧调用，不作离线全序列优化，也不自动补尾部轨迹。[cite:2](#source-2)[cite:6](#source-6)

### L1 步骤：完整逐帧链与离线训练链

#### 八维 xyah，而非 SORT 七维面积状态

令状态与观测分别为

\[
\mathbf x=[x,y,a,h,v_x,v_y,v_a,v_h]^T\in\mathbb R^8,
\qquad \mathbf z=[x,y,a,h]^T\in\mathbb R^4.
\]

x,y 是框中心，a=w/h 是宽高比，**不是面积**；h 是高度。x,y,h 单位为 px，a 无量纲；vx,vy,vh 为 px/frame，va 为 1/frame。四个观测分量都有对应速度。SO01 的 `[u,v,s,r,dot_u,dot_v,dot_s]` 是另一模型，不能复用其面积平方根逆变换。[cite:1](#source-1)[cite:4](#source-4)

#### 固定代码的系统伪代码

下列是正常有限输入下的源码语义展开；KF 矩阵见 L4。`LAP_MIN` 表示 SciPy 矩形线性指派，保持原码“截顶 → 指派 → 后拒配”，不擅自改成最大合法匹配。集合中的元素始终是原始轨迹／检测索引，而非某一子矩阵行列号。[cite:3](#source-3)[cite:5](#source-5)

```text
MIN_COST(metric, T, D, tau):
    if T or D is empty: return [], T, D
    C = metric(T, D)
    C[C > tau] = tau + 1e-5
    pairs = LAP_MIN(C)
    matches = pairs with C[row,col] <= tau, mapped back to original indices
    unmatched = indices absent from accepted matches
    return matches, unmatched_T, unmatched_D

GATED_APPEARANCE(T, D):
    C[i,j] = min cosine_distance(sample, detection[j].feature)
             over sample in metric.samples[track[i].track_id]
    for i in T:
        d2 = KF_SQUARED_MAHALANOBIS(track[i], detections[D].xyah)
        C[i, d2 > chi2inv95[4]] = 1e5
    return C

MATCH_FRAME(detections):
    confirmed, tentative = partition tracks by confirmed state
    U = all detection indices
    A = []
    for level in 0 .. max_age-1:
        if U is empty: break
        T = confirmed tracks with time_since_update == level+1
        accepted, _, U = MIN_COST(GATED_APPEARANCE, T, U, cosine_threshold)
        A += accepted
    RA = confirmed tracks not matched in A
    TI = tentative + tracks in RA with time_since_update == 1
    old_unmatched = tracks in RA with time_since_update != 1
    B, RI, U = MIN_COST(IOU_COST, TI, U, max_iou_distance)
    return A+B, old_unmatched union RI, U

IOU_COST(T, D):
    C[i,j] = 1e5 if track[i].time_since_update > 1
             else 1 - IoU(track[i].predicted_tlwh, detection[j].tlwh)
    return C

FRAME(detections):
    for each existing track:
        mean, covariance = KF_PREDICT(mean, covariance)
        age += 1
        time_since_update += 1
    M, UT, UD = MATCH_FRAME(detections)
    for (t,d) in M:
        KF_UPDATE(track[t], detection[d].xyah)
        track[t].features.append(detection[d].feature)
        hits += 1
        time_since_update = 0
        if state == Tentative and hits >= n_init: state = Confirmed
    for t in UT:
        if state == Tentative: state = Deleted
        else if time_since_update > max_age: state = Deleted
    for d in UD:
        create KF state from detection[d].xyah with zero velocities
        create track with next_id, hits=1, age=1, time_since_update=0
        state = Tentative
        features = [detection[d].feature]
        next_id += 1
    remove Deleted tracks
    active_ids = IDs of all Confirmed tracks, including temporarily unmatched
    for each Confirmed track:
        append cached features to metric.samples[id]
        if budget is not None: samples[id] = samples[id][-budget:]
        clear track.features
    keep only active_ids in metric.samples
    application saves tracks with Confirmed and time_since_update <= 1
```

`features` 是未送入度量库的暂存，不等于整个 gallery。Tentative 的样本先缓存；确认帧把出生及后续匹配样本一起送入 gallery。Confirmed 轨迹每次匹配后增加样本；无匹配不新增，只保留旧 gallery；删除后该 ID 的 gallery 被清除。`partial_fit` 在这里仅维护最近邻样本集合，**没有梯度下降、没有更新 CNN，也没有在线训练新的距离网络**。所有本帧关联完成后才写新特征，避免用待匹配检测先更新自身再证明相似。[cite:3](#source-3)

#### 生命周期与计数边界

| 状态／计数 | 固定代码语义 | 结果与易错点 |
|---|---|---|
| 出生 | `hits=age=1`、`time_since_update=0`、Tentative；实例 `_next_id` 从1递增 | 出生算一次命中，无 SORT 全局 warmup |
| 预测 | `age` 和 `time_since_update` 各加1 | 总年龄 `age` 不用于 matching cascade；论文 age 对应源码 tsu |
| Tentative 匹配 | hits加1；`hits>=n_init` 时转Confirmed | 默认n_init=3，连续第3次检测即确认；前两次依赖IoU关联 |
| Tentative 漏检 | 当帧直接Deleted，与max_age无关 | 无“先累计3个分散命中再确认”；任何早期漏检都终止 |
| Confirmed 漏检 | 仅在 `tsu>max_age` 时删除 | 等于max_age仍保留；不退回Tentative，hits不会因漏检清零 |
| 再获匹配 | 更新后tsu=0，原ID与Confirmed保持 | 不像SO01重建hit_streak；但关联失败仍可能创建新ID |
| 默认max_age=30 | cascade只检查预测后tsu=1…30，IoU仅检查tsu=1 | 第30次漏检帧末可留存；下一帧tsu=31已无正常关联资格，随后删除，不能照搬SORT“超龄同帧可救回”的结论 |
| n_init=1或0 | 构造函数仍直接写Tentative，只有后续update才检查确认 | 不会出生即确认；最早第二次检测才确认；应限制／明确参数合同 |
| ID与输出 | 每个Tracker从1起编号；应用还需confirmed且tsu≤1 | 不同实例可能重号；长遮挡可内部保留却没有输出，删除后不恢复旧ID |

以上以每帧一次predict/update、默认正max_age为条件。max_age=0等异常参数会改变级联与IoU组合，不作为正常推荐设置。`mark_missed` 不再递增tsu，避免双计时。[cite:3](#source-3)[cite:5](#source-5)[cite:6](#source-6)

#### ReID 离线训练、导出与在线使用

1. **数据来源与切分。** 2017论文§2.4描述在大规模行人重识别数据上训练，数据规模为超过110万张、1261人；与其MARS来源及作者提供的 `mars-small128.pb` 配套说明相核对。MARS官方给出总计1261个身份，训练625个、测试636个，训练框509,914张。**全库人数／图像数不是实际参加训练的数量**；测试身份不应进入训练或阈值选择。[cite:1](#source-1)[cite:2](#source-2)[cite:10](#source-10)
2. **作者训练实现。** 2018固定训练仓库支持MARS和Market1501；`train_mars.py` 从官方训练划分另留10%的**身份**做验证，种子1234，`datasets/util.py:create_validation_split` 按去重身份抽取，不是随机拆10%的相邻帧。它不证明已下载权重采用完全相同的训练轨迹；README的“10% data”应以函数体细化。[cite:8](#source-8)
3. **网络与预处理。** 输入NHWC `B×128×64×3`，两层32通道卷积、池化、三组共六个残差块，空间下采样到16×8、通道128，flatten＋全连接128、归一化得到单位特征。2018代码用ELU、BN、训练dropout keep_prob=0.6；解码后缩放到128×64、除255、训练随机水平翻转。冻结接口明确BGR→RGB，再除255；不得在调用端重复交换通道或重复缩放。[cite:1](#source-1)[cite:8](#source-8)
4. **优化目标。** 默认 `cosine-softmax`：归一化特征与类别权重，乘可学习正scale，用身份分类交叉熵训练；MARS入口每ID取4图，默认batch=128、Adam学习率0.001，另含正则项。仓库也有triplet／magnet可选模式，但不能因此写成默认联合triplet训练。监控其他loss与将其纳入优化目标也不同。训练停止、选checkpoint及阈值标定须按独立验证协议；默认 `number_of_steps=None` 不是已收敛证明。[cite:8](#source-8)[cite:9](#source-9)
5. **导出与推理。** 关闭训练态BN/dropout、去掉训练身份分类头，保留128维归一化embedding。`train_mars.py --mode=freeze` 是作者导出路径；训练身份分类器不输出在线track_id。跟踪时网络冻结，仅提取检测ROI特征与更新gallery；离线MARS tracklet检索的平均池化再归一化，不是在线DS01的gallery最小距离。[cite:2](#source-2)[cite:8](#source-8)

2017原文方法部分没有完整展开上述cosine-softmax优化器与超参数；训练细节来源于**后续作者训练仓库**。尚未下载、校验或运行 `mars-small128.pb`，未取得其训练日志／准确checkpoint血缘；不宣称本次证明了2017实验与当前权重逐bit一致。

### L2 机制：门控、gallery、级联与回退

**运动门与外观不是二选一。** 平移相似但外观不同可以被外观拒绝；外观相似但位置不可能可以被运动门拒绝。运动门使用观测创新及投影协方差，衡量标准化残差，而不是把8维状态直接与4维框相减。相机运动未建模时正确候选也可能被门挡住，外观再好不能在该阶段穿越门；但符合IoU回退资格时，第二阶段可以另行匹配。[cite:1](#source-1)[cite:4](#source-4)

**gallery取最小值，不取平均。** 目标转身后，当前特征可能更像旧姿态样本；最近邻保留多模态外观。反面是一次误配写入错误样本后，未来也可能因为“有一个很像”而接受。固定gallery添加样本不会增大最近邻距离，但FIFO删旧样本可能使距离上升。预算越大不保证越准；它增加计算、内存和污染留存机会。没有置信度、遮挡质量或时间衰减加权。[cite:3](#source-3)

**级联以最近观测优先。** 对相同创新，旧轨迹P变大使Mahalanobis距离变小，反而容易争抢检测。级联先为预测后tsu=1的confirmed轨迹求解，再处理tsu=2…max_age；较老轨迹只能使用剩余检测。同一层仍是全局一对一，不是按列表顺序逐轨贪心；跨层不是一张大矩阵的总成本最优。即使旧轨迹外观成本更低，也不撤销已被新近轨迹拿走的检测。“年龄”不是总寿命，不是ID大小，也不是未匹配次数的终身总和。[cite:1](#source-1)[cite:5](#source-5)

**IoU回退是受限的另一轮关联。** 候选为所有未确认轨迹＋外观阶段未配上的confirmed且tsu=1，检测为尚未分配的检测。原码该轮只用 `1-IoU`，不再次调用Mahalanobis门或余弦门。因此外观突然变差、甚至第一轮运动门拒绝，也可能凭足够重叠恢复；这不是所有旧轨迹都能“兜底”，tsu>1的confirmed不进入。IoU误配后样本仍会进gallery，属于污染风险；额外加双门是可评估的行为修改，不是等价优化。[cite:3](#source-3)[cite:5](#source-5)

**生命周期把证据与资源分开。** Tentative要求连续初始证据，Confirmed容忍短失配，删除回收状态与gallery。保留max_age帧不代表每帧输出，也不代表跨出画再进入能复用已删除ID。相同外观零件、全遮挡后位移太大、重复检测或错误框仍能换ID；ReID嵌入不是身份认证。

### L3 工程权衡：参数、诊断及论文代码差异

#### 参数与推荐核查顺序

| 参数／接口 | 固定值或默认值 | 单位、代价与核查依据 |
|---|---|---|
| `max_cosine_distance` | 应用默认0.2 | 无量纲距离上限，等价理想单位向量dot≥0.8；不是匹配概率80%。须在独立同域身份对及固定检测上校准 |
| `nn_budget` | 类与CLI默认None；README示例显式100；论文gallery=100 | 每个ID的样本数，不是帧时长；None不截断，0的Python切片 `[-0:]` 也不截断，不能拿0表示禁用外观 |
| `max_iou_distance` | Tracker默认0.7 | 上限为1-IoU，理想实数下IoU≥0.3；不能误写IoU必须≥0.7 |
| `max_age` | 默认30 | 未更新帧调用数；也决定cascade层数，不是总年龄上限、秒或跨序列身份保留期 |
| `n_init` | 默认3 | 包括出生的连续检测数；更大减少短假轨但增加输出等待；0／1存在构造与确认分支差异 |
| 运动门 | 4D 9.4877；可选only_position时2D 5.9915 | 平方Mahalanobis的95%分位，不是sqrt距离门，也不是8D 15.507；主Tracker未开启only_position |
| Q/R尺度 | position=1/20、velocity=1/160，相对h | 标准差权重，不是协方差本身；几何尺寸、帧率与相机运动改变适用性 |
| 置信度过滤 | CLI默认0.8，README运行例用0.3 | 应用以 `>=` 保留；不控制P、Q、R，不能误认为低分自动以大R更新 |

顺序：先核对帧号／实际dt与ROI坐标 → 检测质量、类别和去重 → 特征维度、模型版本、预处理与有限性 → 分开记录运动门与外观门 → 校准外观阈值和噪声 → 调n_init/max_age/budget → 剖析网络、距离矩阵与LAP。上述是工程建议，不是未经实测的万能最佳参数。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)[cite:6](#source-6)

| 症状 | 先留证据 | 对照与处理原则 |
|---|---|---|
| 遮挡后外观很像仍断轨 | 预测前后tsu、4D创新、S、dM²与门掩膜 | 查是否过龄、相机运动或框比例／高度触门；不只降余弦阈值 |
| 交叉后错误身份不断自我强化 | 两阶段匹配来源、gallery写入时间、最近邻样本 | 查IoU回退污染与重复检测；冻结低质量样本是另立的加固方案 |
| ReID离线效果好但在线很差 | ROI、BGR/RGB、缩放、范数、模型校验和、训练／现场域 | 单位范数不证明语义正确；行人模型不能直接认证同款工业件 |
| 内部ID存在但结果文件没有 | confirmed、tsu、max_age与输出谓词 | tsu=2…30默认可保留但不输出，不是“程序丢了track对象” |
| 内存持续上涨 | budget实际参数、每ID样本数、active_ids、漏帧调用 | None／0不截断、删除未执行、超大max_age分别排查 |
| 偶发NaN或求解器报错 | 首个非法特征／框、成本矩阵与协方差正定性 | 拒绝无效输入并追因；不要把NaN替换为零成本 |

#### paper 与固定代码不能混写的地方

| 主题 | 2017 paper | 固定作者代码 |
|---|---|---|
| 距离融合 | 式(5) `lambda*dM²+(1-lambda)*dcos`；式(6)两门相乘；文中认为相机运动明显时lambda=0合理 | `gated_metric`直接取gallery余弦距离，运动超门设1e5；没有lambda超参数、没有非零运动代价加权。对应lambda=0的主要设计，但还包含截顶／后拒配细节 |
| 可行候选 | Algorithm 1先指派，再按B留下可行对 | motion gate先写大数，再将所有大于当前tau的成本改成tau+1e-5，LAP后再拒配；不等价于任意可行边最大匹配 |
| gallery | 最近100个描述子 | budget可选，默认None；必须显式设100才有相应截断；并有Tentative缓存与confirmed入库规则 |
| 状态估计 | 8维常速＋4维直接观测，未逐项列所有经验噪声 | 自有Kalman实现，Q/R随h；Cholesky求解；不是SO01的FilterPy参数 |
| 网络训练 | 离线行人重识别、128维归一化、网络结构 | README链接后续2018cosine metric training；不能把所有训练实现细节追认成2017正文已给出 |
| IoU回退／输出 | 最终用未确认及最近失配轨迹做IoU关联 | 回退集合和`tsu>1`拒绝明确；应用另输出confirmed且tsu≤1。输出规则不能由“活跃轨迹”一词推出 |

上表是不同证据层的语义核对，不是判定论文错误；同作者论文／仓库不构成独立性能复现。[cite:1](#source-1)[cite:3](#source-3)[cite:5](#source-5)[cite:8](#source-8)

### L4 数学推导：状态、门控与学习距离

#### 坐标变换、可观测性与噪声尺度

对输入 \(b=(l,t,w,h)\)，正宽高域内

\[
z=(l+w/2,\ t+h/2,\ w/h,\ h)^T,
\quad w=ah,\quad l=x-ah/2,\quad t=y-h/2.
\]

局部观测误差传播为 \(R_z\approx J\Sigma_bJ^T\)，其中

\[
J=\frac{\partial z}{\partial(l,t,w,h)}=
\begin{bmatrix}1&0&1/2&0\\0&1&0&1/2\\0&0&1/h&-w/h^2\\0&0&0&1\end{bmatrix}.
\]

这是本文条件推导，不是作者实际使用的R。高度接近0时比例方向病态，角点／宽高的高斯噪声不自动变为独立的xyah高斯噪声。

固定一步模型为

\[
F=\begin{bmatrix}I_4&I_4\\0&I_4\end{bmatrix},\quad
H=[I_4\ 0],\quad
x^- = Fx^+,\quad P^-=FP^+F^T+Q.
\]

单帧H秩为4，速度不可观测；两时刻 \([H;HF]\) 秩为8（非零已知步长），理想正确关联下可辨识四个速度。这个秩条件不保证交叉目标身份可辨识。若改成真实dt，F右上块应为dtI，并重建相应噪声离散模型；不应只重命名速度单位。

记 \(p=1/20,v=1/160\)，源码用下列**标准差向量逐项平方**构造对角阵：[cite:4](#source-4)

\[
\sigma_{P_0}=[2ph,2ph,10^{-2},2ph,10vh,10vh,10^{-5},10vh],
\]
\[
\sigma_Q=[ph,ph,10^{-2},ph,vh,vh,10^{-5},vh],\qquad
\sigma_R=[ph,ph,10^{-1},ph].
\]

P0用出生测量h；predict的Q用**预测前**状态h；project的R用投影时状态h。初始化均值为 `[z,0,0,0,0]`。各项单位为对应状态／测量单位，协方差是其平方；例如比例观测方差为0.01，不是0.1。h为负时平方仍为正不意味着框合法，模型没有正宽高约束。

令 \(e=z-Hx^-\)，\(S=HP^-H^T+R\)。在正确关联、线性模型和匹配噪声假设下，最小化线性更新误差协方差的迹给出

\[
KS=P^-H^T,\quad K=P^-H^TS^{-1},\quad
x^+=x^-+Ke,\quad P^+=P^--KSK^T.
\]

作者实际用 `cho_factor/cho_solve` 求K，不显式求逆；协方差用减法式，**不是Joseph形式**。浮点中可失去对称／正定性；Joseph、对称化、jitter等是另行验证的数值方案，不能混作原码。P只描述给定模型和关联下的不确定性，不包含完整身份分配不确定性，不是实际误差的确定性上界。

#### 卡方维度来自观测残差，不来自状态长度

若正确匹配的 \(e\sim\mathcal N(0,S)\)、S正定，Cholesky \(S=LL^T\) 白化得 \(q=L^{-1}e\sim\mathcal N(0,I_d)\)，所以

\[
d_M^2=e^TS^{-1}e=\|q\|^2\sim\chi_d^2.
\]

默认xyah残差d=4，门为 \(d_M^2\le9.4877\)。`only_position=True` 先将均值、测量取前2维、S取左上2×2子阵，故d=2、门5.9915；不能只换门限而仍计算4维残差，也不能用8维15.507。95%是正确模型下单个真实匹配残差落门内的名义概率，不是“95%的配对身份正确”，不控制多候选误配率。Q/R启发式、遮挡及相机运动下该概率需实际残差验证。[cite:1](#source-1)[cite:4](#source-4)[cite:5](#source-5)

年龄动机也可由式子看出：同一e且 \(S_{old}=kS_{recent},k>1\)，旧轨迹 \(d_{M,old}^2=d_{M,recent}^2/k\) 反而更小。真正高斯负对数似然还含 \(\tfrac12\log\det S\)，而平方Mahalanobis单独没有此项。级联是一种最近观测优先的启发式，并非严格贝叶斯多假设求解或通用最优性定理。

#### 余弦 gallery、训练目标与梯度

对非零特征f与g，归一化 \(r=f/\|f\|,s=g/\|g\|\)，

\[
d_{cos}(r,s)=1-r^Ts=\tfrac12\|r-s\|^2\in[0,2],
\qquad d_{ij}=\min_{s\in\mathcal R_i}d_{cos}(r_j,s).
\]

范围为0…2而非0…1；代码没有裁剪浮点舍入后的负小数。距离0只表示特征方向相同，不证明真实身份相同。gallery取min不同于平均向量，单位范数的均值也不一定仍为单位向量。虽然这个公式等于一半平方欧氏距离，不能把未归一化f、g的欧氏距离阈值直接搬来。

2018作者实现将训练身份的权重列向量归一化 \(\bar w_k=w_k/\|w_k\|\)，正尺度 \(\kappa=\operatorname{softplus}(s)\)，构造

\[
\ell_k=\kappa\bar w_k^Tr,\quad
p_k=\frac{\exp\ell_k}{\sum_c\exp\ell_c},\quad
\mathcal L_{CE}=-\log p_y.
\]

不含普通softmax的自由类别偏置项；归一化使角度成为可学习判别量，scale控制概率尖锐程度。忽略额外正则、对非零f，有

\[
\frac{\partial\mathcal L}{\partial r}
=\kappa\left(\sum_kp_k\bar w_k-\bar w_y\right),\qquad
\frac{\partial r}{\partial f}=\frac{I-rr^T}{\|f\|},
\]
\[
\frac{\partial\mathcal L}{\partial f}
=\frac{\kappa}{\|f\|}(I-rr^T)
\left(\sum_kp_k\bar w_k-\bar w_y\right).
\]

即梯度经单位球切空间投影，类别权重对同类方向提供监督。此推导不保证非凸训练全局收敛，不保证新身份／新领域仍可分；当f趋零时，理想归一化的雅可比病态，框架epsilon只改变数值定义，不能创造身份信息。这里的学习目标属于后续训练来源，不冒称2017跟踪论文完整推导。[cite:8](#source-8)[cite:9](#source-9)

#### 指派目标、截顶后拒配与复杂度

每层n在轨迹集合Tn、剩余检测U上先解固定基数矩形指派：

\[
\min_X\sum_{ij}\widetilde C_{ij}X_{ij},\quad
X_{ij}\in\{0,1\},\quad\sum_jX_{ij}\le1,\quad\sum_iX_{ij}\le1,
\quad\sum_{ij}X_{ij}=\min(|T_n|,|U|),
\]

其中 \(\widetilde C=C\) 若C≤tau，否则tau+1e-5；再删除所选中超tau的边。它**不保证最大合法匹配基数**。纯代价算术反例：tau=0.2，\(C=\begin{bmatrix}0&0.19\\0.19&1\end{bmatrix}\)，截顶后对角总成本0.20001、交叉0.38；指派选对角，后拒配只留1对，但交叉两条都合法。此为目标函数的代数反例，不是实测视频／伪造数据，不宣称所有这类成本均来自同一个合法框配置。先门控＋允许拒配的另一优化目标不能直接替换而称“结果等价”。[cite:5](#source-5)

M条轨迹、N个检测、每ID至多B个特征、维度D=128时，滤波固定维O(M)，所有gallery最近邻最坏O(MNBD)，样本存储O(MBD)，成本矩阵O(MN)；当前实现按目标逐个计算B×N距离，临时内存与B、N有关。稠密每层LAP可用O(max(mn,nn)³)粗略上界，总体应求各层上界之和，而不是把一次全局LAP复杂度机械当整条流水线；最多还循环max_age层并做一次IoU回退。CNN前向、ROI裁剪和I/O另计，None预算会使B随历史命中增长。本文**未提供通用理论误差界或身份收敛保证**，没有将论文Hz／ID切换改善当作本机测量。

### L5 实现：源码映射、异常边界与有限测试

#### 固定源码入口

以下均使用固定提交链接；源码注释／docstring与函数体不一致时按函数体，例如 `distance` 的“squared distance”泛称不能证明余弦返回平方距离。[cite:3](#source-3)[cite:5](#source-5)

| 固定源码 | 核对入口 |
|---|---|
| [detection.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/detection.py) | float64框、float32特征、to_xyah／to_tlbr |
| [kalman_filter.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/kalman_filter.py) | initiate、predict、project、update、gating_distance及卡方表 |
| [nn_matching.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/nn_matching.py) | 行归一化、min(axis=0)、samples、预算切片／active_targets清理 |
| [linear_assignment.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/linear_assignment.py) | 空集早退、成本截顶、SciPy求解、后拒配、cascade及门控 |
| [iou_matching.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/iou_matching.py) | 连续tlwh IoU、tsu>1写大数 |
| [track.py／tracker.py](https://github.com/nwojke/deep_sort/tree/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort) | Tentative/Confirmed/Deleted、计数时点、两阶段候选、缓存与gallery更新 |
| [deep_sort_app.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort_app.py) | MOT字段、过滤、预测调用、结果输出与CLI默认值 |
| [generate_detections.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/tools/generate_detections.py) | ROI比例调整、裁剪／缩放、空ROI的随机patch、编码输出 |
| [训练网络](https://github.com/nwojke/cosine_metric_learning/blob/eda0daaa5462c61ac44553f2151070fd7e316cc8/nets/deep_sort/network_definition.py) | ELU／BN／dropout、行特征与列权重归一、softplus尺度、预处理 |

#### 空特征、零向量和阈值边界

| 边界 | 固定源码事实／风险 | 加固或验证要求（未实现） |
|---|---|---|
| 无检测 `[]` | min_cost_matching对空集合早退，不调用外观距离；旧轨迹仍预测、miss、删除，gallery按活跃confirmed清理 | 不因N=0跳过整帧；空检测不等于检测有空feature |
| 特征缺失None、错误维度 | Detection仍尝试 `np.asarray(...,float32)`，无统一可选特征分支；标量、NaN或维度冲突可能进入dot／solver | 明确 `(D,)`、固定D>0、有限且非零；不以None冒充“自动退成SORT” |
| D=0、形状 `(N,0)` | 归一化结果仍无元素，dot空内积为0，余弦可变成全1而非报错；这不是合法的单位向量 | 显式拒绝D=0；tau≥1可能让无信息外观通过，默认0.2拒绝也不阻止IoU回退 |
| D>0全零向量 | `_cosine_distance` 直接除L2范数，无epsilon；0/0→NaN，可能使SciPy成本不可用 | 不把NaN填0；非零epsilon除法得到的零向量也不是单位向量，应标记不可用并制定另一关联策略 |
| 空gallery与缺失ID键 | `samples[id]`不存在会KeyError；合法二维空gallery经min(axis=0)会ValueError；无身份距离返回值约定 | 正常confirmed流程保证已有样本；反序列化／绕过生命周期时必须验证此不变量 |
| 非有限外观与门掩膜 | NaN比较 `>tau` 为假，不会被成本截顶可靠消除；运动门恰好遮住部分NaN不代表全面防护 | 先显式有限性检查；应测试NaN、Inf及混合有效／无效候选 |
| budget=0、负数 | 0得到 `[-0:]=[0:]`，不截断；负数变正起点切片，可能删空样本 | 只允许None或正整数；不要把0当禁用gallery的开关 |
| 运动门等号 | `dM²>chi2inv95[d]`才写1e5，等于门通过 | 使用实际成本／距离测试精确等号及nextafter邻点；不是把欧氏残差四舍五入后比较 |
| 外观／IoU等号 | `C>tau`才截顶／拒配，C=tau可接受；C稍大被改tau+1e-5后拒绝 | “可接受”不意味着一定被LAP选中；`1-IoU`浮点减法可能落在理论等号两侧 |
| 大数与异常阈值 | INFTY_COST=1e5是有限数；若tau≥1e5，运动禁边可能被接受；极大tau加1e-5可能舍入回tau | 参数域须有限且与真实距离域一致，不能用任意大数关闭门 |
| 非法几何／协方差 | 无统一w/h合法性、正定性或预测后检查；a/h可被速度推为非正，Cholesky可能失败 | 拒绝零高、负宽高和非有限值，追溯错误关联／Q/R，恢复前重验状态与输出 |
| 空ROI | 提取器裁剪失败后生成随机图块，而非跳过检测或返回无特征 | 这是源码风险，不作为可靠兜底；生产应记录失败并隔离，不能生成看似有效的随机身份特征 |
| 裁剪／图接口 | bbox边界先转整数、裁到图像尺寸减1，切片右下仍exclusive；Graph导入前缀与tensor名字需实测 | 不将这一路径宣称已等价复现训练裁剪；权重、颜色与张量名字逐项核验 |

优化前固定逐帧输入，对照匹配对、unmatched集合、状态、计数、gallery及输出；先剖析再做网络batch、样本矩阵缓存或向量化。不得用平均特征代替min-gallery、仅按总年龄排序、把旧confirmed全部送IoU、提前入库或提前删轨。任何无效特征降级、Joseph更新、先门控最大匹配、类别门、质量筛样或相机运动补偿都须另标为扩展。

#### 可复跑的有限测试

测试只使用下列明确给定的**代数／分支输入**，不是视频数据、训练数据或性能测量。读取7份固定外部核心源码到内存，输出内容SHA-256聚合指纹，执行原类；不下载模型、不运行TensorFlow、不写作者源码或测试文件。使用已经检查存在的项目 `.venv/Scripts/python.exe`、NumPy和SciPy；网络获取失败会终止，不降级成假通过。代码块可提取后经 `python -B -` 在内存运行；联网源码属于该固定作者快照，不建议把任意未知URL代码用这种方式执行。

```python
import concurrent.futures
import hashlib
import subprocess
import sys
import types
import numpy as np
import scipy
from scipy.stats import chi2

SHA = "f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab"
names = ["kalman_filter", "linear_assignment", "iou_matching", "nn_matching", "detection", "track", "tracker"]
def fetch(name):
    suffix = f"deep_sort/{name}.py"
    urls = [f"https://raw.githubusercontent.com/nwojke/deep_sort/{SHA}/{suffix}",
            f"https://cdn.jsdelivr.net/gh/nwojke/deep_sort@{SHA}/{suffix}"]
    for url in urls:
        result = subprocess.run(["curl.exe", "-L", "--fail", "--max-time", "20", "-sS", url], capture_output=True)
        if result.returncode == 0:
            return name, result.stdout
    raise RuntimeError(f"source unavailable: {name}")
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
    sources = dict(pool.map(fetch, names))
digest = hashlib.sha256(b"".join(n.encode() + b"\0" + sources[n] for n in sorted(names))).hexdigest()
package = types.ModuleType("ds01_external")
package.__path__ = []
sys.modules[package.__name__] = package
mods = {}
for name in names:
    module = types.ModuleType(f"{package.__name__}.{name}")
    module.__package__ = package.__name__
    sys.modules[module.__name__] = module
    setattr(package, name, module)
    exec(compile(sources[name], f"{SHA}/{name}.py", "exec"), module.__dict__)
    mods[name] = module
kf = mods["kalman_filter"].KalmanFilter()
la, nn = mods["linear_assignment"], mods["nn_matching"]
Detection, Tracker = mods["detection"].Detection, mods["tracker"].Tracker
Track = mods["track"].Track
passed = []
def check(ok, label):
    assert bool(ok), label
    passed.append(label)
def raises(fn, exc, label):
    try:
        fn()
    except exc:
        passed.append(label)
    else:
        raise AssertionError(label)
def det(feature=(1., 0.)):
    return Detection([0., 0., 10., 20.], 1., np.asarray(feature))
def tracker(n_init=3, max_age=30, budget=100):
    return Tracker(nn.NearestNeighborDistanceMetric("cosine", .2, budget), n_init=n_init, max_age=max_age)
def frame(t, ds):
    t.predict()
    t.update(ds)
def run_cost(cost, tau=.2):
    c = np.asarray(cost, dtype=float)
    return la.min_cost_matching(lambda *args: c.copy(), tau, [None]*len(c), [None]*c.shape[1])
d = det()
m, p = kf.initiate(d.to_xyah())
t = Track(m, p, 1, 3, 30, d.feature)
check(np.allclose(d.to_xyah(), [5, 10, .5, 20]) and np.allclose(t.to_tlwh(), d.tlwh), "xyah roundtrip")
check(m.shape == (8,) and np.all(m[4:] == 0) and np.allclose(np.diag(p), [4,4,1e-4,4,1.5625,1.5625,1e-10,1.5625]), "initial state and covariance")
mp, pp = kf.predict(m, p)
check(np.isclose(pp[0,0], 6.5625) and np.isclose(pp[0,4], 1.5625), "prediction covariance")
zp, sp = kf.project(mp, pp)
mu, pu = kf.update(mp, pp, d.to_xyah())
check(np.isclose(sp[0,0], 7.5625) and np.allclose(mu, mp) and np.linalg.eigvalsh(pu).min() > 0, "projection and zero innovation")
check(abs(chi2.ppf(.95, 4)-9.4877) < 5e-5 and abs(chi2.ppf(.95, 2)-5.9915) < 5e-5, "chi square dimensions")
z = d.to_xyah().copy()
z[2] += 1
check(kf.gating_distance(mp, pp, z[None], True)[0] == 0 and kf.gating_distance(mp, pp, z[None], False)[0] > 9.4877, "position only excludes aspect residual")
class BoundaryKF:
    def gating_distance(self, *args):
        return np.array([9.4877, np.nextafter(9.4877, np.inf)])
c = la.gate_cost_matrix(BoundaryKF(), np.zeros((1,2)), [t], [d,d], [0], [0,1])
check(c[0,0] == 0 and c[0,1] == la.INFTY_COST, "motion equality boundary")
check(np.allclose(nn._cosine_distance([[2,0]], [[7,0],[0,2],[-3,0]]), [[0,1,2]]), "cosine normalization and range")
check(np.allclose(nn._nn_cosine_distance([[1,0],[0,1]], [[0,1]]), [0]), "gallery minimum not mean")
metric = nn.NearestNeighborDistanceMetric("cosine", .2, 2)
metric.partial_fit(np.eye(3), [1,1,1], [1])
check(len(metric.samples[1]) == 2 and np.array_equal(metric.samples[1][0], np.eye(3)[1]), "FIFO budget")
metric.partial_fit([], [], [])
check(metric.samples == {}, "deleted gallery cleanup")
metric = nn.NearestNeighborDistanceMetric("cosine", .2, 0)
metric.partial_fit(np.eye(3), [1,1,1], [1])
check(len(metric.samples[1]) == 3, "zero budget does not truncate")
with np.errstate(invalid="ignore", divide="ignore"):
    bad = nn._cosine_distance([[0.,0.]], [[1.,0.]])
    empty_dim = nn._cosine_distance(np.empty((1,0)), np.empty((1,0)))
check(np.isnan(bad).all() and np.array_equal(empty_dim, [[1.]]), "zero vector versus zero dimension")
raises(lambda: nn._nn_cosine_distance(np.empty((0,2)), np.ones((1,2))), ValueError, "empty gallery rejected by reduction")
raises(lambda: nn.NearestNeighborDistanceMetric("cosine", .2).distance(np.ones((1,2)), [1]), KeyError, "missing gallery key")
raises(lambda: run_cost([[np.nan]]), ValueError, "NaN not repaired by clipping")
check(run_cost([[.2]])[0] == [(0,0)] and run_cost([[np.nextafter(.2, np.inf)]])[0] == [], "appearance equality and nextafter")
check(run_cost([[.7]], .7)[0] == [(0,0)] and run_cost([[np.nextafter(.7, np.inf)]], .7)[0] == [], "IoU cost equality and nextafter")
check(run_cost([[0,.19],[.19,1]])[0] == [(0,0)], "clipping not maximum feasible cardinality")
tr = tracker()
frame(tr, [d])
check(tr.tracks[0].hits == 1 and tr.tracks[0].age == 1 and tr.metric.samples == {}, "birth and tentative cache")
frame(tr, [d]); frame(tr, [d])
check(tr.tracks[0].is_confirmed() and len(tr.metric.samples[1]) == 3 and tr.tracks[0].features == [], "third detection confirmation and gallery flush")
frame(tr, [])
check(tr.tracks[0].is_confirmed() and tr.tracks[0].time_since_update == 1, "first miss remains output eligible")
frame(tr, [])
check(tr.tracks[0].time_since_update == 2 and not (tr.tracks[0].time_since_update <= 1), "second miss retained but not output eligible")
fresh = tracker(); frame(fresh, [d]); frame(fresh, [])
check(fresh.tracks == [], "tentative first miss deletion")
one = tracker(n_init=1); frame(one, [d])
check(one.tracks[0].is_tentative(), "n_init one not immediate confirmation")
frame(one, [d])
check(one.tracks[0].is_confirmed(), "n_init one confirms on update")
ages = [types.SimpleNamespace(time_since_update=1, age=100), types.SimpleNamespace(time_since_update=2, age=2)]
def age_cost(tracks, ds, ti, di):
    return np.array([[.19 if i == 0 else .01 for j in di] for i in ti])
check(la.matching_cascade(age_cost, .2, 30, ages, [None])[0] == [(0,0)], "cascade recent observation priority")
back = tracker(n_init=2); frame(back,[d]); frame(back,[d]); frame(back,[det((0,1))])
check(len(back.tracks) == 1 and back.tracks[0].track_id == 1 and len(back.metric.samples[1]) == 3, "IoU bypasses failed appearance and writes gallery")
back.predict()
original_gate = back.kf.gating_distance
back.kf.gating_distance = lambda mean,cov,measurements,only_position=False: np.full(len(measurements), 1e6)
back.update([d]); back.kf.gating_distance = original_gate
check(len(back.tracks) == 1 and back.tracks[0].time_since_update == 0, "IoU bypasses failed motion gate")
old = tracker(n_init=2); frame(old,[d]); frame(old,[d]); frame(old,[]); frame(old,[det((0,1))])
check(len(old.tracks) == 2 and old.tracks[0].time_since_update == 2, "old confirmed track has no IoU fallback")
expire = tracker(n_init=2); frame(expire,[d]); frame(expire,[d])
for _ in range(30):
    frame(expire,[])
check(len(expire.tracks) == 1 and expire.tracks[0].time_since_update == 30, "retained at max_age equality")
frame(expire,[d])
check(len(expire.tracks) == 1 and expire.tracks[0].track_id == 2 and 1 not in expire.metric.samples, "overage cannot rescue old ID")
empty = tracker(); frame(empty,[])
check(empty.tracks == [] and empty.metric.samples == {}, "empty frame with no tracks")
f = np.array([2.,1.]); w = np.eye(2); scale = 2.; eps = 1e-6
def ce(v):
    r = v / np.linalg.norm(v)
    logits = scale * (w @ r)
    return np.logaddexp.reduce(logits) - logits[0]
r = f/np.linalg.norm(f); logits=scale*(w@r)
prob=np.exp(logits-logits.max()); prob/=prob.sum()
analytic=scale/np.linalg.norm(f)*(np.eye(2)-np.outer(r,r))@(prob@w-w[0])
numeric=np.array([(ce(f+eps*e)-ce(f-eps*e))/(2*eps) for e in np.eye(2)])
check(np.allclose(analytic,numeric,atol=1e-8), "cosine softmax gradient finite difference")
print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}")
print(f"Core source aggregate SHA256: {digest}")
print(f"DS01 finite checks: {len(passed)} passed")
```

这些测试对正常有限框、指定特征／成本、单轨状态机与二轨级联作有限核验；部分门控测试替换距离返回值以精确检查阈值和回退分支，不冒称真实运动噪声统计验证。softmax梯度测试是公式有限差分，不是TensorFlow反传或网络训练。无效输入测试确认原码的失败／退化行为，不代表已修复。应用输出只检查其谓词，未运行可视化／文件写出。

后续验收仍需：固定检测及checkpoint的作者／本地逐帧差分 → 多目标矩形、同价、重复框、缺特征与非有限值 → 模型预处理及CPU/GPU精度对照 → 完整视频按交叉、遮挡、重入、相机运动分层 → IDF1／HOTA、ID切换、碎轨与检测误差分开统计 → 网络、特征距离、LAP、I/O尾延迟和峰值内存剖析。训练／验证／测试按身份与场景隔离，避免相邻视频帧泄漏。跟踪结果不可直接作为执行器动作许可。

### 贯穿 L1–L5 的五问自测

| 层级／问题 | 答案要点 | 条件、易错点与证据入口 |
|---|---|---|
| L1：默认连续3帧检测后，空2帧，内部状态和输出怎样？gallery何时加入出生样本？ | 第1帧Tentative且hits=1；第3帧Confirmed，一次送入3个缓存样本；首个空帧tsu=1仍满足应用输出，第二个空帧tsu=2保留但不输出 | 假设关联成功、每帧一次predict/update；不能套SORT的出生hits=0／warmup。见L1状态机、tracker与app |
| L2：最近轨迹外观成本0.19、旧轨迹0.01，为什么前者仍可能先拿检测？旧轨迹失败后能否IoU补配？ | 分层按预测后tsu从1递增，前层接受后检测移除，不追求跨层总成本最小；仅未确认＋tsu=1未配confirmed进IoU | 总age不是级联年龄；旧confirmed tsu>1不进回退。见论文§2.3、L2与cascade测试 |
| L3：换成同外观零件后ID错误，增大gallery并降低阈值能否保证改善？ | 不保证；查域、ROI／颜色、范数、候选两门与IoU污染；更大gallery增加误近邻与成本，距离阈值降低反而更严格；先同域独立数据标定 | 0.2不是概率；None／0非100，max_age不是输出时长。见L3参数表、训练数据来源 |
| L4：为何8维状态却用4维卡方？lambda=0是否关闭运动？归一化如何改变梯度？ | 残差位于xyah观测空间，白化后平方范数服从chi-square4；lambda=0只去掉代价中运动项，门仍在；给出 `(I-rrᵀ)/norm(f)` 的归一化雅可比 | 仅位置时同步切到2维及5.9915；模型正确且S正定；不是8维15.507或95%身份正确。见L4推导与固定代码 |
| L5：没有特征时传空向量或零向量，是否自然退成SORT？等于阈值是否必匹配？ | 否；D=0可给距离1，D>0零向量归一化出NaN，空gallery可能异常；需显式缺失合同及独立降级策略。等号边可接受但仍受竞争／层次影响，截顶后拒配不保证最大合法基数 | 固定代码没有可靠缺特征退路；IoU能绕过两门但只对特定候选，不能概括成普遍兜底。见L5边界、L4反例与有限测试 |

## 来源层级、获取深度与未完成范围

以下为本次联网核对。**P0**为原始论文、作者源码／官方数据说明，P1为权威二手，P2为专业社区，P3为一般资料；本卡核心事实使用P0，CDN仅改变传输地址不改变作者身份。**B**=书目、**A**=摘要／说明、**T**=实际读到方法／函数正文、**F**=失败。原论文和同作者实现不算独立复现；方法／版本细节按 **[SINGLE-SOURCE]** 解读，多入口不凑“三项独立实验证据”。没有统一工业工况的多源效果排名，不移用论文百分比或Hz。

<a id="source-1"></a>

**[cite:1] 2017 Deep SORT 原文，P0，B+A+T（限定）。** [arXiv书目](https://arxiv.org/abs/1703.07402)、[固定v1方法HTML](https://arxiv.org/html/1703.07402v1)、[ar5iv同文转写](https://ar5iv.labs.arxiv.org/html/1703.07402)。已读§2.1–2.3、式(1)–(6)、Algorithm 1、§2.4网络表及数据描述开头；长响应截断，未审核完整实验／参考项，不声称逐页读取IEEE正式PDF。ICIP出版信息与DOI由作者仓库引用补核。命令行另取全文超时，未把超时当读取成功。

<a id="source-2"></a>

**[cite:2] 作者跟踪仓库版本及说明，P0，T。** [提交解析API](https://api.github.com/repos/nwojke/deep_sort/commits/master) 取得SHA `f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab`；[固定README](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/README.md) 已读MOT／POI输入、N×138、`mars-small128.pb`、TensorFlow1.5模型说明、示例预算100及训练仓库链接。当前requirements为NumPy、OpenCV、SciPy且不锁版本；它不证明2017依赖环境。未下载权重／数据，未验证许可证部署要求或导出图兼容性。

<a id="source-3"></a>

**[cite:3] 固定核心接口、gallery与生命周期，P0，T。** 实际读取同SHA的 [tracker.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/tracker.py)、[track.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/track.py)、[nn_matching.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/nn_matching.py)、[detection.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/detection.py) 全部函数正文；代码行为为该版本事实，不以docstring泛称覆盖边界。

<a id="source-4"></a>

**[cite:4] 固定Kalman，P0，T。** [kalman_filter.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/kalman_filter.py)：读取8维矩阵、初始化、Q/R尺度、投影、Cholesky更新、2D／4D门与卡方表。有限检查用本地SciPy分位数核对表内舍入值，非统计覆盖率实测。

<a id="source-5"></a>

**[cite:5] 固定关联代码，P0，T；传输有F后恢复。** [linear_assignment.py固定CDN镜像](https://cdn.jsdelivr.net/gh/nwojke/deep_sort@f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/linear_assignment.py)、[iou_matching.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/iou_matching.py) 全文读取。linear_assignment最初raw抓取空内容、GitHub页面只返回提交概况，直接请求等待后停止；改固定SHA的CDN取得函数体。没有把GitHub页面概况计为代码已读。

<a id="source-6"></a>

**[cite:6] 固定应用，P0，T。** [deep_sort_app.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort_app.py)：读取检测构造、过滤、NMS调用、predict/update、保存谓词和CLI参数。没有执行应用或展示，不声称测试过图像解码、可视化及结果格式化。

<a id="source-7"></a>

**[cite:7] 固定特征生成，P0，T。** [tools/generate_detections.py](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/tools/generate_detections.py)：读取裁剪、缩放、失败随机patch、图加载和batch编码；未运行TensorFlow与冻结图，不将README“可重导出”当兼容性已验证。

<a id="source-8"></a>

**[cite:8] 作者ReID训练来源，P0，T。** [提交解析API](https://api.github.com/repos/nwojke/cosine_metric_learning/commits/master) 固定 `eda0daaa5462c61ac44553f2151070fd7e316cc8`；同提交 [README](https://raw.githubusercontent.com/nwojke/cosine_metric_learning/eda0daaa5462c61ac44553f2151070fd7e316cc8/README.md)、[train_mars.py](https://raw.githubusercontent.com/nwojke/cosine_metric_learning/eda0daaa5462c61ac44553f2151070fd7e316cc8/train_mars.py)、[datasets/util.py](https://raw.githubusercontent.com/nwojke/cosine_metric_learning/eda0daaa5462c61ac44553f2151070fd7e316cc8/datasets/util.py)、[network_definition.py](https://raw.githubusercontent.com/nwojke/cosine_metric_learning/eda0daaa5462c61ac44553f2151070fd7e316cc8/nets/deep_sort/network_definition.py) 已读；[train_app.py](https://raw.githubusercontent.com/nwojke/cosine_metric_learning/eda0daaa5462c61ac44553f2151070fd7e316cc8/train_app.py)读取参数、trainer、优化器及另取 `_create_loss` 分支，不声称全文逐行审核。默认CE、特征／权重归一、训练态、按身份验证与导出语义来自这些函数；未训练、未核checkpoint，不能从源码可选triplet推断默认使用triplet。

<a id="source-9"></a>

**[cite:9] 2018训练论文，P0，B+A+部分T。** [arXiv:1812.00442v1 HTML](https://arxiv.org/html/1812.00442v1) 实际读到摘要、引言、相关工作与§3标准softmax开头；长页返回在§4前截断，带§4片段的请求仍未取得该节正文。WACV2018书目由作者README核对，arXiv上传为2018-12-02，不当作会议举行日。L4具体cosine参数化以已读作者网络函数及本卡代数推导为证，不虚称已通读该论文§4／全部实验。

<a id="source-10"></a>

**[cite:10] MARS官方数据来源，P0，B+A+T。** [Zheng实验室MARS页面](https://zheng-lab-anu.github.io/Project/project_mars.html)：ECCV2016数据工作、1261身份、训练625／测试636、训练框509,914及命名／划分说明。旧 `liangzheng.com.cn` 页面抓取为空（F），改作者实验室站点读取成功；未下载数据、未审核标注或每个样本。官方数据页是数据规模的独立原始入口，不是Deep SORT性能复现。

## 验证记录与 G47 销项依据

本次只新增 `docs/algorithm_cards/deep_sort.md`；未改源码、依赖、索引、台账与总览。仓库Python文件检索 `DeepSORT|DeepSort|deep_sort|NearestNeighborDistanceMetric` 未发现实现；通用教学卡及本文代码围栏不升级实现状态。系统 `python` 为商店入口，但项目虚拟环境可用，因此没有沿用SO01历史环境限制，也没有安装依赖。

本次从Markdown提取Python围栏到内存执行，实际结果为 **`DS01 finite checks: 34 passed`**；环境为 **Python 3.13.14、NumPy 2.5.4、SciPy 1.18.1**。7份核心源码按测试代码所示算法计算的聚合SHA-256为 `d1b0b9095bf90eb3db4dd7435312193bfe63f7b7c42cad4563b762c114426864`；依赖是本机当前版本，不追认为2017运行环境。没有将代码导入产品目录，也没有生成pyc或单独测试文件。

文档UTF-8、五层／五问、单卡键、8张表列数、4个围栏边界、58处本地链接／锚点和尾空白检查通过；另计31处外部链接，不把计数当作逐条可访问性证明。`git diff --check`通过，只有既有文件LF／CRLF提示；目标文件未跟踪，因此另做直接文件校验，不能只靠Git差异检查。

原有106个非目标、非忽略文件的前后内容指纹一致。算法为Git列出路径后按PowerShell `Sort-Object -Unique` 排序、逐文件SHA-256大写十六进制，再拼接 `path=hash` 以LF连接并取UTF-8 SHA-256；前后均为 `877E9F52144FE4F4A4D480ACE19201CBD2D307350BB61670C44FCBAFA10629C2`。核验期间另观察到非本次操作新增的 `docs/algorithm_cards/bytetrack.md`，首次全清单核验因此不一致；按原106文件范围复核通过，未编辑或回退该新增文件。不能把这一结果写成“整个工作区没有并发变化”；指纹也不覆盖忽略文件、Git元数据或取样间短暂改变。

**销项依据：** G47要求的外观训练来源、运动门与级联已由DS01具体展开；八维xyah、卡方维度、余弦归一与gallery、年龄级联、受限IoU回退、生命周期、论文／代码融合差异和异常边界均有固定来源与有限验证入口，另含L1–L5推导／伪代码及5问答案。可关闭“缺专卡与机制说明”这一内容缺口；台账和旧总览仍显示缺卡属于本次未同步状态，不能声称已重算全库覆盖率。

**保留缺口：** 历史权重血缘及2017发表软件快照、2018论文未读完整节、网络训练／冻结图运行、输入与数值加固、完整多目标视频、MOT身份指标、真实延迟和现场安全。实现仍**未实现**，L3／L5仍**部分**；有限通过不能等同于完整复现、工业可用或身份／控制安全保证。
