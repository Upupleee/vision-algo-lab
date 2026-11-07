# Hungarian 指派算法卡

[学习规范](../algorithm_learning_standard.md) · [覆盖表 G13／12-01](../evolution_coverage.md#12-跟踪运动视觉引导) · [T04 门控关联](remaining_domains.md#t04-门控后一对一全局指派) · [跟踪演进](../tracking_guidance_evolution.md)

## 范围、状态与销项口径

核对日期：2026-11-05；仓库基准：HEAD `918a03e` 加当前工作树。唯一卡键为 `hungarian.md:H01`，只承接 G13，不新增第二张历史卡或扩展卡。本文区分 Kuhn 1955 的原始最大评分问题、现代最小费用势能增广实现，以及跟踪中的拒配建模。

| 项目 | 覆盖与边界 |
|---|---|
| 演进定位 | 数据关联的组合优化内核；从局部抢配／一般 LP 转为资格图上的组合匹配与对偶预算；不是运动模型或完整 MOT |
| L1 | 已核验：现代内核逐步流程、可执行参考代码、完整例题；不是 1955 第4节程序逐行复刻 |
| L2 | 已核验：可行势能、紧边、交替增广及最小松弛更新；原文定理5–6与权威讲义交叉 |
| L3 | 部分：矩形、拒配、禁边、尺度与排障协议明确；没有业务代价校准或性能剖析 |
| L4 | 已核验：限定模型的原对偶、互补松弛、终止与复杂度推导；不声称真实关联必然正确 |
| L5 | 部分：文档内教学参考与可复跑检查器；无生产模块、MOT 接入或真实视频验收 |
| 问答 | 贯穿 L1–L5 的5问，含答案、条件、易错点与证据入口 |
| 来源 | 原论文出版社 B+A、大学镜像 T部分；权威教学正文 T；官方 API T；社区实现解释 T。获取等级不等于可信度等级 |
| 实现 | **仓库无独立 Hungarian 产品内核**。本文代码即使离线执行通过，也只算文档参考验证；现有评测的 SciPy 调用不是手写 Hungarian 或跟踪实现 |
| 销项 | G13“没有具体步骤／正确性／矩形与拒配区分”的内容缺口由本卡承接，可据此将具体方法缺卡转“已有”；历史 ID 保留。覆盖表与索引按本次单文件约束未改，不能声称总表已销账 |

“已核验”只限本文声明的资料、代数与有限检查，不表示所有五层工程验收完成。原论文第4–5节的完整过程、图表与手算例未完整取得；这是历史逐步复刻的保留缺口，不用现代代码冒充已读原版。

<a id="h01"></a>
## H01 Hungarian：原对偶与势能增广路径

### 任务、契约与演进分支

输入为两组对象之间的费用矩阵 \(C\in\mathbb R^{n\times m}\) 和合法边集合 \(E\)。费用必须可加，数值越小越好；列不是“类别概率”，矩阵不是图像坐标。基础输出为互不重复的行列索引对、原始费用总和及未占用对象。固定基数版本要求匹配数为 \(k=\min(n,m)\)，合法图无法达到该基数时返回不可行；**矩形只允许较大一侧剩余，不自动允许双方拒配**。可选拒配版本另给每行未匹配成本 \(\alpha_i\)、每列未匹配成本 \(\beta_j\)。

| 分支 | 已取证的机制 | 与本文实现的关系 |
|---|---|---|
| Kuhn 1955，原始论文 | 第2节：0/1资格矩阵、transfer、essential individuals/jobs、匹配与 adequate budget；第3节：正整数评分的最大化，预算满足 \(a_i+b_j\ge R_{ij}\)，等号定义资格边，满匹配给最优证书；p.89可见单位预算调整 | 原型思想，不是本文最小化符号或 `minv/way` 数组代码。[cite:1](#source-1)[cite:2](#source-2) |
| 现代教材矩阵／图形式 | 行列约减相当于建立可行对偶；零元素图中寻找最大匹配，失败后按可达集改势能 | 划线必须覆盖全部零且达到最小覆盖数，不能随意画线或贪心数零。[cite:3](#source-3)[cite:4](#source-4) |
| 现代逐行势能增广路径 | 保留当前交替树和每列最小松弛，不在每次势能变化后重建整个搜索 | 本文可执行参考。方阵 \(O(n^3)\)，矩形短边 \(r\)、长边 \(s\) 时 \(O(r^2s)\)，由本卡循环计数推导。[cite:5](#source-5) |
| SciPy 当前文档所述实现 | modified Jonker–Volgenant，no initialization；支持矩形与 maximize | 独立交叉求解器，不称 Kuhn1955 的源码或本文算法的逐行实现。[cite:6](#source-6) |

原论文确切书目为 H. W. Kuhn，*The Hungarian Method for the Assignment Problem*，*Naval Research Logistics Quarterly* **2(1–2):83–97，March 1955**，DOI `10.1002/nav.3800020109`。原文已经使用对偶思想，不能说“后来才引入对偶”；后来改变的是组织搜索和维护松弛的方式。König／Egerváry 为原文明确的来源，本文不扩展未经核验的绝对首创史。[cite:1](#source-1)[cite:2](#source-2)

前序瓶颈 → 改变维度 → 代价：逐个最便宜会互抢列；紧边匹配配合对偶更新获得全局目标证书；需要全体候选成本、匹配状态与搜索空间。遗留问题是错误成本、错误门控或错误业务目标仍会被“最优”求解，历史身份连续性也不在静态指派目标内。T04 负责预测、成本、门控与生命周期交接，H01 负责其黑盒位置的求解机制，二者不互相替代。

### L1 步骤：从费用到完整匹配

先固定 \(r\le s\)：若输入行比列多，转置并在输出时交换索引。`u` 是行势能，`v` 是列势能，`p[j]` 是列 j 当前占用行；0号列只作未匹配根的哨兵，真实输入输出均为0基索引。本文使用 \(u_i=\min(0,\min_{j\in E_i}C_{ij})\)、\(v_j=0\) 初始化，直接兼容负费；有一行没有合法边则固定基数问题不可行。

1. 从尚未匹配的下一行起树，置 `p[0]=i`，清空 `used`、`minv`、`way`；已经分配的行列不清空。
2. 到达列 `j0` 后，把其占用行 `p[j0]` 加入树；扫描该行到未访问列的合法边，用约减费用 \(C_{ij}-u_i-v_j\) 更新每列最小松弛 `minv[j]`，改善时记录前驱列 `way[j]=j0`。
3. 在所有未访问列的**累计** `minv` 中取最小有限值 \(\delta\)。没有有限值意味着这棵树再无合法出边，返回不可行；不能继续算无穷减无穷。
4. 树内行势能加 \(\delta\)，已访问真实列势能减 \(\delta\)，未访问列的有限松弛减 \(\delta\)。到达一个最小松弛列：若已占用则回到步骤2；若空闲则找到增广路径。
5. 沿 `way` 反向改写 `p`，翻转路径边，一次增加一个匹配；处理完全部 r 行停止。用原始 C 重新求和，恢复转置前索引并检查禁边与唯一性。

这不是把 `solve_assignment()` 换名当步骤；下方代码完整展开搜索、势能更新、增广和拒配转换。它是现代算法的 **Python 可执行伪代码／教学参考**，仅用标准库，未接入 `traditional/`。

```python
from math import isfinite


class Infeasible(ValueError):
    pass


def matrix(c, ncols=None):
    a = [list(row) for row in c]
    m = len(a[0]) if a else (0 if ncols is None else ncols)
    if type(m) is not int or m < 0 or (ncols is not None and ncols != m):
        raise ValueError("invalid shape")
    if any(len(row) != m for row in a):
        raise ValueError("ragged matrix")
    for row in a:
        for x in row:
            if x is not None and not (type(x) is int or
                    (type(x) is float and isfinite(x))):
                raise ValueError("cost must be finite int/float or None")
    return a, len(a), m


def hungarian(c, ncols=None):
    a, n, m = matrix(c, ncols)
    transposed = n > m
    b = [[a[i][j] for i in range(n)] for j in range(m)] if transposed else a
    r, s = min(n, m), max(n, m)
    u, v, p = [0] * (r + 1), [0] * (s + 1), [0] * (s + 1)
    for i in range(1, r + 1):
        values = [x for x in b[i - 1] if x is not None]
        if not values:
            raise Infeasible("row has no allowed edge")
        u[i] = min(0, min(values))
    for i in range(1, r + 1):
        p[0] = i
        j0 = 0
        used = [False] * (s + 1)
        minv = [None] * (s + 1)
        way = [0] * (s + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = None, -1
            for j in range(1, s + 1):
                if used[j]:
                    continue
                x = b[i0 - 1][j - 1]
                if x is not None:
                    cur = x - u[i0] - v[j]
                    if minv[j] is None or cur < minv[j]:
                        minv[j], way[j] = cur, j0
                if minv[j] is not None and (delta is None or minv[j] < delta):
                    delta, j1 = minv[j], j
            if j1 < 0:
                raise Infeasible("no augmenting path")
            for j in range(s + 1):
                if used[j]:
                    u[p[j]] += delta
                    if j:
                        v[j] -= delta
                elif minv[j] is not None:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    oriented = [(p[j] - 1, j - 1) for j in range(1, s + 1) if p[j]]
    pairs = sorted((j, i) if transposed else (i, j) for i, j in oriented)
    return {"pairs": pairs, "cost": sum(a[i][j] for i, j in pairs),
            "u": u[1:], "v": v[1:], "transposed": transposed}


def optional_assignment(c, alpha, beta):
    alpha, beta = list(alpha), list(beta)
    a, n, m = matrix(c, len(beta))
    if len(alpha) != n:
        raise ValueError("invalid penalties")
    matrix([alpha])
    matrix([beta])
    if any(x is None for x in alpha + beta):
        raise ValueError("penalties must be finite")
    d = [[None] * (m + n) for _ in range(n + m)]
    for i in range(n):
        d[i][:m] = a[i]
        d[i][m + i] = alpha[i]
    for j in range(m):
        d[n + j][j] = beta[j]
        d[n + j][m:] = [0] * n
    result = hungarian(d)
    pairs = [(i, j) for i, j in result["pairs"] if i < n and j < m]
    rows, cols = {i for i, _ in pairs}, {j for _, j in pairs}
    ur = [i for i in range(n) if i not in rows]
    uc = [j for j in range(m) if j not in cols]
    cost = sum(a[i][j] for i, j in pairs) + sum(alpha[i] for i in ur)
    cost += sum(beta[j] for j in uc)
    return {"pairs": pairs, "unmatched_rows": ur, "unmatched_cols": uc,
            "cost": cost, "augmented": d, "certificate": result}
```

#### 完整走例

沿用 T04 的费用 `[[1,2],[1.1,100]]`，无禁边，以下为十进制理想算术，不是浮点逐位相等承诺。

| 阶段 | 树／松弛与更新 | 当前匹配（此表1基） |
|---|---|---|
| 初始 | u=(0,0)，v=(0,0) | 空 |
| 加入行1 | minv=(1,2)，δ=1；u=(1,0)，选空列1 | (1,1) |
| 加入行2 | minv=(1.1,100)，δ=1.1；u=(1,1.1)，到占用列1 | (1,1) |
| 经列1到行1 | 列2的松弛从98.9改为1；δ=1；树内两行各加1、列1减1，u=(2,2.1)，v=(-1,0) | 沿行2→列1→行1→列2增广 |
| 结束 | pairs=[(0,1),(1,0)]，原费用3.1；对偶和2+2.1−1=3.1 | (1,2),(2,1) |

约减矩阵为 `[[0,0],[0,97.9]]`。最优值3.1，而按第一行贪心抢列1得到101。零边不仅有最终选中两条，故“出现零就立即永久分配”不是算法。整数等价例 `[[10,20],[11,1000]]` 用于精确证书测试。

### L2 机制：不变量与错误替换

**三个不变量。** 所有合法边 \(\bar c_{ij}=C_{ij}-u_i-v_j\ge0\)；已匹配边都紧，即 \(\bar c=0\)；`p` 始终是一对一匹配。固定根的一棵交替树包含可达行 S 和已访问列 T，已访问列的匹配行也在 S，且在找到空列前 \(|S|=|T|+1\)。原文用“预算上界＝评分”给最大化证书，现代最小化版本用“费用＝对偶下界”；符号方向不能混用。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

**为何取最小松弛。** 对 S 中行增加势能、T 中列降低势能，仅 S→非T 边的约减费用减少；取这类边的最小松弛使其不变负，至少一条边变紧。S→T 边的变化抵消，非S→T 边只增大；原有匹配边不会失去紧性。步长更大可能破坏对偶可行，步长更小可能没有新紧边，反复空转。

**为何保留 `minv`。** 它是整棵树到各未访问列的最小约减费用，不是最后一行的最小值。即使当前新行到某列是禁边，该列之前通过其他树内行得到的有限松弛仍须参与 δ 选择；把选择逻辑错误地放进“当前边合法”分支会漏掉路径。保留前驱 `way` 才能重排早先匹配，不是仅给根抢一个新列。[cite:5](#source-5)

**零松弛不等于死循环。** 已经存在紧边时，δ 可以为0，但每轮仍标记一个此前未访问列，或者在空列结束。教材中“搜索已经穷尽零边后再调势能”的 δ 严格为正，与本文“每轮选下一列”的零步长不矛盾。删除 `used` 或每轮重置树才可能失去该进度保证。

**行列约减的条件。** 方阵每行每列必选一次，分别减常数只改变所有解的共同基线；矩形每行必选、列可不选，任意列约减会改变选列偏好。例如单行 `[2,1]` 若分别减去列值，得到 `[0,0]`，可能错误选原费用2。要么补齐零成本 dummy 行后统一做方阵变换，要么使用本文满足矩形对偶符号的实现。[cite:4](#source-4)

### L3 工程权衡：矩形、拒配与禁边

#### 三种不同业务目标

| 目标 | 正确建模 | 不能混用的做法 |
|---|---|---|
| 矩形固定基数 | 短边全部匹配，长边允许剩余；长边剩余零成本；转置或补 dummy | 不等于“所有差配都可拒绝” |
| 惩罚型拒配 | 真实边费用＋每个未匹配对象的独立成本，允许匹配数从0到k | 不保证优先匹配最多；高费用真边可能被有意放弃 |
| 先最大合法基数，再最小费用 | 先求合法图最大匹配数 q，再在基数 q 下求最小费用流等 | 不能靠任意调大 α、β 就声称严格字典序最优 |

拒配的原目标为

\[
\min_{x\ge0}\sum_{(i,j)\in E}C_{ij}x_{ij}
+\sum_i\alpha_i(1-\sum_jx_{ij})
+\sum_j\beta_j(1-\sum_ix_{ij}),\quad
\sum_jx_{ij}\le1,\quad\sum_ix_{ij}\le1.
\]

等价于常数 \(\sum_i\alpha_i+\sum_j\beta_j\) 加上 \(\sum(C_{ij}-\alpha_i-\beta_j)x_{ij}\)。没有其他竞争时，匹配优于双边拒配的条件是 \(C_{ij}<\alpha_i+\beta_j\)，**不是只跟某一侧阈值比较**；有竞争时必须全局求解。

`optional_assignment` 用 \((n+m)\times(n+m)\) 方阵 D：左上是合法真实 C；右上只允许行 i 到它自己的 dummy 列，费用 α_i；左下只允许 dummy 行 j 到真实列 j，费用 β_j；右下全0，其他边禁用。任意部分匹配都可补成 D 的满匹配；删去 dummy 后还原相同总目标。这个证明及代码是本卡明确构造，不宣称来自 Kuhn1955。只补到 \(\max(n,m)\) 的0行／列无法表达双方任意拒配。

#### 禁边、数值与调参顺序

- **先门控后优化。** 先求全矩阵最优再丢超门限对一般不等价。例 `[[0,4],[4,6]]`，门限5：无约束选对角成本6，丢6后只剩1对；先禁6可得到两条费用4的合法边，总8。另加拒配成本时还要按新目标比较，不能直接套“最大合法命中”结论。
- **禁边用掩膜而非普通大数。** 参考代码用 `None`，不对其做算术。全禁行是不可行的充分条件，不是必要条件：`[[1,None],[2,None]]` 两行都有边仍违反 Hall 条件。可选拒配有有限 α、β 时总能走全拒配，不应报固定基数不可行。
- **若外部求解器只能接收大 M。** 对 N×N 的完整扩展矩阵，已知一个合法满匹配费用 U、所有合法边的下界 L，则取 \(M>U-(N-1)L\) 才能排除含至少一条 M 的候选；该充分界也使 M>L。没有合法解上界 U 时不承诺此法可行；解后仍检查掩膜，防数值溢出／舍入。不要使用无依据的 `1e9`。
- **负费不必截0。** 本文行最小值初始化确保对偶可行。固定匹配数 k 下统一加 K，只给目标加 kK；拒配问题若只平移真实边则加的是变量 \(K|M|\)，会改变拒配偏好。对收益最大化可取 C=−收益，但禁边掩膜不反号，拒配代价必须按同一目标解释。
- **量纲先于阈值。** Mahalanobis平方距离无量纲，像素距离、1−IoU、负对数似然和外观余弦距离不能直接共用尺度；跨轨迹高斯似然还涉及 logdet 协方差。先核对时间戳、坐标与检测，再成本分布／合法图，再 α、β，最后才换求解器；α、β 与 C 同单位，不提供通用最优常数。
- **退化与重复目标。** 同价矩阵有多个最优解，正确性测目标与约束，不要求不同求解器返回完全相同索引；确定性扫描只固定当前输入顺序，不能保证跨帧 ID 稳定。浮点近同价要记录差距和残差，不用随意加随机噪声掩盖歧义。

精度与耗时的次序：小图先穷举核正确性 → 检查原对偶残差与禁边 → 保留合法连通分量分解（且拒配成本仍可分）→ 再剖析成本构造和求解耗时。稀疏求解器、缓存、批量化只有在等价性对照后才替换；本文无耗时或帧率实测，不能据渐近式承诺实时。

### L4 数学：原对偶、证书与复杂度

#### 方阵线性规划与互补松弛

对合法边求最小费用满匹配，松弛为 LP：

\[
P:\quad \min\sum_{(i,j)\in E}C_{ij}x_{ij},\qquad
\sum_jx_{ij}=1,\quad\sum_ix_{ij}=1,\quad x_{ij}\ge0.
\]

为等式约束引入自由变量 u、v，拉格朗日函数为

\[
\mathcal L=\sum_i u_i+\sum_jv_j+
\sum_{(i,j)\in E}(C_{ij}-u_i-v_j)x_{ij}.
\]

对非负 x 取下确界，若有负系数则为负无穷，否则为 \(\sum u+\sum v\)，得到

\[
D:\quad\max\sum_i u_i+\sum_jv_j,\qquad u_i+v_j\le C_{ij}.
\]

二分图关联矩阵的整数性使可行多面体的顶点对应整数匹配（方阵全连接时为双随机矩阵的置换顶点）；删除禁边不会破坏该结构。因此不是先求任意实数 x 再贪心四舍五入。对原对偶可行解，

\[
P(x)-D(u,v)=\sum_{(i,j)\in E}\bar c_{ij}x_{ij}\ge0.
\]

若匹配全部走紧边则差为0，达到全局最优；反之最优原对偶解满足 \(\bar c_{ij}x_{ij}=0\)。这要求完整原问题可行；仅“部分匹配都是零边”不能推出已完成。上述公式由本卡展开推导，整数性与证书由两份权威讲义支持，并与原文最大化定理5–6交叉。[cite:2](#source-2)[cite:3](#source-3)[cite:4](#source-4)

原文最大化采用 \(a_i+b_j\ge R_{ij}\)、最小预算上界；令 C=−R 时可用 u=−a、v=−b 对照符号，但不能把原文的非负整数预算要求直接挪成现代最小化的自由变量限制。原 p.89 的单位调整是第3节理论论证可见部分；本文不将它概括为已完整核验的原第4节高效步长。[cite:2](#source-2)

#### 矩形对偶与未使用列

当 r≤s，行等式、列不等式 \(\sum_i x_{ij}\le1\)，对偶仍最大化 \(\sum u+\sum v\)，但增加 **\(v_j\le0\)**，u 仍自由。令列占用 \(t_j=\sum_ix_{ij}\)，则

\[
P-D=\sum\bar c_{ij}x_{ij}+\sum_jv_j(t_j-1)\ge0.
\]

所以紧匹配之外，还要求未用列的 v=0。本文 v 从0开始，只降低已访问且被占用列的势能，已匹配列增广后仍被占用，故未用列保留0；行数大于列数时先转置，上述“列”是定向后的较大一侧。直接用方阵的自由 v 和紧边条件验证矩形会漏掉证书的一半。

#### 终止、不可行与复杂度

合法满匹配存在、精确算术条件下，每轮选出一个新列；占用列带入其匹配行，空列结束增广。第 i 次增广前仅 i−1 个占用列，至多经历 i 次扫描，每次 O(s)，累计 \(\sum_{i=1}^rO(is)=O(r^2s)\)。势能调整与回溯不超过该量级，输入费用 O(rs)，核心辅助数组 O(r+s)，本文转置副本最坏另占 O(rs)。拒配方阵 N=n+m 的朴素实现为 O(N³) 时间、O(N²) 矩阵内存；不能把它仍记为原 r×s 的代价。

若没有有限出树松弛，则所有树内行合法邻居都在已访问列集合，且 \(|S|=|T|+1\)，构成 Hall 不足的见证，无法覆盖短边；单纯把 `delta` 设成最大数并继续不构成解。只要一轮访问进度成立，δ=0和负原始费用都不妨碍有限终止；浮点舍入不享有精确算术的无条件证书。

现代实现的三次复杂度由上述循环直接核算，也由社区解释交叉；社区页对“原版四次、后续三次”的历史概括仅作二手线索，本文不把 O(n³) 或无条件 O(n⁴) 冒称为已读到的 Kuhn1955 原文定理。Columbia 讲义重新调用匹配子程序的运行时间还取决于所用子程序，不能与保留树的实现混算。[cite:3](#source-3)[cite:5](#source-5)

#### 数值误差与可辨识性

方阵对偶具有规范自由度 \((u+t,v-t)\)，势能不唯一；多重最优时匹配也不唯一，不表示最优费用不可定义。对于浮点输出，应保存最大对偶违约、匹配边紧性残差、矩形未用列势能和原对偶 gap。若每条合法边违约至多 ε_d、匹配边残差绝对值至多 ε_t，且 v≤0、未用列v=0，则将每行u下调 ε_d 得到可行下界，有条件地得到次优差不超过 \(r(\varepsilon_d+\varepsilon_t)\)；该推导还不包括残差求和本身的浮点误差。没有这些条件，不提供通用机器精度界，更不推导视频 ID 错误率。

### L5 实现对应、边界与验证

**源码对应。** 仓库 [tests/benchmark_pose_ncc.py](../../tests/benchmark_pose_ncc.py) 第9行导入 SciPy，第67–78行在已有检测和真值之间构造距离、求指派后判误差门限。这是定位评测，不是 T04 的门控关联实现，也不能证明本卡拒配内核已集成。当前官方 API 明示修改 JV；不能仅因函数名 `linear_sum_assignment` 称它为 Hungarian。本文不修改该既有代码、不重跑完整姿态基准。[cite:6](#source-6)

**数据布局与输入限制。** 代码为行列表，输出0基 `(row,col)`；内部哨兵1基与真实0基显式分开。`None` 是唯一禁边表示，NaN、±∞、布尔和非内置数值类型一律拒绝，避免排序与最小值传播异常。整数路径使用 Python 任意精度整数；浮点路径允许有限 float，但不保证极大值相减不溢出或极近费用可辨，应缩放到可用动态范围并核验证书。`[]` 默认0×0，0×m 由 `ncols=m` 指明；`[[],[]]` 是2×0，固定基数为0；可选拒配由 β 长度保留空行维度。

**优化前检查。** 合法边核验、行列唯一性、固定基数、原费用回算是独立条件；不从 `-v[0]` 取费用，因为本文哨兵不维护教材的累计目标恒等式，且行势能不一定从0开始。并列时按升序列和严格改善更新前驱，不规定跨求解器索引一致。剪枝与稀疏转换前必须保留全部最优候选或给出等价证明。

#### 可复跑检查器

下面是文档内第二个 Python 块，接在第一个块后运行。输入是明确列出的**数学测试夹具和有限值域穷举**，不是实拍数据、论文性能或模拟场景统计。独立枚举器不用势能／增广路径；可行解检查目标和原对偶证书，不可行解检查失败语义。SciPy 只作额外交叉，非唯一真值。

```python
from itertools import permutations, product
from math import isclose


def brute(c):
    n, m = len(c), len(c[0]) if c else 0
    b = list(map(list, zip(*c))) if n > m else c
    r, s = min(n, m), max(n, m)
    costs = []
    for cols in permutations(range(s), r):
        xs = [b[i][j] for i, j in enumerate(cols)]
        if all(x is not None for x in xs):
            costs.append(sum(xs))
    return min(costs) if costs else None


def certificate(c, result):
    n, m = len(c), len(c[0]) if c else 0
    pairs = result["pairs"]
    assert len(pairs) == min(n, m)
    assert len({i for i, _ in pairs}) == len(pairs)
    assert len({j for _, j in pairs}) == len(pairs)
    assert all(c[i][j] is not None for i, j in pairs)
    assert sum(c[i][j] for i, j in pairs) == result["cost"]
    b = list(map(list, zip(*c))) if result["transposed"] else c
    op = [(j, i) for i, j in pairs] if result["transposed"] else pairs
    u, v = result["u"], result["v"]
    assert all(x <= 0 for x in v)
    assert all(u[i] + v[j] <= x for i, row in enumerate(b)
               for j, x in enumerate(row) if x is not None)
    assert all(u[i] + v[j] == b[i][j] for i, j in op)
    occupied = {j for _, j in op}
    assert all(x == 0 for j, x in enumerate(v) if j not in occupied)
    assert sum(u) + sum(v) == result["cost"]


def brute_optional(c, alpha, beta):
    n, m = len(alpha), len(beta)
    values = []
    for cols in product(range(-1, m), repeat=n):
        chosen = [j for j in cols if j >= 0]
        if len(chosen) != len(set(chosen)):
            continue
        if any(j >= 0 and c[i][j] is None for i, j in enumerate(cols)):
            continue
        total = sum(alpha[i] if j < 0 else c[i][j] for i, j in enumerate(cols))
        total += sum(beta[j] for j in range(m) if j not in chosen)
        values.append(total)
    return min(values)


def run_checks():
    fixed = 0
    for n, m in [(1, 3), (2, 2), (2, 3), (3, 2)]:
        for flat in product([None, -2, 0, 3], repeat=n * m):
            c = [list(flat[i * m:(i + 1) * m]) for i in range(n)]
            expected = brute(c)
            try:
                result = hungarian(c)
            except Infeasible:
                assert expected is None
            else:
                assert expected is not None and result["cost"] == expected
                certificate(c, result)
            fixed += 1
    optional = 0
    for flat in product([None, -2, 0, 3], repeat=4):
        c = [list(flat[:2]), list(flat[2:])]
        alpha, beta = [1, 2], [2, 1]
        result = optional_assignment(c, alpha, beta)
        assert result["cost"] == brute_optional(c, alpha, beta)
        certificate(result["augmented"], result["certificate"])
        optional += 1
    assert hungarian([])["cost"] == 0
    assert hungarian([], ncols=3)["pairs"] == []
    assert hungarian([[], []])["pairs"] == []
    assert hungarian([[10, 20], [11, 1000]])["cost"] == 31
    assert isclose(hungarian([[1, 2], [1.1, 100]])["cost"], 3.1)
    assert hungarian([[10**30, 10**30 + 1], [10**30 + 2, 10**30]])["cost"] == 2 * 10**30
    assert hungarian([[0, 4], [4, None]])["cost"] == 8
    assert optional_assignment([[10]], [2], [3])["pairs"] == []
    assert optional_assignment([[4]], [2], [3])["pairs"] == [(0, 0)]
    assert optional_assignment([[None]], [2], [3])["cost"] == 5
    assert optional_assignment([], [], [2, 3])["cost"] == 5
    assert optional_assignment([[], []], [2, 3], [])["cost"] == 5
    assert optional_assignment([[-1]], [-3], [-2])["cost"] == -5
    rejected = 0
    for c in [[[float("nan")]], [[float("inf")]], [[-float("inf")]],
              [[True]], [[1], [2, 3]]]:
        try:
            hungarian(c)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid input accepted")
    assert rejected == 5
    import scipy
    from scipy.optimize import linear_sum_assignment
    cross = 0
    for c in [[[4, 1, 3], [2, 0, 5], [3, 2, 2]],
              [[1, 2], [1.1, 100]], [[-4, 2, 0], [1, -3, 5]],
              [[1, 2], [2, 1], [0, 4]], [[0, 0], [0, 0]]]:
        ii, jj = linear_sum_assignment(c)
        expected = sum(c[i][j] for i, j in zip(ii, jj))
        assert isclose(hungarian(c)["cost"], expected, rel_tol=1e-12, abs_tol=1e-12)
        cross += 1
    print(f"fixed={fixed}; optional={optional}; boundary=13; invalid={rejected}; scipy={cross}; version={scipy.__version__}")


run_checks()
```

执行方式（仓库根目录，PowerShell；只读取 Markdown，在内存中编译执行，不写脚本或缓存）：

```powershell
@'
from pathlib import Path
import re
p = Path('docs/algorithm_cards/hungarian.md')
blocks = re.findall(r'^```python\n(.*?)^```', p.read_text(encoding='utf-8'), re.M | re.S)
assert len(blocks) == 2
exec(compile('\n'.join(blocks), str(p), 'exec'))
'@ | .\.venv\Scripts\python.exe -B -
```

#### 验证记录与未覆盖项

2026-11-05 实际从本文提取两个 Python 块，在 Windows／PowerShell 5.1、仓库 `.venv` 的 **Python 3.13.14、SciPy 1.18.1** 执行，退出码0。系统 `python` 是不可用的商店别名，改用已有虚拟环境；未安装依赖。原始终端摘要为：

```text
fixed=8512; optional=256; boundary=13; invalid=5; scipy=5; version=1.18.1
```

| 实际检查 | 输入与断言 | 结果及限定 |
|---|---|---|
| 固定基数8512例 | 1×3、2×2、2×3、3×2；每格穷举 `{None,-2,0,3}`，分别64／256／4096／4096例；独立排列枚举判可行与最优值；可行解逐条验对偶、紧边、未用列零势能及gap | 全通过；包括负费、并列、禁边、全禁及Hall不足，不是8512个真实场景 |
| 拒配256例 | 2×2相同有限值域；α=[1,2]、β=[2,1]；独立枚举每行拒配或选列，再验扩展方阵证书 | 全通过；穷举只限这组惩罚，其他惩罚另有下列定点边界 |
| 边界13项 | 三种空形状；T04整数／小数例；10³⁰大整数；门控反例；拒配／接受／全禁；两侧分别为空；负拒配成本 | 全通过；小数例只断言目标近似，未对全部浮点输入证明精确证书 |
| 非法输入5项 | NaN、正无穷、负无穷、布尔、参差矩阵 | 均抛 ValueError，未将非法输入误当禁边 |
| SciPy交叉5例 | 官方3×3费用例、T04例、负费宽矩形、高矩形、全0并列；仅比较目标 | 全通过，容差 rel/abs 均为1e−12；不把JV运行等同本文机制或原版复现 |

验证范围只限文档参考的离线有限输入；未运行1955原程序、未执行 SciPy 内部源码逐行比对、未跑视频、未训练模型、未做速度对比。大规模随机矩阵、极端浮点动态范围、生产异常协议和真实关联质量仍未验证；L3／L5 保留“部分”。

### 五层追问链与参考答案

| 层级与问题 | 答案要点及成立条件 | 易错答案／反例 | 证据入口 |
|---|---|---|---|
| L1：T04 的2×2例如何从根走到最优解？ | 行1先占列1；行2经列1到行1，更新列2松弛并增广，得到交叉配对3.1；禁边时可能不可行或走拒配 | 逐行取最小得到101；只报最优索引而说不清前驱翻转 | L1逐轮表、`hungarian` |
| L2：为什么不能随意减大一点制造更多零？ | δ 为树到外部的最小松弛，保证约减费用非负且原匹配仍紧；保留树可保证访问进度 | δ超过最小松弛使边变负；矩形按列任意约减的 `[2,1]` 反例 | L2不变量；来源3、4 |
| L3：2轨迹、3检测就能拒掉差配吗？ | 固定基数仍强制2对；若允许拒配，设置同量纲 α、β 和 n+m 扩展，先查时间戳／成本／门控 | 把矩形剩余当双边拒配；单边阈值代替 α+β；先解后删不保最大合法基数 | L3目标表与门控反例；T04 |
| L4：如何证明输出全局最优，矩形还缺什么？ | 推导拉格朗日对偶、紧边和 gap=0；矩形 v≤0、未用列v=0；整数性避免 LP 舍入 | 只说“零元素最多”；把原文最大化的≥直接用于最小化；有禁边却默认一定可行 | L4公式；来源2–4；证书函数 |
| L5：禁边、负费、并列和精度如何测？ | 掩膜独立，Hall不足要失败；行最小势能接负费；并列只比目标与合法性；转置索引还原，检查原对偶残差；先穷举再剖析 | NaN当禁边；`1e9`必安全；要求并列时与库索引相同；把 SciPy JV 当 Kuhn1955；通过小例就称 MOT 已实现 | L5检查器与来源6 |

## 来源、获取层级与结论边界

获取时间均为2026-11-05。`B`＝书目，`A`＝摘要，`T`＝取得正文，`T部分`＝正文被截断或只读了特定内容。`P0`＝原始论文／官方文档，`P1`＝权威教学二手材料，`P2`＝专业社区；原文镜像仍是原文内容，但不是出版社授权状态或完整版本一致性的证明。

- <a id="source-1"></a>**[cite:1] Kuhn1955，出版社。P0，B+A。** [Wiley DOI入口](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109)取得标题、作者、March1955、卷期页码、摘要和参考文献；[PDF入口](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nav.3800020109)返回访问／购买页面，未从该入口取得正文。摘要明确最大评分、n人与n任务；不支持现代复杂度归属。
- <a id="source-2"></a>**[cite:2] Kuhn1955，Szeged大学课程原文镜像。P0内容、非出版社托管，T部分。** [Kuhn.pdf](https://www.math.u-szeged.hu/~hajnal/courses/BSc_Opkut/2022/Kuhn.pdf)取得原页83–89的解析内容：第1–2节、原第3节正整数模型、定理5–6及p.89单位预算调整。解析有OCR排版错误，后段截断，未完整取得第4节独立算法和第5节例题；不称通读15页。书目由来源1交叉，原机制由来源3、4在现代符号下交叉。另取得[2010重印的课程镜像](https://deltadbu.github.io/UCAS_algorithm_course/Lectures/Lec10-HungarianMethod-Kuhn.pdf)开头与作者回忆，标明 Springer 2010 章节及原文1955出处；属于同文重印，不计独立算法证据。
- <a id="source-3"></a>**[cite:3] Columbia，IEOR8100 Lecture6: The Hungarian Algorithm。P1，T。** [课程PDF](https://www.columbia.edu/~cs2035/courses/ieor8100.F12/lec6.pdf)已读 LP／对偶、约减费用、互补松弛、算法1、正确性与运行时间小节。其记号 n 为总顶点数、m 为边数，与本文矩阵维数不同；不直接复制其复杂度到本文代码。讲义有完整图／可行／整数数据假设，本卡禁边失败与矩形证明另行展开。
- <a id="source-4"></a>**[cite:4] NPTEL／IIT Delhi，Module9 Lecture6: Duality in Assignment Problems。P1，T。** [官方课程PDF](https://archive.nptel.ac.in/content/storage2/courses/111102012/module9/lect6.pdf)已读自由对偶变量、互补松弛、行列约减和划线更新的对偶解释，以及不平衡指派的dummy讨论。部分表格OCR混乱，仅采用可辨公式与正文；未借该文宣称本文双方拒配构造或浮点误差界是原文结论。
- <a id="source-5"></a>**[cite:5] CP-Algorithms，Hungarian algorithm for solving the assignment problem。P2，T部分。** [正文](https://cp-algorithms.com/graph/hungarian-algorithm.html)页面标记更新2023-12-13；取得势能、紧边、不变量、逐行搜索、`minv`维护及 O(n²m) 解释；返回在实现标题附近截断，未宣称取得并照抄完整 C++ 源码。现代循环结构由本卡独立展开、用穷举与证书验证；历史复杂度叙述属于社区二手线索，不独立锚定1955的渐近结论。
- <a id="source-6"></a>**[cite:6] SciPy 官方 linear_sum_assignment API。P0（软件行为文档），T。** [官方页面](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)明确矩形、maximize、输出索引以及修改 JV／无初始化；列出 D. F. Crouse，*On implementing 2D rectangular assignment algorithms*，IEEE TAES 52(4):1679–1696，August2016，DOI `10.1109/TAES.2016.140952`。本次只核验该官方引文，**未取得 Crouse 原文**，不得把其全文当已读来源；页面访问日期不当作库算法提出年，API“0.17.0加入”也不推断当时内部就是当前实现。

**交叉验证口径。** 原对偶紧边最优证书由原文定理5–6、Columbia和IIT讲义互证，标为 `[CONFIRMED，限定模型]`；1955书目由出版社与镜像一致确认，但两者并非独立研究。矩形证明、拒配构造、误差条件式与测试结论是本卡明确可复算的推导／验证，不冒充三篇论文分别给出的结论。SciPy 内部算法名称属 `[SINGLE-SOURCE，官方API]`；原文第4–5节、原版复杂度逐段考证、生产性能及实际关联质量均保留缺口。不以来源数量或卡片篇幅自动升级实现状态。
