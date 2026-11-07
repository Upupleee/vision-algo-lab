# 标定与坐标算法卡片：逐项 L1–L5

[学习规范](../algorithm_learning_standard.md) · [标定总览](../calibration_coordinates.md) · [预处理卡片](preprocessing.md)

## 范围、坐标与诚实边界

核对日期：2026-11-04。本文件覆盖 **10 项**：Zhang、Tsai–Lenz、Kannala–Brandt（KB）、EPnP、PoseNet、针孔、径向／切向畸变、单应、三角化、ChArUco。RANSAC 作为单应和 PnP 的可选鲁棒外层解释，不将它误算为内参模型。

默认使用右手坐标系、列向量，\({}^AT_B\) 将 B 系点变为 A 系点；相机光轴为 +Z，像素 (u,v) 对应数组 `[v,u]`。K 的焦距、主点用 px；三维点和平移统一用 m，若选 mm 必须全链统一；旋转内部用 rad。除非明确说明，\(X_c=R X_w+t\)，输出的是 world→camera，不是相机在世界中的位置。相机中心为 \(C_w=-R^Tt\)。

现有 [pose_ncc.py](../../traditional/template_matching/pose_ncc.py) 只求模板到图像的二维相似变换；[caliper.py](../../traditional/caliper_measurement/caliper.py) 输出像素坐标和像素 RMS；[synthetic.py](../../data/synthetic.py) 用给定 H 生成图像，不估计 H；[learned_matching.py](../../dl/learned_matching.py) 的 `findHomography` 只是文档计划，函数抛出 `NotImplementedError`。它们均不构成本文件算法的标定／物理坐标链实现。

全部十项**未发现专用实现、标定数据或物理验收链**。外部 OpenCV API 存在不等于仓库接入。文末代码仅为确定性坐标／矩阵／损失函数单元测试，不是相机采集数据、模拟性能实验或算法精度报告；未进行 PoseNet 训练、真实靶标检测、真实标定或机器人落点实验。

覆盖状态采用学习规范原词，逐层保守标“部分”：已有流程、机制、参数依据、数学推导、参考伪代码与数值单元测试，尚未完成完整实现复现、资料交叉复核和独立工程验证。五层追问的前提见对应 L 层，不以有答案代替现场验收。

| ID／算法 | L1 | L2 | L3 | L4 | L5 | 实现状态／验证范围 |
|---|---|---|---|---|---|---|
| C01 Zhang | 部分 | 部分 | 部分 | 部分 | 部分 | 未实现；已知单应约束与 K 恢复单元测试 |
| C02 Tsai–Lenz | 部分 | 部分 | 部分 | 部分 | 部分 | 未实现；非交换 AX=XB 代数测试 |
| C03 KB | 部分 | 部分 | 部分 | 部分 | 部分 | 未实现；角度多项式与反解测试 |
| C04 EPnP | 部分 | 部分 | 部分 | 部分 | 部分 | 未接入；投影约束与外部求解器冒烟 |
| C05 PoseNet | 部分 | 部分 | 部分 | 部分 | 部分 | 无权重／训练；四元数指标与损失测试 |
| C06 针孔 | 部分 | 部分 | 部分 | 部分 | 部分 | 未接入；前投影、射线、相机中心测试 |
| C07 畸变 | 部分 | 部分 | 部分 | 部分 | 部分 | 未接入；前投影与 Newton 往返 |
| C08 单应 | 部分 | 部分 | 部分 | 部分 | 部分 | 未实现估计链；归一化 DLT 测试 |
| C09 三角化 | 部分 | 部分 | 部分 | 部分 | 部分 | 未实现；双视图代数与深度传播测试 |
| C10 ChArUco | 部分 | 部分 | 部分 | 部分 | 部分 | 未接入检测／标定；角点 ID 布局测试 |

## 演进定位与工程主线

持续优化的是**成像模型、可观测性、采样设计、估计鲁棒性、误差传播和坐标链一致性**。工程顺序通常为坐标协议 → 成像模型 → 多姿态采集 → 初值 → 非线性重投影优化 → 留出图检查 → 独立物理控制点；手眼还需运动同步与机器人链闭环。

历史并非一条替代链：针孔／畸变决定成像；Zhang（ICCV 1999、TPAMI 2000）降低内参采集装置要求；KB（TPAMI 2006）扩展大视场模型；EPnP（ICCV 2007、IJCV 2009）在已有内参下估位姿；Tsai–Lenz（1989）连接机器人与相机坐标；PoseNet（ICCV 2015）学习场景重定位。ChArUco 是带 ID 的观测前端，可向几何求解提供对应点；单应是平面投影，三角化是多射线恢复深度。新方法不替代不相同的任务。

## C01 Zhang 平面靶标标定

任务／输入输出／前提：多视图平面点 (X,Y,0) 与像点 → K、畸变、每帧外参；靶标尺度和平整度可信、覆盖多姿态。演进定位：三维标定装置门槛高 → 平面未知姿态约束 → 线性初值及非线性精化；代价为采集设计，遗留参数耦合与靶板误差。实现：未实现；来源：[cite:1](#c-source-1)、[cite:2](#c-source-2)。

### L1 步骤
检测亚像素角点并按板坐标匹配 → 每视图归一化 DLT 得 H → 由正交约束堆 V → SVD 求 B → 恢复 K → 求各视图 R、t → 估畸变初值 → 最小化全图重投影误差。检测不足／V 病态拒绝；优化按相对下降、步长或迭代上限停止并报告原因。

### L2 机制
平面 Z=0 将投影变为 H=K[r₁ r₂ t]。同一相机的 K 跨视图共享，姿态变化提供独立约束。单张正视图无法稳健分离焦距和深度；仅增加同姿态照片不补可观测性。亚像素角点降低定位噪声，不自动修复板翘曲。

### L3 工程权衡
先设计倾斜、方位、距离及全视场覆盖，再决定是否增加畸变阶数。图像残差单位 px，格长和平移用 m；板尺度填错可能仍有低像素残差。先检查残差随半径／方向结构、V 奇异值及留出帧，再调优化器。一般含 skew 的 K 有5自由度，线性解通常需至少3个独立姿态单应；施加零 skew 等先验会改变自由度，不能把最小帧数当工程足够帧数。

### L4 推导
写 \(h_i\) 为 H 的第 i 列，\(B=K^{-T}K^{-1}\)。因为 \(r_1^Tr_2=0\) 且 \(r_1^Tr_1=r_2^Tr_2=1\)，消去每帧尺度得

\[
h_1^TBh_2=0,\qquad h_1^TBh_1-h_2^TBh_2=0.
\]

将对称 B 打包 \(b=(B_{11},B_{12},B_{22},B_{13},B_{23},B_{33})^T\)，有

\[
v_{ij}=(h_{1i}h_{1j},h_{1i}h_{2j}+h_{2i}h_{1j},h_{2i}h_{2j},
h_{3i}h_{1j}+h_{1i}h_{3j},h_{3i}h_{2j}+h_{2i}h_{3j},h_{3i}h_{3j})^T.
\]

每帧追加 \(v_{12}^T\) 与 \((v_{11}-v_{22})^T\)，解 \(Vb=0,\|b\|=1\)。理想可辨条件 rank(V)=5；B 的尺度任意，选符号使其正定。若 \(B=LL^T\) 为 Cholesky 分解，则 \(K=L^{-T}\)，再归一 K₃₃=1。非正定不能无条件加 epsilon 掩盖，应检查观测／模型与病态。

\(\lambda=1/\|K^{-1}h_1\|\)，\(r_1=\lambda K^{-1}h_1,r_2=\lambda K^{-1}h_2,t=\lambda K^{-1}h_3,r_3=r_1\times r_2\)。噪声下把 R 投影到 SO(3)，选择使点在相机前方的尺度符号。最终目标 \(\min_{K,d,R_j,t_j}\sum_{j,i}\|p_{ji}-\pi(K,d,R_jX_i+t_j)\|^2\)。固定规范且 J 满列秩、小高斯观测误差时局部协方差可近似 \(\sigma^2(J^TJ)^{-1}\)，不是全局误差上界。线性 V 有固定6列，可随视图数线性累计；非线性求解依赖参数数目、迭代和稀疏实现，不统一承诺 O(N)。

### L5 伪代码与数值测试
```text
require all_views_use_same_pixel_coordinate_system()
Hs = [normalized_DLT(board_xy, image_xy) for each valid view]
V = stack(v12(H), v11(H)-v22(H) for H in Hs)
b = right_singular_vector_of_smallest_singular_value(V)
require sufficiently_separated_null_direction(V)
B = unpack_symmetric(b); choose_sign_for_positive_definite(B)
K = inverse(transpose(cholesky(B))); K /= K[2,2]
for H in Hs:
    R,t = recover_and_orthogonalize_pose(K,H)
    require positive_depth(board,R,t)
K,d,poses = refine_reprojection(K,zero_distortion,poses,observations,stop_rules)
return parameters, per_view_residuals, convergence_status
```
数值测试：文末固定 K=[[800,0,320],[0,820,240],[0,0,1]]，由五组明确旋转与 t=(0.1,−0.2,3) 构造**单应矩阵测试夹具**，应 rank(V)=5，恢复 K 的绝对误差≤1e−5。该测试不检测图像、不估畸变、不验证非线性优化。另需实际实施：正视重复姿态的拒绝、缩放图同步更新 K、格长单位错配、留出图验证。优化用 QR/SVD 和稀疏求解，不直接求正规方程逆。

五层追问答案：L1“由角点怎么到内参？”→H→V→B→K→外参→联合优化；L2“多拍正视为何不够？”→独立约束不足；L3“低重投影但尺寸错？”→先查格长与平面性；L4“v₁₂ 从何而来？”→r₁、r₂ 的正交约束消尺度；L5“B 不正定？”→观测病态／符号／数值错误，不能只补 epsilon。易错：一张棋盘图恢复任意相机全部参数。

## C02 Tsai–Lenz 手眼标定

任务／前提：眼在手上系统的机器人姿态 \(G_i={}^bT_{g_i}\)、靶标视觉姿态 \(C_i={} ^{c_i}T_t\) → 固定 \(X={}^gT_c\)；靶标相对基座固定、同步采集、运动充分激励。演进定位：1989，末端到相机未知 → AX=XB 相对运动约束 → 分离旋转／平移；代价为运动与位姿误差，遗留旋转误差传播。实现：未实现；来源：[cite:3](#c-source-3)。

### L1 步骤
确认 G、C 方向与单位 → 对有效姿态对求 A=Gⱼ⁻¹Gᵢ、B=CⱼCᵢ⁻¹ → 由旋转四元数线性式求 R_X → 由平移线性式求 t_X → 独立检查 GᵢXCᵢ 是否恒定。无足够非平行旋转或矩阵秩不足拒绝；固定最小二乘求解后结束，可另外用非线性联合精化但不冒称原始分步法。

### L2 机制
固定靶标有 GᵢXCᵢ=T_bt，与 j 帧相等，左右消元得 AX=XB。纯平移不含足够旋转信息；所有旋转同轴会留下不可观测分量。将每个绝对姿态直接代 A、B 而不定义方向会得到另一问题。

### L3 工程权衡
先保证时间同步、单位和机器人姿态准确，再设计多个非平行旋转轴。过小旋转信息弱，接近180°影响参数化；所引 OpenCV4.10.0 会过滤最小四元数向量范数小于0.3或大于1.7的相对运动，这是实现策略，不是普遍最优采样角度。全姿态两两配对 O(m²) 且误差相关，更多对不等于更多独立观测。按平移、角度分别评估，不能混合量纲求单一平均。

### L4 推导
由齐次等式得到

\[
R_AR_X=R_XR_B,\quad (R_A-I)t_X=R_Xt_B-t_A.
\]

令旋转单位四元数为 (w,v)，\(P_A=2v_A=2\sin(\theta_A/2)u_A\)，B 同理。理想 A、B 为共轭旋转，角度相同，取一致四元数符号使 w_A=w_B。四元数乘法向量部分给

\[
w_X(v_A-v_B)+(v_A+v_B)\times v_X=0.
\]

令 \(p'=v_X/w_X\)，得到 \([P_A+P_B]_\times p'=P_B-P_A\)。堆叠后 SVD 最小二乘，恢复 \(v_X=p'/\sqrt{1+\|p'\|^2}\)、\(w_X=1/\sqrt{1+\|p'\|^2}\)，再由四元数构 R_X。接近 w_X=0 的参数化病态须另处理；噪声下角度不再严格相等，分步线性解不保证统计最优。平移式在多个充分激励轴下堆叠求解，若所有 R_A=I，则左边全零，t_X 无法辨识。O(P) 累积固定列数系统，P 为使用的相对运动数；未提供机器人实际落点的通用误差界。

### L5 伪代码与数值测试
```text
rows, rhs, pairs = [], [], []
for i,j in accepted_pose_pairs:
    A = inverse_rigid(G[j]) @ G[i]
    B = C[j] @ inverse_rigid(C[i])
    PA, PB = twice_quaternion_vector(A.R), twice_quaternion_vector(B.R)
    if informative_and_consistent(PA,PB):
        rows.append(skew(PA+PB)); rhs.append(PB-PA); pairs.append((A,B))
require stacked_rotation_matrix_has_rank3(rows)
p = least_squares_SVD(rows,rhs)
v = p/sqrt(1+dot(p,p)); w = 1/sqrt(1+dot(p,p))
RX = quaternion_to_rotation(w,v)
tX = least_squares_SVD(stack(A.R-I), stack(RX@B.t-A.t))
return RX,tX, residuals_AX_minus_XB(pairs,RX,tX)
```
数值测试：设 X 的旋转为 Z轴30°、平移 [1,2,3]，A 分别为 X/Y/Z轴60°并有不同平移，令 B=X⁻¹AX；文末从这些非交换矩阵求回 X，误差≤1e−8。若只测 X=I 或所有平移为0，很容易漏掉左右乘和 t 方向错误。另列纯平移、单轴和单位混用拒绝测试，尚未进行机器人验收。

五层追问答案：L1“输出谁到谁？”→相机到夹爪；L2“为何相对运动？”→消固定基座靶标变换；L3“落点随姿态变化？”→查同步、机器人误差和旋转激励；L4“平移式如何得？”→齐次乘法分块展开；L5“如何检出乘序错？”→非交换变换及逆变换闭环。易错：Tsai 的相机标定与 Tsai–Lenz 手眼是同算法。

## C03 Kannala–Brandt（KB）鱼眼模型

任务／前提：三维射线、鱼眼内参／角度系数 → 像点，或像点反求射线；标定时用覆盖中心至外围的多姿态对应点估参数。演进定位：TPAMI2006，透视加径向项在超大视角不便 → 入射角多项式；代价为外围采样与非线性反解，遗留单调性和外推风险。实现：未实现；来源：[cite:4](#c-source-4)、[cite:2](#c-source-2)。

### L1 步骤
本卡采用 OpenCV 常用四系数径向对称 KB 风格子模型：射线求 θ、方位 → θ_d 多项式 → 像素仿射。反投影：先逆 K → 半径 θ_d → 带区间限制 Newton 求 θ → 单位射线。标定时先选参数初值及有效视场，再迭代最小化重投影误差，达到步长／残差阈值或上限停止；越界、非单调或不收敛输出无效。

### L2 机制
用角度而非直接 tanθ 的径向高次扰动覆盖大视场。KB 原文含更广的建模项，不应把本卡四系数实现等同所有原文变体；也不能将普通 `undistort` 与鱼眼接口混用。外围没有观测就不能可靠确定高阶项。

### L3 工程权衡
θ 用 rad，k 在该角度约定下使用，f 用 px；先冻结模型阶数和有效 FOV，再扩采外围、评估残差，最后考虑更多自由度。θ=0 用极限避免0/0。OpenCV 常见模型由 x=X/Z、y=Y/Z、θ=atan(r) 构造，本卡限制 Z>0 的前半空间；不能由此宣称对后半球所有射线可用。

### L4 推导
径向对称映射中心对称，取奇数角度多项式

\[
\theta_d=\theta+k_1\theta^3+k_2\theta^5+k_3\theta^7+k_4\theta^9,
\quad x_d=\theta_d\cos\phi,\ y_d=\theta_d\sin\phi.
\]

以一阶系数固定为1，将总体尺度放进 f，避免两者任意缩放。像素 \(u=f_x(x_d+\alpha y_d)+c_x,v=f_yy_d+c_y\)。反解半径 ρ_d 时设 F(θ)=θ_d−ρ_d，

\[
F'(\theta)=1+3k_1\theta^2+5k_2\theta^4+7k_3\theta^6+9k_4\theta^8,
\quad \theta_{n+1}=\theta_n-F(\theta_n)/F'(\theta_n).
\]

在已验证区间 F′>0 才有唯一单调逆；Newton 初值与导数退化仍影响收敛，需区间保护。噪声角误差近似 \(\delta\theta\approx\delta\rho_d/F'\)，导数小则放大误差。标定对 k 的 Jacobian 包含 θ³、θ⁵等，近光轴采样使高阶列几乎为零。每点固定迭代 J 的成本 O(J)，标定另含多参数优化；无视场外通用界。

### L5 伪代码与数值测试
```text
theta = atan2(hypot(X,Y),Z); require Z > 0
rho = theta*(1+k1*theta**2+k2*theta**4+k3*theta**6+k4*theta**8)
if hypot(X,Y)==0: xd,yd = 0,0
else: xd,yd = rho*X/hypot(X,Y),rho*Y/hypot(X,Y)
u,v = fx*(xd+alpha*yd)+cx,fy*yd+cy
for inverse_iteration until abs(F)<1e-12 or 30 iterations:
    require derivative_positive_and_step_inside_calibrated_interval()
    theta -= F(theta)/dF(theta)
ray = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
```
测试：k₁=0.1、其余0、θ=0.5，ρ=0.5125；f_x=400、c_x=320、φ=0 时 u=525。反解须回0.5（容差1e−10）。k 全0 得等距映射 ρ=θ，**不是针孔** ρ=tanθ。另测光轴、有效视场边缘、多根参数的拒绝；完整鱼眼标定未执行。

五层追问答案：L1“反投影求什么？”→先角半径逆再射线；L2“k=0 是针孔？”→不是，是等距；L3“边缘误差大？”→先查外围覆盖与模型阶数；L4“唯一逆的条件？”→有效区间严格单调；L5“中心 NaN？”→0/0 分支。易错：任何 KB 参数都可全视场稳定反解。

## C04 EPnP

任务／前提：已知 K、匹配的 n 个3D点与无畸变2D点 → R、t；足够非退化对应，输出 world→camera。演进定位：ICCV2007／IJCV2009，点数增长使 PnP 成本高 → 四控制点及固定维度核心求解；代价为退化处理，遗留噪声／外点敏感。实现：未接入；来源：[cite:5](#c-source-5)、[cite:2](#c-source-2)。

### L1 步骤
去畸变／归一化像点 → 选世界控制点 → 求每点重心系数 α → 堆2n×12齐次矩阵 M → 提取小奇异值子空间 → 由控制点距离求候选 β 并小规模精化 → 恢复相机系控制点及点云 → 刚体配准求 R、t → 选择正深度且重投影最小候选。可外套 RANSAC 并最终重投影精化；零候选则失败。

### L2 机制
每个世界点用固定控制点线性表示，未知数不再随 n 增长。控制点间距离在刚体变换下不变，约束齐次方程的尺度和子空间组合。仅对 M 求最小奇异向量后随意归一，不能得到正确尺度。

### L3 工程权衡
至少4点是接口基本要求，不等于任何4点都可稳定求解。先检查3D构型、对应和 K，再选 RANSAC 重投影门限（px）；归一化坐标时门限不可仍直接用像素数。共线不可充分观测，近共面需相应控制点／平面处理或另选平面 PnP；通用四控制点求逆在严格共面时奇异。EPnP 论文的“非迭代”指主求解规模优势，实际 OpenCV 实现含 β 的小规模 Gauss–Newton，不能理解为程序没有任何迭代。

### L4 推导
\(X_i^w=\sum_{j=1}^4\alpha_{ij}C_j^w,\sum_j\alpha_{ij}=1\)，同样 \(X_i^c=\sum_j\alpha_{ij}C_j^c\)。归一化像点 (x_i,y_i) 满足 \(X_i^c-x_iZ_i^c=0,Y_i^c-y_iZ_i^c=0\)，故每点在 M 中的两行分别由 \([\alpha_{ij},0,-x_i\alpha_{ij}]\)、\([0,\alpha_{ij},-y_i\alpha_{ij}]\) 的四块组成。

设候选控制点向量 \(c=\sum_{l=1}^d\beta_l v_l\)，d≤4；每对控制点 j,k 有

\[
\left\|\sum_l\beta_l(v_{lj}-v_{lk})\right\|^2=\|C_j^w-C_k^w\|^2.
\]

展开为 β_lβ_m 的小型二次系统（6对控制点），用若干近似初始化及固定维度精化选候选，恢复尺度。刚体配准先去均值，若 \(H=X_w^TX_c=U\Sigma V^T\)，则 \(R=V\operatorname{diag}(1,1,\det(VU^T))U^T,t=\bar X_c-R\bar X_w\)，由最小化平方配准误差等价最大化迹得此正交解。M 的构造／固定12列求解和评价随 n 为 O(n)，RANSAC 外层另乘假设数，非线性精化另计。无外点时的算法也非任意构型精度保证。

### L5 伪代码与数值测试
```text
points2 = undistort_to_normalized(points_px,K,d)
controls_w = choose_centroid_and_principal_axis_controls(points3)
require affine_control_basis_not_singular_or_use_planar_branch()
alpha = solve_barycentric(points3,controls_w)
M = stack_normalized_projection_rows(alpha,points2)
basis = smallest_right_singular_subspace(M,dimension_up_to_4)
candidates = []
for beta0 in distance_constraint_initializations(basis,controls_w):
    beta = refine_six_control_distances(beta0,max_iterations=10)
    controls_c = reshape(basis@beta,4,3)
    points_c = alpha@controls_c
    R,t = rigid_alignment_with_det_plus_one(points3,points_c)
    if positive_depth(points3,R,t): candidates.append((R,t))
return candidate_with_minimum_reprojection_error_or_failure(candidates)
```
`distance_constraint_initializations` 应至少覆盖所引源码 `find_betas_approx_1/2/3` 的不同有效子空间候选；这仍是参考分层伪代码，不声称已手写全部 EPnP 内核。

数值测试：明确给定8个非共面点、R=Rodrigues([0.1,−0.2,0.05])、t=[0.2,−0.1,4]、K同C01，计算精确投影作为几何单元测试输入；外部 `solvePnP(...SOLVEPNP_EPNP)` 应恢复 R,t，误差≤1e−6，并正深度。这不是图像匹配或仓库 EPnP 实现。另需外点、共线、共面、K错配和双解测试；本次不把这些待验证项报告为通过。使用 contiguous float64、Nx3／Nx2，优化缓存 K⁻¹并按固定坐标协议比较。

五层追问答案：L1“已知哪些？”→K及3D–2D对应；L2“为何控制点？”→固定维度表达和刚性距离；L3“PnP失败先降阈值？”→先查构型／对应／K；L4“尺度哪里来？”→控制点间已知距离；L5“结果反射？”→SVD刚体配准强制 detR=+1并检查正深度。易错：EPnP 同时完成未知内参标定。

## C05 PoseNet 场景重定位

任务／前提：已训练场景的 RGB 图 → 场景参考系相机位置和旋转；需可信位姿监督与一致坐标约定。演进定位：ICCV2015，显式特征匹配在困难图像不稳 → CNN 直接回归位姿；代价为场景训练，遗留域偏移、尺度与精度限制。实现：未实现，无权重／场景位姿数据；来源：[cite:6](#c-source-6)。

### L1 步骤
固定场景坐标与图像预处理 → 按轨迹／区域切分训练验证 → CNN 输出位置 t̂ 与四元数 q̂ → 按训练定义算损失更新 → 验证早停。推理加载匹配权重并归一化四元数，返回位姿及有效性；零范数或域外输入应拒绝／回退几何方法，不捏造置信度。

### L2 机制
从场景外观学到位置先验，不从单幅未知场景无条件恢复可追溯尺度。相似走廊可能有不同真位姿，回归不能保证消除多解；把平移缩到另一单位却不调损失会改变训练任务。

### L3 工程权衡
位置单位来自监督场景，旋转四元数无量纲，β平衡两项梯度与量纲；先核对 pose 是 camera→world 还是反向、四元数排序和图像裁剪，再调 β。评估按留出空间／轨迹报告平移和角误差，不能随机相邻帧泄漏。论文及后续修订预印本的朝向数值有更正，本文不引用性能数值。

### L4 推导
原始 PoseNet 的代表性目标为 \(L=\|\hat t-t\|_2+\beta\|\hat q-q/\|q\|\|_2\)，不是把它未经说明写为所有项平方。工程可选符号不变的改造 \(L_q=\min(\|\bar q-q\|,\|\bar q+q\|)\)，其中 \(\bar q=\hat q/\|\hat q\|\)，**这是本卡建议的鲁棒表达，不冒称原始损失**。

四元数双覆盖给 q 与−q 同旋转。两单位四元数相对旋转角满足 \(\Delta\theta=2\arccos(|q_1^Tq_2|)\)，来自相对四元数标量部等于 cos(Δθ/2)。若 ||q̂||>0，归一化 Jacobian 为 \((I-\bar q\bar q^T)/\|\hat q\|\)，说明零范数附近梯度病态。平移用平方损失时最优为条件均值，用原始欧氏范数则是条件几何中位数，二者不能混淆。卷积复杂度取决于骨干；非凸训练无全局最优保证，无通用域外位姿误差界。

### L5 伪代码与数值测试
```text
backbone = instantiate_checkpoint_compatible_pose_network()
for epoch until validation_stop_or_limit:
    image,t,q = scene_grouped_training_batch()
    t_hat,q_hat = backbone(image)
    loss = norm(t_hat-t) + beta*norm(q_hat-normalize(q))
    optimizer_step(loss)
load_verified_weights(); set_eval_mode()
t_hat,q_hat = backbone(preprocess_rgb(frame))
if norm(q_hat) < quaternion_threshold: return invalid
return t_hat, normalize(q_hat), declared_pose_frame
```
数值测试：q=[1,0,0,0]、−q 描述同一姿态，角误差为0；未经符号处理的四元数欧氏距离却为2，揭示损失与指标差异。位置预测 [1,2,3]、目标 [1,2,4] 的平移误差为1（监督单位）；目标四元数同号且一致时原式总损失亦为1。文末只检查这些代数事实，不实例化／训练网络。优化必须保持 crop、归一化和权重一致，使用同一冻结集比较。

五层追问答案：L1“它输出什么？”→已知场景坐标系位姿；L2“能替代手眼吗？”→不能，缺少机器人固定变换；L3“换镜头失效？”→预处理／成像域改变；L4“q与−q距离2为何角度0？”→SO(3)双覆盖；L5“归一化爆炸？”→四元数近零范数要拒绝。易错：位置回归输出m单位就证明毫米精度。

## C06 针孔成像

任务／前提：世界3D点、K、R,t → 理想无畸变像点；或像点→相机射线。演进定位：成像几何基础模型，投影与物理尺度关联 → 相似三角形与齐次坐标；代价为理想化，遗留镜头畸变、遮挡与未知深度。实现：无专用投影接口；来源：[cite:2](#c-source-2)、[cite:7](#c-source-7)。

### L1 步骤
X_c=RX_w+t → 检查Z_c>0 → x=X_c/Z_c,y=Y_c/Z_c → K映到像点。反投影 K⁻¹[u,v,1] 得方向，可归一为单位向量；没有深度／平面约束时只返回射线，不返回唯一空间点。

### L2 机制
透视相似三角形给坐标除深度；沿同一射线缩放三维点不会改变像点。R,t 表示 world→camera，t不是世界系相机位置。物体在图上小可能因尺寸小或离相机远，单图不能直接判尺度。

### L3 工程权衡
fx,fy,cx,cy 为 px；明确 skew 是否允许，本卡多数例子设0。裁剪偏移 (u₀,v₀) 后 c减偏移；缩放要同时更新 f和c，并考虑半像素规则。先核对图像几何变换历史再认为“标定漂移”。点深度接近0时投影敏感应拒绝。

### L4 推导
相似三角形得 \(u=f_xX_c/Z_c+s_kY_c/Z_c+c_x,v=f_yY_c/Z_c+c_y\)，即 \(Z_c\tilde p=K[R|t]\tilde X_w\)。零skew下 Jacobian

\[
J_\pi=\begin{bmatrix}f_x/Z&0&-f_xX/Z^2\\0&f_y/Z&-f_yY/Z^2\end{bmatrix}.
\]

由微分得小噪声协方差 \(\Sigma_p\approx J_\pi\Sigma_XJ_\pi^T\)，若K、R,t也随机须扩展。Z→0敏感度发散，且射线尺度为投影零空间。C_w满足RC_w+t=0，故 C_w=−Rᵀt。单点O(1)、n点O(n)，无未知深度的唯一逆。

### L5 伪代码与数值测试
```text
Xc = R @ Xw + t
if Xc.z <= depth_threshold: return invalid
p = K @ (Xc/Xc.z)
ray = solve(K,[u,v,1]); ray /= norm(ray)
camera_center_world = -transpose(R) @ t
```
测试：K=[[800,0,320],[0,800,240],[0,0,1]]，R=I、t=0，X=[1,2,4]→(520,640)；[2,4,8]同像点，反投影方向正比[0.25,0.5,1]。R=I、t=[1,2,3]时相机中心为[−1,−2,−3]。测试Z≤0拒绝和裁剪后主点；优化批投影不改变有效掩膜。

五层追问答案：L1“一个像素返回什么3D？”→射线；L2“为何不能知道深度？”→沿射线尺度不变；L3“resize后偏差？”→更新K及像素中心；L4“t是相机位置吗？”→不是，位置是−Rᵀt；L5“图外巨值？”→查Z和坐标单位。易错：像素小数位表示同等物理精度。

## C07 径向与切向畸变

任务／前提：针孔归一化点与5参数 (k₁,k₂,p₁,p₂,k₃) → 畸变点；逆过程迭代恢复无畸变点。演进定位：理想针孔无法描述径向弯曲／偏心 → 参数化补偿；代价为额外标定和逆解，遗留高阶过拟合。实现：未接入；来源：[cite:2](#c-source-2)。

### L1 步骤
前投影先3D→归一化点 → 径向与切向畸变 → K到像素。逆过程先逆K，再在归一化平面解二维非线性方程；Newton达到残差／步长阈值停止，迭代超限或Jacobian奇异输出无效。图像去畸变应对目标图每个像素寻找原畸变图采样位置，避免前向撒点空洞。

### L2 机制
径向项改变离主点的距离，切向项描述偏心的非径向扰动。系数作用于归一化点，不直接作用于像素半径。先把像素反投影，再应用同一模型才能闭环；已去畸变图不能再带原d求PnP。

### L3 工程权衡
系数依赖归一化坐标定义；fx改变时不能只挪主点。先查残差随半径与象限的形态，再增k阶数，不能以训练残差最低选模型。图像 `remap` 用float32坐标常足够，但标定优化用float64，采样误差和参数误差分别评估。

### L4 推导
令 r²=x²+y²、a=1+k₁r²+k₂r⁴+k₃r⁶，取径向对称的偶次半径展开以及切向二次项：

\[
x_d=xa+2p_1xy+p_2(r^2+2x^2),\quad
y_d=ya+p_1(r^2+2y^2)+2p_2xy.
\]

写 \(b=k_1+2k_2r^2+3k_3r^4\)，导数 \(a_x=2xb,a_y=2yb\)，得到

\[
J=\begin{bmatrix}
a+xa_x+2p_1y+6p_2x & xa_y+2p_1x+2p_2y\\
ya_x+2p_1x+2p_2y & a+ya_y+6p_1y+2p_2x
\end{bmatrix}.
\]

对 F(z)=d(z)−z_obs 做一阶Taylor展开 \(F(z+\Delta)\approx F(z)+J\Delta\)，解 \(J\Delta=-F(z)\) 得Newton步；detJ≈0时局部不可逆，多项式还可能全局折叠。每点迭代J次为O(J)，不保证任意初值全局收敛；逆误差约由J⁻¹放大，非通用理论界。

### L5 伪代码与数值测试
```text
xd,yd = inverse_K(pixel)
x,y = xd,yd
for iteration in range(30):
    f = distort(x,y,coeffs) - [xd,yd]
    if norm(f) < 1e-12: return [x,y],valid
    J = analytic_distortion_jacobian(x,y,coeffs)
    if condition_number(J) > accepted_limit: return invalid
    step = solve(J,f)
    x,y = safeguarded_step([x,y],-step,calibrated_domain)
return invalid
```
测试：x=0.5、y=0、k₁=0.1、其余0，r²=0.25，a=1.025，x_d=0.5125；f=800、c_x=320→u=730，无畸变为720。Newton需回[0.5,0]，容差1e−10。另需切向单项、Jacobian中心差分、零系数、远边缘多根拒绝和图像remap测试；本次仅标明实际执行的前后映射检查。

五层追问答案：L1“先K还是畸变？”→前向归一化→畸变→K；L2“p项作用？”→非径向偏心；L3“加更多k总更好？”→外围不足会过拟合；L4“逆解如何来？”→Taylor线性化求Newton步；L5“重复去畸变？”→检查图像坐标域及d是否已置零。易错：k₁乘像素半径平方。

## C08 单应变换与估计

任务／前提：同平面非退化点对应 → 3×3 H及内点／残差；用于平面映射，不恢复一般3D。演进定位：二维相似／仿射不足以表达透视 → 平面射影变换；代价为8自由度和估计敏感性，遗留离平面高度误差。实现：未实现估计链；数据生成已使用给定H，不算估计。来源：[cite:7](#c-source-7)、[cite:2](#c-source-2)。

### L1 步骤
统一无畸变坐标 → 点集中心化和尺度归一 → 建DLT矩阵 → SVD求h → 反归一H → 检查退化及重投影。可RANSAC筛内点后重估并优化几何误差。4对一般位置点为最小情况；有3点共线的最小样本应拒绝。达到鲁棒迭代上限或置信度目标后停止，仍可能失败。

### L2 机制
H定义齐次关系，只能整体乘任意非零尺度。归一化减少像素大坐标的条件数；代数残差小不等于几何误差小。空间不同高度通常不共用同一H；纯旋转相机视图是另一特殊适用情形。

### L3 工程权衡
RANSAC门限在输入图像坐标下用px，映射到物理平面后使用独立m误差。先检查点分布和离平面高度，再调阈值；全部点集中小区域时外推风险大。不要无条件令H₃₃=1，H₃₃近0时用Frobenius范数等尺度规范。

### L4 推导
由 \(\tilde p'\times H\tilde p=0\)，一对 (x,y)→(u,v) 产生两独立行

\[
[-x,-y,-1,0,0,0,ux,uy,u],\quad
[0,0,0,-x,-y,-1,vx,vy,v].
\]

堆 A 后最小化 \(\|Ah\|^2,\|h\|=1\)，解为最小奇异向量；理想rank(A)=8。源／目标相似归一矩阵T、T′使均值距离为√2，归一解H_n后 \(H=T'^{-1}H_nT\)。这是代数目标的闭式初值，后续应最小化像素传递误差。

RANSAC在N个对应中有I个内点、轮内均匀不放回抽4点且无退化时，一次全内点概率为 \(q=\binom{I}{4}/\binom{N}{4}\)。若各轮独立，K次都失败概率为(1−q)^K，目标成功率p给 \(K\ge\log(1-p)/\log(1-q)\)，取上整。常用q≈w⁴（w=I/N）是大总体近似；轮内独立有放回抽样才有该乘积形式，但重复点不能直接构成有效单应假设。未知内点率、样本退化或模型不适配时，公式不构成实际成功保证。q=0时无法成功，q=1且样本有效时一次即可。DLT固定9列成本O(n)，外层另乘假设数，未给离平面几何误差通用界。

### L5 伪代码与数值测试
```text
srcn,T = normalize_points_mean_distance_sqrt2(src)
dstn,U = normalize_points_mean_distance_sqrt2(dst)
A = stack_two_cross_product_rows(srcn,dstn)
require rank_at_least_8_and_well_conditioned_configuration(A)
Vt = svd(A,full_matrices=True).Vt
Hn = reshape(Vt[-1],3,3)
H = solve(U,Hn@T)
H = normalize_scale_without_forcing_tiny_h33(H)
return H, reprojection_residuals_and_finite_denominator_mask(H)
```
数值测试：单位方形四角经 H=[[2,0,10],[0,3,20],[0,0,1]] 到 (10,20),(12,20),(12,23),(10,23)，归一DLT应恢复这些点，容差1e−8；w=0.5,p=0.99用w⁴近似得到K=72，但若N=8、I=4且无退化，精确q=1/70，应取K=321。关键实现陷阱：恰4对时A为8×9，`full_matrices=False` 的经济SVD可能不给出9维右零空间末向量；本卡测试特意覆盖该情况。分母近0标无效；不得用任意小 epsilon 掩盖地平线点。

五层追问答案：L1“至少几对？”→4对一般位置；L2“为什么归一化？”→改善数值条件；L3“中心准边角差？”→覆盖不足／畸变／离平面；L4“DLT两行来源？”→叉积消齐次尺度；L5“四点SVD错？”→检查完整右零空间。易错：H任何时候都能用h₃₃归一。

## C09 三角化

任务／前提：至少两个已知投影矩阵及对应无畸变像点 → 世界3D点和有效性／不确定度；相机同步、非零视差。演进定位：单射线深度不可知 → 多视线相交／最小重投影；代价为对应和基线设计，遗留近平行射线病态。实现：未实现；来源：[cite:7](#c-source-7)、[cite:2](#c-source-2)。

### L1 步骤
同模型去畸变／校正 → 对每视图堆叉积线性约束 → SVD求齐次X → 检查W并除齐次尺度 → 检查双相机正深度、视线夹角、重投影误差 → 可非线性精化；质量不达标输出无效，不能只返回NaN前的巨大深度。

### L2 机制
一条射线有一个深度自由度，两条不同方向射线提供约束；噪声下通常不精确相交。只最小化代数误差会受尺度影响，重投影精化更贴近像点噪声模型。基线越大提升深度分辨但增加遮挡和对应难度。

### L3 工程权衡
基线B用m，焦距f用px，视差d用px；先确认同步、K、左右顺序与视差符号，再查匹配。近距离选基线需顾及共同视场，远处小视差需拒绝高不确定点。门限来自误差预算和夹角，而不是“一律亚像素即可靠”。

### L4 推导
像点 \(x\sim PX\)，叉积得每视图 \((uP_3-P_1)X=0,(vP_3-P_2)X=0\)，两视图堆4×4 A，以最小奇异向量求齐次X。射线平行或相机同中心时深度不可辨。

对理想水平校正双目，左相机原点、右相机中心(B,0,0)，\(u_L=fX/Z+c_x,u_R=f(X-B)/Z+c_x\)，相减得 \(d=fB/Z\)，故 \(Z=fB/d\)。微分 \(\partial Z/\partial d=-fB/d^2\)，仅视差误差随机时 \(\sigma_Z\approx(fB/d^2)\sigma_d=Z^2\sigma_d/(fB)\)。若B、f也随机须加入 \(\partial Z/\partial B=f/d\)、\(\partial Z/\partial f=B/d\) 及协方差。该一阶标准差非确定性界；小视差时线性近似也可能失效。每点固定视图数SVD为O(1)，总O(n)。

### L5 伪代码与数值测试
```text
A = stack(u1*P1[2]-P1[0], v1*P1[2]-P1[1], u2*P2[2]-P2[0], v2*P2[2]-P2[1])
Xh = smallest_right_singular_vector(A)
if abs(Xh[3]) < homogeneous_threshold: return invalid
X = Xh[:3]/Xh[3]
if not positive_depth_both_cameras(X): return invalid
if ray_angle(X) < minimum_angle or reprojection_error(X) > threshold: return invalid
return refine_reprojection_if_requested(X), uncertainty_model
```
数值测试：f=800、B=0.1、d=40→Z=2m；σ_d=0.2px→σ_Z≈0.01m。文末用X=[0.2,0.1,2]、P₁=K[I|0]、P₂=K[I|(-0.1,0,0)]的精确像点检查SVD回原点，误差≤1e−8。另测d=0、负视差、W≈0、低夹角与外点拒绝，未进行真实双目精度验收。批量SVD优化须保留逐点状态。

五层追问答案：L1“得到点后还查什么？”→深度、夹角和重投影；L2“基线大总好？”→遮挡和匹配困难增加；L3“远处乱跳？”→视差过小及同步／匹配；L4“误差为何随Z²？”→对fB/d求导；L5“巨大有限坐标？”→W小不等于可用点。易错：重投影小便说明深度可信。

## C10 ChArUco 观测与标定

任务／前提：带字典ID的棋盘标定板图像及板定义 → 带ID的亚像素棋盘角点；多视图进一步求K,d及外参。演进定位：完整棋盘难以应对遮挡／点序歧义 → ArUco标记识别加棋盘角点定位；代价为字典／板版本依赖，遗留局部观测不足。实现：未接入；来源：[cite:8](#c-source-8)。不是一种独立的新内参成像模型，算法年份不从OpenCV版本号反推。

### L1 步骤
固定字典、方格数、格长、标记边长及板坐标版本 → 检测标记 → 按板几何插值／精化 ChArUco角点 → 按ID取物点 → 剔除不足／共线／覆盖差帧 → `calibrateCamera` 或匹配鱼眼模型估参数 → 留出帧验收。无检测或模型失败返回状态，不把少角点硬补成完整列表。

### L2 机制
标记ID解除局部遮挡下的对应歧义，棋盘交点提供几何定位。不能把 marker角点ID当 ChArUco棋盘角点ID；两者位置与集合不同。检测到ID不意味着该角点精确或板平整。

### L3 工程权衡
方格和标记尺寸统一m，满足标记边长小于格长并有足够像素边界；先检查打印尺寸、字典、板版本和清晰度，再调检测／角点精化参数。OpenCV4.6.0前后板坐标曾有不兼容变化，旧板还需确认 legacy布局。API／板Z轴方向以所用版本定义为准，不凭示意图猜测。部分可见帧可用，但仍需全局足够分布和姿态。

### L4 推导
对于平面板Z=0，已识别marker支持局部 \(s p=H[X,Y,1]^T\) 预测棋盘角点位置；若有K,d也可按已知姿态投影预测。随后角点精化以局部灰度结构约束位置，ID建立 \(X_i=\mathrm{boardPoints}[id_i]\)。多视图目标仍是 \(\sum_{j,i\in\mathcal V_j}\|p_{ji}-\pi(K,d,R_jX_{id_{ji}}+t_j)\|^2\)，只是每帧可见集合不同。固定K的单帧平面位姿与未知K多帧标定是不同可观测性问题；可见角点共线导致单应退化。检测成本依图像像素与候选数，求解成本依有效观测与迭代；ID正确不能给通用亚像素或毫米误差界。

### L5 伪代码与数值测试
```text
board = CharucoBoard(square_counts, square_length_m, marker_length_m, dictionary)
detector = CharucoDetector(board, version_matched_parameters)
objects, images = [], []
for frame in calibration_frames:
    corners,ids,marker_corners,marker_ids = detector.detectBoard(frame)
    if no_corners_or_bad_geometry(corners,ids): continue
    require unique_and_in_range(ids,board.chessboardCorners)
    object_points,image_points = board.matchImagePoints(corners,ids)
    objects.append(object_points); images.append(image_points)
require sufficient_multiview_coverage(objects,images)
result = calibrateCamera(objects,images,image_size,chosen_flags,stop_criteria)
return result, per_view_errors, board_definition_and_version
```
数值测试：5×4方格、格长0.04m、标记0.02m、`DICT_4X4_50`；内角点应12个，所选现代板定义ID0坐标[0.04,0.04,0]、ID5=[0.08,0.08,0]。输入ID顺序[5,0]必须保持对应输出该顺序，不能偷偷按ID排序却不重排像点。文末检查板对象与ID布局，不生成图片、不测试检测率、更不宣称已标定。另需旧版板、局部遮挡、重复ID、角点边界和毫米／米错配测试。

五层追问答案：L1“先识别谁？”→marker，后取ChArUco棋盘角点；L2“遮挡为何可用？”→ID提供局部对应；L3“轴向突然反了？”→检查板版本与点序；L4“它改了标定目标吗？”→主要改观测集合，仍重投影优化；L5“ID顺序错？”→物点与像点必须同序。易错：有4个任意ID便能完成全部内参标定。

## 可执行数值检查器

环境检查为 Python3.13.14、NumPy2.5.4、OpenCV5.0.0，`cv2.aruco`存在；[requirements.txt](../../requirements.txt)只有最低版本声明，不等于锁定此环境。外部资料固定4.10.0，API冒烟结果不证明跨版本等价。下列所有点、矩阵是明示的确定性测试夹具，不是图像数据集或性能数字。默认float64、`rtol=0`，近零判断只用于测试夹具，生产门限需按尺度设计。运行只向标准输出打印，不落盘。

```python
import math
import numpy as np
import cv2

def check(actual, expected, atol=1e-8):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)

def skew(v):
    x,y,z = v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def transform(R,t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def project(K,R,t,points):
    camera = (R@points.T).T+t
    assert np.all(camera[:,2]>0)
    p = (K@camera.T).T
    return p[:,:2]/p[:,2:]

def vij(H,i,j):
    a,b = H[:,i],H[:,j]
    return np.array([a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[1]*b[1],
                     a[2]*b[0]+a[0]*b[2],a[2]*b[1]+a[1]*b[2],a[2]*b[2]])

K = np.array([[800.,0,320],[0,820,240],[0,0,1]])
rotvecs = [[0.2,0,0],[0,0.3,0],[0.2,-0.3,0.1],[-0.4,0.2,0.1],[0.1,0.4,-0.2]]
rows = []
for rv in rotvecs:
    R = cv2.Rodrigues(np.array(rv,dtype=float))[0]
    H = K@np.column_stack([R[:,0],R[:,1],[0.1,-0.2,3.]])
    rows.extend([vij(H,0,1),vij(H,0,0)-vij(H,1,1)])
V = np.array(rows)
assert np.linalg.matrix_rank(V)==5
b = np.linalg.svd(V,full_matrices=True)[2][-1]
B = np.array([[b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]]])
if B[0,0]<0: B = -B
Kr = np.linalg.inv(np.linalg.cholesky(B).T); Kr /= Kr[2,2]
check(Kr,K,atol=1e-5)

RX = cv2.Rodrigues(np.array([0.,0.,math.pi/6]))[0]
X = transform(RX,[1.,2.,3.])
pairs, rows, rhs = [], [], []
for axis,t in zip(np.eye(3),[[0.2,0.3,0.4],[-0.3,0.1,0.5],[0.1,-0.4,0.2]]):
    A = transform(cv2.Rodrigues(axis*math.pi/3)[0],t)
    BB = np.linalg.inv(X)@A@X
    check(A@X,X@BB)
    vectors = []
    for R in [A[:3,:3],BB[:3,:3]]:
        rv = cv2.Rodrigues(R)[0].ravel(); theta = np.linalg.norm(rv)
        vectors.append(2*math.sin(theta/2)*rv/theta)
    PA,PB = vectors
    rows.append(skew(PA+PB)); rhs.append(PB-PA); pairs.append((A,BB))
AA = np.vstack(rows); assert np.linalg.matrix_rank(AA)==3
p = np.linalg.lstsq(AA,np.concatenate(rhs),rcond=None)[0]
v = p/math.sqrt(1+p@p); w = 1/math.sqrt(1+p@p)
Rr = (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*skew(v)
tr = np.linalg.lstsq(np.vstack([A[:3,:3]-np.eye(3) for A,BB in pairs]),
    np.concatenate([Rr@BB[:3,3]-A[:3,3] for A,BB in pairs]),rcond=None)[0]
check(Rr,RX); check(tr,[1,2,3])

theta = 0.5; rho = theta+0.1*theta**3
check(rho,0.5125); check(400*rho+320,525)
estimate = rho
for _ in range(10): estimate -= (estimate+0.1*estimate**3-rho)/(1+0.3*estimate**2)
check(estimate,theta,atol=1e-10)
assert abs(theta-math.tan(theta))>0.01

points = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                   [0,0,1],[1,0,1],[0,1,1],[1,1,1]],dtype=float)
R = cv2.Rodrigues(np.array([0.1,-0.2,0.05]))[0]; t = np.array([0.2,-0.1,4.])
pixels = np.ascontiguousarray(project(K,R,t,points))
ok,rv,tv = cv2.solvePnP(points,pixels,K,np.zeros(5),flags=cv2.SOLVEPNP_EPNP)
assert ok
check(cv2.Rodrigues(rv)[0],R,atol=1e-6); check(tv.ravel(),t,atol=1e-6)

q = np.array([1.,0,0,0]); qneg = -q
check(2*math.acos(np.clip(abs(q@qneg),0,1)),0)
check(np.linalg.norm(q-qneg),2)
check(np.linalg.norm(np.array([1.,2.,3.])-np.array([1.,2.,4.])),1)

Kp = np.array([[800.,0,320],[0,800,240],[0,0,1]])
check(project(Kp,np.eye(3),np.zeros(3),np.array([[1.,2.,4.],[2.,4.,8.]])),[[520,640],[520,640]])
check(np.linalg.solve(Kp,[520.,640.,1.]),[0.25,0.5,1])
check(-np.eye(3).T@np.array([1.,2.,3.]),[-1,-2,-3])

def distort(z):
    x,y = z; factor = 1+0.1*(x*x+y*y)
    return z*factor

observed = distort(np.array([0.5,0.])); check(observed,[0.5125,0])
check(800*observed[0]+320,730)
z = observed.copy()
for _ in range(10):
    J = (1+0.1*(z@z))*np.eye(2)+0.2*np.outer(z,z)
    z -= np.linalg.solve(J,distort(z)-observed)
check(z,[0.5,0],atol=1e-10)

def normalize_points(points):
    center = points.mean(axis=0)
    scale = math.sqrt(2)/np.mean(np.linalg.norm(points-center,axis=1))
    T = np.array([[scale,0,-scale*center[0]],[0,scale,-scale*center[1]],[0,0,1]])
    return (points-center)*scale,T

src = np.array([[0.,0],[1,0],[1,1],[0,1]])
dst = np.array([[10.,20],[12,20],[12,23],[10,23]])
sn,T = normalize_points(src); dn,U = normalize_points(dst)
rows = []
for (x,y),(u,v) in zip(sn,dn):
    rows.extend([[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])
AA = np.array(rows); assert AA.shape==(8,9) and np.linalg.matrix_rank(AA)==8
Hn = np.linalg.svd(AA,full_matrices=True)[2][-1].reshape(3,3)
H = np.linalg.solve(U,Hn@T); H /= H[2,2]
check(H,[[2,0,10],[0,3,20],[0,0,1]])
assert math.ceil(math.log(1-0.99)/math.log(1-0.5**4))==72

P1 = Kp@np.column_stack([np.eye(3),np.zeros(3)])
P2 = Kp@np.column_stack([np.eye(3),[-0.1,0,0]])
point = np.array([0.2,0.1,2.,1.]); rows = []
for P in [P1,P2]:
    pixel = P@point; u,v = pixel[:2]/pixel[2]
    rows.extend([u*P[2]-P[0],v*P[2]-P[1]])
Xh = np.linalg.svd(np.array(rows))[2][-1]
check(Xh[:3]/Xh[3],point[:3])
check(800*0.1/40,2); check(800*0.1/(40*40)*0.2,0.01)

assert hasattr(cv2,'aruco')
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((5,4),0.04,0.02,dictionary)
corners = board.getChessboardCorners()
assert corners.shape==(12,3)
check(corners[[5,0]],[[0.08,0.08,0],[0.04,0.04,0]],atol=1e-7)
print('C01-C10 数学与外部API单元测试通过；未完成真实标定、检测或训练')
```

## 来源与链接核验

以下记录区分来源身份、访问结果和实现版本；HTTP可达仅说明访问成功，不是内容全部正确或算法已复现。2026-11-04核验：标注“200”的入口取得正文并核对标题／关键公式符号；DOI仅202且空正文者明确受限。OpenCV4.x在线文档本次403，故用可访问的固定4.10.0官方源码／教程；本地执行版本5.0.0另记，绝不写成同版本复现。

- <a id="c-source-1"></a>**[cite:1]** Zhang，[Flexible Camera Calibration By Viewing a Plane From Unknown Orientations](https://www.microsoft.com/en-us/research/publication/flexible-camera-calibration-viewing-plane-unknown-orientations/)，Microsoft Research原始记录，ICCV1999，HTTP200。期刊扩展标题为 *A Flexible New Technique for Camera Calibration*，TPAMI2000；本次其微软页面超时，年代边界另参[领域总览](../calibration_coordinates.md)出版核验记录，不将超时当成功读取。
- <a id="c-source-2"></a>**[cite:2]** [OpenCV4.10.0 calib3d.hpp](https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/include/opencv2/calib3d.hpp)，HTTP200，针孔／畸变／鱼眼公式、标定／PnP／手眼API方向与原始论文索引。是官方实现文档，不是独立论文复现证据。
- <a id="c-source-3"></a>**[cite:3]** Tsai、Lenz，*A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration*，1989，[DOI](https://doi.org/10.1109/70.34770)返回202且空正文，**正文获取受限**；[OpenCV4.10.0 calibration_handeye.cpp](https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/src/calibration_handeye.cpp)返回200，核对`calibrateHandEyeTsai`、相对运动方向、旋转线性式及平移式。公式以可读官方实现交叉核对，不冒称已读到DOI正文。
- <a id="c-source-4"></a>**[cite:4]** Kannala、Brandt，*A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*，TPAMI2006，[DOI](https://doi.org/10.1109/TPAMI.2006.153)返回202且空正文，**原文访问受限**；四系数θ多项式与接口由[cite:2](#c-source-2)官方头文件核对。作者书目站本次证书校验失败，未绕过证书验证。原始全模型与本文子模型严格区分。
- <a id="c-source-5"></a>**[cite:5]** [EPFL EPnP作者项目页](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/)，HTTP200，列ICCV2007与IJCV2009版本；[OpenCV4.10.0 epnp.cpp](https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/src/epnp.cpp)，HTTP200，核对重心坐标、`find_betas_approx_1/2/3`与`gauss_newton`。论文题名的O(n)不包含任意外层鲁棒搜索开销。
- <a id="c-source-6"></a>**[cite:6]** Kendall、Grimes、Cipolla，[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html)，CVF正式ICCV2015入口，HTTP200。本卡不复述版本间被更正的性能数值，也不把学习式重定位称作内参标定。
- <a id="c-source-7"></a>**[cite:7]** Hartley、Zisserman，[Multiple View Geometry in Computer Vision, Second Edition](https://www.robots.ox.ac.uk/~vgg/hzbook/)，作者教材站HTTP200，投影、归一化DLT、退化构型与三角化证据入口。数学小例为本文展开，不是该教材的实测数据。
- <a id="c-source-8"></a>**[cite:8]** OpenCV4.10.0：[ChArUco检测教程源文](https://raw.githubusercontent.com/opencv/opencv/4.10.0/doc/tutorials/objdetect/charuco_detection/charuco_detection.markdown)、[ArUco／ChArUco标定教程源文](https://raw.githubusercontent.com/opencv/opencv/4.10.0/doc/tutorials/objdetect/aruco_calibration/aruco_calibration.markdown)，均HTTP200；含`CharucoDetector`、`matchImagePoints`和4.6.0后的板坐标变化说明。

## 验证范围与验收缺口

本次核验包括：10项逐层结构、显式锚点／相对文件存在、外部链接状态、文末确定性代数检查及外部EPnP和ChArUco板对象API冒烟。**没有**真实角点检测、Zhang含畸变联合优化、机器人标定、鱼眼全视场反投影、带外点PnP／单应、真实双目深度测试或PoseNet训练；这些L5部分均不能改标“已实现／现场验证”。

物理验收应使用未参加拟合的全视场控制点或量具，记录标定图和板／相机／镜头版本，报告方向分量、偏差、重复性、分位数、最大误差及拒绝率。像素重投影RMSE只能表示像点拟合；实际m误差还依赖靶标尺度、镜头状态、机械稳定、同步与坐标链。协方差近似不是确定性上界，优化收敛不是全局正确，输出小数位不代表物理精度。
