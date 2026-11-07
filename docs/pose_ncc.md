# 旋转／尺度 ZNCC 使用说明

## 定位

这是传统匹配基线，不涉及训练或权重。新增接口 `match_pose_ncc` 与旧平移接口并存，没有改变 `find_best_matches` 或 `shape_based.match` 的返回类型。形状匹配原有阈值行为尚未修改；本页的拒识保证仅针对新接口。

依赖复用 `requirements.txt`。当前验证环境为 Python 3.12，未新增第三方依赖。模板与图像要求非空、有限数值的二维灰度数组。模板有效区域至少四个像素且不能为常量；非法输入抛出 `ValueError`，不伪装成“没找到”。

## 可运行示例

在仓库根目录执行以下 Python 代码，使用已有合成图，无需外部图片：

```python
from data.synthetic import make_template_scene
from traditional.template_matching.pose_ncc import match_pose_ncc

image, template, truth = make_template_scene(noise=3.0)
result = match_pose_ncc(
    image,
    template,
    angle_range=(-10, 10),
    angle_step=5,
    scale_range=(0.9, 1.1),
    scale_step=0.1,
    min_score=0.8,
    min_score_gap=0.03,
)
if result["matched"]:
    match = result["matches"][0]
    print(match["x"], match["y"], match["angle_deg"], match["scale"], match["score"])
else:
    print(result["reason"], result["best_score"], result["score_gap"])
```

多目标调用需显式关闭唯一性拒识：`max_count=5, min_score_gap=0, candidate_limit=16`。掩膜传入与原模板同形的 `mask`，仅含 0/1，0 表示忽略，1 表示有效；忽略像素也必须是有限数值。模板旋转插值时自动排除补边和掩膜混合边界。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| angle_range / angle_step | (0,0) / 5 | 角度闭区间与离散步长，单位度；非整除时包含上端点 |
| scale_range / scale_step | (1,1) / 0.1 | 正尺度闭区间与离散步长；1 为原大小 |
| min_score | 0.8 | 原图有效区域 ZNCC 阈值，[-1,1]；不是概率 |
| min_score_gap | 0.03 | 单目标最优与次优空间候选的最低分差；0 禁用 |
| nms_radius | 8 | 空间峰值及跨姿态中心距离抑制半径，原图像素 |
| max_count | 1 | 返回数量上限；多目标需关闭分差拒识 |
| pyramid_levels | 3 | 最大空间金字塔层数，小模板自动减少；最大 8 |
| candidate_limit | 8 | 每姿态保留候选数量，至少覆盖 max_count；启用分差时至少 2 |
| subpixel | True | 原图评分峰的平移抛物线精化；无有限邻点时保留整数位置 |
| exhaustive_verification | True | 金字塔搜索后做原图全分辨率评分复核 |

角度和尺度的笛卡尔积最多 10000 个姿态。默认角度／尺度范围只搜索平移，必须显式扩大范围才会启用旋转／尺度搜索。建议先限制现场合理范围，再减小步长；不要首先降低拒识阈值。

## 返回协议与坐标

顶层返回普通字典，可直接 JSON 序列化：

- `matched`：是否存在通过当前策略的结果；失败时 `matches=[]`，不得直接取 `[0]`。
- `reason`：`matched`、`below_min_score`、`ambiguous` 或 `no_valid_pose`。
- `best_score`、`second_score`、`score_gap`：候选诊断；缺失用 `None`，常量图像没有有效峰。
- `elapsed_ms`：包含输入检查、模板变换、金字塔与原图复核的整次调用耗时。
- `searched_poses`：实际可评分姿态数；尺寸过大或变换后有效区域退化的姿态跳过。

每个匹配结果包含：

| 字段 | 坐标定义 |
|---|---|
| x, y | 原始模板像素 (0,0) 映射后的图像坐标；零角度、单位尺度时与旧接口左上角一致 |
| center_x, center_y | 原模板像素中心 ((w-1)/2,(h-1)/2) 的映射位置 |
| bbox_x, bbox_y | 旋转后包围画布的左上角，与 x/y 不一定相同 |
| angle_deg, scale | 被选中的离散姿态；正角遵循 OpenCV 视觉逆时针约定 |
| transform | 3×3 矩阵，原模板 [x,y,1] 列向量映射到图像 |
| corners | 原模板四角像素中心的映射：左上、右上、右下、左下；不是像素外边界 |
| score | 整像素峰上的 ZNCC；平移亚像素精化后不重新计算分数 |

数组索引仍是 `[y,x]`。评测时必须比较同一种参考点，不能把中心与左上角混用。

## 拒识与搜索边界

- `below_min_score` 仅表示未发现达到阈值的候选，不是物理上证明图中不存在目标。
- `ambiguous` 比较 NMS 后不同空间位置的峰；同一中心的旋转对称歧义、平坦峰形状、语义相似度不在此保证范围。
- 增大 NMS 半径可能把邻近真实目标合并，减小半径可能把同一目标的旁瓣当成竞争位置；必须结合目标尺寸和间距调参。
- `exhaustive_verification=True` 保留原图复核成本，不能声称金字塔已带来速度提升。它避免粗层截断丢掉某姿态的原图最佳峰，但每姿态候选上限和跨姿态合并仍可能限制多目标召回。
- `False` 只在粗层候选周围逐层搜索，属于近似快速模式。高频纹理、密集干扰或过小 candidate_limit 可漏检，也可能遗漏次优峰并错误放行唯一性判断。安全相关定位请保留默认复核。
- 当前仅空间粗到细，不是角度／尺度的自动层级精化。角度和尺度仍由网格决定；缩小步长可改善网格间目标得分，但增加耗时。
- 只支持完整变换包围画布落在图内、统一尺度、平面内旋转。不支持透视、非刚性形变或出画目标；局部遮挡只是有限合成验证，不是遮挡比例保证。
- 极端数值导致 FFT 方差不可靠时使用分块局部直接评分兜底，可能明显变慢。工业图像应保持合理灰度量纲。
- 本版未修改 ML/DL、未接真实相机、未承诺设备节拍或工业级定位精度。

## 验证

```powershell
python -B -W error -m unittest tests.test_pose_ncc
python -B -m tests.test_smoke
python -B -W error -m tests.benchmark_pose_ncc
```

第一项是边界／回归断言；第二项验证既有功能兼容性；第三项输出真实执行测得的合成场景指标，包含故意过粗采样的失败样本，不以退出码表示所有场景都检出。结果及运行环境见 [实验记录](experiments.md)。

以下六章补充原使用说明。实现细节以本仓库代码为准；外部资料用于解释数学和库接口，不代表本接口继承其全部能力。测试名称指向现有断言，不把测试容差当作现场精度承诺。

## 第一章：坐标与变换

### 1.0 任务定义

输入为搜索灰度图、模板、可选二值掩膜及搜索配置；任务是在给定角度／统一尺度网格和合法平移范围内寻找模板实例，输出通过分数与拒识策略的姿态，或明确返回未匹配原因。默认是单目标定位并检查不同空间候选的歧义；多目标任务须显式关闭分差拒识。这是已知模板的定位与有限相似变换估计，不是通用物体分类、任意单应配准或物理尺寸测量。任务验收应预先约定目标数量、坐标参考点、允许误差及误检／漏检口径。

### 1.1 从模板点到图像点

设原模板宽高为 `w,h`，图像坐标向右为 x 正向、向下为 y 正向，数组访问为 `image[y,x]`。像素中心从 0 编号，因此模板中心为

$$
c=\begin{bmatrix}(w-1)/2\\(h-1)/2\end{bmatrix},\qquad
A=s\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}.
$$

其中 `angle_deg` 是度，公式中的 θ 需转换为弧度；s 是正的统一尺度。正角将中心右侧的点转向上方，因而是视觉逆时针，不要套用 y 向上的坐标公式后再次取反。

OpenCV 首先产生绕 c 旋转缩放的映射 `q=A(p-c)+c`。令四角映射的逐坐标最小值为 ℓ，`_warp` 再减去 ℓ，使包围画布从零开始。若评分图整数峰为 `(u,v)`，精化偏移为 `(δx,δy)`，则最终

$$
b=\begin{bmatrix}u+\delta_x\\v+\delta_y\end{bmatrix},\qquad
t=c-Ac-\ell+b,\qquad
H=\begin{bmatrix}A&t\\0&0&1\end{bmatrix},\qquad
\begin{bmatrix}p'_x\\p'_y\\1\end{bmatrix}=H\begin{bmatrix}p_x\\p_y\\1\end{bmatrix}.
$$

`transform` 就是 H，`(x,y)=t`，`(bbox_x,bbox_y)=b`，`(center_x,center_y)=Ac+t`。零角度、任意正尺度时，原点与包围画布左上角重合，中心相差 `s*c`；单位尺度时相差 c。旋转后原点与画布左上角不一定重合。`subpixel=False` 只保证包围画布位置为整数，旋转后的原模板原点仍可能是小数。

### 1.2 画布、边界与 ROI

包围画布尺寸按四角像素中心计算：

$$
w'=\lceil s(|\cos\theta|(w-1)+|\sin\theta|(h-1))-10^{-9}\rceil+1,
$$
$$
h'=\lceil s(|\sin\theta|(w-1)+|\cos\theta|(h-1))-10^{-9}\rceil+1.
$$

`1e-9` 是实现中避免浮点误差多出一列／行的容差。搜索要求完整画布满足 `w'<=W, h'<=H`，有效整数平移范围为 `0<=u<=W-w'`、`0<=v<=H-h'`，评分图形状是 `(H-h'+1,W-w'+1)`。掩膜不能绕过画布必须完整落图的限制；精化也不是出画匹配支持。

`corners` 依次变换 `(0,0),(w-1,0),(w-1,h-1),(0,h-1)`，不是外边界 `(-0.5,-0.5)` 等点。若只传入裁剪 ROI，所有输出相对 ROI；调用方要回到整图，应左乘平移矩阵 `T(roi_x,roi_y)`，并同步平移位置与角点，不能只改中心。该接口没有相机标定或像素到毫米转换。

实现关联：[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 的 `_warp`、`match_pose_ncc`（画布尺寸预检及 `transform` 构造）。测试关联：[test_pose_ncc.py](../tests/test_pose_ncc.py) 的 `affine_truth`、`assert_pose`、`test_rotation_and_scale_search`、`test_image_edges_and_corners`、`test_template_fills_image`、`test_oversized_template_and_pose`。ROI 回填是坐标组合推导，不是现有专项测试的结论。

权威参考：OpenCV [几何变换 API](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)，重点是 `getRotationMatrix2D` 和 `warpAffine`。传给 `warpAffine` 的前向矩阵与其内部逆向采样不是两个相反的输出坐标协议。

## 第二章：ZNCC、掩膜与数值边界

### 2.1 真正参与计算的区域

对某个离散姿态，设变换后模板为 T，二值支持为 M，`Ω={i:M_i=1}`，`n=|Ω|`；在平移 `(u,v)` 下，对应图像窗口像素记为 Iᵢ。均值只在同一个 Ω 上计算：

$$
\mu_T=\frac1n\sum_{i\in\Omega}T_i,\qquad
\mu_I=\frac1n\sum_{i\in\Omega}I_i,
$$
$$
R(u,v)=\frac{\sum_{i\in\Omega}(T_i-\mu_T)(I_i-\mu_I)}
{\sqrt{\sum_{i\in\Omega}(T_i-\mu_T)^2}\sqrt{\sum_{i\in\Omega}(I_i-\mu_I)^2}}.
$$

定义要求两项能量均非零；本实现另外要求有效模板像素至少 4 个。分母是两个去均值向量的范数，不能把其中一个换成除以 n 后的标准差而不补偿系数。有限有效分数裁剪到 `[-1,1]`，越大越相似，但不是置信概率；0 不是“无目标”的专用标记。

在支持与几何对齐固定、没有裁剪饱和／量化损失时，若 `I_i=a*T_i+b` 且模板非恒定，则 `a>0` 得分为 1，`a<0` 得分为 -1，`a=0` 分母退化。一般正增益加常数偏置不改变 ZNCC；这不意味着反光、空间不均匀照明、阴影、非线性 gamma、遮挡和模糊也不影响评分。当前按有符号分数排序，不按绝对值匹配反相目标。

用户掩膜必须同形、有限且仅含 0/1，并非任意浮点权重。模板变换时同时线性插值 `template*mask` 与 `mask`，以支持值除回亮度（支持大于 `1e-9` 时）；最后仅保留 `support>=1-1e-6` 的像素。这会排除补零和有效／无效区域混合边界，而不是将补零当作黑色纹理加入相关。旋转缩放可能改变有效像素数，不能将不同姿态的分数理解为同样本数的统计置信度。

### 2.2 SAD、SSD、NCC 与 ZNCC 的区别

在同一有效区域 Ω 上，常见评分如下；本节将未去均值的归一化互相关称为 NCC，文献中 NCC 也可能泛指去均值版本，阅读时应看公式而非缩写。

| 方法 | 公式 | 优化方向与条件 |
|---|---|---|
| SAD | `Σ abs(Tᵢ−Iᵢ)` | 越小越好；直接受灰度偏置、增益和像素数量影响 |
| SSD | `Σ (Tᵢ−Iᵢ)²` | 越小越好；较大残差的影响被平方放大 |
| NCC | `Σ TᵢIᵢ / (sqrt(Σ Tᵢ²) sqrt(Σ Iᵢ²))` | 越大越好；非零向量下对正增益不变，但一般不对加性偏置不变 |
| ZNCC | 本章去均值公式 R | 越大越好；非退化条件下对正增益及常数偏置不变 |

SSD 可展开为 `ΣTᵢ²+ΣIᵢ²−2ΣTᵢIᵢ`，但图像窗口能量随位置变化，因此不能无条件把“最小 SSD”替换成“最大相关分子”。SAD/SSD 不是本接口可切换的评分模式；这些方法及 OpenCV 模式也没有跨方法通用阈值。参见本章 OpenCV 官方公式参考，其中 SAD 不在 `TemplateMatchModes` 六种模式之内。

### 2.3 FFT 路径与直接评分兜底

`_score` 的分子、局部和、局部平方和都使用翻转核的 `fftconvolve(...,mode="valid")`；新接口不是沿用旧接口的矩形积分图。令 `S=ΣIᵢ`、`Q=ΣIᵢ²`，则去均值能量为 `V=Q-S²/n`。代码先进行幅值归一化、去均值，并将 FFT 算出的 V 下限裁为 0；只有满足

$$
V>256\,\epsilon\,\max(1,|Q|)
$$

的窗口走 FFT 除法，这里的 Q、V 是归一化后数值，ε 为 `float64` 机器精度，不是可调的图像对比度阈值。模板还经参考像素相减、去均值和幅值缩放，以降低大偏置／极端幅值影响；这些数学上不改变相关系数的操作仍受浮点表示限制。

FFT 判为不可靠且整图并非常量时，代码按最多 128 个窗口一批，从原始图像取有效像素，局部幅值归一化、减参考值、去均值，再以向量点积直接评分。仍为常量的窗口保留内部 `-inf`，不会产生有限峰。这里不是用一个很小的分母强行生成高分；代价是低对比度背景或极端动态范围下可能明显变慢。已经因量化丢失的纹理不能靠兜底恢复。

原模板常量、有效像素不足等输入问题在搜索前抛出 `ValueError`；某一姿态变换后退化则跳过该姿态。常量搜索图像可以有已搜索姿态但没有峰，返回 `below_min_score` 且诊断分数为 `None`，不等同于 `no_valid_pose`。内部 `-inf` 不作为顶层诊断分数输出。

实现关联：[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 的 `_gray`、`_warp`、`_score`；对比 [ncc_matching.py](../traditional/template_matching/ncc_matching.py) 的 `match_template_ncc`。测试关联：[test_pose_ncc.py](../tests/test_pose_ncc.py) 的 `test_affine_illumination_and_noise`、`test_mask_excludes_corrupted_pixels`、`test_rotated_scaled_mask`、`test_all_ones_mask_matches_unmasked`、`test_large_offset_and_tiny_gain`、`test_huge_amplitude_self_match`、`test_unchanged_target_on_extremely_bright_background`、`test_no_target_constant_image`、`test_constant_or_insufficient_template`。反相增益的结论来自公式，现有照明测试只覆盖所列正增益样本。

权威参考：J. P. Lewis [Fast Normalized Cross-Correlation](https://scribblethink.org/Work/nvisionInterface/)（归一化原理及变换域计算）；OpenCV [TemplateMatchModes](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html)（区分 `TM_CCOEFF_NORMED` 与未去均值的 `TM_CCORR_NORMED`）；SciPy [fftconvolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html)（卷积、核翻转和 `valid` 输出范围）。本接口不是 `cv2.matchTemplate` 的封装，其输入类型、掩膜限制和退化处理不能直接套用 OpenCV 行为。

## 第三章：姿态网格与空间金字塔

### 3.1 姿态不是连续优化变量

`_grid` 从下端点按正步长采样，再在末项与上端点相差超过绝对容差 `1e-10` 时补入上端点；计数用 `floor((hi-lo)/step+1e-10)`。因此最后一个间隔可能小于步长，范围是带浮点容差的闭区间。角度、尺度两组采样取笛卡尔积逐个搜索，最多 10000 个组合；范围、步长须有限有序，尺度须正。角度不自动模 360 去重，重复方向可能浪费计算。

细化步长只减少离散姿态误差，不保证分数必然过阈值。几何上，距模板中心 r 个像素的点，在固定尺度 s 下因角度偏差 Δθ（弧度）产生位移 `2sr|sin(Δθ/2)|`，小角度时约为 `sr|Δθ|`；固定角度下尺度偏差 Δs 产生位移 `r|Δs|`。这些是几何位移关系，不是本程序误差上界或采样自动选择规则；大模板外围尤其容易受粗网格影响。

### 3.2 每个姿态如何粗到细

- 先在原模板上生成当前角度／尺度的画布和支持，再构建该姿态的空间模板金字塔；并非先搜索粗角度再自动细化角度。
- 图像每层以 `INTER_AREA` 缩到 `((w+1)//2,(h+1)//2)`。模板以相同尺寸规则缩小加权像素和支持，再二值化支持；这是面积重采样，不是 `pyrDown` 的高斯金字塔实现。
- 对模板，若当前最短边小于 16 就不再下降；若下一层有效支持少于 16，或下一层评分检查抛出 `ValueError`，也停在当前层。实际层数按姿态变化，`pyramid_levels` 是含原图层的上限，不保证达到指定值。
- 顶层全图评分，保留最多 `candidate_limit` 个峰，半径换算为 `nms_radius/2**level`。逐层上升时把整数候选坐标乘 2，在各方向 ±4 个该层像素范围内重新评分，并裁到合法位置；每个局部窗口最多取 2 峰，再按分数和层级半径合并、截断到候选上限。

奇数尺寸的真实缩放比例不总是精确 1/2，重采样还会改变高频纹理的相位和对比度。因此“乘 2 并局部搜索”是候选传播策略，不是对所有图像的无损推理。

### 3.3 全分辨率复核的准确含义

当实际顶层 `top>0` 且 `exhaustive_verification=True` 时，`_search_pose` 最后重新计算该姿态原图完整评分图，并用该评分图提取的峰替换粗到细结果，而不只是重算已保留候选的得分。这样避免粗层候选截断漏掉该姿态原图最佳峰，但仍有每姿态候选上限与跨姿态 NMS。

当 `top=0`（主动只用一层或小模板回退）时，初始评分已经是原图完整评分，复核开关不触发额外全图评分；所以 `False` 并不总是局部近似。只有实际使用粗层又关闭复核时，才依赖传播得到的局部候选。该模式既可能漏目标，也可能漏竞争峰从而把本应拒识的情况放行。

全分辨率复核保留每个可用姿态的原图计算成本，不能据“用了金字塔”宣称加速。减小 ROI、收窄物理合理姿态范围通常比盲目增层更有针对性；关闭复核前应单独评估漏检与错误唯一性放行风险。

实现关联：[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 的 `_grid`、`_search_pose` 及图像金字塔构造。测试关联：[test_pose_ncc.py](../tests/test_pose_ncc.py) 的 `test_grid_includes_nondivisible_upper_bounds`、`test_small_template_pyramid_fallback`、`test_pyramid_and_single_level_consistency`、`test_exhaustive_verification_defaults_to_true`、`test_high_frequency_odd_phase_pyramid_recovers_exact_target`、`test_approximate_mode_runs_with_and_without_pyramid`；后一个测试仅证明其指定场景可运行，不证明近似模式普遍等价。网格外目标的粗／细采样对照见 [benchmark_pose_ncc.py](../tests/benchmark_pose_ncc.py) 的 `off_grid` 与 `off_grid_fine`。

权威参考：OpenCV [resize 与 INTER_AREA](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)；Lewis [原文](https://scribblethink.org/Work/nvisionInterface/) 的多分辨率讨论指出粗到细需要足够的低频信息。本实现的 ±4 范围、16 像素停止条件和复核开关来自仓库代码，不是外部论文的性能保证。

## 第四章：平移亚像素精化

### 4.1 三点抛物线

取某一轴上整数峰左右三个评分 `f₋,f₀,f₊`，将其看作自变量 `-1,0,1` 上的二次函数采样。由二次项和一次项求顶点可得

$$
d=f_- -2f_0+f_+,\qquad
\delta=\frac{f_- -f_+}{2d}.
$$

x、y 轴分别拟合，返回顺序为 `(dy,dx)`，再加到画布平移。这不是带 xy 交叉项的二维曲面拟合，更不是对角度、尺度或原图像素进行连续联合优化。抛物线近似在孤立、平滑且局部凹的峰附近才有合理的峰位解释。

### 4.2 实际保护条件

- `match_pose_ncc` 先检查峰附近裁剪到评分图内的最多 3×3 邻域是否全部有限；只要有非有限值，就同时放弃两轴精化。即使异常值只在对角邻点，也会触发这一保护。
- `parabolic_subpixel` 对每个轴单独检查左右邻点是否存在；在评分图边缘，该轴偏移为 0，但另一个轴可能仍可精化。近似模式的评分图可能只是局部搜索窗口，窗口边界同样限制精化。
- 若 `abs(d)<1e-12`，该轴偏移为 0；否则按公式计算并裁剪到 `[-0.5,0.5]`。代码没有额外要求 `d<0`，也没有 Hessian 负定、拟合残差或峰锐度验证；因此不能把“返回小数”当作已认证的稳定极大值。
- 精化后更新 `transform`、原点、中心、画布位置与角点，但不重新计算 `score`；排序、分数阈值和分差仍使用整数峰评分。最终跨姿态 NMS 用精化后的中心，因此开关也可能影响接近半径边界的候选合并。

“亚像素”描述估计坐标的分辨率，而非绝对准确度。模糊、重复纹理、粗姿态网格、插值偏差及噪声会影响峰位；不能把某个测试中 `tolerance=0.2` 等断言转换为真实相机的统一误差指标。

实现关联：[ncc_matching.py](../traditional/template_matching/ncc_matching.py) 的 `parabolic_subpixel` 与 [pose_ncc.py](../traditional/template_matching/pose_ncc.py) 的邻域有限性保护。测试关联：[test_pose_ncc.py](../tests/test_pose_ncc.py) 的 `test_fractional_translation_subpixel`、`test_constant_neighbor_windows_keep_subpixel_pose_finite`、`test_image_edges_and_corners`、`test_template_fills_image`、`test_legacy_ncc_interfaces`。现有测试没有对所有峰形逐一验证曲率条件；三点公式为上述二次函数推导，具体截断策略以本地实现为准。

参考边界：OpenCV [几何变换与插值](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) 解释图像重采样；它与这里对“评分曲线”做抛物线拟合是不同操作，不应混称为同一个亚像素算法。

## 第五章：拒识与多目标

### 5.1 候选、NMS 与分差

`_peaks` 每次取剩余评分最大值，再抑制与该峰距离 **小于等于** 半径的评分点；取到非有限值即停止。这里是贪心取峰，不额外验证严格局部极大或平坦峰形状。每姿态最多保留 `candidate_limit` 个候选，不保证取完所有超过阈值的位置。

所有姿态候选按整数峰分数降序排列，跨姿态只保留中心距离与每个已保留项均 **严格大于** `nms_radius` 的候选。它是中心距离 NMS，不是包围框 IoU，也不使用角度／尺度距离；完全相等的半径距离会被合并。同中心不同姿态可互相抑制，所以不检测同心旋转对称歧义。`subpixel=True` 时此处中心已经精化，而每姿态 `_peaks` 抑制基于整数评分坐标。

设合并后前两名评分为 `b₁,b₂`，则 `score_gap=b₁-b₂`。先取这两个诊断分数，再执行 `min_score` 过滤；因此次优即使低于分数阈值，也可能参与唯一性判断。不存在次优时 `second_score`、`score_gap` 都是 `None`，不是 0 或无穷大，也不会触发分差拒识。

### 5.2 返回原因的判定顺序

| 顺序 | 条件 | 返回或行为 |
|---|---|---|
| 输入检查 | 图像、模板、掩膜或配置非法 | 抛出 `ValueError`，不返回拒识字典 |
| 1 | `searched_poses==0` | `no_valid_pose`，例如所有姿态画布过大／有效支持退化 |
| 2 | 没有评分 `>=min_score` 的保留候选 | `below_min_score`；可有有限低分，也可没有有限峰 |
| 3 | 已有过分数阈值候选，且 `min_score_gap>0`、gap 非空、`gap<min_score_gap` | `ambiguous`，清空 `matches`，保留诊断分数 |
| 4 | 其他情况 | `matched`，输出过阈值候选的前 `max_count` 项 |

等于 `min_score` 会通过分数筛选；等于 `min_score_gap` 不会因分差拒识。分差配置范围为 `[0,2]`，分数范围为 `[-1,1]`；这些闭区间和比较符号来自实现，不代表已为每个浮点等号边界设置专项测试。

`searched_poses` 是实际完成评分搜索的姿态数，不是有峰的姿态数、理论网格组合数或输出数。即使常量图像没有峰，也可以大于 0。所有失败结果必须先检查 `matched` 再读取 `matches[0]`；低分与歧义都不能解释为已证明目标不存在。

### 5.3 多目标的容量与保证

多目标必须显式设置 `max_count>1,min_score_gap=0`，否则抛出 `ValueError`。同时 `candidate_limit>=max_count`；单目标启用分差时 `candidate_limit>=2`，关闭时至少 1。计数参数必须是正整数，布尔值不能充当计数；`nms_radius` 必须为正有限数。

`max_count` 是上限，不是要求返回的数量；`candidate_limit` 是每姿态容量，不是全局目标数保证。周期纹理可占满候选容量，较大半径可合并真实相邻目标，较小半径可把旁瓣当目标。多目标关闭的是分差拒识，不是分数阈值或 NMS；即使开启原图复核，也不能据返回项较少推断真实目标数量。

实现关联：[pose_ncc.py](../traditional/template_matching/pose_ncc.py) 的 `_peaks`、参数检查、`distinct` 合并和返回分支。测试关联：[test_pose_ncc.py](../tests/test_pose_ncc.py) 的 `test_equal_targets_are_ambiguous`、`test_multiple_targets_and_output_limit`、`test_multiple_rotated_scaled_targets`、`test_small_nms_preserves_close_target_ambiguity`、`test_small_nms_returns_both_close_targets`、`test_nms_suppresses_alternative_poses`、`test_invalid_search_configuration`。

权威参考：OpenCV [matchTemplate](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html) 说明相关类评分取最大值的基础；本章 NMS、分差和返回原因是本仓库特有策略，不由该 API 或论文背书为通用拒识保证。

### 5.4 峰旁瓣比：理论扩展，尚未实现

PSR（Peak-to-Sidelobe Ratio）可写成 `PSR=(峰值−旁瓣均值)/旁瓣标准差`。旁瓣统计需要明确排除主峰邻域，并规定统计 ROI、无效值处理和标准差退化行为；排除窗口不同，指标也不可直接比较。分差只比较两个峰，PSR 则比较主峰与一组背景响应的统计差异，两者都不是目标存在概率。

重复纹理可能产生多个尖峰，低噪声背景也可能放大 PSR；它不能代替几何验证或无目标样本标定。本接口没有计算／返回 PSR，也没有将其作为拒识条件。若后续接入，必须先确定每姿态还是跨姿态统计，并单独验证误检和漏检。原始应用参考：Bolme 等 [Visual Object Tracking using Adaptive Correlation Filters](https://doi.org/10.1109/CVPR.2010.5539960)，其跟踪场景阈值不能直接移植为此处模板定位阈值。

## 第六章：现场排障与验证方法

### 6.1 先保留证据，再调参数

建议保留失败原始灰度帧、原模板、掩膜、ROI 原点、完整调用参数、数组形状／类型／有限性／有效区域对比度、依赖版本，以及全部顶层诊断字段。接口不会自动保存这些信息，也不输出评分图、每姿态失败原因或各阶段耗时；仅凭一个 `best_score` 无法区分全部成因。下面是排查建议，不是已经接入真实相机后的实验结论。

| 现象 | 优先核对 | 有针对性的动作与现有证据 |
|---|---|---|
| `ValueError` | 二维非空、NaN/Inf、二值掩膜、有效像素数／常量、配置合法性 | 修复输入，不用降分数阈值掩盖问题；见 `test_invalid_image_and_template`、`test_invalid_masks`、`test_invalid_search_configuration` |
| `no_valid_pose` | 变换后完整画布能否落图、尺度范围、支持退化 | 扩大有物理依据的 ROI 或修正尺度；掩膜并不能支持出画目标；见 `test_oversized_template_and_pose`、`test_search_skips_only_oversized_poses` |
| 分数为空但已搜索姿态 | 图像或窗口是否常量／纹理已量化丢失 | 检查曝光、位深与预处理，不把 `None` 转成可接受的 0；见 `test_no_target_constant_image` |
| 目标可见但低分 | 姿态是否在范围内／网格之间、模糊、反相、饱和、遮挡、模板过旧 | 固定图像先比较合理的粗细姿态网格，再检查成像；参照基准 `off_grid/off_grid_fine` 和 `test_partial_occlusion`，不能推导通用遮挡比例 |
| 高分但 `ambiguous` | 是否真实双目标、重复纹理，NMS 是否过小 | 单目标任务保持拒识并改善模板／ROI；确为多目标才关闭分差；见 `test_equal_targets_are_ambiguous` |
| 多目标少检／同目标重复 | 候选是否截断，半径与实际间距是否匹配 | 联合检查 `candidate_limit`、`max_count` 和半径，不只调其中一个；见两个 `test_small_nms_*` 测试 |
| 同图单层能找到，多层近似模式漏检 | 是否关闭原图复核，高频与奇数相位是否影响粗层 | 恢复 `exhaustive_verification=True`，以单层作对照；见 `test_high_frequency_odd_phase_pyramid_recovers_exact_target` |
| 画框偏移、角度方向反、精化后分数不变 | 是否混用原点／中心／画布、遗漏 ROI 偏移、误解角度符号 | 用 H 统一变换相同参考点；分数不重算是协议行为；见 `assert_pose`、`test_legacy_ncc_interfaces` |
| 坐标抖动或精化仍是整数 | 有限邻域、边界、平坦峰、粗姿态网格 | 固定姿态对比 `subpixel=False/True`，整数结果不一定是故障；见 `test_fractional_translation_subpixel`、`test_constant_neighbor_windows_keep_subpixel_pose_finite` |
| 耗时突增 | 姿态总数、原图复核、有效支持和极端灰度动态范围 | 固定输入比较参数配置，先收窄合理搜索范围；极端值直接评分兜底可能昂贵，不能承诺金字塔加速 |

表中 `test_*` 均位于 [test_pose_ncc.py](../tests/test_pose_ncc.py)，基准场景位于 [benchmark_pose_ncc.py](../tests/benchmark_pose_ncc.py)。改善曝光、光学、模板或 ROI 的建议仍需现场样本验证，不是这些合成测试已覆盖的硬件能力。

### 6.2 如何解释测试与基准

现有单测通过 `affine_truth` 构造几何真值，用 SciPy `map_coordinates` 线性采样生成场景；不是以待测 `_warp` 直接生成全部真值。它们覆盖有限的合成样例、边界与回归，不覆盖真实相机噪声模型、标定误差、批次变化或生产节拍。

[benchmark_pose_ncc.py](../tests/benchmark_pose_ncc.py) 的协议是每场景 3 个固定种子，每个种子预热 1 次、计时 3 次。耗时报告汇总 9 次调用的中位数；检测计数与误差取每种子最后一次结果，不是把 9 次计时当作 9 个独立场景。预测与真值先按中心距离做一对一分配，再要求中心误差 `<=2 px`、圆周角差 `<=7.5°`、尺度绝对误差 `<=0.1` 才计为命中。位置、角度与尺度误差最大值只统计已命中的配对，必须和漏检／误报计数一起读。

`ambiguous` 场景虽放置两个实例，评测期望是拒识，因此将期望列表置空；其目标数不能被当作普通多目标召回分母。`off_grid` 故意使用过粗姿态采样，不能只看程序正常退出就宣称全场景通过。基准不比较关闭复核后的速度，也不提供现场延迟分位数或工业精度认证。

实际部署应在独立且有标注的现场正负样本上联合选择 `min_score`、`min_score_gap`、半径和候选容量，分别统计漏检、误报、歧义拒识及定位误差，并单独验证近似模式。不要依据单张正样本把阈值降到恰好通过，也不要把相关系数当作目标存在概率。

### 6.3 精度、重复性与可复现性

准确度描述估计值与独立真值的接近程度，重复性描述相同测量条件下多次结果的离散程度，两者不能互相替代。固定目标重复采集 N 帧（N 至少为 2），统一使用中心坐标 `p_k`，可报告均值 `p̄=Σp_k/N`、偏差向量 `p̄−p_truth`、定位 RMSE `sqrt(Σ||p_k−p_truth||²/N)`，以及逐轴样本标准差 `s_x=sqrt(Σ(x_k−x̄)²/(N−1))`、`s_y` 同理。角度使用圆周差，尺度使用绝对差；同时报告总帧数、成功匹配数及拒识／漏检，不能只展示成功帧的低误差。

同一数组反复运行主要检查算法可复现性与耗时，不等于重新曝光采集的测量重复性；跨批次、光照、温度或重新装夹的变化应另行分组。固定种子只固定合成输入，不保证跨依赖版本／硬件逐位一致。当前基准没有统计重复采集的坐标标准差，其三次重复用于计时，不能当作重复性认证。像素误差转换为物理误差还需标定及其不确定度；返回小数、测试容差和网格步长均不是现场精度承诺。上述统计是建议的评测协议，不是接口已返回的指标。

### 6.4 前次文档核验记录

原使用说明中的 Python 3.12 为原验证环境记录，予以保留。以下保留前次核验记录，不代表每次文档检查都会重跑：当时使用仓库已有 `.venv`：Windows、Python 3.13.14、NumPy 2.5.4、SciPy 1.18.1、OpenCV 5.0.0；直接执行 `python` 命中了未安装解释器的系统别名，改用虚拟环境解释器后执行：

```powershell
.\.venv\Scripts\python.exe -B -W error -m unittest tests.test_pose_ncc
.\.venv\Scripts\python.exe -B -m tests.test_smoke
```

当时记录的结果为单测 37 项通过、冒烟 8/8 通过，未重新运行完整基准，未新增基准性能数据，也未改写 [实验记录](experiments.md) 的历史结果。OpenCV 4.x 和 SciPy 在线参考是公式／API 资料，不表示当时安装版本与文档发布版本完全一致。
