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
