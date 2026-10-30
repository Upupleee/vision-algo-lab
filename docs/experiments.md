# 实验记录

> 格式：日期 | 场景（`data/synthetic` 参数）| 方法 | 指标 | 结果 | 结论
> 所有实验可用 `python -m tests.test_smoke` 复现（固定随机种子）。

## 基线：传统算法在合成数据上的精度（2026-10-30）

| 方向 | 方法 | 场景 | 指标 | 结果 |
|------|------|------|------|------|
| 模板匹配 | NCC + 抛物线亚像素 | make_template_scene(noise=3.0) | 定位误差 | 0.20 px |
| 模板匹配 | 形状匹配（梯度方向） | make_template_scene(noise=2.0) | 定位误差（整像素） | 0.73 px |
| 卡尺测量 | erf 剖面拟合 + PCA 直线 | make_edge_scene(noise=1.0) | 边缘误差 / 直线 RMS | 0.010 / 0.011 px |
| 亚像素(1D) | 梯度抛物线插值 | make_edge_profile(sigma=1.2) | 定位误差 | 0.023 px |
| 亚像素(1D) | erf 剖面拟合 | 同上 | 定位误差 | ~0.000 px |
| 亚像素(2D) | Devernay 梯度抛物面 | make_edge_scene(noise=0.5) | 中位法向误差 | 0.013 px |
| 亚像素角点 | OpenCV cornerSubPix | make_corner_scene() | 平均误差（初检 0.40 px） | 0.080 px |
| 亚像素角点 | 手写梯度正交最小二乘 | 同上 | 平均误差 | 0.047 px |
| 自动对焦 | 7 种评分函数 | make_focus_stack() | 峰值帧偏差 / 单峰性 | 0 帧 / ≥0.93 |
| 自动对焦 | 三点高斯拟合 | 同上 | 对焦位置误差 | 0.019 |
| 自动对焦 | 爬山法 / 黄金分割 | 同上 | 误差（评测次数） | 0.20 (20) / 0.12 (21) |

## 待做对比实验

- [ ] NCC vs 形状匹配：旋转/尺度鲁棒性、耗时（`tests.metrics.bench_ms`）
- [ ] 卡尺法 vs Devernay：真实离焦图像上的稳定性
- [ ] 传统评分函数 vs ml/RF 回归 vs dl/FocusNet：对焦帧数与精度
- [ ] Zernike 矩（含 A31 模糊补偿）vs Devernay（见 research_notes.md §1）
