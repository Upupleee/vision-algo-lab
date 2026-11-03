"""真实数据集（RTImg/）对焦评估：哪几种清晰度函数在这个数据集上真的可靠。

与合成数据集的差别
------------------
合成数据有解析真值 z_focus；真实数据集只能取「对焦距离最接近目标距离的
那一帧」作为真值（详见 traditional/autofocus/source.py 的说明）。

评估口径
--------
* **命中**：评分曲线的峰值帧 == 真值帧。
* **误差**：峰值帧与真值帧的对焦距离之差，单位 mm。
* **单峰性**：monotonic_ratio，1.0 表示峰值两侧单调下降。
* **一致性**：同一函数在 12 个场景上的命中数。

用法
----
    python -m traditional.autofocus.real_dataset
    python -m traditional.autofocus.real_dataset --show
    python -m traditional.autofocus.real_dataset --roi 0.3,0.15,0.4,0.4
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from traditional.autofocus.focus_measures import ALL_MEASURES, monotonic_ratio
from traditional.autofocus.source import FolderSource

DATASET = "RTImg"
# 默认 ROI：中心 70%，覆盖主体、排除下方桌面的木纹干扰
DEFAULT_ROI = (0.15, 0.05, 0.70, 0.70)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="真实数据集对焦评估")
    ap.add_argument("--data", default=DATASET, help="数据集根目录")
    ap.add_argument("--roi", default="0.15,0.05,0.70,0.70",
                    help="ROI，比例(x,y,w,h) 或绝对像素；传 none 表示全图")
    ap.add_argument("--show", action="store_true", help="弹窗显示（默认只存盘）")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    roi = None
    if args.roi.lower() != "none":
        roi = tuple(float(v) for v in args.roi.split(","))
        if len(roi) != 4:
            print("ROI 需要 4 个分量", file=sys.stderr)
            return 2

    src = FolderSource(args.data, roi=roi)
    subjects = src.subject_ids()
    print(f"数据集：{args.data}   共 {len(src)} 帧 / {len(subjects)} 组")
    print(f"ROI：{'全图' if roi is None else roi}")
    print()

    # ── 逐组逐函数计算 ────────────────────────────────────────
    # 先按组把图像读一遍缓存下来，再喂给所有评价函数，避免重复解码
    curves: dict[tuple[int, str], np.ndarray] = {}
    zs_of: dict[int, np.ndarray] = {}
    gt_of: dict[int, float] = {}
    t_start = time.perf_counter()
    for sid in subjects:
        fr = src.groups[sid]
        zs = np.array([f.focus_dist for f in fr], dtype=float)
        imgs = [src.load(f) for f in fr]
        zs_of[sid] = zs
        gt_of[sid] = src.gt_z(sid)
        for name, fn in ALL_MEASURES.items():
            curves[(sid, name)] = np.array([fn(im) for im in imgs], dtype=float)
    elapsed = time.perf_counter() - t_start
    print(f"计算完成，用时 {elapsed:.1f}s（{len(src)} 帧 × {len(ALL_MEASURES)} 个函数）")
    print()

    # ── 汇总表 ────────────────────────────────────────────────
    names = list(ALL_MEASURES.keys())
    hits = {n: 0 for n in names}
    errs: dict[str, list[float]] = {n: [] for n in names}
    monos: dict[str, list[float]] = {n: [] for n in names}

    header = f"{'组':>4} {'真值mm':>7} " + " ".join(f"{n[:9]:>9}" for n in names)
    print(header)
    print("-" * len(header))
    for sid in subjects:
        zs, gt = zs_of[sid], gt_of[sid]
        cells = []
        for n in names:
            sc = curves[(sid, n)]
            pk = float(zs[int(np.argmax(sc))])
            ok = abs(pk - gt) < 1e-6
            hits[n] += int(ok)
            errs[n].append(abs(pk - gt))
            monos[n].append(float(monotonic_ratio(sc)))
            cells.append(f"{'OK' if ok else f'{abs(pk-gt):.0f}':>9}")
        print(f"{sid:>4} {gt:>7.0f} " + " ".join(cells))
    print("-" * len(header))
    print(f"{'命中':>4} {'':>7} " + " ".join(f"{hits[n]:>6}/{len(subjects):<2}" for n in names))
    print(f"{'平均误差':>2} {'':>7} " + " ".join(f"{np.mean(errs[n]):>9.1f}" for n in names))
    print(f"{'平均单峰':>2} {'':>7} " + " ".join(f"{np.mean(monos[n]):>9.2f}" for n in names))

    # ── 出图 ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    from traditional.viz import AUX, DET, GT, MUTED, metric_text, save, setup

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) 三个代表组的 Tenengrad 曲线
    demo_sets = [s for s in (1, 7, 9) if s in subjects][:3]
    colors = [DET, AUX, "#9C27B0"]
    for i, sid in enumerate(demo_sets):
        zs = zs_of[sid]
        sc = curves[(sid, "tenengrad")]
        span = sc.max() - sc.min()
        norm = (sc - sc.min()) / span if span > 1e-12 else np.zeros_like(sc)
        axes[0].plot(zs, norm, "o-", lw=1.8, ms=5, color=colors[i % 3],
                     label=f"组 {sid}（真值 {gt_of[sid]:.0f}mm）")
        k = int(np.argmax(sc))
        axes[0].plot(zs[k], norm[k], "*", ms=15, color=colors[i % 3],
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        axes[0].axvline(gt_of[sid], color=colors[i % 3], ls=":", lw=1.1, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_title("代表组的 Tenengrad 对焦曲线（星号 = 峰值）", pad=6)
    axes[0].set_xlabel("对焦距离（mm，对数刻度）")
    axes[0].set_ylabel("归一化评分")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylim(-0.06, 1.12)

    # (b) 各函数的命中数与平均误差
    idx = np.arange(len(names))
    hit_v = np.array([hits[n] for n in names])
    err_v = np.array([np.mean(errs[n]) for n in names])
    bars = axes[1].bar(idx, hit_v, 0.62, color=AUX, alpha=0.88)
    for b, v in zip(bars, hit_v):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v}",
                     ha="center", fontsize=8, fontweight="bold")
    axes[1].axhline(len(subjects), color=GT, ls="--", lw=1.2)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, rotation=32, ha="right", fontsize=7.5)
    axes[1].set_ylabel(f"命中组数（共 {len(subjects)} 组）")
    axes[1].set_ylim(0, len(subjects) + 2.2)
    axes[1].set_title("各评价函数在真实数据集上的命中率", pad=6)
    axes[1].grid(True, axis="y", alpha=0.4)
    metric_text(
        axes[1],
        [
            f"平均绝对误差（mm）",
            *[f"{n}  {err_v[i]:.1f}" for i, n in enumerate(names)],
        ],
        loc="lower right",
        size=6.2,
    )

    fig.tight_layout()
    print()
    print("图片 →", save(fig, "real_dataset_focus"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
