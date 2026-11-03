"""统一可视化工具：中文字体、配色、图片输出。

设计约定
--------
* **输出目录** 统一为 ``data/output/``（已在 .gitignore 中忽略，属实验产物）。
* **背景** 用白底 —— 图片是给报告/文档引用的产物，浅底打印和嵌入都更稳妥，
  不跟随 IDE 深色主题。
* **配色语义固定**，所有图保持一致，看图即可判断哪条是真值：
  - 真值（ground truth） → 青绿 ``#0F9D58``，虚线
  - 检测结果（detection） → 橙红 ``#E8543F``，实线
  - 辅助 / 次要        → 蓝 ``#4A90D9``
  - 文本               → 近黑 ``#1A1A1A``

用法
----
    from traditional.viz import setup, save, GT, DET

    setup()                      # 幂等，可重复调用
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ...
    save(fig, "ncc_matching")    # → data/output/ncc_matching.png

命令行加 ``--show`` 可弹窗交互查看（默认只存盘，便于 CI 无人值守）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# 必须在 import pyplot 之前选定后端；有 --show 时才用交互后端。
if "--show" in sys.argv:
    try:
        matplotlib.use("TkAgg")
    except Exception:  # 无 GUI 环境时静默回退
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 配色（语义固定，勿随意改）────────────────────────────────────
GT = "#0F9D58"       # 真值：青绿
DET = "#E8543F"      # 检测：橙红
AUX = "#4A90D9"      # 辅助：蓝
MUTED = "#8A8F98"    # 次要：灰
INK = "#1A1A1A"      # 文本：近黑

# 输出目录：项目根 / data / output
_OUT = Path(__file__).resolve().parent.parent / "data" / "output"

_READY = False


def setup() -> None:
    """设置中文字体与全局样式。幂等。"""
    global _READY
    if _READY:
        return

    plt.rcParams.update(
        {
            # 中文：雅黑优先，缺失则依次回退
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "DengXian",
                "DejaVu Sans",
            ],
            # 中文字体下负号会渲染成方块，必须关掉 unicode 负号
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#B8BDC4",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#DDE1E6",
            "grid.linewidth": 0.7,
            "legend.fontsize": 8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#DDE1E6",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
        }
    )
    _READY = True


def out_dir() -> Path:
    """返回输出目录，不存在则创建。"""
    _OUT.mkdir(parents=True, exist_ok=True)
    return _OUT


def save(fig, name: str, dpi: int = 140) -> Path:
    """保存图片到 data/output/<name>.png。

    命令行带 ``--show`` 时先弹窗（阻塞），关窗后再存盘。
    返回图片路径。
    """
    setup()
    path = out_dir() / f"{name}.png"
    if "--show" in sys.argv:
        plt.show()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return path


def draw_box(ax, x: float, y: float, w: float, h: float, *, color: str = DET,
             ls: str = "-", lw: float = 1.6, label: str | None = None) -> None:
    """在图像坐标（y 向下）上画矩形框，左上角为 (x, y)。"""
    ax.add_patch(
        Rectangle((x, y), w, h, fill=False, edgecolor=color, linestyle=ls,
                  linewidth=lw, label=label, zorder=5)
    )


def draw_cross(ax, x: float, y: float, *, color: str = DET, size: float = 6.0,
               lw: float = 1.4, label: str | None = None) -> None:
    """画十字标记，用于标出亚像素点位。"""
    ax.plot([x - size, x + size], [y, y], color=color, lw=lw, zorder=6,
            solid_capstyle="round")
    ax.plot([x, x], [y - size, y + size], color=color, lw=lw, zorder=6,
            solid_capstyle="round", label=label)


def image_ax(ax, img, title: str, *, vmin=None, vmax=None, cmap: str = "gray") -> None:
    """以正确的图像坐标系显示灰度图（y 向下，等比例）。"""
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def metric_text(ax, lines: list[str], *, loc: str = "upper left", size: float = 8.0) -> None:
    """在坐标轴内贴一块半透明指标文本框。

    loc 支持 upper/lower 与 left/center/right 的组合，例如 "lower right"。
    """
    text = "\n".join(lines)
    va = "top" if loc.startswith("upper") else "bottom"
    if loc.endswith("left"):
        ha, x = "left", 0.03
    elif loc.endswith("right"):
        ha, x = "right", 0.97
    else:
        ha, x = "center", 0.5
    y = 0.97 if va == "top" else 0.03
    ax.text(
        x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=size,
        color=INK, zorder=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#DDE1E6", alpha=0.88),
    )
