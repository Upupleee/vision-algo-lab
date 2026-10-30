"""U-Net 亚像素边缘回归网络。

输出两通道：
- edge heat：边缘概率图（BCE 监督）；
- offset：每像素到边缘的亚像素偏移 (dx, dy)（L1 监督，仅边缘处有效）。

标签生成：由 data/synthetic.make_edge_scene 的解析真值线，对每个像素计算
带符号距离 → 距离 < 1px 的像素为 edge=1，offset = 垂足 - 像素中心。
该范式同样适用于圆/孔（改真值为圆方程）。

依赖: pip install -r requirements-dl.txt
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("需要 PyTorch：pip install -r requirements-dl.txt") from e


class EdgeNet(nn.Module):
    """轻量 U-Net：输入灰度图，输出 (2, H, W)：0 通道 heat，1-2 偏移。"""

    def __init__(self, in_channels: int = 1, base: int = 16):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.d1 = blk(in_channels, base)
        self.d2 = blk(base, base * 2)
        self.d3 = blk(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        y = self.u2(x3) + x2
        y = self.u1(y) + x1
        return self.out(y)


def loss_fn(pred: torch.Tensor, heat: torch.Tensor, offset: torch.Tensor,
            w_heat: float = 1.0, w_off: float = 10.0) -> torch.Tensor:
    """heat 用 BCEWithLogits，offset 用仅在边缘处生效的 L1。"""
    bce = nn.functional.binary_cross_entropy_with_logits(pred[:, 0], heat)
    m = heat > 0.5
    if m.any():
        l1 = nn.functional.l1_loss(pred[:, 1:][m], offset[m])
    else:
        l1 = torch.zeros((), device=pred.device)
    return w_heat * bce + w_off * l1


if __name__ == "__main__":
    net = EdgeNet()
    dummy = torch.randn(1, 1, 128, 128)
    out = net(dummy)
    print("输出形状:", out.shape)  # (1, 3, 128, 128)
