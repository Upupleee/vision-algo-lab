"""CNN 自动对焦：从单帧图像直接回归离焦量（defocus distance）。

相比传统"评分曲线 + 搜索"，CNN 省去多帧扫描，但需要离焦量多样的训练数据。

骨架 TODO：
1. 数据集：make_focus_stack 生成 (frame, defocus=z-z_focus) 对，真实场景用
   变焦扫拍序列；
2. 训练：MSELoss 回归离焦量；推理时 1-2 帧 Non-Linear 搜索即可收敛；
3. 对比实验：与 traditional/autofocus 的曲线法比精度与帧数。

依赖: pip install -r requirements-dl.txt
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("需要 PyTorch：pip install -r requirements-dl.txt") from e


class FocusNet(nn.Module):
    """小型 CNN，输入灰度帧，输出标量离焦量（像素/米，按数据集定义）。"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(-1)


def train_loop(model: nn.Module, loader, epochs: int = 20, lr: float = 1e-3) -> None:
    """TODO: 接入 data/synthetic.make_focus_stack 组装 Dataset 后启用。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for frame, defocus in loader:
            pred = model(frame)
            loss = F.mse_loss(pred, defocus)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    model = FocusNet()
    dummy = torch.randn(2, 1, 96, 96)
    print("输出形状:", model(dummy).shape)  # (2,)
