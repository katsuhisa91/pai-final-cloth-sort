"""番号 1 文字を分類する小さな CNN(約 0.2M パラメータ、CPU で学習・推論できる)。"""

from __future__ import annotations

import torch
from torch import nn

from cloth_sort.sim.synth import CHAR_CLASSES, CHAR_SIZE


class CharNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, len(CHAR_CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (CHAR_SIZE, CHAR_SIZE)
        return self.head(self.features(x))
