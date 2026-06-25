"""
ResNet50-based binary classifier for document forgery detection.

Architecture (thesis Section 3.4.1 + Abdalla et al., 2024):
  - Feature extractor : ResNet50 pretrained on ImageNet (2048-d output)
  - Head              : GlobalAvgPool → Dense(256) → BN → ReLU → Dropout(0.4) → Dense(1) → Sigmoid
  - Input             : ELA image (224 × 224 × 3)
  - Output            : P(genuine) in [0, 1]  — 1.0 = genuine, 0.0 = forged

References:
  - ResNet50        : He et al. (2015), arXiv:1512.03385
  - Transfer learning for forgery : Abdalla, Iqbal & Shehata (2024) — thesis Table 2.1 entry 6
  - ELA + CNN       : Yang et al. (2022), Ji et al. (2022) — thesis Ch. 2
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required: pip install timm>=1.0.0") from exc


class ForgeryDetector(nn.Module):
    """
    Binary forgery classifier built on ResNet50.

    Output → scalar in [0, 1]:
      close to 1.0  → document likely Genuine
      close to 0.0  → document likely Forged
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embed_dim: int = 256,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # ResNet50 outputs 2048-d features when num_classes=0
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim: int = self.encoder.num_features  # 2048 for ResNet50

        self.head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image batch (B, 3, H, W) normalised to ImageNet stats.
        Returns:
            Tensor of shape (B,) with genuineness probabilities.
        """
        features = self.encoder(x)             # (B, 2048)
        return self.head(features).squeeze(1)  # (B,)
