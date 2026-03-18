"""Multimodal fusion strategies for RGB and IR image streams."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class EarlyFusion:
    """Pixel-level early fusion by channel concatenation.

    Concatenates an RGB image (3 channels) with an IR image (1 channel)
    to produce a 4-channel representation suitable for downstream models.
    """

    def __call__(
        self,
        rgb_tensor: torch.Tensor,
        ir_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate *rgb_tensor* and *ir_tensor* along the channel dimension.

        Args:
            rgb_tensor: Float tensor of shape ``(3, H, W)``.
            ir_tensor: Float tensor of shape ``(1, H, W)``.

        Returns:
            Float tensor of shape ``(4, H, W)``.
        """
        if ir_tensor.shape[0] != 1:
            raise ValueError(
                f"IR tensor must have exactly 1 channel, got {ir_tensor.shape[0]}."
            )
        if rgb_tensor.shape[0] != 3:
            raise ValueError(
                f"RGB tensor must have exactly 3 channels, got {rgb_tensor.shape[0]}."
            )
        return torch.cat([rgb_tensor, ir_tensor], dim=0)

    @staticmethod
    def to_tensor(
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy image arrays to normalised float32 tensors.

        Args:
            rgb_img: RGB numpy array of shape ``(H, W, 3)`` uint8 or float.
            ir_img: Grayscale IR numpy array of shape ``(H, W)`` or ``(H, W, 1)``.

        Returns:
            ``(rgb_tensor, ir_tensor)`` where rgb has shape ``(3, H, W)`` and
            ir has shape ``(1, H, W)``, both float32 in ``[0, 1]``.
        """
        rgb = rgb_img.astype(np.float32) / 255.0 if rgb_img.dtype == np.uint8 else rgb_img.astype(np.float32)
        ir = ir_img.astype(np.float32) / 255.0 if ir_img.dtype == np.uint8 else ir_img.astype(np.float32)

        # HWC → CHW
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)

        if ir.ndim == 2:
            ir_t = torch.from_numpy(ir[np.newaxis, :, :])  # (1, H, W)
        else:
            ir_t = torch.from_numpy(ir.transpose(2, 0, 1))  # (1, H, W)

        return rgb_t, ir_t


class MidLevelFusion(nn.Module):
    """Dual-stream CNN mid-level feature fusion.

    Two independent convolutional streams process RGB and IR inputs
    separately; their feature maps are concatenated and passed through a
    classification head.

    Architecture per stream::

        Conv2d(in_ch → 64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
        → Conv2d(64 → 128, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(1)

    The 128-d features from each stream are concatenated (256-d total) and
    fed to a fully-connected classification head.

    Args:
        num_classes: Number of output classes.
        rgb_in_channels: Number of input channels for the RGB stream (default 3).
        ir_in_channels: Number of input channels for the IR stream (default 1).
    """

    def __init__(
        self,
        num_classes: int = 5,
        rgb_in_channels: int = 3,
        ir_in_channels: int = 1,
    ) -> None:
        super().__init__()

        self.rgb_stream = self._build_stream(rgb_in_channels)
        self.ir_stream = self._build_stream(ir_in_channels)
        self.fusion_head = nn.Linear(256, num_classes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Run both streams and return class logits.

        Args:
            rgb: Float tensor of shape ``(B, 3, H, W)``.
            ir: Float tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        rgb_feat = self.rgb_stream(rgb).flatten(1)  # (B, 128)
        ir_feat = self.ir_stream(ir).flatten(1)     # (B, 128)
        fused = torch.cat([rgb_feat, ir_feat], dim=1)  # (B, 256)
        return self.fusion_head(fused)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream(in_channels: int) -> nn.Sequential:
        """Build a single convolutional feature-extraction stream."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
