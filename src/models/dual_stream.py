"""Dual-stream RGB+IR classification model."""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.fusion.fusion import MidLevelFusion
from src.models.detector import SolarPanelDetector


class DualStreamModel(nn.Module):
    """Dual-stream model that processes RGB and IR inputs in parallel.

    Internally uses :class:`~src.fusion.fusion.MidLevelFusion` for feature
    extraction and fusion.

    Args:
        num_classes: Number of fault classes to predict.
    """

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.fusion = MidLevelFusion(
            num_classes=num_classes,
            rgb_in_channels=3,
            ir_in_channels=1,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Compute class logits from paired RGB and IR tensors.

        Args:
            rgb: Float tensor of shape ``(B, 3, H, W)``.
            ir: Float tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.fusion(rgb, ir)

    # ------------------------------------------------------------------
    # High-level predict interface
    # ------------------------------------------------------------------

    def predict(
        self,
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> Dict:
        """Classify a single paired RGB + IR image.

        Args:
            rgb_img: RGB numpy array ``(H, W, 3)`` uint8 or float.
            ir_img: IR numpy array ``(H, W)`` or ``(H, W, 1)`` uint8 or float.
            transform: Optional callable applied to tensors before inference.

        Returns:
            Dict with keys ``class_id``, ``class_name``, ``confidence``,
            ``logits``.
        """
        self.eval()
        rgb_t, ir_t = self._to_tensors(rgb_img, ir_img)

        if transform is not None:
            rgb_t = transform(rgb_t)
            ir_t = transform(ir_t)

        # Add batch dimension
        rgb_b = rgb_t.unsqueeze(0)
        ir_b = ir_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(rgb_b, ir_b)  # (1, num_classes)

        probs = torch.softmax(logits, dim=1)[0]
        class_id = int(probs.argmax().item())
        confidence = float(probs[class_id].item())
        class_name = SolarPanelDetector.FAULT_CLASSES.get(class_id, f"class_{class_id}")

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "logits": logits.squeeze(0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights to *path*.

        Args:
            path: File path for the checkpoint (e.g. ``model.pt``).
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        num_classes: int = 5,
    ) -> "DualStreamModel":
        """Load a saved model from *path*.

        Args:
            path: File path to the saved checkpoint.
            device: Target device.  If ``None`` the device is inferred
                    automatically.
            num_classes: Must match the number of classes used during training.

        Returns:
            Loaded :class:`DualStreamModel` instance in eval mode.
        """
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(num_classes=num_classes)
        state = torch.load(path, map_location=target_device)
        model.load_state_dict(state)
        model.to(target_device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensors(
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
    ) -> tuple:
        """Convert numpy images to float32 tensors (C, H, W) in [0, 1]."""
        rgb = rgb_img.astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        ir = ir_img.astype(np.float32)
        if ir.max() > 1.0:
            ir /= 255.0

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)

        if ir.ndim == 2:
            ir_t = torch.from_numpy(ir[np.newaxis])  # (1, H, W)
        else:
            ir_t = torch.from_numpy(ir.transpose(2, 0, 1))

        return rgb_t, ir_t
