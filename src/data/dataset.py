"""PyTorch Dataset for paired RGB and IR solar panel images."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SolarPanelDataset(Dataset):
    """Dataset that loads paired RGB and IR images for solar panel fault detection.

    Expected directory structure::

        root/
            rgb/      # *.jpg or *.png RGB images
            ir/       # *.jpg or *.png IR/thermal images
            labels/   # *.txt YOLO-format annotation files (optional)

    Args:
        root_dir: Path to the root directory containing rgb/, ir/, and labels/ subdirs.
        transform: Optional transform applied to the RGB image tensor.
        ir_transform: Optional transform applied to the IR image tensor.
        load_labels: Whether to attempt loading YOLO-format label files.
    """

    _IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        ir_transform: Optional[Callable] = None,
        load_labels: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.ir_transform = ir_transform
        self.load_labels = load_labels

        self.rgb_dir = self.root_dir / "rgb"
        self.ir_dir = self.root_dir / "ir"
        self.labels_dir = self.root_dir / "labels"

        self.rgb_paths: List[Path] = sorted(self._collect_images(self.rgb_dir))
        self._validate_and_pair()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_images(self, directory: Path) -> List[Path]:
        """Return all image files under *directory* with supported extensions."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return [
            p
            for p in directory.iterdir()
            if p.suffix.lower() in self._IMAGE_EXTENSIONS
        ]

    def _ir_path_for(self, rgb_path: Path) -> Optional[Path]:
        """Return the corresponding IR path for a given RGB path, or None."""
        stem = rgb_path.stem
        for ext in self._IMAGE_EXTENSIONS:
            candidate = self.ir_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def _label_path_for(self, rgb_path: Path) -> Optional[Path]:
        """Return the corresponding label .txt path, or None if not found."""
        candidate = self.labels_dir / (rgb_path.stem + ".txt")
        if candidate.exists():
            return candidate
        return None

    def _validate_and_pair(self) -> None:
        """Build the final paired list, silently skipping unpaired RGB images."""
        paired: List[Path] = []
        for rgb_path in self.rgb_paths:
            if self._ir_path_for(rgb_path) is not None:
                paired.append(rgb_path)
        self.rgb_paths = paired

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, index: int) -> Dict:
        """Return a dict with rgb tensor, ir tensor, optional label, and file paths.

        Returns:
            dict with keys:
                - ``rgb``: FloatTensor of shape (3, H, W), values in [0, 1]
                - ``ir``: FloatTensor of shape (1, H, W), values in [0, 1]
                - ``label``: FloatTensor of YOLO annotations, or ``None``
                - ``rgb_path``: absolute path string to the RGB image
                - ``ir_path``: absolute path string to the IR image
        """
        rgb_path = self.rgb_paths[index]
        ir_path = self._ir_path_for(rgb_path)  # guaranteed not None after _validate

        # --- Load images ---
        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("L")  # single-channel grayscale

        # --- To tensors (C, H, W), float32 in [0, 1] ---
        rgb_tensor = self._pil_to_tensor(rgb_img, channels=3)
        ir_tensor = self._pil_to_tensor(ir_img, channels=1)

        # --- Apply optional user-supplied transforms ---
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_tensor)
        if self.ir_transform is not None:
            ir_tensor = self.ir_transform(ir_tensor)

        # --- Load YOLO label if requested ---
        label_tensor: Optional[torch.Tensor] = None
        if self.load_labels:
            label_path = self._label_path_for(rgb_path)
            if label_path is not None:
                label_tensor = self._load_yolo_label(label_path)

        return {
            "rgb": rgb_tensor,
            "ir": ir_tensor,
            "label": label_tensor,
            "rgb_path": str(rgb_path.resolve()),
            "ir_path": str(ir_path.resolve()),
        }

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_tensor(img: Image.Image, channels: int) -> torch.Tensor:
        """Convert a PIL image to a float32 tensor of shape (C, H, W)."""
        arr = np.array(img, dtype=np.float32) / 255.0
        if channels == 1:
            arr = arr[:, :, np.newaxis] if arr.ndim == 2 else arr
        # HWC → CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor

    @staticmethod
    def _load_yolo_label(label_path: Path) -> Optional[torch.Tensor]:
        """Parse a YOLO-format .txt file into a float tensor of shape (N, 5).

        Each row is: class_id  cx  cy  w  h  (all normalised to [0, 1]).
        Returns None for empty files.
        """
        rows: List[List[float]] = []
        with open(label_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                rows.append([float(v) for v in parts])
        if not rows:
            return None
        return torch.tensor(rows, dtype=torch.float32)
