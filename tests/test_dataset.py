"""Tests for SolarPanelDataset."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_image(path: Path, h: int = 64, w: int = 64, mode: str = "RGB") -> None:
    """Save a random image to *path*."""
    if mode == "RGB":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))


def _create_dataset_structure(
    tmp_path: Path,
    n: int = 3,
    with_labels: bool = False,
) -> Path:
    """Create a minimal dataset directory under *tmp_path*.

    Returns the root directory.
    """
    root = tmp_path / "dataset"
    (root / "rgb").mkdir(parents=True)
    (root / "ir").mkdir(parents=True)
    if with_labels:
        (root / "labels").mkdir()

    for i in range(n):
        _save_image(root / "rgb" / f"img_{i:03d}.jpg")
        _save_image(root / "ir" / f"img_{i:03d}.jpg", mode="L")
        if with_labels:
            label_file = root / "labels" / f"img_{i:03d}.txt"
            label_file.write_text("0 0.5 0.5 0.3 0.3\n1 0.2 0.7 0.1 0.1\n")

    return root


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_dataset_loading(tmp_path: Path) -> None:
    """SolarPanelDataset must find all N paired images."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=3)
    ds = SolarPanelDataset(root_dir=str(root))

    assert len(ds) == 3


def test_dataset_returns_dict(tmp_path: Path) -> None:
    """__getitem__ must return a dict with the required keys."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=2)
    ds = SolarPanelDataset(root_dir=str(root))
    item = ds[0]

    for key in ("rgb", "ir", "rgb_path", "ir_path"):
        assert key in item, f"Missing key: {key}"


def test_dataset_rgb_shape(tmp_path: Path) -> None:
    """RGB tensor must have shape (3, H, W)."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=1)
    ds = SolarPanelDataset(root_dir=str(root))
    item = ds[0]

    assert item["rgb"].shape[0] == 3, "RGB tensor must have 3 channels"
    assert item["rgb"].ndim == 3


def test_dataset_ir_shape(tmp_path: Path) -> None:
    """IR tensor must have shape (1, H, W)."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=1)
    ds = SolarPanelDataset(root_dir=str(root))
    item = ds[0]

    assert item["ir"].shape[0] == 1, "IR tensor must have 1 channel"
    assert item["ir"].ndim == 3


def test_dataset_paths_are_strings(tmp_path: Path) -> None:
    """rgb_path and ir_path must be strings."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=1)
    ds = SolarPanelDataset(root_dir=str(root))
    item = ds[0]

    assert isinstance(item["rgb_path"], str)
    assert isinstance(item["ir_path"], str)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_dataset_with_labels(tmp_path: Path) -> None:
    """Dataset must load YOLO-format labels when present."""
    from src.data.dataset import SolarPanelDataset
    import torch

    root = _create_dataset_structure(tmp_path, n=3, with_labels=True)
    ds = SolarPanelDataset(root_dir=str(root), load_labels=True)
    item = ds[0]

    assert item["label"] is not None, "Label should be loaded"
    assert isinstance(item["label"], torch.Tensor)
    assert item["label"].shape[1] == 5, "Each label row must have 5 values (cls, cx, cy, w, h)"


def test_dataset_missing_label(tmp_path: Path) -> None:
    """Dataset must return None for label when label file is missing."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=2, with_labels=False)
    ds = SolarPanelDataset(root_dir=str(root), load_labels=True)
    item = ds[0]

    assert item["label"] is None, "Label should be None when no label file exists"


def test_dataset_load_labels_false(tmp_path: Path) -> None:
    """When load_labels=False, label must always be None."""
    from src.data.dataset import SolarPanelDataset

    root = _create_dataset_structure(tmp_path, n=2, with_labels=True)
    ds = SolarPanelDataset(root_dir=str(root), load_labels=False)
    item = ds[0]

    assert item["label"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_dataset_png_images(tmp_path: Path) -> None:
    """Dataset must support .png images."""
    from src.data.dataset import SolarPanelDataset

    root = tmp_path / "dataset"
    (root / "rgb").mkdir(parents=True)
    (root / "ir").mkdir(parents=True)

    _save_image(root / "rgb" / "panel_001.png")
    _save_image(root / "ir" / "panel_001.png", mode="L")

    ds = SolarPanelDataset(root_dir=str(root))
    assert len(ds) == 1
    item = ds[0]
    assert item["rgb"].shape[0] == 3


def test_dataset_unpaired_rgb_skipped(tmp_path: Path) -> None:
    """RGB images without a matching IR file should be silently skipped."""
    from src.data.dataset import SolarPanelDataset

    root = tmp_path / "dataset"
    (root / "rgb").mkdir(parents=True)
    (root / "ir").mkdir(parents=True)

    # Create 2 paired images + 1 RGB-only image
    for i in range(2):
        _save_image(root / "rgb" / f"img_{i:03d}.jpg")
        _save_image(root / "ir" / f"img_{i:03d}.jpg", mode="L")

    _save_image(root / "rgb" / "orphan.jpg")  # no matching IR

    ds = SolarPanelDataset(root_dir=str(root))
    assert len(ds) == 2
