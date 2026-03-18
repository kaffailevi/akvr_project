"""Tests for SolarAugmentation."""

import numpy as np
import pytest


def _make_rgb(h: int = 640, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_ir(h: int = 640, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def test_augmentation_output_shape() -> None:
    """Output shapes must match the configured img_size."""
    from src.data.augmentation import SolarAugmentation

    aug = SolarAugmentation(img_size=(640, 640))
    rgb = _make_rgb()
    ir = _make_ir()
    aug_rgb, aug_ir = aug(rgb, ir)

    assert aug_rgb.shape == (640, 640, 3), f"Unexpected RGB shape: {aug_rgb.shape}"
    assert aug_ir.shape == (640, 640), f"Unexpected IR shape: {aug_ir.shape}"


def test_augmentation_output_shape_non_square() -> None:
    """Non-square target size should be respected."""
    from src.data.augmentation import SolarAugmentation

    aug = SolarAugmentation(img_size=(320, 256))
    rgb = _make_rgb(480, 640)
    ir = _make_ir(480, 640)
    aug_rgb, aug_ir = aug(rgb, ir)

    # img_size is (width, height) per OpenCV convention
    assert aug_rgb.shape[:2] == (256, 320), f"Unexpected RGB shape: {aug_rgb.shape}"
    assert aug_ir.shape == (256, 320), f"Unexpected IR shape: {aug_ir.shape}"


def test_augmentation_both_transformed() -> None:
    """Both RGB and IR outputs must be numpy arrays."""
    from src.data.augmentation import SolarAugmentation

    aug = SolarAugmentation()
    aug_rgb, aug_ir = aug(_make_rgb(), _make_ir())

    assert isinstance(aug_rgb, np.ndarray), "RGB output must be a numpy array"
    assert isinstance(aug_ir, np.ndarray), "IR output must be a numpy array"


def test_clahe_applied() -> None:
    """apply_clahe_to_ir must return an array of the same shape."""
    from src.data.augmentation import SolarAugmentation

    ir = _make_ir(128, 128)
    enhanced = SolarAugmentation.apply_clahe_to_ir(ir)

    assert enhanced.shape == ir.shape, (
        f"CLAHE output shape {enhanced.shape} != input shape {ir.shape}"
    )
    assert enhanced.dtype == np.uint8


def test_clahe_applied_2d_input() -> None:
    """apply_clahe_to_ir must handle a plain 2-D grayscale array."""
    from src.data.augmentation import SolarAugmentation

    ir = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
    out = SolarAugmentation.apply_clahe_to_ir(ir)
    assert out.shape == (64, 64)


def test_augmentation_deterministic_flip() -> None:
    """When flip_p=1.0, both images must be horizontally flipped."""
    import cv2
    from src.data.augmentation import SolarAugmentation

    size = (64, 64)
    aug = SolarAugmentation(img_size=size, flip_p=1.0, rotate_degrees=0, apply_clahe=False, gaussian_blur_p=0.0)

    rgb = _make_rgb(64, 64)
    ir = _make_ir(64, 64)
    aug_rgb, aug_ir = aug(rgb, ir)

    # Images are resized (no-op here) then flipped — build expected the same way
    expected_rgb = cv2.flip(cv2.resize(rgb, size), 1)
    expected_ir = cv2.flip(cv2.resize(ir, size), 1)

    assert np.array_equal(aug_rgb, expected_rgb), "RGB was not flipped as expected"
    assert np.array_equal(aug_ir, expected_ir), "IR was not flipped as expected"


def test_augmentation_no_flip() -> None:
    """When flip_p=0 and no rotation, both images should be unchanged (modulo CLAHE)."""
    import cv2
    from src.data.augmentation import SolarAugmentation

    size = (64, 64)
    aug = SolarAugmentation(img_size=size, flip_p=0.0, rotate_degrees=0, apply_clahe=False, gaussian_blur_p=0.0)

    rgb = _make_rgb(64, 64)
    ir = _make_ir(64, 64)
    aug_rgb, aug_ir = aug(rgb, ir)

    expected_rgb = cv2.resize(rgb, size)
    expected_ir = cv2.resize(ir, size)
    assert np.array_equal(aug_rgb, expected_rgb), "RGB should be unchanged when no augmentation"
    assert np.array_equal(aug_ir, expected_ir), "IR should be unchanged when no augmentation"
