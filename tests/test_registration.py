"""Tests for ImageRegistrator."""

import numpy as np
import pytest


def _make_image(h: int = 128, w: int = 128, channels: int = 3) -> np.ndarray:
    if channels == 1:
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return np.random.randint(0, 256, (h, w, channels), dtype=np.uint8)


# Four corner correspondences on a 128×128 image (identity-like mapping)
_SRC_POINTS = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])
_DST_POINTS = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])


def test_compute_homography() -> None:
    """compute_homography must return a (3, 3) float array."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    H = reg.compute_homography(_SRC_POINTS, _DST_POINTS)

    assert H.shape == (3, 3), f"Unexpected homography shape: {H.shape}"
    assert H.dtype in (np.float32, np.float64), f"Unexpected dtype: {H.dtype}"


def test_compute_homography_too_few_points() -> None:
    """compute_homography must raise ValueError for fewer than 4 points."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    with pytest.raises(ValueError):
        reg.compute_homography(_SRC_POINTS[:3], _DST_POINTS[:3])


def test_warp_image() -> None:
    """Warping with the identity-like H must return an image of the correct size."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    H = reg.compute_homography(_SRC_POINTS, _DST_POINTS)
    img = _make_image(128, 128, 3)

    warped = reg.warp_image(img, H, output_size=(128, 128))

    assert warped.shape == img.shape, f"Warped shape {warped.shape} != original {img.shape}"


def test_warp_image_custom_output_size() -> None:
    """warp_image must respect an explicitly specified output_size."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    H = np.eye(3, dtype=np.float64)
    img = _make_image(128, 128, 3)

    warped = reg.warp_image(img, H, output_size=(64, 64))

    assert warped.shape == (64, 64, 3)


def test_register_with_points() -> None:
    """register() with explicit point pairs must return (image, H) tuple."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    rgb = _make_image(128, 128, 3)
    ir = _make_image(128, 128, 1)

    registered_ir, H = reg.register(rgb, ir, src_points=_SRC_POINTS, dst_points=_DST_POINTS)

    assert isinstance(registered_ir, np.ndarray), "registered_ir must be a numpy array"
    assert H is not None, "H should not be None when explicit points are provided"
    assert H.shape == (3, 3)
    assert registered_ir.shape[:2] == rgb.shape[:2], (
        "Registered IR height/width must match RGB"
    )


def test_register_returns_tuple() -> None:
    """register() must always return a 2-tuple."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    rgb = _make_image(64, 64, 3)
    ir = _make_image(64, 64)
    result = reg.register(rgb, ir, src_points=_SRC_POINTS[:4], dst_points=_DST_POINTS[:4])

    assert len(result) == 2, "register() must return a 2-tuple"


def test_feature_based_registration() -> None:
    """feature_based_registration on identical images must return a valid tuple."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    img = _make_image(128, 128, 3)

    registered_ir, H = reg.feature_based_registration(img, img.copy())

    assert isinstance(registered_ir, np.ndarray)
    # H can be None if ORB finds no matches, but the image is always returned
    if H is not None:
        assert H.shape == (3, 3)


def test_feature_based_registration_grayscale_ir() -> None:
    """feature_based_registration must handle a 2-D grayscale IR image."""
    from src.fusion.registration import ImageRegistrator

    reg = ImageRegistrator()
    rgb = _make_image(64, 64, 3)
    ir = _make_image(64, 64, 1)

    result_img, H = reg.feature_based_registration(rgb, ir)
    assert isinstance(result_img, np.ndarray)
