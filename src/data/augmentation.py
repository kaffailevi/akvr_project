"""Augmentation pipeline for paired RGB + IR solar panel images."""

import random
from typing import Optional, Tuple

import cv2
import numpy as np


class SolarAugmentation:
    """Configurable augmentation pipeline for paired RGB and IR images.

    Geometric transforms (horizontal flip, rotation) are applied **identically**
    to both RGB and IR images to maintain spatial correspondence.  IR-specific
    enhancements (CLAHE, Gaussian blur) are applied only to the IR image.

    Args:
        img_size: Target ``(width, height)`` after resizing.
        apply_clahe: Whether to apply CLAHE to the IR image.
        gaussian_blur_p: Probability of applying Gaussian blur to the IR image.
        flip_p: Probability of applying a random horizontal flip.
        rotate_degrees: Maximum rotation angle (degrees) for random rotation.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        apply_clahe: bool = True,
        gaussian_blur_p: float = 0.3,
        flip_p: float = 0.5,
        rotate_degrees: float = 15.0,
    ) -> None:
        self.img_size = img_size
        self.apply_clahe = apply_clahe
        self.gaussian_blur_p = gaussian_blur_p
        self.flip_p = flip_p
        self.rotate_degrees = rotate_degrees

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self, rgb_img: np.ndarray, ir_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations to a paired RGB + IR image.

        Geometric transforms are applied identically to both images; IR
        enhancements are applied only to *ir_img*.

        Args:
            rgb_img: RGB image as a numpy array (H, W, 3) uint8.
            ir_img: IR/thermal image as a numpy array (H, W) or (H, W, 1) uint8.

        Returns:
            Tuple ``(augmented_rgb, augmented_ir)`` as numpy arrays.
        """
        rgb = self._ensure_hwc(rgb_img)
        ir = self._ensure_grayscale(ir_img)

        # --- Resize both to target size ---
        rgb = cv2.resize(rgb, self.img_size)
        ir = cv2.resize(ir, self.img_size)

        # --- Shared geometric transforms ---
        do_flip = random.random() < self.flip_p
        angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)

        rgb = self._apply_geometric(rgb, do_flip=do_flip, angle=angle)
        ir = self._apply_geometric(ir, do_flip=do_flip, angle=angle)

        # --- IR-specific enhancements ---
        if self.apply_clahe:
            ir = self.apply_clahe_to_ir(ir, clahe=self._clahe)

        if random.random() < self.gaussian_blur_p:
            ir = cv2.GaussianBlur(ir, (5, 5), 0)

        return rgb, ir

    @staticmethod
    def apply_clahe_to_ir(
        ir_img: np.ndarray,
        clahe: Optional[cv2.CLAHE] = None,
    ) -> np.ndarray:
        """Apply CLAHE to an IR (grayscale) image to enhance thermal contrast.

        Args:
            ir_img: Grayscale image as a numpy array (H, W) uint8.
            clahe: Optional pre-created CLAHE object; one is created if None.

        Returns:
            CLAHE-enhanced grayscale image of the same shape.
        """
        img = SolarAugmentation._ensure_grayscale(ir_img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if clahe is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_hwc(img: np.ndarray) -> np.ndarray:
        """Return *img* as an (H, W, C) array."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
        """Return *img* as an (H, W) grayscale array."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 1:
            return img[:, :, 0]
        return img

    @staticmethod
    def _apply_geometric(
        img: np.ndarray, *, do_flip: bool, angle: float
    ) -> np.ndarray:
        """Apply horizontal flip and rotation to *img*."""
        if do_flip:
            img = cv2.flip(img, 1)

        if angle != 0.0:
            h, w = img.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        return img
