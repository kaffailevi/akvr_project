"""Image registration utilities for aligning RGB and IR/thermal images."""

from typing import Optional, Tuple

import cv2
import numpy as np


class ImageRegistrator:
    """Aligns IR images to RGB images using homography-based registration.

    Supports both:
    * Manual correspondence-based registration (explicit point pairs).
    * Automatic feature-based registration via ORB + BFMatcher.
    """

    # Minimum number of good matches required for feature-based registration
    _MIN_MATCH_COUNT: int = 4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> np.ndarray:
        """Compute a 3×3 homography matrix from corresponding point pairs.

        Args:
            src_points: Source points, shape (N, 2) float32.
            dst_points: Destination points, shape (N, 2) float32.

        Returns:
            3×3 homography matrix as float64 numpy array.

        Raises:
            ValueError: If fewer than 4 point pairs are provided or if
                        homography computation fails.
        """
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        if len(src) < 4 or len(dst) < 4:
            raise ValueError(
                f"At least 4 point correspondences required, got {len(src)}."
            )

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("cv2.findHomography failed to compute a valid homography.")

        return H.astype(np.float64)

    def warp_image(
        self,
        image: np.ndarray,
        H: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp *image* using the homography matrix *H*.

        Args:
            image: Input image as a numpy array.
            H: 3×3 homography matrix.
            output_size: ``(width, height)`` of the output.  If ``None``,
                         the input size is used.

        Returns:
            Warped image as a numpy array.
        """
        if output_size is None:
            h, w = image.shape[:2]
            output_size = (w, h)
        return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)

    def register(
        self,
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Align *ir_img* to *rgb_img*.

        When ``src_points`` and ``dst_points`` are both provided the homography
        is computed from those correspondences.  Otherwise automatic ORB-based
        feature matching is attempted.

        Args:
            rgb_img: Reference RGB image (H, W, 3).
            ir_img: IR image to warp (H, W) or (H, W, 1).
            src_points: Source (IR) point correspondences, shape (N, 2).
            dst_points: Destination (RGB) point correspondences, shape (N, 2).

        Returns:
            ``(registered_ir, H)`` – the warped IR image and the homography
            matrix used (or ``None`` when registration was not possible).
        """
        output_size = (rgb_img.shape[1], rgb_img.shape[0])

        if src_points is not None and dst_points is not None:
            H = self.compute_homography(src_points, dst_points)
            registered_ir = self.warp_image(ir_img, H, output_size)
            return registered_ir, H

        return self.feature_based_registration(rgb_img, ir_img)

    def feature_based_registration(
        self,
        rgb_img: np.ndarray,
        ir_img: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Attempt automatic registration using ORB keypoints and BFMatcher.

        Args:
            rgb_img: Reference RGB image.
            ir_img: IR image to warp.

        Returns:
            ``(registered_ir, H)`` where *H* may be ``None`` if not enough
            matching keypoints were found (the original *ir_img* is returned).
        """
        gray_rgb = self._to_gray(rgb_img)
        gray_ir = self._to_gray(ir_img)

        orb = cv2.ORB_create(nfeatures=500)
        kp_rgb, des_rgb = orb.detectAndCompute(gray_rgb, None)
        kp_ir, des_ir = orb.detectAndCompute(gray_ir, None)

        if (
            des_rgb is None
            or des_ir is None
            or len(kp_rgb) < self._MIN_MATCH_COUNT
            or len(kp_ir) < self._MIN_MATCH_COUNT
        ):
            return ir_img.copy(), None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ir, des_rgb)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < self._MIN_MATCH_COUNT:
            return ir_img.copy(), None

        src_pts = np.float32(
            [kp_ir[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_rgb[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return ir_img.copy(), None

        output_size = (rgb_img.shape[1], rgb_img.shape[0])
        registered_ir = self.warp_image(ir_img, H, output_size)
        return registered_ir, H.astype(np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert any supported image array to uint8 grayscale."""
        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray
