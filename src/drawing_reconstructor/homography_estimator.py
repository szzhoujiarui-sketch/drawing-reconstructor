from typing import Tuple

import cv2
import numpy as np


class HomographyEstimator:
    @staticmethod
    def estimate(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_thresh: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(src_pts) < 4:
            raise ValueError(f"Need at least 4 point correspondences, got {len(src_pts)}")
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or H.shape != (3, 3):
            raise RuntimeError("Homography estimation failed")
        if not np.all(np.isfinite(H)):
            raise RuntimeError("Homography matrix contains non-finite values")
        return H, mask

    @staticmethod
    def warp(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(image, H, output_shape, flags=cv2.INTER_LINEAR)
