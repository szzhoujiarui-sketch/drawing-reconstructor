import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

MAX_CANVAS_AREA = 100_000_000
MAX_BLEND_AREA = 100_000_000


class Blender:
    @staticmethod
    def feather_blend(warped_images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
        if not warped_images:
            raise ValueError("No images to blend")

        h = max(im.shape[0] for im in warped_images)
        w = max(im.shape[1] for im in warped_images)
        if h <= 0 or w <= 0:
            raise ValueError(f"Blend dimensions must be positive, got {h}x{w}")
        if not math.isfinite(w) or not math.isfinite(h):
            raise ValueError(f"Blend dimensions must be finite, got {h}x{w}")
        if h * w > MAX_BLEND_AREA:
            raise ValueError(
                f"Blend area {h * w} exceeds limit {MAX_BLEND_AREA} ({h}x{w})"
            )

        acc = np.zeros((h, w, 3), dtype=np.float64)
        wacc = np.zeros((h, w), dtype=np.float64)

        for im, mask in zip(warped_images, masks):
            mask_f = mask.astype(np.float64) / 255.0
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            dist = np.clip(dist, 0, None)
            weight = cv2.GaussianBlur(dist, (21, 21), 0)
            max_w = weight.max()
            if np.isfinite(max_w) and max_w > 0:
                weight = weight / max_w
            else:
                weight = mask_f
            weight = np.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0)
            weight = weight * mask_f

            im_f = im.astype(np.float64)
            for c in range(3):
                acc[:, :, c] += im_f[:, :, c] * weight
            wacc += weight

        wacc = np.maximum(wacc, 1e-8)
        result = (acc / wacc[:, :, np.newaxis]).astype(np.uint8)
        return result

    @staticmethod
    def compute_canvas(
        tiles: List[np.ndarray],
        homographies: List[np.ndarray],
        max_canvas_area: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int]:
        corners = []
        for im, H in zip(tiles, homographies):
            h, w = im.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            corners.append(dst)
        all_pts = np.vstack(corners).reshape(-1, 2)
        if not np.all(np.isfinite(all_pts)):
            raise ValueError("Transformed corner coordinates contain non-finite values")

        min_x_f, min_y_f = np.floor(all_pts.min(axis=0))
        max_x_f, max_y_f = np.ceil(all_pts.max(axis=0))
        width_f = max_x_f - min_x_f
        height_f = max_y_f - min_y_f

        limit = max_canvas_area if max_canvas_area is not None else MAX_CANVAS_AREA
        if limit <= 0:
            raise ValueError(f"Canvas area limit must be positive, got {limit}")
        if not math.isfinite(width_f) or not math.isfinite(height_f):
            raise ValueError(f"Canvas dimensions must be finite, got {width_f}x{height_f}")
        if width_f <= 0 or height_f <= 0:
            raise ValueError(f"Canvas dimensions must be positive, got {width_f}x{height_f}")
        if width_f > limit / height_f:
            raise ValueError(
                f"Canvas area exceeds limit {limit} "
                f"({width_f}x{height_f})"
            )

        min_x, min_y = int(min_x_f), int(min_y_f)
        width, height = int(width_f), int(height_f)
        offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
        return offset, width, height
