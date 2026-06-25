from typing import List, Optional, Tuple

import cv2
import numpy as np


class Blender:
    @staticmethod
    def feather_blend(warped_images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
        if not warped_images:
            raise ValueError("No images to blend")

        h = max(im.shape[0] for im in warped_images)
        w = max(im.shape[1] for im in warped_images)
        acc = np.zeros((h, w, 3), dtype=np.float64)
        wacc = np.zeros((h, w), dtype=np.float64)

        for im, mask in zip(warped_images, masks):
            mask_f = mask.astype(np.float64) / 255.0
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            dist = np.clip(dist, 0, None)
            weight = cv2.GaussianBlur(dist, (21, 21), 0)
            max_w = weight.max()
            if max_w > 0:
                weight = weight / max_w
            weight = weight * mask_f

            im_f = im.astype(np.float64)
            for c in range(3):
                acc[:, :, c] += im_f[:, :, c] * weight
            wacc += weight

        wacc = np.maximum(wacc, 1e-8)
        result = (acc / wacc[:, :, np.newaxis]).astype(np.uint8)
        return result

    @staticmethod
    def compute_canvas(tiles: List[np.ndarray], homographies: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
        corners = []
        for im, H in zip(tiles, homographies):
            h, w = im.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            corners.append(dst)
        all_pts = np.vstack(corners).reshape(-1, 2)
        min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
        return offset, width, height
