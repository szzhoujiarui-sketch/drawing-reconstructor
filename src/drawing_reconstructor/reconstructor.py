from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from drawing_reconstructor.blender import Blender
from drawing_reconstructor.feature_matcher import FeatureMatcher
from drawing_reconstructor.homography_estimator import HomographyEstimator
from drawing_reconstructor.tile_loader import TileLoader


class DrawingReconstructor:
    def __init__(self, detector: str = "sift", ratio_thresh: float = 0.75, ransac_thresh: float = 4.0):
        self.matcher = FeatureMatcher(detector, ratio_thresh)
        self.homography = HomographyEstimator()
        self.blender = Blender()
        self.ransac_thresh = ransac_thresh

    def reconstruct(
        self,
        tiles: List[np.ndarray],
        grid: Optional[Tuple[int, int]] = None,
        overlap_estimate: float = 0.12,
    ) -> np.ndarray:
        if grid is None:
            grid = TileLoader.infer_grid(tiles, (1, len(tiles)))
        rows, cols = grid
        if len(tiles) != rows * cols:
            raise ValueError(f"Expected {rows*cols} tiles for {rows}x{cols} grid, got {len(tiles)}")
        TileLoader.validate_tiles(tiles)

        ref_shape = tiles[0].shape[:2]
        ref_h, ref_w = ref_shape
        canvas_h = int(ref_h * rows * (1 - overlap_estimate) + ref_h * overlap_estimate)
        canvas_w = int(ref_w * cols * (1 - overlap_estimate) + ref_w * overlap_estimate)

        homographies = self._estimate_homographies(tiles, rows, cols)
        offset, canvas_w, canvas_h = Blender.compute_canvas(tiles, homographies)

        warped_images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for tile, H in zip(tiles, homographies):
            H_off = offset @ H
            warped = cv2.warpPerspective(tile, H_off, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.ones(tile.shape[:2], dtype=np.uint8) * 255, H_off, (canvas_w, canvas_h))
            warped_images.append(warped)
            masks.append(mask)

        result = self.blender.feather_blend(warped_images, masks)
        return result

    def _estimate_homographies(self, tiles: List[np.ndarray], rows: int, cols: int) -> List[np.ndarray]:
        center_idx = rows // 2 * cols + cols // 2

        pairwise: Dict[Tuple[int, int], np.ndarray] = {}
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    right_idx = r * cols + (c + 1)
                    H = self._pairwise_homography(tiles[idx], tiles[right_idx])
                    if H is not None:
                        pairwise[(idx, right_idx)] = H
                if r + 1 < rows:
                    down_idx = (r + 1) * cols + c
                    H = self._pairwise_homography(tiles[idx], tiles[down_idx])
                    if H is not None:
                        pairwise[(idx, down_idx)] = H

        homographies: List[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in tiles]
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx == center_idx:
                    continue
                path = self._manhattan_path(idx, center_idx, cols)
                H_total = np.eye(3, dtype=np.float64)
                for i in range(len(path) - 1):
                    a, b = path[i + 1], path[i]
                    if (a, b) in pairwise:
                        H_total = pairwise[(a, b)] @ H_total
                    elif (b, a) in pairwise:
                        H_total = np.linalg.inv(pairwise[(b, a)]) @ H_total
                    else:
                        H_total = None
                        break
                if H_total is not None:
                    homographies[idx] = H_total

        return homographies

    @staticmethod
    def _manhattan_path(start_idx: int, end_idx: int, cols: int) -> List[int]:
        sr, sc = start_idx // cols, start_idx % cols
        er, ec = end_idx // cols, end_idx % cols
        path = [start_idx]
        r, c = sr, sc
        while r != er or c != ec:
            if c < ec:
                c += 1
            elif c > ec:
                c -= 1
            elif r < er:
                r += 1
            elif r > er:
                r -= 1
            path.append(r * cols + c)
        return path

    def _pairwise_homography(self, tile_a: np.ndarray, tile_b: np.ndarray) -> Optional[np.ndarray]:
        feats_a = self.matcher.detect_and_compute(tile_a)
        feats_b = self.matcher.detect_and_compute(tile_b)
        matches = self.matcher.match(feats_a[1], feats_b[1])
        if not matches:
            return None

        src_pts = np.float32([feats_b[0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([feats_a[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(src_pts) < 4:
            return None
        try:
            H, mask = self.homography.estimate(src_pts, dst_pts, self.ransac_thresh)
            if H is not None and mask is not None and int(mask.sum()) >= 4:
                return H
        except RuntimeError:
            pass
        return None
