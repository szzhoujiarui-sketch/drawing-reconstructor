from typing import List, Optional, Tuple

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
        ref_w, ref_h = ref_shape[1], ref_shape[0]
        overlap_px = int(overlap_estimate * ref_w)

        homographies = self._estimate_grid_homographies(tiles, rows, cols, overlap_px)
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

    def _estimate_grid_homographies(
        self, tiles: List[np.ndarray], rows: int, cols: int, overlap_px: int
    ) -> List[np.ndarray]:
        homographies = [np.eye(3, dtype=np.float64) for _ in tiles]
        ref_idx = rows // 2 * cols + cols // 2

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ref_feats = self.matcher.detect_and_compute(tiles[ref_idx])
                cur_feats = self.matcher.detect_and_compute(tiles[idx])
                matches = self.matcher.match(cur_feats[1], ref_feats[1])
                src, dst = self.matcher.get_match_points(matches, cur_feats[0], ref_feats[0])
                if len(src) >= 4:
                    H, _ = self.homography.estimate(src, dst, self.ransac_thresh)
                    homographies[idx] = H

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    right_idx = r * cols + (c + 1)
                    H_lr = self._pairwise_homography(tiles[idx], tiles[right_idx], right=True)
                    if H_lr is not None:
                        homographies[right_idx] = homographies[idx] @ H_lr
                if r + 1 < rows:
                    down_idx = (r + 1) * cols + c
                    H_lr = self._pairwise_homography(tiles[idx], tiles[down_idx], right=False)
                    if H_lr is not None:
                        homographies[down_idx] = homographies[idx] @ H_lr

        return homographies

    def _pairwise_homography(self, tile_a: np.ndarray, tile_b: np.ndarray, right: bool) -> Optional[np.ndarray]:
        feats_a = self.matcher.detect_and_compute(tile_a)
        feats_b = self.matcher.detect_and_compute(tile_b)
        matches = self.matcher.match(feats_a[1], feats_b[1])
        if not matches:
            return None

        kp_a, kp_b = feats_a[0], feats_b[0]
        src_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        if right:
            h_a, w_a = tile_a.shape[:2]
            keep = src_pts[:, 0, 0] < (w_a * 0.5)
            src_pts = src_pts[keep]
            dst_pts = dst_pts[keep]
        else:
            h_a, w_a = tile_a.shape[:2]
            keep = src_pts[:, 0, 1] < (h_a * 0.5)
            src_pts = src_pts[keep]
            dst_pts = dst_pts[keep]

        if len(src_pts) < 4:
            return None
        try:
            H, mask = self.homography.estimate(src_pts, dst_pts, self.ransac_thresh)
            return H
        except RuntimeError:
            return None
