from time import perf_counter
from typing import List, Optional, Tuple

import cv2
import numpy as np

from drawing_reconstructor.blender import Blender, MAX_CANVAS_AREA
from drawing_reconstructor.feature_matcher import FeatureMatcher
from drawing_reconstructor.homography_estimator import HomographyEstimator
from drawing_reconstructor.match_graph import MatchGraph
from drawing_reconstructor.models import MatchEdge, ReconstructionReport, ReconstructionResult
from drawing_reconstructor.tile_loader import TileLoader


class DrawingReconstructor:
    def __init__(
        self,
        detector: str = "sift",
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 4.0,
        min_inliers: int = 4,
        min_confidence: float = 0.05,
        max_canvas_area: Optional[int] = None,
    ):
        self.matcher = FeatureMatcher(detector, ratio_thresh)
        self.homography = HomographyEstimator()
        self.blender = Blender()
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.min_confidence = min_confidence
        self.max_canvas_area = max_canvas_area if max_canvas_area is not None else MAX_CANVAS_AREA

    def reconstruct(
        self,
        tiles: List[np.ndarray],
        grid: Optional[Tuple[int, int]] = None,
        overlap_estimate: float = 0.12,
    ) -> np.ndarray:
        return self.reconstruct_with_report(tiles, grid, overlap_estimate).image

    def reconstruct_with_report(
        self,
        tiles: List[np.ndarray],
        grid: Optional[Tuple[int, int]] = None,
        overlap_estimate: float = 0.12,
    ) -> ReconstructionResult:
        started_at = perf_counter()
        if grid is None:
            grid = TileLoader.infer_grid(tiles, (1, len(tiles)))
        rows, cols = grid
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {grid}")
        if len(tiles) != rows * cols:
            raise ValueError(f"Expected {rows*cols} tiles for {rows}x{cols} grid, got {len(tiles)}")
        TileLoader.validate_tiles(tiles)

        edges = self._build_match_edges(tiles, rows, cols)
        reference_tile = rows // 2 * cols + cols // 2
        plan = MatchGraph(len(tiles), edges).plan_homographies(reference_tile)
        if plan.failed_tiles:
            raise RuntimeError(
                f"Unable to connect all tiles to reference tile {reference_tile}: "
                f"{plan.failed_tiles}"
            )

        homographies = plan.homographies
        offset, canvas_w, canvas_h = Blender.compute_canvas(
            tiles, homographies, max_canvas_area=self.max_canvas_area
        )

        warped_images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for tile, H in zip(tiles, homographies):
            H_off = offset @ H
            warped = cv2.warpPerspective(tile, H_off, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(
                np.ones(tile.shape[:2], dtype=np.uint8) * 255,
                H_off,
                (canvas_w, canvas_h),
            )
            warped_images.append(warped)
            masks.append(mask)

        result = self.blender.feather_blend(warped_images, masks)
        coverage_ratio = self._coverage_ratio(masks)
        elapsed_ms = (perf_counter() - started_at) * 1000
        report = ReconstructionReport(
            grid=grid,
            reference_tile=reference_tile,
            tile_count=len(tiles),
            connected_tiles=plan.connected_tiles,
            failed_tiles=plan.failed_tiles,
            edges=edges,
            selected_paths=plan.selected_paths,
            canvas_size=(canvas_w, canvas_h),
            coverage_ratio=coverage_ratio,
            elapsed_ms=elapsed_ms,
            warnings=plan.warnings,
        )
        return ReconstructionResult(image=result, report=report)

    def _build_match_edges(self, tiles: List[np.ndarray], rows: int, cols: int) -> List[MatchEdge]:
        edges: List[MatchEdge] = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    right_idx = r * cols + (c + 1)
                    edge = self._pairwise_match_edge(idx, right_idx, tiles[idx], tiles[right_idx])
                    if edge is not None:
                        edges.append(edge)
                if r + 1 < rows:
                    down_idx = (r + 1) * cols + c
                    edge = self._pairwise_match_edge(idx, down_idx, tiles[idx], tiles[down_idx])
                    if edge is not None:
                        edges.append(edge)
        return edges

    def _pairwise_match_edge(
        self,
        source_idx: int,
        target_idx: int,
        tile_a: np.ndarray,
        tile_b: np.ndarray,
    ) -> Optional[MatchEdge]:
        feats_a = self.matcher.detect_and_compute(tile_a)
        feats_b = self.matcher.detect_and_compute(tile_b)
        matches = self.matcher.match(feats_a[1], feats_b[1])
        if not matches:
            return None

        kp_a, kp_b = feats_a[0], feats_b[0]
        valid_matches = [
            m for m in matches
            if 0 <= m.queryIdx < len(kp_a) and 0 <= m.trainIdx < len(kp_b)
        ]
        if len(valid_matches) < 4:
            return None

        src_pts = np.float32([kp_b[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_a[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

        if len(src_pts) < 4:
            return None
        try:
            H, mask = self.homography.estimate(src_pts, dst_pts, self.ransac_thresh)
            if H is None or mask is None:
                return None
            inliers = int(mask.sum())
            inlier_ratio = inliers / max(len(matches), 1)
            reprojection_error = self._reprojection_error(src_pts, dst_pts, H, mask)
            confidence = self._edge_confidence(len(matches), inliers, reprojection_error)
            if inliers >= self.min_inliers and confidence >= self.min_confidence:
                return MatchEdge(
                    source=source_idx,
                    target=target_idx,
                    matches=len(matches),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    reprojection_error=reprojection_error,
                    confidence=confidence,
                    homography=H,
                )
        except (RuntimeError, ValueError):
            pass
        return None

    @staticmethod
    def _edge_confidence(matches: int, inliers: int, reprojection_error: float) -> float:
        inlier_ratio = inliers / max(matches, 1)
        support = min(inliers / 30.0, 1.0)
        error_score = 1.0 / (1.0 + max(reprojection_error, 0.0))
        return float(inlier_ratio * 0.6 + support * 0.25 + error_score * 0.15)

    @staticmethod
    def _reprojection_error(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        homography: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        projected = cv2.perspectiveTransform(src_pts, homography)
        errors = np.linalg.norm(projected - dst_pts, axis=2).reshape(-1)
        inlier_mask = mask.reshape(-1).astype(bool)
        if not inlier_mask.any():
            return float("inf")
        return float(errors[inlier_mask].mean())

    @staticmethod
    def _coverage_ratio(masks: List[np.ndarray]) -> float:
        if not masks:
            return 0.0
        covered = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            covered |= mask > 0
        return float(covered.mean())
