from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Tuple

import numpy as np

from drawing_reconstructor.models import HomographyPlan, MatchEdge


@dataclass(frozen=True)
class _DirectedEdge:
    target: int
    cost: float
    confidence: float
    homography_to_current: np.ndarray


class MatchGraph:
    def __init__(self, tile_count: int, edges: Iterable[MatchEdge]):
        self.tile_count = tile_count
        self.edges = list(edges)
        self._adjacency = self._build_adjacency(self.edges)

    def plan_homographies(self, reference_tile: int) -> HomographyPlan:
        if reference_tile < 0 or reference_tile >= self.tile_count:
            raise ValueError(f"Reference tile {reference_tile} is outside 0..{self.tile_count - 1}")

        distances = [float("inf")] * self.tile_count
        previous: Dict[int, Tuple[int, np.ndarray]] = {}
        distances[reference_tile] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, reference_tile)]

        while queue:
            distance, tile_idx = heappop(queue)
            if distance > distances[tile_idx]:
                continue
            for edge in self._adjacency.get(tile_idx, []):
                next_distance = distance + edge.cost
                if next_distance < distances[edge.target]:
                    distances[edge.target] = next_distance
                    previous[edge.target] = (tile_idx, edge.homography_to_current)
                    heappush(queue, (next_distance, edge.target))

        homographies = [np.eye(3, dtype=np.float64) for _ in range(self.tile_count)]
        selected_paths: Dict[int, List[int]] = {reference_tile: [reference_tile]}
        connected_tiles = [reference_tile]
        failed_tiles: List[int] = []

        for tile_idx in range(self.tile_count):
            if tile_idx == reference_tile:
                continue
            if tile_idx not in previous:
                failed_tiles.append(tile_idx)
                continue
            path, homography = self._compose_path(tile_idx, reference_tile, previous)
            selected_paths[tile_idx] = path
            homographies[tile_idx] = homography
            connected_tiles.append(tile_idx)

        warnings = []
        if failed_tiles:
            warnings.append(f"Disconnected tiles: {failed_tiles}")

        return HomographyPlan(
            homographies=homographies,
            connected_tiles=sorted(connected_tiles),
            failed_tiles=failed_tiles,
            selected_paths=selected_paths,
            warnings=warnings,
        )

    @staticmethod
    def _build_adjacency(edges: List[MatchEdge]) -> Dict[int, List[_DirectedEdge]]:
        adjacency: Dict[int, List[_DirectedEdge]] = {}
        for edge in edges:
            cost = 1.0 / max(edge.confidence, 1e-6)
            adjacency.setdefault(edge.source, []).append(
                _DirectedEdge(
                    target=edge.target,
                    cost=cost,
                    confidence=edge.confidence,
                    homography_to_current=edge.homography,
                )
            )
            adjacency.setdefault(edge.target, []).append(
                _DirectedEdge(
                    target=edge.source,
                    cost=cost,
                    confidence=edge.confidence,
                    homography_to_current=np.linalg.inv(edge.homography),
                )
            )
        return adjacency

    @staticmethod
    def _compose_path(
        tile_idx: int,
        reference_tile: int,
        previous: Dict[int, Tuple[int, np.ndarray]],
    ) -> Tuple[List[int], np.ndarray]:
        path = [tile_idx]
        transforms: List[np.ndarray] = []
        current = tile_idx
        while current != reference_tile:
            parent, homography_to_parent = previous[current]
            transforms.append(homography_to_parent)
            current = parent
            path.append(current)

        homography = np.eye(3, dtype=np.float64)
        for transform in transforms:
            homography = transform @ homography
        return path, homography
