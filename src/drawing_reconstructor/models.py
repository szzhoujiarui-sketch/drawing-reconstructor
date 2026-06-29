from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MatchEdge:
    source: int
    target: int
    matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    confidence: float
    homography: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data.pop("homography")
        return data


@dataclass(frozen=True)
class ReconstructionReport:
    grid: Tuple[int, int]
    reference_tile: int
    tile_count: int
    connected_tiles: List[int]
    failed_tiles: List[int]
    edges: List[MatchEdge]
    selected_paths: Dict[int, List[int]]
    canvas_size: Tuple[int, int]
    coverage_ratio: float
    elapsed_ms: float
    warnings: List[str]

    @property
    def successful(self) -> bool:
        return not self.failed_tiles

    def to_dict(self) -> Dict[str, object]:
        return {
            "grid": list(self.grid),
            "reference_tile": self.reference_tile,
            "tile_count": self.tile_count,
            "connected_tiles": self.connected_tiles,
            "failed_tiles": self.failed_tiles,
            "edges": [edge.to_dict() for edge in self.edges],
            "selected_paths": {str(key): value for key, value in self.selected_paths.items()},
            "canvas_size": list(self.canvas_size),
            "coverage_ratio": self.coverage_ratio,
            "elapsed_ms": self.elapsed_ms,
            "warnings": self.warnings,
            "successful": self.successful,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    image: np.ndarray
    report: ReconstructionReport


@dataclass(frozen=True)
class HomographyPlan:
    homographies: List[np.ndarray]
    connected_tiles: List[int]
    failed_tiles: List[int]
    selected_paths: Dict[int, List[int]]
    warnings: List[str]
