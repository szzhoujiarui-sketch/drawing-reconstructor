import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

MAX_TILE_AREA = 50_000_000


class TileLoader:
    @staticmethod
    def load_tiles_from_dir(tiles_dir: str) -> List[np.ndarray]:
        paths = sorted(glob.glob(os.path.join(tiles_dir, "*.png")))
        if not paths:
            raise FileNotFoundError(f"No PNG tiles found in {tiles_dir}")
        tiles = []
        for path in paths:
            tile = cv2.imread(path, cv2.IMREAD_COLOR)
            if tile is None:
                raise ValueError(f"Failed to read tile image: {path}")
            tiles.append(tile)
        return tiles

    @staticmethod
    def infer_grid(
        tiles: List[np.ndarray],
        expected_grid: Tuple[int, int],
        max_tile_area: Optional[int] = None,
    ) -> Tuple[int, int]:
        TileLoader.validate_tiles(tiles, max_tile_area=max_tile_area)
        if expected_grid[0] <= 0 or expected_grid[1] <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {expected_grid}")
        if len(tiles) == expected_grid[0] * expected_grid[1]:
            return expected_grid
        h, w = tiles[0].shape[:2]
        aspect = w / h
        if aspect > 2:
            return 1, len(tiles)
        elif aspect < 0.5:
            return len(tiles), 1
        cols = int(round(np.sqrt(len(tiles) * aspect)))
        cols = max(cols, 1)
        rows = int(np.ceil(len(tiles) / cols))
        return rows, cols

    @staticmethod
    def validate_tiles(tiles: List[np.ndarray], max_tile_area: Optional[int] = None) -> None:
        if not tiles:
            raise ValueError("Tile list is empty")
        if tiles[0] is None:
            raise ValueError("Tile 0 is not a valid image")
        h, w = tiles[0].shape[:2]
        limit = max_tile_area if max_tile_area is not None else MAX_TILE_AREA
        if limit <= 0:
            raise ValueError(f"Tile area limit must be positive, got {limit}")
        for i, t in enumerate(tiles):
            if t is None:
                raise ValueError(f"Tile {i} is not a valid image")
            if t.ndim not in (2, 3):
                raise ValueError(f"Tile {i} must be grayscale or color image, got shape {t.shape}")
            if t.shape[:2] != (h, w):
                raise ValueError(f"Tile {i} size {t.shape[:2]} != reference {h}x{w}")
            tile_h, tile_w = t.shape[:2]
            if tile_h <= 0 or tile_w <= 0:
                raise ValueError(f"Tile {i} dimensions must be positive, got {tile_h}x{tile_w}")
            if tile_h * tile_w > limit:
                raise ValueError(
                    f"Tile {i} area {tile_h * tile_w} exceeds limit {limit} "
                    f"({tile_h}x{tile_w})"
                )
