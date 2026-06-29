import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


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
    def infer_grid(tiles: List[np.ndarray], expected_grid: Tuple[int, int]) -> Tuple[int, int]:
        TileLoader.validate_tiles(tiles)
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
    def validate_tiles(tiles: List[np.ndarray]) -> None:
        if not tiles:
            raise ValueError("Tile list is empty")
        if tiles[0] is None:
            raise ValueError("Tile 0 is not a valid image")
        h, w = tiles[0].shape[:2]
        for i, t in enumerate(tiles):
            if t is None:
                raise ValueError(f"Tile {i} is not a valid image")
            if t.ndim not in (2, 3):
                raise ValueError(f"Tile {i} must be grayscale or color image, got shape {t.shape}")
            if t.shape[:2] != (h, w):
                raise ValueError(f"Tile {i} size {t.shape[:2]} != reference {h}x{w}")
