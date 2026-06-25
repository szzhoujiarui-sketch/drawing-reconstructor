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
        tiles = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
        return tiles

    @staticmethod
    def infer_grid(tiles: List[np.ndarray], expected_grid: Tuple[int, int]) -> Tuple[int, int]:
        if len(tiles) == expected_grid[0] * expected_grid[1]:
            return expected_grid
        h, w = tiles[0].shape[:2]
        aspect = w / h
        if aspect > 2:
            return 1, len(tiles)
        elif aspect < 0.5:
            return len(tiles), 1
        cols = int(round(np.sqrt(len(tiles) * aspect)))
        rows = int(np.ceil(len(tiles) / cols))
        return rows, cols

    @staticmethod
    def validate_tiles(tiles: List[np.ndarray]) -> None:
        if not tiles:
            raise ValueError("Tile list is empty")
        h, w = tiles[0].shape[:2]
        for i, t in enumerate(tiles):
            if t.shape[:2] != (h, w):
                raise ValueError(f"Tile {i} size {t.shape[:2]} != reference {h}x{w}")
