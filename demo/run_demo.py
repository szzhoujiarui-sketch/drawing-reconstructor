import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from drawing_reconstructor import DrawingReconstructor


def generate_technical_drawing(width: int, height: int) -> np.ndarray:
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (50, 50), (width - 50, height - 50), (0, 0, 0), 2)
    for i in range(5):
        y = 80 + i * 60
        cv2.line(canvas, (80, y), (width - 80, y), (0, 0, 0), 1)
    for i in range(4):
        x = 80 + i * ((width - 160) // 3)
        cv2.line(canvas, (x, 80), (x, height - 80), (0, 0, 0), 1)
    cv2.putText(canvas, "TECHNICAL DRAWING SAMPLE", (width // 2 - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas


def add_degradation(image: np.ndarray, noise_level: float = 0.01, blur_sigma: float = 0.5) -> np.ndarray:
    result = image.copy()
    if blur_sigma > 0:
        ksize = int(blur_sigma * 3) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), blur_sigma)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return result


def make_tiles(image: np.ndarray, rows: int, cols: int, overlap_pct: float = 0.12) -> list:
    h, w = image.shape[:2]
    oh = int(h * overlap_pct)
    ow = int(w * overlap_pct)
    tile_h = h // rows + oh // (rows - 1) if rows > 1 else h
    tile_w = w // cols + ow // (cols - 1) if cols > 1 else w

    tiles = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            y0 = r * (tile_h - oh // max(rows - 1, 1))
            x0 = c * (tile_w - ow // max(cols - 1, 1))
            x0 = min(x0, w - tile_w)
            y0 = min(y0, h - tile_h)
            tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
            row_tiles.append(tile)
        tiles.extend(row_tiles)
    return tiles


def main():
    base = os.path.dirname(__file__)
    out = os.path.join(base, "output")
    os.makedirs(out, exist_ok=True)

    scenarios = [
        ("1x4_clean", 1, 4, 0.0),
        ("1x4_handscan", 1, 4, 0.02),
        ("2x4_clean", 2, 4, 0.0),
        ("2x4_handscan", 2, 4, 0.015),
    ]

    reconstructor = DrawingReconstructor(detector="sift")

    for name, rows, cols, noise in scenarios:
        print(f"\n=== Scenario: {name} ({rows}x{cols}) ===")
        full = generate_technical_drawing(1600, 900)
        tiles = make_tiles(full, rows, cols)

        for i, tile in enumerate(tiles):
            degraded = add_degradation(tile, noise_level=noise, blur_sigma=0.8 if noise > 0 else 0.3)
            cv2.imwrite(os.path.join(out, f"{name}_tile_{i:02d}.png"), degraded)

        try:
            result = reconstructor.reconstruct(tiles, grid=(rows, cols))
            cv2.imwrite(os.path.join(out, f"{name}_result.png"), result)

            ref_gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
            res_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            mse = ((ref_gray.astype(float) - cv2.resize(res_gray, (ref_gray.shape[1], ref_gray.shape[0])).astype(float)) ** 2).mean()
            print(f"  Reconstruction saved. MSE vs reference: {mse:.2f}")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
