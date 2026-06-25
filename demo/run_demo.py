import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from drawing_reconstructor import DrawingReconstructor


def generate_technical_drawing(width: int, height: int) -> np.ndarray:
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    margin = 60
    inner_x0, inner_y0 = margin, margin
    inner_x1, inner_y1 = width - margin, height - margin

    cv2.rectangle(canvas, (inner_x0, inner_y0), (inner_x1, inner_y1), (0, 0, 0), 3)

    grid_rows, grid_cols = 12, 16
    for i in range(1, grid_rows):
        y = inner_y0 + i * (inner_y1 - inner_y0) // grid_rows
        cv2.line(canvas, (inner_x0, y), (inner_x1, y), (0, 0, 0), 1)
    for i in range(1, grid_cols):
        x = inner_x0 + i * (inner_x1 - inner_x0) // grid_cols
        cv2.line(canvas, (x, inner_y0), (x, inner_y1), (0, 0, 0), 1)

    for i in range(8):
        cx = inner_x0 + 100 + i * ((inner_x1 - inner_x0 - 200) // 7)
        cy = inner_y0 + 100 + (i % 3) * 120
        r = 25 + (i % 4) * 15
        cv2.circle(canvas, (cx, cy), r, (0, 0, 0), 2)

    for i in range(5):
        x1 = inner_x0 + 80 + i * 60
        y1 = inner_y0 + 80
        x2 = inner_x0 + 80 + i * 60
        y2 = inner_y0 + 200
        cv2.arrowedLine(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1, tipLength=0.3)

    cv2.putText(canvas, "ENGINEERING DRAWING", (inner_x0 + 20, inner_y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "SCALE 1:1", (inner_x1 - 150, inner_y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "DWG NO. 2026-001", (inner_x0 + 20, inner_y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for i in range(3):
        y = inner_y0 + 300 + i * 150
        cv2.line(canvas, (inner_x0 + 50, y), (inner_x1 - 50, y), (0, 0, 0), 2)
        cv2.line(canvas, (inner_x0 + 50, y - 15), (inner_x0 + 50, y + 15), (0, 0, 0), 2)
        cv2.line(canvas, (inner_x1 - 50, y - 15), (inner_x1 - 50, y + 15), (0, 0, 0), 2)

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

    tile_h = (h + oh * (rows - 1)) // rows
    tile_w = (w + ow * (cols - 1)) // cols
    step_y = tile_h - oh
    step_x = tile_w - ow

    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * step_y
            x0 = c * step_x
            y0 = max(0, min(y0, h - tile_h))
            x0 = max(0, min(x0, w - tile_w))
            tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
            tiles.append(tile)
    return tiles


def main():
    base = os.path.dirname(__file__)
    out = os.path.join(base, "output")
    os.makedirs(out, exist_ok=True)

    scenarios = [
        ("1x4_clean", 1, 4, 0.0),
        ("1x4_handscan", 1, 4, 0.003),
        ("2x4_clean", 2, 4, 0.0),
        ("2x4_handscan", 2, 4, 0.003),
    ]

    reconstructor = DrawingReconstructor(detector="sift")

    for name, rows, cols, noise in scenarios:
        print(f"\n=== Scenario: {name} ({rows}x{cols}) ===")
        full = generate_technical_drawing(1600, 900)
        tiles = make_tiles(full, rows, cols)

        for i, tile in enumerate(tiles):
            degraded = add_degradation(tile, noise_level=noise, blur_sigma=0.5 if noise > 0 else 0.3)
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
