# Drawing Reconstructor

Reconstruct full technical drawings from overlapping scan tiles using feature-based image registration and stitching.

## Algorithm

1. **Feature Detection** — SIFT keypoints and descriptors on each tile
2. **Pairwise Matching** — BFMatcher with Lowe's ratio test (0.75) for adjacent tiles only
3. **Homography Composition** — Manhattan-path chaining from each tile to the center tile
4. **Warping** — `cv2.warpPerspective` to a unified canvas
5. **Blending** — Distance-transform feathering for seamless seams

For multi-row grids, adjacent-tile homographies are computed with RANSAC and composed along the shortest Manhattan path to the center tile, avoiding cumulative drift from full-chain propagation.

## Requirements

- Python 3.10+
- OpenCV 4.8+, NumPy, SciPy, scikit-image

```bash
pip install -e ".[dev]"
```

## Usage

```python
from drawing_reconstructor import DrawingReconstructor

reconstructor = DrawingReconstructor(detector="sift")
result = reconstructor.reconstruct(tiles, grid=(2, 4))
```

## Demo

```bash
python demo/run_demo.py
```

Generates synthetic 1x4 and 2x4 tile grids (clean + degraded variants) and writes results to `demo/output/`.

## Project Structure

```
drawing-reconstructor/
├── src/drawing_reconstructor/
│   ├── __init__.py
│   ├── tile_loader.py        # tile I/O and grid inference
│   ├── feature_matcher.py    # SIFT/ORB detection and matching
│   ├── homography_estimator.py
│   ├── blender.py            # feather blending with distance transform
│   └── reconstructor.py      # main pipeline
├── demo/
│   ├── run_demo.py
│   └── output/
├── pyproject.toml
└── README.md
```

## Notes

- `detector="sift"` gives the best quality for technical drawings (patent-free in OpenCV 4.4+)
- For very low-texture drawings, try `ratio_thresh=0.8` to allow more matches
- Overlap estimate of 12% is used for grid inference and pairwise filtering
