# Drawing Reconstructor

Reconstruct full technical drawings from overlapping scan tiles using feature-based image registration and stitching.

[中文](README_zh.md)

## Algorithm

1. **Feature Detection** — SIFT keypoints and descriptors on each tile
2. **Pairwise Matching** — BFMatcher with Lowe's ratio test (0.75) for adjacent tiles
3. **Match Graph** — each reliable adjacent transform becomes a confidence-scored graph edge
4. **Best-Path Homographies** — each tile is mapped to the reference tile through the highest-confidence path
5. **Warping** — `cv2.warpPerspective` to a unified canvas
6. **Blending** — Distance-transform feathering for seamless seams
7. **Diagnostics** — `ReconstructionReport` captures match quality, selected paths, coverage, and timing

For multi-row grids, adjacent-tile homographies are computed with RANSAC, scored by inlier support and reprojection error, then composed through the strongest graph path to the center tile.

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

For diagnostics, use the report-aware API:

```python
from drawing_reconstructor import DrawingReconstructor

reconstructor = DrawingReconstructor(detector="sift")
result = reconstructor.reconstruct_with_report(tiles, grid=(2, 4))

image = result.image
report = result.report.to_dict()
```

## Demo

```bash
python demo/run_demo.py
```

Generates synthetic 1x4 and 2x4 tile grids (clean + degraded variants) and writes result images plus `*_report.json` diagnostics to `demo/output/`.

## Testing

```bash
pytest
```

## Project Structure

```
drawing-reconstructor/
├── src/drawing_reconstructor/
│   ├── __init__.py
│   ├── tile_loader.py        # tile I/O and grid inference
│   ├── feature_matcher.py    # SIFT/ORB detection and matching
│   ├── homography_estimator.py
│   ├── match_graph.py        # confidence-scored path planning
│   ├── models.py             # result and report dataclasses
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
- `reconstruct()` preserves the legacy image-returning API
- `reconstruct_with_report()` returns `ReconstructionResult(image, report)`
- Disconnected or low-confidence tile graphs raise diagnostic errors instead of silently using identity transforms
