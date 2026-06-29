# Drawing Reconstructor

基于特征匹配和图像拼接，从多个重叠扫描切片中重建完整技术图纸。

## 算法流程

1. Feature Detection：对每个切片提取 SIFT keypoints 和 descriptors。
2. Pairwise Matching：只匹配相邻切片，使用 BFMatcher 和 Lowe ratio test。
3. Homography Composition：从每个切片沿 Manhattan path 合成到中心切片坐标系。
4. Warping：使用 `cv2.warpPerspective` 投影到统一画布。
5. Blending：使用 distance-transform feathering 平滑拼接缝。

对于多行网格，系统会对相邻切片使用 RANSAC 计算 homography，并沿到中心切片的最短 Manhattan path 合成变换，减少长链路传播带来的累计漂移。

## 环境要求

- Python 3.10+
- OpenCV 4.8+
- NumPy
- SciPy
- scikit-image

## 安装

```bash
pip install -e ".[dev]"
```

## 使用示例

```python
from drawing_reconstructor import DrawingReconstructor

reconstructor = DrawingReconstructor(detector="sift")
result = reconstructor.reconstruct(tiles, grid=(2, 4))
```

## 运行 Demo

```bash
python demo/run_demo.py
```

该命令会生成 1x4 和 2x4 的合成切片网格，包括 clean 与 degraded 变体，并将结果写入 `demo/output/`。

## 项目结构

```text
drawing-reconstructor/
├── src/drawing_reconstructor/
│   ├── __init__.py
│   ├── tile_loader.py
│   ├── feature_matcher.py
│   ├── homography_estimator.py
│   ├── blender.py
│   └── reconstructor.py
├── demo/
│   ├── run_demo.py
│   └── output/
├── pyproject.toml
└── README.md
```

## 说明

- `detector="sift"` 对技术图纸通常质量更好。
- 对低纹理图纸，可以尝试将 `ratio_thresh` 调整为 `0.8`，以允许更多匹配。
- 网格推断和相邻切片过滤默认使用约 12% 的重叠估计。

## 许可证

本项目使用 MIT License。详见 `LICENSE`。
