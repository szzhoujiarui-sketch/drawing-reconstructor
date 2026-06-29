# Drawing Reconstructor

基于特征匹配和图像拼接，从多个重叠扫描切片中重建完整技术图纸。

## 算法流程

1. Feature Detection：对每个切片提取 SIFT keypoints 和 descriptors。
2. Pairwise Matching：匹配相邻切片，使用 BFMatcher 和 Lowe ratio test。
3. Match Graph：把可靠的相邻 homography 作为带置信度的图边。
4. Best-Path Homographies：每个切片沿最高置信路径映射到中心切片坐标系。
5. Warping：使用 `cv2.warpPerspective` 投影到统一画布。
6. Blending：使用 distance-transform feathering 平滑拼接缝。
7. Diagnostics：通过 `ReconstructionReport` 输出匹配质量、选中路径、覆盖率和耗时。

对于多行网格，系统会对相邻切片使用 RANSAC 计算 homography，根据 inlier 支持度和重投影误差评分，再沿到中心切片的最高置信路径合成变换。

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

诊断场景推荐使用报告 API：

```python
from drawing_reconstructor import DrawingReconstructor

reconstructor = DrawingReconstructor(detector="sift")
result = reconstructor.reconstruct_with_report(tiles, grid=(2, 4))

image = result.image
report = result.report.to_dict()
```

## 运行 Demo

```bash
python demo/run_demo.py
```

该命令会生成 1x4 和 2x4 的合成切片网格，包括 clean 与 degraded 变体，并将结果图片和 `*_report.json` 诊断报告写入 `demo/output/`。

## 测试

```bash
pytest
```

## 项目结构

```text
drawing-reconstructor/
├── src/drawing_reconstructor/
│   ├── __init__.py
│   ├── tile_loader.py
│   ├── feature_matcher.py
│   ├── homography_estimator.py
│   ├── match_graph.py
│   ├── models.py
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
- `reconstruct()` 保持返回图片的兼容 API。
- `reconstruct_with_report()` 返回 `ReconstructionResult(image, report)`。
- 当切片图断连或置信度过低时，系统会抛出诊断错误，避免静默使用 identity transform。

## 许可证

本项目使用 MIT License。详见 `LICENSE`。
