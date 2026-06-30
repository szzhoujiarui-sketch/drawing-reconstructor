import cv2
import numpy as np
import pytest

from drawing_reconstructor import DrawingReconstructor, ReconstructionResult
from drawing_reconstructor.blender import Blender
from drawing_reconstructor.feature_matcher import FeatureMatcher
from drawing_reconstructor.homography_estimator import HomographyEstimator
from drawing_reconstructor.tile_loader import TileLoader


def test_reconstruct_with_report_single_tile():
    tile = np.ones((16, 16, 3), dtype=np.uint8) * 255
    tile[4:12, 4:12] = 0
    reconstructor = DrawingReconstructor()

    result = reconstructor.reconstruct_with_report([tile], grid=(1, 1))

    assert isinstance(result, ReconstructionResult)
    assert result.image.shape == tile.shape
    assert result.report.successful is True
    assert result.report.edges == []
    assert result.report.connected_tiles == [0]
    assert result.report.coverage_ratio == 1.0


def test_reconstruct_keeps_image_returning_api():
    tile = np.ones((8, 8, 3), dtype=np.uint8) * 255
    image = DrawingReconstructor().reconstruct([tile], grid=(1, 1))

    assert isinstance(image, np.ndarray)
    assert image.shape == tile.shape


def test_invalid_grid_fails_fast():
    tile = np.ones((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Grid dimensions"):
        DrawingReconstructor().reconstruct([tile], grid=(0, 1))


def test_disconnected_tiles_raise_diagnostic_error():
    tiles = [np.ones((32, 32, 3), dtype=np.uint8) * value for value in (0, 255)]

    with pytest.raises(RuntimeError, match="Unable to connect all tiles"):
        DrawingReconstructor().reconstruct_with_report(tiles, grid=(1, 2))


def test_tile_loader_rejects_invalid_tiles():
    with pytest.raises(ValueError, match="Tile list is empty"):
        TileLoader.validate_tiles([])

    with pytest.raises(ValueError, match="not a valid image"):
        TileLoader.validate_tiles([None])


def test_compute_canvas_rejects_non_finite_points(monkeypatch):
    def fake_perspective_transform(pts, H):
        return np.array([[[np.inf, 0]], [[0, np.nan]], [[0, 0]], [[1, 1]]], dtype=np.float32)

    monkeypatch.setattr("cv2.perspectiveTransform", fake_perspective_transform)
    tile = np.ones((16, 16, 3), dtype=np.uint8)
    H = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        Blender.compute_canvas([tile], [H])


def test_compute_canvas_rejects_excessive_canvas_area():
    tile = np.ones((16, 16, 3), dtype=np.uint8)
    H = np.array([[500, 0, 0], [0, 500, 0], [0, 0, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="Canvas area"):
        Blender.compute_canvas([tile], [H], max_canvas_area=10000)


def test_compute_canvas_rejects_extreme_finite_coordinates_before_integer_cast():
    tile = np.ones((16, 16, 3), dtype=np.uint8)
    H = np.array([[1e20, 0, 0], [0, 1e20, 0], [0, 0, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="Canvas area"):
        Blender.compute_canvas([tile], [H], max_canvas_area=10000)


def test_compute_canvas_rejects_zero_or_negative_dimensions():
    tile = np.ones((16, 16, 3), dtype=np.uint8)
    H_zero = np.eye(3, dtype=np.float64) * 0
    with pytest.raises(ValueError, match="Canvas dimensions must be positive"):
        Blender.compute_canvas([tile], [H_zero])


def test_feather_blend_rejects_excessive_area():
    class FakeImage:
        shape = (20000, 20000, 3)

    class FakeMask:
        shape = (20000, 20000)

    called_with = []

    def guarded_zeros(*args, **kwargs):
        called_with.append(args)
        raise MemoryError("should not reach allocation")

    original_zeros = np.zeros
    np.zeros = guarded_zeros
    try:
        with pytest.raises(ValueError, match="Blend area"):
            Blender.feather_blend([FakeImage()], [FakeMask()])
    finally:
        np.zeros = original_zeros

    assert len(called_with) == 0


def test_pairwise_match_edge_filters_out_of_bounds_indices(monkeypatch):
    tile_a = np.ones((32, 32, 3), dtype=np.uint8) * 64
    tile_b = np.ones((32, 32, 3), dtype=np.uint8) * 128

    kp = [cv2.KeyPoint(8, 8, 1)]
    des = np.ones((1, 128), dtype=np.float32)
    monkeypatch.setattr(
        FeatureMatcher, "detect_and_compute",
        lambda self, img: (kp, des),
    )

    bad_match = cv2.DMatch(_queryIdx=999, _trainIdx=999, _distance=1.0, _imgIdx=0)
    monkeypatch.setattr(
        FeatureMatcher, "match",
        lambda self, d1, d2: [bad_match],
    )

    reconstructor = DrawingReconstructor()
    result = reconstructor._pairwise_match_edge(0, 1, tile_a, tile_b)
    assert result is None


def test_pairwise_match_edge_keeps_valid_indices(monkeypatch):
    tile_a = np.ones((32, 32, 3), dtype=np.uint8) * 64
    tile_b = np.ones((32, 32, 3), dtype=np.uint8) * 128

    kp = [cv2.KeyPoint(float(x), float(y), 1) for x, y in [(0, 0), (10, 0), (0, 10), (10, 10)]]
    des = np.ones((4, 128), dtype=np.float32)
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=1.0, _imgIdx=0)
        for i in range(4)
    ]

    monkeypatch.setattr(
        FeatureMatcher, "detect_and_compute",
        lambda self, img: (kp, des),
    )
    monkeypatch.setattr(
        FeatureMatcher, "match",
        lambda self, d1, d2: matches,
    )

    reconstructor = DrawingReconstructor()
    result = reconstructor._pairwise_match_edge(0, 1, tile_a, tile_b)
    assert result is not None
    assert result.matches == 4


def test_homography_estimator_rejects_non_finite_matrix(monkeypatch):
    H_nan = np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mask = np.ones((4, 1), dtype=np.uint8)
    monkeypatch.setattr("cv2.findHomography", lambda *a, **kw: (H_nan, mask))
    pts = np.ones((4, 1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-finite"):
        HomographyEstimator.estimate(pts, pts)


def test_estimator_rejects_near_singular_matrix(monkeypatch):
    H_singular = np.array(
        [[1, 2, 3], [2, 4, 6], [0, 0, 1]], dtype=np.float64
    )
    mask = np.ones((4, 1), dtype=np.uint8)
    monkeypatch.setattr("cv2.findHomography", lambda *a, **kw: (H_singular, mask))
    pts = np.ones((4, 1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="singular"):
        HomographyEstimator.estimate(pts, pts)


def test_get_match_points_filters_out_of_bounds_indices():
    kp1 = [cv2.KeyPoint(0, 0, 1)]
    kp2 = [cv2.KeyPoint(0, 0, 1)]
    bad = [cv2.DMatch(_queryIdx=5, _trainIdx=5, _distance=1.0, _imgIdx=0)]
    src, dst = FeatureMatcher.get_match_points(bad, kp1, kp2)
    assert src.size == 0
    assert dst.size == 0


def test_tile_loader_rejects_excessive_tile_area_before_feature_extraction():
    class FakeLargeTile:
        shape = (100000, 100000, 3)
        ndim = 3

    with pytest.raises(ValueError, match="Tile 0 area"):
        TileLoader.validate_tiles([FakeLargeTile()])


def test_feather_blend_rejects_zero_or_negative_dimensions():
    class FakeZeroImage:
        shape = (0, 100, 3)

    class FakeZeroMask:
        shape = (0, 100)

    with pytest.raises(ValueError, match="Blend dimensions must be positive"):
        Blender.feather_blend([FakeZeroImage()], [FakeZeroMask()])
