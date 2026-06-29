import numpy as np
import pytest

from drawing_reconstructor import DrawingReconstructor, ReconstructionResult
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
