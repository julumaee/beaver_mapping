"""Tests for src/polygonizer.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polygonizer import detect_rois, _merge_candidates, MIN_AREA_M2


class _ClassifyCLF:
    """Mock classifier returning a fixed label."""
    def __init__(self, label: int, confidence: float = 0.9):
        self._label = label
        self._conf = confidence

    def predict(self, X):
        return np.full(len(X), self._label, dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, min(self._label, 1)] = self._conf
        return probs


def _write_raster(tmp_path: Path, cx=328_000.0, cy=6_821_000.0, size=1024) -> str:
    half = size // 2
    transform = from_bounds(cx - half, cy - half, cx + half, cy + half, size, size)
    p = tmp_path / "tile.tif"
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(str(p), "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs=CRS.from_epsg(3067),
                       transform=transform) as dst:
        dst.write(data)
    return str(p)


class TestDetectROIs:
    def test_flood_classifier_returns_flood_rois(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        flood_rois = detect_rois(jp2, _ClassifyCLF(label=1))
        assert len(flood_rois) >= 1

    def test_negative_classifier_returns_empty(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        flood_rois = detect_rois(jp2, _ClassifyCLF(label=0))
        assert flood_rois == []

    def test_roi_has_correct_structure(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        flood_rois = detect_rois(jp2, _ClassifyCLF(label=1))
        polygon, conf, area = flood_rois[0]
        assert polygon.geom_type == "Polygon"
        assert 0.0 <= conf <= 1.0
        assert area > 0

    def test_confidence_threshold_filters(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        flood_rois = detect_rois(jp2, _ClassifyCLF(label=1, confidence=0.9),
                                 confidence_threshold=0.95)
        assert flood_rois == []

    def test_min_area_filter(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        flood_rois = detect_rois(jp2, _ClassifyCLF(label=1), min_area_m2=1e12)
        assert flood_rois == []


class TestMergeCandidates:
    def test_empty_returns_empty(self):
        assert _merge_candidates([], MIN_AREA_M2) == []

    def test_adjacent_tiles_merged(self):
        a = box(0, 0, 256, 256)
        b = box(256, 0, 512, 256)
        rois = _merge_candidates([(a, 0.8), (b, 0.9)], min_area_m2=1)
        assert len(rois) == 1
        _, conf, area = rois[0]
        assert abs(area - 256 * 512) < 1
        assert abs(conf - 0.85) < 1e-6

    def test_area_filter(self):
        a = box(0, 0, 10, 10)
        assert _merge_candidates([(a, 0.9)], min_area_m2=200) == []
