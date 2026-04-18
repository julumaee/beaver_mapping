"""Tests for src/polygonizer.py (Task 4.3)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polygonizer import detect_rois, _merge_candidates, MIN_AREA_M2


class _AlwaysPositiveCLF:
    """Mock classifier that always returns label=1 with confidence=0.9."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X)) * 0.9])


class _AlwaysNegativeCLF:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.9, np.zeros(len(X))])


def _write_raster(tmp_path: Path, cx: float = 328_000.0, cy: float = 6_821_000.0, size: int = 1024) -> str:
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
    def test_no_stream_mask_uses_full_raster(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        rois = detect_rois(jp2, _AlwaysPositiveCLF(), stream_mask=None)
        assert len(rois) >= 1

    def test_always_negative_returns_empty(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        rois = detect_rois(jp2, _AlwaysNegativeCLF(), stream_mask=None)
        assert rois == []

    def test_stream_mask_limits_search(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=1024)
        # Tiny mask — only the top-left 512×512 quadrant
        small_mask = box(cx - 512, cy, cx, cy + 512)
        rois_all = detect_rois(jp2, _AlwaysPositiveCLF(), stream_mask=None)
        rois_masked = detect_rois(jp2, _AlwaysPositiveCLF(), stream_mask=small_mask)
        assert len(rois_masked) <= len(rois_all)

    def test_roi_has_three_elements(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        rois = detect_rois(jp2, _AlwaysPositiveCLF())
        poly, conf, area = rois[0]
        assert hasattr(poly, "area")
        assert 0.0 <= conf <= 1.0
        assert area > 0

    def test_min_area_filter(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        rois = detect_rois(jp2, _AlwaysPositiveCLF(), min_area_m2=1e12)
        assert rois == []

    def test_confidence_threshold_respected(self, tmp_path):
        jp2 = _write_raster(tmp_path)
        # _AlwaysPositiveCLF returns confidence=0.9; threshold above that → no rois
        rois = detect_rois(jp2, _AlwaysPositiveCLF(), confidence_threshold=0.95)
        assert rois == []


class TestMergeCandidates:
    def test_empty_input(self):
        assert _merge_candidates([], MIN_AREA_M2) == []

    def test_two_adjacent_tiles_merged(self):
        # Two adjacent 256×256 boxes
        a = box(0, 0, 256, 256)
        b = box(256, 0, 512, 256)
        rois = _merge_candidates([(a, 0.8), (b, 0.9)], min_area_m2=1)
        assert len(rois) == 1
        _, conf, area = rois[0]
        assert abs(area - 256 * 512) < 1
        assert abs(conf - 0.85) < 1e-6

    def test_separated_tiles_stay_separate(self):
        a = box(0, 0, 100, 100)
        b = box(1000, 1000, 1100, 1100)
        rois = _merge_candidates([(a, 0.8), (b, 0.9)], min_area_m2=1)
        assert len(rois) == 2

    def test_area_filter(self):
        a = box(0, 0, 10, 10)  # area = 100
        rois = _merge_candidates([(a, 0.9)], min_area_m2=200)
        assert rois == []
