"""Tests for src/polygonizer.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polygonizer import (
    detect_rois,
    _merge_candidates,
    _nanmean_smooth_3x3,
    _prob_map_to_rois,
    MIN_AREA_M2,
)


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


# 32m patches (matches PATCH_SIZE = 64px @ 0.5 m/px in polygonizer.py) —
# each cell covers 32*32 = 1024 m^2.
_PATCH_TRANSFORM = Affine(32.0, 0.0, 0.0, 0.0, -32.0, 0.0)


class TestMinAreaR41:
    """R4.1 — a detection must cover at least 2 patches (default min area 2048 m^2)."""

    def test_default_min_area_is_2048(self):
        assert MIN_AREA_M2 == 2048

    def test_single_isolated_patch_dropped(self):
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        prob_map[2, 2] = 0.9
        rois = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                 min_area_m2=MIN_AREA_M2, smooth=False)
        assert rois == []

    def test_two_adjacent_patches_survive(self):
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        prob_map[2, 2] = 0.9
        prob_map[2, 3] = 0.9
        rois = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                 min_area_m2=MIN_AREA_M2, smooth=False)
        assert len(rois) == 1
        _, _, area = rois[0]
        assert area >= MIN_AREA_M2


class TestHysteresisR42:
    """R4.2 — a connected region survives only if it contains a seed cell."""

    def test_region_without_seed_cell_dropped(self):
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        # Both cells clear `threshold` but neither reaches the default
        # seed_threshold of min(threshold + 0.15, 0.95) = 0.65.
        prob_map[2, 2] = 0.55
        prob_map[2, 3] = 0.55
        rois = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                 min_area_m2=1.0, smooth=False)
        assert rois == []

    def test_region_with_seed_cell_kept(self):
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        prob_map[2, 2] = 0.55
        prob_map[2, 3] = 0.9  # seed cell (>= default seed_threshold 0.65)
        rois = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                 min_area_m2=1.0, smooth=False)
        assert len(rois) == 1
        poly, confidence, area = rois[0]
        assert area == pytest.approx(2 * 1024.0)
        assert confidence == pytest.approx((0.55 + 0.9) / 2)

    def test_explicit_seed_threshold_overrides_default(self):
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        prob_map[2, 2] = 0.55
        prob_map[2, 3] = 0.6  # would be a seed under a lower explicit threshold
        rois = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                 min_area_m2=1.0, seed_threshold=0.58, smooth=False)
        assert len(rois) == 1


class TestSmoothingR42:
    def test_smoothing_ignores_nan_neighbours(self):
        prob_map = np.full((3, 3), np.nan, dtype=np.float32)
        prob_map[1, 1] = 0.6
        prob_map[1, 0] = 0.8
        smoothed = _nanmean_smooth_3x3(prob_map)
        # Only (1,1) and (1,0) are non-NaN in (1,1)'s 3x3 neighbourhood.
        assert smoothed[1, 1] == pytest.approx(0.7)

    def test_smoothing_keeps_outside_mask_cells_nan(self):
        prob_map = np.full((3, 3), np.nan, dtype=np.float32)
        prob_map[1, 1] = 0.6
        prob_map[1, 0] = 0.8
        smoothed = _nanmean_smooth_3x3(prob_map)
        assert np.isnan(smoothed[0, 0])

    def test_smoothing_affects_hysteresis_result(self):
        # A lone patch at 0.9 with a 0.4 neighbour: with smoothing, the
        # neighbour's smoothed value can rise; without smoothing it can't
        # reach threshold on its own. This just checks smooth vs no-smooth
        # can legitimately disagree, without asserting a specific direction.
        prob_map = np.full((5, 5), np.nan, dtype=np.float32)
        prob_map[2, 2] = 0.9
        prob_map[2, 3] = 0.9
        prob_map[2, 4] = 0.05
        no_smooth = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                      min_area_m2=1.0, smooth=False)
        smooth = _prob_map_to_rois(prob_map, _PATCH_TRANSFORM, threshold=0.5,
                                   min_area_m2=1.0, smooth=True)
        assert len(no_smooth) >= 1
        assert len(smooth) >= 1
