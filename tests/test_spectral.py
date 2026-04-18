"""Tests for src/spectral.py (Task 4.1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral import compute_ndvi, compute_ndwi, extract_features


def _chip(nir, red, grn, size=4):
    """Return a (3, size, size) uint8 chip with constant band values."""
    chip = np.zeros((3, size, size), dtype=np.uint8)
    chip[0] = nir   # NIR
    chip[1] = red   # Red
    chip[2] = grn   # Green
    return chip


class TestNDVI:
    def test_known_value(self):
        # NIR=150, Red=50 → NDVI = 100/200 = 0.5
        chip = _chip(nir=150, red=50, grn=100)
        ndvi = compute_ndvi(chip)
        assert ndvi.shape == (4, 4)
        assert abs(ndvi.mean() - 0.5) < 1e-3

    def test_equal_bands_near_zero(self):
        chip = _chip(nir=100, red=100, grn=100)
        ndvi = compute_ndvi(chip)
        assert abs(ndvi.mean()) < 1e-4

    def test_output_dtype(self):
        chip = _chip(nir=100, red=50, grn=80)
        assert compute_ndvi(chip).dtype == np.float32

    def test_range(self):
        rng = np.random.default_rng(0)
        chip = rng.integers(0, 255, (3, 64, 64), dtype=np.uint8)
        ndvi = compute_ndvi(chip)
        assert ndvi.min() >= -1.0 - 1e-5
        assert ndvi.max() <= 1.0 + 1e-5


class TestNDWI:
    def test_known_value(self):
        # Green=150, NIR=50 → NDWI = 100/200 = 0.5
        chip = _chip(nir=50, red=80, grn=150)
        ndwi = compute_ndwi(chip)
        assert abs(ndwi.mean() - 0.5) < 1e-3

    def test_negative_when_nir_dominates(self):
        chip = _chip(nir=200, red=100, grn=50)
        ndwi = compute_ndwi(chip)
        assert ndwi.mean() < 0

    def test_output_dtype(self):
        chip = _chip(nir=100, red=50, grn=80)
        assert compute_ndwi(chip).dtype == np.float32

    def test_range(self):
        rng = np.random.default_rng(1)
        chip = rng.integers(0, 255, (3, 64, 64), dtype=np.uint8)
        ndwi = compute_ndwi(chip)
        assert ndwi.min() >= -1.0 - 1e-5
        assert ndwi.max() <= 1.0 + 1e-5


class TestExtractFeatures:
    def test_output_shape(self):
        chip = _chip(nir=100, red=80, grn=60, size=64)
        feats = extract_features(chip)
        assert feats.shape == (18,)

    def test_output_dtype(self):
        chip = _chip(nir=100, red=80, grn=60, size=64)
        assert extract_features(chip).dtype == np.float32

    def test_water_chip_high_ndwi_fraction(self):
        # Water: low NIR, high Green → NDWI > 0
        chip = _chip(nir=30, red=40, grn=180, size=64)
        feats = extract_features(chip)
        ndwi_fraction = feats[17]  # last feature: fraction NDWI > 0
        assert ndwi_fraction > 0.9

    def test_vegetation_chip_high_ndvi_fraction(self):
        # Dense vegetation: high NIR, low Red → NDVI > 0.2
        chip = _chip(nir=200, red=40, grn=80, size=64)
        feats = extract_features(chip)
        ndvi_fraction = feats[14]  # fraction NDVI > 0.2
        assert ndvi_fraction > 0.9
