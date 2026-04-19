"""NDWI and NDVI spectral index computation for MML CIR imagery."""

import numpy as np

# MML Vääräväri (CIR) band order (0-indexed): NIR=0, Red=1, Green=2
_NIR = 0
_RED = 1
_GRN = 2

_EPS = 1e-6


def compute_ndvi(chip: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Returns float32 (H, W) in [-1, 1]."""
    nir = chip[_NIR].astype(np.float32)
    red = chip[_RED].astype(np.float32)
    return (nir - red) / (nir + red + _EPS)


def compute_ndwi(chip: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Returns float32 (H, W) in [-1, 1]."""
    nir = chip[_NIR].astype(np.float32)
    grn = chip[_GRN].astype(np.float32)
    return (grn - nir) / (grn + nir + _EPS)


# Size of the central sub-region used for feature extraction.
# Features are computed over this inner window rather than the full chip so that
# the spectral signal from the labeled feature (dam, pond, ghost forest) is not
# diluted by the surrounding 256×256m landscape context.
FEATURE_REGION = 64  # pixels — 32×32m at 0.5m/px


def extract_features(chip: np.ndarray) -> np.ndarray:
    """
    Return a 36-element float32 feature vector from a (bands, H, W) chip.

    Features are computed separately on two spatial scales:
      - Central 64×64px (32m) — captures the feature itself
      - Full chip 512×512px (256m) — captures surrounding landscape context
    Each scale contributes 18 values:
      - Per-band mean, std, 25th and 75th percentile  (3 × 4 = 12)
      - NDVI mean, std, fraction of pixels > 0.2      (3)
      - NDWI mean, std, fraction of pixels > 0.0      (3)
    """
    return np.concatenate([
        _features_for_region(_center_crop(chip, FEATURE_REGION)),
        _features_for_region(chip),
    ])


def _center_crop(chip: np.ndarray, size: int) -> np.ndarray:
    """Return the central (size × size) pixels of chip."""
    _, h, w = chip.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return chip[:, r0 : r0 + size, c0 : c0 + size]


def _features_for_region(region: np.ndarray) -> np.ndarray:
    feats: list[float] = []

    for b in range(region.shape[0]):
        band = region[b].astype(np.float32)
        feats += [
            float(band.mean()),
            float(band.std()),
            float(np.percentile(band, 25)),
            float(np.percentile(band, 75)),
        ]

    ndvi = compute_ndvi(region)
    feats += [float(ndvi.mean()), float(ndvi.std()), float(np.mean(ndvi > 0.2))]

    ndwi = compute_ndwi(region)
    feats += [float(ndwi.mean()), float(ndwi.std()), float(np.mean(ndwi > 0.0))]

    return np.array(feats, dtype=np.float32)
