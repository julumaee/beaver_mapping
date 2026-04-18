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


def extract_features(chip: np.ndarray) -> np.ndarray:
    """
    Return an 18-element float32 feature vector from a (bands, H, W) chip:
      - Per-band mean, std, 25th and 75th percentile  (3 × 4 = 12)
      - NDVI mean, std, fraction of pixels > 0.2      (3)
      - NDWI mean, std, fraction of pixels > 0.0      (3)
    """
    feats: list[float] = []

    for b in range(chip.shape[0]):
        band = chip[b].astype(np.float32)
        feats += [
            float(band.mean()),
            float(band.std()),
            float(np.percentile(band, 25)),
            float(np.percentile(band, 75)),
        ]

    ndvi = compute_ndvi(chip)
    feats += [float(ndvi.mean()), float(ndvi.std()), float(np.mean(ndvi > 0.2))]

    ndwi = compute_ndwi(chip)
    feats += [float(ndwi.mean()), float(ndwi.std()), float(np.mean(ndwi > 0.0))]

    return np.array(feats, dtype=np.float32)
