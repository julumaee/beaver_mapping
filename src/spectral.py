"""NDWI and NDVI spectral index computation for MML CIR imagery."""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as skimage_label, regionprops, perimeter as skimage_perimeter

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


# Three crop sizes for multi-scale feature extraction.
# Small (32px = 16m): tight crop for small features — less dilution by surroundings.
# Medium (64px = 32m): standard crop matching the detection patch size.
# Full chip (512px = 256m): landscape context.
FEATURE_REGION_SM = 32   # pixels — 16×16m at 0.5m/px
FEATURE_REGION_MD = 64   # pixels — 32×32m at 0.5m/px
FEATURE_REGION    = FEATURE_REGION_MD  # kept for backwards compatibility


def extract_features(chip: np.ndarray) -> np.ndarray:
    """
    Return a 99-element float32 feature vector from a (bands, H, W) chip.

    32px and 64px center crops — 35 features each (full set):
      - Per-band mean, std, p25, p75                  (3 × 4 = 12)
      - NDVI mean, std, fraction > 0.2                (3)
      - NDWI mean, std, fraction > 0.0                (3)
      - NDWI gradient std                             (1)
      - GLCM on NIR                                   (4)
      - Connected wet-region stats at 3 NDWI thresholds
        (wet_frac, n_components, max_area_frac, shape_index × 3) (12)

    512px chip — 23 features (topology omitted):
      - Per-band mean, std, p25, p75                  (12)
      - NDVI mean, std, fraction > 0.2                (3)
      - NDWI mean, std, fraction > 0.0                (3)
      - NDWI gradient std                             (1)
      - GLCM on NIR                                   (4)
      Connected-component analysis on a 512×512 image costs ~480 ms per patch
      (~93% of total feature time) and those 12 features had near-zero importance.

    6 cross-scale contrast features (32px−512px and 64px−512px):
      - NDWI mean contrast                            (2)
      - NDWI wet-fraction contrast                    (2)
      - Fine-scale max-blob-area vs landscape wet-frac (2)

    Total: 35 + 35 + 23 + 6 = 99
    """
    _NDWI_MEAN = 15
    _NDWI_FRAC = 17
    _MAX_AREA  = 25  # index within the 35-element per-scale block (sm/md only)

    feats_sm   = _features_for_region(_center_crop(chip, FEATURE_REGION_SM))
    feats_md   = _features_for_region(_center_crop(chip, FEATURE_REGION_MD))
    feats_full = _features_for_region(chip, topology=False)  # 23 features

    # feats_full has no _MAX_AREA; use _NDWI_FRAC as the landscape reference.
    cross = np.array([
        feats_sm[_NDWI_MEAN] - feats_full[_NDWI_MEAN],
        feats_sm[_NDWI_FRAC] - feats_full[_NDWI_FRAC],
        feats_sm[_MAX_AREA]  - feats_full[_NDWI_FRAC],
        feats_md[_NDWI_MEAN] - feats_full[_NDWI_MEAN],
        feats_md[_NDWI_FRAC] - feats_full[_NDWI_FRAC],
        feats_md[_MAX_AREA]  - feats_full[_NDWI_FRAC],
    ], dtype=np.float32)

    return np.concatenate([feats_sm, feats_md, feats_full, cross])


def _center_crop(chip: np.ndarray, size: int) -> np.ndarray:
    """Return the central (size × size) pixels of chip."""
    _, h, w = chip.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return chip[:, r0 : r0 + size, c0 : c0 + size]


def _glcm_features(region: np.ndarray) -> np.ndarray:
    """
    Compute GLCM texture features on the NIR band.

    Uses 64 grey levels and two angles (0°, 90°) at distance 1, then
    averages across angles. Returns [contrast, homogeneity, energy, correlation].

    Wet forest (dead standing trees) has high NIR but coarse texture;
    beaver flood (open water) has very low contrast and high homogeneity.
    """
    nir = region[_NIR].astype(np.float32)
    # Normalise to [0, 63] uint8 for GLCM
    nir_min, nir_max = nir.min(), nir.max()
    if nir_max > nir_min:
        nir_u8 = ((nir - nir_min) / (nir_max - nir_min) * 63).astype(np.uint8)
    else:
        nir_u8 = np.zeros_like(nir, dtype=np.uint8)

    glcm = graycomatrix(
        nir_u8,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=64,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        val = graycoprops(glcm, prop).mean()
        # correlation is NaN when std=0 (uniform patch); replace with 0.
        feats.append(float(np.nan_to_num(val, nan=0.0)))
    return np.array(feats, dtype=np.float32)


def _features_for_region(region: np.ndarray, topology: bool = True) -> np.ndarray:
    feats: list[float] = []

    region_f = region.astype(np.float32)
    for b in range(region_f.shape[0]):
        band = region_f[b]
        p25, p75 = np.percentile(band, [25, 75])  # one sort instead of two
        feats += [float(band.mean()), float(band.std()), float(p25), float(p75)]

    ndvi = compute_ndvi(region)
    feats += [float(ndvi.mean()), float(ndvi.std()), float(np.mean(ndvi > 0.2))]

    ndwi = compute_ndwi(region)
    feats += [float(ndwi.mean()), float(ndwi.std()), float(np.mean(ndwi > 0.0))]

    dy, dx = np.gradient(ndwi)
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    feats.append(float(grad_mag.std()))

    feats += _glcm_features(region).tolist()

    if topology:
        feats += _connected_wet_features(ndwi).tolist()

    return np.array(feats, dtype=np.float32)


def _connected_wet_features(ndwi: np.ndarray) -> np.ndarray:
    """
    Connected-component statistics on the NDWI map at three thresholds.

    For each threshold t in (0.0, 0.1, 0.2) returns:
      - wet pixel fraction
      - number of connected wet components (capped at 255 to avoid outliers)
      - area of the largest wet component as a fraction of total pixels

    Beaver floods produce a single large wet blob; wet forest produces
    many small blobs; dry stream gives near-zero fraction.
    """
    total = ndwi.size
    feats: list[float] = []
    for thresh in (0.0, 0.1, 0.2):
        binary = ndwi > thresh
        wet_frac = float(binary.mean())
        if wet_frac == 0.0:
            feats += [0.0, 0.0, 0.0, 0.0]
            continue
        labeled = skimage_label(binary)
        props = regionprops(labeled)
        n_components = min(len(props), 255)
        max_area_frac = max(p.area for p in props) / total if props else 0.0

        # Perimeter/sqrt(area) of the largest blob — low = compact circular pond,
        # high = irregular wet forest edges.
        # Set to 0 when the blob touches the image boundary (perimeter is
        # undercounted there, making any shape appear artificially compact).
        largest = max(props, key=lambda p: p.area)
        largest_mask = labeled == largest.label
        h, w = largest_mask.shape
        touches_edge = (
            largest_mask[0, :].any() or largest_mask[-1, :].any() or
            largest_mask[:, 0].any() or largest_mask[:, -1].any()
        )
        if touches_edge:
            shape_index = 0.0
        else:
            perim = float(skimage_perimeter(largest_mask))
            shape_index = perim / max(1.0, largest.area ** 0.5)

        feats += [wet_frac, float(n_components), float(max_area_frac), shape_index]
    return np.array(feats, dtype=np.float32)
