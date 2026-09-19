"""NDWI and NDVI spectral index computation for MML CIR imagery."""

import numpy as np
from scipy.ndimage import uniform_filter
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


# Block-length constants — the single source of truth for extract_features'
# layout, so tests and this docstring can derive expected lengths instead of
# hardcoding magic numbers.
CORE_LEN_TOPOLOGY    = 35  # per-scale core block, sm/md (see _features_for_region)
CORE_LEN_NO_TOPOLOGY = 23  # per-scale core block, full 512px chip
R24_EXTRA_LEN        = 5   # NDVI GLCM (4) + low-NDVI textured fraction (1)
CROSS_LEN            = 6

SM_BLOCK_LEN   = CORE_LEN_TOPOLOGY + R24_EXTRA_LEN     # 40
MD_BLOCK_LEN   = CORE_LEN_TOPOLOGY + R24_EXTRA_LEN     # 40
FULL_BLOCK_LEN = CORE_LEN_NO_TOPOLOGY                  # 23 (R2.4 extras omitted — see below)

FEATURE_VECTOR_LENGTH = SM_BLOCK_LEN + MD_BLOCK_LEN + FULL_BLOCK_LEN + CROSS_LEN  # 109


def extract_features(chip: np.ndarray) -> np.ndarray:
    """
    Return a FEATURE_VECTOR_LENGTH (109)-element float32 feature vector from
    a (bands, H, W) chip.

    32px and 64px center crops — 40 features each (SM_BLOCK_LEN/MD_BLOCK_LEN):
      - Core block (35, see _features_for_region): per-band mean/std/p25/p75
        (12), NDVI mean/std/frac>0.2 (3), NDWI mean/std/frac>0.0 (3), NDWI
        gradient std (1), GLCM on NIR (4), connected wet-region stats at 3
        NDWI thresholds (12)
      - R2.4 extras (5, R24_EXTRA_LEN — see _extra_features_for_region):
        GLCM on NDVI — contrast/homogeneity/energy/correlation (4), fraction
        of low-NDVI (<0.1) pixels that are also high-local-variance in NIR,
        i.e. standing dead trees over flooded ground rather than smooth open
        water/bog (1)

    512px full chip — 23 features (CORE_LEN_NO_TOPOLOGY; topology and the
    R2.4 extras omitted):
      - Core block minus topology (23): per-band mean/std/p25/p75 (12),
        NDVI mean/std/frac>0.2 (3), NDWI mean/std/frac>0.0 (3), NDWI
        gradient std (1), GLCM on NIR (4)

    6 cross-scale contrast features (CROSS_LEN; 32px−512px and 64px−512px),
    computed from the *core* blocks only (indices below are unaffected by
    the R2.4 extras appended after them):
      - NDWI mean contrast                            (2)
      - NDWI wet-fraction contrast                     (2)
      - Fine-scale max-blob-area vs landscape wet-frac (2)

    R2.4 measurement notes (see wave2B analysis): NIR/(Red+Green) ratio,
    CIR "saturation" (max-min)/max, brightness/shadow-fraction, and a
    uniform-LBP histogram on NIR were all also tried (at 32/64/512px) but
    had near-zero or negative permutation importance under grouped spatial
    CV and were dropped (R2.5). NDVI GLCM and the low-NDVI texture split
    both measurably improved PR-AUC and were kept, at 32/64px only — at
    512px they cost several ms for no expected gain at landscape scale.

    Total: 40 + 40 + 23 + 6 = 109
    """
    _NDWI_MEAN = 15
    _NDWI_FRAC = 17
    _MAX_AREA  = 25  # index within the 35-element core block (sm/md only)

    sm_region = _center_crop(chip, FEATURE_REGION_SM)
    md_region = _center_crop(chip, FEATURE_REGION_MD)

    ndvi_sm, ndwi_sm = compute_ndvi(sm_region), compute_ndwi(sm_region)
    ndvi_md, ndwi_md = compute_ndvi(md_region), compute_ndwi(md_region)
    ndvi_full, ndwi_full = compute_ndvi(chip), compute_ndwi(chip)

    feats_sm = _features_for_region(sm_region, ndvi=ndvi_sm, ndwi=ndwi_sm)
    feats_md = _features_for_region(md_region, ndvi=ndvi_md, ndwi=ndwi_md)
    feats_full = _features_for_region(chip, topology=False, ndvi=ndvi_full, ndwi=ndwi_full)

    extra_sm = _extra_features_for_region(sm_region, ndvi_sm)
    extra_md = _extra_features_for_region(md_region, ndvi_md)

    # feats_full has no _MAX_AREA; use _NDWI_FRAC as the landscape reference.
    cross = np.array([
        feats_sm[_NDWI_MEAN] - feats_full[_NDWI_MEAN],
        feats_sm[_NDWI_FRAC] - feats_full[_NDWI_FRAC],
        feats_sm[_MAX_AREA]  - feats_full[_NDWI_FRAC],
        feats_md[_NDWI_MEAN] - feats_full[_NDWI_MEAN],
        feats_md[_NDWI_FRAC] - feats_full[_NDWI_FRAC],
        feats_md[_MAX_AREA]  - feats_full[_NDWI_FRAC],
    ], dtype=np.float32)

    return np.concatenate([
        feats_sm, extra_sm,
        feats_md, extra_md,
        feats_full,
        cross,
    ])


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


def _features_for_region(
    region: np.ndarray,
    topology: bool = True,
    ndvi: np.ndarray | None = None,
    ndwi: np.ndarray | None = None,
) -> np.ndarray:
    """Core 35-feature (topology=True) or 23-feature (topology=False) block.
    ndvi/ndwi may be passed in to avoid recomputing them when the caller
    already has them (extract_features reuses them for the R2.4 extras)."""
    feats: list[float] = []

    region_f = region.astype(np.float32)
    for b in range(region_f.shape[0]):
        band = region_f[b]
        p25, p75 = np.percentile(band, [25, 75])  # one sort instead of two
        feats += [float(band.mean()), float(band.std()), float(p25), float(p75)]

    if ndvi is None:
        ndvi = compute_ndvi(region)
    feats += [float(ndvi.mean()), float(ndvi.std()), float(np.mean(ndvi > 0.2))]

    if ndwi is None:
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


# ---------------------------------------------------------------------------
# R2.4 — additional per-pixel/texture features (sm/md scales only)
# ---------------------------------------------------------------------------
#
# R2.4 also tried, at 32/64/512px, but pruned (R2.5) after measuring near-zero
# or negative permutation importance under grouped spatial CV on chips_new
# (see wave2B/importance_new135.csv): NIR/(Red+Green) ratio mean/std, CIR
# "saturation" (max-min)/max mean/std, brightness mean/std + shadow fraction,
# plain low-NDVI pixel fraction (the *textured* low-NDVI fraction below did
# help; the untextured total did not), and a uniform-LBP (P=8, R=1) histogram
# on NIR (all 10 bins were noise or mildly harmful — RF splitting on raw LBP
# codes didn't generalise across the label set).

_LOCAL_VAR_WINDOW = 5  # pixels, for the low-NDVI texture split


def _extra_features_for_region(region: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """R24_EXTRA_LEN (5) surviving R2.4 candidate features for one crop:
    GLCM on NDVI (4) + low-NDVI textured fraction (1)."""
    return np.concatenate([
        _glcm_ndvi_features(ndvi),
        _low_ndvi_textured_fraction(region, ndvi),
    ])


def _low_ndvi_textured_fraction(region: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """
    Fraction of low-NDVI pixels (NDVI < 0.1) that also sit in high
    local-variance NIR areas — standing dead trees over flooded ground are
    coarse/textured, while smooth open water or bog is not.

    Local variance is computed with a fast separable box filter
    (mean-of-squares minus square-of-mean) rather than a per-pixel
    sliding-window std, which would be far too slow.

    Returns [low_ndvi_textured_frac] — a fraction of the low-NDVI pixels
    themselves, 0 when there are none.
    """
    low_ndvi = ndvi < 0.1
    n_low = int(low_ndvi.sum())
    if n_low == 0:
        return np.array([0.0], dtype=np.float32)

    nir = region[_NIR].astype(np.float32)
    local_mean = uniform_filter(nir, size=_LOCAL_VAR_WINDOW)
    local_sq_mean = uniform_filter(nir * nir, size=_LOCAL_VAR_WINDOW)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0.0, None)

    var_thresh = float(np.median(local_var))
    textured = low_ndvi & (local_var > var_thresh)
    return np.array([float(textured.sum()) / n_low], dtype=np.float32)


def _glcm_ndvi_features(ndvi: np.ndarray) -> np.ndarray:
    """
    GLCM texture on NDVI (rather than NIR) — dead standing trees and beaver
    floods can differ in NDVI texture even when NIR brightness is similar.
    Uses 32 grey levels (vs 64 for the NIR GLCM) to control cost.

    Returns [contrast, homogeneity, energy, correlation].
    """
    ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
    if ndvi_max > ndvi_min:
        ndvi_u8 = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 31).astype(np.uint8)
    else:
        ndvi_u8 = np.zeros_like(ndvi, dtype=np.uint8)

    glcm = graycomatrix(
        ndvi_u8,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=32,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        val = graycoprops(glcm, prop).mean()
        feats.append(float(np.nan_to_num(val, nan=0.0)))
    return np.array(feats, dtype=np.float32)
