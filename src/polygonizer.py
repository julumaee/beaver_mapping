"""Sliding-window detection and ROI polygonization for beaver flood areas."""

import numpy as np
import rasterio
import rasterio.features
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import unary_union

from ingestion import TILE_SIZE
from models.random_forest import predict as clf_predict

MIN_AREA_M2 = 500
DETECTION_STRIDE = TILE_SIZE // 2  # 50% overlap — used by the legacy detect_rois path

# Dense patch-level RF segmentation constants
PATCH_SIZE = 64   # pixels — 32 m at 0.5 m/px; matches FEATURE_REGION_MD in spectral.py
PATCH_STRIDE = PATCH_SIZE // 2  # 32 px — 50% overlap ensures small features always
                                 # fall within the center crop of at least one patch
_CTX_OFFSET = (TILE_SIZE - PATCH_SIZE) // 2  # 224 — patch position in 512px context


def detect_rois(
    jp2_path: str,
    clf,
    stream_mask=None,
    confidence_threshold: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> list[tuple]:
    """
    Slide 512×512 tiles over jp2_path and classify each tile with the RF classifier.
    Returns flood_rois: [(polygon_epsg3067, confidence, area_m2), ...]
    """
    effective_mask = _resolve_mask(jp2_path, stream_mask)
    flood_candidates: list[tuple] = []
    tiles_checked = tiles_passed = 0

    with rasterio.open(jp2_path) as src:
        for row_off in range(0, src.height, DETECTION_STRIDE):
            for col_off in range(0, src.width, DETECTION_STRIDE):
                win, win_box = _make_window(src, row_off, col_off)
                tiles_checked += 1
                if not _passes_mask(win_box, effective_mask, src):
                    continue
                tiles_passed += 1

                data = _read_padded(src, win)
                label, confidence = clf_predict(clf, data)
                if label == 1 and confidence >= confidence_threshold:
                    flood_candidates.append((win_box, confidence))

    _print_stats(jp2_path, tiles_checked, tiles_passed, flood_candidates)
    return _merge_candidates(flood_candidates, min_area_m2)


def detect_rois_rf_segmentation(
    jp2_path: str,
    clf,
    stream_mask=None,
    confidence_threshold: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> list[tuple]:
    """
    Dense patch-level RF detection using 64×64 px patches with 512×512 px context.

    Reads the full image once into memory and edge-pads it so every patch can be
    sliced in O(1) without further JP2 seeks — avoids ~150 repeated seeks into the
    compressed tile that made the strip-based approach slow.

    Returns flood_rois: [(polygon_epsg3067, confidence, area_m2), ...]
    """
    from spectral import extract_features

    effective_mask = _resolve_mask(jp2_path, stream_mask)

    with rasterio.open(jp2_path) as src:
        img_h, img_w = src.height, src.width
        img_transform = src.transform
        # prob_map pixels correspond to PATCH_STRIDE × PATCH_STRIDE ground areas
        patch_transform = src.transform * Affine.scale(PATCH_STRIDE)
        print("  Loading image into memory ...")
        full_img = src.read()  # (3, img_h, img_w) uint8 — ~300 MB for a 10 000×10 000 tile

    # Edge-pad so every patch always has a full 512×512 context window.
    # With PATCH_STRIDE=32, _PAD_OFFSET = 512 - 224 = 288 (unchanged from stride=64):
    #   padded_row = pr * PATCH_STRIDE + _PAD_OFFSET
    #   context window: padded[row_start : row_start+512, ...]
    #   patch in context: rows [224:288] — same as before regardless of stride
    _PAD = TILE_SIZE  # 512 px
    _PAD_OFFSET = _PAD - _CTX_OFFSET  # 288
    padded = np.pad(full_img, ((0, 0), (_PAD, _PAD), (_PAD, _PAD)), mode="edge")
    del full_img

    n_patch_rows = (img_h + PATCH_STRIDE - 1) // PATCH_STRIDE
    n_patch_cols = (img_w + PATCH_STRIDE - 1) // PATCH_STRIDE
    prob_map = np.full((n_patch_rows, n_patch_cols), np.nan, dtype=np.float32)
    patches_total = patches_processed = 0

    for pr in range(n_patch_rows):
        batch_feats: list[np.ndarray] = []
        batch_cols: list[int] = []
        row_start = pr * PATCH_STRIDE + _PAD_OFFSET

        for pc in range(n_patch_cols):
            patches_total += 1

            if effective_mask is not None:
                p_h = min(PATCH_SIZE, img_h - pr * PATCH_STRIDE)
                p_w = min(PATCH_SIZE, img_w - pc * PATCH_STRIDE)
                patch_box = box(*rasterio.windows.bounds(
                    Window(pc * PATCH_STRIDE, pr * PATCH_STRIDE, p_w, p_h),
                    img_transform,
                ))
                if not effective_mask.intersects(patch_box):
                    continue

            col_start = pc * PATCH_STRIDE + _PAD_OFFSET
            ctx = padded[:, row_start:row_start + TILE_SIZE, col_start:col_start + TILE_SIZE]
            # ctx: (3, 512, 512) — patch always at [224:288, 224:288] regardless of stride

            batch_feats.append(extract_features(ctx))
            batch_cols.append(pc)

        if batch_feats:
            X = np.array(batch_feats, dtype=np.float32)
            proba = clf.predict_proba(X)[:, 1]
            for pc_idx, pc in enumerate(batch_cols):
                prob_map[pr, pc] = proba[pc_idx]
                patches_processed += 1

    print(f"  Patches checked: {patches_total}, processed (in mask): {patches_processed}")
    return _prob_map_to_rois(prob_map, patch_transform, confidence_threshold, min_area_m2)


def _prob_map_to_rois(
    prob_map: np.ndarray,
    patch_transform,
    threshold: float,
    min_area_m2: float,
) -> list[tuple]:
    """Threshold probability map and polygonize detections."""
    valid = np.where(np.isnan(prob_map), 0.0, prob_map).astype(np.float32)
    binary = (valid >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    rois = []
    for geom_dict, val in rasterio.features.shapes(binary, transform=patch_transform):
        if val != 1:
            continue
        poly = shape(geom_dict)
        if poly.area < min_area_m2:
            continue
        # Mean probability over contributing patches
        mask = rasterio.features.rasterize(
            [(poly, 1)],
            out_shape=prob_map.shape,
            transform=patch_transform,
            dtype=np.uint8,
            all_touched=True,
        )
        probs = valid[mask == 1]
        confidence = float(probs.mean()) if len(probs) > 0 else float(threshold)
        rois.append((poly, confidence, poly.area))

    return rois


def detect_rois_cnn(
    jp2_path: str,
    cnn_model,
    norm_stats: dict | None,
    stream_mask=None,
    confidence_threshold: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> list[tuple]:
    """
    Slide 512×512 tiles over jp2_path and classify in batches with the CNN.
    Returns flood_rois: [(polygon_epsg3067, confidence, area_m2), ...]
    """
    from models.cnn_handler import predict_cnn_batch, BATCH_SIZE

    effective_mask = _resolve_mask(jp2_path, stream_mask)
    pending_chips: list[np.ndarray] = []
    pending_boxes: list = []
    flood_candidates: list[tuple] = []
    tiles_checked = tiles_passed = 0

    def _flush_batch():
        if not pending_chips:
            return
        results = predict_cnn_batch(cnn_model, pending_chips, norm_stats)
        for (label, confidence), win_box in zip(results, pending_boxes):
            if label == 1 and confidence >= confidence_threshold:
                flood_candidates.append((win_box, confidence))
        pending_chips.clear()
        pending_boxes.clear()

    with rasterio.open(jp2_path) as src:
        for row_off in range(0, src.height, DETECTION_STRIDE):
            for col_off in range(0, src.width, DETECTION_STRIDE):
                win, win_box = _make_window(src, row_off, col_off)
                tiles_checked += 1
                if not _passes_mask(win_box, effective_mask, src):
                    continue
                tiles_passed += 1

                pending_chips.append(_read_padded(src, win))
                pending_boxes.append(win_box)

                if len(pending_chips) >= BATCH_SIZE:
                    _flush_batch()

    _flush_batch()
    _print_stats(jp2_path, tiles_checked, tiles_passed, flood_candidates)
    return _merge_candidates(flood_candidates, min_area_m2)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_mask(jp2_path: str, stream_mask):
    """Return None if the mask doesn't overlap this raster (fall back to full scan)."""
    if stream_mask is None:
        return None
    with rasterio.open(jp2_path) as src:
        b = src.bounds
        raster_box = box(b.left, b.bottom, b.right, b.top)
    if not stream_mask.intersects(raster_box):
        print("  No hydrography coverage for this image — scanning full raster")
        return None
    return stream_mask


def _make_window(src, row_off: int, col_off: int):
    win = Window(
        col_off=col_off, row_off=row_off,
        width=min(TILE_SIZE, src.width - col_off),
        height=min(TILE_SIZE, src.height - row_off),
    )
    win_box = box(*rasterio.windows.bounds(win, src.transform))
    return win, win_box


def _passes_mask(win_box, effective_mask, src) -> bool:
    if effective_mask is None:
        return True
    inner_box = win_box.buffer(-(TILE_SIZE * src.transform.a / 4))
    return not inner_box.is_empty and effective_mask.intersects(inner_box)


def _read_padded(src, win: Window) -> np.ndarray:
    data = src.read(window=win)
    if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
        padded = np.zeros((data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype)
        padded[:, : data.shape[1], : data.shape[2]] = data
        data = padded
    return data


def _print_stats(jp2_path, tiles_checked, tiles_passed, flood_candidates):
    print(f"  Tiles checked: {tiles_checked}, passed mask: {tiles_passed}, "
          f"flood candidates: {len(flood_candidates)}")


def _merge_candidates(candidates: list[tuple], min_area_m2: float) -> list[tuple]:
    if not candidates:
        return []
    polys = [p for p, _ in candidates]
    confs = [c for _, c in candidates]
    merged = unary_union(polys)
    geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    rois = []
    for geom in geoms:
        if geom.area < min_area_m2:
            continue
        contributing = [confs[i] for i, p in enumerate(polys) if geom.intersects(p)]
        confidence = float(np.mean(contributing)) if contributing else 0.0
        rois.append((geom, confidence, geom.area))
    return rois
