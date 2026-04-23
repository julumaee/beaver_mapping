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
DETECTION_STRIDE = TILE_SIZE // 2  # 50% overlap

# Dense patch-level RF segmentation constants
PATCH_SIZE = 64  # pixels — 32 m at 0.5 m/px; matches FEATURE_REGION in spectral.py
_CTX_OFFSET = (TILE_SIZE - PATCH_SIZE) // 2  # 224 — distance from context top/left to patch top/left


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

    Slides a 64-pixel patch grid over the image.  For each patch the surrounding
    512×512 context is fed to extract_features() — identical to the chip features
    used during training — so no retraining is required.  A probability map is built
    then polygonized with rasterio.features.shapes.

    Returns flood_rois: [(polygon_epsg3067, confidence, area_m2), ...]
    """
    from spectral import extract_features

    effective_mask = _resolve_mask(jp2_path, stream_mask)

    with rasterio.open(jp2_path) as src:
        img_h, img_w = src.height, src.width
        n_patch_rows = (img_h + PATCH_SIZE - 1) // PATCH_SIZE
        n_patch_cols = (img_w + PATCH_SIZE - 1) // PATCH_SIZE

        prob_map = np.full((n_patch_rows, n_patch_cols), np.nan, dtype=np.float32)
        patch_transform = src.transform * Affine.scale(PATCH_SIZE)

        patches_total = patches_processed = 0

        for pr in range(n_patch_rows):
            # Read a full-width strip covering the 512 px context for this patch row.
            ctx_row_start = pr * PATCH_SIZE - _CTX_OFFSET  # image coords, may be < 0
            read_row_start = max(0, ctx_row_start)
            read_row_end = min(img_h, ctx_row_start + TILE_SIZE)
            read_height = read_row_end - read_row_start

            strip = src.read(window=Window(0, read_row_start, img_w, read_height))
            # Pad vertically to TILE_SIZE
            pad_top = read_row_start - ctx_row_start
            pad_bottom = TILE_SIZE - pad_top - read_height
            strip = np.pad(strip, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="edge")
            # strip: (3, TILE_SIZE, img_w)

            batch_feats: list[np.ndarray] = []
            batch_cols: list[int] = []

            for pc in range(n_patch_cols):
                patches_total += 1

                if effective_mask is not None:
                    p_h = min(PATCH_SIZE, img_h - pr * PATCH_SIZE)
                    p_w = min(PATCH_SIZE, img_w - pc * PATCH_SIZE)
                    patch_box = box(*rasterio.windows.bounds(
                        Window(pc * PATCH_SIZE, pr * PATCH_SIZE, p_w, p_h),
                        src.transform,
                    ))
                    if not effective_mask.intersects(patch_box):
                        continue

                # Horizontal slice of the already-loaded strip
                ctx_col_start = pc * PATCH_SIZE - _CTX_OFFSET  # image coords, may be < 0
                read_col_start = max(0, ctx_col_start)
                read_col_end = min(img_w, ctx_col_start + TILE_SIZE)

                col_slice = strip[:, :, read_col_start:read_col_end]
                pad_left = read_col_start - ctx_col_start
                pad_right = TILE_SIZE - pad_left - (read_col_end - read_col_start)
                ctx = np.pad(col_slice, ((0, 0), (0, 0), (pad_left, pad_right)), mode="edge")
                # ctx: (3, TILE_SIZE, TILE_SIZE) — patch is at ctx[..., 224:288, 224:288]

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
