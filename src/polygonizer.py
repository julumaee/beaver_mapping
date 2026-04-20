"""Sliding-window detection and ROI polygonization for beaver flood areas."""

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union

from ingestion import TILE_SIZE
from models.random_forest import predict as clf_predict

MIN_AREA_M2 = 500
DETECTION_STRIDE = TILE_SIZE // 2  # 50% overlap


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
