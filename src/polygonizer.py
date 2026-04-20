"""Sliding-window detection and ROI polygonization for beaver flood areas."""

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union

from ingestion import TILE_SIZE
from classifier import predict as clf_predict

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
    Slide 512×512 tiles over jp2_path and classify each tile.
    Label 1 = flooded/wet-forest area; label 0 = negative.

    Returns flood_rois: [(polygon_epsg3067, confidence, area_m2), ...]

    If the stream mask doesn't cover this raster, the full image is scanned.
    """
    with rasterio.open(jp2_path) as src:
        b = src.bounds
        raster_box = box(b.left, b.bottom, b.right, b.top)

    if stream_mask is not None and not stream_mask.intersects(raster_box):
        print("  No hydrography coverage for this image — scanning full raster")
        effective_mask = None
    else:
        effective_mask = stream_mask

    flood_candidates: list[tuple] = []
    tiles_checked = tiles_passed = 0

    with rasterio.open(jp2_path) as src:
        for row_off in range(0, src.height, DETECTION_STRIDE):
            for col_off in range(0, src.width, DETECTION_STRIDE):
                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(TILE_SIZE, src.width - col_off),
                    height=min(TILE_SIZE, src.height - row_off),
                )
                win_bounds = rasterio.windows.bounds(win, src.transform)
                win_box = box(*win_bounds)
                tiles_checked += 1

                if effective_mask is not None:
                    inner_box = win_box.buffer(-(TILE_SIZE * src.transform.a / 4))
                    if inner_box.is_empty or not effective_mask.intersects(inner_box):
                        continue
                tiles_passed += 1

                data = src.read(window=win)
                if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                    padded = np.zeros(
                        (data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype
                    )
                    padded[:, :data.shape[1], :data.shape[2]] = data
                    data = padded

                label, confidence = clf_predict(clf, data)
                if label == 1 and confidence >= confidence_threshold:
                    flood_candidates.append((win_box, confidence))

    print(f"  Tiles checked: {tiles_checked}, passed mask: {tiles_passed}, "
          f"flood candidates: {len(flood_candidates)}")

    return _merge_candidates(flood_candidates, min_area_m2)


def _merge_candidates(candidates: list[tuple], min_area_m2: float) -> list[tuple]:
    """Merge overlapping positive tile boxes into ROI polygons."""
    if not candidates:
        return []

    polys = [p for p, _ in candidates]
    confs = [c for _, c in candidates]

    merged = unary_union(polys)
    geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    rois = []
    for geom in geoms:
        area = geom.area
        if area < min_area_m2:
            continue
        contributing = [confs[i] for i, p in enumerate(polys) if geom.intersects(p)]
        confidence = float(np.mean(contributing)) if contributing else 0.0
        rois.append((geom, confidence, area))

    return rois
