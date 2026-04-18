"""Sliding-window detection and ROI polygonization."""

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union

from ingestion import TILE_SIZE, read_metadata
from classifier import predict as clf_predict

MIN_AREA_M2 = 500


def detect_rois(
    jp2_path: str,
    clf,
    stream_mask=None,
    confidence_threshold: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> list[tuple]:
    """
    Slide 512×512 tiles over jp2_path (restricted to stream_mask if provided),
    classify each, merge positive tiles, and return ROIs.

    Returns list of (polygon_epsg3067, confidence, area_m2).
    If stream_mask is None, the full raster extent is used.
    """
    if stream_mask is None:
        meta = read_metadata(jp2_path)
        b = meta["bounds"]
        stream_mask = box(b.left, b.bottom, b.right, b.top)

    candidates: list[tuple] = []

    with rasterio.open(jp2_path) as src:
        for row_off in range(0, src.height, TILE_SIZE):
            for col_off in range(0, src.width, TILE_SIZE):
                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(TILE_SIZE, src.width - col_off),
                    height=min(TILE_SIZE, src.height - row_off),
                )
                win_bounds = rasterio.windows.bounds(win, src.transform)
                win_box = box(*win_bounds)

                if not stream_mask.intersects(win_box):
                    continue

                data = src.read(window=win)
                if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                    padded = np.zeros(
                        (data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype
                    )
                    padded[:, : data.shape[1], : data.shape[2]] = data
                    data = padded

                label, confidence = clf_predict(clf, data)
                if label == 1 and confidence >= confidence_threshold:
                    candidates.append((win_box, confidence))

    return _merge_candidates(candidates, min_area_m2)


def _merge_candidates(
    candidates: list[tuple], min_area_m2: float
) -> list[tuple]:
    """Merge overlapping/adjacent positive tile boxes into ROI polygons."""
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
