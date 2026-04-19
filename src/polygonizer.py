"""Sliding-window detection, ROI polygonization, and dam line extraction."""

import math

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points, unary_union

from ingestion import TILE_SIZE, read_metadata
from classifier import predict as clf_predict

MIN_AREA_M2 = 500
DETECTION_STRIDE = TILE_SIZE // 2  # 50% overlap
DAM_LINE_LENGTH = 60.0             # metres — typical beaver dam width


def detect_rois(
    jp2_path: str,
    clf,
    stream_mask=None,
    stream_lines=None,
    confidence_threshold: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> tuple[list[tuple], list[tuple]]:
    """
    Slide 512×512 tiles over jp2_path and classify each tile as:
      0 = negative, 1 = dam, 2 = flooded area

    Returns (dam_lines, flood_rois) where:
      dam_lines  = [(linestring_epsg3067, confidence), ...]
      flood_rois = [(polygon_epsg3067, confidence, area_m2), ...]

    If stream_mask is None the full raster is scanned.
    stream_lines (virtavesikapea geometry) is used to orient dam lines.
    """
    if stream_mask is None:
        meta = read_metadata(jp2_path)
        b = meta["bounds"]
        stream_mask = box(b.left, b.bottom, b.right, b.top)

    dam_candidates: list[tuple] = []
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

                # Require the inner half of the tile to intersect the stream
                # mask — looser than centroid-only but tighter than any-intersect.
                inner_box = win_box.buffer(-(TILE_SIZE * src.transform.a / 4))
                if inner_box.is_empty or not stream_mask.intersects(inner_box):
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
                if confidence < confidence_threshold:
                    continue

                if label == 1:
                    dam_candidates.append((win_box, confidence))
                elif label == 2:
                    flood_candidates.append((win_box, confidence))

    print(f"  Tiles checked: {tiles_checked}, passed mask: {tiles_passed}, "
          f"dam candidates: {len(dam_candidates)}, flood candidates: {len(flood_candidates)}")

    dam_lines = _build_dam_lines(dam_candidates, stream_lines)
    flood_rois = _merge_candidates(flood_candidates, min_area_m2)
    return dam_lines, flood_rois


def _build_dam_lines(candidates: list[tuple], stream_lines) -> list[tuple]:
    """Merge adjacent dam tiles and compute a line perpendicular to the stream."""
    if not candidates:
        return []

    polys = [p for p, _ in candidates]
    confs = [c for _, c in candidates]

    merged = unary_union(polys)
    geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    lines = []
    for geom in geoms:
        center = geom.centroid
        contributing = [confs[i] for i, p in enumerate(polys) if geom.intersects(p)]
        confidence = float(np.mean(contributing))
        dam_line = _dam_line(center, stream_lines)
        lines.append((dam_line, confidence))

    return lines


def _dam_line(center: Point, stream_lines, length: float = DAM_LINE_LENGTH) -> LineString:
    """
    Return a LineString of given length through center, oriented perpendicular
    to the nearest stream segment.  Falls back to east-west if no stream data.
    """
    if stream_lines is None or stream_lines.is_empty:
        half = length / 2
        return LineString([(center.x - half, center.y), (center.x + half, center.y)])

    # Find which sub-line is nearest
    if hasattr(stream_lines, "geoms"):
        nearest_line = min(stream_lines.geoms, key=lambda g: center.distance(g))
    else:
        nearest_line = stream_lines

    # Get stream direction at the nearest point using project/interpolate
    frac = nearest_line.project(center)
    step = min(5.0, nearest_line.length / 2)
    p1 = nearest_line.interpolate(max(0.0, frac - step))
    p2 = nearest_line.interpolate(min(nearest_line.length, frac + step))

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 0.1:
        perp_dx, perp_dy = 1.0, 0.0
    else:
        # Stream direction: (dx/dist, dy/dist)
        # Perpendicular (dam direction): (-dy/dist, dx/dist)
        perp_dx = -dy / dist
        perp_dy = dx / dist

    half = length / 2
    return LineString([
        (center.x - perp_dx * half, center.y - perp_dy * half),
        (center.x + perp_dx * half, center.y + perp_dy * half),
    ])


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
