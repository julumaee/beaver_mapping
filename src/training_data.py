"""Training data pipeline: KML label parsing, chip extraction, negative sampling."""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import csv
import random

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

from ingestion import TILE_SIZE

# Minimum separation (metres) between a negative sample and any positive point.
NEGATIVE_EXCLUSION_RADIUS = 100

_KML_NS = "http://www.opengis.net/kml/2.2"

_WGS84_TO_ETRS = Transformer.from_crs(4326, 3067, always_xy=True)

# Feature types excluded from training by default.
# wet_forest and beaver_flood are treated as the same positive class (label=1)
# because they are spectrally indistinguishable at 512×512 chip resolution.
DEFAULT_EXCLUDE: frozenset[str] = frozenset({"lodge"})


def parse_kml_labels(kml_path: str) -> list[tuple[Point, str]]:
    """
    Parse a KML or KMZ file and return (centroid_epsg3067, feature_type) pairs.

    feature_type is taken from the Placemark <name> tag, lowercased and stripped.
    Placemarks with no name get type "unknown".
    Geometry centroid is used for Polygons and LineStrings.
    """
    kml_text = _read_kml_text(kml_path)
    root = ET.fromstring(kml_text)

    ns = _KML_NS if root.tag.startswith("{") else ""

    results: list[tuple[Point, str]] = []
    for pm in root.iter(f"{{{ns}}}Placemark" if ns else "Placemark"):
        raw = _extract_coords(pm, ns)
        if not raw:
            continue

        name_el = pm.find(f"{{{ns}}}name" if ns else "name")
        feature_type = name_el.text.strip().lower() if (name_el is not None and name_el.text) else "unknown"

        if len(raw) == 1:
            lon, lat = raw[0]
        else:
            mp = MultiPoint(raw)
            lon, lat = mp.centroid.x, mp.centroid.y

        x, y = _WGS84_TO_ETRS.transform(lon, lat)
        results.append((Point(x, y), feature_type))

    return results


def extract_chips(
    jp2_path: str,
    labeled_points: list[tuple[Point, str]],
    out_dir: str,
    label: int,
    manifest_rows: list[dict] | None = None,
) -> list[str]:
    """
    For each (point, feature_type) extract a TILE_SIZE×TILE_SIZE chip centred
    on that point from jp2_path and save it as a .npy file under out_dir.

    Returns a list of written file paths.  If manifest_rows is provided,
    appends one dict per chip (path, label, feature_type, x, y) to it.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    half = TILE_SIZE // 2

    with rasterio.open(jp2_path) as src:
        for i, (pt, feature_type) in enumerate(labeled_points):
            col, row = ~src.transform * (pt.x, pt.y)
            col, row = int(col), int(row)

            col_off = col - half
            row_off = row - half

            if (
                col_off + TILE_SIZE <= 0
                or row_off + TILE_SIZE <= 0
                or col_off >= src.width
                or row_off >= src.height
            ):
                continue

            # Offset into the padded output where raster data begins.
            pad_col = max(-col_off, 0)
            pad_row = max(-row_off, 0)

            # Raster read window — width/height reduced by the pad so the data
            # always fits exactly into the padded output array.
            win_col = max(col_off, 0)
            win_row = max(row_off, 0)
            win_w = min(TILE_SIZE - pad_col, src.width - win_col)
            win_h = min(TILE_SIZE - pad_row, src.height - win_row)

            win = Window(col_off=win_col, row_off=win_row, width=win_w, height=win_h)
            data = src.read(window=win)

            if pad_col > 0 or pad_row > 0 or win_w < TILE_SIZE or win_h < TILE_SIZE:
                padded = np.zeros(
                    (data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype
                )
                padded[:, pad_row : pad_row + win_h, pad_col : pad_col + win_w] = data
                data = padded

            stem = Path(jp2_path).stem
            tag = "pos" if label == 1 else "neg"
            fname = f"{stem}_{tag}_{i:04d}.npy"
            fpath = out_path / fname
            np.save(str(fpath), data)
            written.append(str(fpath))

            if manifest_rows is not None:
                manifest_rows.append({
                    "path": str(fpath),
                    "label": label,
                    "feature_type": feature_type,
                    "x": pt.x,
                    "y": pt.y,
                })

    return written


def sample_negatives(
    stream_mask,
    positive_points: list[Point],
    n: int,
    rng_seed: int = 42,
    fallback_area=None,
) -> list[Point]:
    """
    Draw n random points that are at least NEGATIVE_EXCLUSION_RADIUS metres
    from every positive point, sampling first from stream_mask then falling
    back to fallback_area (e.g. the imagery bounds) if not enough are found.

    stream_mask must be a Shapely geometry in EPSG:3067.
    """
    exclusion = (
        unary_union([p.buffer(NEGATIVE_EXCLUSION_RADIUS) for p in positive_points])
        if positive_points else None
    )

    def _sample_from(area, needed, rng) -> list[Point]:
        candidate = area.difference(exclusion) if exclusion else area
        if candidate.is_empty:
            return []
        minx, miny, maxx, maxy = candidate.bounds
        found = []
        for _ in range(needed * 500):
            if len(found) >= needed:
                break
            pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if candidate.contains(pt):
                found.append(pt)
        return found

    rng = random.Random(rng_seed)
    samples = _sample_from(stream_mask, n, rng)

    if len(samples) < n and fallback_area is not None:
        remaining = n - len(samples)
        samples += _sample_from(fallback_area, remaining, rng)

    return samples


def build_training_dataset(
    jp2_paths: list[str],
    kml_paths: list[str],
    stream_mask,
    out_dir: str,
    n_negatives: int | None = None,
    rng_seed: int = 42,
    exclude_features: frozenset[str] = DEFAULT_EXCLUDE,
) -> str:
    """
    Orchestrate the full training data pipeline:
      1. Parse all KML labels → (point, feature_type) pairs, filtering excluded types
      2. Sample equal-count (or n_negatives) negative points from stream_mask
      3. Extract chips from each jp2 for both positive and negative points
      4. Write a manifest CSV (columns: path, label, feature_type, x, y) and return its path

    wet_forest and beaver_flood are both treated as label=1 — they are spectrally
    indistinguishable at chip resolution and combining them increases sample count.
    lodge is excluded by default (too small to produce a usable chip-level signal).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    positive_labeled: list[tuple[Point, str]] = []
    for kml_path in kml_paths:
        for pt, ftype in parse_kml_labels(kml_path):
            if ftype not in exclude_features:
                positive_labeled.append((pt, ftype))

    positive_points = [pt for pt, _ in positive_labeled]
    n_neg = n_negatives if n_negatives is not None else len(positive_labeled)

    # Build a fallback area from imagery bounds so negative sampling always
    # has somewhere to draw from if the stream mask is too densely covered
    # by positive exclusion zones.
    imagery_bounds = _imagery_union(jp2_paths)
    negative_points = sample_negatives(
        stream_mask, positive_points, n_neg, rng_seed, fallback_area=imagery_bounds
    )
    negative_labeled = [(pt, "negative") for pt in negative_points]

    manifest_rows: list[dict] = []
    for jp2_path in jp2_paths:
        extract_chips(jp2_path, positive_labeled, out_dir, label=1, manifest_rows=manifest_rows)
        extract_chips(jp2_path, negative_labeled, out_dir, label=0, manifest_rows=manifest_rows)

    manifest_path = out_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return str(manifest_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_kml_text(path: str) -> str:
    """Return KML text from a .kml or .kmz file."""
    p = Path(path)
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError(f"No .kml file found inside {path}")
            with zf.open(kml_names[0]) as f:
                return f.read().decode("utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_coords(placemark_el, ns: str) -> list[tuple[float, float]]:
    """Return a list of (lon, lat) pairs from the first geometry in a Placemark."""
    tag = lambda name: f"{{{ns}}}{name}" if ns else name  # noqa: E731

    for geom_tag in ("Point", "Polygon", "LineString", "MultiGeometry"):
        el = placemark_el.find(f".//{tag(geom_tag)}")
        if el is not None:
            coords_el = el.find(f".//{tag('coordinates')}")
            if coords_el is not None and coords_el.text:
                return _parse_coord_string(coords_el.text)
    return []


def _imagery_union(jp2_paths: list[str]):
    """Return a Shapely geometry covering the union of all jp2 raster extents."""
    from shapely.geometry import box as shapely_box
    boxes = []
    for path in jp2_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            boxes.append(shapely_box(b.left, b.bottom, b.right, b.top))
    return unary_union(boxes)


def _parse_coord_string(text: str) -> list[tuple[float, float]]:
    """Parse a KML coordinates string into (lon, lat) tuples (elevation dropped)."""
    pairs = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pairs
