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
NEGATIVE_EXCLUSION_RADIUS = 300

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

            win = Window(
                col_off=max(col_off, 0),
                row_off=max(row_off, 0),
                width=min(TILE_SIZE, src.width - max(col_off, 0)),
                height=min(TILE_SIZE, src.height - max(row_off, 0)),
            )

            data = src.read(window=win)

            if data.shape[1] != TILE_SIZE or data.shape[2] != TILE_SIZE:
                padded = np.zeros(
                    (data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype
                )
                pad_row = max(-row_off, 0)
                pad_col = max(-col_off, 0)
                padded[
                    :,
                    pad_row : pad_row + data.shape[1],
                    pad_col : pad_col + data.shape[2],
                ] = data
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
) -> list[Point]:
    """
    Draw n random points from inside stream_mask that are at least
    NEGATIVE_EXCLUSION_RADIUS metres away from every positive point.

    stream_mask must be a Shapely geometry in EPSG:3067.
    Returns at most n points (may be fewer if the mask area is small).
    """
    if positive_points:
        exclusion = unary_union(
            [p.buffer(NEGATIVE_EXCLUSION_RADIUS) for p in positive_points]
        )
        candidate_area = stream_mask.difference(exclusion)
    else:
        candidate_area = stream_mask

    if candidate_area.is_empty:
        return []

    minx, miny, maxx, maxy = candidate_area.bounds
    rng = random.Random(rng_seed)
    samples: list[Point] = []

    for _ in range(n * 200):
        if len(samples) >= n:
            break
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if candidate_area.contains(pt):
            samples.append(pt)

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
    negative_points = sample_negatives(stream_mask, positive_points, n_neg, rng_seed)
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
