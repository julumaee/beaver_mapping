"""Training data pipeline: KML label parsing, chip extraction, negative sampling."""

import csv
import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

from ingestion import TILE_SIZE

_KML_NS = "http://www.opengis.net/kml/2.2"
_WGS84_TO_ETRS = Transformer.from_crs(4326, 3067, always_xy=True)

# Feature types excluded from training by default.
DEFAULT_EXCLUDE: frozenset[str] = frozenset({"lodge", "dam"})

# Maps KML feature type names to integer class labels:
#   0 = negative (no beaver activity)
#   1 = flooded area / wet forest (beaver-influenced water)
# Dam points are excluded from training (see DEFAULT_EXCLUDE).
FEATURE_TO_LABEL: dict[str, int] = {
    "negative":     0,
    "wet_forest":   1,
    "beaver_flood": 1,
    "unknown":      1,
}


def parse_kml_labels(kml_path: str) -> list[tuple[Point, str]]:
    """
    Parse a KML or KMZ file and return (centroid_epsg3067, feature_type) pairs.

    feature_type is taken from the Placemark <name> tag, lowercased and stripped.
    Placemarks with no name get type "unknown".
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
        feature_type = (
            name_el.text.strip().lower()
            if (name_el is not None and name_el.text)
            else "unknown"
        )
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
    manifest_rows: list[dict] | None = None,
) -> list[str]:
    """
    For each (point, feature_type) extract a TILE_SIZE×TILE_SIZE chip centred
    on that point and save as a .npy file.  The class label (0/1/2) is derived
    from FEATURE_TO_LABEL[feature_type].

    Returns a list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _TAG = {0: "neg", 1: "dam", 2: "flood"}
    written: list[str] = []
    half = TILE_SIZE // 2

    with rasterio.open(jp2_path) as src:
        for i, (pt, feature_type) in enumerate(labeled_points):
            label = FEATURE_TO_LABEL.get(feature_type, 2)

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

            pad_col = max(-col_off, 0)
            pad_row = max(-row_off, 0)
            win_col = max(col_off, 0)
            win_row = max(row_off, 0)
            win_w = min(TILE_SIZE - pad_col, src.width - win_col)
            win_h = min(TILE_SIZE - pad_row, src.height - win_row)

            data = src.read(window=Window(win_col, win_row, win_w, win_h))

            if pad_col > 0 or pad_row > 0 or win_w < TILE_SIZE or win_h < TILE_SIZE:
                padded = np.zeros((data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype)
                padded[:, pad_row:pad_row + win_h, pad_col:pad_col + win_w] = data
                data = padded

            stem = Path(jp2_path).stem
            fname = f"{stem}_{_TAG.get(label, 'pos')}_{i:04d}.npy"
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
    min_pos_distance: float = 200.0,
    min_neg_spacing: float = 100.0,
    imagery_extent=None,
) -> list[Point]:
    """
    Draw n random points from stream_mask, excluding areas near positive labels
    and avoiding tight clustering of negatives.

    imagery_extent: optional Shapely geometry; sampling is restricted to its
                    bounding box and only points inside it are accepted.
    min_pos_distance: reject candidates within this many metres of any positive.
    min_neg_spacing:  reject candidates within this many metres of an already
                      accepted negative (prevents spatial clustering).
    """
    if stream_mask is None and imagery_extent is None:
        return []

    if imagery_extent is not None:
        minx, miny, maxx, maxy = imagery_extent.bounds
    else:
        minx, miny, maxx, maxy = stream_mask.bounds

    rng = random.Random(rng_seed)
    samples: list[Point] = []

    for _ in range(n * 500):
        if len(samples) >= n:
            break
        pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if imagery_extent is not None and not imagery_extent.contains(pt):
            continue
        if stream_mask is not None and not stream_mask.intersects(pt):
            continue
        if any(pt.distance(pos) < min_pos_distance for pos in positive_points):
            continue
        if any(pt.distance(neg) < min_neg_spacing for neg in samples):
            continue
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
    Orchestrate the full training data pipeline and write a manifest CSV.

    Labels: 0 = negative, 1 = dam, 2 = flooded area (wet_forest / beaver_flood).
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

    imagery_extent = _imagery_union(jp2_paths)
    negative_points = sample_negatives(
        stream_mask, positive_points, n_neg, rng_seed,
        imagery_extent=imagery_extent,
    )
    negative_labeled = [(pt, "negative") for pt in negative_points]

    manifest_rows: list[dict] = []
    for jp2_path in jp2_paths:
        extract_chips(jp2_path, positive_labeled, out_dir, manifest_rows=manifest_rows)
        extract_chips(jp2_path, negative_labeled, out_dir, manifest_rows=manifest_rows)

    manifest_path = out_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return str(manifest_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _imagery_union(jp2_paths: list[str]):
    from shapely.geometry import box as _box
    boxes = []
    for path in jp2_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            boxes.append(_box(b.left, b.bottom, b.right, b.top))
    return unary_union(boxes)


def _read_kml_text(path: str) -> str:
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
    tag = lambda name: f"{{{ns}}}{name}" if ns else name  # noqa: E731
    for geom_tag in ("Point", "Polygon", "LineString", "MultiGeometry"):
        el = placemark_el.find(f".//{tag(geom_tag)}")
        if el is not None:
            coords_el = el.find(f".//{tag('coordinates')}")
            if coords_el is not None and coords_el.text:
                return _parse_coord_string(coords_el.text)
    return []


def _parse_coord_string(text: str) -> list[tuple[float, float]]:
    pairs = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pairs
