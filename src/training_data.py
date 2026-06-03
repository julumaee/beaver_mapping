"""Training data pipeline: KML label parsing, chip extraction, negative sampling.

Generic-label training workflow
--------------------------------
Labels can be placed anywhere in the imagery — not just confirmed beaver
territories.  The stream filter (--hydro) is applied at detection time, not
training time, so the model learns visual patterns and domain knowledge is
applied separately.

Recommended label types:
  dead_forest  — standing dead trees (killed by beaver flooding)
  flood        — open water impoundment (any scale)
  wet_forest   — saturated/flooded forest (older, kept for compatibility)
  beaver_flood — confirmed beaver open water (kept for compatibility)
  negative     — explicit hard negative (stream-adjacent, no beaver activity)

Types excluded from training (point-scale features, not area classifiers):
  dam, lodge
"""

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
#   1 = positive (any beaver-associated visual signature)
# Unrecognised label names default to 1 so new types work without code changes.
FEATURE_TO_LABEL: dict[str, int] = {
    "negative":     0,
    # Generic labels — can be placed anywhere in imagery, not just beaver sites:
    "dead_forest":  1,  # standing dead trees killed by beaver flooding
    "flood":        1,  # any open water impoundment
    # Legacy / specific labels kept for backwards compatibility:
    "wet_forest":   1,
    "beaver_flood": 1,
    "unknown":      1,
}


def parse_kml_labels(kml_path: str) -> list[tuple[Point, str]]:
    """
    Parse a KML or KMZ file and return (centroid_epsg3067, feature_type) pairs.

    Feature type resolution (in priority order):
    1. Enclosing <Folder> name — Google Earth folder structure is the primary
       way to categorise placemarks (e.g. a folder named "Dead Forest" gives
       feature type "dead_forest" to all placemarks inside it).
    2. Placemark <name> tag — used only for root-level placemarks not inside
       any named folder.
    3. "unknown" — fallback when neither is present (treated as class 1).

    Label names are normalised: lowercased, leading/trailing whitespace removed,
    spaces and hyphens replaced with underscores (so "Dead Forest" → "dead_forest").
    """
    kml_text = _read_kml_text(kml_path)
    root = ET.fromstring(kml_text)
    ns = _KML_NS if root.tag.startswith("{") else ""

    results: list[tuple[Point, str]] = []
    doc = root.find(f"{{{ns}}}Document" if ns else "Document")
    _parse_kml_element(doc if doc is not None else root, ns, None, results)
    return results


def _normalise_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_kml_element(
    el: ET.Element,
    ns: str,
    folder_type: str | None,
    results: list[tuple[Point, str]],
) -> None:
    """Recursively walk KML elements, propagating the enclosing folder name."""
    tag = lambda name: f"{{{ns}}}{name}" if ns else name  # noqa: E731
    for child in el:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "Folder":
            name_el = child.find(tag("name"))
            this_folder = (
                _normalise_label(name_el.text)
                if (name_el is not None and name_el.text)
                else folder_type
            )
            _parse_kml_element(child, ns, this_folder, results)
        elif local == "Placemark":
            if folder_type is not None:
                ftype = folder_type
            else:
                name_el = child.find(tag("name"))
                ftype = (
                    _normalise_label(name_el.text)
                    if (name_el is not None and name_el.text)
                    else "unknown"
                )
            raw = _extract_coords(child, ns)
            if not raw:
                continue
            if len(raw) == 1:
                lon, lat = raw[0]
            else:
                mp = MultiPoint(raw)
                lon, lat = mp.centroid.x, mp.centroid.y
            x, y = _WGS84_TO_ETRS.transform(lon, lat)
            results.append((Point(x, y), ftype))


def extract_chips(
    jp2_path: str,
    labeled_points: list[tuple[Point, str]],
    out_dir: str,
    manifest_rows: list[dict] | None = None,
    augment_positives: int = 6,
    augment_max_offset: int = 24,
    rng_seed: int = 42,
) -> list[str]:
    """
    For each (point, feature_type) extract a TILE_SIZE×TILE_SIZE chip centred
    on that point and save as a .npy file.  The class label (0/1) is derived
    from FEATURE_TO_LABEL[feature_type].

    Positive chips (label > 0) are augmented with augment_positives additional
    chips extracted at random pixel offsets (±augment_max_offset). This simulates
    the detection grid misalignment and multiplies positive training samples without
    requiring new labels. x/y in the manifest stays at the original label point so
    spatial CV groups augmented chips with their source territory.

    Returns a list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _TAG = {0: "neg", 1: "pos"}
    written: list[str] = []
    half = TILE_SIZE // 2
    rng = random.Random(rng_seed)

    with rasterio.open(jp2_path) as src:
        for i, (pt, feature_type) in enumerate(labeled_points):
            label = FEATURE_TO_LABEL.get(feature_type, 1)

            col, row = ~src.transform * (pt.x, pt.y)
            col, row = int(col), int(row)

            # Build list of (col_offset, row_offset, aug_index) to extract.
            # Index -1 = original (no offset); 0..N-1 = augmented.
            offsets: list[tuple[int, int, int]] = [(0, 0, -1)]
            if label > 0 and augment_positives > 0:
                for aug_i in range(augment_positives):
                    dc = rng.randint(-augment_max_offset, augment_max_offset)
                    dr = rng.randint(-augment_max_offset, augment_max_offset)
                    offsets.append((dc, dr, aug_i))

            for dc, dr, aug_idx in offsets:
                col_off = col + dc - half
                row_off = row + dr - half

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
                    full = np.zeros((data.shape[0], TILE_SIZE, TILE_SIZE), dtype=data.dtype)
                    full[:, pad_row:pad_row + win_h, pad_col:pad_col + win_w] = data
                    if pad_row > 0:
                        full[:, :pad_row, :] = full[:, pad_row:pad_row + 1, :]
                    if pad_col > 0:
                        full[:, :, :pad_col] = full[:, :, pad_col:pad_col + 1]
                    bottom = pad_row + win_h
                    right = pad_col + win_w
                    if bottom < TILE_SIZE:
                        full[:, bottom:, :] = full[:, bottom - 1:bottom, :]
                    if right < TILE_SIZE:
                        full[:, :, right:] = full[:, :, right - 1:right]
                    data = full

                stem = Path(jp2_path).stem
                aug_suffix = "" if aug_idx < 0 else f"_aug{aug_idx}"
                fname = f"{stem}_{_TAG.get(label, 'pos')}_{i:04d}{aug_suffix}.npy"
                fpath = out_path / fname
                np.save(str(fpath), data)
                written.append(str(fpath))

                if manifest_rows is not None:
                    manifest_rows.append({
                        "path": str(fpath),
                        "label": label,
                        "feature_type": feature_type,
                        # Always store original label coordinates so spatial CV
                        # groups augmented chips with their source territory.
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
    stream_mask=None,
    out_dir: str = "chips",
    n_negatives: int | None = None,
    rng_seed: int = 42,
    exclude_features: frozenset[str] = DEFAULT_EXCLUDE,
    augment_positives: int = 6,
    augment_max_offset: int = 24,
    hydro_path: str | None = None,
) -> str:
    """
    Orchestrate the full training data pipeline and write a manifest CSV.

    hydro_path: when given, negative samples are drawn using a per-tile stream
                mask (same bbox-scoped loading as detect).  Avoids loading
                millions of features globally when imagery spans many tiles.
    stream_mask: legacy — passed directly to sample_negatives.  Ignored when
                 hydro_path is set.
    augment_positives: number of extra offset chips per positive label point.
    augment_max_offset: maximum pixel shift in each direction for augmentation.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_labeled: list[tuple[Point, str]] = []
    for kml_path in kml_paths:
        for pt, ftype in parse_kml_labels(kml_path):
            if ftype not in exclude_features:
                all_labeled.append((pt, ftype))

    # Count only true positives for auto-negative balance — hard negatives
    # already in all_labeled must not inflate the auto-sample count.
    n_true_pos = sum(1 for _, ftype in all_labeled if FEATURE_TO_LABEL.get(ftype, 1) == 1)
    positive_points = [pt for pt, _ in all_labeled]
    n_neg = n_negatives if n_negatives is not None else n_true_pos

    if hydro_path is not None:
        negative_points = _sample_negatives_per_tile(
            hydro_path, jp2_paths, positive_points, n_neg, rng_seed,
        )
    else:
        imagery_extent = _imagery_union(jp2_paths)
        negative_points = sample_negatives(
            stream_mask, positive_points, n_neg, rng_seed,
            imagery_extent=imagery_extent,
        )
    negative_labeled = [(pt, "negative") for pt in negative_points]

    manifest_rows: list[dict] = []
    for jp2_path in jp2_paths:
        extract_chips(
            jp2_path, all_labeled, out_dir,
            manifest_rows=manifest_rows,
            augment_positives=augment_positives,
            augment_max_offset=augment_max_offset,
            rng_seed=rng_seed,
        )
        extract_chips(
            jp2_path, negative_labeled, out_dir,
            manifest_rows=manifest_rows,
            augment_positives=0,  # negatives are already spatially varied
        )

    _warn_skipped(all_labeled + negative_labeled, manifest_rows)

    manifest_path = out_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return str(manifest_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn_skipped(
    all_labeled: list[tuple[Point, str]],
    manifest_rows: list[dict],
) -> None:
    """Print a warning for any labeled point that produced no chips in any tile."""
    extracted_xy = {(float(r["x"]), float(r["y"])) for r in manifest_rows}
    skipped = [(pt, ftype) for pt, ftype in all_labeled
               if (pt.x, pt.y) not in extracted_xy]
    for pt, ftype in skipped:
        print(f"  WARNING: '{ftype}' label at EPSG:3067 ({pt.x:.0f}, {pt.y:.0f}) "
              f"falls outside all imagery tiles — skipped")
    if skipped:
        print(f"  {len(skipped)} label(s) skipped. "
              f"Add imagery tiles that cover these locations, or remove the labels.")


def _sample_negatives_per_tile(
    hydro_path: str,
    jp2_paths: list[str],
    positive_points: list[Point],
    n_total: int,
    rng_seed: int,
) -> list[Point]:
    """Sample negatives per JP2 tile using a bbox-scoped stream mask each time."""
    from shapely.geometry import box as _box
    from masking import build_stream_mask, BUFFER_METERS

    n_tiles = len(jp2_paths)
    n_per_tile = max(1, (n_total + n_tiles - 1) // n_tiles)

    all_negatives: list[Point] = []
    for i, jp2_path in enumerate(jp2_paths):
        with rasterio.open(jp2_path) as src:
            b = src.bounds
        bbox = (b.left - BUFFER_METERS, b.bottom - BUFFER_METERS,
                b.right + BUFFER_METERS, b.top + BUFFER_METERS)
        tile_mask = build_stream_mask(hydro_path, bbox=bbox)
        if tile_mask.is_empty:
            continue
        tile_extent = _box(b.left, b.bottom, b.right, b.top)
        negs = sample_negatives(
            tile_mask, positive_points, n_per_tile,
            rng_seed=rng_seed + i,
            imagery_extent=tile_extent,
        )
        all_negatives.extend(negs)

    return all_negatives[:n_total]


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
