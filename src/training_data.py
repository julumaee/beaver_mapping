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
  auto_negative — auto-sampled negative (assigned internally, do not use as a
                 label name — see sample_negatives / build_training_dataset)

Types excluded from training (point-scale features, not area classifiers):
  dam, lodge, other

Any label name not in FEATURE_TO_LABEL and not excluded above is treated as
unknown: build_training_dataset drops it and prints a warning rather than
silently training on it as a positive.
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
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ingestion import TILE_SIZE

_KML_NS = "http://www.opengis.net/kml/2.2"
_WGS84_TO_ETRS = Transformer.from_crs(4326, 3067, always_xy=True)

# Feature types excluded from training by default.
DEFAULT_EXCLUDE: frozenset[str] = frozenset({"lodge", "dam", "other"})

# Maps KML feature type names to integer class labels:
#   0 = negative (no beaver activity)
#   1 = positive (any beaver-associated visual signature)
# Names not present here (and not in DEFAULT_EXCLUDE) are treated as unknown:
# build_training_dataset drops them and prints a warning rather than silently
# treating them as positive — misspelt or unrecognised folder/placemark names
# used to inflate the positive class.
FEATURE_TO_LABEL: dict[str, int] = {
    # Negative class — folder/placemark name variants all map to 0:
    "negative":       0,
    "negatives":      0,
    "hard_negative":  0,
    "hard_negatives": 0,
    "auto_negative":  0,  # auto-sampled negatives (distinct from hand-labelled ones)
    # Generic positive labels — place anywhere in imagery:
    "dead_forest":    1,  # standing dead trees killed by beaver flooding
    "flood":          1,  # any open water impoundment
    "flooded_areas":  1,  # Google Earth folder name variant
    # Legacy / specific labels kept for backwards compatibility:
    "wet_forest":     1,
    "beaver_flood":   1,
}


def parse_kml_labels(kml_path: str, max_polygon_samples: int = 10) -> list[tuple[Point, str]]:
    """
    Parse a KML or KMZ file and return (point_epsg3067, feature_type) pairs.

    Feature type resolution (in priority order):
    1. Enclosing <Folder> name — Google Earth folder structure is the primary
       way to categorise placemarks (e.g. a folder named "Dead Forest" gives
       feature type "dead_forest" to all placemarks inside it).
    2. Placemark <name> tag — used only for root-level placemarks not inside
       any named folder.
    3. "unknown" — fallback when neither is present. build_training_dataset
       drops "unknown" (and any other unrecognised type name) with a warning
       rather than treating it as a positive.

    Geometry handling:
    - Point placemarks yield a single sample point.
    - LineString placemarks yield the centroid (unchanged behaviour).
    - Polygon (and MultiGeometry containing Polygons) placemarks yield several
      sample points inside the polygon — see _polygon_sample_points — instead
      of collapsing to a single centroid, which can fall outside a concave
      (e.g. crescent-shaped) flood polygon.

    Label names are normalised: lowercased, leading/trailing whitespace removed,
    spaces and hyphens replaced with underscores (so "Dead Forest" → "dead_forest").
    """
    results, _no_augment_ids = _parse_kml_labels_meta(kml_path, max_polygon_samples)
    return results


def _parse_kml_labels_meta(
    kml_path: str, max_polygon_samples: int = 10,
) -> tuple[list[tuple[Point, str]], set[int]]:
    """
    Like parse_kml_labels but also returns a set of id(Point) for sample points
    that came from a polygon placemark that yielded >= 2 samples. Those points
    already provide spatial diversity from the polygon itself and should not
    additionally receive offset augmentation in extract_chips.
    """
    kml_text = _read_kml_text(kml_path)
    root = ET.fromstring(kml_text)
    ns = _KML_NS if root.tag.startswith("{") else ""

    results: list[tuple[Point, str]] = []
    no_augment_ids: set[int] = set()
    doc = root.find(f"{{{ns}}}Document" if ns else "Document")
    _parse_kml_element(doc if doc is not None else root, ns, None, results,
                       no_augment_ids, max_polygon_samples)
    return results, no_augment_ids


def _normalise_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_kml_element(
    el: ET.Element,
    ns: str,
    folder_type: str | None,
    results: list[tuple[Point, str]],
    no_augment_ids: set[int],
    max_polygon_samples: int = 10,
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
            _parse_kml_element(child, ns, this_folder, results,
                               no_augment_ids, max_polygon_samples)
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
            geom = _extract_placemark_geometry(child, ns)
            if geom is None:
                continue
            kind, payload = geom

            if kind == "polygon":
                for shell, holes in payload:
                    pts = _polygon_sample_points(shell, holes, max_polygon_samples)
                    if not pts:
                        continue
                    if len(pts) >= 2:
                        no_augment_ids.update(id(p) for p in pts)
                    for p in pts:
                        results.append((p, ftype))
                continue

            if kind == "point":
                lon, lat = payload
            else:  # linestring -> centroid (unchanged behaviour)
                mp = MultiPoint(payload)
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
    no_augment_ids: set[int] | None = None,
) -> list[str]:
    """
    For each (point, feature_type) extract a TILE_SIZE×TILE_SIZE chip centred
    on that point and save as a .npy file.  The class label (0/1) is derived
    from FEATURE_TO_LABEL[feature_type]; points whose feature_type is not in
    FEATURE_TO_LABEL are skipped with a printed warning (callers should
    normally filter these out before calling extract_chips — see
    build_training_dataset — this is a defensive backstop).

    Chips are augmented with augment_positives additional chips extracted at
    random pixel offsets (±augment_max_offset), for both positive labels and
    hand-labelled negatives (any class-0 type other than "auto_negative",
    which is already spatially varied by construction). This simulates the
    detection grid misalignment and multiplies training samples without
    requiring new labels. Points listed in no_augment_ids (id(Point)) are
    skipped for augmentation — used for polygon-derived multi-sample points,
    which already provide spatial diversity from the polygon itself.

    x/y in the manifest stays at the original label point so spatial CV
    groups augmented chips with their source territory.

    Returns a list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _TAG = {0: "neg", 1: "pos"}
    _no_augment = no_augment_ids or set()
    written: list[str] = []
    half = TILE_SIZE // 2
    rng = random.Random(rng_seed)

    with rasterio.open(jp2_path) as src:
        for i, (pt, feature_type) in enumerate(labeled_points):
            label = FEATURE_TO_LABEL.get(feature_type)
            if label is None:
                print(f"  WARNING: unrecognised feature_type '{feature_type}' "
                      f"— skipping label point")
                continue

            col, row = ~src.transform * (pt.x, pt.y)
            col, row = int(col), int(row)

            should_augment = (
                augment_positives > 0
                and feature_type != "auto_negative"
                and id(pt) not in _no_augment
            )

            # Build list of (col_offset, row_offset, aug_index) to extract.
            # Index -1 = original (no offset); 0..N-1 = augmented.
            offsets: list[tuple[int, int, int]] = [(0, 0, -1)]
            if should_augment:
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

    Uses an STRtree over positive_points and a uniform spatial grid over the
    accepted negatives so rejection sampling stays fast for large n (~1000+)
    instead of scanning every prior point on every candidate draw.
    """
    if stream_mask is None and imagery_extent is None:
        return []
    if n <= 0:
        return []

    if imagery_extent is not None:
        minx, miny, maxx, maxy = imagery_extent.bounds
    else:
        minx, miny, maxx, maxy = stream_mask.bounds

    rng = random.Random(rng_seed)
    samples: list[Point] = []

    pos_tree = STRtree(positive_points) if positive_points else None

    # Uniform grid over accepted negatives: cell size == min_neg_spacing, so
    # any point within min_neg_spacing of a candidate lies in one of the 3x3
    # neighbouring cells — avoids O(n^2) distance checks as samples grows.
    cell = min_neg_spacing if min_neg_spacing > 0 else 1.0
    grid: dict[tuple[int, int], list[Point]] = {}

    def _cell_of(pt: Point) -> tuple[int, int]:
        return int(pt.x // cell), int(pt.y // cell)

    def _too_close_to_positive(pt: Point) -> bool:
        if pos_tree is None:
            return False
        for idx in pos_tree.query(pt.buffer(min_pos_distance)):
            if pt.distance(positive_points[idx]) < min_pos_distance:
                return True
        return False

    def _too_close_to_accepted(pt: Point) -> bool:
        cx, cy = _cell_of(pt)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for other in grid.get((gx, gy), ()):
                    if pt.distance(other) < min_neg_spacing:
                        return True
        return False

    for _ in range(n * 500):
        if len(samples) >= n:
            break
        pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if imagery_extent is not None and not imagery_extent.contains(pt):
            continue
        if stream_mask is not None and not stream_mask.intersects(pt):
            continue
        if _too_close_to_positive(pt):
            continue
        if _too_close_to_accepted(pt):
            continue
        samples.append(pt)
        grid.setdefault(_cell_of(pt), []).append(pt)

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
    hydro_flood_samples: int = 0,
    hydro_negatives: bool = True,
    neg_ratio: float = 1.0,
    max_polygon_samples: int = 10,
) -> str:
    """
    Orchestrate the full training data pipeline and write a manifest CSV.

    hydro_path: path to hydrography data; used for tulvaalue extraction and/or
                stream-mask negative sampling depending on the flags below.
    hydro_flood_samples: number of positive chips to auto-extract from the
                tulvaalue layer in hydro_path (0 = disabled).
    hydro_negatives: when True (default) and hydro_path is set, half of the
                auto-sampled negatives come from the stream corridor (hard,
                stream-adjacent negatives) and half from the full imagery
                extent, so the model sees both. When False, all auto-negatives
                come from the full imagery extent.
    stream_mask: legacy — passed directly to sample_negatives when hydro_path
                 is not set.
    augment_positives: number of extra offset chips per positive label point
                (and per hand-labelled negative — see extract_chips).
    augment_max_offset: maximum pixel shift in each direction for augmentation.
    neg_ratio: auto-negative chip count relative to the number of positive
                CHIPS after augmentation (default 1.0 = as many auto-negative
                chips as positive chips). Ignored when n_negatives is given
                explicitly.
    max_polygon_samples: cap on sample points drawn from one Polygon
                placemark (see _polygon_sample_points).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    imagery_extent = _imagery_union(jp2_paths)

    all_labeled: list[tuple[Point, str]] = []
    no_augment_ids: set[int] = set()
    unknown_counts: dict[str, int] = {}
    for kml_path in kml_paths:
        points, kml_no_augment_ids = _parse_kml_labels_meta(kml_path, max_polygon_samples)
        no_augment_ids |= kml_no_augment_ids
        for pt, ftype in points:
            if ftype in exclude_features:
                continue
            if ftype not in FEATURE_TO_LABEL:
                unknown_counts[ftype] = unknown_counts.get(ftype, 0) + 1
                continue
            all_labeled.append((pt, ftype))

    if unknown_counts:
        print("  WARNING: unrecognised label type(s) excluded from training:")
        for ftype, count in sorted(unknown_counts.items()):
            print(f"    '{ftype}': {count} placemark(s)")
        print("    Rename the folder/placemark to a recognised type, or add "
              "it to FEATURE_TO_LABEL in training_data.py.")

    if hydro_path is not None and hydro_flood_samples > 0:
        flood_pts = _load_tulvaalue_points(
            hydro_path, imagery_extent, hydro_flood_samples, rng_seed,
        )
        all_labeled.extend(flood_pts)
        print(f"  Auto-extracted {len(flood_pts)} flood chips from tulvaalue "
              f"(requested {hydro_flood_samples})")

    positive_points = [pt for pt, _ in all_labeled]

    if n_negatives is not None:
        n_neg = n_negatives
    else:
        # Estimate positive CHIP count after augmentation (matches what
        # extract_chips will produce, modulo points that fall outside a tile)
        # — hard negatives already in all_labeled must not inflate this count.
        expected_pos_chips = sum(
            1 if id(pt) in no_augment_ids else 1 + augment_positives
            for pt, ftype in all_labeled if FEATURE_TO_LABEL[ftype] == 1
        )
        n_neg = max(1, round(expected_pos_chips * neg_ratio))

    if hydro_path is not None and hydro_negatives:
        n_stream = n_neg // 2
        n_extent = n_neg - n_stream
        stream_negatives = _sample_negatives_per_tile(
            hydro_path, jp2_paths, positive_points, n_stream, rng_seed,
        )
        extent_negatives = sample_negatives(
            None, positive_points, n_extent, rng_seed=rng_seed + 9973,
            imagery_extent=imagery_extent,
        )
        negative_points = stream_negatives + extent_negatives
        print(f"  Auto-negatives: {len(stream_negatives)} stream-corridor, "
              f"{len(extent_negatives)} full-extent (requested {n_neg})")
    else:
        negative_points = sample_negatives(
            stream_mask, positive_points, n_neg, rng_seed,
            imagery_extent=imagery_extent,
        )
    negative_labeled = [(pt, "auto_negative") for pt in negative_points]

    manifest_rows: list[dict] = []
    for jp2_path in jp2_paths:
        extract_chips(
            jp2_path, all_labeled, out_dir,
            manifest_rows=manifest_rows,
            augment_positives=augment_positives,
            augment_max_offset=augment_max_offset,
            rng_seed=rng_seed,
            no_augment_ids=no_augment_ids,
        )
        extract_chips(
            jp2_path, negative_labeled, out_dir,
            manifest_rows=manifest_rows,
            augment_positives=0,  # auto-negatives are already spatially varied
        )

    _warn_skipped(all_labeled + negative_labeled, manifest_rows)

    manifest_path = out_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "feature_type", "x", "y"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    type_counts: dict[str, int] = {}
    for r in manifest_rows:
        type_counts[r["feature_type"]] = type_counts.get(r["feature_type"], 0) + 1
    print(f"  Final chip counts ({len(manifest_rows)} total):")
    for ftype, count in sorted(type_counts.items()):
        print(f"    {ftype}: {count}")

    return str(manifest_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tulvaalue_points(
    hydro_path: str,
    imagery_extent,
    n: int,
    rng_seed: int = 42,
) -> list[tuple[Point, str]]:
    """
    Sample n chip centres from tulvaalue (flooded area) polygons in the
    hydrography data that overlap the imagery extent.  Returns
    [(Point_epsg3067, "flood"), ...].
    """
    import fiona
    import geopandas as gpd
    import pandas as pd
    from masking import _resolve_files

    bbox = imagery_extent.bounds
    gdfs: list[gpd.GeoDataFrame] = []
    for f in _resolve_files(hydro_path):
        try:
            layers = fiona.listlayers(str(f))
        except Exception:
            continue
        if "tulvaalue" not in layers:
            continue
        gdf = gpd.read_file(str(f), layer="tulvaalue", bbox=bbox)
        if len(gdf) == 0:
            continue
        if gdf.crs is not None and gdf.crs.to_epsg() != 3067:
            gdf = gdf.to_crs(epsg=3067)
        gdfs.append(gdf[["geometry"]])

    if not gdfs:
        print("  WARNING: No tulvaalue features found in hydrography data "
              "— no auto-flood chips extracted")
        return []

    combined = (
        gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        if len(gdfs) > 1 else gdfs[0]
    )
    polys = [g for g in combined.geometry if g is not None and g.is_valid and not g.is_empty]
    if not polys:
        print("  WARNING: tulvaalue layer found but contains no valid polygons")
        return []

    total_area_ha = sum(p.area for p in polys) / 10_000
    print(f"  tulvaalue: {len(polys)} flood polygon(s), "
          f"{total_area_ha:.1f} ha within imagery extent — sampling {n} chip centres ...")

    rng = random.Random(rng_seed)
    areas = [p.area for p in polys]
    points: list[tuple[Point, str]] = []

    for _ in range(n * 200):
        if len(points) >= n:
            break
        poly = rng.choices(polys, weights=areas)[0]
        minx, miny, maxx, maxy = poly.bounds
        candidate = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not poly.contains(candidate):
            continue
        if not imagery_extent.contains(candidate):
            continue
        points.append((candidate, "flood"))

    if len(points) < n:
        print(f"  WARNING: Only sampled {len(points)}/{n} tulvaalue points "
              f"— flood polygons may be small or sparse within imagery")
    return points


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
    """Sample negatives using one stream mask scoped to the union of all tile bboxes.

    Building one mask (not one per tile) avoids N_tiles × N_files file opens
    while still avoiding the global 2.3M-feature load — the union bbox is
    much smaller than the full hydrography extent.
    """
    from shapely.geometry import box as _box
    from masking import build_stream_mask, BUFFER_METERS

    # Compute union bbox of all tiles in one pass.
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    tile_extents = []
    for jp2_path in jp2_paths:
        with rasterio.open(jp2_path) as src:
            b = src.bounds
        minx, miny = min(minx, b.left),  min(miny, b.bottom)
        maxx, maxy = max(maxx, b.right), max(maxy, b.top)
        tile_extents.append(_box(b.left, b.bottom, b.right, b.top))

    union_bbox = (minx - BUFFER_METERS, miny - BUFFER_METERS,
                  maxx + BUFFER_METERS, maxy + BUFFER_METERS)
    union_mask = build_stream_mask(hydro_path, bbox=union_bbox)
    if union_mask.is_empty:
        return []

    n_tiles = len(jp2_paths)
    n_per_tile = max(1, (n_total + n_tiles - 1) // n_tiles)

    all_negatives: list[Point] = []
    for i, tile_extent in enumerate(tile_extents):
        negs = sample_negatives(
            union_mask, positive_points, n_per_tile,
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


def _extract_placemark_geometry(placemark_el, ns: str):
    """
    Extract the usable geometry from a Placemark, searching descendants so
    geometries nested inside a <MultiGeometry> are found too.

    Returns one of:
      ("point", (lon, lat))
      ("linestring", [(lon, lat), ...])
      ("polygon", [(shell, holes), ...])   # one entry per Polygon found;
                                            # shell/holes are [(lon, lat), ...]
    or None if no usable geometry is present.
    """
    tag = lambda name: f"{{{ns}}}{name}" if ns else name  # noqa: E731

    def _ring_coords(ring_el):
        coords_el = ring_el.find(f".//{tag('coordinates')}")
        if coords_el is None or not coords_el.text:
            return []
        return _parse_coord_string(coords_el.text)

    def _polygon_from_el(poly_el):
        outer = poly_el.find(f"{tag('outerBoundaryIs')}/{tag('LinearRing')}")
        if outer is None:
            return None
        shell = _ring_coords(outer)
        if len(shell) < 3:
            return None
        holes = []
        for inner in poly_el.findall(f"{tag('innerBoundaryIs')}/{tag('LinearRing')}"):
            hole = _ring_coords(inner)
            if len(hole) >= 3:
                holes.append(hole)
        return shell, holes

    polygon_els = placemark_el.findall(f".//{tag('Polygon')}")
    if polygon_els:
        polys = [p for p in (_polygon_from_el(pe) for pe in polygon_els) if p is not None]
        if polys:
            return "polygon", polys

    point_el = placemark_el.find(f".//{tag('Point')}")
    if point_el is not None:
        coords_el = point_el.find(f".//{tag('coordinates')}")
        if coords_el is not None and coords_el.text:
            pairs = _parse_coord_string(coords_el.text)
            if pairs:
                return "point", pairs[0]

    linestring_el = placemark_el.find(f".//{tag('LineString')}")
    if linestring_el is not None:
        coords_el = linestring_el.find(f".//{tag('coordinates')}")
        if coords_el is not None and coords_el.text:
            pairs = _parse_coord_string(coords_el.text)
            if pairs:
                return "linestring", pairs

    return None


def _polygon_sample_points(
    shell_lonlat: list[tuple[float, float]],
    holes_lonlat: list[list[tuple[float, float]]],
    max_samples: int = 10,
    target_cell_m: float = 40.0,
    min_spacing_m: float = 30.0,
    rng_seed: int = 42,
) -> list[Point]:
    """
    Build a Polygon in EPSG:3067 from WGS84 ring coordinates and return sample
    points inside it (always inside, even for concave/crescent shapes).

    Sample count is roughly proportional to polygon area (one point per
    target_cell_m x target_cell_m cell), minimum 1, capped at max_samples.
    Points are spaced >= min_spacing_m apart via rejection sampling. For the
    single-sample case, representative_point() is used instead of the
    centroid so the point is guaranteed to fall inside the polygon.
    """
    shell_3067 = [_WGS84_TO_ETRS.transform(lon, lat) for lon, lat in shell_lonlat]
    holes_3067 = [
        [_WGS84_TO_ETRS.transform(lon, lat) for lon, lat in hole]
        for hole in holes_lonlat
    ]
    try:
        poly = Polygon(shell_3067, holes_3067)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return []
    if poly.is_empty or poly.area <= 0:
        return []
    # buffer(0) on a self-intersecting ring can yield a MultiPolygon — use the
    # largest part.
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    target_n = max(1, round(poly.area / (target_cell_m * target_cell_m)))
    n = min(max_samples, target_n)

    if n <= 1:
        return [poly.representative_point()]

    rng = random.Random(rng_seed)
    minx, miny, maxx, maxy = poly.bounds
    samples: list[Point] = []
    max_attempts = n * 200
    for _ in range(max_attempts):
        if len(samples) >= n:
            break
        cand = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not poly.contains(cand):
            continue
        if any(cand.distance(s) < min_spacing_m for s in samples):
            continue
        samples.append(cand)

    if not samples:
        samples = [poly.representative_point()]
    return samples


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
