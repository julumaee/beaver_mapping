"""Tests for src/training_data.py."""

import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_data import (
    DEFAULT_EXCLUDE,
    FEATURE_TO_LABEL,
    build_training_dataset,
    extract_chips,
    parse_kml_labels,
    sample_negatives,
    _parse_kml_labels_meta,
    _polygon_sample_points,
)

_KML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>{name}</name>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>
  </Document>
</kml>"""

_KML_NO_NAME = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>
  </Document>
</kml>"""

_LON, _LAT = 23.761, 61.498
_TO_WGS84 = Transformer.from_crs(3067, 4326, always_xy=True)


def _write_kml(tmp_path: Path, lon=_LON, lat=_LAT, name="dam") -> str:
    p = tmp_path / "labels.kml"
    p.write_text(_KML_TEMPLATE.format(name=name, lon=lon, lat=lat), encoding="utf-8")
    return str(p)


def _write_kmz(tmp_path: Path, lon=_LON, lat=_LAT, name="dam") -> str:
    text = _KML_TEMPLATE.format(name=name, lon=lon, lat=lat)
    p = tmp_path / "labels.kmz"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("doc.kml", text)
    return str(p)


def _write_raster(tmp_path: Path, cx: float, cy: float, size: int = 2048) -> str:
    half = size // 2
    transform = from_bounds(cx - half, cy - half, cx + half, cy + half, size, size)
    p = tmp_path / "tile.tif"
    data = np.random.randint(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(str(p), "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs=CRS.from_epsg(3067),
                       transform=transform) as dst:
        dst.write(data)
    return str(p)


def _write_tulvaalue_gpkg(tmp_path: Path, cx: float, cy: float, size: float = 500.0) -> str:
    """Write a minimal GPKG with one tulvaalue polygon centred at (cx, cy)."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box as _box
    poly = _box(cx - size, cy - size, cx + size, cy + size)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3067")
    p = tmp_path / "hydro.gpkg"
    gdf.to_file(str(p), layer="tulvaalue", driver="GPKG")
    return str(p)


def _write_folder_kml(tmp_path: Path, folder_name: str, lon=_LON, lat=_LAT,
                      placemark_name: str = "Placemark 1") -> str:
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f'<Folder><name>{folder_name}</name>'
        f'<Placemark><name>{placemark_name}</name>'
        f'<Point><coordinates>{lon},{lat},0</coordinates></Point>'
        '</Placemark></Folder>'
        '</Document></kml>'
    )
    p = tmp_path / "folder.kml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _write_polygon_kml(tmp_path: Path, cx: float, cy: float, half_size: float = 100.0,
                       name: str = "flood") -> str:
    """Write a KML with a single square Polygon placemark, half_size in metres,
    centred at (cx, cy) in EPSG:3067."""
    corners_3067 = [
        (cx - half_size, cy - half_size),
        (cx + half_size, cy - half_size),
        (cx + half_size, cy + half_size),
        (cx - half_size, cy + half_size),
        (cx - half_size, cy - half_size),
    ]
    corners_lonlat = [_TO_WGS84.transform(x, y) for x, y in corners_3067]
    coords_text = " ".join(f"{lon},{lat},0" for lon, lat in corners_lonlat)
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f'<Placemark><name>{name}</name>'
        '<Polygon><outerBoundaryIs><LinearRing>'
        f'<coordinates>{coords_text}</coordinates>'
        '</LinearRing></outerBoundaryIs></Polygon>'
        '</Placemark></Document></kml>'
    )
    p = tmp_path / "polygon.kml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _crescent_ring_3067(cx: float, cy: float) -> list[tuple[float, float]]:
    """
    Build a thin crescent/C-shaped ring (difference of two near-coincident
    circles) centred at (cx, cy) in EPSG:3067, small enough to sample a
    single point but concave enough that the centroid falls outside it —
    exercising the representative_point() fallback.
    """
    from shapely.geometry import Point as _P
    big = _P(cx, cy).buffer(20, quad_segs=64)
    small = _P(cx + 3, cy).buffer(19, quad_segs=64)
    crescent = big.difference(small).simplify(0.5, preserve_topology=True)
    assert not crescent.contains(crescent.centroid), "test fixture must be concave"
    return list(crescent.exterior.coords)


def _write_crescent_polygon_kml(tmp_path: Path, cx: float, cy: float,
                                name: str = "flood") -> str:
    """Write a KML with a small concave (crescent) Polygon placemark."""
    ring_3067 = _crescent_ring_3067(cx, cy)
    corners_lonlat = [_TO_WGS84.transform(x, y) for x, y in ring_3067]
    coords_text = " ".join(f"{lon},{lat},0" for lon, lat in corners_lonlat)
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f'<Placemark><name>{name}</name>'
        '<Polygon><outerBoundaryIs><LinearRing>'
        f'<coordinates>{coords_text}</coordinates>'
        '</LinearRing></outerBoundaryIs></Polygon>'
        '</Placemark></Document></kml>'
    )
    p = tmp_path / "crescent.kml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _write_stream_gpkg(tmp_path: Path, cx: float, cy: float, length: float = 3000.0) -> str:
    """Write a minimal GPKG with a virtavesikapea line through (cx, cy)."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString
    line = LineString([(cx - length / 2, cy), (cx + length / 2, cy)])
    gdf = gpd.GeoDataFrame({"geometry": [line]}, crs="EPSG:3067")
    p = tmp_path / "stream.gpkg"
    gdf.to_file(str(p), layer="virtavesikapea", driver="GPKG")
    return str(p)


def _multi_feature_kml(tmp_path: Path, cx: float, cy: float) -> str:
    lon, lat = _TO_WGS84.transform(cx, cy)
    features = ["dam", "wet_forest", "beaver_flood", "lodge"]
    placemarks = "\n".join(
        f"<Placemark><name>{f}</name>"
        f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
        for f in features
    )
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        f"<Document>{placemarks}</Document></kml>"
    )
    p = tmp_path / "multi.kml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestParseKmlLabels:
    def test_returns_point_and_feature_type(self, tmp_path):
        pt, ftype = parse_kml_labels(_write_kml(tmp_path, name="dam"))[0]
        assert isinstance(pt, Point)
        assert ftype == "dam"

    def test_reprojected_to_3067(self, tmp_path):
        pt, _ = parse_kml_labels(_write_kml(tmp_path))[0]
        assert 100_000 < pt.x < 800_000
        assert 6_600_000 < pt.y < 7_800_000

    def test_no_name_gives_unknown(self, tmp_path):
        p = tmp_path / "noname.kml"
        p.write_text(_KML_NO_NAME.format(lon=_LON, lat=_LAT), encoding="utf-8")
        _, ftype = parse_kml_labels(str(p))[0]
        assert ftype == "unknown"

    def test_feature_type_lowercased(self, tmp_path):
        _, ftype = parse_kml_labels(_write_kml(tmp_path, name="Wet_Forest"))[0]
        assert ftype == "wet_forest"

    def test_kmz_parsed_identically(self, tmp_path):
        kml_res = parse_kml_labels(_write_kml(tmp_path, name="beaver_flood"))
        kmz_res = parse_kml_labels(_write_kmz(tmp_path, name="beaver_flood"))
        kpt, kft = kml_res[0]
        zpt, zft = kmz_res[0]
        assert abs(kpt.x - zpt.x) < 1
        assert kft == zft == "beaver_flood"

    def test_folder_name_used_as_feature_type(self, tmp_path):
        _, ftype = parse_kml_labels(
            _write_folder_kml(tmp_path, "Dead Forest")
        )[0]
        assert ftype == "dead_forest"

    def test_folder_name_overrides_placemark_name(self, tmp_path):
        _, ftype = parse_kml_labels(
            _write_folder_kml(tmp_path, "flood", placemark_name="Placemark 42")
        )[0]
        assert ftype == "flood"

    def test_folder_name_normalised_spaces_to_underscores(self, tmp_path):
        _, ftype = parse_kml_labels(
            _write_folder_kml(tmp_path, "Flooded Areas")
        )[0]
        assert ftype == "flooded_areas"

    def test_root_placemark_without_folder_uses_placemark_name(self, tmp_path):
        _, ftype = parse_kml_labels(_write_kml(tmp_path, name="negative"))[0]
        assert ftype == "negative"


class TestExtractChips:
    def test_chip_written_with_correct_label(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "wet_forest")],
                      str(tmp_path / "chips"), manifest_rows=rows,
                      augment_positives=0)
        assert len(rows) == 1
        assert rows[0]["label"] == FEATURE_TO_LABEL["wet_forest"]   # == 1
        assert rows[0]["feature_type"] == "wet_forest"

    def test_flood_label(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "wet_forest")],
                      str(tmp_path / "chips"), manifest_rows=rows)
        assert rows[0]["label"] == FEATURE_TO_LABEL["wet_forest"]  # == 1

    def test_negative_label(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "negative")],
                      str(tmp_path / "chips"), manifest_rows=rows)
        assert rows[0]["label"] == 0

    def test_outside_raster_skipped(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        written = extract_chips(jp2, [(Point(cx + 1_000_000, cy), "wet_forest")],
                                str(tmp_path / "far"))
        assert written == []

    def test_chip_shape(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        written = extract_chips(jp2, [(Point(cx, cy), "wet_forest")],
                                str(tmp_path / "chips"))
        assert np.load(written[0]).shape == (3, 512, 512)

    def test_unmapped_feature_type_skipped_with_warning(self, tmp_path, capsys):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        written = extract_chips(jp2, [(Point(cx, cy), "totally_unrecognised")],
                                str(tmp_path / "chips"))
        assert written == []
        assert "unrecognised feature_type 'totally_unrecognised'" in capsys.readouterr().out


class TestFeatureToLabel:
    def test_positive_labels_are_1(self):
        for label in ("wet_forest", "beaver_flood", "dead_forest", "flood"):
            assert FEATURE_TO_LABEL[label] == 1, f"{label} should be class 1"

    def test_negative_is_0(self):
        assert FEATURE_TO_LABEL["negative"] == 0

    def test_auto_negative_is_0(self):
        assert FEATURE_TO_LABEL["auto_negative"] == 0

    def test_unknown_type_not_mapped(self):
        # Unrecognised/misspelt names must NOT silently default to positive —
        # build_training_dataset drops them instead (see TestUnknownLabels).
        assert "unrecognised_type" not in FEATURE_TO_LABEL
        assert "unknown" not in FEATURE_TO_LABEL


class TestSampleNegatives:
    def _mask(self):
        return box(320_000, 6_815_000, 330_000, 6_825_000)

    def test_returns_requested_count(self):
        assert len(sample_negatives(self._mask(), [], n=10)) == 10

    def test_all_points_inside_mask(self):
        mask = self._mask()
        for pt in sample_negatives(mask, [], n=20, rng_seed=0):
            assert mask.contains(pt)

    def test_deterministic_with_same_seed(self):
        mask = self._mask()
        a = sample_negatives(mask, [], n=10, rng_seed=7)
        b = sample_negatives(mask, [], n=10, rng_seed=7)
        assert [p.wkt for p in a] == [p.wkt for p in b]

    def test_empty_mask_returns_empty(self):
        from shapely.geometry import Point as Pt
        empty = Pt(0, 0).buffer(0)
        assert sample_negatives(empty, [], n=10) == []


class TestBuildTrainingDataset:
    def _setup(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="wet_forest")
        mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        return jp2, kml, mask

    def test_manifest_has_all_label_values(self, tmp_path):
        jp2, kml, mask = self._setup(tmp_path)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=1)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        labels = {int(r["label"]) for r in rows}
        assert 1 in labels   # flood
        assert 0 in labels   # negative

    def test_lodge_excluded_by_default(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        kml = _multi_feature_kml(tmp_path, cx, cy)
        mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        ftypes = {r["feature_type"] for r in rows}
        assert "lodge" not in ftypes

    def test_default_exclude_contains_lodge(self):
        assert "lodge" in DEFAULT_EXCLUDE

    def test_default_exclude_contains_dam(self):
        assert "dam" in DEFAULT_EXCLUDE

    def test_dead_forest_label_produces_class1_chip(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="dead_forest")
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert any(r["feature_type"] == "dead_forest" and int(r["label"]) == 1
                   for r in rows)

    def test_flood_label_produces_class1_chip(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert any(r["feature_type"] == "flood" and int(r["label"]) == 1
                   for r in rows)

    def test_no_hydro_samples_negatives_from_imagery_extent(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=3)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        assert len(neg_rows) == 3

    def test_tulvaalue_chips_extracted_as_flood(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="dead_forest")
        gpkg = _write_tulvaalue_gpkg(tmp_path, cx, cy, size=400.0)
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            n_negatives=0, hydro_path=gpkg, hydro_flood_samples=5,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        flood_rows = [r for r in rows if r["feature_type"] == "flood" and int(r["label"]) == 1]
        assert len(flood_rows) > 0

    def test_tulvaalue_zero_samples_skipped(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="dead_forest")
        gpkg = _write_tulvaalue_gpkg(tmp_path, cx, cy)
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            n_negatives=0, hydro_path=gpkg, hydro_flood_samples=0,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert not any(r["feature_type"] == "flood" for r in rows)


class TestUnknownLabelsDropped:
    """R1.3 — unrecognised label types are excluded, not treated as positive."""

    def test_unknown_type_excluded_with_warning(self, tmp_path, capsys):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        kml = _write_folder_kml(tmp_path, "Mystery Folder")  # -> "mystery_folder"
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert rows == []
        out = capsys.readouterr().out
        assert "mystery_folder" in out
        assert "1 placemark" in out

    def test_multiple_unknown_types_counted_separately(self, tmp_path, capsys):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        placemarks = "".join(
            f"<Placemark><name>{name}</name>"
            f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
            for name in ("typo_flod", "typo_flod", "possible_dam")
        )
        text = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2">'
            f"<Document>{placemarks}</Document></kml>"
        )
        kml = tmp_path / "unknowns.kml"
        kml.write_text(text, encoding="utf-8")
        manifest = build_training_dataset([jp2], [str(kml)], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert rows == []
        out = capsys.readouterr().out
        assert "'typo_flod': 2 placemark" in out
        assert "'possible_dam': 1 placemark" in out

    def test_known_types_still_used_alongside_unknown(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        placemarks = (
            f"<Placemark><name>flood</name>"
            f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
            f"<Placemark><name>gibberish_xyz</name>"
            f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
        )
        text = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2">'
            f"<Document>{placemarks}</Document></kml>"
        )
        kml = tmp_path / "mixed.kml"
        kml.write_text(text, encoding="utf-8")
        manifest = build_training_dataset([jp2], [str(kml)], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        ftypes = {r["feature_type"] for r in rows}
        assert ftypes == {"flood"}


class TestAutoNegativeNaming:
    """R1.2/R1.4 — auto-sampled negatives get their own feature_type."""

    def test_auto_negatives_named_auto_negative(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=3)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        assert len(neg_rows) == 3
        assert all(r["feature_type"] == "auto_negative" for r in neg_rows)

    def test_hand_labelled_negative_keeps_its_own_type(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="hard_negatives")
        manifest = build_training_dataset([jp2], [kml], None,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert all(r["feature_type"] == "hard_negatives" for r in rows)
        assert len(rows) >= 1


class TestPolygonLabels:
    """R1.1 — Polygon placemarks yield several sample points, not one centroid."""

    def test_small_polygon_yields_single_point_inside(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=5.0)
        points = parse_kml_labels(kml)
        assert len(points) == 1
        pt, ftype = points[0]
        assert ftype == "flood"
        poly = Polygon([(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)])
        assert poly.buffer(0.5).contains(pt)  # small floating-point transform tolerance

    def test_large_polygon_yields_multiple_points_capped_and_inside(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=300.0)  # 600x600 m
        points = parse_kml_labels(kml, max_polygon_samples=10)
        assert 2 <= len(points) <= 10
        poly = Polygon([(cx - 300, cy - 300), (cx + 300, cy - 300),
                        (cx + 300, cy + 300), (cx - 300, cy + 300)]).buffer(0.5)
        for pt, ftype in points:
            assert ftype == "flood"
            assert poly.contains(pt)

    def test_polygon_samples_respect_cap(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=300.0)
        points = parse_kml_labels(kml, max_polygon_samples=3)
        assert len(points) <= 3

    def test_polygon_samples_spaced_apart(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=300.0)
        points = [pt for pt, _ in parse_kml_labels(kml, max_polygon_samples=10)]
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                assert a.distance(b) >= 30.0 - 1e-6

    def test_crescent_polygon_sample_inside_concave_shape(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_crescent_polygon_kml(tmp_path, cx, cy)
        points = parse_kml_labels(kml)
        assert len(points) == 1
        pt, ftype = points[0]
        assert ftype == "flood"
        ring_3067 = _crescent_ring_3067(cx, cy)
        poly = Polygon(ring_3067)
        assert not poly.contains(poly.centroid)  # sanity: fixture is concave
        assert poly.buffer(0.5).contains(pt)     # sampled point is still inside

    def test_multi_sample_polygon_points_marked_no_augment(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=300.0)
        points, no_augment_ids = _parse_kml_labels_meta(kml, max_polygon_samples=10)
        assert len(points) >= 2
        assert all(id(pt) in no_augment_ids for pt, _ in points)

    def test_single_sample_polygon_not_marked_no_augment(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        kml = _write_polygon_kml(tmp_path, cx, cy, half_size=5.0)
        points, no_augment_ids = _parse_kml_labels_meta(kml)
        assert len(points) == 1
        assert id(points[0][0]) not in no_augment_ids


class TestExtractChipsNoAugment:
    def test_no_augment_ids_skips_offset_augmentation(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        pt = Point(cx, cy)
        rows: list[dict] = []
        extract_chips(jp2, [(pt, "wet_forest")], str(tmp_path / "chips"),
                      manifest_rows=rows, augment_positives=3, no_augment_ids={id(pt)})
        assert len(rows) == 1

    def test_hand_labelled_negative_is_augmented(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "hard_negatives")], str(tmp_path / "chips"),
                      manifest_rows=rows, augment_positives=3)
        assert len(rows) == 4  # 1 original + 3 augmented

    def test_auto_negative_is_never_augmented(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "auto_negative")], str(tmp_path / "chips"),
                      manifest_rows=rows, augment_positives=3)
        assert len(rows) == 1


class TestNegRatio:
    """R1.4 — auto-negative count scales with positive CHIP count, not point count."""

    def test_neg_ratio_scales_with_augmented_positive_chips(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            augment_positives=2, neg_ratio=1.0,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        pos_rows = [r for r in rows if int(r["label"]) == 1]
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        # 1 positive point + 2 augmented = 3 positive chips; neg_ratio 1.0 -> 3 negatives
        assert len(pos_rows) == 3
        assert len(neg_rows) == 3

    def test_neg_ratio_two_doubles_negatives(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            augment_positives=2, neg_ratio=2.0,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        assert len(neg_rows) == 6

    def test_explicit_n_negatives_overrides_neg_ratio(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            augment_positives=2, neg_ratio=5.0, n_negatives=1,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        assert len(neg_rows) == 1

    def test_hydro_negatives_split_half_stream_half_extent(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="flood")
        gpkg = _write_stream_gpkg(tmp_path, cx, cy, length=3000.0)
        manifest = build_training_dataset(
            [jp2], [kml], None, str(tmp_path / "ds"),
            augment_positives=5, neg_ratio=1.0,
            hydro_path=gpkg, hydro_negatives=True,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        neg_rows = [r for r in rows if int(r["label"]) == 0]
        # 1 positive point + 5 augmented = 6 positive chips -> 6 auto-negatives requested
        assert len(neg_rows) == 6


class TestSampleNegativesAtScale:
    def test_respects_min_pos_distance_and_spacing_at_scale(self):
        mask = box(320_000, 6_815_000, 330_000, 6_825_000)
        positives = [Point(322_000, 6_817_000)]
        samples = sample_negatives(mask, positives, n=150, min_pos_distance=200,
                                   min_neg_spacing=100)
        assert len(samples) == 150
        for i, p in enumerate(samples):
            assert p.distance(positives[0]) >= 200
            for q in samples[i + 1:]:
                assert p.distance(q) >= 100
