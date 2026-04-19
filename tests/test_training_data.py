"""Tests for src/training_data.py (Feature 3)."""

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
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_data import (
    DEFAULT_EXCLUDE,
    NEGATIVE_EXCLUSION_RADIUS,
    build_training_dataset,
    extract_chips,
    parse_kml_labels,
    sample_negatives,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task 3.1 — KML label parsing
# ---------------------------------------------------------------------------

class TestParseKmlLabels:
    def test_returns_point_and_feature_type(self, tmp_path):
        kml = _write_kml(tmp_path, name="dam")
        results = parse_kml_labels(kml)
        assert len(results) == 1
        pt, ftype = results[0]
        assert isinstance(pt, Point)
        assert ftype == "dam"

    def test_reprojected_to_3067(self, tmp_path):
        kml = _write_kml(tmp_path)
        pt, _ = parse_kml_labels(kml)[0]
        assert 100_000 < pt.x < 800_000
        assert 6_600_000 < pt.y < 7_800_000

    def test_no_name_gives_unknown(self, tmp_path):
        p = tmp_path / "noname.kml"
        p.write_text(_KML_NO_NAME.format(lon=_LON, lat=_LAT), encoding="utf-8")
        _, ftype = parse_kml_labels(str(p))[0]
        assert ftype == "unknown"

    def test_feature_type_lowercased(self, tmp_path):
        kml = _write_kml(tmp_path, name="Wet_Forest")
        _, ftype = parse_kml_labels(kml)[0]
        assert ftype == "wet_forest"

    def test_kmz_parsed_identically(self, tmp_path):
        kml_res = parse_kml_labels(_write_kml(tmp_path, name="beaver_flood"))
        kmz_res = parse_kml_labels(_write_kmz(tmp_path, name="beaver_flood"))
        assert len(kml_res) == len(kmz_res) == 1
        kpt, kft = kml_res[0]
        zpt, zft = kmz_res[0]
        assert abs(kpt.x - zpt.x) < 1
        assert kft == zft == "beaver_flood"

    def test_multiple_placemarks(self, tmp_path):
        text = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark><name>dam</name><Point><coordinates>23.0,61.0,0</coordinates></Point></Placemark>
    <Placemark><name>wet_forest</name><Point><coordinates>24.0,62.0,0</coordinates></Point></Placemark>
    <Placemark><name>lodge</name><Point><coordinates>25.0,63.0,0</coordinates></Point></Placemark>
  </Document>
</kml>"""
        p = tmp_path / "multi.kml"
        p.write_text(text, encoding="utf-8")
        results = parse_kml_labels(str(p))
        assert len(results) == 3
        ftypes = [ft for _, ft in results]
        assert ftypes == ["dam", "wet_forest", "lodge"]


# ---------------------------------------------------------------------------
# Task 3.2 — Chip extraction
# ---------------------------------------------------------------------------

class TestExtractChips:
    def test_chips_written_for_interior_point(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        seed = [(Point(cx, cy), "dam")]
        written = extract_chips(jp2, seed, str(tmp_path / "chips"), label=1)
        assert len(written) == 1
        assert np.load(written[0]).shape == (3, 512, 512)

    def test_chip_outside_raster_skipped(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        far = [(Point(cx + 1_000_000, cy), "dam")]
        written = extract_chips(jp2, far, str(tmp_path / "chips_far"), label=1)
        assert written == []

    def test_manifest_row_has_feature_type(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "beaver_flood")],
                      str(tmp_path / "chips"), label=1, manifest_rows=rows)
        assert len(rows) == 1
        assert rows[0]["feature_type"] == "beaver_flood"
        assert rows[0]["label"] == 1

    def test_negative_manifest_row(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "negative")],
                      str(tmp_path / "chips"), label=0, manifest_rows=rows)
        assert rows[0]["label"] == 0
        assert rows[0]["feature_type"] == "negative"

    def test_edge_chip_padded_to_tile_size(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        with rasterio.open(jp2) as src:
            edge_x = src.bounds.left + 10
            edge_y = src.bounds.top - 10
        written = extract_chips(jp2, [(Point(edge_x, edge_y), "dam")],
                                str(tmp_path / "chips_edge"), label=0)
        if written:
            assert np.load(written[0]).shape == (3, 512, 512)


# ---------------------------------------------------------------------------
# Task 3.3 — Negative sampling
# ---------------------------------------------------------------------------

class TestSampleNegatives:
    def _mask(self):
        return box(320_000, 6_815_000, 330_000, 6_825_000)

    def test_returns_requested_count(self):
        assert len(sample_negatives(self._mask(), [], n=10)) == 10

    def test_points_inside_mask(self):
        mask = self._mask()
        for pt in sample_negatives(mask, [], n=30):
            assert mask.contains(pt)

    def test_exclusion_zone_respected(self):
        mask = self._mask()
        centroid = mask.centroid
        for pt in sample_negatives(mask, [centroid], n=20, rng_seed=0):
            assert pt.distance(centroid) >= NEGATIVE_EXCLUSION_RADIUS - 1

    def test_deterministic_with_same_seed(self):
        mask = self._mask()
        a = sample_negatives(mask, [], n=10, rng_seed=7)
        b = sample_negatives(mask, [], n=10, rng_seed=7)
        assert [p.wkt for p in a] == [p.wkt for p in b]


# ---------------------------------------------------------------------------
# Integration — build_training_dataset
# ---------------------------------------------------------------------------

class TestBuildTrainingDataset:
    def _setup(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        lon, lat = _TO_WGS84.transform(cx, cy)
        kml = _write_kml(tmp_path, lon=lon, lat=lat, name="dam")
        mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        return jp2, kml, mask, cx, cy

    def test_manifest_has_both_labels(self, tmp_path):
        jp2, kml, mask, *_ = self._setup(tmp_path)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=1)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert {int(r["label"]) for r in rows} == {0, 1}

    def test_manifest_has_feature_type_column(self, tmp_path):
        jp2, kml, mask, *_ = self._setup(tmp_path)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=1)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        assert all("feature_type" in r for r in rows)

    def test_lodge_excluded_by_default(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        kml = _multi_feature_kml(tmp_path, cx, cy)
        mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        feature_types = {r["feature_type"] for r in rows if int(r["label"]) == 1}
        assert "lodge" not in feature_types
        assert "dam" in feature_types

    def test_wet_forest_and_beaver_flood_both_label_1(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        kml = _multi_feature_kml(tmp_path, cx, cy)
        mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        manifest = build_training_dataset([jp2], [kml], mask,
                                          str(tmp_path / "ds"), n_negatives=0)
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["feature_type"] in ("wet_forest", "beaver_flood"):
                assert int(r["label"]) == 1

    def test_default_exclude_contains_lodge(self):
        assert "lodge" in DEFAULT_EXCLUDE
