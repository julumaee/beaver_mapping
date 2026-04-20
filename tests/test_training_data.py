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
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_data import (
    DEFAULT_EXCLUDE,
    FEATURE_TO_LABEL,
    build_training_dataset,
    extract_chips,
    parse_kml_labels,
    sample_negatives,
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


class TestExtractChips:
    def test_chip_written_with_correct_label(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        rows: list[dict] = []
        extract_chips(jp2, [(Point(cx, cy), "wet_forest")],
                      str(tmp_path / "chips"), manifest_rows=rows)
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
        written = extract_chips(jp2, [(Point(cx + 1_000_000, cy), "dam")],
                                str(tmp_path / "far"))
        assert written == []

    def test_chip_shape(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_raster(tmp_path, cx, cy, size=2048)
        written = extract_chips(jp2, [(Point(cx, cy), "dam")],
                                str(tmp_path / "chips"))
        assert np.load(written[0]).shape == (3, 512, 512)


class TestFeatureToLabel:
    def test_wet_forest_and_beaver_flood_are_1(self):
        assert FEATURE_TO_LABEL["wet_forest"] == 1
        assert FEATURE_TO_LABEL["beaver_flood"] == 1

    def test_negative_is_0(self):
        assert FEATURE_TO_LABEL["negative"] == 0


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
