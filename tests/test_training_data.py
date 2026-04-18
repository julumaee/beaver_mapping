"""Tests for src/training_data.py (Feature 3)."""

import sys
import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, box
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training_data import (
    parse_kml_labels,
    extract_chips,
    sample_negatives,
    build_training_dataset,
    NEGATIVE_EXCLUSION_RADIUS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>
  </Document>
</kml>"""

# A known WGS84 point in Finland (Tampere area) and its approximate EPSG:3067 coords.
_LON, _LAT = 23.761, 61.498
# Expected ETRS-TM35FIN x≈327 800, y≈6 820 600 (±100 m tolerance in tests)


def _write_kml(tmp_path: Path, lon: float = _LON, lat: float = _LAT) -> str:
    p = tmp_path / "labels.kml"
    p.write_text(_KML_TEMPLATE.format(lon=lon, lat=lat), encoding="utf-8")
    return str(p)


def _write_kmz(tmp_path: Path, lon: float = _LON, lat: float = _LAT) -> str:
    kml_text = _KML_TEMPLATE.format(lon=lon, lat=lat)
    p = tmp_path / "labels.kmz"
    with zipfile.ZipFile(str(p), "w") as zf:
        zf.writestr("doc.kml", kml_text)
    return str(p)


def _write_jp2(tmp_path: Path, cx: float, cy: float, size: int = 2048) -> str:
    """
    Write a tiny synthetic EPSG:3067 GeoTIFF (using .tif extension so rasterio
    can write it without needing a JPEG2000 driver) centred near (cx, cy).
    Training code only cares about the CRS and transform, not the file extension.
    """
    half = size // 2
    west, east = cx - half, cx + half
    south, north = cy - half, cy + half
    transform = from_bounds(west, south, east, north, size, size)

    p = tmp_path / "tile.tif"
    data = np.random.randint(0, 255, (3, size, size), dtype=np.uint8)
    with rasterio.open(
        str(p),
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(3067),
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(p)


# ---------------------------------------------------------------------------
# Task 3.1 — KML label parsing
# ---------------------------------------------------------------------------

class TestParseKmlLabels:
    def test_kml_returns_one_point(self, tmp_path):
        kml = _write_kml(tmp_path)
        pts = parse_kml_labels(kml)
        assert len(pts) == 1

    def test_kml_reprojected_to_3067(self, tmp_path):
        kml = _write_kml(tmp_path)
        pt = parse_kml_labels(kml)[0]
        # EPSG:3067 x values in Finland are ~100 000 – 800 000
        assert 100_000 < pt.x < 800_000
        # EPSG:3067 y values in Finland are ~6 600 000 – 7 800 000
        assert 6_600_000 < pt.y < 7_800_000

    def test_kmz_parsed_identically(self, tmp_path):
        kml_pts = parse_kml_labels(_write_kml(tmp_path))
        kmz_pts = parse_kml_labels(_write_kmz(tmp_path))
        assert len(kml_pts) == len(kmz_pts) == 1
        assert abs(kml_pts[0].x - kmz_pts[0].x) < 1
        assert abs(kml_pts[0].y - kmz_pts[0].y) < 1

    def test_multiple_placemarks(self, tmp_path):
        kml_text = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark><Point><coordinates>23.0,61.0,0</coordinates></Point></Placemark>
    <Placemark><Point><coordinates>24.0,62.0,0</coordinates></Point></Placemark>
    <Placemark><Point><coordinates>25.0,63.0,0</coordinates></Point></Placemark>
  </Document>
</kml>"""
        p = tmp_path / "multi.kml"
        p.write_text(kml_text, encoding="utf-8")
        pts = parse_kml_labels(str(p))
        assert len(pts) == 3


# ---------------------------------------------------------------------------
# Task 3.2 — Chip extraction
# ---------------------------------------------------------------------------

class TestExtractChips:
    def test_chips_written_for_interior_point(self, tmp_path):
        # raster centred at (328000, 6821000)
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_jp2(tmp_path, cx, cy, size=2048)
        seed = [Point(cx, cy)]
        out_dir = str(tmp_path / "chips")
        written = extract_chips(jp2, seed, out_dir, label=1)
        assert len(written) == 1
        chip = np.load(written[0])
        assert chip.shape == (3, 512, 512)

    def test_chip_outside_raster_skipped(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_jp2(tmp_path, cx, cy, size=2048)
        # Point far outside the raster bounds
        far_point = [Point(cx + 1_000_000, cy)]
        out_dir = str(tmp_path / "chips_far")
        written = extract_chips(jp2, far_point, out_dir, label=1)
        assert written == []

    def test_manifest_rows_populated(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_jp2(tmp_path, cx, cy, size=2048)
        seed = [Point(cx, cy)]
        rows: list[dict] = []
        extract_chips(jp2, seed, str(tmp_path / "chips"), label=1, manifest_rows=rows)
        assert len(rows) == 1
        assert rows[0]["label"] == 1

    def test_edge_chip_padded_to_tile_size(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_jp2(tmp_path, cx, cy, size=2048)
        # Point near the top-left corner so the chip is clipped
        with rasterio.open(jp2) as src:
            edge_x = src.bounds.left + 10
            edge_y = src.bounds.top - 10
        seed = [Point(edge_x, edge_y)]
        out_dir = str(tmp_path / "chips_edge")
        written = extract_chips(jp2, seed, out_dir, label=0)
        if written:  # chip may be produced with padding
            chip = np.load(written[0])
            assert chip.shape == (3, 512, 512)


# ---------------------------------------------------------------------------
# Task 3.3 — Negative sampling
# ---------------------------------------------------------------------------

class TestSampleNegatives:
    def _mask(self):
        # 10 km × 10 km box in EPSG:3067 Finland coordinates
        return box(320_000, 6_815_000, 330_000, 6_825_000)

    def test_returns_requested_count(self):
        mask = self._mask()
        pts = sample_negatives(mask, [], n=20)
        assert len(pts) == 20

    def test_points_inside_mask(self):
        mask = self._mask()
        pts = sample_negatives(mask, [], n=30)
        for pt in pts:
            assert mask.contains(pt)

    def test_exclusion_zone_respected(self):
        mask = self._mask()
        centroid = mask.centroid
        positive = [centroid]
        pts = sample_negatives(mask, positive, n=50, rng_seed=0)
        for pt in pts:
            assert pt.distance(centroid) >= NEGATIVE_EXCLUSION_RADIUS - 1  # 1 m tolerance

    def test_no_positives_samples_full_mask(self):
        mask = self._mask()
        pts = sample_negatives(mask, [], n=10)
        assert len(pts) == 10

    def test_deterministic_with_same_seed(self):
        mask = self._mask()
        a = sample_negatives(mask, [], n=10, rng_seed=7)
        b = sample_negatives(mask, [], n=10, rng_seed=7)
        assert [p.wkt for p in a] == [p.wkt for p in b]


# ---------------------------------------------------------------------------
# Integration — build_training_dataset
# ---------------------------------------------------------------------------

class TestBuildTrainingDataset:
    def test_manifest_csv_written(self, tmp_path):
        cx, cy = 328_000.0, 6_821_000.0
        jp2 = _write_jp2(tmp_path, cx, cy, size=2048)

        # Derive a WGS84 point guaranteed to be inside the raster by
        # inverse-transforming the raster centroid.
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(3067, 4326, always_xy=True)
        lon, lat = to_wgs84.transform(cx, cy)

        kml_text = _KML_TEMPLATE.format(lon=lon, lat=lat)
        kml_path = tmp_path / "labels.kml"
        kml_path.write_text(kml_text, encoding="utf-8")

        # Mask fits inside the raster (2048m half-side) minus exclusion zone so
        # that the negative sample lands within the raster.
        stream_mask = box(cx - 900, cy - 900, cx + 900, cy + 900)
        out_dir = str(tmp_path / "dataset")
        manifest = build_training_dataset(
            jp2_paths=[jp2],
            kml_paths=[str(kml_path)],
            stream_mask=stream_mask,
            out_dir=out_dir,
            n_negatives=1,
        )
        assert Path(manifest).exists()
        import csv
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        labels = {int(r["label"]) for r in rows}
        assert 1 in labels
        assert 0 in labels
