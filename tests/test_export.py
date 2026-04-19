"""Tests for src/export.py."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from export import export_kml, _confidence_tier

_KML_NS = "http://www.opengis.net/kml/2.2"
_POLY = box(328_000, 6_821_000, 328_256, 6_821_256)
_LINE = LineString([(328_000, 6_821_128), (328_060, 6_821_128)])

_DAMS = [(_LINE, 0.85), (_LINE, 0.65)]
_FLOODS = [(_POLY, 0.85, _POLY.area), (_POLY, 0.65, _POLY.area), (_POLY, 0.45, _POLY.area)]


def _parse(path: str):
    return ET.parse(path).getroot()


class TestExportKML:
    def test_file_created(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        assert Path(out).exists()

    def test_placemark_count(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        pms = _parse(out).findall(f".//{{{_KML_NS}}}Placemark")
        assert len(pms) == len(_DAMS) + len(_FLOODS)

    def test_empty_both(self, tmp_path):
        out = str(tmp_path / "empty.kml")
        export_kml([], [], out)
        pms = _parse(out).findall(f".//{{{_KML_NS}}}Placemark")
        assert len(pms) == 0

    def test_dam_uses_linestring(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, [], out)
        root = _parse(out)
        ls = root.findall(f".//{{{_KML_NS}}}LineString")
        assert len(ls) == len(_DAMS)

    def test_flood_uses_polygon(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml([], _FLOODS, out)
        root = _parse(out)
        polys = root.findall(f".//{{{_KML_NS}}}Polygon")
        assert len(polys) == len(_FLOODS)

    def test_coordinates_in_wgs84_range(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        for coords_el in _parse(out).iter(f"{{{_KML_NS}}}coordinates"):
            for token in coords_el.text.split():
                lon, lat, _ = token.split(",")
                assert 19 < float(lon) < 32
                assert 59 < float(lat) < 71

    def test_output_is_valid_xml(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        ET.parse(out)

    def test_styles_present(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        styles = {s.get("id") for s in _parse(out).findall(f".//{{{_KML_NS}}}Style")}
        assert "dam" in styles
        assert "flood_high" in styles
        assert "flood_medium" in styles
        assert "flood_low" in styles

    def test_parent_dirs_created(self, tmp_path):
        out = str(tmp_path / "a" / "b" / "out.kml")
        export_kml(_DAMS, _FLOODS, out)
        assert Path(out).exists()


class TestConfidenceTier:
    def test_high(self):
        assert _confidence_tier(0.9) == "high"
        assert _confidence_tier(0.8) == "high"

    def test_medium(self):
        assert _confidence_tier(0.79) == "medium"
        assert _confidence_tier(0.6) == "medium"

    def test_low(self):
        assert _confidence_tier(0.59) == "low"
