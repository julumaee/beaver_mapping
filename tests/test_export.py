"""Tests for src/export.py (Tasks 5.1 and 5.2)."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from export import export_kml, _confidence_tier

_KML_NS = "http://www.opengis.net/kml/2.2"

# A polygon in EPSG:3067 (central Finland area)
_POLY = box(328_000, 6_821_000, 328_256, 6_821_256)
_ROIS = [(_POLY, 0.85, _POLY.area), (_POLY, 0.65, _POLY.area), (_POLY, 0.45, _POLY.area)]


def _parse_kml(path: str):
    return ET.parse(path).getroot()


class TestExportKML:
    def test_file_created(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        assert Path(out).exists()

    def test_placemark_count(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        root = _parse_kml(out)
        pms = root.findall(f".//{{{_KML_NS}}}Placemark")
        assert len(pms) == 3

    def test_coordinates_in_wgs84_range(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        root = _parse_kml(out)
        for coords_el in root.iter(f"{{{_KML_NS}}}coordinates"):
            for token in coords_el.text.split():
                lon, lat, _ = token.split(",")
                assert -180 <= float(lon) <= 180
                assert -90 <= float(lat) <= 90
                # Finnish coordinates should be roughly here
                assert 19 < float(lon) < 32
                assert 59 < float(lat) < 71

    def test_empty_rois(self, tmp_path):
        out = str(tmp_path / "empty.kml")
        export_kml([], out)
        root = _parse_kml(out)
        pms = root.findall(f".//{{{_KML_NS}}}Placemark")
        assert len(pms) == 0

    def test_output_is_valid_xml(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        ET.parse(out)  # raises if invalid

    def test_parent_dirs_created(self, tmp_path):
        out = str(tmp_path / "nested" / "dir" / "out.kml")
        export_kml(_ROIS, out)
        assert Path(out).exists()


class TestStyling:
    def test_style_elements_present(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        root = _parse_kml(out)
        styles = root.findall(f".//{{{_KML_NS}}}Style")
        style_ids = {s.get("id") for s in styles}
        assert {"high", "medium", "low"} == style_ids

    def test_style_url_matches_confidence(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        root = _parse_kml(out)
        pms = root.findall(f".//{{{_KML_NS}}}Placemark")
        urls = [pm.find(f"{{{_KML_NS}}}styleUrl").text for pm in pms]
        assert urls == ["#high", "#medium", "#low"]

    def test_description_contains_confidence_and_area(self, tmp_path):
        out = str(tmp_path / "out.kml")
        export_kml(_ROIS, out)
        root = _parse_kml(out)
        desc = root.find(f".//{{{_KML_NS}}}description").text
        assert "Confidence" in desc
        assert "Area" in desc


class TestConfidenceTier:
    def test_high(self):
        assert _confidence_tier(0.9) == "high"
        assert _confidence_tier(0.8) == "high"

    def test_medium(self):
        assert _confidence_tier(0.79) == "medium"
        assert _confidence_tier(0.6) == "medium"

    def test_low(self):
        assert _confidence_tier(0.59) == "low"
        assert _confidence_tier(0.0) == "low"
