"""Tests for the pure helper functions behind the Map tab in src/app.py.

These cover the G0.1 (hydrography window capping), G0.3 (HTML size guard) and
G1.7 (label colour/type grouping) fixes without needing to launch Gradio or
touch real imagery/hydrography data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import app


class TestBboxCenter:
    def test_center_of_bbox(self):
        assert app._bbox_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


class TestCapHydroWindow:
    def test_small_bbox_untouched(self):
        bbox = (0.0, 0.0, 5000.0, 5000.0)
        window, capped = app._cap_hydro_window(bbox)
        assert window == bbox
        assert capped is False

    def test_exact_max_span_untouched(self):
        bbox = (0.0, 0.0, app._HYDRO_MAX_SPAN_M, app._HYDRO_MAX_SPAN_M)
        window, capped = app._cap_hydro_window(bbox)
        assert window == bbox
        assert capped is False

    def test_large_bbox_capped_to_max_span_centred(self):
        # 138x138 km extent, like the real detections.kml in the bug report.
        bbox = (0.0, 0.0, 138_000.0, 138_000.0)
        window, capped = app._cap_hydro_window(bbox)
        assert capped is True
        minx, miny, maxx, maxy = window
        assert maxx - minx == app._HYDRO_MAX_SPAN_M
        assert maxy - miny == app._HYDRO_MAX_SPAN_M
        # Window is centred on the original bbox's centre.
        cx, cy = app._bbox_center(bbox)
        assert (minx + maxx) / 2.0 == cx
        assert (miny + maxy) / 2.0 == cy

    def test_capped_window_stays_within_original_bbox(self):
        # An off-centre, non-square bbox — window must not exceed original extent.
        bbox = (0.0, 0.0, 50_000.0, 12_000.0)
        window, capped = app._cap_hydro_window(bbox)
        assert capped is True
        minx, miny, maxx, maxy = window
        assert minx >= 0.0 and maxx <= 50_000.0
        assert miny >= 0.0 and maxy <= 12_000.0
        assert maxx - minx <= app._HYDRO_MAX_SPAN_M
        assert maxy - miny <= app._HYDRO_MAX_SPAN_M


class TestHtmlSizeMb:
    def test_ascii(self):
        assert app._html_size_mb("a" * 1_000_000) == 1.0

    def test_empty(self):
        assert app._html_size_mb("") == 0.0


class TestFilterDetections:
    _FEATURES = [
        {"name": "a", "confidence": 0.9, "model": "rf", "points": [(0, 0)]},
        {"name": "b", "confidence": 0.6, "model": "rf", "points": [(0, 0)]},
        {"name": "c", "confidence": None, "model": None, "points": [(0, 0)]},
    ]

    def test_zero_threshold_keeps_all(self):
        assert app._filter_detections(self._FEATURES, 0.0) == self._FEATURES

    def test_threshold_drops_low_confidence(self):
        kept = app._filter_detections(self._FEATURES, 0.75)
        names = {f["name"] for f in kept}
        assert names == {"a", "c"}  # b dropped, c kept (no confidence to judge)


class TestLabelColorFor:
    def test_known_groups_get_fixed_colors(self):
        fallback: dict = {}
        assert app._label_color_for("dead_forest", fallback) == app._LABEL_COLORS["dead_forest"]
        assert app._label_color_for("hard_negatives", fallback) == app._LABEL_COLORS["negative"]
        assert app._label_color_for("beaver_flood", fallback) == app._LABEL_COLORS["flood"]
        assert app._label_color_for("flooded_areas", fallback) == app._LABEL_COLORS["flood"]
        assert fallback == {}  # known types never touch the fallback palette

    def test_unknown_type_gets_stable_fallback_color(self):
        fallback: dict = {}
        c1 = app._label_color_for("mystery_type", fallback)
        c2 = app._label_color_for("mystery_type", fallback)
        assert c1 == c2
        assert c1 in app._LABEL_FALLBACK_PALETTE

    def test_distinct_unknown_types_get_distinct_colors(self):
        fallback: dict = {}
        c1 = app._label_color_for("type_a", fallback)
        c2 = app._label_color_for("type_b", fallback)
        assert c1 != c2


class TestParseDetectionsKml:
    _KML = """<?xml version="1.0"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark>
    <name>Flooded area 1 [RF]</name>
    <description>Model: RF
Confidence: 0.91
Area: 512 m2</description>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      25.1234567,62.1234567,0 25.1235,62.1234567,0 25.1235,62.1235,0 25.1234567,62.1234567,0
    </coordinates></LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
</Document></kml>"""

    def test_parses_single_placemark(self, tmp_path):
        kml_path = tmp_path / "det.kml"
        kml_path.write_text(self._KML)
        features = app._parse_detections_kml(str(kml_path))
        assert len(features) == 1
        f = features[0]
        assert f["confidence"] == 0.91
        assert f["model"] == "RF"
        assert len(f["points"]) == 4
        # Rounded to 5 decimals.
        lat, lon = f["points"][0]
        assert lat == round(62.1234567, 5)
        assert lon == round(25.1234567, 5)

    def test_missing_file_returns_empty(self):
        assert app._parse_detections_kml("/no/such/file.kml") == []

    def test_bbox_from_features(self, tmp_path):
        kml_path = tmp_path / "det.kml"
        kml_path.write_text(self._KML)
        features = app._parse_detections_kml(str(kml_path))
        bbox = app._detections_bbox_3067(features)
        assert bbox is not None
        minx, miny, maxx, maxy = bbox
        assert minx < maxx and miny < maxy
