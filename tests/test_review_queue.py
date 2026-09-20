"""Tests for src/review_queue.py (G2.1 — detection review queue)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import review_queue
import training_data


_KML_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
_KML_FOOTER = "</Document></kml>\n"


def _placemark(name: str, confidence: float | None, area_m2: float | None,
               lon: float, lat: float, model: str | None = None, size: float = 0.0005) -> str:
    """A square placemark around (lon, lat), matching export.export_kml's
    <name>/<description>/<Polygon> shape (Confidence:/Area:[/Model:] lines)."""
    desc_lines = []
    if model:
        desc_lines.append(f"Model: {model.upper()}")
    if confidence is not None:
        desc_lines.append(f"Confidence: {confidence:.2f}")
    if area_m2 is not None:
        desc_lines.append(f"Area: {area_m2:.0f} m²")
    desc = "\n".join(desc_lines)
    h = size
    coords = (
        f"{lon - h:.6f},{lat - h:.6f},0 {lon + h:.6f},{lat - h:.6f},0 "
        f"{lon + h:.6f},{lat + h:.6f},0 {lon - h:.6f},{lat + h:.6f},0 {lon - h:.6f},{lat - h:.6f},0"
    )
    return (
        f"<Placemark><name>{name}</name><description>{desc}</description>"
        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}"
        "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n"
    )


def _write_detections_kml(tmp_path, placemarks: list[str], filename: str = "detections.kml") -> str:
    path = tmp_path / filename
    path.write_text(_KML_HEADER + "".join(placemarks) + _KML_FOOTER, encoding="utf-8")
    return str(path)


# A fixed point roughly in the middle of Finland, used across tests so
# EPSG:3067 coordinates stay realistic.
LON0, LAT0 = 26.0, 63.0


class TestParseDetectionsKml:
    def test_missing_file_returns_empty(self, tmp_path):
        assert review_queue.parse_detections_kml(str(tmp_path / "nope.kml")) == []

    def test_blank_path_returns_empty(self):
        assert review_queue.parse_detections_kml("") == []

    def test_empty_kml_returns_empty(self, tmp_path):
        path = _write_detections_kml(tmp_path, [])
        assert review_queue.parse_detections_kml(path) == []

    def test_unparseable_kml_returns_empty(self, tmp_path):
        path = tmp_path / "broken.kml"
        path.write_text("<kml><Document><Placemark>", encoding="utf-8")
        assert review_queue.parse_detections_kml(str(path)) == []

    def test_parses_confidence_area_model_and_centroid(self, tmp_path):
        pm = _placemark("Flooded area 1", 0.72, 4096, LON0, LAT0, model="rf")
        path = _write_detections_kml(tmp_path, [pm])
        items = review_queue.parse_detections_kml(path)
        assert len(items) == 1
        it = items[0]
        assert it.confidence == 0.72
        assert it.area_m2 == 4096
        assert it.model == "rf"
        assert abs(it.lon - LON0) < 0.001
        assert abs(it.lat - LAT0) < 0.001
        # EPSG:3067 coordinates should be plausible Finnish easting/northing.
        assert 100_000 < it.x < 700_000
        assert 6_600_000 < it.y < 7_800_000

    def test_missing_confidence_or_model_are_none(self, tmp_path):
        pm = _placemark("Flooded area 1", None, 2048, LON0, LAT0)
        path = _write_detections_kml(tmp_path, [pm])
        items = review_queue.parse_detections_kml(path)
        assert items[0].confidence is None
        assert items[0].model is None

    def test_same_location_yields_same_id_across_runs(self, tmp_path):
        """A re-detect that reproduces (almost) the same polygon should map
        onto the same review id, so prior decisions still apply."""
        pm1 = _placemark("Flooded area 1", 0.6, 2048, LON0, LAT0)
        # Tiny jitter in the polygon footprint, as a re-run's hysteresis might produce.
        pm2 = _placemark("Flooded area 1", 0.61, 2100, LON0 + 0.00001, LAT0, size=0.00052)
        path1 = _write_detections_kml(tmp_path, [pm1], "run1.kml")
        path2 = _write_detections_kml(tmp_path, [pm2], "run2.kml")
        id1 = review_queue.parse_detections_kml(path1)[0].id
        id2 = review_queue.parse_detections_kml(path2)[0].id
        assert id1 == id2

    def test_distinct_locations_yield_distinct_ids(self, tmp_path):
        pm1 = _placemark("A", 0.6, 2048, LON0, LAT0)
        pm2 = _placemark("B", 0.6, 2048, LON0 + 0.01, LAT0 + 0.01)
        path = _write_detections_kml(tmp_path, [pm1, pm2])
        items = review_queue.parse_detections_kml(path)
        assert items[0].id != items[1].id


class TestOrderItems:
    def _items(self):
        mk = review_queue.ReviewItem
        return [
            mk(id="a", name="a", lon=0, lat=0, x=0, y=0, confidence=0.5, area_m2=100, model=None),
            mk(id="b", name="b", lon=0, lat=0, x=1, y=0, confidence=0.95, area_m2=500, model=None),
            mk(id="c", name="c", lon=0, lat=0, x=2, y=0, confidence=0.05, area_m2=50, model=None),
            mk(id="d", name="d", lon=0, lat=0, x=3, y=0, confidence=None, area_m2=None, model=None),
        ]

    def test_uncertain_default_sorts_by_distance_to_threshold(self):
        ordered = review_queue.order_items(self._items(), mode="uncertain", threshold=0.5)
        # a is exactly at threshold (0 distance), then b (0.45), then c (0.45 tie broken
        # by id), unparsed confidence always sorts last.
        assert [it.id for it in ordered] == ["a", "b", "c", "d"]

    def test_confidence_mode_highest_first(self):
        ordered = review_queue.order_items(self._items(), mode="confidence")
        assert [it.id for it in ordered] == ["b", "a", "c", "d"]

    def test_area_mode_largest_first(self):
        ordered = review_queue.order_items(self._items(), mode="area")
        assert [it.id for it in ordered] == ["b", "a", "c", "d"]

    def test_unknown_mode_raises(self):
        import pytest
        with pytest.raises(ValueError):
            review_queue.order_items(self._items(), mode="bogus")


class TestStatePersistence:
    def test_load_state_missing_file_returns_empty(self, tmp_path):
        state = review_queue.load_state(str(tmp_path / "review_state.json"))
        assert state == {"version": 1, "decisions": {}}

    def test_save_then_load_round_trips(self, tmp_path):
        path = str(tmp_path / "review_state.json")
        state = {"version": 1, "decisions": {"1_2": {"decision": "beaver"}}}
        review_queue.save_state(path, state)
        loaded = review_queue.load_state(path)
        assert loaded == state

    def test_corrupt_state_file_returns_empty(self, tmp_path):
        path = tmp_path / "review_state.json"
        path.write_text("{not json", encoding="utf-8")
        state = review_queue.load_state(str(path))
        assert state == {"version": 1, "decisions": {}}


class TestReviewQueue:
    def _kml(self, tmp_path, n=3):
        pms = [
            _placemark(f"Flooded area {i}", conf, 2048, LON0 + i * 0.01, LAT0)
            for i, conf in enumerate([0.9, 0.5, 0.1], start=1)
        ][:n]
        return _write_detections_kml(tmp_path, pms)

    def test_empty_kml_queue_is_empty(self, tmp_path):
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(str(tmp_path / "missing.kml"), state_path)
        assert q.total() == 0
        assert q.current() is None
        assert q.position() == (0, 0)
        assert q.progress_text() == "0 reviewed, 0 flagged as beaver"

    def test_default_order_is_most_uncertain_first(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path, threshold=0.5)
        # confidences 0.9, 0.5, 0.1 -> distances 0.4, 0.0, 0.4 -> 0.5-conf item first.
        assert q.current().confidence == 0.5

    def test_position_reports_rank_and_total(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        assert q.position() == (1, 3)

    def test_decide_advances_queue_and_persists(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path, threshold=0.5)
        first = q.current()
        q.decide(first.id, "beaver", label_type="dead_forest", kml_path=kml)
        assert first.id not in q.pending_ids()
        second = q.current()
        assert second is not None
        assert second.id != first.id
        assert q.reviewed_count() == 1
        assert q.beaver_count() == 1
        assert q.progress_text() == "1 reviewed, 1 flagged as beaver"

        # State file was written to disk.
        assert Path(state_path).exists()
        reloaded_state = review_queue.load_state(state_path)
        rec = reloaded_state["decisions"][first.id]
        assert rec["decision"] == "beaver"
        assert rec["label_type"] == "dead_forest"
        assert rec["kml"] == kml
        assert "timestamp" in rec

    def test_decide_default_beaver_type(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        item = q.current()
        q.decide(item.id, "beaver")
        assert q.decisions[item.id]["label_type"] == review_queue.DEFAULT_BEAVER_TYPE

    def test_decide_unknown_decision_raises(self, tmp_path):
        import pytest
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        item = q.current()
        with pytest.raises(ValueError):
            q.decide(item.id, "maybe")

    def test_resume_hides_already_decided_items(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q1 = review_queue.ReviewQueue.load(kml, state_path)
        first = q1.current()
        q1.decide(first.id, "not_beaver", kml_path=kml)

        # A brand new session (e.g. GUI restarted) loading the same KML/state.
        q2 = review_queue.ReviewQueue.load(kml, state_path)
        assert first.id not in q2.pending_ids()
        assert q2.total() == 3
        assert q2.reviewed_count() == 1
        assert all(it.id != first.id for it in q2.items.values() if it.id in q2.pending_ids())

    def test_skip_also_removes_item_from_pending(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        item = q.current()
        q.decide(item.id, "skip")
        assert item.id not in q.pending_ids()
        assert q.skip_count() == 1
        # Skips don't count as flagged-beaver but do count as reviewed.
        assert q.reviewed_count() == 1
        assert q.beaver_count() == 0

    def test_next_and_previous_move_cursor_without_deciding(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        first = q.current()
        second = q.next()
        assert second is not None
        assert second.id != first.id
        assert q.reviewed_count() == 0  # next() does not decide anything
        back = q.previous()
        assert back.id == first.id

    def test_position_advances_on_plain_navigation(self, tmp_path):
        """Position should reflect where you are browsing, not just how many
        decisions have been recorded — otherwise clicking Next repeatedly
        without deciding would look like it isn't moving."""
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        assert q.position() == (1, 3)
        q.next()
        assert q.position() == (2, 3)
        q.next()
        assert q.position() == (3, 3)
        q.previous()
        assert q.position() == (2, 3)

    def test_all_reviewed_queue_returns_none(self, tmp_path):
        kml = self._kml(tmp_path)
        state_path = str(tmp_path / "review_state.json")
        q = review_queue.ReviewQueue.load(kml, state_path)
        for _ in range(3):
            item = q.current()
            assert item is not None
            q.decide(item.id, "skip")
        assert q.current() is None
        assert q.position() == (3, 3)


class TestExportReviewKml:
    def test_blank_labels_dir_returns_empty_string(self):
        assert review_queue.export_review_kml({}, "") == ""

    def test_export_and_round_trip_through_training_data(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        x1, y1 = 400000.0, 7000000.0
        x2, y2 = 400100.0, 7000100.0
        x3, y3 = 400200.0, 7000200.0
        decisions = {
            review_queue._make_id(x1, y1): {
                "decision": "beaver", "label_type": "dead_forest", "kml": "a.kml",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            review_queue._make_id(x2, y2): {
                "decision": "beaver", "label_type": "beaver_flood", "kml": "a.kml",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            review_queue._make_id(x3, y3): {
                "decision": "not_beaver", "kml": "a.kml",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            "999999_9999999": {  # skip — must not be exported
                "decision": "skip", "kml": "a.kml", "timestamp": "2026-01-01T00:00:00+00:00",
            },
        }
        out_path = review_queue.export_review_kml(decisions, str(labels_dir))
        assert out_path == str(labels_dir / "review.kml")
        assert Path(out_path).exists()

        parsed = training_data.parse_kml_labels(out_path)
        by_type: dict[str, int] = {}
        for _point, ftype in parsed:
            by_type[ftype] = by_type.get(ftype, 0) + 1
        assert by_type == {"dead_forest": 1, "beaver_flood": 1, "hard_negatives": 1}

        # Every returned type is one training_data actually classifies, and
        # with the expected class.
        for _point, ftype in parsed:
            assert ftype in training_data.FEATURE_TO_LABEL
        assert training_data.FEATURE_TO_LABEL["dead_forest"] == 1
        assert training_data.FEATURE_TO_LABEL["beaver_flood"] == 1
        assert training_data.FEATURE_TO_LABEL["hard_negatives"] == 0

        # Positions round-trip to within the id rounding tolerance.
        points_by_type: dict[str, tuple[float, float]] = {
            ftype: (pt.x, pt.y) for pt, ftype in parsed
        }
        px, py = points_by_type["dead_forest"]
        assert abs(px - x1) < review_queue._COORD_ROUND_M
        assert abs(py - y1) < review_queue._COORD_ROUND_M

    def test_export_with_no_exportable_decisions_still_writes_valid_kml(self, tmp_path):
        labels_dir = tmp_path / "labels"
        decisions = {"1_2": {"decision": "skip", "kml": "", "timestamp": "x"}}
        out_path = review_queue.export_review_kml(decisions, str(labels_dir))
        assert Path(out_path).exists()
        assert training_data.parse_kml_labels(out_path) == []

    def test_queue_export_kml_method_uses_current_decisions(self, tmp_path):
        pms = [_placemark("Flooded area 1", 0.6, 2048, LON0, LAT0)]
        kml = _write_detections_kml(tmp_path, pms)
        state_path = str(tmp_path / "review_state.json")
        labels_dir = tmp_path / "labels"
        q = review_queue.ReviewQueue.load(kml, state_path)
        item = q.current()
        q.decide(item.id, "beaver", label_type="beaver_flood", kml_path=kml)
        out_path = q.export_kml(str(labels_dir))
        parsed = training_data.parse_kml_labels(out_path)
        assert len(parsed) == 1
        assert parsed[0][1] == "beaver_flood"


class TestDefaultStatePath:
    def test_default_state_path(self, tmp_path):
        assert review_queue.default_state_path(str(tmp_path)) == str(tmp_path / "review_state.json")
