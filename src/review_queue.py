"""Detection review queue (active learning), G2.1.

Turns the manual verification work done in "Map & Review" into training
data: load a detections KML, present items to a human reviewer in a useful
order, persist their beaver / not-beaver / skip decisions, and export the
accepted/rejected ones to a labels KML that ``training_data.parse_kml_labels``
can read straight back in for the next training run.

Pure logic, no Gradio — the GUI in ``app.py`` is a thin wrapper around
``ReviewQueue``.

ID scheme
---------
Each item's id encodes its centroid position in EPSG:3067, rounded to the
nearest ``_COORD_ROUND_M`` metres: ``"<x>_<y>"``. Re-running detection on the
same imagery reproduces centroids for the same physical flood area to within
a few metres (the polygon's exact boundary can shift a little between runs,
but its centroid barely moves), so decisions keyed this way survive a
re-detect — the whole point of an active-learning loop. It also means a
decision can be turned back into a KML placemark from the state file alone,
without needing the original detections KML re-parsed.

State file (``<project>/review_state.json``)
---------------------------------------------
::

    {
      "version": 1,
      "decisions": {
        "<id>": {
          "decision": "beaver" | "not_beaver" | "skip",
          "label_type": "dead_forest" | "beaver_flood",   # only for "beaver"
          "timestamp": "2026-09-20T12:34:56.789012+00:00",  # ISO 8601 UTC
          "kml": "/path/to/the/detections.kml/this/decision/came/from"
        },
        ...
      }
    }

Export (``<labels_dir>/review.kml``)
-------------------------------------
A KML with one ``<Folder>`` per label type actually used
(``dead_forest``/``beaver_flood`` for "beaver" decisions — the reviewer's
per-item choice — and ``hard_negatives`` for "not_beaver"), each containing
one ``<Placemark>``/``<Point>`` per decision. "skip" decisions are not
training labels and are not exported. The folder name is exactly what
``training_data.parse_kml_labels`` reads as the label type.
"""

from __future__ import annotations

import datetime
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Polygon

_KML_NS = "http://www.opengis.net/kml/2.2"
_WGS84_TO_3067 = Transformer.from_crs(4326, 3067, always_xy=True)
_3067_TO_WGS84 = Transformer.from_crs(3067, 4326, always_xy=True)

# ID granularity — tolerates small centroid drift between detection re-runs
# while still telling apart distinct nearby flood areas (min detection area
# is 2048 m^2, i.e. tens of metres across).
_COORD_ROUND_M = 5.0

DECISIONS = ("beaver", "not_beaver", "skip")
ORDER_MODES = ("uncertain", "confidence", "area")

DEFAULT_BEAVER_TYPE = "beaver_flood"
BEAVER_LABEL_TYPES = ("dead_forest", "beaver_flood")  # offered to the reviewer
NOT_BEAVER_FOLDER = "hard_negatives"

STATE_FILENAME = "review_state.json"
EXPORT_FILENAME = "review.kml"

_STATE_VERSION = 1


@dataclass(frozen=True)
class ReviewItem:
    """One detection polygon, reduced to what the review queue needs."""
    id: str
    name: str
    lon: float
    lat: float
    x: float
    y: float
    confidence: float | None
    area_m2: float | None
    model: str | None


# --------------------------------------------------------------------------- #
# ID <-> coordinates
# --------------------------------------------------------------------------- #

def _make_id(x: float, y: float) -> str:
    rx = round(x / _COORD_ROUND_M) * _COORD_ROUND_M
    ry = round(y / _COORD_ROUND_M) * _COORD_ROUND_M
    return f"{rx:.0f}_{ry:.0f}"


def _id_to_xy(item_id: str) -> tuple[float, float]:
    x_str, y_str = item_id.split("_")
    return float(x_str), float(y_str)


# --------------------------------------------------------------------------- #
# Loading detections
# --------------------------------------------------------------------------- #

def parse_detections_kml(kml_path: str) -> list[ReviewItem]:
    """Parse a detections KML (as written by ``export.export_kml``) into
    review items.

    A missing file, unparseable XML, or a KML with no usable Placemarks all
    return an empty list rather than raising — callers should treat that as
    "nothing to review", not an error.
    """
    if not kml_path or not Path(kml_path).exists():
        return []
    try:
        root = ET.parse(kml_path).getroot()
    except ET.ParseError:
        return []

    items: list[ReviewItem] = []
    for i, pm in enumerate(root.iter(f"{{{_KML_NS}}}Placemark")):
        name = pm.findtext(f"{{{_KML_NS}}}name") or f"Detection {i + 1}"
        desc = pm.findtext(f"{{{_KML_NS}}}description") or ""
        conf_m = re.search(r"Confidence:\s*([\d.]+)", desc)
        area_m = re.search(r"Area:\s*([\d.]+)", desc)
        model_m = re.search(r"Model:\s*(\w+)", desc)
        confidence = float(conf_m.group(1)) if conf_m else None
        area_m2 = float(area_m.group(1)) if area_m else None
        model = model_m.group(1).lower() if model_m else None

        coords_raw = pm.findtext(f".//{{{_KML_NS}}}coordinates") or ""
        lons: list[float] = []
        lats: list[float] = []
        for part in coords_raw.strip().split():
            vals = part.split(",")
            if len(vals) >= 2:
                try:
                    lons.append(float(vals[0]))
                    lats.append(float(vals[1]))
                except ValueError:
                    continue
        if len(lons) < 3:
            continue

        xs, ys = _WGS84_TO_3067.transform(lons, lats)
        try:
            cx, cy = Polygon(zip(xs, ys)).centroid.coords[0]
        except Exception:
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        clon, clat = _3067_TO_WGS84.transform(cx, cy)

        items.append(ReviewItem(
            id=_make_id(cx, cy), name=name, lon=clon, lat=clat, x=cx, y=cy,
            confidence=confidence, area_m2=area_m2, model=model,
        ))
    return items


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #

def order_items(
    items: list[ReviewItem], mode: str = "uncertain", threshold: float = 0.5,
) -> list[ReviewItem]:
    """Sort review items for presentation.

    mode:
      "uncertain"  — default. |confidence - threshold| ascending (most
                     uncertain / most useful to label first). Items with no
                     parsed confidence sort last.
      "confidence" — highest confidence first (also last for unparsed).
      "area"       — largest area first (also last for unparsed).

    Ties are broken by id, so the order is reproducible across reloads.
    """
    if mode == "confidence":
        def key(it):
            return (0 if it.confidence is not None else 1, -(it.confidence or 0.0), it.id)
    elif mode == "area":
        def key(it):
            return (0 if it.area_m2 is not None else 1, -(it.area_m2 or 0.0), it.id)
    elif mode == "uncertain":
        def key(it):
            if it.confidence is None:
                return (1, 0.0, it.id)
            return (0, abs(it.confidence - threshold), it.id)
    else:
        raise ValueError(f"Unknown ordering mode: {mode!r} (expected one of {ORDER_MODES})")
    return sorted(items, key=key)


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #

def load_state(state_path: str) -> dict:
    """Load the review state JSON, or a fresh empty one if it doesn't exist
    or can't be parsed (never raises — a corrupt state file shouldn't block
    review, just lose its history)."""
    p = Path(state_path)
    if not p.exists():
        return {"version": _STATE_VERSION, "decisions": {}}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return {"version": _STATE_VERSION, "decisions": {}}
    if not isinstance(data, dict) or "decisions" not in data or not isinstance(data["decisions"], dict):
        return {"version": _STATE_VERSION, "decisions": {}}
    return data


def save_state(state_path: str, state: dict) -> None:
    """Write the review state JSON atomically (write to a temp file, then
    rename) so a crash mid-write can't corrupt an existing state file."""
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Export to a labels KML
# --------------------------------------------------------------------------- #

def export_review_kml(decisions: dict, labels_dir: str) -> str:
    """Write ``<labels_dir>/review.kml`` from a decisions dict (the
    ``state["decisions"]`` mapping produced by :class:`ReviewQueue`).

    Beaver decisions go in a folder named by their per-item ``label_type``
    (default ``beaver_flood``); not-beaver decisions go in ``hard_negatives``;
    skips are excluded (they are not a training label). Positions are
    recovered from the decision's item id, so the original detections KML
    does not need to be re-parsed. Always writes a (possibly folder-less but
    still well-formed) KML, even with zero exportable decisions, so
    downstream tooling can rely on the file existing after a call. Returns
    the written path, or "" if ``labels_dir`` is blank.
    """
    if not labels_dir:
        return ""

    folders: dict[str, list[tuple[str, float, float]]] = {}
    for item_id, rec in decisions.items():
        decision = rec.get("decision")
        if decision == "beaver":
            folder = rec.get("label_type") or DEFAULT_BEAVER_TYPE
        elif decision == "not_beaver":
            folder = NOT_BEAVER_FOLDER
        else:
            continue
        try:
            x, y = _id_to_xy(item_id)
        except Exception:
            continue
        lon, lat = _3067_TO_WGS84.transform(x, y)
        folders.setdefault(folder, []).append((item_id, lon, lat))

    ET.register_namespace("", _KML_NS)
    kml = ET.Element(f"{{{_KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{_KML_NS}}}Document")
    ET.SubElement(doc, f"{{{_KML_NS}}}name").text = "CastorDetector Review Decisions"

    for folder_name in sorted(folders):
        folder_el = ET.SubElement(doc, f"{{{_KML_NS}}}Folder")
        ET.SubElement(folder_el, f"{{{_KML_NS}}}name").text = folder_name
        for item_id, lon, lat in folders[folder_name]:
            pm = ET.SubElement(folder_el, f"{{{_KML_NS}}}Placemark")
            ET.SubElement(pm, f"{{{_KML_NS}}}name").text = f"review_{item_id}"
            point = ET.SubElement(pm, f"{{{_KML_NS}}}Point")
            ET.SubElement(point, f"{{{_KML_NS}}}coordinates").text = f"{lon:.6f},{lat:.6f},0"

    out_path = Path(labels_dir) / EXPORT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    with open(out_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)
    return str(out_path)


# --------------------------------------------------------------------------- #
# The queue itself
# --------------------------------------------------------------------------- #

class ReviewQueue:
    """A detections KML reduced to an orderable, resumable review session.

    ``order`` is the full presentation order, fixed at load time. ``cursor``
    indexes into the *pending* (undecided) subsequence of that order — once
    an item gets a decision it drops out of ``pending_ids()``, so the item
    that was "next" slides into the current cursor slot automatically.
    Previous/Next move the cursor without deciding anything.
    """

    def __init__(
        self,
        items: list[ReviewItem],
        state: dict,
        state_path: str,
        order_mode: str = "uncertain",
        threshold: float = 0.5,
    ) -> None:
        self.items: dict[str, ReviewItem] = {it.id: it for it in items}
        self.order: list[str] = [it.id for it in order_items(items, order_mode, threshold)]
        self.state = state
        self.state_path = state_path
        self.order_mode = order_mode
        self.threshold = threshold
        self.cursor = 0

    @classmethod
    def load(
        cls,
        kml_path: str,
        state_path: str,
        order_mode: str = "uncertain",
        threshold: float = 0.5,
    ) -> "ReviewQueue":
        items = parse_detections_kml(kml_path)
        state = load_state(state_path)
        return cls(items, state, state_path, order_mode, threshold)

    @property
    def decisions(self) -> dict:
        return self.state.setdefault("decisions", {})

    def pending_ids(self) -> list[str]:
        decisions = self.decisions
        return [i for i in self.order if i not in decisions]

    def total(self) -> int:
        return len(self.order)

    def current(self) -> ReviewItem | None:
        pending = self.pending_ids()
        if not pending:
            return None
        self.cursor = max(0, min(self.cursor, len(pending) - 1))
        return self.items[pending[self.cursor]]

    def next(self) -> ReviewItem | None:
        pending = self.pending_ids()
        if pending:
            self.cursor = min(self.cursor + 1, len(pending) - 1)
        return self.current()

    def previous(self) -> ReviewItem | None:
        self.cursor = max(self.cursor - 1, 0)
        return self.current()

    def decide(
        self, item_id: str, decision: str, label_type: str | None = None, kml_path: str = "",
    ) -> None:
        if decision not in DECISIONS:
            raise ValueError(f"Unknown decision: {decision!r} (expected one of {DECISIONS})")
        record = {
            "decision": decision,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "kml": kml_path,
        }
        if decision == "beaver":
            record["label_type"] = label_type or DEFAULT_BEAVER_TYPE
        self.decisions[item_id] = record
        save_state(self.state_path, self.state)
        # Cursor is left as-is: the decided item drops out of pending_ids(),
        # so whatever was next slides into this cursor slot on its own.

    def position(self) -> tuple[int, int]:
        """(1-based rank of the current item within the fixed presentation
        order, total items). Tracks both decisions (earlier items dropping
        out shifts the current item's rank up) and plain Previous/Next
        browsing (moving the cursor moves the rank too, since pending items
        are a subsequence of ``order``). (0, 0) when the queue is empty;
        (total, total) once everything has been decided."""
        total = self.total()
        if total == 0:
            return 0, 0
        item = self.current()
        if item is None:
            return total, total
        return self.order.index(item.id) + 1, total

    def reviewed_count(self) -> int:
        return len(self.decisions)

    def beaver_count(self) -> int:
        return sum(1 for d in self.decisions.values() if d.get("decision") == "beaver")

    def not_beaver_count(self) -> int:
        return sum(1 for d in self.decisions.values() if d.get("decision") == "not_beaver")

    def skip_count(self) -> int:
        return sum(1 for d in self.decisions.values() if d.get("decision") == "skip")

    def progress_text(self) -> str:
        return f"{self.reviewed_count()} reviewed, {self.beaver_count()} flagged as beaver"

    def export_kml(self, labels_dir: str) -> str:
        return export_review_kml(self.decisions, labels_dir)


def default_state_path(project_dir: str) -> str:
    return str(Path(project_dir) / STATE_FILENAME)
