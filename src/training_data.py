"""Training data pipeline: KML label parsing, chip extraction, negative sampling."""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Point, MultiPoint

_KML_NS = "http://www.opengis.net/kml/2.2"

_WGS84_TO_ETRS = Transformer.from_crs(4326, 3067, always_xy=True)


def parse_kml_labels(kml_path: str) -> list[Point]:
    """
    Parse a KML or KMZ file and return Placemark centroids in EPSG:3067.

    Placemarks may contain Point, Polygon, or LineString geometry; the centroid
    of each is used so callers get a uniform list of seed coordinates.
    """
    kml_text = _read_kml_text(kml_path)
    root = ET.fromstring(kml_text)

    ns = _KML_NS if root.tag.startswith("{") else ""

    coords_3067: list[Point] = []
    for pm in root.iter(f"{{{ns}}}Placemark" if ns else "Placemark"):
        raw = _extract_coords(pm, ns)
        if not raw:
            continue
        if len(raw) == 1:
            lon, lat = raw[0]
        else:
            mp = MultiPoint(raw)
            lon, lat = mp.centroid.x, mp.centroid.y
        x, y = _WGS84_TO_ETRS.transform(lon, lat)
        coords_3067.append(Point(x, y))

    return coords_3067


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_kml_text(path: str) -> str:
    """Return KML text from a .kml or .kmz file."""
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


def _extract_coords(placemark_el, ns: str) -> list[tuple[float, float]]:
    """Return a list of (lon, lat) pairs from the first geometry in a Placemark."""
    tag = lambda name: f"{{{ns}}}{name}" if ns else name  # noqa: E731

    for geom_tag in ("Point", "Polygon", "LineString", "MultiGeometry"):
        el = placemark_el.find(f".//{tag(geom_tag)}")
        if el is not None:
            coords_el = el.find(f".//{tag('coordinates')}")
            if coords_el is not None and coords_el.text:
                return _parse_coord_string(coords_el.text)
    return []


def _parse_coord_string(text: str) -> list[tuple[float, float]]:
    """Parse a KML coordinates string into (lon, lat) tuples (elevation dropped)."""
    pairs = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pairs
