"""KML export for detected beaver ROIs."""

import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import mapping

_ETRS_TO_WGS84 = Transformer.from_crs(3067, 4326, always_xy=True)

_KML_NS = "http://www.opengis.net/kml/2.2"

# KML color: AABBGGRR
_TIER_COLORS = {
    "high":   "ff00ff00",  # green
    "medium": "ff00ffff",  # yellow  (aabbggrr: ff=alpha, 00=blue, ff=green, ff=red → yellow)
    "low":    "ff0000ff",  # red
}


def _confidence_tier(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _reproject_polygon(polygon, transformer: Transformer):
    """Return polygon exterior ring as 'lon,lat,0 ...' KML coordinate string."""
    coords = []
    for x, y in polygon.exterior.coords:
        lon, lat = transformer.transform(x, y)
        coords.append(f"{lon:.6f},{lat:.6f},0")
    return " ".join(coords)


def _add_styles(doc: ET.Element) -> None:
    """Append Style elements for each confidence tier."""
    for tier, color in _TIER_COLORS.items():
        style = ET.SubElement(doc, "Style", id=tier)
        line = ET.SubElement(style, "LineStyle")
        ET.SubElement(line, "color").text = color
        ET.SubElement(line, "width").text = "2"
        poly = ET.SubElement(style, "PolyStyle")
        ET.SubElement(poly, "color").text = _alpha_fill(color)
        ET.SubElement(poly, "outline").text = "1"


def _alpha_fill(color: str) -> str:
    """Return semi-transparent version (alpha 80) of a KML color string."""
    return "80" + color[2:]


def export_kml(rois: list[tuple], output_path: str) -> None:
    """
    Write ROIs to a styled KML file.

    rois: list of (polygon_epsg3067, confidence, area_m2)
    Each Placemark is color-coded by confidence tier and includes area metadata.
    """
    ET.register_namespace("", _KML_NS)
    kml = ET.Element(f"{{{_KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{_KML_NS}}}Document")
    ET.SubElement(doc, f"{{{_KML_NS}}}name").text = "CastorDetector Results"

    _add_styles(doc)

    for i, (polygon, confidence, area_m2) in enumerate(rois, 1):
        tier = _confidence_tier(confidence)

        pm = ET.SubElement(doc, f"{{{_KML_NS}}}Placemark")
        ET.SubElement(pm, f"{{{_KML_NS}}}name").text = f"Beaver ROI {i}"
        ET.SubElement(pm, f"{{{_KML_NS}}}description").text = (
            f"Confidence: {confidence:.2f}\nArea: {area_m2:.0f} m²"
        )
        ET.SubElement(pm, f"{{{_KML_NS}}}styleUrl").text = f"#{tier}"

        poly_el = ET.SubElement(pm, f"{{{_KML_NS}}}Polygon")
        outer = ET.SubElement(poly_el, f"{{{_KML_NS}}}outerBoundaryIs")
        ring = ET.SubElement(outer, f"{{{_KML_NS}}}LinearRing")
        ET.SubElement(ring, f"{{{_KML_NS}}}coordinates").text = (
            _reproject_polygon(polygon, _ETRS_TO_WGS84)
        )

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)
