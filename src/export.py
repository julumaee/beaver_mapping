"""KML export for detected beaver ROIs (flood polygons and dam lines)."""

import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer

_ETRS_TO_WGS84 = Transformer.from_crs(3067, 4326, always_xy=True)
_KML_NS = "http://www.opengis.net/kml/2.2"

# Flood polygon styles — confidence-coded filled polygons (AABBGGRR)
_FLOOD_COLORS = {
    "high":   ("ff00ff00", "8000ff00"),  # (line, fill) green
    "medium": ("ff00ffff", "8000ffff"),  # yellow
    "low":    ("ff0000ff", "800000ff"),  # red
}

# Dam line style — thick brown line, no fill
_DAM_LINE_COLOR = "ff1478ff"   # brown in AABBGGRR
_DAM_LINE_WIDTH = "3"


def _confidence_tier(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def export_kml(
    dam_lines: list[tuple],
    flood_rois: list[tuple],
    output_path: str,
) -> None:
    """
    Write dam lines and flood ROIs to a styled KML file.

    dam_lines  : [(linestring_epsg3067, confidence), ...]
    flood_rois : [(polygon_epsg3067, confidence, area_m2), ...]
    """
    ET.register_namespace("", _KML_NS)
    kml = ET.Element(f"{{{_KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{_KML_NS}}}Document")
    ET.SubElement(doc, f"{{{_KML_NS}}}name").text = "CastorDetector Results"

    _add_styles(doc)

    for i, (line, confidence) in enumerate(dam_lines, 1):
        _add_dam_placemark(doc, i, line, confidence)

    for i, (polygon, confidence, area_m2) in enumerate(flood_rois, 1):
        _add_flood_placemark(doc, i, polygon, confidence, area_m2)

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


def _add_styles(doc: ET.Element) -> None:
    # Dam line style
    style = ET.SubElement(doc, "Style", id="dam")
    line = ET.SubElement(style, "LineStyle")
    ET.SubElement(line, "color").text = _DAM_LINE_COLOR
    ET.SubElement(line, "width").text = _DAM_LINE_WIDTH

    # Flood polygon styles per confidence tier
    for tier, (line_color, fill_color) in _FLOOD_COLORS.items():
        style = ET.SubElement(doc, "Style", id=f"flood_{tier}")
        ls = ET.SubElement(style, "LineStyle")
        ET.SubElement(ls, "color").text = line_color
        ET.SubElement(ls, "width").text = "2"
        ps = ET.SubElement(style, "PolyStyle")
        ET.SubElement(ps, "color").text = fill_color
        ET.SubElement(ps, "outline").text = "1"


def _add_dam_placemark(doc: ET.Element, i: int, line, confidence: float) -> None:
    pm = ET.SubElement(doc, f"{{{_KML_NS}}}Placemark")
    ET.SubElement(pm, f"{{{_KML_NS}}}name").text = f"Dam {i}"
    ET.SubElement(pm, f"{{{_KML_NS}}}description").text = (
        f"Confidence: {confidence:.2f}"
    )
    ET.SubElement(pm, f"{{{_KML_NS}}}styleUrl").text = "#dam"

    ls_el = ET.SubElement(pm, f"{{{_KML_NS}}}LineString")
    ET.SubElement(ls_el, f"{{{_KML_NS}}}tessellate").text = "1"
    ET.SubElement(ls_el, f"{{{_KML_NS}}}coordinates").text = (
        _coords_from_line(line)
    )


def _add_flood_placemark(
    doc: ET.Element, i: int, polygon, confidence: float, area_m2: float
) -> None:
    tier = _confidence_tier(confidence)
    pm = ET.SubElement(doc, f"{{{_KML_NS}}}Placemark")
    ET.SubElement(pm, f"{{{_KML_NS}}}name").text = f"Flooded area {i}"
    ET.SubElement(pm, f"{{{_KML_NS}}}description").text = (
        f"Confidence: {confidence:.2f}\nArea: {area_m2:.0f} m²"
    )
    ET.SubElement(pm, f"{{{_KML_NS}}}styleUrl").text = f"#flood_{tier}"

    poly_el = ET.SubElement(pm, f"{{{_KML_NS}}}Polygon")
    outer = ET.SubElement(poly_el, f"{{{_KML_NS}}}outerBoundaryIs")
    ring = ET.SubElement(outer, f"{{{_KML_NS}}}LinearRing")
    ET.SubElement(ring, f"{{{_KML_NS}}}coordinates").text = (
        _coords_from_polygon(polygon)
    )


def _coords_from_line(line) -> str:
    parts = []
    for x, y in line.coords:
        lon, lat = _ETRS_TO_WGS84.transform(x, y)
        parts.append(f"{lon:.6f},{lat:.6f},0")
    return " ".join(parts)


def _coords_from_polygon(polygon) -> str:
    parts = []
    for x, y in polygon.exterior.coords:
        lon, lat = _ETRS_TO_WGS84.transform(x, y)
        parts.append(f"{lon:.6f},{lat:.6f},0")
    return " ".join(parts)
