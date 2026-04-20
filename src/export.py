"""KML export for detected beaver flood ROI polygons."""

import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer

_ETRS_TO_WGS84 = Transformer.from_crs(3067, 4326, always_xy=True)
_KML_NS = "http://www.opengis.net/kml/2.2"

# KML colors are AABBGGRR.
# Confidence tiers used when model_type is unknown (legacy / RF-only export).
_CONFIDENCE_COLORS = {
    "high":   ("ff00ff00", "8000ff00"),  # green
    "medium": ("ff00ffff", "8000ffff"),  # yellow
    "low":    ("ff0000ff", "800000ff"),  # red
}

# Model-type colors: RF=red, CNN=blue, both=purple.
_MODEL_COLORS = {
    "rf":   ("ff0000ff", "800000ff"),   # red
    "cnn":  ("ffff0000", "80ff0000"),   # blue
    "both": ("ffff00ff", "80ff00ff"),   # purple
}


def export_kml(flood_rois: list[tuple], output_path: str) -> None:
    """
    Write flood ROI polygons to a styled KML file.

    Each element of flood_rois may be:
      (polygon, confidence, area_m2)              — confidence-coded colour (legacy)
      (polygon, confidence, area_m2, model_type)  — model-type colour (rf/cnn/both)
    """
    ET.register_namespace("", _KML_NS)
    kml = ET.Element(f"{{{_KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{_KML_NS}}}Document")
    ET.SubElement(doc, f"{{{_KML_NS}}}name").text = "CastorDetector Results"

    _add_styles(doc)

    for i, roi in enumerate(flood_rois, 1):
        polygon, confidence, area_m2 = roi[0], roi[1], roi[2]
        model_type = roi[3] if len(roi) > 3 else None
        _add_placemark(doc, i, polygon, confidence, area_m2, model_type)

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


def _add_styles(doc: ET.Element) -> None:
    for style_id, (line_color, fill_color) in {
        **{f"flood_{tier}": colors for tier, colors in _CONFIDENCE_COLORS.items()},
        **{f"model_{mtype}": colors for mtype, colors in _MODEL_COLORS.items()},
    }.items():
        style = ET.SubElement(doc, "Style", id=style_id)
        ls = ET.SubElement(style, "LineStyle")
        ET.SubElement(ls, "color").text = line_color
        ET.SubElement(ls, "width").text = "2"
        ps = ET.SubElement(style, "PolyStyle")
        ET.SubElement(ps, "color").text = fill_color
        ET.SubElement(ps, "outline").text = "1"


def _add_placemark(
    doc: ET.Element,
    i: int,
    polygon,
    confidence: float,
    area_m2: float,
    model_type: str | None,
) -> None:
    if model_type in _MODEL_COLORS:
        style_url = f"#model_{model_type}"
        label = f"Flooded area {i} [{model_type.upper()}]"
        description = (
            f"Model: {model_type.upper()}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Area: {area_m2:.0f} m²"
        )
    else:
        tier = _confidence_tier(confidence)
        style_url = f"#flood_{tier}"
        label = f"Flooded area {i}"
        description = f"Confidence: {confidence:.2f}\nArea: {area_m2:.0f} m²"

    pm = ET.SubElement(doc, f"{{{_KML_NS}}}Placemark")
    ET.SubElement(pm, f"{{{_KML_NS}}}name").text = label
    ET.SubElement(pm, f"{{{_KML_NS}}}description").text = description
    ET.SubElement(pm, f"{{{_KML_NS}}}styleUrl").text = style_url

    poly_el = ET.SubElement(pm, f"{{{_KML_NS}}}Polygon")
    outer = ET.SubElement(poly_el, f"{{{_KML_NS}}}outerBoundaryIs")
    ring = ET.SubElement(outer, f"{{{_KML_NS}}}LinearRing")
    ET.SubElement(ring, f"{{{_KML_NS}}}coordinates").text = _coords_from_polygon(polygon)


def _confidence_tier(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _coords_from_polygon(polygon) -> str:
    parts = []
    for x, y in polygon.exterior.coords:
        lon, lat = _ETRS_TO_WGS84.transform(x, y)
        parts.append(f"{lon:.6f},{lat:.6f},0")
    return " ".join(parts)
