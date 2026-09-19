"""Gradio web UI for CastorDetector."""
import sys
from pathlib import Path

# Key third-party modules the app needs; import names differ from pip package
# names for a couple of these (scikit-learn -> sklearn, scikit-image -> skimage).
_REQUIRED_MODULES = ("gradio", "folium", "rasterio", "geopandas", "sklearn", "skimage")


def _check_dependencies() -> None:
    missing = []
    for mod in _REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(
            f"Missing dependencies: {', '.join(missing)}\n"
            "Install them with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

_check_dependencies()

import argparse
import csv
import json
import queue
import tempfile
import threading
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

# --------------------------------------------------------------------------- #
# Persistent settings
# --------------------------------------------------------------------------- #

_SETTINGS_PATH = Path(__file__).parent.parent / "data" / "settings.json"

_SETTINGS_KEYS = [
    "rf_imagery", "rf_labels", "rf_model", "rf_hydro", "rf_chips",
    "rf_flood_samples", "rf_hydro_negatives",
    "cnn_imagery", "cnn_labels", "cnn_model", "cnn_norm_stats", "cnn_hydro",
    "cnn_epochs", "cnn_lr",
    "det_imagery", "det_output", "det_rf_model", "det_cnn_model",
    "det_norm_stats", "det_hydro",
    "det_method", "det_threshold",
    "ev_manifest", "ev_rf_model",
    "ev_radius", "ev_per_class",
    "cmp_manifest", "cmp_rf_model", "cmp_cnn_model", "cmp_norm_stats",
    "cmp_test_frac",
    "diag_imagery", "diag_rf_model",
    "ov_imagery", "ov_labels", "ov_models_dir", "ov_chips",
    "map_kml", "map_labels", "map_hydro",
]


def _load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            with open(_SETTINGS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_settings(settings: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def handle_save_settings(*values) -> str:
    settings = dict(zip(_SETTINGS_KEYS, values))
    try:
        _save_settings(settings)
        return f"Defaults saved to {_SETTINGS_PATH}"
    except Exception as exc:
        return f"ERROR saving settings: {exc}"


_s = _load_settings()


_MAX_LOG_HISTORY = 5


def _append_log_history(log: str, history: list) -> tuple[list, str]:
    """Prepend the completed run log to the history list (newest first)."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"─── {ts} ───\n{(log or '').rstrip()}"
    history = list(history or [])[-(_MAX_LOG_HISTORY - 1):]
    history.append(entry)
    return history, "\n\n".join(reversed(history))


def _file_to_path(f) -> str:
    """Return a file path string from whatever gr.UploadButton hands back."""
    if f is None:
        return ""
    if isinstance(f, list):
        f = f[0] if f else None
        if f is None:
            return ""
    return f.name if hasattr(f, "name") else str(f)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _find_files(path: str, suffix: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == suffix.lower():
        # User passed a single file path — give a clear error if it doesn't exist
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path!r}")
        return [str(p)]
    if not p.exists():
        return []
    if p.is_file():
        return [str(p)]
    return sorted(str(f) for f in p.rglob(f"*{suffix}"))


class _NullContext:
    """Context manager that creates and returns a fixed directory path."""
    def __init__(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self._path = path

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *_) -> None:
        pass


class _QueueWriter:
    """Redirect stdout lines into a queue for live streaming."""
    def __init__(self, q: queue.Queue) -> None:
        self._q = q
        self._buf = ""

    def write(self, s: str) -> None:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(line + "\n")

    def flush(self) -> None:
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""


def _stream(fn, *args, **kwargs):
    """Run fn in a background thread, yielding its stdout output line-by-line."""
    q: queue.Queue = queue.Queue()

    def _run() -> None:
        old = sys.stdout
        sys.stdout = _QueueWriter(q)
        try:
            fn(*args, **kwargs)
        except SystemExit as exc:
            if exc.code:
                q.put(f"ERROR: {exc.code}\n")
        except Exception as exc:
            q.put(f"ERROR: {exc}\n")
        finally:
            sys.stdout = old
            q.put(None)

    threading.Thread(target=_run, daemon=True).start()
    accumulated = ""
    while True:
        line = q.get()
        if line is None:
            break
        accumulated += line
        yield accumulated


# --------------------------------------------------------------------------- #
# Train RF backend
# --------------------------------------------------------------------------- #

def _do_train_rf(
    imagery_dir: str,
    labels_dir: str,
    model_path: str,
    hydro_dir: str,
    chip_dir: str,
    augment: int,
    flood_samples: int,
    hydro_negatives: bool,
) -> None:
    from cli import cmd_train
    cmd_train(argparse.Namespace(
        imagery=imagery_dir,
        labels=labels_dir,
        model=model_path,
        hydro=hydro_dir or None,
        chip_dir=chip_dir or None,
        augment_positives=augment,
        flood_samples=flood_samples,
        no_hydro_negatives=not hydro_negatives,
    ))


def handle_train_rf(
    imagery_dir: str,
    labels_dir: str,
    model_path: str,
    hydro_dir: str,
    chip_dir: str,
    augment: float,
    flood_samples: float,
    hydro_negatives: bool,
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory or .jp2 file is required."; return
    if not labels_dir or not labels_dir.strip():
        yield "ERROR: Labels directory is required."; return
    if not model_path or not model_path.strip():
        yield "ERROR: Model output path is required."; return
    if int(flood_samples) > 0 and not (hydro_dir and hydro_dir.strip()):
        yield "ERROR: Hydrography directory is required when flood samples > 0."; return
    yield from _stream(
        _do_train_rf,
        imagery_dir.strip(), labels_dir.strip(), model_path.strip(),
        hydro_dir.strip() if hydro_dir else "",
        chip_dir.strip() if chip_dir else "",
        int(augment), int(flood_samples), bool(hydro_negatives),
    )


# --------------------------------------------------------------------------- #
# Train CNN backend
# --------------------------------------------------------------------------- #

def _do_train_cnn(
    imagery_dir: str,
    labels_dir: str,
    model_path: str,
    norm_stats_path: str,
    hydro_dir: str,
    epochs: int,
    lr: float,
) -> None:
    from cli import cmd_cnn_train
    cmd_cnn_train(argparse.Namespace(
        imagery=imagery_dir,
        labels=labels_dir,
        model=model_path,
        norm_stats=norm_stats_path,
        hydro=hydro_dir or None,
        epochs=epochs,
        lr=lr,
    ))


def handle_train_cnn(
    imagery_dir: str,
    labels_dir: str,
    model_path: str,
    norm_stats_path: str,
    hydro_dir: str,
    epochs: float,
    lr: float,
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory or .jp2 file is required."; return
    if not labels_dir or not labels_dir.strip():
        yield "ERROR: Labels directory is required."; return
    if not model_path or not model_path.strip():
        yield "ERROR: Model output path is required."; return
    if not norm_stats_path or not norm_stats_path.strip():
        yield "ERROR: Norm stats path is required."; return
    yield from _stream(
        _do_train_cnn,
        imagery_dir.strip(), labels_dir.strip(), model_path.strip(),
        norm_stats_path.strip(),
        hydro_dir.strip() if hydro_dir else "",
        int(epochs), float(lr),
    )


# --------------------------------------------------------------------------- #
# Detect backend
# --------------------------------------------------------------------------- #

def _do_detect(
    imagery_dir: str,
    method: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    hydro_dir: str,
    threshold: float,
    output_path: str,
) -> None:
    from cli import cmd_detect
    cmd_detect(argparse.Namespace(
        imagery=imagery_dir,
        method=method,
        rf_model=rf_model_path or None,
        cnn_model=cnn_model_path or None,
        norm_stats=norm_stats_path or None,
        hydro=hydro_dir or None,
        threshold=threshold,
        output=output_path,
    ))


def handle_detect(
    imagery_dir: str,
    method: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    hydro_dir: str,
    threshold: float,
    output_path: str,
    progress: gr.Progress = gr.Progress(),
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory or .jp2 file is required.", None; return
    if not output_path or not output_path.strip():
        yield "ERROR: Output KML path is required.", None; return
    if method in ("rf", "both") and (not rf_model_path or not rf_model_path.strip()):
        yield "ERROR: RF model path is required for this method.", None; return
    if method in ("cnn", "both") and (not cnn_model_path or not cnn_model_path.strip()):
        yield "ERROR: CNN model path is required for this method.", None; return
    if method in ("cnn", "both") and (not norm_stats_path or not norm_stats_path.strip()):
        yield "ERROR: Norm stats path is required for this method.", None; return
    out = output_path.strip()
    try:
        total_tiles = len(_find_files(imagery_dir.strip(), ".jp2"))
    except Exception:
        total_tiles = 0
    tiles_done = 0
    progress(0, desc=f"0 / {total_tiles} tiles")
    last_log = ""
    for log in _stream(
        _do_detect,
        imagery_dir.strip(), method,
        rf_model_path.strip(), cnn_model_path.strip(), norm_stats_path.strip(),
        hydro_dir.strip(), float(threshold), out,
    ):
        new = log[len(last_log):]
        tiles_done += new.count("Processing ")
        if total_tiles > 0:
            progress(min(tiles_done / total_tiles, 0.99), desc=f"{tiles_done} / {total_tiles} tiles")
        last_log = log
        yield log, None, gr.update()
    progress(1.0, desc="Done")
    kml_exists = out and Path(out).exists()
    yield last_log, (out if kml_exists else None), (out if kml_exists else gr.update())


# --------------------------------------------------------------------------- #
# Map view
# --------------------------------------------------------------------------- #

_MAP_NS = "http://www.opengis.net/kml/2.2"

# Folder-based label type (as returned by training_data.parse_kml_labels) -> colour
# group. Mirrors training_data.FEATURE_TO_LABEL's name variants so the map/Overview
# agree with what training actually uses, plus dam/lodge/other (excluded from
# training but still labelled in KML).
_LABEL_TYPE_GROUPS: dict[str, str] = {
    "dead_forest":    "dead_forest",
    "flood":          "flood",
    "flooded_areas":  "flood",
    "beaver_flood":   "flood",
    "wet_forest":     "wet_forest",
    "negative":       "negative",
    "negatives":      "negative",
    "hard_negative":  "negative",
    "hard_negatives": "negative",
    "dam":            "dam",
    "lodge":          "lodge",
    "other":          "other",
}
_LABEL_COLORS: dict[str, str] = {
    "dead_forest": "#8b4513",  # brown — standing dead trees
    "flood":       "#00aaff",  # blue — open water
    "wet_forest":  "#ff7700",  # orange — saturated forest
    "negative":    "#888888",  # grey — hard negative
    "dam":         "#a0522d",  # sienna
    "lodge":       "#654321",  # dark brown
    "other":       "#bbbbbb",  # light grey
}
# Cycled through for label types not covered above, so new folder names still render.
_LABEL_FALLBACK_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6",
]


def _label_color_for(ftype: str, fallback_assignment: dict[str, str]) -> str:
    """Colour for a label type: fixed colour for known groups, else a stable
    colour drawn from the fallback palette (assigned once per type per call)."""
    group = _LABEL_TYPE_GROUPS.get(ftype)
    if group is not None:
        return _LABEL_COLORS[group]
    if ftype not in fallback_assignment:
        fallback_assignment[ftype] = (
            _LABEL_FALLBACK_PALETTE[len(fallback_assignment) % len(_LABEL_FALLBACK_PALETTE)]
        )
    return fallback_assignment[ftype]


def _confidence_color(conf: float | None) -> str:
    if conf is None:
        return "#888888"
    if conf >= 0.85:
        return "#00cc44"  # green
    if conf >= 0.75:
        return "#ffcc00"  # yellow
    if conf >= 0.65:
        return "#ff4400"  # red-orange
    return "#888888"      # grey — below typical useful threshold


def _detection_stats(kml_path: str) -> str:
    """Parse a detections KML and return a confidence/area summary string."""
    import re
    if not kml_path or not Path(kml_path).exists():
        return ""
    counts = {"≥ 0.85": 0, "0.75–0.85": 0, "0.65–0.75": 0, "< 0.65": 0}
    total_area = 0.0
    n = 0
    try:
        root = ET.parse(kml_path).getroot()
        for pm in root.iter(f"{{{_MAP_NS}}}Placemark"):
            desc = pm.findtext(f"{{{_MAP_NS}}}description") or ""
            cm = re.search(r"Confidence:\s*([\d.]+)", desc)
            am = re.search(r"Area:\s*([\d.]+)", desc)
            if cm:
                conf = float(cm.group(1))
                n += 1
                if conf >= 0.85:   counts["≥ 0.85"]    += 1
                elif conf >= 0.75: counts["0.75–0.85"] += 1
                elif conf >= 0.65: counts["0.65–0.75"] += 1
                else:              counts["< 0.65"]    += 1
            if am:
                total_area += float(am.group(1))
    except Exception as exc:
        return f"Error reading stats: {exc}"
    if n == 0:
        return "No detections in KML."
    lines = [
        f"Total detections : {n}",
        f"Total area       : {total_area / 1e4:.2f} ha  ({total_area:.0f} m²)",
        "",
        "Confidence breakdown:",
        f"  ≥ 0.85   (green)  : {counts['≥ 0.85']}",
        f"  0.75–0.85 (yellow): {counts['0.75–0.85']}",
        f"  0.65–0.75 (red)   : {counts['0.65–0.75']}",
        f"  < 0.65   (grey)   : {counts['< 0.65']}",
    ]
    return "\n".join(lines)


def _parse_detections_kml(kml_path: str) -> list[dict]:
    """Parse a detections KML exactly once.

    Returns a list of {"name", "confidence" (float|None), "model" (str|None),
    "points" (list of (lat, lon) rounded to 5 decimals, ~1 m)} — enough to both
    render the layer and compute its bbox without re-parsing the file.
    """
    import re
    features: list[dict] = []
    try:
        root = ET.parse(kml_path).getroot()
    except Exception as exc:
        print(f"Warning: could not parse detections KML: {exc}")
        return features
    for pm in root.iter(f"{{{_MAP_NS}}}Placemark"):
        name = pm.findtext(f"{{{_MAP_NS}}}name") or "Detection"
        desc = pm.findtext(f"{{{_MAP_NS}}}description") or ""
        conf_m = re.search(r"Confidence:\s*([\d.]+)", desc)
        confidence = float(conf_m.group(1)) if conf_m else None
        model_m = re.search(r"Model:\s*(\w+)", desc)
        model = model_m.group(1) if model_m else None
        coords_raw = pm.findtext(f".//{{{_MAP_NS}}}coordinates") or ""
        points: list[tuple[float, float]] = []
        for part in coords_raw.strip().split():
            vals = part.split(",")
            if len(vals) >= 2:
                points.append((round(float(vals[1]), 5), round(float(vals[0]), 5)))
        if len(points) >= 3:
            features.append({
                "name": name, "confidence": confidence, "model": model, "points": points,
            })
    return features


def _filter_detections(features: list[dict], threshold: float) -> list[dict]:
    """Keep only features with confidence >= threshold (features with no parsed
    confidence are always kept, since we can't judge them)."""
    if not threshold:
        return features
    return [f for f in features if f["confidence"] is None or f["confidence"] >= threshold]


def _detections_bbox_3067(features: list[dict]):
    """Return (minx, miny, maxx, maxy) in EPSG:3067 from already-parsed detection
    features, or None."""
    if not features:
        return None
    lats = [p[0] for f in features for p in f["points"]]
    lons = [p[1] for f in features for p in f["points"]]
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 3067, always_xy=True)
    xs, ys = t.transform(lons, lats)
    return (min(xs), min(ys), max(xs), max(ys))


def _labels_bbox_3067(labels_dir: str):
    """Return (minx, miny, maxx, maxy) in EPSG:3067 from all labels in a directory
    (via training_data.parse_kml_labels, already EPSG:3067), or None."""
    import training_data
    xs: list[float] = []
    ys: list[float] = []
    for kml_path in _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz"):
        try:
            for point, _ in training_data.parse_kml_labels(kml_path):
                xs.append(point.x)
                ys.append(point.y)
        except Exception as exc:
            print(f"Warning: could not parse {kml_path}: {exc}")
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = bbox
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)


_HYDRO_MAX_SPAN_M = 10_000.0  # cap the hydrography load window to 10x10 km


def _cap_hydro_window(
    bbox: tuple[float, float, float, float],
    max_span: float = _HYDRO_MAX_SPAN_M,
) -> tuple[tuple[float, float, float, float], bool]:
    """Clip bbox to at most max_span x max_span metres, centred on the bbox's own
    centre. Returns (window_bbox, was_capped); bbox is returned unchanged if it
    already fits."""
    minx, miny, maxx, maxy = bbox
    width, height = maxx - minx, maxy - miny
    if width <= max_span and height <= max_span:
        return bbox, False
    cx, cy = _bbox_center(bbox)
    half = max_span / 2.0
    window = (
        max(minx, cx - half), max(miny, cy - half),
        min(maxx, cx + half), min(maxy, cy + half),
    )
    return window, True


_MAX_MAP_HTML_MB = 30.0


def _html_size_mb(html: str) -> float:
    return len(html.encode("utf-8")) / 1e6


def _add_detections_geojson_layer(m, features: list[dict]) -> list[tuple[float, float]]:
    """Render all detections as ONE folium.GeoJson layer (not one Polygon per
    feature) — the only way to keep HTML size manageable for thousands of
    detections. Style/tooltip are driven by per-feature properties."""
    import folium
    if not features:
        return []
    geo_features = []
    bounds: list[tuple[float, float]] = []
    for f in features:
        ring = [[lon, lat] for lat, lon in f["points"]]
        geo_features.append({
            "type": "Feature",
            "properties": {
                "name": f["name"],
                "confidence": f["confidence"],
                "model": f["model"] or "",
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
        bounds.extend(f["points"])
    geojson = {"type": "FeatureCollection", "features": geo_features}

    def _style(feature):
        color = _confidence_color(feature["properties"].get("confidence"))
        return {"color": color, "fillColor": color, "fillOpacity": 0.35, "weight": 1.5}

    group = folium.FeatureGroup(name="Detections", show=True)
    folium.GeoJson(
        geojson,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "model", "confidence"],
            aliases=["Detection", "Model", "Confidence"],
        ),
    ).add_to(group)
    group.add_to(m)
    return bounds


def _add_labels_layer(m, labels_dir: str) -> tuple[list[tuple[float, float]], dict[str, tuple[int, str]]]:
    """Render training-label points. Type comes from training_data.parse_kml_labels
    (KML *folder* name), matching how training actually derives the class — not
    the placemark name, which was the old (wrong) behaviour.

    Returns (bounds, {type: (count, colour)}) so the caller can build a legend
    from the types actually present.
    """
    import folium
    import training_data
    from pyproj import Transformer

    group = folium.FeatureGroup(name="Training labels", show=True)
    bounds: list[tuple[float, float]] = []
    type_counts: dict[str, int] = {}
    fallback_colors: dict[str, str] = {}
    to_wgs84 = Transformer.from_crs(3067, 4326, always_xy=True)

    for kml_path in _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz"):
        try:
            for point, ftype in training_data.parse_kml_labels(kml_path):
                lon, lat = to_wgs84.transform(point.x, point.y)
                pt = (lat, lon)
                bounds.append(pt)
                type_counts[ftype] = type_counts.get(ftype, 0) + 1
                color = _label_color_for(ftype, fallback_colors)
                folium.CircleMarker(
                    location=pt,
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    weight=1.5,
                    tooltip=ftype or "unknown",
                ).add_to(group)
        except Exception as exc:
            print(f"Warning: could not parse {kml_path}: {exc}")
    group.add_to(m)
    legend = {t: (n, _label_color_for(t, fallback_colors)) for t, n in type_counts.items()}
    return bounds, legend


_HYDRO_SIMPLIFY_M = 5.0          # metres; normal (small-extent) simplify tolerance
_HYDRO_SIMPLIFY_CAPPED_M = 30.0  # metres; heavier simplify when the window was capped
_HYDRO_LAYERS = ("virtavesialue", "virtavesikapea")
# Narrow-stream lines are skipped entirely when the window was capped — they
# dominate vertex count for little visual payoff at that zoom level.
_HYDRO_LAYERS_CAPPED = ("virtavesialue",)


def _add_hydro_layer(
    m,
    hydro_dir: str,
    bbox_3067: tuple,
    capped: bool = False,
) -> list[tuple[float, float]]:
    import fiona
    import folium
    import geopandas as gpd
    import pandas as pd
    from masking import _resolve_files

    files = _resolve_files(hydro_dir)
    if not files:
        return []

    wanted_layers = _HYDRO_LAYERS_CAPPED if capped else _HYDRO_LAYERS
    simplify_m = _HYDRO_SIMPLIFY_CAPPED_M if capped else _HYDRO_SIMPLIFY_M

    gdfs: list[gpd.GeoDataFrame] = []
    for f in files:
        try:
            available = fiona.listlayers(str(f))
            layers = [l for l in wanted_layers if l in available] or available[:1]
        except Exception:
            layers = [None]
        for layer in layers:
            try:
                kw: dict = {"bbox": bbox_3067}
                gdf = (
                    gpd.read_file(str(f), layer=layer, **kw)
                    if layer else
                    gpd.read_file(str(f), **kw)
                )
                if gdf.crs is not None and gdf.crs.to_epsg() != 3067:
                    gdf = gdf.to_crs(epsg=3067)
                gdfs.append(gdf[["geometry"]])
            except Exception as exc:
                print(f"  Warning: could not read hydro layer from {f}: {exc}")

    if not gdfs:
        return []

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0].reset_index(drop=True),
        crs="EPSG:3067",
    )

    # Simplify in projected CRS (metres) before reprojection — much smaller GeoJSON
    combined = combined.copy()
    combined["geometry"] = combined.geometry.simplify(simplify_m, preserve_topology=True)
    combined = combined[~combined.geometry.is_empty & combined.geometry.notna()]
    if combined.empty:
        return []

    combined_wgs84 = combined.to_crs(epsg=4326)

    vectors_group = folium.FeatureGroup(name="Hydrography", show=True)
    folium.GeoJson(
        combined_wgs84.__geo_interface__,
        style_function=lambda _: {
            "color": "#1a6aa8",
            "fillColor": "#4da6ff",
            "fillOpacity": 0.30,
            "weight": 1.5,
        },
    ).add_to(vectors_group)
    vectors_group.add_to(m)

    minx, miny, maxx, maxy = combined_wgs84.total_bounds
    return [(miny, minx), (maxy, maxx)]


def _build_map(
    kml_path: str,
    labels_dir: str = "",
    hydro_dir: str = "",
    basemap: str = "Satellite",
    show_detections: bool = True,
    show_labels: bool = True,
    show_hydro: bool = True,
    confidence_threshold: float = 0.0,
) -> str:
    import folium
    satellite_url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services"
        "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    m = folium.Map(location=[65.0, 26.0], zoom_start=6, tiles=None)

    if basemap == "Satellite":
        folium.TileLayer(satellite_url, attr="Esri", name="Satellite").add_to(m)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.TileLayer(satellite_url, attr="Esri", name="Satellite").add_to(m)

    bounds: list[tuple[float, float]] = []

    # Parse the detections KML once — reused for both the layer and the hydro bbox.
    detection_features: list[dict] = []
    if kml_path and Path(kml_path).exists():
        detection_features = _parse_detections_kml(kml_path)
        if confidence_threshold:
            detection_features = _filter_detections(detection_features, confidence_threshold)

    hydro_notice: str | None = None
    if show_hydro and hydro_dir:
        source_bbox = _detections_bbox_3067(detection_features) if detection_features else None
        source_label = "detections"
        if source_bbox is None and labels_dir:
            source_bbox = _labels_bbox_3067(labels_dir)
            source_label = "labels"
        if source_bbox is not None:
            window_bbox, was_capped = _cap_hydro_window(source_bbox)
            bounds.extend(_add_hydro_layer(m, hydro_dir, window_bbox, capped=was_capped))
            if was_capped:
                span_km = _HYDRO_MAX_SPAN_M / 1000.0
                hydro_notice = (
                    f"Hydrography shown only for the central {span_km:.0f}×{span_km:.0f} km "
                    f"of the {source_label} extent — reload with a smaller KML/area for full coverage."
                )
        else:
            hydro_notice = (
                "Hydrography skipped — a Detections KML or Labels directory is "
                "required to define the load area."
            )

    if show_detections and detection_features:
        bounds.extend(_add_detections_geojson_layer(m, detection_features))

    label_legend: dict[str, tuple[int, str]] = {}
    if show_labels and labels_dir:
        lbl_bounds, label_legend = _add_labels_layer(m, labels_dir)
        bounds.extend(lbl_bounds)

    folium.LayerControl(collapsed=False).add_to(m)

    if bounds:
        lats = [b[0] for b in bounds]
        lons = [b[1] for b in bounds]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    if hydro_notice:
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;top:10px;right:10px;z-index:9999;'
            'background:#fff3cd;padding:8px 12px;border-radius:4px;'
            'border:1px solid #ffc107;font-size:12px;max-width:320px">'
            f'{hydro_notice}</div>'
        ))

    labels_legend_html = "".join(
        f'<span style="color:{color}">&#9679;</span> {ftype} ({count}) &nbsp;'
        for ftype, (count, color) in sorted(label_legend.items())
    ) or "(none loaded)"

    legend_html = f"""
    <div style="
        position:fixed;bottom:30px;left:30px;z-index:9999;
        background:rgba(255,255,255,0.9);padding:10px 14px;
        border-radius:6px;border:1px solid #ccc;font-size:12px;line-height:1.8;max-width:360px">
      <b>Detections (confidence)</b><br>
      <span style="color:#00cc44">&#9632;</span> ≥ 0.85 &nbsp;
      <span style="color:#ffcc00">&#9632;</span> 0.75–0.85 &nbsp;
      <span style="color:#ff4400">&#9632;</span> 0.65–0.75 &nbsp;
      <span style="color:#888888">&#9632;</span> &lt; 0.65<br>
      <b>Labels</b><br>
      {labels_legend_html}<br>
      <b>Hydrography</b><br>
      <span style="color:#1a6aa8">&#9644;</span> streams / water bodies
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # Click handler: store lat/lon in window globals so the Diagnose button can
    # read them. This script runs inside the folium map's iframe (m._repr_html_()
    # wraps everything in <iframe srcdoc="...">), which has its own `window` —
    # separate from the parent Gradio page where the Diagnose button's JS runs.
    # Write to window.parent too (same-origin srcdoc) so the button can see it.
    map_var = m.get_name()
    click_js = f"""
    <div id="map-click-coords" style="text-align:center;font-size:12px;color:#555;padding:4px 0">
      Click on the map to select a point for diagnosis
    </div>
    <script>
    (function() {{
      var poll = setInterval(function() {{
        if (typeof {map_var} !== 'undefined') {{
          clearInterval(poll);
          {map_var}.on('click', function(e) {{
            window._mapClickLat = e.latlng.lat;
            window._mapClickLon = e.latlng.lng;
            try {{
              window.parent._mapClickLat = e.latlng.lat;
              window.parent._mapClickLon = e.latlng.lng;
            }} catch (err) {{}}
            var el = document.getElementById('map-click-coords');
            if (el) el.textContent = 'Selected: ' + e.latlng.lat.toFixed(6)
                                     + ', ' + e.latlng.lng.toFixed(6);
          }});
        }}
      }}, 100);
    }})();
    </script>"""
    m.get_root().html.add_child(folium.Element(click_js))

    html = f'<div style="height:580px">{m._repr_html_()}</div>'
    size_mb = _html_size_mb(html)
    if size_mb > _MAX_MAP_HTML_MB:
        return (
            "<p style='color:#c00;padding:1em'><b>Too much to draw:</b> "
            f"{len(detection_features)} detections produced {size_mb:.1f} MB of map HTML "
            f"(limit {_MAX_MAP_HTML_MB:.0f} MB). Raise the confidence threshold or use a "
            "smaller detections file, then reload the map.</p>"
        )
    return html


def handle_export_filtered_kml(kml_path: str, threshold: float):
    """Re-write the detections KML keeping only placemarks above the confidence threshold."""
    import re
    kml_path = (kml_path or "").strip()
    if not kml_path or not Path(kml_path).exists():
        return None
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        ns = _MAP_NS
        for parent in list(root.iter()):
            to_remove = []
            for child in parent:
                if child.tag != f"{{{ns}}}Placemark":
                    continue
                desc = child.findtext(f"{{{ns}}}description") or ""
                m = re.search(r"Confidence:\s*([\d.]+)", desc)
                if m and float(m.group(1)) < threshold:
                    to_remove.append(child)
            for child in to_remove:
                parent.remove(child)
        src = Path(kml_path)
        out_path = src.parent / f"{src.stem}_conf{threshold:.2f}.kml"
        ET.register_namespace("", ns)
        tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
        return str(out_path)
    except Exception:
        return None


def handle_load_map(
    kml_path: str,
    labels_dir: str,
    hydro_dir: str,
    basemap: str,
    show_detections: bool,
    show_labels: bool,
    show_hydro: bool,
    confidence_threshold: float = 0.0,
) -> str:
    kml_path   = (kml_path   or "").strip()
    labels_dir = (labels_dir or "").strip()
    hydro_dir  = (hydro_dir  or "").strip()
    if not kml_path and not labels_dir and not hydro_dir:
        return (
            "<p style='color:#888;padding:1em'>"
            "Specify at least one data source, then click Load Map."
            "</p>"
        )
    try:
        return _build_map(kml_path, labels_dir, hydro_dir, basemap,
                          show_detections, show_labels, show_hydro,
                          float(confidence_threshold or 0.0))
    except Exception as exc:
        return f"<p style='color:red'><b>ERROR:</b> {exc}</p>"


# --------------------------------------------------------------------------- #
# Evaluate RF backend
# --------------------------------------------------------------------------- #

def _do_evaluate_rf(
    manifest_path: str,
    rf_model_path: str,
    cluster_radius: float,
    per_class: bool,
) -> None:
    from cli import cmd_evaluate_rf
    cmd_evaluate_rf(argparse.Namespace(
        manifest=manifest_path,
        rf_model=rf_model_path or None,
        cluster_radius=cluster_radius,
        n_splits=5,
        oof_path=None,     # defaults to <manifest_dir>/oof.csv — read back by _confusion_matrix_image
        cache_dir=None,    # defaults to the manifest's own directory
        no_cache=False,
        per_class=per_class,
    ))


def _confusion_matrix_image(manifest_path: str, rf_model_path: str):
    """Build a confusion matrix from the out-of-fold predictions CSV written by
    the last evaluate_rf_spatial run (<manifest_dir>/oof.csv) — NOT from the
    model's own training data, which would be in-sample and near-perfect."""
    if not manifest_path or not manifest_path.strip():
        return None
    try:
        oof_path = Path(manifest_path.strip()).resolve().parent / "oof.csv"
    except Exception:
        return None
    if not oof_path.exists():
        return None
    try:
        import io as _io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from models.evaluate import _recommended_thresholds

        with open(oof_path) as f:
            oof_rows = list(csv.DictReader(f))
        if not oof_rows:
            return None

        y = np.array([int(r["label"]) for r in oof_rows])
        prob = np.array([float(r["prob"]) for r in oof_rows])
        if len(set(y.tolist())) > 1:
            threshold, _, _ = _recommended_thresholds(y, prob)
        else:
            threshold = 0.5
        y_pred = (prob >= threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"])
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Spatial CV (out-of-fold), threshold {threshold:.2f}")
        fig.tight_layout()

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image
        return np.array(Image.open(buf))
    except Exception:
        return None


def _chip_to_image(chip: np.ndarray) -> np.ndarray:
    """CIR chip (bands, H, W) → false-colour RGB uint8 (H, W, 3): NIR→R, Red→G, Green→B."""
    rgb = np.stack([chip[0], chip[1], chip[2]], axis=-1).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    rgb = ((rgb - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    return rgb


def _colormap_image(data: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """2-D float array → RGB uint8 using a matplotlib colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = plt.get_cmap(cmap)(norm(data))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _probability_heatmap(chip: np.ndarray, clf) -> np.ndarray:
    """Sliding-window RF probability map: divide chip into 32×32 patches, predict each."""
    import cv2
    from spectral import extract_features
    patch = 32
    h, w = chip.shape[1], chip.shape[2]
    rows_n, cols_n = h // patch, w // patch
    probs = np.zeros((rows_n, cols_n), dtype=np.float32)
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 0
    for r in range(rows_n):
        for c in range(cols_n):
            sub = chip[:, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
            feats = extract_features(sub).reshape(1, -1)
            probs[r, c] = clf.predict_proba(feats)[0][pos_idx]
    prob_map = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    return _colormap_image(prob_map, "hot", vmin=0, vmax=1)


def _do_diagnose(
    lon: float,
    lat: float,
    imagery_dir: str,
    rf_model_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pyproj import Transformer
    from diagnose_point import find_covering_jp2, extract_chip_at
    from spectral import compute_ndwi, compute_ndvi, extract_features
    from models.random_forest import load_model, predict

    transformer = Transformer.from_crs(4326, 3067, always_xy=True)
    x, y = transformer.transform(lon, lat)
    print(f"WGS84    : {lat:.6f}°N  {lon:.6f}°E")
    print(f"EPSG:3067: x={x:.1f}  y={y:.1f}")

    jp2s = find_covering_jp2(imagery_dir, x, y)
    if not jp2s:
        raise ValueError("No .jp2 tile covers this point. Check the imagery directory and coordinates.")
    print(f"Tile: {Path(jp2s[0]).name}")

    chip = extract_chip_at(jp2s[0], x, y)
    print(f"Chip: {chip.shape}  dtype={chip.dtype}")
    for i, name in enumerate(["NIR", "Red", "Green"]):
        b = chip[i]
        print(f"  {name}: min={b.min()}  max={b.max()}  mean={b.mean():.1f}")

    clf = load_model(rf_model_path)
    label, confidence = predict(clf, chip)
    label_name = {0: "negative", 1: "flood"}.get(label, "unknown")
    print(f"\nPrediction : {label_name} (class {label})  confidence={confidence:.3f}")

    feats = extract_features(chip).reshape(1, -1)
    probs = clf.predict_proba(feats)[0]
    print("Per-class probabilities:")
    for cls, prob in zip(clf.classes_, probs):
        cls_name = {0: "negative", 1: "flood"}.get(int(cls), str(cls))
        print(f"  {cls_name}: {prob:.3f}")

    chip_img = _chip_to_image(chip)
    ndwi_img = _colormap_image(compute_ndwi(chip), "RdBu",   vmin=-1, vmax=1)
    ndvi_img = _colormap_image(compute_ndvi(chip), "RdYlGn", vmin=-1, vmax=1)
    prob_img = _probability_heatmap(chip, clf)
    return chip_img, ndwi_img, ndvi_img, prob_img


def handle_diagnose(
    lon: float,
    lat: float,
    imagery_dir: str,
    rf_model_path: str,
):
    import io as _io
    if not imagery_dir or not imagery_dir.strip():
        return None, None, None, None, "ERROR: Imagery directory or .jp2 file is required."
    if not rf_model_path or not rf_model_path.strip():
        return None, None, None, None, "ERROR: RF model path is required."
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    chip_img = ndwi_img = ndvi_img = prob_img = None
    error: str = ""
    try:
        chip_img, ndwi_img, ndvi_img, prob_img = _do_diagnose(
            float(lon), float(lat),
            imagery_dir.strip(), rf_model_path.strip(),
        )
    except Exception as exc:
        error = str(exc)
    finally:
        sys.stdout = old
    log = buf.getvalue() or ""
    if error:
        log += f"\nERROR: {error}"
    return chip_img, ndwi_img, ndvi_img, prob_img, log or "No output."


def handle_map_click_followup(
    status_msg: str,
    lon: float,
    lat: float,
    imagery_dir: str,
    rf_model_path: str,
):
    """After the map-click JS fills diag_lon/diag_lat: switch to the Diagnose Point
    tab and run the diagnosis automatically — but only if a point was actually
    clicked (status_msg starts with "Selected:"; see the map_diagnose_btn wiring).
    Otherwise leave everything as-is so the "click the map first" message stands.
    """
    if not (status_msg or "").startswith("Selected:"):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    chip_img, ndwi_img, ndvi_img, prob_img, log = handle_diagnose(
        lon, lat, imagery_dir, rf_model_path
    )
    return gr.Tabs(selected="diagnose"), chip_img, ndwi_img, ndvi_img, prob_img, log


def _do_overview(
    imagery_dir: str,
    labels_dir: str,
    models_dir: str,
    chips_dir: str,
) -> str:
    import datetime
    lines: list[str] = []
    warnings: list[str] = []

    def _fmt_size(path: str) -> str:
        try:
            s = Path(path).stat().st_size
            if s > 1e9: return f"{s/1e9:.1f} GB"
            if s > 1e6: return f"{s/1e6:.1f} MB"
            return f"{s/1e3:.0f} KB"
        except Exception:
            return "?"

    def _fmt_mtime(path: str) -> str:
        try:
            ts = Path(path).stat().st_mtime
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "?"

    # ---- Imagery ----
    lines.append("=== Imagery ===")
    if imagery_dir:
        jp2s = _find_files(imagery_dir, ".jp2")
        if jp2s:
            total_mb = sum(Path(f).stat().st_size for f in jp2s) / 1e6
            lines.append(f"  {len(jp2s)} .jp2 file(s)  ({total_mb:.0f} MB total)")
            for f in jp2s[:5]:
                lines.append(f"    {Path(f).name}  ({_fmt_size(f)})")
            if len(jp2s) > 5:
                lines.append(f"    ... and {len(jp2s) - 5} more")
        else:
            lines.append(f"  No .jp2 files found in {imagery_dir!r}")
            warnings.append(f"No imagery found in {imagery_dir!r}")
    else:
        lines.append("  (not specified)")

    # ---- Labels ----
    lines.append("\n=== Labels ===")
    if labels_dir:
        kml_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")
        if kml_files:
            lines.append(f"  {len(kml_files)} KML/KMZ file(s)")
            # Feature type comes from the enclosing KML *folder* name — the same
            # resolution training_data.parse_kml_labels uses to derive the class,
            # not the placemark name (which training ignores when inside a folder).
            import training_data
            counts: dict[str, int] = {}
            for kp in kml_files:
                try:
                    for _, ftype in training_data.parse_kml_labels(kp):
                        counts[ftype] = counts.get(ftype, 0) + 1
                except Exception:
                    pass
            for k, v in sorted(counts.items()):
                lines.append(f"    {k}: {v}")
            if not counts:
                warnings.append("No placemark points found in KML files")
        else:
            lines.append(f"  No KML/KMZ files found in {labels_dir!r}")
            warnings.append(f"No labels found in {labels_dir!r}")
    else:
        lines.append("  (not specified)")

    # ---- Models ----
    lines.append("\n=== Models ===")
    if models_dir:
        p = Path(models_dir)
        if p.exists():
            model_files = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix in (".pkl", ".pth", ".json")
            )
            if model_files:
                for f in model_files:
                    lines.append(f"  {f.name}  ({_fmt_size(str(f))}  modified {_fmt_mtime(str(f))})")
            else:
                lines.append(f"  No model files (.pkl/.pth/.json) found in {models_dir!r}")
        else:
            lines.append(f"  Directory not found: {models_dir!r}")
    else:
        lines.append("  (not specified)")

    # ---- Training chips ----
    lines.append("\n=== Training Chips ===")
    if chips_dir:
        manifest_path = Path(chips_dir) / "manifest.csv"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    rows = list(csv.DictReader(f))
                positive = [r for r in rows if int(r["label"]) == 1]
                negative = [r for r in rows if int(r["label"]) == 0]
                by_type: dict[str, int] = {}
                for r in positive:
                    ft = r.get("feature_type", "unknown")
                    by_type[ft] = by_type.get(ft, 0) + 1
                type_str = "  ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
                lines.append(f"  {len(rows)} chips total")
                lines.append(f"  Positive: {len(positive)}  ({type_str})")
                lines.append(f"  Negative: {len(negative)}")
            except Exception as exc:
                lines.append(f"  Error reading manifest: {exc}")
        else:
            lines.append(f"  No manifest.csv found in {chips_dir!r}")
    else:
        lines.append("  (not specified)")

    if warnings:
        lines.append("\n=== Warnings ===")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


def handle_overview(
    imagery_dir: str,
    labels_dir: str,
    models_dir: str,
    chips_dir: str,
) -> str:
    try:
        return _do_overview(
            (imagery_dir or "").strip(),
            (labels_dir  or "").strip(),
            (models_dir  or "").strip(),
            (chips_dir   or "").strip(),
        )
    except Exception as exc:
        return f"ERROR: {exc}"


def _load_chip_gallery(chips_dir: str, n_per_class: int = 12) -> list:
    """Return [(image_array, caption), ...] for a sample of chips from manifest.csv."""
    import random
    manifest = Path(chips_dir) / "manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    positives = [r for r in rows if int(r["label"]) == 1]
    negatives = [r for r in rows if int(r["label"]) == 0]
    sample = (
        random.sample(positives, min(n_per_class, len(positives))) +
        random.sample(negatives, min(n_per_class, len(negatives)))
    )
    gallery = []
    for r in sample:
        try:
            chip = np.load(r["path"])
            img  = _chip_to_image(chip)
            ft   = r.get("feature_type", "negative") if int(r["label"]) == 1 else "negative"
            gallery.append((img, ft))
        except Exception:
            pass
    return gallery


def handle_chip_gallery(chips_dir: str) -> list:
    if not chips_dir or not chips_dir.strip():
        return []
    try:
        return _load_chip_gallery(chips_dir.strip())
    except Exception:
        return []


def _do_evaluate_compare(
    manifest_path: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    test_fraction: float,
) -> None:
    from cli import cmd_evaluate
    cmd_evaluate(argparse.Namespace(
        manifest=manifest_path,
        rf_model=rf_model_path,
        cnn_model=cnn_model_path,
        norm_stats=norm_stats_path,
        test_fraction=test_fraction,
    ))


def handle_evaluate_compare(
    manifest_path: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    test_fraction: float,
):
    if not manifest_path or not manifest_path.strip():
        yield "ERROR: Manifest CSV path is required."; return
    if not rf_model_path or not rf_model_path.strip():
        yield "ERROR: RF model path is required."; return
    if not cnn_model_path or not cnn_model_path.strip():
        yield "ERROR: CNN model path is required."; return
    if not norm_stats_path or not norm_stats_path.strip():
        yield "ERROR: Norm stats path is required."; return
    yield from _stream(
        _do_evaluate_compare,
        manifest_path.strip(), rf_model_path.strip(),
        cnn_model_path.strip(), norm_stats_path.strip(),
        float(test_fraction),
    )


def handle_evaluate_rf(
    manifest_path: str,
    rf_model_path: str,
    cluster_radius: float,
    per_class: bool,
):
    if not manifest_path or not manifest_path.strip():
        yield "ERROR: Manifest CSV path is required."; return
    # RF model path is optional here — CV folds are always trained fresh
    # (see models.random_forest.make_classifier); it's only echoed for reference.
    yield from _stream(
        _do_evaluate_rf,
        manifest_path.strip(), (rf_model_path or "").strip(),
        float(cluster_radius), bool(per_class),
    )


# --------------------------------------------------------------------------- #
# Gradio layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="CastorDetector") as demo:
    gr.Markdown("# CastorDetector\nBeaver activity detection in MML aerial imagery.")
    with gr.Row():
        save_btn    = gr.Button("Save as defaults", variant="secondary", scale=0)
        save_status = gr.Textbox(label="", interactive=False, scale=1, max_lines=1,
                                 show_label=False, placeholder="")
    with gr.Tabs() as main_tabs:

        # ------------------------------------------------------------------ #
        # Train RF
        # ------------------------------------------------------------------ #
        with gr.Tab("Train RF"):
            gr.Markdown(
                "## Train Random Forest\n"
                "Extract chips from labelled imagery and train a Random Forest classifier."
            )
            with gr.Row():
                rf_imagery = gr.Textbox(label="Imagery directory or .jp2 file", placeholder="data/imagery/",   value=_s.get("rf_imagery", ""))
                rf_labels  = gr.Textbox(label="Labels directory",  placeholder="data/labels/",    value=_s.get("rf_labels",  ""))
            with gr.Row():
                rf_model  = gr.Textbox(label="Model output path (.pkl)", placeholder="data/models/model.pkl",   value=_s.get("rf_model", ""))
                rf_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/", value=_s.get("rf_hydro", ""))
            with gr.Row():
                rf_chips   = gr.Textbox(label="Chip directory (optional, enables evaluate-rf)", placeholder="data/chips/", value=_s.get("rf_chips", ""))
                rf_augment = gr.Slider(minimum=0, maximum=12, value=6, step=1,
                                       label="Augment positives (extra offset chips per label)")
            with gr.Row():
                rf_flood_samples = gr.Number(
                    label="Flood samples from hydrography (tulvaalue, 0 = disabled)",
                    value=int(_s.get("rf_flood_samples", 0)), precision=0, minimum=0,
                )
                rf_hydro_negatives = gr.Checkbox(
                    label="Restrict auto-negatives to stream corridor (requires hydrography)",
                    value=bool(_s.get("rf_hydro_negatives", False)),
                )
            with gr.Row():
                rf_btn  = gr.Button("Train RF", variant="primary")
                rf_stop = gr.Button("Stop", variant="stop")
            rf_log = gr.Textbox(label="Log", lines=15, interactive=False)
            rf_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                rf_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            rf_event = rf_btn.click(
                fn=handle_train_rf,
                inputs=[rf_imagery, rf_labels, rf_model, rf_hydro, rf_chips,
                        rf_augment, rf_flood_samples, rf_hydro_negatives],
                outputs=rf_log,
            )

        # ------------------------------------------------------------------ #
        # Train CNN
        # ------------------------------------------------------------------ #
        with gr.Tab("Train CNN"):
            gr.Markdown(
                "## Train CNN (Prithvi-EO-1.0-100M)\n"
                "Fine-tune the pretrained geospatial foundation model on your labelled chips.\n"
                "> **Note:** Downloads ~454 MB of pretrained weights from HuggingFace on first run."
            )
            with gr.Row():
                cnn_imagery    = gr.Textbox(label="Imagery directory or .jp2 file", placeholder="data/imagery/",                value=_s.get("cnn_imagery",    ""))
                cnn_labels     = gr.Textbox(label="Labels directory",         placeholder="data/labels/",                 value=_s.get("cnn_labels",     ""))
            with gr.Row():
                cnn_model      = gr.Textbox(label="Model output path (.pth)", placeholder="data/models/beaver_cnn_v1.pth", value=_s.get("cnn_model",      ""))
                cnn_norm_stats = gr.Textbox(label="Norm stats path (.json)",  placeholder="data/models/norm_stats.json",   value=_s.get("cnn_norm_stats", ""))
            with gr.Row():
                cnn_hydro = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/",         value=_s.get("cnn_hydro",      ""))
            with gr.Row():
                cnn_epochs = gr.Number(value=int(_s.get("cnn_epochs", 30)),       label="Epochs",        precision=0)
                cnn_lr     = gr.Number(value=float(_s.get("cnn_lr", 0.001)), label="Learning rate")
            with gr.Row():
                cnn_btn  = gr.Button("Train CNN", variant="primary")
                cnn_stop = gr.Button("Stop", variant="stop")
            cnn_log = gr.Textbox(label="Log", lines=15, interactive=False)
            cnn_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                cnn_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            cnn_event = cnn_btn.click(
                fn=handle_train_cnn,
                inputs=[cnn_imagery, cnn_labels, cnn_model, cnn_norm_stats, cnn_hydro, cnn_epochs, cnn_lr],
                outputs=cnn_log,
            )

        # ------------------------------------------------------------------ #
        # Detect & Export
        # ------------------------------------------------------------------ #
        with gr.Tab("Detect & Export"):
            gr.Markdown(
                "## Detect & Export\n"
                "Run the trained model on imagery and export detections as a KML file."
            )
            with gr.Row():
                det_imagery = gr.Textbox(label="Imagery directory or .jp2 file", placeholder="data/imagery/",             value=_s.get("det_imagery",    ""))
                det_output  = gr.Textbox(label="Output KML path",   placeholder="data/output/detections.kml", value=_s.get("det_output",     ""))
            with gr.Row():
                det_method = gr.Dropdown(choices=["rf", "cnn", "both"], value=_s.get("det_method", "rf"), label="Method")
                det_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/", value=_s.get("det_hydro", ""))
            with gr.Row():
                with gr.Column():
                    det_rf_model      = gr.Textbox(label="RF model path (.pkl)",  placeholder="data/models/model.pkl",         value=_s.get("det_rf_model",  ""))
                    det_rf_model_btn  = gr.UploadButton("📂 Browse", file_types=[".pkl"],  size="sm")
                with gr.Column():
                    det_cnn_model     = gr.Textbox(label="CNN model path (.pth)", placeholder="data/models/beaver_cnn_v1.pth", value=_s.get("det_cnn_model", ""))
                    det_cnn_model_btn = gr.UploadButton("📂 Browse", file_types=[".pth"],  size="sm")
            with gr.Row():
                with gr.Column():
                    det_norm_stats     = gr.Textbox(label="Norm stats path (.json)", placeholder="data/models/norm_stats.json", value=_s.get("det_norm_stats", ""))
                    det_norm_stats_btn = gr.UploadButton("📂 Browse", file_types=[".json"], size="sm")
                with gr.Column():
                    det_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=float(_s.get("det_threshold", 0.5)), step=0.05,
                                              label="Confidence threshold")
            with gr.Row():
                det_btn  = gr.Button("Detect & Export KML", variant="primary")
                det_stop = gr.Button("Stop", variant="stop")
            det_log   = gr.Textbox(label="Log", lines=15, interactive=False)
            det_stats = gr.Textbox(label="Statistics", lines=8, interactive=False)
            det_file  = gr.File(label="Download KML", interactive=False)
            det_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                det_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            # det_btn.click() is wired after the Map tab so map_kml is in scope

        # ------------------------------------------------------------------ #
        # Evaluate RF
        # ------------------------------------------------------------------ #
        with gr.Tab("Evaluate RF"):
            gr.Markdown(
                "## Evaluate RF\n"
                "Assess model quality using pooled out-of-fold spatial cross-validation: "
                "label points within the cluster radius are grouped into the same fold "
                "(avoiding the spatial autocorrelation leak a random split introduces), each "
                "fold's held-out probabilities are pooled before computing metrics, and results "
                "are reported at both a chip level and a point level (chips from the same label "
                "point, including augmented copies, are averaged together).\n"
                "Reports ROC-AUC, PR-AUC, a recommended (max-F1) threshold, and a breakdown by "
                "feature type. Results are written to `oof.csv` next to the manifest."
            )
            with gr.Row():
                with gr.Column():
                    ev_manifest     = gr.Textbox(label="Manifest CSV path",    placeholder="data/chips/manifest.csv", value=_s.get("ev_manifest", ""))
                    ev_manifest_btn = gr.UploadButton("📂 Browse", file_types=[".csv"], size="sm")
                with gr.Column():
                    ev_rf_model     = gr.Textbox(label="RF model path (.pkl, optional — for reference only; CV folds are always trained fresh)", placeholder="data/models/model.pkl",   value=_s.get("ev_rf_model", ""))
                    ev_rf_model_btn = gr.UploadButton("📂 Browse", file_types=[".pkl"], size="sm")
            with gr.Row():
                ev_radius    = gr.Slider(minimum=100, maximum=2000, value=float(_s.get("ev_radius", 500)), step=50,
                                         label="Cluster radius (metres)")
                ev_per_class = gr.Checkbox(label="Per-feature-type breakdown (recall for positive types, specificity for negative types)", value=bool(_s.get("ev_per_class", False)))
            with gr.Row():
                ev_btn  = gr.Button("Evaluate RF", variant="primary")
                ev_stop = gr.Button("Stop", variant="stop")
            ev_log = gr.Textbox(label="Results", lines=20, interactive=False)
            ev_cm  = gr.Image(label="Confusion matrix — spatial CV (out-of-fold)", type="numpy", height=320)
            ev_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                ev_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            ev_event = ev_btn.click(
                fn=handle_evaluate_rf,
                inputs=[ev_manifest, ev_rf_model, ev_radius, ev_per_class],
                outputs=ev_log,
            )

        # ------------------------------------------------------------------ #
        # Evaluate RF vs CNN
        # ------------------------------------------------------------------ #
        with gr.Tab("Evaluate RF vs CNN"):
            gr.Markdown(
                "## Evaluate RF vs CNN\n"
                "Compare the two saved models on a spatially-held-out slice of the training "
                "manifest.\n"
                "> **⚠ IN-SAMPLE — these chips were used to train the saved models; numbers "
                "are optimistic.** Both models were fit on the full manifest, so no split of "
                "it is truly held out. Pass `--test-manifest` on the CLI with chips built from "
                "separate, unseen tiles for a genuine out-of-sample comparison. "
                "Use the **Evaluate RF** tab for spatially rigorous cross-validation of the RF model."
            )
            with gr.Row():
                with gr.Column():
                    cmp_manifest     = gr.Textbox(label="Manifest CSV path",       placeholder="data/chips/manifest.csv",       value=_s.get("cmp_manifest",   ""))
                    cmp_manifest_btn = gr.UploadButton("📂 Browse", file_types=[".csv"], size="sm")
                with gr.Column():
                    cmp_rf_model     = gr.Textbox(label="RF model path (.pkl)",    placeholder="data/models/model.pkl",         value=_s.get("cmp_rf_model",   ""))
                    cmp_rf_model_btn = gr.UploadButton("📂 Browse", file_types=[".pkl"], size="sm")
            with gr.Row():
                with gr.Column():
                    cmp_cnn_model     = gr.Textbox(label="CNN model path (.pth)",   placeholder="data/models/beaver_cnn_v1.pth", value=_s.get("cmp_cnn_model",  ""))
                    cmp_cnn_model_btn = gr.UploadButton("📂 Browse", file_types=[".pth"],  size="sm")
                with gr.Column():
                    cmp_norm_stats     = gr.Textbox(label="Norm stats path (.json)", placeholder="data/models/norm_stats.json",   value=_s.get("cmp_norm_stats", ""))
                    cmp_norm_stats_btn = gr.UploadButton("📂 Browse", file_types=[".json"], size="sm")
            cmp_test_frac = gr.Slider(minimum=0.1, maximum=0.5, value=float(_s.get("cmp_test_frac", 0.2)), step=0.05,
                                      label="Test fraction")
            with gr.Row():
                cmp_btn  = gr.Button("Evaluate", variant="primary")
                cmp_stop = gr.Button("Stop", variant="stop")
            cmp_log = gr.Textbox(label="Results", lines=12, interactive=False)
            cmp_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                cmp_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            cmp_event = cmp_btn.click(
                fn=handle_evaluate_compare,
                inputs=[cmp_manifest, cmp_rf_model, cmp_cnn_model, cmp_norm_stats, cmp_test_frac],
                outputs=cmp_log,
            )

        # ------------------------------------------------------------------ #
        # Diagnose Point
        # ------------------------------------------------------------------ #
        with gr.Tab("Diagnose Point", id="diagnose"):
            gr.Markdown(
                "## Diagnose Point\n"
                "Extract the chip at a known WGS84 location, run the RF classifier, "
                "and visualise the spectral signature. Useful for understanding false "
                "positives and missed detections."
            )
            with gr.Row():
                diag_lon       = gr.Number(value=25.0,  label="Longitude (WGS84)")
                diag_lat       = gr.Number(value=62.0,  label="Latitude (WGS84)")
            with gr.Row():
                diag_imagery  = gr.Textbox(label="Imagery directory or .jp2 file", placeholder="data/imagery/", value=_s.get("diag_imagery", ""))
                with gr.Column():
                    diag_rf_model     = gr.Textbox(label="RF model path (.pkl)", placeholder="data/models/model.pkl", value=_s.get("diag_rf_model", ""))
                    diag_rf_model_btn = gr.UploadButton("📂 Browse", file_types=[".pkl"], size="sm")
            diag_btn = gr.Button("Diagnose", variant="primary")
            with gr.Row():
                diag_chip = gr.Image(label="CIR chip (NIR=R, Red=G, Green=B)", type="numpy")
                diag_ndwi = gr.Image(label="NDWI  (blue=water, red=dry)",       type="numpy")
                diag_ndvi = gr.Image(label="NDVI  (green=veg, red=bare)",        type="numpy")
                diag_prob = gr.Image(label="RF probability map (bright=flood)",  type="numpy")
            diag_log = gr.Textbox(label="Prediction & band stats", lines=12, interactive=False)
            diag_btn.click(
                fn=handle_diagnose,
                inputs=[diag_lon, diag_lat, diag_imagery, diag_rf_model],
                outputs=[diag_chip, diag_ndwi, diag_ndvi, diag_prob, diag_log],
            )

        # ------------------------------------------------------------------ #
        # Overview
        # ------------------------------------------------------------------ #
        with gr.Tab("Overview"):
            gr.Markdown(
                "## Data Overview\n"
                "Scan your data directories to verify what is available before training or detection."
            )
            with gr.Row():
                ov_imagery    = gr.Textbox(label="Imagery directory or .jp2 file", placeholder="data/imagery/", value=_s.get("ov_imagery",    ""))
                ov_labels     = gr.Textbox(label="Labels directory",          placeholder="data/labels/",  value=_s.get("ov_labels",     ""))
            with gr.Row():
                ov_models_dir = gr.Textbox(label="Models directory",          placeholder="data/models/",  value=_s.get("ov_models_dir", ""))
                ov_chips      = gr.Textbox(label="Chip directory (optional)", placeholder="data/chips/",   value=_s.get("ov_chips",      ""))
            ov_btn = gr.Button("Scan", variant="primary")
            ov_out = gr.Textbox(label="Summary", lines=22, interactive=False)
            gr.Markdown("### Chip sample (CIR false-colour)")
            ov_gallery = gr.Gallery(
                label="Training chips — positives then negatives (up to 12 each)",
                columns=6, height=320, object_fit="contain",
            )
            ov_event = ov_btn.click(
                fn=handle_overview,
                inputs=[ov_imagery, ov_labels, ov_models_dir, ov_chips],
                outputs=ov_out,
            )
            ov_event.then(
                fn=handle_chip_gallery,
                inputs=[ov_chips],
                outputs=[ov_gallery],
            )

        # ------------------------------------------------------------------ #
        # Map
        # ------------------------------------------------------------------ #
        with gr.Tab("Map"):
            gr.Markdown(
                "## Results Map\n"
                "View detection polygons and training label points on an interactive map."
            )
            with gr.Row():
                map_kml    = gr.Textbox(label="Detections KML path", placeholder="data/output/detections.kml", value=_s.get("map_kml",    ""))
                map_labels = gr.Textbox(label="Labels directory",    placeholder="data/labels/",               value=_s.get("map_labels", ""))
            with gr.Row():
                map_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/", value=_s.get("map_hydro", ""))
                map_basemap = gr.Dropdown(choices=["Satellite", "OpenStreetMap"], value="Satellite",
                                          label="Base map")
            with gr.Row():
                map_show_det    = gr.Checkbox(label="Show detections",      value=True)
                map_show_labels = gr.Checkbox(label="Show training labels", value=True)
                map_show_hydro  = gr.Checkbox(label="Show hydrography",     value=True)
            map_btn  = gr.Button("Load Map", variant="primary")
            map_html = gr.HTML()
            map_diagnose_btn = gr.Button(
                "Diagnose selected point (click map first)", variant="secondary"
            )
            map_click_status = gr.Textbox(
                label="", interactive=False, show_label=False, max_lines=1,
                value="Click a point on the map, then press the button above.",
            )
            gr.Markdown("### Export filtered detections")
            with gr.Row():
                map_filter_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                    label="Minimum confidence to keep (applies to map load & export)",
                )
                map_export_btn = gr.Button("Export filtered KML", variant="secondary", scale=0)
            map_export_file = gr.File(label="Filtered KML download", interactive=False)
            map_btn.click(
                fn=handle_load_map,
                inputs=[map_kml, map_labels, map_hydro, map_basemap,
                        map_show_det, map_show_labels, map_show_hydro,
                        map_filter_threshold],
                outputs=map_html,
            )
            # Click-to-diagnose: the folium map is embedded via an <iframe srcdoc="...">
            # (see _build_map's click_js), which has its own `window` — separate from
            # this parent Gradio page, where this button's JS runs. The iframe's click
            # handler writes to window.parent too, so plain window._mapClickLat/Lon
            # here (in the parent) sees it. Falls back to the current lon/lat plus a
            # "click first" message if nothing has been clicked yet.
            map_click_event = map_diagnose_btn.click(
                fn=None,
                inputs=[diag_lon, diag_lat],
                outputs=[diag_lon, diag_lat, map_click_status],
                js="""
                (lon, lat) => {
                  const clat = window._mapClickLat, clon = window._mapClickLon;
                  if (clat === undefined || clon === undefined) {
                    return [lon, lat, 'Click a point on the map first, then press this button.'];
                  }
                  return [clon, clat, 'Selected: ' + clat.toFixed(6) + ', ' + clon.toFixed(6)
                          + ' - running diagnosis...'];
                }
                """,
            )
            # Switch to the Diagnose Point tab and run the diagnosis automatically —
            # only if a point was actually clicked (see handle_map_click_followup).
            map_click_event.then(
                fn=handle_map_click_followup,
                inputs=[map_click_status, diag_lon, diag_lat, diag_imagery, diag_rf_model],
                outputs=[main_tabs, diag_chip, diag_ndwi, diag_ndvi, diag_prob, diag_log],
            )
            map_export_btn.click(
                fn=handle_export_filtered_kml,
                inputs=[map_kml, map_filter_threshold],
                outputs=[map_export_file],
            )

    # Wire detect button here so map_kml is in scope
    det_event = det_btn.click(
        fn=handle_detect,
        inputs=[det_imagery, det_method, det_rf_model, det_cnn_model,
                det_norm_stats, det_hydro, det_threshold, det_output],
        outputs=[det_log, det_file, map_kml],
    )

    det_event.then(fn=_detection_stats, inputs=[det_output], outputs=[det_stats])

    # Confusion matrix — render after Evaluate RF completes
    ev_event.then(fn=_confusion_matrix_image,
                  inputs=[ev_manifest, ev_rf_model],
                  outputs=[ev_cm])

    # Run log history — append completed log to each tab's accordion
    rf_event.then(fn=_append_log_history,
                  inputs=[rf_log,  rf_history_state],
                  outputs=[rf_history_state,  rf_history_text])
    cnn_event.then(fn=_append_log_history,
                   inputs=[cnn_log, cnn_history_state],
                   outputs=[cnn_history_state, cnn_history_text])
    det_event.then(fn=_append_log_history,
                   inputs=[det_log, det_history_state],
                   outputs=[det_history_state, det_history_text])
    ev_event.then(fn=_append_log_history,
                  inputs=[ev_log,  ev_history_state],
                  outputs=[ev_history_state,  ev_history_text])
    cmp_event.then(fn=_append_log_history,
                   inputs=[cmp_log, cmp_history_state],
                   outputs=[cmp_history_state, cmp_history_text])

    # Stop buttons
    rf_stop.click(fn=None,  cancels=[rf_event])
    cnn_stop.click(fn=None, cancels=[cnn_event])
    det_stop.click(fn=None, cancels=[det_event])
    ev_stop.click(fn=None,  cancels=[ev_event])
    cmp_stop.click(fn=None, cancels=[cmp_event])

    save_btn.click(
        fn=handle_save_settings,
        inputs=[
            rf_imagery, rf_labels, rf_model, rf_hydro, rf_chips,
            rf_flood_samples, rf_hydro_negatives,
            cnn_imagery, cnn_labels, cnn_model, cnn_norm_stats, cnn_hydro,
            cnn_epochs, cnn_lr,
            det_imagery, det_output, det_rf_model, det_cnn_model,
            det_norm_stats, det_hydro,
            det_method, det_threshold,
            ev_manifest, ev_rf_model,
            ev_radius, ev_per_class,
            cmp_manifest, cmp_rf_model, cmp_cnn_model, cmp_norm_stats,
            cmp_test_frac,
            diag_imagery, diag_rf_model,
            ov_imagery, ov_labels, ov_models_dir, ov_chips,
            map_kml, map_labels, map_hydro,
        ],
        outputs=save_status,
    )


    # Browse-button wirings — populate adjacent textbox with selected file path
    det_rf_model_btn.upload( fn=_file_to_path, inputs=[det_rf_model_btn],  outputs=[det_rf_model])
    det_cnn_model_btn.upload(fn=_file_to_path, inputs=[det_cnn_model_btn], outputs=[det_cnn_model])
    det_norm_stats_btn.upload(fn=_file_to_path, inputs=[det_norm_stats_btn], outputs=[det_norm_stats])
    ev_manifest_btn.upload(  fn=_file_to_path, inputs=[ev_manifest_btn],   outputs=[ev_manifest])
    ev_rf_model_btn.upload(  fn=_file_to_path, inputs=[ev_rf_model_btn],   outputs=[ev_rf_model])
    cmp_manifest_btn.upload( fn=_file_to_path, inputs=[cmp_manifest_btn],  outputs=[cmp_manifest])
    cmp_rf_model_btn.upload( fn=_file_to_path, inputs=[cmp_rf_model_btn],  outputs=[cmp_rf_model])
    cmp_cnn_model_btn.upload(fn=_file_to_path, inputs=[cmp_cnn_model_btn], outputs=[cmp_cnn_model])
    cmp_norm_stats_btn.upload(fn=_file_to_path, inputs=[cmp_norm_stats_btn], outputs=[cmp_norm_stats])
    diag_rf_model_btn.upload(fn=_file_to_path, inputs=[diag_rf_model_btn], outputs=[diag_rf_model])

demo.queue()

if __name__ == "__main__":
    demo.launch()
