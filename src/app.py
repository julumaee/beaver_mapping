"""Gradio web UI for CastorDetector."""
import subprocess
import sys
from pathlib import Path

def _ensure_dependencies() -> None:
    req = Path(__file__).parent.parent / "requirements.txt"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
    )

_ensure_dependencies()

import csv
import queue
import tempfile
import threading
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _find_files(path: str, suffix: str) -> list[str]:
    p = Path(path)
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
) -> None:
    from training_data import build_training_dataset
    from models.random_forest import train

    jp2_files = _find_files(imagery_dir, ".jp2")
    kml_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")

    if not jp2_files:
        raise ValueError(f"No .jp2 files found in {imagery_dir!r}")
    if not kml_files:
        raise ValueError(f"No KML/KMZ files found in {labels_dir!r}")

    stream_mask = None
    if hydro_dir:
        from masking import build_stream_mask
        print(f"Building stream mask from {hydro_dir} ...")
        stream_mask = build_stream_mask(hydro_dir)

    chip_ctx = _NullContext(chip_dir) if chip_dir else tempfile.TemporaryDirectory()
    with chip_ctx as cd:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            stream_mask=stream_mask,
            out_dir=cd,
            augment_positives=augment,
        )
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        flood = [r for r in rows if int(r["label"]) == 1]
        neg   = [r for r in rows if int(r["label"]) == 0]
        print(f"  Flood chips   : {len(flood)}")
        print(f"  Negative chips: {len(neg)}")

        if not flood:
            raise ValueError(
                "No positive chips extracted. "
                "Check that your imagery tiles cover the labelled feature locations."
            )

        print("Training Random Forest ...")
        train(manifest, model_path)

    print(f"Model saved to {model_path}")
    if chip_dir:
        print(f"Chips and manifest saved to {chip_dir}/")


def handle_train_rf(
    imagery_dir: str,
    labels_dir: str,
    model_path: str,
    hydro_dir: str,
    chip_dir: str,
    augment: float,
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory is required."; return
    if not labels_dir or not labels_dir.strip():
        yield "ERROR: Labels directory is required."; return
    if not model_path or not model_path.strip():
        yield "ERROR: Model output path is required."; return
    yield from _stream(
        _do_train_rf,
        imagery_dir.strip(), labels_dir.strip(), model_path.strip(),
        hydro_dir.strip() if hydro_dir else "",
        chip_dir.strip() if chip_dir else "",
        int(augment),
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
    from training_data import build_training_dataset
    from models.cnn_train import train_cnn

    jp2_files = _find_files(imagery_dir, ".jp2")
    kml_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")

    if not jp2_files:
        raise ValueError(f"No .jp2 files found in {imagery_dir!r}")
    if not kml_files:
        raise ValueError(f"No KML/KMZ files found in {labels_dir!r}")

    stream_mask = None
    if hydro_dir:
        from masking import build_stream_mask
        print(f"Building stream mask from {hydro_dir} ...")
        stream_mask = build_stream_mask(hydro_dir)

    with tempfile.TemporaryDirectory() as cd:
        print("Extracting training chips ...")
        manifest = build_training_dataset(
            jp2_paths=jp2_files,
            kml_paths=kml_files,
            stream_mask=stream_mask,
            out_dir=cd,
        )
        print(f"Training CNN (epochs={epochs}, lr={lr}) ...")
        train_cnn(
            manifest_path=manifest,
            model_path=model_path,
            norm_stats_path=norm_stats_path,
            epochs=epochs,
            lr=lr,
        )

    print(f"CNN model saved to {model_path}")
    print(f"Norm stats  saved to {norm_stats_path}")


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
        yield "ERROR: Imagery directory is required."; return
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
    from polygonizer import detect_rois_rf_segmentation, detect_rois_cnn
    from export import export_kml

    jp2_files = _find_files(imagery_dir, ".jp2")
    if not jp2_files:
        raise ValueError(f"No .jp2 files found in {imagery_dir!r}")

    stream_mask = None
    if hydro_dir:
        from masking import build_stream_mask
        print(f"Building stream mask from {hydro_dir} ...")
        stream_mask = build_stream_mask(hydro_dir)

    rf_clf = cnn_model = norm_stats = None

    if method in ("rf", "both"):
        from models.random_forest import load_model as load_rf
        print(f"Loading RF model from {rf_model_path} ...")
        rf_clf = load_rf(rf_model_path)

    if method in ("cnn", "both"):
        import json
        from models.cnn_handler import load_cnn
        cnn_model = load_cnn(cnn_model_path)
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

    all_rois: list = []

    for jp2_path in jp2_files:
        print(f"Processing {jp2_path} ...")
        if method == "rf":
            rois = detect_rois_rf_segmentation(jp2_path, rf_clf, stream_mask, threshold)
            print(f"  RF detections: {len(rois)}")
            all_rois.extend(rois)
        elif method == "cnn":
            rois = detect_rois_cnn(jp2_path, cnn_model, norm_stats, stream_mask, threshold)
            print(f"  CNN detections: {len(rois)}")
            all_rois.extend(rois)
        elif method == "both":
            rf_rois  = detect_rois_rf_segmentation(jp2_path, rf_clf, stream_mask, threshold)
            cnn_rois = detect_rois_cnn(jp2_path, cnn_model, norm_stats, stream_mask, threshold)
            print(f"  RF: {len(rf_rois)}  CNN: {len(cnn_rois)}")
            tagged: list = []
            matched: set[int] = set()
            for rf_p, rf_c, rf_a in rf_rois:
                found = False
                for j, (cnn_p, cnn_c, cnn_a) in enumerate(cnn_rois):
                    if rf_p.intersects(cnn_p):
                        merged = rf_p.union(cnn_p)
                        tagged.append((merged, max(rf_c, cnn_c), merged.area, "both"))
                        matched.add(j)
                        found = True
                        break
                if not found:
                    tagged.append((rf_p, rf_c, rf_a, "rf"))
            for j, (cnn_p, cnn_c, cnn_a) in enumerate(cnn_rois):
                if j not in matched:
                    tagged.append((cnn_p, cnn_c, cnn_a, "cnn"))
            all_rois.extend(tagged)

    print(f"Exporting {len(all_rois)} detection(s) to {output_path} ...")
    export_kml(all_rois, output_path)
    print("Done.")


def handle_detect(
    imagery_dir: str,
    method: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    hydro_dir: str,
    threshold: float,
    output_path: str,
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory is required.", None; return
    if not output_path or not output_path.strip():
        yield "ERROR: Output KML path is required.", None; return
    if method in ("rf", "both") and (not rf_model_path or not rf_model_path.strip()):
        yield "ERROR: RF model path is required for this method.", None; return
    if method in ("cnn", "both") and (not cnn_model_path or not cnn_model_path.strip()):
        yield "ERROR: CNN model path is required for this method.", None; return
    if method in ("cnn", "both") and (not norm_stats_path or not norm_stats_path.strip()):
        yield "ERROR: Norm stats path is required for this method.", None; return
    out = output_path.strip()
    last_log = ""
    for log in _stream(
        _do_detect,
        imagery_dir.strip(), method,
        rf_model_path.strip(), cnn_model_path.strip(), norm_stats_path.strip(),
        hydro_dir.strip(), float(threshold), out,
    ):
        last_log = log
        yield log, None, gr.update()
    kml_exists = out and Path(out).exists()
    yield last_log, (out if kml_exists else None), (out if kml_exists else gr.update())


# --------------------------------------------------------------------------- #
# Map view
# --------------------------------------------------------------------------- #

_MAP_NS = "http://www.opengis.net/kml/2.2"

_DETECTION_COLORS = {
    "model_rf":   "#e03030",
    "model_cnn":  "#3030e0",
    "model_both": "#a030a0",
}

_LABEL_COLORS = {
    "wet_forest":   "#ff7700",
    "beaver_flood": "#00aaff",
    "negative":     "#888888",
    "dam":          "#8b4513",
    "lodge":        "#654321",
}
_LABEL_DEFAULT_COLOR = "#ffcc00"


def _add_detections_layer(m, kml_path: str) -> list[tuple[float, float]]:
    import folium
    group = folium.FeatureGroup(name="Detections", show=True)
    bounds: list[tuple[float, float]] = []
    try:
        root = ET.parse(kml_path).getroot()
        for pm in root.iter(f"{{{_MAP_NS}}}Placemark"):
            style_id = (pm.findtext(f"{{{_MAP_NS}}}styleUrl") or "").lstrip("#")
            color = _DETECTION_COLORS.get(style_id, "#e07030")
            name  = pm.findtext(f"{{{_MAP_NS}}}name") or "Detection"
            desc  = (pm.findtext(f"{{{_MAP_NS}}}description") or "").replace("\n", "<br>")
            coords_raw = pm.findtext(f".//{{{_MAP_NS}}}coordinates") or ""
            points: list[tuple[float, float]] = []
            for part in coords_raw.strip().split():
                vals = part.split(",")
                if len(vals) >= 2:
                    pt = (float(vals[1]), float(vals[0]))  # (lat, lon)
                    points.append(pt)
                    bounds.append(pt)
            if len(points) >= 3:
                folium.Polygon(
                    locations=points,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.30,
                    weight=2,
                    tooltip=folium.Tooltip(f"<b>{name}</b><br>{desc}"),
                ).add_to(group)
    except Exception as exc:
        print(f"Warning: could not parse detections KML: {exc}")
    group.add_to(m)
    return bounds


def _add_labels_layer(m, labels_dir: str) -> list[tuple[float, float]]:
    import folium
    group = folium.FeatureGroup(name="Training labels", show=True)
    bounds: list[tuple[float, float]] = []
    kml_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")
    for kml_path in kml_files:
        try:
            if kml_path.endswith(".kmz"):
                with zipfile.ZipFile(kml_path) as z:
                    inner = next(n for n in z.namelist() if n.endswith(".kml"))
                    with z.open(inner) as f:
                        root = ET.parse(f).getroot()
            else:
                root = ET.parse(kml_path).getroot()
            for pm in root.iter(f"{{{_MAP_NS}}}Placemark"):
                label = (pm.findtext(f"{{{_MAP_NS}}}name") or "").strip().lower()
                point_el = pm.find(f".//{{{_MAP_NS}}}Point")
                if point_el is None:
                    continue
                coords_raw = (point_el.findtext(f"{{{_MAP_NS}}}coordinates") or "").strip()
                if not coords_raw:
                    continue
                parts = coords_raw.split(",")
                if len(parts) < 2:
                    continue
                lon = float(parts[0].split()[0])
                lat = float(parts[1])
                pt = (lat, lon)
                bounds.append(pt)
                color = _LABEL_COLORS.get(label, _LABEL_DEFAULT_COLOR)
                folium.CircleMarker(
                    location=pt,
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    weight=1.5,
                    tooltip=label or "unknown",
                ).add_to(group)
        except Exception as exc:
            print(f"Warning: could not parse {kml_path}: {exc}")
    group.add_to(m)
    return bounds


def _add_hydro_layer(m, hydro_dir: str) -> list[tuple[float, float]]:
    import folium
    import geopandas as gpd
    import pandas as pd
    from masking import _load_stream_layers, _resolve_files, BUFFER_METERS

    files = _resolve_files(hydro_dir)
    if not files:
        return []

    gdfs: list[gpd.GeoDataFrame] = []
    for f in files:
        for _, gdf in _load_stream_layers(f):
            if gdf.crs is None:
                continue
            if gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            gdfs.append(gdf[["geometry"]])
    if not gdfs:
        return []

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0].reset_index(drop=True),
        crs="EPSG:3067",
    )
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

    buffered = combined.copy()
    buffered["geometry"] = buffered.geometry.buffer(BUFFER_METERS)
    buffered = buffered[~buffered.geometry.is_empty].to_crs(epsg=4326)
    buffer_group = folium.FeatureGroup(name=f"Stream buffer ({BUFFER_METERS} m)", show=False)
    folium.GeoJson(
        buffered.__geo_interface__,
        style_function=lambda _: {
            "color": "#1a6aa8",
            "fillColor": "#4da6ff",
            "fillOpacity": 0.12,
            "weight": 0,
        },
    ).add_to(buffer_group)
    buffer_group.add_to(m)

    bounds: list[tuple[float, float]] = []
    for geom in combined_wgs84.geometry:
        if geom is not None and not geom.is_empty:
            minx, miny, maxx, maxy = geom.bounds
            bounds.extend([(miny, minx), (maxy, maxx)])
    return bounds


def _build_map(
    kml_path: str,
    labels_dir: str = "",
    hydro_dir: str = "",
    basemap: str = "Satellite",
    show_detections: bool = True,
    show_labels: bool = True,
    show_hydro: bool = True,
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

    if show_hydro and hydro_dir:
        bounds.extend(_add_hydro_layer(m, hydro_dir))

    if show_detections and kml_path and Path(kml_path).exists():
        bounds.extend(_add_detections_layer(m, kml_path))

    if show_labels and labels_dir:
        bounds.extend(_add_labels_layer(m, labels_dir))

    folium.LayerControl(collapsed=False).add_to(m)

    if bounds:
        lats = [b[0] for b in bounds]
        lons = [b[1] for b in bounds]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    legend_html = """
    <div style="
        position:fixed;bottom:30px;left:30px;z-index:9999;
        background:rgba(255,255,255,0.9);padding:10px 14px;
        border-radius:6px;border:1px solid #ccc;font-size:12px;line-height:1.8">
      <b>Detections</b><br>
      <span style="color:#e03030">&#9632;</span> RF &nbsp;
      <span style="color:#3030e0">&#9632;</span> CNN &nbsp;
      <span style="color:#a030a0">&#9632;</span> Both<br>
      <b>Labels</b><br>
      <span style="color:#ff7700">&#9679;</span> wet_forest &nbsp;
      <span style="color:#00aaff">&#9679;</span> beaver_flood<br>
      <span style="color:#888888">&#9679;</span> negative &nbsp;
      <span style="color:#8b4513">&#9679;</span> dam<br>
      <b>Hydrography</b><br>
      <span style="color:#1a6aa8">&#9644;</span> streams / water bodies
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return f'<div style="height:580px">{m._repr_html_()}</div>'


def handle_load_map(
    kml_path: str,
    labels_dir: str,
    hydro_dir: str,
    basemap: str,
    show_detections: bool,
    show_labels: bool,
    show_hydro: bool,
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
                          show_detections, show_labels, show_hydro)
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
    if per_class:
        from models.evaluate import evaluate_rf_per_class
        evaluate_rf_per_class(
            manifest_path=manifest_path,
            rf_model_path=rf_model_path,
            cluster_radius=cluster_radius,
        )
    else:
        from models.evaluate import evaluate_rf_spatial
        evaluate_rf_spatial(
            manifest_path=manifest_path,
            rf_model_path=rf_model_path,
            cluster_radius=cluster_radius,
        )


def _do_evaluate_compare(
    manifest_path: str,
    rf_model_path: str,
    cnn_model_path: str,
    norm_stats_path: str,
    test_fraction: float,
) -> None:
    from models.evaluate import evaluate_models
    evaluate_models(
        manifest_path=manifest_path,
        rf_model_path=rf_model_path,
        cnn_model_path=cnn_model_path,
        norm_stats_path=norm_stats_path,
        test_fraction=test_fraction,
    )


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
    if not rf_model_path or not rf_model_path.strip():
        yield "ERROR: RF model path is required."; return
    yield from _stream(
        _do_evaluate_rf,
        manifest_path.strip(), rf_model_path.strip(),
        float(cluster_radius), bool(per_class),
    )


# --------------------------------------------------------------------------- #
# Gradio layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="CastorDetector") as demo:
    gr.Markdown("# CastorDetector\nBeaver activity detection in MML aerial imagery.")
    with gr.Tabs():

        # ------------------------------------------------------------------ #
        # Train RF
        # ------------------------------------------------------------------ #
        with gr.Tab("Train RF"):
            gr.Markdown(
                "## Train Random Forest\n"
                "Extract chips from labelled imagery and train a Random Forest classifier."
            )
            with gr.Row():
                rf_imagery = gr.Textbox(label="Imagery directory", placeholder="data/imagery/")
                rf_labels  = gr.Textbox(label="Labels directory",  placeholder="data/labels/")
            with gr.Row():
                rf_model  = gr.Textbox(label="Model output path (.pkl)", placeholder="data/models/model.pkl")
                rf_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/")
            with gr.Row():
                rf_chips   = gr.Textbox(label="Chip directory (optional, enables evaluate-rf)", placeholder="data/chips/")
                rf_augment = gr.Slider(minimum=0, maximum=12, value=6, step=1,
                                       label="Augment positives (extra offset chips per label)")
            rf_btn = gr.Button("Train RF", variant="primary")
            rf_log = gr.Textbox(label="Log", lines=15, interactive=False)
            rf_btn.click(
                fn=handle_train_rf,
                inputs=[rf_imagery, rf_labels, rf_model, rf_hydro, rf_chips, rf_augment],
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
                cnn_imagery    = gr.Textbox(label="Imagery directory",        placeholder="data/imagery/")
                cnn_labels     = gr.Textbox(label="Labels directory",         placeholder="data/labels/")
            with gr.Row():
                cnn_model      = gr.Textbox(label="Model output path (.pth)", placeholder="data/models/beaver_cnn_v1.pth")
                cnn_norm_stats = gr.Textbox(label="Norm stats path (.json)",  placeholder="data/models/norm_stats.json")
            with gr.Row():
                cnn_hydro = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/")
            with gr.Row():
                cnn_epochs = gr.Number(value=30,    label="Epochs",        precision=0)
                cnn_lr     = gr.Number(value=0.001, label="Learning rate")
            cnn_btn = gr.Button("Train CNN", variant="primary")
            cnn_log = gr.Textbox(label="Log", lines=15, interactive=False)
            cnn_btn.click(
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
                det_imagery = gr.Textbox(label="Imagery directory", placeholder="data/imagery/")
                det_output  = gr.Textbox(label="Output KML path",   placeholder="data/output/detections.kml")
            with gr.Row():
                det_method = gr.Dropdown(choices=["rf", "cnn", "both"], value="rf", label="Method")
                det_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/")
            with gr.Row():
                det_rf_model  = gr.Textbox(label="RF model path (.pkl)",  placeholder="data/models/model.pkl")
                det_cnn_model = gr.Textbox(label="CNN model path (.pth)", placeholder="data/models/beaver_cnn_v1.pth")
            with gr.Row():
                det_norm_stats = gr.Textbox(label="Norm stats path (.json)", placeholder="data/models/norm_stats.json")
                det_threshold  = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                           label="Confidence threshold")
            det_btn  = gr.Button("Detect & Export KML", variant="primary")
            det_log  = gr.Textbox(label="Log", lines=15, interactive=False)
            det_file = gr.File(label="Download KML", interactive=False)
            # det_btn.click() is wired after the Map tab so map_kml is in scope

        # ------------------------------------------------------------------ #
        # Evaluate RF
        # ------------------------------------------------------------------ #
        with gr.Tab("Evaluate RF"):
            gr.Markdown(
                "## Evaluate RF\n"
                "Assess model quality using spatial leave-one-cluster-out cross-validation.\n"
                "Label points within the cluster radius are grouped into the same fold, "
                "avoiding the spatial autocorrelation leak that a random split introduces."
            )
            with gr.Row():
                ev_manifest  = gr.Textbox(label="Manifest CSV path", placeholder="data/chips/manifest.csv")
                ev_rf_model  = gr.Textbox(label="RF model path (.pkl)", placeholder="data/models/model.pkl")
            with gr.Row():
                ev_radius    = gr.Slider(minimum=100, maximum=2000, value=500, step=50,
                                         label="Cluster radius (metres)")
                ev_per_class = gr.Checkbox(label="Per-class breakdown (wet_forest / beaver_flood)", value=False)
            ev_btn = gr.Button("Evaluate RF", variant="primary")
            ev_log = gr.Textbox(label="Results", lines=20, interactive=False)
            ev_btn.click(
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
                "Compare both models on a random held-out split of the training manifest.\n"
                "> **Note:** Uses a random split — results are indicative. "
                "Use the **Evaluate RF** tab for spatially rigorous cross-validation."
            )
            with gr.Row():
                cmp_manifest   = gr.Textbox(label="Manifest CSV path",      placeholder="data/chips/manifest.csv")
                cmp_rf_model   = gr.Textbox(label="RF model path (.pkl)",   placeholder="data/models/model.pkl")
            with gr.Row():
                cmp_cnn_model  = gr.Textbox(label="CNN model path (.pth)",  placeholder="data/models/beaver_cnn_v1.pth")
                cmp_norm_stats = gr.Textbox(label="Norm stats path (.json)", placeholder="data/models/norm_stats.json")
            cmp_test_frac = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05,
                                      label="Test fraction")
            cmp_btn = gr.Button("Evaluate", variant="primary")
            cmp_log = gr.Textbox(label="Results", lines=12, interactive=False)
            cmp_btn.click(
                fn=handle_evaluate_compare,
                inputs=[cmp_manifest, cmp_rf_model, cmp_cnn_model, cmp_norm_stats, cmp_test_frac],
                outputs=cmp_log,
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
                map_kml    = gr.Textbox(label="Detections KML path", placeholder="data/output/detections.kml")
                map_labels = gr.Textbox(label="Labels directory",    placeholder="data/labels/")
            with gr.Row():
                map_hydro  = gr.Textbox(label="Hydrography directory (optional)", placeholder="data/hydrography/")
                map_basemap = gr.Dropdown(choices=["Satellite", "OpenStreetMap"], value="Satellite",
                                          label="Base map")
            with gr.Row():
                map_show_det    = gr.Checkbox(label="Show detections",      value=True)
                map_show_labels = gr.Checkbox(label="Show training labels", value=True)
                map_show_hydro  = gr.Checkbox(label="Show hydrography",     value=True)
            map_btn  = gr.Button("Load Map", variant="primary")
            map_html = gr.HTML()
            map_btn.click(
                fn=handle_load_map,
                inputs=[map_kml, map_labels, map_hydro, map_basemap,
                        map_show_det, map_show_labels, map_show_hydro],
                outputs=map_html,
            )

    # Wire detect button here so map_kml is in scope
    det_btn.click(
        fn=handle_detect,
        inputs=[det_imagery, det_method, det_rf_model, det_cnn_model,
                det_norm_stats, det_hydro, det_threshold, det_output],
        outputs=[det_log, det_file, map_kml],
    )


demo.queue()

if __name__ == "__main__":
    demo.launch()
