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
import datetime
import json
import os
import queue
import tempfile
import threading
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

# --------------------------------------------------------------------------- #
# Project paths — G1.1: one shared Project panel, everything else derived
# --------------------------------------------------------------------------- #

def _default_data_dir() -> Path:
    """Locate the data directory holding imagery/, labels/ and hydrography/.

    data/ is gitignored, so a git worktree of this repo doesn't have one; fall
    back to the main checkout's data/ so the GUI still opens on the real data.
    Override with the CASTOR_DATA environment variable.
    """
    env = os.environ.get("CASTOR_DATA")
    if env:
        return Path(env)

    repo_root = Path(__file__).parent.parent
    candidates = [repo_root / "data"]
    parts = repo_root.parts
    if ".claude" in parts:                      # .../<main checkout>/.claude/worktrees/<name>
        candidates.append(Path(*parts[:parts.index(".claude")]) / "data")
    for candidate in candidates:
        if (candidate / "imagery").is_dir() or (candidate / "labels").is_dir():
            return candidate
    return repo_root / "data"


def _default_subdir(name: str) -> str:
    """Path to a standard data subdirectory, or "" when it doesn't exist."""
    path = DEFAULT_DATA_DIR / name
    return str(path) if path.is_dir() else ""


DEFAULT_DATA_DIR = _default_data_dir()
DEFAULT_PROJECT_DIR = str(DEFAULT_DATA_DIR)
RF_MODEL_FILENAME = "model.pkl"
CNN_MODEL_FILENAME = "beaver_cnn_v1.pth"
NORM_STATS_FILENAME = "norm_stats.json"


def derive_paths(project_dir: str, rf_model_override: str = "") -> dict:
    """Derive every standard sub-path from a project directory.

    Everything lives under <project_dir>: models/, chips/, output/. The RF
    model file itself can be overridden (Advanced: override RF model path)
    without changing where anything else lives; its metadata sidecar
    (<model>.json, written by the training pipeline) follows it.
    """
    project_dir = (project_dir or "").strip() or DEFAULT_PROJECT_DIR
    p = Path(project_dir)
    models_dir = p / "models"
    chips_dir = p / "chips"

    override = (rf_model_override or "").strip()
    rf_model = Path(override) if override else models_dir / RF_MODEL_FILENAME

    return {
        "project_dir": str(p),
        "models_dir": str(models_dir),
        "rf_model": str(rf_model),
        "rf_sidecar": str(rf_model.with_suffix(".json")),
        "chips_dir": str(chips_dir),
        "manifest": str(chips_dir / "manifest.csv"),
        "oof_csv": str(chips_dir / "oof.csv"),
        "cnn_model": str(models_dir / CNN_MODEL_FILENAME),
        "norm_stats": str(models_dir / NORM_STATS_FILENAME),
        "output_dir": str(p / "output"),
    }


def make_output_path(project_dir: str, method: str) -> str:
    """Timestamped detection output path — G3.x."""
    paths = derive_paths(project_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return str(Path(paths["output_dir"]) / f"detections_{method}_{ts}.kml")


def list_output_kmls(project_dir: str) -> list[str]:
    """KML files under <project>/output, newest first — G3.x past-runs dropdown."""
    out_dir = Path(derive_paths(project_dir)["output_dir"])
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.kml"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [str(f) for f in files]


def list_model_files(models_dir: str, suffix: str) -> list[str]:
    """Files with the given suffix under models_dir — G3.x dropdown instead of Browse."""
    p = Path(models_dir)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob(f"*{suffix}"))


def read_model_sidecar(model_path: str) -> dict | None:
    """Read <model>.json next to an RF model, if present. The training
    pipeline (another agent, same wave) is expected to write keys including
    created, n_chips, chips_by_type, feature_length, recommended_threshold
    (float|None) and cv (dict with roc_auc, pr_auc, per_type, ... or None).
    Returns None if the sidecar doesn't exist or can't be parsed — callers
    must treat that as "no metadata available" rather than an error."""
    if not model_path:
        return None
    try:
        sidecar = Path(model_path).with_suffix(".json")
    except Exception:
        return None
    if not sidecar.exists():
        return None
    try:
        with open(sidecar) as f:
            return json.load(f)
    except Exception:
        return None


def build_status_line(rf_model_path: str, labels_dir: str = "") -> str:
    """Header status line — G3.x: 'Model: trained <date> · <n> chips · CV
    PR-AUC x · recommended threshold y', plus a staleness warning if any
    label file is newer than the model."""
    meta = read_model_sidecar(rf_model_path)
    if not meta:
        if rf_model_path and Path(rf_model_path).exists():
            return "**Model:** trained (no metadata sidecar found — run Evaluate for CV stats)"
        return "**Model:** not trained yet — use the Train tab."

    parts = []
    created = meta.get("created")
    if created:
        parts.append(f"trained {created}")
    n_chips = meta.get("n_chips")
    if n_chips is not None:
        parts.append(f"{n_chips} chips")
    cv = meta.get("cv") if isinstance(meta.get("cv"), dict) else None
    if cv and cv.get("pr_auc") is not None:
        parts.append(f"CV PR-AUC {cv['pr_auc']:.2f}")
    rt = meta.get("recommended_threshold")
    if rt is not None:
        parts.append(f"recommended threshold {rt:.2f}")

    line = "**Model:** " + (" · ".join(parts) if parts else "trained (empty metadata)")

    try:
        if labels_dir and rf_model_path and Path(rf_model_path).exists():
            model_mtime = Path(rf_model_path).stat().st_mtime
            label_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")
            newest_label = max((Path(f).stat().st_mtime for f in label_files), default=0.0)
            if newest_label > model_mtime:
                line += "  \n⚠️ Labels changed since this model was trained — consider retraining."
    except Exception:
        pass
    return line


# --------------------------------------------------------------------------- #
# Persistent settings — G1.4
# --------------------------------------------------------------------------- #

def _settings_path() -> Path:
    """Read CASTOR_SETTINGS fresh on every call (not cached at import time) so
    tests/dev tooling can point the app at a tmp file for the whole process."""
    env = os.environ.get("CASTOR_SETTINGS")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "data" / "settings.json"


_SETTINGS_KEYS = [
    # Project panel
    "project_dir", "imagery_dir", "labels_dir", "hydro_dir", "rf_model_override",
    # Train
    "train_augment", "train_flood_samples", "train_hydro_negatives",
    "train_neg_ratio", "train_run_cv",
    # Evaluate
    "ev_radius", "ev_n_splits", "ev_per_class",
    "audit_low", "audit_high", "audit_include_auto_neg",
    # Detect
    "det_threshold", "det_min_area", "det_seed_threshold", "det_no_smooth",
    "det_output_override",
    # Map & Review
    "map_basemap", "map_show_det", "map_show_labels", "map_show_hydro",
    "map_show_audit", "map_conf_threshold", "map_kml",
    "diag_lon", "diag_lat",
    # Experimental (CNN)
    "cnn_epochs", "cnn_lr", "cmp_test_frac",
    "exp_det_method", "exp_det_threshold", "exp_det_output_override",
]

# Legacy (pre-G1.1) settings.json had ~30 per-tab path fields instead of one
# shared Project panel. Presence of any of these marks an old-format file.
_LEGACY_KEY_HINTS = (
    "rf_imagery", "rf_labels", "rf_hydro", "det_imagery", "det_hydro",
    "ov_imagery", "ov_labels", "map_labels", "map_hydro", "det_rf_model",
    "ev_rf_model", "cnn_imagery", "cnn_labels", "cnn_hydro", "diag_imagery",
)


def migrate_settings(old: dict) -> dict:
    """Migrate a legacy (pre-G1.1) settings.json to the new project-panel
    schema. Already-migrated settings (containing "project_dir") and
    unrecognised/empty settings both pass through unchanged, so this is safe
    to call on any settings.json this app has ever written."""
    if not old:
        return {}
    if "project_dir" in old:
        return dict(old)
    if not any(k in old for k in _LEGACY_KEY_HINTS):
        return dict(old)

    imagery_dir = (
        old.get("rf_imagery") or old.get("ov_imagery") or old.get("det_imagery")
        or old.get("cnn_imagery") or old.get("diag_imagery") or ""
    )
    labels_dir = (
        old.get("rf_labels") or old.get("ov_labels") or old.get("map_labels")
        or old.get("cnn_labels") or ""
    )
    hydro_dir = (
        old.get("rf_hydro") or old.get("det_hydro") or old.get("map_hydro")
        or old.get("cnn_hydro") or ""
    )

    project_dir = DEFAULT_PROJECT_DIR
    if imagery_dir:
        parent = str(Path(imagery_dir).parent)
        if parent not in ("", "."):
            project_dir = parent

    default_rf_model = str(Path(project_dir) / "models" / RF_MODEL_FILENAME)
    rf_model_override = ""
    for key in ("det_rf_model", "rf_model", "ev_rf_model", "diag_rf_model"):
        val = old.get(key)
        if val and val != default_rf_model:
            rf_model_override = val
            break

    det_method = old.get("det_method")
    if det_method not in ("rf", "cnn", "both"):
        det_method = "rf"

    return {
        "project_dir": project_dir,
        "imagery_dir": imagery_dir,
        "labels_dir": labels_dir,
        "hydro_dir": hydro_dir,
        "rf_model_override": rf_model_override,
        "train_flood_samples": int(old.get("rf_flood_samples", 0) or 0),
        "train_hydro_negatives": bool(old.get("rf_hydro_negatives", False)),
        "exp_det_method": det_method if det_method in ("cnn", "both") else "cnn",
        "det_threshold": float(old.get("det_threshold", 0.5) or 0.5),
        "ev_radius": float(old.get("ev_radius", 500) or 500),
        "ev_per_class": bool(old.get("ev_per_class", False)),
        "cnn_epochs": int(old.get("cnn_epochs", 30) or 30),
        "cnn_lr": float(old.get("cnn_lr", 0.001) or 0.001),
        "cmp_test_frac": float(old.get("cmp_test_frac", 0.2) or 0.2),
        "map_kml": old.get("map_kml", ""),
    }


def _load_settings() -> dict:
    try:
        path = _settings_path()
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            return migrate_settings(raw)
    except Exception:
        pass
    return {}


def _save_settings(settings: dict) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def handle_autosave(*values) -> str:
    """G1.4 — autosave on change; replaces the old explicit 'Save as defaults' button."""
    settings = dict(zip(_SETTINGS_KEYS, values))
    try:
        _save_settings(settings)
        return f"Settings saved {datetime.datetime.now().strftime('%H:%M:%S')}"
    except Exception as exc:
        return f"ERROR saving settings: {exc}"


_s = _load_settings()

# Saved value wins; otherwise fall back to the detected data directory.
_PROJECT_DIR_VALUE = _s.get("project_dir") or DEFAULT_PROJECT_DIR


_MAX_LOG_HISTORY = 5


def _append_log_history(log: str, history: list) -> tuple[list, str]:
    """Prepend the completed run log to the history list (newest first)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"─── {ts} ───\n{(log or '').rstrip()}"
    history = list(history or [])[-(_MAX_LOG_HISTORY - 1):]
    history.append(entry)
    return history, "\n\n".join(reversed(history))


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
    hydro_dir: str,
    project_dir: str,
    rf_model_override: str,
    augment: int,
    flood_samples: int,
    hydro_negatives: bool,
    neg_ratio: float,
    run_cv: bool = True,
) -> None:
    from cli import cmd_train
    paths = derive_paths(project_dir, rf_model_override)
    Path(paths["chips_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    cmd_train(argparse.Namespace(
        imagery=imagery_dir,
        labels=labels_dir,
        model=paths["rf_model"],
        hydro=hydro_dir or None,
        # G1.5 — always keep chips in the project chip dir so Evaluate works right away.
        chip_dir=paths["chips_dir"],
        augment_positives=augment,
        flood_samples=flood_samples,
        no_hydro_negatives=not hydro_negatives,
        neg_ratio=neg_ratio,
        no_cv=not run_cv,
    ))


def handle_train_rf(
    imagery_dir: str,
    labels_dir: str,
    hydro_dir: str,
    project_dir: str,
    rf_model_override: str,
    augment: float,
    flood_samples: float,
    hydro_negatives: bool,
    neg_ratio: float,
    run_cv: bool = True,
    progress: gr.Progress = gr.Progress(),
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory is required — set it in the Project panel above."
        return
    if not labels_dir or not labels_dir.strip():
        yield "ERROR: Labels directory is required — set it in the Project panel above."
        return
    if int(flood_samples) > 0 and not (hydro_dir and hydro_dir.strip()):
        yield "ERROR: Hydrography directory is required when flood samples > 0."
        return
    progress(0.02, desc="Extracting training chips ...")
    last_log = ""
    for log in _stream(
        _do_train_rf,
        imagery_dir.strip(), labels_dir.strip(), (hydro_dir or "").strip(),
        project_dir.strip(), (rf_model_override or "").strip(),
        int(augment), int(flood_samples), bool(hydro_negatives), float(neg_ratio),
        bool(run_cv),
    ):
        new = log[len(last_log):]
        if "Training Random Forest" in new:
            progress(0.7, desc="Training Random Forest ...")
        elif "Model saved to" in new:
            progress(0.95, desc="Finishing ...")
        last_log = log
        yield log
    progress(1.0, desc="Done")
    yield last_log


def handle_post_train_cv(
    run_cv: bool,
    project_dir: str,
    rf_model_override: str,
    cluster_radius: float,
    n_splits: float,
    per_class: bool,
):
    """After Train RF finishes: chain into Evaluate RF if the "Run spatial
    cross-validation after training" checkbox is on (G1.2). Leaves the
    Evaluate log untouched (gr.update(), a no-op) when the box is unchecked."""
    if not run_cv:
        yield gr.update()
        return
    yield "Training finished — running spatial cross-validation ...\n"
    for log in handle_evaluate_rf(project_dir, rf_model_override, cluster_radius, n_splits, per_class):
        yield log


# --------------------------------------------------------------------------- #
# Train CNN backend (Experimental)
# --------------------------------------------------------------------------- #

def _do_train_cnn(
    imagery_dir: str,
    labels_dir: str,
    hydro_dir: str,
    project_dir: str,
    epochs: int,
    lr: float,
) -> None:
    from cli import cmd_cnn_train
    paths = derive_paths(project_dir)
    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    cmd_cnn_train(argparse.Namespace(
        imagery=imagery_dir,
        labels=labels_dir,
        model=paths["cnn_model"],
        norm_stats=paths["norm_stats"],
        hydro=hydro_dir or None,
        epochs=epochs,
        lr=lr,
    ))


def handle_train_cnn(
    imagery_dir: str,
    labels_dir: str,
    hydro_dir: str,
    project_dir: str,
    epochs: float,
    lr: float,
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory is required — set it in the Project panel above."
        return
    if not labels_dir or not labels_dir.strip():
        yield "ERROR: Labels directory is required — set it in the Project panel above."
        return
    yield from _stream(
        _do_train_cnn,
        imagery_dir.strip(), labels_dir.strip(), (hydro_dir or "").strip(),
        project_dir.strip(), int(epochs), float(lr),
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
    min_area: float,
    seed_threshold: float | None,
    no_smooth: bool,
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
        min_area=min_area,
        seed_threshold=seed_threshold,
        no_smooth=no_smooth,
    ))


def handle_detect(
    imagery_dir: str,
    method: str,
    hydro_dir: str,
    project_dir: str,
    rf_model_override: str,
    threshold: float,
    output_override: str,
    min_area: float,
    seed_threshold,
    no_smooth: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not imagery_dir or not imagery_dir.strip():
        yield "ERROR: Imagery directory is required — set it in the Project panel above.", None, gr.update()
        return
    paths = derive_paths(project_dir, rf_model_override)
    if method in ("rf", "both") and not Path(paths["rf_model"]).exists():
        yield f"ERROR: RF model not found at {paths['rf_model']}. Train a model first.", None, gr.update()
        return
    if method in ("cnn", "both") and not Path(paths["cnn_model"]).exists():
        yield f"ERROR: CNN model not found at {paths['cnn_model']}. Train a CNN first.", None, gr.update()
        return

    out = (output_override or "").strip() or make_output_path(project_dir, method)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    seed = float(seed_threshold) if seed_threshold not in (None, "") else None

    try:
        total_tiles = len(_find_files(imagery_dir.strip(), ".jp2"))
    except Exception:
        total_tiles = 0
    tiles_done = 0
    progress(0, desc=f"0 / {total_tiles} tiles")
    last_log = ""
    for log in _stream(
        _do_detect,
        imagery_dir.strip(), method, paths["rf_model"], paths["cnn_model"], paths["norm_stats"],
        (hydro_dir or "").strip(), float(threshold), out,
        float(min_area), seed, bool(no_smooth),
    ):
        new = log[len(last_log):]
        tiles_done += new.count("Processing ")
        if total_tiles > 0:
            progress(min(tiles_done / total_tiles, 0.99), desc=f"{tiles_done} / {total_tiles} tiles")
        last_log = log
        yield log, None, gr.update()
    progress(1.0, desc="Done")
    kml_exists = out and Path(out).exists()
    if kml_exists:
        dropdown_update = gr.update(choices=list_output_kmls(project_dir), value=out)
    else:
        dropdown_update = gr.update()
    yield last_log, (out if kml_exists else None), dropdown_update


def handle_refresh_threshold(project_dir: str, rf_model_override: str):
    """G1.2 — threshold defaults to the sidecar's recommended_threshold when
    available, shown as 'recommended: 0.33'."""
    paths = derive_paths(project_dir, rf_model_override)
    meta = read_model_sidecar(paths["rf_model"])
    rt = (meta or {}).get("recommended_threshold")
    if rt is not None:
        return gr.update(value=float(rt), label=f"Confidence threshold (recommended: {rt:.2f})")
    return gr.update(label="Confidence threshold (recommended: n/a — train & evaluate first)")


def handle_refresh_status(project_dir: str, rf_model_override: str, labels_dir: str):
    paths = derive_paths(project_dir, rf_model_override)
    return build_status_line(paths["rf_model"], labels_dir)


def handle_refresh_rf_model_choices(project_dir: str):
    paths = derive_paths(project_dir)
    return gr.update(choices=list_model_files(paths["models_dir"], ".pkl"))


# --------------------------------------------------------------------------- #
# Map view
# --------------------------------------------------------------------------- #

_MAP_NS = "http://www.opengis.net/kml/2.2"

# Folder-based label type (as returned by training_data.parse_kml_labels) -> colour
# group. Mirrors training_data.FEATURE_TO_LABEL's name variants so the map/Data
# check agree with what training actually uses, plus dam/lodge/other (excluded
# from training but still labelled in KML).
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


def _add_audit_layer(
    m,
    oof_csv: str,
    low: float = 0.2,
    high: float = 0.8,
    include_auto_neg: bool = False,
) -> list[tuple[float, float]]:
    """Render suspicious label points (R1.5 / G2.2) as red rings with a
    tooltip. Reuses the same map (and hence the same click handler) as
    everything else, so clicking one of these rings still populates
    window._mapClickLat/Lon for the Diagnose button below."""
    import folium
    import label_audit
    from pyproj import Transformer

    if not oof_csv or not Path(oof_csv).exists():
        return []
    try:
        rows = label_audit.audit_labels(oof_csv, low=low, high=high, include_auto_negatives=include_auto_neg)
    except Exception as exc:
        print(f"Warning: could not audit labels: {exc}")
        return []
    if not rows:
        return []

    to_wgs84 = Transformer.from_crs(3067, 4326, always_xy=True)
    group = folium.FeatureGroup(name="Label audit (suspicious)", show=True)
    bounds: list[tuple[float, float]] = []
    for r in rows:
        lon, lat = to_wgs84.transform(r["x"], r["y"])
        pt = (lat, lon)
        bounds.append(pt)
        label_name = "positive" if r["label"] == 1 else "negative"
        folium.CircleMarker(
            location=pt,
            radius=11,
            color="#ff0000",
            fill=False,
            weight=2.5,
            tooltip=(
                f"{r['feature_type']} ({label_name}) — "
                f"mean OOF prob {r['mean_prob']:.2f} over {r['n_chips']} chip(s)"
            ),
        ).add_to(group)
    group.add_to(m)
    return bounds


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
    oof_csv: str = "",
    show_audit: bool = False,
    audit_low: float = 0.2,
    audit_high: float = 0.8,
    audit_include_auto_neg: bool = False,
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

    if show_audit and oof_csv:
        bounds.extend(_add_audit_layer(m, oof_csv, audit_low, audit_high, audit_include_auto_neg))

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
      <span style="color:#1a6aa8">&#9644;</span> streams / water bodies<br>
      <b>Label audit</b><br>
      <span style="color:#ff0000">&#9711;</span> suspicious (label disagrees with model)
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
      Click on the map to select a point for diagnosis (works on detections, labels, and audit rings too)
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
    show_audit: bool,
    confidence_threshold: float,
    project_dir: str,
    rf_model_override: str,
    audit_low: float = 0.2,
    audit_high: float = 0.8,
    audit_include_auto_neg: bool = False,
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
    paths = derive_paths(project_dir, rf_model_override)
    oof_csv = paths["oof_csv"] if show_audit else ""
    try:
        return _build_map(kml_path, labels_dir, hydro_dir, basemap,
                          show_detections, show_labels, show_hydro,
                          float(confidence_threshold or 0.0),
                          oof_csv=oof_csv, show_audit=bool(show_audit),
                          audit_low=float(audit_low), audit_high=float(audit_high),
                          audit_include_auto_neg=bool(audit_include_auto_neg))
    except Exception as exc:
        return f"<p style='color:red'><b>ERROR:</b> {exc}</p>"


# --------------------------------------------------------------------------- #
# Evaluate RF backend
# --------------------------------------------------------------------------- #

def _do_evaluate_rf(
    project_dir: str,
    rf_model_override: str,
    cluster_radius: float,
    n_splits: int,
    per_class: bool,
) -> None:
    from cli import cmd_evaluate_rf
    paths = derive_paths(project_dir, rf_model_override)
    rf_model = paths["rf_model"] if Path(paths["rf_model"]).exists() else None
    cmd_evaluate_rf(argparse.Namespace(
        manifest=paths["manifest"],
        rf_model=rf_model,
        cluster_radius=cluster_radius,
        n_splits=n_splits,
        oof_path=paths["oof_csv"],
        cache_dir=paths["chips_dir"],
        no_cache=False,
        per_class=per_class,
    ))


def handle_evaluate_rf(
    project_dir: str,
    rf_model_override: str,
    cluster_radius: float,
    n_splits: float,
    per_class: bool,
):
    paths = derive_paths(project_dir, rf_model_override)
    if not Path(paths["manifest"]).exists():
        yield f"ERROR: No training manifest at {paths['manifest']}. Run Train first (with chips kept)."
        return
    yield from _stream(
        _do_evaluate_rf,
        project_dir.strip(), (rf_model_override or "").strip(),
        float(cluster_radius), int(n_splits), bool(per_class),
    )


def handle_confusion_matrix(project_dir: str, rf_model_override: str):
    paths = derive_paths(project_dir, rf_model_override)
    return _confusion_matrix_image(paths["manifest"], paths["rf_model"])


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


def handle_audit_table(
    project_dir: str,
    rf_model_override: str,
    low: float = 0.2,
    high: float = 0.8,
    include_auto_neg: bool = False,
) -> list[list]:
    """R1.5 / G2.2 — suspicious label points as table rows: type, lat, lon,
    mean OOF probability, label, chip count."""
    import label_audit
    paths = derive_paths(project_dir, rf_model_override)
    oof_path = paths["oof_csv"]
    if not Path(oof_path).exists():
        return []
    try:
        rows = label_audit.audit_labels(
            oof_path, low=float(low), high=float(high), include_auto_negatives=bool(include_auto_neg),
        )
    except Exception as exc:
        print(f"Warning: could not audit labels: {exc}")
        return []
    if not rows:
        return []
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(3067, 4326, always_xy=True)
    table = []
    for r in rows:
        lon, lat = to_wgs84.transform(r["x"], r["y"])
        label_name = "positive" if r["label"] == 1 else "negative"
        table.append([r["feature_type"], label_name, round(lat, 6), round(lon, 6),
                      round(r["mean_prob"], 3), r["n_chips"]])
    return table


# --------------------------------------------------------------------------- #
# Evaluate RF vs CNN backend (Experimental)
# --------------------------------------------------------------------------- #

def _do_evaluate_compare(
    project_dir: str,
    rf_model_override: str,
    test_fraction: float,
) -> None:
    from cli import cmd_evaluate
    paths = derive_paths(project_dir, rf_model_override)
    cmd_evaluate(argparse.Namespace(
        manifest=paths["manifest"],
        rf_model=paths["rf_model"],
        cnn_model=paths["cnn_model"],
        norm_stats=paths["norm_stats"],
        test_manifest=None,
        test_fraction=test_fraction,
    ))


def handle_evaluate_compare(
    project_dir: str,
    rf_model_override: str,
    test_fraction: float,
):
    paths = derive_paths(project_dir, rf_model_override)
    if not Path(paths["manifest"]).exists():
        yield f"ERROR: No training manifest at {paths['manifest']}. Run Train (RF) with chips first."
        return
    if not Path(paths["rf_model"]).exists():
        yield f"ERROR: No RF model at {paths['rf_model']}. Train RF first."
        return
    if not Path(paths["cnn_model"]).exists():
        yield f"ERROR: No CNN model at {paths['cnn_model']}. Train a CNN first."
        return
    yield from _stream(
        _do_evaluate_compare,
        project_dir.strip(), (rf_model_override or "").strip(), float(test_fraction),
    )


# --------------------------------------------------------------------------- #
# Diagnose Point (lives inside the Map & Review tab)
# --------------------------------------------------------------------------- #

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
        return None, None, None, None, "ERROR: Imagery directory is required — set it in the Project panel above."
    if not rf_model_path or not rf_model_path.strip() or not Path(rf_model_path).exists():
        return None, None, None, None, f"ERROR: RF model not found at {rf_model_path!r}. Train a model first."
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


def handle_diagnose_proj(
    lon: float,
    lat: float,
    imagery_dir: str,
    project_dir: str,
    rf_model_override: str,
):
    paths = derive_paths(project_dir, rf_model_override)
    return handle_diagnose(lon, lat, imagery_dir, paths["rf_model"])


def handle_map_click_followup(
    status_msg: str,
    lon: float,
    lat: float,
    imagery_dir: str,
    project_dir: str,
    rf_model_override: str,
):
    """After the map-click JS fills diag_lon/diag_lat: run the diagnosis
    automatically — but only if a point was actually clicked (status_msg
    starts with "Selected:"; see the map_diagnose_btn wiring). Otherwise
    leave everything as-is so the "click the map first" message stands.
    Diagnose Point lives in the same tab as the map now, so no tab switch
    is needed (unlike the old separate-tab layout)."""
    if not (status_msg or "").startswith("Selected:"):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    chip_img, ndwi_img, ndvi_img, prob_img, log = handle_diagnose_proj(
        lon, lat, imagery_dir, project_dir, rf_model_override
    )
    return chip_img, ndwi_img, ndvi_img, prob_img, log


# --------------------------------------------------------------------------- #
# Data check (Overview + validation) — G1.2
# --------------------------------------------------------------------------- #

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


def _do_data_check(
    imagery_dir: str,
    labels_dir: str,
    hydro_dir: str,
    project_dir: str,
) -> str:
    import training_data

    paths = derive_paths(project_dir)
    lines: list[str] = []
    warnings: list[str] = []

    # ---- Imagery ----
    lines.append("=== Imagery ===")
    jp2s: list[str] = []
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
        lines.append("  (not specified — set Imagery directory in the Project panel)")

    # ---- Labels ----
    lines.append("\n=== Labels ===")
    all_points: list[tuple] = []
    if labels_dir:
        kml_files = _find_files(labels_dir, ".kml") + _find_files(labels_dir, ".kmz")
        if kml_files:
            lines.append(f"  {len(kml_files)} KML/KMZ file(s)")
            counts: dict[str, int] = {}
            for kp in kml_files:
                try:
                    for point, ftype in training_data.parse_kml_labels(kp):
                        counts[ftype] = counts.get(ftype, 0) + 1
                        all_points.append((point, ftype))
                except Exception as exc:
                    warnings.append(f"Could not parse {kp}: {exc}")
            unknown_total = 0
            for k, v in sorted(counts.items()):
                tag = ""
                if k not in training_data.FEATURE_TO_LABEL and k not in training_data.DEFAULT_EXCLUDE:
                    tag = "  (unrecognised — dropped by training)"
                    unknown_total += v
                lines.append(f"    {k}: {v}{tag}")
            if unknown_total:
                warnings.append(
                    f"{unknown_total} label(s) have an unrecognised folder/type name and "
                    "will be dropped by training"
                )
            if not counts:
                warnings.append("No placemark points found in KML files")
        else:
            lines.append(f"  No KML/KMZ files found in {labels_dir!r}")
            warnings.append(f"No labels found in {labels_dir!r}")
    else:
        lines.append("  (not specified — set Labels directory in the Project panel)")

    # ---- Labels outside imagery coverage ----
    if all_points and jp2s:
        try:
            import rasterio
            tile_bboxes = []
            for f in jp2s:
                with rasterio.open(f) as src:
                    b = src.bounds
                    tile_bboxes.append((b.left, b.bottom, b.right, b.top))
            outside = 0
            for point, _ in all_points:
                x, y = point.x, point.y
                if not any(minx <= x <= maxx and miny <= y <= maxy for minx, miny, maxx, maxy in tile_bboxes):
                    outside += 1
            if outside:
                warnings.append(f"{outside} label(s) fall outside any imagery tile's coverage")
        except Exception as exc:
            warnings.append(f"Could not check label/tile coverage: {exc}")

    # ---- Hydrography ----
    lines.append("\n=== Hydrography ===")
    if hydro_dir:
        try:
            from masking import _resolve_files
            files = _resolve_files(hydro_dir)
            if files:
                lines.append(f"  {len(files)} file(s) found")
            else:
                lines.append(f"  No hydrography files found in {hydro_dir!r}")
                warnings.append(f"No hydrography files found in {hydro_dir!r}")
        except Exception as exc:
            warnings.append(f"Could not read hydrography: {exc}")
    else:
        lines.append("  (not specified — optional, but recommended for stream-filtered detection)")

    # ---- Models ----
    lines.append("\n=== Models ===")
    models_dir = paths["models_dir"]
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
        lines.append(f"  Directory not found: {models_dir!r} (created on first train)")

    # ---- Training chips ----
    lines.append("\n=== Training Chips ===")
    manifest_path = Path(paths["manifest"])
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
        lines.append(f"  No manifest.csv yet at {manifest_path} — run Train to create it")

    if warnings:
        lines.append("\n=== Warnings ===")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


def handle_data_check(
    imagery_dir: str,
    labels_dir: str,
    hydro_dir: str,
    project_dir: str,
) -> str:
    try:
        return _do_data_check(
            (imagery_dir or "").strip(), (labels_dir or "").strip(),
            (hydro_dir or "").strip(), (project_dir or "").strip(),
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


def handle_chip_gallery(project_dir: str) -> list:
    if not project_dir or not project_dir.strip():
        return []
    try:
        paths = derive_paths(project_dir.strip())
        return _load_chip_gallery(paths["chips_dir"])
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Gradio layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="CastorDetector") as demo:
    gr.Markdown("# CastorDetector\nBeaver activity detection in MML aerial imagery.")

    _project_configured = bool(
        _s.get("imagery_dir") or _s.get("labels_dir")
        or _default_subdir("imagery") or _default_subdir("labels")
    )

    # ------------------------------------------------------------------ #
    # G1.1 — Shared Project panel (collapsible once set)
    # ------------------------------------------------------------------ #
    with gr.Accordion("Project", open=not _project_configured) as project_accordion:
        with gr.Row():
            proj_imagery = gr.Textbox(
                label="Imagery directory", placeholder="data/imagery/",
                value=_s.get("imagery_dir") or _default_subdir("imagery"),
            )
            proj_labels = gr.Textbox(
                label="Labels directory", placeholder="data/labels/",
                value=_s.get("labels_dir") or _default_subdir("labels"),
            )
        with gr.Row():
            proj_hydro = gr.Textbox(
                label="Hydrography directory (optional)", placeholder="data/hydrography/",
                value=_s.get("hydro_dir") or _default_subdir("hydrography"),
            )
            proj_dir = gr.Textbox(
                label="Project directory", placeholder=DEFAULT_PROJECT_DIR,
                value=_PROJECT_DIR_VALUE,
                info="Models, chips, and detection outputs are all derived from this directory.",
            )
        with gr.Accordion("Advanced: override model path", open=False):
            proj_rf_model_override = gr.Dropdown(
                label="RF model file (.pkl) — default: <project>/models/model.pkl",
                choices=list_model_files(
                    derive_paths(_PROJECT_DIR_VALUE)["models_dir"], ".pkl"
                ),
                value=_s.get("rf_model_override", ""),
                allow_custom_value=True,
                info="Leave blank to use the default. Pick a saved .pkl to compare models without retraining.",
            )
            proj_refresh_models_btn = gr.Button("Refresh list", size="sm", scale=0)
        status_line = gr.Markdown(build_status_line(
            derive_paths(_PROJECT_DIR_VALUE,
                         _s.get("rf_model_override", ""))["rf_model"],
            _s.get("labels_dir", ""),
        ))

    save_status = gr.Textbox(
        label="", interactive=False, max_lines=1, show_label=False,
        placeholder="Settings are saved automatically as you edit them.",
    )

    with gr.Tabs() as main_tabs:

        # ------------------------------------------------------------------ #
        # 1. Data check
        # ------------------------------------------------------------------ #
        with gr.Tab("1. Data check"):
            gr.Markdown(
                "## Data Check\n"
                "Validate your project before training or detection: confirms paths exist, "
                "counts tiles and labels by type, flags unrecognised label types and labels "
                "outside imagery coverage, and confirms hydrography is available."
            )
            dc_btn = gr.Button("Scan", variant="primary")
            dc_out = gr.Textbox(label="Summary", lines=24, interactive=False)
            gr.Markdown("### Chip sample (CIR false-colour)")
            dc_gallery = gr.Gallery(
                label="Training chips — positives then negatives (up to 12 each)",
                columns=6, height=320, object_fit="contain",
            )
            dc_event = dc_btn.click(
                fn=handle_data_check,
                inputs=[proj_imagery, proj_labels, proj_hydro, proj_dir],
                outputs=dc_out,
            )
            dc_event.then(fn=handle_chip_gallery, inputs=[proj_dir], outputs=[dc_gallery])

        # ------------------------------------------------------------------ #
        # 2. Train
        # ------------------------------------------------------------------ #
        with gr.Tab("2. Train"):
            gr.Markdown(
                "## Train Random Forest\n"
                "Extract chips from labelled imagery and train the Random Forest classifier. "
                "Chips are always kept in `<project>/chips/` so Evaluate can run right away."
            )
            tr_run_cv = gr.Checkbox(
                label="Run spatial cross-validation after training", value=bool(_s.get("train_run_cv", True)),
                info="Recommended — populates the Evaluate tab automatically when training finishes.",
            )
            with gr.Accordion("Advanced", open=False):
                tr_augment = gr.Slider(
                    minimum=0, maximum=12, value=int(_s.get("train_augment", 6)), step=1,
                    label="Augment positives (extra offset chips per label)",
                    info="More augmentation helps with few labels but can overfit to a single feature.",
                )
                tr_flood_samples = gr.Number(
                    label="Flood samples from hydrography", value=int(_s.get("train_flood_samples", 0)),
                    precision=0, minimum=0,
                    info="Extra positives sampled from mapped flood areas (MML tulvaalue). Requires hydrography.",
                )
                tr_hydro_negatives = gr.Checkbox(
                    label="Restrict auto-negatives to stream corridor",
                    value=bool(_s.get("train_hydro_negatives", False)),
                    info="Requires hydrography. Leave off (default) when using generic dead_forest/flood labels "
                         "so negatives are sampled from the full imagery extent, per the training workflow.",
                )
                tr_neg_ratio = gr.Slider(
                    minimum=0.25, maximum=4.0, value=float(_s.get("train_neg_ratio", 1.0)), step=0.25,
                    label="Negative:positive ratio",
                    info="Auto-negative chip count relative to positives after augmentation.",
                )
            with gr.Row():
                tr_btn  = gr.Button("Train RF", variant="primary")
                tr_stop = gr.Button("Stop", variant="stop")
            tr_log = gr.Textbox(label="Log", lines=15, interactive=False)
            tr_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                tr_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            tr_event = tr_btn.click(
                fn=handle_train_rf,
                inputs=[proj_imagery, proj_labels, proj_hydro, proj_dir, proj_rf_model_override,
                        tr_augment, tr_flood_samples, tr_hydro_negatives, tr_neg_ratio, tr_run_cv],
                outputs=tr_log,
            )

        # ------------------------------------------------------------------ #
        # 3. Evaluate
        # ------------------------------------------------------------------ #
        with gr.Tab("3. Evaluate"):
            gr.Markdown(
                "## Evaluate RF\n"
                "Pooled out-of-fold spatial cross-validation: label points within the cluster "
                "radius are grouped into the same fold (avoiding the spatial-autocorrelation "
                "leak a random split introduces), each fold's held-out probabilities are pooled "
                "before computing metrics, and results are reported at both a chip level and a "
                "point level. Reports ROC-AUC, PR-AUC, a recommended (max-F1) threshold, and a "
                "per-feature-type breakdown. Results are written to `chips/oof.csv`."
            )
            with gr.Accordion("Advanced", open=False):
                ev_radius = gr.Slider(
                    minimum=100, maximum=2000, value=float(_s.get("ev_radius", 500)), step=50,
                    label="Cluster radius (metres)",
                    info="Label points within this distance are treated as one spatial cluster.",
                )
                ev_n_splits = gr.Slider(
                    minimum=2, maximum=10, value=int(_s.get("ev_n_splits", 5)), step=1,
                    label="CV folds", info="Capped automatically at the number of spatial clusters.",
                )
                ev_per_class = gr.Checkbox(
                    label="Per-feature-type breakdown", value=bool(_s.get("ev_per_class", True)),
                    info="Recall for positive types, specificity for negative types.",
                )
                gr.Markdown("**Label audit thresholds**")
                with gr.Row():
                    audit_low = gr.Slider(
                        minimum=0.0, maximum=0.5, value=float(_s.get("audit_low", 0.2)), step=0.05,
                        label="Flag positives below this probability",
                    )
                    audit_high = gr.Slider(
                        minimum=0.5, maximum=1.0, value=float(_s.get("audit_high", 0.8)), step=0.05,
                        label="Flag negatives above this probability",
                    )
                audit_include_auto_neg = gr.Checkbox(
                    label="Include auto-sampled negatives in the audit",
                    value=bool(_s.get("audit_include_auto_neg", False)),
                    info="Auto-negatives come from random sampling, not a human decision — off by default.",
                )
            with gr.Row():
                ev_btn  = gr.Button("Evaluate RF", variant="primary")
                ev_stop = gr.Button("Stop", variant="stop")
            ev_log = gr.Textbox(label="Results", lines=20, interactive=False)
            ev_cm  = gr.Image(label="Confusion matrix — spatial CV (out-of-fold)", type="numpy", height=320)
            gr.Markdown(
                "### Label audit — suspicious label points\n"
                "Positives the model consistently scores low, and hand-labelled negatives it "
                "scores high. Worth a second look in Google Earth — some are mislabels."
            )
            ev_audit_table = gr.Dataframe(
                headers=["type", "label", "lat", "lon", "mean OOF prob", "n chips"],
                interactive=False,
            )
            ev_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                ev_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            ev_event = ev_btn.click(
                fn=handle_evaluate_rf,
                inputs=[proj_dir, proj_rf_model_override, ev_radius, ev_n_splits, ev_per_class],
                outputs=ev_log,
            )

        # ------------------------------------------------------------------ #
        # 4. Detect
        # ------------------------------------------------------------------ #
        with gr.Tab("4. Detect"):
            gr.Markdown(
                "## Detect (Random Forest)\n"
                "Run the trained RF model on imagery and export detections as a timestamped KML file."
            )
            dt_method_state = gr.State("rf")
            det_threshold_default = float((read_model_sidecar(
                derive_paths(_PROJECT_DIR_VALUE,
                             _s.get("rf_model_override", ""))["rf_model"]
            ) or {}).get("recommended_threshold") or _s.get("det_threshold", 0.5) or 0.5)
            det_threshold_label = "Confidence threshold"
            _det_meta = read_model_sidecar(derive_paths(
                _PROJECT_DIR_VALUE, _s.get("rf_model_override", ""))["rf_model"])
            if _det_meta and _det_meta.get("recommended_threshold") is not None:
                det_threshold_label = f"Confidence threshold (recommended: {_det_meta['recommended_threshold']:.2f})"
            dt_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=det_threshold_default, step=0.05,
                label=det_threshold_label,
            )
            with gr.Accordion("Advanced", open=False):
                dt_min_area = gr.Number(
                    label="Minimum detection area (m²)", value=float(_s.get("det_min_area", 2048.0)),
                    info="Roughly 2 RF patches; a single 64px patch is 1024 m².",
                )
                dt_seed_threshold = gr.Number(
                    label="Seed threshold (blank = auto: threshold + 0.15)",
                    value=_s.get("det_seed_threshold", None),
                    info="A region is kept only if it contains a cell at or above this confidence.",
                )
                dt_no_smooth = gr.Checkbox(
                    label="Disable probability-map smoothing", value=bool(_s.get("det_no_smooth", False)),
                    info="Turns off 3x3 NaN-aware smoothing before hysteresis thresholding.",
                )
                dt_output = gr.Textbox(
                    label="Output KML path (blank = auto-timestamped in <project>/output/)",
                    placeholder="(auto)", value=_s.get("det_output_override", ""),
                )
            with gr.Row():
                dt_btn  = gr.Button("Detect & Export KML", variant="primary")
                dt_stop = gr.Button("Stop", variant="stop")
            dt_log   = gr.Textbox(label="Log", lines=15, interactive=False)
            dt_stats = gr.Textbox(label="Statistics", lines=8, interactive=False)
            dt_file  = gr.File(label="Download KML", interactive=False)
            dt_history_state = gr.State([])
            with gr.Accordion("Previous runs", open=False):
                dt_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
            # dt_btn.click() wired after Map & Review so mp_kml_dropdown is in scope

        # ------------------------------------------------------------------ #
        # 5. Map & Review
        # ------------------------------------------------------------------ #
        with gr.Tab("5. Map & Review"):
            gr.Markdown(
                "## Map & Review\n"
                "View detection polygons, training labels, hydrography, and suspicious label "
                "audit points on an interactive map. Click anywhere on the map (including a "
                "marker) to diagnose that point below."
            )
            with gr.Row():
                mp_kml_dropdown = gr.Dropdown(
                    label="Detections KML (past runs, newest first)",
                    choices=list_output_kmls(_PROJECT_DIR_VALUE),
                    value=_s.get("map_kml", ""), allow_custom_value=True,
                )
                mp_refresh_btn = gr.Button("Refresh list", size="sm", scale=0)
            with gr.Row():
                mp_basemap = gr.Dropdown(
                    choices=["Satellite", "OpenStreetMap"], value=_s.get("map_basemap", "Satellite"),
                    label="Base map",
                )
                mp_filter_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=float(_s.get("map_conf_threshold", 0.75)), step=0.05,
                    label="Minimum confidence to show/export",
                )
            with gr.Row():
                mp_show_det    = gr.Checkbox(label="Show detections",      value=bool(_s.get("map_show_det", True)))
                mp_show_labels = gr.Checkbox(label="Show training labels", value=bool(_s.get("map_show_labels", True)))
                mp_show_hydro  = gr.Checkbox(label="Show hydrography",     value=bool(_s.get("map_show_hydro", True)))
                mp_show_audit  = gr.Checkbox(
                    label="Show label audit (suspicious points)", value=bool(_s.get("map_show_audit", False)),
                    info="Red rings — run Evaluate first to generate oof.csv.",
                )
            mp_btn  = gr.Button("Load Map", variant="primary")
            mp_html = gr.HTML()
            mp_diagnose_btn = gr.Button(
                "Diagnose selected point (click map first)", variant="secondary"
            )
            mp_click_status = gr.Textbox(
                label="", interactive=False, show_label=False, max_lines=1,
                value="Click a point on the map, then press the button above.",
            )
            gr.Markdown("### Export filtered detections")
            mp_export_btn = gr.Button("Export filtered KML (above threshold)", variant="secondary", scale=0)
            mp_export_file = gr.File(label="Filtered KML download", interactive=False)

            gr.Markdown("### Diagnose Point")
            gr.Markdown(
                "Extract the chip at a known WGS84 location, run the RF classifier, and "
                "visualise the spectral signature — useful for false positives and misses."
            )
            with gr.Row():
                dg_lon = gr.Number(value=float(_s.get("diag_lon", 25.0)), label="Longitude (WGS84)")
                dg_lat = gr.Number(value=float(_s.get("diag_lat", 62.0)), label="Latitude (WGS84)")
                dg_btn = gr.Button("Diagnose", variant="primary")
            with gr.Row():
                dg_chip = gr.Image(label="CIR chip (NIR=R, Red=G, Green=B)", type="numpy")
                dg_ndwi = gr.Image(label="NDWI  (blue=water, red=dry)",       type="numpy")
                dg_ndvi = gr.Image(label="NDVI  (green=veg, red=bare)",        type="numpy")
                dg_prob = gr.Image(label="RF probability map (bright=flood)",  type="numpy")
            dg_log = gr.Textbox(label="Prediction & band stats", lines=12, interactive=False)

            dg_btn.click(
                fn=handle_diagnose_proj,
                inputs=[dg_lon, dg_lat, proj_imagery, proj_dir, proj_rf_model_override],
                outputs=[dg_chip, dg_ndwi, dg_ndvi, dg_prob, dg_log],
            )
            mp_btn.click(
                fn=handle_load_map,
                inputs=[mp_kml_dropdown, proj_labels, proj_hydro, mp_basemap,
                        mp_show_det, mp_show_labels, mp_show_hydro, mp_show_audit,
                        mp_filter_threshold, proj_dir, proj_rf_model_override,
                        audit_low, audit_high, audit_include_auto_neg],
                outputs=mp_html,
            )
            mp_refresh_btn.click(
                fn=lambda pd: gr.update(choices=list_output_kmls(pd)),
                inputs=[proj_dir], outputs=[mp_kml_dropdown],
            )
            # Click-to-diagnose: the folium map is embedded via an <iframe srcdoc="...">
            # (see _build_map's click_js), which has its own `window` — separate from
            # this parent Gradio page, where this button's JS runs. The iframe's click
            # handler writes to window.parent too, so plain window._mapClickLat/Lon
            # here (in the parent) sees it. Falls back to the current lon/lat plus a
            # "click first" message if nothing has been clicked yet.
            mp_click_event = mp_diagnose_btn.click(
                fn=None,
                inputs=[dg_lon, dg_lat],
                outputs=[dg_lon, dg_lat, mp_click_status],
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
            mp_click_event.then(
                fn=handle_map_click_followup,
                inputs=[mp_click_status, dg_lon, dg_lat, proj_imagery, proj_dir, proj_rf_model_override],
                outputs=[dg_chip, dg_ndwi, dg_ndvi, dg_prob, dg_log],
            )
            mp_export_btn.click(
                fn=handle_export_filtered_kml,
                inputs=[mp_kml_dropdown, mp_filter_threshold],
                outputs=[mp_export_file],
            )

        # ------------------------------------------------------------------ #
        # 6. Experimental (CNN)
        # ------------------------------------------------------------------ #
        with gr.Tab("6. Experimental (CNN)"):
            gr.Markdown(
                "## Experimental: Prithvi-EO CNN\n"
                "> **⚠ Slow & experimental.** CPU inference is roughly 40 min/tile without a "
                "hydrography mask, ~15 min with one. Training downloads ~454 MB of pretrained "
                "weights from HuggingFace on first run. The Random Forest pipeline (tabs 1–5) "
                "is the primary, supported workflow."
            )
            with gr.Accordion("Train CNN", open=False):
                with gr.Row():
                    xp_epochs = gr.Number(value=int(_s.get("cnn_epochs", 30)), label="Epochs", precision=0)
                    xp_lr     = gr.Number(value=float(_s.get("cnn_lr", 0.001)), label="Learning rate")
                with gr.Row():
                    xp_train_btn  = gr.Button("Train CNN", variant="primary")
                    xp_train_stop = gr.Button("Stop", variant="stop")
                xp_train_log = gr.Textbox(label="Log", lines=15, interactive=False)
                xp_train_history_state = gr.State([])
                with gr.Accordion("Previous runs", open=False):
                    xp_train_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
                xp_train_event = xp_train_btn.click(
                    fn=handle_train_cnn,
                    inputs=[proj_imagery, proj_labels, proj_hydro, proj_dir, xp_epochs, xp_lr],
                    outputs=xp_train_log,
                )

            with gr.Accordion("Evaluate RF vs CNN", open=False):
                gr.Markdown(
                    "> **⚠ IN-SAMPLE** — both models were fit on the full manifest, so no split "
                    "of it is truly held out. Use the **3. Evaluate** tab for spatially rigorous "
                    "RF cross-validation."
                )
                xp_test_frac = gr.Slider(
                    minimum=0.1, maximum=0.5, value=float(_s.get("cmp_test_frac", 0.2)), step=0.05,
                    label="Test fraction",
                )
                with gr.Row():
                    xp_cmp_btn  = gr.Button("Evaluate", variant="primary")
                    xp_cmp_stop = gr.Button("Stop", variant="stop")
                xp_cmp_log = gr.Textbox(label="Results", lines=12, interactive=False)
                xp_cmp_history_state = gr.State([])
                with gr.Accordion("Previous runs", open=False):
                    xp_cmp_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
                xp_cmp_event = xp_cmp_btn.click(
                    fn=handle_evaluate_compare,
                    inputs=[proj_dir, proj_rf_model_override, xp_test_frac],
                    outputs=xp_cmp_log,
                )

            with gr.Accordion("Detect (CNN / both)", open=False):
                xp_det_method = gr.Dropdown(
                    choices=["cnn", "both"], value=_s.get("exp_det_method", "cnn"),
                    label="Method", info="'both' runs RF and CNN and tags agreement.",
                )
                xp_det_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=float(_s.get("exp_det_threshold", 0.5)), step=0.05,
                    label="Confidence threshold",
                )
                xp_det_output = gr.Textbox(
                    label="Output KML path (blank = auto-timestamped in <project>/output/)",
                    placeholder="(auto)", value=_s.get("exp_det_output_override", ""),
                )
                with gr.Row():
                    xp_det_btn  = gr.Button("Detect & Export KML", variant="primary")
                    xp_det_stop = gr.Button("Stop", variant="stop")
                xp_det_log  = gr.Textbox(label="Log", lines=15, interactive=False)
                xp_det_file = gr.File(label="Download KML", interactive=False)
                xp_det_history_state = gr.State([])
                with gr.Accordion("Previous runs", open=False):
                    xp_det_history_text = gr.Textbox(label="", lines=10, interactive=False, show_label=False)
                xp_det_event = xp_det_btn.click(
                    fn=handle_detect,
                    inputs=[proj_imagery, xp_det_method, proj_hydro, proj_dir, proj_rf_model_override,
                            xp_det_threshold, xp_det_output, dt_min_area, dt_seed_threshold, dt_no_smooth],
                    outputs=[xp_det_log, xp_det_file, mp_kml_dropdown],
                )
                xp_det_event.then(
                    fn=_append_log_history, inputs=[xp_det_log, xp_det_history_state],
                    outputs=[xp_det_history_state, xp_det_history_text],
                )

    # ------------------------------------------------------------------ #
    # Cross-tab wiring
    # ------------------------------------------------------------------ #

    # Detect RF button — wired here so mp_kml_dropdown is in scope
    dt_event = dt_btn.click(
        fn=handle_detect,
        inputs=[proj_imagery, dt_method_state, proj_hydro, proj_dir, proj_rf_model_override,
                dt_threshold, dt_output, dt_min_area, dt_seed_threshold, dt_no_smooth],
        outputs=[dt_log, dt_file, mp_kml_dropdown],
    )
    dt_event.then(fn=_detection_stats, inputs=[dt_file], outputs=[dt_stats])
    dt_event.then(fn=_append_log_history, inputs=[dt_log, dt_history_state],
                  outputs=[dt_history_state, dt_history_text])

    # Post-training: chain into Evaluate RF when the checkbox is on
    tr_cv_event = tr_event.then(
        fn=handle_post_train_cv,
        inputs=[tr_run_cv, proj_dir, proj_rf_model_override, ev_radius, ev_n_splits, ev_per_class],
        outputs=[ev_log],
    )
    tr_cv_event.then(fn=handle_confusion_matrix, inputs=[proj_dir, proj_rf_model_override], outputs=[ev_cm])
    tr_cv_event.then(
        fn=handle_audit_table,
        inputs=[proj_dir, proj_rf_model_override, audit_low, audit_high, audit_include_auto_neg],
        outputs=[ev_audit_table],
    )
    tr_event.then(fn=_append_log_history, inputs=[tr_log, tr_history_state],
                  outputs=[tr_history_state, tr_history_text])
    tr_event.then(fn=handle_refresh_status, inputs=[proj_dir, proj_rf_model_override, proj_labels],
                  outputs=[status_line])
    tr_event.then(fn=handle_refresh_threshold, inputs=[proj_dir, proj_rf_model_override],
                  outputs=[dt_threshold])
    tr_event.then(fn=handle_refresh_rf_model_choices, inputs=[proj_dir], outputs=[proj_rf_model_override])

    # Evaluate RF button — confusion matrix + audit table + history
    ev_event.then(fn=handle_confusion_matrix, inputs=[proj_dir, proj_rf_model_override], outputs=[ev_cm])
    ev_event.then(
        fn=handle_audit_table,
        inputs=[proj_dir, proj_rf_model_override, audit_low, audit_high, audit_include_auto_neg],
        outputs=[ev_audit_table],
    )
    ev_event.then(fn=_append_log_history, inputs=[ev_log, ev_history_state],
                  outputs=[ev_history_state, ev_history_text])

    xp_cmp_event.then(fn=_append_log_history, inputs=[xp_cmp_log, xp_cmp_history_state],
                      outputs=[xp_cmp_history_state, xp_cmp_history_text])

    # Refresh status/threshold/model-list whenever the project changes
    proj_refresh_models_btn.click(fn=handle_refresh_rf_model_choices, inputs=[proj_dir],
                                  outputs=[proj_rf_model_override])
    for _trigger in (proj_dir, proj_rf_model_override, proj_labels):
        _trigger.change(fn=handle_refresh_status, inputs=[proj_dir, proj_rf_model_override, proj_labels],
                        outputs=[status_line])
    for _trigger in (proj_dir, proj_rf_model_override):
        _trigger.change(fn=handle_refresh_threshold, inputs=[proj_dir, proj_rf_model_override],
                        outputs=[dt_threshold])

    # Stop buttons
    tr_stop.click(fn=None, cancels=[tr_event])
    ev_stop.click(fn=None, cancels=[ev_event])
    dt_stop.click(fn=None, cancels=[dt_event])
    xp_train_stop.click(fn=None, cancels=[xp_train_event])
    xp_cmp_stop.click(fn=None, cancels=[xp_cmp_event])
    xp_det_stop.click(fn=None, cancels=[xp_det_event])

    # ------------------------------------------------------------------ #
    # G1.4 — autosave settings on change (replaces "Save as defaults")
    # ------------------------------------------------------------------ #
    _AUTOSAVE_COMPONENTS = [
        proj_dir, proj_imagery, proj_labels, proj_hydro, proj_rf_model_override,
        tr_augment, tr_flood_samples, tr_hydro_negatives, tr_neg_ratio, tr_run_cv,
        ev_radius, ev_n_splits, ev_per_class,
        audit_low, audit_high, audit_include_auto_neg,
        dt_threshold, dt_min_area, dt_seed_threshold, dt_no_smooth, dt_output,
        mp_basemap, mp_show_det, mp_show_labels, mp_show_hydro,
        mp_show_audit, mp_filter_threshold, mp_kml_dropdown,
        dg_lon, dg_lat,
        xp_epochs, xp_lr, xp_test_frac,
        xp_det_method, xp_det_threshold, xp_det_output,
    ]
    assert len(_AUTOSAVE_COMPONENTS) == len(_SETTINGS_KEYS), (
        f"_AUTOSAVE_COMPONENTS ({len(_AUTOSAVE_COMPONENTS)}) must line up 1:1 with "
        f"_SETTINGS_KEYS ({len(_SETTINGS_KEYS)})"
    )
    # Text/number fields save on blur or Enter, sliders on release, and the rest
    # on change. Using .change everywhere fired one save per keystroke, which
    # queued up behind long-running jobs and made the UI look stuck.
    def _autosave_triggers(component):
        if isinstance(component, (gr.Textbox, gr.Number)):
            return [component.blur, component.submit]
        if isinstance(component, gr.Slider):
            return [component.release]
        return [component.change]

    gr.on(
        triggers=[t for c in _AUTOSAVE_COMPONENTS for t in _autosave_triggers(c)],
        fn=handle_autosave,
        inputs=_AUTOSAVE_COMPONENTS,
        outputs=[save_status],
        queue=False,
        show_progress="hidden",
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
