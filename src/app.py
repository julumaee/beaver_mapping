"""Gradio web UI for CastorDetector."""
import csv
import io
import sys
import tempfile
from pathlib import Path

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


def _capture(fn, *args, **kwargs) -> str:
    """Run fn synchronously, capture its stdout, return as string."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
        return buf.getvalue() or "Done."
    except Exception as exc:
        return (buf.getvalue() or "") + f"\nERROR: {exc}"
    finally:
        sys.stdout = old


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
) -> str:
    return _capture(
        _do_train_rf,
        imagery_dir.strip(), labels_dir.strip(), model_path.strip(),
        hydro_dir.strip(), chip_dir.strip(), int(augment),
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
) -> str:
    return _capture(
        _do_train_cnn,
        imagery_dir.strip(), labels_dir.strip(), model_path.strip(),
        norm_stats_path.strip(), hydro_dir.strip(), int(epochs), float(lr),
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
            rf_log = gr.Textbox(label="Log", lines=15, interactive=False, show_copy_button=True)
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
            cnn_log = gr.Textbox(label="Log", lines=15, interactive=False, show_copy_button=True)
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
            det_btn = gr.Button("Detect & Export KML", variant="primary")
            det_log = gr.Textbox(label="Log", lines=15, interactive=False, show_copy_button=True)


demo.queue()

if __name__ == "__main__":
    demo.launch()
