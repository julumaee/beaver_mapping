"""Gradio web UI for CastorDetector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr


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
