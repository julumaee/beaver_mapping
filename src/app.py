"""Gradio web UI for CastorDetector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr


with gr.Blocks(title="CastorDetector") as demo:
    gr.Markdown("# CastorDetector\nBeaver activity detection in MML aerial imagery.")
    with gr.Tabs():
        with gr.Tab("Train RF"):
            gr.Markdown("## Train Random Forest")
        with gr.Tab("Train CNN"):
            gr.Markdown("## Train CNN (Prithvi-EO)")
        with gr.Tab("Detect & Export"):
            gr.Markdown("## Detect & Export")


demo.queue()

if __name__ == "__main__":
    demo.launch()
