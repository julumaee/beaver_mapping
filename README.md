# CastorDetector — Beaver Activity Detection in MML Aerial Imagery

CLI tool for detecting beaver activity in Finnish National Land Survey (MML) aerial imagery using spectral analysis.

## How It Works

The pipeline slides a 512×512px window over MML `.jp2` tiles, classifies each window as beaver activity or background, and exports detections as KML polygons for verification in Google Earth.

The primary detection model is the **Random Forest (RF)**, trained on manually labelled point observations (`wet_forest`, `beaver_flood`) placed on top of beaver-influenced landscape features in Google Earth. A CNN (Prithvi-EO) path also exists for experimental comparison.

**RF feature vector (70 elements, computed at two spatial scales):**
- Per-band mean, std, p25, p75 (NIR, Red, Green)
- NDVI and NDWI statistics + high-value pixel fractions
- NDWI spatial gradient std (water-edge sharpness)
- GLCM texture on NIR (contrast, homogeneity, energy, correlation)
- Connected wet-region stats at three NDWI thresholds: wet fraction, component count, largest blob area fraction, blob shape index

## Data

| Path | Contents |
|---|---|
| `data/imagery/` | MML JPEG2000 tiles (`.jp2`), EPSG:3067, Vääräväri (CIR) band order: NIR, Red, Green |
| `data/hydrography/` | MML Virtavesi vector files (`.gpkg` / `.shp`), directory accepted |
| `data/labels/` | Google Earth ground truth (`.kml` / `.kmz`) with point placemarks named `wet_forest`, `beaver_flood`, or `dam` |
| `data/output/` | Generated KML detection files |
| `data/models/` | Trained model weights |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins the CPU-only PyTorch wheel. If you have an NVIDIA GPU, replace the `--extra-index-url` line with the appropriate CUDA wheel URL from pytorch.org.
>
> All commands below assume the venv is active. If not, prefix with `.venv/bin/python`.

## GUI

A Gradio-based control panel wraps every CLI command below in a browser UI — recommended for interactive use. The CLI (see sections below) is better suited to scripting and automation.

```bash
python src/app.py
```

Opens `http://localhost:7860`. A single **Project** panel at the top (imagery/labels/hydrography/project directory, collapsible once set) drives everything below it — models, chips, and detection outputs are all derived from the project directory (`<project>/models/model.pkl`, `<project>/chips/`, `<project>/output/`), so there is one place to set paths instead of one per tab. A header status line shows `Model: trained <date> · <n> chips · CV PR-AUC x · recommended threshold y` and warns if your labels have changed since the model was last trained. Settings are saved automatically as you edit them (to `data/settings.json`, or the file named by the `CASTOR_SETTINGS` env var) — there is no separate "save" step, and old-format settings files are migrated automatically on first load.

The workflow is ordered left to right; each tab streams log output live and has a **Stop** button to cancel an in-flight run:

| Tab | Purpose |
|---|---|
| **1. Data check** | Validate the project before running anything: paths exist, tile/label counts by type, unrecognised label types, labels outside imagery coverage, hydrography presence, plus a chip gallery preview |
| **2. Train** | Extract chips (always kept under `<project>/chips/`) and train the Random Forest; optionally chains straight into spatial cross-validation when finished |
| **3. Evaluate** | Pooled out-of-fold spatial cross-validation — confusion matrix, per-feature-type breakdown, and a **label audit** table of suspicious label points (positives the model scores low, hand-labelled negatives it scores high) |
| **4. Detect** | Run the RF model on imagery and export a timestamped KML (`detections_rf_<YYYYmmdd-HHMM>.kml`); threshold defaults to the model's recommended value when available |
| **5. Map & Review** | Interactive satellite/OSM map — detections, training labels, hydrography, and toggleable label-audit rings; a "past runs" dropdown lists `<project>/output/*.kml`, newest first; click any point (including a marker) to run **Diagnose Point**, shown directly below the map |
| **6. Experimental (CNN)** | Prithvi-EO CNN training, RF-vs-CNN comparison, and CNN/"both" detection — collapsed by default and marked slow (CPU inference ~15–40 min/tile) |

An **Advanced** accordion in the Project panel lets you override the RF model file (e.g. to compare a saved checkpoint) without touching the default `<project>/models/model.pkl`; each tab has its own **Advanced** accordion for the less-common flags (augmentation, flood samples, negative ratio, CV folds/cluster radius, min detection area, seed threshold, smoothing, CNN epochs/LR), each with inline help text.

## Random Forest

### Label format

Place point placemarks in Google Earth and name them:
- `wet_forest` — flooded/saturated forest with dead standing trees
- `beaver_flood` — open water impoundment behind a beaver dam
- `negative` — stream-adjacent area with no beaver activity (optional; the pipeline also auto-samples negatives)

Placemarks named `dam` or `lodge` are excluded from training by default.

### Train

```bash
python src/cli.py train \
  --imagery data/imagery/ \
  --labels data/labels/ \
  --model data/models/model.pkl \
  --hydro data/hydrography/ \       # optional but recommended
  --chip-dir data/chips/            # optional: keep chips for evaluate-rf
```

The training pipeline:
1. Parses `wet_forest` and `beaver_flood` labels from all KML/KMZ files in `--labels`
2. Auto-samples an equal number of negatives from the stream corridor, excluding any point within 200 m of a positive and enforcing 100 m minimum spacing between negatives
3. Extracts 70-element feature vectors per chip and trains a balanced Random Forest

### Detect

```bash
python src/cli.py detect \
  --imagery data/imagery/ \
  --output data/output/detections_rf.kml \
  --method rf \
  --rf-model data/models/model.pkl \
  --hydro data/hydrography/ \
  --threshold 0.5                  # optional, default 0.5
```

### Evaluate RF (spatial cross-validation)

Standard random splits are unreliable for spatial data. Use spatial leave-one-cluster-out CV instead — label points within 500 m of each other form one fold:

```bash
# Overall metrics (accuracy, precision, recall, F1 — mean/min/max across folds)
python src/cli.py evaluate-rf \
  --manifest data/chips/manifest.csv \
  --rf-model data/models/model.pkl

# Per-class breakdown (wet_forest vs beaver_flood separately)
python src/cli.py evaluate-rf \
  --manifest data/chips/manifest.csv \
  --rf-model data/models/model.pkl \
  --per-class

# Adjust cluster radius if your territories are closer/further apart
python src/cli.py evaluate-rf \
  --manifest data/chips/manifest.csv \
  --rf-model data/models/model.pkl \
  --cluster-radius 300
```

> **Note:** The manifest CSV is written into the chip directory. By default `train` uses a temp directory that is deleted after training. Pass `--chip-dir data/chips/` to keep chips and the manifest on disk for use with `evaluate-rf`.

## CNN (Prithvi-EO-1.0-100M)

The CNN downloads ~454MB of pretrained weights from HuggingFace on first run. Place them manually in `data/models/` to avoid download issues:

```bash
# Recommended: authenticate first for reliable download speeds
huggingface-cli login
huggingface-cli download ibm-nasa-geospatial/prithvi-eo-1.0-100M \
  Prithvi_EO_V1_100M.pt prithvi_mae.py \
  --local-dir data/models/
```

### Train

```bash
python src/cli.py cnn-train \
  --imagery data/imagery/ \
  --labels data/labels/ \
  --model data/models/beaver_cnn_v1.pth \
  --norm-stats data/models/norm_stats.json \
  --hydro data/hydrography/ \
  --epochs 30 \
  --lr 0.001
```

Norm stats are computed from your training chips and saved automatically. Training prints `train_loss` and `val_acc` per epoch; the best checkpoint is saved.

### Detect

```bash
python src/cli.py detect \
  --imagery data/imagery/ \
  --output data/output/detections_cnn.kml \
  --method cnn \
  --cnn-model data/models/beaver_cnn_v1.pth \
  --norm-stats data/models/norm_stats.json \
  --hydro data/hydrography/
```

> **Performance:** CNN inference runs at batch_size=1 on CPU (~40 min per 12km² tile without hydro mask, ~15 min with mask). Use `--hydro` to restrict processing to stream corridors.

## Run Both Models and Compare

```bash
python src/cli.py detect \
  --imagery data/imagery/ \
  --output data/output/detections_both.kml \
  --method both \
  --rf-model data/models/model.pkl \
  --cnn-model data/models/beaver_cnn_v1.pth \
  --norm-stats data/models/norm_stats.json \
  --hydro data/hydrography/
```

KML colour coding:
- **Red** — RF only
- **Blue** — CNN only
- **Purple** — both models agree (high confidence)

## Evaluate RF vs CNN

Runs both models on a held-out split of the training manifest and prints accuracy, precision, recall, and F1.

```bash
python src/cli.py evaluate \
  --manifest data/chips/manifest.csv \
  --rf-model data/models/model.pkl \
  --cnn-model data/models/beaver_cnn_v1.pth \
  --norm-stats data/models/norm_stats.json
```

## Diagnose a Single Point

Extracts and visualises the chip at a known location and shows the RF classifier result.

```bash
python src/diagnose_point.py \
  --lon 25.123 \
  --lat 62.456 \
  --imagery data/imagery/ \
  --model data/models/model.pkl \
  --out chip_debug.png
```

## Output

All detection KML files open directly in Google Earth. Each polygon includes:
- Confidence score
- Area in m²
- Model source (when using `--method both`)
