# CastorDetector — Beaver Activity Detection in MML Aerial Imagery

GUI and CLI tool for detecting beaver activity in Finnish National Land Survey (MML) aerial imagery using spectral analysis.

## How It Works

The detector scans MML `.jp2` tiles in 64×64 px patches (32×32 m), each classified using its surrounding 512×512 px context. It merges positive patches into polygons and exports them as KML for verification in Google Earth.

The primary detection model is a **Random Forest (RF)**. It is trained on point labels placed in Google Earth (`dead_forest`, `beaver_flood`/`flood`, `hard_negatives`) plus automatically sampled negatives. A CNN (Prithvi-EO) path also exists for experimental comparison.

**RF feature vector (109 elements, three spatial scales).** Computed on 32 px and 64 px centre crops and on the 512 px context:
- Per-band mean, std, p25, p75 (NIR, Red, Green)
- NDVI and NDWI statistics + high-value pixel fractions
- NDWI spatial gradient std (water-edge sharpness)
- GLCM texture on NIR, and on NDVI at 32/64 px
- Low-NDVI-in-textured-area fraction (dead standing trees) at 32/64 px
- Connected wet-region stats at three NDWI thresholds (32/64 px only)
- 6 cross-scale contrast features (fine scale − landscape scale)

See `src/spectral.py` for the exact layout. Changing the feature vector makes previously trained models incompatible; `detect` refuses them with a "retrain" message.

## Data

| Path | Contents |
|---|---|
| `data/imagery/` | MML JPEG2000 tiles (`.jp2`), EPSG:3067, Vääräväri (CIR) band order: NIR, Red, Green |
| `data/hydrography/` | MML Virtavesi vector files (`.gpkg` / `.shp`), directory accepted |
| `data/labels/` | Google Earth ground truth (`.kml` / `.kmz`); the label type comes from the enclosing **folder** name (see [Label format](#label-format)) |
| `data/chips/` | Extracted training chips, `manifest.csv`, feature cache and `oof.csv` (cross-validation predictions) |
| `data/output/` | Generated KML detection files |
| `data/models/` | Trained model weights plus a `model.json` metadata file next to each RF model |

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

Put placemarks in Google Earth **folders**. The folder name is the label type; it is lowercased and spaces become underscores, so "Dead Forest" becomes `dead_forest`. Placemarks outside any folder use their own name.

| Type (folder name) | Class | Meaning |
|---|---|---|
| `dead_forest` | positive | Standing dead trees killed by beaver flooding |
| `beaver_flood`, `flood`, `flooded_areas` | positive | Open-water impoundment |
| `wet_forest` | positive | Saturated forest (legacy) |
| `hard_negatives`, `negative` | negative | Look-alikes near streams: bogs, ditches, lakes, clearcuts |
| `dam`, `lodge`, `other` | excluded | Not used for training |

Any other name is **excluded with a warning**; it is not silently treated as positive. Point and polygon placemarks both work, and a polygon yields up to 10 sample points spread inside it.

### Train

```bash
python src/cli.py train \
  --imagery data/imagery/ \
  --labels data/labels/ \
  --model data/models/model.pkl \
  --hydro data/hydrography/ \
  --chip-dir data/chips/
```

The training pipeline:
1. Parses labels from all KML/KMZ files in `--labels`. Positives and hand-labelled negatives get 6 extra chips each at small random offsets (`--augment-positives`).
2. Auto-samples `auto_negative` points: one per positive chip by default (`--neg-ratio`). They are drawn half from the stream corridor and half from the whole imagery extent, at least 200 m from any label and 100 m apart.
3. Trains the Random Forest (300 trees, `min_samples_leaf=3`, balanced class weights).
4. Runs spatial cross-validation (skip with `--no-cv`). It writes `model.json` next to the model with the recommended threshold and CV metrics.

To try other classifier settings, run `tune` (below) and pass its output with `--classifier-config data/chips/tuning.json`.

### Detect

```bash
python src/cli.py detect \
  --imagery data/imagery/ \
  --output data/output/detections_rf.kml \
  --method rf \
  --rf-model data/models/model.pkl \
  --hydro data/hydrography/
```

- **Threshold:** defaults to the recommended threshold stored in `model.json` (else 0.5); override with `--threshold`.
- **Clean-up:** the patch probability map is smoothed. A region is kept only if it contains at least one patch above a stricter seed threshold (`--seed-threshold`, default threshold + 0.15) and covers at least 2048 m² (`--min-area`). Disable the smoothing with `--no-smooth`.

### Evaluate RF (spatial cross-validation)

Random train/test splits are unreliable for spatial data. `evaluate-rf` instead groups label points within 500 m into spatial clusters and runs 5-fold grouped cross-validation. All metrics come from the pooled out-of-fold predictions.

```bash
python src/cli.py evaluate-rf --manifest data/chips/manifest.csv --per-class
```

It reports:
- ROC-AUC and PR-AUC;
- the recommended threshold (max F1) and a high-recall threshold;
- precision/recall/F1 per chip and per label point;
- recall or specificity for each label type.

Out-of-fold predictions are written to `data/chips/oof.csv`, which the GUI's label audit uses. Features are cached next to the manifest; pass `--no-cache` to recompute them.

### Tune the classifier

```bash
python src/cli.py tune --manifest data/chips/manifest.csv
```

Runs the same spatial cross-validation for several RandomForest, ExtraTrees and HistGradientBoosting settings. It prints a ranked table and writes `tuning.json` for `train --classifier-config`.

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

Scores both saved models and prints accuracy, precision, recall and F1.

```bash
python src/cli.py evaluate \
  --manifest data/chips/manifest.csv \
  --rf-model data/models/model.pkl \
  --cnn-model data/models/beaver_cnn_v1.pth \
  --norm-stats data/models/norm_stats.json \
  --test-manifest path/to/other_tiles/manifest.csv   # recommended
```

Without `--test-manifest`, it holds out spatial clusters of the training manifest. The saved models were trained on those same chips, so the numbers are **in-sample and optimistic**. For a real comparison, build a manifest from tiles that were not used for training.

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
