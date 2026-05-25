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
