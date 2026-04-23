# CastorDetector — Beaver Activity Detection in MML Aerial Imagery

CLI tool for detecting beaver activity in Finnish National Land Survey (MML) aerial imagery using spectral analysis and deep learning.

## How It Works

The pipeline slides a 512×512px window over MML `.jp2` tiles, classifies each window as beaver activity or background, and exports detections as KML polygons for verification in Google Earth.

Two detection models are available:

- **Random Forest (RF)** — uses NDWI/NDVI spectral indices. Fast, works well on CPU, good baseline.
- **CNN (Prithvi-EO)** — frozen Prithvi-EO-1.0-100M ViT-Base encoder with a trainable classification head. Slower on CPU but captures spatial context.

Both models can be run independently or together, with agreement detections highlighted in a third colour.

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins the CPU-only PyTorch wheel. If you have an NVIDIA GPU, replace the `--extra-index-url` line with the appropriate CUDA wheel URL from pytorch.org.

## Random Forest

### Train

```bash
python src/cli.py train \
  --imagery data/imagery/ \
  --labels data/labels/ \
  --model data/models/model.pkl \
  --hydro data/hydrography/        # optional but recommended
```

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
