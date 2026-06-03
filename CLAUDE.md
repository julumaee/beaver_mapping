# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CastorDetector (Finland MML Edition)** — a CLI tool to detect beaver activity in Finnish National Land Survey (MML) aerial imagery.

- Input: MML 5x5km JPEG2000 (`.jp2`) files in EPSG:3067 (ETRS-TM35FIN)
- Masking: MML Hydrography (Virtavesi) vector data (GeoPackage/Shapefile)
- Ground truth: KML/KMZ placemarks exported from Google Earth — generic labels (`dead_forest`, `flood`) placed anywhere in imagery, plus hard negatives (`negative`) near streams
- Output: KML files (EPSG:4326) for verification in Google Earth

## Development Commands

```bash
pip install -r requirements.txt   # install dependencies
python -m pytest tests/            # run tests
python src/cli.py --help           # run the CLI
```

## Directory Structure

```
data/
  imagery/        # .jp2 input files (EPSG:3067, CIR band order: NIR, Red, Green)
  hydrography/    # MML Virtavesi vector data (.gpkg or .shp directory)
  labels/         # KML/KMZ ground truth placemarks
  chips/          # extracted training chips + manifest.csv (written by train --chip-dir)
  models/         # trained model weights (.pkl for RF, .pth for CNN, norm_stats.json)
  output/         # generated KML detection files
src/
  cli.py              # argparse CLI: train, detect, evaluate-rf, evaluate, cnn-train
  ingestion.py        # .jp2 metadata validation, TILE_SIZE constant
  masking.py          # hydrography vector load, 100m stream buffer mask
  spectral.py         # NDWI/NDVI computation, 99-element feature vector extraction
  training_data.py    # KML label parsing, chip extraction, negative sampling
  polygonizer.py      # merge detections into ROI polygons, minimum-area filter
  export.py           # reproject to EPSG:4326, write KML with confidence colours
  diagnose_point.py   # visualise chip + RF prediction at a single lon/lat
  models/
    random_forest.py  # RF train/predict, spatial leave-one-cluster-out CV
    evaluate.py       # RF and CNN evaluation: accuracy, precision, recall, F1, per-class
    cnn_train.py      # Prithvi-EO-1.0-100M fine-tuning loop
    cnn_handler.py    # CNN inference wrapper
    cnn_dataset.py    # PyTorch Dataset for chip loading
tests/
  test_classifier.py
  test_export.py
  test_polygonizer.py
  test_spectral.py
  test_training_data.py
```

## Key Technical Constraints

**Coordinate systems:**
- All MML raster/vector data is in EPSG:3067 (ETRS-TM35FIN)
- Use `pyproj.Transformer` (not the deprecated `pyproj.transform`) for CRS conversions
- KML output must be in EPSG:4326 (WGS84)

**MML band order (0-indexed in rasterio/numpy):**
- Vääräväri (CIR): Band 0 = NIR, Band 1 = Red, Band 2 = Green
- Väri (RGB): Band 0 = Red, Band 1 = Green, Band 2 = Blue

**Memory management:**
- Never load a full 5km tile into memory
- Use `rasterio.windows.Window` for windowed reads; chip size is 512×512px (TILE_SIZE in `ingestion.py`)
- Only process tiles that intersect the hydrography stream buffer mask

## Architecture

The pipeline has five implemented stages:

1. **Ingestion** (`ingestion.py`) — read `.jp2` metadata, validate CRS, define TILE_SIZE
2. **Masking** (`masking.py`) — load Virtavesi vectors, build 100m stream buffer, restrict tile processing
3. **Training data** (`training_data.py`) — parse KML/KMZ labels → reproject to EPSG:3067 → extract 512×512 chips; auto-sample equal-count negatives from stream corridor (200m exclusion zone around positives, 100m minimum spacing)
4. **Detection** (`spectral.py`, `models/`) — compute 70-element feature vectors (per-band stats, NDWI/NDVI, GLCM texture, connected wet-region stats at three thresholds); run RF or CNN classifier; merge detections into ROI polygons (`polygonizer.py`)
5. **Export** (`export.py`) — reproject to EPSG:4326, write KML with confidence-coded colours (Red=RF, Blue=CNN, Purple=both)

**Feature vector (99 elements, multi-scale):**
- 32px and 64px center crops (35 features each): per-band mean/std/p25/p75, NDVI stats, NDWI stats, NDWI gradient std, GLCM texture, connected wet-region stats at 3 thresholds
- 512px full chip (23 features): same minus connected-component analysis (too slow, near-zero importance)
- 6 cross-scale contrast features (32px−512px, 64px−512px): NDWI mean, wet fraction, max blob area — beaver floods are wet at all scales; ditches are wet only at fine scale

**Label classes:**
- `dead_forest` → class 1 — standing dead trees killed by beaver flooding (generic, place anywhere)
- `flood` → class 1 — open water impoundment of any kind (generic, place anywhere)
- `wet_forest` → class 1 — saturated/flooded forest (legacy, kept for compatibility)
- `beaver_flood` → class 1 — confirmed beaver open water (legacy, kept for compatibility)
- `negative` → class 0 — hard negative; place near false-positive areas (streams, bogs, ditches)
- `dam`, `lodge` → excluded (point-scale features, not suitable for area classification)

**Training workflow (generic-label approach):**
- Label `dead_forest` and `flood` anywhere in the imagery — no confirmed beaver territory needed
- Train **without** `--hydro` so auto-sampled negatives come from the full imagery extent
- Keep some hand-labeled `negative` points near streams as hard negatives
- Run `detect` **with** `--hydro` — stream filter applied at inference, not training time

**Evaluation:**
- Use spatial leave-one-cluster-out CV (`evaluate-rf`), not random splits — label points within 500m form one fold
- Per-class precision/recall available via `--per-class` flag

**CNN:**
- Prithvi-EO-1.0-100M pretrained weights (~454MB), downloaded from HuggingFace or placed manually in `data/models/`
- CPU inference is slow (~40 min/tile without hydro mask, ~15 min with mask)

Core dependencies: `rasterio`, `fiona`, `pyproj`, `geopandas`, `shapely`, `scikit-learn`, `scikit-image`, `torch`
