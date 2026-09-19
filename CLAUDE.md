# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CastorDetector (Finland MML Edition)** — a CLI + Gradio GUI tool to detect beaver activity in Finnish National Land Survey (MML) aerial imagery.

- Input: MML 5x5km JPEG2000 (`.jp2`) files in EPSG:3067 (ETRS-TM35FIN)
- Masking: MML Hydrography (Virtavesi) vector data (GeoPackage/Shapefile)
- Ground truth: KML/KMZ placemarks exported from Google Earth, typed by folder — `dead_forest`, `beaver_flood`/`flood` anywhere in imagery, plus `hard_negatives` near streams
- Roadmap: `beaver-detector-roadmap-v4-accuracy-gui.md` (RF accuracy + GUI plan; tick items off as they land)
- Output: KML files (EPSG:4326) for verification in Google Earth

## Development Commands

```bash
.venv/bin/pip install -r requirements.txt   # install dependencies
.venv/bin/python -m pytest tests/ -q        # run tests (pytest.ini disables stray ROS pytest plugins)
.venv/bin/python src/cli.py --help          # CLI: train, tune, detect, evaluate-rf, evaluate, cnn-train
.venv/bin/python src/app.py                 # Gradio GUI at http://localhost:7860
```

`python` is not on PATH on the dev machine — use `.venv/bin/python`. The shell's global `PYTHONPATH` includes ROS Jazzy packages; that's why `pytest.ini` disables the `launch_testing`/`launch_ros` plugins.

## Directory Structure

```
data/
  imagery/        # .jp2 input files (EPSG:3067, CIR band order: NIR, Red, Green)
  hydrography/    # MML Virtavesi vector data (.gpkg or .shp directory)
  labels/         # KML/KMZ ground truth placemarks
  chips/          # training chips + manifest.csv, features cache, oof.csv (CV predictions), tuning.json
  models/         # model.pkl + model.json metadata sidecar (RF), .pth + norm_stats.json (CNN)
  output/         # generated KML detection files (GUI: detections_<method>_<timestamp>.kml)
  settings.json   # GUI settings, auto-saved (override path with CASTOR_SETTINGS env var)
src/
  app.py              # Gradio GUI: Project panel + Data check → Train → Evaluate → Detect → Map & Review → Experimental (CNN)
  cli.py              # argparse CLI: train, tune, detect, evaluate-rf, evaluate, cnn-train (GUI delegates to cmd_*)
  ingestion.py        # .jp2 metadata validation, TILE_SIZE constant
  masking.py          # hydrography vector load, 100m stream buffer mask
  spectral.py         # NDWI/NDVI computation, feature vector (FEATURE_VECTOR_LENGTH = 109)
  training_data.py    # KML label parsing (folder = type), polygon sampling, chip extraction, negative sampling
  polygonizer.py      # patch-level RF detection, smoothing + hysteresis, min-area filter, ROI polygons
  export.py           # reproject to EPSG:4326, write KML with confidence colours
  diagnose_point.py   # visualise chip + RF prediction at a single lon/lat
  label_audit.py      # flag suspicious labels from out-of-fold CV predictions
  models/
    random_forest.py  # make_classifier (single config source), train, model.json sidecar, feature-length check
    evaluate.py       # pooled out-of-fold spatial CV, per-type breakdown, tuning, RF vs CNN
    cnn_train.py      # Prithvi-EO-1.0-100M fine-tuning loop
    cnn_handler.py    # CNN inference wrapper
    cnn_dataset.py    # PyTorch Dataset for chip loading
tests/                # pytest suite (test_<module>.py per module, plus test_app_*.py for GUI helpers)
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
3. **Training data** (`training_data.py`) — parse KML/KMZ labels (type = folder name) → reproject to EPSG:3067 → extract 512×512 chips (positives and hand-labelled negatives get ±24 px offset augmentation; polygon labels yield up to 10 interior samples); auto-sample `auto_negative` points = `neg_ratio` × positive chips, half stream corridor / half full extent when `--hydro` is set (200 m from labels, 100 m spacing)
4. **Detection** (`spectral.py`, `models/`, `polygonizer.py`) — classify every 64 px patch (512 px context) with the RF → 3×3 NaN-aware smoothing → hysteresis (seed = threshold + 0.15) → min area 2048 m² → ROI polygons. Threshold defaults to `model.json`'s `recommended_threshold`
5. **Export** (`export.py`) — reproject to EPSG:4326, write KML with confidence-coded colours (Red=RF, Blue=CNN, Purple=both)

**Feature vector (109 elements, `spectral.FEATURE_VECTOR_LENGTH`):**
- 32px and 64px center crops (40 each): per-band mean/std/p25/p75, NDVI stats, NDWI stats, NDWI gradient std, GLCM on NIR, connected wet-region stats at 3 thresholds (35 core) + GLCM on NDVI and low-NDVI-textured fraction (5, dead-forest cues)
- 512px full chip (23): core minus connected-component analysis
- 6 cross-scale contrast features (32px−512px, 64px−512px)
- Never hardcode the length elsewhere; changing it invalidates trained models (`check_feature_length` makes `detect` fail with a "retrain" message). New features are adopted only if pooled spatial CV improves.
- RF features are rotation/flip invariant — rotation augmentation does not help the RF.

**Label classes** (`training_data.FEATURE_TO_LABEL`; unknown names are excluded with a warning, never defaulted to positive):
- class 1: `dead_forest`, `flood`, `flooded_areas`, `beaver_flood`, `wet_forest`
- class 0: `hard_negatives`, `hard_negative`, `negative`, `negatives` (hand-labelled), `auto_negative` (auto-sampled)
- excluded: `dam`, `lodge`, `other`

**Evaluation:**
- `evaluate-rf` = spatial clusters (500 m) → StratifiedGroupKFold (5) → pooled out-of-fold probabilities → ROC-AUC, PR-AUC, recommended (max-F1) threshold, chip- and point-level metrics, per-type recall/specificity; writes `oof.csv`. Never average per-fold metrics (single-class folds score 0).
- Point-level metrics (augmented chips merged) are the honest numbers; chip-level is inflated by augmentation.
- `make_classifier` is the single classifier config for train, CV and tune (default RF 300 trees, min_samples_leaf=3, balanced).

**CNN:**
- Prithvi-EO-1.0-100M pretrained weights (~454MB), downloaded from HuggingFace or placed manually in `data/models/`
- CPU inference is slow (~40 min/tile without hydro mask, ~15 min with mask)

Core dependencies: `rasterio`, `fiona`, `pyproj`, `geopandas`, `shapely`, `scikit-learn`, `scikit-image`, `torch`
