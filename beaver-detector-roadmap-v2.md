# Implementation Roadmap: CastorDetector (Finland MML Edition) v2

This document outlines the step-by-step development of a CLI application to detect beaver activity using Finnish National Land Survey (MML) imagery.

## Project Context
- **Input Imagery:** MML 5x5km JPEG2000 (.jp2) files.
- **Native CRS:** EPSG:3067 (ETRS-TM35FIN).
- **Masking Data:** MML Hydrography (Virtavesi) vector data (GeoPackage or Shapefile).
- **Ground Truth:** ~100 manually identified territories in Google Earth (KML/KMZ).
- **Output Goal:** KML files (EPSG:4326) for verification in Google Earth.

---

## Feature 1: Project Scaffolding & Geospatial Env
**Goal:** Setup environment to handle Finnish CRS and Vector data.

- [ ] **Task 1.1:** Initialize Git and Python environment.
    - *Commit message:* "feat: initial project structure"
- [ ] **Task 1.2:** Create `requirements.txt` including: `rasterio`, `fiona`, `pyproj`, `geopandas`, `shapely`, `opencv-python`, and `scikit-learn`.
    - *Commit message:* "feat: add geospatial and ML dependencies"
- [ ] **Task 1.3:** Setup folder structure: `/data/imagery`, `/data/hydrography`, `/data/labels`, `/data/output`, `/src`.
    - *Commit message:* "chore: setup directory structure"

## Feature 2: MML Data Ingestion & Masking
**Goal:** Efficiently handle MML files and focus only on areas near water.

- [ ] **Task 2.1:** Implement a metadata reader that identifies the CRS (EPSG:3067) and bounds of the .jp2 files using `rasterio`.
    - *Commit message:* "feat: add EPSG:3067 coordinate support"
- [ ] **Task 2.2:** Build a "Stream Buffer" utility. Load MML hydrography vectors and create a spatial mask (e.g., 100m buffer) to limit the search area.
    - *Commit message:* "feat: implement stream-masking logic to reduce search space"
- [ ] **Task 2.3:** Implement windowed reading for .jp2 files so the app only processes 512x512px tiles that intersect with the "Stream Buffer."
    - *Commit message:* "feat: optimize tiling to follow hydrography mask"

## Feature 3: Training Data Pipeline (Leveraging Manual Labels)
**Goal:** Convert your 100 Google Earth points into "Training Chips."

- [ ] **Task 3.1:** Write a script to parse your KML labels and convert Google Earth coordinates (WGS84) back to MML coordinates (ETRS-TM35FIN).
    - *Commit message:* "feat: kml label coordinate transformation"
- [ ] **Task 3.2:** "Chip Extractor": Automatically crop 512x512px image "chips" around your 100 known beaver territories from the .jp2 files.
    - *Commit message:* "feat: implement training chip extraction"
- [ ] **Task 3.3:** Generate "Negative Samples": Extract 100 chips from stream segments where you know there is *no* beaver activity.
    - *Commit message:* "feat: generate negative samples for training"

## Feature 4: Detection Logic
**Goal:** Analyze pixels to find beaver signatures (Dams, Ponds, Ghost Forests).

- [ ] **Task 4.1:** Implement spectral index calculation. For MML Vääräväri (CIR): calculate NDWI (Water) and NDVI (Vegetation Health) to isolate flooded areas and dead wood.
    - *Commit message:* "feat: add NIR-based spectral index analysis"
- [ ] **Task 4.2:** Implement a classifier (e.g., Random Forest or basic CNN) that uses the chips from Feature 3 to identify beaver-modified landscapes.
    - *Commit message:* "feat: implement classifier for beaver signatures"
- [ ] **Task 4.3:** Create a "Polygonizer": Merge neighboring detected pixels into "Regions of Interest" (ROI) and filter by minimum size.
    - *Commit message:* "feat: implement detection clustering and polygonization"

## Feature 5: Google Earth Export & CLI
**Goal:** Finalize the tool for production use.

- [ ] **Task 5.1:** Implement KML Export functionality that transforms detected polygons back to EPSG:4326 for Google Earth.
    - *Commit message:* "feat: export detections to WGS84 KML"
- [ ] **Task 5.2:** Add visual metadata to KML: Color-code polygons by "Detection Confidence" and include area size in the balloon description.
    - *Commit message:* "feat: add metadata and styling to KML output"
- [ ] **Task 5.3:** Build a batch-processing CLI that scans a directory of .jp2 files and outputs a single combined KML.
    - *Commit message:* "feat: finalize batch processing CLI"

---

## Technical Instructions for AI Assistant
- **Coordinate Transformation:** Use `pyproj.Transformer` for high-precision conversion between ETRS-TM35FIN (MML) and WGS84 (Google Earth).
- **MML Band Order (Vääräväri):** Band 1 = NIR, Band 2 = Red, Band 3 = Green.
- **MML Band Order (Väri):** Band 1 = Red, Band 2 = Green, Band 3 = Blue.
- **Memory Optimization:** Use `rasterio.windows.Window` to process sub-tiles; do not load the whole 5km tile into memory.
