# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CastorDetector (Finland MML Edition)** — a CLI tool to detect beaver activity in Finnish National Land Survey (MML) aerial imagery.

- Input: MML 5x5km JPEG2000 (`.jp2`) files in EPSG:3067 (ETRS-TM35FIN)
- Masking: MML Hydrography (Virtavesi) vector data (GeoPackage/Shapefile)
- Ground truth: ~100 manually identified territories exported from Google Earth (KML/KMZ)
- Output: KML files (EPSG:4326) for verification in Google Earth

## Development Commands

Once the project is scaffolded, expected commands will be:

```bash
pip install -r requirements.txt   # install dependencies
python -m pytest tests/            # run tests
python src/cli.py --help           # run the CLI
```

## Planned Directory Structure

```
data/
  imagery/        # .jp2 input files
  hydrography/    # MML vector data
  labels/         # KML/KMZ ground truth
  output/         # generated KML detections
src/              # Python source code
tests/
requirements.txt
```

## Key Technical Constraints

**Coordinate systems:**
- All MML raster/vector data is in EPSG:3067 (ETRS-TM35FIN)
- Use `pyproj.Transformer` (not the deprecated `pyproj.transform`) for CRS conversions
- KML output must be in EPSG:4326 (WGS84)

**MML band order:**
- Vääräväri (CIR): Band 1 = NIR, Band 2 = Red, Band 3 = Green
- Väri (RGB): Band 1 = Red, Band 2 = Green, Band 3 = Blue

**Memory management:**
- Never load a full 5km tile into memory
- Use `rasterio.windows.Window` for windowed/tiled reads (512×512px tiles)
- Only process tiles that intersect the hydrography stream buffer mask

## Architecture

The pipeline has five stages (see `beaver-detector-roadmap-v2.md` for full task breakdown):

1. **Ingestion** — read `.jp2` metadata, validate CRS
2. **Masking** — load hydrography vectors, create 100m stream buffer, restrict tile processing to buffered area
3. **Training data** — parse KML labels → reproject to EPSG:3067 → extract 512×512 chips around known territories; generate equal-count negative samples from stream areas
4. **Detection** — compute NDWI/NDVI spectral indices on CIR imagery, run classifier (Random Forest or CNN) on chips, merge neighboring detections into ROI polygons (minimum-area filter)
5. **Export** — reproject polygons to EPSG:4326, write KML with confidence-coded colors and area metadata; batch CLI processes a directory of `.jp2` files into one combined KML

Core dependencies: `rasterio`, `fiona`, `pyproj`, `geopandas`, `shapely`, `opencv-python`, `scikit-learn`
