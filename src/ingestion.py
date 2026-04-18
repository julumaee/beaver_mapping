"""MML .jp2 metadata reading and windowed tile iteration."""

import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS
from shapely.geometry import box

EXPECTED_CRS = CRS.from_epsg(3067)
TILE_SIZE = 512


def read_metadata(jp2_path: str) -> dict:
    """Return CRS, bounds, and shape of a .jp2 file."""
    with rasterio.open(jp2_path) as src:
        if src.crs != EXPECTED_CRS:
            raise ValueError(f"Expected EPSG:3067, got {src.crs} in {jp2_path}")
        return {
            "path": jp2_path,
            "crs": src.crs,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "count": src.count,
        }


def iter_masked_windows(jp2_path: str, mask_geometry):
    """
    Yield (Window, transform) for each 512x512 tile that intersects mask_geometry.
    mask_geometry must be in EPSG:3067.
    """
    with rasterio.open(jp2_path) as src:
        for row_off in range(0, src.height, TILE_SIZE):
            for col_off in range(0, src.width, TILE_SIZE):
                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(TILE_SIZE, src.width - col_off),
                    height=min(TILE_SIZE, src.height - row_off),
                )
                win_transform = rasterio.windows.transform(win, src.transform)
                win_bounds = rasterio.windows.bounds(win, src.transform)
                win_box = box(*win_bounds)
                if mask_geometry.intersects(win_box):
                    yield win, win_transform
