"""Stream buffer mask from MML hydrography vector data."""

import geopandas as gpd
from shapely.ops import unary_union

BUFFER_METERS = 100


def build_stream_mask(hydro_path: str) -> object:
    """
    Load MML hydrography vectors and return a single merged Shapely geometry
    representing a BUFFER_METERS buffer around all stream features (EPSG:3067).
    """
    gdf = gpd.read_file(hydro_path)

    if gdf.crs is None:
        raise ValueError(f"No CRS found in {hydro_path}")
    if gdf.crs.to_epsg() != 3067:
        gdf = gdf.to_crs(epsg=3067)

    buffered = gdf.geometry.buffer(BUFFER_METERS)
    return unary_union(buffered)
