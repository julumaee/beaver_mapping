"""Stream buffer mask from MML hydrography vector data."""

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

BUFFER_METERS = 100
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}


def build_stream_mask(hydro_path: str) -> object:
    """
    Load MML hydrography vectors and return a single merged Shapely geometry
    representing a BUFFER_METERS buffer around all stream features (EPSG:3067).

    hydro_path may be a single vector file or a directory; all recognised vector
    files found directly inside a directory are loaded and merged.
    """
    files = _resolve_files(hydro_path)
    if not files:
        raise ValueError(f"No vector files found at {hydro_path}")

    gdfs = []
    for f in files:
        gdf = gpd.read_file(f)
        if gdf.crs is None:
            raise ValueError(f"No CRS found in {f}")
        if gdf.crs.to_epsg() != 3067:
            gdf = gdf.to_crs(epsg=3067)
        gdfs.append(gdf)

    combined = gpd.pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    return unary_union(combined.geometry.buffer(BUFFER_METERS))


def _resolve_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _VECTOR_SUFFIXES)
