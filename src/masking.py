"""Stream buffer mask from MML hydrography vector data."""

from pathlib import Path

import fiona
import geopandas as gpd
from shapely.ops import unary_union

BUFFER_METERS = 100
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}

# MML GeoPackage layers that represent flowing water (streams and stream areas).
# Lakes (jarvi) and other features are intentionally excluded — beaver activity
# is concentrated along stream corridors, not open lake shores.
_MML_STREAM_LAYERS = ("virtavesialue", "virtavesikapea")


def build_stream_mask(hydro_path: str) -> object:
    """
    Load MML hydrography vectors and return a single merged Shapely geometry
    representing a BUFFER_METERS buffer around all stream features (EPSG:3067).

    hydro_path may be a single vector file or a directory; all recognised vector
    files found directly inside a directory are loaded and merged.
    For MML GeoPackages only the stream layers (virtavesialue, virtavesikapea)
    are loaded; single-layer files (e.g. Shapefile) are loaded as-is.
    """
    files = _resolve_files(hydro_path)
    if not files:
        raise ValueError(f"No vector files found at {hydro_path}")

    gdfs = []
    for f in files:
        for gdf in _load_stream_layers(f):
            if gdf.crs is None:
                raise ValueError(f"No CRS found in {f}")
            if gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            gdfs.append(gdf)

    if not gdfs:
        raise ValueError(f"No stream layers found in {hydro_path}")

    combined = gpd.pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    return unary_union(combined.geometry.buffer(BUFFER_METERS))


def _load_stream_layers(path: Path) -> list[gpd.GeoDataFrame]:
    """Return GeoDataFrames for the relevant stream layers in a vector file."""
    try:
        available = fiona.listlayers(str(path))
    except Exception:
        # Single-layer format (e.g. Shapefile) — load as-is.
        return [gpd.read_file(path)]

    layers = [l for l in _MML_STREAM_LAYERS if l in available]
    if not layers:
        # GeoPackage doesn't contain expected MML layers — fall back to default.
        return [gpd.read_file(path)]

    return [gpd.read_file(path, layer=l) for l in layers]


def load_stream_lines(hydro_path: str):
    """
    Return a merged Shapely geometry of stream centrelines (virtavesikapea)
    for use in computing dam line orientations.  Returns None if no line
    layers are found.
    """
    files = _resolve_files(hydro_path)
    gdfs = []
    for f in files:
        try:
            available = fiona.listlayers(str(f))
        except Exception:
            continue
        if "virtavesikapea" in available:
            gdf = gpd.read_file(f, layer="virtavesikapea")
            if gdf.crs is not None and gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            gdfs.append(gdf)
    if not gdfs:
        return None
    combined = gpd.pd.concat(gdfs, ignore_index=True)
    return combined.geometry.unary_union


def _resolve_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _VECTOR_SUFFIXES)
