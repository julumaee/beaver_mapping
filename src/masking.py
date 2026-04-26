"""Stream buffer mask from MML hydrography vector data."""

from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

BUFFER_METERS = 50
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}

_MML_STREAM_LAYERS = ("virtavesialue", "virtavesikapea")


class StreamMask:
    """
    Buffered stream GeoDataFrame with a spatial index for fast per-patch queries.

    Replaces the old unary_union approach: no upfront merge is required, so
    startup is instant.  intersects() uses the R-tree sindex to find candidates
    then does exact geometry tests only on those.
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        self._gdf = gdf

    def intersects(self, geom) -> bool:
        candidates = self._gdf.sindex.query(geom)
        if len(candidates) == 0:
            return False
        return bool(self._gdf.iloc[candidates].intersects(geom).any())


def build_stream_mask(hydro_path: str) -> StreamMask:
    """
    Load MML hydrography vectors, buffer by BUFFER_METERS, and return a
    StreamMask backed by a spatial index.

    hydro_path may be a single vector file or a directory.  For MML GeoPackages
    only the stream layers (virtavesialue, tulvaalue, virtavesikapea) are loaded.
    """
    files = _resolve_files(hydro_path)
    if not files:
        raise ValueError(f"No vector files found at {hydro_path}")

    gdfs = []
    for f in files:
        print(f"  Reading {f.name} ...")
        for gdf in _load_stream_layers(f):
            if gdf.crs is None:
                raise ValueError(f"No CRS found in {f}")
            if gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            gdfs.append(gdf)

    if not gdfs:
        raise ValueError(f"No stream layers found in {hydro_path}")

    if len(gdfs) > 1:
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        combined = gdfs[0]

    print(f"  Buffering {len(combined)} features by {BUFFER_METERS} m ...")
    combined = combined[["geometry"]].copy()
    combined["geometry"] = combined.geometry.buffer(BUFFER_METERS)
    combined = combined[~combined.geometry.is_empty & combined.geometry.notna()]
    print(f"  Stream mask ready ({len(combined)} buffered features).")
    return StreamMask(combined)


def _load_stream_layers(path: Path) -> list[gpd.GeoDataFrame]:
    """Return GeoDataFrames for the relevant stream layers in a vector file."""
    try:
        available = fiona.listlayers(str(path))
    except Exception:
        return [gpd.read_file(path)]

    layers = [l for l in _MML_STREAM_LAYERS if l in available]
    if not layers:
        return [gpd.read_file(path)]

    return [gpd.read_file(path, layer=l) for l in layers]


def load_stream_lines(hydro_path: str):
    """
    Return a merged Shapely geometry of stream centrelines (virtavesikapea)
    for dam orientation computation.  Returns None if no line layers are found.
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
    if len(gdfs) > 1:
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        combined = gdfs[0]
    return combined.geometry.unary_union


def _resolve_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _VECTOR_SUFFIXES)
