"""Stream buffer mask from MML hydrography vector data."""

from pathlib import Path
from typing import Optional, Tuple

import fiona
import geopandas as gpd
import pandas as pd

BUFFER_METERS = 50
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}

_MML_AREA_LAYERS = ("virtavesialue",)  # polygon water bodies
_MML_LINE_LAYERS = ("virtavesikapea",) # narrow waterway lines (all included)

# tasosijainti == -1 means underground culvert — not visible from aerial imagery.
_SURFACE_ONLY = 0


class StreamMask:
    """
    Buffered stream GeoDataFrame with a spatial index for fast per-patch queries.

    intersects() uses the R-tree sindex to find candidates then does exact
    geometry tests only on those.
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        self._gdf = gdf

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(self._gdf.total_bounds)

    @property
    def is_empty(self) -> bool:
        return len(self._gdf) == 0

    def intersects(self, geom) -> bool:
        candidates = self._gdf.sindex.query(geom)
        if len(candidates) == 0:
            return False
        return bool(self._gdf.iloc[candidates].intersects(geom).any())

    def count_intersecting(self, geom) -> int:
        """Return the number of buffered features that intersect geom."""
        candidates = self._gdf.sindex.query(geom)
        if len(candidates) == 0:
            return 0
        return int(self._gdf.iloc[candidates].intersects(geom).sum())


def build_stream_mask(
    hydro_path: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> StreamMask:
    """
    Load MML hydrography vectors and return a StreamMask backed by a spatial index.

    Both virtavesialue (polygon water bodies) and virtavesikapea (narrow waterway
    lines) are included — beavers can colonise any watercourse. Underground culverts
    (tasosijainti == -1) are excluded as they are not visible in aerial imagery.

    bbox: optional (minx, miny, maxx, maxy) in EPSG:3067 to spatially pre-filter
          features at load time.  Pass the JP2 tile bounds (+ BUFFER_METERS margin)
          to load only the features relevant to a single tile — reduces load from
          millions of features down to thousands for a 6×6 km tile.
    """
    files = _resolve_files(hydro_path)
    if not files:
        raise ValueError(f"No vector files found at {hydro_path}")

    gdfs: list[gpd.GeoDataFrame] = []
    total_raw = 0

    for f in files:
        print(f"  Reading {f.name} ...")
        for _layer_name, gdf in _load_stream_layers(f, bbox=bbox):
            if len(gdf) == 0:
                continue
            if gdf.crs is None:
                raise ValueError(f"No CRS found in {f}")
            if gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            # Drop underground culverts — not visible from aerial imagery.
            if "tasosijainti" in gdf.columns:
                gdf = gdf[gdf["tasosijainti"] == _SURFACE_ONLY]
            total_raw += len(gdf)
            gdfs.append(gdf[["geometry"]])

    if not gdfs:
        raise ValueError(f"No stream layers found in {hydro_path}")

    if len(gdfs) == 1:
        combined = gdfs[0].reset_index(drop=True)
    else:
        combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    print(f"  Buffering {len(combined)} features by {BUFFER_METERS} m ...")
    combined = combined.copy()
    combined["geometry"] = combined.geometry.buffer(BUFFER_METERS)
    combined = combined[~combined.geometry.is_empty & combined.geometry.notna()].reset_index(drop=True)
    print(f"  Stream mask ready ({len(combined)} buffered features).")
    return StreamMask(combined)


def _load_stream_layers(
    path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Return [(layer_name, GeoDataFrame), ...] for relevant stream layers."""
    all_layers = _MML_AREA_LAYERS + _MML_LINE_LAYERS
    kwargs = {"bbox": bbox} if bbox is not None else {}
    try:
        available = fiona.listlayers(str(path))
    except Exception:
        return [("unknown", gpd.read_file(path, **kwargs))]

    layers = [l for l in all_layers if l in available]
    if not layers:
        return [("unknown", gpd.read_file(path, **kwargs))]

    return [(l, gpd.read_file(path, layer=l, **kwargs)) for l in layers]


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
