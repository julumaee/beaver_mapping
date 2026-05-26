"""Stream buffer mask from MML hydrography vector data."""

from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

BUFFER_METERS = 50
_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".fgb"}

_MML_AREA_LAYERS = ("virtavesialue",)   # polygon water bodies — always included
_MML_LINE_LAYERS = ("virtavesikapea",)  # narrow waterway lines — filtered by proximity to area layers

# virtavesikapea includes both natural streams and forest drainage ditches.
# Only keep line features within this distance of a virtavesialue polygon so
# that isolated ditch networks in drained forests are excluded.
STREAM_CONNECTIVITY_M = 500


class StreamMask:
    """
    Buffered stream GeoDataFrame with a spatial index for fast per-patch queries.

    Replaces the old unary_union approach: no upfront merge is required, so
    startup is instant.  intersects() uses the R-tree sindex to find candidates
    then does exact geometry tests only on those.
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


def build_stream_mask(hydro_path: str, connectivity_m: float = STREAM_CONNECTIVITY_M) -> StreamMask:
    """
    Load MML hydrography vectors and return a StreamMask backed by a spatial index.

    virtavesialue (stream/river area polygons) are always included.
    virtavesikapea (narrow waterway lines) are filtered to only those within
    STREAM_CONNECTIVITY_M of a virtavesialue polygon — this excludes the vast
    majority of forest drainage ditches which form isolated networks.
    """
    files = _resolve_files(hydro_path)
    if not files:
        raise ValueError(f"No vector files found at {hydro_path}")

    area_gdfs: list[gpd.GeoDataFrame] = []
    line_gdfs: list[gpd.GeoDataFrame] = []

    for f in files:
        print(f"  Reading {f.name} ...")
        for layer_name, gdf in _load_stream_layers(f):
            if gdf.crs is None:
                raise ValueError(f"No CRS found in {f}")
            if gdf.crs.to_epsg() != 3067:
                gdf = gdf.to_crs(epsg=3067)
            if layer_name in _MML_AREA_LAYERS:
                area_gdfs.append(gdf[["geometry"]])
            else:
                line_gdfs.append(gdf[["geometry"]])

    if not area_gdfs and not line_gdfs:
        raise ValueError(f"No stream layers found in {hydro_path}")

    def _concat(gdfs):
        if not gdfs:
            return None
        if len(gdfs) == 1:
            return gdfs[0].reset_index(drop=True)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    areas = _concat(area_gdfs)
    lines = _concat(line_gdfs)

    # Filter narrow waterway lines to those connected to real water bodies.
    if lines is not None and areas is not None and connectivity_m > 0:
        print(f"  Filtering {len(lines)} narrow waterway lines by proximity to "
              f"{len(areas)} stream areas (threshold {connectivity_m} m) ...")
        area_buffered = gpd.GeoDataFrame(
            geometry=areas.geometry.buffer(connectivity_m), crs=areas.crs
        )
        joined = gpd.sjoin(lines, area_buffered, how="inner", predicate="intersects")
        lines = lines.iloc[joined.index.unique()].reset_index(drop=True)
        print(f"  {len(lines)} connected waterway lines retained.")

    parts = [g for g in (areas, lines) if g is not None]
    combined = _concat(parts)

    print(f"  Buffering {len(combined)} features by {BUFFER_METERS} m ...")
    combined = combined.copy()
    combined["geometry"] = combined.geometry.buffer(BUFFER_METERS)
    combined = combined[~combined.geometry.is_empty & combined.geometry.notna()].reset_index(drop=True)
    print(f"  Stream mask ready ({len(combined)} buffered features).")
    return StreamMask(combined)


def _load_stream_layers(path: Path) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Return [(layer_name, GeoDataFrame), ...] for relevant stream layers."""
    all_layers = _MML_AREA_LAYERS + _MML_LINE_LAYERS
    try:
        available = fiona.listlayers(str(path))
    except Exception:
        return [("unknown", gpd.read_file(path))]

    layers = [l for l in all_layers if l in available]
    if not layers:
        return [("unknown", gpd.read_file(path))]

    return [(l, gpd.read_file(path, layer=l)) for l in layers]


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
