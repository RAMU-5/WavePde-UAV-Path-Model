"""Population density raster ingest (WorldPop primary, LandScan fallback).

Output is a `(nx, ny)` float32 array aligned with `aura.geom.grid.SimGrid`,
units = persons / cell at the simulation grid resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PopulationRaster:
    density: np.ndarray   # (nx, ny) float32, persons per cell
    crs: str = "EPSG:32650"
    source: str = "worldpop"


def load_worldpop(tif_path: str, grid,
                  resampling: str = "bilinear") -> PopulationRaster:
    """Read WorldPop GeoTIFF, reproject to grid CRS, resample to grid resolution."""
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_origin

    rs_lookup = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    rs_method = rs_lookup.get(resampling, Resampling.bilinear)

    target_transform = from_origin(
        grid.origin_xy[0],
        grid.origin_xy[1] + grid.size_xy[1],
        grid.dx, grid.dy,
    )
    with rasterio.open(tif_path) as src:
        src_data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            src_data = np.where(src_data == nodata, 0.0, src_data)
        # Reproject + resample to (ny_rows, nx_cols).
        dst = np.zeros((grid.ny, grid.nx), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=grid.crs,
            resampling=rs_method,
        )
    # rasterio uses (rows=y, cols=x); we want (nx, ny).
    density = dst.T.astype(np.float32)
    # WorldPop is persons / 100m^2-cell. Rescale to persons / sim cell area.
    # WorldPop 2020 unconstrained ~100 m grid in WGS84; we approximate scale
    # by ratio of cell areas after reprojection.
    scale = (grid.dx * grid.dy) / (100.0 * 100.0)
    density *= float(scale)
    return PopulationRaster(density=density, crs=grid.crs, source="worldpop")


def load_landscan(tif_path: str, grid) -> PopulationRaster:
    """Same shape as load_worldpop but tags source='landscan'."""
    pr = load_worldpop(tif_path, grid, resampling="bilinear")
    return PopulationRaster(density=pr.density, crs=pr.crs, source="landscan")


def synth_pop(grid, mean: float = 5.0, hotspots: int = 5,
              hotspot_strength: float = 30.0,
              hotspot_sigma_m: float = 800.0,
              seed: int = 0) -> PopulationRaster:
    """Quick offline synthetic density for tests when no GeoTIFF is available."""
    rng = np.random.default_rng(seed)
    nx, ny = grid.nx, grid.ny
    base = np.full((nx, ny), mean, dtype=np.float32)
    centers = grid.cell_centers_xy()  # (nx, ny, 2)
    for _ in range(hotspots):
        cx = rng.uniform(grid.origin_xy[0], grid.origin_xy[0] + grid.size_xy[0])
        cy = rng.uniform(grid.origin_xy[1], grid.origin_xy[1] + grid.size_xy[1])
        d2 = (centers[..., 0] - cx) ** 2 + (centers[..., 1] - cy) ** 2
        base += hotspot_strength * np.exp(-0.5 * d2 / hotspot_sigma_m ** 2).astype(np.float32)
    return PopulationRaster(density=base, crs=grid.crs, source="synthetic")


def save(pr: PopulationRaster, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, density=pr.density, crs=pr.crs, source=pr.source)


def load(path: str | Path) -> PopulationRaster:
    z = np.load(path, allow_pickle=False)
    return PopulationRaster(
        density=z["density"].astype(np.float32),
        crs=str(z["crs"]),
        source=str(z["source"]),
    )
