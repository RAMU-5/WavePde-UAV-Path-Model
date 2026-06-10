# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Voxelise the 2D building_height raster into a 3D occupancy volume.

Output shape (D, H, W) uint8, D=64 (256 m at 4 m grid), matching the model's
training distribution.  Heights >256 m are clipped (~10 k cells in our crop).
"""

from pathlib import Path
import numpy as np
import time

SRC = Path(__file__).resolve().parents[2] / ".cache" / "shenzhen-osm" / "nanshan"
IN_NPZ = SRC / "raster_4m_4096.npz"
OUT_NPY = SRC / "geometry_64x4096x4096.npy"
DEPTH = 64
RES = 4.0


def main():
    t0 = time.time()
    d = np.load(IN_NPZ, allow_pickle=True)
    bh = d["building_height"].astype(np.float32)
    H, W = bh.shape
    print(f"[load] building_height {H} x {W}  max={bh.max():.1f} m", flush=True)

    z_layers = np.arange(DEPTH, dtype=np.float32) * RES         # (D,)  0,4,...,252
    geom = (z_layers[:, None, None] < bh[None, :, :]).astype(np.uint8)  # (D,H,W)
    print(f"[vox]  shape={geom.shape} dtype={geom.dtype}", flush=True)

    occ = int(geom.sum())
    nz = int((bh > DEPTH * RES).sum())
    print(f"  occupied cells: {occ:,}  ~{occ*64/1e9:.3f} G m³")
    print(f"  building cells clipped to z={DEPTH}: {nz:,}", flush=True)

    np.save(OUT_NPY, geom)
    print(f"[save] {OUT_NPY}  ({OUT_NPY.stat().st_size/1e9:.2f} GB)")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
