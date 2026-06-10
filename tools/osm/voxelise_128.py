# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Voxelise the 2D building_height raster into (128, 4096, 4096) uint8.

Matches the convention of geometry_2048x2048.npy in SCALED-UrbanFlowv3:
shape (D, H, W), dtype uint8, 1 inside solid (building/ground), 0 in air.
Depth = 128 cells @ 4 m = 512 m vertical extent. Max building height in our
crop is 392.5 m, so no clipping at this depth.
"""

from pathlib import Path
import numpy as np
import time

SRC = Path(__file__).resolve().parents[2] / ".cache" / "shenzhen-osm" / "nanshan"
IN_NPZ = SRC / "raster_4m_4096.npz"
OUT_DIR = Path(
    "/leonardo_scratch/large/userexternal/yli00001/SCALED_Project/"
    "SCALED-UrbanFlowv3/data/geometry"
)
OUT_NPY = OUT_DIR / "NANSHAN_128x4096x4096.npy"

DEPTH = 128
RES = 4.0


def main():
    t0 = time.time()
    d = np.load(IN_NPZ, allow_pickle=True)
    bh = d["building_height"].astype(np.float32)
    H, W = bh.shape
    print(f"[load] building_height {H} x {W}  max={bh.max():.1f} m", flush=True)
    assert (H, W) == (4096, 4096), f"expected 4096x4096, got {H}x{W}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build incrementally to keep peak RAM manageable (final ~2.15 GB).
    # Write directly to a memory-mapped array.
    geom = np.lib.format.open_memmap(
        OUT_NPY, mode="w+", dtype=np.uint8, shape=(DEPTH, H, W)
    )
    print(f"[vox]  -> {OUT_NPY}  ({DEPTH}*{H}*{W} uint8 = {DEPTH*H*W/1e9:.2f} GB)", flush=True)

    occ = 0
    for k in range(DEPTH):
        z = k * RES
        slab = (z < bh).astype(np.uint8)
        geom[k] = slab
        occ += int(slab.sum())
        if k % 16 == 0:
            print(f"  z={k:3d}  ({z:6.1f} m)  occ_so_far={occ:,}", flush=True)
    geom.flush()
    del geom

    print(f"  total occupied cells: {occ:,}  ~{occ*64/1e9:.3f} G m³")
    n_clip = int((bh > DEPTH * RES).sum())
    print(f"  building cells above z={DEPTH} (clipped): {n_clip:,}")
    print(f"[done] {OUT_NPY}  ({OUT_NPY.stat().st_size/1e9:.2f} GB)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
