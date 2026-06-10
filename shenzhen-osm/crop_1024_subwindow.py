# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Pick the densest 1024x1024 sub-window of the (64,4096,4096) voxel volume
and save it as NANSHAN_64x1024x1024.npy in the UrbanFlowv3 geometry dir."""

from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parent / "nanshan"
IN_NPY = SRC / "geometry_64x4096x4096.npy"
IN_NPZ = SRC / "raster_4m_4096.npz"

OUT_DIR = Path(
    "/leonardo_scratch/large/userexternal/yli00001/SCALED_Project/"
    "SCALED-UrbanFlowv3/data/geometry"
)
OUT_NPY = OUT_DIR / "NANSHAN_64x1024x1024.npy"
OUT_PNG = SRC / "nanshan_1024_subwindow.png"
WIN = 1024


def main():
    print(f"[load] {IN_NPY}", flush=True)
    geom = np.load(IN_NPY)
    D, H, W = geom.shape
    print(f"  shape={geom.shape}", flush=True)

    # also load 2D building_height for scoring (taller cells count more)
    nz = np.load(IN_NPZ, allow_pickle=True)
    bh = nz["building_height"].astype(np.float32)            # (4096,4096)
    bld = nz["buildings"].astype(np.int32)                   # 0/1
    weight = bld + (bh > 30).astype(np.int32) * 2 + (bh > 100).astype(np.int32) * 2
    ii = weight.cumsum(0).cumsum(1)
    pad = np.zeros((H + 1, W + 1), dtype=ii.dtype)
    pad[1:, 1:] = ii

    def s(r0, c0):
        return pad[r0+WIN, c0+WIN] - pad[r0, c0+WIN] - pad[r0+WIN, c0] + pad[r0, c0]

    best = (-1, 0, 0)
    step = 32
    for r in range(0, H - WIN + 1, step):
        for c in range(0, W - WIN + 1, step):
            v = s(r, c)
            if v > best[0]:
                best = (v, r, c)
    s0, r0, c0 = best
    for r in range(max(0, r0 - step), min(H - WIN, r0 + step) + 1):
        for c in range(max(0, c0 - step), min(W - WIN, c0 + step) + 1):
            v = s(r, c)
            if v > best[0]:
                best = (v, r, c)
    score, r0, c0 = best
    r1, c1 = r0 + WIN, c0 + WIN
    print(f"[pick] sub-window rows {r0}:{r1} cols {c0}:{c1}  score={int(score):,}", flush=True)

    sub = geom[:, r0:r1, c0:c1].copy()
    print(f"  sub shape={sub.shape} dtype={sub.dtype}  occupied={int(sub.sum()):,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, sub)
    print(f"[save] {OUT_NPY}  ({OUT_NPY.stat().st_size/1e6:.1f} MB)", flush=True)

    # preview of this slice
    sub_bh = bh[r0:r1, c0:c1]
    sub_bld = bld[r0:r1, c0:c1]
    rgb = np.full((WIN, WIN, 3), 244, dtype=np.uint8)
    rgb[sub_bld > 0] = (140, 50, 50)
    # tint by height
    tall = sub_bh > 100
    rgb[tall] = (255, 90, 50)
    very = sub_bh > 200
    rgb[very] = (255, 240, 60)
    img = Image.fromarray(rgb)
    img.save(OUT_PNG, optimize=True)
    print(f"[save] {OUT_PNG}")

    # save a small json of the sub-window's location for later
    meta = {
        "subwindow_rows": [int(r0), int(r1)],
        "subwindow_cols": [int(c0), int(c1)],
        "subwindow_score": int(score),
        "resolution_m": 4.0,
        "depth_cells": int(D),
        "depth_m": int(D * 4),
    }
    import json
    (OUT_DIR / "NANSHAN_64x1024x1024.meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
