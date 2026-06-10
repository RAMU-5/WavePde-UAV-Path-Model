# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Cut a 4096x4096 window (at 4 m/grid -> 16.384 km x 16.384 km) from the
full Nanshan raster, placed to maximise total building footprint inside."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[2] / ".cache" / "shenzhen-osm" / "nanshan"
IN_NPZ = SRC / "raster_4m.npz"
OUT_NPZ = SRC / "raster_4m_4096.npz"
OUT_PNG = SRC / "raster_4m_4096_preview.png"
WIN = 4096
RES = 4.0


def main():
    d = np.load(IN_NPZ, allow_pickle=True)
    buildings = d["buildings"]
    H, W = buildings.shape
    a, b, c, dd, e, f = d["transform"].tolist()  # a=4, e=-4, c=minx, f=maxy
    print(f"[load] full {H} x {W} (cells), 4m grid", flush=True)

    if H < WIN or W < WIN:
        raise SystemExit(f"full grid too small for {WIN}x{WIN}")

    # weight: buildings + 0.25 * (building_height > 30) bonus for tall ones
    bh = d["building_height"]
    weight = buildings.astype(np.int32) + (bh > 30).astype(np.int32) * 2
    # integral image
    ii = weight.cumsum(0).cumsum(1)
    pad = np.zeros((H + 1, W + 1), dtype=ii.dtype)
    pad[1:, 1:] = ii

    def window_sum(r0, c0):
        r1, c1 = r0 + WIN, c0 + WIN
        return pad[r1, c1] - pad[r0, c1] - pad[r1, c0] + pad[r0, c0]

    # search top-left in a coarse grid first then refine
    best = (-1, 0, 0)
    step = 32
    for r in range(0, H - WIN + 1, step):
        for c in range(0, W - WIN + 1, step):
            s = window_sum(r, c)
            if s > best[0]:
                best = (s, r, c)
    # refine
    s0, r0, c0 = best
    for r in range(max(0, r0 - step), min(H - WIN, r0 + step) + 1):
        for c in range(max(0, c0 - step), min(W - WIN, c0 + step) + 1):
            s = window_sum(r, c)
            if s > best[0]:
                best = (s, r, c)
    score, r0, c0 = best
    r1, c1 = r0 + WIN, c0 + WIN
    print(f"[pick] window rows {r0}:{r1}  cols {c0}:{c1}  score={int(score):,}", flush=True)

    arrays = {}
    for k in ("buildings", "building_height", "water", "roads",
              "landuse_class", "boundary"):
        arrays[k] = d[k][r0:r1, c0:c1].copy()

    new_minx = c + c0 * RES
    new_maxy = f + e * r0
    new_maxx = new_minx + WIN * RES
    new_miny = new_maxy + WIN * e
    new_transform = np.array([a, b, new_minx, dd, e, new_maxy], dtype=np.float64)
    new_bounds = np.array([new_minx, new_miny, new_maxx, new_maxy], dtype=np.float64)

    np.savez_compressed(
        OUT_NPZ,
        **arrays,
        transform=new_transform,
        bounds=new_bounds,
        crs=d["crs"],
        res=d["res"],
        landuse_classes=d["landuse_classes"],
        window_rc=np.array([r0, c0, r1, c1], dtype=np.int64),
    )
    print(f"[save] {OUT_NPZ}  ({OUT_NPZ.stat().st_size/1e6:.1f} MB)", flush=True)

    # stats
    bcount = int((arrays["buildings"] > 0).sum())
    hmax = float(arrays["building_height"].max())
    print(f"  buildings: {bcount:,} cells  ~{bcount*16/1e6:.2f} km²")
    print(f"  height max: {hmax:.1f} m  (>30m cells: {int((arrays['building_height']>30).sum()):,})")

    # preview
    rgb = np.full((WIN, WIN, 3), 244, dtype=np.uint8)
    rgb[arrays["boundary"] == 0] = (255, 255, 255)
    rgb[arrays["landuse_class"] > 0] = (230, 226, 211)
    rgb[arrays["water"] == 1] = (166, 206, 227)
    rgb[arrays["roads"] == 1] = (90, 90, 90)
    rgb[arrays["buildings"] == 1] = (140, 50, 50)
    img = Image.fromarray(rgb)
    img = img.resize((2048, 2048), Image.NEAREST)
    img.save(OUT_PNG, optimize=True)
    print(f"[save] {OUT_PNG}")


if __name__ == "__main__":
    main()
