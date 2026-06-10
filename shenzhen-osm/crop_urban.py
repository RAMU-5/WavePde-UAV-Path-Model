# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "scipy"]
# ///
"""Crop the rasterized stack to the dense urban region (largest building cluster).

We dilate the building mask to merge nearby blocks, take the largest connected
component, then crop all layers to its bounding box + small margin.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = Path(__file__).resolve().parent / "nanshan"
IN_NPZ = SRC / "raster_4m.npz"
OUT_NPZ = SRC / "raster_4m_urban.npz"
OUT_PNG = SRC / "raster_4m_urban_preview.png"

RES = 4.0
DILATE_M = 200.0          # merge clusters within ~200 m
MARGIN_M = 100.0          # padding around the chosen region

def main():
    d = np.load(IN_NPZ, allow_pickle=True)
    buildings = d["buildings"]
    transform = d["transform"]
    a, b, c, dd, e, f = transform.tolist()   # a=res, e=-res, c=minx, f=maxy
    H, W = buildings.shape

    # 1) dilate to merge nearby blocks
    k = int(round(DILATE_M / RES))
    print(f"[step] dilate buildings by {k} cells (~{k*RES:.0f} m)", flush=True)
    dilated = ndi.binary_dilation(buildings > 0, iterations=k)

    # 2) connected components, pick largest
    lab, n = ndi.label(dilated)
    sizes = ndi.sum(dilated, lab, index=np.arange(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    print(f"[step] {n} clusters, biggest has {int(sizes[big-1]):,} dilated cells", flush=True)

    mask = lab == big
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    # 3) margin
    m = int(round(MARGIN_M / RES))
    r0 = max(0, r0 - m); r1 = min(H, r1 + m)
    c0 = max(0, c0 - m); c1 = min(W, c1 + m)
    print(f"[step] crop rows {r0}:{r1}  cols {c0}:{c1}  -> {r1-r0} x {c1-c0}", flush=True)

    # 4) crop every array; rebuild transform (origin = top-left of crop)
    new_minx = c + c0 * RES
    new_maxy = f + e * r0               # e is negative
    new_transform = np.array([a, b, new_minx, dd, e, new_maxy], dtype=np.float64)
    new_minx_f = new_minx
    new_maxy_f = new_maxy
    new_maxx_f = new_minx + (c1 - c0) * RES
    new_miny_f = new_maxy + (r1 - r0) * e   # going down
    new_bounds = np.array(
        [new_minx_f, new_miny_f, new_maxx_f, new_maxy_f], dtype=np.float64
    )

    arrays = {}
    for k_ in ("buildings", "building_height", "water", "roads",
               "landuse_class", "boundary"):
        arrays[k_] = d[k_][r0:r1, c0:c1].copy()

    np.savez_compressed(
        OUT_NPZ,
        **arrays,
        transform=new_transform,
        bounds=new_bounds,
        crs=d["crs"],
        res=d["res"],
        landuse_classes=d["landuse_classes"],
    )
    print(f"[save] {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.1f} MB)", flush=True)

    # 5) preview
    h, w = arrays["buildings"].shape
    rgb = np.full((h, w, 3), 244, dtype=np.uint8)
    rgb[arrays["boundary"] == 0] = (255, 255, 255)
    rgb[arrays["landuse_class"] > 0] = (230, 226, 211)
    rgb[arrays["water"] == 1] = (166, 206, 227)
    rgb[arrays["roads"] == 1] = (90, 90, 90)
    rgb[arrays["buildings"] == 1] = (140, 50, 50)
    img = Image.fromarray(rgb)
    if max(img.size) > 4096:
        s = 4096 / max(img.size)
        img = img.resize((int(img.size[0]*s), int(img.size[1]*s)), Image.NEAREST)
    img.save(OUT_PNG, optimize=True)
    print(f"[save] {OUT_PNG} ({img.size})", flush=True)

    # building stats
    bcount = int((arrays["buildings"] > 0).sum())
    print(f"  buildings in crop: {bcount:,} cells (~{bcount*16/1e6:.2f} km²)")
    print(f"  height max: {float(arrays['building_height'].max()):.1f} m")

if __name__ == "__main__":
    main()
