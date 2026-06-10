import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import osm2geojson

SRC = Path.home() / "shenzhen-osm"
OUT = SRC / "nanshan"
OUT.mkdir(exist_ok=True)

NANSHAN_REL_ID = 5664195
LAYERS = ["water", "landuse", "roads", "pois", "buildings"]


def fetch_boundary():
    out = OUT / "boundary.geojson"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] boundary exists")
        return gpd.read_file(out)

    query = f"""
        [out:json][timeout:120];
        relation({NANSHAN_REL_ID});
        out geom;
    """
    print("fetching Nanshan boundary...")
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={"User-Agent": "claude-code-osm/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        osm = json.loads(r.read())
    gj = osm2geojson.json2geojson(osm)
    with open(out, "w") as f:
        json.dump(gj, f, ensure_ascii=False)
    print(f"  wrote {out.name}")
    return gpd.read_file(out)


def clip_layer(name, mask_geom):
    src = SRC / f"{name}.geojson"
    dst = OUT / f"{name}.geojson"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {name}")
        return
    print(f"[clip] {name}", flush=True)
    t0 = time.time()
    g = gpd.read_file(src)
    n_before = len(g)
    # use intersects predicate first (fast), then clip geometries
    hit = g[g.intersects(mask_geom)]
    clipped = gpd.clip(hit, mask_geom, keep_geom_type=False)
    clipped = clipped[~clipped.geometry.is_empty]
    clipped.to_file(dst, driver="GeoJSON")
    print(
        f"   {n_before} -> {len(clipped)} features, "
        f"{dst.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s"
    )


def main():
    boundary = fetch_boundary()
    mask = boundary.geometry.union_all()
    for layer in LAYERS:
        clip_layer(layer, mask)


if __name__ == "__main__":
    main()
