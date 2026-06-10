import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import osm2geojson

OUT_DIR = Path(__file__).resolve().parents[2] / ".cache" / "shenzhen-osm"
OUT_DIR.mkdir(exist_ok=True)

SHENZHEN_REL_ID = 3464353
AREA_ID = 3600000000 + SHENZHEN_REL_ID  # 3603464353

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

LAYERS = {
    "boundary": f"""
        [out:json][timeout:300];
        relation({SHENZHEN_REL_ID});
        out geom;
    """,
    "water": f"""
        [out:json][timeout:600];
        area({AREA_ID})->.a;
        (
          way["natural"="water"](area.a);
          relation["natural"="water"](area.a);
          way["waterway"](area.a);
          relation["waterway"](area.a);
        );
        out geom;
    """,
    "landuse": f"""
        [out:json][timeout:600];
        area({AREA_ID})->.a;
        (
          way["landuse"](area.a);
          relation["landuse"](area.a);
        );
        out geom;
    """,
    "roads": f"""
        [out:json][timeout:900];
        area({AREA_ID})->.a;
        (
          way["highway"](area.a);
        );
        out geom;
    """,
    "pois": f"""
        [out:json][timeout:900];
        area({AREA_ID})->.a;
        (
          node["amenity"](area.a);
          node["shop"](area.a);
          node["tourism"](area.a);
          node["leisure"](area.a);
          way["amenity"](area.a);
          way["shop"](area.a);
          way["tourism"](area.a);
          way["leisure"](area.a);
        );
        out geom;
    """,
    "buildings": f"""
        [out:json][timeout:1800];
        area({AREA_ID})->.a;
        (
          way["building"](area.a);
          relation["building"](area.a);
        );
        out geom;
    """,
}


def fetch(query, timeout=1800):
    last_err = None
    for ep in ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"  -> {ep} (try {attempt + 1})", flush=True)
                data = urllib.parse.urlencode({"data": query}).encode()
                req = urllib.request.Request(
                    ep, data=data, headers={"User-Agent": "claude-code-osm/1.0"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except Exception as e:
                msg = str(e)[:200]
                print(f"     failed: {msg}", flush=True)
                last_err = e
                time.sleep(15)
    raise last_err


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    layers = {only: LAYERS[only]} if only else LAYERS
    for name, query in layers.items():
        out = OUT_DIR / f"{name}.geojson"
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {name} (exists, {out.stat().st_size/1e6:.1f} MB)", flush=True)
            continue
        print(f"[fetch] {name}", flush=True)
        t0 = time.time()
        osm = fetch(query)
        n = len(osm.get("elements", []))
        print(f"   got {n} elements in {time.time()-t0:.0f}s", flush=True)
        gj = osm2geojson.json2geojson(osm)
        nfeat = len(gj.get("features", []))
        with open(out, "w") as f:
            json.dump(gj, f, ensure_ascii=False)
        print(
            f"   wrote {out.name}: {nfeat} features, {out.stat().st_size/1e6:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
