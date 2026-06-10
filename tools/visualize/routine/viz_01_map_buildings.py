"""Viz 1: Nanshan map + building footprints + heights.

Reads OSM geojson directly (raw source) and the domain.npz (for the bbox).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OSM_BUILDINGS = ROOT / "shenzhen-osm" / "nanshan" / "buildings.geojson"
OSM_ROADS = ROOT / "shenzhen-osm" / "nanshan" / "roads.geojson"
DOMAIN = ROOT / "input" / "domain.npz"
OUT = ROOT / "output" / "figures" / "routine_system" / "01_map_buildings.png"


def parse_tags(s):
    if isinstance(s, dict): return s
    if isinstance(s, str) and s.startswith("{"):
        try: return eval(s)
        except Exception: return {}
    return {}


def building_height_m(tags):
    h = tags.get("height")
    if h:
        try: return float(str(h).split()[0])
        except Exception: pass
    lv = tags.get("building:levels")
    if lv:
        try: return 3.0 * float(lv)
        except Exception: pass
    return 9.0


def main():
    z = np.load(DOMAIN, allow_pickle=False)
    bbox = z["domain_bbox_lonlat"]   # (lon0, lat0, lon1, lat1)
    lon0, lat0, lon1, lat1 = bbox

    polys, heights = [], []
    with open(OSM_BUILDINGS) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [poly[0] for poly in g["coordinates"]]
        else:
            continue
        tags = parse_tags(feat.get("properties", {}).get("tags", ""))
        h = building_height_m(tags)
        for ring in rings:
            if len(ring) < 4: continue
            ring_arr = np.array(ring)
            # Inside our domain bbox?
            if (ring_arr[:, 0].max() < lon0 or ring_arr[:, 0].min() > lon1 or
                ring_arr[:, 1].max() < lat0 or ring_arr[:, 1].min() > lat1):
                continue
            polys.append(ring_arr); heights.append(h)
    print(f"[viz1] {len(polys)} buildings inside domain")

    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_facecolor("#f4f1ec")
    pc = PolyCollection(
        polys,
        array=np.asarray(heights),
        cmap="copper",
        norm=plt.Normalize(vmin=5, vmax=120),
        edgecolor="black", linewidths=0.15,
    )
    ax.add_collection(pc)
    cbar = plt.colorbar(pc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("building height (m)")

    # Roads on top
    if OSM_ROADS.exists():
        with open(OSM_ROADS) as f:
            gj_r = json.load(f)
        n_roads = 0
        for feat in gj_r["features"]:
            g = feat["geometry"]
            coords = []
            if g["type"] == "LineString":
                coords = [g["coordinates"]]
            elif g["type"] == "MultiLineString":
                coords = g["coordinates"]
            for c in coords:
                arr = np.array(c)
                if arr[:, 0].max() < lon0 or arr[:, 0].min() > lon1: continue
                if arr[:, 1].max() < lat0 or arr[:, 1].min() > lat1: continue
                ax.plot(arr[:, 0], arr[:, 1], "-", color="#444",
                        lw=0.4, alpha=0.5, zorder=2)
                n_roads += 1
        print(f"[viz1] {n_roads} road segments")

    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    lat_mid = 0.5 * (lat0 + lat1)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    Lx_km = (lon1 - lon0) * 111 * np.cos(np.deg2rad(lat_mid))
    Ly_km = (lat1 - lat0) * 111
    ax.set_title(f"Nanshan upper-right — buildings + roads  "
                 f"({Lx_km:.1f} km × {Ly_km:.1f} km, "
                 f"{len(polys)} buildings)", fontsize=12)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
