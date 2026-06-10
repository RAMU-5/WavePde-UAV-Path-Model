"""Viz 2: 118 deployed stations on the Nanshan map (with buildings).

Reads the same OSM source as viz 1 + the station list from domain.npz.
Stations are coloured by which deployment policy chose them
(block centroid vs hex backfill — read from station_source).
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
DOMAIN = ROOT / "input" / "domain.npz"
OUT = ROOT / "output" / "figures" / "routine_system" / "02_stations.png"


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
    bbox = z["domain_bbox_lonlat"]
    lon0, lat0, lon1, lat1 = bbox
    station_lonlat = z["station_lonlat"]
    station_source = z["station_source"]
    station_n_b = z["station_n_buildings_within_max"]
    n = len(station_lonlat)
    print(f"[viz2] {n} stations to plot")

    # Buildings underlay
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
            if (ring_arr[:, 0].max() < lon0 or ring_arr[:, 0].min() > lon1 or
                ring_arr[:, 1].max() < lat0 or ring_arr[:, 1].min() > lat1):
                continue
            polys.append(ring_arr); heights.append(h)

    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_facecolor("#f4f1ec")
    pc = PolyCollection(polys, array=np.asarray(heights),
                        cmap="copper", norm=plt.Normalize(5, 120),
                        edgecolor="black", linewidths=0.1, alpha=0.6)
    ax.add_collection(pc)

    # Stations split by source
    src_arr = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in station_source])
    is_block = np.array([("block" in s) for s in src_arr])
    is_hex = ~is_block

    if is_block.any():
        ax.scatter(station_lonlat[is_block, 0],
                   station_lonlat[is_block, 1],
                   s=70, marker="o", facecolor="#5fc8ff",
                   edgecolor="black", linewidths=0.6, zorder=10,
                   label=f"block centroid ({is_block.sum()})")
    if is_hex.any():
        ax.scatter(station_lonlat[is_hex, 0],
                   station_lonlat[is_hex, 1],
                   s=70, marker="^", facecolor="#ff7755",
                   edgecolor="black", linewidths=0.6, zorder=10,
                   label=f"hex backfill ({is_hex.sum()})")

    # Station IDs as small text
    for i, (lo, la) in enumerate(station_lonlat):
        ax.text(lo, la, str(i), fontsize=4, ha="center", va="center",
                color="black", zorder=11)

    cbar = plt.colorbar(pc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("building height (m)")

    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    lat_mid = 0.5 * (lat0 + lat1)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    Lx_km = (lon1 - lon0) * 111 * np.cos(np.deg2rad(lat_mid))
    Ly_km = (lat1 - lat0) * 111
    ax.set_title(f"118 stations deployed via greedy set cover  "
                 f"({Lx_km:.1f}×{Ly_km:.1f} km, max coverage 450m)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
