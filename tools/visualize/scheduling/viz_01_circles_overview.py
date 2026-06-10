"""Viz 1 (scheduling): overview of business circles + residential stations.

Shows:
  - Building footprints (grey)
  - 7 business circles as coloured discs (radius scaled, label = name)
  - 118 stations coloured by residential_score
  - Each circle's "primary" station highlighted
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

from business_circles import (load_circles_with_stations, demand_profile,
                              NANSHAN_CIRCLES)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOMAIN = ROOT / "input" / "domain.npz"
OSM_BUILDINGS = ROOT / "shenzhen-osm" / "nanshan" / "buildings.geojson"
OUT_DIR = ROOT / "output" / "figures" / "scheduling_system"

CIRCLE_COLORS = {
    "office":        "#5fc8ff",
    "mall":          "#ff7755",
    "tech":          "#3affb3",
    "entertainment": "#ffaa44",
}


def _parse_tags(s):
    if isinstance(s, dict): return s
    if isinstance(s, str) and s.startswith("{"):
        try: return eval(s)
        except Exception: return {}
    return {}


def _load_buildings(bbox):
    lon0, lat0, lon1, lat1 = bbox
    with open(OSM_BUILDINGS) as f:
        gj = json.load(f)
    polys = []
    for feat in gj["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon": rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [poly[0] for poly in g["coordinates"]]
        else: continue
        for ring in rings:
            if len(ring) < 4: continue
            arr = np.array(ring)
            if (arr[:, 0].max() < lon0 or arr[:, 0].min() > lon1 or
                arr[:, 1].max() < lat0 or arr[:, 1].min() > lat1):
                continue
            polys.append(arr)
    return polys


def main():
    info = load_circles_with_stations()
    bbox = info["bbox_lonlat"]
    lon0, lat0, lon1, lat1 = bbox
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0))

    polys = _load_buildings(bbox)
    print(f"[viz1-sched] {len(polys)} buildings, "
          f"{len(NANSHAN_CIRCLES)} circles")

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0])
    ax = fig.add_subplot(gs[0])
    ax_demand = fig.add_subplot(gs[1])

    ax.set_facecolor("#f4f1ec")

    # Building footprints
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(polys, facecolor="#777777",
                        edgecolor="black", linewidths=0.08, alpha=0.45)
    ax.add_collection(pc)

    # Stations coloured by residential score
    rs = info["residential_score"]
    sx = info["station_xy"][:, 0]
    sy = info["station_xy"][:, 1]
    # Convert local m back to lon/lat for plotting on lon/lat axes
    s_lon = sx / LON_M + lon0
    s_lat = sy / LAT_M + lat0
    sc = ax.scatter(s_lon, s_lat, c=rs, cmap="viridis", s=44,
                    vmin=0, vmax=1, edgecolor="black", linewidths=0.4,
                    zorder=8, label=f"118 stations (color = residential)")
    cbar_st = plt.colorbar(sc, ax=ax, fraction=0.024, pad=0.01,
                           location="left")
    cbar_st.set_label("residential score (1=pure residential, 0=in circle)")

    # Business circles as patches
    lat_mid = 0.5 * (lat0 + lat1)
    aspect_ratio = 1.0 / np.cos(np.deg2rad(lat_mid))
    for k, c in enumerate(info["circle_meta"]):
        # radius in deg lon (account for aspect distortion)
        r_deg = c.radius_m / LON_M
        col = CIRCLE_COLORS[c.profile]
        circle_patch = MplCircle((c.lon, c.lat), r_deg, color=col,
                                 alpha=0.45, zorder=6,
                                 edgecolor="black", linewidth=1.2)
        ax.add_patch(circle_patch)
        ax.scatter(c.lon, c.lat, s=180, marker="*", c=col,
                   edgecolor="black", linewidths=1.0, zorder=10)
        # Label
        ax.text(c.lon, c.lat + r_deg + 0.0006,
                f"{c.name_en}\n(w={c.weight:.1f}, {c.profile})",
                fontsize=8, ha="center", va="bottom",
                fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=col, alpha=0.85, edgecolor="black",
                          linewidth=0.5))
        # Highlight nearest station
        st = info["circle_to_station"][k]
        ax.scatter(s_lon[st], s_lat[st], s=160, marker="s",
                   facecolor="none", edgecolor=col,
                   linewidths=2.5, zorder=11)

    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(aspect_ratio)
    Lx_km = (lon1 - lon0) * LON_M / 1000
    Ly_km = (lat1 - lat0) * LAT_M / 1000
    ax.set_title(f"Nanshan scheduling system — 7 business circles + 118 stations  "
                 f"({Lx_km:.1f}×{Ly_km:.1f} km)", fontsize=12)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.3)

    # Legend by profile
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CIRCLE_COLORS[k], edgecolor="black",
                     label=k) for k in CIRCLE_COLORS]
    handles.append(plt.Line2D([0], [0], marker="s", color="w",
                              markerfacecolor="none", markeredgecolor="black",
                              markersize=10, label="primary station"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    # Demand profile panel (right)
    t = np.arange(0, 24, 0.1)
    for prof, col in CIRCLE_COLORS.items():
        y = np.array([demand_profile(ti, prof) for ti in t])
        ax_demand.plot(t, y, color=col, lw=2, label=prof)
    ax_demand.set_xlim(0, 24); ax_demand.set_xticks(np.arange(0, 25, 4))
    ax_demand.set_xlabel("hour of day")
    ax_demand.set_ylabel("demand multiplier λ(t)")
    ax_demand.set_title("24h demand profiles by circle type")
    ax_demand.legend(loc="upper left")
    ax_demand.grid(True, alpha=0.3)
    ax_demand.axvspan(11.5, 13.0, alpha=0.1, color="red", label="lunch")
    ax_demand.axvspan(17.5, 20.0, alpha=0.1, color="orange", label="dinner")
    ax_demand.text(12.25, 1.6, "lunch peak", ha="center", color="red")
    ax_demand.text(18.75, 1.6, "dinner peak", ha="center", color="orange")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "01_circles_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
