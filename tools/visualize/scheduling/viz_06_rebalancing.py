"""Viz 6 (scheduling): rebalancing / deadhead dynamics.

  - Net inflow heatmap: station × time, colour = (inbound − outbound)
    deadhead UAVs per 5-min bin
  - Stacked area: outbound moves over time (top 10 source stations)
  - Stacked area: inbound moves over time (top 10 destination stations)
  - Map view: arrows from total source → total destination (top 30 flows)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle as MplCircle

from business_circles import (load_circles_with_stations, NANSHAN_CIRCLES)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM = ROOT / "output" / "figures" / "scheduling_system" / "sim_log.npz"
OSM_BUILDINGS = ROOT / ".cache" / "shenzhen-osm" / "nanshan" / "buildings.geojson"
OUT = ROOT / "output" / "figures" / "scheduling_system" / "06_rebalancing.png"


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
    z = np.load(SIM, allow_pickle=False)
    info = load_circles_with_stations()
    if "log_deadhead_outbound" not in z.files:
        print("[viz6] sim_log.npz has no rebalancing data — re-run sim")
        return
    out = z["log_deadhead_outbound"]   # (n_ticks, n_st)
    inb = z["log_deadhead_inbound"]
    tick_sec = float(z["tick_sec"])
    t_start = float(z["sim_t_start"])
    n_ticks, n_st = out.shape
    t_h = t_start + np.arange(n_ticks) * tick_sec / 3600.0
    n_total = int(z["n_deadhead_total"])
    print(f"[viz6] {n_total:,} deadhead moves total")

    # Bin to 5-min
    bin_size = max(1, int(300 / tick_sec))
    n_bins = n_ticks // bin_size
    out_bin = out[:n_bins*bin_size].reshape(
        n_bins, bin_size, n_st).sum(axis=1)
    inb_bin = inb[:n_bins*bin_size].reshape(
        n_bins, bin_size, n_st).sum(axis=1)
    t_bins = t_h[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
    net_bin = inb_bin - out_bin

    # Most active stations
    out_total = out.sum(axis=0)
    inb_total = inb.sum(axis=0)
    top_out = np.argsort(-out_total)[:10]
    top_in = np.argsort(-inb_total)[:10]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.4],
                          width_ratios=[1.0, 1.0])

    # Heatmap of net inflow per station × time
    ax_h = fig.add_subplot(gs[0, :])
    sort_by_net = np.argsort(net_bin.sum(axis=0))   # most-net-loss first
    show = net_bin[:, sort_by_net].T
    vmax = max(abs(show.min()), abs(show.max()), 1)
    im = ax_h.imshow(show, aspect="auto", origin="lower",
                     cmap="RdBu", vmin=-vmax, vmax=vmax,
                     extent=[t_bins[0], t_bins[-1], 0, n_st])
    cbar = plt.colorbar(im, ax=ax_h, fraction=0.025)
    cbar.set_label("net inbound − outbound deadheads per 5-min")
    ax_h.set_xlabel("hour of day")
    ax_h.set_ylabel("station (sorted: top = net-receivers, bot = net-senders)")
    ax_h.set_title("Net deadhead flow per station per 5-min bin")

    # Outbound time series (top senders)
    ax_o = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(n_bins)
    for s in top_out:
        ax_o.fill_between(t_bins, bottom, bottom + out_bin[:, s],
                          alpha=0.7, label=f"st {s} ({out_total[s]})")
        bottom += out_bin[:, s]
    ax_o.set_xlabel("hour of day")
    ax_o.set_ylabel("UAVs out per 5-min")
    ax_o.set_title("Top 10 source stations (sending UAVs out)")
    ax_o.legend(fontsize=7, loc="upper right", ncol=2)
    ax_o.grid(True, alpha=0.3)

    # Inbound time series (top receivers)
    ax_i = fig.add_subplot(gs[1, 1])
    bottom = np.zeros(n_bins)
    for s in top_in:
        ax_i.fill_between(t_bins, bottom, bottom + inb_bin[:, s],
                          alpha=0.7, label=f"st {s} ({inb_total[s]})")
        bottom += inb_bin[:, s]
    ax_i.set_xlabel("hour of day")
    ax_i.set_ylabel("UAVs in per 5-min")
    ax_i.set_title("Top 10 destination stations (receiving deadheads)")
    ax_i.legend(fontsize=7, loc="upper right", ncol=2)
    ax_i.grid(True, alpha=0.3)

    # Map view: station-to-station deadhead flow magnitudes
    ax_m = fig.add_subplot(gs[2, :])
    bbox = info["bbox_lonlat"]
    polys = _load_buildings(bbox)
    pc = PolyCollection(polys, facecolor="#aaaaaa", edgecolor="black",
                        linewidths=0.05, alpha=0.35)
    ax_m.add_collection(pc)

    # Aggregate per-pair flows from outbound + inbound? We don't have
    # pair-level logs, but we can show: per station, outbound (red arrow
    # leaving) and inbound (blue arrow entering), magnitude = total over sim.
    sxy = info["station_lonlat"]
    # Top sources (red)
    if out_total.max() > 0:
        s_max = out_total.max()
        for s in top_out:
            sz = 30 + 250 * (out_total[s] / s_max)
            ax_m.scatter(sxy[s, 0], sxy[s, 1], s=sz,
                         marker="s", c="#cc3344", edgecolor="black",
                         linewidths=0.6, alpha=0.85, zorder=10)
            ax_m.text(sxy[s, 0], sxy[s, 1] + 0.0006,
                      f"-{out_total[s]}", fontsize=7, ha="center",
                      color="#cc3344", fontweight="bold")
    # Top destinations (blue)
    if inb_total.max() > 0:
        i_max = inb_total.max()
        for s in top_in:
            sz = 30 + 250 * (inb_total[s] / i_max)
            ax_m.scatter(sxy[s, 0], sxy[s, 1], s=sz,
                         marker="^", c="#3388cc", edgecolor="black",
                         linewidths=0.6, alpha=0.85, zorder=10)
            ax_m.text(sxy[s, 0], sxy[s, 1] - 0.0009,
                      f"+{inb_total[s]}", fontsize=7, ha="center",
                      color="#3388cc", fontweight="bold")

    # Business circle markers
    for c in NANSHAN_CIRCLES:
        ax_m.scatter(c.lon, c.lat, s=160, marker="*",
                     c="#ffd633", edgecolor="black", linewidths=0.6,
                     zorder=11)

    # All stations dim
    ax_m.scatter(sxy[:, 0], sxy[:, 1], s=4, c="white",
                 edgecolor="black", linewidths=0.2, alpha=0.5,
                 zorder=4)

    lon0, lat0, lon1, lat1 = bbox
    lat_mid = 0.5 * (lat0 + lat1)
    ax_m.set_xlim(lon0, lon1); ax_m.set_ylim(lat0, lat1)
    ax_m.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))
    ax_m.set_xticks([]); ax_m.set_yticks([])
    ax_m.set_title("Geographic deadhead flow  "
                   "■ red = senders (surplus), ▲ blue = receivers (deficit), "
                   "★ yellow = business circles")

    fig.suptitle(f"Rebalancing / deadhead repositioning — "
                 f"{n_total:,d} moves over the simulated 4h window",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
