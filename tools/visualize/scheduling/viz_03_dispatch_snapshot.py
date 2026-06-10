"""Viz 3 (scheduling): map snapshot during the simulation.

Picks the busiest 5-min window and shows:
  - Building footprints
  - Business circles (coloured discs)
  - All 118 stations as small dots
  - Active UAV trajectories (in-flight at the snapshot time):
      • magenta dashed line: UAV→pickup
      • orange solid line:   pickup→customer
  - Pending orders not yet assigned: red triangles at pickup, blue dots at
    customer
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle as MplCircle, Patch

from business_circles import (load_circles_with_stations,
                              NANSHAN_CIRCLES)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIM = ROOT / "output" / "figures" / "scheduling_system" / "sim_log.npz"
OSM_BUILDINGS = ROOT / "shenzhen-osm" / "nanshan" / "buildings.geojson"
OUT = ROOT / "output" / "figures" / "scheduling_system" / "03_dispatch_snapshot.png"

CIRCLE_COLORS = {"office": "#5fc8ff", "mall": "#ff7755",
                 "tech": "#3affb3", "entertainment": "#ffaa44"}


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
    bbox = info["bbox_lonlat"]
    lon0, lat0, lon1, lat1 = bbox
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0))

    station_lonlat = info["station_lonlat"]
    log_busy = z["log_busy"]
    tick_sec = float(z["tick_sec"])
    t_start = float(z["sim_t_start"])

    # Pick the tick with most busy UAVs (peak)
    peak_tick = int(np.argmax(log_busy))
    t_peak_sec = peak_tick * tick_sec
    t_peak_h = t_start + t_peak_sec / 3600.0
    print(f"[viz3-sched] peak at tick {peak_tick} (t={t_peak_h:.2f}h, "
          f"{log_busy[peak_tick]} UAVs busy)")

    # Find orders in flight at the snapshot
    o_pickup = z["order_pickup_station"]
    o_customer = z["order_customer_station"]
    o_created = z["order_created_t"]
    o_assigned = z["order_assigned_t"]
    o_pickup_t = z["order_pickup_t"]
    o_delivered = z["order_delivered_t"]
    n_orders = len(o_pickup)

    # Vectorised classification at t_peak_sec
    created_before = o_created <= t_peak_sec
    delivered_before = (o_delivered >= 0) & (o_delivered <= t_peak_sec)
    active = created_before & ~delivered_before
    assigned_before = (o_assigned >= 0) & (o_assigned <= t_peak_sec)
    pickup_before = (o_pickup_t >= 0) & (o_pickup_t <= t_peak_sec)
    pending_mask = active & ~assigned_before
    to_pickup_mask = active & assigned_before & ~pickup_before
    to_dest_mask = active & pickup_before & ~delivered_before
    pending_now = np.where(pending_mask)[0]
    in_flight_to_pickup = np.where(to_pickup_mask)[0]
    in_flight_to_dest = np.where(to_dest_mask)[0]
    # Cap displayed lines (drawing 30k segments is slow + visually unreadable)
    MAX_LINES = 800
    if len(in_flight_to_pickup) > MAX_LINES:
        in_flight_to_pickup = np.random.default_rng(0).choice(
            in_flight_to_pickup, MAX_LINES, replace=False)
    if len(in_flight_to_dest) > MAX_LINES:
        in_flight_to_dest = np.random.default_rng(1).choice(
            in_flight_to_dest, MAX_LINES, replace=False)
    if len(pending_now) > 600:
        pending_now = np.random.default_rng(2).choice(pending_now, 600,
                                                       replace=False)
    print(f"  pending (waiting) ......... {len(pending_now)}")
    print(f"  UAV → pickup .............. {len(in_flight_to_pickup)}")
    print(f"  pickup → customer ......... {len(in_flight_to_dest)}")

    polys = _load_buildings(bbox)

    fig, ax = plt.subplots(figsize=(15, 14))
    ax.set_facecolor("#f4f1ec")

    pc = PolyCollection(polys, facecolor="#888888",
                        edgecolor="black", linewidths=0.06, alpha=0.4)
    ax.add_collection(pc)

    # Stations small dots
    ax.scatter(station_lonlat[:, 0], station_lonlat[:, 1], s=10,
               c="white", edgecolor="black", linewidths=0.3,
               zorder=4)

    # Business circles
    for k, c in enumerate(NANSHAN_CIRCLES):
        col = CIRCLE_COLORS[c.profile]
        r_deg = c.radius_m / LON_M
        ax.add_patch(MplCircle((c.lon, c.lat), r_deg, color=col,
                               alpha=0.45, zorder=5))
        ax.text(c.lon, c.lat + r_deg + 0.0006, c.name_en,
                fontsize=8, ha="center", va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=col, alpha=0.85,
                          edgecolor="black", linewidth=0.5))

    # In-flight legs
    segs_pickup = [[station_lonlat[o_pickup[i]],
                    station_lonlat[o_customer[i]]]
                   for i in in_flight_to_pickup]
    # For "going to pickup", we don't know which UAV station; approximate by
    # pickup→customer line (the order's full route)
    if segs_pickup:
        lc = LineCollection(segs_pickup, colors="#ff44dd",
                            linestyles="dashed", linewidths=1.4, alpha=0.7,
                            zorder=7,
                            label=f"UAV → pickup ({len(segs_pickup)})")
        ax.add_collection(lc)

    segs_dest = [[station_lonlat[o_pickup[i]],
                  station_lonlat[o_customer[i]]]
                 for i in in_flight_to_dest]
    if segs_dest:
        lc2 = LineCollection(segs_dest, colors="#ff8833",
                             linewidths=1.8, alpha=0.85, zorder=8,
                             label=f"pickup → customer ({len(segs_dest)})")
        ax.add_collection(lc2)

    # Pending pickups
    if len(pending_now) > 0:
        p_xy = station_lonlat[o_pickup[pending_now]]
        ax.scatter(p_xy[:, 0], p_xy[:, 1], s=80, marker="^",
                   facecolor="red", edgecolor="black", linewidths=0.6,
                   zorder=10, label=f"pending pickups ({len(pending_now)})")
        c_xy = station_lonlat[o_customer[pending_now]]
        ax.scatter(c_xy[:, 0], c_xy[:, 1], s=40, marker="o",
                   facecolor="#5588ff", edgecolor="black", linewidths=0.4,
                   zorder=9, label="pending customers")

    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    lat_mid = 0.5 * (lat0 + lat1)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(
        f"Dispatch snapshot at {t_peak_h:.2f}h  "
        f"(peak load: {int(log_busy[peak_tick])} UAVs in flight, "
        f"{len(pending_now)} orders pending)",
        fontsize=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
