"""Nanshan business circles (商圈) inside our domain bbox.

The domain is the upper-right corner of Nanshan district:
    bbox = (113.900, 22.520) — (113.965, 22.585)  ~6.6km × 7.2km

We hard-code the 7 major commercial / office clusters that fall inside
this bbox, along with their type (which drives the demand profile),
geographic centre, characteristic radius, and a relative peak-load
weight.

Order demand model
------------------
Each circle generates Poisson-distributed orders over time. The hourly
rate is:
    λ(t) = base · weight · profile(t, type)
where profile(t, type) is a 24-hour shape (e.g. lunch-heavy for offices,
evening-heavy for entertainment, balanced for malls).

Customer (delivery) destinations
--------------------------------
Each of the 118 stations gets a `residential_score` ∈ [0, 1]:
    1.0 — pure residential
    0.0 — inside a business circle
Customers are sampled from this distribution.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOMAIN = ROOT / "wave_test_large" / "domain.npz"

# Demand profile types (24h, evaluated 0.0-24.0)
ProfileT = Literal["office", "mall", "tech", "entertainment"]


@dataclass(frozen=True)
class BusinessCircle:
    name: str          # human-readable name
    name_en: str
    lon: float
    lat: float
    radius_m: float    # characteristic radius (for visualisation + match)
    weight: float      # relative order rate (1.0 = baseline)
    profile: ProfileT


# ---- 7 major business circles inside our bbox ----------------------------
NANSHAN_CIRCLES = [
    BusinessCircle(
        name="海岸城 / Coastal City",
        name_en="coastal_city",
        lon=113.948, lat=22.534, radius_m=300, weight=1.5, profile="mall"),
    BusinessCircle(
        name="腾讯滨海 / 软件园 (Tencent HQ + Software Park)",
        name_en="tencent_softpark",
        lon=113.937, lat=22.541, radius_m=350, weight=2.2, profile="office"),
    BusinessCircle(
        name="万象天地 / MixC World",
        name_en="mixc_world",
        lon=113.957, lat=22.537, radius_m=280, weight=1.4, profile="mall"),
    BusinessCircle(
        name="欢乐海岸 / OCT Bay",
        name_en="oct_bay",
        lon=113.964, lat=22.527, radius_m=320, weight=1.0,
        profile="entertainment"),
    BusinessCircle(
        name="南山中心商业区 / Nanshan Central",
        name_en="nanshan_central",
        lon=113.928, lat=22.535, radius_m=280, weight=1.1, profile="office"),
    BusinessCircle(
        name="粤海科技园北 / Yuehai Tech Park (North)",
        name_en="yuehai_north",
        lon=113.928, lat=22.555, radius_m=350, weight=1.6, profile="tech"),
    BusinessCircle(
        name="天利中央广场 / Tianli Plaza (Houhai office)",
        name_en="tianli_plaza",
        lon=113.961, lat=22.541, radius_m=240, weight=1.0, profile="office"),
]


def demand_profile(t_h: float, profile: ProfileT) -> float:
    """24-hour demand multiplier (mean ~0.5; peaks ~1.5).

    Sum of two Gaussians (lunch + dinner) per type, plus background.
    """
    def gauss(t, mu, sig): return np.exp(-0.5 * ((t - mu) / sig) ** 2)
    bg = 0.10
    if profile == "office":
        return bg + 1.4 * gauss(t_h, 12.0, 0.8) + 0.5 * gauss(t_h, 19.0, 1.0)
    if profile == "mall":
        return bg + 0.7 * gauss(t_h, 12.5, 1.2) + 1.2 * gauss(t_h, 18.5, 1.3)
    if profile == "tech":
        return bg + 1.3 * gauss(t_h, 12.0, 0.7) + 0.8 * gauss(t_h, 19.5, 1.0) \
             + 0.4 * gauss(t_h, 22.5, 1.2)   # late-night coding snacks
    if profile == "entertainment":
        return bg + 0.4 * gauss(t_h, 12.5, 1.5) + 1.5 * gauss(t_h, 19.0, 1.5) \
             + 0.6 * gauss(t_h, 21.5, 1.0)
    return bg


# ---- Project to local metres + match each circle to its nearest station --

def _lonlat_to_local_m(lon, lat, lon0, lat0):
    LAT_M = 111000.0
    LON_M = LAT_M * np.cos(np.deg2rad(lat0))
    return ((np.asarray(lon) - lon0) * LON_M,
            (np.asarray(lat) - lat0) * LAT_M)


def load_circles_with_stations():
    """Returns dict with:
        circles_xy    : (K, 2) — (x, y) in metres
        circle_meta   : list of BusinessCircle
        station_xy    : (N, 2) m
        station_lonlat: (N, 2)
        circle_to_station : (K,) int — index of station nearest each circle
        residential_score : (N,) float in [0, 1]
            high = residential / good customer drop-off
            low  = inside / next to a circle
    """
    z = np.load(DOMAIN, allow_pickle=False)
    bbox = z["domain_bbox_lonlat"]
    lon0, lat0 = float(bbox[0]), float(bbox[1])
    station_xy = z["stations_xy_m"][:, :2]
    station_lonlat = z["station_lonlat"]
    n_st = len(station_xy)

    K = len(NANSHAN_CIRCLES)
    cx, cy = _lonlat_to_local_m(
        np.array([c.lon for c in NANSHAN_CIRCLES]),
        np.array([c.lat for c in NANSHAN_CIRCLES]),
        lon0, lat0)
    circles_xy = np.column_stack([cx, cy])

    # Match each circle to its nearest station
    diff = station_xy[None, :, :] - circles_xy[:, None, :]
    d_st_circle = np.linalg.norm(diff, axis=2)              # (K, N)
    circle_to_station = d_st_circle.argmin(axis=1).astype(int)

    # Residential score per station: 1 - max(circle proximity)
    # If any circle is within 1.5·radius of a station, score → 0
    score = np.ones(n_st)
    for k, c in enumerate(NANSHAN_CIRCLES):
        d = d_st_circle[k]                                 # (N,)
        prox = np.exp(-(d / (1.5 * c.radius_m)) ** 2)
        score = np.minimum(score, 1.0 - 0.95 * prox)
    score = np.clip(score, 0.05, 1.0)

    return dict(
        circles_xy=circles_xy,
        circle_meta=NANSHAN_CIRCLES,
        station_xy=station_xy,
        station_lonlat=station_lonlat,
        circle_to_station=circle_to_station,
        residential_score=score,
        bbox_lonlat=bbox,
        n_st=n_st,
    )


if __name__ == "__main__":
    info = load_circles_with_stations()
    print(f"[circles] {len(info['circle_meta'])} business circles, "
          f"{info['n_st']} stations")
    for k, c in enumerate(info["circle_meta"]):
        st = info["circle_to_station"][k]
        print(f"  {k}: {c.name_en:18s}  → station {st:3d}  "
              f"(weight {c.weight:.1f}, profile {c.profile})")
    rs = info["residential_score"]
    print(f"[residential] score range [{rs.min():.2f}, {rs.max():.2f}], "
          f"top-5 residential stations: "
          f"{np.argsort(-rs)[:5].tolist()}")

    # Show 24h demand by profile
    t = np.arange(0, 24, 0.5)
    for prof in ["office", "mall", "tech", "entertainment"]:
        peak_t = t[np.argmax([demand_profile(ti, prof) for ti in t])]
        peak = max(demand_profile(ti, prof) for ti in t)
        print(f"[demand] {prof:14s} peak at {peak_t:4.1f}h "
              f"(λ_mult={peak:.2f})")
