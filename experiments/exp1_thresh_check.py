"""Diagnostic: is the heavy positive tail of the 5 m optimality gap a
min_peak_amp detection artifact or real wavelength physics?

Re-solves a few sources on domain_5m.npz with the production config
EXCEPT min_peak_amp lowered 1e-6 -> 1e-9, then prints T at all stations
next to the production pair_matrix_5m values. If the bad pairs collapse
to the ground-truth times, the production threshold misses weak first
arrivals at 5 m (amplitudes are (5/50)^3 = 1e-3 of the 50 m runs the
threshold was tuned on); if they stay, the wave genuinely routes around
sub-wavelength passages (E4.1 territory).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from experiments.exp1_dijkstra_gt import nudge_stations  # noqa: E402
import model.wave_api as wave_api  # noqa: E402
from model.wave_api import WaveConfig, build_static_fields  # noqa: E402

SOURCES = [80, 70, 10]
MIN_PEAK = 1e-9

z = np.load(ROOT / "input" / "domain_5m.npz", allow_pickle=False)
mask = z["mask"]
dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
stations = z["stations_xy_m"]
ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)

pm = np.load(ROOT / "output" / "wave" / "pair_matrix_5m.npz")
Tw = pm["T_matrix"]

cfg = WaveConfig(min_peak_amp=MIN_PEAK)
static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
out = {}
for s in SOURCES:
    T = wave_api.solve_forward(mask, dx, dy, dz,
                               (int(ix[s]), int(iy[s]), int(iz[s])),
                               cfg=cfg, static=static)
    tv = T[ix, iy, iz]
    out[f"T_{s}"] = tv
    diff = Tw[s] - tv
    big = np.where(np.isfinite(diff) & (diff > 5.0))[0]
    print(f"[src {s}] stations where production T is >5s LATER "
          f"than 1e-9-threshold T: {len(big)}")
    for b in big[:15]:
        print(f"   dst {b:3d}: prod {Tw[s, b]:7.1f}  lowthr {tv[b]:7.1f}  "
              f"diff {diff[b]:+7.1f}")
np.savez_compressed(ROOT / "output" / "runs" / "exp1_dijkstra" /
                    "thresh_check.npz", st_ix=ix, st_iy=iy, st_iz=iz, **out)
print("[done]")
