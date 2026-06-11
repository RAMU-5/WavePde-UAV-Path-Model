"""Re-solve the full 118-source 5 m wave matrix with the corrected peak
threshold (E2.1 follow-up).

The production pair_matrix_5m.npz was computed with min_peak_amp = 1e-6
— a value tuned on the 50 m grid. Source amplitudes scale with dx^3
(sum=1 cell normalisation), so at 5 m every far/shadowed first arrival
sits ~1000x lower and the detector waited for a later, stronger packet:
the ground-truth comparison showed +5...+230% phantom gaps on 71-86
stations per source, which collapse to |gap| p95 < 7% with the scaled
threshold 1e-6 * (dx/50)^3 = 1e-9 (validated in exp1_thresh_check).

Caveat of the lower threshold: cells within ~9 cells (45 m) of the
source can latch onto the source's own spatial tail; station pairs are
>= 700 m apart, so the matrix is unaffected.

Writes output/wave/pair_matrix_5m_v2.npz (same layout as the aggregate
script; the production file is kept for the before/after comparison).
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

from exp1_dijkstra_gt import nudge_stations  # noqa: E402
import wave_api  # noqa: E402
from wave_api import WaveConfig, build_static_fields  # noqa: E402

OUT = ROOT / "output" / "wave" / "pair_matrix_5m_v2.npz"
CHK = ROOT / "output" / "wave" / "pair_matrix_5m_v2_chk.npz"
LO_DIST_M, HI_DIST_M = 700.0, 3000.0

z = np.load(ROOT / "input" / "domain_5m.npz", allow_pickle=False)
mask = z["mask"]
dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
stations = z["stations_xy_m"]
n_st = len(stations)
ix, iy, iz = nudge_stations(mask, stations, dx, dy, dz)

MIN_PEAK = 1e-6 * (dx / 50.0) ** 3
cfg = WaveConfig(min_peak_amp=MIN_PEAK)
print(f"[cfg] min_peak_amp = {MIN_PEAK:.2e}, n_steps={cfg.n_steps}",
      flush=True)

if CHK.exists():
    c = np.load(CHK)
    T_at = c["T_at_stations"]
    done = c["done"]
    print(f"[resume] {done.sum()}/{n_st}", flush=True)
else:
    T_at = np.full((n_st, n_st), np.inf, dtype=np.float32)
    done = np.zeros(n_st, dtype=bool)

static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
t0 = time.time()
for s in range(n_st):
    if done[s]:
        continue
    t1 = time.time()
    T = wave_api.solve_forward(mask, dx, dy, dz,
                               (int(ix[s]), int(iy[s]), int(iz[s])),
                               cfg=cfg, static=static)
    T_at[s, :] = T[ix, iy, iz]
    done[s] = True
    np.savez_compressed(CHK, T_at_stations=T_at, done=done)
    print(f"[src {s:3d}] {time.time()-t1:5.1f}s  "
          f"finite {np.isfinite(T_at[s]).sum()}/{n_st}  "
          f"elapsed {(time.time()-t0)/60:.1f}min", flush=True)

diff = stations[:, None, :2] - stations[None, :, :2]
distances_m = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
np.fill_diagonal(distances_m, 0.0)
valid = ((distances_m >= LO_DIST_M) & (distances_m <= HI_DIST_M)
         & np.isfinite(T_at))
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                     capture_output=True, text=True).stdout.strip()
np.savez_compressed(
    OUT, T_matrix=T_at, valid_mask=valid, distances_m=distances_m,
    stations_xy_m=stations, station_lonlat=z["station_lonlat"],
    n_pairs_valid=np.int32(valid.sum()),
    n_pairs_with_T=np.int32(np.isfinite(T_at).sum()),
    min_peak_amp=MIN_PEAK, git_sha=sha)
print(f"[done] -> {OUT}", flush=True)
