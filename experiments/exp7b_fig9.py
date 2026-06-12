"""Re-trace the Fig.9 multirotor path with the normalised-convolution
smoothed tracer (exp5 lesson: raw -grad T descent on the noisy 5 m
field stalls — exp7's first attempt gave truncated_187m). One forward
solve from station 50, smooth, trace to the same destination 49,
overwrite output/runs/exp7_efficiency/fig9_multirotor.npz.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "experiments"))

import wave_api  # noqa: E402
from wave_api import WaveConfig  # noqa: E402
from solve_coupled import trace_path  # noqa: E402
from exp1_dijkstra_gt import polyline_time  # noqa: E402
from exp5_broadband import path_collides  # noqa: E402
from exp7_efficiency import production_station_cells, FIG9_SRC  # noqa: E402

OUT = ROOT / "output" / "runs" / "exp7_efficiency" / "fig9_multirotor.npz"
DST = 49        # exp7's pick: farthest valid band partner of src 50


def main():
    z = np.load(ROOT / "input" / "domain_5m.npz")
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    cfg = WaveConfig(min_peak_amp=1e-6 * (dx / 50.0) ** 3)
    ix, iy, iz = production_station_cells(mask, z["stations_xy_m"],
                                          dx, dy, dz)
    t0 = time.time()
    T = wave_api.solve_forward(mask, dx, dy, dz,
                               (int(ix[FIG9_SRC]), int(iy[FIG9_SRC]),
                                int(iz[FIG9_SRC])), cfg=cfg)
    print(f"[solve] {time.time()-t0:.0f}s", flush=True)

    Tt = np.asarray(T, dtype=np.float64)
    fin = np.isfinite(Tt)
    num = gaussian_filter(np.where(fin, Tt, 0.0), 2.0)
    den = gaussian_filter(fin.astype(np.float64), 2.0)
    Tsm = np.where(fin, num / np.maximum(den, 1e-9), Tt[fin].max() + 1e6)

    p0 = np.array([(ix[FIG9_SRC] + .5) * dx, (iy[FIG9_SRC] + .5) * dy,
                   (iz[FIG9_SRC] + .5) * dz])
    p1 = np.array([(ix[DST] + .5) * dx, (iy[DST] + .5) * dy,
                   (iz[DST] + .5) * dz])
    path, status = trace_path(Tsm, p0, p1, dx, dy, dz)
    coll = path_collides(path, mask, dx, dy, dz)
    pt = polyline_time(path.astype(np.float64), dz) \
        if status == "reached" and not coll else np.inf
    print(f"[trace] {status} collides={coll} path_time={pt:.1f}s "
          f"T_at_dst={T[ix[DST], iy[DST], iz[DST]]:.1f}s", flush=True)

    iz_slice = 10
    np.savez_compressed(
        OUT, T_slice=T[:, :, iz_slice], mask_slice=mask[:, :, iz_slice],
        path=path, path_status=status, path_collides=coll,
        path_time_s=pt, src=FIG9_SRC, dst=DST, dx=dx, dz=dz,
        z_m=(iz_slice + .5) * dz,
        st_xyz=np.stack([(ix + .5) * dx, (iy + .5) * dy,
                         (iz + .5) * dz], 1))
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
