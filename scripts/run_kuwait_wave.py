"""Single wind-aware wave solve on the Kuwait CBD domain.

There is no airflow/CFD surrogate in this repo; the "wind" is a uniform
coefficient field fed to the convected wave PDE. This runs one forward
solve from a chosen source on input/kuwait_cbd_1024.npz and saves the
arrival-time field T(x).

Usage (on a GPU node):
    python scripts/run_kuwait_wave.py --wind 5 0 0
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = str(ROOT / "_jax_cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
sys.path.insert(0, str(ROOT / "model"))
import jax  # noqa: E402
import wave_api  # noqa: E402
from wave_api import WaveConfig, build_static_fields  # noqa: E402


def nudge_free(mask, ix, iy, iz):
    """Move (ix,iy,iz) out of a building: horizontal square shells, then up."""
    nx, ny, nz = mask.shape
    if not mask[ix, iy, iz]:
        return ix, iy, iz
    for r in range(1, 40):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                a, b = ix + di, iy + dj
                if 0 < a < nx - 1 and 0 < b < ny - 1 and not mask[a, b, iz]:
                    return a, b, iz
    while iz < nz - 1 and mask[ix, iy, iz]:
        iz += 1
    return ix, iy, iz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="kuwait_cbd_1024.npz")
    p.add_argument("--wind", type=float, nargs=3, default=[5.0, 0.0, 0.0],
                   metavar=("U", "V", "W"), help="uniform wind vector (m/s)")
    p.add_argument("--src", type=int, nargs=3, default=None,
                   metavar=("IX", "IY", "IZ"),
                   help="source cell idx; default = domain centre at ground")
    p.add_argument("--out", default="kuwait_T.npz")
    args = p.parse_args()

    print(f"[jax] devices: {jax.devices()}", flush=True)
    z = np.load(ROOT / "input" / args.domain, allow_pickle=False)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    print(f"[domain] {args.domain}  grid {nx}x{ny}x{nz}  "
          f"dx={dx} dy={dy} dz={dz}  fill {mask.mean()*100:.2f}%", flush=True)

    if args.src is None:
        ix, iy, iz = nx // 2, ny // 2, max(1, int(round(5.0 / dz)))
    else:
        ix, iy, iz = args.src
    ix, iy, iz = nudge_free(mask, ix, iy, iz)
    print(f"[src] cell ({ix},{iy},{iz})  = "
          f"({ix*dx:.0f}m, {iy*dy:.0f}m, {iz*dz:.0f}m)", flush=True)

    min_peak = 1e-6 * (dx / 50.0) ** 3
    cfg = WaveConfig(wind=tuple(args.wind), min_peak_amp=min_peak)
    print(f"[cfg] wind={cfg.wind} m/s  c0={cfg.c0}  n_steps={cfg.n_steps}  "
          f"min_peak_amp={min_peak:.2e}", flush=True)

    t0 = time.time()
    static = build_static_fields(mask, dx, dy, dz, cfg=cfg)
    T = wave_api.solve_forward(mask, dx, dy, dz, (ix, iy, iz),
                               cfg=cfg, static=static)
    dt_solve = time.time() - t0

    finite = np.isfinite(T)
    print(f"[solve] {dt_solve:.1f}s  reached {finite.mean()*100:.1f}% of cells  "
          f"T[finite] max={T[finite].max():.1f}s", flush=True)

    out = ROOT / "output" / "wave" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, T_full=T.astype(np.float32),
        src_idx=np.array([ix, iy, iz]), wind=np.array(cfg.wind),
        dx=dx, dy=dy, dz=dz)
    print(f"[saved] {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
