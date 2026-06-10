"""Raw trace experiment — no path smoothing, no phase-3 descent ramp,
no z-endpoint relax. Pure gradient-descent on T, plus straight-line
phase-1 climb-out. This isolates the wave-routing's intrinsic
geometry: longer wavelength SHOULD give smoother paths per paper §2.9.

Usage:
    python exp2_aircraft_raw.py short    # τ=6, σ=4
    python exp2_aircraft_raw.py long     # τ=20, σ=10
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_THIS / "_jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import exp2_aircraft as EA

CONFIGS = {
    "short": {"tau": 6,  "sigma_xy": 4.0,
              "label": "RAW short λ (τ=6, σ_xy=4)"},
    "long":  {"tau": 20, "sigma_xy": 10.0,
              "label": "RAW long λ (τ=20, σ_xy=10)"},
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "short"
    cfg = CONFIGS[name]
    print(f"========== {cfg['label']} ==========")
    EA.AC_PULSE_TAU = cfg["tau"]
    EA.AC_SOURCE_SIGMA_CELLS = cfg["sigma_xy"]

    # Disable path smoothing — return path verbatim
    def no_smooth(path_xyz, *args, **kwargs):
        return path_xyz.copy()
    EA.smooth_path = no_smooth

    # Use peak-normalised source so amplitude survives at far cells
    def gaussian_source_3d_peak(shape, idx, sigma_cells):
        nx, ny, nz = shape
        ix = jnp.arange(nx)[:, None, None]
        iy = jnp.arange(ny)[None, :, None]
        iz = jnp.arange(nz)[None, None, :]
        r2 = ((ix - idx[0]) ** 2 + (iy - idx[1]) ** 2 + (iz - idx[2]) ** 2)
        g = jnp.exp(-0.5 * r2 / (sigma_cells ** 2))
        return (g / g.max()).astype(jnp.float32)
    EA.gaussian_source_3d = gaussian_source_3d_peak

    out_png = EA.OUT / f"aircraft_raw_{name}_lambda.png"
    out_npz = EA.OUT / f"aircraft_raw_{name}_lambda.npz"
    ORIG_savefig = plt.Figure.savefig

    def patched_savefig(self, fname, *a, **kw):
        if str(fname).endswith(".png"):
            fname = str(out_png)
        return ORIG_savefig(self, fname, *a, **kw)
    plt.Figure.savefig = patched_savefig

    ORIG_savez = np.savez_compressed

    def patched_savez(file, *a, **kw):
        if str(file).endswith(".npz"):
            file = str(out_npz)
        return ORIG_savez(file, *a, **kw)
    np.savez_compressed = patched_savez

    EA.main()


if __name__ == "__main__":
    main()
