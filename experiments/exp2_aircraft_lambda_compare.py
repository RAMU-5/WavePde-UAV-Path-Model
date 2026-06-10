"""Controlled comparison: aircraft routing with SHORT vs LONG wavelength,
both using phase-3 descent ramp + relaxed z endpoints. Isolates the
pure wavelength effect on path geometry (xy AND z).

Two figures:
  visualization/3d_aircraft/aircraft_short_lambda.png
  visualization/3d_aircraft/aircraft_long_lambda_only.png
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil

_THIS = Path(__file__).resolve().parent
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_THIS.parent / "_jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import exp2_aircraft as EA
import exp2_aircraft_long_lambda as LL  # imports & monkeypatches everything

# ===== Sweep configurations (always with phase-3 ramp + relaxed z) =====
CONFIGS = [
    {"name": "short_lambda", "tau": 6,  "sigma_xy": 4.0,
     "label": "short λ (τ=6, σ_xy=4)"},
    {"name": "long_lambda_only", "tau": 20, "sigma_xy": 10.0,
     "label": "long λ (τ=20, σ_xy=10)"},
]

OUT_DIR = EA.OUT


def main():
    summary = []
    for cfg in CONFIGS:
        print(f"\n========== {cfg['label']} ==========")
        EA.AC_PULSE_TAU = cfg["tau"]
        EA.AC_SOURCE_SIGMA_CELLS = cfg["sigma_xy"]
        # path smoothing kept FIXED at 8 km — same for both runs so
        # the difference is only from wavelength.
        EA.PATH_SMOOTH_SIGMA_M = 8000.0

        # Patches from LL are already in place
        # (gaussian_source_3d, smooth_path, trace_aircraft).

        # Redirect savefig + savez to per-config filenames.
        out_png = OUT_DIR / f"aircraft_{cfg['name']}.png"
        out_npz = OUT_DIR / f"aircraft_{cfg['name']}.npz"
        ORIG_savefig = plt.Figure.savefig

        def make_savefig(out_png_path):
            def _patched(self, fname, *a, **kw):
                fname = str(fname)
                if fname.endswith(".png"):
                    fname = str(out_png_path)
                return ORIG_savefig(self, fname, *a, **kw)
            return _patched

        plt.Figure.savefig = make_savefig(out_png)

        ORIG_savez = np.savez_compressed

        def make_savez(out_npz_path):
            def _patched(file, *a, **kw):
                file = str(file)
                if file.endswith(".npz"):
                    file = str(out_npz_path)
                return ORIG_savez(file, *a, **kw)
            return _patched

        np.savez_compressed = make_savez(out_npz)

        EA.main()

        plt.Figure.savefig = ORIG_savefig
        np.savez_compressed = ORIG_savez

        # Collect path stats
        z = np.load(out_npz)
        per_path = []
        for i in range(int(z["n_paths"])):
            p = z[f"path_{i}_xyz"] / 1000
            dz = np.diff(p[:, 2])
            # xy curvature on the smoothed path (ignore phase 1, which
            # is straight)
            n_p1 = max(2, int(EA.PHASE1_LENGTH_M
                              / (0.3 * min(EA.DX, EA.DY, EA.DZ))))
            p_p2 = p[n_p1 - 1:]   # phase 2 onward
            if p_p2.shape[0] > 5:
                seg = np.diff(p_p2[:, :2], axis=0)
                seg_len = np.linalg.norm(seg, axis=1)
                seg_len = np.maximum(seg_len, 1e-6)
                tan = seg / seg_len[:, None]
                dt = np.diff(tan, axis=0)
                dn = np.linalg.norm(dt, axis=1)
                midlen = 0.5 * (seg_len[:-1] + seg_len[1:])
                kappa_xy = dn / midlen
                k_xy_max = float(kappa_xy.max())
                k_xy_p95 = float(np.percentile(kappa_xy, 95))
            else:
                k_xy_max = 0.0; k_xy_p95 = 0.0
            R_xy_min = 1e-3 / max(k_xy_max, 1e-12)   # km

            # z stats over phase-2
            zs_p2 = p_p2[:, 2]
            dz_p2 = np.diff(zs_p2)
            per_path.append({
                "i": i,
                "z_max_km": float(zs_p2.max()),
                "z_min_km": float(zs_p2.min()),
                "max_abs_dz_m_per_step": float(np.abs(dz_p2).max() * 1000)
                                         if dz_p2.size else 0.0,
                "k_xy_max_per_km": k_xy_max * 1000,    # 1/km
                "k_xy_p95_per_km": k_xy_p95 * 1000,
                "R_xy_min_km": R_xy_min,
            })

        summary.append({"cfg": cfg, "per_path": per_path})

    # Print comparison
    print("\n========== COMPARISON ==========")
    print(f"{'config':<30s} {'path':>4s} "
          f"{'κ_xy_max (1/km)':>16s} {'R_xy_min (km)':>14s} "
          f"{'z_max (km)':>10s} {'max|dz/step| (m)':>16s}")
    for s in summary:
        for pp in s["per_path"]:
            print(f"{s['cfg']['label']:<30s} {pp['i']:>4d} "
                  f"{pp['k_xy_max_per_km']:>16.4f} "
                  f"{pp['R_xy_min_km']:>14.2f} "
                  f"{pp['z_max_km']:>10.2f} "
                  f"{pp['max_abs_dz_m_per_step']:>16.0f}")
        # Median across paths
        kmed = np.median([pp['k_xy_max_per_km'] for pp in s['per_path']])
        rmed = np.median([pp['R_xy_min_km'] for pp in s['per_path']])
        zmed = np.median([pp['z_max_km'] for pp in s['per_path']])
        dzmed = np.median([pp['max_abs_dz_m_per_step']
                          for pp in s['per_path']])
        print(f"{s['cfg']['label']:<30s} {'med':>4s} "
              f"{kmed:>16.4f} {rmed:>14.2f} "
              f"{zmed:>10.2f} {dzmed:>16.0f}")
        print()


if __name__ == "__main__":
    main()
