"""Read all T_source_NNN.npz produced by the SLURM array, build the
118×118 travel-time matrix M where M[i, j] = travel time from station i
to station j. Apply the [700m, 3000m] distance filter.

Outputs:
    pair_matrix.npz with:
        T_matrix      : (N, N) float32  — full source→dest matrix
        valid_mask    : (N, N) bool     — pairs satisfying 700m≤d≤3000m
        distances_m   : (N, N) float32  — Euclidean station distances
        n_pairs_valid : int
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR.parent / "output" / "wave"
LO_DIST_M, HI_DIST_M = 700.0, 3000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="domain.npz")
    p.add_argument("--prefix", default="T_source_",
                   help="Filename prefix; default reads T_source_NNN.npz; "
                        "use 'T_source_5m_' for 5m run.")
    p.add_argument("--out", default="pair_matrix.npz")
    args = p.parse_args()

    z = np.load(THIS_DIR.parent / "input" / args.domain, allow_pickle=False)
    stations_xy_m = z["stations_xy_m"]
    n_st = len(stations_xy_m)
    print(f"[aggregate] expecting {n_st} sources from {args.prefix}*.npz")

    T_matrix = np.full((n_st, n_st), np.inf, dtype=np.float32)
    found = []
    missing = []
    for i in range(n_st):
        # Try with and without zero-padding for compatibility with the
        # 5m array script (which uses unpadded SLURM_ARRAY_TASK_ID).
        path = OUT_DIR / f"{args.prefix}{i:03d}.npz"
        if not path.exists():
            path = OUT_DIR / f"{args.prefix}{i}.npz"
        if not path.exists():
            missing.append(i)
            continue
        a = np.load(path)
        T_matrix[i, :] = a["T_at_stations"]
        found.append(i)
    print(f"[aggregate] found {len(found)}/{n_st}, missing {len(missing)}")
    if missing:
        print(f"[aggregate] missing source idxs: {missing[:20]}"
              + (" ..." if len(missing) > 20 else ""))

    # Pairwise distances
    diff = stations_xy_m[:, None, :2] - stations_xy_m[None, :, :2]
    distances_m = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    np.fill_diagonal(distances_m, 0.0)

    # Valid pair mask: distance in [700m, 3000m]
    valid_mask = (distances_m >= LO_DIST_M) & (distances_m <= HI_DIST_M)
    n_valid = int(valid_mask.sum())
    n_finite = int(np.isfinite(T_matrix[valid_mask]).sum())
    print(f"[aggregate] {n_valid} pairs in [{LO_DIST_M:.0f}m, "
          f"{HI_DIST_M:.0f}m]  ({n_finite}/{n_valid} have finite T = "
          f"{n_finite/max(n_valid,1)*100:.1f}%)")

    # Quick stats on the valid travel times
    finite_T = T_matrix[valid_mask & np.isfinite(T_matrix)]
    if finite_T.size:
        print(f"[aggregate] travel-time min/median/max = "
              f"{finite_T.min():.1f}s / {np.median(finite_T):.1f}s / "
              f"{finite_T.max():.1f}s")
        # vs. straight-line / 15 m/s lower bound
        d_finite = distances_m[valid_mask & np.isfinite(T_matrix)]
        T_min_theory = d_finite / 15.0
        ratio = finite_T / np.maximum(T_min_theory, 1e-6)
        print(f"[aggregate] T / (d/15m·s⁻¹) ratio: "
              f"median={np.median(ratio):.2f}, "
              f"p95={np.percentile(ratio, 95):.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / args.out
    np.savez_compressed(
        out,
        T_matrix=T_matrix,
        valid_mask=valid_mask,
        distances_m=distances_m,
        stations_xy_m=stations_xy_m,
        station_lonlat=z["station_lonlat"],
        n_pairs_valid=np.int32(n_valid),
        n_pairs_with_T=np.int32(n_finite),
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
