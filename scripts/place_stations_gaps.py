"""Re-place N stations in the OPEN GAPS BETWEEN buildings (not on top of
building clusters), embedded in the urban fabric.

Method:
  B    = 2D building footprint (mask.any(z))
  Dfree= distance_transform_edt(~B)  -> clearance (cells) from nearest
         building for every free cell
  urban= B dilated by ~URBAN_R m     -> "near some building"
  candidates = free cells that are (a) inside `urban`, and
               (b) have clearance in [CLR_LO, CLR_HI] cells
               => genuine streets/courtyards between buildings, off the
                  walls but not out in the empty padding.
  Weight each candidate by local building density (buildings within a
  DENS_R m window) and run weighted k-means(N); snap each centroid to
  the nearest candidate cell so every station provably sits in a gap.

Outputs input/kuwait_stations_v2.npz (+ json + placement PNG).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from scipy.ndimage import (distance_transform_edt, binary_dilation,
                           uniform_filter, generate_binary_structure,
                           iterate_structure)
from scipy.cluster.vq import kmeans2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STATION_Z_M = 5.0
LAT_M = 111000.0

URBAN_R_M = 150.0    # a gap must have buildings within this radius
DENS_R_M = 150.0     # density window for weighting
CLR_LO_M = 12.0      # min clearance from buildings (off the walls)
CLR_HI_M = 60.0      # max clearance (still a gap, not open desert)


def disk(r_cells):
    s = iterate_structure(generate_binary_structure(2, 1), int(r_cells))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="kuwait_cbd_1024.npz")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="kuwait_stations_v2.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    z = dict(np.load(ROOT / "input" / args.domain, allow_pickle=False))
    mask = z["mask"]; dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    lon0, lat0, lon1, lat1 = z["domain_bbox_lonlat"]
    LON_M = LAT_M * np.cos(np.deg2rad(0.5 * (lat0 + lat1)))

    B = mask.any(axis=2)                       # (nx,ny) building footprint
    Dfree = distance_transform_edt(~B)         # clearance in cells
    urban = binary_dilation(B, structure=disk(round(URBAN_R_M / dx)))
    clr_m = Dfree * dx
    cand = urban & (~B) & (clr_m >= CLR_LO_M) & (clr_m <= CLR_HI_M)
    print(f"[gap] {int(cand.sum())} candidate gap cells "
          f"(clearance {CLR_LO_M:.0f}-{CLR_HI_M:.0f} m, within "
          f"{URBAN_R_M:.0f} m of buildings)", flush=True)

    dens = uniform_filter(B.astype(np.float32),
                          size=int(round(2 * DENS_R_M / dx)))   # frac built
    ci, cj = np.where(cand)
    w = dens[ci, cj] + 1e-3
    pts = np.column_stack([ci * dx, cj * dy]).astype(float)

    rng = np.random.default_rng(args.seed)
    samp = rng.choice(len(pts), size=min(60000, 10 * len(pts)),
                      p=w / w.sum(), replace=True)
    cent, _ = kmeans2(pts[samp], args.n, seed=args.seed, minit="++",
                      missing="warn")

    # snap each centroid to nearest candidate cell
    st = []
    for cx, cy in cent:
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        k = int(np.argmin(d2))
        st.append((ci[k], cj[k]))
    st = np.array(sorted(set(map(tuple, st))))   # dedup
    sx, sy = st[:, 0], st[:, 1]
    iz = max(1, int(round(STATION_Z_M / dz)))
    n_st = len(st)

    stations_xy_m = np.column_stack([sx * dx, sy * dy,
                                     np.full(n_st, iz * dz)]).astype(float)
    station_lonlat = np.column_stack([lon0 + sx * dx / LON_M,
                                      lat0 + sy * dy / LAT_M]).astype(float)
    clr_st = clr_m[sx, sy]
    on_building = B[sx, sy].sum()
    diff = stations_xy_m[:, None, :2] - stations_xy_m[None, :, :2]
    D = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(D, 0)
    iu = np.triu_indices(n_st, 1)
    band = (D[iu] >= 700) & (D[iu] <= 3000)
    print(f"[stations] placed {n_st}  on-building={on_building}  "
          f"clearance min/med/max = {clr_st.min():.0f}/"
          f"{np.median(clr_st):.0f}/{clr_st.max():.0f} m", flush=True)
    print(f"[stations] pair dist med={np.median(D[iu]):.0f}m  "
          f"valid[700,3000]={band.sum()}/{len(iu[0])}", flush=True)

    z["stations_xy_m"] = stations_xy_m
    z["station_lonlat"] = station_lonlat
    z["station_z_m"] = STATION_Z_M
    np.savez_compressed(ROOT / "input" / args.out, **z)
    rec = dict(domain=args.domain, n_stations=int(n_st),
               placement="inter-building gaps (EDT clearance band)",
               clearance_band_m=[CLR_LO_M, CLR_HI_M],
               n_valid_pairs=int(band.sum()),
               stations=[dict(id=i, lon=float(station_lonlat[i, 0]),
                              lat=float(station_lonlat[i, 1]),
                              x_m=float(stations_xy_m[i, 0]),
                              y_m=float(stations_xy_m[i, 1]),
                              clearance_m=float(clr_st[i]))
                         for i in range(n_st)])
    (ROOT / "input" / "stations_kuwait_v2.json").write_text(json.dumps(rec, indent=1))
    print(f"[saved] input/{args.out} + stations_kuwait_v2.json", flush=True)

    # figure
    foot = B.T.astype(float); foot[foot == 0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 10))
    ext = [0, nx * dx, 0, ny * dy]
    ax.imshow((clr_m * (~B)).T, origin="lower", cmap="bone", extent=ext,
              vmax=80)
    ax.imshow(foot, origin="lower", cmap="autumn", vmin=0, vmax=2,
              extent=ext, alpha=0.7)
    ax.scatter(stations_xy_m[:, 0], stations_xy_m[:, 1], c="lime", s=45,
               edgecolor="k", lw=0.6, zorder=5)
    ax.set_title(f"Kuwait CBD — {n_st} stations in inter-building gaps "
                 f"(clearance {CLR_LO_M:.0f}-{CLR_HI_M:.0f} m)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(ROOT / "output" / "wave" / "kuwait_stations_v2.png", dpi=120)
    print("[saved] output/wave/kuwait_stations_v2.png", flush=True)


if __name__ == "__main__":
    main()
