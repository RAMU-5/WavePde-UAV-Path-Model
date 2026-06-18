"""Place N stations in the Kuwait CBD domain near building clusters and
embed them into a station-bearing domain npz, mirroring the Nanshan
experiment (stations sit on building clusters; nearest pairs <700m are
later excluded by the pair-distance band).

Placement: weighted k-means (k=N) over the 2D building footprint cells,
so centroids concentrate where buildings are; each centroid is snapped
to a free ground cell. Outputs:
    input/kuwait_stations.npz  — mask + grid + stations_xy_m + station_lonlat
    input/stations_kuwait.json — record (lon/lat, building counts)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[1]
STATION_Z_M = 5.0           # ground-level pickup, as in build_domain.py
LAT_M = 111000.0


def nudge_free_ground(mask, ix, iy, iz, max_r=60):
    nx, ny, nz = mask.shape
    ix = int(np.clip(ix, 1, nx - 2)); iy = int(np.clip(iy, 1, ny - 2))
    if not mask[ix, iy, iz]:
        return ix, iy, iz
    for r in range(1, max_r + 1):
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
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="kuwait_stations.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    z = dict(np.load(ROOT / "input" / args.domain, allow_pickle=False))
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    lon0, lat0, lon1, lat1 = z["domain_bbox_lonlat"]
    lat_c = 0.5 * (lat0 + lat1)
    LON_M = LAT_M * np.cos(np.deg2rad(lat_c))

    # 2D building footprint cells, weighted by column height (n filled z).
    foot = mask.sum(axis=2)                      # (nx, ny) int
    ix_b, iy_b = np.where(foot > 0)
    w = foot[ix_b, iy_b].astype(float)
    pts = np.column_stack([ix_b * dx, iy_b * dy])   # metres
    print(f"[footprint] {len(pts)} building columns")

    # Weighted k-means via weighted resampling (kmeans2 has no weights).
    rng = np.random.default_rng(args.seed)
    samp = rng.choice(len(pts), size=min(40000, 8 * len(pts)),
                      p=w / w.sum(), replace=True)
    data = pts[samp]
    cent, label = kmeans2(data, args.n, seed=args.seed, minit="++", missing="warn")
    print(f"[kmeans] {len(cent)} centroids")

    iz0 = max(1, int(round(STATION_Z_M / dz)))
    st_ix, st_iy, st_iz = [], [], []
    for cx, cy in cent:
        ix, iy, iz = nudge_free_ground(mask, round(cx / dx), round(cy / dy), iz0)
        st_ix.append(ix); st_iy.append(iy); st_iz.append(iz)
    st_ix = np.array(st_ix); st_iy = np.array(st_iy); st_iz = np.array(st_iz)

    # Drop accidental duplicates (centroids that snapped to same cell).
    _, uniq = np.unique(np.column_stack([st_ix, st_iy]), axis=0,
                        return_index=True)
    uniq = np.sort(uniq)
    st_ix, st_iy, st_iz = st_ix[uniq], st_iy[uniq], st_iz[uniq]
    n_st = len(st_ix)

    stations_xy_m = np.column_stack(
        [st_ix * dx, st_iy * dy, st_iz * dz]).astype(np.float64)
    station_lonlat = np.column_stack([
        lon0 + st_ix * dx / LON_M,
        lat0 + st_iy * dy / LAT_M]).astype(np.float64)

    # Diagnostics: pairwise distances and valid [700,3000]m band.
    diff = stations_xy_m[:, None, :2] - stations_xy_m[None, :, :2]
    D = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(D, 0.0)
    iu = np.triu_indices(n_st, 1)
    band = (D[iu] >= 700) & (D[iu] <= 3000)
    print(f"[stations] {n_st} placed (after dedup)")
    print(f"[stations] pair dist min/median/max = "
          f"{D[iu].min():.0f}/{np.median(D[iu]):.0f}/{D[iu].max():.0f} m")
    print(f"[stations] valid pairs in [700,3000]m: {band.sum()} / {len(iu[0])}")

    z["stations_xy_m"] = stations_xy_m
    z["station_lonlat"] = station_lonlat
    z["station_z_m"] = STATION_Z_M
    out = ROOT / "input" / args.out
    np.savez_compressed(out, **z)
    print(f"[saved] {out}  ({n_st} stations embedded)")

    rec = dict(
        domain=args.domain, n_stations=int(n_st),
        valid_pair_band_m=[700, 3000],
        n_valid_pairs=int(band.sum()),
        stations=[dict(id=i, lon=float(station_lonlat[i, 0]),
                       lat=float(station_lonlat[i, 1]),
                       x_m=float(stations_xy_m[i, 0]),
                       y_m=float(stations_xy_m[i, 1]),
                       z_m=float(stations_xy_m[i, 2]))
                  for i in range(n_st)],
    )
    js = ROOT / "input" / "stations_kuwait.json"
    js.write_text(json.dumps(rec, indent=1))
    print(f"[saved] {js}")


if __name__ == "__main__":
    main()
