"""Check whether traced UAV paths pass through building voxels.

Fully vectorised (no per-path Python loops):
  (1) vertex test   : round every path vertex to a cell, test mask.
  (2) segment graze : oversample each segment K times (catch a vertex
      that steps across a building corner between two free cells).
Per-file processing keeps memory bounded.
"""
import argparse
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = 6  # sub-samples per segment (gamma step ~1.5m -> ~0.25m resolution)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="kuwait_stations.npz")
    p.add_argument("--dbdir", default="database_kuwait")
    args = p.parse_args()
    z = np.load(ROOT / "input" / args.domain)
    mask = z["mask"]
    dx, dy, dz = float(z["dx"]), float(z["dy"]), float(z["dz"])
    nx, ny, nz = mask.shape
    DB = ROOT / "output" / args.dbdir

    def inside(pts):
        ix = np.clip(np.round(pts[:, 0] / dx).astype(np.int32), 0, nx - 1)
        iy = np.clip(np.round(pts[:, 1] / dy).astype(np.int32), 0, ny - 1)
        iz = np.clip(np.round(pts[:, 2] / dz).astype(np.int32), 0, nz - 1)
        return mask[ix, iy, iz]

    tot_paths = tot_pts = 0
    hit_v_paths = hit_v_pts = 0
    hit_s_paths = 0
    worst = []

    for i in range(64):
        f = DB / f"paths_{i:03d}.npz"
        if not f.exists():
            continue
        a = np.load(f, allow_pickle=False)
        dst = a["dst_indices"]; off = a["path_offsets"]; pc = a["paths_concat"]
        status = a["path_status"].astype(str)
        tot_pts += len(pc)
        for k in range(len(dst)):
            seg = pc[off[k]:off[k + 1]]
            if len(seg) == 0:
                continue
            tot_paths += 1
            # (1) vertex test — vectorised over the whole path
            vmask = inside(seg)
            nv = int(vmask.sum())
            if nv:
                hit_v_paths += 1; hit_v_pts += nv
            # (2) segment graze — oversample all segments at once
            if len(seg) >= 2:
                a0 = seg[:-1]; a1 = seg[1:]
                t = np.linspace(0.0, 1.0, K, endpoint=False)[1:]   # interiors
                samp = (a0[:, None, :]
                        + (a1 - a0)[:, None, :] * t[None, :, None]
                        ).reshape(-1, 3)
                ns = int(inside(samp).sum()) + nv
            else:
                ns = nv
            if ns:
                hit_s_paths += 1
                worst.append((ns, i, int(dst[k]), status[k], len(seg)))
        print(f"[{i:02d}] done, cum paths={tot_paths}", flush=True)

    print("\n========== COLLISION REPORT ==========", flush=True)
    print(f"paths checked : {tot_paths}")
    print(f"path vertices : {tot_pts}")
    print("--- (1) vertex / cell-centre test ---")
    print(f"paths with >=1 vertex inside a building : {hit_v_paths} "
          f"({hit_v_paths/max(tot_paths,1)*100:.2f}%)")
    print(f"vertices inside buildings               : {hit_v_pts} "
          f"({hit_v_pts/max(tot_pts,1)*100:.4f}%)")
    print(f"--- (2) segment graze test (K={K} oversample) ---")
    print(f"paths grazing a building along a segment : {hit_s_paths} "
          f"({hit_s_paths/max(tot_paths,1)*100:.2f}%)")
    worst.sort(reverse=True)
    print("\ntop 12 worst (n_inside_samples, src, dst, status, n_vertices):")
    for w in worst[:12]:
        print("  ", w)

    tag = args.dbdir.replace("database_", "")
    out = ROOT / "output" / "wave" / f"collision_summary_{tag}.npz"
    np.savez(out,
             hit_vertex_paths=hit_v_paths, hit_vertex_pts=hit_v_pts,
             hit_segment_paths=hit_s_paths,
             tot_paths=tot_paths, tot_pts=tot_pts,
             worst=np.array(worst, dtype=object))
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
