"""Anisotropic FMM driver — wraps Mirebeau's AGD HFM library.

Public surface:
  - solve_single_source(field, grid, src_xyz)            -> ArrivalField
  - extract_geodesic(arrival, grid, dst_xyz, dt=0.5)     -> NominalTrajectory
  - solve_all_pairs(field, grid, station_xys)            -> AllPairsDB
  - resolve_affected_sources(field, grid, db, sources)   -> updated AllPairsDB

agd is lazily imported. If the CUDA ``Rander3`` backend is unavailable, the
solver falls back to the sparse isotropic Dijkstra implementation below.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from aura.cost.metric import RanderField
from aura.geom.grid import SimGrid
from aura.schemas import NominalTrajectory


@dataclass
class ArrivalField:
    """T(x) — minimum-cost arrival time over the grid for one source."""
    T: np.ndarray                          # (nx, ny, nz) float32
    source_xyz: Tuple[float, float, float]
    metric_version: int = 0


@dataclass
class AllPairsDB:
    """Sparse upper-triangular dict (src, dst) -> NominalTrajectory."""
    pairs: Dict[Tuple[int, int], NominalTrajectory] = dc_field(default_factory=dict)
    arrivals: Dict[int, ArrivalField] = dc_field(default_factory=dict)
    grid: SimGrid | None = None
    metric_version: int = 0

    def get(self, src: int, dst: int) -> NominalTrajectory | None:
        return self.pairs.get((src, dst)) or self.pairs.get((dst, src))

    def keys(self):
        return list(self.pairs.keys())


def _ensure_agd(backend: str = "cpu") -> bool:
    """Return whether the requested AGD backend can solve ``Rander3``.

    The FileHFM CPU distribution does not provide a three-dimensional
    Randers model.  AGD's CUDA implementation does, so an importable Python
    package alone is not sufficient evidence that the requested solve will
    work.
    """
    if backend != "cuda":
        return False
    try:
        import agd  # noqa: F401
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _solve_agd_rander(field: RanderField, grid: SimGrid,
                      src_xyz: Tuple[float, float, float],
                      backend: str = "cpu") -> np.ndarray:
    """Call AGD HFM, return arrival field (nx, ny, nz)."""
    if backend != "cuda":
        raise RuntimeError("AGD Rander3 is available only through its CUDA backend")

    from agd import Eikonal, Metrics
    nx, ny, nz = grid.shape3
    omega = np.moveaxis(field.rander_omega, -1, 0)  # (3, nx, ny, nz)
    riemann = np.zeros((3, 3, nx, ny, nz), dtype=np.float32)
    diag = field.riemann_diag  # (nx, ny, nz, 3)
    riemann[0, 0] = diag[..., 0]
    riemann[1, 1] = diag[..., 1]
    riemann[2, 2] = diag[..., 2]
    metric = Metrics.Rander(riemann, omega)
    hfm = Eikonal.dictIn({
        "mode": "gpu_transfer",
        "model": "Rander3",
        "metric": metric,
        "seeds": np.asarray([src_xyz], dtype=np.float64),
        "exportValues": 1,
        "verbosity": 0,
    })
    hfm.SetRect([
        [grid.origin_xy[0], grid.origin_xy[0] + grid.size_xy[0]],
        [grid.origin_xy[1], grid.origin_xy[1] + grid.size_xy[1]],
        [grid.z_min, grid.z_max],
    ], dims=(nx, ny, nz))
    if field.walls is not None:
        hfm["walls"] = field.walls
    out = hfm.Run()
    T = np.asarray(out["values"], dtype=np.float32)
    return T.reshape((nx, ny, nz))


def solve_single_source(field: RanderField, grid: SimGrid,
                        src_xyz: Tuple[float, float, float],
                        backend: str = "cpu") -> ArrivalField:
    if not _ensure_agd(backend):
        T = _fallback_isotropic_arrival(field, grid, src_xyz)
    else:
        T = _solve_agd_rander(field, grid, src_xyz, backend=backend)
    return ArrivalField(T=T, source_xyz=src_xyz,
                        metric_version=field.metric_version)


def _build_grid_graph(field: RanderField, grid: SimGrid):
    """Build a 6-connectivity sparse cost graph over the full 3D grid.

    Returns (csr_graph, walls_flat) for downstream scipy Dijkstra calls.
    Edge cost = step_meters * 0.5 * (inv_v_eff[src] + inv_v_eff[dst]),
    midpoint average of rho/v_cruise so the metric stays consistent.
    Edges into wall cells are dropped entirely.
    """
    from scipy.sparse import csr_matrix
    nx, ny, nz = grid.shape3
    inv_v = (field.rho * (1.0 / field.v_cruise)).astype(np.float64)
    walls = (field.walls if field.walls is not None
             else np.zeros((nx, ny, nz), dtype=bool))
    walls_flat = walls.ravel()
    flat = np.arange(nx * ny * nz, dtype=np.int64).reshape(nx, ny, nz)

    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []
    for di, dj, dk, step in (
        (1, 0, 0, grid.dx),
        (0, 1, 0, grid.dy),
        (0, 0, 1, grid.dz),
    ):
        src_idx = flat[: nx - di, : ny - dj, : nz - dk].ravel()
        dst_idx = flat[di:, dj:, dk:].ravel()
        inv_v_src = inv_v[: nx - di, : ny - dj, : nz - dk].ravel()
        inv_v_dst = inv_v[di:, dj:, dk:].ravel()
        w = step * 0.5 * (inv_v_src + inv_v_dst)
        keep = ~(walls_flat[src_idx] | walls_flat[dst_idx])
        src_idx = src_idx[keep]
        dst_idx = dst_idx[keep]
        w = w[keep]
        rows_all.append(np.concatenate([src_idx, dst_idx]))
        cols_all.append(np.concatenate([dst_idx, src_idx]))
        data_all.append(np.concatenate([w, w]))
    n = nx * ny * nz
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)
    g = csr_matrix((data, (rows, cols)), shape=(n, n))
    return g, walls_flat


def _fallback_isotropic_arrival(field: RanderField, grid: SimGrid,
                                src_xyz: Tuple[float, float, float]) -> np.ndarray:
    """scipy.csgraph multi-source Dijkstra wrapper — single source case.

    NOT a substitute for the real AGD anisotropic Eikonal — wind asymmetry
    (Rander offset) is ignored. But it scales to ~5M cells in seconds and
    respects building wall masks, which is what we need for fallback dev.
    """
    from scipy.sparse.csgraph import dijkstra
    nx, ny, nz = grid.shape3
    g, walls_flat = _build_grid_graph(field, grid)
    src_ijk = grid.world_to_index(np.asarray(src_xyz)[None, :])[0]
    src_lin = int((src_ijk[0] * ny + src_ijk[1]) * nz + src_ijk[2])
    if walls_flat[src_lin]:
        return np.full((nx, ny, nz), np.inf, dtype=np.float32)
    dist = dijkstra(g, indices=src_lin, directed=False,
                    return_predecessors=False)
    return dist.reshape(nx, ny, nz).astype(np.float32)


def _fallback_isotropic_batch(field: RanderField, grid: SimGrid,
                              src_xyzs: np.ndarray) -> np.ndarray:
    """Multi-source batch — re-uses the sparse graph across all sources.

    src_xyzs: (S, 3) world coords. Returns (S, nx, ny, nz) arrival times.
    """
    from scipy.sparse.csgraph import dijkstra
    nx, ny, nz = grid.shape3
    g, walls_flat = _build_grid_graph(field, grid)
    ijk = grid.world_to_index(np.asarray(src_xyzs))  # (S, 3)
    src_lin = (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]
    src_lin = src_lin.astype(np.int64)
    valid = ~walls_flat[src_lin]
    if not valid.all():
        # Snap invalid sources up to the nearest non-wall z-cell.
        for s in np.where(~valid)[0]:
            i, j, k = ijk[s]
            for k2 in range(nz):
                lin = int((i * ny + j) * nz + k2)
                if not walls_flat[lin]:
                    src_lin[s] = lin
                    break
    dist = dijkstra(g, indices=src_lin, directed=False,
                    return_predecessors=False)  # (S, n)
    return dist.reshape(-1, nx, ny, nz).astype(np.float32)


def _snap_dst_to_navigable(arrival: ArrivalField, grid: SimGrid,
                           dst_xyz: Tuple[float, float, float]
                           ) -> Tuple[float, float, float]:
    """Snap dst z up/down to the cell with smallest finite arrival time."""
    ijk = grid.world_to_index(np.asarray(dst_xyz)[None, :])[0]
    ix, iy = int(ijk[0]), int(ijk[1])
    nz = arrival.T.shape[2]
    col = arrival.T[ix, iy]
    if not np.any(np.isfinite(col)):
        return dst_xyz
    iz_best = int(np.argmin(np.where(np.isfinite(col), col, np.inf)))
    z_centers = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
    return (float(dst_xyz[0]), float(dst_xyz[1]), float(z_centers[iz_best]))


def extract_geodesic(arrival: ArrivalField, grid: SimGrid,
                     dst_xyz: Tuple[float, float, float],
                     trajectory_dt_s: float = 0.5,
                     v_cruise: float = 15.0,
                     max_steps: int = 10000,
                     gradient_T=None) -> NominalTrajectory:
    """Steepest-descent of T from dst back to source, then resampled.

    Avoids depending on AGD's geodesic exporter — works for both AGD and
    fallback arrivals. The returned trajectory is in source->dest order.

    `gradient_T` is an optional precomputed (gx, gy, gz) tuple of np.gradient
    output for `arrival.T` — pass it when running many destinations against
    the same source so the gradient isn't recomputed N-1 times.
    """
    nx, ny, nz = arrival.T.shape
    dst_xyz = _snap_dst_to_navigable(arrival, grid, dst_xyz)
    pos = np.asarray(dst_xyz, dtype=np.float64)
    if gradient_T is None:
        # Numerical gradient of T. NaN around wall boundaries is expected.
        with np.errstate(invalid="ignore"):
            gT = np.gradient(arrival.T, grid.dx, grid.dy, grid.dz)
    else:
        gT = gradient_T
    gx, gy, gz = gT  # each (nx, ny, nz)

    path: list[Tuple[float, float, float]] = [tuple(pos.tolist())]
    src_xyz = np.asarray(arrival.source_xyz, dtype=np.float64)
    step_h = max(grid.dx, grid.dy, grid.dz) * 0.5
    x_lo = grid.origin_xy[0] + 1e-3
    x_hi = grid.origin_xy[0] + grid.size_xy[0] - 1e-3
    y_lo = grid.origin_xy[1] + 1e-3
    y_hi = grid.origin_xy[1] + grid.size_xy[1] - 1e-3
    z_lo = grid.z_min + 1e-3
    z_hi = grid.z_max - 1e-3
    for _ in range(max_steps):
        if np.linalg.norm(pos - src_xyz) < step_h:
            path.append(tuple(src_xyz.tolist()))
            break
        ijk = grid.world_to_index(pos[None, :])[0]
        ix, iy, iz = int(ijk[0]), int(ijk[1]), int(ijk[2])
        g = np.asarray([gx[ix, iy, iz], gy[ix, iy, iz], gz[ix, iy, iz]])
        if not np.all(np.isfinite(g)):
            break
        n = np.linalg.norm(g)
        if n < 1e-9:
            break
        pos = pos - step_h * g / n
        # Clip into the grid AABB so a stuck backtrace cannot drift outside
        # the simulation domain (otherwise altitude can balloon to 1000m+).
        pos[0] = min(max(pos[0], x_lo), x_hi)
        pos[1] = min(max(pos[1], y_lo), y_hi)
        pos[2] = min(max(pos[2], z_lo), z_hi)
        path.append(tuple(pos.tolist()))
    path.reverse()  # now source -> dest

    arr = np.asarray(path, dtype=np.float64)
    # Drop any non-finite rows (gradient at wall boundaries).
    finite_rows = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_rows]
    if len(arr) < 2:
        return NominalTrajectory(src=-1, dst=-1, xyz=arr.tolist(),
                                 dt_s=trajectory_dt_s,
                                 metric_version=arrival.metric_version)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(cum[-1])
    if not np.isfinite(total_len) or total_len <= 0.0:
        return NominalTrajectory(src=-1, dst=-1, xyz=arr.tolist(),
                                 dt_s=trajectory_dt_s,
                                 metric_version=arrival.metric_version)
    total_t = total_len / float(v_cruise)
    n_samples = max(2, int(np.ceil(total_t / trajectory_dt_s)) + 1)
    ts = np.linspace(0.0, total_len, n_samples)
    sampled = np.stack(
        [np.interp(ts, cum, arr[:, c]) for c in range(arr.shape[1])], axis=1
    )
    return NominalTrajectory(src=-1, dst=-1, xyz=sampled.tolist(),
                             dt_s=trajectory_dt_s,
                             metric_version=arrival.metric_version)


def _direct_distance_xy(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _coverage_dst_indices(station_xys: np.ndarray, sid: int,
                          coverage_radius_m: float) -> list[int]:
    """Indices of stations within `coverage_radius_m` of station sid (excludes self)."""
    src = station_xys[sid]
    diff = station_xys - src
    d2 = (diff ** 2).sum(axis=1)
    out = np.where(d2 <= coverage_radius_m ** 2)[0].tolist()
    return [i for i in out if i != sid]


def _solve_chunk_worker(payload):
    """multiprocessing worker — solves a chunk of source indices.

    payload = (field, grid, station_xys, sid_list, station_z_m,
               coverage_radius_m, trajectory_dt_s)
    Returns dict {(sid, did): NominalTrajectory} for this chunk.
    """
    (field, grid, station_xys, sid_list, station_z_m,
     coverage_radius_m, trajectory_dt_s) = payload
    src_xyzs = np.column_stack([
        station_xys[sid_list, 0],
        station_xys[sid_list, 1],
        np.full(len(sid_list), float(station_z_m)),
    ])
    T_all = _fallback_isotropic_batch(field, grid, src_xyzs)
    out: dict = {}
    arrivals_out: dict = {}
    for k, sid in enumerate(sid_list):
        arrival = ArrivalField(T=T_all[k],
                               source_xyz=tuple(src_xyzs[k].tolist()),
                               metric_version=field.metric_version)
        arrivals_out[sid] = arrival
        with np.errstate(invalid="ignore"):
            gT = np.gradient(arrival.T, grid.dx, grid.dy, grid.dz)
        if coverage_radius_m is None or coverage_radius_m <= 0:
            dst_ids = [i for i in range(len(station_xys)) if i != sid]
        else:
            dst_ids = _coverage_dst_indices(station_xys, sid,
                                            coverage_radius_m)
        for did in dst_ids:
            dx, dy = station_xys[did]
            traj = extract_geodesic(
                arrival, grid,
                (float(dx), float(dy), float(station_z_m)),
                trajectory_dt_s=trajectory_dt_s,
                v_cruise=field.v_cruise,
                gradient_T=gT,
            )
            traj.src = int(sid)
            traj.dst = int(did)
            out[(int(sid), int(did))] = traj
    return out, arrivals_out


def solve_all_pairs(field: RanderField, grid: SimGrid,
                    station_xys: np.ndarray, station_z_m: float,
                    trajectory_dt_s: float = 0.5,
                    backend: str = "cpu",
                    progress_cb=None,
                    coverage_radius_m: float | None = None,
                    n_workers: int = 1) -> AllPairsDB:
    """Per-source FMM, then back-trace to every other station within range.

    Args:
      coverage_radius_m: if set, each source only computes trajectories to
        stations within this Euclidean distance. None / <=0 = all-pairs.
      n_workers: if >1, splits sources across processes via multiprocessing.

    Strategy when AGD is unavailable:
      Build the 3D grid graph once per worker, run scipy multi-source Dijkstra
      on that worker's sources, back-trace with gradient cache. 9-12x speedup
      with 16 cores vs single-threaded for 100 sources.
    """
    db = AllPairsDB(grid=grid, metric_version=field.metric_version)
    n = len(station_xys)
    have_agd = _ensure_agd(backend)
    if have_agd:
        for sid in range(n):
            sx, sy = station_xys[sid]
            src_xyz = (float(sx), float(sy), float(station_z_m))
            arrival = solve_single_source(field, grid, src_xyz, backend=backend)
            db.arrivals[sid] = arrival
            with np.errstate(invalid="ignore"):
                gT = np.gradient(arrival.T, grid.dx, grid.dy, grid.dz)
            if coverage_radius_m is None or coverage_radius_m <= 0:
                dst_ids = [i for i in range(n) if i != sid]
            else:
                dst_ids = _coverage_dst_indices(station_xys, sid,
                                                coverage_radius_m)
            for did in dst_ids:
                dx, dy = station_xys[did]
                traj = extract_geodesic(
                    arrival, grid,
                    (float(dx), float(dy), float(station_z_m)),
                    trajectory_dt_s=trajectory_dt_s,
                    v_cruise=field.v_cruise, gradient_T=gT,
                )
                traj.src = sid
                traj.dst = did
                db.pairs[(sid, did)] = traj
            if progress_cb:
                progress_cb(sid + 1, n)
        return db

    # Fallback path: single-process or multi-process scipy Dijkstra.
    if n_workers <= 1:
        result, arrivals = _solve_chunk_worker(
            (field, grid, station_xys, list(range(n)), station_z_m,
             coverage_radius_m, trajectory_dt_s)
        )
        db.pairs.update(result)
        db.arrivals.update(arrivals)
        if progress_cb:
            progress_cb(n, n)
        return db

    import multiprocessing as mp
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks = [list(range(i, min(i + chunk_size, n)))
              for i in range(0, n, chunk_size)]
    payloads = [
        (field, grid, station_xys, ch, station_z_m,
         coverage_radius_m, trajectory_dt_s)
        for ch in chunks
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(n_workers, len(chunks))) as pool:
        for k, (chunk_pairs, chunk_arrivals) in enumerate(
                pool.imap_unordered(_solve_chunk_worker, payloads)):
            db.pairs.update(chunk_pairs)
            db.arrivals.update(chunk_arrivals)
            if progress_cb:
                done = sum(len(c) for c in chunks[: k + 1])
                progress_cb(done, n)
    return db


def save_db(db: AllPairsDB, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(path / "trajectories.pkl", "wb") as f:
        pickle.dump({"pairs": db.pairs, "metric_version": db.metric_version},
                    f, protocol=pickle.HIGHEST_PROTOCOL)


def load_db(path: str | Path) -> AllPairsDB:
    import pickle
    path = Path(path)
    with open(path / "trajectories.pkl", "rb") as f:
        d = pickle.load(f)
    return AllPairsDB(pairs=d["pairs"], metric_version=d["metric_version"])
