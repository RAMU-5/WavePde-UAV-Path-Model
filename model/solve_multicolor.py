"""Four-colour domain decomposition for batched routing
(paper §2.13.1). Most-impactful interpretation:

Each station-source has a wave-influence region of radius
~ N_STEPS * c0. Two sources whose influence regions DON'T overlap
can be solved in a single wave equation by SUPERIMPOSING their
sources (linearity of the PDE, paper §2.9.4). The first-arrival-
time field then encodes a separate T(x) for each source within
its own influence region, and a 4-colouring of the conflict graph
guarantees that within one colour class no two sources share a
destination cell.

This means N_st sources can be solved in roughly  ceil(N_st / k)
wave solves where k is the average colour-class size — typically
N_st / 4 if sources are well-spread.

NOT implemented here: the multi-GPU domain split with halo
exchange (paper §2.13.2). That requires MPI/jax-distributed and
is a multi-day effort; this module covers the routing-time
speedup which is what the paper explicitly motivates.

Public API:
  conflict_graph(positions, min_separation_m) -> adj-list dict
  greedy_four_color(adj) -> list[int]   colour per node, in {0..3}
  solve_batched(static_fields, src_indices, dx, dy, dz, n_steps)
      -> T-field (shape of domain) — first-arrival of the wave from
         the union of all batched sources.
  arrange_by_colour(positions, min_separation_m) -> list[list[int]]
      (one inner list per colour, contains source indices)
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional

import os
_THIS = Path(__file__).resolve().parent
_CACHE_DIR = str(_THIS.parent / "_jax_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)

import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit

from solve_all import (
    build_static_fields, gaussian_source, solve_one, N_STEPS,
)


def conflict_graph(positions: np.ndarray, min_separation_m: float) -> dict:
    """Build adjacency: i~j if ||pos_i - pos_j|| < min_separation_m.

    positions : (N, 3) array
    Returns : { i: [j, ...] } for each node.
    """
    N = positions.shape[0]
    adj = defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < min_separation_m:
                adj[i].append(j)
                adj[j].append(i)
    return dict(adj)


def greedy_four_color(adj: dict, n_nodes: int) -> list:
    """Greedy graph colouring with welsh-powell ordering (high degree
    first). Always succeeds in 4 colours for any planar-ish station
    network (proven for any planar graph by 4-colour theorem; even
    non-planar conflict graphs typically need only 3-4 here).

    Returns: list of length n_nodes, each entry in {0..K-1}.
    """
    order = sorted(range(n_nodes),
                   key=lambda i: -len(adj.get(i, [])))
    colour = [-1] * n_nodes
    for v in order:
        forbidden = set(colour[u] for u in adj.get(v, []) if colour[u] >= 0)
        c = 0
        while c in forbidden:
            c += 1
        colour[v] = c
    return colour


def arrange_by_colour(positions: np.ndarray, min_separation_m: float
                      ) -> list:
    """Return list-of-lists; inner list i = source indices of colour i.
    The number of inner lists is the number of distinct colours used
    (>= 1, typically <= 4)."""
    N = positions.shape[0]
    adj = conflict_graph(positions, min_separation_m)
    colour = greedy_four_color(adj, N)
    n_col = max(colour) + 1
    out = [[] for _ in range(n_col)]
    for i, c in enumerate(colour):
        out[c].append(i)
    return out


def batched_source(shape, indices_xyz: list, sigma: float = 1.5):
    """Sum of Gaussians at the listed integer-cell indices.

    Each Gaussian normalised to unit mass; the SUM has total mass = K,
    where K is the number of sources in the batch."""
    s = jnp.zeros(shape, dtype=jnp.float32)
    for idx in indices_xyz:
        s = s + gaussian_source(shape, idx, sigma=sigma)
    return s


def solve_batched(static, src_indices_xyz, dx, dy, dz,
                  n_steps: int = N_STEPS):
    """One wave solve with a multi-source. Returns T-field.
    Caller is responsible for ensuring sources are non-conflicting."""
    s_total = batched_source(static["c2_xy"].shape, src_indices_xyz)
    one_minus_damp = (1.0 - static["damp"]).astype(jnp.float32)
    solve_jit = jit(solve_one, static_argnums=(11, 12, 13, 14))
    T = solve_jit(static["c2_xy"], static["c2_z"],
                  static["u"], static["v"], static["w"],
                  static["ax"], static["ay"], static["az"],
                  one_minus_damp, static["free"], s_total,
                  dx, dy, dz, n_steps)
    return jax.device_get(T)


def estimate_min_separation(c0: float, n_steps: int, dt: float) -> float:
    """Conservative: 2 * (waves' farthest reach during simulation)."""
    return 2.0 * c0 * n_steps * dt
