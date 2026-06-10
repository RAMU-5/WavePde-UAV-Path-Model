"""Delivery order generator: Poisson with population + ToD modulation."""

from __future__ import annotations

from typing import List

import numpy as np

from aura.schemas import Order, Station


def time_of_day_modulation(t_s: float, day_period_s: float = 1800.0) -> float:
    """Two-peak sinusoidal mod (matches aura_toy_archive/tasks.py)."""
    phase = (t_s / day_period_s) * 2.0 * np.pi
    return 1.0 + 0.7 * np.sin(phase) ** 2


def maybe_spawn_order(orders: List[Order], stations: List[Station],
                      t: float, dt: float, rng: np.random.Generator,
                      lambda_per_min: float = 50.0,
                      deadline_horizon_s: float = 1800.0,
                      pkg_kg_lo: float = 0.2,
                      pkg_kg_hi: float = 2.0) -> Order | None:
    """Sample at most one order per call from a Poisson process.

    Origin/destination weighted by station capacity (proxy for population).
    Pattern: aura_toy_archive/tasks.py:maybe_spawn_order.
    """
    p_unit = lambda_per_min / 60.0
    p = p_unit * time_of_day_modulation(t) * dt
    if rng.random() > p:
        return None
    weights = np.asarray([s.pop_voronoi for s in stations], dtype=np.float64)
    if weights.sum() < 1e-9:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    o, d = rng.choice(len(stations), size=2, replace=False, p=weights)
    pkg = float(rng.uniform(pkg_kg_lo, pkg_kg_hi))
    order = Order(
        id=len(orders),
        origin=int(o),
        destination=int(d),
        born_at_s=float(t),
        deadline_s=float(t) + float(deadline_horizon_s),
        payload_kg=pkg,
    )
    orders.append(order)
    return order
