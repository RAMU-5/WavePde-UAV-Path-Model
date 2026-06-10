"""Multi-type Poisson order generation orchestrator.

For each registered TypeSpec, on every 1 Hz tick draw `k = Poisson(λ · m(t) · dt)`
new orders, sample (origin, destination) pairs from the type-specific weight
vectors, reject pairs >5 km apart or with no nominal in `db`, and append.

Each type carries its own RNG split off the master seed so adding/removing one
type does not perturb the order sequence of the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from aura.fmm.agd_solver import AllPairsDB
from aura.missions.order_types import TypeSpec
from aura.network.poi_features import features_to_arrays
from aura.schemas import Order, Station


@dataclass
class OrderBookCounters:
    """Per-type running tallies (debug/KPI)."""
    spawned: Dict[str, int] = field(default_factory=dict)
    rejected_geometry: Dict[str, int] = field(default_factory=dict)
    rejected_no_traj: Dict[str, int] = field(default_factory=dict)


class OrderBook:
    def __init__(self,
                 specs: List[TypeSpec],
                 stations: List[Station],
                 db: AllPairsDB,
                 master_rng: np.random.Generator,
                 coverage_radius_m: float = 5000.0,
                 max_resample: int = 5):
        self.specs = specs
        self.stations = stations
        self.db = db
        self.coverage_radius_m = float(coverage_radius_m)
        self.max_resample = int(max_resample)
        # Per-type RNG split for reproducibility-stable ordering.
        self.rngs = {
            s.name: np.random.default_rng(int(master_rng.integers(2**62)))
            for s in specs
        }
        # Cache feature arrays + per-type origin/dest weight vectors.
        feat = features_to_arrays(stations)
        self._origin_w = {}
        self._dest_w = {}
        for s in specs:
            ow = np.asarray(s.weight_fn_origin(feat), dtype=np.float64)
            dw = np.asarray(s.weight_fn_dest(feat), dtype=np.float64)
            ow = np.clip(ow, 0.0, None)
            dw = np.clip(dw, 0.0, None)
            if ow.sum() <= 0:
                ow = np.ones_like(ow)
            if dw.sum() <= 0:
                dw = np.ones_like(dw)
            self._origin_w[s.name] = ow / ow.sum()
            self._dest_w[s.name] = dw / dw.sum()
        self.station_xy = np.asarray([s.xy_m for s in stations])
        self.counters = OrderBookCounters(
            spawned={s.name: 0 for s in specs},
            rejected_geometry={s.name: 0 for s in specs},
            rejected_no_traj={s.name: 0 for s in specs},
        )

    def _draw_pair(self, spec: TypeSpec) -> Optional[tuple[int, int]]:
        """Sample (origin, destination) station ids; return None on persistent failure."""
        rng = self.rngs[spec.name]
        ow = self._origin_w[spec.name]
        dw = self._dest_w[spec.name]
        K = len(self.stations)
        for _ in range(self.max_resample):
            o = int(rng.choice(K, p=ow))
            d = int(rng.choice(K, p=dw))
            if o == d:
                continue
            dx = self.station_xy[o, 0] - self.station_xy[d, 0]
            dy = self.station_xy[o, 1] - self.station_xy[d, 1]
            if dx * dx + dy * dy > self.coverage_radius_m ** 2:
                self.counters.rejected_geometry[spec.name] += 1
                continue
            if self.db.get(o, d) is None:
                self.counters.rejected_no_traj[spec.name] += 1
                continue
            return (o, d)
        return None

    def step(self, orders: List[Order], t_s: float, dt_s: float) -> int:
        """Spawn new orders this tick; return count of new orders appended."""
        added = 0
        for spec in self.specs:
            rng = self.rngs[spec.name]
            m_t = float(spec.tod_curve(t_s))
            mean_count = spec.base_lambda_per_min / 60.0 * m_t * dt_s
            if mean_count <= 0:
                continue
            k = int(rng.poisson(mean_count))
            for _ in range(k):
                pair = self._draw_pair(spec)
                if pair is None:
                    continue
                o, d = pair
                payload = float(rng.uniform(spec.payload_kg_lo, spec.payload_kg_hi))
                req_class = int(spec.required_class_fn(payload))
                order = Order(
                    id=len(orders),
                    origin=o, destination=d,
                    born_at_s=float(t_s),
                    deadline_s=float(t_s) + float(spec.deadline_horizon_s),
                    payload_kg=payload,
                    type=spec.name,
                    priority=int(spec.priority),
                    required_class=req_class,
                )
                orders.append(order)
                self.counters.spawned[spec.name] += 1
                added += 1
        return added
