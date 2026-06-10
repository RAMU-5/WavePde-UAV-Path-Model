"""KPI logger and helpers for Stage 1 regression.

Reportable KPIs (matches plan §Verification):
  - throughput: deliveries per minute
  - sla_pct: orders delivered before deadline / total
  - avg_delivery_s: completed orders' mean (completed_at - born_at)
  - energy_proxy_total: sum of headwind-corrected path lengths times rho avg
  - tube_violations: count of UAV samples with ||pos - ref|| > eps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import csv
import numpy as np

from aura.schemas import Order


@dataclass
class TypeBucket:
    seen: int = 0
    completed: int = 0
    late: int = 0
    failed: int = 0
    delivery_times_s: List[float] = field(default_factory=list)


@dataclass
class KpiAccumulator:
    n_orders_seen: int = 0
    n_completed: int = 0
    n_late: int = 0
    n_failed: int = 0
    delivery_times_s: List[float] = field(default_factory=list)
    tube_violations: int = 0
    tube_samples: int = 0
    energy_proxy_total: float = 0.0
    per_type: Dict[str, TypeBucket] = field(default_factory=dict)
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    class_active_samples: List[tuple] = field(default_factory=list)  # (t, n_active_per_class)

    def _bucket(self, order_type: str) -> TypeBucket:
        b = self.per_type.get(order_type)
        if b is None:
            b = TypeBucket()
            self.per_type[order_type] = b
        return b

    def record_complete(self, order: Order, completed_at_s: float) -> None:
        self.n_completed += 1
        dt = completed_at_s - order.born_at_s
        self.delivery_times_s.append(dt)
        b = self._bucket(order.type)
        b.completed += 1
        b.delivery_times_s.append(dt)
        if completed_at_s > order.deadline_s:
            self.n_late += 1
            b.late += 1

    def record_failed(self, order: Order) -> None:
        self.n_failed += 1
        self._bucket(order.type).failed += 1

    def record_seen(self, n: int = 1, order_type: str = "_total") -> None:
        self.n_orders_seen += n
        if order_type != "_total":
            self._bucket(order_type).seen += n

    def record_tube_sample(self, deviation_m: float, eps_m: float) -> None:
        self.tube_samples += 1
        if deviation_m > eps_m:
            self.tube_violations += 1

    def record_energy(self, e: float) -> None:
        self.energy_proxy_total += float(e)

    def record_reject(self, reasons: Dict[str, int]) -> None:
        for k, v in reasons.items():
            self.reject_reasons[k] = self.reject_reasons.get(k, 0) + int(v)

    def record_class_active(self, t_s: float, n_active_per_class: tuple) -> None:
        self.class_active_samples.append((float(t_s), tuple(int(n) for n in n_active_per_class)))

    def sla_pct(self) -> float:
        served = self.n_completed
        if served == 0:
            return 1.0
        return 1.0 - self.n_late / served

    def sla_pct_type(self, t: str) -> float:
        b = self.per_type.get(t)
        if b is None or b.completed == 0:
            return 1.0
        return 1.0 - b.late / b.completed

    def avg_delivery_s(self) -> float:
        if not self.delivery_times_s:
            return 0.0
        return float(np.mean(self.delivery_times_s))

    def throughput_per_min(self, sim_duration_s: float) -> float:
        if sim_duration_s <= 0:
            return 0.0
        return self.n_completed / (sim_duration_s / 60.0)

    def tube_violation_rate(self) -> float:
        if self.tube_samples == 0:
            return 0.0
        return self.tube_violations / self.tube_samples

    def to_dict(self, sim_duration_s: float) -> Dict[str, float]:
        d = {
            "orders_seen": self.n_orders_seen,
            "orders_completed": self.n_completed,
            "orders_late": self.n_late,
            "orders_failed": self.n_failed,
            "throughput_per_min": self.throughput_per_min(sim_duration_s),
            "avg_delivery_s": self.avg_delivery_s(),
            "sla_pct": self.sla_pct(),
            "tube_violation_rate": self.tube_violation_rate(),
            "energy_proxy_total": self.energy_proxy_total,
        }
        for k, v in self.reject_reasons.items():
            d[f"reject_{k}"] = v
        for tname, b in self.per_type.items():
            d[f"{tname}_seen"] = b.seen
            d[f"{tname}_completed"] = b.completed
            d[f"{tname}_late"] = b.late
            d[f"{tname}_failed"] = b.failed
            d[f"{tname}_sla"] = self.sla_pct_type(tname)
            if b.delivery_times_s:
                arr = np.asarray(b.delivery_times_s)
                d[f"{tname}_p50_s"] = float(np.percentile(arr, 50))
                d[f"{tname}_p90_s"] = float(np.percentile(arr, 90))
                d[f"{tname}_p99_s"] = float(np.percentile(arr, 99))
        return d


def write_csv(accumulator: KpiAccumulator, sim_duration_s: float,
              path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    d = accumulator.to_dict(sim_duration_s)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        w.writeheader()
        w.writerow(d)
