"""Type specifications + time-of-day modulation + spatial weight functions
for the 4 Stage 1 order categories.

Calibrated to Shenzhen Nanshan core (~300k residents, ~25k orders/day).
Defaults locked in the plan file:

  food         55%   λ_avg = 9.5/min   payload 0.3-1.5 kg   30 min deadline   small
  urgent_doc    5%   λ_avg = 0.9/min   payload 0.05-0.5 kg  15 min deadline   small
  parcel       30%   λ_avg = 5.2/min   payload 0.5-4.0 kg    2h deadline      small
  heavy_cargo  10%   λ_avg = 1.7/min   payload 3.0-9.0 kg    4h deadline      large
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


SECONDS_PER_DAY = 86400.0
SMALL_CLASS = 0
LARGE_CLASS = 1
SMALL_PAYLOAD_MAX_KG = 4.0


@dataclass
class TypeSpec:
    name: str
    base_lambda_per_min: float            # 24h-average λ across the whole region
    payload_kg_lo: float
    payload_kg_hi: float
    deadline_horizon_s: float
    priority: int                         # 0 = highest
    required_class_fn: Callable[[float], int]
    tod_curve: Callable[[float], float]
    weight_fn_origin: Callable[[dict], np.ndarray]
    weight_fn_dest: Callable[[dict], np.ndarray]
    color: str = "#888888"                # for the dashboard viz


# ---------- ToD curves -------------------------------------------------------

def _gaussian_mix(t_s: float, baseline: float,
                  peaks: list[tuple[float, float, float]]) -> float:
    """t_s in seconds; peaks = list of (mu_h, A, sigma_h)."""
    h = (t_s % SECONDS_PER_DAY) / 3600.0
    out = baseline
    for mu, A, sigma in peaks:
        out += A * np.exp(-0.5 * ((h - mu) / sigma) ** 2)
    return float(out)


def tod_food(t_s: float) -> float:
    return _gaussian_mix(t_s, 0.15, [
        (12.0, 3.2, 0.9),    # lunch peak
        (18.5, 3.6, 1.1),    # dinner peak
        (8.0, 0.6, 0.8),     # breakfast bump
        (22.0, 0.7, 1.3),    # late-night
    ])


def tod_urgent(t_s: float) -> float:
    return _gaussian_mix(t_s, 0.10, [
        (10.0, 2.5, 1.2),
        (15.0, 2.0, 1.3),
    ])


def tod_parcel(t_s: float) -> float:
    return _gaussian_mix(t_s, 0.30, [
        (10.0, 1.5, 2.0),
        (15.0, 1.8, 2.2),
    ])


def tod_heavy(t_s: float) -> float:
    return _gaussian_mix(t_s, 0.20, [
        (6.0, 2.0, 1.0),
        (14.0, 1.2, 1.5),
    ])


# ---------- Weight functions -------------------------------------------------

def w_food_origin(arrs: dict) -> np.ndarray:
    return 1.0 + 4.0 * arrs["n_food_pois"]


def w_food_dest(arrs: dict) -> np.ndarray:
    return arrs["n_residential_units"].clip(min=1.0)


def w_urgent_origin(arrs: dict) -> np.ndarray:
    return 1.0 + 2.0 * arrs["n_shop_pois"] + 3.0 * arrs["is_hub"]


def w_urgent_dest(arrs: dict) -> np.ndarray:
    return 1.0 + 2.0 * arrs["n_shop_pois"] + arrs["n_residential_units"] / 1000.0


def w_parcel_origin(arrs: dict) -> np.ndarray:
    return 1.0 + 8.0 * arrs["is_hub"] + arrs["n_shop_pois"]


def w_parcel_dest(arrs: dict) -> np.ndarray:
    return arrs["n_residential_units"].clip(min=1.0) + 0.3 * arrs["n_shop_pois"]


def w_heavy_origin(arrs: dict) -> np.ndarray:
    # Only hubs originate heavy_cargo. Add tiny epsilon so empty-hub case
    # never produces all-zero weights.
    return arrs["is_hub"].astype(np.float64) + 1e-6


def w_heavy_dest(arrs: dict) -> np.ndarray:
    return arrs["n_shop_pois"] + 0.5 * arrs["is_hub"] + 1e-6


# ---------- payload-to-class fns --------------------------------------------

def class_payload_small(_: float) -> int:
    return SMALL_CLASS


def class_payload_large(_: float) -> int:
    return LARGE_CLASS


def class_auto(payload_kg: float) -> int:
    return SMALL_CLASS if payload_kg <= SMALL_PAYLOAD_MAX_KG else LARGE_CLASS


# ---------- Default Stage 1 specs --------------------------------------------

def default_specs() -> List[TypeSpec]:
    return [
        TypeSpec(
            name="food",
            base_lambda_per_min=9.5,
            payload_kg_lo=0.3, payload_kg_hi=1.5,
            deadline_horizon_s=1800.0,
            priority=1,
            required_class_fn=class_payload_small,
            tod_curve=tod_food,
            weight_fn_origin=w_food_origin,
            weight_fn_dest=w_food_dest,
            color="#d62728",  # red
        ),
        TypeSpec(
            name="urgent_doc",
            base_lambda_per_min=0.9,
            payload_kg_lo=0.05, payload_kg_hi=0.5,
            deadline_horizon_s=900.0,
            priority=0,
            required_class_fn=class_payload_small,
            tod_curve=tod_urgent,
            weight_fn_origin=w_urgent_origin,
            weight_fn_dest=w_urgent_dest,
            color="#9467bd",  # purple
        ),
        TypeSpec(
            name="parcel",
            base_lambda_per_min=5.2,
            payload_kg_lo=0.5, payload_kg_hi=4.0,
            deadline_horizon_s=7200.0,
            priority=2,
            required_class_fn=class_auto,
            tod_curve=tod_parcel,
            weight_fn_origin=w_parcel_origin,
            weight_fn_dest=w_parcel_dest,
            color="#1f77b4",  # blue
        ),
        TypeSpec(
            name="heavy_cargo",
            base_lambda_per_min=1.7,
            payload_kg_lo=3.0, payload_kg_hi=9.0,
            deadline_horizon_s=14400.0,
            priority=3,
            required_class_fn=class_payload_large,
            tod_curve=tod_heavy,
            weight_fn_origin=w_heavy_origin,
            weight_fn_dest=w_heavy_dest,
            color="#2ca02c",  # green
        ),
    ]
