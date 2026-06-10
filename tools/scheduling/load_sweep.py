"""Load sweep: run greedy and mean-field dispatchers across a range of
order arrival rates, holding fleet size fixed.

Goal: identify the load regime where mean-field's global view starts to
matter (UAV-bound regime), vs the regime where greedy is good enough
(UAV-rich regime).

At each load level we report:
  - completion %
  - mean / p95 wait and delivery time
  - peak pending queue length
  - per-dispatch wall-clock
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from rolling_sim import (SimConfig, run_simulation,
                         dispatch_greedy, dispatch_mf)


LOAD_LEVELS = [3_000, 6_000, 9_000, 12_000]    # orders / hour
N_UAVS = 500
SIM_HOURS = (18.0, 19.0)
SEED = 0


def fmt_pct(x): return f"{x:5.1f}%"
def fmt_s(x):   return f"{x:7.1f}"
def fmt_ms(x):  return f"{x:8.2f}"
def fmt_pp(d):  return f"{d:+5.2f}pp"


def run_one(target, dispatcher, name):
    cfg = SimConfig(n_uavs=N_UAVS, sim_hours=SIM_HOURS,
                    target_orders_per_hour=target, seed=SEED)
    print(f"  [{name:10s}] load={target}/h ... ", end="", flush=True)
    t0 = time.time()
    k = run_simulation(cfg, dispatcher, name, verbose=False)
    print(f"done in {time.time()-t0:5.1f}s "
          f"({k['n_orders']} orders, {k['n_delivered']} delivered, "
          f"compl {k['completion_pct']:.1f}%)")
    return k


def main():
    rows = []
    for target in LOAD_LEVELS:
        print(f"\n--- Load = {target} orders/hour ---")
        kg = run_one(target, dispatch_greedy, "greedy")
        km = run_one(target, dispatch_mf, "mean-field")
        rows.append((target, kg, km))

    # ---- Comparison table
    print("\n" + "=" * 100)
    print(" LOAD SWEEP — greedy vs mean-field")
    print("=" * 100)
    hdr = (f"{'load/hr':>8s}  {'method':>10s}  {'orders':>6s}  "
           f"{'compl%':>6s}  {'wait/s':>7s}  {'p95w/s':>7s}  "
           f"{'deliv/s':>8s}  {'p95d/s':>8s}  {'pkpend':>6s}  "
           f"{'util':>6s}  {'disp/ms':>8s}")
    print(hdr)
    for target, kg, km in rows:
        for label, k in (("greedy", kg), ("mean-fld", km)):
            print(f"{target:>8d}  {label:>10s}  {k['n_orders']:>6d}  "
                  f"{k['completion_pct']:>5.1f}%  "
                  f"{k['mean_wait_pickup_s']:>7.1f}  "
                  f"{k['p95_wait_pickup_s']:>7.1f}  "
                  f"{k['mean_delivery_s']:>8.1f}  "
                  f"{k['p95_delivery_s']:>8.1f}  "
                  f"{k['peak_pending']:>6d}  "
                  f"{k['fleet_util_mean']*100:>5.1f}%  "
                  f"{k['mean_dispatch_ms']:>8.2f}")

    # ---- Delta summary (MF − greedy)
    print("\n" + "=" * 100)
    print(" DELTA (mean-field − greedy)")
    print("=" * 100)
    print(f"{'load/hr':>8s}  {'Δcompl':>8s}  {'Δwait':>9s}  {'Δdeliv':>9s}  "
          f"{'Δpkpend':>9s}  {'slowdown':>10s}")
    for target, kg, km in rows:
        d_compl = km["completion_pct"] - kg["completion_pct"]
        d_wait = (km["mean_wait_pickup_s"] - kg["mean_wait_pickup_s"]) \
                  / max(kg["mean_wait_pickup_s"], 1e-9) * 100
        d_deliv = (km["mean_delivery_s"] - kg["mean_delivery_s"]) \
                   / max(kg["mean_delivery_s"], 1e-9) * 100
        d_pkpend = km["peak_pending"] - kg["peak_pending"]
        slow = km["mean_dispatch_ms"] / max(kg["mean_dispatch_ms"], 1e-9)
        print(f"{target:>8d}  {d_compl:>+7.2f}pp  "
              f"{d_wait:>+8.2f}%  {d_deliv:>+8.2f}%  "
              f"{d_pkpend:>+9d}  {slow:>9.0f}×")


if __name__ == "__main__":
    main()
