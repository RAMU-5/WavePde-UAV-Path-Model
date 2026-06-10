#!/bin/bash
#SBATCH --job-name=mf_chg
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D25_028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_charge_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_charge_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone

PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python

echo "=== Node ==="; hostname; nvidia-smi -L

echo
echo "=== Paper-scale: batching + rebalance + 5-min CHARGING ==="
echo "=== 7080 UAV, 118 stations, 1h dinner peak ==="
$PY -u -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import time
from rolling_sim_v2 import (SimConfigV2, run_simulation,
                              greedy_dispatcher, mf_gpu_dispatcher)

variants = [
    ('greedy + rb (no chg)', greedy_dispatcher, True,  0.0),
    ('greedy + rb + chg5m ', greedy_dispatcher, True,  300.0),
    ('mf-gpu + rb (no chg)', mf_gpu_dispatcher, True,  0.0),
    ('mf-gpu + rb + chg5m ', mf_gpu_dispatcher, True,  300.0),
]

print('=' * 120)
print(f\"  {'load/h':>7s}  {'variant':>22s}  {'compl%':>6s}  {'wait/s':>7s}  \"
      f\"{'deliv/s':>8s}  {'pkpend':>6s}  {'mean_batch':>10s}  {'util':>5s}  \"
      f\"{'pk_chg':>6s}  {'mean_chg':>8s}  {'deadhead':>8s}  {'wall/s':>6s}\")
print('=' * 120)
results = []
for target in [60_000, 85_000, 120_000]:
    print(f'-- Target {target}/h --')
    for vname, fn, rebal, charge in variants:
        cfg = SimConfigV2(n_uavs=7080, sim_hours=(18.0, 19.0),
                          target_orders_per_hour=target, seed=0,
                          max_batch_size=3, batch_detour_sec=150.0,
                          enable_rebalance=rebal,
                          rebalance_max_moves=1500,
                          charge_sec_per_trip=charge)
        t0 = time.time()
        k = run_simulation(cfg, fn, vname.strip(), verbose=False)
        print(f\"  {target:>7d}  {vname}  \"
              f\"{k['completion_pct']:>5.1f}%  \"
              f\"{k['mean_wait_pickup_s']:>7.1f}  \"
              f\"{k['mean_delivery_s']:>8.1f}  \"
              f\"{k['peak_pending']:>6d}  \"
              f\"{k['mean_batch_size']:>10.2f}  \"
              f\"{k['fleet_util_mean']*100:>4.1f}%  \"
              f\"{k['peak_charging_count']:>6d}  \"
              f\"{k['mean_charging_count']:>8.1f}  \"
              f\"{k['n_deadhead_total']:>8d}  \"
              f\"{k['sim_wallclock_s']:>6.1f}\")
        results.append((target, vname.strip(), k))

print()
print('=' * 110)
print(' Charging impact summary (greedy+rb baseline; same for MF+rb)')
print('=' * 110)
for target in [60_000, 85_000, 120_000]:
    g0 = next(k for t,n,k in results if t==target and n=='greedy + rb (no chg)')
    g1 = next(k for t,n,k in results if t==target and n=='greedy + rb + chg5m')
    m0 = next(k for t,n,k in results if t==target and n=='mf-gpu + rb (no chg)')
    m1 = next(k for t,n,k in results if t==target and n=='mf-gpu + rb + chg5m')
    print(f'{target}/h: greedy+rb: nochg={g0[\"completion_pct\"]:.1f}% chg5m={g1[\"completion_pct\"]:.1f}% (Δ{g1[\"completion_pct\"]-g0[\"completion_pct\"]:+.1f}pp)   '
          f'mf+rb: nochg={m0[\"completion_pct\"]:.1f}% chg5m={m1[\"completion_pct\"]:.1f}% (Δ{m1[\"completion_pct\"]-m0[\"completion_pct\"]:+.1f}pp)   '
          f'ΔMF-over-greedy-with-charging={m1[\"completion_pct\"]-g1[\"completion_pct\"]:+.1f}pp')
"

echo; echo "=== Done ==="; date
