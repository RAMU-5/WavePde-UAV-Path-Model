#!/bin/bash
#SBATCH --job-name=mf_full
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D25_028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_full_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_full_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone

PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python

echo "=== Node ==="
hostname; nvidia-smi -L

echo
echo "=== Paper-scale: batching + predictive rebalance ==="
echo "=== 7080 UAV, 118 stations, 1h dinner peak, 4 dispatcher variants ==="
$PY -u -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import time
from rolling_sim_v2 import (SimConfigV2, run_simulation,
                              greedy_dispatcher, mf_gpu_dispatcher)

print('=' * 110)
print(f\"  {'load/h':>7s}  {'variant':>16s}  {'compl%':>6s}  {'wait/s':>7s}  \"
      f\"{'deliv/s':>8s}  {'pkpend':>6s}  {'mean_batch':>10s}  {'util':>5s}  \"
      f\"{'deadhead':>8s}  {'disp/ms':>8s}  {'wall/s':>7s}\")
print('=' * 110)
variants = [
    ('greedy            ', greedy_dispatcher, False),
    ('greedy + rebalance', greedy_dispatcher, True),
    ('mf-gpu            ', mf_gpu_dispatcher, False),
    ('mf-gpu + rebalance', mf_gpu_dispatcher, True),
]
all_results = []
for target in [60_000, 85_000, 120_000]:
    print(f'-- Target {target}/h --')
    for vname, fn, rebal in variants:
        cfg = SimConfigV2(n_uavs=7080, sim_hours=(18.0, 19.0),
                          target_orders_per_hour=target, seed=0,
                          max_batch_size=3, batch_detour_sec=150.0,
                          enable_rebalance=rebal,
                          rebalance_max_moves=1500)
        t0 = time.time()
        k = run_simulation(cfg, fn, vname.strip(), verbose=False)
        print(f\"  {target:>7d}  {vname}  \"
              f\"{k['completion_pct']:>5.1f}%  \"
              f\"{k['mean_wait_pickup_s']:>7.1f}  \"
              f\"{k['mean_delivery_s']:>8.1f}  \"
              f\"{k['peak_pending']:>6d}  \"
              f\"{k['mean_batch_size']:>10.2f}  \"
              f\"{k['fleet_util_mean']*100:>4.1f}%  \"
              f\"{k['n_deadhead_total']:>8d}  \"
              f\"{k['mean_dispatch_ms']:>8.1f}  \"
              f\"{k['sim_wallclock_s']:>7.1f}\")
        all_results.append((target, vname.strip(), k))

print()
print('=' * 110)
print(' Comparison vs paper Table 3: paper 4h, batching + rebalance, ~96% completion')
print('=' * 110)
for target in [60_000, 85_000, 120_000]:
    g_no = next(k for t,n,k in all_results if t==target and n=='greedy')
    g_rb = next(k for t,n,k in all_results if t==target and n=='greedy + rebalance')
    m_no = next(k for t,n,k in all_results if t==target and n=='mf-gpu')
    m_rb = next(k for t,n,k in all_results if t==target and n=='mf-gpu + rebalance')
    print(f'{target}/h: greedy={g_no[\"completion_pct\"]:.1f}% '
          f'greedy+rb={g_rb[\"completion_pct\"]:.1f}% '
          f'mf={m_no[\"completion_pct\"]:.1f}% '
          f'mf+rb={m_rb[\"completion_pct\"]:.1f}%   '
          f'(Δrb_greedy={g_rb[\"completion_pct\"]-g_no[\"completion_pct\"]:+.1f}pp, '
          f'ΔMF over greedy+rb={m_rb[\"completion_pct\"]-g_rb[\"completion_pct\"]:+.1f}pp)')
"

echo
echo "=== Done ==="
date
