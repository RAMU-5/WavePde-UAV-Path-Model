#!/bin/bash
#SBATCH --job-name=mf_batch
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D25_028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_batch_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_batch_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone

PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python

echo "=== Node ==="
hostname; nvidia-smi -L

echo
echo "=== Paper-scale rolling sim WITH BATCHING (max_batch=3) ==="
echo "=== 7080 UAV, 118 stations, 1h dinner peak ==="
$PY -u -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import time
from rolling_sim_v2 import (SimConfigV2, run_simulation,
                              greedy_dispatcher, mf_gpu_dispatcher)

print('=' * 100)
print(f\"  {'load/h':>7s}  {'method':>10s}  {'orders':>6s}  {'delivered':>9s}  \"
      f\"{'compl%':>6s}  {'wait/s':>7s}  {'deliv/s':>8s}  {'pkpend':>6s}  \"
      f\"{'mean_batch':>10s}  {'util':>5s}  {'disp/ms':>8s}  {'wall/s':>7s}\")
print('=' * 100)
results = []
for target in [60_000, 85_000, 120_000]:
    for name, fn in [('greedy', greedy_dispatcher),
                     ('mf-gpu', mf_gpu_dispatcher)]:
        cfg = SimConfigV2(n_uavs=7080, sim_hours=(18.0, 19.0),
                          target_orders_per_hour=target, seed=0,
                          max_batch_size=3, batch_detour_sec=150.0)
        t0 = time.time()
        k = run_simulation(cfg, fn, name, verbose=False)
        print(f\"  {target:>7d}  {name:>10s}  {k['n_orders']:>6d}  \"
              f\"{k['n_delivered']:>9d}  \"
              f\"{k['completion_pct']:>5.1f}%  \"
              f\"{k['mean_wait_pickup_s']:>7.1f}  \"
              f\"{k['mean_delivery_s']:>8.1f}  \"
              f\"{k['peak_pending']:>6d}  \"
              f\"{k['mean_batch_size']:>10.2f}  \"
              f\"{k['fleet_util_mean']*100:>4.1f}%  \"
              f\"{k['mean_dispatch_ms']:>8.1f}  \"
              f\"{k['sim_wallclock_s']:>7.1f}\")
        results.append((target, name, k))

print()
print('=' * 100)
print(' COMPARISON vs paper Table 3 (7080 UAV, 4h window, batching enabled)')
print(' paper reports: 342498 orders / 4h, mean_batch ~2.21, ~96% completion')
print('=' * 100)
for target in [60_000, 85_000, 120_000]:
    g = next(k for t,n,k in results if t==target and n=='greedy')
    m = next(k for t,n,k in results if t==target and n=='mf-gpu')
    dcompl = m['completion_pct'] - g['completion_pct']
    dwait = (m['mean_wait_pickup_s'] - g['mean_wait_pickup_s']) / \
            max(g['mean_wait_pickup_s'], 1e-9) * 100
    print(f\"  {target}/h:  greedy {g['completion_pct']:.1f}%  vs  mf {m['completion_pct']:.1f}%  \"
          f\"(Δ{dcompl:+.1f}pp)   wait Δ{dwait:+.1f}%   \"
          f\"batch~{g['mean_batch_size']:.2f}/{m['mean_batch_size']:.2f}\")
"

echo
echo "=== Done ==="
date
