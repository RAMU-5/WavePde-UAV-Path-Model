#!/bin/bash
#SBATCH --job-name=mf_paper
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D25_028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_scale_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/paper_scale_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone

PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python

echo "=== Node ==="
hostname
nvidia-smi -L
echo

echo "=== Paper-scale sanity (small parity) ==="
$PY -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from mf_scheduler_torch import (mf_hybrid_lsa_assignment_t,
                                mf_hybrid_lsa_fast_t, MFConfigT)
# verify fast path matches the original on small case
rng = np.random.default_rng(0)
print('Fast-path parity check (8x8):')
for t in range(5):
    C = rng.uniform(1.0, 10.0, (8, 8))
    cfg = MFConfigT(theta0=3.0, gamma=0.99, n_iter=300, eta=0.4,
                    perturb=0.1, seed=t, device='cuda')
    col_full, _ = mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg)
    col_fast, _ = mf_hybrid_lsa_fast_t(C, lam_one=50.0, cfg=cfg)
    hu = linear_sum_assignment(C)[1]
    f_cost = C[np.arange(8), col_full].sum()
    g_cost = C[np.arange(8), col_fast].sum()
    h_cost = C[np.arange(8), hu].sum()
    print(f'  t{t}: hungarian={h_cost:.2f} full={f_cost:.2f} fast={g_cost:.2f}')
"

echo
echo "=== Fast-path scaling check (huge sizes) ==="
$PY -c "
import sys, time, torch, numpy as np
sys.path.insert(0, 'tools/scheduling')
from mf_scheduler_torch import mf_hybrid_lsa_fast_t, MFConfigT

rng = np.random.default_rng(0)
print(f\"  {'N×M':>12s}  {'gpu/ms':>10s}\")
for N, M in [(1000,1000),(3000,3000),(5000,3000),(7000,5000),(7000,10000)]:
    C = rng.uniform(1.0, 100.0, (N, M))
    cfg = MFConfigT(theta0=20.0, gamma=0.97, n_iter=60, eta=0.4,
                    perturb=0.1, seed=0, device='cuda')
    # warmup
    mf_hybrid_lsa_fast_t(C, lam_one=200.0, cfg=cfg); torch.cuda.synchronize()
    t0 = time.time()
    mf_hybrid_lsa_fast_t(C, lam_one=200.0, cfg=cfg); torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    print(f'  {N}×{M:<5d}    {ms:>10.1f}')
"

echo
echo "=== Phase: paper-scale rolling sim (7080 UAV, 1h dinner peak) ==="
$PY -c "
import sys, time
import numpy as np
sys.path.insert(0, 'tools/scheduling')
from rolling_sim import (SimConfig, run_simulation, dispatch_greedy,
                          build_dispatch_cost)
from mf_scheduler_torch import mf_hybrid_lsa_fast_t, MFConfigT

# GPU MF dispatcher using the memory-efficient fast path
def dispatch_mf_gpu_fast(idle_uav_ids, idle_uav_stations,
                          pending_order_ids, pending_pickups, pending_customers,
                          pending_ages, T_eff, age_weight=0.5, seed=0):
    if len(idle_uav_ids) == 0 or len(pending_order_ids) == 0:
        return []
    N, M = len(idle_uav_ids), len(pending_order_ids)
    C, feas = build_dispatch_cost(idle_uav_stations,
                                   pending_pickups, pending_customers, T_eff)
    age_bonus = age_weight * pending_ages.astype(np.float64)
    C_eff = C - age_bonus[None, :]
    sentinel = 5e5
    Cext = np.hstack([C_eff, np.full((N, 1), sentinel)])
    cfg = MFConfigT(theta0=50.0, gamma=0.97, n_iter=60, eta=0.4,
                    perturb=0.1, tol=1e-5, seed=seed, device='cuda')
    task_of_uav, _ = mf_hybrid_lsa_fast_t(Cext, lam_one=200.0, cfg=cfg)
    leg1 = T_eff[idle_uav_stations[:, None], pending_pickups[None, :]]
    leg2 = T_eff[pending_pickups, pending_customers]
    assignments = []
    for i in range(N):
        j = int(task_of_uav[i])
        if j < 0 or j >= M:
            continue
        if not feas[i, j]:
            continue
        assignments.append((int(idle_uav_ids[i]), int(pending_order_ids[j]),
                            float(leg1[i, j]), float(leg2[j])))
    return assignments

print('=' * 90)
print('PAPER-SCALE ROLLING SIM — 7080 UAV, 118 stations, 1h dinner peak')
print('Targets: paper Table 3 is 342498 orders / 4h = 85k orders/h average')
print('We run 1h window at 85k orders/h to test peak-hour dispatch behaviour')
print('=' * 90)

for target_per_hour in [60_000, 85_000, 120_000]:
    print(f'\\n--- Target {target_per_hour} orders/h ---')
    cfg = SimConfig(n_uavs=7080, sim_hours=(18.0, 19.0),
                    target_orders_per_hour=target_per_hour, seed=0)
    for name, disp in [('greedy', dispatch_greedy),
                       ('mf-gpu', dispatch_mf_gpu_fast)]:
        t0 = time.time()
        k = run_simulation(cfg, disp, name, verbose=False)
        print(f'  [{name:8s}] orders={k[\"n_orders\"]:6d}  '
              f'compl={k[\"completion_pct\"]:.1f}%  '
              f'wait={k[\"mean_wait_pickup_s\"]:.1f}s  '
              f'deliv={k[\"mean_delivery_s\"]:.1f}s  '
              f'pkpend={k[\"peak_pending\"]:5d}  '
              f'util={k[\"fleet_util_mean\"]*100:.1f}%  '
              f'disp/ms={k[\"mean_dispatch_ms\"]:.1f}  '
              f'wall={k[\"sim_wallclock_s\"]:.1f}s')
"

echo
echo "=== Done ==="
date
