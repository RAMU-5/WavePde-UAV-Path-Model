#!/bin/bash
#SBATCH --job-name=mf_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D25_028
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/gpu_bench_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D25_028/Levistone/tools/scheduling/gpu_bench_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone

PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python

echo "=== Node info ==="
hostname
nvidia-smi -L
$PY -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo
echo "=== Phase 1: torch parity + scaling benchmark ==="
# Need a slightly bigger sizes list for the GPU run since we have memory now.
$PY -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import numpy as np, time, torch
from mf_scheduler_torch import (mf_hybrid_lsa_assignment_t, MFConfigT)
from mf_scheduler import mf_hybrid_lsa_assignment, MFConfig

print('=' * 72)
print('PARITY: numpy vs torch-GPU on K=1 hybrid-LSA')
print('=' * 72)
rng = np.random.default_rng(0)
for t in range(5):
    C = rng.uniform(1.0, 10.0, (10, 10))
    cfg_np = MFConfig(theta0=3.0, gamma=0.99, n_iter=300, eta=0.4,
                      perturb=0.1, seed=t, update='jacobi')
    col_np, _ = mf_hybrid_lsa_assignment(C, lam_one=50.0, cfg=cfg_np)
    cfg_g = MFConfigT(theta0=3.0, gamma=0.99, n_iter=300, eta=0.4,
                      perturb=0.1, seed=t, device='cuda')
    col_g, _ = mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_g)
    np_cost = float(C[np.arange(10), col_np].sum())
    g_cost = float(C[np.arange(10), col_g].sum())
    print(f'  trial {t}: numpy={np_cost:.3f} gpu={g_cost:.3f} '
          f\"  {('OK' if abs(np_cost-g_cost)/np_cost < 0.15 else 'DIFF')}\")

print()
print('=' * 72)
print('SCALING: torch CPU vs torch GPU wall-clock per dispatch (60 iter)')
print('=' * 72)
print(f\"  {'N×M':>10s}  {'cpu/ms':>10s}  {'gpu/ms':>10s}  {'speedup':>9s}\")
sizes = [(30,30),(100,100),(250,100),(500,200),(1000,300),
         (2000,500),(4000,500),(7000,500)]
for N, M in sizes:
    C = rng.uniform(1.0, 100.0, (N, M))

    cfg_t = MFConfigT(theta0=10.0, gamma=0.97, n_iter=60, eta=0.4,
                      perturb=0.1, seed=0, device='cpu')
    t0 = time.time()
    mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_t)
    cpu_ms = (time.time() - t0) * 1000

    cfg_g = MFConfigT(theta0=10.0, gamma=0.97, n_iter=60, eta=0.4,
                      perturb=0.1, seed=0, device='cuda')
    # warmup
    mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_g)
    torch.cuda.synchronize()
    t0 = time.time()
    mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_g)
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t0) * 1000

    sp = cpu_ms / gpu_ms
    print(f'  {N}×{M:<3d}     {cpu_ms:>10.1f}  {gpu_ms:>10.1f}  {sp:>8.1f}×')
"

echo
echo "=== Phase 2: rolling_sim with GPU MF dispatcher ==="
# Patch rolling_sim to optionally use GPU
$PY -c "
import sys; sys.path.insert(0, 'tools/scheduling')
import numpy as np, time
from rolling_sim import SimConfig, run_simulation, dispatch_greedy
from mf_scheduler_torch import mf_hybrid_lsa_assignment_t, MFConfigT

def dispatch_mf_gpu(idle_uav_ids, idle_uav_stations,
                    pending_order_ids, pending_pickups, pending_customers,
                    pending_ages, T_eff, age_weight=0.5, seed=0):
    from rolling_sim import build_dispatch_cost
    if len(idle_uav_ids) == 0 or len(pending_order_ids) == 0:
        return []
    N = len(idle_uav_ids); M = len(pending_order_ids)
    C, feas = build_dispatch_cost(idle_uav_stations,
                                   pending_pickups, pending_customers, T_eff)
    age_bonus = age_weight * pending_ages.astype(np.float64)
    C_eff = C - age_bonus[None, :]
    sentinel = 5e5
    Cext = np.hstack([C_eff, np.full((N, 1), sentinel)])
    cfg = MFConfigT(theta0=50.0, gamma=0.97, n_iter=60, eta=0.4,
                    perturb=0.1, tol=1e-5, seed=seed, device='cuda')
    task_of_uav, _ = mf_hybrid_lsa_assignment_t(Cext, lam_one=200.0, cfg=cfg)
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

# Run load sweep on GPU for greedy comparison
print('LOAD SWEEP — GPU MF vs greedy (1h window, 500 UAV)')
print('=' * 90)
print(f\"{'load/hr':>8s}  {'method':>10s}  {'orders':>6s}  {'compl%':>6s}  \"
      f\"{'wait/s':>7s}  {'deliv/s':>8s}  {'pkpend':>6s}  {'disp/ms':>8s}  {'wall/s':>7s}\")
for target in [3000, 6000, 9000, 12000]:
    cfg = SimConfig(n_uavs=500, sim_hours=(18.0, 19.0),
                    target_orders_per_hour=target, seed=0)
    for name, disp in [('greedy', dispatch_greedy),
                       ('mf-gpu', dispatch_mf_gpu)]:
        k = run_simulation(cfg, disp, name, verbose=False)
        print(f\"{target:>8d}  {name:>10s}  {k['n_orders']:>6d}  \"
              f\"{k['completion_pct']:>5.1f}%  \"
              f\"{k['mean_wait_pickup_s']:>7.1f}  \"
              f\"{k['mean_delivery_s']:>8.1f}  \"
              f\"{k['peak_pending']:>6d}  \"
              f\"{k['mean_dispatch_ms']:>8.2f}  \"
              f\"{k['sim_wallclock_s']:>7.2f}\")
"

echo
echo "=== Done ==="
date
