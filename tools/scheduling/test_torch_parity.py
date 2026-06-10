"""Verify torch CPU implementation matches numpy implementation on
the basic assignment / scheduling problems.

Also benchmarks CPU vs GPU (if CUDA available) on increasing problem sizes.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf_scheduler import (mean_field_assignment_2d, mf_hybrid_lsa_assignment,
                          MFConfig)
from mf_scheduler_torch import (mean_field_assignment_2d_t,
                                mf_hybrid_lsa_assignment_t,
                                MFConfigT)


def test_parity_k1(N=8, M=8, n_trials=5, seed=0):
    """Compare numpy vs torch CPU on K=1 hybrid-LSA assignment."""
    print("=" * 72)
    print(f"PARITY TEST  numpy vs torch-CPU   K=1 hybrid-LSA  "
          f"N={N} M={M} trials={n_trials}")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    all_match = True
    for t in range(n_trials):
        C = rng.uniform(1.0, 10.0, size=(N, M)).astype(np.float64)
        hu = linear_sum_assignment(C)[1]
        hu_cost = float(C[np.arange(N), hu].sum())

        # numpy: keep Seidel for compatibility with test_mf_basic
        cfg_np = MFConfig(theta0=3.0, gamma=0.99, n_iter=300, eta=0.4,
                          perturb=0.1, seed=t, update="jacobi")
        col_np, _ = mf_hybrid_lsa_assignment(C, lam_one=50.0, cfg=cfg_np)
        np_cost = float(C[np.arange(N), col_np].sum())

        # torch CPU
        cfg_t = MFConfigT(theta0=3.0, gamma=0.99, n_iter=300, eta=0.4,
                          perturb=0.1, seed=t, device="cpu")
        col_t, _ = mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_t)
        t_cost = float(C[np.arange(N), col_t].sum())

        match = abs(np_cost - t_cost) / max(hu_cost, 1e-9) < 0.10
        all_match = all_match and match
        print(f"  trial {t}: hungarian={hu_cost:.3f}  "
              f"numpy={np_cost:.3f}  torch-cpu={t_cost:.3f}  "
              f"{'OK' if match else 'DIFF'}")

    print(f"\n  numpy vs torch-CPU agreement: "
          f"{'PASS — within 10%' if all_match else 'FAIL'}")
    return all_match


def benchmark_scaling():
    """Benchmark torch CPU vs torch GPU (if available) across problem sizes."""
    has_gpu = torch.cuda.is_available()
    print("\n" + "=" * 72)
    print(f"WALL-CLOCK BENCHMARK  (CUDA available: {has_gpu})")
    if has_gpu:
        print(f"  device: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  memory: {gpu_mem:.1f} GB")
    print("=" * 72)

    sizes = [(30, 30), (100, 100), (250, 100), (500, 200),
             (800, 250)]
    rng = np.random.default_rng(0)

    print(f"  {'N×M':>10s}  {'numpy/ms':>9s}  {'torch-cpu/ms':>13s}  "
          f"{'torch-gpu/ms':>13s}  {'gpu_speedup':>11s}")
    print(f"  {'-'*10:>10s}  {'-'*9:>9s}  {'-'*13:>13s}  "
          f"{'-'*13:>13s}  {'-'*11:>11s}")
    for N, M in sizes:
        C = rng.uniform(1.0, 100.0, size=(N, M)).astype(np.float64)

        # numpy timing (only for small sizes to avoid blowing up wallclock)
        if N * M < 200_000:
            cfg_np = MFConfig(theta0=10.0, gamma=0.97, n_iter=60,
                              eta=0.4, perturb=0.1, seed=0, update="jacobi")
            t0 = time.time()
            mf_hybrid_lsa_assignment(C, lam_one=50.0, cfg=cfg_np)
            np_ms = (time.time() - t0) * 1000
            np_str = f"{np_ms:9.1f}"
        else:
            np_str = "       --"

        # torch CPU
        cfg_t = MFConfigT(theta0=10.0, gamma=0.97, n_iter=60,
                          eta=0.4, perturb=0.1, seed=0, device="cpu")
        t0 = time.time()
        mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_t)
        cpu_ms = (time.time() - t0) * 1000

        # torch GPU
        if has_gpu:
            cfg_g = MFConfigT(theta0=10.0, gamma=0.97, n_iter=60,
                              eta=0.4, perturb=0.1, seed=0, device="cuda")
            # warmup
            mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_g)
            torch.cuda.synchronize()
            t0 = time.time()
            mf_hybrid_lsa_assignment_t(C, lam_one=50.0, cfg=cfg_g)
            torch.cuda.synchronize()
            gpu_ms = (time.time() - t0) * 1000
            speedup = cpu_ms / gpu_ms
            gpu_str = f"{gpu_ms:13.1f}"
            sp_str = f"{speedup:10.1f}×"
        else:
            gpu_str = "         n/a"
            sp_str = "        n/a"

        size_str = f"{N}×{M}"
        print(f"  {size_str:>10s}  {np_str:>9s}  {cpu_ms:>13.1f}  "
              f"{gpu_str:>13s}  {sp_str:>11s}")


def main():
    ok = test_parity_k1()
    benchmark_scaling()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
