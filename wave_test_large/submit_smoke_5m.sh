#!/bin/bash
#SBATCH --job-name=wave-5m-smoke
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/smoke_5m_%j.out
#SBATCH --error=slurm_logs/smoke_5m_%j.err

# 5 m-grid wave smoke test (one source, one solve).
# Domain: 1333 × 1443 × 42 = 80.8 M cells, ~10-15 GB GPU memory.
# Expected: 25-40 min wall on A100.

set -euo pipefail
cd /leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large
mkdir -p slurm_logs

echo "=== 5m smoke test on $(hostname) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

VENV=/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large/venv
# jax[cuda12] must be installed from the LOGIN node (compute nodes
# have no outbound network). One-time setup:
#   $VENV/bin/pip install --no-cache-dir "jax[cuda12]"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u smoke_5m.py --domain domain_5m.npz --src-idx 48

echo "=== done ==="
date
