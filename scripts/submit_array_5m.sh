#!/bin/bash
#SBATCH --job-name=wave-5m-array
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=0-117%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/slurm_logs/src5m_%a.out
#SBATCH --error=output/slurm_logs/src5m_%a.err

# Build the 118 × 118 forward T-matrix at 5 m resolution.
# Each task = one source on its own A100, ~3-4 min wall (measured
# 50.2 s for JIT-compile + 4500-step solve in smoke_5m, plus ~30s
# overheads). %32 caps concurrent tasks at 32 to reuse the JAX
# compile cache across waves.
# Total wall time: ~12-15 min for all 118 sources at 32-way concurrency.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv
# jax[cuda12] must be pre-installed from the login node.

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u -m model.solve_one_source \
    --domain domain_5m.npz \
    --source-idx ${SLURM_ARRAY_TASK_ID} \
    --no-full \
    --out T_source_5m_${SLURM_ARRAY_TASK_ID}.npz

echo "=== done ${SLURM_ARRAY_TASK_ID} ==="
date
