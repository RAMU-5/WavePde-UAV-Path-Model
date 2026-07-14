#!/bin/bash
#SBATCH --job-name=exp0-analytic
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=output/slurm_logs/exp0_%j.out
#SBATCH --error=output/slurm_logs/exp0_%j.err

# Tier-1 analytic accuracy suite (E1.1–E1.4, paper §5.2 / Fig.1–3).
# ~25 JIT-compiled configs; solves themselves are seconds each on A100.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== exp0 analytic suite on $(hostname) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u -m experiments.exp0_analytic

echo "=== done ==="
date
