#!/bin/bash
#SBATCH --job-name=kuwait-wave
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/slurm_logs/kuwait_wave_%j.out
#SBATCH --error=output/slurm_logs/kuwait_wave_%j.err

# One wind-aware convected-wave solve on the Kuwait CBD 1024^2 domain.
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== $(hostname) ==="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u scripts/run_kuwait_wave.py \
    --domain kuwait_cbd_1024.npz \
    --wind 5 0 0 \
    --out kuwait_T.npz

echo "=== done ==="; date
