#!/bin/bash
#SBATCH --job-name=kuwait-db-v2
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=0-63%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=output/slurm_logs/kuwait_db_v2_%a.out
#SBATCH --error=output/slurm_logs/kuwait_db_v2_%a.err

# 64-station path database on the Kuwait CBD 1024^2 domain.
# Each task = one source: baseline no-wind wave solve + -∇T path trace
# to every valid [700,3000]m destination. %16 reuses the JAX cache.
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u scripts/build_database_kuwait.py \
    --domain kuwait_stations_v2.npz \
    --outdir database_kuwait_v2 \
    --source-idx ${SLURM_ARRAY_TASK_ID}

echo "=== done ${SLURM_ARRAY_TASK_ID} ==="; date
