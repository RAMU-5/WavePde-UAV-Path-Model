#!/bin/bash
#SBATCH --job-name=exp8-scheduler
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_logs/exp8_%j.out
#SBATCH --error=output/slurm_logs/exp8_%j.err

# E5.1 scheduler comparison (hungarian / greedy / mf_soft / mf_hybrid)
# on the real v2 cost matrix, sizes 10..5000. Pure CPU.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cpu
venv/bin/python -u experiments/exp8_scheduler.py
