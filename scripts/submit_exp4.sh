#!/bin/bash
#SBATCH --job-name=exp4-wind
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_logs/exp4_%j.out
#SBATCH --error=output/slurm_logs/exp4_%j.err

# E4.3 spatially varying wind on the 50 m domain: wave vs wind-aware
# Dijkstra under vortex wind + reciprocity-vs-divergence sweep.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
venv/bin/python -u experiments/exp4_wind.py
