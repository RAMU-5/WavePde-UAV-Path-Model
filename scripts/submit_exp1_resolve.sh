#!/bin/bash
#SBATCH --job-name=exp1-resolve-5m
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm_logs/exp1rs_%j.out
#SBATCH --error=output/slurm_logs/exp1rs_%j.err

# Full 118-source 5m wave matrix with the corrected (dx-scaled) peak
# threshold; per-source checkpointing, resumable.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
venv/bin/python -u -m experiments.exp1_resolve_5m
