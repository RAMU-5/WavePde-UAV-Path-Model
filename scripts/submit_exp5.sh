#!/bin/bash
#SBATCH --job-name=exp5-broadband
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm_logs/exp5_%j.out
#SBATCH --error=output/slurm_logs/exp5_%j.err

# E4.2 broadband ablation: short/long/broadband path tracing on the
# 20 hardest (long-lambda) pairs of the 5 m domain.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
venv/bin/python -u -m experiments.exp5_broadband
