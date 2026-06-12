#!/bin/bash
#SBATCH --job-name=exp6-closure
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_logs/exp6_%j.out
#SBATCH --error=output/slurm_logs/exp6_%j.err

# E4.5 coupled-iteration convergence (5 inits x 10 pairs) +
# E4.4 wind-inversion closure (vortex truth -> tracks -> recovered
# field -> re-solve) on the 50 m domain.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
venv/bin/python -u experiments/exp6_closure.py
