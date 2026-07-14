#!/bin/bash
#SBATCH --job-name=exp3-wavelength
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm_logs/exp3_%j.out
#SBATCH --error=output/slurm_logs/exp3_%j.err

# E4.1 wavelength sweep: 6 tau x 12 GT sources on the 5 m domain
# + per-tau empty-domain bias companions; checkpointed, resumable.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
venv/bin/python -u -m experiments.exp3_wavelength_gap
