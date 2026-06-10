#!/bin/bash
#SBATCH --job-name=sk_wind
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=output/slurm_logs/sk_wind_%j.out
#SBATCH --error=output/slurm_logs/sk_wind_%j.err

set -euo pipefail
mkdir -p output/slurm_logs

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv

cd /leonardo_work/EUHPC_D25_028/Levistone

# Make sure jax can find CUDA (the GPU node has CUDA12 toolkit)
unset JAX_PLATFORMS

$VENV/bin/python experiments/exp_sk_wind.py \
    --t 3000 \
    --hi 320 --wi 320 \
    --nx 384 --ny 384 --nz 48 \
    --dx 4.0 \
    --n_steps 4500 \
    --wind_scale 15.0 \
    --src 236 228 4 \
    --dst 1304 1304 4 \
    --vis_z 6 \
    --reg_low_factor 0.05 \
    --reg_z_low 15.0 \
    --reg_z_full 20.0
