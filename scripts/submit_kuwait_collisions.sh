#!/bin/bash
#SBATCH --job-name=kuwait-collide
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm_logs/kuwait_collide_%j.out
#SBATCH --error=output/slurm_logs/kuwait_collide_%j.err

# Check whether traced UAV paths pass through building voxels.
# numpy-only; runs on a real compute node (boost is the budgeted
# partition) rather than the contended login/serial partition.
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
echo "=== $(hostname) ==="; date
/leonardo_work/EUHPC_D25_028/miniconda3/bin/python3 -u \
    scripts/check_collisions_kuwait.py
echo "=== done ==="; date
