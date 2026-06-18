#!/bin/bash
#SBATCH --job-name=kuwait-stcheck
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=output/slurm_logs/kuwait_stcheck_%j.out
#SBATCH --error=output/slurm_logs/kuwait_stcheck_%j.err
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python3
echo "=== $(hostname) ==="; date
$PY -u scripts/viz_stations_kuwait.py --domain kuwait_stations_v2.npz
$PY -u scripts/viz_stations_kuwait.py --domain kuwait_stations.npz
echo "=== done ==="; date
