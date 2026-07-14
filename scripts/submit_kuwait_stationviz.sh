#!/bin/bash
#SBATCH --job-name=kuwait-stationviz
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/slurm_logs/kuwait_stationviz_%j.out
#SBATCH --error=output/slurm_logs/kuwait_stationviz_%j.err

# (1) re-place 64 stations in inter-building gaps; (2) visualise current
# routes + collisions. numpy/matplotlib only; runs on a compute node.
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python3

echo "=== $(hostname) ==="; date
echo "--- placing stations in gaps ---"
$PY -u -m scripts.place_stations_gaps --n 64
echo "--- visualising collisions (current routes) ---"
$PY -u -m scripts.viz_collisions_kuwait
echo "=== done ==="; date
