#!/bin/bash
#SBATCH --job-name=kuwait-post-v2
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/slurm_logs/kuwait_post_v2_%j.out
#SBATCH --error=output/slurm_logs/kuwait_post_v2_%j.err

# Post-process the v2 (gap-placed stations) path database:
# pair matrix + route figure, collision report, collision figure.
set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/leonardo_work/EUHPC_D25_028/miniconda3/bin/python3
DOM=kuwait_stations_v2.npz
DB=database_kuwait_v2

echo "=== $(hostname) ==="; date
$PY -u scripts/aggregate_kuwait.py       --domain $DOM --dbdir $DB
$PY -u scripts/check_collisions_kuwait.py --domain $DOM --dbdir $DB
$PY -u scripts/viz_collisions_kuwait.py   --domain $DOM --dbdir $DB
echo "=== done ==="; date
