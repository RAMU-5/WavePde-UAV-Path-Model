#!/bin/bash
#SBATCH --job-name=exp1-dij-5m
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --output=output/slurm_logs/exp1_5m_%j.out
#SBATCH --error=output/slurm_logs/exp1_5m_%j.err

# E2.1 ground truth on the 5 m domain (80.8 M cells, ~1.8 G directed
# edges): anisotropic Dijkstra + metric-aware string pulling from 12
# spread source stations. CPU-only workload (scipy); the GPU is idle —
# boost is the only partition our QOS covers. Memory peak ~100 GB
# during the COO->CSR conversion.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== exp1 dijkstra 5m on $(hostname) ==="
date

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

$VENV/bin/python -u -m experiments.exp1_dijkstra_gt \
    --domain domain_5m.npz \
    --sources 0,10,20,30,40,50,60,70,80,90,100,110

echo "=== done ==="
date
