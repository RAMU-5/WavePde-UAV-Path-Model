#!/bin/bash
#SBATCH --job-name=back-5m-val
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=output/slurm_logs/back5m_%j.out
#SBATCH --error=output/slurm_logs/back5m_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== 5m backward validation on $(hostname) ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1

VENV=/leonardo_work/EUHPC_D25_028/Levistone/venv

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

$VENV/bin/python -u validation/validate_backward_real.py \
    --domain domain_5m.npz \
    --pair-file pair_matrix_5m.npz \
    --n-pairs 6 \
    --n-origins-per-dst 8 \
    --min-t-fwd 80

echo "=== done ==="
date
