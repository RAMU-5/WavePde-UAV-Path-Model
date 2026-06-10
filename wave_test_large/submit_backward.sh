#!/bin/bash
#SBATCH --job-name=wave-back-val
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/back_val_%j.out
#SBATCH --error=slurm_logs/back_val_%j.err

set -euo pipefail
cd /leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large
mkdir -p slurm_logs

echo "=== backward-wave validation on $(hostname) ==="
date

# Use the local venv on /leonardo_scratch (fast Lustre tier) instead
# of the home-conda env. Conda env startup hits site.py for hundreds
# of pkgs and stalls under load (5-15 min); the venv has 4 pkgs and
# starts in 0.02s.
VENV=/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large/venv

# JAX on GPU when available; falls back to CPU otherwise.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

$VENV/bin/python -u validate_backward_real.py \
    --pair-file pair_matrix.npz --n-pairs 6 --n-origins-per-dst 8 \
    --min-t-fwd 60

echo "=== done ==="
date
