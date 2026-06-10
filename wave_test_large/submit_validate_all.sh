#!/bin/bash
#SBATCH --job-name=wave-val-all
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_logs/val_all_%j.out
#SBATCH --error=slurm_logs/val_all_%j.err

# Real-domain validation of all six new features (#2/#3/#4/#5/#11/#16)
# in one slurm job. Total wall time ~3-4 minutes after JIT warm-up.

set -euo pipefail
cd /leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large
mkdir -p slurm_logs

echo "=== validate_all_real on $(hostname) ==="
date

VENV=/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large/venv
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

$VENV/bin/python -u validate_all_real.py

echo "=== done ==="
date
