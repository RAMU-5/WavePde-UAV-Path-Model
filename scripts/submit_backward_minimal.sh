#!/bin/bash
#SBATCH --job-name=back-min-val
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=output/slurm_logs/back_min_%j.out
#SBATCH --error=output/slurm_logs/back_min_%j.err

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== minimal numpy backward-symmetry test on $(hostname) ==="
date
/leonardo/home/userexternal/yli00001/miniconda3/bin/python -u -m experiments.validation.validate_backward_minimal
echo "=== done ==="
date
