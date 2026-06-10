#!/bin/bash
#SBATCH --job-name=db-build
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-117%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/db_%a.out
#SBATCH --error=slurm_logs/db_%a.err

set -euo pipefail
cd /leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/wave_test_large
mkdir -p slurm_logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 -u build_database.py --source-idx ${SLURM_ARRAY_TASK_ID}
