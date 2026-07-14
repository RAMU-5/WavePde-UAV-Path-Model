#!/bin/bash
#SBATCH --job-name=wave-1src
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-117%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm_logs/source_%a.out
#SBATCH --error=output/slurm_logs/source_%a.err

# Each array task solves wave for one source station.
# %32 caps concurrent tasks at 32 to avoid overloading the JAX cache dir
# during initial compile. After ~1 min the cache is populated and all
# subsequent tasks read it instantly.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
mkdir -p output/slurm_logs

echo "=== task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
date
echo "Python: $(which python3)"

# Limit JAX to 1 thread per task; we already have 4 cpus reserved which
# helps the BLAS-side ops without spawning 128 threads on a shared node.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"

python3 -u -m model.solve_one_source --source-idx ${SLURM_ARRAY_TASK_ID} --no-full

echo "=== done $SLURM_ARRAY_TASK_ID ==="
date
