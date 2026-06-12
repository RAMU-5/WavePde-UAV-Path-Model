#!/bin/bash
#SBATCH --job-name=exp7-efficiency
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_logs/exp7_%j.out
#SBATCH --error=output/slurm_logs/exp7_%j.err

# E3.1 adjoint accuracy + E3.2 multicolor crosstalk/colouring +
# E3.3 throughput (50m / 5m / 2.5m subdomain) + Fig.9 multirotor data.

set -euo pipefail
cd /leonardo_work/EUHPC_D25_028/Levistone
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# fresh compile cache so the "cold" timing is a true compile, not a
# disk-cache hit (solve_all setdefaults a persistent cache dir)
export JAX_COMPILATION_CACHE_DIR=$(mktemp -d /tmp/jaxcache_XXXX)
venv/bin/python -u experiments/exp7_efficiency.py
