#!/usr/bin/env bash
#SBATCH --job-name=aura-traj
#SBATCH --account=euhpc_d25_028
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --output=output/runs/slurm/traj-%j.out
#SBATCH --error=output/runs/slurm/traj-%j.err

# Aura Stage 1 — all-pairs nominal trajectory build on a CPU compute node.
# Avoids hammering the login node with the 5–6 min scipy.csgraph Dijkstra +
# 9900 geodesic backtraces. Submit:
#     sbatch aura/slurm/build_trajectories.slurm
#
# Env: assumes the existing yli00001 miniconda Python is on PATH. The script
# enters the project root, runs the CLI, and dumps logs to output/runs/slurm/.

set -euo pipefail

PROJECT_ROOT="/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone"
cd "$PROJECT_ROOT"
mkdir -p output/runs/slurm

ASSETS=${ASSETS:-output/runs/v1}
OUT=${OUT:-output/runs/v1/traj}

echo "[slurm] node=$(hostname) jobid=${SLURM_JOB_ID:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"
echo "[slurm] PWD=$PWD"
echo "[slurm] ASSETS=$ASSETS  OUT=$OUT"
echo "[slurm] python=$(which python)"
date -u +"[slurm] start %Y-%m-%dT%H:%M:%SZ"

# Hint scipy/numpy to use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

COVERAGE_M=${COVERAGE_M:-5000}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}

python -u -m aura.cli.build_trajectories \
  --assets "$ASSETS" --out "$OUT" \
  --coverage-m "$COVERAGE_M" --workers "$WORKERS"

date -u +"[slurm] done  %Y-%m-%dT%H:%M:%SZ"
