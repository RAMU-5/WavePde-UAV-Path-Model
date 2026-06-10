#!/usr/bin/env bash
# Aura Stage 1 install script.
# Run interactively after reviewing — does NOT run automatically.
#
# Assumes the existing yli00001 miniconda env at /leonardo/home/userexternal/yli00001/miniconda3
# is the target. Creates a dedicated env "aura-stage1" so SCALED env is not perturbed.

set -euo pipefail

ENV_NAME="aura-stage1"
PY_VERSION="3.11"  # 3.13 is too new for some geo libs (rasterio wheels)

CONDA_BIN="/leonardo/home/userexternal/yli00001/miniconda3/bin/conda"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[aura] creating conda env $ENV_NAME (python=$PY_VERSION)"
"$CONDA_BIN" create -y -n "$ENV_NAME" python="$PY_VERSION"

source "/leonardo/home/userexternal/yli00001/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[aura] tier A core (geo stack first via conda-forge — wheels are unreliable)"
"$CONDA_BIN" install -y -c conda-forge \
  numpy scipy matplotlib pandas shapely pyproj \
  geopandas rasterio rioxarray zarr numcodecs networkx \
  pyyaml tqdm rich

echo "[aura] tier A pip-only"
pip install \
  "osmnx>=1.9" \
  "agd>=0.2.13" \
  "do-mpc>=4.6" "casadi>=3.6" \
  "alns>=5.1" \
  "scikit-learn-extra>=0.3" \
  "typer>=0.12"

echo "[aura] dev"
pip install pytest pytest-cov pre-commit ruff

echo "[aura] sanity"
python - <<'PY'
import importlib
mods = ["numpy","scipy","shapely","geopandas","rasterio","pyproj",
        "osmnx","agd","do_mpc","casadi","alns","sklearn_extra",
        "yaml","zarr"]
ok = []
miss = []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        miss.append((m, type(e).__name__))
print("OK:", ", ".join(ok))
if miss:
    print("MISSING:", miss)
    raise SystemExit(1)
PY

echo "[aura] done. activate with: conda activate $ENV_NAME"
