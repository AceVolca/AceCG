#!/bin/bash
#SBATCH --job-name=acecg_solver_win
#SBATCH --account=pi-gavoth
#SBATCH --partition=gavoth
#SBATCH --qos=gavoth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=1-15
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch dopc_windows.sbatch <config.json> <output_root>" >&2
  exit 2
fi

CONFIG="$1"
OUTROOT="$2"
WINDOW="${SLURM_ARRAY_TASK_ID}"
WINDOW_PAD=$(printf "%03d" "${WINDOW}")

ACECG_ROOT="${ACECG_ROOT:-/beagle3/gavoth/weizhixue/programs/AceCG}"
PYTHON_BIN="${ACECG_PYTHON:-/beagle3/gavoth/weizhixue/conda/envs/mscgtest/bin/python}"
export PYTHONPATH="${ACECG_PYTHONPATH:-$(dirname "${ACECG_ROOT}")}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export ACECG_TRACE_STEPS="${ACECG_TRACE_STEPS:-0}"
export ACECG_TQDM="${ACECG_TQDM:-1}"

mkdir -p "$OUTROOT/windows/window_${WINDOW_PAD}"

ARGS=(
  --config "$CONFIG"
  --window "$WINDOW"
  --outdir "$OUTROOT/windows/window_${WINDOW_PAD}"
)

if [[ "${ACECG_PARALLEL:-1}" == "1" ]]; then
  ARGS+=(--parallel --n-parts "${ACECG_N_PARTS:-8}" --n-workers "${ACECG_N_WORKERS:-8}" --worker-blas-threads "${ACECG_WORKER_BLAS_THREADS:-1}")
fi

"$PYTHON_BIN" "${ACECG_ROOT}/tests/solver/run_fm_solver.py" "${ARGS[@]}"
