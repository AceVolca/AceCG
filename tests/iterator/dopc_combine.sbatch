#!/bin/bash
#SBATCH --job-name=acecg_iter_combine
#SBATCH --account=pi-gavoth
#SBATCH --partition=gavoth
#SBATCH --qos=gavoth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch dopc_combine.sbatch <config.json> <output_root>" >&2
  exit 2
fi

CONFIG="$1"
OUTROOT="$2"

ACECG_ROOT="${ACECG_ROOT:-/beagle3/gavoth/weizhixue/programs/AceCG}"
PYTHON_BIN="${ACECG_PYTHON:-/beagle3/gavoth/weizhixue/conda/envs/mscgtest/bin/python}"
export PYTHONPATH="${ACECG_PYTHONPATH:-$(dirname "${ACECG_ROOT}")}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export ACECG_BLAS_THREADS="${ACECG_BLAS_THREADS:-${OPENBLAS_NUM_THREADS}}"
export ACECG_TRACE_STEPS="${ACECG_TRACE_STEPS:-0}"
export ACECG_TQDM="${ACECG_TQDM:-1}"

mkdir -p "$OUTROOT/combine"

# Combine step is the same for solver and iterator —
# it sums per-window XtX/XtY matrices and solves via FMMatrixSolver.
"$PYTHON_BIN" "${ACECG_ROOT}/tests/solver/run_fm_combine.py" \
  --config "$CONFIG" \
  --windows-root "$OUTROOT/windows" \
  --outdir "$OUTROOT/combine" \
  --blas-threads "${ACECG_BLAS_THREADS}"
