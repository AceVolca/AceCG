#!/usr/bin/env bash
# Submit the full FM solver pipeline: per-window accumulation + combine.
# Usage: bash submit_jobs.sh [output_root]
set -euo pipefail

ACECG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ACECG_ROOT
export ACECG_PYTHON="${ACECG_PYTHON:-/beagle3/gavoth/weizhixue/conda/envs/mscgtest/bin/python}"
CFG="${ACECG_ROOT}/tests/data/dopc_fm_input_acecg.json"
RUNROOT="${1:-${ACECG_ROOT}/tests/_runs/solver_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUNROOT}"

echo "[SOLVER] runroot=${RUNROOT}"

WINDOWS_SBATCH_ARGS=()
[[ -n "${ACECG_WINDOWS_EXCLUDE:-}" ]] && WINDOWS_SBATCH_ARGS+=(--exclude "${ACECG_WINDOWS_EXCLUDE}")
[[ -n "${ACECG_WINDOWS_NODELIST:-}" ]] && WINDOWS_SBATCH_ARGS+=(--nodelist "${ACECG_WINDOWS_NODELIST}")

COMBINE_SBATCH_ARGS=()
[[ -n "${ACECG_COMBINE_EXCLUDE:-}" ]] && COMBINE_SBATCH_ARGS+=(--exclude "${ACECG_COMBINE_EXCLUDE}")
[[ -n "${ACECG_COMBINE_NODELIST:-}" ]] && COMBINE_SBATCH_ARGS+=(--nodelist "${ACECG_COMBINE_NODELIST}")

WIN_JOB=$(sbatch --parsable --export=ALL "${WINDOWS_SBATCH_ARGS[@]}" \
  "${ACECG_ROOT}/tests/solver/dopc_windows.sbatch" "${CFG}" "${RUNROOT}")
echo "[SOLVER] windows job=${WIN_JOB}"

COMB_JOB=$(sbatch --parsable --dependency=afterok:${WIN_JOB} --export=ALL "${COMBINE_SBATCH_ARGS[@]}" \
  "${ACECG_ROOT}/tests/solver/dopc_combine.sbatch" "${CFG}" "${RUNROOT}")
echo "[SOLVER] combine job=${COMB_JOB}"

cat > "${RUNROOT}/metadata.json" <<EOF
{
  "runroot": "${RUNROOT}",
  "config": "${CFG}",
  "engine": "matrix",
  "env_name": "${CONDA_DEFAULT_ENV:-unknown}",
  "windows_job_id": "${WIN_JOB}",
  "combine_job_id": "${COMB_JOB}",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[SOLVER] metadata written to ${RUNROOT}/metadata.json"
echo "[SOLVER] ACECG_ROOT=${ACECG_ROOT}"
echo "[SOLVER] ACECG_PYTHON=${ACECG_PYTHON}"
echo "[SOLVER] Monitor: squeue -u \$USER ; sacct -j ${WIN_JOB},${COMB_JOB} --format=JobID,State,Elapsed"
