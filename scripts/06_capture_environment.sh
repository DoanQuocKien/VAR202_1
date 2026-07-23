#!/usr/bin/env bash
# Snapshot the exact environment used to produce results: conda/pip package
# versions, GPU driver/CUDA version, and the exact commit of the baseline
# repo (it's a live clone from GitHub, so this pins down which version was
# actually used). Run this any time before a submission you might need to
# defend later - contest rule 10.3 says top-ranked teams may be asked to
# prove reproducibility with exactly this kind of information.
#
# Usage:
#   bash scripts/06_capture_environment.sh

set -euo pipefail

ENV_NAME="var2026-3dgs"
OUT_DIR="environment_snapshot"
mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

echo "== Capturing conda package list =="
if command -v conda >/dev/null 2>&1; then
  conda list -n "${ENV_NAME}" --export > "${OUT_DIR}/conda_list_${TS}.txt" \
    || echo "  [warn] could not export conda list for env '${ENV_NAME}'"
  echo "== Capturing pip freeze =="
  conda run -n "${ENV_NAME}" pip freeze > "${OUT_DIR}/pip_freeze_${TS}.txt" \
    || echo "  [warn] could not run pip freeze in env '${ENV_NAME}'"
else
  echo "  [warn] conda not found, skipping package list capture"
fi

echo "== Capturing GPU/driver info =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_DIR}/nvidia_smi_${TS}.txt" 2>&1
else
  echo "nvidia-smi not found on this machine" > "${OUT_DIR}/nvidia_smi_${TS}.txt"
fi

echo "== Capturing baseline repo commit =="
{
  if [ -d external/gaussian-splatting/.git ]; then
    echo "remote: $(git -C external/gaussian-splatting remote get-url origin 2>/dev/null || echo unknown)"
    echo "commit: $(git -C external/gaussian-splatting rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "submodules:"
    git -C external/gaussian-splatting submodule status 2>/dev/null || echo "  (none found)"
  else
    echo "external/gaussian-splatting is not a git repo (run scripts/00_setup_env.sh first)"
  fi
} > "${OUT_DIR}/baseline_commit_${TS}.txt"

echo ""
echo "Done. Snapshot written to ${OUT_DIR}/:"
ls -la "${OUT_DIR}" | grep "${TS}" || true
