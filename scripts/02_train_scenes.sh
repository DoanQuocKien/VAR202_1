#!/usr/bin/env bash
# Train the (unmodified) baseline 3D Gaussian Splatting model for every scene
# listed in a scenes config yaml.
#
# Usage:
#   bash scripts/02_train_scenes.sh [scenes_config.yaml] [extra train.py args...]
#
# Examples:
#   bash scripts/02_train_scenes.sh configs/scenes.yaml
#   # local-validation run (holds out ~1/8 of train images per scene):
#   bash scripts/02_train_scenes.sh configs/scenes.yaml --eval
#
# Reads the [train] block in the yaml for iterations/sh_degree/eval defaults,
# but any extra args you pass on the command line are appended last and win.

set -euo pipefail

CONFIG="${1:-configs/scenes.example.yaml}"
shift || true
EXTRA_ARGS=("$@")

REPO="external/gaussian-splatting"
DATA_ROOT="data"
OUTPUT_ROOT="output"
LOG_DIR="logs/train"
mkdir -p "${LOG_DIR}"

if [ ! -f "${REPO}/train.py" ]; then
  echo "ERROR: ${REPO}/train.py not found. Run scripts/00_setup_env.sh first." >&2
  exit 1
fi

# Parse the yaml with python (avoids a hard dependency on yq).
read -r -d '' PARSE_PY << 'EOF' || true
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
scenes = cfg.get("scenes", [])
t = cfg.get("train", {})
print(" ".join(scenes))
print(t.get("iterations", 30000))
print(t.get("sh_degree", 3))
print("true" if t.get("eval", False) else "false")
EOF

mapfile -t PARSED < <(python -c "${PARSE_PY}" "${CONFIG}")
SCENES=(${PARSED[0]})
ITERATIONS="${PARSED[1]}"
SH_DEGREE="${PARSED[2]}"
EVAL_FLAG="${PARSED[3]}"

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "ERROR: no scenes listed in ${CONFIG}" >&2
  exit 1
fi

echo "Scenes to train: ${SCENES[*]}"
echo "iterations=${ITERATIONS} sh_degree=${SH_DEGREE} eval(config default)=${EVAL_FLAG}"

# Optional per-scene extra args from the yaml's scene_extra_args: block
# (e.g. bonsai: ["-r", "1"] to disable auto-downscaling for one scene only).
read -r -d '' SCENE_EXTRA_PY << 'EOF' || true
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
extra = cfg.get("scene_extra_args", {}).get(sys.argv[2], [])
print(" ".join(str(x) for x in extra))
EOF

for scene in "${SCENES[@]}"; do
  SRC="${DATA_ROOT}/${scene}/train"
  DST="${OUTPUT_ROOT}/${scene}"

  if [ ! -d "${SRC}" ]; then
    echo "SKIP ${scene}: ${SRC} not found"
    continue
  fi

  echo "=== Training ${scene} ==="
  # The baseline only writes a checkpoint at iterations listed in
  # --save_iterations (default [7000, 30000]). If --iterations is set to
  # anything else (e.g. a shortened run), it silently finishes without saving
  # ANY checkpoint unless we explicitly add that value here too.
  ARGS=(-s "${SRC}" -m "${DST}" --iterations "${ITERATIONS}" --sh_degree "${SH_DEGREE}"
        --save_iterations "${ITERATIONS}")
  if [ "${EVAL_FLAG}" = "true" ]; then
    ARGS+=(--eval)
  fi

  SCENE_EXTRA="$(python -c "${SCENE_EXTRA_PY}" "${CONFIG}" "${scene}")"
  if [ -n "${SCENE_EXTRA}" ]; then
    echo "  extra args for ${scene}: ${SCENE_EXTRA}"
    # shellcheck disable=SC2206
    ARGS+=(${SCENE_EXTRA})
  fi

  ARGS+=("${EXTRA_ARGS[@]}")

  # Persist a full training log per scene (terminal output scrolls away and
  # is lost once the session ends). Contest rule 10.3 says top-ranked teams
  # may be asked to prove reproducibility with training logs, so keep them.
  LOG_FILE="${LOG_DIR}/${scene}_$(date +%Y%m%d_%H%M%S).log"
  {
    echo "# scene=${scene} config=${CONFIG} started=$(date -Iseconds)"
    echo "# command: python ${REPO}/train.py ${ARGS[*]}"
  } > "${LOG_FILE}"
  echo "  logging to ${LOG_FILE}"

  # pipefail (set above) makes this correctly propagate train.py's exit code,
  # not tee's, so a failed run still stops the script under set -e.
  python "${REPO}/train.py" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "Done. Checkpoints under ${OUTPUT_ROOT}/<scene>/point_cloud/iteration_<N>/point_cloud.ply"
echo "Training logs under ${LOG_DIR}/"
