#!/usr/bin/env bash
# Apply the LPIPS-fine-tune step to submission scenes, re-render, re-distort
# scenes with lens distortion, and repackage the submission zip.
#
# Usage: bash scripts/12_apply_finetune_and_rebuild.sh

set -euo pipefail

REPO="external/gaussian-splatting"
FT_ITERS=5000
LAMBDA_LPIPS=0.1
OUT_ROOT="submission_build_ft"

# Extract lens parameters dynamically if data folder exists
if [ -d "data" ]; then
  echo "Extracting lens parameters dynamically..."
  python scripts/extract_lens_params.py --data_dir data --out configs/lens_params.json || true
fi

SCENES=(HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 bonsai chair)

mkdir -p "${OUT_ROOT}"

for scene in "${SCENES[@]}"; do
  echo ""
  echo "=================== ${scene} ==================="
  MODEL_DIR="output/${scene}"

  if [ ! -d "${MODEL_DIR}/point_cloud" ]; then
    echo "SKIP ${scene}: No trained model output directory at ${MODEL_DIR}"
    continue
  fi

  BASE_ITER=$(ls "${MODEL_DIR}/point_cloud" | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "[${scene}] base checkpoint iteration_${BASE_ITER}"

  echo "[${scene}] fine-tuning ${FT_ITERS} iters (lambda_lpips=${LAMBDA_LPIPS}) ..."
  python scripts/11_finetune_perceptual.py \
    --repo_path "${REPO}" \
    --source_path "data/${scene}/train" --model_path "${MODEL_DIR}" \
    --load_iteration "${BASE_ITER}" \
    --iterations "${FT_ITERS}" --lambda_lpips "${LAMBDA_LPIPS}"

  NEW_ITER=$((BASE_ITER + FT_ITERS))

  echo "[${scene}] rendering iteration_${NEW_ITER} ..."
  python scripts/03_render_novel_views.py \
    --repo_path "${REPO}" \
    --scene_dir "data/${scene}" --model_path "${MODEL_DIR}" \
    --out_dir "submission_build_ft_raw/${scene}" --iteration "${NEW_ITER}"

  # Parse distortion parameters dynamically using python helper
  PARAM_JSON=$(python -c '
import json, sys
try:
    data = json.load(open("configs/lens_params.json"))
    s = data.get(sys.argv[1], {})
    if s.get("has_distortion", False):
        print(f"{s[\"f\"]} {s[\"cx\"]} {s[\"cy\"]} {s[\"k\"]}")
except Exception:
    pass
' "${scene}" 2>/dev/null || true)

  if [ -n "${PARAM_JSON}" ]; then
    read -r F_VAL CX_VAL CY_VAL K_VAL <<< "${PARAM_JSON}"
    echo "[${scene}] redistorting dynamically (f=${F_VAL} cx=${CX_VAL} cy=${CY_VAL} k=${K_VAL}) ..."
    python scripts/10_redistort_renders.py \
      --renders_dir "submission_build_ft_raw/${scene}" \
      --out_dir "${OUT_ROOT}/${scene}" \
      --f "${F_VAL}" --cx "${CX_VAL}" --cy "${CY_VAL}" --k "${K_VAL}"
  else
    echo "[${scene}] no distortion parameters found or camera is SIMPLE_PINHOLE - copying renders as-is"
    mkdir -p "${OUT_ROOT}/${scene}"
    cp "submission_build_ft_raw/${scene}"/* "${OUT_ROOT}/${scene}/"
  fi
done

echo ""
echo "=== Packaging submission ==="
python scripts/04_make_submission.py \
  --submission_dir "${OUT_ROOT}" --data_dir data \
  --out_zip submission_round2_ft.zip

echo ""
echo "Done. submission_round2_ft.zip ready - check its size is under 350MB:"
ls -lh submission_round2_ft.zip
