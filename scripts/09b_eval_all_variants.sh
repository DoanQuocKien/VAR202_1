#!/usr/bin/env bash
# Generate post-processing variants of an already-rendered scene and score
# every one (plus the untouched original) against a local GT benchmark, so we
# can see in one shot whether any cheap image post-processing helps LPIPS/SSIM
# without touching the trained model.
#
# Usage:
#   bash scripts/09b_eval_all_variants.sh <scene> <renders_dir>
# Example:
#   bash scripts/09b_eval_all_variants.sh HCM0193 submission_build/HCM0193_gtcheck

set -euo pipefail
SCENE="${1:?Usage: bash scripts/09b_eval_all_variants.sh <scene> <renders_dir>}"
RENDERS_DIR="${2:?Usage: bash scripts/09b_eval_all_variants.sh <scene> <renders_dir>}"
GT_DIR="local_eval_gt/${SCENE}"
OUT_ROOT="submission_build_variants/${SCENE}"

if [ ! -d "${GT_DIR}" ]; then
  echo "ERROR: ${GT_DIR} not found (need real ground-truth to score against)." >&2
  exit 1
fi

echo "=== [1/2] generating post-processing variants ==="
python scripts/09_postprocess_variants.py \
  --renders_dir "${RENDERS_DIR}" \
  --train_images_dir "data/${SCENE}/train/images" \
  --out_root "${OUT_ROOT}"

echo ""
echo "=== [2/2] scoring original + every variant ==="
echo ""
echo "--- original (no post-processing) ---"
python scripts/05_eval_metrics.py --pred_dir "${RENDERS_DIR}" --gt_dir "${GT_DIR}" --psnr_max 30 | tail -3

for vdir in "${OUT_ROOT}"/*/; do
  vname="$(basename "${vdir}")"
  echo ""
  echo "--- ${vname} ---"
  python scripts/05_eval_metrics.py --pred_dir "${vdir}" --gt_dir "${GT_DIR}" --psnr_max 30 | tail -3
done
