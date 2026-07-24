#!/usr/bin/env bash
# Grid search over densification parameters on benchmark scene HCM0193 to find optimal PSNR/SSIM/LPIPS settings.
set -euo pipefail

REPO="external/gaussian-splatting"
SCENE="HCM0193"
SRC="data/${SCENE}/train"
EVAL_GT="local_eval_gt_raw/${SCENE}"

# Default parameters for HCM0193 benchmark scene
DIST_F=925.1842594361348
DIST_K=0.00795193982469231
DIST_CX=660.0
DIST_CY=494.5

GRAD_THRESHOLDS=(0.00020 0.00015 0.00010)
DENSIFY_UNTIL=(15000 20000 25000)
PERCENT_DENSE=(0.01)

# Only test the unexplored MIDDLE GROUND configs.
# README §6.5 confirms default (0.00020, 15000) already tried → no change.
# README §6.5 confirms fully aggressive → OOM crash.
# Only the middle range below is unexplored:
CONFIGS=(
  "0.00015 20000"
  "0.00010 25000"
)

RESULTS_FILE="logs/sweep_results.csv"
mkdir -p logs

echo "grad_threshold,densify_until,percent_dense,psnr,ssim,lpips,score" > "${RESULTS_FILE}"

for CONFIG in "${CONFIGS[@]}"; do
  read -r gt du <<< "${CONFIG}"
  pd="0.01"
      TAG="gt${gt}_du${du}_pd${pd}"
      MODEL_DIR="output/sweep_${TAG}"
      RENDER_DIR="submission_build/sweep_${TAG}"
      REDIST_DIR="submission_build/sweep_${TAG}_redist"

      echo "=========================================="
      echo "Running sweep config: ${TAG}"
      echo "=========================================="

      # Track VRAM usage in background
      nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 30 > "logs/vram_sweep_${TAG}.csv" 2>/dev/null &
      VRAM_PID=$!

      # 1. Train model with custom densification params
      set +e
      python "${REPO}/train.py" \
        -s "${SRC}" -m "${MODEL_DIR}" \
        --iterations 30000 --sh_degree 3 --save_iterations 30000 \
        --densify_grad_threshold "${gt}" \
        --densify_until_iter "${du}" \
        --percent_dense "${pd}" \
        2>&1 | tee "logs/sweep_${TAG}.log"
      TRAIN_EXIT_CODE=$?
      set -e

      kill ${VRAM_PID} 2>/dev/null || true

      if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
        echo "${gt},${du},${pd},OOM/CRASH,OOM/CRASH,OOM/CRASH,OOM/CRASH" >> "${RESULTS_FILE}"
        echo "  => Run failed (OOM/Crash)"
        continue
      fi

      # 2. Render novel test views
      python scripts/03_render_novel_views.py \
        --repo_path "${REPO}" \
        --scene_dir "data/${SCENE}" --model_path "${MODEL_DIR}" \
        --out_dir "${RENDER_DIR}"

      # 3. Apply lens redistortion
      python scripts/10_redistort_renders.py \
        --renders_dir "${RENDER_DIR}" --out_dir "${REDIST_DIR}" \
        --f "${DIST_F}" --cx "${DIST_CX}" --cy "${DIST_CY}" --k "${DIST_K}"

      # 4. Evaluate against GT using competition psnr_max=50 formula
      SCORE_LINE=$(python scripts/05_eval_metrics.py \
        --pred_dir "${REDIST_DIR}" --gt_dir "${EVAL_GT}" --psnr_max 50 \
        2>/dev/null | grep "^MEAN" | awk '{print $2","$3","$4","$5}')

      echo "${gt},${du},${pd},${SCORE_LINE}" >> "${RESULTS_FILE}"
      echo "  => Results (PSNR,SSIM,LPIPS,Score): ${SCORE_LINE}"
done

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Top 5 configurations ranked by score:"
sort -t, -k7 -nr "${RESULTS_FILE}" | head -6
