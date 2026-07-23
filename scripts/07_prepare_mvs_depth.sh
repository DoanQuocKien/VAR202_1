#!/usr/bin/env bash
# Compute per-image depth maps via COLMAP's own dense MVS (patch_match_stereo)
# directly on the training images + sparse model already used for 3DGS
# training. This is NOT a pretrained monocular depth network - it's a classic
# geometric algorithm (plane-sweep multi-view stereo) run only on the images
# and camera poses the contest already gave us, so it stays inside the "no
# external data" rule (10) without needing organizer clarification.
#
# Usage:
#   bash scripts/07_prepare_mvs_depth.sh <scene_name>
# Example:
#   bash scripts/07_prepare_mvs_depth.sh HCM0421
#
# Requires the `colmap` CLI built with CUDA support (for patch_match_stereo).
# Check/install first if missing:
#   apt-get update && apt-get install -y colmap
#   colmap patch_match_stereo --help   # should not error
#
# After this finishes, train with depth supervision via:
#   python external/gaussian-splatting/train.py -s data/<scene>/train \
#     -m output/<scene>_canary_depth --iterations 30000 --sh_degree 3 \
#     --save_iterations 30000 --eval --depths depths \
#     2>&1 | tee logs/canary/depth_run.log

set -euo pipefail
SCENE="${1:?Usage: bash scripts/07_prepare_mvs_depth.sh <scene_name>}"
SRC="data/${SCENE}/train"
WORK="mvs_workspace/${SCENE}"

if ! command -v colmap >/dev/null 2>&1; then
  echo "ERROR: 'colmap' CLI not found on this machine. Install it first:" >&2
  echo "  apt-get update && apt-get install -y colmap" >&2
  echo "Then verify CUDA dense-reconstruction support before rerunning this script." >&2
  exit 1
fi

if [ ! -d "${SRC}/sparse/0" ]; then
  echo "ERROR: ${SRC}/sparse/0 not found. Run scripts/00b_prepare_data.py first." >&2
  exit 1
fi

echo "=== Setting up MVS workspace (symlinks only, no data duplication) ==="
echo "    (data/${SCENE}/train/images/ is already pixel-undistorted PINHOLE by"
echo "     00b_prepare_data.py, so it's safe to feed directly into patch_match_stereo"
echo "     without a separate 'colmap image_undistorter' pass - poses stay pixel-aligned.)"
rm -rf "${WORK}"
mkdir -p "${WORK}/sparse"
ln -s "$(realpath "${SRC}/images")" "${WORK}/images"
for f in cameras.bin images.bin points3D.bin; do
  ln -s "$(realpath "${SRC}/sparse/0/${f}")" "${WORK}/sparse/${f}"
done

echo "=== [1/2] colmap patch_match_stereo (${SCENE}) - dense per-pixel matching, can take a while ==="
colmap patch_match_stereo \
  --workspace_path "${WORK}" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 1600

echo "=== [2/2] converting depth maps to the baseline's --depths format ==="
python scripts/07b_convert_mvs_depth.py --scene_dir "${SRC}" --mvs_workspace "${WORK}"

echo ""
echo "Done. Depths written to ${SRC}/depths/, params at ${SRC}/sparse/0/depth_params.json"
echo ""
echo "Canary train command:"
echo "  mkdir -p logs/canary"
echo "  python external/gaussian-splatting/train.py -s ${SRC} -m output/${SCENE}_canary_depth \\"
echo "    --iterations 30000 --sh_degree 3 --save_iterations 30000 --eval --depths depths \\"
echo "    2>&1 | tee logs/canary/depth_run.log"
