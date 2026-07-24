#!/usr/bin/env bash
# Setup baseline gaussian-splatting (graphdeco-inria) + pipeline dependencies.
# Run from VAR2026_BTS_NVS/ : bash scripts/00_setup_env.sh

set -euo pipefail

REPO_URL="https://github.com/graphdeco-inria/gaussian-splatting.git"
EXTERNAL_DIR="external/gaussian-splatting"
ENV_NAME="var2026-3dgs"

echo "== 1/4: cloning baseline repo (with submodules) =="
if [ -d "${EXTERNAL_DIR}/.git" ]; then
  echo "Repo already present at ${EXTERNAL_DIR}, syncing submodules..."
  git -C "${EXTERNAL_DIR}" submodule update --init --recursive
else
  mkdir -p external
  git clone --recursive "${REPO_URL}" "${EXTERNAL_DIR}"
fi

CONDA_RUN=""

echo "== 2/4: checking Python & PyTorch environment =="
# If the current environment (e.g. Docker container or active virtual environment)
# already has PyTorch with CUDA available, use it directly!
if python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "  Active environment has working PyTorch + CUDA. Using active environment directly!"
  CONDA_RUN=""
elif command -v conda >/dev/null 2>&1; then
  if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda env '${ENV_NAME}' already exists."
  else
    echo "  Creating conda environment '${ENV_NAME}'..."
    sed -e "s/^name: .*/name: ${ENV_NAME}/" -e "/submodules\//d" "${EXTERNAL_DIR}/environment.yml" > /tmp/var2026_env.yml
    (cd "${EXTERNAL_DIR}" && conda env create -f /tmp/var2026_env.yml)
  fi
  CONDA_RUN="conda run -n ${ENV_NAME}"
  # Ensure mkl is installed to prevent symbol issues in legacy conda envs
  ${CONDA_RUN} pip install mkl || true
else
  echo "  Conda not found & active PyTorch not detected — creating virtualenv..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install plyfile tqdm
  CONDA_RUN=""
fi

echo "== 3/4: installing pipeline dependencies =="
if [ -n "${CONDA_RUN:-}" ]; then
  ${CONDA_RUN} pip install -r requirements.txt
else
  pip install -r requirements.txt
fi

echo "== 4/4: compiling CUDA rasterizer submodules =="
check_and_install_submodule () {
  local import_name="$1"
  local submodule_dir="$2"
  if ${CONDA_RUN:-} python -c "import ${import_name}" >/dev/null 2>&1; then
    echo "  ${import_name}: already installed, skipping"
  else
    echo "  ${import_name}: compiling from ${submodule_dir}..."
    if [ -n "${CONDA_RUN:-}" ]; then
      ${CONDA_RUN} pip install "${EXTERNAL_DIR}/${submodule_dir}"
    else
      pip install "${EXTERNAL_DIR}/${submodule_dir}"
    fi
  fi
}

check_and_install_submodule "diff_gaussian_rasterization" "submodules/diff-gaussian-rasterization"
check_and_install_submodule "simple_knn._C" "submodules/simple-knn"
if [ -d "${EXTERNAL_DIR}/submodules/fused-ssim" ]; then
  check_and_install_submodule "fused_ssim" "submodules/fused-ssim"
fi

echo "== Sanity check =="
if [ -n "${CONDA_RUN:-}" ]; then
  ${CONDA_RUN} python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
  ${CONDA_RUN} python -c "import diff_gaussian_rasterization; import simple_knn._C; print('Rasterizer + simple_knn import OK')"
else
  python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
  python -c "import diff_gaussian_rasterization; import simple_knn._C; print('Rasterizer + simple_knn import OK')"
fi

echo ""
echo "Setup completed successfully!"
