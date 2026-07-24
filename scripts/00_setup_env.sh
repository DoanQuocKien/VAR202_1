#!/usr/bin/env bash
# Setup baseline gaussian-splatting (graphdeco-inria) + pipeline dependencies.
# Run from VAR2026_BTS_NVS/ : bash scripts/00_setup_env.sh
#
# Requires: git, an NVIDIA GPU + driver, and either conda (recommended) or
# python3-venv. See comments below for the pip/venv-only path.

set -euo pipefail

REPO_URL="https://github.com/graphdeco-inria/gaussian-splatting.git"
EXTERNAL_DIR="external/gaussian-splatting"
ENV_NAME="var2026-3dgs"

echo "== 1/4: cloning baseline repo (with submodules) =="
if [ -d "${EXTERNAL_DIR}/.git" ]; then
  echo "Repo already present at ${EXTERNAL_DIR}, pulling latest + syncing submodules..."
  git -C "${EXTERNAL_DIR}" pull
  git -C "${EXTERNAL_DIR}" submodule update --init --recursive
else
  mkdir -p external
  git clone --recursive "${REPO_URL}" "${EXTERNAL_DIR}"
fi

echo "== 2/4: creating environment =="
if command -v conda >/dev/null 2>&1; then
  if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists, skipping creation."
  else
    # The repo ships its own environment.yml (torch + cudatoolkit pinned to a
    # version that matches the CUDA submodules). Its pip section lists the
    # submodules by RELATIVE path (e.g. submodules/diff-gaussian-rasterization),
    # which pip resolves relative to the current working directory at install
    # time — so we must run conda env create from inside external/gaussian-splatting,
    # not from the repo root, or pip can't find them.
    sed -e "s/^name: .*/name: ${ENV_NAME}/" -e "/submodules\//d" "${EXTERNAL_DIR}/environment.yml" > /tmp/var2026_env.yml
    (cd "${EXTERNAL_DIR}" && conda env create -f /tmp/var2026_env.yml)
  fi
  echo "Activate with: conda activate ${ENV_NAME}"
  CONDA_RUN="conda run -n ${ENV_NAME}"
else
  echo "conda not found — falling back to venv + manual pip install."
  echo "This path is more fragile: diff-gaussian-rasterization and simple-knn"
  echo "are native CUDA extensions and need a torch build that matches your"
  echo "local CUDA toolkit. Adjust the --index-url below to your CUDA version"
  echo "(see https://pytorch.org/get-started/locally/)."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install plyfile tqdm
  pip install "${EXTERNAL_DIR}/submodules/diff-gaussian-rasterization"
  pip install "${EXTERNAL_DIR}/submodules/simple-knn"
  CONDA_RUN=""
fi

echo "== 3/4: installing pipeline-specific dependencies =="
if [ -n "${CONDA_RUN:-}" ]; then
  ${CONDA_RUN} pip install mkl
  ${CONDA_RUN} pip install -r requirements.txt
else
  pip install -r requirements.txt
fi

echo "== 3.5/5: ensuring CUDA compiler (nvcc) is available =="
# environment.yml only pulls in `cudatoolkit=11.6`, which is the CUDA *runtime*
# (libcudart etc.) — it does NOT include nvcc or the CUDA headers needed to
# compile the diff-gaussian-rasterization / simple-knn / fused-ssim extensions
# from source. On machines where CUDA wasn't separately installed system-wide
# (common on WSL2), building those extensions fails with:
#   OSError: CUDA_HOME environment variable is not set
# Fix: install the matching nvcc + dev headers *into this conda env* from
# NVIDIA's conda channel, and point CUDA_HOME at the env itself.
if [ -n "${CONDA_RUN:-}" ]; then
  ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
  if [ -x "${ENV_PREFIX}/bin/nvcc" ]; then
    echo "  nvcc already installed in '${ENV_NAME}', skipping"
  else
    echo "  nvcc not found in '${ENV_NAME}' — installing CUDA 11.6 dev toolkit (matches cudatoolkit=11.6 pinned in environment.yml)..."
    conda install -n "${ENV_NAME}" -y -c "nvidia/label/cuda-11.6.2" cuda-nvcc cuda-cudart-dev cuda-cccl
  fi
  export CUDA_HOME="${ENV_PREFIX}"
  # torch's cpp_extension looks for CUDA_HOME/lib64 on Linux; NVIDIA's conda
  # packages install to CUDA_HOME/lib, so add a lib64 symlink if missing.
  if [ -d "${ENV_PREFIX}/lib" ] && [ ! -e "${ENV_PREFIX}/lib64" ]; then
    ln -s "${ENV_PREFIX}/lib" "${ENV_PREFIX}/lib64" 2>/dev/null || true
  fi
  echo "  CUDA_HOME=${CUDA_HOME}"
fi

# nvcc alone isn't enough — torch's build backend also needs a host C++
# compiler (g++/gcc) to be present as a plain system binary named exactly
# "g++"/"gcc" in PATH. This is a system package (not something conda/pip
# can install), and installing it needs sudo, which this script can't
# answer interactively — so fail fast with clear instructions instead of
# letting it die deep inside a pip build log.
if ! command -v g++ >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
  echo ""
  echo "ERROR: no C++ compiler (g++/gcc) found on this system (WSL/Linux)."
  echo "The diff-gaussian-rasterization / simple-knn / fused-ssim extensions need"
  echo "one to compile from source. Install it once (one-time, needs sudo):"
  echo ""
  echo "    sudo apt update && sudo apt install -y build-essential"
  echo ""
  echo "Then re-run: bash scripts/00_setup_env.sh"
  exit 1
fi

# CUDA 11.6 (pinned by environment.yml) only supports host compilers up to
# GCC 11. Newer Ubuntu releases (WSL default) ship a newer default g++
# (12/13) via build-essential, which nvcc will refuse with "unsupported GNU
# version". Detect that up front instead of failing deep in a pip build log.
GXX_MAJOR_VERSION="$(g++ -dumpversion | cut -d. -f1)"
if [ "${GXX_MAJOR_VERSION}" -gt 11 ]; then
  echo ""
  echo "ERROR: default g++ is version ${GXX_MAJOR_VERSION}, but CUDA 11.6 only supports up to g++ 11."
  echo "Install g++ 11 alongside it and make it the default (one-time, needs sudo):"
  echo ""
  echo "    sudo apt update && sudo apt install -y gcc-11 g++-11"
  echo "    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100"
  echo "    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100"
  echo "    sudo update-alternatives --set gcc /usr/bin/gcc-11"
  echo "    sudo update-alternatives --set g++ /usr/bin/g++-11"
  echo ""
  echo "Then re-run: bash scripts/00_setup_env.sh"
  exit 1
fi

echo "== 4/5: checking CUDA rasterizer extensions =="
# These are compiled from source (submodules/diff-gaussian-rasterization,
# submodules/simple-knn, submodules/fused-ssim). If a previous run of this
# script failed partway through conda env create (e.g. the pip section of
# environment.yml errored out), the conda packages (torch etc.) can end up
# installed while these three never get built — and since the env already
# exists on a rerun, the create step above is skipped entirely and this would
# otherwise go unnoticed. So check explicitly every run and (re)install
# whichever one is missing, regardless of whether the env was just created.
check_and_install_submodule () {
  local import_name="$1"
  local submodule_dir="$2"
  if ${CONDA_RUN:-} python -c "import ${import_name}" >/dev/null 2>&1; then
    echo "  ${import_name}: already installed, skipping"
  else
    echo "  ${import_name}: missing, building from ${submodule_dir} (compiles CUDA code, can take a few minutes)..."
    (cd "${EXTERNAL_DIR}" && CUDA_HOME="${CUDA_HOME:-}" ${CONDA_RUN:-} pip install "./${submodule_dir}")
  fi
}
check_and_install_submodule "diff_gaussian_rasterization" "submodules/diff-gaussian-rasterization"
check_and_install_submodule "simple_knn._C" "submodules/simple-knn"
if [ -d "${EXTERNAL_DIR}/submodules/fused-ssim" ]; then
  check_and_install_submodule "fused_ssim" "submodules/fused-ssim"
fi

echo "== 5/5: sanity check =="
if [ -n "${CONDA_RUN:-}" ]; then
  ${CONDA_RUN} python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  ${CONDA_RUN} python -c "import diff_gaussian_rasterization; import simple_knn._C; print('rasterizer + simple_knn import OK')"
else
  python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  python -c "import diff_gaussian_rasterization; import simple_knn._C; print('rasterizer + simple_knn import OK')"
fi

echo "Setup done. Next: python scripts/01_validate_scenes.py --data_dir data"
