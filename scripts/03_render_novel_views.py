#!/usr/bin/env python3
"""
Render RGB images at the novel test poses given in test_poses.csv, using a
trained 3D Gaussian Splatting model.

Why this script exists: the baseline repo's own render.py only re-renders
camera poses that are already registered in the scene's COLMAP images.bin
(it loads them via Scene -> readColmapSceneInfo). The contest's test poses are
NOT in images.bin — they are brand-new viewpoints supplied separately in
test_poses.csv. This script builds a camera for each CSV row from scratch,
using the exact same pose convention the baseline uses when it reads
images.bin (see external/gaussian-splatting/scene/dataset_readers.py,
function readColmapCameras), so a model trained with the stock train.py
renders correctly at these new poses with no changes to the baseline code.

CSV convention (must match COLMAP images.bin convention exactly):
    image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
    qvec/tvec define the WORLD-TO-CAMERA transform:  X_cam = R(qvec) @ X_world + tvec

Usage (single scene):
    python scripts/03_render_novel_views.py \
        --repo_path external/gaussian-splatting \
        --scene_dir data/scene_001 \
        --model_path output/scene_001 \
        --out_dir submission_build/scene_001

Usage (all scenes found under --data_dir, matched against --output_root):
    python scripts/03_render_novel_views.py --all \
        --repo_path external/gaussian-splatting \
        --data_dir data --output_root output --submission_root submission_build
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision  # noqa: F401  (still needed by the baseline repo imports)
from PIL import Image as PILImage

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import find_test_csv as _find_test_csv, TEST_CSV_NAMES


def find_test_csv(scene_dir: Path) -> Path:
    res = _find_test_csv(scene_dir)
    if res is not None:
        return res
    raise FileNotFoundError(
        f"No test pose file found under {scene_dir / 'test'} "
        f"(looked for {' or '.join(TEST_CSV_NAMES)})"
    )


def find_latest_iteration(model_path: Path) -> int:
    pc_dir = model_path / "point_cloud"
    if not pc_dir.is_dir():
        raise FileNotFoundError(f"No point_cloud/ under {model_path} — did training finish?")
    iters = []
    for p in pc_dir.iterdir():
        if p.is_dir() and p.name.startswith("iteration_"):
            try:
                iters.append(int(p.name.split("_")[1]))
            except ValueError:
                pass
    if not iters:
        raise FileNotFoundError(f"No iteration_* checkpoints under {pc_dir}")
    return max(iters)


def load_test_poses(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    required = ["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                "fx", "fy", "cx", "cy", "width", "height"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return df


def build_minicam(row, MiniCam, qvec2rotmat, getWorld2View2, getProjectionMatrix, focal2fov,
                   znear=0.01, zfar=100.0, warn_offcenter=True, supersample=1):
    qvec = np.array([row.qw, row.qx, row.qy, row.qz], dtype=np.float64)
    tvec = np.array([row.tx, row.ty, row.tz], dtype=np.float64)
    width, height = int(row.width), int(row.height)
    fx, fy = float(row.fx), float(row.fy)
    cx, cy = float(row.cx), float(row.cy)

    if warn_offcenter:
        dx = abs(cx - width / 2.0)
        dy = abs(cy - height / 2.0)
        if dx > 0.02 * width or dy > 0.02 * height:
            print(f"  [warn] {row.image_name}: principal point (cx={cx:.1f}, cy={cy:.1f}) "
                  f"is off-center relative to image size ({width}x{height}). "
                  f"The baseline projection matrix assumes a centered principal point; "
                  f"see README section 12.", file=sys.stderr)

    # Same convention as dataset_readers.readColmapCameras: R stored transposed,
    # T stored as-is (world-to-camera). getWorld2View2 undoes the transpose internally.
    R = np.transpose(qvec2rotmat(qvec))
    T = tvec

    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    # Supersampling: FoV is unchanged, just rasterize onto an N-times-larger
    # pixel grid. Valid because the baseline projection matrix only depends on
    # FoV + aspect (principal point assumed centered), so scaling W,H together
    # is exact - equivalent to scaling f, cx, cy by N in COLMAP terms.
    width, height = width * supersample, height * supersample

    return MiniCam(
        width=width, height=height, fovy=FovY, fovx=FovX,
        znear=znear, zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def render_scene(repo_path: Path, scene_dir: Path, model_path: Path, out_dir: Path,
                  iteration: int, sh_degree: int, white_background: bool, antialiasing: bool = False,
                  supersample: int = 1):
    sys.path.insert(0, str(repo_path))
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from scene.colmap_loader import qvec2rotmat
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
    from gaussian_renderer import render as gs_render
    from argparse import Namespace

    csv_path = find_test_csv(scene_dir)
    df = load_test_poses(csv_path)

    if iteration <= 0:
        iteration = find_latest_iteration(model_path)
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ply_path}")

    print(f"[{scene_dir.name}] loading checkpoint iteration {iteration}: {ply_path}")
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path), use_train_test_exp=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                      debug=False, antialiasing=antialiasing)

    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row in df.itertuples(index=False):
            cam = build_minicam(row, MiniCam, qvec2rotmat, getWorld2View2, getProjectionMatrix, focal2fov,
                                 supersample=supersample)
            rendering = gs_render(cam, gaussians, pipe, background, use_trained_exp=False)["render"]
            out_path = out_dir / str(row.image_name)
            if out_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                out_path = out_path.with_suffix(".png")
            # Do NOT use torchvision.utils.save_image here: for .jpg outputs it
            # falls through to PIL's DEFAULT JPEG quality (75), which visibly
            # degrades the render with block/ringing artifacts before scoring
            # (verified: submitted files byte-match a q=75 re-encode). SSIM and
            # especially LPIPS punish exactly this kind of compression noise.
            # Save via PIL with explicit near-lossless quality instead.
            arr = (rendering.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
            pil_img = PILImage.fromarray(arr.permute(1, 2, 0).cpu().numpy())
            if out_path.suffix.lower() in (".jpg", ".jpeg"):
                pil_img.save(str(out_path), quality=98, subsampling=0)
            else:
                pil_img.save(str(out_path))

    print(f"[{scene_dir.name}] wrote {len(df)} images to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_path", type=str, required=True, help="path to cloned gaussian-splatting repo")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = latest available checkpoint")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--antialiasing", action="store_true",
                     help="enable the Mip-Splatting-style 2D rasterizer filter at RENDER time only "
                          "(this is independent of whether the checkpoint was TRAINED with "
                          "--antialiasing - safe to try on any existing checkpoint without retraining)")
    ap.add_argument("--supersample", type=int, default=1,
                     help="render at N times the CSV resolution (FoV unchanged). Meant to feed "
                          "10_redistort_renders.py --supersample N, which folds the downsample "
                          "into its single remap for a sharper, less-interpolated final image.")

    ap.add_argument("--all", action="store_true", help="process every scene under --data_dir")
    ap.add_argument("--scenes", type=str, nargs="*",
                     help="in --all mode, only process these scene names (default: every scene "
                          "folder under --data_dir - useful to avoid mixing e.g. public_set "
                          "practice scenes into a private_set1 submission_build/)")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_root", type=str, default="output")
    ap.add_argument("--submission_root", type=str, default="submission_build")

    ap.add_argument("--scene_dir", type=str, help="single-scene mode: data/<scene>")
    ap.add_argument("--model_path", type=str, help="single-scene mode: output/<scene>")
    ap.add_argument("--out_dir", type=str, help="single-scene mode: submission_build/<scene>")

    args = ap.parse_args()
    repo_path = Path(args.repo_path)
    if not (repo_path / "gaussian_renderer" / "__init__.py").is_file():
        print(f"ERROR: {repo_path} doesn't look like the gaussian-splatting repo "
              f"(run scripts/00_setup_env.sh first)", file=sys.stderr)
        sys.exit(1)

    if args.all:
        data_dir = Path(args.data_dir)
        output_root = Path(args.output_root)
        submission_root = Path(args.submission_root)
        scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        if args.scenes:
            wanted = set(args.scenes)
            scenes = [p for p in scenes if p.name in wanted]
        if not scenes:
            print(f"ERROR: no scenes under {data_dir}", file=sys.stderr)
            sys.exit(1)
        for scene_dir in scenes:
            model_path = output_root / scene_dir.name
            out_dir = submission_root / scene_dir.name
            if not model_path.is_dir():
                print(f"SKIP {scene_dir.name}: no trained model at {model_path}")
                continue
            render_scene(repo_path, scene_dir, model_path, out_dir,
                         args.iteration, args.sh_degree, args.white_background, args.antialiasing,
                         args.supersample)
    else:
        if not (args.scene_dir and args.model_path and args.out_dir):
            ap.error("single-scene mode requires --scene_dir, --model_path and --out_dir "
                      "(or use --all for batch mode)")
        render_scene(repo_path, Path(args.scene_dir), Path(args.model_path), Path(args.out_dir),
                     args.iteration, args.sh_degree, args.white_background, args.antialiasing,
                     args.supersample)


if __name__ == "__main__":
    main()
