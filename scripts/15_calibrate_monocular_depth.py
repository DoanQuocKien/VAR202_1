#!/usr/bin/env python3
"""
Calibrate Depth Anything V2 monocular depth maps using Camera Orbit Center distance.
Since points3D.bin contains dummy points, we use the 3D camera translation vectors
(tx, ty, tz) from images.bin to compute the camera-to-target distance for each shot.
Close-up shots get smaller scale, distant shots get larger scale -> aligning monocular depth
across all orbiting drone camera angles!
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image


def read_colmap_sparse(sparse_dir: Path) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "gaussian-splatting"))
    from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary
    
    pts3d: Dict[int, Any] = read_points3D_binary(sparse_dir / "points3D.bin")  # type: ignore
    imgs: Dict[int, Any] = read_extrinsics_binary(sparse_dir / "images.bin")    # type: ignore
    cams: Dict[int, Any] = read_intrinsics_binary(sparse_dir / "cameras.bin")    # type: ignore
    return pts3d, imgs, cams


def calibrate_scene_orbital(scene_train_dir: Path):
    sparse_dir = scene_train_dir / "sparse" / "0"
    depths_dir = scene_train_dir / "depths"
    
    if not (sparse_dir / "images.bin").is_file():
        print(f"[{scene_train_dir.parent.name}] No images.bin under {sparse_dir}, skipping.")
        return

    pts3d, imgs, cams = read_colmap_sparse(sparse_dir)

    # Compute camera centers in world space: C = -R^T @ t
    from scene.colmap_loader import qvec2rotmat
    cam_centers = {}
    for img_id, img_info in imgs.items():
        stem = Path(img_info.name).stem
        R = qvec2rotmat(img_info.qvec)
        t = img_info.tvec
        C = -np.transpose(R) @ t
        cam_centers[stem] = C

    if not cam_centers:
        return

    # Scene target center is the mean of all camera positions
    all_C = np.array(list(cam_centers.values()))
    target_center = np.mean(all_C, axis=0)

    # Distance from each camera to scene center
    distances = {stem: float(np.linalg.norm(C - target_center)) for stem, C in cam_centers.items()}
    mean_dist = float(np.mean(list(distances.values())))

    print(f"[{scene_train_dir.parent.name}] Calibrating {len(distances)} camera depths via orbital distance (mean dist: {mean_dist:.2f}m)...")

    depth_params = {}
    for stem, dist in distances.items():
        # Scale monocular relative depth proportionally to camera distance
        scale = dist / (mean_dist + 1e-6)
        offset = 0.0
        depth_params[stem] = {"scale": float(scale), "offset": float(offset)}

    # Save depth_params.json
    with open(sparse_dir / "depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"[{scene_train_dir.parent.name}] Done. Calibrated orbital depth_params.json saved.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if args.scenes:
        scenes = [p for p in scenes if p.name in set(args.scenes)]

    for scene_dir in scenes:
        train_dir = scene_dir / "train"
        if (train_dir / "sparse" / "0").is_dir():
            calibrate_scene_orbital(train_dir)


if __name__ == "__main__":
    main()
