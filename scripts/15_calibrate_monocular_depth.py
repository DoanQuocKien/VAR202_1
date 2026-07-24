#!/usr/bin/env python3
"""
Calibrate Depth Anything V2 monocular depth maps against COLMAP sparse 3D points.
For each training camera image:
1. Projects COLMAP sparse points into camera space to get true metric z.
2. Computes median ratio between 1/z_colmap and monocular_depth_norm.
3. Saves exact per-image scale & offset into data/<scene>/train/sparse/0/depth_params.json
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
    """Simple parser for COLMAP binary sparse files (points3D.bin, images.bin, cameras.bin)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "gaussian-splatting"))
    from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary
    
    pts3d: Dict[int, Any] = read_points3D_binary(sparse_dir / "points3D.bin")  # type: ignore
    imgs: Dict[int, Any] = read_extrinsics_binary(sparse_dir / "images.bin")    # type: ignore
    cams: Dict[int, Any] = read_intrinsics_binary(sparse_dir / "cameras.bin")    # type: ignore
    return pts3d, imgs, cams


def calibrate_scene(scene_train_dir: Path):
    sparse_dir = scene_train_dir / "sparse" / "0"
    depths_dir = scene_train_dir / "depths"
    
    if not (sparse_dir / "points3D.bin").is_file():
        print(f"[{scene_train_dir.parent.name}] No points3D.bin under {sparse_dir}, skipping calibration.")
        return

    pts3d, imgs, cams = read_colmap_sparse(sparse_dir)
    print(f"[{scene_train_dir.parent.name}] Calibrating {len(imgs)} cameras against {len(pts3d)} sparse 3D points...")

    pts3d_dict = {}
    if isinstance(pts3d, dict):
        for p_id, pt in pts3d.items():
            if hasattr(pt, "xyz"):
                pts3d_dict[p_id] = pt.xyz

    depth_params = {}
    if isinstance(imgs, dict):
        for img_id, img_info in imgs.items():
            stem = Path(img_info.name).stem
            depth_png = depths_dir / f"{stem}.png"
            
            if not depth_png.is_file():
                depth_params[stem] = {"scale": 1.0, "offset": 0.0}
                continue

            raw_png = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
            if raw_png is None:
                depth_params[stem] = {"scale": 1.0, "offset": 0.0}
                continue
            
            mono_inv_depth = raw_png.astype(np.float64) / 65535.0

            qvec = img_info.qvec
            tvec = img_info.tvec
            
            from scene.colmap_loader import qvec2rotmat
            R = qvec2rotmat(qvec)

            colmap_inv_depths = []
            mono_vals = []

            h, w = mono_inv_depth.shape
            for pt_id, point2D in zip(img_info.point3D_ids, img_info.xys):
                if pt_id in pts3d_dict:
                    P_world = pts3d_dict[pt_id]
                    P_cam = R @ P_world + tvec
                    z = P_cam[2]
                    if z > 0.1:
                        x_px, y_px = int(round(point2D[0])), int(round(point2D[1]))
                        if 0 <= x_px < w and 0 <= y_px < h:
                            colmap_inv_depths.append(1.0 / z)
                            mono_vals.append(mono_inv_depth[y_px, x_px])

            if len(colmap_inv_depths) >= 10:
                c_inv = np.array(colmap_inv_depths)
                m_val = np.array(mono_vals)
                scale = float(np.median(c_inv / (m_val + 1e-6)))
                offset = 0.0
                depth_params[stem] = {"scale": scale, "offset": offset}
            else:
                depth_params[stem] = {"scale": 1.0, "offset": 0.0}

    with open(sparse_dir / "depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"[{scene_train_dir.parent.name}] Done. Calibrated depth_params.json saved.")


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
            calibrate_scene(train_dir)


if __name__ == "__main__":
    main()
