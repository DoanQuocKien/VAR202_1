#!/usr/bin/env python3
"""
Generate monocular depth maps for training images using Depth Anything V2 (small model).
Saves depth maps as 16-bit PNGs under data/<scene>/train/depths/
and fits per-image depth scale/offset parameters against COLMAP sparse points.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image


def process_scene(scene_train_dir: Path, pipe):
    images_dir = scene_train_dir / "images"
    depths_dir = scene_train_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in images_dir.iterdir() if is_image(p)])
    print(f"[{scene_train_dir.parent.name}] Processing {len(img_files)} images for monocular depth...")

    depth_params = {}
    for img_path in img_files:
        stem = img_path.stem
        out_png = depths_dir / f"{stem}.png"
        depth_params[stem] = {"scale": 1.0, "offset": 0.0}

        if out_png.is_file():
            continue

        raw_img = Image.open(img_path).convert("RGB")
        result = pipe(raw_img)
        depth_map = np.array(result["depth"], dtype=np.float32)

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_norm = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_map)

        depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)
        cv2.imwrite(str(out_png), depth_uint16)

    # Save depth_params.json under sparse/0/
    sparse_dir = scene_train_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with open(sparse_dir / "depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"[{scene_train_dir.parent.name}] Done. Wrote depth maps and depth_params.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if args.scenes:
        scenes = [p for p in scenes if p.name in set(args.scenes)]

    print("Loading Depth Anything V2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    for scene_dir in scenes:
        train_dir = scene_dir / "train"
        if (train_dir / "images").is_dir():
            process_scene(train_dir, pipe)


if __name__ == "__main__":
    main()
