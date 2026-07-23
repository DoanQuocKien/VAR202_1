#!/usr/bin/env python3
"""
Extract lens distortion parameters from cameras.bin across all prepared scenes
and centralize them into configs/lens_params.json.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

CAMERA_MODEL_NPARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def read_camera(cameras_bin: Path):
    with open(cameras_bin, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        if num_cameras != 1:
            raise ValueError(f"{cameras_bin}: expected 1 camera, found {num_cameras}")
        camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
        name, n_params = CAMERA_MODEL_NPARAMS.get(model_id, (f"UNKNOWN_{model_id}", 0))
        params = struct.unpack("<" + "d" * n_params, fid.read(8 * n_params)) if n_params else []
    return {
        "camera_id": camera_id,
        "model": name,
        "width": width,
        "height": height,
        "params": list(params),
    }


def main():
    ap = argparse.ArgumentParser(description="Extract camera parameters from dataset into JSON")
    ap.add_argument("--data_dir", type=str, default="data", help="Directory containing scene folders")
    ap.add_argument("--out", type=str, default="configs/lens_params.json", help="Output JSON path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if not data_dir.is_dir():
        print(f"Error: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    result = {}
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        cam_bin = scene_dir / "train" / "sparse" / "0" / "cameras.bin"
        if not cam_bin.exists():
            continue
        
        try:
            cam = read_camera(cam_bin)
            if cam["model"] == "SIMPLE_RADIAL":
                f, cx, cy, k = cam["params"]
                result[scene_dir.name] = {
                    "model": cam["model"],
                    "f": f,
                    "cx": cx,
                    "cy": cy,
                    "k": k,
                    "width": cam["width"],
                    "height": cam["height"],
                    "has_distortion": True
                }
            elif cam["model"] in ("PINHOLE", "SIMPLE_PINHOLE"):
                result[scene_dir.name] = {
                    "model": cam["model"],
                    "width": cam["width"],
                    "height": cam["height"],
                    "has_distortion": False
                }
        except Exception as e:
            print(f"[warn] Failed to read camera for {scene_dir.name}: {e}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Extracted parameters for {len(result)} scenes -> {out_path}")


if __name__ == "__main__":
    main()
