#!/usr/bin/env python3
"""
Convert COLMAP MVS (patch_match_stereo) per-image geometric depth maps into the
16-bit PNG + depth_params.json format the baseline gaussian-splatting repo's
--depths flag expects (see utils/make_depth_scale.py and scene/dataset_readers.py
/ scene/cameras.py in graphdeco-inria/gaussian-splatting).

Why this script exists instead of using the baseline's own make_depth_scale.py:
that script is built for a MONOCULAR depth network (e.g. Depth Anything), which
only produces depth up to an unknown per-image scale/shift, so it fits a robust
affine (scale, offset) to calibrate each image against the COLMAP sparse points.
Our depth source is different: COLMAP's own dense MVS reconstruction
(patch_match_stereo), which is ALREADY in the same metric/coordinate system as
the sparse point cloud (same camera-space z, since both come from the same
COLMAP model) - no cross-modal calibration is needed. We only need a per-image
linear rescale so the value range fits into a uint16 PNG without clipping, and
we record the exact inverse of that rescale in depth_params.json (scale/offset)
so the baseline loader reconstructs the true metric inverse depth:

    final_invdepth = (raw_uint16 / 65536) * scale + offset      (baseline's own formula,
                                                                    see scene/cameras.py)

Usage:
    python scripts/07b_convert_mvs_depth.py --scene_dir data/HCM0421/train \
        --mvs_workspace mvs_workspace/HCM0421
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_colmap_mvs_bin(path: Path) -> np.ndarray:
    """Read a COLMAP mvs/Mat<float> binary file (stereo/depth_maps/*.bin).
    Format (verified against src/colmap/mvs/mat.cc Mat<float>::Write/Read):
    ASCII header "{width}&{height}&{depth}&" followed immediately by raw
    little-endian float32 data, flat index = slice*width*height + row*width + col
    (row-major within each slice, slice-major across channels). depth=1 for
    depth maps (3 for normal maps, not used here)."""
    with open(path, "rb") as f:
        header = bytearray()
        amp = 0
        while amp < 3:
            b = f.read(1)
            if not b:
                raise EOFError(f"Unexpected EOF reading header of {path}")
            header += b
            if b == b"&":
                amp += 1
        width, height, depth = (int(x) for x in header.decode("ascii").strip("&").split("&"))
        data = np.fromfile(f, dtype="<f4")
    expected = width * height * depth
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} floats (w={width} h={height} d={depth}), got {data.size}")
    if depth == 1:
        return data.reshape((height, width))
    return data.reshape((depth, height, width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="e.g. data/HCM0421/train (must contain sparse/0/ and images/)")
    ap.add_argument("--mvs_workspace", required=True, help="workspace passed to colmap patch_match_stereo (contains stereo/depth_maps/)")
    ap.add_argument("--out_depths_subdir", default="depths", help="subfolder under --scene_dir to write PNG depth maps to")
    ap.add_argument("--min_depth", type=float, default=1e-3, help="depths below this (or <=0) are treated as invalid/missing (COLMAP marks failed matches as 0)")
    ap.add_argument("--valid_percentile", type=float, default=99.5,
                     help="per-image inverse-depth percentile used to set the uint16 quantization range "
                          "(clips the top ~0.5%% of noisy near-camera outliers instead of letting them "
                          "blow out the value range for the whole image)")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    depth_maps_dir = Path(args.mvs_workspace) / "stereo" / "depth_maps"
    if not depth_maps_dir.is_dir():
        raise SystemExit(f"ERROR: {depth_maps_dir} not found - did colmap patch_match_stereo run and finish?")

    out_depths_dir = scene_dir / args.out_depths_subdir
    out_depths_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(depth_maps_dir.glob("*.geometric.bin"))
    if not bin_files:
        raise SystemExit(f"ERROR: no *.geometric.bin files in {depth_maps_dir}")

    depth_params = {}
    n_ok, n_skipped = 0, 0
    for bin_path in bin_files:
        # filenames look like "00012.png.geometric.bin" (original image filename + suffix)
        image_name = bin_path.name[: -len(".geometric.bin")]
        stem = Path(image_name).stem

        depth = read_colmap_mvs_bin(bin_path)  # (H, W) float32, meters in COLMAP's own scale, 0 = no estimate
        valid = depth > args.min_depth
        if valid.sum() < 100:
            print(f"  [skip] {image_name}: only {int(valid.sum())} valid MVS depth pixels")
            n_skipped += 1
            continue

        invdepth = np.zeros_like(depth, dtype=np.float32)
        invdepth[valid] = 1.0 / depth[valid]

        s = float(np.percentile(invdepth[valid], args.valid_percentile))
        if s <= 0:
            print(f"  [skip] {image_name}: degenerate scale (s={s})")
            n_skipped += 1
            continue

        # raw/65536 * s ~= invdepth  =>  raw = round(invdepth/s * 65536), quantized to uint16.
        # Invalid pixels stay at raw=0 -> after the loader's `* scale + offset` they map to
        # invdepth=0 (i.e. "infinitely far"), a reasonable default for sky/holes in this
        # outdoor-drone dataset rather than a true per-pixel mask (the baseline's depth_mask
        # is a per-IMAGE reliable/unreliable flag only, not per-pixel - see scene/cameras.py).
        raw = np.zeros_like(depth, dtype=np.float64)
        raw[valid] = (invdepth[valid] / s) * 65536.0
        raw = np.clip(raw, 0, 65535).astype(np.uint16)

        out_path = out_depths_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), raw)

        depth_params[stem] = {"scale": s, "offset": 0.0}
        n_ok += 1

    depth_params_path = scene_dir / "sparse" / "0" / "depth_params.json"
    with open(depth_params_path, "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"\nDone. {n_ok} depth maps written to {out_depths_dir}/ ({n_skipped} skipped - insufficient MVS coverage).")
    print(f"depth_params.json -> {depth_params_path}")
    print(f"Train with: --depths {args.out_depths_subdir}")


if __name__ == "__main__":
    main()
