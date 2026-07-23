#!/usr/bin/env python3
"""
Re-apply the original camera's SIMPLE_RADIAL distortion to pinhole renders,
so submitted images are geometrically aligned with the organizers' RAW
(distorted) ground-truth test photos.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image

INTERP_MAP = {
    "lanczos4": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
}


def build_redistort_maps(width: int, height: int, f: float, cx: float, cy: float, k: float,
                          supersample: int = 1):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)

    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64),
                          np.arange(height, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)

    und_norm = cv2.undistortPoints(pts, K, dist)  # (N,1,2), normalized
    map_x_1x = und_norm[:, 0, 0] * f + cx
    map_y_1x = und_norm[:, 0, 1] * f + cy
    n = supersample
    map_x = ((map_x_1x + 0.5) * n - 0.5).astype(np.float32).reshape(height, width)
    map_y = ((map_y_1x + 0.5) * n - 0.5).astype(np.float32).reshape(height, width)
    return map_x, map_y


def _process_single_image(p: Path, map_x: np.ndarray, map_y: np.ndarray, interp, out_dir: Path, jpeg_quality: int):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [warn] failed to read {p}, skipping", file=sys.stderr)
        return False
    out = cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_dir / p.name), out,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    else:
        cv2.imwrite(str(out_dir / p.name), out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--f", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--jpeg_quality", type=int, default=98)
    ap.add_argument("--interpolation", default="lanczos4", choices=["lanczos4", "cubic", "linear"],
                     help="cv2 interpolation algorithm for remap")
    ap.add_argument("--supersample", type=int, default=1,
                     help="input renders are N x the target resolution")
    args = ap.parse_args()

    renders_dir = Path(args.renders_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in renders_dir.iterdir() if is_image(p)])
    if not files:
        raise SystemExit(f"ERROR: no images in {renders_dir}")

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"ERROR: failed to read first image {files[0]}")
    h_in, w_in = first.shape[:2]
    n = args.supersample
    if w_in % n or h_in % n:
        raise SystemExit(f"ERROR: input {w_in}x{h_in} not divisible by --supersample {n}")
    w, h = w_in // n, h_in // n  # output (submission) resolution
    print(f"{len(files)} renders {w_in}x{h_in} (supersample={n}) -> output {w}x{h}; "
          f"building redistortion maps (f={args.f:.1f} cx={args.cx} cy={args.cy} k={args.k:.6f}, interp={args.interpolation})")
    map_x, map_y = build_redistort_maps(w, h, args.f, args.cx, args.cy, args.k, supersample=n)

    interp = INTERP_MAP[args.interpolation]
    count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_process_single_image, p, map_x, map_y, interp, out_dir, args.jpeg_quality)
            for p in files
        ]
        for f in as_completed(futures):
            if f.result():
                count += 1

    print(f"Done. {count} redistorted images -> {out_dir}")


if __name__ == "__main__":
    main()
