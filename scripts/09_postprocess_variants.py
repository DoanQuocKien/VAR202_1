#!/usr/bin/env python3
"""
Generate a handful of cheap post-processing variants of already-rendered
novel-view images, to test (against a local GT benchmark) whether any of them
improve LPIPS/SSIM without touching the trained model at all.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image


def unsharp_mask(img: np.ndarray, amount: float, radius: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_train_color_stats(train_images_dir: Path, sample_n: int = 40):
    files = sorted([p for p in train_images_dir.iterdir() if is_image(p)])
    if len(files) > sample_n:
        idx = np.linspace(0, len(files) - 1, sample_n).astype(int)
        files = [files[i] for i in idx]
    means, stds = [], []
    for p in files:
        img_raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"  [warn] failed to read {p}, skipping for color stats", file=sys.stderr)
            continue
        img = img_raw.astype(np.float32)
        means.append(img.reshape(-1, 3).mean(axis=0))
        stds.append(img.reshape(-1, 3).std(axis=0))
    if not means:
        raise ValueError(f"No valid images found in {train_images_dir}")
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def match_color(img: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        m, s = img[..., c].mean(), img[..., c].std() + 1e-6
        out[..., c] = (img[..., c] - m) / s * ref_std[c] + ref_mean[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--train_images_dir", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    renders_dir = Path(args.renders_dir)
    out_root = Path(args.out_root)
    files = sorted([p for p in renders_dir.iterdir() if is_image(p)])
    print(f"{len(files)} rendered images in {renders_dir}")

    print("Computing train-image color stats (target for color matching)...")
    ref_mean, ref_std = compute_train_color_stats(Path(args.train_images_dir))
    print(f"  train mean(BGR)={ref_mean}, std(BGR)={ref_std}")

    variants = {
        "sharpen_light": lambda im: unsharp_mask(im, amount=0.3),
        "sharpen_medium": lambda im: unsharp_mask(im, amount=0.6),
        "sharpen_strong": lambda im: unsharp_mask(im, amount=1.0),
        "color_match": lambda im: match_color(im, ref_mean, ref_std),
        "sharpen_medium_color_match": lambda im: match_color(unsharp_mask(im, amount=0.6), ref_mean, ref_std),
    }

    for vname, fn in variants.items():
        vdir = out_root / vname
        vdir.mkdir(parents=True, exist_ok=True)
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [warn] failed to read {p}, skipping for variant {vname}", file=sys.stderr)
                continue
            out = fn(img)
            cv2.imwrite(str(vdir / p.name), out)
        print(f"  wrote variant '{vname}' -> {vdir}")

    print("\nDone. Score each variant with 05_eval_metrics.py, e.g.:")
    for vname in variants:
        print(f"  python scripts/05_eval_metrics.py --pred_dir {out_root/vname} "
              f"--gt_dir local_eval_gt/HCM0193 --psnr_max 30")


if __name__ == "__main__":
    main()
