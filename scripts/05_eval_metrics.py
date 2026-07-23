#!/usr/bin/env python3
"""
Compute LPIPS / SSIM / PSNR and the contest's combined score for a folder of
predicted images against a folder of ground-truth images with matching
filenames. Intended for LOCAL validation (see README section 8) — you won't
have real ground truth for test_poses.csv, so use this against a held-out
split of your own train images to get a proxy score before submitting.

Score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*PSNR_norm
PSNR_norm = clamp(PSNR / psnr_max, 0, 1)

Note: the organizers haven't published which LPIPS backbone / exact psnr_max
they use on the leaderboard. This script defaults to LPIPS-alex and
psnr_max=30 (both configurable) — treat the resulting number as a relative
proxy for comparing your own runs, not as the literal leaderboard score.

Usage:
    python scripts/05_eval_metrics.py --pred_dir path/to/renders --gt_dir path/to/gt
    python scripts/05_eval_metrics.py --pred_dir renders --gt_dir gt --lpips_net vgg --psnr_max 35
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image


def load_image_as_float(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float64) / 255.0
    return arr  # HWC, [0,1]


def to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    # lpips expects NCHW float tensor in [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return t * 2.0 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--psnr_max", type=float, default=30.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = {p.name: p for p in pred_dir.iterdir() if is_image(p)}
    gt_files = {p.name: p for p in gt_dir.iterdir() if is_image(p)}

    common = sorted(set(pred_files) & set(gt_files))
    missing_pred = sorted(set(gt_files) - set(pred_files))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    if missing_pred:
        print(f"[warn] {len(missing_pred)} GT image(s) have no prediction, skipped: {missing_pred[:5]}{'...' if len(missing_pred) > 5 else ''}", file=sys.stderr)
    if missing_gt:
        print(f"[warn] {len(missing_gt)} prediction(s) have no GT, skipped: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}", file=sys.stderr)
    if not common:
        print("ERROR: no matching filenames between pred_dir and gt_dir", file=sys.stderr)
        sys.exit(1)

    import lpips
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(args.device)
    lpips_model.eval()

    rows = []
    with torch.no_grad():
        for name in common:
            pred = load_image_as_float(pred_files[name])
            gt = load_image_as_float(gt_files[name])

            if pred.shape != gt.shape:
                print(f"[warn] {name}: shape mismatch pred {pred.shape} vs gt {gt.shape}, skipping", file=sys.stderr)
                continue

            psnr_val = sk_psnr(gt, pred, data_range=1.0)
            ssim_val = sk_ssim(gt, pred, data_range=1.0, channel_axis=2)

            pt = to_lpips_tensor(pred).to(args.device)
            gtt = to_lpips_tensor(gt).to(args.device)
            lpips_val = lpips_model(pt, gtt).item()

            psnr_norm = float(np.clip(psnr_val / args.psnr_max, 0.0, 1.0))
            score = 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm

            rows.append((name, psnr_val, ssim_val, lpips_val, score))

    if not rows:
        print("ERROR: no image pairs were successfully scored", file=sys.stderr)
        sys.exit(1)

    print(f"{'image':30s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'score':>8s}")
    for name, p, s, l, sc in rows:
        print(f"{name:30s} {p:8.3f} {s:8.4f} {l:8.4f} {sc:8.4f}")

    arr = np.array([[p, s, l, sc] for _, p, s, l, sc in rows])
    mean_psnr, mean_ssim, mean_lpips, mean_score = arr.mean(axis=0)
    print("-" * 66)
    print(f"{'MEAN':30s} {mean_psnr:8.3f} {mean_ssim:8.4f} {mean_lpips:8.4f} {mean_score:8.4f}")
    print(f"\n{len(rows)}/{len(common)} images scored. psnr_max={args.psnr_max}, lpips_net={args.lpips_net}")


if __name__ == "__main__":
    main()
