#!/usr/bin/env python3
"""
Fine-tune an already-trained 3DGS checkpoint with a perceptual (LPIPS) loss
term added, WITHOUT retraining from scratch.

Why: the contest score is 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm, i.e. the
single heaviest term is LPIPS - but the baseline train.py optimizes only
L1 + 0.2*D-SSIM and never optimizes LPIPS at all. Adding a perceptual loss is
the standard trick for pushing LPIPS down; fine-tuning from the existing 30k
checkpoint (a few thousand iters, low LR, no densification) gets most of the
benefit at ~1/6 the cost of a full retrain.

Also exposes --lambda_dssim so the SSIM-vs-L1 balance can be tested in the
same run (baseline default 0.2; the score weights SSIM at 0.3 so a higher
value may trade a little PSNR for more SSIM profitably).

LPIPS is computed on a random square crop per iteration (default 512px) to
keep VRAM/time reasonable; L1 and SSIM stay full-image. Uses the VGG backbone
(standard for training; also avoids overfitting to the alex net we happen to
use in local eval).

The result is saved as a NEW iteration folder (load_iteration + iterations),
so the original checkpoint stays untouched and 03_render_novel_views.py picks
the fine-tuned one up automatically (it uses the latest iteration by default).

Usage (benchmark scene first, as always):
    python scripts/11_finetune_perceptual.py \
        --repo_path external/gaussian-splatting \
        --source_path data/HCM0193 \
        --model_path output/HCM0193 \
        --iterations 5000 --lambda_lpips 0.1
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_path", required=True)
    ap.add_argument("--source_path", required=True, help="data/<scene>")
    ap.add_argument("--model_path", required=True, help="output/<scene>")
    ap.add_argument("--load_iteration", type=int, default=-1,
                     help="-1 = latest checkpoint")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--lambda_dssim", type=float, default=0.2,
                     help="baseline default 0.2; try 0.4 to push SSIM harder")
    ap.add_argument("--lambda_lpips", type=float, default=0.1)
    ap.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"])
    ap.add_argument("--lpips_crop", type=int, default=512,
                     help="LPIPS computed on a random crop of this size (0 = full image)")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.insert(0, str(Path(args.repo_path)))
    from argparse import Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    import lpips as lpips_lib

    # --- mirror ModelParams / OptimizationParams / PipelineParams namespaces ---
    dataset = Namespace(
        sh_degree=args.sh_degree, source_path=args.source_path,
        model_path=args.model_path, images="images", depths="",
        resolution=-1, white_background=args.white_background,
        train_test_exp=False, data_device="cuda", eval=False,
    )
    # Fine-tune LRs: xyz continues at (roughly) its end-of-training decayed
    # value instead of restarting high - a high xyz LR at this stage would
    # destroy the converged geometry. Other params use baseline constants
    # (they were never scheduled in the baseline anyway).
    opt = Namespace(
        iterations=args.iterations,
        position_lr_init=1.6e-6, position_lr_final=1.6e-7,
        position_lr_delay_mult=0.01, position_lr_max_steps=args.iterations,
        feature_lr=0.0025, opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001,
        exposure_lr_init=0.01, exposure_lr_final=0.001,
        exposure_lr_delay_steps=0, exposure_lr_delay_mult=0.0,
        percent_dense=0.01, lambda_dssim=args.lambda_dssim,
        densification_interval=100, opacity_reset_interval=3000,
        densify_from_iter=500, densify_until_iter=0,  # densification fully off
        densify_grad_threshold=0.0002,
        depth_l1_weight_init=1.0, depth_l1_weight_final=0.01,
        random_background=False, optimizer_type="default",
    )
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False, antialiasing=False)

    gaussians = GaussianModel(args.sh_degree)
    load_iter = None if args.load_iteration == 0 else args.load_iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=True)
    start_iter = scene.loaded_iter
    print(f"Loaded checkpoint iteration {start_iter} "
          f"({gaussians.get_xyz.shape[0]} gaussians)")

    # load_ply doesn't restore spatial_lr_scale (stays 0 -> xyz LR would be 0);
    # recompute it from the scene extent exactly like create_from_pcd does.
    gaussians.spatial_lr_scale = scene.cameras_extent

    # load_ply also never sets up the per-image exposure-correction parameter
    # (that only happens in create_from_pcd, i.e. fresh training from a point
    # cloud) - but training_setup() unconditionally builds an optimizer over
    # self._exposure. Reproduce create_from_pcd's init here so fine-tuning an
    # existing checkpoint doesn't crash. We never use it (use_trained_exp=False
    # everywhere), so its exact values don't matter - it just needs to exist.
    train_cams = scene.getTrainCameras()
    gaussians.pretrained_exposures = None
    gaussians.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(train_cams)}
    exposure_init = torch.eye(3, 4, device="cuda")[None].repeat(len(train_cams), 1, 1)
    gaussians._exposure = torch.nn.Parameter(exposure_init.requires_grad_(True))

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    lpips_model = None
    if args.lambda_lpips > 0:
        lpips_model = lpips_lib.LPIPS(net=args.lpips_net).cuda()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
        lpips_model.eval()

    viewpoint_stack = []
    t0 = time.time()
    ema_loss = None
    for it in range(1, args.iterations + 1):
        gaussians.update_learning_rate(it)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(cam, gaussians, pipe, background, use_trained_exp=False,
                            separate_sh=False)
        image = render_pkg["render"]
        gt = cam.original_image.cuda()

        Ll1 = l1_loss(image, gt)
        ssim_val = ssim(image, gt)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_val)

        if lpips_model is not None:
            _, H, W = image.shape
            c = args.lpips_crop
            if c and c < min(H, W):
                y0 = random.randint(0, H - c)
                x0 = random.randint(0, W - c)
                pred_c = image[:, y0:y0 + c, x0:x0 + c]
                gt_c = gt[:, y0:y0 + c, x0:x0 + c]
            else:
                pred_c, gt_c = image, gt
            lp = lpips_model(pred_c.unsqueeze(0) * 2 - 1,
                             gt_c.unsqueeze(0) * 2 - 1).squeeze()
            loss = loss + args.lambda_lpips * lp

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
        if it % 250 == 0 or it == args.iterations:
            dt = time.time() - t0
            print(f"  iter {it}/{args.iterations}  ema_loss={ema_loss:.5f}  "
                  f"L1={Ll1.item():.5f} SSIM={ssim_val.item():.4f}  "
                  f"[{dt/it:.2f}s/it, ETA {(args.iterations-it)*dt/it/60:.1f}min]",
                  flush=True)

    out_iter = start_iter + args.iterations
    scene.save(out_iter)
    print(f"\nDone. Fine-tuned checkpoint saved as iteration_{out_iter} "
          f"under {args.model_path}/point_cloud/")
    print("Render it with 03_render_novel_views.py (picks latest iteration by "
          f"default, or pass --iteration {out_iter} explicitly).")


if __name__ == "__main__":
    main()
