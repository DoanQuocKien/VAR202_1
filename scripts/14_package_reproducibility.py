#!/usr/bin/env python3
"""
Package all reproducibility requirements per Contest Rule 10.3:
1. Source code & inference scripts
2. Configurations (cfg_args, CLI flags)
3. Installed packages & exact environment freeze
4. Model checkpoints (latest point_cloud.ply per scene)
5. Training logs (logs/*.log)
"""
import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_tar", default="reproducibility_package.tar.gz")
    ap.add_argument("--model_dir", default="output")
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print("=== Step 1: Exporting environment freeze ===")
    freeze_path = repo_root / "requirements_freeze.txt"
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        with open(freeze_path, "w") as f:
            f.write(res.stdout)
        print(f"  Wrote environment freeze to {freeze_path}")
    except Exception as e:
        print(f"  [warn] Could not run pip freeze: {e}")

    print("\n=== Step 2: Collecting checkpoints and logs ===")
    model_dir = Path(args.model_dir)
    logs_dir = Path(args.logs_dir)

    tar_path = repo_root / args.output_tar
    with tarfile.open(tar_path, "w:gz") as tar:
        # 1. Add Codebase & Configs
        for item in ["README.md", "requirements.txt", "requirements_freeze.txt", "scripts"]:
            p = repo_root / item
            if p.exists():
                print(f"  Adding code/config: {item}")
                tar.add(p, arcname=f"codebase/{item}")

        # 2. Add Training Logs
        if logs_dir.is_dir():
            print(f"  Adding logs directory: {logs_dir}")
            tar.add(logs_dir, arcname="logs")

        # 3. Add Model Checkpoints (latest point_cloud.ply only)
        if model_dir.is_dir():
            for scene_dir in sorted(model_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                scene_name = scene_dir.name
                pc_dir = scene_dir / "point_cloud"
                if not pc_dir.is_dir():
                    continue

                # Find latest iteration folder
                iters = []
                for p in pc_dir.iterdir():
                    if p.is_dir() and p.name.startswith("iteration_"):
                        try:
                            iters.append((int(p.name.split("_")[1]), p))
                        except ValueError:
                            pass
                if not iters:
                    continue
                latest_iter, latest_folder = max(iters, key=lambda x: x[0])
                ply_file = latest_folder / "point_cloud.ply"

                if ply_file.is_file():
                    arc_target = f"checkpoints/{scene_name}/iteration_{latest_iter}/point_cloud.ply"
                    print(f"  Adding checkpoint: {scene_name} (iter {latest_iter}) -> {arc_target}")
                    tar.add(ply_file, arcname=arc_target)

                # Add scene configs if present
                for cfg_file in ["cfg_args", "cameras.json"]:
                    cfg_path = scene_dir / cfg_file
                    if cfg_path.is_file():
                        tar.add(cfg_path, arcname=f"checkpoints/{scene_name}/{cfg_file}")

    tar_size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"  SUCCESS! Reproducibility package created:")
    print(f"  {tar_path.name} ({tar_size_mb:.2f} MB)")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
