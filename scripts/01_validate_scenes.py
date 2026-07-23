#!/usr/bin/env python3
"""
Validate that every scene under --data_dir matches the contest's expected layout
BEFORE you spend GPU hours training on it.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image, find_test_csv, TEST_CSV_NAMES

REQUIRED_COLS = [
    "image_name", "qw", "qx", "qy", "qz",
    "tx", "ty", "tz", "fx", "fy", "cx", "cy", "width", "height",
]


def validate_scene(scene_dir: Path, min_images: int, max_images: int,
                    min_poses: int, max_poses: int):
    errors = []
    name = scene_dir.name

    images_dir = scene_dir / "train" / "images"
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    test_csv = find_test_csv(scene_dir)

    if not images_dir.is_dir():
        errors.append(f"[{name}] missing train/images/")
    else:
        n_images = sum(1 for p in images_dir.iterdir() if is_image(p))
        if n_images == 0:
            errors.append(f"[{name}] train/images/ has no image files")
        elif not (min_images <= n_images <= max_images):
            errors.append(
                f"[{name}] train/images/ has {n_images} images "
                f"(expected roughly {min_images}-{max_images} — sanity check this, "
                f"or pass --min_train_images/--max_train_images to adjust)"
            )

    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / fname).is_file():
            errors.append(f"[{name}] missing train/sparse/0/{fname}")

    if test_csv is None:
        errors.append(f"[{name}] missing test/{{{'|'.join(TEST_CSV_NAMES)}}}")
    else:
        try:
            df = pd.read_csv(test_csv, skipinitialspace=True)
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_cols:
                errors.append(f"[{name}] {test_csv.name} missing columns: {missing_cols}")
            else:
                if not (min_poses <= len(df) <= max_poses):
                    errors.append(
                        f"[{name}] {test_csv.name} has {len(df)} rows "
                        f"(expected roughly {min_poses}-{max_poses} target views — sanity check this)"
                    )
                if df["image_name"].duplicated().any():
                    dupes = df["image_name"][df["image_name"].duplicated()].tolist()
                    errors.append(f"[{name}] duplicate image_name in {test_csv.name}: {dupes}")
                for col in ("width", "height"):
                    if (df[col] <= 0).any():
                        errors.append(f"[{name}] non-positive {col} in {test_csv.name}")
        except Exception as e:
            errors.append(f"[{name}] failed to parse {test_csv.name}: {e}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--min_train_images", type=int, default=100)
    ap.add_argument("--max_train_images", type=int, default=300)
    ap.add_argument("--min_test_poses", type=int, default=30)
    ap.add_argument("--max_test_poses", type=int, default=80)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    scenes = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not scenes:
        print(f"ERROR: no scene folders found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_errors = []
    for scene_dir in scenes:
        errs = validate_scene(scene_dir, args.min_train_images, args.max_train_images,
                              args.min_test_poses, args.max_test_poses)
        all_errors.extend(errs)
        status = "OK" if not errs else f"{len(errs)} issue(s)"
        print(f"{scene_dir.name}: {status}")

    if all_errors:
        print("\n--- Issues found ---")
        for e in all_errors:
            print(" -", e)
        sys.exit(1)
    else:
        print(f"\nAll {len(scenes)} scene(s) look valid.")


if __name__ == "__main__":
    main()
