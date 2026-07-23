#!/usr/bin/env python3
"""
Package submission_build/<scene>/*.png into submission_round1.zip, validating
each scene against its test pose csv first (right file count, names, sizes)
so you don't find out about a mismatch after the deadline.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import find_test_csv as _find_test_csv, TEST_CSV_NAMES

REQUIRED_COLS = ["image_name", "width", "height"]


def find_test_csv(scene_name: str, data_dir: Path):
    return _find_test_csv(data_dir / scene_name)


def validate_scene(scene_name: str, scene_out_dir: Path, test_csv):
    errors = []
    if test_csv is None:
        return [f"[{scene_name}] no {{{'|'.join(TEST_CSV_NAMES)}}} found under data/{scene_name}/test/, cannot validate"]

    df = pd.read_csv(test_csv, skipinitialspace=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return [f"[{scene_name}] {test_csv.name} missing columns: {missing}"]

    if not scene_out_dir.is_dir():
        return [f"[{scene_name}] no rendered output dir at {scene_out_dir}"]

    produced = {p.name: p for p in scene_out_dir.iterdir() if p.is_file()}

    expected_names = set()
    for row in df.itertuples(index=False):
        name = str(row.image_name)
        if Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            name = str(Path(name).with_suffix(".png"))
        expected_names.add(name)

        if name not in produced:
            errors.append(f"[{scene_name}] missing rendered image: {name}")
            continue

        try:
            with Image.open(produced[name]) as im:
                w, h = im.size
            if (w, h) != (int(row.width), int(row.height)):
                errors.append(
                    f"[{scene_name}] {name} size {w}x{h} != expected {int(row.width)}x{int(row.height)}"
                )
        except Exception as e:
            errors.append(f"[{scene_name}] failed to open {name}: {e}")

    extra = set(produced.keys()) - expected_names
    if extra:
        errors.append(f"[{scene_name}] extra files not in {test_csv.name} (will still be zipped, remove if unwanted): {sorted(extra)}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission_dir", type=str, default="submission_build")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_zip", type=str, default="submission_round1.zip")
    ap.add_argument("--force", action="store_true", help="zip anyway even if validation finds errors")
    args = ap.parse_args()

    submission_dir = Path(args.submission_dir)
    data_dir = Path(args.data_dir)

    scene_dirs = sorted([p for p in submission_dir.iterdir() if p.is_dir()])
    if not scene_dirs:
        print(f"ERROR: no scene folders under {submission_dir}", file=sys.stderr)
        sys.exit(1)

    all_errors = []
    for scene_out_dir in scene_dirs:
        scene_name = scene_out_dir.name
        test_csv = find_test_csv(scene_name, data_dir)
        errs = validate_scene(scene_name, scene_out_dir, test_csv)
        all_errors.extend(errs)
        print(f"{scene_name}: {'OK' if not errs else str(len(errs)) + ' issue(s)'}")

    if all_errors:
        print("\n--- Issues found ---")
        for e in all_errors:
            print(" -", e)
        if not args.force:
            print("\nFix these before zipping, or re-run with --force to zip anyway.")
            sys.exit(1)

    out_zip = Path(args.out_zip)
    with zipfile.ZipFile(out_zip, "w") as zf:
        for scene_out_dir in scene_dirs:
            for f in sorted(scene_out_dir.iterdir()):
                if f.is_file():
                    arcname = f"{scene_out_dir.name}/{f.name}"
                    # JPEGs are already compressed; storing them saves CPU time
                    compress = zipfile.ZIP_STORED if f.suffix.lower() in (".jpg", ".jpeg") else zipfile.ZIP_DEFLATED
                    zf.write(f, arcname, compress_type=compress)

    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_zip} with {len(scene_dirs)} scene(s) ({zip_size_mb:.2f} MB).")
    if zip_size_mb > 350.0:
        print(f"WARNING: Zip size ({zip_size_mb:.2f} MB) exceeds the 350 MB contest limit!", file=sys.stderr)


if __name__ == "__main__":
    main()
