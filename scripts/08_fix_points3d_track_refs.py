#!/usr/bin/env python3
"""
Fix points3D.bin so pycolmap.Reconstruction() (used by gsplat's dataset
loader) can load it without crashing with IndexError: Image with ID N does
not exist.

Why this is needed: 00b_prepare_data.py filters images.bin down to only the
images actually distributed under train/images/ (some contest scenes were
COLMAP-reconstructed with extra photos that were never shipped - see that
script's docstring). It deliberately leaves points3D.bin untouched, because
the ORIGINAL gaussian-splatting repo's own parser never cross-checks a
point's per-observation image_id against the images table - it only reads
xyz/rgb to seed the initial point cloud, so stale track references there are
harmless. pycolmap.Reconstruction() is stricter: it validates every track
entry's image_id against the images table and raises on any dangling
reference. This script removes just those dangling track entries (and drops
any point left with zero observations afterwards), so the model loads
cleanly under pycolmap while keeping the exact same physical point cloud
already used to train the baseline submission - no effect on that pipeline
at all (it never reads the track data).

Keeps a one-time backup (points3D.bin.orig) so this is safe to re-run and
never loses the original file.

Usage:
    python scripts/08_fix_points3d_track_refs.py --sparse_dir data/HCM0421/train/sparse/0
"""
import argparse
import shutil
import struct
from pathlib import Path


def read_kept_image_ids(images_bin: Path) -> set:
    """Same binary layout as 00b_prepare_data.py's images.bin reader."""
    ids = set()
    with open(images_bin, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("<idddddddi", fid.read(64))[0]
            ids.add(image_id)
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            fid.read(24 * num_points2d)
    return ids


def filter_points3d(points3d_bin: Path, out_path: Path, kept_image_ids: set):
    n_in, n_out, n_track_dropped = 0, 0, 0
    kept_points = []
    with open(points3d_bin, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            n_in += 1
            header = fid.read(43)  # point3D_id(Q) + xyz(ddd) + rgb(BBB) + error(d)
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_raw = fid.read(8 * track_length)
            pairs = struct.unpack("<" + "ii" * track_length, track_raw) if track_length else ()
            kept_pairs = []
            for i in range(track_length):
                image_id, point2d_idx = pairs[2 * i], pairs[2 * i + 1]
                if image_id in kept_image_ids:
                    kept_pairs.append((image_id, point2d_idx))
                else:
                    n_track_dropped += 1
            if not kept_pairs:
                continue  # fully-orphaned point (only ever seen in a dropped image) - drop it
            n_out += 1
            kept_points.append((header, kept_pairs))

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(kept_points)))
        for header, kept_pairs in kept_points:
            out.write(header)
            out.write(struct.pack("<Q", len(kept_pairs)))
            for image_id, point2d_idx in kept_pairs:
                out.write(struct.pack("<ii", image_id, point2d_idx))

    return n_in, n_out, n_track_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse_dir", required=True, help="e.g. data/HCM0421/train/sparse/0")
    args = ap.parse_args()

    sparse_dir = Path(args.sparse_dir)
    images_bin = sparse_dir / "images.bin"
    points3d_bin = sparse_dir / "points3D.bin"
    backup = sparse_dir / "points3D.bin.orig"

    if not backup.exists():
        shutil.copy2(points3d_bin, backup)
        print(f"Backed up original -> {backup}")
    else:
        print(f"Backup already exists at {backup} (re-running from it, original is safe)")

    kept_ids = read_kept_image_ids(images_bin)
    print(f"{len(kept_ids)} image ids present in images.bin")

    tmp_out = sparse_dir / "points3D.bin.new"
    n_in, n_out, n_dropped = filter_points3d(backup, tmp_out, kept_ids)
    tmp_out.replace(points3d_bin)

    print(f"points3D.bin: {n_in} points in -> {n_out} points out "
          f"({n_in - n_out} fully-orphaned points dropped, "
          f"{n_dropped} dangling track entries removed)")


if __name__ == "__main__":
    main()
