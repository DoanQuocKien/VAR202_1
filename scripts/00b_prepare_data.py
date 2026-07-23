#!/usr/bin/env python3
"""
Reorganize the organizer's raw data dump (phase1/<public_set|private_set1>/<scene>/...)
into this pipeline's expected data/<scene>/ layout, and fix a camera-model
incompatibility with the baseline repo along the way.
"""
import argparse
import shutil
import struct
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

# Import shared pipeline utilities
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import is_image, find_test_csv

CAMERA_MODEL_NPARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
PINHOLE_MODEL_ID = 1
UNDISTORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def read_single_camera(cameras_bin: Path):
    with open(cameras_bin, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        cams = []
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
            model_name, n_params = CAMERA_MODEL_NPARAMS[model_id]
            params = struct.unpack("<" + "d" * n_params, fid.read(8 * n_params))
            cams.append({
                "camera_id": camera_id, "model_id": model_id, "model_name": model_name,
                "width": width, "height": height, "params": list(params),
            })
    if len(cams) != 1:
        raise ValueError(f"{cameras_bin}: expected exactly 1 camera, found {len(cams)}")
    return cams[0]


def write_pinhole_cameras_bin(out_path: Path, camera_id: int, width: int, height: int,
                               fx: float, fy: float, cx: float, cy: float):
    with open(out_path, "wb") as fid:
        fid.write(struct.pack("<Q", 1))  # num_cameras
        fid.write(struct.pack("<iiQQ", camera_id, PINHOLE_MODEL_ID, width, height))
        fid.write(struct.pack("<dddd", fx, fy, cx, cy))


def build_K_and_dist(cam: dict):
    model, params = cam["model_name"], cam["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, None, f, f, cx, cy
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K, None, fx, fy, cx, cy
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k, 0, 0, 0, 0], dtype=np.float64)
        return K, dist, f, f, cx, cy
    if model == "RADIAL":
        f, cx, cy, k1, k2 = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
        return K, dist, f, f, cx, cy
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
        return K, dist, fx, fy, cx, cy
    raise ValueError(f"Unhandled camera model for undistortion: {model}")


def _undistort_single_file(p: Path, dst_dir: Path, K: np.ndarray, dist: np.ndarray):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [warn] failed to read {p}, skipping", file=sys.stderr)
        return False
    und = cv2.undistort(img, K, dist, None, K)
    cv2.imwrite(str(dst_dir / p.name), und)
    return True


def undistort_dir(src_dir: Path, dst_dir: Path, K: np.ndarray, dist: np.ndarray):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src_dir.iterdir() if is_image(p)])
    count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_undistort_single_file, p, dst_dir, K, dist) for p in files]
        for f in as_completed(futures):
            if f.result():
                count += 1
    return count


def copy_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src_dir.iterdir() if is_image(p)])
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(shutil.copy2, p, dst_dir / p.name) for p in files]
        for f in as_completed(futures):
            f.result()
    return len(files)


def read_images_binary_raw(images_bin: Path):
    entries = []
    with open(images_bin, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack(
                "<idddddddi", fid.read(64))
            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            points2d_bytes = fid.read(24 * num_points2d)
            entries.append({
                "image_id": image_id, "qvec": (qw, qx, qy, qz), "tvec": (tx, ty, tz),
                "camera_id": camera_id, "name": name_bytes.decode("utf-8"),
                "num_points2d": num_points2d, "points2d_bytes": points2d_bytes,
            })
    return entries


def write_images_binary_filtered(out_path: Path, entries):
    with open(out_path, "wb") as fid:
        fid.write(struct.pack("<Q", len(entries)))
        for e in entries:
            qw, qx, qy, qz = e["qvec"]
            tx, ty, tz = e["tvec"]
            fid.write(struct.pack("<idddddddi", e["image_id"], qw, qx, qy, qz,
                                   tx, ty, tz, e["camera_id"]))
            fid.write(e["name"].encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", e["num_points2d"]))
            fid.write(e["points2d_bytes"])


def filter_images_bin_to_existing(images_bin_src: Path, out_path: Path, kept_names: set):
    entries = read_images_binary_raw(images_bin_src)
    kept = [e for e in entries if e["name"] in kept_names]
    dropped = len(entries) - len(kept)
    write_images_binary_filtered(out_path, kept)
    return len(entries), len(kept), dropped


def prepare_scene(scene_src: Path, scene_name: str, out_dir: Path, gt_out_dir):
    train_images_src = scene_src / "train" / "images"
    sparse_src = scene_src / "train" / "sparse" / "0"
    test_src = scene_src / "test"

    if not train_images_src.is_dir() or not sparse_src.is_dir():
        print(f"SKIP {scene_name}: missing train/images or train/sparse/0")
        return

    cam = read_single_camera(sparse_src / "cameras.bin")

    out_scene = out_dir / scene_name
    out_train_images = out_scene / "train" / "images"
    out_sparse = out_scene / "train" / "sparse" / "0"
    out_test = out_scene / "test"
    out_sparse.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    if cam["model_name"] in UNDISTORTED_MODELS:
        print(f"{scene_name}: camera already {cam['model_name']} (undistorted), copying as-is")
        n = copy_dir(train_images_src, out_train_images)
        shutil.copy2(sparse_src / "cameras.bin", out_sparse / "cameras.bin")
    else:
        print(f"{scene_name}: camera is {cam['model_name']} {cam['params']} -> undistorting {cam['width']}x{cam['height']}")
        K, dist, fx, fy, cx, cy = build_K_and_dist(cam)
        n = undistort_dir(train_images_src, out_train_images, K, dist)
        write_pinhole_cameras_bin(out_sparse / "cameras.bin", cam["camera_id"],
                                   cam["width"], cam["height"], fx, fy, cx, cy)

    kept_names = {p.name for p in out_train_images.iterdir()}
    total, kept, dropped = filter_images_bin_to_existing(
        sparse_src / "images.bin", out_sparse / "images.bin", kept_names)
    if dropped:
        print(f"  {scene_name}: images.bin had {total} registered images, "
              f"{dropped} don't exist in train/images/ (kept {kept}) - dropped them")

    shutil.copy2(sparse_src / "points3D.bin", out_sparse / "points3D.bin")
    print(f"{scene_name}: {n} train images ready")

    test_csv = find_test_csv(test_src)
    if test_csv is not None:
        shutil.copy2(test_csv, out_test / test_csv.name)
    else:
        print(f"  [warn] {scene_name}: no test_pose(s).csv found under {test_src}")

    test_images_src = test_src / "images"
    if test_images_src.is_dir() and gt_out_dir is not None:
        gt_scene_dir = Path(gt_out_dir) / scene_name
        if cam["model_name"] in UNDISTORTED_MODELS:
            n_gt = copy_dir(test_images_src, gt_scene_dir)
        else:
            K, dist, *_ = build_K_and_dist(cam)
            n_gt = undistort_dir(test_images_src, gt_scene_dir, K, dist)
        print(f"  {scene_name}: {n_gt} ground-truth test images -> {gt_scene_dir} (for local eval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=str, help="organizer's raw zip, e.g. VAI_NVS_DATA.zip or VAI_NVS_DATA_ROUND2.zip")
    ap.add_argument("--src_root", type=str, help="already-extracted folder (alternative to --zip_path)")
    ap.add_argument("--split", type=str, default=None, choices=["public_set", "private_set1"],
                     help="ONLY for older zips using phase1/<split>/<scene>")
    ap.add_argument("--scenes", type=str, nargs="*", help="optional: only prepare these scene names")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--gt_out_dir", type=str, default="local_eval_gt")
    args = ap.parse_args()

    if not args.zip_path and not args.src_root:
        ap.error("provide either --zip_path or --src_root")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir = args.gt_out_dir if args.gt_out_dir else None

    with ExitStack() as stack:
        if args.zip_path:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            extract_root = Path(tmp_dir)
            with zipfile.ZipFile(args.zip_path) as zf:
                all_names = zf.namelist()
                has_phase1 = any(n.startswith("phase1/") for n in all_names)
                if has_phase1:
                    if not args.split:
                        ap.error(f"{args.zip_path} uses old layout - pass --split public_set or --split private_set1")
                    prefix = f"phase1/{args.split}/"
                    print(f"Extracting {args.split} scenes from {args.zip_path}...")
                else:
                    prefix = ""
                    print(f"Extracting scenes from {args.zip_path} (flat layout)...")
                members = [m for m in all_names if m.startswith(prefix) and "__MACOSX" not in m
                           and m.split("/")[-1] != ".DS_Store"]
                if args.scenes:
                    wanted = set(args.scenes)
                    members = [m for m in members if m[len(prefix):].split("/", 1)[0] in wanted]
                zf.extractall(extract_root, members)
            split_root = (extract_root / "phase1" / args.split) if has_phase1 else extract_root
        elif args.split:
            split_root = Path(args.src_root) / args.split
            if not split_root.is_dir():
                split_root = Path(args.src_root)
        else:
            split_root = Path(args.src_root)

        if not split_root.is_dir():
            print(f"ERROR: {split_root} not found", file=sys.stderr)
            sys.exit(1)

        scene_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])
        if args.scenes:
            wanted = set(args.scenes)
            scene_dirs = [p for p in scene_dirs if p.name in wanted]

        if not scene_dirs:
            print(f"ERROR: no scenes found under {split_root}", file=sys.stderr)
            sys.exit(1)

        for scene_dir in scene_dirs:
            prepare_scene(scene_dir, scene_dir.name, out_dir, gt_out_dir)

    print(f"\nDone. {len(scene_dirs)} scene(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
