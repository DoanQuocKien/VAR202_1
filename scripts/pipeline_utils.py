#!/usr/bin/env python3
"""
Shared constants and helper functions for VAR202_1 pipeline scripts.
"""
import sys
from pathlib import Path

# Canonical set of supported image extensions (case-insensitive checks should use .suffix.lower())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Standard test pose CSV filenames used across competition data releases
TEST_CSV_NAMES = ("test_pose.csv", "test_poses.csv")


def is_image(p: Path) -> bool:
    """Return True if path suffix matches a supported image format."""
    return p.suffix.lower() in IMAGE_EXTS


def find_test_csv(scene_dir: Path) -> Path | None:
    """
    Find test pose CSV under scene_dir/test/.
    Checks for both test_pose.csv and test_poses.csv.
    Returns Path if found, None otherwise.
    """
    test_dir = scene_dir / "test" if (scene_dir / "test").is_dir() else scene_dir
    for fname in TEST_CSV_NAMES:
        p = test_dir / fname
        if p.is_file():
            return p
    return None
