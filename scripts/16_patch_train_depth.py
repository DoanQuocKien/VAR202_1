#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def patch_train():
    repo_dir = Path(__file__).parent.parent / "external" / "gaussian-splatting"
    train_path = repo_dir / "train.py"
    train_depth_path = repo_dir / "train_depth.py"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        sys.exit(1)
        
    content = train_path.read_text(encoding="utf-8")
    
    # 1. Inject Pearson Correlation Loss function at the top
    pearson_func = """
def pearson_correlation_loss(pred, target):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    cov = (pred_centered * target_centered).sum()
    var_pred = (pred_centered**2).sum()
    var_target = (target_centered**2).sum()
    corr = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
    return 1.0 - corr
"""
    if "pearson_correlation_loss" not in content:
        content = content.replace("from utils.loss_utils import l1_loss, ssim", 
                                  "from utils.loss_utils import l1_loss, ssim\n" + pearson_func)
    
    # 2. Find the exact L1 depth loss calculation line
    # Usually looks like: Ll1depth = l1_loss(rendered_depth, viewpoint_cam.depth)
    match = re.search(r'(Ll1depth\s*=\s*)l1_loss\((.*?)\)', content)
    
    if match:
        old_line = match.group(0)
        new_line = f"{match.group(1)}pearson_correlation_loss({match.group(2)})"
        content = content.replace(old_line, new_line)
        train_depth_path.write_text(content, encoding="utf-8")
        print(f"SUCCESS: Replaced L1 depth loss with Pearson loss!")
        print(f"  Old: {old_line}")
        print(f"  New: {new_line}")
        print(f"Saved to: {train_depth_path}")
    else:
        print("FAILED: Could not find Ll1depth = l1_loss(...) pattern.")
        print("Here are lines 90-130 to inspect manually:")
        lines = content.split('\n')
        for i, line in enumerate(lines[90:130]):
            print(f"{i+90}: {line}")

if __name__ == "__main__":
    patch_train()
