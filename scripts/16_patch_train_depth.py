#!/usr/bin/env python3
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
def pearson_correlation_loss(pred, target, mask):
    # Only compute over valid masked pixels
    pred_valid = pred[mask > 0]
    target_valid = target[mask > 0]
    
    if pred_valid.numel() < 10:
        return torch.tensor(0.0, device=pred.device)
        
    pred_centered = pred_valid - pred_valid.mean()
    target_centered = target_valid - target_valid.mean()
    
    cov = (pred_centered * target_centered).sum()
    var_pred = (pred_centered**2).sum()
    var_target = (target_centered**2).sum()
    
    corr = cov / (torch.sqrt(var_pred * var_target) + 1e-8)
    return 1.0 - corr
"""
    if "pearson_correlation_loss" not in content:
        content = content.replace("from utils.loss_utils import l1_loss, ssim", 
                                  "from utils.loss_utils import l1_loss, ssim\n" + pearson_func)
    
    # 2. String replacement for the depth loss
    old_line = "Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()"
    new_line = "Ll1depth_pure = pearson_correlation_loss(invDepth, mono_invdepth, depth_mask)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        train_depth_path.write_text(content, encoding="utf-8")
        print(f"SUCCESS: Replaced absolute L1 depth loss with Pearson loss!")
        print(f"Saved to: {train_depth_path}")
    else:
        print("FAILED: Exact string not found. Did you already patch it?")

if __name__ == "__main__":
    patch_train()
