#!/usr/bin/env python3
from pathlib import Path

def dump_train_lines():
    repo_dir = Path(__file__).parent.parent / "external" / "gaussian-splatting"
    train_path = repo_dir / "train.py"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return
        
    content = train_path.read_text(encoding="utf-8")
    lines = content.split('\n')
    
    # We know line 147 is: Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
    # So the calculation must be between line 130 and 147.
    snippet = "\n".join(lines[125:150])
    
    out_path = Path("depth_snippet.txt")
    out_path.write_text(snippet, encoding="utf-8")
    print(f"Saved lines 125-150 to {out_path.absolute()}")
    print("Please run: cat depth_snippet.txt")

if __name__ == "__main__":
    dump_train_lines()
