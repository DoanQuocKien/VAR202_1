# VAR202_1 Optimization Playbook — Agent Handoff Document

> **Purpose**: Complete implementation guide for a teammate agent to maximize competition score before deadline.  
> **Repo**: `d:\VAR202_1`  
> **Deadline**: 30/07/2026  
> **Current best score**: **72.52340** / 100  
> **Projected ceiling (all fixes)**: **75.5 – 79.0** (conservative – optimistic)

---

## 0. Scoring Formula & Current Decomposition

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × (PSNR / 50)
```

### Current raw metrics (from leaderboard submission):

| Metric | Raw Value | Weight | Component | % of Total |
|--------|-----------|--------|-----------|------------|
| PSNR | 24.61 dB | 0.3 × (val/50) | 0.14766 | 20.4% |
| SSIM | 0.8149 | 0.3 × val | 0.24447 | 33.7% |
| LPIPS | 0.1673 | 0.4 × (1−val) | 0.33308 | 45.9% |
| **Total** | | | **0.72521** | **100%** |

### Marginal value of each metric (how much 1 unit of improvement moves the score):

| Improvement | Score Δ | Points gained |
|---|---|---|
| PSNR +1 dB | +0.006 | +0.60 |
| SSIM +0.01 | +0.003 | +0.30 |
| LPIPS −0.01 | +0.004 | +0.40 |

> [!IMPORTANT]
> PSNR has the **highest marginal score per unit** (0.6 pts/dB), and PSNR+SSIM together have 0.6 weight. The densification sweep (which primarily improves PSNR and SSIM) is therefore the highest-leverage single change.

---

## 1. Score Impact Summary — All Fixes Ranked

| Priority | Task | Category | Est. Score Δ | Confidence | Time |
|:---:|---|---|:---:|:---:|:---:|
| **🥇 P0** | Densification parameter sweep | Algorithmic | **+2.0 to +5.0** | Medium | 6-12h GPU |
| **🥈 P1** | Auto-extract & centralize lens params | Pipeline | **+0.0 to +0.3** | High | 1h |
| **🥉 P2** | BICUBIC vs LANCZOS4 redistortion A/B test | Quality | **−0.1 to +0.3** | Low | 30min |
| P3 | ZIP_STORED for JPEGs (speed, not score) | Speed | 0 (saves ~2min) | High | 15min |
| P4 | Parallelize I/O loops (undistort, redistort) | Speed | 0 (saves ~5min) | High | 1h |
| P5 | Fix cv2.imread null checks | Robustness | 0 | High | 15min |
| P6 | Fix TemporaryDirectory leak | Robustness | 0 | High | 5min |
| P7 | Shared utils.py module | Maintenance | 0 | High | 45min |

### Projected Score Scenarios

```
Current:      72.52  ─────────────────────────────────▌
Conservative: 75.50  ─────────────────────────────────────────▌  (+3.0)
Moderate:     77.20  ────────────────────────────────────────────────▌  (+4.7)
Optimistic:   79.00  ──────────────────────────────────────────────────────▌  (+6.5)
```

| Scenario | PSNR | SSIM | LPIPS | Score |
|----------|------|------|-------|-------|
| Current | 24.61 | 0.8149 | 0.1673 | 72.52 |
| Conservative (+densify only) | 26.50 | 0.8350 | 0.1570 | **75.50** |
| Moderate (+densify +interp fix) | 27.50 | 0.8500 | 0.1500 | **77.20** |
| Optimistic (+densify +fine-tune v2) | 28.50 | 0.8650 | 0.1450 | **79.00** |

---

## 2. Task Specifications (for implementing agent)

---

### 🥇 TASK P0: Densification Parameter Sweep

**Estimated score gain**: +2.0 to +5.0 points  
**Time budget**: 6-12h GPU  
**Risk**: Medium (requires full retrain per config; may OOM on some configs)

#### Background

The current model uses baseline defaults:
- `--densify_grad_threshold 0.0002`
- `--densify_until_iter 15000`
- `--percent_dense 0.01`

Two extremes were tried (README §6.5): default = no change, aggressive = OOM crash. The **middle ground is completely unexplored** and is where the biggest gains likely are.

#### Implementation

**Step 1**: Create a sweep script. Run on the benchmark scene `HCM0193` (the only scene with ground-truth for local scoring).

```bash
#!/usr/bin/env bash
# scripts/sweep_densification.sh — Grid search over densification params
set -euo pipefail

REPO="external/gaussian-splatting"
SCENE="HCM0193"
SRC="data/${SCENE}/train"
EVAL_GT="local_eval_gt_raw/${SCENE}"  # RAW (distorted) GT for accurate scoring

# Lens params for HCM0193 (from README §9)
DIST_F=925.1842594361348
DIST_K=0.00795193982469231
DIST_CX=660.0
DIST_CY=494.5

# --- SWEEP GRID ---
GRAD_THRESHOLDS=(0.00010 0.00012 0.00015 0.00018 0.0002)
DENSIFY_UNTIL=(15000 20000 25000)
PERCENT_DENSE=(0.005 0.01 0.02)

RESULTS_FILE="sweep_results.csv"
echo "grad_threshold,densify_until,percent_dense,psnr,ssim,lpips,score" > "${RESULTS_FILE}"

for gt in "${GRAD_THRESHOLDS[@]}"; do
  for du in "${DENSIFY_UNTIL[@]}"; do
    for pd in "${PERCENT_DENSE[@]}"; do
      TAG="gt${gt}_du${du}_pd${pd}"
      MODEL_DIR="output/sweep_${TAG}"
      RENDER_DIR="submission_build/sweep_${TAG}"
      REDIST_DIR="submission_build/sweep_${TAG}_redist"

      echo "=== ${TAG} ==="

      # 1. Train
      python "${REPO}/train.py" \
        -s "${SRC}" -m "${MODEL_DIR}" \
        --iterations 30000 --sh_degree 3 --save_iterations 30000 \
        --densify_grad_threshold "${gt}" \
        --densify_until_iter "${du}" \
        --percent_dense "${pd}" \
        2>&1 | tee "logs/sweep_${TAG}.log"

      # Check for OOM / crash
      if [ $? -ne 0 ]; then
        echo "${gt},${du},${pd},OOM,OOM,OOM,OOM" >> "${RESULTS_FILE}"
        continue
      fi

      # 2. Render
      python scripts/03_render_novel_views.py \
        --repo_path "${REPO}" \
        --scene_dir "data/${SCENE}" --model_path "${MODEL_DIR}" \
        --out_dir "${RENDER_DIR}"

      # 3. Redistort
      python scripts/10_redistort_renders.py \
        --renders_dir "${RENDER_DIR}" --out_dir "${REDIST_DIR}" \
        --f "${DIST_F}" --cx "${DIST_CX}" --cy "${DIST_CY}" --k "${DIST_K}"

      # 4. Score
      SCORE_LINE=$(python scripts/05_eval_metrics.py \
        --pred_dir "${REDIST_DIR}" --gt_dir "${EVAL_GT}" --psnr_max 50 \
        2>/dev/null | grep "^MEAN" | awk '{print $2","$3","$4","$5}')

      echo "${gt},${du},${pd},${SCORE_LINE}" >> "${RESULTS_FILE}"
      echo "  => ${SCORE_LINE}"
    done
  done
done

echo ""
echo "=== SWEEP COMPLETE ==="
echo "Results in ${RESULTS_FILE}"
sort -t, -k7 -nr "${RESULTS_FILE}" | head -10
```

**Step 2**: Pick the best config from `sweep_results.csv`.

**Step 3**: Retrain ALL 7 scenes with that config:
```bash
# In 02_train_scenes.sh or via scene_extra_args in configs/scenes_round2.yaml:
# Add to configs/scenes_round2.yaml:
train:
  iterations: 30000
  sh_degree: 3
  eval: false
  densify_grad_threshold: <BEST_VALUE>    # from sweep
  densify_until_iter: <BEST_VALUE>        # from sweep
  percent_dense: <BEST_VALUE>             # from sweep
```

**Step 4**: Fine-tune all 7 scenes with LPIPS (script 12), re-render, redistort, package.

#### Key constraints
- Each training run takes ~1-2h on RTX 4090
- Full sweep (5×3×3 = 45 configs) = 45-90h → **too long**
- Recommended: Start with a coarse sweep (5×3×1 = 15 configs, fix `percent_dense=0.01`), then refine around the best region
- Coarse sweep ≈ 15-30h → fits in the remaining time window with a single GPU
- **Monitor VRAM**: Log `nvidia-smi` during training. If VRAM approaches 24GB, that config will OOM on larger scenes even if HCM0193 fits

#### VRAM monitoring helper (add to sweep script)
```bash
# Background VRAM logger — start before each training run
nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
  --format=csv -l 30 > "logs/vram_sweep_${TAG}.csv" &
VRAM_PID=$!
# ... training ...
kill ${VRAM_PID} 2>/dev/null || true
```

---

### 🥈 TASK P1: Auto-Extract & Centralize Lens Parameters

**Estimated score gain**: +0.0 to +0.3 (eliminates human copy-paste errors)  
**Time budget**: 1h  

#### Problem
Lens parameters (f, k, cx, cy) are hardcoded in:
1. [12_apply_finetune_and_rebuild.sh](file:///d:/VAR202_1/scripts/12_apply_finetune_and_rebuild.sh#L18-L25) — bash associative arrays
2. [README.md](file:///d:/VAR202_1/README.md#L181-L188) — markdown table
3. Manual CLI invocations

#### Implementation

**File to create**: `scripts/extract_lens_params.py`

```python
#!/usr/bin/env python3
"""Extract lens distortion parameters from cameras.bin and write to JSON."""
import json, struct, argparse
from pathlib import Path

CAMERA_MODEL_NPARAMS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8),
}

def read_camera(cameras_bin):
    with open(cameras_bin, "rb") as fid:
        num = struct.unpack("<Q", fid.read(8))[0]
        _, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
        name, n = CAMERA_MODEL_NPARAMS[model_id]
        params = struct.unpack("<" + "d" * n, fid.read(8 * n))
    return {"model": name, "width": width, "height": height, "params": list(params)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="configs/lens_params.json")
    args = ap.parse_args()
    
    result = {}
    for scene_dir in sorted(Path(args.data_dir).iterdir()):
        cam_bin = scene_dir / "train" / "sparse" / "0" / "cameras.bin"
        if not cam_bin.exists():
            continue
        cam = read_camera(cam_bin)
        if cam["model"] == "SIMPLE_RADIAL":
            f, cx, cy, k = cam["params"]
            result[scene_dir.name] = {"f": f, "cx": cx, "cy": cy, "k": k,
                                       "width": cam["width"], "height": cam["height"]}
        elif cam["model"] in ("PINHOLE", "SIMPLE_PINHOLE"):
            result[scene_dir.name] = {"model": cam["model"], "no_distortion": True}
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {len(result)} scenes to {args.out}")

if __name__ == "__main__":
    main()
```

**Then modify** [12_apply_finetune_and_rebuild.sh](file:///d:/VAR202_1/scripts/12_apply_finetune_and_rebuild.sh) to read from `configs/lens_params.json` instead of hardcoded arrays.

---

### 🥉 TASK P2: Redistortion Interpolation A/B Test

**Estimated score gain**: −0.1 to +0.3  
**Time budget**: 30min  

#### Implementation

**File to modify**: [10_redistort_renders.py](file:///d:/VAR202_1/scripts/10_redistort_renders.py#L104)

Add a `--interpolation` flag and test `INTER_CUBIC` vs `INTER_LANCZOS4`:

```python
# Add to argparse:
ap.add_argument("--interpolation", default="lanczos4",
                choices=["lanczos4", "cubic", "linear"],
                help="cv2 interpolation for remap")

# In the loop:
INTERP_MAP = {"lanczos4": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC,
              "linear": cv2.INTER_LINEAR}
interp = INTERP_MAP[args.interpolation]
out = cv2.remap(img, map_x, map_y, interpolation=interp,
                borderMode=cv2.BORDER_REPLICATE)
```

**Test on HCM0193**:
```bash
for interp in lanczos4 cubic; do
  python scripts/10_redistort_renders.py \
    --renders_dir submission_build/HCM0193 \
    --out_dir submission_build/HCM0193_${interp} \
    --f 925.1842594361348 --cx 660.0 --cy 494.5 --k 0.00795193982469231 \
    --interpolation ${interp}
  python scripts/05_eval_metrics.py \
    --pred_dir submission_build/HCM0193_${interp} \
    --gt_dir local_eval_gt_raw/HCM0193 --psnr_max 50
done
```

---

### TASK P3: ZIP_STORED for JPEGs

**File**: [04_make_submission.py](file:///d:/VAR202_1/scripts/04_make_submission.py#L117-L122)

Replace:
```python
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for scene_out_dir in scene_dirs:
        for f in sorted(scene_out_dir.iterdir()):
            if f.is_file():
                arcname = f"{scene_out_dir.name}/{f.name}"
                zf.write(f, arcname)
```

With:
```python
with zipfile.ZipFile(out_zip, "w") as zf:
    for scene_out_dir in scene_dirs:
        for f in sorted(scene_out_dir.iterdir()):
            if f.is_file():
                arcname = f"{scene_out_dir.name}/{f.name}"
                # JPEGs are already compressed — deflating them wastes CPU
                compress = (zipfile.ZIP_STORED if f.suffix.lower() in (".jpg", ".jpeg")
                           else zipfile.ZIP_DEFLATED)
                zf.write(f, arcname, compress_type=compress)
```

> [!WARNING]
> Verify the resulting zip stays under **350MB** (the contest limit). If ZIP_STORED pushes it over, revert to ZIP_DEFLATED.

---

### TASK P4: Parallelize I/O Loops

**Files to modify**:
- [00b_prepare_data.py](file:///d:/VAR202_1/scripts/00b_prepare_data.py#L138-L148) — `undistort_dir()`
- [10_redistort_renders.py](file:///d:/VAR202_1/scripts/10_redistort_renders.py#L99-L111) — main loop

Use `concurrent.futures.ThreadPoolExecutor` (I/O-bound work):

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_one(p, map_x, map_y, interp, border, out_dir, jpeg_quality):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return p.name, False
    out = cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=border)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_dir / p.name), out,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    else:
        cv2.imwrite(str(out_dir / p.name), out)
    return p.name, True

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_one, p, map_x, map_y, ...) for p in files]
    for f in as_completed(futures):
        name, ok = f.result()
        if not ok:
            print(f"  [warn] failed to read {name}")
```

---

### TASK P5: Fix cv2.imread Null Checks

**Files**:
- [09_postprocess_variants.py](file:///d:/VAR202_1/scripts/09_postprocess_variants.py#L42) — `compute_train_color_stats()` line 42
- [09_postprocess_variants.py](file:///d:/VAR202_1/scripts/09_postprocess_variants.py#L85) — variant generation loop line 85
- [10_redistort_renders.py](file:///d:/VAR202_1/scripts/10_redistort_renders.py#L100) — main loop line 100

Add null check after every `cv2.imread`:
```python
img = cv2.imread(str(p), cv2.IMREAD_COLOR)
if img is None:
    print(f"  [warn] failed to read {p}, skipping", file=sys.stderr)
    continue
```

---

### TASK P6: Fix TemporaryDirectory Leak

**File**: [00b_prepare_data.py](file:///d:/VAR202_1/scripts/00b_prepare_data.py#L309-L357)

Replace the manual `tmp_ctx` pattern with proper context management:

```python
# BEFORE (leaks on exception):
tmp_ctx = None
if args.zip_path:
    tmp_ctx = tempfile.TemporaryDirectory()
    extract_root = Path(tmp_ctx.name)
    ...
for scene_dir in scene_dirs:
    prepare_scene(...)
if tmp_ctx is not None:
    tmp_ctx.cleanup()

# AFTER (always cleans up):
from contextlib import ExitStack
with ExitStack() as stack:
    if args.zip_path:
        tmp_ctx = stack.enter_context(tempfile.TemporaryDirectory())
        extract_root = Path(tmp_ctx)
        ...
    for scene_dir in scene_dirs:
        prepare_scene(...)
```

---

### TASK P7: Shared Utils Module

**File to create**: `scripts/pipeline_utils.py`

```python
"""Shared constants and helpers for VAR202_1 pipeline scripts."""
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEST_CSV_NAMES = ("test_pose.csv", "test_poses.csv")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def find_test_csv(scene_dir: Path) -> Path | None:
    """Find test pose CSV under scene_dir/test/. Returns None if not found."""
    for fname in TEST_CSV_NAMES:
        p = scene_dir / "test" / fname
        if p.is_file():
            return p
    return None
```

Then replace all 4 copies of `find_test_csv` and 5 copies of `IMAGE_EXTS` with imports from this module.

---

## 3. Score Impact Model — Detailed Projections

### How each metric flows into score

```
     ┌─────────────────────────────────────────────────────────────┐
     │              Score = 0.4×(1−L) + 0.3×S + 0.3×(P/50)        │
     │                                                             │
     │  Densification  ──→  PSNR ↑↑  (+2 to +4 dB)               │
     │                 ──→  SSIM ↑↑  (+0.02 to +0.05)            │
     │                 ──→  LPIPS ↓   (−0.01 to −0.02)  indirect │
     │                                                             │
     │  Fine-tune v2   ──→  LPIPS ↓↓  (already diminishing)       │
     │  (on better ckpt)──→  PSNR ↑   (+0.5 to +1 dB) compound  │
     │                                                             │
     │  Interpolation  ──→  SSIM ↑    (+0.002 to +0.01)          │
     │  fix            ──→  LPIPS ↓   (−0.002 to −0.005)         │
     │                                                             │
     │  Bug/robustness ──→  No direct metric change               │
     │  fixes          ──→  Prevents silent failures              │
     └─────────────────────────────────────────────────────────────┘
```

### Compound effect: Fine-tune on BETTER checkpoint

The current fine-tune (+0.3 points) was applied to a baseline-default checkpoint. If the checkpoint is already better (from improved densification), fine-tuning starts from a higher quality model → the LPIPS loss gradient landscape is different → potentially larger gains from fine-tuning.

Estimated compound bonus: **+0.3 to +0.8 additional points** beyond the sum of individual effects.

---

## 4. Recommended Execution Order (Time-Boxed to 8 Days)

```
Day 1-2:  P1 (auto-extract lens params, 1h)
          P5+P6 (bug fixes, 20min)
          P0 START: Launch coarse densification sweep on HCM0193
              Grid: 5 grad_threshold × 3 densify_until = 15 configs
              Expected wall-clock: ~20-30h on single RTX 4090

Day 3:    P0 ANALYZE: Review sweep_results.csv
          P0 REFINE: Narrow grid around top-3 configs (3-6 more runs)
          P2: BICUBIC vs LANCZOS4 test while waiting (30min)

Day 4-5:  P0 APPLY: Retrain ALL 7 scenes with best config (~10-14h)
          Then fine-tune all 7 with script 12 (~7h)

Day 6:    Render + redistort + package new submission
          Submit to leaderboard for real scoring

Day 7:    If score improved: try second-best config as sanity check
          If not: investigate why (VRAM? different scene characteristics?)

Day 8:    Final submission with best variant
          P3+P4+P7 (speed + maintenance, do AFTER final submission is locked in)
```

---

## 5. Files Reference Map

| File | Lines | Role | Modification Priority |
|---|---|---|---|
| [00_setup_env.sh](file:///d:/VAR202_1/scripts/00_setup_env.sh) | 163 | Env setup | None |
| [00b_prepare_data.py](file:///d:/VAR202_1/scripts/00b_prepare_data.py) | 364 | Data conversion | P6 (temp dir fix) |
| [01_validate_scenes.py](file:///d:/VAR202_1/scripts/01_validate_scenes.py) | 139 | Validation | P7 (shared utils) |
| [02_train_scenes.sh](file:///d:/VAR202_1/scripts/02_train_scenes.sh) | 114 | Training loop | P0 (add densify params) |
| [03_render_novel_views.py](file:///d:/VAR202_1/scripts/03_render_novel_views.py) | 249 | Rendering | P7 (shared utils) |
| [04_make_submission.py](file:///d:/VAR202_1/scripts/04_make_submission.py) | 129 | ZIP packaging | P3 (ZIP_STORED) |
| [05_eval_metrics.py](file:///d:/VAR202_1/scripts/05_eval_metrics.py) | 116 | Scoring | P7 (shared utils) |
| [09_postprocess_variants.py](file:///d:/VAR202_1/scripts/09_postprocess_variants.py) | 98 | Post-processing | P5 (null checks) |
| [10_redistort_renders.py](file:///d:/VAR202_1/scripts/10_redistort_renders.py) | 118 | Lens redistortion | P2 (interp test), P5 |
| [11_finetune_perceptual.py](file:///d:/VAR202_1/scripts/11_finetune_perceptual.py) | 189 | LPIPS fine-tune | None (works well) |
| [12_apply_finetune_and_rebuild.sh](file:///d:/VAR202_1/scripts/12_apply_finetune_and_rebuild.sh) | 76 | Full pipeline | P0+P1 (densify+lens) |
| [configs/scenes_round2.yaml](file:///d:/VAR202_1/configs/scenes_round2.yaml) | 32 | Scene config | P0 (add densify keys) |

---

## 6. Critical Constraints & Gotchas

> [!CAUTION]
> **Do NOT change these — they are validated and working:**
> - JPEG quality=98, subsampling=0 in rendering (script 03)
> - `lambda_lpips=0.1`, `lambda_dssim=0.2` in fine-tuning (script 11)
> - Fine-tune iterations = 5000 (saturates at ~5000, see README §6.4)
> - The baseline `train.py` itself — **do not modify it** (contest rules)

> [!WARNING]
> **VRAM boundaries** (RTX 4090 24GB, ~17GB used at baseline defaults):
> - Lowering `densify_grad_threshold` → more Gaussians → more VRAM
> - Raising `densify_until_iter` → longer densification → more Gaussians
> - If peak VRAM exceeds ~22GB on HCM0193, it WILL OOM on larger BTS scenes
> - Always log VRAM during sweep (see Task P0 implementation)

> [!NOTE]
> **Scoring accuracy**: `05_eval_metrics.py` uses `--psnr_max 30` by default for local benchmarking. The **real leaderboard** uses `psnr_max=50`. When comparing to leaderboard scores, always pass `--psnr_max 50`. This only affects the PSNR normalization component, not SSIM or LPIPS.

---

## 7. Quick Validation Checklist

Before any leaderboard submission:

- [ ] All 7 scenes render without errors
- [ ] All 5 BTS scenes are redistorted (bonsai/chair are NOT)
- [ ] Image sizes match `test_poses.csv` dimensions (1320×989 for BTS)
- [ ] ZIP size < 350MB
- [ ] Spot-check 1 image visually (no black frames, no artifacts)
- [ ] Run `04_make_submission.py` validation (should report `OK` for all scenes)
- [ ] HCM0193 local benchmark score is ≥ previous best before committing to full submission
