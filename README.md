# VAR2026 BTS Novel View Synthesis

Pipeline cho cuộc thi **Viettel AI Race 2026** — Vòng 1 sơ loại (đề bài: tái dựng
3D scene trạm BTS từ ảnh drone và sinh ảnh RGB tại các góc nhìn mới/chưa từng
chụp). Xây trên baseline chính thức
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(graphdeco-inria) — `train.py` gốc **không sửa đổi**; toàn bộ code trong repo
này chỉ là glue code để chuyển đổi data thi + sinh ảnh đúng định dạng nộp bài,
cộng với vài kỹ thuật tăng điểm được kiểm chứng qua benchmark thật (mục 6).

## 0. Trạng thái hiện tại (checkpoint đầu tiên lên GitHub)

| | |
|---|---|
| Deadline | 30/07/2026 (chỉ tính lần nộp **cuối cùng**, nộp lại thoải mái) |
| Giới hạn file nộp | 350MB/zip |
| Bản nộp tốt nhất hiện tại | **72.7025 điểm** (7/7 scene chấm được) |
| File tương ứng | `submission_round2_moderate_densify.zip` (323MB) |
| Cấu hình bản tốt nhất | full-res, sh_degree=3, 30000 iter (`--densify_grad_threshold 0.00015 --densify_until_iter 20000`) + fine-tune LPIPS nhẹ (5000 iter) + redistort ống kính + JPEG q98 — xem mục 6 & 8 |
| Việc còn dang dở | Thử tiếp `--densify_grad_threshold 0.00012` (giữa 0.00015 và 0.00010) — xem mục 8 |

Chi tiết điểm số qua từng lần nộp ở mục 7.

## 1. Cấu trúc thư mục

```
VAR2026_BTS_NVS/
├── README.md
├── requirements.txt              # deps riêng cho scripts/*.py (không gồm torch/CUDA)
├── .gitignore
├── configs/
│   ├── scenes.example.yaml       # mẫu
│   ├── scenes_round2.yaml        # 5 scene BTS + bonsai/chair — DÙNG CHO BẢN NỘP THẬT
│   └── scenes_private1.yaml      # data private_set1 cũ (đã bị BTC thay bằng round2)
├── external/
│   └── gaussian-splatting/       # KHÔNG commit — clone bằng scripts/00_setup_env.sh
├── data/                         # KHÔNG commit — sinh bằng scripts/00b_prepare_data.py
│   └── <scene>/
│       ├── train/images/, train/sparse/0/{cameras,images,points3D}.bin
│       └── test/test_pose(s).csv
├── output/                       # KHÔNG commit — checkpoint train.py (point_cloud.ply)
├── submission_build*/            # KHÔNG commit — ảnh PNG/JPG đã render, đúng cấu trúc nộp
├── local_eval_gt(_raw)/          # KHÔNG commit — ground-truth thật của scene HCM0193 (để tự chấm điểm)
├── logs/, logs_round2_gpu_run/   # KHÔNG commit — log train
├── environment_snapshot/         # KHÔNG commit — snapshot môi trường (rule 10.3)
└── scripts/                      # TOÀN BỘ CODE — xem bảng script ở mục 3
```

**Chỉ có `scripts/`, `configs/`, `README.md`, `requirements.txt` được commit.**
Mọi thứ khác (data ảnh, checkpoint, ảnh render, file zip nộp bài, log, snapshot môi
trường) đều to (hàng chục MB đến hàng chục GB) và **tái tạo được từ code + data gốc
BTC phát**, nên bị `.gitignore` loại — xem mục 4 để biết cách tái tạo từ đầu.

## 2. Cài đặt môi trường

```bash
git clone <URL repo này>
cd VAR2026_BTS_NVS
bash scripts/00_setup_env.sh
```

Script này:
1. Clone `graphdeco-inria/gaussian-splatting` (kèm submodule CUDA rasterizer) vào
   `external/gaussian-splatting/`.
2. Tạo conda env `var2026-3dgs` từ `environment.yml` gốc của baseline (torch +
   CUDA build sẵn cho rasterizer).
3. Cài thêm `requirements.txt` của repo này (pandas, lpips, scikit-image, opencv,
   pyyaml...) vào cùng env.

Yêu cầu hạ tầng: GPU NVIDIA CUDA ≥7.0, khuyến nghị ≥16GB VRAM cho train full-res
sh_degree=3 (xem mục 5 về việc phải thuê GPU rời — máy đội hiện tại 12GB VRAM
không đủ chạy full-res). Linux/WSL2 (baseline build native CUDA extension).

## 3. Tham chiếu toàn bộ script (chạy đúng theo thứ tự số)

| Script | Vai trò |
|---|---|
| `00_setup_env.sh` | Clone baseline + tạo conda env (mục 2) |
| `00b_prepare_data.py` | Giải nén zip BTC phát, gỡ méo ảnh (SIMPLE_RADIAL→PINHOLE), ghi ra `data/<scene>/` |
| `01_validate_scenes.py` | Kiểm tra nhanh mỗi scene đủ ảnh/pose trước khi tốn thời gian train |
| `02_train_scenes.sh` | Loop `train.py` gốc của baseline qua toàn bộ scene trong 1 file config yaml |
| `03_render_novel_views.py` | **Bước quan trọng nhất** — dựng camera ảo từ `test_pose(s).csv` và render (baseline gốc không hỗ trợ render pose mới hoàn toàn) |
| `04_make_submission.py` | Validate + đóng gói `submission_build*/` thành 1 file zip đúng cấu trúc nộp |
| `05_eval_metrics.py` | Tính PSNR/SSIM/LPIPS + công thức điểm cuộc thi (mục 6) trên 1 scene có ground-truth |
| `06_capture_environment.sh` | Snapshot conda/pip list + nvidia-smi + commit baseline repo, phòng BTC yêu cầu chứng minh reproducibility (rule 10.3) |
| `07_prepare_mvs_depth.sh` / `07b_convert_mvs_depth.py` | **Bỏ dở** — thử depth regularization qua COLMAP MVS, bị chặn vì bản COLMAP có sẵn (apt/conda-forge) không build kèm CUDA, cần COLMAP tự build từ source mới chạy được `patch_match_stereo` |
| `08_fix_points3d_track_refs.py` | Vá `points3D.bin` khi dùng loader nghiêm ngặt hơn (`pycolmap`, ví dụ cho gsplat) — baseline gốc không cần |
| `09_postprocess_variants.py` / `09b_eval_all_variants.sh` | Thử hậu xử lý ảnh (sharpen, color-match) — **đã thử, không cải thiện điểm**, giữ lại để tham khảo |
| `10_redistort_renders.py` | **Fix tăng điểm lớn nhất** — áp lại đúng độ méo ống kính gốc lên ảnh render trước khi nộp (mục 6) |
| `11_finetune_perceptual.py` | Fine-tune checkpoint đã train xong với thêm loss LPIPS (không cần train lại từ đầu) — mục 6 |
| `12_apply_finetune_and_rebuild.sh` | Tự động hoá: fine-tune + render + redistort + đóng gói cho cả 7 scene cùng lúc |

## 4. Quy trình đầy đủ, từ data thô đến file nộp

```bash
# 1. Chuẩn bị data (data thi thật hiện dùng VAI_NVS_DATA_ROUND2.zip — tải từ cổng
#    thi, KHÔNG có trong repo này vì quá nặng, xem mục 1 ghi chú "trạng thái")
python scripts/00b_prepare_data.py \
  --zip_path VAI_NVS_DATA_ROUND2.zip --out_dir data --gt_out_dir local_eval_gt

# 2. Kiểm tra data
python scripts/01_validate_scenes.py --data_dir data

# 3. Train từng scene (baseline gốc, không sửa)
bash scripts/02_train_scenes.sh configs/scenes_round2.yaml
#   Nếu GPU <16GB VRAM và OOM/paging chậm bất thường: thêm `-r 2` (train ở nửa
#   độ phân giải — không ảnh hưởng độ phân giải ảnh nộp cuối, xem comment trong
#   configs/scenes_round2.yaml)

# 4. (Khuyến nghị) Fine-tune nhẹ với loss LPIPS — xem mục 6 vì sao bước này đáng làm
bash scripts/12_apply_finetune_and_rebuild.sh   # tự làm luôn bước 5+6+7 cho cả 7 scene

# --- Hoặc làm thủ công từng bước (nếu không dùng script 12) ---
# 5. Render ảnh tại pose test
python scripts/03_render_novel_views.py --all \
  --repo_path external/gaussian-splatting \
  --data_dir data --output_root output --submission_root submission_build

# 6. Redistort (áp lại méo ống kính — BẮT BUỘC với 5 scene BTS, xem mục 6)
python scripts/10_redistort_renders.py \
  --renders_dir submission_build/<scene> --out_dir submission_build_final/<scene> \
  --f <fx scene> --cx 660.0 --cy 494.5 --k <k scene>   # tham số mỗi scene ở mục 6

# 7. Đóng gói
python scripts/04_make_submission.py \
  --submission_dir submission_build_final --data_dir data \
  --out_zip submission_round2_ft.zip
```

## 5. Vì sao cần thuê GPU rời

Train full-res + `sh_degree=3` (chất lượng cao nhất baseline hỗ trợ) làm số
Gaussian tăng nhanh qua densification, vượt VRAM GPU máy đội (12GB) → Windows/WDDM
âm thầm paging sang RAM thay vì báo lỗi, làm tốc độ chậm đi ~75 lần thay vì crash
rõ ràng. Giải pháp đã dùng: thuê 1× RTX 4090 24GB trên Vast.ai (~17GB VRAM thực tế
dùng khi train full-res sh=3 cho scene BTS lớn nhất, còn dư ~7GB — có thể là chỗ để
thử tăng mật độ Gaussian, xem mục 8).

Lưu ý chi phí: Vast.ai tính phí theo **thời gian instance ở trạng thái "running"**,
không phụ thuộc GPU có đang được dùng hay không, và không phụ thuộc có đang kết nối
SSH hay không — nhớ **Stop** (dừng tính phí compute, giữ lại ổ đĩa) hoặc **Destroy**
(dừng toàn bộ phí kể cả storage, nhưng **xoá vĩnh viễn dữ liệu trên đó**) khi không
dùng. Trước khi Destroy, luôn tải về: file submission zip cuối, `logs/train/*.log`,
và chạy `scripts/06_capture_environment.sh` rồi tải `environment_snapshot/` — 3 thứ
này là bằng chứng reproducibility (rule 10.3), không tái tạo lại được sau khi xoá
instance.

## 6. Công thức điểm & các kỹ thuật đã kiểm chứng giúp tăng điểm

**Công thức điểm** (dò ngược chính xác từ 1 kết quả leaderboard thật):

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × (PSNR / 50)
```

Trọng số PSNR+SSIM (0.6) nặng hơn LPIPS (0.4) — cần cân nhắc khi đánh đổi giữa các
kỹ thuật (ví dụ: đừng tăng lambda LPIPS quá tay, xem cảnh báo bên dưới).

### 6.1. Camera model thật là SIMPLE_RADIAL (có méo ống kính), không phải PINHOLE

`cameras.bin` thật dùng COLMAP model `SIMPLE_RADIAL` (hệ số méo `k`), nhưng
baseline `train.py` chỉ chấp nhận `PINHOLE`/`SIMPLE_PINHOLE` (crash ngay nếu không
xử lý). `00b_prepare_data.py` gỡ méo ảnh train bằng OpenCV trước khi đưa vào
baseline — bắt buộc, không có lựa chọn khác vì baseline không hỗ trợ méo ảnh.

### 6.2. [FIX LỚN NHẤT — điểm +5.5] Phải áp lại méo ống kính TRƯỚC khi nộp

Vì (6.1) train trên ảnh đã gỡ méo, ảnh render ra cũng là ảnh "pinhole lý tưởng"
(không méo) — nhưng ảnh ground-truth thật của BTC là ảnh chụp gốc (CÓ méo ống
kính). Nộp thẳng ảnh render pinhole sẽ bị lệch hình học so với GT, nặng nhất ở rìa
ảnh, khiến SSIM/LPIPS thấp giả tạo. Xác nhận bằng cách so ảnh render với 2 loại GT
của scene `HCM0193` (loại có ảnh test thật, dùng để tự chấm điểm): so với GT đã gỡ
méo ra SSIM 0.856, còn so với GT gốc (còn méo) chỉ ra SSIM 0.687 — gần khớp con số
leaderboard thật (SSIM 0.71) hơn nhiều → BTC chấm trên ảnh GỐC còn méo.

**Fix**: `scripts/10_redistort_renders.py` áp lại đúng méo ống kính (bằng
`cv2.undistortPoints` nghịch đảo + `cv2.remap`) lên ảnh render trước khi nộp.
Tham số ống kính từng scene (trích từ `cameras.bin` thật, `cx/cy` đều là tâm ảnh
1320×989 = 660/494.5):

| Scene | f | k |
|---|---|---|
| HCM0421 | 926.3566353543065 | 0.008944892859727953 |
| HCM0539 | 925.4471372744322 | 0.008103330827489163 |
| HCM0540 | 926.7065722882213 | 0.008867352646003301 |
| HCM0644 | 925.5047492526122 | 0.009033684294378248 |
| HCM0674 | 925.3208375156581 | 0.008810618762187384 |
| bonsai, chair | — (SIMPLE_PINHOLE, không méo) | copy thẳng, không redistort |

Kết quả thật trên leaderboard: 66.669 → **72.206**.

### 6.3. [FIX +1.4 điểm] JPEG quality mặc định của PIL/torchvision quá thấp

`torchvision.utils.save_image()` lưu `.jpg` ở quality=75 mặc định (nén mạnh, có
artifact block/ringing) — SSIM và đặc biệt LPIPS phạt nặng loại nhiễu nén này.
Đã sửa: lưu bằng PIL trực tiếp với `quality=98, subsampling=0` trong
`03_render_novel_views.py`. Kết quả thật: 65.214 → **66.669**.

### 6.4. [Cải thiện nhỏ, +0.3 điểm] Fine-tune thêm với loss LPIPS

Baseline gốc chỉ tối ưu `L1 + 0.2×D-SSIM`, **không bao giờ tối ưu LPIPS trực tiếp**
dù LPIPS là thành phần nặng nhất trong công thức điểm (0.4). `11_finetune_perceptual.py`
nạp lại checkpoint đã train xong (30000 iter) và fine-tune thêm 5000 iteration với
loss có thêm hạng tử LPIPS (`lambda_lpips=0.1`, tính trên crop ngẫu nhiên 512px để
tiết kiệm VRAM/thời gian; `densify_until_iter=0` — tắt densify khi fine-tune, chỉ
tinh chỉnh giá trị Gaussian có sẵn).

**Cảnh báo đã kiểm chứng — đừng tăng lambda_lpips quá tay**: thử `lambda_lpips=0.25`
+ `lambda_dssim=0.3` (đẩy mạnh hơn) làm LPIPS giảm thêm nhưng PSNR/SSIM giảm nhiều
hơn phần lãi → **điểm tổng còn tệ hơn cả không fine-tune**. `lambda_lpips=0.1` giữ
nguyên `lambda_dssim=0.2` mặc định là điểm cân bằng tốt nhất tìm được; chạy dài hơn
(10000 iter) gần như không cải thiện thêm (đã bão hoà — xem benchmark HCM0193:
0.8492 ở 5000 iter, 0.8493 ở 10000 iter).

Kết quả thật trên leaderboard: 72.206 → **72.52340** (PSNR 22.08→24.61, SSIM
70.04→81.49, LPIPS 22.62→16.73).

### 6.5. Các hướng đã thử và KHÔNG hiệu quả (đóng lại, không cần thử lại)

| Hướng | Kết quả |
|---|---|
| Train 60000 iteration thay vì 30000 | Không cải thiện đáng kể, tốn gấp đôi thời gian |
| gsplat MCMC densification strategy | Thua baseline gốc |
| Depth regularization (COLMAP MVS) | Bị chặn — COLMAP bản có sẵn (apt/conda-forge) không build kèm CUDA cho `patch_match_stereo` |
| Hậu xử lý ảnh: sharpen / color-match (`09_postprocess_variants.py`) | Mọi biến thể đều thua bản gốc không xử lý |
| Antialiasing (Mip-Splatting filter) — cả lúc train và chỉ lúc render | Thua rõ rệt, đặc biệt nếu chỉ bật lúc render trên checkpoint train không bật (SSIM 0.856→0.726) |
| Supersample (render 2x rồi downsample khi redistort) | Thua bản 1x — Lanczos4 không chống-alias đủ tốt khi downsample gộp chung với gỡ méo |
| Dịch tâm ảnh (cx,cy) nửa pixel khi redistort | Không đổi gì (quá nhỏ so với sai số nội suy) |
| Tune densify_grad_threshold: nhẹ / mạnh | Nhẹ = không đổi; mạnh = OOM crash. **Vùng giữa 2 mức này CHƯA được thử** — xem mục 8 |

## 7. Lịch sử điểm số (leaderboard thật)

| Ngày | Cấu hình | PSNR | SSIM | LPIPS | Điểm |
|---|---|---|---|---|---|
| — | Bản train đầu tiên (baseline mặc định) | — | — | — | 65.214 |
| 20/07/2026 | + JPEG quality fix (q98) | — | — | — | 66.669 |
| 20/07/2026 | + redistort ống kính (mục 6.2) | — | — | — | 72.206 |
| 20/07/2026 23:41 | + fine-tune LPIPS nhẹ (mục 6.4) | 24.61 | 81.49 | 16.73 | 72.52340 |
| 24/07/2026 | + moderate densification (`--densify_grad_threshold 0.00015 --densify_until_iter 20000`) — **bản tốt nhất hiện tại** | 24.579 | 81.727 | 16.407 | **72.7025** |

## 8. Kết quả thử nghiệm densification & hướng tiếp theo

### 8.1. Đã thử — moderate densification (+0.18 điểm)

| Cấu hình | grad_threshold | densify_until | Kết quả |
|---|---|---|---|
| Default | 0.00020 | 15000 | baseline (không đổi) |
| **Moderate** ✅ | **0.00015** | **20000** | **+0.18 điểm (72.523 → 72.703)** |
| Aggressive | 0.00010 | 25000 | Chưa thử (nguy cơ OOM) |

Cơ chế: SSIM tăng (81.49→81.73) và LPIPS giảm (16.73→16.41) nhờ nhiều Gaussian hơn
tái tạo chi tiết tốt hơn; PSNR hơi giảm nhẹ (24.61→24.58) — trade-off chấp nhận được.
Peak VRAM khi train moderate: **~19.5/24 GB** (RTX 4090), không OOM.

### 8.2. Hướng còn mở

- **`--densify_grad_threshold 0.00012`** (giữa moderate và aggressive) — có thể thêm
  điểm mà không OOM, nhưng cần thử. Peak VRAM ước ~21-22 GB.
- **Aggressive (`0.00010`, `25000`)** — rủi ro OOM cao hơn trên 24 GB VRAM. Nếu muốn
  thử: chuẩn bị fallback về moderate nếu crash.
- **`--percent_dense`** (mặc định 0.01) — chưa thử thay đổi, ít ảnh hưởng hơn 2 param trên.

Lưu ý: **không còn HCM0193 làm benchmark nội bộ** (Round 2 không có GT ảnh test),
nên mỗi lần thử phải nộp thẳng lên leaderboard để biết kết quả thật.

## 9. Tự đánh giá trước khi nộp (dùng scene `HCM0193`)

`HCM0193` là scene benchmark có ảnh ground-truth thật (từ `public_set` của
`VAI_NVS_DATA.zip` — zip Vòng 1 gốc, khác với `VAI_NVS_DATA_ROUND2.zip` dùng để
train 5 scene BTS thật). Dùng scene này để thử bất kỳ ý tưởng mới nào trước khi
áp dụng lên 7 scene thật, tránh tốn GPU/rủi ro nộp bài vào ý tưởng chưa kiểm chứng:

```bash
python scripts/00b_prepare_data.py \
  --zip_path VAI_NVS_DATA.zip --split public_set --scenes HCM0193 \
  --out_dir data --gt_out_dir local_eval_gt_raw   # GT KHÔNG gỡ méo — xem mục 6.2

python external/gaussian-splatting/train.py -s data/HCM0193/train -m output/HCM0193
python scripts/03_render_novel_views.py \
  --repo_path external/gaussian-splatting \
  --scene_dir data/HCM0193 --model_path output/HCM0193 \
  --out_dir submission_build/HCM0193

python scripts/10_redistort_renders.py \
  --renders_dir submission_build/HCM0193 --out_dir submission_build/HCM0193_redist \
  --f 925.1842594361348 --cx 660.0 --cy 494.5 --k 0.00795193982469231

python scripts/05_eval_metrics.py \
  --pred_dir submission_build/HCM0193_redist --gt_dir local_eval_gt_raw/HCM0193 --psnr_max 30
```

Mốc hiện tại trên `HCM0193` (bản 1x + redistort, chưa fine-tune): **PSNR 25.116 /
SSIM 0.8268 / LPIPS 0.1270 / score 0.8484**. Lưu ý `--psnr_max 30` ở đây chỉ là
thang đo cục bộ để so sánh tương đối giữa các thử nghiệm — công thức điểm thật của
BTC dùng `PSNR_max=50` (mục 6).

## 10. Lưu ý bám sát luật thi

- **Không dùng dữ liệu ngoài**: toàn bộ script chỉ dùng data BTC phát.
- **Không chỉnh sửa ảnh thủ công**: mọi ảnh trong `submission_build*/` do
  `03_render_novel_views.py` + `10_redistort_renders.py` tự sinh — cả 2 bước này
  đều là biến đổi tự động, xác định trước (deterministic), áp dụng đều cho mọi
  ảnh, không có bước chỉnh tay từng ảnh nào trong pipeline.
- **Khả năng tái lập (rule 10.3)**: giữ `output/<scene>/cfg_args` (train.py tự
  ghi), checkpoint `point_cloud.ply`, `logs/train/*.log`, và
  `environment_snapshot/` (mục 5) — 4 thứ BTC có thể yêu cầu nộp nếu vào top cao.
