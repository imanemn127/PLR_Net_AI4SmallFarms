# PLR-Net on AI4SmallFarms

This repository adapts [PLR-Net](https://github.com/mengmengli01/PLR-Net-demo)
— a Point-Line-Region interactive multi-task model originally designed for agricultural
parcel delineation from high-resolution GF-2 imagery (0.8 m/px) — to work with
**Sentinel-2** imagery (10 m/px) using the
[AI4SmallFarms](https://doi.org/10.17026/dans-xy6-ngg6) dataset
(Vietnam and Cambodia).

It is **not a reimplementation**. The model, backbone, encoder, and post-processing all
come from the original PLR-Net demo codebase. All original files are preserved;
adaptation code is integrated directly into the existing modules.
Several components missing from the original demo (`transforms.py`,
`multi_task_head.py`, training loop) were reconstructed by consulting the
[HiSup](https://github.com/SarahwXU/HiSup) codebase, which shares a common
architecture with PLR-Net.

---

## Why this is non-trivial

The original model was designed for GF-2 imagery at 0.8 m/px (uint8, RGB).
Sentinel-2 is 10 m/px, uint16, reflectance values in [0, 10000].
That gap creates several concrete problems:

| Problem | What breaks | Fix |
|---------|-------------|-----|
| uint16 reflectance | `io.imread().astype(float)` reads raw counts, normalisation wrong | Read with `rasterio`, divide by 10000, keep in [0, 1] |
| `TO_255: True` in original config | Multiplies [0, 1] values by 255, destroying normalisation | Set `TO_255: False`; transforms adapted for float input |
| `THC/THC.h` removed in PyTorch ≥ 1.9 | CUDA extension `afm_op` fails to compile | Replace with `c10/cuda/CUDAException.h`; use `data_ptr<T>()` and `C10_CUDA_CHECK` |
| `_download_url_to_file` removed from `torch.hub` | `model_zoo.py` crashes at import | Replace with `torch.hub.download_url_to_file` + `urllib.parse.urlparse` |
| Non-ASCII characters in source | `SyntaxError` at import in `polygon.py` and `encoder.py` | Remove offending characters |
| `np.long` removed in NumPy 1.24 | `TypeError` in dataset `__getitem__` | Replace with `np.int64` throughout |
| `hisup.*` imports in `train.py` | `ModuleNotFoundError` — package is named `PLRNet` | Replace all 9 imports |
| `ai4sf_train/val/test` not in catalog | `KeyError` at dataset build time | Add entries to `paths_catalog.py`; fix routing for `'val' in name` |
| `TestDataset` uses `PIL.Image.open` | Crashes on TIF uint16 | Rewrite `__getitem__` with rasterio; build same annotation fields as `TrainDataset` so `forward_train` works during validation |
| `forward()` always calls `forward_test` | No losses returned during training | Add `forward_train()` with 5 losses; dispatch on `self.training` |
| `transforms.py` missing from PLR-Net-demo | `ImportError` at dataset build | Recreated from HiSup reference |
| `multi_task_head.py` missing from PLR-Net-demo | `ModuleNotFoundError` at backbone build | Recreated from HiSup reference |
| Training loop missing entirely | — | Written from scratch, using HiSup `train.py` as starting point |

---

## Dataset

Sentinel-2 Level-2A, bands B4/B3/B2 (Red, Green, Blue), patches **256 × 256 px**.

Two versions were built at different `MIN_AREA` thresholds:

| Version | Min area | Train ann | Val ann | Test ann |
|---------|----------|-----------|---------|----------|
| `ai4sf_256px_area100` | 100 px² | 8 205 | ~2 000 | ~1 500 |
| `ai4sf_256px_area50` | 50 px² | 27 331 | ~7 000 | ~5 000 |

Current active dataset: **`ai4sf_256px_area50`** (set in `PLRNet/config/paths_catalog.py`).

```
data/ai4sf_256px_area50/
  train_coco.json      98 patches    27 331 annotations
  val_coco.json        24 patches
  test_coco.json       18 patches
  train/patches_256/
  validate/patches_256/
  test/patches_256/
```

Normalisation stats computed on training split (reflectance in [0, 1]):

| Channel | Mean   | Std    |
|---------|--------|--------|
| B4 (R)  | 0.1036 | 0.0540 |
| B3 (G)  | 0.0983 | 0.0346 |
| B2 (B)  | 0.0688 | 0.0309 |

---

## Repository structure

```
PLR-Net/
├── config-files/
│   └── PLR-Net.yaml              training configuration
├── PLRNet/
│   ├── backbones/                BsiNet-v2 + multi-task head (reconstructed)
│   ├── config/
│   │   ├── defaults.py
│   │   ├── paths_catalog.py      ai4sf_train/val/test entries added
│   │   └── dataset.py
│   ├── csrc/lib/afm_op/          custom CUDA Attraction Field Map op
│   ├── dataset/
│   │   ├── transforms.py         created: Resize/ToTensor/Normalize for Sentinel-2
│   │   ├── train_dataset.py      rasterio TIF reader; np.int64 fix
│   │   └── test_dataset.py       rewritten: same fields as TrainDataset, no augmentation
│   ├── detector.py               forward_train() with 5 losses; Dropout2d(p=0.1) in heads
│   ├── encoder.py                SyntaxError fix
│   ├── solver.py
│   └── utils/
│       ├── model_zoo.py          torch.hub private symbols replaced
│       ├── polygon.py            SyntaxError fix; top-K reduced 600→300
│       └── metrics/              cIoU, Polis, junction eval
├── scripts/
│   ├── build_coco_dataset.py     Sentinel-2 TIF tiles → 256px COCO patches
│   ├── compute_normalization.py  per-channel mean/std on training patches
│   ├── dataset_stats.py          annotation statistics and area threshold analysis
│   ├── tif_to_png.py             uint16 TIF → uint8 PNG for quick visualisation
│   ├── infer_one.py              single-image forward pass smoke test
│   ├── train.py                  training loop (written for this adaptation)
│   ├── test.py                   evaluation script
│   ├── eval_testA.py             val-set inference: mask IoU, junction recall, snap rate
│   └── plot_losses_plrnet_ai4sf.py  plot metrics.csv curves
└── tools/
    ├── evaluation.py
    └── mask2json.py
```

---

## Setup

### 1. Create the environment

```bash
conda create -n ai4sf python=3.11 -y
conda activate ai4sf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install rasterio "yacs==0.1.8" --no-deps
```

### 2. Compile the CUDA AFM extension

```bash
cd PLRNet/csrc/lib/afm_op
python setup.py build_ext --inplace
cd ../../../..
```

The compiled `.so` is placed in `PLRNet/csrc/lib/afm_op/` and loaded automatically
at import time. Deprecation warnings from cuSPARSE are expected and harmless.

### 3. Set paths

Edit `config-files/PLR-Net.yaml` and set `OUTPUT_DIR` to an absolute path.

---

## Dataset preparation

The AI4SmallFarms dataset is available at:
> https://doi.org/10.17026/dans-xy6-ngg6

Sentinel-2 tiles (multi-band GeoTIFF) and field polygons (GeoPackage) for Vietnam and
Cambodia. The raw download uses `validate/` as the split name (not `val/`).

```
sentinel-2-asia/
├── train/images/      *.tif
├── validate/images/   *.tif
├── test/images/       *.tif
└── reference/         *_fields.gpkg
```

### Build the COCO patches

```bash
python scripts/build_coco_dataset.py
```

Splits each tile into non-overlapping 256×256 px patches, clips field polygons to each
patch boundary, converts UTM to pixel coordinates, filters annotations below `MIN_AREA_PX`
(set at the top of the script — currently 50 px²), and writes `train_coco.json`,
`val_coco.json`, `test_coco.json` plus the patch images.

### Compute normalisation stats

```bash
python scripts/compute_normalization.py
```

Update `PIXEL_MEAN` and `PIXEL_STD` in `config-files/PLR-Net.yaml` with the output.

### Smoke test

```bash
python scripts/infer_one.py
```

Single forward pass (inference mode). Saves mask and polygon overlay to
`PLR-Net_output/infer_one/`.

---

## Training

```bash
python scripts/train.py --config-file config-files/PLR-Net.yaml
# change validation frequency (default: every 5 epochs)
python scripts/train.py --config-file config-files/PLR-Net.yaml --val-every 10
```

Each run creates a timestamped directory under `OUTPUT_DIR`:

```
OUTPUT_DIR/YYYY-MM-DD_HH-MM-SS/
├── train.log
├── config.yml
├── metrics.csv
├── checkpoints/
│   ├── latest.pth           overwritten every epoch
│   ├── epoch_5.pth          saved at each validation epoch
│   └── best_val_loss.pth    updated whenever val_loss improves
└── visualizations/
    ├── val/    {epoch:03d}_{patch_name}.png   GT | Pred side-by-side
    └── train/  {epoch:03d}_{patch_name}.png   same, 2 fixed patches
```

`metrics.csv` columns:

| Column | Description |
|--------|-------------|
| `epoch` | epoch index |
| `train_loss` | weighted total training loss |
| `w_loss_*` | weighted component loss (train) |
| `val_loss` | weighted total val loss (every `val_every` epochs) |
| `val_w_loss_*` | weighted component loss (val) |
| `val_mask_iou` | mean IoU of `remask_pred` vs GT mask (threshold 0.5, val patches only) |

### Loss weights (current — Run 7 and onwards)

| Loss | Weight | Role |
|------|--------|------|
| `loss_jloc` | 8.0 | junction classification — bg / concave / convex |
| `loss_joff` | 0.0 | junction offset regression (disabled — see run history) |
| `loss_mask` | 1.0 | binary segmentation mask BCE |
| `loss_afm` | 0.1 | attraction field map L1 |
| `loss_remask` | 1.0 | refined mask BCE |

### Plot training curves

```bash
python scripts/plot_losses_plrnet_ai4sf.py                  # latest run
python scripts/plot_losses_plrnet_ai4sf.py /path/to/run     # specific run
```

Produces `loss_curves_plrnet.png` with 4 panels: total loss (train vs val),
weighted component losses (train), weighted component losses (val), val mask IoU.

---

## Monitoring

```bash
tail -f OUTPUT_DIR/YYYY-MM-DD_HH-MM-SS/train.log
watch -n 1 nvidia-smi
```

---

## Training history

### Run 1 — baseline (150 epochs)

| Loss | Train (epoch 1 → 150) | Val (epoch 1 → 150) |
|------|----------------------|---------------------|
| `loss_mask` | 0.54 → 0.08 | 0.47 → 0.72 |
| `loss_jloc` | 4.07 → 0.33 | 0.46 → 0.49 |
| `loss_remask` | 0.72 → 0.62 | ~0.69 (flat) |
| `loss_afm` | 0.43 → 0.25 | ~0.41 (slow) |
| `loss_joff` | 0.127 (flat) | 0.127 (flat) |
| `val_total_loss` | — | rises after epoch 50: ~2.08 → ~2.40 |

**Diagnosis:** massive overfitting on the mask and junction branches; AFM and offset
branches learn almost nothing. No coherent polygons in validation visualisations.
Root cause: only 98 training patches, no dropout, no D4 augmentation.

### Run 2 — regularisation fixes (stopped at epoch 79/150)

Three changes applied after run 1:

| Change | Where | Why |
|--------|-------|-----|
| D4 augmentation (`ROTATE_F: True`) | `config-files/PLR-Net.yaml` | 98 patches × 8 orientations = ~784 effective configs; fields have no preferred orientation |
| `Dropout2d(p=0.1)` in heads | `PLRNet/detector.py` — `_make_conv` | Forces each channel to learn independently; applied to mask, jloc, afm heads |
| Top-K junctions 600 → 300 | `PLRNet/utils/polygon.py` | Reduces false-positive junction candidates; article value for the original dataset |

| Loss (weighted) | Train (ep 1 → 79) | Val (ep 5 → 75) |
|------|-------------------|-----------------|
| total | 6.09 → 1.93 | 2.18 → 2.15 |
| `w_loss_jloc` | 4.27 → 0.42 | 0.46 → 0.43 |
| `w_loss_mask` | 0.54 → 0.32 | 0.48 → 0.49 |
| `w_loss_afm` | 0.43 → 0.39 | 0.42 → 0.40 |
| `w_loss_remask` | 0.72 → 0.68 | 0.69 → 0.71 |
| `w_loss_joff` | 0.127 → 0.122 | 0.123 → 0.119 |

**Diagnosis:** The train/val gap is well controlled (1.93 vs 2.15 at epoch 75, gap ~0.20) —
D4 + Dropout successfully prevented the mask overfitting seen in run 1. The `w_loss_mask`
on val stays nearly flat (0.48 → 0.49) while train improves (0.54 → 0.32), which
indicates the mask branch is no longer memorising. Visually, large rectangular shapes
begin to appear on dense patches by epoch 50 and loosely follow field edges at epoch 75,
but boundary precision remains poor and the model still produces many spurious polygons
on sparse patches. The `loss_joff` branch barely moves (0.127 → 0.122) — junction
offset regression is not learning, which limits vertex placement accuracy.

### Run 3 — MIN_AREA 100→50 + val mask IoU tracking (150 epochs, complete)

Two changes compared to run 2:

| Change | Where | Why |
|--------|-------|-----|
| `MIN_AREA` 100 → 50 px² | `scripts/build_coco_dataset.py` | Recovers small fields; annotation count 8 205 → 27 331 on train (×3.3) |
| Val mask IoU added to metrics | `scripts/train.py`, `detector.py` | Measures segmentation quality independently of loss magnitude |

Dataset rebuilt: `data/ai4sf_256px_area50/` (98 train / 24 val / 18 test patches).

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 5 → 150) |
|------|-------------------|-----------------|
| total | 6.47 → 2.28 | 3.13 → 3.09 |
| `w_loss_jloc` | 4.43 → 0.89 | 1.11 → 1.08 |
| `w_loss_mask` | 0.65 → 0.25 | 0.63 → 0.70 |
| `w_loss_afm` | 0.56 → 0.42 | 0.57 → 0.51 |
| `w_loss_remask` | 0.70 → 0.60 | 0.70 → 0.68 |
| `w_loss_joff` | 0.127 → 0.122 | 0.126 → 0.122 |
| **`val_mask_iou`** | — | **0.430 (ep 5) → 0.417 (ep 150), best 0.457 (ep 25)** |

**Diagnosis:** Annotation count tripled (8 205 → 27 331 train) which raised the absolute loss
level but generalisation remains stable — train/val gap ~0.81 at epoch 150.
`val_mask_iou` peaks at 0.457 (epoch 25) then declines to 0.417; mask loss keeps
decreasing while IoU does not follow, indicating the model overfits annotation density
rather than learning sharper boundaries. `val_w_loss_jloc` spikes at late epochs (1.13
at epoch 145), likely due to the higher density of small polygons. `loss_joff` flat
across all three runs (0.127 → 0.122) — junction offset regression has not learned and
remains the main bottleneck for polygon accuracy.

Note: `paths_catalog.py` pointed to `ai4sf_256px_area50` for this run. Confirmed retroactively at run 7.

### Run 4 — `loss_joff` weight 0.25 → 1.0 (125 epochs)

Note: a Git stash + rebase after run 3 restored `paths_catalog.py` to its default value,
pointing to area100 (8 205 train ann) instead of area50 (27 331). Runs 4, 5, and 6 all
trained on area100 as a result — metrics are not directly comparable to run 3. Detected
retroactively at run 7.

Before relaunching I wanted to understand why `loss_joff` has been flat since the start (0.127 → 0.122 across 3 runs). I wrote `scripts/diag_joff.py` to inspect the targets directly in the data pipeline — do the offsets actually exist in the GT, or is it a data problem?

The script loads a few batches from the train loader, calls the encoder to materialise the GT tensors, and prints stats on `joff_gt` restricted to junction pixels only.

Result over 5 batches:

```
Total junction pixels : 1452 / 327680 (0.44%)
joff_x  min=-0.500  max=0.499  mean=0.006  std=0.294
joff_y  min=-0.500  max=0.499  mean=-0.011  std=0.292
fraction |offset| > 1e-6 : 1.000
```

The offsets are there, well distributed, spanning the full ±0.5 range. So it's not a data problem — the branch *could* learn.

The real issue: junction pixels are only **0.44 %** of the total. The joff gradient is microscopic compared to `loss_jloc × 8.0` which covers every pixel. At weight 0.25, the optimiser essentially ignores the joff head.

| Change | Where | Why |
|--------|-------|-----|
| `loss_joff` weight 0.25 → 1.0 | `config-files/PLR-Net.yaml` | Junction pixels = 0.44% of image; offset head starved of gradient at weight 0.25 |

| Loss (weighted) | Train (ep 1 → 125) | Val (ep 5 → 125) |
|------|-------------------|-----------------|
| total | 6.47 → 1.97 | 2.57 → 2.52 |
| `w_loss_joff` | 0.509 → 0.382 | 0.492 → 0.491 |
| `w_loss_jloc` | 4.27 → 0.385 | 0.463 → 0.442 |
| `w_loss_mask` | 0.544 → 0.213 | 0.490 → 0.489 |
| `w_loss_afm` | 0.433 → 0.332 | 0.426 → 0.407 |
| `w_loss_remask` | 0.720 → 0.658 | 0.694 → 0.696 |
| **`val_mask_iou`** | — | **0.132 (ep 5) → 0.167 (ep 125), best 0.230 (ep 115)** |

**Diagnosis:** `loss_joff` finally learns (0.509 → 0.382 train) confirming the branch can
move with enough weight. But `val_mask_iou` is very volatile (0.066 at ep 10, 0.230 at
ep 115) and overall much lower than run 3 best (0.457) — the mask branch is losing
gradient to the joff head. `loss_jloc` (weight 8.0) and `loss_joff` (weight 1.0) share
the same junction feature map; raising joff without rebalancing jloc shifts the
representation away from corner classification. Visually, Vietnam val produces large
polygons that do not match the small GT parcels — the model merges many fields into
coarse shapes. Cambodia val generates dense false positives scattered across the entire
urban area with no correspondence to the GT annotations.

**Next — Run 5:** keep `loss_joff: 1.0`, reduce `loss_jloc: 8.0 → 4.0` to rebalance
the two heads on the junction feature map without sacrificing mask quality.

### Run 5 — `loss_jloc` weight 8.0 → 4.0, `loss_joff` 1.0 (150 epochs, complete)

Note: same dataset issue as run 4 — area100 at launch, not area50 as intended (see run 4 note).

| Change | Where | Why |
|--------|-------|-----|
| `loss_jloc` weight 8.0 → 4.0 | `config-files/PLR-Net.yaml` | `loss_jloc` and `loss_joff` share the same junction feature map; reducing jloc weight frees gradient budget for the offset head |

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 5 → 150) |
|------|-------------------|-----------------|
| total | 4.34 → 1.78 | 2.31 → 2.32 |
| `w_loss_joff` | 0.509 → 0.388 | 0.492 → 0.484 |
| `w_loss_jloc` | 2.14 → 0.197 | 0.231 → 0.222 |
| `w_loss_mask` | 0.543 → 0.215 | 0.469 → 0.504 |
| `w_loss_afm` | 0.433 → 0.324 | 0.420 → 0.410 |
| `w_loss_remask` | 0.720 → 0.659 | 0.694 → 0.700 |
| **`val_mask_iou`** | — | **0.178 (ep 5) → 0.198 (ep 150), best 0.210 (ep 85)** |

**Diagnosis:** Reducing `loss_jloc` to 4.0 did not accelerate `loss_joff` learning
(0.509 → 0.388 over 150 epochs, same pace as run 4). `val_mask_iou` best is 0.210
at epoch 85 — identical to run 4 best — so rebalancing brought no improvement on the
offset side. After epoch 100 the gap on `w_loss_mask` widens (train 0.215, val
0.504–0.541), indicating the mask branch starts overfitting again. Visually, Vietnam
val generates large polygons covering whole zones that do not match the small GT
parcels — the model is not learning fine boundaries. Cambodia val produces scattered
false positives across the entire urban area with no correspondence to the few GT
annotations. The conclusion across runs 3–5: with only 0.44% of pixels active,
`loss_joff` cannot be improved by reweighting alone regardless of how `loss_jloc`
is set. Runs 4 and 5 both degraded `val_mask_iou` relative to run 3 (best 0.457).

**Next — Run 6:** disable `loss_joff` (weight 0.0), restore `loss_jloc: 8.0`. Goal:
recover the mask IoU of run 3 (best 0.457) without the joff head competing for gradients.

### Run 6 — `loss_joff: 0.0`, `loss_jloc: 8.0`, dataset area100 (125 epochs)

| Change | Where | Why |
|--------|-------|-----|
| `loss_joff` weight 1.0 → 0.0 | `config-files/PLR-Net.yaml` | Runs 4–5 showed joff head degrades mask IoU; disabling removes the competition |
| `loss_jloc` weight 4.0 → 8.0 | `config-files/PLR-Net.yaml` | Restore original balance after run 5 experiment |

Note: `paths_catalog.py` still pointed to area100 at launch (same issue as runs 4–5).

| Loss (weighted) | Train (ep 1 → 125) | Val (ep 5 → 125) |
|------|-------------------|-----------------|
| total | 5.97 → 1.63 | 2.04 → 2.00 |
| `w_loss_joff` | 0.0 (disabled) | 0.0 (disabled) |
| `w_loss_jloc` | 4.27 → 0.391 | 0.456 → 0.435 |
| `w_loss_mask` | 0.543 → 0.238 | 0.473 → 0.462 |
| `w_loss_afm` | 0.433 → 0.341 | 0.419 → 0.406 |
| `w_loss_remask` | 0.722 → 0.662 | 0.693 → 0.694 |
| **`val_mask_iou`** | — | **0.112 (ep 5) → 0.150 (ep 125), best 0.220 (ep 85)** |

**Diagnosis:** best IoU is 0.220 — well below run 3 (0.457). At this point the dataset
issue was not yet identified; the low IoU was attributed to the loss configuration. Val
Cambodia ep 125: dense false positives covering the whole scene, no GT match. Val Vietnam
ep 125: large coarse polygons, missing fine field edges.

**Next — Run 7:** fix `paths_catalog.py` to area50, keep `loss_joff: 0.0`, `loss_jloc: 8.0`.

### Run 7 — `loss_joff: 0.0`, `loss_jloc: 8.0`, dataset area50 (150 epochs, complete)

| Change | Where | Why |
|--------|-------|-----|
| Dataset area100 → area50 | `PLRNet/config/paths_catalog.py` | Identified that runs 4–6 had used area100; reverted to area50 to isolate the dataset effect |

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 5 → 150) |
|------|-------------------|-----------------|
| total | 6.35 → 2.26 | 2.99 → 2.92 |
| `w_loss_joff` | 0.0 (disabled) | 0.0 (disabled) |
| `w_loss_jloc` | 4.44 → 0.912 | 1.105 → 1.067 |
| `w_loss_mask` | 0.646 → 0.300 | 0.621 → 0.650 |
| `w_loss_afm` | 0.560 → 0.434 | 0.569 → 0.512 |
| `w_loss_remask` | 0.700 → 0.610 | 0.690 → 0.687 |
| **`val_mask_iou`** | — | **0.426 (ep 5) → 0.441 (ep 150), best 0.450 (ep 15)** |

**Diagnosis:** IoU reaches 0.426 at ep 5 and peaks at 0.450 (ep 15), confirming area50
is the key variable. After ep 15 it oscillates 0.39–0.45 with no upward trend.
`val_w_loss_mask` rises (0.621 → 0.650) while train mask loss falls (0.646 → 0.300) —
mask branch overfitting. The model fails on low-contrast uniform-texture patches (Cambodia
small paddy grids): near-zero detections despite clear GT. IoU ceiling at ~0.45 is not a
loss-weight issue.

### Test A — post-processing only, no retraining (checkpoint epoch 95)

The idea: before launching another training run, test whether tweaking the post-processing
thresholds can recover some polygon quality from the existing Run 7 checkpoint.

Two changes in `PLRNet/utils/polygon.py`:

| Change | Before | After | Why |
|--------|--------|-------|-----|
| NMS score threshold | 0.008 | 0.004 | Try to recover corners the network predicts with low confidence |
| Contour–junction matching distance | 5 px | 3 px | Avoid snapping to wrong junctions on neighbouring parcels |

Evaluated on all 24 val images with `scripts/eval_testA.py` (epoch 95 checkpoint):

| Metric | Value |
|--------|-------|
| Mean mask IoU | 0.399 |
| Mean GT polygons / image | 283.4 |
| Mean predicted polygons / image | 3.8 |
| Junction candidates / image | 600 (hard cap, hit on every image) |
| Snap rate | 1.00 |
| Junction recall @3 px | 0.36 |
| Junction recall @5 px | 0.54 |
| Junction recall @8 px | 0.74 |

**What the numbers mean:**

The most striking thing is the gap between 3.8 predicted polygons and 283 GT polygons.
The mask branch merges all adjacent parcels into a handful of large blobs — once the mask
has merged them, there is no way to split them back out in post-processing. Snap rate = 1.00
means each detected blob does find junctions nearby, so the matching step is not the problem.

The NMS threshold change did nothing because the 600-candidate cap is already hit at 0.008.
The network produces a diffuse heatmap with many weak activations rather than sharp peaks on
real corners — so flooding it with more candidates at 0.004 just adds more noise, not better corners.

Junction recall @3px = 0.36 with 600 candidates on a 256×256 image is barely better than
random. If the predictions were uniformly random, you'd expect roughly one point every 11 px,
giving ~0.30 recall at 3 px. So the jloc branch is placing corners slightly better than
chance, but not by much — it has not learned to produce confident, localised predictions.

**Actions taken after Test A:**

- NMS threshold reverted to 0.008 (lowering had no effect, cap was already hit)
- Matching threshold kept at 3 px (snap rate unaffected, no regression)
- Root cause confirmed: the mask branch needs to separate adjacent parcels;
  the jloc branch needs more training time at a high learning rate → Test B

---

## Troubleshooting

**`ModuleNotFoundError: PLRNet.csrc.lib.afm_op.CUDA`**
Extension not compiled. Run `python setup.py build_ext --inplace` from
`PLRNet/csrc/lib/afm_op/`.

**`THC/THC.h: No such file or directory`**
THC headers removed in PyTorch ≥ 1.9. Already fixed in this repo — just rebuild.

**`KeyError: 'edges_positive'` during validation**
The val dataset was not producing annotation fields compatible with `forward_train`.
Fixed in `test_dataset.py`.

**`NotImplementedError` in `DatasetCatalog.get`**
`ai4sf_val` routed incorrectly — needed `'val' in name` check. Fixed in `paths_catalog.py`.

**Images very dark in visualisations**
Percentile stretch (p2–p98) applied per channel at display time. Check `TO_255: False`
is set in the yaml.

**`CUDA out of memory`**
Lower `IMS_PER_BATCH` in `config-files/PLR-Net.yaml`.

---

## Citation

```bibtex
@article{li2025plrnet,
  author  = {Mengmeng Li and Chengwen Lu and Mengjing Lin and Xiaolong Xiu
             and Jiang Long and Xiaoqin Wang},
  title   = {Extracting vectorized agricultural parcels from high-resolution
             satellite images using a Point-Line-Region interactive multitask model},
  journal = {Computers and Electronics in Agriculture},
  volume  = {231},
  pages   = {109953},
  year    = {2025},
  doi     = {10.1016/j.compag.2025.109953}
}

@article{xu2023hisup,
  title   = {HiSup: Accurate polygonal mapping of buildings in satellite imagery
             with hierarchical supervision},
  author  = {Bowen Xu and Jiakun Xu and Nan Xue and Gui-Song Xia},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {198},
  pages   = {284--296},
  year    = {2023},
  doi     = {10.1016/j.isprsjprs.2023.03.006}
}

@dataset{ai4smallfarms2024,
  title = {AI4SmallFarms},
  year  = {2024},
  doi   = {10.17026/dans-xy6-ngg6},
  url   = {https://doi.org/10.17026/dans-xy6-ngg6}
}
```

Original model: [mengmengli01/PLR-Net-demo](https://github.com/mengmengli01/PLR-Net-demo)  
HiSup reference: [SarahwXU/HiSup](https://github.com/SarahwXU/HiSup)
