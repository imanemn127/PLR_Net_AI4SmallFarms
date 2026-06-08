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

Sentinel-2 Level-2A, bands B4/B3/B2 (Red, Green, Blue), patches **256 × 256 px**,
minimum polygon area **100 px²**.

```
data/ai4sf_256px_area100/
  train_coco.json      98 patches    77 818 annotations
  val_coco.json        24 patches    20 168 annotations
  test_coco.json       18 patches    15 313 annotations
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
patch boundary, converts UTM to pixel coordinates, filters annotations below 100 px²,
and writes `train_coco.json`, `val_coco.json`, `test_coco.json` plus the patch images.

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

### Loss weights (default)

| Loss | Weight | Role |
|------|--------|------|
| `loss_jloc` | 8.0 | junction classification — bg / concave / convex |
| `loss_joff` | 0.25 | junction offset regression |
| `loss_mask` | 1.0 | binary segmentation mask BCE |
| `loss_afm` | 0.1 | attraction field map L1 |
| `loss_remask` | 1.0 | refined mask BCE |

### Plot training curves

```bash
python scripts/plot_losses_plrnet_ai4sf.py                  # latest run
python scripts/plot_losses_plrnet_ai4sf.py /path/to/run     # specific run
```

Produces `loss_curves_plrnet.png` with 3 panels: total loss (train vs val),
weighted component losses (train), weighted component losses (val).

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
