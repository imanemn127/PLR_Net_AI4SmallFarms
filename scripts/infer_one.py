#!/usr/bin/env python3
"""
infer_one.py  —  PLR-Net / AI4SmallFarms
=========================================
Run a single forward pass of PLR-Net on one image (PNG or GeoTIFF) and
save the predicted segmentation mask and polygon overlay.

This script bypasses the full test_pipelines / DatasetCatalog machinery so
that you can quickly verify the environment (CUDA, imports, model loading)
without needing a trained checkpoint or a properly configured dataset.

When no checkpoint exists yet (cold start), the model runs with random
weights — the mask and polygons will be meaningless, but the pipeline
running without errors confirms the environment is healthy.

Outputs go to /mnt/DATA/IMANE/PLR-Net_output/infer_one/ by default.

Usage:
  # Minimal — random weights, one image
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/infer_one.py \
      --image data/sample_test.png \
      --config config-files/PLR-Net.yaml

  # With a trained checkpoint
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/infer_one.py \
      --image data/sample_test.png \
      --config config-files/PLR-Net.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/train/model_00050.pth

  # Directly from a GeoTIFF (converted internally)
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/infer_one.py \
      --image data/train/patches_256/0_vietnam_00000_00000.tif \
      --config config-files/PLR-Net.yaml
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from skimage import io

# Make sure the project root is on the Python path when called from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector

DEFAULT_OUTPUT = "/mnt/DATA/IMANE/PLR-Net_output/infer_one"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load a PNG, JPG, or 3-band GeoTIFF.

    All images are returned as float64 in [0, 1] — the same scale
    expected by the training pipeline (TO_255: False, mean/std on [0, 1]).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3]).astype(np.float64) / 10000.0  # (3,H,W) → [0,1]
        return data.transpose(1, 2, 0)   # (H, W, 3)

    # PNG / JPG: uint8 [0-255] → [0, 1]
    image = io.imread(path).astype(np.float64) / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image[:, :, :3]


# ---------------------------------------------------------------------------
# Pre-processing  (mirrors PLRNet/dataset/build.py transforms)
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, cfg) -> torch.Tensor:
    """
    Resize to (IMAGE.HEIGHT x IMAGE.WIDTH), normalise, return (1,3,H,W) tensor.
    """
    h = cfg.DATASETS.IMAGE.HEIGHT
    w = cfg.DATASETS.IMAGE.WIDTH
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    mean = np.array(cfg.DATASETS.IMAGE.PIXEL_MEAN, dtype=np.float32)
    std  = np.array(cfg.DATASETS.IMAGE.PIXEL_STD,  dtype=np.float32)
    if cfg.DATASETS.IMAGE.TO_255:
        mean = mean * 255.0
        std  = std  * 255.0

    image = (image.astype(np.float32) - mean) / std
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(image_orig: np.ndarray, output: dict, out_dir: str) -> None:
    """Save predicted mask, polygon overlay and junction map to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    mask = output["mask_pred"][0]                          # (H, W) float [0,1]
    h, w = mask.shape

    # 1. Raw mask probability map
    cv2.imwrite(
        os.path.join(out_dir, "mask_pred.png"),
        (mask * 255).astype(np.uint8)
    )

    # 2. Polygon overlay on the original image
    # Percentile stretch (p2-p98) per channel for display only — does not
    # affect the normalisation used by the model.
    vis_float = image_orig.copy()
    for c in range(vis_float.shape[2]):
        p2, p98 = np.percentile(vis_float[:, :, c], (2, 98))
        if p98 > p2:
            vis_float[:, :, c] = (vis_float[:, :, c] - p2) / (p98 - p2)
    vis = cv2.resize((vis_float * 255).clip(0, 255).astype(np.uint8), (w, h))
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255)]
    polys = output["polys_pred"][0] if output["polys_pred"] else []
    for i, poly in enumerate(polys):
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True,
                      color=colors[i % len(colors)], thickness=1)
    cv2.imwrite(os.path.join(out_dir, "polygons_overlay.png"), vis)

    # 3. Junction map
    juncs = output["juncs_pred"][0]                        # (N, 2)
    junc_map = np.zeros((h, w), dtype=np.uint8)
    for x, y in juncs:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        cv2.circle(junc_map, (xi, yi), 2, 255, -1)
    cv2.imwrite(os.path.join(out_dir, "junctions.png"), junc_map)

    print(f"  mask_pred.png        → {out_dir}")
    print(f"  polygons_overlay.png → {out_dir}")
    print(f"  junctions.png        → {out_dir}")
    print(f"  Detected polygons    : {len(polys)}")
    print(f"  Detected junctions   : {len(juncs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-image PLR-Net inference — environment sanity check"
    )
    parser.add_argument("--image",      required=True,
                        help="Input PNG, JPG, or GeoTIFF")
    parser.add_argument("--config",     required=True,
                        help="YAML config file (e.g. config-files/PLR-Net.yaml)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (optional — omit for smoke test)")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    # Config
    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    print(f"\n[infer_one]  device     : {device}")
    print(f"[infer_one]  image      : {args.image}")
    print(f"[infer_one]  output dir : {args.output}")

    # Model
    model = BuildingDetector(cfg, test=True).to(device)

    if args.checkpoint is not None:
        print(f"\n  Loading checkpoint : {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model" in state:
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print("  Checkpoint loaded.")
    else:
        print("\n  No checkpoint — random weights (smoke test only).")

    model.eval()

    # Image
    print(f"\n  Loading image ...")
    image_orig = load_image(args.image)
    print(f"  Shape  : {image_orig.shape}  dtype={image_orig.dtype}")

    tensor = preprocess(image_orig, cfg).to(device)
    print(f"  Tensor : {tensor.shape}  device={tensor.device}")

    # Forward pass
    print(f"\n  Running forward pass ...")
    meta = [{
        "filename": os.path.basename(args.image),
        "height":   cfg.DATASETS.IMAGE.HEIGHT,
        "width":    cfg.DATASETS.IMAGE.WIDTH,
    }]
    with torch.no_grad():
        output, _ = model(tensor, meta)

    print("  Forward pass OK")
    print(f"  Output keys : {list(output.keys())}")

    # Save
    print(f"\n  Saving results ...")
    save_results(image_orig, output, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
