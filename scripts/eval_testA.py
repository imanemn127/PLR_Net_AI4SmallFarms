#!/usr/bin/env python3
"""
eval_testA.py -- Test A: post-processing evaluation (no retraining needed)
Run inference on all 24 val images using the best Run 7 checkpoint
and the modified polygon.py thresholds (NMS 0.004, matching 3px).

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/eval_testA.py \
      --config  config-files/PLR-Net.yaml \
      --checkpoint /home/imane/DATA/PLR-Net_output/PLR-Net/2026-06-08_14-05-29/checkpoints/best_val_loss.pth \
      --output  /home/imane/DATA/PLR-Net_output/TestA
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pycocotools.coco import COCO
from skimage.measure import label, regionprops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.spatial.distance import cdist

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from PLRNet.utils.metrics.cIoU import calc_IoU


# Thresholds for junction recall (fraction of GT corners matched within N px)
RECALL_THRESHOLDS = [3, 5, 8]   # pixels


def compute_junction_recall(gt_polys, pred_juncs, thresholds=RECALL_THRESHOLDS):
    """
    For each threshold T in `thresholds`, compute the fraction of GT polygon
    corners that have at least one predicted junction within T pixels.

    gt_polys   : list of (N,2) arrays (GT polygon vertices from COCO)
    pred_juncs : (M,2) numpy array of predicted junction coordinates
    Returns    : dict {T: recall_float} — NaN if there are no GT corners.

    Why this matters: if junction_recall@3px is low (< 0.3) but mask IoU is
    decent (> 0.4), it confirms the network can find the blobs but cannot
    place corners precisely — the bottleneck is jloc, not the mask branch.
    """
    # Collect all GT corner coordinates (skip duplicate closing vertex)
    all_gt = []
    for poly in gt_polys:
        pts = np.array(poly).reshape(-1, 2)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=0.5):
            pts = pts[:-1]
        all_gt.append(pts)

    if not all_gt:
        return {t: float('nan') for t in thresholds}

    gt_corners = np.vstack(all_gt)  # (n_gt_corners, 2)

    if len(pred_juncs) == 0:
        return {t: 0.0 for t in thresholds}

    # cdist computes all pairwise distances: shape (n_gt, n_pred)
    min_dists = cdist(gt_corners, pred_juncs).min(axis=1)  # closest pred per GT corner

    return {t: float((min_dists <= t).mean()) for t in thresholds}


def load_and_preprocess(img_path, cfg):
    """Load a Sentinel-2 GeoTIFF and return (tensor (1,3,H,W), display uint8 array)."""
    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 10000.0

    # Percentile stretch for display only
    disp = arr.transpose(1, 2, 0).copy()
    for c in range(3):
        p2, p98 = np.percentile(disp[:, :, c], (2, 98))
        if p98 > p2:
            disp[:, :, c] = (disp[:, :, c] - p2) / (p98 - p2)
    disp = (np.clip(disp, 0, 1) * 255).astype(np.uint8)

    mean = np.array(cfg.DATASETS.IMAGE.PIXEL_MEAN, dtype=np.float32)
    std  = np.array(cfg.DATASETS.IMAGE.PIXEL_STD,  dtype=np.float32)
    img_norm = (arr.transpose(1, 2, 0) - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, disp


def build_gt_mask(coco_obj, img_id, h, w):
    """Build a binary (H,W) mask from COCO annotations."""
    ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
    anns    = coco_obj.loadAnns(ids=ann_ids)
    mask    = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        mask = np.clip(mask + coco_obj.annToMask(ann), 0, 1).astype(np.uint8)
    return mask


def render_comparison(disp_img, gt_polys, pred_polys, juncs, iou, img_name, out_dir):
    """Save a side-by-side GT | Pred figure with polygons and junctions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#111111')

    for ax in axes:
        ax.imshow(disp_img)
        ax.axis('off')

    for poly in gt_polys:
        pts = np.array(poly).reshape(-1, 2)
        closed = np.vstack([pts, pts[0]])
        axes[0].plot(closed[:, 0], closed[:, 1], '-', color='#00ff00', linewidth=1.2)
    axes[0].set_title(f"GT  ({len(gt_polys)} polygons)", color='white', fontsize=8)

    for poly in pred_polys:
        pts = np.array(poly).reshape(-1, 2)
        if len(pts) < 2:
            continue
        closed = np.vstack([pts, pts[0]])
        axes[1].plot(closed[:, 0], closed[:, 1], '-', color='#ff6600', linewidth=1.2)

    if len(juncs) > 0:
        axes[1].scatter(juncs[:, 0], juncs[:, 1],
                        s=6, c='red', marker='+', linewidths=0.8, zorder=5)

    axes[1].set_title(
        f"Pred  {len(pred_polys)} poly | {len(juncs)} junctions\nmask IoU={iou:.3f}",
        color='white', fontsize=7
    )

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(out_dir, f"{img_name}.png")
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="/home/imane/DATA/PLR-Net_output/TestA")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    viz_dir = os.path.join(args.output, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    print(f"\n[TestA]  device      : {device}")
    print(f"[TestA]  checkpoint  : {args.checkpoint}")

    model = BuildingDetector(cfg, test=True).to(device)
    ckpt_raw  = torch.load(args.checkpoint, map_location=device)
    ckpt_epoch = torch.load(args.checkpoint, map_location='cpu').get('epoch', '?')
    state = ckpt_raw.get("model", ckpt_raw)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[TestA]  Checkpoint loaded (epoch {ckpt_epoch})")

    data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "ai4sf_256px_area50"
    )
    ann_file = os.path.join(data_root, "val_coco.json")
    coco_obj = COCO(ann_file)
    img_ids  = coco_obj.getImgIds()
    print(f"[TestA]  {len(img_ids)} validation images\n")

    iou_list         = []
    n_gt_poly_list   = []
    n_pred_poly_list = []
    n_juncs_list     = []
    snap_rate_list   = []
    recall_lists     = {t: [] for t in RECALL_THRESHOLDS}  # per-threshold recall

    for img_id in img_ids:
        img_info  = coco_obj.loadImgs(ids=[img_id])[0]
        img_path  = os.path.join(data_root, img_info['file_name'])
        h, w      = img_info['height'], img_info['width']
        img_name  = os.path.splitext(os.path.basename(img_info['file_name']))[0]

        if not os.path.exists(img_path):
            print(f"  [SKIP] not found: {img_path}")
            continue

        tensor, disp = load_and_preprocess(img_path, cfg)
        tensor = tensor.to(device)

        with torch.no_grad():
            output, _ = model(tensor)

        mask_pred  = output['mask_pred'][0]
        juncs      = output['juncs_pred'][0]
        pred_polys = output['polys_pred'][0] if output['polys_pred'] else []

        gt_mask  = build_gt_mask(coco_obj, img_id, h, w)
        pred_bin = (mask_pred > 0.5).astype(np.uint8)
        iou      = calc_IoU(pred_bin, gt_mask)

        gt_polys = [np.array(seg).reshape(-1, 2)
                    for ann in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id]))
                    for seg in ann['segmentation']]

        n_blobs   = len(regionprops(label(pred_bin)))
        n_snapped = len(pred_polys)

        # Junction recall at each threshold
        recalls = compute_junction_recall(gt_polys, juncs)

        iou_list.append(iou)
        n_gt_poly_list.append(len(gt_polys))
        n_pred_poly_list.append(len(pred_polys))
        n_juncs_list.append(len(juncs))
        snap_rate_list.append(n_snapped / max(n_blobs, 1))
        for t in RECALL_THRESHOLDS:
            if not np.isnan(recalls[t]):
                recall_lists[t].append(recalls[t])

        render_comparison(disp, gt_polys, pred_polys, juncs, iou, img_name, viz_dir)

        recall_str = "  ".join(f"R@{t}px={recalls[t]:.2f}" for t in RECALL_THRESHOLDS)
        print(f"  {img_name:38s}  IoU={iou:.3f}  GT={len(gt_polys):3d}  "
              f"Pred={len(pred_polys):3d}  Juncs={len(juncs):3d}  "
              f"snap={n_snapped}/{n_blobs}  {recall_str}")

    print("\n" + "="*60)
    print("TEST A SUMMARY")
    print("="*60)
    print(f"  Images processed      : {len(iou_list)}")
    print(f"  Mean mask IoU         : {np.mean(iou_list):.4f}  (std={np.std(iou_list):.4f})")
    print(f"  Mean GT polys/image   : {np.mean(n_gt_poly_list):.1f}")
    print(f"  Mean pred polys/image : {np.mean(n_pred_poly_list):.1f}")
    print(f"  Mean junctions/image  : {np.mean(n_juncs_list):.1f}  (std={np.std(n_juncs_list):.1f})")
    print(f"  Mean snap rate        : {np.mean(snap_rate_list):.3f}  (1.0 = all blobs snapped)")
    for t in RECALL_THRESHOLDS:
        vals = recall_lists[t]
        mean_r = np.mean(vals) if vals else float('nan')
        print(f"  Junction recall @{t:2d}px : {mean_r:.4f}  ({len(vals)} images with GT corners)")
    print(f"\n  NOTE: recall@3px < 0.30 with IoU > 0.40 → bottleneck is jloc, not mask branch")
    print(f"\n  Visualizations saved to: {viz_dir}")
    print("="*60)

    summary = {
        "test": "A",
        "nms_threshold": 0.004,
        "matching_threshold_px": 3,
        "checkpoint_epoch": ckpt_epoch,
        "n_images": len(iou_list),
        "mean_iou": float(np.mean(iou_list)),
        "std_iou": float(np.std(iou_list)),
        "mean_gt_polys": float(np.mean(n_gt_poly_list)),
        "mean_pred_polys": float(np.mean(n_pred_poly_list)),
        "mean_juncs": float(np.mean(n_juncs_list)),
        "mean_snap_rate": float(np.mean(snap_rate_list)),
        "junction_recall": {
            f"@{t}px": float(np.mean(recall_lists[t])) if recall_lists[t] else float('nan')
            for t in RECALL_THRESHOLDS
        },
    }
    summary_path = os.path.join(args.output, "summary_testA.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary          : {summary_path}")


if __name__ == "__main__":
    main()
