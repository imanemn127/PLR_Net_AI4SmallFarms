#!/usr/bin/env python3
"""
Diagnostic script for joff_gt (junction offset targets).

Loads one batch from the train dataset and prints statistics on joff_gt
restricted to junction pixels (where jloc_gt > 0).

Usage:
    python scripts/diag_joff.py --config-file config-files/PLR-Net.yaml
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from PLRNet.config import cfg
from PLRNet.dataset import build_train_dataset
from PLRNet.detector import BuildingDetector
from PLRNet.utils.comm import to_single_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--n-batches', default=5, type=int,
                        help='Number of batches to inspect (default: 5)')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = cfg.MODEL.DEVICE
    model  = BuildingDetector(cfg).to(device)
    model.eval()

    train_loader = build_train_dataset(cfg)

    all_offsets   = []   # joff values at junction pixels, across batches
    total_juncs   = 0
    total_pixels  = 0

    print(f"\nInspecting {args.n_batches} batches from train split...\n")

    for i, (images, annotations) in enumerate(train_loader):
        if i >= args.n_batches:
            break

        images      = images.to(device)
        annotations = to_single_device(annotations, device)

        with torch.no_grad():
            targets, _ = model.encoder(annotations)
            targets     = {k: v.to(device) for k, v in targets.items()}

        jloc_gt = targets['jloc'].squeeze(1)   # (B, H, W)  long
        joff_gt = targets['joff']               # (B, 2, H, W) float

        junc_mask = (jloc_gt > 0)              # True where a junction pixel exists
        n_juncs   = junc_mask.sum().item()
        n_pixels  = junc_mask.numel()

        # joff at junction pixels: shape (N, 2)
        off_vals = joff_gt.permute(0, 2, 3, 1)[junc_mask]  # (N, 2)

        total_juncs  += n_juncs
        total_pixels += n_pixels
        all_offsets.append(off_vals.cpu())

        print(f"Batch {i+1}:")
        print(f"  junction pixels : {n_juncs} / {n_pixels} "
              f"({100*n_juncs/n_pixels:.2f}%)")
        if n_juncs > 0:
            print(f"  joff_x  min={off_vals[:,0].min():.4f}  "
                  f"max={off_vals[:,0].max():.4f}  "
                  f"mean={off_vals[:,0].mean():.4f}  "
                  f"std={off_vals[:,0].std():.4f}")
            print(f"  joff_y  min={off_vals[:,1].min():.4f}  "
                  f"max={off_vals[:,1].max():.4f}  "
                  f"mean={off_vals[:,1].mean():.4f}  "
                  f"std={off_vals[:,1].std():.4f}")
            nonzero_frac = (off_vals.abs() > 1e-6).float().mean().item()
            print(f"  fraction |offset| > 1e-6 : {nonzero_frac:.3f}")
        else:
            print("  NO junction pixels in this batch!")
        print()

    # Global summary
    if all_offsets:
        all_off = torch.cat(all_offsets, dim=0)
        print("=" * 50)
        print(f"GLOBAL SUMMARY over {args.n_batches} batches")
        print(f"  Total junction pixels : {total_juncs} / {total_pixels} "
              f"({100*total_juncs/total_pixels:.2f}%)")
        print(f"  joff_x  min={all_off[:,0].min():.4f}  "
              f"max={all_off[:,0].max():.4f}  "
              f"mean={all_off[:,0].mean():.4f}  "
              f"std={all_off[:,0].std():.4f}")
        print(f"  joff_y  min={all_off[:,1].min():.4f}  "
              f"max={all_off[:,1].max():.4f}  "
              f"mean={all_off[:,1].mean():.4f}  "
              f"std={all_off[:,1].std():.4f}")
        nonzero_frac = (all_off.abs() > 1e-6).float().mean().item()
        print(f"  fraction |offset| > 1e-6 : {nonzero_frac:.3f}")
        print()

        # Interpretation
        print("INTERPRETATION:")
        if nonzero_frac < 0.05:
            print("  -> Offsets are essentially zero.")
            print("     Cause: junction coordinates were snapped to pixel grid during")
            print("     patch creation (integer coords -> off_x = x - floor(x) - 0.5 ≈ -0.5 or 0).")
            print("     Recommendation: set LOSS_WEIGHTS.loss_joff: 0.0 in the YAML.")
        elif all_off.std() < 0.05:
            print("  -> Offsets are non-zero but nearly constant (low variance).")
            print("     The branch has very little to learn. Consider disabling loss_joff.")
        else:
            print("  -> Offsets have meaningful variance — the branch CAN learn.")
            print("     If loss_joff is still flat, try increasing its weight (e.g. 1.0).")

if __name__ == '__main__':
    main()
