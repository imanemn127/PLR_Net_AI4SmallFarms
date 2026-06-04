#!/usr/bin/env python3
"""
dataset_stats.py  —  PLR-Net / AI4SmallFarms
=============================================
Prints statistics on the generated COCO JSON files to help choose
MIN_AREA_PX and understand the annotation density before training.
The output is also saved automatically to a text file.

Usage:
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/dataset_stats.py
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/dataset_stats.py --split val
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/dataset_stats.py --split all --output my_stats.txt
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

DATA_ROOT = "/mnt/DATA/IMANE/PLR-Net/data/ai4sf_256px_area100"

JSON_FILES = {
    "train": os.path.join(DATA_ROOT, "train_coco.json"),
    "val":   os.path.join(DATA_ROOT, "val_coco.json"),
    "test":  os.path.join(DATA_ROOT, "test_coco.json"),
}


def summary_row(label: str, arr: np.ndarray) -> str:
    """Return a compact summary string without percentiles."""
    return (f"  {label:<22s}  "
            f"min={arr.min():.1f}  "
            f"median={np.median(arr):.1f}  "
            f"max={arr.max():.1f}  "
            f"mean={arr.mean():.1f}")


def analyze(json_path: str, split: str, out_streams):
    """Write statistics to all given streams (console and file)."""
    # Helper to write to all streams at once
    def write(*args, **kwargs):
        for stream in out_streams:
            print(*args, file=stream, **kwargs)
            stream.flush()

    with open(json_path) as f:
        d = json.load(f)

    n_images = len(d["images"])
    n_anns   = len(d["annotations"])

    areas   = np.array([a["area"]                       for a in d["annotations"]])
    widths  = np.array([a["bbox"][2]                    for a in d["annotations"]])
    heights = np.array([a["bbox"][3]                    for a in d["annotations"]])
    n_verts = np.array([len(a["segmentation"][0]) // 2  for a in d["annotations"]])

    # bbox diagonal as proxy for "effective side length"
    diag = np.sqrt(widths**2 + heights**2)

    ann_per_img = defaultdict(int)
    for a in d["annotations"]:
        ann_per_img[a["image_id"]] += 1
    counts = np.array(list(ann_per_img.values()))

    write(f"\n{'='*70}")
    write(f"  Split : {split.upper()}  —  {json_path}")
    write(f"{'='*70}")
    write(f"  Images      : {n_images}")
    write(f"  Annotations : {n_anns}")
    write(f"  Avg ann/img : {n_anns/n_images:.1f}")

    write(f"\n--- Annotations per image ---")
    write(summary_row("ann/image", counts))

    write(f"\n--- Polygon area (px²) ---")
    write(summary_row("area (px²)", areas))

    write(f"\n--- Bounding-box dimensions (px) ---")
    write(summary_row("bbox width  (px)", widths))
    write(summary_row("bbox height (px)", heights))
    write(summary_row("bbox diag   (px)", diag))

    write(f"\n--- Vertices per polygon ---")
    write(summary_row("vertices/polygon", n_verts))

    write(f"\n--- Junction density (for Encoder jmap 256×256) ---")
    # avg junctions per patch = avg ann/patch × avg verts/ann
    avg_juncs = (n_anns / n_images) * n_verts.mean()
    density_pct = avg_juncs / (256 * 256) * 100
    write(f"  Avg junctions/patch : {avg_juncs:.0f}")
    write(f"  % pixels occupied   : {density_pct:.2f}%  "
          f"({'HIGH — risk of junction collision' if density_pct > 5 else 'OK'})")

    write(f"\n--- Effect of MIN_AREA_PX filter on annotation count ---")
    write(f"  {'Threshold':>12s}  {'Kept':>8s}  {'%':>6s}  {'Avg ann/patch':>14s}  "
          f"{'Median bbox diag (px)':>22s}  {'Avg juncs/patch':>16s}  {'Junction density %':>17s}")
    for thr in [16, 30, 50, 100, 150, 200, 400]:
        mask      = areas >= thr
        kept      = mask.sum()
        pct       = kept / n_anns * 100
        kept_per_patch = kept / n_images
        med_diag  = np.median(diag[mask]) if mask.sum() > 0 else 0
        avg_v     = n_verts[mask].mean()  if mask.sum() > 0 else 0
        avg_j     = kept_per_patch * avg_v
        density   = avg_j / (256 * 256) * 100
        write(f"  {thr:>12d}  {kept:>8d}  {pct:>5.1f}%  {kept_per_patch:>14.1f}  "
              f"{med_diag:>22.1f}  {avg_j:>16.1f}  {density:>16.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="train")
    parser.add_argument("--output", "-o", type=str, default="dataset_stats.txt",
                        help="Output text file (default: dataset_stats.txt)")
    args = parser.parse_args()

    splits = JSON_FILES if args.split == "all" else {args.split: JSON_FILES[args.split]}

    # Open the output file once for all splits
    with open(args.output, "w", encoding="utf-8") as f_out:
        for split, path in splits.items():
            if not os.path.isfile(path):
                print(f"  SKIP {split} — file not found: {path}")
                continue
            # Pass both stdout and the file as output streams
            analyze(path, split, out_streams=[sys.stdout, f_out])

    print(f"\nStatistics saved to {args.output}")


if __name__ == "__main__":
    main()