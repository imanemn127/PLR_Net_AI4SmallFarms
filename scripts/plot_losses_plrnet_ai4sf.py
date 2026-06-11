#!/usr/bin/env python3
"""
Plot training curves from a PLR-Net metrics.csv file.

Usage:
    python plot_losses_plrnet_ai4sf.py                     # auto-detect latest run
    python plot_losses_plrnet_ai4sf.py /path/to/run/folder # specific run
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


# Individual loss names as written in metrics.csv
LOSS_NAMES = ["loss_jloc", "loss_joff", "loss_mask", "loss_afm", "loss_remask"]

COLORS = {
    "loss_jloc"   : "#e41a1c",
    "loss_joff"   : "#377eb8",
    "loss_mask"   : "#4daf4a",
    "loss_afm"    : "#ff7f00",
    "loss_remask" : "#984ea3",
}


def find_latest_run():
    """Return the most recent timestamped run folder that contains a metrics.csv."""
    base    = "/mnt/DATA/IMANE/PLR-Net_output/PLR-Net"
    pattern = os.path.join(base, "*/metrics.csv")
    runs    = sorted(glob.glob(pattern))
    if not runs:
        return None
    return os.path.dirname(runs[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss curves from a PLR-Net metrics.csv file."
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Run folder containing metrics.csv. If omitted, the latest run is used.",
    )
    args = parser.parse_args()

    # -------- determine run directory --------
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            sys.exit(
                "No runs found under /mnt/DATA/IMANE/PLR-Net_output/PLR-Net\n"
                "Run training first or pass the run folder as argument."
            )
        print(f"Auto-detected latest run: {run_dir}")

    # -------- read CSV --------
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, na_values=["", " "])
    for col in ["epoch", "train_loss", "val_loss", "val_mask_iou", "train_mask_iou"] + \
               ["w_" + k for k in LOSS_NAMES] + \
               ["val_w_" + k for k in LOSS_NAMES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    val_df = df.dropna(subset=["val_loss"])

    # -------- layout: 4 subplots --------
    #   [0] Total loss (train vs val)
    #   [1] Individual train losses
    #   [2] Individual val losses (at validation epochs only)
    #   [3] Val mask IoU
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # --- subplot 0 : total loss train vs val ---
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"],
            color="steelblue", linewidth=1.4, label="Train loss")
    if len(val_df):
        ax.plot(val_df["epoch"], val_df["val_loss"],
                color="tomato", linewidth=1.4,
                marker="o", markersize=4, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted total loss")
    ax.set_title("Total Loss (train vs val)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- subplot 1 : weighted train losses per component ---
    ax = axes[1]
    for loss in LOSS_NAMES:
        col = "w_" + loss
        if col in df.columns:
            ax.plot(df["epoch"], df[col],
                    color=COLORS[loss], linewidth=1.2, label=loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss")
    ax.set_title("Train Losses (weighted, per component)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- subplot 2 : weighted val losses per component ---
    ax = axes[2]
    any_val = False
    for loss in LOSS_NAMES:
        col = "val_w_" + loss
        if col in df.columns:
            sub = df.dropna(subset=[col])
            if len(sub):
                ax.plot(sub["epoch"], sub[col],
                        color=COLORS[loss], linewidth=1.2,
                        marker="o", markersize=4, label=loss)
                any_val = True
    if not any_val:
        ax.text(0.5, 0.5, "No validation data yet",
                ha='center', va='center', transform=ax.transAxes, color='grey')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss")
    ax.set_title("Val Losses (weighted, per component)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- subplot 3 : mask IoU (train vs val) ---
    ax = axes[3]
    has_data = False
    if "train_mask_iou" in df.columns:
        train_iou_df = df.dropna(subset=["train_mask_iou"])
        if len(train_iou_df):
            ax.plot(train_iou_df["epoch"], train_iou_df["train_mask_iou"],
                    color="steelblue", linewidth=1.2, alpha=0.7, label="Train mask IoU")
            has_data = True
    if "val_mask_iou" in df.columns:
        iou_df = df.dropna(subset=["val_mask_iou"])
        if len(iou_df):
            ax.plot(iou_df["epoch"], iou_df["val_mask_iou"],
                    color="#2ca02c", linewidth=1.4,
                    marker="o", markersize=4, label="Val mask IoU")
            has_data = True
    if has_data:
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No IoU data yet",
                ha='center', va='center', transform=ax.transAxes, color='grey')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Mask IoU (train vs val)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "loss_curves_plrnet.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    print(
        f"Epochs plotted : {int(df['epoch'].min())}–{int(df['epoch'].max())}  "
        f"|  Val checkpoints : {len(val_df)}"
    )


if __name__ == "__main__":
    main()
