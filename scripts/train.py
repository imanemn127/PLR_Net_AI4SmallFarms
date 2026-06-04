import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import argparse
import logging
import random
import numpy as np
import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from PLRNet.dataset import build_train_dataset, build_test_dataset
from PLRNet.utils.comm import to_single_device
from PLRNet.solver import make_lr_scheduler, make_optimizer
from PLRNet.utils.logger import setup_logger
from PLRNet.utils.miscellaneous import save_config
from PLRNet.utils.metric_logger import MetricLogger

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
VAL_EVERY = 5   # run validation every N epochs
N_VIZ     = 2   # images to visualize per val/train run


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        return sum(self.loss_weights[k] * loss_dict[k]
                   for k in self.loss_weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Training PLR-Net')
    parser.add_argument("--config-file", metavar="FILE", type=str, default=None)
    parser.add_argument("--clean", default=False, action='store_true')
    parser.add_argument("--val-every", default=VAL_EVERY, type=int,
                        help="Run validation every N epochs (default: 5)")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_metrics_csv(csv_path, loss_names):
    fieldnames = (
        ['epoch', 'train_loss']
        + ['w_' + k for k in loss_names]
        + ['val_loss']
        + ['val_w_' + k for k in loss_names]
    )
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return fieldnames


def append_metrics_csv(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


# ------------------------------------------------------------------ #
#  Image utilities
# ------------------------------------------------------------------ #
def tensor_to_display(image_tensor, mean, std):
    """Normalised CxHxW tensor → display-ready uint8 HxWx3 array."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    for c in range(img.shape[2]):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            img[:, :, c] = (img[:, :, c] - p2) / (p98 - p2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def draw_polygons(ax, polys, color, label):
    """Draw a list of Nx2 polygon arrays on a matplotlib axis."""
    for poly in polys:
        if len(poly) < 3:
            continue
        pts = np.array(poly)
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color=color, linewidth=1.2)
    return mpatches.Patch(color=color, label=label)


# ------------------------------------------------------------------ #
#  Validation visualization  (GT from COCO object)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_val(model, val_loader, epoch, output_dir, mean, std,
                  n_images=N_VIZ):
    """Side-by-side GT | Pred for N_VIZ val images."""
    model.eval()
    device  = next(model.parameters()).device
    viz_dir = os.path.join(output_dir, 'visualizations', 'val')
    os.makedirs(viz_dir, exist_ok=True)

    coco_obj = val_loader.dataset.coco
    saved = 0

    for images, annotations in val_loader:
        if saved >= n_images:
            break
        images_dev = images.to(device)
        output, _  = model(images_dev)

        for b in range(images.shape[0]):
            if saved >= n_images:
                break

            ann      = annotations[b]
            img_name = os.path.splitext(
                os.path.basename(ann.get('filename', str(saved))))[0]
            img_id   = ann.get('img_id', None)
            img_disp = tensor_to_display(images[b], mean, std)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
            fig.patch.set_facecolor('black')
            for ax in axes:
                ax.imshow(img_disp)
                ax.axis('off')

            patches = []

            # left — GT polygons from COCO
            if img_id is not None:
                ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
                anns    = coco_obj.loadAnns(ids=ann_ids)
                gt_polys = []
                for a in anns:
                    for seg in a['segmentation']:
                        gt_polys.append(np.array(seg).reshape(-1, 2))
                p = draw_polygons(axes[0], gt_polys, '#00ff00', 'GT')
                patches.append(p)
            axes[0].set_title(f"GT  epoch {epoch:03d}", color='white', fontsize=8)

            # right — predicted polygons
            pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
            p2 = draw_polygons(axes[1], pred_polys, '#ff6600', 'Pred')
            patches.append(p2)
            axes[1].set_title(f"Pred  epoch {epoch:03d}", color='white', fontsize=8)

            if patches:
                axes[1].legend(handles=patches, loc='upper right',
                               fontsize=6, framealpha=0.5)

            plt.tight_layout(pad=0.3)
            out_path = os.path.join(viz_dir, f"{epoch:03d}_{img_name}.png")
            plt.savefig(out_path, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            saved += 1


# ------------------------------------------------------------------ #
#  Train visualization  (GT from mask via cv2.findContours)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_train(model, fixed_train_batches, epoch, output_dir, mean, std):
    """
    Side-by-side GT | Pred for a fixed set of training samples.
    GT is derived from ann['mask'] using cv2.findContours (no COCO object needed).
    The same samples are used every epoch so evolution is trackable.
    Output: {output_dir}/visualizations/train/{epoch:03d}_{img_name}.png
    """
    model.eval()
    device  = next(model.parameters()).device
    viz_dir = os.path.join(output_dir, 'visualizations', 'train')
    os.makedirs(viz_dir, exist_ok=True)

    for images, annotations in fixed_train_batches:
        images_dev = images.to(device)
        output, _  = model(images_dev)

        for b in range(images.shape[0]):
            ann      = annotations[b]
            img_name = os.path.splitext(
                os.path.basename(ann.get('filename', f"sample_{b}")))[0]
            img_disp = tensor_to_display(images[b], mean, std)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
            fig.patch.set_facecolor('black')
            for ax in axes:
                ax.imshow(img_disp)
                ax.axis('off')

            patches = []

            # left — GT from binary mask via contours
            mask = ann.get('mask', None)
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                mask_u8 = (mask_np * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                gt_polys = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
                p = draw_polygons(axes[0], gt_polys, '#00ff00', 'GT')
                patches.append(p)
            axes[0].set_title(f"GT  epoch {epoch:03d}", color='white', fontsize=8)

            # right — predicted polygons
            pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
            p2 = draw_polygons(axes[1], pred_polys, '#ff6600', 'Pred')
            patches.append(p2)
            axes[1].set_title(f"Pred  epoch {epoch:03d}", color='white', fontsize=8)

            if patches:
                axes[1].legend(handles=patches, loc='upper right',
                               fontsize=6, framealpha=0.5)

            plt.tight_layout(pad=0.3)
            out_path = os.path.join(viz_dir, f"{epoch:03d}_{img_name}.png")
            plt.savefig(out_path, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)

    model.train()


# ------------------------------------------------------------------ #
#  Validation loss pass
# ------------------------------------------------------------------ #
#  Validation loss pass
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(model, val_loader, loss_reducer, device, loss_names):
    model.eval()
    sums  = {k: 0.0 for k in loss_names}
    total = 0.0
    n     = 0
    for images, annotations in val_loader:
        images      = images.to(device)
        annotations = to_single_device(annotations, device)
        loss_dict, _ = model.forward_train(images, annotations)
        weighted = loss_reducer(loss_dict)
        for k in loss_names:
            sums[k] += loss_dict[k].item()
        total += weighted.item()
        n     += 1
    avg       = {k: sums[k] / max(n, 1) for k in loss_names}
    avg_total = total / max(n, 1)
    return avg_total, avg


# ------------------------------------------------------------------ #
#  Train loop
# ------------------------------------------------------------------ #
def train(cfg, output_dir, val_every):
    logger     = logging.getLogger("training")
    device     = cfg.MODEL.DEVICE
    mean       = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std        = list(cfg.DATASETS.IMAGE.PIXEL_STD)

    model = BuildingDetector(cfg).to(device)

    train_dataset            = build_train_dataset(cfg)
    val_dataset, val_ann_file = build_test_dataset(cfg)

    optimizer    = make_optimizer(cfg, model)
    scheduler    = make_lr_scheduler(cfg, optimizer)
    loss_reducer = LossReducer(cfg)
    loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    loss_names   = list(loss_weights.keys())

    max_epoch  = cfg.SOLVER.MAX_EPOCH
    epoch_size = len(train_dataset)

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_val_loss = float('inf')

    def save_checkpoint(name, current_epoch):
        state = {
            'epoch'     : current_epoch,
            'model'     : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }
        path = os.path.join(checkpoints_dir, name)
        torch.save(state, path)
        logger.info(f"Checkpoint saved → checkpoints/{name}")

    csv_path   = os.path.join(output_dir, 'metrics.csv')
    fieldnames = init_metrics_csv(csv_path, loss_names)  # loss_names still needed for w_ columns

    # Pre-fetch N_VIZ fixed training batches (same every epoch)
    fixed_train_batches = []
    _train_iter = iter(train_dataset)
    for _ in range(N_VIZ):
        try:
            fixed_train_batches.append(next(_train_iter))
        except StopIteration:
            break

    start_time = time.time()
    end        = time.time()

    for epoch in range(1, max_epoch + 1):
        meters = MetricLogger(" ")
        model.train()

        epoch_loss_sums = {k: 0.0 for k in loss_names}
        epoch_total     = 0.0
        n_batches       = 0

        for it, (images, annotations) in enumerate(train_dataset):
            data_time   = time.time() - end
            images      = images.to(device)
            annotations = to_single_device(annotations, device)

            loss_dict, _ = model(images, annotations)
            total_loss   = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_red = {k: v.item() for k, v in loss_dict.items()}
                loss_red      = total_loss.item()
                meters.update(loss=loss_red, **loss_dict_red)
                for k in loss_names:
                    epoch_loss_sums[k] += loss_dict_red.get(k, 0.0)
                epoch_total += loss_red
                n_batches   += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if it % 20 == 0 or it + 1 == epoch_size:
                eta_seconds = meters.time.global_avg * (
                    epoch_size * (max_epoch - epoch + 1) - it + 1)
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}", "epoch: {epoch}", "iter: {iter}",
                        "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}\n",
                    ]).format(
                        eta=str(datetime.timedelta(seconds=int(eta_seconds))),
                        epoch=epoch, iter=it, meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024**2,
                    )
                )

        scheduler.step()

        avg_losses = {k: epoch_loss_sums[k] / max(n_batches, 1) for k in loss_names}
        avg_total  = epoch_total / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # always overwrite latest
        save_checkpoint('latest.pth', epoch)

        # --- validation + visualizations every val_every epochs ---
        val_total  = float('nan')
        val_losses = {k: float('nan') for k in loss_names}

        if epoch % val_every == 0:
            logger.info(f"=== Validation at epoch {epoch} ===")
            val_total, val_losses = validate(
                model, val_dataset, loss_reducer, device, loss_names)
            logger.info(
                "Val total_loss: {:.4f}  |  {}".format(
                    val_total,
                    "  ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
            )
            # checkpoint at this val epoch
            save_checkpoint(f'epoch_{epoch}.pth', epoch)

            # best val loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint('best_val_loss.pth', epoch)
                logger.info(f"New best val loss: {best_val_loss:.4f}")

            # val visualization
            visualize_val(model, val_dataset, epoch, output_dir,
                          mean, std, n_images=N_VIZ)
            # train visualization (fixed samples, same every epoch)
            visualize_train(model, fixed_train_batches, epoch,
                            output_dir, mean, std)

        # --- write metrics.csv ---
        row = {'epoch': epoch, 'train_loss': round(avg_total, 6)}
        for k in loss_names:
            row['w_' + k] = round(avg_losses[k] * loss_weights[k], 6)
        row['val_loss'] = round(val_total, 6) if not np.isnan(val_total) else ''
        for k in loss_names:
            row['val_w_' + k] = (round(val_losses[k] * loss_weights[k], 6)
                                 if not np.isnan(val_losses[k]) else '')
        append_metrics_csv(csv_path, fieldnames, row)

        logger.info(
            "Epoch {:03d} | train_loss: {:.4f} | val_loss: {} | lr: {:.6f}".format(
                epoch, avg_total,
                f"{val_total:.4f}" if not np.isnan(val_total) else "-",
                current_lr,
            )
        )

    total_time = time.time() - start_time
    logger.info("Total training time: {} ({:.4f} s / epoch)".format(
        str(datetime.timedelta(seconds=int(total_time))),
        total_time / max_epoch,
    ))


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(cfg.OUTPUT_DIR, timestamp)

    if os.path.isdir(cfg.OUTPUT_DIR) and args.clean:
        import shutil
        shutil.rmtree(cfg.OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('training', output_dir, out_file='train.log')
    logger.info(args)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Validation every {args.val_every} epochs, "
                f"visualizing {N_VIZ} train + {N_VIZ} val images")

    with open(args.config_file, "r") as cf:
        logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    save_config(cfg, os.path.join(output_dir, 'config.yml'))
    set_random_seed(args.seed, True)
    train(cfg, output_dir, val_every=args.val_every)
