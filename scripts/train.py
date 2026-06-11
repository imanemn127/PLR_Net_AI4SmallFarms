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
from PLRNet.utils.metrics.cIoU import calc_IoU

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
        ['epoch', 'train_loss', 'train_mask_iou']
        + ['w_' + k for k in loss_names]
        + ['val_loss']
        + ['val_w_' + k for k in loss_names]
        + ['val_mask_iou']
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
#  visualization 
# ------------------------------------------------------------------ #
def _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir):
    """Save one GT | Pred side-by-side figure."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.imshow(img_disp)
        ax.axis('off')
    patches = [
        draw_polygons(axes[0], gt_polys,   '#00ff00', 'GT'),
        draw_polygons(axes[1], pred_polys, '#ff6600', 'Pred'),
    ]
    axes[0].set_title(f"GT   epoch {epoch:03d}", color='white', fontsize=8)
    axes[1].set_title(f"Pred epoch {epoch:03d}", color='white', fontsize=8)
    axes[1].legend(handles=patches, loc='upper right', fontsize=6, framealpha=0.5)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(viz_dir, f"{epoch:03d}_{img_name}.png"),
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def select_viz_indices_val(val_loader):
    """
    Scan val_loader once (no shuffle) and return the indices of:
      - the image with the most GT polygons (dense)
      - the image with the fewest GT polygons >= 1 (sparse)
    Returns list of (flat_index, images_batch, b_in_batch, annotation).
    Called once before training starts.
    """
    coco_obj = val_loader.dataset.coco
    best = {'dense': (None, -1), 'sparse': (None, float('inf'))}
    flat_idx = 0
    entries = {}

    for images, annotations in val_loader:
        for b in range(images.shape[0]):
            ann    = annotations[b]
            img_id = ann.get('img_id', None)
            count  = (sum(len(a['segmentation'])
                          for a in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id])))
                      if img_id is not None else 0)
            entries[flat_idx] = (images, b, ann, count)
            if count > best['dense'][1]:
                best['dense'] = (flat_idx, count)
            if 0 < count < best['sparse'][1]:
                best['sparse'] = (flat_idx, count)
            flat_idx += 1

    selected = []
    for key in ('dense', 'sparse'):
        idx = best[key][0]
        if idx is not None:
            selected.append(entries[idx][:3])  # (images, b, ann)
    return selected


def select_viz_indices_train(train_dataset_obj, transform):
    """
    Scan the raw train dataset once with shuffle=False and return:
      - index of the sample with the most GT mask contours (dense)
      - index of the sample with the fewest GT mask contours >= 1 (sparse)
    Returns list of (images_tensor, b=0, annotation) loaded at those indices.
    Called once before training starts.
    """
    from torch.utils.data import DataLoader as _DL
    from PLRNet.dataset.train_dataset import collate_fn as _train_collate
    probe_loader = _DL(train_dataset_obj, batch_size=1,
                       shuffle=False, collate_fn=_train_collate,
                       num_workers=0)
    best = {'dense': (None, -1), 'sparse': (None, float('inf'))}
    entries = {}

    coco_obj = train_dataset_obj.coco
    img_ids  = train_dataset_obj.images  # ordered list of COCO image IDs
    for idx, (images, annotations) in enumerate(probe_loader):
        ann    = annotations[0]
        img_id = img_ids[idx]
        ann['filename'] = coco_obj.loadImgs(ids=[img_id])[0]['file_name']
        mask = ann.get('mask', None)
        count = 0
        if mask is not None:
            m = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            contours, _ = cv2.findContours(
                (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = len([c for c in contours if len(c) >= 3])
        entries[idx] = (images, 0, ann, count)
        if count > best['dense'][1]:
            best['dense'] = (idx, count)
        if 0 < count < best['sparse'][1]:
            best['sparse'] = (idx, count)

    selected = []
    for key in ('dense', 'sparse'):
        idx = best[key][0]
        if idx is not None:
            selected.append(entries[idx][:3])  # (images, b, ann)
    return selected


# ------------------------------------------------------------------ #
#  Validation visualization  (GT from COCO object)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_val(model, val_viz_entries, coco_obj, epoch, output_dir, mean, std):
    """
    Save the 2 pre-selected val images (dense + sparse GT).
    val_viz_entries is computed once before training by select_viz_indices_val().
    Output: visualizations/val/{epoch:03d}_{full_patch_name}.png
    """
    model.eval()
    device   = next(model.parameters()).device
    viz_dir  = os.path.join(output_dir, 'visualizations', 'val')
    os.makedirs(viz_dir, exist_ok=True)

    for images, b, ann in val_viz_entries:
        img_id   = ann.get('img_id', None)
        img_name = os.path.splitext(os.path.basename(ann.get('filename', 'val')))[0]
        img_disp = tensor_to_display(images[b], mean, std)
        output, _ = model(images.to(device))

        gt_polys = []
        if img_id is not None:
            gt_polys = [np.array(seg).reshape(-1, 2)
                        for a in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id]))
                        for seg in a['segmentation']]
        pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
        _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir)


# ------------------------------------------------------------------ #
#  Train visualization  (GT from mask via cv2.findContours)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_train(model, train_viz_entries, epoch, output_dir, mean, std):
    """
    Save the 2 pre-selected train images (dense + sparse GT).
    train_viz_entries is computed once before training by select_viz_indices_train().
    Output: visualizations/train/{epoch:03d}_{full_patch_name}.png
    """
    model.eval()
    device  = next(model.parameters()).device
    viz_dir = os.path.join(output_dir, 'visualizations', 'train')
    os.makedirs(viz_dir, exist_ok=True)

    for images, b, ann in train_viz_entries:
        img_name = os.path.splitext(
            os.path.basename(ann.get('filename', 'train')))[0]
        img_disp = tensor_to_display(images[b], mean, std)
        output, _ = model(images.to(device))

        mask = ann.get('mask', None)
        gt_polys = []
        if mask is not None:
            m = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            contours, _ = cv2.findContours(
                (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_polys = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
        pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
        _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir)

    model.train()


# ------------------------------------------------------------------ #
#  Validation loss pass
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(model, val_loader, loss_reducer, device, loss_names):
    model.eval()
    sums  = {k: 0.0 for k in loss_names}
    total = 0.0
    iou_sum = 0.0  # accumulate IoU over all images
    n     = 0
    
    for images, annotations in val_loader:
        images      = images.to(device)
        annotations = to_single_device(annotations, device)

        # forward pass (extras contains refined mask logits)
        loss_dict, extras = model.forward_train(images, annotations)

        weighted = loss_reducer(loss_dict)
        for k in loss_names:
            sums[k] += loss_dict[k].item()
        total += weighted.item()

        # per‑image IoU inside the batch
        remask  = extras['remask_pred'].sigmoid()
        mask_gt = torch.stack([a['mask'].squeeze() for a in annotations]).to(device)

        for b in range(remask.size(0)):
            pred_bin = (remask[b] > 0.5).cpu().numpy()
            gt_bin   = (mask_gt[b] > 0.5).cpu().numpy()
            iou_sum += calc_IoU(pred_bin, gt_bin)

        n += remask.size(0)    # count images, not batches

    # compute averages after all batches
    avg       = {k: sums[k] / max(n, 1) for k in loss_names}
    avg_total = total / max(n, 1)
    avg_iou   = iou_sum / max(n, 1)    # mean IoU over all images
    return avg_total, avg, avg_iou


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
    fieldnames = init_metrics_csv(csv_path, loss_names)

    # Pre-select visualization samples once before training starts
    # val: scan val_loader (no shuffle) → dense + sparse by COCO polygon count
    # train: scan raw dataset with shuffle=False → dense + sparse by mask contour count
    val_coco_obj      = val_dataset.dataset.coco
    val_viz_entries   = select_viz_indices_val(val_dataset)
    train_viz_entries = select_viz_indices_train(train_dataset.dataset, None)
    logger.info(f"Viz samples selected — val: {len(val_viz_entries)}, "
                f"train: {len(train_viz_entries)}")

    start_time = time.time()
    end        = time.time()

    for epoch in range(1, max_epoch + 1):
        meters = MetricLogger(" ")
        model.train()

        epoch_loss_sums  = {k: 0.0 for k in loss_names}
        epoch_total      = 0.0
        epoch_iou_sum    = 0.0
        epoch_n_images   = 0
        n_batches        = 0

        for it, (images, annotations) in enumerate(train_dataset):
            data_time   = time.time() - end
            images      = images.to(device)
            annotations = to_single_device(annotations, device)

            loss_dict, extras = model(images, annotations)
            total_loss        = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_red = {k: v.item() for k, v in loss_dict.items()}
                loss_red      = total_loss.item()
                meters.update(loss=loss_red, **loss_dict_red)
                for k in loss_names:
                    epoch_loss_sums[k] += loss_dict_red.get(k, 0.0)
                epoch_total += loss_red
                n_batches   += 1

                # accumulate train mask IoU using the remask logit already computed
                remask   = extras['remask_pred'].sigmoid()
                mask_gt  = torch.stack([a['mask'].squeeze() for a in annotations]).to(device)
                for b in range(remask.size(0)):
                    pred_bin = (remask[b] > 0.5).cpu().numpy()
                    gt_bin   = (mask_gt[b] > 0.5).cpu().numpy()
                    epoch_iou_sum  += calc_IoU(pred_bin, gt_bin)
                    epoch_n_images += 1

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

        avg_losses     = {k: epoch_loss_sums[k] / max(n_batches, 1) for k in loss_names}
        avg_total      = epoch_total / max(n_batches, 1)
        avg_train_iou  = epoch_iou_sum / max(epoch_n_images, 1)
        current_lr     = optimizer.param_groups[0]["lr"]

        # always overwrite latest
        save_checkpoint('latest.pth', epoch)

        # --- validation + visualizations every val_every epochs ---
        val_total  = float('nan')
        val_losses = {k: float('nan') for k in loss_names}
        val_iou    = float('nan')

        if epoch % val_every == 0:
            logger.info(f"=== Validation at epoch {epoch} ===")
            val_total, val_losses, val_iou = validate(
                model, val_dataset, loss_reducer, device, loss_names)
            logger.info(
                "Val total_loss: {:.4f}  |  {}".format(
                    val_total,
                    "  ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
            )
            logger.info(f"Val mask_iou: {val_iou:.4f}")
            # checkpoint at this val epoch
            save_checkpoint(f'epoch_{epoch}.pth', epoch)

            # best val loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint('best_val_loss.pth', epoch)
                logger.info(f"New best val loss: {best_val_loss:.4f}")

            # visualizations — same 2 images every epoch (dense + sparse GT)
            visualize_val(model, val_viz_entries, val_coco_obj, epoch, output_dir, mean, std)
            visualize_train(model, train_viz_entries, epoch, output_dir, mean, std)

        # --- write metrics.csv ---
        row = {'epoch': epoch, 'train_loss': round(avg_total, 6),
               'train_mask_iou': round(avg_train_iou, 6)}
        for k in loss_names:
            row['w_' + k] = round(avg_losses[k] * loss_weights[k], 6)
        row['val_loss'] = round(val_total, 6) if not np.isnan(val_total) else ''
        for k in loss_names:
            row['val_w_' + k] = (round(val_losses[k] * loss_weights[k], 6)
                                 if not np.isnan(val_losses[k]) else '')
        row['val_mask_iou'] = round(val_iou, 6) if epoch % val_every == 0 else ''
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
