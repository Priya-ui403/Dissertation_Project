"""
YOLOPv2 Training Script — Train from scratch on BDD100K.
Multi-task: object detection + drivable area seg + lane line seg.

Usage:
    python train.py --data-root ../bdd100k_data --epochs 100 --batch-size 8
"""

import argparse
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from lib.models import build_model
from lib.dataset import BDD100KDataset, collate_fn
from lib.data_download import ensure_bdd100k, KAGGLE_DATASET_DEFAULT
from lib.loss import MultiTaskLoss
from utils.utils import AverageMeter, SegmentationMetric, select_device


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOPv2 Training')
    parser.add_argument('--data-root', type=str, default='../bdd100k_data',
                        help='Path to BDD100K data root')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0', help='cuda device or cpu')
    parser.add_argument('--gpu-ids', type=str, default='1,3,4',
                        help='Comma-separated GPU ids for DataParallel')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--width-mult', type=float, default=0.75,
                        help='Model width multiplier (0.5=small, 0.75=medium, 1.0=large)')
    parser.add_argument('--depth-mult', type=float, default=0.67,
                        help='Model depth multiplier')
    parser.add_argument('--save-dir', type=str, default='runs/train')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--nc', type=int, default=10, help='Number of detection classes')
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--det-weight', type=float, default=1.0)
    parser.add_argument('--da-weight', type=float, default=1.0)
    parser.add_argument('--ll-weight', type=float, default=1.0)
    parser.add_argument('--kaggle-dataset', type=str, default=KAGGLE_DATASET_DEFAULT,
                        help='Kaggle dataset slug to download if --data-root is missing. '
                             'Default: solesensei/solesensei_bdd100k '
                             '(https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)')
    parser.add_argument('--force-download', action='store_true',
                        help='Re-download the dataset from Kaggle even if a local copy exists.')
    return parser.parse_args()


def cosine_lr_scheduler(optimizer, lrf, epochs):
    """Cosine annealing LR scheduler."""
    lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def warmup_lr(optimizer, ni, nw, lr0, warmup_bias_lr=0.1, warmup_momentum=0.8, momentum=0.937):
    """Linear warmup for learning rate and momentum."""
    xi = [0, nw]
    for j, x in enumerate(optimizer.param_groups):
        # Bias lr falls from warmup_bias_lr to lr0, others rise from 0 to lr0
        x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr']])
        if 'momentum' in x:
            x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args, nw):
    model.train()
    loss_meter = AverageMeter()
    det_meter = AverageMeter()
    da_meter = AverageMeter()
    ll_meter = AverageMeter()

    nb = len(dataloader)
    t0 = time.time()

    for i, (imgs, targets, da_masks, ll_masks) in enumerate(dataloader):
        ni = i + nb * epoch  # number integrated batches (since train start)

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        da_masks = da_masks.to(device)
        ll_masks = ll_masks.to(device)

        # Warmup
        if ni <= nw:
            xi = [0, nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = float(
                    (ni / nw) * x['initial_lr'] if j != 2
                    else ((1 - ni / nw) * 0.1 + (ni / nw) * x['initial_lr'])
                )

        # Forward
        with autocast(enabled=(device.type != 'cpu')):
            det_out, da_pred, ll_pred = model(imgs)
            loss, l_det, l_da, l_ll, _ = criterion(det_out, da_pred, ll_pred,
                                                     targets, da_masks, ll_masks)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Metrics
        bs = imgs.size(0)
        loss_meter.update(loss.item(), bs)
        det_meter.update(l_det, bs)
        da_meter.update(l_da, bs)
        ll_meter.update(l_ll, bs)

        if i % 50 == 0:
            mem = f'{torch.cuda.memory_reserved(device) / 1e9:.1f}GB' if device.type == 'cuda' else 'CPU'
            lr = optimizer.param_groups[0]['lr']
            print(f'  Epoch {epoch} [{i}/{nb}]  '
                  f'loss: {loss_meter.avg:.4f}  '
                  f'det: {det_meter.avg:.4f}  da: {da_meter.avg:.4f}  ll: {ll_meter.avg:.4f}  '
                  f'lr: {lr:.6f}  mem: {mem}  '
                  f'time: {time.time() - t0:.1f}s')

    return loss_meter.avg


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    da_metric = SegmentationMetric(2)
    ll_metric = SegmentationMetric(2)

    for imgs, targets, da_masks, ll_masks in dataloader:
        imgs = imgs.to(device)
        da_masks_np = da_masks.numpy()
        ll_masks_np = ll_masks.numpy().astype(int)

        det_out, da_pred, ll_pred = model(imgs)

        # Drivable area
        if da_pred.shape[2:] != da_masks.shape[1:]:
            da_pred = nn.functional.interpolate(da_pred, size=da_masks.shape[1:],
                                                 mode='bilinear', align_corners=True)
        da_pred_cls = da_pred.argmax(1).cpu().numpy()
        da_metric.addBatch(da_pred_cls.flatten(), da_masks_np.flatten())

        # Lane line
        if ll_pred.shape[2:] != ll_masks.shape[1:]:
            ll_pred = nn.functional.interpolate(ll_pred, size=ll_masks.shape[1:],
                                                 mode='bilinear', align_corners=True)
        ll_pred_cls = (ll_pred.squeeze(1).sigmoid() > 0.5).long().cpu().numpy()
        ll_metric.addBatch(ll_pred_cls.flatten(), ll_masks_np.flatten())

    da_miou = da_metric.meanIntersectionOverUnion()
    da_acc = da_metric.pixelAccuracy()
    ll_iou = ll_metric.IntersectionOverUnion()
    ll_acc = ll_metric.lineAccuracy()

    print(f'  [VAL] DA mIoU: {da_miou:.4f}  DA Acc: {da_acc:.4f}  '
          f'LL IoU: {ll_iou:.4f}  LL Acc: {ll_acc:.4f}')

    return da_miou, ll_iou


def main():
    args = parse_args()

    # Multi-GPU setup
    gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda:0')  # primary device (maps to first in CUDA_VISIBLE_DEVICES)
    print(f'Using GPUs: {gpu_ids} (CUDA_VISIBLE_DEVICES={args.gpu_ids})')
    print(f'Primary device: {device}')

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model (from scratch — no pretrained weights)
    print("\n=== Building YOLOPv2 from scratch (no pretrained weights) ===")
    model = build_model(nc=args.nc, width_mult=args.width_mult, depth_mult=args.depth_mult)
    model = model.to(device)

    # Dataset (auto-download from Kaggle if not present locally)
    print("\n=== Loading dataset ===")
    args.data_root = ensure_bdd100k(
        args.data_root,
        kaggle_dataset=args.kaggle_dataset,
        force_download=args.force_download,
    )
    train_dataset = BDD100KDataset(args.data_root, split='train', img_size=args.img_size, augment=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Loss (build BEFORE wrapping model in DataParallel so it can access model.detect)
    criterion = MultiTaskLoss(
        model, nc=args.nc,
        det_weight=args.det_weight,
        da_weight=args.da_weight,
        ll_weight=args.ll_weight,
    ).to(device)

    # Wrap model in DataParallel for multi-GPU
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f'Using DataParallel on {n_gpus} GPUs')
        model = nn.DataParallel(model)
    else:
        print(f'Using single GPU')

    # Optimizer (SGD with weight decay)
    pg0, pg1, pg2 = [], [], []  # no decay, weight decay, biases
    for k, v in model.named_parameters():
        if '.bias' in k:
            pg2.append(v)
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)
        else:
            pg0.append(v)

    optimizer = optim.SGD(pg0, lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    for g in optimizer.param_groups:
        g['initial_lr'] = args.lr

    # Scheduler
    scheduler = cosine_lr_scheduler(optimizer, lrf=0.01, epochs=args.epochs)

    # Mixed precision
    scaler = GradScaler(enabled=(device.type != 'cpu'))

    # Warmup iterations
    nw = max(round(args.warmup_epochs * len(train_loader)), 1000)

    # Resume
    start_epoch = 0
    best_metric = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['model']
        # Handle DataParallel state dict
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n=== Training for {args.epochs} epochs ===")
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Image size:   {args.img_size}")
    print(f"  Device:       {device} ({n_gpus} GPUs)")
    print(f"  Save dir:     {save_dir}\n")

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs - 1} ---")

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                    device, epoch, args, nw)
        scheduler.step()

        t_epoch = time.time() - t_start
        print(f'  Epoch {epoch} done in {t_epoch:.1f}s  avg_loss: {avg_loss:.4f}')

        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric,
            }, save_dir / f'epoch_{epoch}.pt')

    # Save final
    save_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': args.epochs - 1,
        'model': save_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_metric': best_metric,
    }, save_dir / 'last.pt')
    print(f'\nTraining complete. Best metric: {best_metric:.4f}')
    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    main()
