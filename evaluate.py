"""
YOLOPv2 Comprehensive Evaluation Pipeline.

Tasks:
  1. Quantitative metrics: mAP (detection), mIoU (drivable area), IoU+Acc (lane line)
  2. Visual testing: overlay predictions on 50 random images
  3. Hard example mining: top 10 images with highest loss

Usage:
    CUDA_VISIBLE_DEVICES=4 python evaluate.py --weights runs/train/last.pt --data-root ../bdd100k_data
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.models import build_model
from lib.dataset import BDD100KDataset, collate_fn
from lib.data_download import ensure_bdd100k, KAGGLE_DATASET_DEFAULT
from lib.loss import MultiTaskLoss


# ---------------------------------------------------------------------------
# Detection metrics helpers
# ---------------------------------------------------------------------------

def xywh2xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """NMS on inference output [bs, num_anchors, 5+nc]."""
    output = []
    for xi, x in enumerate(prediction):
        # Filter by confidence
        obj_conf = x[:, 4]
        mask = obj_conf > conf_thres
        x = x[mask]

        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        # Compute class confidence
        cls_conf, cls_idx = x[:, 5:].max(1, keepdim=True)
        x_conf = x[:, 4:5] * cls_conf  # obj_conf * cls_conf

        # [x1,y1,x2,y2, conf, cls]
        boxes = xywh2xyxy(x[:, :4])
        detections = torch.cat([boxes, x_conf, cls_idx.float()], dim=1)

        # Sort by confidence
        detections = detections[detections[:, 4].argsort(descending=True)]

        if detections.shape[0] > max_det * 3:
            detections = detections[:max_det * 3]

        # NMS per class
        keep = []
        unique_cls = detections[:, 5].unique()
        for c in unique_cls:
            dc = detections[detections[:, 5] == c]
            indices = _nms(dc[:, :4], dc[:, 4], iou_thres)
            keep.append(dc[indices])

        if keep:
            keep = torch.cat(keep, dim=0)
            keep = keep[keep[:, 4].argsort(descending=True)][:max_det]
        else:
            keep = torch.zeros((0, 6), device=x.device)

        output.append(keep)
    return output


def _nms(boxes, scores, iou_threshold):
    """Simple NMS implementation."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-7)
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)


def compute_ap(recall, precision):
    """Compute AP from recall and precision arrays (VOC-style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(all_detections, all_ground_truths, nc=10, iou_thresholds=None):
    """
    Compute mAP@0.5 and mAP@0.5:0.95.
    all_detections: list of [N, 6] tensors per image [x1,y1,x2,y2,conf,cls]
    all_ground_truths: list of [M, 5] tensors per image [cls, cx, cy, w, h] (normalized)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    aps_per_threshold = {t: [] for t in iou_thresholds}

    for c in range(nc):
        # Gather all detections and GTs for this class
        dets_all = []
        n_gt = 0
        gt_matched = []

        for img_idx, (dets, gts) in enumerate(zip(all_detections, all_ground_truths)):
            # Get GTs for this class
            if gts.shape[0] > 0:
                gt_mask = gts[:, 0] == c
                gt_boxes = gts[gt_mask][:, 1:]  # cx, cy, w, h (normalized)
                n_gt += gt_boxes.shape[0]
            else:
                gt_boxes = np.zeros((0, 4))

            # Get detections for this class
            if dets.shape[0] > 0:
                det_mask = dets[:, 5] == c
                det_cls = dets[det_mask]
                for d in det_cls:
                    dets_all.append((img_idx, d[4].item(), d[:4].cpu().numpy()))
            gt_matched.append(gt_boxes)

        if n_gt == 0:
            for t in iou_thresholds:
                aps_per_threshold[t].append(0.0)
            continue

        # Sort detections by confidence
        dets_all.sort(key=lambda x: x[1], reverse=True)

        for t in iou_thresholds:
            tp = np.zeros(len(dets_all))
            fp = np.zeros(len(dets_all))
            matched_gt = {i: set() for i in range(len(all_ground_truths))}

            for det_idx, (img_idx, conf, det_box) in enumerate(dets_all):
                gt_boxes = gt_matched[img_idx]
                if gt_boxes.shape[0] == 0:
                    fp[det_idx] = 1
                    continue

                # Convert GT from cx,cy,w,h (normalized) to x1,y1,x2,y2 (pixel 640)
                gt_xyxy = np.zeros((gt_boxes.shape[0], 4))
                gt_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * 640
                gt_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * 640
                gt_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * 640
                gt_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * 640

                # Compute IoU with all GT boxes
                ious = _box_iou_np(det_box, gt_xyxy)
                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]

                if best_iou >= t and best_iou_idx not in matched_gt[img_idx]:
                    tp[det_idx] = 1
                    matched_gt[img_idx].add(best_iou_idx)
                else:
                    fp[det_idx] = 1

            # Compute precision/recall
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (n_gt + 1e-7)
            precision = tp_cum / (tp_cum + fp_cum + 1e-7)
            ap = compute_ap(recall, precision)
            aps_per_threshold[t].append(ap)

    # Per-class AP arrays
    per_class_ap50 = np.array(aps_per_threshold[0.5])
    per_class_ap50_95 = np.mean(
        np.stack([aps_per_threshold[t] for t in iou_thresholds], axis=0),
        axis=0,
    )

    # Average across classes
    map50 = float(per_class_ap50.mean()) if per_class_ap50.size > 0 else 0.0
    map50_95 = float(per_class_ap50_95.mean()) if per_class_ap50_95.size > 0 else 0.0

    return map50, map50_95, per_class_ap50, per_class_ap50_95


def compute_prf1(all_detections, all_ground_truths, nc=10, iou_thresh=0.5):
    """
    Compute global Precision, Recall, F1 at IoU=iou_thresh across all classes.
    Detections are already confidence-filtered by NMS. Returns dict with
    overall + per-class metrics.
    """
    per_class = {}
    tp_total = fp_total = fn_total = 0

    for c in range(nc):
        tp = fp = fn = 0
        # Per-image
        for dets, gts in zip(all_detections, all_ground_truths):
            # GT boxes for class c (cx, cy, w, h normalized -> 640 pixel xyxy)
            if gts.shape[0] > 0:
                gt_mask = gts[:, 0] == c
                gt_c = gts[gt_mask][:, 1:]
            else:
                gt_c = np.zeros((0, 4))

            if gt_c.shape[0] > 0:
                gt_xyxy = np.zeros((gt_c.shape[0], 4))
                gt_xyxy[:, 0] = (gt_c[:, 0] - gt_c[:, 2] / 2) * 640
                gt_xyxy[:, 1] = (gt_c[:, 1] - gt_c[:, 3] / 2) * 640
                gt_xyxy[:, 2] = (gt_c[:, 0] + gt_c[:, 2] / 2) * 640
                gt_xyxy[:, 3] = (gt_c[:, 1] + gt_c[:, 3] / 2) * 640
            else:
                gt_xyxy = np.zeros((0, 4))

            # Predictions for class c
            if dets.shape[0] > 0:
                det_mask = dets[:, 5] == c
                det_c = dets[det_mask]
                # Sort by confidence desc
                det_c = det_c[det_c[:, 4].argsort(descending=True)]
                det_boxes = det_c[:, :4].cpu().numpy()
            else:
                det_boxes = np.zeros((0, 4))

            matched = set()
            for d in det_boxes:
                if gt_xyxy.shape[0] == 0:
                    fp += 1
                    continue
                ious = _box_iou_np(d, gt_xyxy)
                best = np.argmax(ious)
                if ious[best] >= iou_thresh and best not in matched:
                    tp += 1
                    matched.add(best)
                else:
                    fp += 1
            fn += gt_xyxy.shape[0] - len(matched)

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}
        tp_total += tp
        fp_total += fp
        fn_total += fn

    prec_o = tp_total / max(tp_total + fp_total, 1)
    rec_o = tp_total / max(tp_total + fn_total, 1)
    f1_o = 2 * prec_o * rec_o / max(prec_o + rec_o, 1e-12)
    return {
        "overall": {"precision": prec_o, "recall": rec_o, "f1": f1_o,
                    "tp": tp_total, "fp": fp_total, "fn": fn_total},
        "per_class": per_class,
    }


def _box_iou_np(box, boxes):
    """Compute IoU between a single box and multiple boxes (numpy, xyxy format)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-7)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

class SegMetric:
    """Confusion matrix-based segmentation metric."""
    def __init__(self, num_classes):
        self.nc = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add_batch(self, pred, gt):
        """pred, gt: flattened arrays of class indices."""
        mask = (gt >= 0) & (gt < self.nc)
        self.cm += np.bincount(
            self.nc * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.nc ** 2
        ).reshape(self.nc, self.nc)

    def miou(self):
        intersection = np.diag(self.cm)
        union = self.cm.sum(axis=1) + self.cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-7)
        return np.nanmean(iou)

    def pixel_accuracy(self):
        return np.diag(self.cm).sum() / (self.cm.sum() + 1e-7)

    def class_iou(self):
        intersection = np.diag(self.cm)
        union = self.cm.sum(axis=1) + self.cm.sum(axis=0) - intersection
        return intersection / (union + 1e-7)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

DET_CLASSES = ['car', 'bus', 'truck', 'person', 'rider', 'bike',
               'motor', 'traffic light', 'traffic sign', 'train']

COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 255, 0), (128, 128, 0), (0, 0, 255)
]


def visualize_predictions(img, det_boxes, da_mask, ll_mask, save_path):
    """
    Overlay predictions onto image and save.
    img: HxWx3 BGR
    det_boxes: [N, 6] x1,y1,x2,y2,conf,cls (pixel coords for 640x640)
    da_mask: HxW binary
    ll_mask: HxW binary
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    # Drivable area overlay (green)
    da_overlay = np.zeros_like(vis)
    da_overlay[da_mask == 1] = (0, 180, 0)
    vis = cv2.addWeighted(vis, 0.7, da_overlay, 0.3, 0)

    # Lane lines overlay (blue)
    vis[ll_mask == 1] = (255, 50, 50)

    # Detection boxes
    for det in det_boxes:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{DET_CLASSES[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, vis)


# ---------------------------------------------------------------------------
# Decode model output (inference mode)
# ---------------------------------------------------------------------------

def decode_detections(model, imgs, conf_thres=0.25, iou_thres=0.45):
    """Run model in eval mode and decode detection outputs."""
    model.eval()
    with torch.no_grad():
        det_out, da_pred, ll_pred = model(imgs)

    # det_out in eval mode: (concatenated_preds [bs, total, no], raw_per_layer)
    # The Detect head already applies sigmoid in eval mode (see lib/models.py
    # Detect.forward -> y = x[i].sigmoid()), and decodes box xy/wh into pixel
    # coordinates. So `pred` is fully decoded: [x_center, y_center, w, h,
    # obj_conf, cls_conf_0, ..., cls_conf_{nc-1}] with all confs in [0, 1].
    if isinstance(det_out, tuple):
        pred = det_out[0]  # [bs, total_anchors, no]
    else:
        pred = det_out

    # NMS directly on the already-sigmoided predictions
    nms_out = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    # DA seg
    da_seg = da_pred.argmax(1).cpu().numpy()  # [bs, H, W]

    # LL seg
    ll_seg = (ll_pred.squeeze(1).sigmoid() > 0.5).long().cpu().numpy()  # [bs, H, W]

    return nms_out, da_seg, ll_seg


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='YOLOPv2 Evaluation Pipeline')
    parser.add_argument('--weights', type=str, default='runs/train/last.pt')
    parser.add_argument('--data-root', type=str, default='../bdd100k_data')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--num-vis', type=int, default=50, help='Number of images to visualize')
    parser.add_argument('--num-hard', type=int, default=10, help='Number of hard examples')
    parser.add_argument('--nc', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='runs/test')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Which seg split to evaluate on.')
    parser.add_argument('--require-labels', action='store_true',
                        help='Filter the seg split to only images with JSON det/lane labels.')
    parser.add_argument('--kaggle-dataset', type=str, default=KAGGLE_DATASET_DEFAULT,
                        help='Kaggle dataset slug to download if --data-root is missing. '
                             'Default: solesensei/solesensei_bdd100k '
                             '(https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)')
    parser.add_argument('--force-download', action='store_true',
                        help='Re-download the dataset from Kaggle even if a local copy exists.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / 'visualizations'
    hard_dir = output_dir / 'hard_examples'
    vis_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model from {args.weights}...")
    model = build_model(nc=args.nc, width_mult=0.75, depth_mult=0.67)
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint")

    # --- Load Dataset (auto-download from Kaggle if not present locally) ---
    print(f"Loading evaluation dataset (split={args.split}, require_labels={args.require_labels})...")
    args.data_root = ensure_bdd100k(
        args.data_root,
        kaggle_dataset=args.kaggle_dataset,
        force_download=args.force_download,
    )
    val_dataset = BDD100KDataset(
        args.data_root, split=args.split, img_size=args.img_size,
        augment=False, require_labels=args.require_labels,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    print(f"  {len(val_dataset)} evaluation images")

    # Hard-example mining uses the full multi-task loss as the ranking signal.
    # If the subset has no detection GT for an image, the det loss term is
    # handled gracefully (build_targets returns empty tensors).
    loss_model = build_model(nc=args.nc, width_mult=0.75, depth_mult=0.67)
    loss_model.load_state_dict(ckpt['model'])
    loss_model = loss_model.to(device)
    loss_model.train()
    criterion = MultiTaskLoss(loss_model, nc=args.nc).to(device)

    # --- Evaluation ---
    print("\n=== Running Evaluation ===")

    # Metrics
    da_metric = SegMetric(num_classes=2)
    ll_metric = SegMetric(num_classes=2)

    # Detection storage for mAP computation
    all_detections = []
    all_ground_truths = []

    # Hard example mining (multi-task loss)
    per_image_losses = []  # (idx, loss_value, fname)

    # Visualization indices (random 50)
    vis_indices = set(random.sample(range(len(val_dataset)), min(args.num_vis, len(val_dataset))))
    vis_count = 0

    global_img_idx = 0
    t0 = time.time()

    for batch_idx, (imgs, targets, da_masks, ll_masks) in enumerate(val_loader):
        bs = imgs.shape[0]
        imgs_dev = imgs.to(device)
        targets_dev = targets.to(device)
        da_masks_dev = da_masks.to(device)
        ll_masks_dev = ll_masks.to(device)

        # --- Detection + Segmentation inference ---
        nms_out, da_seg, ll_seg = decode_detections(
            model, imgs_dev, conf_thres=args.conf_thres, iou_thres=args.iou_thres
        )

        # --- Forward through loss_model (train mode) for per-image loss ---
        loss_model.train()
        with torch.no_grad():
            det_out_train, da_pred_train, ll_pred_train = loss_model(imgs_dev)

        # --- Per-image multi-task loss for hard-example mining ---
        for i in range(bs):
            img_idx_global = global_img_idx + i
            img_targets = targets_dev[targets_dev[:, 0] == i].clone()
            img_targets[:, 0] = 0  # reset to img-0 inside the single-image batch
            try:
                loss_val, _, _, _, _ = criterion(
                    det_out_train[i:i+1], da_pred_train[i:i+1], ll_pred_train[i:i+1],
                    img_targets, da_masks_dev[i:i+1], ll_masks_dev[i:i+1],
                )
                loss_scalar = loss_val.item()
            except Exception:
                loss_scalar = 0.0
            per_image_losses.append(
                (img_idx_global, loss_scalar, val_dataset.image_files[img_idx_global])
            )

        # --- Segmentation metrics ---
        da_gt = da_masks.numpy()
        ll_gt = ll_masks.numpy()

        for i in range(bs):
            da_metric.add_batch(da_seg[i].flatten(), da_gt[i].flatten())
            ll_metric.add_batch(ll_seg[i].flatten(), ll_gt[i].astype(int).flatten())

        # --- Detection storage for mAP ---
        for i in range(bs):
            all_detections.append(nms_out[i].cpu() if nms_out[i].shape[0] > 0 else torch.zeros((0, 6)))
            img_mask = targets[:, 0] == i
            img_gt = targets[img_mask][:, 1:].numpy()  # [cls, cx, cy, w, h]
            all_ground_truths.append(img_gt)

        # --- Visualization ---
        for i in range(bs):
            img_idx_global = global_img_idx + i
            if img_idx_global in vis_indices:
                # Recover original image (denormalize)
                img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                dets = nms_out[i].cpu().numpy() if nms_out[i].shape[0] > 0 else np.zeros((0, 6))
                save_path = str(vis_dir / f"vis_{img_idx_global:04d}.jpg")
                visualize_predictions(img_bgr, dets, da_seg[i], ll_seg[i], save_path)
                vis_count += 1

        global_img_idx += bs

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {global_img_idx}/{len(val_dataset)} images...")

    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s ({len(val_dataset)/elapsed:.1f} img/s)")

    # --- Detection mAP ---
    total_gt_boxes = int(sum(g.shape[0] for g in all_ground_truths))
    print(f"\n=== Computing Detection mAP (total GT boxes: {total_gt_boxes}) ===")
    if total_gt_boxes == 0:
        print("  No GT boxes available — skipping mAP.")
        map50 = map50_95 = veh_map50 = veh_map5095 = 0.0
        per_class_metrics = {}
    else:
        map50, map50_95, per_cls_ap50, per_cls_ap50_95 = compute_map(
            all_detections, all_ground_truths, nc=args.nc
        )
        print(f"  mAP@0.5      (all {args.nc} classes): {map50:.4f}")
        print(f"  mAP@0.5:0.95 (all {args.nc} classes): {map50_95:.4f}")
        per_class_metrics = {}
        print("  Per-class AP:")
        for i, cls_name in enumerate(DET_CLASSES[:args.nc]):
            ap50_i = float(per_cls_ap50[i]) if i < len(per_cls_ap50) else 0.0
            ap5095_i = float(per_cls_ap50_95[i]) if i < len(per_cls_ap50_95) else 0.0
            per_class_metrics[cls_name] = {
                "AP@0.5": round(ap50_i, 4),
                "AP@0.5:0.95": round(ap5095_i, 4),
            }
            print(f"    {cls_name:15s} AP@0.5: {ap50_i:.4f}  AP@0.5:0.95: {ap5095_i:.4f}")
        vehicle_classes = ['car', 'bus', 'truck']
        veh_idx = [DET_CLASSES.index(c) for c in vehicle_classes
                   if c in DET_CLASSES[:args.nc]]
        if veh_idx:
            veh_map50 = float(np.mean([per_cls_ap50[i] for i in veh_idx]))
            veh_map5095 = float(np.mean([per_cls_ap50_95[i] for i in veh_idx]))
        else:
            veh_map50 = veh_map5095 = 0.0
        print(f"  Vehicle (car/bus/truck) mAP@0.5:      {veh_map50:.4f}")
        print(f"  Vehicle (car/bus/truck) mAP@0.5:0.95: {veh_map5095:.4f}")

    # --- Precision / Recall / F1 at IoU 0.5 ---
    prf1 = None
    if total_gt_boxes > 0:
        print(f"\n=== Precision / Recall / F1 @ IoU 0.5 (conf>={args.conf_thres}) ===")
        prf1 = compute_prf1(all_detections, all_ground_truths,
                            nc=args.nc, iou_thresh=0.5)
        o = prf1["overall"]
        print(f"  Overall  P: {o['precision']:.4f}  R: {o['recall']:.4f}  "
              f"F1: {o['f1']:.4f}  (TP={o['tp']}, FP={o['fp']}, FN={o['fn']})")
        for c in range(args.nc):
            d = prf1["per_class"][c]
            name = DET_CLASSES[c] if c < len(DET_CLASSES) else f"cls{c}"
            print(f"    {name:15s} P: {d['precision']:.4f}  R: {d['recall']:.4f}  "
                  f"F1: {d['f1']:.4f}  (TP={d['tp']}, FP={d['fp']}, FN={d['fn']})")

    # --- Segmentation Metrics ---
    print("\n=== Segmentation Metrics ===")
    da_miou = da_metric.miou()
    da_acc = da_metric.pixel_accuracy()
    da_class_iou = da_metric.class_iou()

    ll_miou = ll_metric.miou()
    ll_acc = ll_metric.pixel_accuracy()
    ll_class_iou = ll_metric.class_iou()

    print(f"  Drivable Area:")
    print(f"    mIoU:     {da_miou:.4f}")
    print(f"    Accuracy: {da_acc:.4f}")
    print(f"    BG IoU:   {da_class_iou[0]:.4f}  DA IoU: {da_class_iou[1]:.4f}")
    print(f"  Lane Line:")
    print(f"    mIoU:     {ll_miou:.4f}")
    print(f"    Accuracy: {ll_acc:.4f}")
    print(f"    BG IoU:   {ll_class_iou[0]:.4f}  LL IoU: {ll_class_iou[1]:.4f}")

    # --- Save Metrics ---
    detection_block = {
        "mAP@0.5": round(float(map50), 4),
        "mAP@0.5:0.95": round(float(map50_95), 4),
        "vehicle_mAP@0.5": round(float(veh_map50), 4),
        "vehicle_mAP@0.5:0.95": round(float(veh_map5095), 4),
        "per_class": per_class_metrics,
        "total_gt_boxes": total_gt_boxes,
        "conf_threshold_for_PRF1": args.conf_thres,
    }
    if prf1 is not None:
        detection_block["overall_PRF1@IoU0.5"] = {
            "precision": round(prf1["overall"]["precision"], 4),
            "recall": round(prf1["overall"]["recall"], 4),
            "f1": round(prf1["overall"]["f1"], 4),
            "tp": prf1["overall"]["tp"],
            "fp": prf1["overall"]["fp"],
            "fn": prf1["overall"]["fn"],
        }
        detection_block["per_class_PRF1@IoU0.5"] = {
            (DET_CLASSES[c] if c < len(DET_CLASSES) else f"cls{c}"): {
                "precision": round(prf1["per_class"][c]["precision"], 4),
                "recall": round(prf1["per_class"][c]["recall"], 4),
                "f1": round(prf1["per_class"][c]["f1"], 4),
                "tp": prf1["per_class"][c]["tp"],
                "fp": prf1["per_class"][c]["fp"],
                "fn": prf1["per_class"][c]["fn"],
            }
            for c in range(args.nc)
        }
    metrics = {
        "detection": detection_block,
        "drivable_area_segmentation": {
            "mIoU": round(float(da_miou), 4),
            "pixel_accuracy": round(float(da_acc), 4),
            "background_IoU": round(float(da_class_iou[0]), 4),
            "drivable_IoU": round(float(da_class_iou[1]), 4),
        },
        "lane_line_segmentation": {
            "mIoU": round(float(ll_miou), 4),
            "pixel_accuracy": round(float(ll_acc), 4),
            "background_IoU": round(float(ll_class_iou[0]), 4),
            "lane_IoU": round(float(ll_class_iou[1]), 4),
        },
        "meta": {
            "weights": args.weights,
            "split": args.split,
            "require_labels": args.require_labels,
            "num_eval_images": len(val_dataset),
            "img_size": args.img_size,
            "conf_threshold": args.conf_thres,
            "iou_threshold": args.iou_thres,
            "inference_time_sec": round(elapsed, 2),
            "fps": round(len(val_dataset) / elapsed, 2),
        }
    }

    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # --- Hard Example Mining ---
    print(f"\n=== Hard Example Mining (top {args.num_hard}) ===")
    per_image_losses.sort(key=lambda x: x[1], reverse=True)
    hard_examples = per_image_losses[:args.num_hard]

    hard_info = []
    for rank, (idx, loss_val, fname) in enumerate(hard_examples):
        print(f"  #{rank+1}: {fname} (loss: {loss_val:.4f})")
        hard_info.append({"rank": rank + 1, "filename": fname, "loss": round(loss_val, 4), "index": idx})

        # Use the dataset's __getitem__ to get a properly letterboxed tensor
        # (matches the training-time pre-processing).
        try:
            img_tensor, _, _, _ = val_dataset[idx]  # img_tensor: [3, H, W] float in [0, 1]
        except Exception as e:
            print(f"    Warning: failed to load hard example {fname}: {e}")
            continue

        img_batch = img_tensor.unsqueeze(0).to(device)
        nms_out_h, da_seg_h, ll_seg_h = decode_detections(
            model, img_batch, conf_thres=args.conf_thres, iou_thres=args.iou_thres
        )

        # Recover BGR uint8 image for drawing
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        dets_h = nms_out_h[0].cpu().numpy() if nms_out_h[0].shape[0] > 0 else np.zeros((0, 6))
        save_path = str(hard_dir / f"hard_{rank+1:02d}_{fname}")
        visualize_predictions(img_vis, dets_h, da_seg_h[0], ll_seg_h[0], save_path)

    # Save hard examples info
    with open(hard_dir / 'hard_examples_info.json', 'w') as f:
        json.dump(hard_info, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Visualizations saved: {vis_dir} ({vis_count} images)")
    print(f"  Hard examples saved:  {hard_dir} ({len(hard_examples)} images)")
    print(f"  Metrics file:         {metrics_path}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
