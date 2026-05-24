"""
Lane-Line diagnostic:
  1. Sweep the LL sigmoid threshold and report IoU / pixel accuracy at each.
  2. Save side-by-side panels (image | GT mask | predicted heatmap | thresholded
     mask) for a handful of samples to visually diagnose where the model fails.

Usage:
    CUDA_VISIBLE_DEVICES=4 python diagnose_ll.py \
        --weights runs/train/last.pt --data-root ../bdd100k_data
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.models import build_model
from lib.dataset import BDD100KDataset, collate_fn


# -------------------------------------------------------------------- helpers


def ll_iou_from_logits(ll_logits, ll_gt, threshold):
    """
    ll_logits: [B, 1, H, W] raw logits (or [B, H, W])
    ll_gt:     [B, H, W] in {0, 1}
    threshold: scalar in (0, 1) applied to sigmoid(logits)
    Returns (intersection, gt_sum, pred_sum, correct_pixels, total_pixels)
    """
    if ll_logits.dim() == 4:
        ll_logits = ll_logits.squeeze(1)
    prob = ll_logits.sigmoid()
    pred = (prob > threshold).to(torch.uint8)
    gt = ll_gt.to(torch.uint8)
    inter = (pred & gt).sum().item()
    gt_sum = gt.sum().item()
    pred_sum = pred.sum().item()
    correct = (pred == gt).sum().item()
    total = gt.numel()
    return inter, gt_sum, pred_sum, correct, total


def heatmap_overlay(img_bgr, prob_map, alpha=0.55):
    """Overlay a probability heatmap (HxW float in [0,1]) on a BGR image."""
    hm = (prob_map * 255).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0)


def mask_overlay(img_bgr, mask, color=(0, 255, 255), alpha=0.55):
    """Overlay a binary mask on a BGR image with the given color."""
    overlay = img_bgr.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def make_panel(img_bgr, gt_mask, prob_map, pred_mask, title):
    """Compose a 1x4 panel: original | GT overlay | prob heatmap | pred overlay."""
    pieces = [
        img_bgr,
        mask_overlay(img_bgr, gt_mask, color=(0, 255, 0)),
        heatmap_overlay(img_bgr, prob_map),
        mask_overlay(img_bgr, pred_mask, color=(0, 255, 255)),
    ]
    labels = ["original", "GT (green)", "pred prob (jet)", "pred>thr (yellow)"]
    h, w = img_bgr.shape[:2]
    pad = 28
    for i, (p, lab) in enumerate(zip(pieces, labels)):
        cv2.rectangle(p, (0, 0), (w, pad), (0, 0, 0), -1)
        cv2.putText(p, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    panel = np.hstack(pieces)
    # Top title bar
    title_bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([title_bar, panel])


# ---------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/last.pt")
    parser.add_argument("--data-root", type=str, default="../bdd100k_data")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num-panels", type=int, default=12,
                        help="Number of side-by-side diagnostic panels to save")
    parser.add_argument("--panel-thr", type=float, default=0.3,
                        help="Threshold used when drawing the binary mask in panels")
    parser.add_argument("--output-dir", type=str, default="runs/test/ll_diagnostics")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which seg split to evaluate on.")
    parser.add_argument("--require-labels", action="store_true",
                        help="Restrict the split to images with JSON det/lane labels.")
    parser.add_argument(
        "--thresholds", type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated list of sigmoid thresholds to sweep.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {args.weights}...")
    model = build_model(nc=10, width_mult=0.75, depth_mult=0.67)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint")

    # --- Dataset ---
    print(f"Loading dataset (split={args.split}, require_labels={args.require_labels})...")
    val_dataset = BDD100KDataset(
        args.data_root, split=args.split, img_size=args.img_size,
        augment=False, require_labels=args.require_labels,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    print(f"  {len(val_dataset)} val images")

    # --- Sweep state ---
    thresholds = [float(t) for t in args.thresholds.split(",")]
    stats = {t: {"inter": 0, "gt": 0, "pred": 0, "correct": 0, "total": 0} for t in thresholds}

    # Track per-image max prob and gt prevalence to understand the distribution
    max_probs = []   # max sigmoid logit per image
    mean_probs = []  # mean sigmoid logit per image
    gt_prev = []     # fraction of positive GT pixels per image

    # For panels we pick the highest-positive-GT images we encounter
    panel_candidates = []  # (gt_pos_count, idx, img_tensor, gt_mask, prob_map)

    print(f"\nSweeping thresholds: {thresholds}")
    global_idx = 0
    with torch.no_grad():
        for imgs, _targets, _da, ll_masks in val_loader:
            imgs_dev = imgs.to(device)
            ll_gt_dev = ll_masks.to(device)
            _, _, ll_pred = model(imgs_dev)  # [B, 1, H, W]
            ll_logits = ll_pred.squeeze(1)   # [B, H, W]
            ll_prob = ll_logits.sigmoid()

            for t in thresholds:
                inter, gs, ps, corr, tot = ll_iou_from_logits(ll_logits, ll_gt_dev, t)
                stats[t]["inter"] += inter
                stats[t]["gt"] += gs
                stats[t]["pred"] += ps
                stats[t]["correct"] += corr
                stats[t]["total"] += tot

            B = imgs.shape[0]
            for i in range(B):
                p = ll_prob[i]
                max_probs.append(p.max().item())
                mean_probs.append(p.mean().item())
                gt_pos = int(ll_gt_dev[i].sum().item())
                total_px = int(ll_gt_dev[i].numel())
                gt_prev.append(gt_pos / max(total_px, 1))

                # Track candidates for panels: prefer images where the model
                # has the strongest LL response. (When val GT is empty we want
                # to inspect predictions; if GT exists we still surface the
                # most active images, which are usually the most informative.)
                pred_pos = float((p > 0.3).sum().item())
                rank_key = max(pred_pos, float(gt_pos))
                panel_candidates.append((rank_key, global_idx + i, imgs[i].cpu(),
                                         ll_masks[i].cpu(), p.cpu()))
            global_idx += B

    # --- Compute final metrics per threshold ---
    print("\n=== LL Threshold Sweep ===")
    print(f"{'thr':>6} | {'IoU':>7} | {'precision':>9} | {'recall':>7} | {'pixel_acc':>9} | {'pred_frac':>9}")
    sweep_results = []
    for t in thresholds:
        s = stats[t]
        iou = s["inter"] / max(s["gt"] + s["pred"] - s["inter"], 1)
        prec = s["inter"] / max(s["pred"], 1)
        rec = s["inter"] / max(s["gt"], 1)
        acc = s["correct"] / max(s["total"], 1)
        pred_frac = s["pred"] / max(s["total"], 1)
        print(f"{t:>6.3f} | {iou:>7.4f} | {prec:>9.4f} | {rec:>7.4f} | {acc:>9.4f} | {pred_frac:>9.6f}")
        sweep_results.append({
            "threshold": t,
            "IoU": round(iou, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "pixel_accuracy": round(acc, 6),
            "predicted_positive_fraction": round(pred_frac, 8),
        })

    # Best-IoU threshold
    best = max(sweep_results, key=lambda r: r["IoU"])
    print(f"\nBest IoU: {best['IoU']:.4f} at threshold {best['threshold']}")

    # --- Diagnostic statistics ---
    max_probs = np.asarray(max_probs)
    mean_probs = np.asarray(mean_probs)
    gt_prev = np.asarray(gt_prev)
    diag_stats = {
        "max_sigmoid_per_image": {
            "min": float(max_probs.min()),
            "median": float(np.median(max_probs)),
            "mean": float(max_probs.mean()),
            "max": float(max_probs.max()),
            "p90": float(np.percentile(max_probs, 90)),
            "p99": float(np.percentile(max_probs, 99)),
        },
        "mean_sigmoid_per_image": {
            "min": float(mean_probs.min()),
            "median": float(np.median(mean_probs)),
            "mean": float(mean_probs.mean()),
            "max": float(mean_probs.max()),
        },
        "gt_positive_fraction_per_image": {
            "min": float(gt_prev.min()),
            "median": float(np.median(gt_prev)),
            "mean": float(gt_prev.mean()),
            "max": float(gt_prev.max()),
        },
    }
    print("\n=== Output distribution ===")
    print(json.dumps(diag_stats, indent=2))

    # --- Save sweep + diag to JSON ---
    out = {
        "weights": args.weights,
        "num_val_images": len(val_dataset),
        "sweep": sweep_results,
        "best": best,
        "distribution": diag_stats,
    }
    sweep_path = out_dir / "ll_threshold_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    # --- Pick samples with most GT lane pixels for panels ---
    panel_candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = panel_candidates[: args.num_panels]
    print(f"\nWriting {len(chosen)} diagnostic panels to {panels_dir}/ "
          f"(threshold={args.panel_thr})...")
    for rank, (gt_pos, idx, img_t, gt_t, prob_t) in enumerate(chosen):
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gt_mask = gt_t.numpy().astype(np.uint8)
        prob_map = prob_t.numpy().astype(np.float32)
        pred_mask = (prob_map > args.panel_thr).astype(np.uint8)
        title = (f"#{rank+1}  idx={idx}  file={val_dataset.image_files[idx]}  "
                 f"GT+px={gt_pos}  pred+px={int(pred_mask.sum())}  thr={args.panel_thr}")
        panel = make_panel(img_bgr, gt_mask, prob_map, pred_mask, title)
        cv2.imwrite(str(panels_dir / f"panel_{rank+1:02d}_idx{idx:04d}.jpg"), panel)

    print("Done.")


if __name__ == "__main__":
    main()
