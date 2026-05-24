"""
Reconstruct training loss curves post-hoc from the saved checkpoints.

The training script (`train.py`) did not write a results log to disk, but it
checkpointed every 5 epochs. This script loads each checkpoint, runs it on a
fixed sample of the labeled training data, and computes the multi-task loss
components (box, obj, cls, drivable-area, lane-line, total). It then plots
each component vs epoch and writes them to `runs/test/plots/`.

NOTE: This is NOT identical to the loss the model saw during training (no
augmentation noise, no DataParallel batch mixing, etc.), but the shape of
the curves is a faithful representation of how training progressed.

Usage:
    CUDA_VISIBLE_DEVICES=4 python recover_train_curves.py \
        --ckpt-dir runs/train --data-root ../bdd100k_data
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lib.models import build_model
from lib.dataset import BDD100KDataset, collate_fn
from lib.loss import MultiTaskLoss


def find_checkpoints(ckpt_dir):
    """Return sorted list of (epoch, path) tuples for epoch_*.pt files."""
    pat = re.compile(r"epoch_(\d+)\.pt$")
    out = []
    for p in Path(ckpt_dir).iterdir():
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


@torch.no_grad()
def loss_on_sample(model, criterion, loader, device, max_batches=None):
    """Average multi-task losses over `loader`. Returns dict of mean components."""
    model.train()  # need training-mode forward (returns det_concat for criterion)
    sums = {"total": 0.0, "det": 0.0, "da": 0.0, "ll": 0.0,
            "box": 0.0, "obj": 0.0, "cls": 0.0}
    n_batches = 0
    for bi, (imgs, targets, da_masks, ll_masks) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = imgs.to(device)
        targets = targets.to(device)
        da_masks = da_masks.to(device)
        ll_masks = ll_masks.to(device)
        det_out, da_pred, ll_pred = model(imgs)
        try:
            total, l_det, l_da, l_ll, det_items = criterion(
                det_out, da_pred, ll_pred, targets, da_masks, ll_masks
            )
        except Exception as e:
            print(f"    batch {bi} criterion error: {e}")
            continue
        bs = imgs.shape[0]
        sums["total"] += float(total.item()) / max(bs, 1)
        sums["det"] += float(l_det) / max(bs, 1)
        sums["da"] += float(l_da)
        sums["ll"] += float(l_ll)
        # det_items: tensor([lbox, lobj, lcls])
        if hasattr(det_items, "shape") and det_items.shape[0] >= 3:
            sums["box"] += float(det_items[0].item())
            sums["obj"] += float(det_items[1].item())
            sums["cls"] += float(det_items[2].item())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}, n_batches


def plot_curves(epochs, series, out_path, title, ylabel="loss"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys in series.items():
        ax.plot(epochs, ys, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="runs/train")
    p.add_argument("--data-root", type=str, default="../bdd100k_data")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=320,
                   help="Number of training images to evaluate each checkpoint on.")
    p.add_argument("--max-batches", type=int, default=None,
                   help="If set, override n-samples by limiting batch count.")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--output-dir", type=str, default="runs/test/plots")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fixed training-data sample ---
    print("Loading labeled training subset (require_labels=True)...")
    full = BDD100KDataset(args.data_root, split="train", img_size=args.img_size,
                          augment=False, require_labels=True)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(full), size=min(args.n_samples, len(full)), replace=False).tolist()
    subset = Subset(full, idxs)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"  Eval sample size: {len(subset)}")

    # --- Iterate checkpoints ---
    ckpts = find_checkpoints(args.ckpt_dir)
    print(f"Found {len(ckpts)} checkpoints: {[e for e, _ in ckpts]}")

    epochs = []
    history = {"total": [], "det": [], "da": [], "ll": [],
               "box": [], "obj": [], "cls": []}

    for epoch, path in ckpts:
        print(f"\nCheckpoint epoch {epoch}: {path.name}")
        t0 = time.time()
        model = build_model(nc=10, width_mult=0.75, depth_mult=0.67)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        criterion = MultiTaskLoss(model, nc=10).to(device)

        sums, nb = loss_on_sample(model, criterion, loader, device,
                                  max_batches=args.max_batches)
        elapsed = time.time() - t0
        print(f"  batches={nb}  elapsed={elapsed:.1f}s  losses={ {k: round(v,4) for k,v in sums.items()} }")
        epochs.append(epoch)
        for k in history:
            history[k].append(sums[k])

        del model, criterion
        torch.cuda.empty_cache()

    # --- Save raw JSON ---
    raw_path = out_dir / "train_curves.json"
    with open(raw_path, "w") as f:
        json.dump({"epochs": epochs, **history,
                   "note": ("Reconstructed by running each saved checkpoint over "
                            f"a fixed sample of {len(subset)} labeled training images. "
                            "This approximates training-set loss but excludes augmentation.")},
                  f, indent=2)
    print(f"\nSaved raw curves to {raw_path}")

    # --- Plot ---
    plot_curves(epochs, {"total": history["total"]},
                out_dir / "loss_total.png",
                "Reconstructed Total Training Loss vs Epoch")
    plot_curves(epochs,
                {"detection": history["det"], "drivable_area": history["da"],
                 "lane_line": history["ll"]},
                out_dir / "loss_task.png",
                "Reconstructed Per-Task Loss vs Epoch")
    plot_curves(epochs,
                {"box": history["box"], "obj": history["obj"], "cls": history["cls"]},
                out_dir / "loss_detection_components.png",
                "Reconstructed Detection Loss Components vs Epoch")
    print(f"Saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
