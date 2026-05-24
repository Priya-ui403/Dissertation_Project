"""
Generate qualitative visualizations of the trained YOLOPv2 model:
  - `runs/test/visualizations/` : 50 random images from the labeled test set
  - `runs/test/visualizations/challenging_conditions/night/`
  - `runs/test/visualizations/challenging_conditions/rain/`
  - `runs/test/visualizations/challenging_conditions/crowded/`

Each output is the input image with predicted bboxes + drivable-area overlay
+ lane-line overlay drawn on top.

Usage:
    CUDA_VISIBLE_DEVICES=4 python visualize_challenging.py \
        --weights runs/train/last.pt --data-root ../bdd100k_data
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.models import build_model
from lib.dataset import BDD100KDataset, collate_fn

# Re-use the helpers from evaluate.py
from evaluate import (
    DET_CLASSES, COLORS,
    decode_detections, visualize_predictions,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/train/last.pt")
    p.add_argument("--data-root", type=str, default="../bdd100k_data")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--num-random", type=int, default=50)
    p.add_argument("--per-category", type=int, default=12,
                   help="Number of images to save per challenging-condition category.")
    p.add_argument("--crowded-min-objects", type=int, default=33,
                   help="Minimum #GT objects in an image to count as 'crowded'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=str, default="runs/test/visualizations")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    chal_root = out_root / "challenging_conditions"
    night_dir = chal_root / "night"
    rain_dir = chal_root / "rain"
    crowd_dir = chal_root / "crowded"
    for d in (night_dir, rain_dir, crowd_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading {args.weights}...")
    model = build_model(nc=10, width_mult=0.75, depth_mult=0.67)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # --- Dataset: labeled training subset (acts as our held-out test set) ---
    print("Loading labeled subset (split=train, require_labels=True)...")
    ds = BDD100KDataset(args.data_root, split="train", img_size=args.img_size,
                        augment=False, require_labels=True)
    n = len(ds)
    print(f"  {n} labeled images")

    # --- Categorize images via JSON attributes ---
    label_lookup = ds.label_lookup
    night_idx, rain_idx, crowd_idx = [], [], []
    for i, fname in enumerate(ds.image_files):
        entry = label_lookup.get(fname, {})
        attrs = entry.get("attributes", {})
        n_obj = sum(1 for l in entry.get("labels", []) if "box2d" in l)
        tod = attrs.get("timeofday", "")
        weather = attrs.get("weather", "")
        if tod == "night":
            night_idx.append(i)
        if weather == "rainy":
            rain_idx.append(i)
        if n_obj >= args.crowded_min_objects:
            crowd_idx.append(i)

    print(f"  Categorized: night={len(night_idx)}  rain={len(rain_idx)}  "
          f"crowded(>={args.crowded_min_objects} obj)={len(crowd_idx)}")

    # Pick random sample for general visualization
    random_idx = sorted(random.sample(range(n), min(args.num_random, n)))
    # Pick subsets for challenging
    def pick(lst, k):
        return random.sample(lst, min(k, len(lst)))
    night_pick = set(pick(night_idx, args.per_category))
    rain_pick = set(pick(rain_idx, args.per_category))
    crowd_pick = set(pick(crowd_idx, args.per_category))

    # Combine all indices we need to visualize
    needed = set(random_idx) | night_pick | rain_pick | crowd_pick
    print(f"  Total unique images to render: {len(needed)}")

    # --- Inference loop ---
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    global_idx = 0
    n_saved = {"random": 0, "night": 0, "rain": 0, "crowded": 0}
    for imgs, _targets, _da, _ll in loader:
        bs = imgs.shape[0]
        # Skip batches with no needed images
        batch_indices = list(range(global_idx, global_idx + bs))
        if not any(i in needed for i in batch_indices):
            global_idx += bs
            continue

        imgs_dev = imgs.to(device)
        nms_out, da_seg, ll_seg = decode_detections(
            model, imgs_dev, conf_thres=args.conf_thres, iou_thres=args.iou_thres
        )

        for i in range(bs):
            gi = global_idx + i
            if gi not in needed:
                continue
            img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            dets = nms_out[i].cpu().numpy() if nms_out[i].shape[0] > 0 else np.zeros((0, 6))

            fname = ds.image_files[gi]
            if gi in random_idx:
                save = out_root / f"vis_{gi:05d}_{fname}"
                visualize_predictions(img_bgr.copy(), dets, da_seg[i], ll_seg[i], str(save))
                n_saved["random"] += 1
            if gi in night_pick:
                save = night_dir / f"night_{gi:05d}_{fname}"
                visualize_predictions(img_bgr.copy(), dets, da_seg[i], ll_seg[i], str(save))
                n_saved["night"] += 1
            if gi in rain_pick:
                save = rain_dir / f"rain_{gi:05d}_{fname}"
                visualize_predictions(img_bgr.copy(), dets, da_seg[i], ll_seg[i], str(save))
                n_saved["rain"] += 1
            if gi in crowd_pick:
                save = crowd_dir / f"crowded_{gi:05d}_{fname}"
                visualize_predictions(img_bgr.copy(), dets, da_seg[i], ll_seg[i], str(save))
                n_saved["crowded"] += 1
        global_idx += bs

    # --- Manifest ---
    manifest = {
        "random_sample_count": n_saved["random"],
        "challenging_conditions": {
            "night": {
                "matched_images": len(night_idx),
                "saved": n_saved["night"],
                "filter": "attributes.timeofday == 'night'",
            },
            "rain": {
                "matched_images": len(rain_idx),
                "saved": n_saved["rain"],
                "filter": "attributes.weather == 'rainy'",
            },
            "crowded": {
                "matched_images": len(crowd_idx),
                "saved": n_saved["crowded"],
                "filter": f"#box2d labels >= {args.crowded_min_objects}",
            },
        },
        "output_root": str(out_root),
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Saved:", n_saved)
    print(f"Manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
