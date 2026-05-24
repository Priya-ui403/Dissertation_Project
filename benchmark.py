"""
Real-time performance benchmark for YOLOPv2.
Reports inference latency (ms/frame), throughput (FPS), parameter count,
and on-disk checkpoint size. Excludes data loading; uses random dummy tensors.

Usage:
    CUDA_VISIBLE_DEVICES=4 python benchmark.py --weights runs/train/last.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from lib.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/last.pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=100,
                        help="Number of dummy frames to time (batch=1).")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated batch sizes to benchmark.")
    parser.add_argument("--half", action="store_true", help="Use FP16.")
    parser.add_argument("--output", type=str, default="runs/test/benchmark.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading {args.weights}...")
    model = build_model(nc=10, width_mult=0.75, depth_mult=0.67)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    if args.half:
        model = model.half()

    # --- Param count & size ---
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_bytes = os.path.getsize(args.weights)
    # Pure parameter footprint (fp32)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Checkpoint size: {ckpt_bytes/1e6:.2f} MB")
    print(f"  Params+buffers (fp32): {(param_bytes+buffer_bytes)/1e6:.2f} MB\n")

    batch_results = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        print(f"--- batch size {bs} ---")
        # Dummy input
        x = torch.randn(bs, 3, args.img_size, args.img_size, device=device)
        if args.half:
            x = x.half()

        # Warmup
        with torch.no_grad():
            for _ in range(args.n_warmup):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time
        n_iters = max(args.n_iters // max(bs, 1), 10)  # keep total frames ~constant
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
        else:
            t0 = time.time()
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(x)
            elapsed = time.time() - t0

        total_frames = n_iters * bs
        ms_per_frame = (elapsed / total_frames) * 1000
        ms_per_batch = (elapsed / n_iters) * 1000
        fps = total_frames / elapsed
        print(f"  iters={n_iters}  total_frames={total_frames}  elapsed={elapsed:.2f}s")
        print(f"  ms/frame: {ms_per_frame:.3f}  ms/batch: {ms_per_batch:.3f}  FPS: {fps:.1f}")
        batch_results.append({
            "batch_size": bs,
            "iterations": n_iters,
            "total_frames": total_frames,
            "elapsed_sec": round(elapsed, 4),
            "ms_per_frame": round(ms_per_frame, 4),
            "ms_per_batch": round(ms_per_batch, 4),
            "fps": round(fps, 2),
        })

    out = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "weights": args.weights,
        "img_size": args.img_size,
        "half_precision": args.half,
        "model": {
            "total_parameters": n_params,
            "total_parameters_M": round(n_params / 1e6, 3),
            "trainable_parameters": n_trainable,
            "checkpoint_size_MB": round(ckpt_bytes / 1e6, 3),
            "param_plus_buffer_MB_fp32": round((param_bytes + buffer_bytes) / 1e6, 3),
        },
        "batches": batch_results,
        "single_frame": next((b for b in batch_results if b["batch_size"] == 1), None),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved benchmark to {args.output}")


if __name__ == "__main__":
    main()
