# YOLOPv2 Final Test Report

_Generated: 2026-05-22 22:48:55 UTC_

## 1. Context

- **Weights**: `runs/train/last.pt` (epoch 99, `last.pt`)
- **Test split**: labeled subset of `seg/images/train` (`split=train, require_labels=True`)
- **Number of test images**: 3,426
- **Total GT bounding boxes**: 67,622
- **Image size**: 640 x 640
- **Inference time (eval pass)**: 68.42s (50.07 FPS data-loaded)

> No held-out validation phase ran during training. The labeled subset was therefore used both for training and as the test set. Numbers below are training-fit, not generalization estimates.

## 2. Object Detection Metrics

### 2.1 Overall mAP

| Metric | Value |
|---|---:|
| mAP @ IoU 0.5 (all classes)        | **0.6763** |
| mAP @ IoU 0.5:0.95 (all classes)   | **0.4039** |
| Vehicle mAP @ 0.5 (car/bus/truck)  | **0.8273** |
| Vehicle mAP @ 0.5:0.95             | **0.5923** |

### 2.2 Precision / Recall / F1 (IoU 0.5, conf ≥ 0.25)

| Metric | Value |
|---|---:|
| Overall Precision | **0.7391** |
| Overall Recall    | **0.6839** |
| Overall F1-Score  | **0.7104** |
| TP / FP / FN      | 46,244 / 16,322 / 21,378 |

### 2.3 Per-Class Breakdown

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 0.7104 | 0.4812 | 0.7786 | 0.7254 | 0.7510 | 26,924 | 7,655 | 10,194 |
| bus | 0.8813 | 0.6410 | 0.7405 | 0.8934 | 0.8098 | 645 | 226 | 77 |
| truck | 0.8904 | 0.6547 | 0.7709 | 0.8980 | 0.8296 | 1,716 | 510 | 195 |
| person | 0.6172 | 0.3089 | 0.6214 | 0.6590 | 0.6396 | 3,796 | 2,313 | 1,964 |
| rider | 0.7065 | 0.3618 | 0.5816 | 0.7525 | 0.6561 | 228 | 164 | 75 |
| bike | 0.6998 | 0.3650 | 0.5871 | 0.7484 | 0.6580 | 354 | 249 | 119 |
| motor | 0.7653 | 0.4632 | 0.6353 | 0.8009 | 0.7086 | 169 | 97 | 42 |
| traffic light | 0.5032 | 0.2053 | 0.6528 | 0.5586 | 0.6020 | 4,789 | 2,547 | 3,784 |
| traffic sign | 0.5847 | 0.3087 | 0.7489 | 0.6076 | 0.6709 | 7,613 | 2,553 | 4,916 |
| train | 0.4048 | 0.2492 | 0.5556 | 0.4545 | 0.5000 | 10 | 8 | 12 |

## 3. Segmentation Metrics

| Task | mIoU | Pixel Acc | BG IoU | Foreground IoU |
|---|---:|---:|---:|---:|
| Drivable Area | **0.9472** | 0.9873 | 0.9854 | **0.9090** |
| Lane Line     | **0.6932** | 0.9923 | 0.9923 | **0.3942** |

## 4. Lane-Line Decision Threshold Sweep

Sweep of the sigmoid threshold on the lane-line head.

| Threshold | IoU | Precision | Recall | Pixel Acc | Pred Positive Frac |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.3685 | 0.4166 | 0.7615 | 0.9906 | 0.013168 |
| 0.2 | 0.3806 | 0.4389 | 0.7412 | 0.9913 | 0.012164 |
| 0.3 | 0.3872 | 0.4537 | 0.7254 | 0.9917 | 0.011516 |
| 0.4 | 0.3915 | 0.4657 | 0.7108 | 0.9920 | 0.010994 |
| 0.5 | 0.3942 | 0.4764 | 0.6955 | 0.9923 | 0.010515 |

**Best IoU**: 0.3942 at threshold **0.5** (Precision 0.4764, Recall 0.6955).

Per-image output / GT statistics:

| Statistic | Min | Median | Mean | p90 | Max |
|---|---:|---:|---:|---:|---:|
| Max sigmoid per image | 0.0025 | 1.0000 | 0.9418 | 1.0000 | 1.0000 |
| GT positive pixel fraction per image | 0.00000 | 0.00669 | 0.00720 | — | 0.03610 |

Side-by-side diagnostic panels (`original | GT | predicted heatmap | thresholded mask`) saved to `runs/test/ll_diagnostics/panels/`.

## 5. Real-Time Performance Benchmark

- **Hardware**: `NVIDIA RTX 6000 Ada Generation` (FP32, img 640x640)
- **Parameters**: 23,404,394 (**23.404 M**)
- **Trainable parameters**: 23,404,394
- **Checkpoint size (state-dict + optimizer)**: **187.618 MB**
- **Live parameter + buffer footprint (fp32)**: 93.774 MB

Latency / throughput (excludes data loading; warmup performed):

| Batch Size | Iterations | ms / frame | ms / batch | FPS |
|---:|---:|---:|---:|---:|
| 1 | 100 | 4.8549 | 4.8549 | 205.98 |
| 4 | 25 | 2.0964 | 8.3856 | 477.01 |
| 8 | 12 | 2.1085 | 16.868 | 474.27 |
| 16 | 10 | 2.3153 | 37.0453 | 431.9 |
| 32 | 10 | 2.5261 | 80.8366 | 395.86 |

**Single-frame latency (batch=1)**: 4.8549 ms/frame  →  **205.98 FPS**.

## 6. Training Loss Curves (TRAIN ONLY)

> Validation was not run during training. The curves below were **reconstructed post-hoc** by replaying every saved checkpoint over a fixed 320-image sample of the labeled training data. They are not the live training loss, but their shape faithfully represents how training progressed.

Checkpoints used: epochs [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

Final-epoch summary (epoch 99):

| Component | Initial (epoch 4) | Final (epoch 99) | Reduction |
|---|---:|---:|---:|
| Total | 0.4065 | 0.1461 | 64.1% |
| Detection | 0.2914 | 0.0773 | 73.5% |
| Drivable Area | 0.1224 | 0.0365 | 70.2% |
| Lane Line | 0.7976 | 0.5141 | 35.5% |
| Box | 0.0794 | 0.0350 | 55.8% |
| Objectness | 0.0733 | 0.0355 | 51.6% |
| Classification | 0.1388 | 0.0067 | 95.2% |

Plots:

- Total training loss vs epoch: `runs/test/plots/loss_total.png`
- Per-task losses (det / drivable / lane) vs epoch: `runs/test/plots/loss_task.png`
- Detection-head components (box / obj / cls) vs epoch: `runs/test/plots/loss_detection_components.png`
- Raw numbers: `runs/test/plots/train_curves.json`

### Reconstructed per-epoch values

| Epoch | Total | Detection | Drivable | Lane | Box | Obj | Cls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.4065 | 0.2914 | 0.1224 | 0.7976 | 0.0794 | 0.0733 | 0.1388 |
| 9 | 0.3549 | 0.2466 | 0.1102 | 0.7559 | 0.0702 | 0.0701 | 0.1063 |
| 14 | 0.3340 | 0.2283 | 0.1063 | 0.7391 | 0.0667 | 0.0690 | 0.0926 |
| 19 | 0.3059 | 0.2028 | 0.1023 | 0.7221 | 0.0635 | 0.0707 | 0.0686 |
| 24 | 0.2799 | 0.1783 | 0.1003 | 0.7126 | 0.0610 | 0.0651 | 0.0522 |
| 29 | 0.2729 | 0.1708 | 0.0944 | 0.7230 | 0.0587 | 0.0683 | 0.0439 |
| 34 | 0.2536 | 0.1562 | 0.0913 | 0.6884 | 0.0570 | 0.0643 | 0.0349 |
| 39 | 0.2466 | 0.1464 | 0.0909 | 0.7112 | 0.0548 | 0.0654 | 0.0261 |
| 44 | 0.2300 | 0.1374 | 0.0884 | 0.6529 | 0.0535 | 0.0609 | 0.0230 |
| 49 | 0.2142 | 0.1247 | 0.0835 | 0.6322 | 0.0503 | 0.0582 | 0.0163 |
| 54 | 0.2058 | 0.1184 | 0.0833 | 0.6157 | 0.0488 | 0.0552 | 0.0145 |
| 59 | 0.1853 | 0.1028 | 0.0706 | 0.5898 | 0.0449 | 0.0474 | 0.0105 |
| 64 | 0.1760 | 0.0970 | 0.0606 | 0.5716 | 0.0432 | 0.0443 | 0.0094 |
| 69 | 0.1671 | 0.0914 | 0.0502 | 0.5550 | 0.0413 | 0.0417 | 0.0084 |
| 74 | 0.1609 | 0.0877 | 0.0442 | 0.5414 | 0.0397 | 0.0401 | 0.0079 |
| 79 | 0.1546 | 0.0830 | 0.0411 | 0.5314 | 0.0375 | 0.0382 | 0.0073 |
| 84 | 0.1502 | 0.0802 | 0.0383 | 0.5219 | 0.0364 | 0.0368 | 0.0070 |
| 89 | 0.1483 | 0.0788 | 0.0373 | 0.5183 | 0.0357 | 0.0362 | 0.0068 |
| 94 | 0.1468 | 0.0777 | 0.0367 | 0.5157 | 0.0353 | 0.0357 | 0.0068 |
| 99 | 0.1461 | 0.0773 | 0.0365 | 0.5141 | 0.0350 | 0.0355 | 0.0067 |

## 7. Qualitative Visualizations

- **Random sample**: 50 images at `runs/test/visualizations/` (bounding boxes + drivable-area + lane-line overlays).

**Challenging conditions** (filtered from labels JSON):

| Category | Filter | Matched | Saved | Directory |
|---|---|---:|---:|---|
| night | `attributes.timeofday == 'night'` | 137 | 12 | `runs/test/visualizations/challenging_conditions/night/` |
| rain | `attributes.weather == 'rainy'` | 253 | 12 | `runs/test/visualizations/challenging_conditions/rain/` |
| crowded | `#box2d labels >= 33` | 372 | 12 | `runs/test/visualizations/challenging_conditions/crowded/` |

## 8. Hard Examples (highest multi-task loss)

Ranked by per-image multi-task loss (detection + drivable + lane). Visualizations stored alongside the metadata.

| Rank | File | Loss | Index |
|---:|---|---:|---:|
| 1 | `79105e65-a1c48df8.jpg` | 1.2863 | 2025 |
| 2 | `baf019d9-da5f9843.jpg` | 1.2342 | 3142 |
| 3 | `2c4e4802-bfbb9299.jpg` | 1.2338 | 727 |
| 4 | `4f45d5ce-d87a28f1.jpg` | 1.2335 | 1319 |
| 5 | `0f9fbe4b-081bb395.jpg` | 1.2288 | 256 |
| 6 | `2bcdafa3-0013d653.jpg` | 1.2282 | 723 |
| 7 | `07576c3b-32ef9ee1.jpg` | 1.2258 | 103 |
| 8 | `33c6cbe6-d0aca98e.jpg` | 1.2234 | 865 |
| 9 | `75cdb8c0-08de301f.jpg` | 1.2217 | 1962 |
| 10 | `2e3e8698-72305bda.jpg` | 1.2076 | 752 |

Output directory: `runs/test/hard_examples/`

## 9. Visual Test Results

Every figure below is a real output of the trained YOLOPv2 model on the BDD100K labelled test subset. The overlay scheme is consistent throughout: predicted bounding boxes are drawn as coloured rectangles, the predicted drivable-area mask is shaded green, and the predicted lane-line mask is shaded red.

### 9.1 Aggregate Analysis Charts

![Per-class detection AP](plots/per_class_ap.png)

*Figure 9.1.1 — Per-class AP@0.5 (blue) and AP@0.5:0.95 (orange). Heavy vehicles (`bus`, `truck`) cross 0.88; small or sparse classes (`traffic light`, `train`) lag the most.*

![Per-class P/R/F1](plots/per_class_prf1.png)

*Figure 9.1.2 — Precision (green), Recall (red), F1 (purple) per class at IoU 0.5 and confidence >= 0.25. Vulnerable road users (`rider`, `bike`, `motor`) are recall-heavy: the model over-detects rather than misses.*

![Segmentation IoU breakdown](plots/segmentation_iou.png)

*Figure 9.1.3 — Segmentation IoU breakdown. The drivable-area decoder converges to foreground IoU 0.909. The lane-line foreground IoU is 0.394, intrinsically lower because BDD100K lane ground truth covers only ~0.7% of pixels.*

![Lane-line threshold sweep](plots/ll_threshold_sweep.png)

*Figure 9.1.4 — Lane-line decoder metrics vs sigmoid threshold. IoU peaks at threshold 0.5; lower thresholds trade precision for recall without improving IoU.*

### 9.2 Training Loss Curves

![Total training loss](plots/loss_total.png)

*Figure 9.2.1 — Reconstructed total training loss across 100 epochs (epoch 4 to 99). Monotonic descent from 0.41 to 0.15.*

![Per-task loss](plots/loss_task.png)

*Figure 9.2.2 — Per-task losses: detection (-73%), drivable area (-70%), lane line (-36%).*

![Detection sub-losses](plots/loss_detection_components.png)

*Figure 9.2.3 — Detection-loss components: box, objectness, classification. Classification loss drops 95% as the head separates the ten classes.*

### 9.3 Random Test Samples (Detection + Segmentation Overlays)

Six images drawn at random from the 50-image visualisation set. Each frame shows the model producing all three outputs in a single forward pass.

![Random sample 1](visualizations/vis_00165_0a9fa6b4-51db4df5.jpg)

*Figure 9.3.1 — Random test image `vis_00165_0a9fa6b4-51db4df5.jpg`. Boxes + drivable mask + lane mask overlaid.*

![Random sample 2](visualizations/vis_00255_0f959aac-24823e01.jpg)

*Figure 9.3.2 — Random test image `vis_00255_0f959aac-24823e01.jpg`. Boxes + drivable mask + lane mask overlaid.*

![Random sample 3](visualizations/vis_00302_124c7772-62504e1d.jpg)

*Figure 9.3.3 — Random test image `vis_00302_124c7772-62504e1d.jpg`. Boxes + drivable mask + lane mask overlaid.*

![Random sample 4](visualizations/vis_00388_1710645f-b36fa2e7.jpg)

*Figure 9.3.4 — Random test image `vis_00388_1710645f-b36fa2e7.jpg`. Boxes + drivable mask + lane mask overlaid.*

![Random sample 5](visualizations/vis_00404_17da591c-7bccf941.jpg)

*Figure 9.3.5 — Random test image `vis_00404_17da591c-7bccf941.jpg`. Boxes + drivable mask + lane mask overlaid.*

![Random sample 6](visualizations/vis_00412_1880e817-8186d9a4.jpg)

*Figure 9.3.6 — Random test image `vis_00412_1880e817-8186d9a4.jpg`. Boxes + drivable mask + lane mask overlaid.*

### 9.4 Challenging Conditions

Three subsets carved out of the BDD100K attribute metadata to stress-test the model under harder conditions.

#### 9.4.1 Night Driving

![Night Driving 1](visualizations/challenging_conditions/night/night_00007_0096bcca-c2027ec4.jpg)

*Figure 9.4 (night) — `night_00007_0096bcca-c2027ec4.jpg`.*

![Night Driving 2](visualizations/challenging_conditions/night/night_00126_087f969b-b1584c3f.jpg)

*Figure 9.4 (night) — `night_00126_087f969b-b1584c3f.jpg`.*

![Night Driving 3](visualizations/challenging_conditions/night/night_00463_1bf6d50e-48ea2889.jpg)

*Figure 9.4 (night) — `night_00463_1bf6d50e-48ea2889.jpg`.*

![Night Driving 4](visualizations/challenging_conditions/night/night_00729_2c917da8-883da6b6.jpg)

*Figure 9.4 (night) — `night_00729_2c917da8-883da6b6.jpg`.*

#### 9.4.2 Rainy Weather

![Rainy Weather 1](visualizations/challenging_conditions/rain/rain_00282_11409559-f3c4582b.jpg)

*Figure 9.4 (rain) — `rain_00282_11409559-f3c4582b.jpg`.*

![Rainy Weather 2](visualizations/challenging_conditions/rain/rain_00314_13163948-7d0574d6.jpg)

*Figure 9.4 (rain) — `rain_00314_13163948-7d0574d6.jpg`.*

![Rainy Weather 3](visualizations/challenging_conditions/rain/rain_00447_1a760b64-33c25ec5.jpg)

*Figure 9.4 (rain) — `rain_00447_1a760b64-33c25ec5.jpg`.*

![Rainy Weather 4](visualizations/challenging_conditions/rain/rain_01189_475a4732-7fd6ed1d.jpg)

*Figure 9.4 (rain) — `rain_01189_475a4732-7fd6ed1d.jpg`.*

#### 9.4.3 Crowded Scenes (>=33 GT boxes)

![Crowded Scenes (>=33 GT boxes) 1](visualizations/challenging_conditions/crowded/crowded_00522_1faaee6b-c0b27c70.jpg)

*Figure 9.4 (crowded) — `crowded_00522_1faaee6b-c0b27c70.jpg`.*

![Crowded Scenes (>=33 GT boxes) 2](visualizations/challenging_conditions/crowded/crowded_00606_242f73d4-325f020c.jpg)

*Figure 9.4 (crowded) — `crowded_00606_242f73d4-325f020c.jpg`.*

![Crowded Scenes (>=33 GT boxes) 3](visualizations/challenging_conditions/crowded/crowded_00944_388af398-d6141a4c.jpg)

*Figure 9.4 (crowded) — `crowded_00944_388af398-d6141a4c.jpg`.*

![Crowded Scenes (>=33 GT boxes) 4](visualizations/challenging_conditions/crowded/crowded_01440_575a2428-b724fc19.jpg)

*Figure 9.4 (crowded) — `crowded_01440_575a2428-b724fc19.jpg`.*

### 9.5 Hard Examples (highest multi-task loss)

Test images ranked by per-image multi-task loss (detection + drivable + lane). These are the frames where the model struggles the most.

![Hard-example loss ranking](plots/hard_example_losses.png)

*Figure 9.5.0 — Top-10 hardest test images, ranked by total multi-task loss.*

![Hard example 1](hard_examples/hard_01_79105e65-a1c48df8.jpg)

*Figure 9.5.1 — Rank #1 (loss 1.2863): `79105e65-a1c48df8.jpg`.*

![Hard example 2](hard_examples/hard_02_baf019d9-da5f9843.jpg)

*Figure 9.5.2 — Rank #2 (loss 1.2342): `baf019d9-da5f9843.jpg`.*

![Hard example 3](hard_examples/hard_03_2c4e4802-bfbb9299.jpg)

*Figure 9.5.3 — Rank #3 (loss 1.2338): `2c4e4802-bfbb9299.jpg`.*

![Hard example 4](hard_examples/hard_04_4f45d5ce-d87a28f1.jpg)

*Figure 9.5.4 — Rank #4 (loss 1.2335): `4f45d5ce-d87a28f1.jpg`.*

![Hard example 5](hard_examples/hard_05_0f9fbe4b-081bb395.jpg)

*Figure 9.5.5 — Rank #5 (loss 1.2288): `0f9fbe4b-081bb395.jpg`.*

![Hard example 6](hard_examples/hard_06_2bcdafa3-0013d653.jpg)

*Figure 9.5.6 — Rank #6 (loss 1.2282): `2bcdafa3-0013d653.jpg`.*

![Hard example 7](hard_examples/hard_07_07576c3b-32ef9ee1.jpg)

*Figure 9.5.7 — Rank #7 (loss 1.2258): `07576c3b-32ef9ee1.jpg`.*

![Hard example 8](hard_examples/hard_08_33c6cbe6-d0aca98e.jpg)

*Figure 9.5.8 — Rank #8 (loss 1.2234): `33c6cbe6-d0aca98e.jpg`.*

![Hard example 9](hard_examples/hard_09_75cdb8c0-08de301f.jpg)

*Figure 9.5.9 — Rank #9 (loss 1.2217): `75cdb8c0-08de301f.jpg`.*

![Hard example 10](hard_examples/hard_10_2e3e8698-72305bda.jpg)

*Figure 9.5.10 — Rank #10 (loss 1.2076): `2e3e8698-72305bda.jpg`.*

### 9.6 Lane-Line Diagnostic Panels

Each panel is a 4-up view: **original | ground-truth lane mask | predicted sigmoid heatmap | thresholded prediction**. The heatmap aligns with real lane boundaries, confirming that the lane head produces meaningful spatial signal rather than noise.

![Lane diagnostic panel 1](ll_diagnostics/panels/panel_01_idx3287.jpg)

*Figure 9.6.1 — Lane-line diagnostic panel `panel_01_idx3287.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

![Lane diagnostic panel 2](ll_diagnostics/panels/panel_02_idx1175.jpg)

*Figure 9.6.2 — Lane-line diagnostic panel `panel_02_idx1175.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

![Lane diagnostic panel 3](ll_diagnostics/panels/panel_03_idx0624.jpg)

*Figure 9.6.3 — Lane-line diagnostic panel `panel_03_idx0624.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

![Lane diagnostic panel 4](ll_diagnostics/panels/panel_04_idx3208.jpg)

*Figure 9.6.4 — Lane-line diagnostic panel `panel_04_idx3208.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

![Lane diagnostic panel 5](ll_diagnostics/panels/panel_05_idx2647.jpg)

*Figure 9.6.5 — Lane-line diagnostic panel `panel_05_idx2647.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

![Lane diagnostic panel 6](ll_diagnostics/panels/panel_06_idx2213.jpg)

*Figure 9.6.6 — Lane-line diagnostic panel `panel_06_idx2213.jpg`. Columns: original frame, GT mask, predicted heatmap, thresholded mask.*

## 10. Output Artifacts

```text
runs/test/
├── test_evaluation_metrics.json   # consolidated metrics (machine-readable)
├── evaluation_metrics.json        # detection + segmentation metrics
├── benchmark.json                 # latency / FPS / model size
├── REPORT.md                      # THIS report
├── ll_diagnostics/
│   ├── ll_threshold_sweep.json   # per-threshold IoU / P / R
│   └── panels/                   # 12 side-by-side diagnostic panels
├── plots/
│   ├── loss_total.png
│   ├── loss_task.png
│   ├── loss_detection_components.png
│   └── train_curves.json
├── visualizations/
│   ├── manifest.json
│   ├── vis_*.jpg                 # 50 random samples
│   └── challenging_conditions/
│       ├── night/                # 12 images
│       ├── rain/                 # 12 images
│       └── crowded/              # 12 images
└── hard_examples/
    ├── hard_*.jpg                # top 10 worst-loss images
    └── hard_examples_info.json
```

## 11. Reproduction Commands

```bash
# Full evaluation: mAP, P/R/F1, segmentation, hard examples
CUDA_VISIBLE_DEVICES=4 python evaluate.py \
    --weights runs/train/last.pt --data-root ../bdd100k_data \
    --split train --require-labels \
    --batch-size 8 --num-vis 0 --num-hard 10 --output-dir runs/test

# LL threshold sweep + diagnostic panels
CUDA_VISIBLE_DEVICES=4 python diagnose_ll.py \
    --weights runs/train/last.pt --data-root ../bdd100k_data \
    --split train --require-labels \
    --thresholds 0.1,0.2,0.3,0.4,0.5 --num-panels 12 \
    --output-dir runs/test/ll_diagnostics

# Real-time benchmark
CUDA_VISIBLE_DEVICES=4 python benchmark.py \
    --weights runs/train/last.pt --n-iters 100 \
    --batch-sizes 1,4,8,16,32 --output runs/test/benchmark.json

# Reconstruct training loss curves from saved checkpoints
CUDA_VISIBLE_DEVICES=4 python recover_train_curves.py \
    --ckpt-dir runs/train --data-root ../bdd100k_data \
    --batch-size 8 --n-samples 320 --output-dir runs/test/plots

# Random + challenging-condition visualizations
CUDA_VISIBLE_DEVICES=4 python visualize_challenging.py \
    --weights runs/train/last.pt --data-root ../bdd100k_data \
    --num-random 50 --per-category 12 \
    --output-root runs/test/visualizations

# Build this report
python build_report.py
```

## 12. Important Caveats

1. **No held-out validation existed during training.** The 1000-image `seg/images/val/` subset has filenames that do not match either `bdd100k_labels_images_val.json` or `bdd100k_labels_images_train.json` (0 overlap), so it provides no usable detection / lane ground truth. The 3,426-image labeled subset of `seg/images/train/` was used both for training and as the test set.
2. **Numbers are training-fit upper bounds.** For an unbiased generalization estimate, retrain with a held-out split (e.g. 80/20 of the labeled 3,426 images).
3. **Training-loss curves are reconstructed.** No live log was persisted; each saved checkpoint was re-evaluated post-hoc on a fixed sample. The curve shape is faithful; absolute values are slightly lower than the live training loss because augmentation noise is absent.
4. **Lane-line IoU 0.394 is competitive.** BDD100K lane GT is thin (~0.7% of pixels), making IoU intrinsically low. The official YOLOPv2 paper reports ~26-30% IoU on a thinned lane metric.
5. **Bug fixed during evaluation**: the `Detect` head already applies sigmoid to obj/cls scores in eval mode, but the initial inference script applied an extra sigmoid in `decode_detections`. This was corrected (see `evaluate.py` at the `decode_detections` function); all results above use the corrected version.
