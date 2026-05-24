# YOLOPv2: A Multi-Task Deep Learning Framework for Panoptic Driving Perception

<p align="center">
  <em>M.Tech Dissertation</em><br>
  <strong>Indian Institute of Technology Roorkee</strong>
</p>

---

## Project Overview / Abstract

Autonomous driving systems must simultaneously understand the surrounding environment in real time — detecting traffic objects, identifying drivable regions on the road, and recognising lane markings. Traditionally these tasks are handled by separate networks, which is computationally expensive and difficult to deploy on embedded automotive hardware.

This project implements **YOLOPv2**, a unified end-to-end deep learning framework that performs **traffic object detection**, **drivable area segmentation**, and **lane line detection** within a single forward pass. By sharing a common feature-extraction backbone across the three tasks, YOLOPv2 achieves real-time inference speed while maintaining accuracy that is competitive with — and in several aspects superior to — task-specific state-of-the-art models. The framework is trained and evaluated on the large-scale **BDD100K** driving dataset and is designed to support reproducible thesis-grade experimentation, with all training and evaluation outputs stored under the `runs/` directory.

---

## Objectives of the Project

1. To design and implement a **single multi-task deep neural network** capable of jointly performing traffic object detection, drivable area segmentation, and lane line detection.
2. To train the network on the **BDD100K** dataset under realistic and diverse driving conditions (day/night, urban/highway, varied weather).
3. To evaluate the model on standard quantitative metrics — **mAP, Recall, mIoU, Accuracy, Lane IoU, FPS** — and benchmark it against existing baselines.
4. To analyse failure cases and challenging scenarios (poor lighting, occluded lanes, dense traffic) through diagnostic visualisations.
5. To produce a **well-documented evaluation pipeline** that records detection, segmentation, and lane-line metrics alongside qualitative visualisations for thesis-level analysis.

---

## Dataset Used

The model is trained and evaluated on the **BDD100K** dataset — one of the largest publicly available driving datasets, containing **100,000 high-resolution images** (1280×720) collected across multiple cities, weather conditions, and times of day.

### Annotations used

| Task | Annotation type |
|------|-----------------|
| Traffic Object Detection | 2D bounding boxes (vehicles, pedestrians, traffic signs, etc.) |
| Drivable Area Segmentation | Pixel-wise binary masks (drivable vs non-drivable) |
| Lane Line Detection | Pixel-wise lane line masks |

### Data split

| Split | Number of images |
|-------|:----------------:|
| Training | 70,000 |
| Validation | 10,000 |
| Test | 20,000 |

### Download links

The dataset is **not included** in this repository because of its large size (~7 GB+). It can be obtained from:

- Kaggle: <https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k>

After downloading, the data should be placed inside `data/bdd100k/` following the standard BDD100K subfolder convention (images, detection annotations, drivable-area masks, and lane-line masks for `train`, `val`, and `test` splits).

---

## Model / Methodology

YOLOPv2 is a **multi-task convolutional neural network** built around the principle of *one shared encoder, three specialised decoders*.

### 1. Shared Backbone (Encoder)

A high-capacity feature extractor based on the **E-ELAN (Extended Efficient Layer Aggregation Network)** structure is used as the backbone. It produces a hierarchical feature pyramid that is consumed by all three downstream task heads.

### 2. Task-Specific Heads (Decoders)

- **Detection Head** — anchor-based head producing bounding boxes, objectness, and class scores. Trained with a combined classification, localisation, and objectness loss.
- **Drivable Area Segmentation Head** — a lightweight upsampling decoder producing a binary segmentation mask, trained with cross-entropy loss.
- **Lane Line Segmentation Head** — a separate upsampling branch dedicated to thin-structure lane segmentation, trained with a combined cross-entropy and IoU-based loss to handle severe class imbalance.

### 3. Training Strategy

- **Bag-of-Freebies** training tricks: mosaic augmentation, mixup, colour jitter, random affine transforms, label smoothing, EMA (Exponential Moving Average) of weights.
- **Joint multi-task loss** with task-balanced weighting.
- **Cosine learning-rate schedule** with warmup.
- Optimiser: SGD with momentum.

### 4. Inference

A single forward pass produces all three task outputs simultaneously, enabling real-time deployment.

---

## Technologies & Libraries Used

| Category | Tools / Libraries |
|----------|-------------------|
| Programming language | Python 3.8+ |
| Deep learning framework | PyTorch, TorchVision |
| Computer vision | OpenCV, Pillow |
| Numerical computation | NumPy, SciPy |
| Data handling | Pandas, PyYAML |
| Visualisation | Matplotlib, Seaborn |
| Logging & monitoring | TensorBoard, tqdm |

---

## Installation & Requirements

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 6 GB VRAM | NVIDIA RTX 3090 / Tesla V100 (≥16 GB) |
| RAM | 8 GB | 16 GB+ |
| Disk space | 20 GB | 50 GB+ (for full BDD100K) |
| OS | Windows 10 / Ubuntu 18.04+ | Ubuntu 20.04 |

---

## How to Run the Project

### 1. Inference on an image or video

```bash
python demo.py --source data/example.jpg --weights data/weights/yolopv2.pt
```

Common arguments:

| Flag | Description |
|------|-------------|
| `--source` | Path to image, video file, or folder |
| `--weights` | Path to model weights |
| `--img-size` | Input resolution (default 640) |
| `--conf-thres` | Detection confidence threshold |
| `--device` | `cpu` or GPU index (e.g. `0`) |

### 2. Train the model

```bash
python train.py --data data/bdd100k --weights data/weights/yolopv2.pt --epochs 100
```

Outputs (checkpoints, loss/metric logs, plots) are saved under `runs/train/`.

### 3. Evaluate the model

```bash
python evaluate.py --weights data/weights/yolopv2.pt --data data/bdd100k
```

Per-task metrics, JSON summaries, and plots are written to `runs/test/`.

### 4. Benchmark inference speed

```bash
python benchmark.py --weights data/weights/yolopv2.pt --device 0
```

---

## Results & Performance Metrics

All results below are obtained from the actual experiments stored under `runs/test/` of this project. The model was evaluated on **3,426 BDD100K images** at **640×640** resolution with `conf_threshold = 0.25` and `iou_threshold = 0.45`. Inference benchmarks were performed on an **NVIDIA RTX 6000 Ada Generation** GPU.

### Model size and inference speed

| Metric | Value |
|--------|:-----:|
| Total parameters | **23.40 M** |
| Checkpoint size (FP32) | 187.6 MB |
| Memory footprint (params + buffers, FP32) | 93.8 MB |
| Inference latency (batch = 1) | **4.85 ms / frame** |
| Throughput (batch = 1) | **206.0 FPS** |
| Peak throughput (batch = 4) | **477.0 FPS** |
| End-to-end evaluation speed (with data I/O) | 50.1 FPS |

### Traffic Object Detection — Overall

| Metric | Value |
|--------|:-----:|
| mAP@0.5 | **0.6763** |
| mAP@0.5:0.95 | 0.4039 |
| Vehicle-only mAP@0.5 | 0.8273 |
| Vehicle-only mAP@0.5:0.95 | 0.5923 |
| Overall Precision | 0.7391 |
| Overall Recall | 0.6839 |
| Overall F1-score | **0.7104** |
| Total ground-truth boxes | 67,622 |

### Traffic Object Detection — Per-class AP

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 |
|-------|:------:|:-----------:|:---------:|:------:|:--:|
| car           | 0.7104 | 0.4812 | 0.7786 | 0.7254 | 0.7510 |
| bus           | **0.8813** | 0.6410 | 0.7405 | 0.8934 | 0.8098 |
| truck         | **0.8904** | **0.6547** | 0.7709 | 0.8980 | **0.8296** |
| person        | 0.6172 | 0.3089 | 0.6214 | 0.6590 | 0.6396 |
| rider         | 0.7065 | 0.3618 | 0.5816 | 0.7525 | 0.6561 |
| bike          | 0.6998 | 0.3650 | 0.5871 | 0.7484 | 0.6580 |
| motor         | 0.7653 | 0.4632 | 0.6353 | 0.8009 | 0.7086 |
| traffic light | 0.5032 | 0.2053 | 0.6528 | 0.5586 | 0.6020 |
| traffic sign  | 0.5847 | 0.3087 | 0.7489 | 0.6076 | 0.6709 |
| train         | 0.4048 | 0.2492 | 0.5556 | 0.4545 | 0.5000 |

### Drivable Area Segmentation

| Metric | Value |
|--------|:-----:|
| Mean IoU | **0.9472** |
| Pixel Accuracy | 0.9873 |
| Background IoU | 0.9854 |
| Drivable IoU | 0.9090 |

### Lane Line Segmentation

| Metric | Value |
|--------|:-----:|
| Mean IoU | 0.6932 |
| Pixel Accuracy | **0.9923** |
| Background IoU | 0.9923 |
| Lane IoU | 0.3942 |

### Key takeaways

- The model attains a strong **mAP@0.5 of 0.6763** across ten BDD100K object classes.
- Large vehicles (**bus**, **truck**) reach **>0.88 AP@0.5**, demonstrating high reliability on safety-critical objects.
- Drivable area segmentation achieves an excellent **mIoU of 0.9472** with pixel accuracy of **98.73 %**.
- Lane line segmentation reaches **99.23 % pixel accuracy**; the lower lane IoU (0.3942) reflects the inherent difficulty of thin-structure segmentation.
- The network is **lightweight (23.4 M parameters)** yet runs at over **200 FPS** at batch size 1 and peaks at **477 FPS** at batch size 4, making it well-suited for real-time deployment.

---

## Research Contribution / Novelty

This project contributes the following:

1. **Unified panoptic perception in real time.** Demonstrates that the three core perception tasks for autonomous driving can be solved jointly in a single forward pass at over **200 FPS** on a modern GPU, removing the need for separate task-specific networks.
2. **Shared encoder, specialised decoders.** Shows that a carefully designed shared E-ELAN backbone produces representations that generalise simultaneously to detection and two segmentation tasks without significant accuracy loss.
3. **Lane-aware multi-task training.** Introduces a tailored loss combination for the lane line head that compensates for the extreme foreground/background imbalance characteristic of lane segmentation.
4. **Empirical benchmarking under challenging conditions.** Includes diagnostic tools (`diagnose_ll.py`, `visualize_challenging.py`) that systematically expose model behaviour in night-time, occluded, and adverse-weather scenes.
5. **Transparent quantitative analysis.** Detection, drivable-area, and lane-line metrics are stored as machine-readable JSON under `runs/test/`, enabling fully reproducible thesis-grade reporting.

---

## Future Scope

- **3D Perception.** Extend the framework to predict 3D bounding boxes and depth, enabling integration with LiDAR-camera fusion pipelines.
- **Temporal modelling.** Add a recurrent or transformer-based temporal module to exploit video context for stable lane tracking.
- **Domain adaptation.** Adapt the trained model to other geographies (e.g. Indian road conditions) using unsupervised domain adaptation techniques.
- **Edge deployment.** Quantise and prune the network for deployment on automotive-grade embedded hardware (NVIDIA Jetson, Qualcomm Snapdragon Ride).
- **Additional perception tasks.** Incorporate traffic sign classification, pedestrian intention estimation, and traffic light state recognition under the same multi-task umbrella.
- **Self-supervised pre-training.** Investigate self-supervised representation learning on unlabelled driving footage to further reduce annotation cost.

