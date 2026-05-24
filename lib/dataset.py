"""
BDD100K Dataset for YOLOPv2 multi-task training.
Provides:
  - Images (RGB)
  - Detection labels (bounding boxes + class)
  - Drivable area segmentation masks
  - Lane line segmentation masks (generated from polygon annotations)
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# BDD100K detection categories -> indices
DET_CLASSES = ['car', 'bus', 'truck', 'person', 'rider', 'bike',
               'motor', 'traffic light', 'traffic sign', 'train']
CLS2IDX = {c: i for i, c in enumerate(DET_CLASSES)}

# BDD100K segmentation train_id mapping for drivable area
# In seg labels: 0 = direct drivable, 1 = alternative drivable, 255 = ignore
# We map: background=0, drivable=1


class BDD100KDataset(Dataset):
    """
    Multi-task BDD100K dataset.

    Directory structure expected:
        data_root/
          bdd100k/
            images/100k/{train,val}/  <- full resolution images (1280x720)
            labels/bdd100k_labels_images_{train,val}.json  <- detection + lane + drivable area
            seg/images/{train,val}/  <- segmentation subset images
            seg/labels/{train,val}/*_train_id.png  <- semantic segmentation labels
    """

    def __init__(self, data_root, split='train', img_size=640, augment=True,
                 require_labels=False):
        super().__init__()
        assert split in ('train', 'val')
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.data_root = Path(data_root) / 'bdd100k'

        # --- Paths ---
        self.img_dir = self.data_root / 'images' / '100k' / split
        self.seg_img_dir = self.data_root / 'seg' / 'images' / split
        self.seg_label_dir = self.data_root / 'seg' / 'labels' / split
        # --- Load JSON labels (load ALL splits for full coverage) ---
        print(f"Loading BDD100K {split} labels...")
        self.label_lookup = {}
        for s in ('train', 'val'):
            lf = self.data_root / 'labels' / f'bdd100k_labels_images_{s}.json'
            if lf.exists():
                with open(lf, 'r') as f:
                    for entry in json.load(f):
                        self.label_lookup[entry['name']] = entry

        # Use the segmentation subset; include images even without detection labels
        # unless `require_labels=True`, in which case we keep only those that
        # have an entry in `label_lookup` (i.e. valid det/lane JSON labels).
        seg_images = sorted(os.listdir(self.seg_img_dir))
        self.image_files = []
        n_with_det = 0
        for fname in seg_images:
            seg_label = self.seg_label_dir / f'{Path(fname).stem}_train_id.png'
            if not seg_label.exists():
                continue
            has_det = fname in self.label_lookup
            if require_labels and not has_det:
                continue
            self.image_files.append(fname)
            if has_det:
                n_with_det += 1

        print(f"  Found {len(self.image_files)} seg images, {n_with_det} with det labels "
              f"({split}, require_labels={require_labels})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]

        # --- Load image ---
        img_path = str(self.seg_img_dir / fname)
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to load image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # original size: 720 x 1280

        # --- Load drivable area segmentation ---
        stem = Path(fname).stem
        seg_path = str(self.seg_label_dir / f'{stem}_train_id.png')
        da_seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if da_seg is None:
            da_seg = np.zeros((h0, w0), dtype=np.uint8)
        # Map: 0=direct drivable, 1=alternative drivable -> both become 1 (drivable)
        # Everything else -> 0 (background), 255 -> 0 (ignore becomes background)
        da_mask = np.zeros_like(da_seg, dtype=np.uint8)
        da_mask[(da_seg == 0) | (da_seg == 1)] = 1

        # --- Load lane line labels (from JSON poly2d) ---
        ll_mask = np.zeros((h0, w0), dtype=np.uint8)
        entry = self.label_lookup.get(fname, {'labels': []})
        for label in entry.get('labels', []):
            if label['category'] == 'lane':
                for poly in label.get('poly2d', []):
                    pts = np.array(poly['vertices'], dtype=np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(ll_mask, [pts], isClosed=poly.get('closed', False),
                                      color=1, thickness=4)

        # --- Load detection labels ---
        det_labels = []
        for label in entry.get('labels', []):
            cat = label['category']
            if cat in CLS2IDX and 'box2d' in label:
                box = label['box2d']
                x1 = box['x1'] / w0
                y1 = box['y1'] / h0
                x2 = box['x2'] / w0
                y2 = box['y2'] / h0
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                if bw > 0 and bh > 0:
                    det_labels.append([CLS2IDX[cat], cx, cy, bw, bh])

        det_labels = np.array(det_labels, dtype=np.float32) if det_labels else np.zeros((0, 5), dtype=np.float32)

        # --- Resize ---
        img, da_mask, ll_mask, det_labels = self._resize(img, da_mask, ll_mask, det_labels)

        # --- Augment ---
        if self.augment:
            img, da_mask, ll_mask, det_labels = self._augment(img, da_mask, ll_mask, det_labels)

        # --- To tensors ---
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        da_mask = torch.from_numpy(da_mask).long()
        ll_mask = torch.from_numpy(ll_mask).float()

        # det_labels: [N, 5] with [class, cx, cy, w, h] normalized
        det_labels = torch.from_numpy(det_labels)

        return img, det_labels, da_mask, ll_mask

    def _resize(self, img, da_mask, ll_mask, det_labels):
        """Letterbox resize to self.img_size maintaining aspect ratio."""
        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        new_w, new_h = int(w * r), int(h * r)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        da_mask = cv2.resize(da_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        left = pad_w // 2

        img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        da_mask = cv2.copyMakeBorder(da_mask, top, pad_h - top, left, pad_w - left,
                                     cv2.BORDER_CONSTANT, value=0)
        ll_mask = cv2.copyMakeBorder(ll_mask, top, pad_h - top, left, pad_w - left,
                                     cv2.BORDER_CONSTANT, value=0)

        # Adjust det_labels (they are already normalized to 0-1 of original image)
        if len(det_labels) > 0:
            # Convert from image-normalized to pixel coords in new size
            det_labels = det_labels.copy()
            # Scale and shift
            det_labels[:, 1] = det_labels[:, 1] * new_w / self.img_size + left / self.img_size  # cx
            det_labels[:, 2] = det_labels[:, 2] * new_h / self.img_size + top / self.img_size    # cy
            det_labels[:, 3] = det_labels[:, 3] * new_w / self.img_size  # w
            det_labels[:, 4] = det_labels[:, 4] * new_h / self.img_size  # h

        return img, da_mask, ll_mask, det_labels

    def _augment(self, img, da_mask, ll_mask, det_labels):
        """Simple data augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            da_mask = np.fliplr(da_mask).copy()
            ll_mask = np.fliplr(ll_mask).copy()
            if len(det_labels) > 0:
                det_labels[:, 1] = 1.0 - det_labels[:, 1]

        # Color jitter
        if random.random() > 0.5:
            img = img.astype(np.float32)
            img *= random.uniform(0.7, 1.3)
            img = np.clip(img, 0, 255).astype(np.uint8)

        # HSV augmentation
        if random.random() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img, da_mask, ll_mask, det_labels


def collate_fn(batch):
    """
    Custom collate: detection labels have variable length per image,
    so we prepend an image index to each label row.
    """
    imgs, det_labels, da_masks, ll_masks = zip(*batch)
    imgs = torch.stack(imgs, 0)
    da_masks = torch.stack(da_masks, 0)
    ll_masks = torch.stack(ll_masks, 0)

    # Prepend image index to detection labels
    targets = []
    for i, labels in enumerate(det_labels):
        if len(labels) > 0:
            idx_col = torch.full((len(labels), 1), i, dtype=torch.float32)
            targets.append(torch.cat([idx_col, labels], dim=1))
    if targets:
        targets = torch.cat(targets, 0)  # [N_total, 6]: img_idx, cls, cx, cy, w, h
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, targets, da_masks, ll_masks
