"""
Multi-task loss for YOLOPv2 training.
  1. Detection loss (YOLO-style: box + obj + cls)
  2. Drivable area segmentation loss (Cross-Entropy)
  3. Lane line segmentation loss (Binary Cross-Entropy + Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    """Compute IoU (and variants) between box1 and box2."""
    if not x1y1x2y2:
        # xywh -> xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - (rho2 / c2 + v * alpha)
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    return iou


# ---------------------------------------------------------------------------
# YOLO Detection Loss (build targets + compute loss)
# ---------------------------------------------------------------------------

class YOLODetectionLoss(nn.Module):
    def __init__(self, model, nc=10):
        super().__init__()
        det = model.detect
        self.nc = nc
        self.nl = det.nl
        self.na = det.na
        self.no = nc + 5
        self.anchors = det.anchors  # [nl, na, 2] already divided by stride
        self.stride = model.stride

        # Compute grid sizes from strides (assumes 640x640 input)
        img_size = 640
        self._grid_sizes = [(img_size // int(s), img_size // int(s)) for s in self.stride]

        self.balance = [4.0, 1.0, 0.4]  # P3, P4, P5 objectness balance
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))

        # Loss gains
        self.box_gain = 0.05
        self.obj_gain = 1.0
        self.cls_gain = 0.5

    def _unflatten_preds(self, det_concat):
        """Reconstruct per-level preds from flat [bs, total_anchors, no] tensor."""
        bs = det_concat.shape[0]
        preds = []
        offset = 0
        for (ny, nx) in self._grid_sizes:
            count = self.na * ny * nx
            p = det_concat[:, offset:offset + count, :]
            p = p.view(bs, self.na, ny, nx, self.no)
            preds.append(p)
            offset += count
        return preds

    def forward(self, det_concat, targets):
        """
        det_concat: [bs, total_anchors, no] flat tensor from model training output
        targets: [N, 6] = [img_idx, cls, cx, cy, w, h] (normalized 0-1)
        """
        preds = self._unflatten_preds(det_concat)
        device = preds[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(preds, targets)

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]

            if n:
                ps = pi[b, a, gj, gi]  # [n, no]
                # Box regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], x1y1x2y2=False, CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # Objectness target
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Class loss
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], 0.0, device=device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= self.box_gain
        lobj *= self.obj_gain
        lcls *= self.cls_gain
        bs = preds[0].shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, preds, targets):
        """Assign targets to anchors for each detection layer."""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets_with_ai = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # [na, nt, 7]

        g = 0.5  # grid offset
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

        for i in range(self.nl):
            anchors_i = self.anchors[i]  # [na, 2]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain = [nx, ny, nx, ny]

            t = targets_with_ai.clone()
            if nt:
                t[..., 2:6] *= gain[2:6]

                # Filter by anchor ratio
                r = t[..., 4:6] / anchors_i[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < 4.0  # ratio threshold
                t = t[j]

                # Grid offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j_left, k_up = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                j_right, k_down = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j_list = torch.stack((torch.ones_like(j_left), j_left, k_up, j_right, k_down))
                t = t.repeat((5, 1, 1))[j_list]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j_list]
            else:
                t = targets_with_ai[0]
                offsets = 0

            # Extract
            if t.shape[0] > 0:
                b, c = t[:, :2].long().T
                gxy = t[:, 2:4]
                gwh = t[:, 4:6]
                gij = (gxy - offsets).long()
                gi, gj = gij.T
                gi = gi.clamp_(0, int(gain[2]) - 1)
                gj = gj.clamp_(0, int(gain[3]) - 1)

                a = t[:, 6].long()
                indices.append((b, a, gj, gi))
                tbox.append(torch.cat((gxy - gij.float(), gwh), 1))
                anch.append(anchors_i[a])
                tcls.append(c)
            else:
                indices.append((torch.zeros(0, dtype=torch.long, device=targets.device),) * 4)
                tbox.append(torch.zeros((0, 4), device=targets.device))
                anch.append(torch.zeros((0, 2), device=targets.device))
                tcls.append(torch.zeros(0, dtype=torch.long, device=targets.device))

        return tcls, tbox, indices, anch


# ---------------------------------------------------------------------------
# Segmentation Losses
# ---------------------------------------------------------------------------

class DrivableAreaLoss(nn.Module):
    """Cross-entropy loss for drivable area segmentation."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, pred, target):
        """
        pred: [B, 2, H, W]
        target: [B, H, W] with values in {0, 1}
        """
        # Resize pred to match target if needed
        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True)
        return self.ce(pred, target)


class LaneLineLoss(nn.Module):
    """BCE + Dice loss for lane line segmentation."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W]
        target: [B, H, W] with values in {0, 1}
        """
        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True)
        pred = pred.squeeze(1)  # [B, H, W]
        bce = self.bce(pred, target)
        dice = self._dice_loss(pred.sigmoid(), target)
        return bce + dice

    @staticmethod
    def _dice_loss(pred, target, smooth=1.0):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# ---------------------------------------------------------------------------
# Combined Multi-Task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Combined loss = det_weight * det_loss + da_weight * da_loss + ll_weight * ll_loss
    """
    def __init__(self, model, nc=10, det_weight=1.0, da_weight=1.0, ll_weight=1.0):
        super().__init__()
        self.det_loss = YOLODetectionLoss(model, nc=nc)
        self.da_loss = DrivableAreaLoss()
        self.ll_loss = LaneLineLoss()
        self.det_weight = det_weight
        self.da_weight = da_weight
        self.ll_weight = ll_weight

    def forward(self, det_preds, da_pred, ll_pred, targets, da_target, ll_target):
        """
        det_preds: [bs, total_anchors, no] flat tensor (training mode) — unflattened internally
        da_pred: [bs, 2, H, W]
        ll_pred: [bs, 1, H, W]
        targets: [N, 6] detection targets
        da_target: [bs, H, W] drivable area mask
        ll_target: [bs, H, W] lane line mask
        """
        l_det, det_items = self.det_loss(det_preds, targets)
        l_da = self.da_loss(da_pred, da_target)
        l_ll = self.ll_loss(ll_pred, ll_target)

        total = self.det_weight * l_det + self.da_weight * l_da + self.ll_weight * l_ll
        return total, l_det.item(), l_da.item(), l_ll.item(), det_items
