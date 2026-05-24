"""
YOLOPv2 Model Architecture
Multi-task network for:
  1. Traffic object detection (YOLO head)
  2. Drivable area segmentation
  3. Lane line segmentation

Architecture based on YOLOPv2 paper: E-ELAN backbone + multi-task heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv: Conv2d + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck: 1x1 conv -> 3x3 conv with residual"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block (YOLOv7 / YOLOPv2 style)"""
    def __init__(self, c1, c2, c3, n=1):
        """
        c1: input channels
        c2: intermediate channels
        c3: output channels
        n: number of bottleneck repeats per branch
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(
                nn.Sequential(*[Conv(c2, c2, 3, 1) for _ in range(n)])
            )
        self.cv3 = Conv(c2 * 4, c3, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        outs = [x1, x2]
        prev = x2
        for blk in self.blocks:
            prev = blk(prev)
            outs.append(prev)
        return self.cv3(torch.cat(outs, dim=1))


class MP(nn.Module):
    """MaxPool down-sample block (like YOLOv7)"""
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.mp = nn.MaxPool2d(2, 2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)

    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv3(self.cv2(x))
        return torch.cat([x1, x2], dim=1)


class SPPCSPC(nn.Module):
    """SPP-CSPC: Spatial Pyramid Pooling - Cross Stage Partial Connections"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y = [x1]
        y.extend(m(x1) for m in self.m)
        y = self.cv6(self.cv5(torch.cat(y, 1)))
        return self.cv7(torch.cat([y, self.cv2(x)], 1))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class CSPDarknetBackbone(nn.Module):
    """
    CSPDarknet-style backbone producing 3 feature maps at strides 8, 16, 32.
    Channel widths scaled by width_mult.
    """
    def __init__(self, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        base = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width_mult), 16) for c in base]
        n = [max(round(3 * depth_mult), 1) for _ in range(4)]

        # stem
        self.stem = Conv(3, ch[0], 6, 2, 2)

        # stage1: stride 4
        self.stage1 = nn.Sequential(Conv(ch[0], ch[1], 3, 2), C3(ch[1], ch[1], n[0]))
        # stage2: stride 8
        self.stage2 = nn.Sequential(Conv(ch[1], ch[2], 3, 2), C3(ch[2], ch[2], n[1]))
        # stage3: stride 16
        self.stage3 = nn.Sequential(Conv(ch[2], ch[3], 3, 2), C3(ch[3], ch[3], n[2]))
        # stage4: stride 32
        self.stage4 = nn.Sequential(Conv(ch[3], ch[4], 3, 2), C3(ch[4], ch[4], n[3]))

        self.spp = SPPCSPC(ch[4], ch[4])

        self.out_channels = [ch[2], ch[3], ch[4]]  # stride 8, 16, 32

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)   # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        c5 = self.spp(c5)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# Neck (FPN + PAN)
# ---------------------------------------------------------------------------

class FPN_PAN(nn.Module):
    """Feature Pyramid Network + Path Aggregation Network"""
    def __init__(self, in_channels, depth_mult=1.0):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels
        n = max(round(3 * depth_mult), 1)

        # Top-down (FPN)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.fpn_conv1 = Conv(c5_ch, c4_ch, 1, 1)
        self.fpn_c3_1 = C3(c4_ch * 2, c4_ch, n, shortcut=False)

        self.fpn_conv2 = Conv(c4_ch, c3_ch, 1, 1)
        self.fpn_c3_2 = C3(c3_ch * 2, c3_ch, n, shortcut=False)

        # Bottom-up (PAN)
        self.pan_down1 = Conv(c3_ch, c3_ch, 3, 2)
        self.pan_c3_1 = C3(c3_ch + c4_ch, c4_ch, n, shortcut=False)

        self.pan_down2 = Conv(c4_ch, c4_ch, 3, 2)
        self.pan_c3_2 = C3(c4_ch + c5_ch, c5_ch, n, shortcut=False)

        self.out_channels = [c3_ch, c4_ch, c5_ch]

    def forward(self, c3, c4, c5):
        # FPN top-down
        p5 = c5
        p4 = self.fpn_c3_1(torch.cat([self.up(self.fpn_conv1(p5)), c4], 1))
        p3 = self.fpn_c3_2(torch.cat([self.up(self.fpn_conv2(p4)), c3], 1))

        # PAN bottom-up
        n3 = p3
        n4 = self.pan_c3_1(torch.cat([self.pan_down1(n3), p4], 1))
        n5 = self.pan_c3_2(torch.cat([self.pan_down2(n4), p5], 1))

        return n3, n4, n5


# ---------------------------------------------------------------------------
# Detection Head (YOLO-style)
# ---------------------------------------------------------------------------

class Detect(nn.Module):
    """YOLOv5-style detection head"""
    stride = None

    def __init__(self, nc=10, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # outputs per anchor
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


# ---------------------------------------------------------------------------
# Segmentation Heads
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Lightweight segmentation head with upsampling."""
    def __init__(self, in_channels_list, num_classes=2):
        super().__init__()
        c = 64
        # Process each FPN level
        self.conv_s8 = nn.Sequential(Conv(in_channels_list[0], c, 1, 1), Conv(c, c, 3, 1))
        self.conv_s16 = nn.Sequential(Conv(in_channels_list[1], c, 1, 1), Conv(c, c, 3, 1))
        self.conv_s32 = nn.Sequential(Conv(in_channels_list[2], c, 1, 1), Conv(c, c, 3, 1))

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.fuse = nn.Sequential(
            Conv(c * 3, c, 3, 1),
            Conv(c, c, 3, 1),
        )
        self.up_final = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def forward(self, features):
        p3, p4, p5 = features
        s8 = self.conv_s8(p3)
        s16 = self.up2(self.conv_s16(p4))
        s32 = self.up4(self.conv_s32(p5))
        fused = self.fuse(torch.cat([s8, s16, s32], dim=1))
        out = self.classifier(self.up_final(fused))
        return out


# ---------------------------------------------------------------------------
# Full YOLOPv2 Model
# ---------------------------------------------------------------------------

class YOLOPv2(nn.Module):
    """
    Multi-task panoptic driving perception model.
    Tasks:
        1. Object detection (vehicles, pedestrians, etc.)
        2. Drivable area segmentation
        3. Lane line segmentation
    """

    # BDD100K detection classes
    DET_CLASSES = ['car', 'bus', 'truck', 'person', 'rider', 'bike',
                   'motor', 'traffic light', 'traffic sign', 'train']

    # Anchors (YOLOv5-m sized, 3 scales)
    ANCHORS = [
        [10, 13, 16, 30, 33, 23],       # P3/8
        [30, 61, 62, 45, 59, 119],       # P4/16
        [116, 90, 156, 198, 373, 326],   # P5/32
    ]

    def __init__(self, nc=10, width_mult=0.75, depth_mult=0.67, anchors=None):
        super().__init__()
        self.nc = nc
        anchors = anchors or self.ANCHORS

        # Backbone
        self.backbone = CSPDarknetBackbone(width_mult=width_mult, depth_mult=depth_mult)
        ch = self.backbone.out_channels

        # Neck
        self.neck = FPN_PAN(ch, depth_mult=depth_mult)
        neck_ch = self.neck.out_channels

        # Detection head
        self.detect = Detect(nc=nc, anchors=anchors, ch=neck_ch)

        # Segmentation heads
        self.da_seg_head = SegmentationHead(neck_ch, num_classes=2)  # drivable area: 2 classes
        self.ll_seg_head = SegmentationHead(neck_ch, num_classes=1)  # lane line: binary

        # Initialize
        self._init_detect_biases()
        self._init_weights()

        # Compute strides
        self._build_strides()

    def _build_strides(self):
        m = self.detect
        s = 256
        dummy = torch.zeros(1, 3, s, s)
        with torch.no_grad():
            c3, c4, c5 = self.backbone(dummy)
            n3, n4, n5 = self.neck(c3, c4, c5)
        features = [n3, n4, n5]
        m.stride = torch.tensor([s / x.shape[-2] for x in features])
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride

    def _init_detect_biases(self):
        for mi, s in zip(self.detect.m, self.detect.stride or [8, 16, 32]):
            b = mi.bias.view(self.detect.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        # Backbone
        c3, c4, c5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(c3, c4, c5)

        # Detection
        det_out = self.detect([n3, n4, n5])

        # Segmentation
        features = (n3, n4, n5)
        da_seg = self.da_seg_head(features)
        ll_seg = self.ll_seg_head(features)

        if self.training:
            # det_out is list of [bs,na,ny,nx,no] — concat into single tensor for DataParallel
            # Flatten each to [bs, -1, no] then concat on dim=1
            flat = [d.view(d.shape[0], -1, d.shape[-1]) for d in det_out]
            det_concat = torch.cat(flat, dim=1)  # [bs, total_anchors, no]
            # Store split sizes so loss can reconstruct
            self._det_split_sizes = [d.shape[0] * d.shape[1] * d.shape[2] * d.shape[3] // d.shape[0] // d.shape[-1]
                                     for d in det_out]  # na*ny*nx per level
            self._det_shapes = [d.shape for d in det_out]
            return det_concat, da_seg, ll_seg
        else:
            return det_out, da_seg, ll_seg


def build_model(nc=10, width_mult=0.75, depth_mult=0.67):
    """Factory function to build YOLOPv2."""
    model = YOLOPv2(nc=nc, width_mult=width_mult, depth_mult=depth_mult)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"YOLOPv2 created: {n_params / 1e6:.1f}M parameters")
    return model
