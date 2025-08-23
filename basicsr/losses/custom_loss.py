import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY, ARCH_REGISTRY

# ============================================================
# Utilities
# ============================================================

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Convert Bx3xHxW RGB to Bx1xHxW grayscale in [0,1] (differentiable)."""
    if x.shape[1] == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class CharbonnierLoss(nn.Module):
    """Robust L1 (a.k.a. pseudo-Huber). Stable for SEM edges/textures."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, use_norm=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=not use_norm)]
        if use_norm:
            layers += [nn.GroupNorm(8, out_ch)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# Learnable Edge Extractor + EMA
# ============================================================

@ARCH_REGISTRY.register()
class LearnableEdgeExtractor(nn.Module):
    """Small CNN that learns edge maps robustly for SEM (grayscale-friendly)."""
    def __init__(self, in_ch: int = 1, base_ch: int = 32, use_norm: bool = False):
        super().__init__()
        def cb(i, o):
            layers = [nn.Conv2d(i, o, 3, 1, 1, bias=not use_norm)]
            if use_norm:
                layers += [nn.GroupNorm(8, o)]
            layers += [nn.LeakyReLU(inplace=True)]
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            cb(in_ch, base_ch),
            cb(base_ch, base_ch),
            nn.Conv2d(base_ch, 1, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] == 3:
            x = rgb_to_gray(x)
        return self.net(x)


@LOSS_REGISTRY.register()
class LeeLoss(nn.Module):
    """Lee loss with EMA teacher to avoid trivial collapse.

    Args:
        edge_extractor: learnable student extractor (trainable)
        criterion: 'charbonnier' | 'l1' | 'l2'
        ema_tau: EMA momentum for teacher update (close to 1)
        detach_on_eval: if True, skip teacher update when model.eval()
    """
    def __init__(self,
                 edge_extractor: nn.Module = None,
                 loss_weight: float = 1.0,
                 criterion: str = 'charbonnier',
                 ema_tau: float = 0.999,
                 detach_on_eval: bool = True):
        super().__init__()
        self.edge_extractor = edge_extractor or LearnableEdgeExtractor(in_ch=1)
        self.student = self.edge_extractor
        self.teacher = copy.deepcopy(self.student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.loss_weight = loss_weight
        self.ema_tau = ema_tau
        self.detach_on_eval = detach_on_eval
        if criterion == 'charbonnier':
            self.crit = CharbonnierLoss()
        elif criterion == 'l1':
            self.crit = nn.L1Loss()
        elif criterion == 'l2':
            self.crit = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'charbonnier', 'l1' or 'l2'")

    @torch.no_grad()
    def _update_teacher(self):
        tau = self.ema_tau
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

    def update_teacher(self):
        return self._update_teacher()

    def forward(self, pred, target):
        e_pred = self.student(pred)
        with torch.no_grad():
            e_tgt = self.teacher(target)
        loss = self.crit(e_pred, e_tgt) * self.loss_weight
        if self.training or not self.detach_on_eval:
            self._update_teacher()
        return loss


# ============================================================
# Learnable Texture Extractor
# ============================================================



@ARCH_REGISTRY.register()
class LTE(nn.Module):
    """Multi-scale low-level texture extractor for SEM.
    Produces aligned feature maps at 3 receptive-field levels.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, use_norm: bool = False):
        super().__init__()
        # Level 1 (native scale): local edges/textures
        self.l1 = nn.Sequential(
            ConvBlock(in_ch, base_ch, use_norm=use_norm),
            ConvBlock(base_ch, base_ch, use_norm=use_norm),
        )
        # Level 2: 1/2 scale processing then upsample back (wider RF)
        self.l2_down = nn.AvgPool2d(2)
        self.l2 = nn.Sequential(
            ConvBlock(in_ch, base_ch * 2, use_norm=use_norm),
            ConvBlock(base_ch * 2, base_ch * 2, use_norm=use_norm),
        )
        # Level 3: dilated convs at native scale (even wider RF)
        self.l3 = nn.Sequential(
            ConvBlock(in_ch, base_ch * 2, d=2, p=2, use_norm=use_norm),
            ConvBlock(base_ch * 2, base_ch * 2, d=4, p=4, use_norm=use_norm),
        )
        # Heads to align channel dims for loss
        self.head1 = nn.Conv2d(base_ch, base_ch, 1)
        self.head2 = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.head3 = nn.Conv2d(base_ch * 2, base_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] == 3:
            x = rgb_to_gray(x)
        # Level 1
        f1 = self.l1(x)
        f1h = self.head1(f1)
        # Level 2
        x2 = self.l2_down(x)
        f2 = self.l2(x2)
        f2_up = F.interpolate(f2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        f2h = self.head2(f2_up)
        # Level 3
        f3 = self.l3(x)
        f3h = self.head3(f3)
        return {'l1': f1h, 'l2': f2h, 'l3': f3h}


@LOSS_REGISTRY.register()
class LteLoss(nn.Module):
    """LTE feature-matching loss with EMA teacher to avoid collapse.

    Args:
        lte: learnable student LTE (trainable)
        layer_weights: dict for {'l1','l2','l3'}
        criterion: 'charbonnier' | 'l1' | 'l2'
        ema_tau: EMA momentum for teacher
    """
    def __init__(self,
                 lte: nn.Module,
                 layer_weights=None,
                 criterion: str = 'charbonnier',
                 loss_weight: float = 1.0,
                 ema_tau: float = 0.999,
                 detach_on_eval: bool = True):
        super().__init__()
        self.lte = lte
        self.student = self.lte
        self.teacher = copy.deepcopy(lte).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights or {'l1': 1.0, 'l2': 1.0, 'l3': 1.0}
        self.ema_tau = ema_tau
        self.detach_on_eval = detach_on_eval
        if criterion == 'charbonnier':
            self.crit = CharbonnierLoss()
        elif criterion == 'l1':
            self.crit = nn.L1Loss()
        elif criterion == 'l2':
            self.crit = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'charbonnier', 'l1' or 'l2'")

    @torch.no_grad()
    def _update_teacher(self):
        tau = self.ema_tau
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

    def update_teacher(self):
        return self._update_teacher()

    def forward(self, pred, target, **kwargs):
        f_p = self.student(pred)           # with grad
        with torch.no_grad():
            f_t = self.teacher(target)     # no grad
        loss = 0.0
        for k, w in self.layer_weights.items():
            if w == 0:
                continue
            loss = loss + w * self.crit(f_p[k], f_t[k])
        loss = loss * self.loss_weight
        if self.training or not self.detach_on_eval:
            self._update_teacher()
        return loss
