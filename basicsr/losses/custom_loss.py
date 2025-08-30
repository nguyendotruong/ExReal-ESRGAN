import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        luma = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        return luma.unsqueeze(1)
    return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, use_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d)
        self.activation  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))



class MTE(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, use_norm: bool = False):
        super().__init__()
        self.l1 = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch)
        )
        self.l2_down = nn.AvgPool2d(2)
        self.l2 = nn.Sequential(
            ConvBlock(in_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2)
        )
        self.l3 = nn.Sequential(
            ConvBlock(in_ch, base_ch * 2, d=2, p=2),
            ConvBlock(base_ch * 2, base_ch * 2, d=4, p=4)
        )
        self.head1 = nn.Conv2d(base_ch, base_ch, 1)
        self.head2 = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.head3 = nn.Conv2d(base_ch * 2, base_ch, 1)


    def forward(self, x):
        x_gray = rgb_to_gray(x)
        f1 = self.l1(x_gray)
        f1h = self.head1(f1)
        x2 = self.l2_down(x_gray)
        f2 = self.l2(x2)
        f2_up = F.interpolate(f2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        f2h = self.head2(f2_up)
        f3 = self.l3(x_gray)
        f3h = self.head3(f3)
        return {'l1': f1h, 'l2': f2h, 'l3': f3h}


@LOSS_REGISTRY.register()
class MteLoss(nn.Module):
    """
    Perceptual loss using a PRE-TRAINED and FROZEN MTE model.

    Args:
        layer_weights (dict): Weights for different layers.
        mte_pretrain_path (str): Path to the pretrained MTE weights.
        use_input_norm (bool): Whether to normalize the input.
        range_norm (bool): Whether to norm [-1,1] to [0,1] before normalization.
        criterion (str): 'l1', 'l2', or 'charbonnier'.
        loss_weight (float): Loss weight.
        base_ch (int): Base channels of the MTE network.
    """
    def __init__(self,
                 layer_weights,
                 mte_pretrain_path,
                 use_input_norm=True,
                 range_norm=False,
                 criterion='l1',
                 loss_weight=1.0,
                 base_ch=32):
        super().__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.mte_net = MTE(in_ch=1, base_ch=base_ch)

        if not os.path.exists(mte_pretrain_path):
            raise FileNotFoundError(f"MTE pretrained model not found at {mte_pretrain_path}")
        state_dict = torch.load(mte_pretrain_path, map_location=lambda storage, loc: storage)
        self.mte_net.load_state_dict(state_dict)

        self.mte_net.eval()
        for param in self.mte_net.parameters():
            param.requires_grad = False

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        elif criterion == 'charbonnier':
            self.criterion = CharbonnierLoss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.5]).view(1, 1, 1, 1))
            self.register_buffer('std', torch.Tensor([0.5]).view(1, 1, 1, 1))

    def forward(self, pred, target):
        self.mte_net.to(pred.device)

        if self.use_input_norm:
            if self.range_norm:
                 pred = (pred + 1) / 2
                 target = (target + 1) / 2
            pred = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
            target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        pred_features = self.mte_net(pred)
        target_features = self.mte_net(target)

        loss = 0.0
        for key, weight in self.layer_weights.items():
            if weight > 0 and key in pred_features:
                loss += weight * self.criterion(pred_features[key], target_features[key])

        return self.loss_weight * loss






class SobelEdge(nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()
        gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)
        gy = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        self.register_buffer('weight_x', gx)
        self.register_buffer('weight_y', gy)

    def forward(self, x):
        if x.shape[1] == 3:
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

        edge_x = F.conv2d(x, self.weight_x, padding=1)
        edge_y = F.conv2d(x, self.weight_y, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edge


@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, criterion: str = 'l1'):
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobel = SobelEdge()

        if criterion.lower() == 'l1':
            self.crit = nn.L1Loss()
        elif criterion.lower() in ['l2', 'mse']:
            self.crit = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'l1' or 'l2'/'mse'")

    def forward(self, pred, target):
        e_pred = self.sobel(pred)
        e_tgt = self.sobel(target)
        loss = self.crit(e_pred, e_tgt) * self.loss_weight
        return loss

