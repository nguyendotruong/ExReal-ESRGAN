import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

# ------------------------------
# Edge Loss
# ------------------------------
@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        # Laplacian kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8
        self.register_buffer('kernel', kernel)

    def forward(self, x, y, **kwargs):
        kernel = self.kernel.expand(x.shape[1], 1, 3, 3)
        x_edge = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        y_edge = F.conv2d(y, kernel, padding=1, groups=y.shape[1])
        return self.loss_weight * F.l1_loss(x_edge, y_edge)


# ------------------------------
# LTE Loss (Perceptual feature-based)
# ------------------------------
@LOSS_REGISTRY.register()
class LteLoss(nn.Module):
    def __init__(self,
                 layer_weights={'conv1_2': 1.0, 'conv2_2': 1.0, 'conv3_4': 1.0},
                 loss_weight=1.0,
                 criterion='l1'):
        super(LteLoss, self).__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=False,
            requires_grad=False
        )

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'Unsupported criterion: {criterion}')

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target.detach())

        loss = 0
        for k in pred_feats.keys():
            loss += self.layer_weights[k] * self.criterion(pred_feats[k], target_feats[k])
        return self.loss_weight * loss
