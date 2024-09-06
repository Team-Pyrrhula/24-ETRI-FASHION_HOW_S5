import torch
import torch.nn as nn

from .register import register_criterion

@register_criterion("OHEMLoss")

class OHEMLoss(nn.Module):
    def __init__(self, ratio=0.7, weight=None):
        super(OHEMLoss, self).__init__()
        self.ratio = ratio
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        losses = self.criterion(inputs, targets)
        num_samples = int(self.ratio * losses.numel())
        hard_losses, _ = losses.topk(num_samples)
        return hard_losses.mean()