import torch
import torch.nn as nn

from .register import register_criterion

@register_criterion("EpochWeightLoss")
class DynamicWeightedLoss(nn.Module):
    def __init__(self, singularity, weight=None, device='cuda'):
        super().__init__()
        self.device = device
        self.singularity = singularity

        if  weight is not None:
            self.weight = weight.to(device)
        else:
            self.weight = weight

        self.criterion = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def forward(self, inputs, targets):
        losses = self.criterion(inputs, targets)
        return (losses)

    def update_epoch(self, epoch):
        if epoch == self.singularity:
            self.criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='mean').to(self.device)
            print(f"Loss weight is Change -> {self.weight} !!!")
