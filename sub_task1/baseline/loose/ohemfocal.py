import torch
import torch.nn as nn
import torch.nn.functional as F
from .focal import FocalLoss
from .register import register_criterion

@register_criterion("OHEMFocalLoss")

class OHEMFocalLoss(nn.Module):
    def __init__(self,  ratio=0.7, alpha=1, gamma=2, reduction='mean', weight=None):
        super(OHEMFocalLoss, self).__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none', weight=weight)

    def forward(self, inputs, targets):
        """
        :param inputs: Logits from the model, shape (batch_size, num_classes).
        :param targets: Ground truth labels, shape (batch_size).
        :return: OHEM loss with Focal Loss.
        """
        losses = self.focal_loss(inputs, targets)

        # Sort the losses in descending order (hard examples first)
        num_samples = len(losses)
        sorted_losses, _ = torch.sort(losses, descending=True)

        # Select the top `ratio` fraction of hard examples
        num_hard_examples = int(self.ratio * num_samples)
        if num_hard_examples == 0:
            num_hard_examples = 1

        hard_loss = sorted_losses[:num_hard_examples]

        # Apply the specified reduction
        if self.reduction == 'mean':
            return hard_loss.mean()
        elif self.reduction == 'sum':
            return hard_loss.sum()
        else:
            return hard_loss