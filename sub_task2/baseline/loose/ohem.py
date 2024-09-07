import torch
import torch.nn as nn

from .register import register_criterion

@register_criterion("OHEMLoss")

class OHEMLoss(nn.Module):
    def __init__(self, ratio=0.7, weight=None, reduction='mean'):
        """
        :param ratio: Ratio of hard examples to mine (float between 0 and 1).
        :param weight: A manual rescaling weight given to each class. Shape: (num_classes,).
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum').
        """
        super(OHEMLoss, self).__init__()
        self.ratio = ratio
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        """
        :param inputs: Logits from the model, shape (batch_size, num_classes).
        :param targets: Ground truth labels, shape (batch_size).
        :return: OHEM loss.
        """
        losses = self.criterion(inputs, targets)

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