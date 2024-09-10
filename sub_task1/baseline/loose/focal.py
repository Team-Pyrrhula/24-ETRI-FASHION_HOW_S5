import torch
import torch.nn as nn
import torch.nn.functional as F

from .register import register_criterion

@register_criterion("Focalloss")
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, logits=False, reduction='mean', weight=None, device='cuda'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        if  weight is not None:
            self.weight = weight.to(device)
        else:
            self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)

        # F.cross_entropy는 softmax를 자동적으로 포함하고 있기 때문에
        # logit 값을 구하기 위해서 지수함수 이용
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option.")