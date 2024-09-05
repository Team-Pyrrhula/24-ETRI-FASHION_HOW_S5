import torch
import torch.nn as nn
import torch.nn.functional as F

from .register import register_criterion

@register_criterion("OHEMFocalLoss")

class OHEMFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ratio=0.7, weight=None):
        super(OHEMFocalLoss, self).__init__()
        self.alpha = alpha  # focal loss의 alpha 값
        self.gamma = gamma  # focal loss의 gamma 값
        self.ratio = ratio  # OHEM에서 사용할 어려운 예제 비율
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal Loss 계산
        pt = torch.exp(-ce_loss)  # pt는 예측 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # focal loss 적용

        # OHEM: 가장 어려운 예제 선택
        num_samples = int(self.ratio * focal_loss.numel())
        hard_losses, _ = focal_loss.topk(num_samples)

        return hard_losses.mean()  # 평균 loss 반환