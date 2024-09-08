import torch
import torch.nn as nn

from .register import register_criterion

@register_criterion("DynamicWeightedLoss")

class DynamicWeightedLoss(nn.Module):
    def __init__(self, initial_weights, num_epochs):
        super().__init__()
        self.initial_weights = initial_weights
        self.num_epochs = num_epochs
        self.current_epoch = 0

    def forward(self, inputs, targets):
        # 동적 가중치 계산
        alpha = self.current_epoch / self.num_epochs
        current_weights = self.initial_weights * (1 - alpha) + torch.ones_like(self.initial_weights) * alpha
        return nn.functional.cross_entropy(inputs, targets, weight=current_weights)

    def update_epoch(self, epoch):
        self.current_epoch = epoch

"""
# 사용 예시
initial_weights = compute_class_weights(y_train)  # 초기 가중치 계산
criterion = DynamicWeightedLoss(initial_weights, num_epochs=100)

for epoch in range(100):
    # 훈련 루프
    ...
    criterion.update_epoch(epoch)
"""