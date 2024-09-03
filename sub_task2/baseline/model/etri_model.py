import torch.nn as nn
import timm

class ETRI_model_color(nn.Module):
    def __init__(self, config):
        super(ETRI_model_color, self).__init__()
        self.model = timm.create_model(
            config.MODEL, config.PRETRAIN, num_classes=19
        )

    def forward(self, x):
        out = self.model(x)
        return out