import torch.nn.functional as F
import torch.nn as nn
import timm
from einops import rearrange
from collections import OrderedDict

class mobile_vit_s(nn.Module):
    def __init__(self, config):
        super(mobile_vit_s, self).__init__()

        self.encoder = timm.create_model(
            config.ENCODER, True, num_classes=0
        )
        # output shape [batch_size, 640, 7, 7] if input (batch_size, 224, 224)
        self.encoder.head.global_pool = nn.Identity()  
        self.encoder.head.flatten = nn.Identity()

    def forward(self, x):
        encoded_features = self.encoder(x)
        return (encoded_features)

class fast_vit_t8(nn.Module):
    def __init__(self, config):
        super(fast_vit_t8, self).__init__()

        self.encoder = timm.create_model(
            config.ENCODER, True, num_classes=0
        )
        self.encoder = self._get_layers_until('final_conv')

    def _get_layers_until(self, stop_layer_name):
        layers = OrderedDict()
        for name, module in self.encoder.named_children():
            if name == stop_layer_name:
                break
            layers[name] = module

        return nn.Sequential(layers)

    def forward(self, x):
        encoded_features = self.encoder(x)
        return (encoded_features)