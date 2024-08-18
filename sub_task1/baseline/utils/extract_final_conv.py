import timm
import torch.nn as nn
from collections import OrderedDict

def extract_final_conv(config):
    model = timm.create_model(
        config.MODEL, True, num_classes=0
    )
    final_conv = OrderedDict()
    for name, module in model.named_children():
        if (name == 'final_conv'):
            final_conv[name] = module
        else:
            continue
    final_conv['head'] = model.head
    return (nn.Sequential(final_conv))  