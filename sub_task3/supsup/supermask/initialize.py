# torch
import torch
import torch.nn as nn

# built-in library
import math


def mask_init(module):
    """kaiming uniform distribution을 따르는 값들로 weights를 초기화
    """
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    
    return scores

def signed_constant(module):
    """weights를 동일한 값(std)으로 설정하되, 부호를 무작위로 부여하는 초기화 방법
    """
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in') # ?
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    
    module.weight.data = module.weight.data.sign() * std