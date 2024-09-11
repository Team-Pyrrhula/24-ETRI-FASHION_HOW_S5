# torch
import torch.autograd as autograd

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        """binary mask 생성
        """
        return (scores >= 0).float()
    
    @staticmethod
    def backward(ctx, g):
        """gradient를 그대로 반환
        """
        return g
