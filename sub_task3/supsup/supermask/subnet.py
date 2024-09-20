# torch
import torch.autograd as autograd

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        """binary mask 생성
        """
        # TODO: 0이 아닌 threshold 값 설정(for generalization)
        return (scores >= 0).float()
    
    @staticmethod
    def backward(ctx, g):
        """gradient를 그대로 반환
        """
        return g
