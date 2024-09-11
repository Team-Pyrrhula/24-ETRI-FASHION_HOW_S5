# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom modules
from supermask.initialize import mask_init, signed_constant
from supermask.subnet import GetSubnet


class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, **kwargs):
        """note: 레이어는 한 개, 스코어는 num_tasks만큼 생성
        """
        super().__init__(*args, **kwargs)
        
        self.num_tasks = num_tasks
        
        # task별로 layer의 score를 따로 기록
        self.scores = nn.ParameterList([
            nn.Parameter(mask_init(self)) for _ in range(num_tasks)
        ])
        
        self.weight.requires_grad = False
        signed_constant(self)
        
    @torch.no_grad()
    def cache_masks(self):
        """subnet의 binary mask를 등록
        """
        self.register_buffer(
            "stacked",
            torch.stack([
                GetSubnet.apply(self.scores[j]) 
                for j in range(self.num_tasks)
            ])
        )
    
    # TODO: 마스크로 포워드하기
    def forward(self, x):
        """생성자에 없는 변수들은 클래스 밖에서 선언
        주의) task는 0부터 시작해야 함
        """
        # inference (with no task id)
        if self.task < 0:
            alpha_weights = self.alphas[:self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            
            # TODO: inference 알고리즘 분석
            subnet = (
                alpha_weights[idxs] *
                    self.stacked[:self.num_tasks_learned][idxs]
            ).sum(dim=0)
            
        # train
        else:
            subnet = GetSubnet.apply(self.scores[self.task])
        
        subnet_w = self.weight * subnet # masking 후 subnet weight를 구해서
        x = F.linear(x, subnet_w, self.bias) # linear transformation
        
        return x