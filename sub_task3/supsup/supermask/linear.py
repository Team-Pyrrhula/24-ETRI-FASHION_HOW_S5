# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom modules
from supermask.initialize import mask_init, signed_constant
from supermask.subnet import GetSubnet


class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, 
                 scenario: dict = {'mode': 'train', 
                                   'pred_option': 'scores', 
                                   'num_tasks': 6},  **kwargs):
        """
        note:
            - 레이어는 한 개, 스코어는 num_tasks만큼 생성
            - pred mode가 아닌 경우에는, pred_option과 상관없이 무조건 scores만 이용하여 예측
            - pred mode이면서 pred_option이 masks인 경우, buffer에 등록된 binary masks만 이용하여 예측
        """
        super().__init__(*args, **kwargs)
        
        self.mode = scenario['mode']
        self.pred_option = scenario['pred_option']
        self.num_tasks = scenario['num_tasks']
        
        # scores parameters 선언 및 초기화
        self.scores = nn.ParameterList([
            nn.Parameter(mask_init(self)) for _ in range(self.num_tasks)
            ])
        
        self.weight.requires_grad = False
        signed_constant(self)
        
        # 예측 시 binary masks만 사용하고자 한다면
        if self.mode == 'pred' and self.pred_option == 'masks':
            # 1. scores parameters는 불필요하므로 삭제하고
            self.rm_scores()

            # 2. 학습 과정에서 저장한 binary masks를 불러오기 위해, 동일한 이름의 빈 텐서를 버퍼에 등록
            self.register_buffer(
                "stacked",
                torch.stack([
                    torch.zeros(*args).transpose(1, 0).to(torch.int8)
                    for _ in range(self.num_tasks)
                ])
            )
        
    @torch.no_grad()
    def cache_masks(self):
        """subnet의 binary mask를 등록
        """
        self.register_buffer(
            "stacked",
            torch.stack([
                GetSubnet.apply(self.scores[j]).to(torch.int8) # TODO: 정상 동작하는지 검증하기 
                for j in range(self.num_tasks)
            ])
        )

    @torch.no_grad()
    def rm_scores(self):
        """scores parameter를 삭제
        """
        delattr(self, 'scores')
    
    def forward(self, x):
        """생성자에 없는 변수들은 클래스 밖에서 선언
        note: task는 0부터 시작해야 함
        """
        # inference (without task id) / note: not used in sub_task3
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

        # inference (with task id)
        else:
            # only use <binary masks>, not scores
            if self.mode == 'pred' and self.pred_option == 'masks':
                subnet = torch.tensor(self.stacked[self.task], 
                                      dtype=torch.float32)

            # only use <scores>, not binary masks
            else: subnet = GetSubnet.apply(self.scores[self.task])
        
        subnet_w = self.weight * subnet # masking 후 subnet weight를 구해서
        x = F.linear(x, subnet_w, self.bias) # linear transformation
        
        return x
