# torch
import torch
import torch.nn as nn

# custom modules
from supermask.requirement import RequirementMLP
from supermask.policy import PolicyNet


class SupSupMLP(nn.Module):
    """Supermasks in Superposition(SupSup) 방식을 적용한 MLP 모델 클래스입니다.
    """
    def __init__(self, mem_size: int = 16, emb_size: int = 128, out_size: int = 300,
                 meta_size: int = 4, coordi_size: int = 4, num_rnk: int = 3,
                 req_node: str = '[2000,1000,500]', eval_node: str = '[6000,6000,200][2000]',
                 use_batch_norm: bool = False, use_dropout: bool = False, zero_prob: float = 0.5,
                 use_multimodal: bool = False, img_feat_size: int = 4096, 
                 mode: str = 'train', pred_option: str = 'scores', num_tasks: int = 6) -> None:
        """전체 아키텍처를 build합니다.

        Args:
            mem_size (int, optional): 대화문 임베딩 벡터의 길이입니다. Defaults to 16.
            emb_size (int, optional): 임베딩 벡터의 크기입니다. Defaults to 128.
            out_size (int, optional): 최종 출력 벡터의 크기입니다. Defaults to 300.
            meta_size (int, optional): 메타데이터 특징의 개수입니다. Defaults to 4.
            coordi_size (int, optional): 하나의 패션 조합을 구성하는 아이템의 개수입니다. Defaults to 4.
            num_rnk (int, optional): 순위를 평가할 패션 코디 조합의 개수입니다. Defaults to 3.
            req_node (str, optional): RequirementMLP의 노드 구성입니다. Defaults to '[2000,1000,500]'.
            eval_node (str, optional): PolicyNet의 노드 구성입니다. Defaults to '[6000,6000,200][2000]'.
            use_batch_norm (bool, optional): batch normalization을 적용할지 선택합니다. Defaults to False.
            use_dropout (bool, optional): dropout을 적용할지 선택합니다. Defaults to False.
            zero_prob (float, optional): dropout 확률을 설정합니다. Defaults to 0.5.
            use_multimodal (bool, optional): 텍스트와 함께 이미지 데이터를 사용할지 선택합니다. Defaults to False.
            img_feat_size (int, optional): 이미지 데이터를 나타내는 피처의 크기입니다. Defaults to 4096.
            mode (str, optional): 
                - 실행할 모드를 선택합니다. 
                - train/eval/test/pred의 네 가지 모드가 존재합니다. 
                - Defaults to 'train'.

            pred_option (str, optional): 
                - 추론 방식을 선택합니다. 
                - 'scores'와 'masks'가 존재하며, 'scores' 선택 시 edge pop-up score로,
                    'masks' 선택 시 binary mask로 추론하게 됩니다.
                - Defaults to 'scores'.
                - note: 'masks' 옵션을 사용하기 위해선, 모델을 학습할 때도 'masks' 옵션을 설정해주어야 합니다.

            num_tasks (int, optional): 학습하고자 하는 총 task(dataset)의 개수입니다. Defaults to 6.
        """
        super().__init__()

        self._mem_size = mem_size
        self._emb_size = emb_size

        # class instance for requirement estimation
        self._requirement = RequirementMLP(mem_size, emb_size, req_node,
                                           use_dropout, zero_prob,
                                           use_batch_norm, out_size, 
                                           mode, pred_option, num_tasks)

        # class instance for ranking
        self._policy = PolicyNet(emb_size, out_size, meta_size, coordi_size,
                                 num_rnk, eval_node, use_batch_norm,
                                 use_dropout, zero_prob, use_multimodal,
                                 img_feat_size, mode, pred_option, num_tasks)

    def forward(self, dlg: torch.tensor, crd: torch.tensor) -> tuple:
        """just forward

        Args:
            dlg (torch.tensor): 대화문 임베딩 벡터입니다.
            crd (torch.tensor): 패션 코디 조합의 임베딩 벡터입니다.

        Returns:
            tuple:
                - logits: forward 결과로 계산된 값입니다.
                - preds: 모델이 예측한 순위입니다.
        """
        dlg = dlg.view(-1, self._mem_size * self._emb_size)

        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        preds = torch.argmax(logits, 1)

        return logits, preds
    

# Test code
if __name__ == "__main__":
    mlp = SupSupMLP()
    for name, param in mlp.named_parameters():
        if param.requires_grad:
            print(name)