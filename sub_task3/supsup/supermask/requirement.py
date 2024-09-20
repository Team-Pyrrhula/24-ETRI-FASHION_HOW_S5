'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2023, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.06.16.
'''
# torch
import torch
import torch.nn as nn

# custom modules
from supermask.linear import MultitaskMaskLinear


class RequirementMLP(nn.Module):
    """대화문 임베딩 벡터로부터 사용자의 요구사항에 대한 feature를 추출하는 MLP 모델입니다.
    """
    def __init__(self, mem_size: int = 16, emb_size: int = 128,
                 architecture: str = '[2000,1000,500]',
                 use_dropout: bool = False, zero_prob: float = 0.25,
                 use_batchnorm: bool = False, out_size: int = 300,
                 mode: str = 'train', pred_option: str = 'scores', num_tasks: int = 6) -> None:
        """RequirementMLP 아키텍처를 build합니다.

        Args:
            mem_size (int, optional): 대화문 임베딩 벡터의 길이입니다. Defaults to 16.
            emb_size (int, optional): 임베딩 벡터의 크기입니다. Defaults to 128.
            architecture (str, optional): RequirementMLP의 노드 구성입니다. Defaults to '[2000,1000,500]'.
            use_dropout (bool, optional): dropout을 적용할지 선택합니다. Defaults to False.
            zero_prob (float, optional): dropout 확률을 설정합니다. Defaults to 0.25.
            use_batchnorm (bool, optional): batch normalization을 적용할지 선택합니다. Defaults to False.
            out_size (int, optional): 최종 출력 벡터의 크기입니다. Defaults to 300.
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

        super(RequirementMLP, self).__init__()
        
        self.emb_size = emb_size
        self.num_in = mem_size * emb_size

        # 모델 생성
        model = []
        num_in = self.num_in

        architecture = list(map(int, architecture[1:-1].split(',')))
        n_layer = len(architecture)

        for i in range(n_layer + 1):
            if i == n_layer:
                num_out = out_size
                model.append(
                    MultitaskMaskLinear(
                        num_in,
                        num_out,
                        scenario={'mode': mode,
                                  'pred_option': pred_option,
                                  'num_tasks': num_tasks},
                        bias=False
                    )
                )

            else:
                num_out = architecture[i]
                model.append(
                    MultitaskMaskLinear(
                        num_in,
                        num_out,
                        scenario={'mode': mode,
                                  'pred_option': pred_option,
                                  'num_tasks': num_tasks},
                        bias=False
                    )
                )
                model.append(nn.ReLU())

                if use_dropout:
                    model.append(nn.Dropout(p=zero_prob))

                if use_batchnorm:
                    model.append(nn.BatchNorm1d())

                num_in = num_out

        self.model = nn.Sequential(*model)
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        """just forward

        Args:
            x (torch.tensor): 대화문 임베딩 벡터입니다.

        Returns:
            torch.tensor: 대화문 임베딩 벡터로부터 추출한 feature입니다.
        """
        req = self.model(x)

        return req
    

# Test Code
if __name__ == "__main__":
    print(RequirementMLP())