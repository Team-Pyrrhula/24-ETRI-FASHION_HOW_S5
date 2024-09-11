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
import torch.nn as nn
import torch.nn.functional as F

# custom modules
from supermask.linear import MultitaskMaskLinear


class RequirementMLP(nn.Module):
    def __init__(self, mem_size: int = 16, emb_size: int = 128,
                 architecture: str = '[2000,1000,500]',
                 use_dropout: bool = False, zero_prob: float = 0.25,
                 use_batchnorm: bool = False, out_size: int = 300,
                 num_tasks: int = 6) -> None:
        super(RequirementMLP, self).__init__()
        
        self.mem_size = mem_size
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
                        num_tasks=num_tasks,
                        bias=False
                    )
                )

            else:
                num_out = architecture[i]
                model.append(
                    MultitaskMaskLinear(
                        num_in,
                        num_out,
                        num_tasks=num_tasks,
                        bias=False
                    )
                )
                model.append(nn.ReLU())

                if use_dropout:
                    model.append(nn.Dropout(p=zero_prob))

                # TODO: BatchNorm 파라미터도 score를 부여해야 하나?
                if use_batchnorm:
                    model.append(nn.BatchNorm1d())

                num_in = num_out

        self.model = nn.Sequential(*model)

    def forward(self, x):
        req = self.model(x)

        return req
    

# Test Code
if __name__ == "__main__":
    print(RequirementMLP())