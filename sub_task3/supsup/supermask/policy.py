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

# built-in library
import math

# custom modules
from supermask.linear import MultitaskMaskLinear


class PolicyNet(nn.Module):
    def __init__(self, emb_size: int = 128, out_size: int = 300, 
                 meta_size: int = 4, coordi_size: int = 4, num_rnk: int = 3,
                 eval_node: str = '[6000,6000,200][2000]',
                 use_batch_norm: bool = False, use_dropout: bool = False, 
                 zero_prob: float = 0.5, use_multimodal: bool = False,
                 img_feat_size: int = 4096, num_tasks: int = 6, name='PolicyNet'):
        
        super(PolicyNet, self).__init__()
        self._num_rnk = num_rnk

        # 모델 생성에 필요한 hparams 정의
        buf = eval_node[1:-1].split('][')
        
        num_hid_eval = list(map(int, buf[0].split(',')))
        num_hid_rnk = list(map(int, buf[1].split(',')))

        num_hid_layer_eval = len(num_hid_eval)
        num_hid_layer_rnk = len(num_hid_rnk)

        ##### evaluation MLP #####
        mlp_eval_list = []
        num_in = (emb_size * meta_size * coordi_size) + out_size

        if use_multimodal:
            num_in += img_feat_size

        count_eval = 0
        for i in range(num_hid_layer_eval):
            num_out = num_hid_eval[i]

            mlp_eval_list.append(
                MultitaskMaskLinear(
                    num_in,
                    num_out,
                    num_tasks=num_tasks,
                    bias=False
                )
            )
            mlp_eval_list.append(
                nn.ReLU()
            )
            
            if use_batch_norm:
                mlp_eval_list.append(
                    nn.BatchNorm1d(num_out)
                )
                
            if use_dropout:
                mlp_eval_list.append(
                    nn.Dropout(p=zero_prob)
                )
                
            count_eval += (num_in * num_out + num_out)
            num_in = num_out

        eval_out_node = num_out
        self._mlp_eval = nn.Sequential(*mlp_eval_list)
        ##### evaluation MLP #####

        ##### ranking MLP #####
        mlp_rnk_list = []
        num_in = (eval_out_node * self._num_rnk) + out_size

        for i in range(num_hid_layer_rnk + 1):
            if i == num_hid_layer_rnk:
                num_out = math.factorial(self._num_rnk)

                mlp_rnk_list.append(
                    MultitaskMaskLinear(
                        num_in,
                        num_out,
                        num_tasks=num_tasks,
                        bias=False
                    )
                )
                
            else:
                num_out = num_hid_rnk[i]

                mlp_rnk_list.append(
                    MultitaskMaskLinear(
                        num_in,
                        num_out,
                        num_tasks=num_tasks,
                        bias=False
                    )
                )
                mlp_rnk_list.append(
                    nn.ReLU()
                )

                if use_batch_norm:
                    mlp_rnk_list.append(nn.BatchNorm1d(num_out))

                if use_dropout:
                    mlp_rnk_list.append(nn.Dropout(p=zero_prob))

            count_eval += (num_in * num_out + num_out)
            num_in = num_out

        self._mlp_rnk = nn.Sequential(*mlp_rnk_list) 
        ##### ranking MLP #####

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """        
        crd_and_req = torch.cat((crd, req), 1)
        evl = self._mlp_eval(crd_and_req)

        return evl
    
    def _ranking_coordi(self, in_rnk):
        """
        rank candidates         
        """        
        out_rnk = self._mlp_rnk(in_rnk)

        return out_rnk
    
    def forward(self, req, crd):
        """
        build graph for evaluation and ranking         
        """
        # (bs, 3, 2048) -> (3, bs, 2048)
        crd_tr = torch.transpose(crd, 1, 0)

        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_tr[i], req)

            if i == 0:
                in_rnk = crd_eval
            
            else:
                in_rnk = torch.cat((in_rnk, crd_eval), dim=1)

        in_rnk = torch.cat((in_rnk, req), 1)                
        out_rnk = self._ranking_coordi(in_rnk)

        return out_rnk


# Test code
if __name__ == "__main__":
    print(PolicyNet())