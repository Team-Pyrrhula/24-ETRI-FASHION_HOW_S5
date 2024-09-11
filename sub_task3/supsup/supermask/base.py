# torch
import torch
import torch.nn as nn

# custom modules
from supermask.requirement import RequirementMLP
from supermask.policy import PolicyNet


# TODO: 구현 마무리
class SupSupMLP(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, mem_size: int = 16, emb_size: int = 128, out_size: int = 300,
                 meta_size: int = 4, coordi_size: int = 4, num_rnk: int = 3,
                 req_node: str = '[2000,1000,500]', eval_node: str = '[6000,6000,200][2000]',
                 use_batch_norm: bool = False, use_dropout: bool = False, zero_prob: float = 0.5,
                 use_multimodal: bool = False, img_feat_size: int = 4096, num_tasks: int = 6):
        """
        initialize and declare variables
        """
        super().__init__()

        self._mem_size = mem_size
        self._emb_size = emb_size

        # class instance for requirement estimation
        self._requirement = RequirementMLP(mem_size, emb_size, req_node,
                                           use_dropout, zero_prob,
                                           use_batch_norm, out_size, num_tasks)

        # class instance for ranking
        self._policy = PolicyNet(emb_size, out_size, meta_size, coordi_size,
                                 num_rnk, eval_node, use_batch_norm,
                                 use_dropout, zero_prob, use_multimodal,
                                 img_feat_size, num_tasks)

    def forward(self, dlg, crd):
        """
        build graph
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