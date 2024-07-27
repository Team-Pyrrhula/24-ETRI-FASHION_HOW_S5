import torch.nn as nn
import torch
import timm

class ETRI_model(nn.Module):
    def __init__(self, config, pretrain:bool=True):
        super(ETRI_model, self).__init__()
        self.model = timm.create_model(
            config.MODEL, pretrain, num_classes=0
        )

        #model output check
        self.in_features = self._check_output_size()

        #classifier append
        self.daily = nn.Linear(self.in_features, config.INFO['label_1_num'])
        self.gender = nn.Linear(self.in_features, config.INFO['label_2_num'])
        self.embel = nn.Linear(self.in_features, config.INFO['label_3_num'])

    def _check_output_size(self):
        dummy_input = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            output_size = self.model(dummy_input).shape[1]
        return (output_size)

    def forward(self, x):
        out = self.model(x)
        l1 = self.daily(out)
        l2 = self.gender(out)
        l3 = self.embel(out)
        return (l1, l2, l3)