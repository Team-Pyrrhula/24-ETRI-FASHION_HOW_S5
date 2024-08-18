import torch.nn as nn
import torch
import timm
from utils import extract_final_conv
from .mae_model import MAE_Model


class ETRI_MAE_model(nn.Module):
    def __init__(self, config, encoder, final_conv, freeze:bool=True):
        super(ETRI_MAE_model, self).__init__()
        self.config = config
        #freeze opt
        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False

        self.model = nn.Sequential(
            encoder,
            final_conv,
        )

        #model output check
        self.in_features = self._check_output_size()

        #deepler classifier append
        if config.DEEP_HEAD:
            self.daily = self._make_deep_classifier(self.in_features // 2, config.INFO['label_1_num'])
            self.gender = self._make_deep_classifier(self.in_features // 2, config.INFO['label_2_num'])
            self.embel = self._make_deep_classifier(self.in_features // 2, config.INFO['label_3_num'])
        else:
            self.daily = nn.Linear(self.in_features, config.INFO['label_1_num'])
            self.gender = nn.Linear(self.in_features, config.INFO['label_2_num'])
            self.embel = nn.Linear(self.in_features, config.INFO['label_3_num'])

    def _make_deep_classifier(self, hidden_layer, output_size):
        layers = nn.Sequential(
            nn.Linear(self.in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, output_size)
        )
        return (layers)

    def _check_output_size(self):
        dummy_input = torch.randn(3, 3, 224, 224).to(self.config.DEVICE)
        with torch.no_grad():
            output_size = self.model(dummy_input).shape[1]
        return (output_size)

    def forward(self, x):
        out = self.model(x)
        l1 = self.daily(out)
        l2 = self.gender(out)
        l3 = self.embel(out)
        return (l1, l2, l3)
    
class ETRI_MAE_Inference_model(nn.Module):
    def __init__(self, config, mae_config, deep_head:bool=True):
        super(ETRI_MAE_Inference_model, self).__init__()
        encoder = MAE_Model(mae_config).encoder
        final_conv = extract_final_conv(config)

        self.config = config
        self.model = nn.Sequential(encoder, final_conv)
        del encoder, final_conv
 
        #model output check
        self.in_features = self._check_output_size()

        #deepler classifier append
        if config.DEEP_HEAD:  
            self.daily = self._make_deep_classifier(self.in_features // 2, config.INFO['label_1_num'])
            self.gender = self._make_deep_classifier(self.in_features // 2, config.INFO['label_2_num'])
            self.embel = self._make_deep_classifier(self.in_features // 2, config.INFO['label_3_num'])
        else:
            self.daily = nn.Linear(self.in_features, config.INFO['label_1_num'])
            self.gender = nn.Linear(self.in_features, config.INFO['label_2_num'])
            self.embel = nn.Linear(self.in_features, config.INFO['label_3_num'])

    def _make_classifier(self, hidden_layer, output_size):
        layers = nn.Sequential(
            nn.Linear(self.in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, output_size)
        )
        return (layers)

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