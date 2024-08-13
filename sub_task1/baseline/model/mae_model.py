import torch.nn.functional as F
import torch.nn as nn
import torch
import timm
from einops import rearrange

from .mae_encoder import mobile_vit_s, fast_vit_t8
    
_ENCODER_DICT = {
    'mobilevit_s.cvnets_in1k' :mobile_vit_s,
    'fastvit_t8.apple_in1k': fast_vit_t8,
}

class MAE_Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, img_size):
        super(MAE_Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = 32  # Assuming a patch size used during encoding

        # Transformer layers for decoding
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])

        # Projection to image channels
        self.proj = nn.Linear(embed_dim, 384)  # Adjust to match the encoder's output channels

        # Upsampling layers to bring the image back to original size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 192, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),   # Output: [batch_size, 96, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),    # Output: [batch_size, 48, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),     # Output: [batch_size, 3, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, x, memory):
        batch_size, embed_dim, height, width = x.size()

        # Flatten spatial dimensions and apply transformer layers
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, embed_dim)
        memory_flat = memory.permute(0, 2, 3, 1).reshape(batch_size * height * width, embed_dim)
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x_flat = layer(x_flat, memory_flat)
        # Apply projection layer
        x = self.proj(x_flat)  # Project to the target number of channels (e.g., 384)
        # 4D
        x = x.reshape(batch_size, height, width, embed_dim).permute(0, 3, 1, 2)
        # Apply upsampling
        x = self.upsample(x)

        return x
    
class MAE_Model(nn.Module):
    def __init__(self, config, mask_ratio:float=0.75):
        super(MAE_Model, self).__init__()
        self.config = config
        # encoder check
        assert (config.ENCODER in _ENCODER_DICT), f"{config.ENCODER} is not supply encoder model !!! "
        self.encoder = _ENCODER_DICT[config.ENCODER](config)
        self.embed_size = self._check_output_size()
        self.decoder = MAE_Decoder(
            self.embed_size,
            self.config.NUM_HEADS,
            self.config.HIDDEN_DIM,
            self.config.NUM_LAYERS,
            self.config.RESIZE
        )
        self.mask_ratio = mask_ratio

    def _check_output_size(self):
        dummy_input = torch.randn(3, 3, self.config.RESIZE, self.config.RESIZE)
        with torch.no_grad():
            output_size = self.encoder(dummy_input).shape[1]
        return (output_size)
    
    def forward(self, x):
        x = self.encoder(x)
        tgt = x
        result = self.decoder(tgt, x)
        return result

        