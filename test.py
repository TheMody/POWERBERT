import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, output_attention = True)
src = torch.rand(32, 10, 512)
out = encoder_layer(src)

print(out.shape)