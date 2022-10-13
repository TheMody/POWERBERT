import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, output_attention = True)
src = torch.rand(32,1,10).long()
src.to(device)
print(src.type())
out = torch.matmul(src.transpose(1,2),src)

print(out.shape)