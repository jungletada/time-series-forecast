import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channels) if configs.rev else None
        self.individual = configs.individual

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [B, L, D]
        x_enc = self.rev(x_enc, 'norm') if self.rev else x_enc
        x_enc = self.dropout(x_enc)
        if self.individual:
            pred = x_enc.new_zeros(x_enc.size(0), self.Linear[0].out_features, x_enc.size(2))
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x_enc[:, :, idx])
        else:
            pred = self.Linear(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred

