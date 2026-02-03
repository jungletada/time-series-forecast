import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from layers.Embed import DataEmbedding


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    """
    TCN adapted for Time-Series-Library.
    Temporal convolutional networks: A unified approach to action segmentation
    @inproceedings{lea2016temporal,
        title={Temporal convolutional networks: A unified approach to action segmentation},
        author={Lea, Colin and Vidal, Rene and Reiter, Austin and Hager, Gregory D},
        booktitle={European conference on computer vision},
        pages={47--54},
        year={2016},
        organization={Springer}
        }
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # TCN Parameters
        self.num_channels = [configs.d_model] * configs.e_layers  # Stack layers with d_model depth
        self.kernel_size = configs.kernel_size
        self.dropout = configs.dropout

        # Embedding Layer
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            self.dropout)

        # TCN Encoder
        self.tcn = TemporalConvNet(
            configs.d_model, 
            self.num_channels, 
            kernel_size=self.kernel_size, 
            dropout=self.dropout)

        # Forecasting Heads
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Project from d_model to c_out
            self.projection = nn.Linear(configs.d_model, configs.c_out)
            # Project time dimension: seq_len -> pred_len
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len)

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out)
            
        if self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout_layer = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [Batch, Seq_Len, Enc_In]
        """
        # 1. Normalization (Non-stationary Transformer logic)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 2. Embedding
        # Output: [Batch, Seq_Len, d_model]
        if x_mark_enc is not None and torch.all(x_mark_enc == 0):
            x_mark_enc = None
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 3. TCN Encoder
        # TCN expects [Batch, Channels, Seq_Len], so we permute
        enc_out = enc_out.permute(0, 2, 1) 
        enc_out = self.tcn(enc_out)
        # Back to [Batch, Seq_Len, d_model]
        enc_out = enc_out.permute(0, 2, 1)

        # 4. Prediction
        # [Batch, Seq_Len, d_model] -> [Batch, Seq_Len, c_out]
        dec_out = self.projection(enc_out)
        
        # Project Time Dimension: [Batch, Seq_Len, c_out] -> [Batch, c_out, Seq_Len] -> Linear -> [Batch, c_out, Pred_Len]
        dec_out = self.predict_linear(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 5. De-Normalization
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, Pred_Len, D]
        
        if self.task_name == 'imputation':
            # Simplified for imputation (reconstruct input)
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out = enc_out.permute(0, 2, 1)
            enc_out = self.tcn(enc_out)
            enc_out = enc_out.permute(0, 2, 1)
            dec_out = self.projection(enc_out)
            return dec_out

        if self.task_name == 'anomaly_detection':
            # Simplified for anomaly detection (reconstruct input)
            enc_out = self.enc_embedding(x_enc, None)
            enc_out = enc_out.permute(0, 2, 1)
            enc_out = self.tcn(enc_out)
            enc_out = enc_out.permute(0, 2, 1)
            dec_out = self.projection(enc_out)
            return dec_out

        if self.task_name == 'classification':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out = enc_out.permute(0, 2, 1)
            enc_out = self.tcn(enc_out)
            enc_out = enc_out.permute(0, 2, 1)
            
            output = self.act(enc_out)
            output = self.dropout_layer(output)
            output = output * x_mark_enc.unsqueeze(-1)
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output)
            return output
            
        return None