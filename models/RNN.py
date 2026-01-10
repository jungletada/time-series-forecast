import torch
import torch.nn as nn
from layers.Embed import DataEmbedding

class RNNBlock(nn.Module):
    """
    Standard RNN/LSTM/GRU wrapper.
    """
    def __init__(self, configs):
        super(RNNBlock, self).__init__()
        self.input_size = configs.d_model
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        # Allow selecting type via configs, default to LSTM
        self.rnn_type = getattr(configs, 'rnn_type', 'RNN') 

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        else: # Vanilla RNN
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            )

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        # output: [Batch, Seq_Len, hidden_size]
        out, _ = self.rnn(x)
        return out

class Model(nn.Module):
    """
    RNN/LSTM/GRU adapted for Time-Series-Library.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Embedding Layer
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout)

        # RNN Encoder
        self.rnn_block = RNNBlock(configs)

        # Forecasting Heads
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Project from hidden_size (d_model) to c_out
            self.projection = nn.Linear(configs.d_model, configs.c_out)
            # Project time dimension: seq_len -> pred_len (Direct forecasting strategy)
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

        # 3. RNN Encoder
        # Output: [Batch, Seq_Len, d_model]
        enc_out = self.rnn_block(enc_out)

        # 4. Prediction
        # [Batch, Seq_Len, d_model] -> [Batch, Seq_Len, c_out]
        dec_out = self.projection(enc_out)
        
        # Project Time Dimension: [Batch, Seq_Len, c_out] -> [Batch, c_out, Seq_Len] -> Linear -> [Batch, c_out, Pred_Len]
        # Transpose to apply linear layer on time dimension
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
            # Reconstruct input
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out = self.rnn_block(enc_out)
            dec_out = self.projection(enc_out)
            return dec_out

        if self.task_name == 'anomaly_detection':
            # Reconstruct input
            enc_out = self.enc_embedding(x_enc, None)
            enc_out = self.rnn_block(enc_out)
            dec_out = self.projection(enc_out)
            return dec_out

        if self.task_name == 'classification':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out = self.rnn_block(enc_out)
            
            output = self.act(enc_out)
            output = self.dropout_layer(output)
            output = output * x_mark_enc.unsqueeze(-1)
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output)
            return output
            
        return None