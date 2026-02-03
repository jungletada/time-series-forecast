import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from layers.Embed import DataEmbedding
from layers.StandardNorm import Normalize
from layers.nda import DecompInputAdapter 


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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
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
    TCN with Decomposition Adapter (TCN-NDA)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        
        # Decomposition Parameters
        self.decomp_k = getattr(configs, 'decomp_k', 3)

        # TCN Parameters
        # 注意: TCN 的输入通道数现在是 d_model (来自 Adapter)
        self.num_channels = [configs.d_model] * configs.e_layers 
        self.kernel_size = configs.kernel_size
        self.dropout = configs.dropout

        # ============================================================
        # 1. 初始化通用分解适配器
        # ============================================================
        # TCN 是 Point-wise 的，所以我们使用 mode='timestep' (或 patch_len=1)
        self.adapter = DecompInputAdapter(
            d_model=self.d_model,
            patch_len=1,     # TCN 逐点处理，Patch Length 设为 1
            stride=1,
            decomp_k=self.decomp_k,
            dropout=self.dropout,
            pos_embed=False, # TCN 对位置敏感，通常不需要显式 PosEmbed，或者由 Adapter 内部加
            mode='timestep'  # 标记为时间步模式
        )
        
        # 2. TCN Encoder
        self.tcn = TemporalConvNet(
            self.d_model,  # Input Channels = Adapter Output dim
            self.num_channels, 
            kernel_size=self.kernel_size, 
            dropout=self.dropout
        )

        # 3. Forecasting Heads
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Adapter 输出是 [B*C, T, D]
            # 我们需要映射回 [B, C, Pred_Len]
            # 策略：先 TCN 提取特征 -> Flatten -> Linear -> Reshape
            
            # 方式 A: 直接映射所有时间步 (Seq_Len * d_model -> Pred_Len)
            # 这比较暴力，但有效
            self.head = nn.Linear(self.seq_len * self.d_model, self.pred_len)
            
            # 或者方式 B: 类似原版 TCN，先 Project 到 c_out 再 Project Time
            # 但这里我们是 Channel Independent，所以 Output 应该是 1 (单变量)
            # self.projection = nn.Linear(self.d_model, 1) 
            # self.predict_linear = nn.Linear(self.seq_len, self.pred_len)

        else:
            raise ValueError(f"Task {self.task_name} not supported in this adapter version.")
            
        # 归一化 (不使用 RevIN，因为我们有手动归一化)
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

    def forecast(self, x_decomp, masks, x_dec, x_mark_dec):
        """
        x_decomp: [B, T, C, K]
        """
        B, T, C, K = x_decomp.shape
        
        # ==========================================
        # 1. 归一化 (保留分量相对幅度)
        # ==========================================
        x_raw = x_decomp.sum(dim=-1) # [B, T, C]
        mean = x_raw.mean(dim=1, keepdim=True).unsqueeze(-1) # [B, 1, C, 1]
        std = x_raw.std(dim=1, keepdim=True).unsqueeze(-1)   # [B, 1, C, 1]
        x_decomp = (x_decomp - mean) / (std + 1e-5)

        # ==========================================
        # 2. Adapter 处理
        # ==========================================
        # Input: [B, T, C, K]
        # DecompInputAdapter (patch_len=1) 内部逻辑：
        # -> Permute/Reshape: [B*C, T, K]
        # -> Unfold (size=1): [B*C, T, K, 1]
        # -> Proj & Gate: [B*C, T, D]
        enc_out = self.adapter(x_decomp) # [B*C, T, D]

        # ==========================================
        # 3. TCN 处理
        # ==========================================
        # TCN 期望输入: [Batch, Channels, Seq_Len]
        # 这里的 Batch 是 B*C, Channels 是 D, Seq_Len 是 T
        enc_out = enc_out.reshape(B*C, T, -1)
        enc_out = enc_out.permute(0, 2, 1) # [B*C, D, T]
        
        enc_out = self.tcn(enc_out) # [B*C, D, T]
        
        # ==========================================
        # 4. Prediction Head
        # ==========================================
        # [B*C, D, T] -> [B*C, T, D]
        enc_out = enc_out.permute(0, 2, 1)
        
        # Flatten: [B*C, T*D]
        enc_out = enc_out.reshape(B * C, -1)
        
        # Linear Projection: [B*C, Pred_Len]
        # 注意：这里我们预测的是每个 Channel 的未来值，所以 Output dim 是 Pred_Len (隐含 Channel=1)
        out = self.head(enc_out) 
        
        # Reshape back to [B, C, Pred_Len] -> [B, Pred_Len, C]
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # ==========================================
        # 5. 反归一化
        # ==========================================
        mean = mean.squeeze(-1) # [B, 1, C]
        std = std.squeeze(-1)
        out = out * std + mean
        
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out # [B, Pred_Len, C]
        return None