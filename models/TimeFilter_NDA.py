import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone
from layers.nda import DecompInputAdapter

class PatchEmbed(nn.Module):
    """
    Original Patch Embedding for TimeFilter
    """
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x)  # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.args = configs
        
        # 1. 基础配置
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len # Non-overlap
        
        # 2. 获取分解参数 K
        self.decomp_k = getattr(configs, 'decomp_k', 3) 
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)

        # TimeFilter 超参
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # ============================================================
        # 3. 初始化通用分解适配器 (Adapter)
        # ============================================================
        self.adapter = DecompInputAdapter(
            d_model=self.dim,
            patch_len=self.patch_len,
            stride=self.stride,
            decomp_k=self.decomp_k,
            dropout=configs.dropout,
            pos_embed=configs.pos,  # Adapter 内部加位置编码
            mode='patch'            # 设为 patch 模式适配 TimeFilter
        )

        # ============================================================
        # 4. Backbone (TimeFilter)
        # ============================================================
        # TimeFilter 核心，输入标准的 Embeddings
        self.backbone = TimeFilter_Backbone(
            self.dim, self.n_vars, configs.d_ff,
            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
            self.seq_len * self.n_vars // self.patch_len
        )

        # ============================================================
        # 5. Prediction Head
        # ============================================================
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 将 Patch Embeddings 展平 -> 映射回 Pred_Len
            self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        else:
            # 可以在此扩展 Classification 等其他 Head
            raise ValueError(f"Task {self.task_name} not supported yet.")

        # 用于手动归一化，不使用 RevIN 模块
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

    def _get_mask(self, device):
        # TimeFilter 特有的掩码生成逻辑 (保持原样)
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(device)
            ST = torch.ones(L).to(dtype).to(device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks

    def forecast(self, x_decomp, masks, x_dec, x_mark_dec):
        """
        x_decomp: [B, T, C, K] - 输入的分解分量
        """
        B, T, C, K = x_decomp.shape
        
        # ==========================================
        # 1. 归一化 (Instance Normalization)
        # ==========================================
        # 还原原始信号用于计算统计量 (Original ≈ Sum(Components))
        x_raw = x_decomp.sum(dim=-1) # [B, T, C]
        
        # 计算均值和标准差: [B, 1, C, 1]
        mean = x_raw.mean(dim=1, keepdim=True).unsqueeze(-1)
        std = x_raw.std(dim=1, keepdim=True).unsqueeze(-1)
        
        # 归一化输入 (保留分量间的相对关系)
        x_decomp_norm = (x_decomp - mean) / (std + 1e-8)
        
        # ==========================================
        # 2. Adapter 特征提取
        # ==========================================
        # Adapter 融合 K 个分量 -> D_model
        # Input: [B, T, C, K] -> Output: [B * C, Num_Patches, D]
        enc_out = self.adapter(x_decomp_norm) 
        
        # ==========================================
        # 3. Backbone (TimeFilter) 处理
        # ==========================================
        # TimeFilter 期望输入: [B, Total_Nodes, D]
        # 我们将 Variables (C) 和 Patches (N) 视为图的节点
        
        # Reshape: [B * C, N, D] -> [B, C * N, D]
        enc_out = enc_out.reshape(B, C * self.num_patches, self.dim)

        # 传入 Backbone (注意 masks 需匹配节点数)
        # self._get_mask 会根据当前的 C*N 生成对应的图掩码
        enc_out, moe_loss = self.backbone(enc_out, self._get_mask(enc_out.device), self.alpha)
        
        # ==========================================
        # 4. Prediction Head
        # ==========================================
        # Backbone Output: [B, C * N, D]
        
        # 恢复维度以便 Head 处理: [B, C, N, D]
        enc_out = enc_out.reshape(B, C, self.num_patches, self.dim)
        
        # Flatten Patches: [B, C, N * D]
        enc_out = enc_out.flatten(start_dim=-2)
        
        # Linear Projection: [B, C, N * D] -> [B, C, Pred_Len]
        # 注意: 此时 self.head 应该是 nn.Linear(N*D, Pred_Len)
        out = self.head(enc_out)
        
        # 调整维度匹配 Target: [B, C, Pred_Len] -> [B, Pred_Len, C]
        out = out.permute(0, 2, 1)

        # ==========================================
        # 5. 反归一化
        # ==========================================
        # 调整 mean/std 维度以匹配 [B, Pred_Len, C]
        mean = mean.squeeze(-1) # [B, 1, C]
        std = std.squeeze(-1)   # [B, 1, C]
        
        out = out * std + mean
        
        # # ==========================================
        # # 6. 残差连接 (Residual Connection)
        # # ==========================================
        # # 加上输入序列的最后一个值，强制模型学习趋势增量
        # last_val = x_raw[:, -1:, :] # [B, 1, C]
        # out = out + last_val
        
        return out, moe_loss
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, moe_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], moe_loss
        return None
    