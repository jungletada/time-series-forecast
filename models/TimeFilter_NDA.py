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
        self.nda_patch = configs.nda_patch
        self.adapter = DecompInputAdapter(
            d_model=self.dim,
            patch_len=self.nda_patch,
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
        x_decomp: [B, T, C, K] - 原始输入
        """
        B, T, C, K = x_decomp.shape
        
        # ==========================================
        # 1. 归一化 (保留分量相对幅度)
        # ==========================================
        # 还原原始信号用于计算统计量
        x_raw = x_decomp.sum(dim=-1) # [B, T, C]
        
        # 计算均值和标准差 (Instance Normalization)
        mean = x_raw.mean(dim=1, keepdim=True).unsqueeze(-1) # [B, 1, C, 1]
        std = x_raw.std(dim=1, keepdim=True).unsqueeze(-1)   # [B, 1, C, 1]
        
        # 归一化所有分量
        x_decomp = (x_decomp - mean) / (std + 1e-6)

        # 1. Adapter: [B, T, C, K] -> [B * C, Num_Patches, D]
        enc_out = self.adapter(x_decomp) 
        
        # TimeFilter Backbone 期望的输入是 [B, Total_Nodes, D]
        # 其中 Total_Nodes = Num_Variables * Num_Patches_Per_Var
        
        # 2. 将 C 和 Num_Patches 展平合并: [B, C * Num_Patches, D]
        # 这样总节点数就是 C * N = 42 (如果 C=7, N=6)
        enc_out = enc_out.reshape(B, C * self.num_patches, self.dim)

        # 2. Backbone: 输入 [B, 42, D]
        # 注意：_get_mask 也需要根据新的维度生成 mask
        # TimeFilter 内部会处理这个长序列的图结构
        enc_out, moe_loss = self.backbone(enc_out, self._get_mask(enc_out.device), self.alpha)

        # 3. Output Head: [B, C * Num_Patches, D] -> [B, C, Pred_Len]
        # 恢复维度以便 Head 处理
        enc_out = enc_out.reshape(B, C, self.num_patches, self.dim) # [B, C, N, D]
        enc_out = enc_out.flatten(start_dim=-2) # [B, C, N*D]
        out = self.head(enc_out) # [B, C, Pred_Len] -> Head 是 Linear(N*D, Pred_Len)
        out = out.permute(0, 2, 1) # [B, Pred_Len, C]

        # ==========================================
        # 5. 反归一化
        # ==========================================
        mean = mean.squeeze(-1) # [B, 1, C]
        std = std.squeeze(-1)
        out = out * std + mean
        
        return out, moe_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, moe_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], moe_loss
        return None