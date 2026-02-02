import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding

class ComponentAwarePatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True, decomp_k=3):
        super().__init__()
        print(f">>>>> TimeFilter_CADE: patch_len: {patch_len}, decomp_k: {decomp_k}, dim: {dim} (Dynamic Mode)\n")
        
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.decomp_k = decomp_k
        self.dim = dim
        self.pos = pos

        # 1. 独立投影 (Decoupled Projection)
        self.component_projs = nn.ModuleList([
            nn.Linear(patch_len, dim) for _ in range(decomp_k)
        ])

        # 2. 动态门控网络 (Dynamic Gating Network)
        self.gating_net = nn.Sequential(
            nn.Linear(dim * decomp_k, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, decomp_k),
        )
        self.softmax = nn.Softmax(dim=-1)

        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, L, K]
        
        # 1. Unfold 展开时间维度
        # 输入 x: [B, L, K] -> Unfold dim 1 -> 输出: [B, Num_Patches, K, Patch_Len]
        # 注意：unfold 将新生成的维度 (Patch_Len) 放在最后
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # 2. 独立投影 (Independent Projection)
        embeddings_list = []
        for k in range(self.decomp_k):
            # 关键修正：切片维度应该是第 2 维 (K)，保留第 3 维 (Patch_Len)
            # x_patched[:, :, k, :] -> [B, Num_Patches, Patch_Len]
            component_patch = x_patched[:, :, k, :] 
            
            # Linear: [..., Patch_Len] -> [..., Dim]
            emb = self.component_projs[k](component_patch)
            embeddings_list.append(emb)

        # 堆叠起来: [B, Num_Patches, K, Dim]
        stacked_emb = torch.stack(embeddings_list, dim=2)

        # 3. 计算动态权重 (Dynamic Weighting)
        # Flatten Context: [B, Num_Patches, K * Dim]
        context = stacked_emb.flatten(start_dim=2)
        
        # Calculate Weights: [B, Num_Patches, K]
        attn_weights = self.gating_net(context)
        attn_weights = self.softmax(attn_weights) 
        
        # Expand for broadcast: [B, Num_Patches, K, 1]
        attn_weights = attn_weights.unsqueeze(-1)

        # 4. 加权融合 (Weighted Fusion)
        # Sum over K: [B, Num_Patches, Dim]
        x_final = (stacked_emb * attn_weights).sum(dim=2)

        if self.pos:
            x_final += self.pe(x_final)
            
        return x_final
    
class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True, decomp_k=3):
        super().__init__()
        print(f">>>>> TimeFilter_NDA: patch_len: {patch_len}, decomp_k: {decomp_k}, dim: {dim}\n\n")
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        # 关键修改：输入维度从 patch_len 变为 patch_len * K
        self.patch_proj = nn.Linear(self.patch_len * decomp_k, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, N_total, K]  (N_total 是展平后的总时间点数 C*T)
        # 1. Unfold 展开时间维度: [B, Num_Patches, Patch_Len, K]
        # 注意：unfold 作用在维度 1 (N_total)，类似于滑动窗口
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # 2. Flatten Patch 和 K 维度: [B, Num_Patches, Patch_Len * K]
        x = x.flatten(start_dim=-2) 
        # 3. Projection: [B, Num_Patches, D]
        x = self.patch_proj(x)
        if self.pos:
            x += self.pe(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.args = configs
        # 获取分解分量数，默认为 3 (High, Mid, Low)
        self.decomp_k = getattr(configs, 'decomp_k', 3) 
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # L

        # Filter Hyperparams
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # ============================================================
        # 关键修改 1: 使用 ComponentAwarePatchEmbed
        # ============================================================
        # 注意：这里我们传入 decomp_k，不再使用普通的 Linear
        self.patch_embed = ComponentAwarePatchEmbed(
            dim=self.dim, 
            patch_len=self.patch_len, 
            stride=self.stride, 
            pos=configs.pos, 
            decomp_k=self.decomp_k
        )

        # TimeFilter Backbone (保持不变)
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
                                            self.seq_len * self.n_vars // self.patch_len)

        # Prediction Heads
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(self.dim * self.num_patches, self.seq_len)
        elif self.task_name == 'classification':
            self.num_patches = int((self.seq_len * configs.enc_in - self.patch_len) / self.stride + 1)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.dim * self.num_patches, configs.num_class)

        # Normalization (不使用RevIN，使用手动归一化)
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

    def _get_mask(self, device):
        # 生成 TimeFilter 需要的掩码 (逻辑保持不变)
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
        # x_decomp input shape: [B, T, C, K]
        B, T, C, K = x_decomp.shape
        
        # ==========================================
        # 1. 归一化 (保留分量相对幅度)
        # ==========================================
        # 还原原始信号用于计算统计量
        x_raw = x_decomp.sum(dim=-1)  # [B, T, C]
        
        # 计算均值和标准差
        mean = x_raw.mean(dim=1, keepdim=True)  # [B, 1, C]
        std = x_raw.std(dim=1, keepdim=True)    # [B, 1, C]
        
        # 扩展维度以适配 K
        mean_k = mean.unsqueeze(-1) # [B, 1, C, 1]
        std_k = std.unsqueeze(-1)   # [B, 1, C, 1]
        
        # 归一化所有分量
        x_decomp = (x_decomp - mean_k) / (std_k + 1e-5)

        # ==========================================
        # 2. 维度适配 ComponentAwarePatchEmbed
        # ==========================================
        # 目标输入形状: [B, N_total, K] 其中 N_total = C * T
        # 原始 TimeFilter 将多变量展平处理，我们也遵循此逻辑
        
        # [B, T, C, K] -> [B, C, T, K]
        x = x_decomp.permute(0, 2, 1, 3)
        
        # [B, C, T, K] -> [B, C*T, K]
        x = x.reshape(B, C * T, K)

        # ==========================================
        # 3. Component-Aware Embedding
        # ==========================================
        # 输入: [B, C*T, K]
        # 输出: [B, Num_Patches, D] (已融合 K 个分量)
        x = self.patch_embed(x) 

        # ==========================================
        # 4. Backbone (TimeFilter 核心)
        # ==========================================
        # 输入: [B, Num_Patches, D]
        # 此时 x 已经是标准的 d_model 维度，Backbone 无需知道 K 的存在
        x, moe_loss = self.backbone(x, self._get_mask(x.device), self.alpha)

        # ==========================================
        # 5. Prediction Head
        # ==========================================
        # [B, C, T/P, D]
        x = x.reshape(B, self.n_vars, self.num_patches, self.dim)
        # Flatten for Linear Head: [B, C, Num_Patches * D]
        x = x.flatten(start_dim=-2)
        # Linear Projection: [B, C, Pred_Len]
        x = self.head(x)
        # [B, Pred_Len, C]
        x = x.permute(0, 2, 1)

        # ==========================================
        # 6. 反归一化
        # ==========================================
        # 预测结果是原始序列 (Sum of components)，直接用之前的 mean/std 反归一化
        x = x * std + mean
        
        return (x, moe_loss)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, moe_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return (dec_out[:, -self.pred_len:, :], moe_loss)
        return None