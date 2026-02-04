import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding


class DecompInputAdapterv1(nn.Module):
    """
    通用时序分解适配器 (Universal Decomposition Input Adapter)
    
    能够将形状为 [B, T, C, K] 的多变量时序分解信号，转换为标准深度学习模型
    可接受的 Embedding 格式。
    
    Modes:
      - 'patch': 适用于 Patch-based 模型 (如 TimeFilter, PatchTST)。
                 输出: [B * C, Num_Patches, d_model]
      - 'timestep': 适用于 Point-wise 模型 (如 LSTM, Transformer, Linear)。
                 输出: [B, T, C, d_model] 或 [B, T, C * d_model] (取决于后续reshape)
    """
    def __init__(self, 
                 d_model,        # 目标嵌入维度
                 patch_len=16,   # Patch 长度 (如果是 timestep 模式设为 1)
                 stride=None,    # Patch 步长
                 decomp_k=3,     # 分解分量数量 (K)
                 dropout=0.1, 
                 pos_embed=True, # 是否在 Adapter 内部加位置编码
                 mode='patch'    # 'patch' or 'timestep'
                 ):
        super().__init__()
        self.mode = mode
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.decomp_k = decomp_k
        self.d_model = d_model
        self.pos_embed = pos_embed
        
        print(f">>>>> Init DecompInputAdapter: mode={mode}, patch_len={patch_len}, k={decomp_k}, d_model={d_model}")

        # ============================================================
        # 1. 独立投影 (Decoupled Projection / MLP)
        # 针对每个分量 k，学习从 patch_len 到 d_model 的映射
        # ============================================================
        self.component_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_len, d_model),
                nn.Dropout(dropout)
            ) for _ in range(decomp_k)
        ])

        # ============================================================
        # 2. 动态门控网络 (Dynamic Gating Network - Expand-Reduce)
        # 输入维度: d_model * K
        # 作用: 动态决定每个分量的重要性
        # ============================================================
        input_dim = d_model * decomp_k
        
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim, decomp_k),
            nn.GELU(),
        )
        self.softmax = nn.Softmax(dim=-1)

        # 位置编码 (可选)
        if self.pos_embed:
            self.pe = PositionalEmbedding(d_model, max_len=5000)

    def forward(self, x):
        """
        Input x: [B, T, C, K] - 原始分解信号
        Output:  [B * C, Num_Patches, d_model] (if mode='patch')
        """
        B, T, C, K = x.shape
        assert K == self.decomp_k, f"Input component count {K} mismatch with initialized {self.decomp_k}"

        # -------------------------------------------------------
        # Step 1: 维度准备
        # 将 B 和 C 合并处理，视为独立的单变量序列 (Channel Independent)
        # [B, T, C, K] -> [B, C, T, K] -> [B * C, T, K]
        # -------------------------------------------------------
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, K)

        # -------------------------------------------------------
        # Step 2: Patching / Unfold
        # [B*C, T, K] -> [B*C, Num_Patches, K, Patch_Len]
        # -------------------------------------------------------
        # 注意：unfold 作用于 dim 1 (Time)，输出把 patch 维放到最后
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # -------------------------------------------------------
        # Step 3: 独立投影 (Independent Projection)
        # -------------------------------------------------------
        embeddings_list = []
        for k in range(self.decomp_k):
            # 取出第 k 个分量: [B*C, Num_Patches, Patch_Len]
            component_patch = x_patched[:, :, k, :] 
            
            # 投影: -> [B*C, Num_Patches, d_model]
            emb = self.component_projs[k](component_patch)
            embeddings_list.append(emb)

        # Stack: [B*C, Num_Patches, K, d_model]
        stacked_emb = torch.stack(embeddings_list, dim=2)

        # -------------------------------------------------------
        # Step 4: 动态门控与融合 (Dynamic Gating & Fusion)
        # -------------------------------------------------------
        # Context: [B*C, Num_Patches, K * d_model]
        context = stacked_emb.flatten(start_dim=2)
        
        # Weights: [B*C, Num_Patches, K, 1]
        weights = self.softmax(self.gating_net(context)).unsqueeze(-1)
        
        # Weighted Sum: [B*C, Num_Patches, d_model]
        x_final = (stacked_emb * weights).sum(dim=2)

        # -------------------------------------------------------
        # Step 5: 位置编码 & 输出
        # -------------------------------------------------------
        if self.pos_embed:
            x_final = x_final + self.pe(x_final)          #  [B * C, Num_Patches, D]
            
        x_final = x_final.reshape(B, C, -1, self.d_model) # [B, C, Num_Patches, D]
        return x_final

class DecompInputAdapterv2(nn.Module):
    def __init__(self, 
                 d_model,        
                 patch_len=16,   
                 stride=None,    
                 decomp_k=3,     
                 dropout=0.1, 
                 pos_embed=True, 
                 mode='patch'    
                 ):
        super().__init__()
        self.mode = mode
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.decomp_k = decomp_k
        self.d_model = d_model
        self.pos_embed = pos_embed
        
        print(f">>>>> Init DecompInputAdapter: mode={mode}, patch_len={patch_len}, k={decomp_k}, d_model={d_model}")

        # ============================================================
        # 1. 独立投影 (Decoupled Projection / MLP)
        # ============================================================
        self.component_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_len, d_model),
                nn.LayerNorm(d_model), # [新增] 稳定分布
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(decomp_k)
        ])

        # ============================================================
        # 2. 动态门控网络 (Dynamic Gating Network - Expand-Reduce)
        # 修复：增加了中间层，增强非线性拟合能力
        # ============================================================
        input_dim = d_model * decomp_k
        hidden_dim = d_model // 2 # 或者 input_dim // 2
        
        self.gating_net = nn.Sequential(
            nn.LayerNorm(input_dim),        # [新增] 输入归一化
            nn.Linear(input_dim, hidden_dim), # [修改] 先降维/升维提取特征
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decomp_k)   # [修改] 再映射到权重
        )
        self.softmax = nn.Softmax(dim=-1)

        if self.pos_embed:
            self.pe = PositionalEmbedding(d_model, max_len=5000)

    def forward(self, x):
        B, T, C, K = x.shape
        assert K == self.decomp_k, f"Input component count {K} mismatch"

        # 1. Channel Independent Flattening
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, K)

        # 2. Patching
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # 3. Independent Projection
        embeddings_list = []
        for k in range(self.decomp_k):
            component_patch = x_patched[:, :, k, :] 
            emb = self.component_projs[k](component_patch)
            embeddings_list.append(emb)

        # [B*C, N, K, D]
        stacked_emb = torch.stack(embeddings_list, dim=2)

        # 4. Dynamic Gating
        context = stacked_emb.flatten(start_dim=2) # [B*C, N, K*D]
        
        # [关键] 这一步现在通过两层 MLP 计算，能力更强
        weights = self.softmax(self.gating_net(context)).unsqueeze(-1) # [B*C, N, K, 1]
        
        # Weighted Fusion
        x_final = (stacked_emb * weights).sum(dim=2) # [B*C, N, D]

        # 5. Pos Encoding
        if self.pos_embed:
            x_final = x_final + self.pe(x_final)
            
        # Reshape to [B, C, N, D]
        x_final = x_final.reshape(B, C, -1, self.d_model)
        
        return x_final

class DecompInputAdapter(nn.Module):
    def __init__(self, 
                 d_model,        
                 patch_len=16,   
                 stride=None,    
                 decomp_k=3,     
                 dropout=0.1, 
                 pos_embed=True, 
                 mode='patch'    
                 ):
        super().__init__()
        self.mode = mode
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.decomp_k = decomp_k
        self.d_model = d_model
        self.pos_embed = pos_embed
        
        print(f">>>>> Init DecompInputAdapter (MEAN FUSION): mode={mode}, patch_len={patch_len}, k={decomp_k}, d_model={d_model}")

        # 1. 独立投影 (保留)
        # 我们依然需要将不同分量映射到相同的 d_model 维度，才能进行平均
        self.component_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_len, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(decomp_k)
        ])

        # ============================================================
        # [修改] 移除 Gating Network
        # 我们不再需要学习权重，所以注释掉这部分参数，节省显存并防止过拟合
        # ============================================================
        # input_dim = d_model * decomp_k
        # hidden_dim = d_model // 2
        # self.gating_net = nn.Sequential(
        #     nn.LayerNorm(input_dim),
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, decomp_k)
        # )
        # self.softmax = nn.Softmax(dim=-1)

        if self.pos_embed:
            self.pe = PositionalEmbedding(d_model, max_len=5000)

    def forward(self, x):
        B, T, C, K = x.shape
        assert K == self.decomp_k, f"Input component count {K} mismatch"

        # 1. Channel Independent Flattening
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, K)

        # 2. Patching
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # 3. Independent Projection
        embeddings_list = []
        for k in range(self.decomp_k):
            component_patch = x_patched[:, :, k, :] 
            emb = self.component_projs[k](component_patch)
            embeddings_list.append(emb)

        # [B*C, N, K, D]
        stacked_emb = torch.stack(embeddings_list, dim=2)

        # ============================================================
        # [修改] 融合策略：直接平均 (Mean Fusion)
        # 替代原本的 (stacked_emb * weights).sum(dim=2)
        # ============================================================
        
        # Dim 2 是 K 维度，求平均表示同等看待 Trend, Seasonality 和 Noise
        x_final = stacked_emb.mean(dim=2) # [B*C, N, D]

        # 5. Pos Encoding
        if self.pos_embed:
            x_final = x_final + self.pe(x_final)
            
        # Reshape to [B, C, N, D] (为了适配 Model 里的 reshape 逻辑)
        # 但为了保持接口一致性，这里还原成 Channel 独立的 N 形式
        x_final = x_final.reshape(B, C, -1, self.d_model)
        
        return x_final
    