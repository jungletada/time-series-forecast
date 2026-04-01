import os
import random
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange
# from utils.tools import plot_mat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


class FullAttention_ablation(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, SF_mode=1, softmax_flag=1, weight_plus=0, outside_softmax=0,
                 plot_mat_flag=False, save_folder='./', plot_grad_flag=False):  # './utils/corr_mat/traffic.npy'
        super(FullAttention_ablation, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.SF_mode = SF_mode
        self.softmax_flag = softmax_flag
        self.token_num = token_num
        self.outside_softmax = outside_softmax  # False  #
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.plot_grad_flag = plot_grad_flag
        self.save_folder = os.path.join(save_folder)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention_ablation: {self.weight_plus}')
        print(f'self.softmax_flag in FullAttention_ablation: {self.softmax_flag}')
        print(f'self.outside_softmax in FullAttention_ablation: {self.outside_softmax}')

        if not self.SF_mode:
            print('Vanilla attention is used...')
        else:
            print('Enhanced attention is used...')

        if self.SF_mode and self.token_num is not None:
            # [1,1,N,1]
            if self.softmax_flag:
                self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

            init_weight_mat = (torch.eye(self.token_num) * 1.0 +
                               torch.randn(self.token_num, self.token_num) * 1.0)
            # ablation
            # init_weight_mat = (torch.eye(self.token_num) * 0.0 +
            #                    torch.randn(self.token_num, self.token_num) * 1.0)
            self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # this_device = queries.device

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        ori_attn_mat = None
        if self.SF_mode:
            if not self.training and self.plot_mat_flag:
                ori_attn_mat = torch.softmax(A, dim=-1)

        # attention matrix adjustment; 240507
        if self.SF_mode and self.token_num is not None:
            # 2d
            if self.softmax_flag:
                weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
            else:
                # use scale or not
                weight_mat = F.softplus(self.weight_mat)  # / sqrt(self.token_num)

        if self.SF_mode and weight_mat is not None:
            if self.outside_softmax:
                if self.weight_plus:
                    A = A.softmax(dim=-1) + weight_mat
                else:
                    A = A.softmax(dim=-1) * weight_mat
                A = F.normalize(A, p=1, dim=-1)
            else:
                if self.weight_plus:
                    A = A + weight_mat
                else:
                    # ablation: this is a better choice
                    A = A * weight_mat

                # attention matrix [b,h,l,s]
                A = torch.softmax(A, dim=-1)

        else:
            A = torch.softmax(A, dim=-1)

        # # plot mat
        # if not self.training and self.plot_mat_flag and random.random() < 1:  # and A.shape[-1] > 10
        #     batch_idx = random.randint(0, A.shape[0] - 1)
        #     head_idx = random.randint(0, A.shape[1] - 1)

        #     # final
        #     att_mat_2d = A[batch_idx, head_idx, :, :]
        #     time_or_channel = 'channel'
        #     plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
        #              save_folder=self.save_folder)

        #     if ori_attn_mat is not None and self.SF_mode:
        #         ori_att_mat_2d = ori_attn_mat[batch_idx, head_idx, :, :]
        #         time_or_channel = 'channel'
        #         plot_mat(ori_att_mat_2d, str_cat='ori_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_"
        #                                                               f"{time_or_channel}",
        #                  save_folder=self.save_folder)

        #     if weight_mat is not None and self.SF_mode:
        #         batch_idx2 = min(batch_idx, weight_mat.shape[0] - 1)
        #         head_idx2 = min(head_idx, weight_mat.shape[1] - 1)
        #         weight_mat_2d = weight_mat[batch_idx2, head_idx2, :, :]

        #         plot_mat(weight_mat_2d, str_cat='adding_mat_2d',
        #                  str0=f"batch_{batch_idx2}_head_{head_idx2}_{time_or_channel}",
        #                  save_folder=self.save_folder)

        #     print(f'Attention matrix in FullAttention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # # plot gradient
        # if self.plot_grad_flag and random.random() < 0.01 and self.weight_mat.grad is not None:
        #     batch_idx = random.randint(0, self.weight_mat.shape[0] - 1)
        #     head_idx = random.randint(0, self.weight_mat.shape[1] - 1)

        #     # final
        #     weight_mat_grad = self.weight_mat.grad[batch_idx, head_idx, :, :]
        #     time_or_channel = 'channel'
        #     plot_mat(weight_mat_grad, str_cat='weight_grad_mat',
        #              str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
        #              save_folder=self.save_folder)

        #     print(f'Attention gradient matrix in FullAttention has been saved to {self.save_folder}...')

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

