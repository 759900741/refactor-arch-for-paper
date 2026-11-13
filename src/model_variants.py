# -*- coding: utf-8 -*-
"""
model_variants.py
=================
阶段六「消融 / 模型对比」所需的模型变体，统一 `forward(x)->(B, horizon)` 接口。

变体清单：
    bilstm_itransformer  —— 本文方法（BiLSTM 逐变量编码 + 变量维自注意力 + 残差头）
    bilstm               —— 纯 BiLSTM（多元时间序列 BiLSTM，无注意力）
    transformer          —— BiLSTM 前端 + 时序维 Transformer（论文 "BiLSTM-Transformer"）
    informer             —— BiLSTM 前端 + 时序维 Informer（ProbSparse 自注意力，"BiLSTM-Informer"）

另：iTransformer-only（无 BiLSTM 前端）与 均值池化头 通过主类
    BiLSTMiTransformer 的 `use_bilstm=False` / `head_mode="mean"` 开关实现，
    由 build_variant 的 cfg 注入。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from model_bilstm_itransformer import BiLSTMiTransformer


# ---------------------------------------------------------------------------
# 纯 BiLSTM（多元时序，特征拼接后过双向 LSTM）
# ---------------------------------------------------------------------------
class BiLSTMBaseline(nn.Module):
    def __init__(self, n_feat, horizon, hidden=32, d_model=32, dropout=0.1,
                 lstm_layers=1, target_idx=0, residual=False):
        super().__init__()
        self.n_feat = n_feat
        self.horizon = horizon
        self.target_idx = target_idx
        self.residual = residual
        self.in_proj = nn.Linear(n_feat, hidden)
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(4 * hidden, horizon)  # [mean; last] × 双向

    def forward(self, x):
        B, L, F = x.shape
        z = self.in_proj(x)                       # (B, L, hidden)
        h, _ = self.bilstm(z)                     # (B, L, 2*hidden)
        h = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)   # (B, 4*hidden)
        out = self.head(h)                        # (B, horizon)
        if self.residual:
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# 时序维 Transformer / Informer 共用件
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _BiLSTMFrontend(nn.Module):
    """BiLSTM 前端：把 (B,L,d_model) 在时间维用双向 LSTM 增强，输出 (B,L,d_model)。"""

    def __init__(self, d_model, hidden, num_layers=1, dropout=0.0):
        super().__init__()
        self.bilstm = nn.LSTM(d_model, hidden, num_layers=int(num_layers), batch_first=True,
                              bidirectional=True,
                              dropout=float(dropout) if int(num_layers) > 1 else 0.0)
        self.merge = nn.Linear(2 * hidden, d_model)

    def forward(self, z):
        h, _ = self.bilstm(z)        # (B, L, 2*hidden)
        return self.merge(h) + z     # 残差增强


class TransformerTemporal(nn.Module):
    """BiLSTM 前端 + 时序维 Transformer（标准自注意力在时间维）。"""

    def __init__(self, n_feat, horizon, hidden=32, d_model=32, n_heads=2, e_layers=1,
                 d_ff=64, dropout=0.1, lstm_layers=1, target_idx=0, residual=False):
        super().__init__()
        self.n_feat = n_feat
        self.horizon = horizon
        self.target_idx = target_idx
        self.residual = residual
        self.in_proj = nn.Linear(n_feat, d_model)
        self.front = _BiLSTMFrontend(d_model, hidden, num_layers=lstm_layers, dropout=dropout)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, L, F = x.shape
        z = self.in_proj(x)             # (B, L, d_model)
        z = self.front(z)               # (B, L, d_model)  + BiLSTM 时间维增强
        z = self.pos(z)
        z = self.encoder(z)             # (B, L, d_model)
        z = z[:, -1]                    # 取末步
        out = self.head(z)              # (B, horizon)
        if self.residual:
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# Informer（ProbSparse 自注意力，简化实现：输出语义正确，不优化计算量）
# ---------------------------------------------------------------------------
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)   # (B,h,L,d_k)
        k = self.wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scale = self.d_k ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale               # (B,h,L,L)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                          # (B,h,L,d_k)

        # 稀疏度度量 M(q_i) = max - mean，取 top-u 个 query 做完整注意力，其余用 V 均值替代
        M = scores.max(dim=-1)[0] - scores.mean(dim=-1)                      # (B,h,L)
        u = max(1, min(L, int(self.factor * math.log(L + 1))))
        topk = torch.topk(M, u, dim=-1).indices                              # (B,h,u)
        selected = torch.zeros(B, self.n_heads, L, device=x.device)
        selected.scatter_(2, topk, 1.0)
        v_mean = v.mean(dim=2, keepdim=True)                                 # (B,h,1,d_k)
        sel = selected.unsqueeze(-1)                                         # (B,h,L,1)
        out = out * sel + v_mean * (1 - sel)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(self.dropout(out))


class _ProbSparseEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class InformerTemporal(nn.Module):
    """BiLSTM 前端 + 时序维 Informer（ProbSparse 自注意力）。"""

    def __init__(self, n_feat, horizon, hidden=32, d_model=32, n_heads=2, e_layers=1,
                 d_ff=64, dropout=0.1, lstm_layers=1, target_idx=0, residual=False):
        super().__init__()
        self.n_feat = n_feat
        self.horizon = horizon
        self.target_idx = target_idx
        self.residual = residual
        self.in_proj = nn.Linear(n_feat, d_model)
        self.front = _BiLSTMFrontend(d_model, hidden, num_layers=lstm_layers, dropout=dropout)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(
            *[_ProbSparseEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, L, F = x.shape
        z = self.in_proj(x)
        z = self.front(z)
        z = self.pos(z)
        z = self.encoder(z)             # (B, L, d_model)
        z = z[:, -1]
        out = self.head(z)
        if self.residual:
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# 变体工厂（trainer.build_model 调用）
# ---------------------------------------------------------------------------
def build_variant(model_type, n_feat, horizon, cfg):
    h = cfg.get("hidden", 32)
    dm = cfg.get("d_model", 32)
    nh = cfg.get("n_heads", 2)
    el = cfg.get("e_layers", 1)
    dff = cfg.get("d_ff", 64)
    do = cfg.get("dropout", 0.1)
    ll = cfg.get("lstm_layers", 1)
    res = cfg.get("residual", True)
    tgt = cfg.get("target_idx", 0)
    hm = cfg.get("head_mode", "target")
    ub = cfg.get("use_bilstm", True)
    fu = cfg.get("fusion", "serial")
    mvh = cfg.get("mv_hidden")
    mvl = int(cfg.get("mv_layers", 1) or 1)

    if model_type == "bilstm_itransformer":
        return BiLSTMiTransformer(n_feat, horizon, hidden=h, d_model=dm, n_heads=nh,
                                  e_layers=el, d_ff=dff, dropout=do, lstm_layers=ll,
                                  target_idx=tgt, residual=res, head_mode=hm, use_bilstm=ub,
                                  fusion=fu, mv_hidden=mvh, mv_layers=mvl)
    if model_type == "bilstm":
        return BiLSTMBaseline(n_feat, horizon, hidden=h, d_model=dm, dropout=do,
                              lstm_layers=ll, target_idx=tgt, residual=res)
    if model_type == "transformer":
        return TransformerTemporal(n_feat, horizon, hidden=h, d_model=dm, n_heads=nh,
                                   e_layers=el, d_ff=dff, dropout=do, lstm_layers=ll,
                                   target_idx=tgt, residual=res)
    if model_type == "informer":
        return InformerTemporal(n_feat, horizon, hidden=h, d_model=dm, n_heads=nh,
                                e_layers=el, d_ff=dff, dropout=do, lstm_layers=ll,
                                target_idx=tgt, residual=res)
    raise ValueError(f"未知 model_type: {model_type}")


if __name__ == "__main__":
    torch.manual_seed(0)
    cfgs = dict(hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1,
                lstm_layers=1, residual=True, target_idx=0)
    for mt in ("bilstm_itransformer", "bilstm", "transformer", "informer"):
        m = build_variant(mt, 16, 4, cfgs)
        x = torch.randn(8, 96, 16)
        out = m(x)
        loss = out.sum()
        loss.backward()
        print(f"{mt:20s} out={tuple(out.shape)} params={sum(p.numel() for p in m.parameters()):6d}")
    # iTransformer-only 与 均值池化头
    m2 = build_variant("bilstm_itransformer", 16, 4, {**cfgs, "use_bilstm": False})
    m3 = build_variant("bilstm_itransformer", 16, 4, {**cfgs, "head_mode": "mean"})
    print(f"itransformer-only   out={tuple(m2(torch.randn(8, 96, 16)).shape)}")
    print(f"mean-head           out={tuple(m3(torch.randn(8, 96, 16)).shape)}")
