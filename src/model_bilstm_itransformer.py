# -*- coding: utf-8 -*-
"""
model_bilstm_itransformer.py
============================
BiLSTM-iTransformer 混合预测模型（PyTorch，兼容 Python 3.8 / torch 2.4.1）。

融合逻辑（两种融合方式，由 fusion 开关控制）：
  A. 串行分支（论文原始结构，fusion="serial"）
     1) BiLSTM：把每个变量各自的时间序列(L)编码为时序嵌入 -> 每个变量一个"token"；
     2) iTransformer：把所有变量当 token，在**变量维**做 self-attention，
        建模多源特征间的全局相关性（倒置 Transformer 思想）；
     3) 投影头：取目标变量 token（或变量维池化）输出未来 H 步。
  B. 并行多变量时序分支（fusion="parallel"|"gate"，本文补强）
     1) 多变量联合 BiLSTM：逐时间步混合全部变量后过 BiLSTM，直接建模跨变量短时轨迹；
     2) 与 A 的预测按步做凸组合（parallel）或逐维门控（gate）融合。
     该分支严格包含纯 BiLSTM 的建模路径，用于弥补串行结构在**短预测步长**下
     因"变量 token 只保留时序摘要"而损失近期动态的缺陷。

接口：forward(x)  x:(B, L, F)  ->  (B, horizon)
      forward_with_aux(x) -> (out, p_parallel)（仅有并行分支时返回两支预测，供辅助损失）
供阶段三（MIV 承载网络）、阶段四/五（最终模型）、阶段六（架构消融 B6）复用。
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BiLSTMiTransformer(nn.Module):
    def __init__(self, n_feat: int, horizon: int, hidden: int = 32,
                 d_model: int = 32, n_heads: int = 2, e_layers: int = 1,
                 d_ff: int = 64, dropout: float = 0.1, lstm_layers: int = 1,
                 target_idx: int = 0, residual: bool = False,
                 use_bilstm: bool = True, head_mode: str = "target",
                 fusion: str = "serial", mv_hidden: int | None = None,
                 mv_layers: int = 1):
        super().__init__()
        self.n_feat = n_feat
        self.horizon = horizon
        self.target_idx = target_idx
        self.residual = residual
        self.use_bilstm = use_bilstm
        self.head_mode = head_mode
        self.fusion = fusion

        if use_bilstm:
            # 1) 逐变量 BiLSTM：输入单变量序列 (B*F, L, 1)
            self.bilstm = nn.LSTM(
                input_size=1, hidden_size=hidden, num_layers=lstm_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            # 2) 时序嵌入 -> 变量 token（均值池化 + 末步拼接，保留近期动态）
            self.embed = nn.Linear(4 * hidden, d_model)
        else:
            # iTransformer-only：不含 BiLSTM 前端，用逐时间步线性投影 + 时序均值池化生成变量 token
            # （不依赖 L 的具体取值，便于复用于 96/24 两种窗口）
            self.bilstm = None
            self.tok_proj = nn.Linear(1, d_model)
        self.var_pos = nn.Parameter(torch.zeros(1, n_feat, d_model))
        nn.init.normal_(self.var_pos, mean=0.0, std=0.02)

        # 3) iTransformer 编码器（变量维 self-attention）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        # 4) 输出头
        self.head = nn.Linear(d_model, horizon)

        # 5) 并行多变量时序分支（仅 fusion != "serial" 时创建，保证旧 state_dict 键集合不变）
        self.mv_bilstm = None
        if fusion in ("parallel", "gate"):
            h_mv = int(mv_hidden or hidden)
            self.mv_proj = nn.Linear(n_feat, h_mv)
            self.mv_bilstm = nn.LSTM(
                input_size=h_mv, hidden_size=h_mv, num_layers=int(mv_layers),
                batch_first=True, bidirectional=True, dropout=dropout if mv_layers > 1 else 0.0,
            )
            self.mv_head = nn.Linear(4 * h_mv, horizon)
            if fusion == "gate":
                self.gate = nn.Linear(2 * horizon, horizon)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)   # 初始 g=0.5，两支等权起步

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """把 (B, L, F) 编码为 (B, F, d_model) 的变量 token。"""
        B, L, F = x.shape
        if self.use_bilstm:
            s = x.permute(0, 2, 1).contiguous().reshape(B * F, L, 1)  # (B*F, L, 1)
            h, _ = self.bilstm(s)                                      # (B*F, L, 2H)
            h = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)        # (B*F, 4H)
            tok = self.embed(h).reshape(B, F, -1)                      # (B, F, d_model)
        else:
            # iTransformer-only：逐变量把 L 步序列经线性层映射到 d_model
            s = x.permute(0, 2, 1).contiguous().reshape(B * F, L)      # (B*F, L)
            # 分 L 步线性投影再平均（等价于对序列做全局线性组合 + 均值）
            tok = self.tok_proj(s.unsqueeze(-1)).reshape(B * F, L, -1).mean(dim=1)
            tok = tok.reshape(B, F, -1)                                 # (B, F, d_model)
        return tok + self.var_pos[:, :F, :]

    def _serial_head(self, x: torch.Tensor) -> torch.Tensor:
        """串行分支：变量 token -> 变量维注意力 -> 目标 token -> 输出 (B, horizon)。"""
        tok = self._tokenize(x)               # (B, F, d_model)
        z = self.encoder(tok)                 # (B, F, d_model)
        if self.head_mode == "mean":
            z = self.norm(z.mean(dim=1))      # 变量维均值池化头
        else:
            z = self.norm(z[:, self.target_idx])  # 目标变量 token
        return self.head(z)                   # (B, horizon)

    def _parallel_head(self, x: torch.Tensor) -> torch.Tensor:
        """并行分支：多变量联合 BiLSTM -> [mean; last] -> 输出 (B, horizon)。"""
        h, _ = self.mv_bilstm(self.mv_proj(x))                   # (B, L, 2*h_mv)
        f = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)       # (B, 4*h_mv)
        return self.mv_head(f)                                   # (B, horizon)

    def _fuse(self, p_s: torch.Tensor, p_p: torch.Tensor) -> torch.Tensor:
        if self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([p_s, p_p], dim=-1)))  # (B, horizon)
            return g * p_s + (1.0 - g) * p_p
        return 0.5 * (p_s + p_p)

    def forward_with_aux(self, x: torch.Tensor):
        """返回 (融合预测, 并行分支预测)。无并行分支时 aux 为 None。"""
        p_s = self._serial_head(x)
        if self.mv_bilstm is None:
            out = p_s
            aux = None
        else:
            p_p = self._parallel_head(x)
            out = self._fuse(p_s, p_p)
            aux = p_p
        if self.residual:
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_aux(x)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for fu in ("serial", "parallel", "gate"):
        m = BiLSTMiTransformer(n_feat=22, horizon=4, fusion=fu, residual=True)
        x = torch.randn(8, 96, 22)
        out, aux = m.forward_with_aux(x)
        out.sum().backward()
        print(f"fusion={fu:9s} out={tuple(out.shape)} aux={None if aux is None else tuple(aux.shape)} "
              f"params={sum(p.numel() for p in m.parameters())}")
    # 兼容性：serial 模式的键集合不含并行分支
    a = BiLSTMiTransformer(22, 4, fusion="serial")
    b = BiLSTMiTransformer(22, 4, fusion="gate")
    ka, kb = set(a.state_dict()), set(b.state_dict())
    print("serial-only keys:", sorted(ka - kb), "| gate-only keys:", sorted(kb - ka)[:5], "...")
