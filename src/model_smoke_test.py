# -*- coding: utf-8 -*-
"""
model_smoke_test.py
===================
阶段四「入口校验」的算力可行性探针（必须在 Python 3.8 + torch 环境执行）。

做四件事并把结果落盘为 JSON，供 3.13 环境的 stage_gate.py 读取：
  1. 模型可实例化、可前向、可反向（梯度非空）；
  2. 支持 config 字典注入（阶段五 TE-RIME 需要按超参向量重建模型）；
  3. 实测单 batch 耗时 -> 估算每个 epoch / 整阶段训练时长，判断 CPU 预算可行性；
  4. 对比「全特征 vs 选中特征」的参数量，量化特征筛选带来的结构收益。

输出：reports/gates/model_smoke.json
运行：cd src && "<py3.8>" model_smoke_test.py
"""
from __future__ import annotations
import os
import sys
import json
import time
import platform
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_bilstm_itransformer import BiLSTMiTransformer

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "reports", "gates")
os.makedirs(OUT, exist_ok=True)

# 阶段四拟采用的容量（比阶段三承载网络更强，但保持 CPU 可行）
CAP = dict(hidden=32, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1)
BATCH = 128
PLAN_EPOCHS = {"jingmen": 30, "lanzhou": 30}


def variants_smoke():
    """阶段六变体模型冒烟：供 stage_gate 的 validate_stage6_entry 读取。"""
    from model_variants import build_variant
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1,
               lstm_layers=1, residual=True, target_idx=0)
    cases = {
        "bilstm": ("bilstm", {}),
        "transformer": ("transformer", {}),
        "informer": ("informer", {}),
        "itransformer_only": ("bilstm_itransformer", {"use_bilstm": False}),
        "mean_head": ("bilstm_itransformer", {"head_mode": "mean"}),
        "parallel": ("bilstm_itransformer", {"fusion": "parallel"}),
        "gate": ("bilstm_itransformer", {"fusion": "gate"}),
    }
    out = {}
    for k, (mt, over) in cases.items():
        try:
            m = build_variant(mt, 22, 4, {**cfg, **over}).to(dev)
            x = torch.randn(8, 96, 22, device=dev)
            y = torch.randn(8, 4, device=dev)
            pred = m(x)
            torch.nn.functional.mse_loss(pred, y).backward()
            ok = tuple(pred.shape) == (8, 4) and all(
                p.grad is not None for p in m.parameters() if p.requires_grad)
            out[k] = dict(ok=bool(ok), params=int(sum(p.numel() for p in m.parameters())),
                          out_shape=list(pred.shape), device=dev)
        except Exception as e:  # noqa: BLE001
            out[k] = dict(ok=False, error=str(e)[:200], device=dev)
    out["_meta"] = dict(device=dev, cuda=torch.cuda.is_available(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        torch=torch.__version__)
    with open(os.path.join(OUT, "model_variants_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("  变体冒烟: " + ", ".join(f"{k}={v.get('ok')}" for k, v in out.items() if not k.startswith("_")))
    return out


def main():
    torch.set_num_threads(4)
    rep = dict(python=platform.python_version(), torch=torch.__version__,
               threads=int(torch.get_num_threads()), cap=CAP, batch=BATCH, items={})

    for name in ("jingmen", "lanzhou"):
        st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        L, H, stride = st["L"], st["H"], st["stride"]
        n_tr = st["windows"]["train"]["X_shape"][0]
        dsel = np.load(os.path.join(P, "miv", f"{name}_train_sel.npz"), allow_pickle=True)
        F_sel = dsel["X"].shape[2]
        F_all = st["n_features"]

        model = BiLSTMiTransformer(F_sel, H, **CAP)
        n_par = sum(p.numel() for p in model.parameters())
        m_all = BiLSTMiTransformer(F_all, H, **CAP)
        n_par_all = sum(p.numel() for p in m_all.parameters())

        x = torch.randn(BATCH, L, F_sel)
        y = torch.randn(BATCH, H)
        # 前向 + 反向（含一次优化器步，模拟真实开销）
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ts = []
        for _ in range(3):          # 前 2 次预热，取第 3 次
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            ts.append(time.perf_counter())
        per_batch = ts[-1] - ts[-2]
        nb = int(np.ceil(n_tr / BATCH))
        ep_sec = per_batch * nb
        grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)

        rep["items"][name] = dict(
            L=int(L), H=int(H), F_sel=int(F_sel), F_all=int(F_all), n_train=int(n_tr),
            params_sel=int(n_par), params_all=int(n_par_all),
            param_drop=round(1 - n_par / n_par_all, 4),
            batches_per_epoch=int(nb), sec_per_batch=round(float(per_batch), 4),
            est_epoch_sec=round(float(ep_sec), 1),
            plan_epochs=int(PLAN_EPOCHS[name]),
            est_total_min=round(float(ep_sec * PLAN_EPOCHS[name]) / 60.0, 1),
            grad_ok=bool(grad_ok),
            out_shape=list(model(x).shape),
            backend="cpu",
        )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep["backend"] = dev
    with open(os.path.join(OUT, "model_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    variants_smoke()

    print(f"python={rep['python']} torch={rep['torch']} threads={rep['threads']}")
    for k, v in rep["items"].items():
        print(f"  {k}: Fsel={v['F_sel']}/{v['F_all']} params={v['params_sel']}(-{v['param_drop']*100:.1f}%) "
              f"per_batch={v['sec_per_batch']}s epoch={v['est_epoch_sec']}s "
              f"plan={v['plan_epochs']}ep -> {v['est_total_min']}min grad_ok={v['grad_ok']}")
    print(f"报告: {os.path.relpath(os.path.join(OUT, 'model_smoke.json'), BASE)}")


if __name__ == "__main__":
    main()
