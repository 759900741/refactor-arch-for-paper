# -*- coding: utf-8 -*-
"""
stage3_miv.py
=============
阶段三：MIV 特征筛选（严格对齐论文 Eq.9–14）。

流程：
  3.1 载入/对齐（npz+json；不依赖 pandas）
  3.2 共享模型 BiLSTM-iTransformer
  3.3 训练两个承载网络 F_ori(原始X) / F_pca(RPCA 的 L)
  3.4 MIV：X1=X(1+δ), X2=X(1-δ), Miv=(1/K)Σ|F(X1)-F(X2)|
  3.5 α、θ（Eq.12）
  3.6 在 RPCA 数据上按 |Miv_pca|≥θ 选择（Eq.13-14），强制保留目标列
  3.7 生成选中特征的下游滑窗 Xsel
  3.8 图 + 统计 JSON

仅用 numpy/json/torch/matplotlib（Python 3.8 + torch cpu）。
运行：cd src && "<py3.8>" stage3_miv.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_bilstm_itransformer import BiLSTMiTransformer

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
W = os.path.join(P, "windows")
RP = os.path.join(P, "rpca")
MV = os.path.join(P, "miv")
FIG = os.path.join(BASE, "reports", "figs")
os.makedirs(MV, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---- 超参（承载网络：轻量，CPU 可行）----
SEED = 42
DELTA = 0.1           # 扰动系数 δ
MAX_TRAIN = 3000      # 承载网络子采样上限
EPOCHS = 20
PATIENCE = 5
BATCH = 128
LR = 1e-3
HIDDEN = 32
K_MIV = 800           # 计算 MIV 的样本数

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

CONFIG = {
    "jingmen": dict(label="荆门市2024", target="负荷"),
    "lanzhou": dict(label="兰州市2023", target="electricity(KW)"),
}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_windows(M, L, H, stride):
    """与阶段一完全一致的滑窗；返回 Xw (N,L,F), ends。"""
    N = len(M) - L - H + 1
    idx = list(range(0, N, stride))
    Xw = np.stack([M[i:i + L] for i in idx], axis=0).astype(np.float32)
    ends = np.array(idx) + L - 1
    return Xw, ends


def _device():
    s = os.environ.get("WB_DEVICE", "auto")
    if s == "auto":
        s = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(s)


def train_carrier(Xtr, ytr, Xval, yval, horizon, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = _device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    nfeat = Xtr.shape[2]
    model = BiLSTMiTransformer(nfeat, horizon, hidden=HIDDEN, d_model=HIDDEN,
                               n_heads=2, e_layers=1, d_ff=HIDDEN * 2, dropout=0.1,
                               target_idx=0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()

    # 子采样控制 CPU 成本
    if len(Xtr) > MAX_TRAIN:
        ridx = np.random.RandomState(seed).permutation(len(Xtr))[:MAX_TRAIN]
        Xtr, ytr = Xtr[ridx], ytr[ridx]

    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xval_t = torch.tensor(Xval, device=device); yval_t = torch.tensor(yval, device=device)
    n = len(Xtr_t)
    hist = dict(train_loss=[], val_rmse=[])
    best = float("inf"); best_state = None; wait = 0; ep_used = 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0; nb = 0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            pv = model(Xval_t)
            vrmse = float(torch.sqrt(torch.mean((pv - yval_t) ** 2)).item())
        hist["train_loss"].append(tot / max(nb, 1)); hist["val_rmse"].append(vrmse)
        ep_used = ep + 1
        if vrmse < best - 1e-6:
            best = vrmse; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist, best, ep_used


@torch.no_grad()
def compute_miv(model, X, delta, K, bs=BATCH):
    """Eq.9-11：逐特征乘性扰动，Miv=(1/K)Σ|F(X1)-F(X2)|。"""
    device = next(model.parameters()).device
    Xs = torch.tensor(X[:K], dtype=torch.float32, device=device)
    F = Xs.shape[2]
    miv = np.zeros(F, dtype=np.float64)
    for f in range(F):
        xp = Xs.clone(); xp[:, :, f] = xp[:, :, f] * (1 + delta)
        xm = Xs.clone(); xm[:, :, f] = xm[:, :, f] * (1 - delta)
        tot = 0.0; cnt = 0
        for i in range(0, len(Xs), bs):
            yp = model(xp[i:i + bs]); ym = model(xm[i:i + bs])
            d = (yp - ym).abs().mean(dim=1)   # 对 H 求均 -> 每样本影响
            tot += float(d.sum().item()); cnt += len(d)
        miv[f] = tot / max(cnt, 1)
    return miv


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def process(name, cfg):
    print(f"\n{'='*66}\n阶段三 MIV：{cfg['label']} ({name})")
    st = load_json(os.path.join(P, f"{name}_stats.json"))
    feat = st["feature_cols"]; L = st["L"]; H = st["H"]; stride = st["stride"]
    target = cfg["target"]; tgt_idx = feat.index(target)
    F = len(feat)
    print(f"  feature={F}, target={target}@{tgt_idx}, L={L}, H={H}, stride={stride}")

    # 3.1 载入原始窗口 + 用 L 重建 RPCA 窗口
    Xo, Xp, y = {}, {}, {}
    for s in ("train", "val", "test"):
        d = np.load(os.path.join(W, f"{name}_{s}.npz"))
        Xo[s], y[s] = d["X"], d["y"]
        dl = np.load(os.path.join(RP, f"{name}_L_{s}.npz"))
        Lmat = dl["L"]; cols = [str(c) for c in dl["cols"]]
        assert cols == feat, "RPCA 列序与 feature_cols 不一致"
        Xp[s], _ = build_windows(Lmat, L, H, stride)
        assert Xp[s].shape[0] == Xo[s].shape[0], \
            f"{s} 窗口数不对齐: pca={Xp[s].shape[0]} vs ori={Xo[s].shape[0]}"
    print(f"  对齐完成: Xo/Xp train={Xo['train'].shape}, y={y['train'].shape}")

    # 3.3+3.4 训练承载网络 + MIV（带缓存：向量已算过时跳过训练，便于重调阈值）
    cache = os.path.join(MV, f"{name}_miv_cache.npz")
    kk = min(K_MIV, Xo["train"].shape[0])
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        miv_ori, miv_pca = d["miv_ori"], d["miv_pca"]
        hist_o = d["hist_o"].item(); hist_p = d["hist_p"].item()
        v_o = float(d["v_o"]); v_p = float(d["v_p"])
        ep_o = int(d["ep_o"]); ep_p = int(d["ep_p"])
        print(f"  命中 MIV 缓存，跳过训练与扰动计算 (F_ori={v_o:.4f}, F_pca={v_p:.4f})")
    else:
        torch.manual_seed(SEED)
        m_ori, hist_o, v_o, ep_o = train_carrier(Xo["train"], y["train"], Xo["val"], y["val"], H, SEED)
        print(f"  F_ori 训练: val_rmse={v_o:.4f} ({ep_o} epochs)")
        m_pca, hist_p, v_p, ep_p = train_carrier(Xp["train"], y["train"], Xp["val"], y["val"], H, SEED + 1)
        print(f"  F_pca 训练: val_rmse={v_p:.4f} ({ep_p} epochs)")
        miv_ori = compute_miv(m_ori, Xo["train"], DELTA, kk)
        miv_pca = compute_miv(m_pca, Xp["train"], DELTA, kk)
        print(f"  MIV 计算完成 (K={kk}); Miv_ori max={miv_ori.max():.5f}, Miv_pca max={miv_pca.max():.5f}")
        np.savez_compressed(cache, miv_ori=miv_ori, miv_pca=miv_pca,
                            hist_o=np.array(hist_o, dtype=object),
                            hist_p=np.array(hist_p, dtype=object),
                            v_o=v_o, v_p=v_p, ep_o=ep_o, ep_p=ep_p)

    # 3.5 α, θ（Eq.12 原式；统计范围排除目标列——它按 Eq.14 强制保留，
    #     且其对预测的扰动影响远大于其他特征，纳入会严重抬高/扭曲阈值）
    ex = np.ones(len(feat), bool); ex[tgt_idx] = False
    alpha = (np.abs(miv_ori[ex]).max() /
             (np.abs(miv_pca[ex]).max() + np.abs(miv_ori[ex]).max() + 1e-12))
    theta = alpha * miv_pca[ex].mean() + (1 - alpha) * miv_ori[ex].mean()
    print(f"  α={alpha:.4f}, θ={theta:.6f} (排除目标列; Miv_pca 区间 [{miv_pca[ex].min():.5f},{miv_pca[ex].max():.5f}])")

    # 3.6 选择（Eq.13-14），强制保留目标列
    sel_mask = np.abs(miv_pca) >= theta
    sel_mask[tgt_idx] = True
    note = "阈值θ(排除目标列统计)"
    if sel_mask.sum() < 3:
        thr2 = np.median(np.abs(miv_pca))
        sel_mask = np.abs(miv_pca) >= thr2; sel_mask[tgt_idx] = True
        note = "选中<3，改用中位数阈值补足"
    sel_idx = np.where(sel_mask)[0].tolist()
    sel_cols = [feat[i] for i in sel_idx]
    print(f"  选中 {len(sel_idx)}/{len(feat)} 个特征 ({note})")

    # 3.7 生成下游选中数据（来自 RPCA 的 L 窗口）
    for s in ("train", "val", "test"):
        Xsel = Xp[s][:, :, sel_idx].astype(np.float32)
        np.savez_compressed(os.path.join(MV, f"{name}_{s}_sel.npz"),
                            X=Xsel, y=y[s].astype(np.float32),
                            cols=np.array(sel_cols), L=L, H=H)
        Xselo = Xo[s][:, :, sel_idx].astype(np.float32)
        np.savez_compressed(os.path.join(MV, f"{name}_{s}_selorig.npz"),
                            X=Xselo, y=y[s].astype(np.float32),
                            cols=np.array(sel_cols), L=L, H=H)
    print(f"  下游数据已保存: {name}_{{train,val,test}}_sel.npz (Fsel={len(sel_idx)})")

    # 3.8 统计 + 图
    stats = dict(
        dataset=name, label=cfg["label"], target=target, tgt_idx=int(tgt_idx),
        n_feat=int(F), L=int(L), H=int(H), stride=int(stride),
        delta=DELTA, K_miv=int(kk), alpha=float(alpha), theta=float(theta),
        theta_scope="excl_target", select_note=note,
        carrier=dict(val_rmse_ori=float(v_o), val_rmse_pca=float(v_p),
                     epochs_ori=int(ep_o), epochs_pca=int(ep_p), hidden=HIDDEN,
                     max_train=MAX_TRAIN, batch=BATCH, lr=LR),
        miv_ori={feat[i]: float(miv_ori[i]) for i in range(F)},
        miv_pca={feat[i]: float(miv_pca[i]) for i in range(F)},
        selected=sel_cols, n_selected=int(len(sel_idx)),
    )
    with open(os.path.join(MV, f"{name}_miv_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    make_figs(name, cfg["label"], feat, miv_ori, miv_pca, theta, sel_mask,
              hist_o, hist_p, tgt_idx)
    print(f"  {cfg['label']} 完成。")
    return stats


def make_figs(name, label, feat, miv_ori, miv_pca, theta, sel_mask, ho, hp, tgt_idx):
    # MIV 柱状
    order = np.argsort(-miv_pca)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(feat))))
    yv = np.arange(len(feat))
    ax.barh(yv, miv_pca[order], color=["#c53030" if sel_mask[i] else "#a0aec0" for i in order], label="Miv_pca")
    ax.plot(miv_ori[order], yv, "k.", ms=4, label="Miv_ori")
    ax.axvline(theta, color="blue", ls="--", lw=1, label=f"θ={theta:.4f}")
    ax.set_yticks(yv); ax.set_yticklabels([feat[i] for i in order], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("MIV"); ax.set_title(f"{label} MIV 与阈值（红色=选中）")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"miv_{name}_bar.png"), dpi=130); plt.close(fig)

    # 训练曲线（缓存命中且无历史时跳过）
    if ho and hp and len(ho.get("train_loss", [])) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3))
        axes[0].plot(ho["train_loss"], label="F_ori"); axes[0].plot(hp["train_loss"], label="F_pca")
        axes[0].set_title("训练损失"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
        axes[1].plot(ho["val_rmse"], label="F_ori"); axes[1].plot(hp["val_rmse"], label="F_pca")
        axes[1].set_title("验证 RMSE(缩放域)"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
        fig.suptitle(f"{label} 承载网络训练曲线"); fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"miv_{name}_train.png"), dpi=130); plt.close(fig)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(CONFIG.keys())
    out = [process(n, CONFIG[n]) for n in names]
    print("\n阶段三完成。")
    for s in out:
        print(f"  {s['dataset']}: Fsel={s['n_selected']}/{s['n_feat']}, α={s['alpha']:.3f}, θ={s['theta']:.5f}, "
              f"val_rmse(pca)={s['carrier']['val_rmse_pca']:.4f}")


if __name__ == "__main__":
    main()
