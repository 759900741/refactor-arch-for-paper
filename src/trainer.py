# -*- coding: utf-8 -*-
"""
trainer.py
==========
可复用的训练/评估组件（Python 3.8 + torch CPU，仅依赖 numpy/torch）。

设计目标：阶段四（基线训练）与阶段五（TE-RIME 超参搜索）共用同一套数据加载、
训练循环与指标口径，保证「搜索时评估的」与「最终交付的」完全一致。

对外接口：
    load_split(name)                  -> dict   # X/y/cols/L/H/stride/y_range/time
    build_model(n_feat, horizon, cfg) -> nn.Module
    fit_model(d, cfg, log=print)      -> (model, hist, best_val_rmse)
    predict(model, X)                 -> np.ndarray
    metrics(y_true, y_pred)           -> dict
    inverse(y, lo, hi)                -> np.ndarray
"""
from __future__ import annotations
import os
import json
import time
import copy
import hashlib
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn

from model_bilstm_itransformer import BiLSTMiTransformer

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
MVD = os.path.join(P, "miv")

MODEL_KEYS = ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "dropout", "lstm_layers",
              "residual", "head_mode", "use_bilstm", "model_type",
              "fusion", "mv_hidden", "mv_layers")

# 阶段四基线超参（阶段五 TE-RIME 在此基础上搜索）
# 取值为 CUDA 迁移前的 CPU 预算稳定配置：小容量 + 学习率 1e-3 + 残差输出头
# 注：d_model/d_ff/e_layers 明显低于研究方案中的论文规格（128/256/2），交由 tuner 归位
DEFAULT_CFG = dict(
    hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1, lstm_layers=1,
    residual=True, head_mode="target", use_bilstm=True, model_type="bilstm_itransformer",
    fusion="serial", mv_hidden=None, mv_layers=1, aux_weight=0.0,
    lr=1e-3, weight_decay=0.0, batch=128, epochs=60, patience=12, seed=42,
    device="auto", monitor="val_mape",
)

FREQ_DELTA = {"15min": timedelta(minutes=15), "30min": timedelta(minutes=30),
              "1h": timedelta(hours=1), "1H": timedelta(hours=1)}


def _load_array(name, split, variant, add_lag=False):
    """按变体载入 X/y/cols。variant: sel / selorig / full；add_lag 仅在 sel/selorig 下追加 lag 两列。"""
    st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
    if variant in ("sel", "selorig"):
        f = np.load(os.path.join(MVD, f"{name}_{split}_{variant}.npz"), allow_pickle=True)
        X, y = f["X"].astype(np.float32), f["y"].astype(np.float32)
        mst = json.load(open(os.path.join(MVD, f"{name}_miv_stats.json"), encoding="utf-8"))
        cols = list(mst["selected"])
        if add_lag:
            full = np.load(os.path.join(P, "windows", f"{name}_{split}.npz"), allow_pickle=True)
            fcols = st["feature_cols"]
            lag_cols = [c for c in ("lag_1d", "lag_1w") if c in fcols]
            lag_idx = [fcols.index(c) for c in lag_cols]
            X = np.concatenate([X, full["X"][:, :, lag_idx].astype(np.float32)], axis=2)
            cols = cols + lag_cols
        return X, y, cols
    elif variant == "full":
        f = np.load(os.path.join(P, "windows", f"{name}_{split}.npz"), allow_pickle=True)
        return f["X"].astype(np.float32), f["y"].astype(np.float32), list(st["feature_cols"])
    elif variant == "lfull":
        # RPCA 低秩矩阵 L（全特征）重建窗口（消融 A1 用）
        rp = np.load(os.path.join(P, "rpca", f"{name}_L_{split}.npz"), allow_pickle=True)
        M = rp["L"].astype(np.float32)
        cols = [str(c) for c in rp["cols"]]
        L, H, stride = int(st["L"]), int(st["H"]), int(st["stride"])
        N = len(M) - L - H + 1
        idx = list(range(0, N, stride))
        X = np.stack([M[i:i + L] for i in idx], axis=0).astype(np.float32)
        w = np.load(os.path.join(P, "windows", f"{name}_{split}.npz"), allow_pickle=True)
        y = w["y"].astype(np.float32)
        assert X.shape[0] == y.shape[0], f"lfull 窗口数不对齐: {X.shape[0]} vs {y.shape[0]}"
        return X, y, cols
    raise ValueError(f"未知 variant: {variant}")


def load_split(name, variant="sel", add_lag=False):
    """载入滑窗数据并补出每窗口的结束时刻。

    variant="sel"     -> RPCA 低秩矩阵 L 的选中列（去噪输入）
    variant="selorig" -> 原始矩阵的选中列
    variant="full"    -> 原始矩阵全特征（消融 A0 用）
    add_lag=True      -> 在 sel/selorig 基础上追加 lag_1d/lag_1w 两列（消融 A5 用）
    """
    st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
    mst = json.load(open(os.path.join(MVD, f"{name}_miv_stats.json"), encoding="utf-8"))
    L, H, stride = int(st["L"]), int(st["H"]), int(st["stride"])
    dt = FREQ_DELTA[st["time_report"]["freq"]]
    d = dict(L=L, H=H, stride=stride, target=mst["target"], cols=[], variant=variant,
             add_lag=add_lag, y_range=None, time={})
    for s in ("train", "val", "test"):
        X, y, cols = _load_array(name, s, variant, add_lag)
        d[s] = dict(X=X, y=y)
        d["cols"] = cols
        n = len(X)
        start = datetime.strptime(st["split_boundaries"][s][0], "%Y-%m-%d %H:%M:%S")
        base = start + dt * (L - 1)
        d["time"][s] = [base + dt * (i * stride) for i in range(n)]
    d["trainset_rows"] = int(st["windows"]["train"]["rows"])
    d["name"] = name
    try:
        lo, hi = load_y_range(name)
        d["y_range"] = (lo, hi)
    except Exception:  # noqa: BLE001  缺少 y_range json 时退化为无原始量纲指标
        d["y_range"] = None
    return d


def load_y_range(name):
    """读取 y 反归一化区间（由 3.13 环境从 scaler pkl 导出为 JSON，3.8 无法直接反序列化）。"""
    p = os.path.join(P, f"{name}_y_range.json")
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    return float(r["min"]), float(r["max"])


def resolve_device(cfg=None):
    """解析设备：cfg['device'] > 环境变量 WB_DEVICE > "auto"（有 CUDA 用 cuda，否则 cpu）。"""
    s = (cfg or {}).get("device") or os.environ.get("WB_DEVICE") or "auto"
    if s == "auto":
        s = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(s)


def build_model(n_feat, horizon, cfg):
    """按 cfg['model_type'] 分发到对应变体并迁移到目标设备；target_idx 由 cfg 提供。"""
    from model_variants import build_variant
    return build_variant(cfg.get("model_type", "bilstm_itransformer"), n_feat, horizon, cfg).to(resolve_device(cfg))


def _to_t(a, device="cpu"):
    return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


def auto_batch(cfg, dataset=None):
    """按容量确定 batch（不进搜索空间）：逐变量 LSTM 的 B*F 条序列是显存瓶颈。"""
    h = max(int(cfg.get("hidden") or 32), int(cfg.get("mv_hidden") or 0))
    layers = int(cfg.get("lstm_layers") or 1)
    if dataset == "lanzhou":
        return 128
    if h >= 128 and layers >= 2:
        return 64
    if h >= 96:
        return 96
    return 128


def fit_model(d, cfg, log=print, monitor=None):
    """训练至多 cfg['epochs'] 轮，按 monitor 早停并回滚最优权重。

    monitor:
        "val_rmse" —— 缩放域验证 RMSE（阶段四早期口径，保留兼容）
        "val_mape" —— 原始量纲验证 MAPE（默认，与超参搜索目标一致）
    """
    cfg = {**DEFAULT_CFG, **cfg}
    monitor = monitor or cfg.get("monitor") or "val_rmse"
    if cfg.get("target_idx") is None:
        cfg["target_idx"] = d["cols"].index(d["target"])
    device = resolve_device(cfg)
    seed = int(cfg["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr = _to_t(d["train"]["X"], device), _to_t(d["train"]["y"], device)
    Xva, yva = _to_t(d["val"]["X"], device), _to_t(d["val"]["y"], device)
    y_range = d.get("y_range") or load_y_range(d["name"]) if d.get("name") else d.get("y_range")
    model = build_model(Xtr.shape[2], ytr.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(cfg["epochs"]), 1), eta_min=cfg["lr"] * 0.05)
    lossf = nn.MSELoss()
    aux_w = float(cfg.get("aux_weight") or 0.0)
    has_aux = hasattr(model, "forward_with_aux") and aux_w > 0

    n = len(Xtr); bs = int(cfg["batch"])
    hist = dict(train_loss=[], val_rmse=[], val_mape=[], val_rmse_orig=[],
                lr=[], epoch_sec=[], monitor=monitor)
    best = float("inf"); best_state = None; wait = 0; ep_best = 0
    for ep in range(int(cfg["epochs"])):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            if has_aux:
                out, aux = model.forward_with_aux(Xtr[b])
                loss = lossf(out, ytr[b])
                if aux is not None:
                    loss = loss + aux_w * lossf(aux, ytr[b])
            else:
                loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        vs = val_scores(model, Xva, yva, y_range)
        score = vs["mape"] if monitor == "val_mape" else vs["rmse"]
        hist["train_loss"].append(tot / max(nb, 1))
        hist["val_rmse"].append(vs["rmse"]); hist["val_mape"].append(vs["mape"])
        hist["val_rmse_orig"].append(vs["rmse_orig"])
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["epoch_sec"].append(round(time.perf_counter() - t0, 2))
        sched.step()
        if score < best - 1e-6:
            best, ep_best, wait = score, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if log and (ep % 5 == 0 or ep == int(cfg["epochs"]) - 1 or wait == 0):
            log(f"    ep{ep+1:02d} loss={hist['train_loss'][-1]:.5f} "
                f"val_rmse={vs['rmse']:.5f} val_mape={vs['mape']:.3f}% "
                f"lr={hist['lr'][-1]:.2e} ({hist['epoch_sec'][-1]}s)")
        if wait >= int(cfg["patience"]):
            if log:
                log(f"    早停于 ep{ep+1}（最优 ep{ep_best}, {monitor}={best:.5f}）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_epoch"] = ep_best
    hist["best_score"] = float(best)
    return model, hist, float(best)


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    device = next(model.parameters()).device
    out = []
    Xt = _to_t(X, device)
    for i in range(0, len(Xt), bs):
        out.append(model(Xt[i:i + bs]).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_rmse(model, Xt, yt, bs=256):
    model.eval()
    tot = 0.0; cnt = 0
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            p = model(Xt[i:i + bs])
            tot += float(((p - yt[i:i + bs]) ** 2).sum().item())
            cnt += p.numel()
    return float(np.sqrt(tot / max(cnt, 1)))


@torch.no_grad()
def val_scores(model, Xt, yt, y_range=None, bs=256):
    """一次前向同时给出缩放域 RMSE、原始量纲 RMSE/MAPE（早停与搜索共用）。"""
    model.eval()
    ps = []
    for i in range(0, len(Xt), bs):
        ps.append(model(Xt[i:i + bs]))
    p = torch.cat(ps, dim=0)
    e = (p - yt)
    rmse = float(torch.sqrt((e ** 2).mean()).item())
    out = dict(rmse=rmse, mape=float("nan"), rmse_orig=float("nan"))
    if y_range is not None:
        lo, hi = float(y_range[0]), float(y_range[1])
        yt_o = yt.detach().cpu().numpy().astype(np.float64) * (hi - lo) + lo
        p_o = p.detach().cpu().numpy().astype(np.float64) * (hi - lo) + lo
        out["rmse_orig"] = float(np.sqrt(np.mean((yt_o - p_o) ** 2)))
        out["mape"] = float(np.mean(np.abs(yt_o - p_o) / np.maximum(np.abs(yt_o), 1e-8)) * 100)
    return out


def inverse(y, lo, hi):
    return np.asarray(y) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# 配置规范化 / 指纹 / 权重重载（调参缓存与阶段四-六一致性校验共用）
# ---------------------------------------------------------------------------
#: 各 model_type 实际生效的超参键（其余键即使传入也不影响前向，须从指纹中剥离）
ACTIVE_KEYS = {
    "bilstm_itransformer": ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "dropout",
                            "lstm_layers", "residual", "head_mode", "use_bilstm",
                            "fusion", "mv_hidden", "mv_layers"),
    "bilstm": ("hidden", "dropout", "lstm_layers", "residual"),
    "transformer": ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "dropout",
                    "lstm_layers", "residual"),
    "informer": ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "dropout",
                 "lstm_layers", "residual"),
}
_TRAIN_KEYS = ("lr", "weight_decay", "batch", "epochs", "patience", "seed", "aux_weight")


def canonical_cfg(cfg):
    """剥离不适用键、修正 n_heads 整除性、数值取定长，保证同一语义只对应一个指纹。"""
    cfg = {**DEFAULT_CFG, **cfg}
    mt = cfg.get("model_type", "bilstm_itransformer")
    keys = ACTIVE_KEYS.get(mt, ACTIVE_KEYS["bilstm_itransformer"]) + _TRAIN_KEYS
    out = {"model_type": mt}
    for k in keys:
        v = cfg.get(k)
        if v is None:
            out[k] = None
        elif isinstance(v, bool):
            out[k] = bool(v)
        elif isinstance(v, float):
            out[k] = round(v, 8)
        elif isinstance(v, int):
            out[k] = int(v)
        else:
            out[k] = v
    if out.get("fusion") in (None, "serial"):
        out.pop("mv_hidden", None); out.pop("mv_layers", None)
    dm = out.get("d_model"); nh = out.get("n_heads")
    if isinstance(dm, int) and isinstance(nh, int) and nh and dm % nh:
        out["n_heads"] = 1 if dm <= 2 else 2
    if out.get("d_model") and out.get("d_ff") and out["d_ff"] % out["d_model"]:
        out["d_ff"] = int(round(out["d_ff"] / out["d_model"])) * out["d_model"]
    return out


def cfg_hash(cfg, n=12):
    """配置指纹（仅含生效键），用于调参缓存键与阶段四/六一致性校验。"""
    s = json.dumps(canonical_cfg(cfg), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def load_model_from_ckpt(path, map_location=None):
    """从阶段四权重文件重建模型：严格按 ckpt 中的实际 cfg 与 cols 维度实例化。"""
    ck = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    cfg = ck["cfg"]
    cols = ck.get("cols")
    n_feat = len(cols) if cols else ck["n_feat"]
    model = build_model(n_feat, ck["H"], cfg)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, cfg, ck


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    e = y - p
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-8)) * 100)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    rpd = float(np.std(y, ddof=1) / rmse) if rmse > 0 else float("nan")
    den = np.sum((np.abs(p - y.mean()) + np.abs(y - y.mean())) ** 2)
    ia = float(1 - np.sum(e ** 2) / den) if den > 0 else float("nan")
    vae = float(np.var(np.abs(e)))  # 绝对误差方差（论文稳定性指标 Table 7）
    return dict(rmse=rmse, mae=mae, mape=mape, r2=r2, rpd=rpd, ia=ia, vae=vae, n=int(y.size))
