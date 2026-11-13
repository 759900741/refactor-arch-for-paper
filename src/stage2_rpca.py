# -*- coding: utf-8 -*-
"""
stage2_rpca.py
==============
阶段二主流程：RPCA 降维去噪。

流程：
  2.1 载入/分块/切分
  2.2 IALM 分解(仅训练集) + 2.2b c 网格选参
  2.3 SVD 选主成分(95%)
  2.4 投影去噪(全部子集，P 仅来自训练集)
  2.5 拼接透传块恢复原始列序
  2.6 效果度量
  2.7 保存 + 绘图

产物：data/processed/rpca/*.npz|*.json, reports/figs/rpca_*.png
用法：cd src && python stage2_rpca.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

import utils_data as U
import utils_rpca as R

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
RP = os.path.join(P, "rpca")
FIG = os.path.join(BASE, "reports", "figs")
os.makedirs(RP, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_GRID = [0.5, 1.0, 1.5, 2.0]
ENERGY = 0.95

CONFIG = {
    "jingmen": dict(
        label="荆门市2024", target="负荷",
        rpca_cols=["负荷", "max_temperature", "low_temperature", "avg_temperature", "avg_shidu",
                   "jiangyuliang", "wind_speed", "cloud_cover", "surface_pressure",
                   "shortwave_radiation", "dew_point", "lag_1d", "lag_1w"],
    ),
    "lanzhou": dict(
        label="兰州市2023", target="electricity(KW)",
        rpca_cols=["electricity(KW)", "cold(ton)", "heat(mmBTU)", "Temperature", "Wind Speed",
                   "Wind Direction", "Surface Albedo", "Pressure", "Precipitable Water",
                   "Solar Zenith Angle", "Relative Humidity", "Dew Point", "Cloud Type",
                   "lag_1d", "lag_1w"],
    ),
}


def _corr_cols(M, y):
    """逐列与 y 的 Pearson 相关。"""
    M = np.asarray(M, float); y = np.asarray(y, float)
    out = np.zeros(M.shape[1])
    ys = y - y.mean()
    for j in range(M.shape[1]):
        x = M[:, j] - M[:, j].mean()
        den = np.sqrt((x * x).sum() * (ys * ys).sum()) + 1e-12
        out[j] = (x * ys).sum() / den
    return out


def _overlap(row_energy, is_anom):
    true_idx = np.where(is_anom)[0]
    if len(true_idx) == 0:
        return float("nan"), 0
    top = np.argsort(-row_energy)[:len(true_idx)]
    return float(len(np.intersect1d(top, true_idx)) / len(true_idx)), int(len(true_idx))


def _target_metrics(S, tgt_idx, is_anom):
    """基于目标列 |S| 的诊断：召回重合率 + 异常/非异常分离比。"""
    rT = np.abs(S[:, tgt_idx])
    true_idx = np.where(is_anom)[0]
    k = len(true_idx)
    if k == 0:
        return float("nan"), float("nan")
    top = np.argsort(-rT)[:k]
    overlap = float(len(np.intersect1d(top, true_idx)) / k)
    ratio = float(rT[is_anom].mean() / (rT[~is_anom].mean() + 1e-12))
    return overlap, ratio


def process(name, cfg):
    print(f"\n{'='*66}\n阶段二 RPCA：{cfg['label']} ({name})")
    st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
    feat_cols = st["feature_cols"]
    target = cfg["target"]
    rpca_cols = cfg["rpca_cols"]
    miss = [c for c in rpca_cols if c not in feat_cols]
    assert not miss, f"rpca_cols 不在 feature_cols: {miss}"
    pass_cols = [c for c in feat_cols if c not in set(rpca_cols)]
    assert len(pass_cols) + len(rpca_cols) == len(feat_cols)

    df = pd.read_parquet(os.path.join(P, f"{name}_clean.parquet"))
    tr, va, te = U.split_chrono(df, (0.8, 0.1, 0.1))
    splits = {"train": tr, "val": va, "test": te}
    sx = joblib.load(os.path.join(P, f"scaler_{name}_X.pkl"))

    X = {s: U.transform_X(part, feat_cols, sx) for s, part in splits.items()}
    rpca_idx = [feat_cols.index(c) for c in rpca_cols]
    pass_idx = [feat_cols.index(c) for c in pass_cols]
    tgt_idx = rpca_cols.index(target)

    Xtr_blk = X["train"][:, rpca_idx]
    y_tr = Xtr_blk[:, tgt_idx]
    is_anom_tr = tr["is_anomaly"].to_numpy().astype(bool)
    m, n = Xtr_blk.shape
    print(f"  训练块 Xtr_blk={Xtr_blk.shape} | rpca块={n} 透传={len(pass_cols)} | 目标'{target}'")

    # ---- 2.2 / 2.2b c 网格选参 ----
    table = []
    cache = {}
    for c in C_GRID:
        lam = c / np.sqrt(max(m, n))
        L, S, info = R.rpca_ialm(Xtr_blk, lam, tol=1e-6, max_iter=500)
        cache[c] = (L, S, info, lam)
        sparsity = float(np.mean(np.abs(S) < 1e-6))
        recon = info["recon_rel_err"]
        cb = np.abs(_corr_cols(Xtr_blk, y_tr))
        ca = np.abs(_corr_cols(L, y_tr))
        mask = (cb > 0.05)
        mask[tgt_idx] = False
        sig_ret = float(ca[mask].mean() / (cb[mask].mean() + 1e-12)) if mask.any() else 1.0
        ovl_t, ratio_t = _target_metrics(S, tgt_idx, is_anom_tr)
        table.append(dict(c=c, lam=float(lam), iters=info["iters"], converged=info["converged"],
                          recon_rel_err=float(recon), sparsity=sparsity,
                          signal_ret=sig_ret, anomaly_overlap=ovl_t, anomaly_ratio=ratio_t))
        print(f"    c={c:<4} lam={lam:.5f} it={info['iters']:<3} conv={info['converged']!s:<5} "
              f"recon={recon:.4f} sparse={sparsity:.3f} sigret={sig_ret:.3f} "
              f"ovl_t={ovl_t:.3f} ratio={ratio_t:.2f}")

    valid = [r for r in table if r["signal_ret"] >= 0.90 and r["sparsity"] >= 0.5]
    if valid:
        # 以「异常分离比」为主价值指标，兼顾稀疏度接近 0.9
        best = max(valid, key=lambda r: (r["anomaly_ratio"] if r["anomaly_ratio"] == r["anomaly_ratio"] else -1,
                                         -abs(r["sparsity"] - 0.9)))
        note = "网格优选(异常分离比↑)"
    else:
        best = next(r for r in table if r["c"] == 1.0)
        note = "无满足约束的c，回落c=1"
    c_sel = best["c"]; lam_sel = best["lam"]
    L_tr, S_tr, info_tr, _ = cache[c_sel]
    print(f"  选定 c={c_sel}（{note}）, lam={lam_sel:.5f}")

    # ---- 2.3 SVD 选 k ----
    sv = R.svd_select_k(L_tr, ENERGY)
    V, k, cum = sv["V"], sv["k"], sv["cum"]
    rank_L = int(np.linalg.matrix_rank(L_tr))
    print(f"  SVD: rank(L_tr)={rank_L}, 选 k={k}, 累积能量@k={cum[k-1]:.4f}")

    # ---- 2.4 逐段 RPCA 去噪：L = X - S（全秩，仅去稀疏异常）----
    # 说明：V_k/P 仅作贡献率分析；下游数据用逐段 IALM 的 L（每段只用自身数据，
    # 跨段零信息流，同样严格防泄漏），避免 k 维投影对目标列的过度平滑。
    Pmat = V @ V.T
    Lblk, Sblk = {}, {}
    for s in ("train", "val", "test"):
        Xb = X[s][:, rpca_idx]
        if s == "train":
            Lb, Sb = L_tr, S_tr
        else:
            Lb, Sb, _ = R.rpca_ialm(Xb, lam_sel, tol=1e-6, max_iter=500)
        Lblk[s], Sblk[s] = Lb, Sb
    removed_energy = float(np.linalg.norm(S_tr, "fro") / (np.linalg.norm(Xtr_blk, "fro") + 1e-12))
    load_corr_after = float(abs(_corr_cols(L_tr, y_tr)[tgt_idx]))

    # ---- 2.5 拼接恢复列序 ----
    Lfull = {}
    for s, part in splits.items():
        M = np.empty((len(part), len(feat_cols)), dtype=np.float32)
        for i, col in enumerate(feat_cols):
            if col in set(rpca_cols):
                M[:, i] = Lblk[s][:, rpca_cols.index(col)]
            else:
                M[:, i] = X[s][:, feat_cols.index(col)]
        Lfull[s] = M

    # ---- 2.6 度量：前后相关性 / MI / 异常重合 / 保真度 ----
    mi_before = mutual_info_regression(Xtr_blk, y_tr, random_state=0)
    mi_after = mutual_info_regression(Lblk["train"], y_tr, random_state=0)
    cb = np.abs(_corr_cols(Xtr_blk, y_tr)); ca = np.abs(_corr_cols(Lblk["train"], y_tr))
    mask = cb > 0.05          # 含目标列在内（目标列保真度必须纳入考核）
    sig_ret_all = float(ca[mask].mean() / (cb[mask].mean() + 1e-12))
    sig_ba = {}
    for j, col in enumerate(rpca_cols):
        sig_ba[col] = dict(pearson_before=float(cb[j]), pearson_after=float(ca[j]),
                           MI_before=float(mi_before[j]), MI_after=float(mi_after[j]))
    overlap_sel, ratio_sel = _target_metrics(S_tr, tgt_idx, is_anom_tr)
    n_anom = int(is_anom_tr.sum())
    sparsity_sel = best["sparsity"]

    # ---- 2.7 保存 ----
    for s, part in splits.items():
        np.savez_compressed(os.path.join(RP, f"{name}_L_{s}.npz"),
                            L=Lfull[s], cols=np.array(feat_cols), time=part.index.astype(str).to_numpy())
        # RPCA 真稀疏分量 S = X - L（逐段，诊断/异常定位用）
        np.savez_compressed(os.path.join(RP, f"{name}_S_{s}.npz"),
                            S=Sblk[s].astype(np.float32), block_cols=np.array(rpca_cols))
    # 清理旧命名产物，避免混淆
    for stale in [f"{name}_R_{s}.npz" for s in splits] + [f"{name}_Ssparse_train.npz"]:
        p = os.path.join(RP, stale)
        if os.path.exists(p):
            os.remove(p)
    np.savez_compressed(os.path.join(RP, f"{name}_rpca_model.npz"),
                        V_k=V, P=Pmat, singular_values=sv["singular_values"],
                        k=k, lam=lam_sel, c=c_sel, rank=rank_L)

    stats = dict(
        dataset=name, label=cfg["label"], n_train=int(m), F_rpca=int(n),
        rpca_cols=rpca_cols, passthrough_cols=pass_cols,
        c_grid=C_GRID, c_table=table, c_selected=float(c_sel), c_note=note,
        lam=float(lam_sel), iters=int(info_tr["iters"]), converged=bool(info_tr["converged"]),
        rank_L=rank_L, k=int(k), cum_energy=float(cum[k-1]), energy_threshold=ENERGY,
        removed_energy=float(removed_energy), load_corr_after=float(load_corr_after),
        S_sparsity=float(sparsity_sel), signal_ret_mean=float(sig_ret_all),
        anomaly_overlap=float(overlap_sel) if overlap_sel == overlap_sel else None,
        anomaly_ratio=float(ratio_sel) if ratio_sel == ratio_sel else None,
        n_anomaly_train=int(n_anom), signal_before_after=sig_ba,
        windows_rows={s: int(len(part)) for s, part in splits.items()},
    )
    U.save_json(stats, os.path.join(RP, f"{name}_rpca_stats.json"))

    # ---- 绘图 ----
    make_figs(name, cfg, sv, S_tr, Xtr_blk, Lblk["train"], rpca_cols, tgt_idx, cb, ca)
    print(f"  完成：去噪能量比={removed_energy:.4f}, 负荷保真corr={load_corr_after:.4f}, "
          f"S稀疏={sparsity_sel:.3f}, 信号保持(含目标)={sig_ret_all:.3f}, 异常重合={overlap_sel:.2f}")
    return stats


def make_figs(name, cfg, sv, S_tr, Xtr, Ltr, rpca_cols, tgt_idx, cb, ca):
    label = cfg["label"]
    # 1 奇异值谱 + 累积能量
    fig, ax1 = plt.subplots(figsize=(8, 3))
    s = sv["singular_values"]
    ax1.bar(range(1, len(s) + 1), sv["ratio"], color="#4299e1", alpha=.7, label="贡献率")
    ax1.set_xlabel("主成分序号"); ax1.set_ylabel("贡献率")
    ax2 = ax1.twinx(); ax2.plot(range(1, len(s) + 1), sv["cum"], "r-o", ms=3, label="累积")
    ax2.axhline(.95, color="gray", ls="--", lw=.8); ax2.set_ylabel("累积贡献率")
    ax1.set_title(f"{label} RPCA 主成分能量（k={sv['k']}）")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rpca_{name}_energy.png"), dpi=130); plt.close(fig)

    # 2 S 稀疏热力图（特征×时间，下采样）
    T = S_tr.shape[0]; step = max(1, T // 1500)
    Ms = S_tr[::step].T
    fig, ax = plt.subplots(figsize=(11, 3.2))
    vmax = np.percentile(np.abs(Ms), 99) + 1e-9
    im = ax.imshow(Ms, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(rpca_cols))); ax.set_yticklabels(rpca_cols, fontsize=7)
    ax.set_title(f"{label} 稀疏分量 S（特征×时间，越深越异常）")
    fig.colorbar(im, ax=ax, fraction=.02); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"rpca_{name}_S.png"), dpi=130); plt.close(fig)

    # 3 负荷去噪前后
    N = min(len(Xtr), 96 * 7 if name == "jingmen" else 24 * 7)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(range(N), Xtr[:N, tgt_idx], color="#a0aec0", lw=.8, label="原始(标准化)")
    ax.plot(range(N), Ltr[:N, tgt_idx], color="#c53030", lw=1.0, label="RPCA去噪")
    ax.set_title(f"{label} 负荷去噪前后（前 {N} 步）"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rpca_{name}_denoise.png"), dpi=130); plt.close(fig)

    # 4 相关性前后
    fig, ax = plt.subplots(figsize=(8, max(3, .28 * len(rpca_cols))))
    order = np.argsort(np.abs(ca))
    yy = np.arange(len(rpca_cols))
    ax.barh(yy - .2, np.abs(cb[order]), height=.4, color="#a0aec0", label="去噪前")
    ax.barh(yy + .2, np.abs(ca[order]), height=.4, color="#c53030", label="去噪后")
    ax.set_yticks(yy); ax.set_yticklabels([rpca_cols[i] for i in order], fontsize=7)
    ax.set_title(f"{label} 特征与目标 |相关系数| 前后对比"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rpca_{name}_corr.png"), dpi=130); plt.close(fig)


def main():
    all_stats = [process(n, c) for n, c in CONFIG.items()]
    print("\n阶段二完成。")
    for s in all_stats:
        print(f"  {s['dataset']}: c={s['c_selected']}, k={s['k']}, cum={s['cum_energy']:.3f}, "
              f"S稀疏={s['S_sparsity']:.3f}, 信号保持={s['signal_ret_mean']:.3f}")


if __name__ == "__main__":
    main()
