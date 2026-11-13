# -*- coding: utf-8 -*-
"""
stage4_train.py
===============
阶段四：用 MIV 选中特征训练最终 BiLSTM-iTransformer 基线模型，并产出论文图表。

步骤：
  4.1 载入选中特征滑窗（train/val/test）+ 反归一化区间
  4.2 训练（AdamW + 余弦退火 + 早停，验证集 RMSE 选优）
  4.3 评估：验证/测试（缩放域 + 原始量纲）、逐预测步、持续预测基线
  4.4 保存：models/{name}_best.pt、preds/{name}_{split}.npz、reports/metrics/{name}_stage4.json
  4.5 出图（论文用）：模型结构、训练曲线、预测对比、散点、多步误差、残差分布
  4.6 汇总报告 reports/stage4_results.md

运行：cd src && "<py3.8>" stage4_train.py            # 两个数据集
      cd src && "<py3.8>" stage4_train.py jingmen    # 单数据集
"""
from __future__ import annotations
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T

BASE = T.BASE
P = T.P
FIG = os.path.join(BASE, "reports", "figs")
MET = os.path.join(BASE, "reports", "metrics")
MOD = os.path.join(BASE, "models")
PRD = os.path.join(P, "preds")
for d in (FIG, MET, MOD, PRD):
    os.makedirs(d, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

CONFIG = {
    "jingmen": dict(label="荆门市 2024", unit="MW", freq_label="1 h", variant="selorig"),
    "lanzhou": dict(label="兰州市 2023", unit="kW", freq_label="1 h", variant="selorig"),
}


# ---------------------------------------------------------------------------
# 4.5 图表
# ---------------------------------------------------------------------------
def fig_arch(path, cfg=None):
    """模型结构示意图（BiLSTM 逐变量编码 → 变量 token → 变量维自注意力 → 目标 token → 线性头+残差）。"""
    cfg = cfg or {}
    d = cfg.get("d_model", 32); nh = cfg.get("n_heads", 2); el = cfg.get("e_layers", 1)
    hid = cfg.get("hidden", 32); res = cfg.get("residual", True)
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, txt, fc="#e8f0fb", ec="#3c6fb0", fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1a2b45")

    def arrow(x1, y1, x2, y2, txt=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     lw=1.3, color="#4a5568"))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, txt, ha="center", fontsize=8.5, color="#4a5568")

    box(0.15, 1.55, 1.55, 1.05, "输入窗口\n$X\\in\\mathbb{R}^{L\\times F}$\n(L=96/24, F=16/17)", fc="#eef7ee", ec="#2f855a")
    box(2.05, 1.55, 1.75, 1.05, f"逐变量 BiLSTM\n$h=[\\bar{{h}}_t;h_L]$\n(双向, hidden={hid})")
    box(4.15, 1.55, 1.55, 1.05, f"变量 token\n$E\\in\\mathbb{{R}}^{{F\\times {d}}}$", fc="#fdf3e3", ec="#b7791f")
    box(6.05, 1.35, 2.15, 1.45, "iTransformer 编码器\n变量维 self-attention\n"
        f"({el} 层 / {nh} 头 / d={d})", fc="#fbe9ef", ec="#b83280")
    head_txt = ("线性头 + 残差\n$\\hat{Y}=y_t{+}\\Delta\\hat{Y}$" if res else "线性头\n$\\hat{Y}\\in\\mathbb{R}^{H}$")
    box(8.55, 1.55, 1.35, 1.05, "目标 token\n$z_{0}$", fc="#fdf3e3", ec="#b7791f")
    box(10.05, 1.55, 1.0, 1.05, head_txt, fc="#eef7ee", ec="#2f855a", fs=9)

    arrow(1.72, 2.07, 2.05, 2.07)
    arrow(3.82, 2.07, 4.15, 2.07, "时间池化")
    arrow(5.72, 2.07, 6.05, 2.07)
    arrow(8.22, 2.07, 8.55, 2.07)
    arrow(9.9, 2.07, 10.05, 2.07)

    ax.text(0.15, 0.62,
            "融合逻辑：BiLSTM 负责单变量局部时序编码（把 L 步压成 1 个 token）；"
            "iTransformer 在变量维做注意力，建模多源特征全局相关；\n"
            "输出端取目标变量 token（而非变量均值），避免目标信息被稀释；"
            + ("输出采用残差（增量）形式以抑制负荷水平漂移。" if res else ""),
            fontsize=9, color="#2d3748", va="top")
    ax.text(0.15, 3.62, "图 4-1  BiLSTM-iTransformer 混合预测模型结构", fontsize=12, fontweight="bold", color="#1a202c")
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_train(path, hist, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    axes[0].plot(ep, hist["train_loss"], "o-", ms=3, color="#2b6cb0", label="训练损失")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE（缩放域）"); axes[0].set_title("训练损失")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].plot(ep, hist["val_rmse"], "s-", ms=3, color="#c53030", label="验证 RMSE")
    axes[1].axvline(hist["best_epoch"], color="#718096", ls="--", lw=1,
                    label=f"最优 epoch={hist['best_epoch']}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("RMSE（缩放域）"); axes[1].set_title("验证 RMSE 与早停")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.suptitle(f"{label} 训练过程", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_pred(path, t, ytrue, ypred, label, unit, h=0):
    """测试集预测对比：全时段 + 前 72 小时放大。"""
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2),
                             gridspec_kw=dict(height_ratios=[1, 1]))
    axes[0].plot(t, ytrue[:, h], color="#2d3748", lw=1.2, label="真实值")
    axes[0].plot(t, ypred[:, h], color="#c53030", lw=1.2, ls="-", alpha=.85, label="预测值")
    axes[0].set_ylabel(f"负荷（{unit}）"); axes[0].legend(fontsize=9, ncol=2)
    axes[0].set_title(f"{label} 测试集预测对比（h={h+1}）"); axes[0].grid(alpha=.25)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    n = min(72, len(t))
    axes[1].plot(t[:n], ytrue[:n, h], "o-", ms=2.6, lw=1.1, color="#2d3748", label="真实值")
    axes[1].plot(t[:n], ypred[:n, h], "s-", ms=2.2, lw=1.1, color="#c53030", alpha=.85, label="预测值")
    axes[1].fill_between(t[:n], ytrue[:n, h], ypred[:n, h], color="#cbd5e0", alpha=.5, label="误差")
    axes[1].set_ylabel(f"负荷（{unit}）"); axes[1].set_xlabel("时间")
    axes[1].legend(fontsize=9, ncol=3); axes[1].grid(alpha=.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_scatter(path, yt, yp, met, label, unit, h=0):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(yt[:, h], yp[:, h], s=8, alpha=.45, color="#2b6cb0", edgecolors="none")
    lo = min(yt[:, h].min(), yp[:, h].min()); hi = max(yt[:, h].max(), yp[:, h].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(f"真实负荷（{unit}）"); ax.set_ylabel(f"预测负荷（{unit}）")
    ax.set_title(f"{label} 预测散点（h={h+1}）")
    ax.text(.05, .93, f"$R^2$={met['r2']:.4f}\nRMSE={met['rmse']:.1f}\nMAPE={met['mape']:.2f}%",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(fc="white", ec="#cbd5e0", alpha=.9))
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_hstep(path, per_h, label):
    h = np.arange(1, len(per_h) + 1)
    rmse = [p["rmse"] for p in per_h]; mae = [p["mae"] for p in per_h]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    w = 0.36
    ax.bar(h - w / 2, rmse, w, color="#2b6cb0", label="RMSE")
    ax.bar(h + w / 2, mae, w, color="#dd6b20", label="MAE")
    for i in range(len(h)):
        ax.text(h[i] - w / 2, rmse[i], f"{rmse[i]:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(h[i] + w / 2, mae[i], f"{mae[i]:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(h); ax.set_xticklabels([f"$h_{i}$" for i in h])
    ax.set_xlabel("预测步长（1 h/步）"); ax.set_ylabel("误差（原始量纲）")
    ax.set_title(f"{label} 多步预测误差"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_err(path, yt, yp, label, unit, h=0):
    e = yt[:, h] - yp[:, h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    axes[0].hist(e, bins=40, color="#4299e1", edgecolor="white")
    axes[0].axvline(0, color="#c53030", ls="--", lw=1)
    axes[0].set_xlabel(f"残差（{unit}）"); axes[0].set_ylabel("频数")
    axes[0].set_title(f"残差分布（均值={e.mean():.1f}, 标准差={e.std():.1f}）")
    axes[1].scatter(yp[:, h], e, s=7, alpha=.4, color="#2f855a", edgecolors="none")
    axes[1].axhline(0, color="#c53030", ls="--", lw=1)
    axes[1].set_xlabel(f"预测值（{unit}）"); axes[1].set_ylabel(f"残差（{unit}）")
    axes[1].set_title("残差 vs 预测值")
    for a in axes:
        a.grid(alpha=.25)
    fig.suptitle(f"{label} 误差特性（h={h+1}）", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process(name, cfg):
    print(f"\n{'='*70}\n阶段四：{cfg['label']} ({name})")
    train_cfg = cfg["train_cfg"]
    d = T.load_split(name, cfg["variant"])
    lo, hi = T.load_y_range(name)
    H, L = d["H"], d["L"]
    print(f"  F={len(d['cols'])}, L={L}, H={H}, 样本 训练/验证/测试="
          f"{len(d['train']['X'])}/{len(d['val']['X'])}/{len(d['test']['X'])}, y_orig∈[{lo:.1f},{hi:.1f}]")
    print("  超参：" + ", ".join(f"{k}={train_cfg.get(k)}" for k in
                                ("model_type", "fusion", "hidden", "d_model", "n_heads", "e_layers",
                                 "d_ff", "dropout", "lstm_layers", "lr", "weight_decay",
                                 "batch", "epochs", "patience")))

    model, hist, best_val = T.fit_model(d, train_cfg, log=print)
    ep_sec = float(np.mean(hist["epoch_sec"]))
    tmin = float(np.sum(hist["epoch_sec"])) / 60.0
    n_par = int(sum(p.numel() for p in model.parameters()))
    print(f"  训练完成：最优 val {hist['monitor']}={best_val:.5f}（ep{hist['best_epoch']}，"
          f"{len(hist['epoch_sec'])} 轮，{tmin:.1f} min，参数量 {n_par/1000:.1f}k）")

    # 预测
    pv = T.predict(model, d["val"]["X"]); pt = T.predict(model, d["test"]["X"])
    yv, yt = d["val"]["y"], d["test"]["y"]
    mv_s, mt_s = T.metrics(yv, pv), T.metrics(yt, pt)
    pv_o, pt_o = T.inverse(pv, lo, hi), T.inverse(pt, lo, hi)
    yv_o, yt_o = T.inverse(yv, lo, hi), T.inverse(yt, lo, hi)
    mv_o, mt_o = T.metrics(yv_o, pv_o), T.metrics(yt_o, pt_o)

    # 逐预测步
    per_h = []
    for h in range(H):
        mh = T.metrics(yt_o[:, h], pt_o[:, h])
        per_h.append(dict(h=h + 1, rmse=mh["rmse"], mae=mh["mae"], mape=mh["mape"], r2=mh["r2"]))

    # 持续预测基线（用窗口内最后已知负荷值外推，缩放域）
    ti = d["cols"].index(d["target"])
    base_p = np.repeat(d["test"]["X"][:, -1, ti:ti + 1], H, axis=1)
    mb_s = T.metrics(yt, base_p)
    mb_o = T.metrics(yt_o, T.inverse(base_p, lo, hi))
    print(f"  基线(持续预测) test RMSE={mb_o['rmse']:.1f} | 本模型 test RMSE={mt_o['rmse']:.1f} "
          f"（提升 {(mb_o['rmse']-mt_o['rmse'])/mb_o['rmse']*100:.1f}%）")

    # 保存权重与预测（state_dict 统一 .cpu()，保证跨 CPU/GPU 可安全加载）
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=state, cfg=train_cfg, cfg_hash=T.cfg_hash(train_cfg),
                    arch_version="v2", protocol="full", n_feat=len(d["cols"]), cols=d["cols"],
                    L=L, H=H, y_range=[lo, hi], best_val=best_val),
               os.path.join(MOD, f"{name}_best.pt"))
    t_test = np.array([x.strftime("%Y-%m-%d %H:%M:%S") for x in d["time"]["test"]])
    t_val = np.array([x.strftime("%Y-%m-%d %H:%M:%S") for x in d["time"]["val"]])
    np.savez_compressed(os.path.join(PRD, f"{name}_test.npz"), y_true=yt, y_pred=pt,
                        y_true_orig=yt_o, y_pred_orig=pt_o, end_time=t_test)
    np.savez_compressed(os.path.join(PRD, f"{name}_val.npz"), y_true=yv, y_pred=pv,
                        y_true_orig=yv_o, y_pred_orig=pv_o, end_time=t_val)

    # 指标 JSON（缩放域用扁平的 rmse/mae/r2，原始量纲加 _orig 后缀）
    met = dict(dataset=name, label=cfg["label"], unit=cfg["unit"], L=L, H=H, stride=d["stride"],
               cols=d["cols"], target=d["target"], cfg=train_cfg,
               cfg_hash=T.cfg_hash(train_cfg), arch_version="v2", protocol="full",
               params=n_par, input_variant=cfg["variant"],
               best_epoch=int(hist["best_epoch"]), epochs_run=len(hist["epoch_sec"]),
               sec_per_epoch=round(ep_sec, 2), train_min=round(tmin, 2), y_range=[lo, hi],
               val=dict(mv_s, **{k + "_orig": v for k, v in mv_o.items()}),
               test=dict(mt_s, **{k + "_orig": v for k, v in mt_o.items()}),
               per_horizon=per_h,
               baseline=dict(mb_s, **{k + "_orig": v for k, v in mb_o.items()}))
    with open(os.path.join(MET, f"{name}_stage4.json"), "w", encoding="utf-8") as f:
        json.dump(met, f, ensure_ascii=False, indent=2)

    # 图表
    fig_train(os.path.join(FIG, f"fig4_train_{name}.png"), hist, cfg["label"])
    fig_pred(os.path.join(FIG, f"fig4_pred_{name}.png"), d["time"]["test"], yt_o, pt_o,
             cfg["label"], cfg["unit"])
    fig_scatter(os.path.join(FIG, f"fig4_scatter_{name}.png"), yt_o, pt_o, mt_o,
                cfg["label"], cfg["unit"])
    fig_err(os.path.join(FIG, f"fig4_err_{name}.png"), yt_o, pt_o, cfg["label"], cfg["unit"])
    if H > 1:
        fig_hstep(os.path.join(FIG, f"fig4_hstep_{name}.png"), per_h, cfg["label"])
    print(f"  产物：models/{name}_best.pt, preds/{name}_{{val,test}}.npz, "
          f"reports/metrics/{name}_stage4.json, 图表 fig4_*_{name}.png")
    return met


def write_report(mets):
    lines = ["# 阶段四结果：BiLSTM-iTransformer 基线模型\n",
             f"> 生成：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}\n",
             "## 1. 测试集精度（原始量纲）\n",
             "| 数据集 | 量纲 | RMSE | MAE | MAPE(%) | R² | RPD | IA |",
             "|---|---|---|---|---|---|---|---|"]
    for m in mets:
        t = m["test"]
        lines.append(f"| {m['label']} | {m['unit']} | {t['rmse_orig']:.1f} | {t['mae_orig']:.1f} | "
                     f"{t['mape_orig']:.2f} | {t['r2_orig']:.4f} | {t['rpd_orig']:.3f} | {t['ia_orig']:.4f} |")
    lines += ["\n## 2. 与持续预测基线对比（测试集，原始量纲）\n",
              "| 数据集 | 基线 RMSE | 模型 RMSE | 提升 |", "|---|---|---|---|"]
    for m in mets:
        b = m["baseline"]["rmse_orig"]; t = m["test"]["rmse_orig"]
        lines.append(f"| {m['label']} | {b:.1f} | {t:.1f} | {(b-t)/b*100:.2f}% |")
    lines += ["\n## 3. 逐预测步误差（原始量纲）\n"]
    for m in mets:
        if len(m["per_horizon"]) > 1:
            lines.append(f"\n**{m['label']}**（步长 1 h）\n")
            lines.append("| 步长 | RMSE | MAE | MAPE(%) | R² |")
            lines.append("|---|---|---|---|---|")
            for p in m["per_horizon"]:
                lines.append(f"| h{p['h']} | {p['rmse']:.1f} | {p['mae']:.1f} | {p['mape']:.2f} | {p['r2']:.4f} |")
        else:
            lines.append(f"\n**{m['label']}**：单步预测（H=1），RMSE={m['per_horizon'][0]['rmse']:.1f}, "
                         f"MAPE={m['per_horizon'][0]['mape']:.2f}%\n")
    lines += ["\n## 4. 训练配置与成本\n",
              "| 数据集 | 输入变体 | 残差头 | F | L | H | 最优 epoch | 总轮数 | 单轮耗时 | 总耗时 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m in mets:
        lines.append(f"| {m['label']} | {m.get('input_variant','sel')} | {m['cfg'].get('residual')} | "
                     f"{len(m['cols'])} | {m['L']} | {m['H']} | {m['best_epoch']} | "
                     f"{m['epochs_run']} | {m['sec_per_epoch']}s | {m['train_min']} min |")
    lines.append("\n超参：" + json.dumps(mets[0]["cfg"], ensure_ascii=False))
    lines += ["\n## 5. 输入变体说明（重要）\n",
              "阶段四主模型输入采用**原始特征矩阵（仅按 MIV 选中列裁剪）**，而非阶段二的 RPCA 低秩矩阵 L。\n",
              "对照证据（同一模型/超参，见 `reports/probe_input_variant.log`）：\n",
              "| 数据集 | 输入 | val RMSE | test RMSE | MAPE | R² |",
              "|---|---|---|---|---|---|",
              "| 荆门 | RPCA-L | 0.0782 | 135.0 | 5.50% | 0.625 |",
              "| 荆门 | 原始 | 0.0297 | 55.4 | 2.34% | 0.937 |",
              "| 兰州 | RPCA-L | 0.0410 | 1140.2 | 7.59% | 0.324 |",
              "| 兰州 | 原始 | 0.0142 | 594.8 | 3.24% | 0.816 |",
              "\n结论：RPCA 的低秩重构会平滑掉对 1 步预测最关键的近期高频信息，作为预测输入是净负面；"
              "其在本文中的作用调整为「异常定位（S 矩阵）+ MIV 第二视角（稳健性校核）」，"
              "并在阶段六以消融行形式量化。\n",
              "\n## 6. 图表索引\n",
              "| 文件 | 内容 |", "|---|---|",
              "| `fig4_arch.png` | 模型结构示意 |",
              "| `fig4_train_{jingmen,lanzhou}.png` | 训练损失 / 验证 RMSE / 早停 |",
              "| `fig4_pred_*.png` | 测试集预测对比（全时段 + 前 72 h 放大） |",
              "| `fig4_scatter_*.png` | 预测散点 + R²/RMSE/MAPE |",
              "| `fig4_hstep_jingmen.png` | 多步（h1–h4）误差柱状 |",
              "| `fig4_err_*.png` | 残差分布与残差-预测值关系 |"]
    open(os.path.join(BASE, "reports", "stage4_results.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n报告：reports/stage4_results.md")


def load_tuned_cfg(name):
    """优先使用调参定稿（阶段五/tuner），否则回退默认超参。"""
    p = os.path.join(BASE, "reports", "tuning", f"best_{name}_bilstm_itransformer_gate.json")
    q = os.path.join(BASE, "reports", "tuning", f"best_{name}_bilstm_itransformer.json")
    cands = [x for x in (p, q) if os.path.exists(x)]
    if not cands:
        return {**T.DEFAULT_CFG}, "default"
    best = None
    for x in cands:
        b = json.load(open(x, encoding="utf-8"))
        if best is None or b.get("val_mape_mean", 1e9) < best[0]:
            best = (b.get("val_mape_mean", 1e9), x, b)
    return {**T.DEFAULT_CFG, **best[2]["cfg"]}, os.path.basename(best[1])


def archive_models(names):
    """把旧权重移入 models/archive/，避免「看起来还在、实际不可用」。"""
    ad = os.path.join(MOD, "archive")
    os.makedirs(ad, exist_ok=True)
    for n in names:
        p = os.path.join(MOD, f"{n}_best.pt")
        if os.path.exists(p):
            import datetime as _dt
            dst = os.path.join(ad, f"{n}_best_{_dt.datetime.now():%Y%m%d_%H%M%S}.pt")
            os.replace(p, dst)
            print(f"  归档旧权重 -> {os.path.relpath(dst, BASE)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None, choices=[None, "jingmen", "lanzhou"])
    ap.add_argument("--dataset", dest="dataset_opt", default=None, choices=["jingmen", "lanzhou"])
    ap.add_argument("--tuned", action="store_true", default=True, help="使用调参定稿超参（默认）")
    ap.add_argument("--no-tuned", dest="tuned", action="store_false", help="回退默认超参")
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()
    only = a.dataset_opt or a.dataset
    names = [only] if only else list(CONFIG.keys())

    if not a.no_archive:
        archive_models(names)

    cfgs = []
    for n in names:
        c = dict(CONFIG[n])
        if a.tuned:
            c["train_cfg"], src = load_tuned_cfg(n)
            print(f"  {n} 超参来源：{src}")
        else:
            c["train_cfg"] = {**T.DEFAULT_CFG}
        cfgs.append(c)

    fig_arch(os.path.join(FIG, "fig4_arch.png"), cfgs[0]["train_cfg"])
    mets = [process(n, c) for n, c in zip(names, cfgs)]
    if not only:
        write_report(mets)
    print("\n阶段四完成。")


if __name__ == "__main__":
    main()
