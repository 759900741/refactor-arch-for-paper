# -*- coding: utf-8 -*-
"""
stage6_experiments.py
=====================
阶段六：消融 / 模型对比 / 跨场景泛化实验统一编排器。

用法：
    cd src && py -3.8 stage6_experiments.py --mode fe        # 特征工程消融（Table 6/7）
    cd src && py -3.8 stage6_experiments.py --mode arch      # 架构消融（含 B6 串行/并行）
    cd src && py -3.8 stage6_experiments.py --mode compare   # 模型对比（Table 8）
    cd src && py -3.8 stage6_experiments.py --mode generalize# 跨场景泛化（Table 9/10）
    cd src && py -3.8 stage6_experiments.py --mode all       # 全部
    选项：--dataset jingmen|lanzhou  --epochs N  --force  --dry-run  --no-tuned

超参来源（v2 口径）：
    - 本文模型 / 对比模型：一律读 reports/tuning/best_{ds}_{key}.json（调参定稿结果）；
      compare / generalize 行直接复用 reports/metrics/tuned_{ds}_{key}.json（免重训）。
    - 特征消融 A0–A3、架构消融 B2/B3/B5 共用本文最终 cfg，只改 variant / head_mode /
      use_bilstm / residual——消融要隔离结构变量，而不是比谁调参运气好。
    - 缓存键含配置指纹：reports/metrics/stage6_cache_v2.json，key = dataset|tag|protocol|cfg_hash。
      阶段四「完整模型」行仅在 cfg_hash 与 protocol 都吻合时才复用其预测，否则重训。
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T

BASE = T.BASE
P = T.P
MET = os.path.join(BASE, "reports", "metrics")
FIG = os.path.join(BASE, "reports", "figs")
TUN = os.path.join(BASE, "reports", "tuning")
CACHE = os.path.join(MET, "stage6_cache_v2.json")
os.makedirs(MET, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.set_num_threads(4)

CONFIG = {"jingmen": dict(label="荆门市 2024", unit="MW"),
          "lanzhou": dict(label="兰州市 2023", unit="kW")}

PEAK_H = (8, 9, 10, 11, 12, 18, 19, 20, 21)
VALLEY_H = (23, 0, 1, 2, 3, 4, 5)

#: 对比模型对应的调参 key（与 tuner.model_key 保持一致）
CMP_KEYS = [("BiLSTM", "bilstm"), ("BiLSTM-Transformer", "transformer"),
            ("BiLSTM-Informer", "informer")]


# ---------------------------------------------------------------------------
# 超参来源
# ---------------------------------------------------------------------------
def _best_path(ds, key):
    return os.path.join(TUN, f"best_{ds}_{key}.json")


def hybrid_key(ds="jingmen"):
    """本文模型的调参 key：并行门控版本若存在且更优则用它（由 tuner.report 决定）。"""
    gate = _best_path(ds, "bilstm_itransformer_gate")
    serial = _best_path(ds, "bilstm_itransformer")
    cands = []
    for p, k in ((gate, "bilstm_itransformer_gate"), (serial, "bilstm_itransformer")):
        if os.path.exists(p):
            b = json.load(open(p, encoding="utf-8"))
            cands.append((b.get("val_mape_mean", 1e9), k))
    if not cands:
        return "bilstm_itransformer"
    return min(cands)[1]


def tuned_cfg(ds, key=None, required=True):
    """读取调参定稿的完整有效 cfg（含 epochs/batch/monitor/seed）。"""
    key = key or hybrid_key(ds)
    p = _best_path(ds, key)
    if os.path.exists(p):
        b = json.load(open(p, encoding="utf-8"))
        return {**T.DEFAULT_CFG, **b["cfg"]}, key
    if required:
        raise FileNotFoundError(f"缺少调参结果 {p}，请先运行 tuner.py finalize")
    return {**T.DEFAULT_CFG}, key


def _exp(tag, variant, model_type, residual=True, head_mode="target", use_bilstm=True,
         add_lag=False, fusion=None):
    return dict(tag=tag, variant=variant, add_lag=add_lag,
                cfg=dict(model_type=model_type, residual=residual,
                         head_mode=head_mode, use_bilstm=use_bilstm,
                         **({} if fusion is None else {"fusion": fusion})))


def fe_experiments(name, fusion=None):
    """特征工程消融：A0 无FE / A1 仅RPCA-L / A2 MIV(完整) / A3 +lag。"""
    return [
        _exp("A0_无特征工程", "full", "bilstm_itransformer", fusion=fusion),
        _exp("A1_仅RPCA_L", "lfull", "bilstm_itransformer", fusion=fusion),
        _exp("A2_MIV完整", "selorig", "bilstm_itransformer", fusion=fusion),
        _exp("A3_加lag", "selorig", "bilstm_itransformer", add_lag=True, fusion=fusion),
    ]


def arch_experiments(name, fusion=None):
    """架构消融：B1 完整 / B2 均值池化头 / B3 iTransformer-only / B4 BiLSTM-only /
    B5 无残差 / B6 融合方式对调（串行 vs 并行门控）。"""
    flip = None
    if fusion in ("gate", "parallel"):
        flip = "serial"
    elif fusion == "serial":
        flip = "gate"
    out = [
        _exp("B1_完整", "selorig", "bilstm_itransformer", fusion=fusion),
        _exp("B2_均值池化头", "selorig", "bilstm_itransformer", head_mode="mean", fusion=fusion),
        _exp("B3_iTransformerOnly", "selorig", "bilstm_itransformer", use_bilstm=False, fusion=fusion),
        _exp("B4_BiLSTMOnly", "selorig", "bilstm", fusion=fusion),
        _exp("B5_无残差", "selorig", "bilstm_itransformer", residual=False, fusion=fusion),
    ]
    if flip:
        tag = "B6_并行门控融合" if flip == "gate" else "B6_串行融合"
        out.append(_exp(tag, "selorig", "bilstm_itransformer", fusion=flip))
    return out


def compare_experiments(name, fusion=None):
    """模型对比：BiLSTM / BiLSTM-Transformer / BiLSTM-Informer / 本文。"""
    return [_exp(t, "selorig", k) for t, k in CMP_KEYS] + \
           [_exp("本文_BiLSTM-iTransformer", "selorig", "bilstm_itransformer", fusion=fusion)]


def tag_key(tag, fusion=None):
    """tag -> tuner 的调参 key（用于取 tuned 结果）。"""
    for t, k in CMP_KEYS:
        if tag == t:
            return k
    if tag.startswith("本文"):
        return "bilstm_itransformer_gate" if fusion in ("gate", "parallel") else "bilstm_itransformer"
    return None


# ---------------------------------------------------------------------------
# 指标与分块
# ---------------------------------------------------------------------------
def hour_group(h):
    if h in PEAK_H:
        return "peak"
    if h in VALLEY_H:
        return "valley"
    return "flat"


def compute_split_metrics(y_true_orig, y_pred_orig, end_time):
    """整体 + 工作日/周末 + 峰/平/谷 分块指标（原始量纲）。"""
    yt = np.asarray(y_true_orig, dtype=np.float64)
    yp = np.asarray(y_pred_orig, dtype=np.float64)
    out = dict(overall=T.metrics(yt, yp))
    work = []; wknd = []; grp = dict(peak=[], valley=[], flat=[])
    for i, ts in enumerate(end_time):
        dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        (work if dt.weekday() < 5 else wknd).append(i)
        grp[hour_group(dt.hour)].append(i)
    out["workday"] = T.metrics(yt[work], yp[work]) if work else None
    out["weekend"] = T.metrics(yt[wknd], yp[wknd]) if wknd else None
    for k in ("peak", "valley", "flat"):
        out[k] = T.metrics(yt[grp[k]], yp[grp[k]]) if grp[k] else None
    return out


def seed_from_stage4(name, tag, expect_hash):
    """「完整模型」行：仅当阶段四的 cfg 指纹与当前调参定稿一致时复用其预测。"""
    mpath = os.path.join(MET, f"{name}_stage4.json")
    if not os.path.exists(mpath):
        return None
    m = json.load(open(mpath, encoding="utf-8"))
    if m.get("cfg_hash") != expect_hash or m.get("protocol") != "full":
        return None
    pred = os.path.join(P, "preds", f"{name}_test.npz")
    if not os.path.exists(pred):
        return None
    res = dict(dataset=name, label=CONFIG[name]["label"], tag=tag, seeded=True,
               model_type=m["cfg"]["model_type"], variant=m.get("input_variant", "selorig"),
               residual=m["cfg"].get("residual"), head_mode=m["cfg"].get("head_mode"),
               use_bilstm=m["cfg"].get("use_bilstm"), fusion=m["cfg"].get("fusion"),
               add_lag=False, cfg=m["cfg"], cfg_hash=m["cfg_hash"],
               L=m["L"], H=m["H"], F=len(m["cols"]), params=m.get("params"),
               best_epoch=m["best_epoch"], train_min=m.get("train_min"),
               val_rmse=m["val"]["rmse"], protocol="full")
    pr = np.load(pred, allow_pickle=True)
    res["test"] = compute_split_metrics(pr["y_true_orig"], pr["y_pred_orig"], pr["end_time"])
    pv = np.load(os.path.join(P, "preds", f"{name}_val.npz"), allow_pickle=True)
    res["val_overall"] = T.metrics(pv["y_true_orig"], pv["y_pred_orig"])
    return res


def load_tuned_result(ds, key):
    p = os.path.join(MET, f"tuned_{ds}_{key}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    r["seeded"] = False
    return r


# ---------------------------------------------------------------------------
# 单次实验
# ---------------------------------------------------------------------------
def run_one(name, exp, base_cfg, epochs_override=None, log=print):
    lo, hi = T.load_y_range(name)
    d = T.load_split(name, exp["variant"], add_lag=exp["add_lag"])
    cfg = {**T.DEFAULT_CFG, **base_cfg, **exp["cfg"]}
    if exp["cfg"].get("fusion") is None:
        cfg["fusion"] = base_cfg.get("fusion", "serial")
    if cfg.get("fusion") == "serial":
        cfg["mv_hidden"] = None
    if epochs_override:
        cfg["epochs"] = int(epochs_override)
    cfg["batch"] = T.auto_batch(cfg, name) if not epochs_override else cfg.get("batch", 128)
    cfg["target_idx"] = d["cols"].index(d["target"])
    log(f"  [{exp['tag']}] F={len(d['cols'])}, model={cfg['model_type']}, "
        f"fusion={cfg.get('fusion')}, residual={cfg['residual']}, head={cfg['head_mode']}, "
        f"use_bilstm={cfg['use_bilstm']}, hidden={cfg.get('hidden')}, d_model={cfg.get('d_model')}")
    model, hist, best = T.fit_model(d, cfg, log=log)
    pt = T.predict(model, d["test"]["X"])
    yt_o = T.inverse(d["test"]["y"], lo, hi)
    pt_o = T.inverse(pt, lo, hi)
    et = np.array([x.strftime("%Y-%m-%d %H:%M:%S") for x in d["time"]["test"]])
    return dict(dataset=name, label=CONFIG[name]["label"], tag=exp["tag"], seeded=False,
                model_type=cfg["model_type"], variant=exp["variant"],
                residual=cfg["residual"], head_mode=cfg["head_mode"],
                use_bilstm=cfg["use_bilstm"], fusion=cfg.get("fusion"), add_lag=exp["add_lag"],
                F=len(d["cols"]), L=d["L"], H=d["H"], cfg=cfg,
                cfg_hash=T.cfg_hash(cfg), protocol="full",
                params=int(sum(p.numel() for p in model.parameters())),
                best_epoch=int(hist["best_epoch"]),
                train_min=round(float(np.sum(hist["epoch_sec"])) / 60.0, 2),
                val_rmse=float(best),
                test=compute_split_metrics(yt_o, pt_o, et),
                val_overall=T.metrics(T.inverse(d["val"]["y"], lo, hi),
                                      T.inverse(T.predict(model, d["val"]["X"]), lo, hi)))


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def load_cache():
    if not os.path.exists(CACHE):
        return {}
    d = json.load(open(CACHE, encoding="utf-8"))
    return d.get("entries", d)


def save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({"schema": "v2", "entries": cache}, f, ensure_ascii=False, indent=2)


def ckey(name, tag, cfg, protocol="full"):
    return f"{name}|{tag}|{protocol}|{T.cfg_hash(cfg)}"


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def fig_ablation_bar(results, title, fname, metric="mape"):
    rows = [r for r in results if r is not None]
    if not rows:
        return
    labels = [r["tag"] for r in rows]
    vals = [r["test"]["overall"][metric] for r in rows]
    best = int(np.argmin(vals))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rows)), 4.2))
    colors = ["#c53030" if i == best else "#4299e1" for i in range(len(rows))]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"测试集 {metric.upper()}（原始量纲）")
    ax.set_title(title + f"（最优={labels[best]}）")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


def fig_residual(results, dataset, unit, fname):
    r = next((x for x in results if x and x.get("seeded")), None)
    if r is None:
        return
    pr = np.load(os.path.join(P, "preds", f"{dataset}_test.npz"), allow_pickle=True)
    yt, yp = pr["y_true_orig"][:, 0], pr["y_pred_orig"][:, 0]
    tt = [datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S") for s in pr["end_time"]]
    n = min(168, len(tt))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4))
    axes[0].plot(tt[:n], yt[:n], lw=1.2, color="#2d3748", label="真实")
    axes[0].plot(tt[:n], yp[:n], lw=1.0, color="#c53030", alpha=.85, label="预测")
    axes[0].legend(fontsize=8); axes[0].set_ylabel(f"负荷（{unit}）")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    axes[1].plot(tt[:n], (yt - yp)[:n], lw=.9, color="#2f855a")
    axes[1].axhline(0, color="#718096", ls="--", lw=.8)
    axes[1].set_ylabel(f"残差（{unit}）"); axes[1].set_xlabel("时间")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    for a in axes:
        a.grid(alpha=.25)
    fig.suptitle(f"{CONFIG[dataset]['label']} 完整模型测试集残差（前 7 天）", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------
def table_md(title, results, metrics=("rmse", "mae", "mape", "r2", "rpd", "ia")):
    lines = [f"\n## {title}\n",
             "| 变体 | " + " | ".join(m.upper() for m in metrics) + " | 参数量 | 配置指纹 |",
             "|" + "---|" * (len(metrics) + 3)]
    for r in results:
        if r is None:
            continue
        o = r["test"]["overall"]
        cells = []
        for m in metrics:
            v = o[m]
            cells.append(f"{v:.2f}" if m in ("mape", "rmse", "mae", "rpd", "ia") else f"{v:.4f}")
        p = r.get("params")
        lines.append(f"| {r['tag']} | " + " | ".join(cells)
                     + f" | {p//1000}k | {str(r.get('cfg_hash'))[:8]} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def collect(mode, dataset, epochs_override, force, tuned=True):
    cache = load_cache()
    hcfg, hkey = tuned_cfg(dataset, required=tuned)
    fusion = hcfg.get("fusion", "serial")

    if mode == "fe":
        exps = [(dataset, e) for e in fe_experiments(dataset, fusion)]
        base = hcfg
    elif mode == "arch":
        exps = [(dataset, e) for e in arch_experiments(dataset, fusion)]
        base = hcfg
    elif mode in ("compare", "generalize"):
        exps = [(dataset, e) for e in compare_experiments(dataset, fusion)]
        base = hcfg
    elif mode == "all":
        return None
    else:
        raise ValueError(mode)

    results = []
    for name, exp in exps:
        key = tag_key(exp["tag"], fusion)
        # 1) compare/generalize 且已有 tuned 结果 -> 直接复用（test 只评一次）
        if mode in ("compare", "generalize") and tuned:
            tr = load_tuned_result(name, key)
            if tr is not None and not force:
                results.append(tr)
                print(f"  [tuned] {name} | {exp['tag']} <- tuned_{name}_{key}.json")
                continue
        # 2) 完整模型行：阶段四权重指纹吻合则直接复用
        if exp["tag"] in ("A2_MIV完整", "B1_完整", "本文_BiLSTM-iTransformer") and tuned \
                and exp["variant"] == "selorig":
            cfg_hash = T.cfg_hash({**hcfg, **exp["cfg"]})
            ck = ckey(name, exp["tag"], {**hcfg, **exp["cfg"]})
            if ck in cache and not force:
                results.append(cache[ck]); print(f"  [skip] {ck[:40]} 已缓存"); continue
            seeded = seed_from_stage4(name, exp["tag"], cfg_hash)
            if seeded is not None:
                cache[ck] = seeded; save_cache(cache); results.append(seeded)
                print(f"  [seed] {name} | {exp['tag']} <- 阶段四基线（指纹一致）")
                continue
        # 3) 训练
        exp_cfg = {**hcfg, **exp["cfg"]}
        ck = ckey(name, exp["tag"], exp_cfg)
        if ck in cache and not force:
            print(f"  [skip] {exp['tag']} 已缓存"); results.append(cache[ck]); continue
        res = run_one(name, exp, base, epochs_override)
        cache[ck] = res; save_cache(cache); results.append(res)
    return results


def run_mode(mode, dataset, epochs_override, force, tuned=True):
    results = collect(mode, dataset, epochs_override, force, tuned)
    if results is None:
        return
    if mode in ("fe", "arch"):
        fig_ablation_bar(results, f"{mode} 消融测试集 MAPE", f"fig6_{mode}_mape.png")
        fig_ablation_bar(results, f"{mode} 消融测试集 RMSE", f"fig6_{mode}_rmse.png", "rmse")
    if mode in ("compare", "generalize"):
        fig_ablation_bar(results, f"{mode} 测试集 MAPE", f"fig6_{mode}_mape.png")
        fig_residual(results, dataset, CONFIG[dataset]["unit"], f"fig6_{mode}_residual.png")

    md = [f"# 阶段六 {mode} 结果（{CONFIG[dataset]['label']}）\n",
          f"> 生成：{datetime.now():%Y-%m-%d %H:%M:%S} ｜ 超参：tuned={'是' if tuned else '否'}\n"]
    md.append(table_md("测试集整体指标（原始量纲）", results))
    md.append("\n## 分时段/工作日指标\n")
    md.append("| 变体 | 工作日 MAPE | 周末 MAPE | 峰 MAPE | 平 MAPE | 谷 MAPE |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        if r is None:
            continue
        g = lambda k: (r["test"].get(k) or {}).get("mape", float("nan"))  # noqa: E731
        md.append(f"| {r['tag']} | {g('workday'):.2f} | {g('weekend'):.2f} | "
                  f"{g('peak'):.2f} | {g('flat'):.2f} | {g('valley'):.2f} |")
    out = os.path.join(MET, f"stage6_{mode}.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(BASE, "reports", f"stage6_{mode}.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"  结果: {os.path.relpath(out, BASE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fe", choices=["fe", "arch", "compare", "generalize", "all"])
    ap.add_argument("--dataset", default="jingmen", choices=["jingmen", "lanzhou"])
    ap.add_argument("--epochs", default=None, help="覆盖训练轮数（快速验证用）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只列出实验不训练")
    ap.add_argument("--no-tuned", action="store_true", help="不使用调参定稿，回退默认超参")
    args = ap.parse_args()
    tuned = not args.no_tuned

    if args.dry_run:
        for ds, m in (("jingmen", "fe"), ("jingmen", "arch"),
                      (args.dataset, "compare"), ("lanzhou", "generalize")):
            hcfg, hkey = tuned_cfg(ds, required=False)
            print(f"\n== {ds} / {m}  (本文 cfg key={hkey}, fusion={hcfg.get('fusion')})")
            for e in (fe_experiments(ds) if m == "fe" else
                      arch_experiments(ds, hcfg.get("fusion")) if m == "arch" else
                      compare_experiments(ds, hcfg.get("fusion"))):
                print(f"   {e['tag']:<24} variant={e['variant']:<8} {e['cfg']}")
        return

    if args.mode == "all":
        for m, ds in (("fe", "jingmen"), ("arch", "jingmen"),
                      ("compare", args.dataset), ("generalize", "lanzhou")):
            print(f"\n{'='*64}\n模式 {m}（dataset={ds}）")
            run_mode(m, ds, args.epochs, args.force, tuned)
    else:
        run_mode(args.mode, args.dataset, args.epochs, args.force, tuned)


if __name__ == "__main__":
    main()
