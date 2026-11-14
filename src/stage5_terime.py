# -*- coding: utf-8 -*-
"""
stage5_terime.py
================
阶段五：TE-RIME 自适应超参数优化 + 优化算法对比（论文表 5 / 图 10）。

复用 `tuner` 的搜索空间与评估器，保证「搜索时评估的」与「最终交付的」完全一致：
    - 同空间：space_for(model_key) + join_space/repair，5 种算法边界完全一致；
    - 同口径：tuner.evaluate（fast 协议、验证集 MAPE、绝不读 test）；
    - 同预算：pop × iters 的唯一评估次数，5 种算法逐一相同。

用法：
    py -3.8 stage5_terime.py --dataset jingmen --model bilstm_itransformer --algo all --evals 100
    py -3.8 stage5_terime.py --dataset jingmen --model bilstm_itransformer --algo terime --evals 500
    py -3.8 stage5_terime.py --dataset jingmen --report
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T
import tuner as U
import metaheuristics as MH

BASE = T.BASE
MET = os.path.join(BASE, "reports", "metrics")
FIG = os.path.join(BASE, "reports", "figs")
os.makedirs(MET, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ALGOS = ["pso", "bwo", "gwo", "rime", "terime"]
CNAME = {"pso": "PSO", "bwo": "BWO", "gwo": "GWO", "rime": "RIME", "terime": "TE-RIME"}
COLOR = {"pso": "#4299e1", "bwo": "#38a169", "gwo": "#d69e2e",
         "rime": "#805ad5", "terime": "#c53030"}


def write_json(path, value):
    """Replace a complete JSON checkpoint atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def verification_identity(dataset, key, cfg, seeds):
    """Identify the candidate, data, backend and every full-training seed."""
    backend = str(T.resolve_device(cfg))
    return dict(dataset=dataset, key=key, protocol=U.FULL.as_dict(),
                fingerprints=[U.fingerprint(dataset, "selorig", False,
                              U._effective(cfg, dataset, U.FULL, sd), U.FULL, backend)
                              for sd in seeds], seeds=list(seeds))


def decode_cfg(u, dims, key):
    raw = U.repair(U.join_space(dims, u), key)
    raw.setdefault("model_type", "bilstm_itransformer" if "bilstm_itransformer" in key else key)
    return raw


def run_one(dataset, key, algo, evals, pop, seed, protocol=U.FAST, patience=None,
            warmstart=False, tag=None, output_dir=None, train_seed=None,
            restart_after=None, deadline_at=None, cache_dir=None, refine=True):
    dims = U.space_for(key)
    train_seed = seed if train_seed is None else train_seed
    folder = output_dir or MET
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(folder, f"stage5_{dataset}_{key}_{algo}{suffix}.json")
    ledger_path = path.replace(".json", "_ledger.json")
    space_hash = hashlib.sha1(json.dumps([d.to_json() for d in dims],
                                       sort_keys=True).encode()).hexdigest()[:12]
    # restart_after 只对 TE-RIME 生效（S6 准对立重启），基线算法保持原样
    restart_eff = restart_after if algo == "terime" else None
    request = dict(dataset=dataset, key=key, algo=algo, budget=evals, pop=pop,
                   search_seed=seed, train_seed=train_seed, protocol=protocol.as_dict(),
                   space_hash=space_hash, restart_after=restart_eff,
                   algorithm_version=MH.ALGORITHM_VERSION, warmstart=warmstart,
                   patience=patience, data=U._dcache_fp(dataset, "selorig", False),
                   code=U.CODE_VERSION, backend=str(T.resolve_device()),
                   isolate=os.path.basename(cache_dir.rstrip("/\\")) if cache_dir else None)
    ledger = dict(request=request, records=[], started_at=time.time())
    if os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as stream:
            ledger = json.load(stream)
        if ledger["request"] != request:
            raise ValueError("Checkpoint request differs; choose a new output directory/tag")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("complete") and previous.get("request") == request:
            print(f"  [completed] {algo} {previous['evals']}/{evals}", flush=True)
            return previous
    cursor = 0

    def identity(u):
        cfg = decode_cfg(np.asarray(u), dims, key)
        return T.cfg_hash({**T.DEFAULT_CFG, **cfg})

    def f(u):
        nonlocal cursor
        cfg = decode_cfg(np.asarray(u), dims, key)
        cfg_id = identity(u)
        if cursor < len(ledger["records"]):
            saved = ledger["records"][cursor]
            if saved["cfg_hash"] != cfg_id or not np.array_equal(saved["u"], u):
                raise ValueError("Deterministic checkpoint replay diverged")
            cursor += 1
            return saved["val_mape"]
        if deadline_at is not None and time.time() >= deadline_at:
            raise T.TrainingDeadlineExceeded({})
        result = U.evaluate(dataset, cfg, protocol, seed=train_seed,
                            log=lambda *a, **kw: None, deadline_at=deadline_at,
                            cache_dir=cache_dir)
        record = dict(index=cursor + 1, u=np.asarray(u).tolist(), cfg_hash=cfg_id,
                      cfg=result["cfg"], val_mape=float(result["val_mape"]),
                      cache_hit=result["cache_hit"], training_secs=result["secs"],
                      fingerprint=result["fp"], elapsed_secs=time.time() - ledger["started_at"])
        ledger["records"].append(record)
        cursor += 1
        write_json(ledger_path, ledger)
        print(f"    {algo} {cursor}/{evals}: val={record['val_mape']:.5f}% "
              f"cache={record['cache_hit']} train={record['training_secs']:.1f}s", flush=True)
        return record["val_mape"]

    x0 = None
    if warmstart:
        if "warmstart_u" not in ledger:
            cfgs = U.top_cfgs(dataset, key, pop)
            ledger["warmstart_u"] = [U.u_for(dims, cfg).tolist() for cfg in cfgs]
            write_json(ledger_path, ledger)
        x0 = ledger["warmstart_u"]
        print(f"  warm-start：注入 {len(x0)} 个 fast 最优配置作为初始种群")

    print(f"\n[{CNAME[algo]}] {dataset} | {key} | dims={len(dims)} | pop={pop} | 预算 {evals} 次唯一评估")
    try:
        r = MH.run(algo, len(dims), f, pop=pop, iters=max(1, evals // max(pop, 1)), seed=seed,
                   budget=evals, patience=patience, x0=x0, identity=identity,
                   restart_after=restart_eff,
                   grid=[len(d.values) for d in dims], refine=refine)
    except T.TrainingDeadlineExceeded as error:
        write_json(path.replace(".json", "_interrupted.json"),
                   dict(request=request, status="incomplete", history=error.history,
                        completed_evals=len(ledger["records"]), stopped_at=time.time()))
        raise
    best_cfg = decode_cfg(r["best_u"], dims, key)
    out = dict(dataset=dataset, key=key, algo=algo, model_type=best_cfg.get("model_type"),
               best_fit=r["best_fit"], best_cfg=T.canonical_cfg({**T.DEFAULT_CFG, **best_cfg}),
               history=r["history"], evals=r["evals"],
               wall_sec=round(time.time() - ledger["started_at"], 2),
               execution_wall_sec=r["wall_sec"], seed=seed,
               pop=pop, iters=r["iters"], budget=r["budget"], protocol=protocol.tag,
               space_hash=space_hash, train_seed=train_seed, request=request,
               algorithm_version=r["algorithm_version"], proposals=r["proposals"],
               initial_u=r["initial_u"], restarts=r["restarts"],
               probes=r.get("probes", 0), qobl_restarts=r.get("qobl_restarts", 0),
               op_stats=r.get("op_stats"), op_family_log=r.get("op_family_log"),
               complete=r["evals"] == evals,
               cache_hits=sum(x["cache_hit"] for x in ledger["records"]),
               new_trainings=sum(not x["cache_hit"] for x in ledger["records"]),
               train_secs=round(sum(x["training_secs"] for x in ledger["records"]
                                    if not x["cache_hit"]), 1),
               evaluation_history=ledger["records"],
               created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"  -> best val_mape={r['best_fit']:.4f}%  evals={r['evals']}  {r['wall_sec']:.0f}s")
    print(f"     最优 cfg: {U._brief(best_cfg)}")
    write_json(path, out)
    return out


def report(dataset, key):
    rows = []
    for a in ALGOS:
        p = os.path.join(MET, f"stage5_{dataset}_{key}_{a}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p, encoding="utf-8")))
    if not rows:
        print("  无阶段五结果"); return
    rows.sort(key=lambda r: r["best_fit"])
    ref = rows[0]["best_fit"]

    # 各算法「最优解」轻量定稿：把 cfg + 关键口径单独落一份，便于归档/跨机复用
    # （整份 stage5_*.json 含逐次台账，体积大；摘要文件小到可直接读）
    for r in rows:
        write_json(os.path.join(U.TUN, f"stage5_best_{dataset}_{key}_{r['algo']}.json"),
                   dict(dataset=dataset, key=key, algo=r["algo"], algo_name=CNAME[r["algo"]],
                        best_fit=round(float(r["best_fit"]), 5), best_cfg=r["best_cfg"],
                        space_hash=r.get("space_hash"), budget=r.get("budget"),
                        evals=r.get("evals"), pop=r.get("pop"), iters=r.get("iters"),
                        protocol=r.get("protocol"), seed=r.get("seed"),
                        new_trainings=r.get("new_trainings"), train_secs=r.get("train_secs"),
                        algorithm_version=r.get("algorithm_version"),
                        source=f"reports/metrics/stage5_{dataset}_{key}_{r['algo']}.json",
                        created_at=time.strftime("%Y-%m-%d %H:%M:%S")))

    def evals_to(h, t):
        for x in h:
            if x["best_fit"] <= t:
                return x["evals"]
        return h[-1]["evals"]

    def auc(h):
        """best-so-far 相对评估数的归一化面积（越小越快），用于 anytime 比较。"""
        xs = np.array([x["evals"] for x in h], dtype=float)
        ys = np.array([x["best_fit"] for x in h], dtype=float)
        span = max(xs[-1] - xs[0], 1.0)
        base = max(ys[0] - ys[-1], 1e-9)
        return float(np.trapz((ys[0] - ys) / base, xs) / span)

    e101 = {r["algo"]: evals_to(r["history"], ref * 1.01) for r in rows}
    e102 = {r["algo"]: evals_to(r["history"], ref * 1.02) for r in rows}

    lines = [f"# 阶段五 优化算法对比 —— {U.LABEL[dataset]} / {U._tag_of(key)}\n",
             f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 同空间、同初始种群、同评估口径、同预算\n",
             f"> 算法版本：{rows[0].get('algorithm_version', '—')} ｜ 空间指纹：{rows[0].get('space_hash', '—')}"
             f" ｜ 缓存：{'各算法独立（同等初始快照）' if rows[0].get('request', {}).get('isolate') else '共享'}\n",
             "| 算法 | 最优验证 MAPE(%) | 相对最优 | 唯一评估 | 达1.01×评估 | 达1.02×评估 | 曲线AUC | 新训练数 | 训练耗时(s) | 最优超参 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {CNAME[r['algo']]} | {r['best_fit']:.4f} | "
                     f"{(r['best_fit'] - ref) / ref * 100:+.2f}% | {r['evals']} | "
                     f"{e101[r['algo']]} | {e102[r['algo']]} | {auc(r['history']):.3f} | "
                     f"{r.get('new_trainings', '—')} | {r.get('train_secs', r['wall_sec']):.0f} | "
                     f"{U._brief(r['best_cfg'])} |")
    lines.append("")
    lines.append("> 说明：`达1.01×/1.02×评估` 为首次达到「全局最优 ×1.01 / ×1.02」所需唯一评估数；"
                 "`曲线AUC` 为 best-so-far 归一化面积（越小=越早收敛）；"
                 "`新训练数/训练耗时` 为真实发生的训练，排除缓存命中，用于等算力比较。")
    lines.append("")
    with open(os.path.join(BASE, "reports", "stage5_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.2))
    for r in rows:
        h = r["history"]
        ax[0].plot([x["evals"] for x in h], [x["best_fit"] for x in h],
                   label=CNAME[r["algo"]], color=COLOR[r["algo"]], lw=1.4)
    ax[0].set_xlabel("唯一评估次数（等价训练次数）"); ax[0].set_ylabel("最优验证集 MAPE（%）")
    ax[0].set_title("收敛曲线（等预算）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

    names = [CNAME[r["algo"]] for r in rows][::-1]
    vals = [r["best_fit"] for r in rows][::-1]
    cols = [COLOR[r["algo"]] for r in rows][::-1]
    ax[1].barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax[1].text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax[1].set_xlabel("最优验证集 MAPE（%）")
    ax[1].set_title("各算法最终精度（越低越好）")
    ax[1].set_xlim(0, max(vals) * 1.15)
    ax[1].grid(axis="x", alpha=.25)

    e_vals = [e102[r["algo"]] for r in rows][::-1]
    ax[2].barh(names, e_vals, color=cols)
    for i, v in enumerate(e_vals):
        ax[2].text(v, i, f" {v}", va="center", fontsize=8)
    ax[2].set_xlabel("达到最优×1.02 所需唯一评估次数")
    ax[2].set_title("达阈效率（越小越快）")
    ax[2].set_xlim(0, max(e_vals) * 1.2)
    ax[2].grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig5_convergence_{dataset}.png"), dpi=150)
    plt.close(fig)
    print(f"  报告：reports/stage5_results.md ｜ 图：reports/figs/fig5_convergence_{dataset}.png")

    # 注：定稿仲裁（full 复验 + 替换决策）由独立子命令 --arbitrate 触发，
    # 避免「表5 对比阶段」就误触发约 30min 的 full 复验；且仲裁应基于 TE-RIME 长跑结果。


def verify_full(dataset, key, cfg, seeds=(42, 43)):
    """full 协议 × 多 seed 复验，返回 (mean, std, per_seed)。"""
    vals = []
    for sd in seeds:
        r = U.evaluate(dataset, cfg, U.FULL, seed=sd, log=lambda *a, **k: None)
        vals.append(r["val_mape"])
        print(f"    full 复验 seed={sd} val_mape={r['val_mape']:.4f}% ({r['secs']:.0f}s)")
    return float(np.mean(vals)), float(np.std(vals)), vals


def arbitrate(dataset, key, te_cfg, te_fast_val, space_hash, te_evals):
    """full 复验 + 定稿仲裁：TE-RIME 更优才替换现有定稿，否则沿用。可断点（已有 full 结果则跳过）。"""
    best_te = os.path.join(U.TUN, f"best_{dataset}_{key}_terime.json")
    expected_identity = verification_identity(dataset, key, te_cfg, (42, 43))
    if os.path.exists(best_te):
        b = json.load(open(best_te, encoding="utf-8"))
        if b.get("val_mape_full_mean") is not None and b.get("verification_identity") == expected_identity:
            print(f"  [仲裁] 已有 full 复验结果，跳过（val_mape_full_mean={b['val_mape_full_mean']}）")
            return b
    print(f"\n[仲裁] {dataset} | {key} | full×{len((42, 43))} seed 复验中...")
    m, s, vals = verify_full(dataset, key, te_cfg, seeds=(42, 43))

    existing_path = os.path.join(U.TUN, f"best_{dataset}_{key}.json")
    base = None
    ex = None
    if os.path.exists(existing_path):
        ex = json.load(open(existing_path, encoding="utf-8"))
        base = ex.get("val_mape_mean", ex.get("val_mape"))
    replace = (base is None) or (m < base - 1e-4)

    te_note = dict(dataset=dataset, key=key, cfg=te_cfg, cfg_hash=T.cfg_hash(te_cfg),
                   verification_identity=expected_identity,
                   val_mape_fast=round(float(te_fast_val), 4),
                   val_mape_full_mean=round(m, 4), val_mape_full_std=round(s, 4),
                   val_full_seeds=vals, seeds=[42, 43], source="terime",
                   space_hash=space_hash, evals=int(te_evals),
                   created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(best_te, "w", encoding="utf-8") as f:
        json.dump(te_note, f, ensure_ascii=False, indent=1)

    lines = ["# 阶段五 定稿仲裁报告\n",
             f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "## 对比\n",
             "| 项 | 现有定稿（随机搜索） | TE-RIME |",
             "|---|---|---|"]
    if base is not None:
        lines.append(f"| full 验证 MAPE(%) | {base:.4f} | {m:.4f} |")
    lines += [f"| full 验证 std | — | {s:.4f} |",
              f"| TE-RIME fast 验证 MAPE(%) | — | {te_fast_val:.4f} |",
              "",
              f"**结论**：{'替换现有定稿（TE-RIME 更优）' if replace else '沿用现有定稿（TE-RIME 未更优）'}。\n",
              "注：fast 协议会系统性低估本文模型收敛能力（已实测 +1.19pp），故以 full 复验为准。\n"]
    with open(os.path.join(BASE, "reports", "stage5_decision.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if replace and ex is not None:
        ex["cfg"] = te_cfg
        ex["cfg_hash"] = T.cfg_hash(te_cfg)
        ex["val_mape_mean"] = round(m, 4)
        ex["val_mape_std"] = round(s, 4)
        ex["val_seeds"] = vals
        ex["source"] = "terime"
        with open(existing_path, "w", encoding="utf-8") as f:
            json.dump(ex, f, ensure_ascii=False, indent=1)
        print(f"  => 替换现有定稿：TE-RIME {m:.4f} < 现有 {base if base is not None else 'N/A'}")

        # 定稿被替换后，tuned_*.json 与 preds 仍是旧 cfg 的产物。若不标记，
        # 阶段六会静默复用到旧配置的结论 —— 这里显式置 stale 并要求重跑 finalize。
        tp = os.path.join(MET, f"tuned_{dataset}_{key}.json")
        if os.path.exists(tp):
            old = json.load(open(tp, encoding="utf-8"))
            old["stale"] = True
            old["stale_reason"] = (f"定稿已替换为 TE-RIME {T.cfg_hash(te_cfg)[:8]}"
                                   f"（full {m:.4f}%），需重跑 tuner.finalize 更新 test 与预测")
            old["superseded_by"] = T.cfg_hash(te_cfg)
            with open(tp, "w", encoding="utf-8") as f:
                json.dump(old, f, ensure_ascii=False, indent=1)
            print(f"     已标记 tuned_{dataset}_{key}.json 为 stale（需重跑 finalize）")
    else:
        print(f"  => 沿用现有定稿：TE-RIME {m:.4f} 未优于现有 {base if base is not None else 'N/A'}")
    return te_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jingmen")
    ap.add_argument("--model", default="bilstm_itransformer")
    ap.add_argument("--fusion", default="serial", choices=["serial", "gate", "parallel"])
    ap.add_argument("--algo", default="terime", choices=ALGOS + ["all"])
    ap.add_argument("--evals", type=int, default=100, help="唯一评估次数预算")
    ap.add_argument("--pop", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--protocol", default="fast", choices=list(U.PROTOCOLS))
    ap.add_argument("--warmstart", action="store_true", help="初始种群注入 fast 最优配置（仅长跑用）")
    ap.add_argument("--arbitrate", action="store_true", help="对 TE-RIME 长跑最优做 full 复验 + 定稿仲裁")
    ap.add_argument("--tag", default=None, help="结果文件名后缀（长跑用 --tag long，避免覆盖表5等预算结果）")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--isolate", action="store_true",
                    help="每个算法使用独立缓存目录（同等初始快照），消除缓存热度对耗时的污染")
    ap.add_argument("--restart-after", type=int, default=3,
                    help="TE-RIME 连续停滞多少完整代触发准对立重启（S6），0/负数关闭")
    ap.add_argument("--no-refine", action="store_true", help="关闭 TE-RIME 坐标精修（S5，消融用）")
    a = ap.parse_args()
    key = U.model_key(a.model, a.fusion)
    if a.report:
        report(a.dataset, key); return
    if a.arbitrate:
        # 优先读长跑结果（--tag long），否则读表5等预算结果
        tp = os.path.join(MET, f"stage5_{a.dataset}_{key}_terime_long.json")
        if not os.path.exists(tp):
            tp = os.path.join(MET, f"stage5_{a.dataset}_{key}_terime.json")
        if not os.path.exists(tp):
            print(f"  缺 TE-RIME 结果（stage5_..._terime[_long].json），请先跑 TE-RIME")
            return
        te = json.load(open(tp, encoding="utf-8"))
        arbitrate(a.dataset, key, te["best_cfg"], te["best_fit"], te["space_hash"], te["evals"])
        return
    algos = ALGOS if a.algo == "all" else [a.algo]
    restart = a.restart_after if a.restart_after and a.restart_after > 0 else None
    for al in algos:
        cdir = None
        if a.isolate:
            cdir = os.path.join(U.TUN, "cache_stage5", f"{a.dataset}_{key}_{al}_s{a.seed}")
            if not os.path.isdir(cdir):
                raise SystemExit(f"  隔离缓存目录不存在：{cdir}\n"
                                 f"  请先把统一缓存快照复制到该目录（模拟五算法同等起跑线），"
                                 f"例如：cp -r reports/tuning/cache/. {cdir}/")
        run_one(a.dataset, key, al, a.evals, a.pop, a.seed, U.PROTOCOLS[a.protocol],
                a.patience, warmstart=a.warmstart, tag=a.tag,
                restart_after=restart, cache_dir=cdir, refine=not a.no_refine)
    report(a.dataset, key)


if __name__ == "__main__":
    main()
