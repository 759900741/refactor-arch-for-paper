# -*- coding: utf-8 -*-
"""
tuner.py
========
通用超参调参器（任意 model_type × 任意 dataset，共用同一协议）。

设计三原则：
    1) 同空间语义 —— 所有模型共用一套「公共维度 + 架构专属维度」，架构专属键在该模型上才生效；
    2) 同评估协议 —— 同一 Protocol、同一等间隔子采样索引、同一 seed、同一 monitor（验证集 MAPE）；
    3) 同预算     —— 每个模型固定 n_trials，固定 fast → medium → full 三级晋升名额（48 → 12 → 3）。

隔离：调参阶段**物理上不加载 test 集**（load 时即 pop 掉），finalize 才允许评估一次 test。

缓存：一试验一文件 reports/tuning/cache/{fp}.json，fp 由「代码版本 + 数据指纹 + 规范配置 + 协议」
决定；缓存内容只含验证集指标，绝不含 test。

用法：
    py -3.8 tuner.py search   --dataset jingmen --model bilstm_itransformer --trials 48
    py -3.8 tuner.py search   --dataset jingmen --model bilstm_itransformer --fusion gate
    py -3.8 tuner.py refine   --dataset jingmen --model bilstm_itransformer --top 12
    py -3.8 tuner.py finalize --dataset jingmen --model bilstm_itransformer --seeds 42,43,44
    py -3.8 tuner.py report   --dataset jingmen
    py -3.8 tuner.py verify   --dataset jingmen
"""
from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
import hashlib
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T

BASE = T.BASE
TUN = os.path.join(BASE, "reports", "tuning")
CACHE = os.path.join(TUN, "cache")
os.makedirs(CACHE, exist_ok=True)

CODE_VERSION = "arch-v2"          # 架构/训练循环变更时必须手工 bump，使旧缓存自动失效
SCHEMA = "tune-v2"

LABEL = {"jingmen": "荆门市 2024", "lanzhou": "兰州市 2023"}
UNIT = {"jingmen": "MW", "lanzhou": "kW"}


# ---------------------------------------------------------------------------
# 评估协议
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    tag: str
    n_train: int | None       # None = 全量
    n_val: int | None
    epochs: int
    patience: int

    def as_dict(self):
        return asdict(self)


FAST = Protocol("fast", 1536, 256, 24, 6)
MEDIUM = Protocol("medium", 3072, 512, 40, 10)
FULL = Protocol("full", None, None, 60, 12)
PROTOCOLS = {"fast": FAST, "medium": MEDIUM, "full": FULL}


# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------
class Dim:
    """一维离散参数（全类别化，保证空间有限、可复现、缓存可命中）。"""

    def __init__(self, name, values, doc=""):
        self.name = name
        self.values = list(values)
        self.doc = doc

    def decode(self, u):
        i = int(min(max(u, 0.0), 0.999999) * len(self.values))
        return self.values[i]

    def index(self, v):
        try:
            return self.values.index(v)
        except ValueError:
            # 兼容浮点近似
            for i, x in enumerate(self.values):
                if isinstance(x, float) and isinstance(v, (int, float)) and abs(x - v) < 1e-9:
                    return i
            return 0

    def encode(self, v):
        i = self.index(v)
        return (i + 0.5) / len(self.values)

    def to_json(self):
        return dict(name=self.name, values=self.values, doc=self.doc)


LR = Dim("lr", [3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3], "学习率（论文列为待优化项，无数值）")
WD = Dim("weight_decay", [0.0, 1e-5, 1e-4, 1e-3], "L2 正则（论文列为待优化项）")
DO = Dim("dropout", [0.01, 0.05, 0.1, 0.2], "丢弃率（论文 0.01）")
HID = Dim("hidden", [32, 64, 96, 128], "BiLSTM 隐藏维（论文未给，128 为推定锚点）")
DM = Dim("d_model", [64, 96, 128], "iTransformer 模型维（论文 128）")
NH = Dim("n_heads", [4, 8], "注意力头数（论文 8）")
EL = Dim("e_layers", [1, 2], "编码器层数（论文 2）")
FF = Dim("ff_mult", [2, 4], "前馈扩张倍数（论文 256=2×128）")
LL = Dim("lstm_layers", [1, 2], "LSTM 层数（论文 2）")
MVH = Dim("mv_hidden", [64, 128], "并行分支隐藏维（本文新增，无论文依据）")
MVL = Dim("mv_layers", [1, 2], "并行分支层数（本文新增）")

COMMON = [LR, WD, DO, HID, LL]
SPACE = {
    "bilstm": COMMON,
    "bilstm_itransformer": [LR, WD, DO, HID, DM, NH, EL, FF, LL],
    "bilstm_itransformer_gate": [LR, WD, DO, HID, DM, NH, EL, FF, LL, MVH, MVL],
    "transformer": [LR, WD, DO, HID, DM, NH, EL, FF, LL],
    "informer": [LR, WD, DO, HID, DM, NH, EL, FF, LL],
}


def model_key(model, fusion="serial"):
    if model == "bilstm_itransformer" and fusion in ("gate", "parallel"):
        return f"bilstm_itransformer_{fusion}"
    return model


def space_for(key):
    return SPACE[key]


def join_space(dims, u):
    """u∈[0,1]^D -> 原始参数字典（ff_mult/d_model -> 实际 d_ff）。"""
    raw = {d.name: d.decode(u[i]) for i, d in enumerate(dims)}
    if "ff_mult" in raw:
        raw["d_ff"] = int(raw.pop("ff_mult") * raw.get("d_model", 128))
    return raw


def u_for(dims, cfg):
    u = np.zeros(len(dims))
    raw = dict(cfg)
    if "ff_mult" in raw and "d_model" in raw and raw["d_model"]:
        raw["ff_mult"] = int(round(raw["d_ff"] / raw["d_model"]))
    for i, d in enumerate(dims):
        u[i] = d.encode(raw.get(d.name, d.values[0]))
    return u


def repair(raw, key):
    """n_heads 必须整除 d_model；d_ff 必须是 d_model 的整数倍。"""
    if "d_model" in raw and "n_heads" in raw:
        dm, nh = raw["d_model"], raw["n_heads"]
        if dm % nh:
            raw["n_heads"] = 4 if dm % 4 == 0 else 2
    if "d_model" in raw and "d_ff" in raw:
        dm = raw["d_model"]
        raw["d_ff"] = int(max(1, round(raw["d_ff"] / dm)) * dm)
    if key.endswith("_gate") or key.endswith("_parallel"):
        raw["fusion"] = key.split("_", 2)[2] if key.count("_") >= 2 else "gate"
    else:
        raw["fusion"] = "serial"
    return raw


# ---------------------------------------------------------------------------
# 锚点
# ---------------------------------------------------------------------------
def anchors_for(key):
    """论文规格 / 当前基线 / 中间容量 等确定性锚点，覆盖随机采样覆盖不到的角落。"""
    base = dict(model_type="bilstm_itransformer" if "bilstm_itransformer" in key else key)
    cur = dict(hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1,
               lstm_layers=1, lr=1e-3, weight_decay=0.0)
    paper = dict(hidden=128, d_model=128, n_heads=8, e_layers=2, d_ff=256, dropout=0.01,
                 lstm_layers=2, lr=1e-3, weight_decay=0.0)
    mid = dict(hidden=64, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.05,
               lstm_layers=2, lr=1e-3, weight_decay=1e-5)
    lst = []
    if key == "bilstm":
        lst = [dict(hidden=32, dropout=0.1, lstm_layers=1, lr=1e-3, weight_decay=0.0),
               dict(hidden=128, dropout=0.01, lstm_layers=2, lr=1e-3, weight_decay=0.0),
               dict(hidden=64, dropout=0.05, lstm_layers=2, lr=7e-4, weight_decay=1e-5)]
    else:
        lst = [dict(cur), dict(paper),
               dict(paper, lr=5e-4), dict(paper, weight_decay=1e-4),
               dict(mid), dict(paper, dropout=0.05)]
        if key.endswith("_gate"):
            lst = [dict(x, mv_hidden=128, mv_layers=1) for x in lst]
            lst.append(dict(paper, mv_hidden=64, mv_layers=2))
        for x in lst:
            x["model_type"] = "bilstm_itransformer" if "bilstm_itransformer" in key else key
    return [{**base, **x} for x in lst]


# ---------------------------------------------------------------------------
# 数据（带缓存 + test 隔离）
# ---------------------------------------------------------------------------
_DCACHE = {}


def data_of(dataset, variant="selorig", add_lag=False, need_test=False):
    k = (dataset, variant, add_lag)
    if k not in _DCACHE:
        d = T.load_split(dataset, variant, add_lag=add_lag)
        _DCACHE[k] = d
    d = _DCACHE[k]
    if not need_test:
        out = {kk: vv for kk, vv in d.items() if kk != "test"}
        return out
    return d


def _sub_idx(n, k):
    if k is None or k >= n:
        return None
    return np.unique(np.linspace(0, n - 1, int(k)).round().astype(int))


def apply_protocol(d, p: Protocol):
    """按协议对 train/val 做等间隔子采样，返回新的浅层字典（不改原数据）。"""
    it = _sub_idx(len(d["train"]["X"]), p.n_train)
    iv = _sub_idx(len(d["val"]["X"]), p.n_val)
    out = dict(d)
    if it is not None:
        out["train"] = dict(X=d["train"]["X"][it], y=d["train"]["y"][it])
    if iv is not None:
        out["val"] = dict(X=d["val"]["X"][iv], y=d["val"]["y"][iv])
    return out


def data_fingerprint(d):
    y = d["train"]["y"]
    return dict(shape=[int(x) for x in d["train"]["X"].shape],
                y_sum=round(float(y.sum()), 4),
                n_val=int(len(d["val"]["X"])), n_feat=len(d["cols"]))


# ---------------------------------------------------------------------------
# 指纹与评估
# ---------------------------------------------------------------------------
def fingerprint(dataset, variant, add_lag, cfg, p: Protocol, backend):
    payload = dict(schema=SCHEMA, code=CODE_VERSION, backend=backend,
                   data=dict(dataset=dataset, variant=variant, add_lag=add_lag,
                             **_dcache_fp(dataset, variant, add_lag)),
                   model=T.canonical_cfg(cfg), protocol=p.as_dict())
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _dcache_fp(dataset, variant, add_lag):
    d = _DCACHE.get((dataset, variant, add_lag))
    if d is None:
        d = data_of(dataset, variant, add_lag)
        _DCACHE[(dataset, variant, add_lag)] = d
    return data_fingerprint(d)


def _effective(cfg, dataset, p: Protocol, seed):
    c = {**T.DEFAULT_CFG, **cfg}
    c["epochs"] = p.epochs
    c["patience"] = p.patience
    c["seed"] = seed
    c["batch"] = T.auto_batch(c, dataset)
    c["monitor"] = "val_mape"
    return c


def evaluate(dataset, cfg, p: Protocol = FAST, variant="selorig", add_lag=False,
             seed=42, force=False, log=None):
    """训练一次并返回验证集指标。绝不解包 test 集。"""
    cfg = {**cfg}
    eff = _effective(cfg, dataset, p, seed)
    fp = fingerprint(dataset, variant, add_lag, eff, p, str(T.resolve_device(eff)))
    path = os.path.join(CACHE, f"{fp}.json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        r["cache_hit"] = True
        return r

    d = apply_protocol(data_of(dataset, variant, add_lag, need_test=False), p)
    t0 = time.perf_counter()
    oom = False
    try:
        model, hist, best = T.fit_model(d, eff, log=log)
    except RuntimeError as e:  # noqa: BLE001  CUDA OOM -> batch 减半重试一次
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        eff["batch"] = max(16, int(eff["batch"]) // 2)
        oom = True
        model, hist, best = T.fit_model(d, eff, log=log)

    vs = T.val_scores(model, T._to_t(d["val"]["X"], T.resolve_device(eff)),
                      T._to_t(d["val"]["y"], T.resolve_device(eff)), d.get("y_range"))
    res = dict(fp=fp, dataset=dataset, variant=variant, add_lag=add_lag, protocol=p.tag,
               cfg=T.canonical_cfg(eff), fusion=eff.get("fusion"), model_type=eff.get("model_type"),
               val_mape=round(float(vs["mape"]), 5), val_rmse=round(float(vs["rmse"]), 6),
               val_rmse_orig=round(float(vs["rmse_orig"]), 4),
               best_epoch=int(hist["best_epoch"]), epochs_run=len(hist["train_loss"]),
               secs=round(time.perf_counter() - t0, 2), oom_retry=oom,
               params=int(sum(x.numel() for x in model.parameters())),
               backend=str(T.resolve_device(eff)), cache_hit=False,
               created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def latin_hypercube(n, ndim, rng):
    """确定性拉丁超立方：每维分层后随机置换。"""
    u = np.zeros((n, ndim))
    for j in range(ndim):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def sample_cfgs(key, n, seed=42, include_anchors=True):
    dims = space_for(key)
    rng = np.random.RandomState(seed)
    out = []
    seen = set()
    if include_anchors:
        for a in anchors_for(key):
            a = repair(dict(a), key)
            h = T.cfg_hash({**T.DEFAULT_CFG, **a})
            if h not in seen:
                seen.add(h); out.append(a)
    n_lhs = max(n - len(out), 0)
    if n_lhs:
        for u in latin_hypercube(n_lhs, len(dims), rng):
            raw = repair(join_space(dims, u), key)
            raw.setdefault("model_type", "bilstm_itransformer" if "bilstm_itransformer" in key else key)
            h = T.cfg_hash({**T.DEFAULT_CFG, **raw})
            if h in seen:
                continue
            seen.add(h); out.append(raw)
    return out[:n]


# ---------------------------------------------------------------------------
# 搜索 / 复核 / 定稿
# ---------------------------------------------------------------------------
def _trials_path(dataset, key):
    return os.path.join(TUN, f"{dataset}_{key}_trials.json")


def load_trials(dataset, key):
    p = _trials_path(dataset, key)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_trials(dataset, key, rows):
    with open(_trials_path(dataset, key), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def top_cfgs(dataset, key, n=5, protocol_tag="fast"):
    """从试验记录取 protocol_tag 下 val_mape 最低的 n 个 canonical cfg，
    并补回 model_type/fusion（规避 canonical_cfg 丢键的坑），供 warm-start 编码。"""
    rows = [r for r in load_trials(dataset, key) if r["protocol"] == protocol_tag]
    rows.sort(key=lambda r: r["val_mape"])
    out = []
    for r in rows[:n]:
        c = dict(r["cfg"])
        c["model_type"] = r.get("model_type") or c.get("model_type")
        if r.get("fusion"):
            c["fusion"] = r["fusion"]
        out.append(c)
    return out


def search(dataset, key, trials=48, protocol=FAST, seed=42, force=False,
           max_minutes=None, verbose=True):
    rows = load_trials(dataset, key)
    done = {r["fp"] for r in rows}
    cand = sample_cfgs(key, trials, seed=seed)
    t_start = time.perf_counter()
    print(f"\n[search] {dataset} | {key} | {protocol.tag} | 候选 {len(cand)} 组 | 空间维度 {len(space_for(key))}")
    for i, cfg in enumerate(cand, 1):
        if max_minutes and (time.perf_counter() - t_start) / 60 > max_minutes:
            print(f"  达到 --max-minutes={max_minutes}，提前停止（已完成 {i-1}/{len(cand)}）")
            break
        r = evaluate(dataset, cfg, protocol, seed=seed, force=force,
                     log=(lambda *a, **k: None) if not verbose else print)
        if r["fp"] in done and not force:
            print(f"  [{i:02d}/{len(cand)}] 缓存命中 val_mape={r['val_mape']:.4f}")
            continue
        rows.append(r); done.add(r["fp"])
        save_trials(dataset, key, rows)
        print(f"  [{i:02d}/{len(cand)}] val_mape={r['val_mape']:.4f}% val_rmse={r['val_rmse']:.5f} "
              f"{r['secs']:.0f}s ep{r['best_epoch']} p={r['params']//1000}k"
              + ("  (OOM重试)" if r["oom_retry"] else ""))
    rows = sorted(load_trials(dataset, key), key=lambda r: r["val_mape"])
    save_trials(dataset, key, rows)
    if rows:
        print(f"  当前最优 val_mape={rows[0]['val_mape']:.4f}%  cfg={_brief(rows[0]['cfg'])}")
    return rows


def refine(dataset, key, top=12, sigma=0.12, seed=43, max_minutes=None,
           protocol=None, n_perturb=2):
    """fast 前 top 名（+ 可选邻域扰动）用更高一级协议复核排序。"""
    rows = [r for r in load_trials(dataset, key) if r["protocol"] == "fast"]
    rows = sorted(rows, key=lambda r: r["val_mape"])[:top]
    if not rows:
        print("  无 fast 结果可复核"); return []
    proto = protocol or MEDIUM
    dims = space_for(key)
    rng = np.random.RandomState(seed)
    def _from_record(r):
        c = dict(r["cfg"])
        c["model_type"] = r.get("model_type") or c.get("model_type")
        if r.get("fusion"):
            c["fusion"] = r["fusion"]
        return c

    cand = [_from_record(r) for r in rows]
    for r in rows[:n_perturb]:                        # 局部扰动，每维 ±1 档
        u = u_for(dims, _from_record(r))
        for _ in range(2):
            v = u.copy()
            j = rng.randint(len(dims))
            d = dims[j]
            if d.name == "ff_mult":
                cur = int(round(r["cfg"]["d_ff"] / max(r["cfg"]["d_model"], 1)))
            else:
                cur = r["cfg"].get(d.name, d.values[0])
            i = int(np.clip(d.index(cur) + rng.choice([-1, 1]), 0, len(d.values) - 1))
            v[j] = (i + 0.5) / len(d.values)
            raw = repair(join_space(dims, v), key)
            raw.setdefault("model_type", "bilstm_itransformer" if "bilstm_itransformer" in key else key)
            cand.append(raw)
    print(f"\n[refine] {dataset} | {key} | {proto.tag} | 候选 {len(cand)} 组")
    t0 = time.perf_counter()
    out = []
    for i, cfg in enumerate(cand, 1):
        if max_minutes and (time.perf_counter() - t0) / 60 > max_minutes:
            break
        r = evaluate(dataset, cfg, proto, seed=42, log=lambda *a, **k: None)
        out.append(r)
        print(f"  [{i:02d}/{len(cand)}] val_mape={r['val_mape']:.4f}% ({r['secs']:.0f}s)"
              + (" [cache]" if r["cache_hit"] else ""))
    merged = {r["fp"]: r for r in load_trials(dataset, key)}
    for r in out:
        merged[r["fp"]] = r
    save_trials(dataset, key, sorted(merged.values(), key=lambda r: r["val_mape"]))
    return out


def finalize(dataset, key, top=3, seeds=(42, 43, 44), max_minutes=None):
    """top-N（按 medium/最优）在 full 协议 × 多 seed 下复训，定稿并**唯一一次**评估 test。"""
    rows = load_trials(dataset, key)
    pool = [r for r in rows if r["protocol"] == "medium"] or rows
    pool = sorted(pool, key=lambda r: r["val_mape"])[:top]
    if not pool:
        print("  无候选可定稿"); return None
    print(f"\n[finalize] {dataset} | {key} | full | top{len(pool)} × seeds{list(seeds)}")
    scored = []
    keep = {}                       # cfg_hash -> (model, hist, val_scores)，供 test 复用，避免重复训练
    for r in pool:
        cfg_r = dict(r["cfg"])
        cfg_r["model_type"] = r.get("model_type") or cfg_r.get("model_type")
        if r.get("fusion"):
            cfg_r["fusion"] = r["fusion"]
        ms, rms = [], []
        for i_sd, sd in enumerate(seeds):
            if i_sd == 0:
                # 首个 seed 直接训练并**保留模型**，后续 test 预测复用它（省一次全量训练）
                eff0 = _effective(cfg_r, dataset, FULL, sd)
                d_tr = apply_protocol(data_of(dataset, "selorig", False, need_test=False), FULL)
                model0, hist0, best0 = T.fit_model(d_tr, eff0, log=lambda *a, **k: None)
                vs = T.val_scores(model0, T._to_t(d_tr["val"]["X"], T.resolve_device(eff0)),
                                  T._to_t(d_tr["val"]["y"], T.resolve_device(eff0)), d_tr.get("y_range"))
                rr = dict(val_mape=round(float(vs["mape"]), 5), val_rmse=round(float(vs["rmse"]), 6),
                          secs=round(float(np.sum(hist0["epoch_sec"])), 1), cache_hit=False,
                          params=int(sum(x.numel() for x in model0.parameters())))
                keep[T.cfg_hash(eff0)] = (model0, hist0, eff0)
            else:
                rr = evaluate(dataset, cfg_r, FULL, seed=sd, log=lambda *a, **k: None)
            ms.append(rr["val_mape"]); rms.append(rr["val_rmse"])
            print(f"  seed={sd} val_mape={rr['val_mape']:.4f}% val_rmse={rr['val_rmse']:.5f} "
                  f"({rr['secs']:.0f}s, p={rr['params']//1000}k)" + (" [cache]" if rr["cache_hit"] else ""))
        scored.append(dict(cfg=cfg_r, mean=float(np.mean(ms)), std=float(np.std(ms)),
                           rmse_mean=float(np.mean(rms)), seeds=list(seeds), vals=ms))
    scored.sort(key=lambda x: x["mean"])
    best = scored[0]

    # ---- 唯一一次 test 评估（复用 seeds[0] 已训练的模型，不重复训练）----
    eff = _effective(best["cfg"], dataset, FULL, seeds[0])
    h_eff = T.cfg_hash(eff)
    d = data_of(dataset, "selorig", False, need_test=True)
    if h_eff in keep:
        model, hist, eff = keep[h_eff]
    else:
        model, hist, _ = T.fit_model(d, eff, log=lambda *a, **k: None)
    lo, hi = T.load_y_range(dataset)
    ps_test = T.predict(model, d["test"]["X"])
    ps_val = T.predict(model, d["val"]["X"])
    pt = T.inverse(ps_test, lo, hi)
    yt = T.inverse(d["test"]["y"], lo, hi)
    pv = T.inverse(ps_val, lo, hi)
    yv = T.inverse(d["val"]["y"], lo, hi)
    et = [x.strftime("%Y-%m-%d %H:%M:%S") for x in d["time"]["test"]]

    from stage6_experiments import compute_split_metrics
    pred_dir = os.path.join(BASE, "data", "processed", "preds")
    np.savez_compressed(os.path.join(pred_dir, f"tuned_{dataset}_{key}_test.npz"),
                        y_true=d["test"]["y"], y_pred=ps_test,
                        y_true_orig=yt, y_pred_orig=pt, end_time=np.array(et))
    np.savez_compressed(os.path.join(pred_dir, f"tuned_{dataset}_{key}_val.npz"),
                        y_true=d["val"]["y"], y_pred=ps_val,
                        y_true_orig=yv, y_pred_orig=pv)

    res = dict(dataset=dataset, label=LABEL[dataset], tag=_tag_of(key), seeded=False,
               model_type=eff["model_type"], variant="selorig",
               residual=eff["residual"], head_mode=eff["head_mode"],
               use_bilstm=eff["use_bilstm"], fusion=eff.get("fusion"), add_lag=False,
               F=len(d["cols"]), L=d["L"], H=d["H"], cfg=eff,
               cfg_hash=T.cfg_hash(eff), params=int(sum(x.numel() for x in model.parameters())),
               best_epoch=int(hist["best_epoch"]),
               train_min=round(float(np.sum(hist["epoch_sec"])) / 60.0, 2),
               val_rmse=float(hist["best_score"]),
               val_mape_mean=best["mean"], val_mape_std=best["std"], val_seeds=best["vals"],
               test=compute_split_metrics(yt, pt, et),
               val_overall=T.metrics(yv, pv),
               test_eval_count=1, protocol="full")
    note = dict(dataset=dataset, key=key, cfg=eff, cfg_hash=T.cfg_hash(eff),
                val_mape_mean=best["mean"], val_mape_std=best["std"], val_seeds=best["vals"],
                seeds=list(seeds), best_epoch=res["best_epoch"], params=res["params"],
                test_mape=res["test"]["overall"]["mape"], test_rmse=res["test"]["overall"]["rmse"],
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(TUN, f"best_{dataset}_{key}.json"), "w", encoding="utf-8") as f:
        json.dump(note, f, ensure_ascii=False, indent=1)
    with open(os.path.join(BASE, "reports", "metrics", f"tuned_{dataset}_{key}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"  => best val_mape={best['mean']:.4f}±{best['std']:.4f}% | "
          f"test RMSE={res['test']['overall']['rmse']:.2f}{UNIT[dataset]} "
          f"MAPE={res['test']['overall']['mape']:.2f}% | cfg={_brief(eff)}")
    return res


def _tag_of(key):
    return {"bilstm": "BiLSTM", "transformer": "BiLSTM-Transformer",
            "informer": "BiLSTM-Informer", "bilstm_itransformer": "本文_BiLSTM-iTransformer",
            "bilstm_itransformer_gate": "本文_并行门控"}.get(key, key)


def _brief(cfg):
    ks = ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "dropout", "lstm_layers",
          "lr", "weight_decay", "fusion", "mv_hidden", "mv_layers")
    return ", ".join(f"{k}={cfg.get(k)}" for k in ks if cfg.get(k) not in (None,))


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report(dataset, keys=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    keys = keys or sorted({os.path.basename(p).split(f"{dataset}_", 1)[1].rsplit("_trials", 1)[0]
                           for p in os.listdir(TUN) if p.startswith(f"{dataset}_") and p.endswith("_trials.json")})
    bests = []
    lines = [f"# {LABEL[dataset]} 超参搜索汇总\n", f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "| 模型 | 最优 val MAPE | ± std | 参数量 | test RMSE | test MAPE% | 关键超参 |",
             "|---|---|---|---|---|---|---|"]
    for k in keys:
        p = os.path.join(TUN, f"best_{dataset}_{k}.json")
        if not os.path.exists(p):
            continue
        b = json.load(open(p, encoding="utf-8"))
        bests.append(b)
        lines.append(f"| {_tag_of(k)} | {b['val_mape_mean']:.4f} | {b['val_mape_std']:.4f} | "
                     f"{b['params']//1000}k | {b['test_rmse']:.2f} | {b['test_mape']:.2f} | "
                     f"{_brief(b['cfg'])} |")
        trows = load_trials(dataset, k)
        if trows:
            fr = [r for r in trows if r["protocol"] == "fast"] or trows
            xs = np.arange(1, len(fr) + 1)
            bs = np.minimum.accumulate([r["val_mape"] for r in fr])
            plt.plot(xs, bs, marker="o", ms=2, lw=1.2, label=_tag_of(k))
    lines.append("")
    if bests:
        winner = min(bests, key=lambda b: b["val_mape_mean"])
        lines.append(f"**验证集最优**：{_tag_of(winner['key'])}（val MAPE {winner['val_mape_mean']:.4f}%）\n")
    with open(os.path.join(TUN, f"summary_{dataset}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if len(bests) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        names = [_tag_of(b["key"]) for b in bests]
        vals = [b["val_mape_mean"] for b in bests]
        errs = [b["val_mape_std"] for b in bests]
        order = np.argsort(vals)
        axes[0].bar([names[i] for i in order], [vals[i] for i in order],
                    yerr=[errs[i] for i in order], capsize=3,
                    color=["#c53030" if i == order[0] else "#4299e1" for i in order])
        for i, idx in enumerate(order):
            axes[0].text(i, vals[idx], f"{vals[idx]:.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("验证集 MAPE（%）"); axes[0].set_title("各模型调参后精度（3 seeds）")
        plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right", fontsize=8)
        axes[1].set_xlabel("试验序号"); axes[1].set_ylabel("best-so-far 验证 MAPE（%）")
        axes[1].set_title("主流搜索收敛轨迹（fast）"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(os.path.join(TUN, f"summary_{dataset}.png"), dpi=150)
        plt.close(fig)
    print(f"  报告：reports/tuning/summary_{dataset}.md")
    return bests


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def calibrate(dataset, keys=None, seed=42):
    """协议标定：同一配置在 fast/medium/full 三级协议下的验证 MAPE，
    用于量化「快速协议是否低估某模型的收敛能力」（排序偏差诊断）。"""
    keys = keys or ["bilstm_itransformer", "bilstm"]
    out = {}
    for key in keys:
        cfg = anchors_for(key)[0]
        row = dict(key=key, cfg=T.canonical_cfg({**T.DEFAULT_CFG, **cfg}), levels={})
        for p in (FAST, MEDIUM, FULL):
            r = evaluate(dataset, cfg, p, seed=seed, log=lambda *a, **k: None)
            row["levels"][p.tag] = dict(val_mape=r["val_mape"], secs=r["secs"],
                                        epochs_run=r["epochs_run"], best_epoch=r["best_epoch"])
            print(f"  {key:26s} {p.tag:7s} val_mape={r['val_mape']:.4f}% "
                  f"({r['secs']:.0f}s, {r['epochs_run']}轮)")
        f, m, fl = (row["levels"][k]["val_mape"] for k in ("fast", "medium", "full"))
        row["bias_fast_vs_full"] = round(f - fl, 4)
        row["bias_medium_vs_full"] = round(m - fl, 4)
        out[key] = row
    p = os.path.join(TUN, f"calibration_{dataset}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  协议标定 -> {os.path.relpath(p, BASE)}")
    return out


def verify(dataset="jingmen", key="bilstm_itransformer"):
    """断言：①同 cfg 第二次命中缓存且指标一致；②调参不加载 test。"""
    cfg = {**T.DEFAULT_CFG, "hidden": 32, "d_model": 32, "n_heads": 2, "e_layers": 1,
           "d_ff": 64, "lstm_layers": 1}
    r1 = evaluate(dataset, cfg, FAST, log=lambda *a, **k: None)
    r2 = evaluate(dataset, cfg, FAST, log=lambda *a, **k: None)
    ok_cache = (r2["cache_hit"] and abs(r1["val_mape"] - r2["val_mape"]) < 1e-9)
    d = data_of(dataset, "selorig", False, need_test=False)
    ok_no_test = "test" not in d
    print(f"  缓存一致性: {'PASS' if ok_cache else 'FAIL'} ({r1['val_mape']:.4f} vs {r2['val_mape']:.4f})")
    print(f"  test 隔离 : {'PASS' if ok_no_test else 'FAIL'}")
    return ok_cache and ok_no_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "refine", "finalize", "report", "verify", "calibrate"])
    ap.add_argument("--dataset", default="jingmen", choices=list(LABEL))
    ap.add_argument("--model", default="bilstm_itransformer",
                    choices=["bilstm_itransformer", "bilstm", "transformer", "informer"])
    ap.add_argument("--fusion", default="serial", choices=["serial", "gate", "parallel"])
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--protocol", default="fast", choices=list(PROTOCOLS))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--n-perturb", type=int, default=2, help="复核时做局部扰动的 top 数（0=不扰动）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    key = model_key(a.model, a.fusion)
    if a.cmd == "search":
        search(a.dataset, key, a.trials, PROTOCOLS[a.protocol], a.seed, a.force,
               a.max_minutes, verbose=not a.quiet)
    elif a.cmd == "refine":
        refine(a.dataset, key, a.top, max_minutes=a.max_minutes,
               protocol=PROTOCOLS[a.protocol], n_perturb=a.n_perturb)
    elif a.cmd == "finalize":
        finalize(a.dataset, key, top=min(3, a.top), seeds=[int(x) for x in a.seeds.split(",")])
    elif a.cmd == "report":
        report(a.dataset)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.dataset, key) else 1)
    elif a.cmd == "calibrate":
        calibrate(a.dataset, ["bilstm_itransformer", "bilstm"])


if __name__ == "__main__":
    main()
