# -*- coding: utf-8 -*-
"""
stage1_preprocess.py
====================
阶段一主流程：数据理解与预处理。

对两份数据集（荆门 2024 / 兰州 2023）依次执行：
    加载解析 -> 时间规整 -> 缺失分层 -> 异常标记 -> 日历特征
    -> 滞后特征 -> 时序划分(8:1:1) -> MinMax归一化(仅训集fit) -> 滑窗 -> 报告/图

输出：
    data/processed/{name}_clean.parquet        清洗后（未归一化）全量特征
    data/processed/scaler_{name}_{X,y}.pkl     归一化器
    data/processed/windows/{name}_{split}.npz  滑窗样本
    reports/data_quality.md
    reports/figs/{name}_*.png

用法：python src/stage1_preprocess.py
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import utils_data as U

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(BASE, "data", "processed")
WIN_DIR = os.path.join(PROC_DIR, "windows")
FIG_DIR = os.path.join(BASE, "reports", "figs")
os.makedirs(WIN_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

USE_LAGS = True          # 决策1：跨日/跨周滞后特征开关
NORM = "minmax"          # 决策5：MinMax[0,1]

DATASETS = {
    "jingmen": dict(
        name="jingmen", label="荆门市2024", file_pattern="荆门",
        time_col="times", target="负荷", freq="15min",
        steps_per_day=96, steps_per_week=672,
        L=96, H=4, stride=4,          # 多步：stride=H 防目标重叠
        anomaly="global", has_day_flags=True,
    ),
    "lanzhou": dict(
        name="lanzhou", label="兰州市2023", file_pattern="兰州",
        time_col="date", target="electricity(KW)", freq="1h",
        steps_per_day=24, steps_per_week=168,
        L=24, H=1, stride=1,          # 单步：stride=1
        anomaly="hampel", has_day_flags=False, win=25, k=3.0,
    ),
}

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 主处理函数
# ---------------------------------------------------------------------------
def process_dataset(cfg: dict) -> dict:
    print(f"\n{'='*66}\n处理数据集：{cfg['label']} ({cfg['name']})")
    # 1.1 加载解析
    path = U.find_csv(BASE, cfg["file_pattern"])
    raw = U.load_csv(path)
    print(f"  原始: {raw.shape} <- {os.path.basename(path)}")

    # 1.2 时间规整
    df, rep_time = U.parse_and_regularize(raw, cfg["time_col"], cfg["freq"])
    print(f"  时间规整: {rep_time}")

    # 1.3 缺失分层（当前应不触发）
    value_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    df, miss_log, drop_mask = U.tiered_fill(df[value_cols])
    print(f"  缺失分层: 处理段数={len(miss_log)}, 删除行={int(drop_mask.sum())}")

    # 1.4 异常标记
    y = df[cfg["target"]].astype(float)
    if cfg["anomaly"] == "global":
        flag, ainfo = U.detect_anomaly_global(y)
    elif cfg["anomaly"] == "hampel":
        flag, ainfo = U.detect_anomaly_hampel(y, win=cfg.get("win", 25), k=cfg.get("k", 3.0))
    else:
        flag, ainfo = U.detect_anomaly_seasonal(df, y)
    df["is_anomaly"] = flag.astype(int)
    print(f"  异常标记: {ainfo}")

    # 1.5 日历特征
    df = U.add_calendar_features(df)
    df = U.add_day_type(df, has_flags=cfg["has_day_flags"])

    # 1.6 滞后特征（跨日/跨周）
    df = U.add_lag_features(df, cfg["target"], cfg["steps_per_day"],
                            cfg["steps_per_week"], enabled=USE_LAGS)

    # 特征列：数值列，排除 is_anomaly；删除常量列与原始 weekday(与 dayofweek 重复)
    if "weekday" in df.columns:
        df = df.drop(columns=["weekday"])
    feat_cols = [c for c in df.columns
                 if c != "is_anomaly" and np.issubdtype(df[c].dtype, np.number)]
    const_cols = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]
    feat_cols = [c for c in feat_cols if c not in const_cols]
    if const_cols:
        df = df.drop(columns=const_cols)
    print(f"  特征列 {len(feat_cols)} 个；剔除常量列 {const_cols}")

    # 删除滞后导致的起始 NaN 行（仅序列最前面几步）
    n_before = len(df)
    df_model = df.dropna(subset=feat_cols).copy()
    n_dropped = n_before - len(df_model)
    print(f"  丢弃起始 NaN 行: {n_dropped} (剩余 {len(df_model)})")

    # 1.7 时序划分 8:1:1（先划分）
    train, val, test = U.split_chrono(df_model, (0.8, 0.1, 0.1))
    splits = {"train": train, "val": val, "test": test}

    # 1.8 归一化（仅在训练集 fit）
    sx, sy = U.fit_scalers(train, feat_cols, cfg["target"])
    joblib.dump(sx, os.path.join(PROC_DIR, f"scaler_{cfg['name']}_X.pkl"))
    joblib.dump(sy, os.path.join(PROC_DIR, f"scaler_{cfg['name']}_y.pkl"))

    # 1.9 构造滑窗并按 split 保存
    win_report = {}
    for split, part in splits.items():
        Xs = U.transform_X(part, feat_cols, sx)
        ys = U.transform_y(part, cfg["target"], sy)
        Xw, yw, ends = U.make_windows(Xs, ys, cfg["L"], cfg["H"], cfg["stride"])
        end_times = part.index[ends].astype(str).to_numpy()
        np.savez_compressed(
            os.path.join(WIN_DIR, f"{cfg['name']}_{split}.npz"),
            X=Xw, y=yw, end_time=end_times,
        )
        win_report[split] = dict(rows=len(part), X_shape=list(Xw.shape),
                                 y_shape=list(yw.shape))
        print(f"  [{split}] rows={len(part)} windows X={Xw.shape} y={yw.shape}")

    # 保存清洗后全量（未归一化）供复核
    df_model.to_parquet(os.path.join(PROC_DIR, f"{cfg['name']}_clean.parquet"))

    # 统计 & 保存
    stats = dict(
        dataset=cfg["name"], label=cfg["label"], file=os.path.basename(path),
        time_report=rep_time, anomaly=ainfo,
        n_features=len(feat_cols), feature_cols=feat_cols,
        const_dropped=const_cols, n_dropped_start=n_dropped,
        split_boundaries={k: [str(v.index.min()), str(v.index.max())] for k, v in splits.items()},
        windows=win_report, L=cfg["L"], H=cfg["H"], stride=cfg["stride"],
        use_lags=USE_LAGS, norm=NORM,
    )
    U.save_json(stats, os.path.join(PROC_DIR, f"{cfg['name']}_stats.json"))

    # 图表
    make_figures(cfg, df_model, anchor=df if "is_anomaly" in df else df_model)
    return stats


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def make_figures(cfg, df_model, anchor):
    name, label, tgt = cfg["name"], cfg["label"], cfg["target"]
    figdir = FIG_DIR

    # 图1：目标时序 + 异常点
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(df_model.index, df_model[tgt], lw=0.6, color="#2b6cb0", label="负荷")
    an = df_model[df_model["is_anomaly"] == 1]
    ax.scatter(an.index, an[tgt], s=6, color="#c53030", label="异常标记", zorder=3)
    ax.set_title(f"{label} 目标时序与异常标记"); ax.set_ylabel(tgt)
    ax.legend(loc="upper right", fontsize=8); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{name}_load_ts.png"), dpi=130); plt.close(fig)

    # 图2：月均
    mmean = df_model[tgt].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar([str(x.month) for x in mmean.index], mmean.values, color="#4299e1")
    ax.set_title(f"{label} 月均负荷"); ax.set_xlabel("月份"); ax.set_ylabel(tgt)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{name}_monthly.png"), dpi=130); plt.close(fig)

    # 图3：典型日曲线（按 day_type）
    if "day_type" in df_model.columns:
        d = df_model.copy(); d["h"] = d.index.hour
        fig, ax = plt.subplots(figsize=(8, 3))
        label_map = {0: "周末", 1: "工作日", 2: "节假日"}
        for dt, g in d.groupby("day_type"):
            prof = g.groupby("h")[tgt].mean()
            ax.plot(prof.index, prof.values, marker="o", ms=2, label=label_map.get(dt, str(dt)))
        ax.set_title(f"{label} 典型日负荷曲线"); ax.set_xlabel("小时"); ax.set_ylabel(tgt)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{name}_daily_profile.png"), dpi=130); plt.close(fig)

    # 图4：特征-目标相关性（条形）
    feats = [c for c in df_model.columns if c != "is_anomaly" and np.issubdtype(df_model[c].dtype, np.number)]
    corr = df_model[feats].corr()[tgt].drop(tgt).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(corr))))
    colors = ["#c53030" if v < 0 else "#2f855a" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_title(f"{label} 各特征与目标相关系数"); ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{name}_corr.png"), dpi=130); plt.close(fig)

    # 图5：目标分布
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(df_model[tgt].values, bins=50, color="#4299e1")
    axes[0].set_title("直方图")
    axes[1].boxplot(df_model[tgt].values, orientation="vertical")
    axes[1].set_title("箱线图")
    fig.suptitle(f"{label} 目标分布")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{name}_target_dist.png"), dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def write_report(all_stats):
    lines = ["# 数据质量与预处理报告（阶段一自动生成）\n",
             f"> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} ｜ 滞后特征={USE_LAGS} ｜ 归一化={NORM}\n"]
    for s in all_stats:
        w = s["windows"]
        lines.append(f"\n## {s['label']}（{s['dataset']}）\n")
        lines.append(f"- 文件：`{s['file']}`")
        lines.append(f"- 时间规整：{s['time_report']}")
        lines.append(f"- 异常检测：{s['anomaly']}")
        lines.append(f"- 特征数：{s['n_features']}；剔除常量列：{s['const_dropped']}；丢弃起始NaN行：{s['n_dropped_start']}")
        lines.append(f"- 划分边界：")
        for k, v in s["split_boundaries"].items():
            lines.append(f"    - {k}: {v[0]} ~ {v[1]}（{w[k]['rows']} 行）")
        lines.append(f"- 滑窗（L={s['L']}, H={s['H']}, stride={s['stride']}）：")
        for k in ("train", "val", "test"):
            lines.append(f"    - {k}: X={w[k]['X_shape']}, y={w[k]['y_shape']}")
        lines.append(f"- 特征列：{', '.join(s['feature_cols'])}")
    lines.append("\n## 说明与注意\n")
    lines.append("- 荆门采用全局 3σ + 差分阈值标记异常；兰州采用局部 Hampel 滤波（win=25, k=3）标记毛刺，二者均避免误删夏季真实峰/换挡段。")
    lines.append("- 异常仅做标记（is_anomaly 列），未删除；是否修正留待阶段二 RPCA。")
    lines.append("- 兰州无节假日日历，day_type 以「周一~周五=工作日、周末=周末」近似（泛化集，可接受）。")
    lines.append("- 兰州 cold/heat 与温度强共线，已保留并待阶段二 MIV 裁决。")
    lines.append("- 归一化仅在训练集 fit，验证/测试仅 transform（防泄漏）。")
    lines.append("- 多步（荆门 H=4）采用 stride=H 避免相邻样本目标重叠。")
    with open(os.path.join(BASE, "reports", "data_quality.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n报告已写入 reports/data_quality.md")


def main():
    all_stats = []
    for cfg in DATASETS.values():
        all_stats.append(process_dataset(cfg))
    write_report(all_stats)
    print("\n阶段一完成。")


if __name__ == "__main__":
    main()
