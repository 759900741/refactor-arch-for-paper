# -*- coding: utf-8 -*-
"""
utils_data.py
=============
阶段一（数据预处理）通用工具函数库。

设计原则：
- 所有"防泄漏"相关操作（划分、归一化、滑窗）都提供独立函数，便于审计。
- 缺失值分层处理、异常检测按论文 4.1 与数据特性分别实现。
- 指标函数（RMSE/MAE/MAPE/R2/RPD/VAE/IA）在后续阶段复用。

作者：xzj 研究方案配套实现
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------------------------
# 1. 数据加载与时间规整
# ----------------------------------------------------------------------------
def find_csv(base_dir: str, pattern: str) -> str:
    """按模式查找数据文件，自动忽略 '副本' 文件。"""
    cands = [
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".csv") and pattern in f and "副本" not in f
    ]
    if not cands:
        raise FileNotFoundError(f"在 {base_dir} 未找到含 '{pattern}' 的 CSV")
    # 若有多个，取文件名最短的（通常为正本）
    cands.sort(key=len)
    return os.path.join(base_dir, cands[0])


def load_csv(path: str) -> pd.DataFrame:
    """读入 CSV，统一 utf-8-sig，避免中文 BOM 乱码。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_and_regularize(df: pd.DataFrame, time_col: str, freq: str):
    """
    解析时间列 -> 升序去重 -> 用 date_range 重建等间隔轴 -> reindex。

    返回:
        df_out : 以 DatetimeIndex 为索引、等间隔连续的 DataFrame
        report : dict，含解析失败数、重复数、补齐的缺失步数等
    """
    n0 = len(df)
    t = pd.to_datetime(df[time_col], errors="coerce")
    parse_fail = int(t.isna().sum())
    df = df.loc[t.notna()].copy()
    df[time_col] = t[t.notna()]

    df = df.sort_values(time_col)
    n_dup = int(df[time_col].duplicated().sum())
    df = df.drop_duplicates(subset=[time_col], keep="first")

    df = df.set_index(time_col)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    n_extra = len(full_idx) - len(df)
    df = df.reindex(full_idx)
    df.index.name = time_col

    # 记录重采样后"整行为空"的步数（即原数据缺失的时间点）
    missing_steps = int(df.isna().all(axis=1).sum())
    report = dict(
        n_raw=n0, parse_fail=parse_fail, n_dup=n_dup,
        n_rows_after=len(df), freq=freq,
        added_steps=max(n_extra, 0), missing_steps=missing_steps,
    )
    return df, report


# ----------------------------------------------------------------------------
# 2. 缺失值分层处理（论文 4.1：<6 插值 / 6~24 移动平均 / >=25 删除）
# ----------------------------------------------------------------------------
def _nan_runs(mask: np.ndarray):
    """返回布尔数组中连续 True 的区间列表 [(start, end_inclusive), ...]。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def tiered_fill(df: pd.DataFrame, short: int = 6, mid: int = 24):
    """
    对每个数值列按连续缺失段长度分档处理：
        L < short      -> 线性插值
        short <= L <= mid -> 邻域移动平均填充
        L > mid        -> 整段行删除（返回删除掩码）

    返回 (df_filled, log, drop_mask)
    """
    df = df.copy()
    drop_mask = np.zeros(len(df), dtype=bool)
    log = []

    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        s = df[col].astype(float)
        mask = s.isna().to_numpy()
        if not mask.any():
            continue
        for a, b in _nan_runs(mask):
            L = b - a + 1
            if L < short:
                # 线性插值；首尾用可用单侧值
                left = s.iloc[a - 1] if a > 0 else np.nan
                right = s.iloc[b + 1] if b + 1 < len(s) else np.nan
                if np.isnan(left) and np.isnan(right):
                    continue
                if np.isnan(left):
                    s.iloc[a:b + 1] = right
                elif np.isnan(right):
                    s.iloc[a:b + 1] = left
                else:
                    s.iloc[a:b + 1] = np.linspace(left, right, L + 2)[1:-1]
                log.append(dict(col=col, start=int(a), end=int(b), len=L, action="interp"))
            elif L <= mid:
                left = s.iloc[max(0, a - 1):a].mean()
                right = s.iloc[b + 1:b + 2].mean()
                vals = [v for v in (left, right) if not np.isnan(v)]
                if not vals:
                    continue
                s.iloc[a:b + 1] = float(np.mean(vals))
                log.append(dict(col=col, start=int(a), end=int(b), len=L, action="movavg"))
            else:
                drop_mask[a:b + 1] = True
                log.append(dict(col=col, start=int(a), end=int(b), len=L, action="drop"))
        df[col] = s

    df = df.loc[~drop_mask]
    return df, log, drop_mask


# ----------------------------------------------------------------------------
# 3. 异常检测与标记（只标记，不删除）
# ----------------------------------------------------------------------------
def detect_anomaly_global(y: pd.Series, k_sigma: float = 3.0, diff_sigma: float = 4.0):
    """
    适用于季节性较温和的数据（荆门）。
    规则：全局 |z|>k_sigma  或  |一阶差分| > mean+diff_sigma*std。
    """
    y = y.astype(float)
    m, s = np.nanmean(y), np.nanstd(y)
    z = (y - m) / (s + 1e-12)
    flag_z = (np.abs(z) > k_sigma).to_numpy()

    d = y.diff().abs()
    thr = np.nanmean(d) + diff_sigma * np.nanstd(d)
    flag_d = (d > thr).fillna(False).to_numpy()

    flag = flag_z | flag_d
    info = dict(method="global", n_sigma=int(flag_z.sum()),
                n_diff=int(flag_d.sum()), diff_threshold=float(thr))
    return flag, info


def detect_anomaly_seasonal(df: pd.DataFrame, y: pd.Series, k_mad: float = 3.5):
    """
    （保留的备选）以 (month, hour) 分组中位数作为季节基线 + 全局 MAD 判定。
    注意：在存在"爬坡/换挡"过渡月的数据上会过度标记，故默认不使用。
    """
    y = y.astype(float)
    tmp = pd.DataFrame({"y": y.to_numpy()}, index=df.index)
    tmp["month"] = df.index.month
    tmp["hour"] = df.index.hour
    base = tmp.groupby(["month", "hour"])["y"].transform("median")
    resid = tmp["y"] - base
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    rz = (resid - med) / (1.4826 * mad + 1e-12)
    flag = (np.abs(rz) > k_mad).to_numpy()
    info = dict(method="seasonal_month_hour", n_flagged=int(flag.sum()),
                mad=float(mad), k_mad=k_mad)
    return flag, info


def detect_anomaly_hampel(y: pd.Series, win: int = 25, k: float = 3.0, min_frac: float = 0.5):
    """
    局部 Hampel 滤波（强鲁棒，适合强季节/换挡数据，如兰州）。
    对每个点，与"局部窗口(居中)的中位数"比较，用局部 MAD 标准化：
        z = |x - median_win| / (1.4826 * MAD_win)
    只抓"相对邻域的突变毛刺"，渐变爬坡/换季不会被误判为异常。
    这是经典的 Hampel identifier，比全局阈值/月度基线更稳健。
    """
    y = pd.Series(y).astype(float)
    med = y.rolling(win, center=True, min_periods=max(3, int(win * min_frac))).median()
    mad = (y - med).abs().rolling(win, center=True, min_periods=max(3, int(win * min_frac))).median()
    z = (y - med).abs() / (1.4826 * mad + 1e-9)
    flag = (z > k).fillna(False).to_numpy()
    info = dict(method="hampel", win=win, k=k, n_flagged=int(flag.sum()))
    return flag, info


# ----------------------------------------------------------------------------
# 4. 日历/时间特征
# ----------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()
    df["hour"] = idx.hour
    df["minute"] = idx.minute
    df["dayofweek"] = idx.day_of_week          # 0=周一 ... 6=周日
    df["month"] = idx.month
    df["dayofyear"] = idx.day_of_year
    df["is_weekend"] = (idx.day_of_week >= 5).astype(int)

    # 周期编码，避免 23 点与 0 点被当成数值远
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * idx.day_of_week / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * idx.day_of_week / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * idx.day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.day_of_year / 365.25)
    return df


def add_day_type(df: pd.DataFrame, has_flags: bool = True) -> pd.DataFrame:
    """
    合成三态 day_type：0=普通周末, 1=工作日, 2=法定节假日。
    has_flags=True 用原 is_workday/is_holiday；False（兰州无此列）按周一~周五=工作日近似。
    """
    df = df.copy()
    if has_flags and "is_workday" in df.columns and "is_holiday" in df.columns:
        dt = np.where(df["is_holiday"].fillna(0).astype(int) == 1, 2,
                      np.where(df["is_workday"].fillna(0).astype(int) == 1, 1, 0))
    else:
        # 无节假日日历时，仅用周末近似（文档中注明局限）
        dt = np.where(df.index.day_of_week >= 5, 0, 1)
    df["day_type"] = dt.astype(int)
    return df


# ----------------------------------------------------------------------------
# 5. 滞后特征（跨日/跨周同时刻）
# ----------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, target: str, steps_per_day: int,
                     steps_per_week: int, enabled: bool = True) -> pd.DataFrame:
    """
    enabled=True 时增加：
        lag_1d = 前 1 天同一时刻的目标值（shift steps_per_day）
        lag_1w = 前 1 周同一时刻的目标值（shift steps_per_week）
    只向后看，不引入未来信息。
    """
    df = df.copy()
    if not enabled:
        return df
    df["lag_1d"] = df[target].shift(steps_per_day)
    df["lag_1w"] = df[target].shift(steps_per_week)
    return df


# ----------------------------------------------------------------------------
# 6. 时序划分 / 归一化 / 滑窗（防泄漏三原则）
# ----------------------------------------------------------------------------
def split_chrono(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    n = len(df)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr = df.iloc[:n_tr]
    va = df.iloc[n_tr:n_tr + n_va]
    te = df.iloc[n_tr + n_va:]
    return tr, va, te


def fit_scalers(train_df: pd.DataFrame, feature_cols, target_col: str):
    """仅在训练集上 fit 的 MinMax[0,1]；X 与 y 分开。"""
    sx = MinMaxScaler((0, 1)).fit(train_df[feature_cols].to_numpy())
    sy = MinMaxScaler((0, 1)).fit(train_df[[target_col]].to_numpy())
    return sx, sy


def transform_X(df: pd.DataFrame, feature_cols, sx) -> np.ndarray:
    return sx.transform(df[feature_cols].to_numpy())


def transform_y(df: pd.DataFrame, target_col: str, sy) -> np.ndarray:
    return sy.transform(df[[target_col]].to_numpy()).ravel()


def make_windows(X: np.ndarray, y: np.ndarray, L: int, H: int, stride: int):
    """
    构造监督样本：
        X_win: (N, L, F)  过去 L 步全部特征
        y_win: (N, H)     未来 H 步目标
    返回 (X_win, y_win, end_idx)，end_idx 记录每个样本窗口结束位置（用于对齐时间）。
    """
    N = len(X) - L - H + 1
    if N <= 0:
        raise ValueError(f"样本不足: len={len(X)}, L={L}, H={H}")
    xs, ys, ends = [], [], []
    for i in range(0, N, stride):
        xs.append(X[i:i + L])
        ys.append(y[i + L:i + L + H])
        ends.append(i + L - 1)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ends)


# ----------------------------------------------------------------------------
# 7. 评估指标（后续阶段复用）
# ----------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    # RPD = SD(obs) / RMSE
    rpd = float(np.std(y_true) / (rmse + 1e-12))
    vae = float(np.var(np.abs(err)))
    # IA (Willmott)
    num = np.sum(err ** 2)
    den = np.sum((np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2)
    ia = 1 - num / (den + 1e-12)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=float(r2), RPD=rpd, VAE=vae, IA=float(ia))


# ----------------------------------------------------------------------------
# 8. 小工具
# ----------------------------------------------------------------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ----------------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 分层缺失处理单元测试
    idx = pd.date_range("2024-01-01", periods=60, freq="15min")
    s = pd.Series(np.arange(60, dtype=float), index=idx)
    s.iloc[3:6] = np.nan     # L=3 -> 插值
    s.iloc[20:30] = np.nan   # L=11 -> 移动平均
    s.iloc[40:55] = np.nan   # L=15 -> 移动平均(=15<=24)
    s.iloc[55:60] = np.nan   # 与上段合并 => L=20 -> 移动平均
    df = s.to_frame("v")
    filled, log, drop = tiered_fill(df)
    print("原始 NaN:", int(df.isna().sum().sum()))
    print("填充后 NaN:", int(filled.isna().sum().sum()), "删除行数:", int(drop.sum()))
    for r in log:
        print(r)
    # 指标冒烟测试
    yt = np.array([1., 2., 3.]); yp = np.array([1.1, 1.9, 3.2])
    print("metrics:", {k: round(v, 4) for k, v in evaluate(yt, yp).items()})
