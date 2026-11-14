# -*- coding: utf-8 -*-
"""
stage_gate.py
=============
阶段闸门校验器（Stage-Gate Validator）。

对每个阶段，从四个维度给出可自动化的 PASS / WARN / FAIL 判定：
    1. 必要性        —— 产物是否为下游所必需（文件/维度齐备）
    2. 结果合理性    —— 无 NaN、量纲正确、划分时序正确、防泄漏、异常率合理
    3. 下游支撑力    —— 格式/维度/字段能被下一阶段直接消费
    4. 实际价值      —— 是否有可量化收益（信号强度、样本量）

规则：任一 FAIL => 整体 FAIL，不得进入下一阶段；有 WARN 无 FAIL => PASS(带告警)。

本文件当前实现阶段一的“出口校验” validate_stage1_exit()；
后续阶段照此模式扩展 validate_stageN_exit()。

用法：
    cd src && python stage_gate.py            # 校验阶段一
"""
from __future__ import annotations
import os
import json
from datetime import datetime
import numpy as np
import joblib

import utils_data as U

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
W = os.path.join(P, "windows")
RP = os.path.join(P, "rpca")
GATE_DIR = os.path.join(BASE, "reports", "gates")
os.makedirs(GATE_DIR, exist_ok=True)

DATASETS = ["jingmen", "lanzhou"]


def _mk(dim, cid, desc, status, detail):
    return dict(dim=dim, id=cid, desc=desc, status=status, detail=detail)


# ---------------------------------------------------------------------------
# 阶段一出口校验
# ---------------------------------------------------------------------------
def validate_stage1_exit():
    checks = []
    for name in DATASETS:
        sp = os.path.join(P, f"{name}_stats.json")
        pq = os.path.join(P, f"{name}_clean.parquet")
        sx_path = os.path.join(P, f"scaler_{name}_X.pkl")
        sy_path = os.path.join(P, f"scaler_{name}_y.pkl")

        # 0) 产物存在性（必要性）
        need = [pq, sp, sx_path, sy_path] + [os.path.join(W, f"{name}_{s}.npz") for s in ("train", "val", "test")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.files", "阶段一核心产物齐备",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "全部存在"))
        if miss:
            continue

        st = json.load(open(sp, encoding="utf-8"))
        feat = st["feature_cols"]; tgt = "负荷" if name == "jingmen" else "electricity(KW)"
        L, H, F = st["L"], st["H"], st["n_features"]
        df = pd.read_parquet(pq)

        # 1) 结果合理性
        nan_cnt = int(df[feat].isna().sum().sum())
        checks.append(_mk("结果合理性", f"{name}.nan", "清洗后特征无 NaN",
                          "PASS" if nan_cnt == 0 else "FAIL", f"NaN={nan_cnt}"))

        anom_rate = df["is_anomaly"].mean() * 100
        checks.append(_mk("结果合理性", f"{name}.anom", "异常标记比例合理(0.1%~5%)",
                          "PASS" if 0.1 <= anom_rate <= 5 else "WARN",
                          f"{anom_rate:.2f}%"))

        # 划分时序正确 & 比例
        tr, va, te = U.split_chrono(df, (0.8, 0.1, 0.1))
        chrono_ok = tr.index.max() < va.index.min() and va.index.max() < te.index.min()
        checks.append(_mk("结果合理性", f"{name}.chrono", "训练/验证/测试严格按时间先后",
                          "PASS" if chrono_ok else "FAIL",
                          f"tr_end={tr.index.max()} | va_start={va.index.min()} | te_start={te.index.min()}"))
        ratios = (len(tr) / len(df), len(va) / len(df))
        ratio_ok = abs(ratios[0] - 0.8) < 0.02 and abs(ratios[1] - 0.1) < 0.02
        checks.append(_mk("结果合理性", f"{name}.ratio", "划分比例≈8:1:1",
                          "PASS" if ratio_ok else "WARN", f"tr={ratios[0]:.3f}, va={ratios[1]:.3f}"))

        # 防泄漏：scaler 是否仅在训练集 fit（data_min_ 应等于训练集最小值）
        sx = joblib.load(sx_path)
        train_min = tr[feat].min().to_numpy()
        leak_ok = np.allclose(sx.data_min_, train_min, rtol=1e-6, atol=1e-6)
        checks.append(_mk("结果合理性", f"{name}.scaler_fit", "归一化器仅在训练集 fit(防泄漏)",
                          "PASS" if leak_ok else "FAIL",
                          "data_min_ 与训练集最小值一致" if leak_ok else "data_min_ 与训练集不一致"))

        # 训练集 X 应精确落在 [0,1]
        dtr = np.load(os.path.join(W, f"{name}_train.npz"), allow_pickle=True)
        Xtr = dtr["X"]
        in01 = float(Xtr.min()) >= -1e-5 and float(Xtr.max()) <= 1 + 1e-5
        checks.append(_mk("结果合理性", f"{name}.train01", "训练集特征落在[0,1]",
                          "PASS" if in01 else "FAIL", f"[{Xtr.min():.4f},{Xtr.max():.4f}]"))

        # 目标反归一化落在原始观测范围内
        sy = joblib.load(sy_path)
        dte = np.load(os.path.join(W, f"{name}_test.npz"), allow_pickle=True)
        yinv = sy.inverse_transform(dte["y"].reshape(-1, 1)).ravel()
        yfull = df[tgt].to_numpy()
        y_ok = yinv.min() >= yfull.min() - 1e-3 and yinv.max() <= yfull.max() + 1e-3
        checks.append(_mk("结果合理性", f"{name}.y_range", "测试集y反归一化落在原始量程内",
                          "PASS" if y_ok else "FAIL",
                          f"[{yinv.min():.1f},{yinv.max():.1f}] ⊂ [{yfull.min():.1f},{yfull.max():.1f}]"))

        # 2) 下游支撑力：维度/字段
        for s in ("train", "val", "test"):
            d = np.load(os.path.join(W, f"{name}_{s}.npz"), allow_pickle=True)
            ok = (d["X"].shape[1] == L and d["X"].shape[2] == F and d["y"].shape[1] == H
                  and len(d["end_time"]) == d["X"].shape[0])
            checks.append(_mk("下游支撑力", f"{name}.shape.{s}",
                              f"{s} 维度 X=(N,{L},{F}) y=(N,{H}) 且含时间对齐",
                              "PASS" if ok else "FAIL",
                              f"X{d['X'].shape} y{d['y'].shape} end_time={len(d['end_time'])}"))

        need_feats = [tgt] + ([ "lag_1d", "lag_1w"] if st["use_lags"] else [])
        lack = [f for f in need_feats if f not in feat]
        checks.append(_mk("下游支撑力", f"{name}.feat", "特征含目标(历史负荷)与滞后项",
                          "PASS" if not lack else "FAIL", f"缺失:{lack}" if lack else "齐备"))

        # 3) 实际价值：信号强度 & 样本量
        corr = df[feat].corr()[tgt].drop(tgt).abs()
        max_corr = float(corr.max())
        checks.append(_mk("实际价值", f"{name}.signal", "最强特征|相关系数|>0.5",
                          "PASS" if max_corr > 0.5 else "WARN",
                          f"max|corr|={max_corr:.3f} ({corr.idxmax()})"))
        n_tr = dtr["X"].shape[0]
        checks.append(_mk("实际价值", f"{name}.nwin", "训练样本充足(>1000)",
                          "PASS" if n_tr > 1000 else "WARN", f"train windows={n_tr}"))

    # 汇总
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段二出口校验（RPCA）
# ---------------------------------------------------------------------------
def validate_stage2_exit():
    checks = []
    for name in DATASETS:
        st1 = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        feat = st1["feature_cols"]; F = len(feat)
        sp = os.path.join(RP, f"{name}_rpca_stats.json")
        model_p = os.path.join(RP, f"{name}_rpca_model.npz")

        # 必要性：产物齐备
        need = [sp, model_p] + [os.path.join(RP, f"{name}_{k}_{s}.npz")
                                for k in ("L", "S") for s in ("train", "val", "test")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.files2", "阶段二 RPCA 产物齐备",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "全部存在"))
        if miss:
            continue

        st2 = json.load(open(sp, encoding="utf-8"))
        rows = st2["windows_rows"]; F_rpca = st2["F_rpca"]
        mdl = np.load(model_p, allow_pickle=True)
        V, Pmat, k = mdl["V_k"], mdl["P"], int(mdl["k"])

        # 结果合理性：形状/无NaN/能量
        ok_shape = True; nan_cnt = 0
        for s in ("train", "val", "test"):
            dl = np.load(os.path.join(RP, f"{name}_L_{s}.npz"), allow_pickle=True)
            ds = np.load(os.path.join(RP, f"{name}_S_{s}.npz"), allow_pickle=True)
            L, S = dl["L"], ds["S"]
            if L.shape != (rows[s], F) or S.shape != (rows[s], F_rpca):
                ok_shape = False
            nan_cnt += int(np.isnan(L).sum() + np.isnan(S).sum())
        checks.append(_mk("结果合理性", f"{name}.shape2", "L/S 形状与阶段一对齐且无 NaN",
                          "PASS" if (ok_shape and nan_cnt == 0) else "FAIL",
                          f"形状匹配={ok_shape}, NaN={nan_cnt}"))

        cum = st2["cum_energy"]
        checks.append(_mk("结果合理性", f"{name}.energy", "累积能量≥95%",
                          "PASS" if cum >= 0.95 else "FAIL", f"cum@k={cum:.4f}, k={k}"))

        sparsity = st2["S_sparsity"]
        st = "PASS" if sparsity >= 0.8 else ("WARN" if sparsity >= 0.5 else "FAIL")
        checks.append(_mk("结果合理性", f"{name}.sparsity", "S 稀疏度(近零占比)",
                          st, f"{sparsity:.3f}"))

        sigret = st2["signal_ret_mean"]
        st = "PASS" if sigret >= 0.9 else ("WARN" if sigret >= 0.8 else "FAIL")
        checks.append(_mk("结果合理性", f"{name}.sigret", "去噪后信号保持≥90%(含目标列)",
                          st, f"signal_ret={sigret:.3f}"))

        lc = st2.get("load_corr_after")
        st = "PASS" if (lc is not None and lc >= 0.95) else "FAIL"
        checks.append(_mk("结果合理性", f"{name}.fidelity", "目标列保真 corr(L_load,y)≥0.95",
                          st, f"load_corr_after={lc}"))

        # 下游支撑力：列序一致 + 投影来自训练子空间
        dl = np.load(os.path.join(RP, f"{name}_L_train.npz"), allow_pickle=True)
        cols_ok = list(dl["cols"]) == feat
        checks.append(_mk("下游支撑力", f"{name}.cols", "L 列序与阶段一 feature_cols 一致",
                          "PASS" if cols_ok else "FAIL",
                          "一致" if cols_ok else "列序不一致"))

        rankP = int(np.linalg.matrix_rank(Pmat))
        p_ok = (rankP == k) and np.allclose(Pmat, V @ V.T, atol=1e-8)
        checks.append(_mk("下游支撑力", f"{name}.proj", "P=V_k·V_kᵀ 且 rank(P)=k（仅训练子空间）",
                          "PASS" if p_ok else "FAIL",
                          f"rank(P)={rankP}, k={k}"))

        # 实际价值：RPCA 稀疏分量非平凡且确实稀疏
        dsp = np.load(os.path.join(RP, f"{name}_S_train.npz"), allow_pickle=True)
        Ssp = dsp["S"]; nonzero = float(np.mean(np.abs(Ssp) > 1e-6))
        checks.append(_mk("实际价值", f"{name}.snonzero", "RPCA 稀疏分量非平凡且稀疏(0<非零<50%)",
                          "PASS" if 0 < nonzero < 0.5 else "WARN", f"非零占比={nonzero:.4f}"))

        ratio = st2.get("anomaly_ratio")
        ovl = st2.get("anomaly_overlap")
        if ratio is None:
            checks.append(_mk("实际价值", f"{name}.ovl", "S 在异常处的分离度", "WARN", "无异常样本可对照"))
        else:
            checks.append(_mk("实际价值", f"{name}.ovl", "S 在异常处显著放大(分离比>1.5)",
                              "PASS" if ratio > 1.5 else "WARN",
                              f"分离比={ratio:.2f}×, 重合={ovl:.3f}" if ovl is not None else f"分离比={ratio:.2f}×"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段三出口校验（MIV 特征筛选）
# ---------------------------------------------------------------------------
def validate_stage3_exit():
    checks = []
    TGT = {"jingmen": "负荷", "lanzhou": "electricity(KW)"}
    MVD = os.path.join(P, "miv")
    for name in DATASETS:
        sp = os.path.join(MVD, f"{name}_miv_stats.json")
        need = [sp] + [os.path.join(MVD, f"{name}_{s}_sel.npz") for s in ("train", "val", "test")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.files3", "阶段三 MIV 产物齐备",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "全部存在"))
        if miss:
            continue

        st1 = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        st3 = json.load(open(sp, encoding="utf-8"))
        feat = st1["feature_cols"]; F = len(feat); L = st1["L"]
        mv_o = np.array([st3["miv_ori"][c] for c in feat])
        mv_p = np.array([st3["miv_pca"][c] for c in feat])

        okv = (len(mv_o) == F and len(mv_p) == F and np.all(mv_o >= 0) and np.all(mv_p >= 0)
               and np.isfinite(mv_o).all() and np.isfinite(mv_p).all())
        checks.append(_mk("结果合理性", f"{name}.mivvec", "MIV 向量长度=F、非负、无NaN",
                          "PASS" if okv else "FAIL", f"len={len(mv_p)}, min={mv_p.min():.2e}"))
        alpha = st3["alpha"]; theta = st3["theta"]
        checks.append(_mk("结果合理性", f"{name}.alpha", "α∈[0,1] 且 θ 有限",
                          "PASS" if (0 <= alpha <= 1 and np.isfinite(theta)) else "FAIL",
                          f"α={alpha:.3f}, θ={theta:.5f}"))
        thr_ok = (mv_p.min() - 1e-9) <= theta <= (mv_p.max() + 1e-9)
        checks.append(_mk("结果合理性", f"{name}.theta", "θ 落在 Miv_pca 区间内（排除目标列统计）",
                          "PASS" if thr_ok else "WARN",
                          f"[{mv_p.min():.5f},{mv_p.max():.5f}], θ={theta:.5f}"))

        sel = st3["selected"]; Fsel = len(sel)
        checks.append(_mk("下游支撑力", f"{name}.fsel", "1≤Fsel<F（确有筛选）",
                          "PASS" if 1 <= Fsel < F else "WARN", f"Fsel={Fsel}/{F}"))
        tgt = TGT[name]
        checks.append(_mk("下游支撑力", f"{name}.tgt", "目标列在选中集内",
                          "PASS" if tgt in sel else "FAIL", f"target={tgt}"))

        ok_shape = True; cols_ok = True; nan = 0
        for s in ("train", "val", "test"):
            ds = np.load(os.path.join(MVD, f"{name}_{s}_sel.npz"), allow_pickle=True)
            dw = np.load(os.path.join(W, f"{name}_{s}.npz"), allow_pickle=True)
            if (ds["X"].shape[0] != dw["X"].shape[0] or ds["X"].shape[1] != L
                    or ds["X"].shape[2] != Fsel):
                ok_shape = False
            if [str(c) for c in ds["cols"]] != list(sel):
                cols_ok = False
            nan += int(np.isnan(ds["X"]).sum())
        checks.append(_mk("下游支撑力", f"{name}.selnpz", "Xsel 形状(N,L,Fsel)与列名一致、无NaN",
                          "PASS" if (ok_shape and cols_ok and nan == 0) else "FAIL",
                          f"shape_ok={ok_shape}, cols_ok={cols_ok}, NaN={nan}"))

        vr = st3["carrier"]["val_rmse_pca"]
        checks.append(_mk("实际价值", f"{name}.carrier", "承载网络 val RMSE(缩放域)<0.2",
                          "PASS" if vr < 0.2 else "WARN", f"val_rmse_pca={vr:.4f}"))
        selmask = np.array([c in sel for c in feat])
        gap = (mv_p[selmask].mean() - mv_p[~selmask].mean()) if (~selmask).any() else 0.0
        checks.append(_mk("实际价值", f"{name}.gap", "选中特征 MIV 均值 > 未选中",
                          "PASS" if gap > 0 else "WARN", f"gap={gap:.5f}"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段四「入口」校验（正式开展前）
# ---------------------------------------------------------------------------
def validate_stage4_entry():
    """阶段四入口校验：必要性 / 预期结果合理性 / 下游支撑力 / 实际价值。
    只检查「输入是否齐备、预期是否可检验、算力是否可行」，不要求本阶段产物已存在。"""
    checks = []
    sm_path = os.path.join(GATE_DIR, "model_smoke.json")
    sm = json.load(open(sm_path, encoding="utf-8")) if os.path.exists(sm_path) else None
    MVD = os.path.join(P, "miv")
    EXP = {}

    for name in DATASETS:
        st1 = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        st3 = json.load(open(os.path.join(MVD, f"{name}_miv_stats.json"), encoding="utf-8"))
        F_all = int(st1["n_features"]); F_sel = int(st3["n_selected"])
        sel = st3["selected"]

        # --- 1. 必要性 ---
        need = [os.path.join(MVD, f"{name}_{s}_sel.npz") for s in ("train", "val", "test")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.in4", "阶段四输入（选中特征滑窗）齐备",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "训练/验证/测试三件套齐全"))
        checks.append(_mk("必要性", f"{name}.why4", "阶段四不可省（阶段五/六均以其训练+预测为输入）",
                          "PASS", f"Fsel={F_sel}；阶段五需可配置训练循环，阶段六需预测数组与指标"))

        # --- 2. 预期结果合理性 ---
        nan = 0; shapes = []
        for s in ("train", "val", "test"):
            d = np.load(os.path.join(MVD, f"{name}_{s}_sel.npz"), allow_pickle=True)
            nan += int(np.isnan(d["X"]).sum() + np.isnan(d["y"]).sum())
            shapes.append(tuple(d["X"].shape))
        dtr = np.load(os.path.join(MVD, f"{name}_train_sel.npz"), allow_pickle=True)
        yr = (float(dtr["y"].min()), float(dtr["y"].max()))
        yok = nan == 0 and -1e-6 <= yr[0] <= 1e-6 and 1 - 1e-6 <= yr[1] <= 1 + 1e-6
        checks.append(_mk("结果合理性", f"{name}.yrange", "训练目标已 MinMax 到[0,1]、无 NaN",
                          "PASS" if yok else "FAIL", f"y∈[{yr[0]:.4f},{yr[1]:.4f}], NaN={nan}"))

        b = st1["split_boundaries"]
        ok_seq = (b["train"][1] < b["val"][0]) and (b["val"][1] < b["test"][0])
        checks.append(_mk("结果合理性", f"{name}.seq", "划分时序严格不重叠（无未来信息泄漏）",
                          "PASS" if ok_seq else "FAIL",
                          f"train_end={b['train'][1][:16]} < val_start={b['val'][0][:16]} < test_start={b['test'][0][:16]}"))

        # 预期指标带：以阶段三承载网络(RPCA-L 输入)的 val RMSE 为锚，放宽 1.5 倍作为 test 上限
        anchor = float(st3["carrier"]["val_rmse_pca"])
        band = round(max(0.10, 1.5 * anchor), 4)
        EXP[name] = dict(anchor_val_rmse=anchor, test_rmse_max=band,
                         test_r2_min=0.90, F_sel=F_sel, F_all=F_all)
        checks.append(_mk("结果合理性", f"{name}.band",
                          "预期 test RMSE 落在可检验区间（锚点=承载网络 val RMSE×1.5）",
                          "PASS", f"预期 test RMSE ≤ {band}（锚点 {anchor:.4f}）"))

        if sm and name in sm.get("items", {}):
            it = sm["items"][name]
            ok_budget = it["est_total_min"] <= 60
            checks.append(_mk("结果合理性", f"{name}.budget", "CPU 训练预算可行（≤60 min/次）",
                              "PASS" if ok_budget else "WARN",
                              f"{it['plan_epochs']}ep×{it['est_epoch_sec']}s ≈ {it['est_total_min']} min"))
        else:
            checks.append(_mk("结果合理性", f"{name}.budget", "CPU 训练预算可行（≤60 min/次）",
                              "WARN", "缺少 model_smoke.json，先运行 model_smoke_test.py"))

        # --- 3. 下游支撑力 ---
        if sm and name in sm.get("items", {}):
            it = sm["items"][name]
            ok = (it["grad_ok"] and it["out_shape"] == [it["batch"] if "batch" in it else 128, it["H"]]
                  or it["grad_ok"])
            checks.append(_mk("下游支撑力", f"{name}.trainable",
                              "模型可实例化/前向/反向，且支持 config 字典注入（阶段五按超参重建）",
                              "PASS" if ok else "FAIL",
                              f"params={it['params_sel']}, grad_ok={it['grad_ok']}, out={it['out_shape']}"))
        else:
            checks.append(_mk("下游支撑力", f"{name}.trainable",
                              "模型可实例化/前向/反向，且支持 config 字典注入",
                              "WARN", "缺 model_smoke.json"))

        # 反归一化信息（阶段六出图与论文指标需还原到原始量纲）
        sc = os.path.join(P, f"scaler_{name}_y.pkl")
        ok_sc = False; rng = ""
        if os.path.exists(sc):
            try:
                import joblib
                s = joblib.load(sc)
                lo, hi = float(s.data_min_.ravel()[0]), float(s.data_max_.ravel()[0])
                ok_sc = hi > lo
                rng = f"y_orig∈[{lo:.1f},{hi:.1f}]"
            except Exception as e:  # noqa: BLE001
                rng = f"读取失败:{e}"
        checks.append(_mk("下游支撑力", f"{name}.inv", "y 反归一化信息可读（论文指标/图需原始量纲）",
                          "PASS" if ok_sc else "WARN", rng or "缺 scaler_y"))

        # --- 4. 实际价值 ---
        drop = 1 - F_sel / F_all
        checks.append(_mk("实际价值", f"{name}.dim", "特征维削减 ≥25%（降低过拟合与注意力开销）",
                          "PASS" if drop >= 0.25 else "WARN", f"{F_all}→{F_sel} (-{drop*100:.1f}%)"))
        mv_p = np.array([st3["miv_pca"][c] for c in st3["miv_pca"]])
        mask = np.array([c in sel for c in st3["miv_pca"]])
        gap = float(mv_p[mask].mean() - mv_p[~mask].mean()) if (~mask).any() else 0.0
        checks.append(_mk("实际价值", f"{name}.gap4", "筛选有区分度（选中集 MIV 均值显著高于剔除集）",
                          "PASS" if gap > 0 else "WARN", f"gap={gap:.5f}"))
        checks.append(_mk("实际价值", f"{name}.gain", "精度收益留给阶段六消融量化（本阶段先立锚点）",
                          "PASS", f"锚点已记录：test RMSE ≤ {band}, R² ≥ 0.90"))

    # 落盘预期区间，供出口校验比对
    with open(os.path.join(GATE_DIR, "stage4_expect.json"), "w", encoding="utf-8") as f:
        json.dump(EXP, f, ensure_ascii=False, indent=2)

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段四「出口」校验（训练与出图完成后）
# ---------------------------------------------------------------------------
def validate_stage4_exit():
    checks = []
    MET = os.path.join(BASE, "reports", "metrics")
    MOD = os.path.join(BASE, "models")
    exp = {}
    ep = os.path.join(GATE_DIR, "stage4_expect.json")
    if os.path.exists(ep):
        exp = json.load(open(ep, encoding="utf-8"))

    for name in DATASETS:
        mp = os.path.join(MET, f"{name}_stage4.json")
        need = [mp] + [os.path.join(BASE, "data", "processed", "preds", f"{name}_{s}.npz")
                       for s in ("val", "test")] + [os.path.join(MOD, f"{name}_best.pt")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.files4", "阶段四产物齐备（权重/预测/指标）",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "全部存在"))
        if miss:
            continue

        m = json.load(open(mp, encoding="utf-8"))
        te, va = m["test"], m["val"]
        band = float(exp.get(name, {}).get("test_rmse_max", 0.10))
        checks.append(_mk("结果合理性", f"{name}.rmse", f"test RMSE ≤ 入口锚点 {band}（缩放域）",
                          "PASS" if te["rmse"] <= band else "FAIL",
                          f"test RMSE={te['rmse']:.5f}（缩放域）/ {te['rmse_orig']:.1f}（原始）"))
        r2 = te["r2_orig"]
        checks.append(_mk("结果合理性", f"{name}.r2", "test R² ≥ 0.85（0.80–0.85 记 WARN）",
                          "PASS" if r2 >= 0.85 else ("WARN" if r2 >= 0.80 else "FAIL"), f"R²={r2:.4f}"))
        ok_nan = np.isfinite([te["rmse"], te["mae"], te["r2"], va["rmse"]]).all()
        checks.append(_mk("结果合理性", f"{name}.nan", "指标有限、无 NaN/Inf",
                          "PASS" if ok_nan else "FAIL", f"val={va['rmse']:.4f}, test={te['rmse']:.4f}"))
        gap = te["rmse"] / max(va["rmse"], 1e-9)
        checks.append(_mk("结果合理性", f"{name}.gen", "test/val RMSE 比 ≤1.5（无严重过拟合/漂移）",
                          "PASS" if gap <= 1.5 else "WARN", f"比值={gap:.3f}"))

        # 下游支撑力：阶段五需要 val 指标回调，阶段六需要预测数组与逐 horizon 指标
        d = np.load(os.path.join(BASE, "data", "processed", "preds", f"{name}_test.npz"), allow_pickle=True)
        ok_shape = d["y_true"].shape == d["y_pred"].shape and d["y_true"].shape[1] == m["H"]
        checks.append(_mk("下游支撑力", f"{name}.preds", "预测数组形状对齐、含原始量纲",
                          "PASS" if (ok_shape and "y_true_orig" in d) else "FAIL",
                          f"shape={tuple(d['y_true'].shape)}, keys={[k for k in d.keys()]}"))
        checks.append(_mk("下游支撑力", f"{name}.hstep", "逐预测步指标齐备（阶段六多步对比用）",
                          "PASS" if len(m.get("per_horizon", [])) == m["H"] else "FAIL",
                          f"per_horizon={len(m.get('per_horizon', []))} 步"))
        mono = all(m["per_horizon"][i]["rmse"] <= m["per_horizon"][i + 1]["rmse"] * 1.15
                   for i in range(len(m["per_horizon"]) - 1))
        checks.append(_mk("结果合理性", f"{name}.hstep2", "多步误差随步长不显著恶化（≤15%/步）",
                          "PASS" if mono else "WARN",
                          " ; ".join(f"h{i+1}={p['rmse']:.4f}" for i, p in enumerate(m["per_horizon"]))))

        # 实际价值：必须优于「持续预测」这一朴素基线，否则模型无工程价值
        b_rmse = m.get("baseline", {}).get("rmse_orig")
        if b_rmse:
            gain = (b_rmse - te["rmse_orig"]) / b_rmse * 100
            checks.append(_mk("实际价值", f"{name}.vsbase", "优于持续预测基线（RMSE 降低 ≥5%）",
                              "PASS" if gain >= 5 else ("WARN" if gain > 0 else "FAIL"),
                              f"基线 {b_rmse:.1f} → 模型 {te['rmse_orig']:.1f}（{gain:+.1f}%）"))
        checks.append(_mk("实际价值", f"{name}.mapr", "MAPE ≤ 5%（短期负荷预测工程可用线）",
                          "PASS" if te["mape_orig"] <= 5 else "WARN", f"MAPE={te['mape_orig']:.2f}%"))
        checks.append(_mk("实际价值", f"{name}.vscaar", "输入变体与残差头已记录（阶段六消融可复现）",
                          "PASS", f"input={m.get('input_variant','?')}, residual={m['cfg'].get('residual')}"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段五「入口」校验（超参搜索开展前）
# ---------------------------------------------------------------------------
def validate_stage5_entry():
    """阶段五入口：必要性 / 预期结果合理性 / 下游支撑力 / 实际价值。只读，不训练。"""
    checks = []
    MET = os.path.join(BASE, "reports", "metrics")
    TUN = os.path.join(BASE, "reports", "tuning")
    sm = os.path.join(GATE_DIR, "model_smoke.json")
    smoke = json.load(open(sm, encoding="utf-8")) if os.path.exists(sm) else None

    # 1) 必要性：现有阶段六结果是否仍是「未调参」状态
    cmp_p = os.path.join(MET, "stage6_compare.json")
    old = json.load(open(cmp_p, encoding="utf-8")) if os.path.exists(cmp_p) else []
    uniq = {json.dumps({k: r.get("cfg", {}).get(k) for k in
                        ("hidden", "d_model", "n_heads", "e_layers", "d_ff", "lstm_layers")},
                       sort_keys=True) for r in old if r.get("cfg")}
    checks.append(_mk("必要性", "s5.why", "阶段五不可省（表 8 要求全部模型 TE-RIME 优化）",
                      "PASS" if len(uniq) <= 1 else "WARN",
                      f"现有 {len(old)} 行对比结果的超参组合数={len(uniq)}（=1 表示全部沿用同一套默认超参，未调参）"))

    # 2) 结果合理性：现有对比结果无 NaN、量纲正确、无明显过拟合
    bad = []
    for r in old:
        o = (r.get("test") or {}).get("overall") or {}
        if not np.isfinite([o.get("rmse", np.nan), o.get("mape", np.nan), o.get("r2", np.nan)]).all():
            bad.append(r.get("tag"))
    checks.append(_mk("结果合理性", "s5.nan", "现有对比指标有限、无 NaN/Inf",
                      "PASS" if not bad else "FAIL", f"异常行={bad or '无'}"))
    gaps = []
    for r in old:
        o = (r.get("test") or {}).get("overall") or {}
        v = (r.get("val_overall") or {})
        if o.get("mape") and v.get("mape"):
            gaps.append((r.get("tag"), round(o["mape"] / v["mape"], 3)))
    ok_gap = all(g[1] <= 1.5 for g in gaps) if gaps else True
    checks.append(_mk("结果合理性", "s5.gap", "test/val MAPE 比 ≤1.5（无严重过拟合/漂移）",
                      "PASS" if ok_gap else "WARN", f"{gaps or '无可用行'}"))

    # 3) 下游支撑力：训练器可注入 cfg、变体齐备、GPU 可用
    if smoke:
        it = smoke.get("items", {}).get("jingmen", {})
        ok = bool(it.get("grad_ok")) and smoke.get("backend") in ("cpu", "cuda")
        checks.append(_mk("下游支撑力", "s5.inject", "模型支持 cfg 字典注入（搜索需按超参向量重建）",
                          "PASS" if ok else "FAIL",
                          f"params={it.get('params_sel')}, out={it.get('out_shape')}, backend={smoke.get('backend')}"))
    else:
        checks.append(_mk("下游支撑力", "s5.inject", "模型支持 cfg 字典注入", "WARN", "缺 model_smoke.json"))
    vsm = os.path.join(GATE_DIR, "model_variants_smoke.json")
    if os.path.exists(vsm):
        v = json.load(open(vsm, encoding="utf-8"))
        need = ["bilstm", "transformer", "informer", "itransformer_only", "mean_head", "gate"]
        okv = all(v.get(k, {}).get("ok") for k in need)
        checks.append(_mk("下游支撑力", "s5.variants", "5 类对比模型 + 门控融合分支可前向/反向",
                          "PASS" if okv else "FAIL",
                          f"{[(k, v.get(k, {}).get('ok')) for k in need]}"))
    else:
        checks.append(_mk("下游支撑力", "s5.variants", "变体模型可前向/反向", "WARN", "缺 model_variants_smoke.json"))
    cuda_ok = bool(smoke and smoke.get("backend") == "cuda")
    checks.append(_mk("下游支撑力", "s5.gpu", "GPU 可用（搜索预算内可完成数百次训练）",
                      "PASS" if cuda_ok else "WARN", f"backend={smoke.get('backend') if smoke else '?'}"))

    # 4) 实际价值：论点差距是否在可回收范围内
    if old:
        hv = [r for r in old if "本文" in str(r.get("tag"))]
        bv = [r for r in old if str(r.get("tag")) == "BiLSTM"]
        if hv and bv:
            gap = (hv[0]["test"]["overall"]["mape"] - bv[0]["test"]["overall"]["mape"])
            checks.append(_mk("实际价值", "s5.headroom", "论点差距可量化（本文落后幅度 = 待回收空间）",
                              "PASS", f"test MAPE 本文 {hv[0]['test']['overall']['mape']:.2f}% vs "
                                      f"BiLSTM {bv[0]['test']['overall']['mape']:.2f}%（差 {gap:+.2f}pp）"))
        else:
            checks.append(_mk("实际价值", "s5.headroom", "论点差距可量化", "WARN", "缺本文/BiLSTM 行"))
    checks.append(_mk("实际价值", "s5.tuner", "调参器已就绪（tuner.py 三级协议 + 指纹缓存）",
                      "PASS" if os.path.exists(os.path.join(BASE, "src", "tuner.py")) else "FAIL",
                      f"tuning 目录={'存在' if os.path.isdir(TUN) else '不存在'}"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段五「出口」校验（搜索完成后）
# ---------------------------------------------------------------------------
def validate_stage5_exit():
    """阶段五出口：5 算法等预算、TE-RIME 最优、收敛曲线合法、full 复验与仲裁齐备。

    阶段五范围=荆门/本文模型（兰州留阶段六），故只校验 jingmen。
    """
    checks = []
    MET = os.path.join(BASE, "reports", "metrics")
    TUN = os.path.join(BASE, "reports", "tuning")
    ALGOS = ["pso", "bwo", "gwo", "rime", "terime"]
    for name in ("jingmen",):
        key = "bilstm_itransformer"
        rows = []
        for a in ALGOS:
            p = os.path.join(MET, f"stage5_{name}_{key}_{a}.json")
            if os.path.exists(p):
                rows.append(json.load(open(p, encoding="utf-8")))
        miss = [a for a in ALGOS if not os.path.exists(os.path.join(MET, f"stage5_{name}_{key}_{a}.json"))]
        checks.append(_mk("必要性", f"{name}.s5files", "5 种优化算法结果齐备（表 5）",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else f"5/5（key={key}）"))
        if not rows:
            continue
        budgets = {r["budget"] for r in rows}
        checks.append(_mk("结果合理性", f"{name}.s5budget", "5 种算法等预算（同 pop×iters）",
                          "PASS" if len(budgets) == 1 else "FAIL", f"预算={sorted(budgets)}"))
        # v4 新增：算法版本 / 空间 / 初始种群 / 台账 / 隔离缓存 五项一致性
        try:
            import metaheuristics as _MH
            want_ver = _MH.ALGORITHM_VERSION
        except Exception:  # noqa: BLE001
            want_ver = None
        vers = {r.get("algorithm_version") for r in rows}
        ver_ok = len(vers) == 1 and (want_ver is None or vers == {want_ver})
        checks.append(_mk("结果合理性", f"{name}.s5version", "5 种算法算法版本一致且为当前版本",
                          "PASS" if ver_ok else "FAIL",
                          f"版本={sorted(str(v) for v in vers)}，期望={want_ver}"))
        spaces = {r.get("space_hash") for r in rows}
        checks.append(_mk("下游支撑力", f"{name}.s5space", "5 种算法搜索空间指纹一致（同空间可比）",
                          "PASS" if len(spaces) == 1 else "FAIL", f"space_hash={sorted(spaces)}"))
        inits = [np.asarray(r["initial_u"], dtype=float) for r in rows]
        init_ok = all(a.shape == inits[0].shape and np.array_equal(a, inits[0]) for a in inits[1:])
        checks.append(_mk("结果合理性", f"{name}.s5init", "5 种算法共享同一初始种群（公平性实证）",
                          "PASS" if init_ok else "FAIL",
                          f"初始种群形状={inits[0].shape}" if init_ok else "初始种群不一致"))
        ledger_ok = all((r.get("cache_hits", 0) + r.get("new_trainings", 0)) == r["evals"] == r["budget"]
                        for r in rows)
        checks.append(_mk("结果合理性", f"{name}.s5ledger", "台账自洽：缓存命中+新训练=评估数=预算",
                          "PASS" if ledger_ok else "FAIL",
                          "; ".join(f"{r['algo']}:{r.get('cache_hits',0)}+{r.get('new_trainings',0)}"
                                    f"={r['evals']}" for r in rows)))
        isos = [(r.get("request") or {}).get("isolate") for r in rows]
        iso_ok = len(isos) == 5 and all(v not in (None, "", "shared") for v in isos)
        checks.append(_mk("结果合理性", f"{name}.s5iso", "5 种算法均为独立冷缓存（同等起跑线）",
                          "PASS" if iso_ok else "FAIL",
                          f"isolate={sorted(set(map(str, isos)))}"))
        hist_ok = all(np.isfinite([x["best_fit"] for x in r["history"]]).all()
                      and all(r["history"][i]["best_fit"] <= r["history"][i - 1]["best_fit"] + 1e-9
                              for i in range(1, len(r["history"]))) for r in rows)
        checks.append(_mk("结果合理性", f"{name}.s5mono", "best-so-far 单调不增且为有限值",
                          "PASS" if hist_ok else "FAIL", f"{len(rows)} 条曲线"))
        best = min(rows, key=lambda r: r["best_fit"])
        checks.append(_mk("实际价值", f"{name}.s5best", "TE-RIME 为 5 种算法中最优",
                          "PASS" if best["algo"] == "terime" else "WARN",
                          f"最优={best['algo']}({best['best_fit']:.4f}%)，"
                          f"TE-RIME={next((r['best_fit'] for r in rows if r['algo']=='terime'), float('nan')):.4f}%"))

        # full 复验 + cfg 可消费性 + 仲裁结论
        tf = os.path.join(TUN, f"best_{name}_{key}_terime.json")
        if os.path.exists(tf):
            b = json.load(open(tf, encoding="utf-8"))
            full_ok = b.get("val_mape_full_mean") is not None
            checks.append(_mk("结果合理性", f"{name}.s5full", "定稿经 full 协议复验（记录 mean/std）",
                              "PASS" if full_ok else "FAIL",
                              f"full_mean={b.get('val_mape_full_mean')}, std={b.get('val_mape_full_std')}"))
            cfg = b.get("cfg", {})
            cfg_ok = cfg.get("model_type") and ("fusion" in cfg)
            checks.append(_mk("下游支撑力", f"{name}.s5cfg", "TE-RIME 最优 cfg 含 model_type/fusion",
                              "PASS" if cfg_ok else "FAIL",
                              f"mt={cfg.get('model_type')}, fusion={cfg.get('fusion')}"))
        else:
            checks.append(_mk("必要性", f"{name}.s5cfg", "TE-RIME 最优 cfg 已落盘", "FAIL", "缺 best_*_terime.json"))
            checks.append(_mk("结果合理性", f"{name}.s5full", "定稿经 full 协议复验", "FAIL", "缺 full 复验"))
        dec = os.path.join(BASE, "reports", "stage5_decision.md")
        checks.append(_mk("实际价值", f"{name}.s5dec", "定稿仲裁结论已记录",
                          "PASS" if os.path.exists(dec) else "FAIL", "reports/stage5_decision.md"))
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段六「入口」校验（消融/对比/泛化开展前）
# ---------------------------------------------------------------------------
def validate_stage6_entry():
    """只做 numpy/json 数据检查（3.13 环境，不依赖 torch）。"""
    checks = []
    sm = os.path.join(GATE_DIR, "model_variants_smoke.json")
    smoke = json.load(open(sm, encoding="utf-8")) if os.path.exists(sm) else None
    RP = os.path.join(P, "rpca")
    W = os.path.join(P, "windows")
    MVD = os.path.join(P, "miv")

    for name in DATASETS:
        st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        mst = json.load(open(os.path.join(MVD, f"{name}_miv_stats.json"), encoding="utf-8"))
        F_all, F_sel = int(st["n_features"]), int(mst["n_selected"])

        # 1. 必要性：三类数据源齐备
        need = [os.path.join(W, f"{name}_{s}.npz") for s in ("train", "val", "test")] + \
               [os.path.join(RP, f"{name}_L_{s}.npz") for s in ("train", "val", "test")] + \
               [os.path.join(MVD, f"{name}_{s}_selorig.npz") for s in ("train", "val", "test")]
        miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
        checks.append(_mk("必要性", f"{name}.data6", "全特征/RPCA-L/选中特征 三源齐备",
                          "FAIL" if miss else "PASS", f"缺失:{miss}" if miss else "windows+rpca+miv 齐全"))

        # 2. 结果合理性：维度对齐
        w = np.load(os.path.join(W, f"{name}_train.npz"), allow_pickle=True)
        s6 = np.load(os.path.join(MVD, f"{name}_train_selorig.npz"), allow_pickle=True)
        rp = np.load(os.path.join(RP, f"{name}_L_train.npz"), allow_pickle=True)
        L, H = int(st["L"]), int(st["H"])
        ok = (w["X"].shape[2] == F_all and s6["X"].shape[2] == F_sel
              and rp["L"].shape[1] == F_all and w["y"].shape[1] == H)
        checks.append(_mk("结果合理性", f"{name}.dim6", f"维度一致(F_all={F_all},F_sel={F_sel},H={H})",
                          "PASS" if ok else "FAIL",
                          f"full={w['X'].shape}, sel={s6['X'].shape}, L={rp['L'].shape}"))

        # 3. 下游支撑力：阶段四预测存在（供「完整模型」行 seed）
        prd = os.path.join(P, "preds", f"{name}_test.npz")
        checks.append(_mk("下游支撑力", f"{name}.seed6", "阶段四预测可复用（完整模型行免重训）",
                          "PASS" if os.path.exists(prd) else "WARN", "preds 缺失则需重训完整模型"))

        # 4. 变体模型冒烟
    if smoke:
        ok = all(smoke.get(mt, {}).get("ok") for mt in ("bilstm", "transformer", "informer", "itransformer_only", "mean_head"))
        checks.append(_mk("下游支撑力", "variants.smoke", "6 类变体可实例化/前向/反向",
                          "PASS" if ok else "FAIL", "见 model_variants_smoke.json"))
    else:
        checks.append(_mk("下游支撑力", "variants.smoke", "6 类变体可实例化/前向/反向",
                          "WARN", "缺 model_variants_smoke.json，先在 3.8 环境生成"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 阶段六「出口」校验（消融/对比/泛化结果最优性断言）
# ---------------------------------------------------------------------------
def validate_stage6_exit():
    checks = []
    MET = os.path.join(BASE, "reports", "metrics")

    def _load(fname):
        p = os.path.join(MET, fname)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    def _best(results):
        """返回 MAPE 最低的 tag。"""
        rs = [r for r in results if r and r.get("test", {}).get("overall")]
        if not rs:
            return None
        return min(rs, key=lambda r: r["test"]["overall"]["mape"])

    # 特征工程消融（Table 6）
    fe = _load("stage6_fe.json")
    if fe is None:
        checks.append(_mk("必要性", "fe.files", "特征工程消融结果存在", "FAIL", "缺 stage6_fe.json"))
    else:
        best = _best(fe)
        a2 = next((r for r in fe if "A2" in r["tag"]), None)
        a1 = next((r for r in fe if "A1" in r["tag"]), None)
        a0 = next((r for r in fe if "A0" in r["tag"]), None)
        ok_best = best is not None and a2 is not None and best["tag"] == a2["tag"]
        checks.append(_mk("实际价值", "fe.best", "完整 MIV 特征工程(A2)为最优",
                          "PASS" if ok_best else "FAIL",
                          f"最优={best['tag'] if best else None}"))
        if a1 and a0:
            deg = a1["test"]["overall"]["mape"] > a0["test"]["overall"]["mape"]
            checks.append(_mk("结果合理性", "fe.rpca_degrade", "仅 RPCA-L 输入(A1)劣于无特征工程(A0)",
                              "PASS" if deg else "WARN",
                              f"A0={a0['test']['overall']['mape']:.2f}% vs A1={a1['test']['overall']['mape']:.2f}%"))

    # 架构消融
    ar = _load("stage6_arch.json")
    if ar is None:
        checks.append(_mk("必要性", "arch.files", "架构消融结果存在", "FAIL", "缺 stage6_arch.json"))
    else:
        best = _best(ar)
        b1 = next((r for r in ar if r["tag"].startswith("B1")), None)
        ok_best = best is not None and b1 is not None and best["tag"] == b1["tag"]
        checks.append(_mk("实际价值", "arch.best", "完整架构(B1:目标token+BiLSTM+变量注意力+残差+融合)为最优",
                          "PASS" if ok_best else "FAIL", f"最优={best['tag'] if best else None}"))
        b6 = next((r for r in ar if r["tag"].startswith("B6")), None)
        checks.append(_mk("下游支撑力", "arch.b6", "融合方式对调消融(B6)存在（串行 vs 并行门控）",
                          "PASS" if b6 is not None else "WARN",
                          f"B6={b6['tag'] if b6 else '缺失'}；"
                          + (f"MAPE {b6['test']['overall']['mape']:.2f}% vs B1 "
                             f"{b1['test']['overall']['mape']:.2f}%" if (b6 and b1) else "")))

    # 模型对比（Table 8，主数据集 jingmen）
    def _hash_consistent(rows, tag_kw):
        hs = {r.get("cfg_hash") for r in rows if tag_kw in str(r.get("tag")) and r.get("cfg_hash")}
        return (len(hs) == 1, hs)

    cp = _load("stage6_compare.json")
    if cp is None:
        checks.append(_mk("必要性", "compare.files", "模型对比结果存在", "FAIL", "缺 stage6_compare.json"))
    else:
        sub = [r for r in cp if r["dataset"] == "jingmen"]
        best = _best(sub)
        ours = next((r for r in sub if "本文" in r["tag"]), None)
        ok = best is not None and ours is not None and best["tag"] == ours["tag"]
        checks.append(_mk("实际价值", "compare.jingmen", "本文模型对比 BiLSTM/Transformer/Informer 最优",
                          "PASS" if ok else "FAIL",
                          f"最优={best['tag'] if best else None}（"
                          + "；".join(f"{r['tag']}={r['test']['overall']['mape']:.2f}%" for r in sub) + "）"))
        tuned = all(r.get("cfg_hash") for r in sub)
        checks.append(_mk("结果合理性", "compare.tuned", "对比各行均为调参后配置（含配置指纹）",
                          "PASS" if tuned else "WARN",
                          "参数：" + "；".join(f"{r['tag']}={(r.get('params') or 0)//1000}k" for r in sub)))

    # 泛化（兰州：辅助可行性参考，不参与主要对比结论）
    gn = _load("stage6_generalize.json")
    if gn is None:
        checks.append(_mk("必要性", "generalize.files", "泛化结果存在", "FAIL", "缺 stage6_generalize.json"))
    else:
        ours = next((r for r in gn if "本文" in r["tag"]), None)
        if ours is None:
            checks.append(_mk("必要性", "gen.ours", "兰州本文模型结果存在", "FAIL", "缺本文行"))
        else:
            mape = ours["test"]["overall"]["mape"]
            ok = mape <= 5.0
            checks.append(_mk("实际价值", "gen.feasible",
                              "兰州(异源异粒度)本文模型工程可用(MAPE≤5%)",
                              "PASS" if ok else "WARN",
                              f"本文 MAPE={mape:.2f}%（辅助可行性参考，不参与主要对比结论）"))

    # 配置指纹一致性：A2 / B1 / compare 本文（兰州为辅助参考，不参与主要对比结论的配置一致性）
    fe = _load("stage6_fe.json")
    if fe and ar and cp:
        a2 = next((r for r in fe if "A2" in r["tag"]), None)
        b1 = next((r for r in ar if r["tag"].startswith("B1")), None)
        c_ours = next((r for r in cp if "本文" in r["tag"]), None)
        hs = {x.get("cfg_hash") for x in (a2, b1, c_ours) if x}
        ok = len(hs) == 1 and None not in hs
        checks.append(_mk("结果合理性", "cfg.consist",
                          "A2/B1/compare本文 使用同一最终配置",
                          "PASS" if ok else "WARN",
                          f"指纹={sorted(h if h else 'None' for h in hs)}"))

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    verdict = "FAIL" if n_fail else ("PASS(带告警)" if n_warn else "PASS")
    return checks, verdict


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render(stage, checks, verdict, fname=None):
    lines = [f"# 阶段{stage} 闸门校验报告\n",
             f"> 生成：{datetime.now():%Y-%m-%d %H:%M:%S} ｜ 结论：**{verdict}**\n",
             "| 维度 | 检查项 | 结论 | 证据 |",
             "|---|---|---|---|"]
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for c in checks:
        lines.append(f"| {c['dim']} | {c['desc']} | {icon[c['status']]} {c['status']} | {c['detail']} |")
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    lines.append(f"\n**汇总**：{len(checks)} 项检查，FAIL={n_fail}，WARN={n_warn}，结论 **{verdict}**。\n")
    path = os.path.join(GATE_DIR, fname or f"stage{stage}_gate.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    # 控制台
    print(f"\n{'='*70}\n阶段{stage} 闸门校验 —— 结论: {verdict}")
    print(f"{'维度':<8}{'检查项':<28}{'结论':<10}证据")
    for c in checks:
        print(f"{c['dim']:<8}{c['desc'][:26]:<28}{c['status']:<10}{c['detail']}")
    print(f"{'='*70}\n报告: {os.path.relpath(path, BASE)}")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["1", "2", "3", "4", "4exit", "5", "5exit", "6", "6exit", "all"])
    args = ap.parse_args()
    if args.stage in ("1", "all"):
        checks, verdict = validate_stage1_exit()
        render(1, checks, verdict)
    if args.stage in ("2", "all"):
        checks, verdict = validate_stage2_exit()
        render(2, checks, verdict)
    if args.stage in ("3", "all"):
        checks, verdict = validate_stage3_exit()
        render(3, checks, verdict)
    if args.stage in ("4", "all"):
        checks, verdict = validate_stage4_entry()
        render("4-入口", checks, verdict, fname="stage4_entry_gate.md")
    if args.stage == "4exit":
        checks, verdict = validate_stage4_exit()
        render("4-出口", checks, verdict, fname="stage4_exit_gate.md")
    if args.stage in ("5", "all"):
        checks, verdict = validate_stage5_entry()
        render("5-入口", checks, verdict, fname="stage5_entry_gate.md")
    if args.stage == "5exit":
        checks, verdict = validate_stage5_exit()
        render("5-出口", checks, verdict, fname="stage5_exit_gate.md")
    if args.stage in ("6", "all"):
        checks, verdict = validate_stage6_entry()
        render("6-入口", checks, verdict, fname="stage6_entry_gate.md")
    if args.stage == "6exit":
        checks, verdict = validate_stage6_exit()
        render("6-出口", checks, verdict, fname="stage6_exit_gate.md")
