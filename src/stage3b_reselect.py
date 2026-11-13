# -*- coding: utf-8 -*-
"""
stage3b_reselect.py
===================
特征重选（阶段六出口校验 FAIL 后的修正）：结合 MIV 重要性排序与负荷预测领域先验，
**强制保留全部气象 + 负荷特征 + 关键日历特征**，仅剔除"明确冗余/低信息"列：

    month / is_holiday / jiangyuliang / dayofyear / is_weekend / lag_1d / lag_1w

理由：MIV 用轻量承载网络 + 边际扰动，对"与 season/hour 共线"的气象特征（温度/风/辐射/露点）
会系统性低估其重要性，导致误删；而这些正是负荷预测的核心驱动因子。

本脚本不重训任何网络，直接从已有数据重建选中特征滑窗：
    selorig  <- windows/{name}_{split}.npz（原始，新子集）
    sel      <- rpca/{name}_L_{split}.npz（RPCA-L，新子集）
并把 miv_stats.json 的 selected 更新为新子集（原 MIV 选择备份到 selected_orig_miv）。

运行：cd src && "<py3.8>" stage3b_reselect.py
"""
from __future__ import annotations
import os
import json
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(BASE, "data", "processed")
W = os.path.join(P, "windows")
RP = os.path.join(P, "rpca")
MV = os.path.join(P, "miv")

# 领域先验：明确冗余/低信息列（其余全部保留）
DROP_COLS = {
    "jingmen": ["month", "is_holiday", "jiangyuliang", "dayofyear", "is_weekend", "lag_1d", "lag_1w"],
    "lanzhou": ["month", "dayofyear", "is_weekend", "lag_1d", "lag_1w"],
}


def build_windows(M, L, H, stride):
    N = len(M) - L - H + 1
    idx = list(range(0, N, stride))
    return np.stack([M[i:i + L] for i in idx], axis=0).astype(np.float32)


def main():
    for name, drop in DROP_COLS.items():
        st = json.load(open(os.path.join(P, f"{name}_stats.json"), encoding="utf-8"))
        feats = list(st["feature_cols"])
        L, H, stride = int(st["L"]), int(st["H"]), int(st["stride"])
        sel = [c for c in feats if c not in drop]
        sel_idx = [feats.index(c) for c in sel]
        print(f"\n{name}: {len(feats)} -> {len(sel)} 特征（剔除 {drop}）")

        # 1) 原始选中特征（selorig）
        for s in ("train", "val", "test"):
            w = np.load(os.path.join(W, f"{name}_{s}.npz"), allow_pickle=True)
            X = w["X"][:, :, sel_idx].astype(np.float32)
            np.savez_compressed(os.path.join(MV, f"{name}_{s}_selorig.npz"),
                                X=X, y=w["y"].astype(np.float32),
                                cols=np.array(sel), L=L, H=H)
            # 2) RPCA-L 选中特征（sel）
            rp = np.load(os.path.join(RP, f"{name}_L_{s}.npz"), allow_pickle=True)
            M = rp["L"].astype(np.float32)
            Xl = build_windows(M, L, H, stride)[:, :, sel_idx]
            np.savez_compressed(os.path.join(MV, f"{name}_{s}_sel.npz"),
                                X=Xl, y=w["y"].astype(np.float32),
                                cols=np.array(sel), L=L, H=H)

        # 3) 更新 miv_stats.json（原 MIV 选择备份）
        mp = os.path.join(MV, f"{name}_miv_stats.json")
        mst = json.load(open(mp, encoding="utf-8"))
        mst["selected_orig_miv"] = mst["selected"]
        mst["n_selected_orig_miv"] = mst["n_selected"]
        mst["selected"] = sel
        mst["n_selected"] = len(sel)
        mst["selection_method"] = "MIV+domain(保留气象/负荷/关键日历，剔除明确冗余)"
        mst["dropped_by_domain"] = drop
        json.dump(mst, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  selected: {sel}")
        print(f"  已更新 {os.path.relpath(mp, BASE)}")

    print("\n重选完成。后续需重跑阶段四（重建完整模型 preds）与阶段六。")
    print("注意：stage3 原 MIV 结论仍保留于 selected_orig_miv 字段，供消融报告对照。")


if __name__ == "__main__":
    main()
