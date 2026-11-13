# -*- coding: utf-8 -*-
"""
probe_input_variant.py（诊断脚本，非正式流水线）
================================================
对照实验：同一模型/超参下，输入用 RPCA 低秩矩阵 L（sel）还是原始矩阵（selorig），
精度差异有多大。用于决定阶段四主输入，并为阶段六消融提供证据。

运行：cd src && "<py3.8>" probe_input_variant.py > ../reports/probe_input_variant.log 2>&1
"""
import numpy as np
import trainer as T

CFG = dict(hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1,
           lr=1e-3, weight_decay=0.0, epochs=40, patience=10, batch=128)


def main():
    print("variant 说明: sel=RPCA-L 输入, selorig=原始输入（同样的 MIV 选中列）", flush=True)
    for n in ("lanzhou", "jingmen"):
        lo, hi = T.load_y_range(n)
        for v in ("sel", "selorig"):
            d = T.load_split(n, v)
            m, h, best = T.fit_model(d, CFG, log=None)
            pt = T.predict(m, d["test"]["X"]); pv = T.predict(m, d["val"]["X"])
            mt = T.metrics(T.inverse(d["test"]["y"], lo, hi), T.inverse(pt, lo, hi))
            mv = T.metrics(T.inverse(d["val"]["y"], lo, hi), T.inverse(pv, lo, hi))
            tp = np.repeat(d["test"]["X"][:, -1, 0:1], d["H"], axis=1)
            mb = T.metrics(T.inverse(d["test"]["y"], lo, hi), T.inverse(tp, lo, hi))
            print(f"{n:8s} {v:8s}: val_rmse={best:.5f} val_R2={mv['r2']:.3f} | "
                  f"test RMSE={mt['rmse']:7.1f} MAPE={mt['mape']:5.2f}% R2={mt['r2']:6.3f} | "
                  f"基线 RMSE={mb['rmse']:7.1f} R2={mb['r2']:6.3f} | best_ep={h['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
