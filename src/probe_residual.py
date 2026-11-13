# -*- coding: utf-8 -*-
"""
probe_residual.py（诊断脚本）
============================
在原始输入 + MIV 选中列的基础上，对比「直接回归目标值」与「残差(增量)输出头」。
残差头：ŷ_{t+h} = y_t + Δ̂，用于抑制负荷水平整体漂移带来的系统性偏差。

运行：cd src && "<py3.8>" probe_residual.py
"""
import numpy as np
import trainer as T

BASE = dict(hidden=32, d_model=32, n_heads=2, e_layers=1, d_ff=64, dropout=0.1,
            lr=1e-3, weight_decay=0.0, epochs=40, patience=10, batch=128)


def main():
    for n in ("lanzhou", "jingmen"):
        lo, hi = T.load_y_range(n)
        d = T.load_split(n, "selorig")
        tp = np.repeat(d["test"]["X"][:, -1, 0:1], d["H"], axis=1)
        mb = T.metrics(T.inverse(d["test"]["y"], lo, hi), T.inverse(tp, lo, hi))
        print(f"== {n} 持续预测基线: RMSE={mb['rmse']:.1f} MAPE={mb['mape']:.2f}% R2={mb['r2']:.4f}", flush=True)
        for tag, extra in (("直接回归", {}), ("残差头", dict(residual=True))):
            m, h, best = T.fit_model(d, {**BASE, **extra}, log=None)
            pt = T.predict(m, d["test"]["X"])
            mt = T.metrics(T.inverse(d["test"]["y"], lo, hi), T.inverse(pt, lo, hi))
            gain = (mb["rmse"] - mt["rmse"]) / mb["rmse"] * 100
            print(f"   {n} {tag}: val={best:.5f} test RMSE={mt['rmse']:7.1f} MAPE={mt['mape']:5.2f}% "
                  f"R2={mt['r2']:.4f} | 相对基线 {gain:+.1f}% | best_ep={h['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
