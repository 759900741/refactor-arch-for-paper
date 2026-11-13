# 阶段3 闸门校验报告

> 生成：2026-11-11 16:30:45 ｜ 结论：**PASS**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 必要性 | 阶段三 MIV 产物齐备 | ✅ PASS | 全部存在 |
| 结果合理性 | MIV 向量长度=F、非负、无NaN | ✅ PASS | len=29, min=3.31e-05 |
| 结果合理性 | α∈[0,1] 且 θ 有限 | ✅ PASS | α=0.517, θ=0.00040 |
| 结果合理性 | θ 落在 Miv_pca 区间内（排除目标列统计） | ✅ PASS | [0.00003,0.05592], θ=0.00040 |
| 下游支撑力 | 1≤Fsel<F（确有筛选） | ✅ PASS | Fsel=16/29 |
| 下游支撑力 | 目标列在选中集内 | ✅ PASS | target=负荷 |
| 下游支撑力 | Xsel 形状(N,L,Fsel)与列名一致、无NaN | ✅ PASS | shape_ok=True, cols_ok=True, NaN=0 |
| 实际价值 | 承载网络 val RMSE(缩放域)<0.2 | ✅ PASS | val_rmse_pca=0.0788 |
| 实际价值 | 选中特征 MIV 均值 > 未选中 | ✅ PASS | gap=0.00387 |
| 必要性 | 阶段三 MIV 产物齐备 | ✅ PASS | 全部存在 |
| 结果合理性 | MIV 向量长度=F、非负、无NaN | ✅ PASS | len=27, min=8.31e-05 |
| 结果合理性 | α∈[0,1] 且 θ 有限 | ✅ PASS | α=0.429, θ=0.00085 |
| 结果合理性 | θ 落在 Miv_pca 区间内（排除目标列统计） | ✅ PASS | [0.00008,0.02781], θ=0.00085 |
| 下游支撑力 | 1≤Fsel<F（确有筛选） | ✅ PASS | Fsel=17/27 |
| 下游支撑力 | 目标列在选中集内 | ✅ PASS | target=electricity(KW) |
| 下游支撑力 | Xsel 形状(N,L,Fsel)与列名一致、无NaN | ✅ PASS | shape_ok=True, cols_ok=True, NaN=0 |
| 实际价值 | 承载网络 val RMSE(缩放域)<0.2 | ✅ PASS | val_rmse_pca=0.0406 |
| 实际价值 | 选中特征 MIV 均值 > 未选中 | ✅ PASS | gap=0.00255 |

**汇总**：18 项检查，FAIL=0，WARN=0，结论 **PASS**。
