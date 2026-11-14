# 阶段五 优化算法对比 —— 荆门市 2024 / 本文_BiLSTM-iTransformer

> 生成：2026-11-13 16:00:47 ｜ 同空间、同初始种群、同评估口径、同预算

> 算法版本：stage5-tev4 ｜ 空间指纹：028bf33c3cd5 ｜ 缓存：各算法独立（同等初始快照）

| 算法 | 最优验证 MAPE(%) | 相对最优 | 唯一评估 | 达1.01×评估 | 达1.02×评估 | 曲线AUC | 新训练数 | 训练耗时(s) | 最优超参 |
|---|---|---|---|---|---|---|---|---|---|
| TE-RIME | 2.2044 | +0.00% | 100 | 35 | 12 | 0.941 | 100 | 5021 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=256, dropout=0.3, lstm_layers=2, lr=0.002, weight_decay=0.0, fusion=serial |
| PSO | 2.2419 | +1.70% | 100 | 100 | 16 | 0.970 | 100 | 3141 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.3, lstm_layers=2, lr=0.003, weight_decay=1e-05, fusion=serial |
| RIME | 2.2439 | +1.79% | 100 | 100 | 12 | 0.977 | 100 | 3305 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.3, lstm_layers=2, lr=0.003, weight_decay=0.0, fusion=serial |
| BWO | 2.2539 | +2.25% | 100 | 100 | 100 | 0.704 | 100 | 3017 | hidden=128, d_model=96, n_heads=4, e_layers=1, d_ff=192, dropout=0.3, lstm_layers=1, lr=0.003, weight_decay=0.0, fusion=serial |
| GWO | 2.4214 | +9.84% | 100 | 100 | 100 | 0.737 | 100 | 4224 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=256, dropout=0.3, lstm_layers=2, lr=0.0015, weight_decay=0.0, fusion=serial |

> 说明：`达1.01×/1.02×评估` 为首次达到「全局最优 ×1.01 / ×1.02」所需唯一评估数；`曲线AUC` 为 best-so-far 归一化面积（越小=越早收敛）；`新训练数/训练耗时` 为真实发生的训练，排除缓存命中，用于等算力比较。
