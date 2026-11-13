# 阶段五 优化算法对比 —— 荆门市 2024 / 本文_BiLSTM-iTransformer

> 生成：2026-11-13 00:14:10 ｜ 同空间、同评估口径、同预算

| 算法 | 最优验证 MAPE(%) | 相对最优 | 唯一评估次数 | 达到 1.02×最优 所需评估 | 耗时(s) | 最优超参 |
|---|---|---|---|---|---|---|
| PSO | 2.2477 | +0.00% | 100 | 10 | 761 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.2, lstm_layers=2, lr=0.003, weight_decay=0.0, fusion=serial |
| RIME | 2.2477 | +0.00% | 100 | 10 | 170 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.2, lstm_layers=2, lr=0.003, weight_decay=0.0, fusion=serial |
| TE-RIME | 2.2477 | +0.00% | 100 | 15 | 2952 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.2, lstm_layers=2, lr=0.003, weight_decay=0.0, fusion=serial |
| BWO | 2.2600 | +0.55% | 100 | 30 | 2305 | hidden=128, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.1, lstm_layers=2, lr=0.003, weight_decay=0.0, fusion=serial |
| GWO | 2.5710 | +14.38% | 100 | 100 | 360 | hidden=96, d_model=64, n_heads=4, e_layers=1, d_ff=128, dropout=0.1, lstm_layers=1, lr=0.001, weight_decay=1e-05, fusion=serial |
