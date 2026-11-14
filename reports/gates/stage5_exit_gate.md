# 阶段5-出口 闸门校验报告

> 生成：2026-11-13 16:28:34 ｜ 结论：**PASS**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 必要性 | 5 种优化算法结果齐备（表 5） | ✅ PASS | 5/5（key=bilstm_itransformer） |
| 结果合理性 | 5 种算法等预算（同 pop×iters） | ✅ PASS | 预算=[100] |
| 结果合理性 | 5 种算法算法版本一致且为当前版本 | ✅ PASS | 版本=['stage5-tev4']，期望=stage5-tev4 |
| 下游支撑力 | 5 种算法搜索空间指纹一致（同空间可比） | ✅ PASS | space_hash=['028bf33c3cd5'] |
| 结果合理性 | 5 种算法共享同一初始种群（公平性实证） | ✅ PASS | 初始种群形状=(5, 9) |
| 结果合理性 | 台账自洽：缓存命中+新训练=评估数=预算 | ✅ PASS | pso:0+100=100; bwo:0+100=100; gwo:0+100=100; rime:0+100=100; terime:0+100=100 |
| 结果合理性 | 5 种算法均为独立冷缓存（同等起跑线） | ✅ PASS | isolate=['jingmen_bilstm_itransformer_bwo_s42', 'jingmen_bilstm_itransformer_gwo_s42', 'jingmen_bilstm_itransformer_pso_s42', 'jingmen_bilstm_itransformer_rime_s42', 'jingmen_bilstm_itransformer_terime_s42'] |
| 结果合理性 | best-so-far 单调不增且为有限值 | ✅ PASS | 5 条曲线 |
| 实际价值 | TE-RIME 为 5 种算法中最优 | ✅ PASS | 最优=terime(2.2044%)，TE-RIME=2.2044% |
| 结果合理性 | 定稿经 full 协议复验（记录 mean/std） | ✅ PASS | full_mean=1.7064, std=0.0281 |
| 下游支撑力 | TE-RIME 最优 cfg 含 model_type/fusion | ✅ PASS | mt=bilstm_itransformer, fusion=serial |
| 实际价值 | 定稿仲裁结论已记录 | ✅ PASS | reports/stage5_decision.md |

**汇总**：12 项检查，FAIL=0，WARN=0，结论 **PASS**。
