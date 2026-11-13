# 阶段5-出口 闸门校验报告

> 生成：2026-11-13 01:19:35 ｜ 结论：**PASS(带告警)**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 必要性 | 5 种优化算法结果齐备（表 5） | ✅ PASS | 5/5（key=bilstm_itransformer） |
| 结果合理性 | 5 种算法等预算（同 pop×iters） | ✅ PASS | 预算=[100] |
| 结果合理性 | best-so-far 单调不增且为有限值 | ✅ PASS | 5 条曲线 |
| 实际价值 | TE-RIME 为 5 种算法中最优 | ⚠️ WARN | 最优=pso(2.2477%)，TE-RIME=2.2477% |
| 结果合理性 | 定稿经 full 协议复验（记录 mean/std） | ✅ PASS | full_mean=1.6189, std=0.0242 |
| 下游支撑力 | TE-RIME 最优 cfg 含 model_type/fusion | ✅ PASS | mt=bilstm_itransformer, fusion=serial |
| 实际价值 | 定稿仲裁结论已记录 | ✅ PASS | reports/stage5_decision.md |

**汇总**：7 项检查，FAIL=0，WARN=1，结论 **PASS(带告警)**。
