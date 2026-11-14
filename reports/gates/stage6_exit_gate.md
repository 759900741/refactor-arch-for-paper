# 阶段6-出口 闸门校验报告

> 生成：2026-11-14 13:25:35 ｜ 结论：**PASS**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 实际价值 | 完整 MIV 特征工程(A2)为最优 | ✅ PASS | 最优=A2_MIV完整 |
| 结果合理性 | 仅 RPCA-L 输入(A1)劣于无特征工程(A0) | ✅ PASS | A0=1.91% vs A1=5.12% |
| 实际价值 | 完整架构(B1:目标token+BiLSTM+变量注意力+残差+融合)为最优 | ✅ PASS | 最优=B1_完整 |
| 下游支撑力 | 融合方式对调消融(B6)存在（串行 vs 并行门控） | ✅ PASS | B6=B6_并行门控融合；MAPE 2.92% vs B1 1.72% |
| 实际价值 | 本文模型对比 BiLSTM/Transformer/Informer 最优 | ✅ PASS | 最优=本文_BiLSTM-iTransformer（BiLSTM=2.36%；BiLSTM-Transformer=3.80%；BiLSTM-Informer=3.49%；本文_BiLSTM-iTransformer=1.72%） |
| 结果合理性 | 对比各行均为调参后配置（含配置指纹） | ✅ PASS | 参数：BiLSTM=664k；BiLSTM-Transformer=803k；BiLSTM-Informer=803k；本文_BiLSTM-iTransformer=730k |
| 实际价值 | 兰州(异源异粒度)本文模型工程可用(MAPE≤5%) | ✅ PASS | 本文 MAPE=2.60%（辅助可行性参考，不参与主要对比结论） |
| 结果合理性 | A2/B1/compare本文 使用同一最终配置 | ✅ PASS | 指纹=['b9b7b9a2e975'] |

**汇总**：8 项检查，FAIL=0，WARN=0，结论 **PASS**。
