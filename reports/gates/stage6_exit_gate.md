# 阶段6-出口 闸门校验报告

> 生成：2026-11-12 14:20:23 ｜ 结论：**FAIL**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 实际价值 | 完整 MIV 特征工程(A2)为最优 | ❌ FAIL | 最优=A0_无特征工程 |
| 结果合理性 | 仅 RPCA-L 输入(A1)劣于无特征工程(A0) | ✅ PASS | A0=2.00% vs A1=5.09% |
| 实际价值 | 完整架构(B1:目标token+BiLSTM+变量注意力+残差)为最优 | ❌ FAIL | 最优=B4_BiLSTMOnly |
| 实际价值 | 本文模型对比 BiLSTM/Transformer/Informer 最优 | ❌ FAIL | 最优=BiLSTM |
| 实际价值 | 兰州(异源异粒度)本文模型仍最优 | ✅ PASS | 最优=本文_BiLSTM-iTransformer |

**汇总**：5 项检查，FAIL=3，WARN=0，结论 **FAIL**。
