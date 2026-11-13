# 阶段5-入口 闸门校验报告

> 生成：2026-11-12 16:42:51 ｜ 结论：**PASS(带告警)**

| 维度 | 检查项 | 结论 | 证据 |
|---|---|---|---|
| 必要性 | 阶段五不可省（表 8 要求全部模型 TE-RIME 优化） | ✅ PASS | 现有 4 行对比结果的超参组合数=1（=1 表示全部沿用同一套默认超参，未调参） |
| 结果合理性 | 现有对比指标有限、无 NaN/Inf | ✅ PASS | 异常行=无 |
| 结果合理性 | test/val MAPE 比 ≤1.5（无严重过拟合/漂移） | ⚠️ WARN | [('BiLSTM', 1.077), ('BiLSTM-Transformer', 1.75), ('BiLSTM-Informer', 1.434), ('本文_BiLSTM-iTransformer', 1.059)] |
| 下游支撑力 | 模型支持 cfg 字典注入（搜索需按超参向量重建） | ✅ PASS | params=85956, out=[128, 4], backend=cuda |
| 下游支撑力 | 5 类对比模型 + 门控融合分支可前向/反向 | ✅ PASS | [('bilstm', True), ('transformer', True), ('informer', True), ('itransformer_only', True), ('mean_head', True), ('gate', True)] |
| 下游支撑力 | GPU 可用（搜索预算内可完成数百次训练） | ✅ PASS | backend=cuda |
| 实际价值 | 论点差距可量化（本文落后幅度 = 待回收空间） | ✅ PASS | test MAPE 本文 2.12% vs BiLSTM 1.95%（差 +0.18pp） |
| 实际价值 | 调参器已就绪（tuner.py 三级协议 + 指纹缓存） | ✅ PASS | tuning 目录=存在 |

**汇总**：8 项检查，FAIL=0，WARN=1，结论 **PASS(带告警)**。
