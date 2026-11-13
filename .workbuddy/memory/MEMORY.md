# MEMORY.md — 项目长期约定

## 项目
《基于 iTransformer-BiLSTM-TE-RIME 模型的 X 区短期电力负荷预测研究》
技术路线：RPCA-MIV 特征工程 → BiLSTM-iTransformer 预测 → TE-RIME 超参数优化。
数据：荆门 2024（15min，主实验）/ 兰州 2023（1h，跨场景泛化）。

## 强制工作流：阶段闸门校验（Stage-Gate）——用户明确要求
每个阶段都必须先校验、后进入下一阶段，且每阶段输出要简要写校验结论。四个维度：
1. **必要性**：该阶段是否为下游所必需，能否省。
2. **结果合理性**：数据/指标是否自洽、无泄漏、量纲正确、无 NaN。
3. **下游支撑力**：产物格式/维度/字段能否被下一阶段直接消费。
4. **实际价值**：是否带来可量化收益（如特征相关性、降噪比例、精度提升）。
规则：
- 任一 FAIL → 明确列出问题 + 调整建议，**不得进入下一阶段**。
- 全部 PASS（允许 WARN）→ 才可进入下一阶段。
- 每阶段结束输出「校验结论（PASS/FAIL + 关键证据）」。
- 校验尽量自动化：统一入口 `src/stage_gate.py`，报告写 `reports/gates/stageN_gate.md`。

## 已确认的阶段一设计决策
- 决策1：加跨日/跨周滞后特征 lag_1d / lag_1w（可配置开关 USE_LAGS）。
- 决策2：is_workday/is_holiday 合成 day_type 三态（0周末/1工作日/2节假日），保留原列。
- 决策3：兰州 cold/heat 保留，标注共线，交阶段二 MIV 裁决。
- 决策4：兰州 6 月高峰原样保留（真实极端）。
- 决策5：归一化 MinMax[0,1]，仅训练集 fit。
- 预测尺度：输入过去 24h，输出未来 1h（荆门 96→4 步，stride=4；兰州 24→1 步，stride=1）。
- 划分：时间顺序 8:1:1，先划分后滑窗。
- 异常策略：荆门全局3σ+差分；兰州 Hampel(win=25,k=3)。

## 已确认的阶段三（MIV）设计决策
- 环境：Python 3.8 + torch 2.4.1+cu121（CUDA，RTX 4060 Laptop 8GB）；gate/校验脚本用 3.13 跑（3.8 pandas 损坏）。device 开关：trainer.resolve_device()，默认 auto→有 CUDA 用 cuda，环境变量 WB_DEVICE 可覆盖。
- 承载网络：轻量 BiLSTM-iTransformer（hidden=d_model=32, 20 epochs, MAX_TRAIN=3000, δ=0.1, K=800）。
- 模型头：取目标变量 token（target_idx=0）输出，不做变量维均值池化；BiLSTM 末步+均值拼接。
- θ 口径：排除目标列后按论文 Eq.12 原式算 α/θ；目标列按 Eq.14 强制保留（其 MIV 高第二名约 60 倍）。
- MIV 缓存：`miv/{name}_miv_cache.npz`，重调阈值免重训。
- 结果：荆门 16/29，兰州 17/27；lag_1d/lag_1w 双双被剔除（信息与窗口内负荷列冗余，阶段六消融验证）。
- 产物：`miv/{name}_{split}_sel.npz`（RPCA-L 基，下游主用）+ `_selorig.npz` + `{name}_miv_stats.json`。

## 已确认的阶段四（基线模型）设计决策
- 新工作流：每阶段先「入口校验」（--stage 4）→ 实现+论文图 →「出口校验」（--stage 4exit）。
- **主输入=原始特征+MIV 选中列（selorig），不用 RPCA-L**：L 作输入精度减半（荆门 55 vs 135），RPCA 角色改为异常定位+MIV 第二视角，L 输入留作阶段六消融。
- 模型加 residual 开关：ŷ=y_t+Δ̂（阶段四基线 True，阶段三承载网络 False）。
- 基线超参（trainer.DEFAULT_CFG）：hidden32/d_model32/1层2头/lr1e-3/wd0/60ep/早停12/batch128。
- 结果：荆门 test 56.1 MW/MAPE 2.28%/R²0.935（超基线+23.9%）；兰州 551.6 kW/2.72%/0.842（+2.3%，测试期为低负荷窄带）。
- MIV 两视角秩相关>0.99，筛选结论稳健。
- 训练器 src/trainer.py 为阶段四/五共用接口（fit_model 按 cfg 字典，供 TE-RIME 调用）。

## 目录约定
- 代码 `src/`；中间数据 `data/processed/`；报告 `reports/`；阶段规格/方案在项目根目录 `*.md`。
- 中文输出；图表需设中文字体（Microsoft YaHei/SimHei）。
