# ADR-0004：SCUT 纯 CPU 覆冰类型末端选择性微调回退与 Shadow 联调治理决策

- **状态**：Accepted（2026-11-25）
- **关联文档**：`ADR-0003-ice-type-head-gating.md`、`docs/cpu_ice_type_baseline.md`、`plans/电网覆冰类型识别_纯CPU代码改进-实现计划.md`

## 1. 背景

根据 `ADR-0003` 与 `电网覆冰类型识别_纯CPU代码改进-实现计划`，我们在纯 CPU（`Ryzen 7 4800U`）环境下基于干净审计后的 `SCUT` 7,244 张有效图像（排除 18 个噪声候选后 train/val/test = `4,964 / 745 / 1,517`），依次验证了：
1. P1 传统 390 维手工特征基线（Linear/RBF SVM）；
2. P2 冻结 `MobileNetV3-Small`（`576` 维）主线及温度标定/三重拒识；
3. P3 终端归一化 ROI 双视角（`1152` 维）消融及“仅解冻最后 1 个阶段（56,448 参数）”的端到端分层微调（`1e-5 / 3e-4`）；
4. P4 纯生产 ONNX Runtime 部署链路（`none_rule / shadow / model` 三态受控策略）。

## 2. 决策内容

### 决策 A：正式回退 P3 末端选择性微调，采用冻结 `MobileNetV3-Small + RBF SVM (C=0.1)` 作为主产物 `v1`
实测对比结果显示，选择性末端微调（`v1_p3_finetune`，CPU 训练 562.38 s，6 轮早停）**不满足晋级条件**：
- 在 `Monitor-Holdout test (1,517)` 上：微调 `Macro-F1 = 0.5478`（`BAcc = 0.5512`），**低于**冻结基线 `v1_p2_rbf`（`Macro-F1 = 0.5539`、`BAcc = 0.5627`），且 `glaze->rime`（`23.22%`）恶化于冻结线（`17.76%`）；
- 在 `Temporal-Holdout test (468, 2018+ 跨年份)` 上：微调模型出现严重跨期退化，`Macro-F1` 骤降至 **`0.4800`**（`BAcc = 0.5336`、`wet_snow` 召回跌至 `30.34%`），而冻结 `v1_p2_rbf` 保持 `0.5985`（`BAcc = 0.6496`），冻结 `v1_p2_logreg` 与 `v1_p3_dual_rbf` 分别达 `0.6376` 与 `0.6442`。
- 依据计划 §3.5 明确回退规则，**不采用微调权重**，将冻结特征链路 `v1_p2_rbf` 晋升为主产物 `artifacts/ice_type/v1` 及 `artifacts/onnx/ice_type.onnx`。

### 决策 B：上线模式严格锁定为 `none_rule`（默認）与可选只读 `shadow`，禁开自动密度折减 `model`
- 虽然 `v1` 达到了 P2 阶段门（`Macro-F1 = 0.5539 >= 0.55`、`BAcc = 0.5627 >= 0.55`、`ECE = 0.0617 <= 0.15`）、双向 `glaze <-> rime` 控制在 `17.76% / 16.84% (<= 20%)`、且 ONNX 导出误差 `9.83e-6`、ORT 推理仅 `3.07 ms (P50)`；
- 但仍**未达到 G2 生产门限 `Macro-F1 >= 0.70`**，且在 2018 跨年/特定局站上存在显著域漂移与 `mixed -> rime (54.76%)` 系统性混淆（具体局站属客户敏感信息，不在此列出）；
- 因此：
  1. `configs/model/pipeline.yaml` 默认保持 `severity.ice_type.mode: none_rule`、`weights.ice_type: null`；
  2. 工程侧在 `RhoPolicy` 与 `InferencePipeline` 中实现受控三态：`none_rule`（零额外算力） / `shadow`（运行 `ice_type.onnx` 填充 `ice_type`、`class_probabilities`、`requires_review` 并写入审计字段，**但 `RhoPolicy` 强制返回标准密度 `900.0 kg/m³`、`rho_source="standard"`**） / `model`（仅当 `not requires_review` 时允许折减密度）；
  3. 永久保持 `heuristic` 模式抛出禁用异常；任何开启 `model` 模式的变更均需未来在现场 `Set-D` / 重标定集上通过 `G2 >= 0.70` 后由新的 ADR 批准。

## 3. 后果与后续工作
- 工程闭环与纯 CPU 推理路径已就绪，不会误伤现网等值冰厚计算（不存在 58% 等值厚度虚假缩水风险）；
- 下阶段精度突破路径已明确指向 **ROI 标注质检 + 微气象特征晚期融合**，而非无额外信息下的更深层微调。
