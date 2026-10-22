# ADR-0003：覆冰类型头（IceTypeGradeNet）冻结，不接线

- 状态：已采纳
- 日期：2026-10-20
- 相关：§10.5、`features/severity/ice_type.py`、`tests/unit/test_severity_ice_type.py`

## 背景

`IceTypeGradeNet`（`features/ice/model.py`）已实现：ROI 图像 + 4 类掩码 →
类型 5 类 + 视觉等级 CORAL。但当前 pipeline **不引用**它，`resolve_ice_type`
只做一条规则：`G0 → none`，其余 `→ unknown`（不猜），密度一律取 `RHO_STANDARD = 900`。

能不能现在就接线替换粗判？三个硬约束给出否定答案：

1. **没有 `ice_type` 标注数据**。随机初始化的 5 类头在 rime/glaze/mixed 之间近似均匀，
   约 2/5 概率输出 `rime`；`rime` 的密度是 350，会让等值冰厚缩水 **57.6%**
   （d0=30 mm、径向 5 mm：5.00 → 2.12 mm）。`d_eq` 直接进 I–IV 预警矩阵，
   缩水会**跨级下调预警**——这是最危险的方向。
2. **训练通路尚不存在**。`data/datasets/` 只有 detection / segmentation / weather，
   没有类型数据集类；`training/steps.py` 没有类型/等级 step；`evaluation/` 没有类型评估；
   全仓库没有 `type_logits` / `grade_logits` 的消费者。
3. **契约缺口**。`rho_source` 只能记录"用了哪个密度"（standard / config_prior /
   ice_type_model），不能记录"这个密度来自一个未验证的模型"。审计链不完整时，
   事后无法区分"密度是算出来的"还是"猜出来的"。

## 决策

### 1. 冻结

- `InferencePipeline` 继续使用 `resolve_ice_type`，`rho_source` 只可能是 `standard`；
- `RhoPolicy.mode` 只允许 `none_rule`，传 `heuristic` / `model` 直接 `ValueError`；
- 防回归测试同时检查**行为**与**配置**：`configs/model/pipeline.yaml` 的
  `severity.ice_type.mode` 必须是 `none_rule`；
- 现阶段**不预埋** `ice_type_gate` 字段、不新增数据集类与训练 step（等数据到位再做，
  避免写出没有被执行过的代码路径）。

### 2. 开闸条件（全部满足才允许接线）

| # | 条件 | 阈值 | 检查方式 |
|---|---|---|---|
| G1 | 数据量 | ≥200 个 ROI 带 `ice_type` 标注；参与折算的每类 ≥50（不足按 `taxonomy.yaml` 的 `merge_rule` 并入 `mixed`） | manifest 统计 |
| G2 | 类型精度 | 类型 macro-F1 ≥ 0.70；ECE ≤ 0.15 | 留出集评估脚本 |
| G3 | 关键混淆 | `glaze ↔ rime` 双向误判率 ≤ 20% | 混淆矩阵 |
| G4 | 契约 | 输出 `rho_source=ice_type_model` + 置信门禁与回退原因字段 | schema 测试 |
| G5 | 回退 | 置信低于门限或类型为 `unknown` 时回退 900，不得使用模型密度 | 单测 |
| G6 | 预警 | 类型引起的等级迁移必须单独出灵敏度报告（列出跨级样本） | 报告 |
| G7 | 单调性 | 同一样本，`rho_ice_type=350` 与 `900` 的 `d_eq` 必须满足单调关系，且不得被记成 `measured` 之外的静默修正 | 单测 |

### 3. 接线顺序（满足 G1–G7 之后）

1. 先离线：只跑类型头，不接密度，比较 macro-F1 / ECE；
2. 再影子模式：并行输出 `ice_type` 但厚度仍用 900，累计 2 周对比；
3. 最后生效：`mode: model` + 门禁字段，预警等级变化需人工复核标记。

## 代价

- 类型恒为 `unknown`，暂时用不上 glaze / rime 的密度差异；
- 但在**安全方向**上是有利的：`unknown` 取 900（最大密度），对 rime 事件会把等值冰厚
  算**偏厚**，属于倾向多发预警而非漏报。反过来若误判为 rime，则会少报 57.6%。

## 收益

- 不用未验证的密度驱动除冰决策；
- 审计链只保留可证明的来源；
- 接线前有可执行、可复现的门槛，而不是"感觉差不多就上"。

## 备选方案与否决理由

| 方案 | 否决理由 |
|---|---|
| 现在直接接线替换粗判 | 随机权重 + ρ=350 使 `d_eq` 缩水 57.6%，跨级改变预警，且无法审计 |
| 用启发式（亮度/纹理阈值）判类型 | 阈值无数据依据，等价于把"猜"包装成"算法"；同样缺 G1–G3 的验证 |
| 先接 CORAL 等级头替代覆盖率分级 | 等级头同样是随机权重；覆盖率分级已有合成验证（0.155 px），不需要替换 |
| 用 `weather` 分类结果推断 `ice_type` | 天气相态与覆冰类型不是一一映射（冻雨可致 glaze 也可致 mixed），属于跨层推断，违反标注规范 §0 |

## 复现

```bash
uv run pytest tests/unit/test_severity_ice_type.py -q
uv run python -c "from grid_ice_vision.features.severity.ice_type import RhoPolicy; RhoPolicy(mode='model')"
# 预期：ValueError: ice_type.mode='model' 尚未实现/未验证
```
