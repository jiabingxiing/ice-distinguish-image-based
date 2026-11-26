# docs/ 索引

远程仓库：<https://github.com/jiabingxiing/ice-distinguish-image-based>（公开，默认分支 `main`）。

本目录承载计划、调研与验证记录。代码本身见仓库根 `src/`。

## 计划与需求（自上而下）

| 文档 | 内容 | 状态 |
|---|---|---|
| `IMPLEMENTATION_PLAN.md` | 合并版实现计划 v1.0（架构、里程碑、验收口径） | 现行基准 |
| `电网覆冰智能识别_v2.0-实现计划.md` | v2.0 实现计划（在 v1.0 基础上迭代） | 现行 |
| `电网覆冰类型识别_纯CPU代码改进-实现计划.md` | 纯 CPU 冰型识别改进实施计划（开源调研、架构、路线图） | 现行 |
| `1125 思考.md` | 纯 CPU 路线的前期思考与数据观察记录 | 参考 |
| `数据可行性需求调研报告_v1.0.md` | 数据可得性、许可与体量调研 | 结论已并入计划 |
| `数据需求清单_v1.0.xlsx` | 逐项数据字段与来源清单 | 配套表格 |
| `current-alternative-plans/plan1..5.md` | 早期候选方案（已归档，保留决策脉络） | 归档 |
| `_data_inventory_build/` | 生成上面 xlsx 的构建脚本 | 工具 |

## 技术决策与验证

| 文档 | 内容 |
|---|---|
| `architecture.md` | 分层架构与依赖方向 |
| `acceptance_targets.md` | 验收指标口径（硬门槛 / 软目标 / 文献参考） |
| `severity_rubric.md` | 覆冰程度分级标准 |
| `annotation_guide.md` | 标注规范 |
| `annotation_effort_estimate.md` | 标注工作量实测估算 |
| `decisions/ADR-*.md` | 架构决策记录 |
| `model_cards/` | 模型卡片 |

## 实测报告

| 文档 | 内容 |
|---|---|
| `cpu_fallback_assessment.md` | CPU 回退可行性与实测时延 |
| `cpu_training_budget.md` | CPU 训练预算 |
| `m3_severity_report.md` | M3 严重程度 Track A 验证 |
| `p2_smoke_report.md` | P2 检测冒烟结果 |
| `public_validation_matrix.md` | 公开数据验证矩阵与许可状态 |
| `weather_p1_report.md` | 天气侧 P1 基线 |
| `field_domain_check.md` | 现场域差检查 |
| `data_license_ledger.md` | 数据许可台账（含待确认项） |

> 说明：`data/splits/*.jsonl` 不入库（含客户台账信息），由
> `scripts/build_scut_manifest.py` 以固定 seed=42 重建，详见 `data/README.md`。