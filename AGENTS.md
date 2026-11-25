# AGENTS.md —— 仓库契约（写给所有参与的人与 Agent）

> 本文件是**硬规则**，违反会被 CI 或 pre-commit 拦下。
> 计划依据：`docs/IMPLEMENTATION_PLAN.md` §9.2、§12、§15。

## 1. 模块边界（不可越界）

| 规则 | 原因 | 检查方式 |
|---|---|---|
| `features/*` 不读写文件、不解析配置、不打印日志 | 保证可单测、可复用 | code review + 单测 |
| `features/severity/` 不 import 任何模型代码 | 物理量算必须能脱离模型独立验证 | `tests/unit/test_thickness.py` 无 torch 依赖 |
| 阈值一律走 `configs/`，`features/` 内禁止魔法数字 | 业务阈值必须可审计、可回滚 | code review |
| `service/` 不 import `training/` | 训练依赖不得进入生产镜像 | code review |
| 跨模块传递必须用 `common/schemas.py` 的 Pydantic 对象 | 契约变更即报错 | `tests/unit/test_schemas.py` |
| 只有 `scripts/prelabel.py` 可 import `ultralytics` | AGPL-3.0 传染风险 | `tests/unit/test_license_guard.py` |

依赖方向：`common → data → features → training → evaluation → application → service`，禁止反向。

## 2. 数据红线

1. **分组键 = `site_id | camera_id | weather_event_id`**，同一 group 不得跨 split。
2. **合成数据只进 train**，`test` 中 `is_synthetic` 必须全为 false。
3. **许可未确认的资源只能进 `data/external/`**（只读、不参与训练）。
4. 每次训练必须绑定一个 manifest；manifest 每条带 `sha256`。
5. 改动划分逻辑后必须跑 `scripts/validate_dataset.py`。

## 3. 指标口径

- 三栏制：**硬门槛 / 软目标 / 文献参考**。文献值只能作对照，写进报告时必须标注"不可比"。
- 覆冰厚度 MAPE 只在 `gt ≥ 1 mm` 样本上计算；0 厚度样本必须剔除并单独报数量。
- 没有 Set-D（现场精标集）之前，**禁止**在任何材料中写"可用于生产"。

## 4. 目录与命名

- 包名 `snake_case`；仓库名 `kebab-case`；目录嵌套 ≤ 4 层。
- 根目录禁止 `train.py` / `test.py` / `utils.py` 等裸脚本；入口统一 `src/grid_ice_vision/cli.py`。
- 禁止 `temp/`、`runs/`、`exp2_final/`、`old_code/` 等噪音目录；实验产物写 `artifacts/`（gitignore）。
- 依赖单一事实来源：`pyproject.toml`。**禁止**同时存在 `setup.py` / `requirements.txt` / `poetry.lock`。
- 权重只放 `artifacts/<task>/<version>/`，不进 `src/`。

## 5. 提交前

```bash
make check     # ruff + mypy + pytest
```

- 新增业务阈值 → 同时改 `configs/` 与文档，并在 PR 描述里说明依据。
- 新增字段到契约 → 给默认值；删改字段 → 升主版本并保留一个版本的兼容解析。
- 修改 `features/severity/` → 必须同时补/改单测（恒等、零、单调、密度四条不可删）。

## 6. 已知取舍（不要"顺手优化"掉）

- **不用 Ultralytics 进生产**：多花 3–5 天自建 GFL 头，换掉 AGPL 风险。
- **不做端到端多任务共享骨干**：小样本下负迁移风险 > 收益，留作 P4 消融。
- **不承诺 INT8**：量化对覆冰边缘 IoU 的损伤未实测，先 FP16。
- **不做 LRP**：运维人员看不懂；卡尺叠加图才是有效解释。
- **不用伪标签**：无现场数据时伪标签无意义。
