# grid-ice-vision

面向**输电线路固定监控与无人机巡检图像**的覆冰 / 降水视觉识别系统。

输出三类结构化结果：**降水相态（视觉）** + **部件覆冰状态** + **覆冰程度（含物理量纲）**，
以统一 JSON 契约接入电网运维平台。

> 本仓库是 `IMPLEMENTATION_PLAN.md`（合并版 v1.0）的代码落地。
> 计划文档在上级目录，本文件只描述**怎么跑**与**边界在哪**。

---

## 1. 先说清楚不做什么

这一节比后面的技术细节都重要。任何对外交付都必须带上这几条。

| # | 不承诺 | 原因 |
|---|---|---|
| 1 | 不承诺从单张 RGB 判定「正在下冻雨」 | 冻雨与普通雨在可见光下表观高度相似；系统只输出「视觉相态」与「疑似 glaze 证据」 |
| 2 | 不承诺无标定条件下的绝对冰厚 | 无尺度参考时 `d_eq_mm = null`，只给覆盖率与视觉等级 |
| 3 | 不承诺冰雹类别的自动化精度 | 样本近乎不可获取；策略为「检测到即告警 + 人工确认」 |
| 4 | 不承诺夜间 / 红外性能 | 无夜间标注数据；夜间 `quality.status` 直接置 `unusable` |
| 5 | 不用图像模型替代运行管理判断 | I–IV 级预警必须人工复核后才可下发处置指令 |
| 6 | MVP 阶段不做端到端多任务共享骨干 | 小样本下负迁移风险高于收益，留作 P4 消融 |
| 7 | 不承诺「无现场数据也能达到生产精度」 | 公开数据只能建立方法与基线；生产精度以现场数据到位为前提 |

**没有 Set-D（现场精标集）之前，不得宣称「可用于生产」。**

---

## 2. 安装

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,serve,deploy]"
```

Linux / macOS 把 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。

## 3. 跑起来

```bash
# 单元测试（< 10 s，不需要任何数据与权重）
uv run pytest tests/unit -q

# 全量检查（ruff + mypy + 全部单测）
make check

# 合成数据自检：验证几何量算链路（不需要真实数据）
uv run python scripts/demo_synthetic.py

# 单图推理（需要权重；无权重时用 --random-weights 只验契约）
uv run giv infer --image samples/demo.jpg --config configs/model/pipeline.yaml

# 启动服务
uv run giv serve --config configs/model/pipeline.yaml --port 8000
```

## 4. 目录约定

```
configs/     所有阈值与超参（features/ 内禁止魔法数字）
src/grid_ice_vision/
  common/        契约（Pydantic）、配置、日志、常量
  data/          manifest、分组划分、Dataset、增强、采样器
  features/      quality | precipitation | detection | ice | severity  ← 只做算法
  training/      引擎、损失、回调、checkpoint
  evaluation/    分类 / 检测 / 分割 / 厚度指标
  application/   pipeline 编排、EKF、预警规则、证据图
  deployment/    ONNX 导出、运行时、benchmark
  service/       FastAPI
scripts/      数据转换、合成、校验、导出
docs/         架构、标注规范、等级 rubric、许可台账、ADR、模型卡
tests/        unit（每次提交）/ integration（每次 PR）/ e2e（每日回归）
```

硬规则见 `AGENTS.md`，违反会被 CI 拦下。

## 5. 模块边界（摘要）

- `features/severity/` **不 import 任何模型代码**，只接受 mask + 标定参数 —— 物理量算必须能脱离模型独立测试。
- `features/*` 不读写文件、不解析配置、不打印日志。
- 所有跨模块传递使用 `common/schemas.py` 的 Pydantic 对象。
- 阈值全部走 `configs/`，业务阈值必须可审计、可回滚。

## 6. 许可与合规

- 生产链路**不使用** Ultralytics（AGPL-3.0）。`scripts/prelabel.py` 是唯一允许 import ultralytics 的地方，其产物只进人工修订流程，不产生生产权重。
- 检测器为 `timm backbone + 自建 GFL 头`（Apache-2.0 / MIT 依赖链）。
- 外部数据集的许可状态见 `docs/data_license_ledger.md`；**许可未确认的资源只能进 `data/external/`（只读、不参与训练）**。

## 7. 版本

- `0.1.0` — P0–P4 工程闭环（方法 + 合成验证）。生产精度未验证。
