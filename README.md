# grid-ice-vision

面向**输电线路固定监控与无人机巡检图像**的覆冰 / 降水视觉识别系统。

输出三类结构化结果：**降水相态（视觉）** + **部件覆冰状态** + **覆冰程度（含物理量纲）**，
以统一 JSON 契约接入电网运维平台。

> 本仓库是 `../plans/IMPLEMENTATION_PLAN.md`（合并版 v1.0）的代码落地，
> 当前实施基线为 `../plans/电网覆冰智能识别_v2.0-实现计划.md`。
> 计划文档在上级 `plans/` 目录，本文件只描述**怎么跑**与**边界在哪**。

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

> 仓库带 `uv.lock`（当前锁 torch 2.15.0 / torchvision 0.29.2 / timm 1.0.31 / numpy 2.5.5）。
> 本机遗留的 `_editable_impl_grid_ice_vision.pth` 记录的是**当前**绝对路径；
> 项目整体换盘或换目录后必须先跑 `uv run python scripts/preflight.py` 核对，否则解释器启动失败。
> 注意用 `uv run` 或 `.venv/Scripts/python.exe`，不要用裸 `python`（本机 PATH 上是 3.13 全局解释器）。

## 3. 跑起来

```bash
# 单元测试（不需要任何数据与权重）
uv run pytest tests/unit -q

# 全量检查（ruff + mypy + 全部测试）；本机没有 make，用 just
just check

# 合成数据自检：验证几何量算链路（不需要真实数据）
uv run python scripts/demo_synthetic.py

# 单图推理（`--image` 必须指向真实存在的文件；configs/model/pipeline.yaml 的 weights 默认全 null，
# 即模型用随机初始化权重，只用于验契约，不产生任何有意义的精度）
uv run giv infer --image <你的图.jpg> --config configs/model/pipeline.yaml

# 启动服务
uv run giv serve --config configs/model/pipeline.yaml --port 8000
```

> `giv infer` 目前**没有** `--random-weights` 选项，仓库内也**没有** `samples/demo.jpg`；
> 自带的合成自检是 `scripts/demo_synthetic.py`。

### 天气分类（P1，WEAPD / CC0）

```bash
# 1. 下载并解压 WEAPD（DOI 10.7910/DVN/M8JQCR，约 620 MB）
#    Windows 无 7-Zip 时可用 conda 自带的 bsdtar：bsdtar -xf weapd.rar

# 2. 建 manifest：11 类 → 8 类，丢 4 类，pHash 近重复聚类分组
uv run python scripts/build_weather_manifest.py \
    --src data/raw/weapd/dataset --out data/manifests/weather_v1.jsonl

# 3. 校验（分组泄漏 + 哈希）
uv run python scripts/validate_dataset.py --manifest data/manifests/weather_v1.jsonl

# 4. 训练（CPU 约 6 min/epoch）
uv run giv train precip --config configs/train/precip.yaml

# 5. 评估：逐类 P/R/F1 + 混淆矩阵 + ECE
uv run python scripts/eval_weather.py --config configs/train/precip.yaml --split test
```

> 预训练权重需经镜像：`configure_hf_mirror()` 已在 `train` 入口自动调用。
> 若手动跑脚本，先 `export HF_ENDPOINT=https://hf-mirror.com`。

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
docs/         架构、标注规范、等级 rubric、许可台账、ADR、模型卡、公开数据验证矩阵、CPU 训练预算
tests/        unit（每次提交）/ integration（每次 PR）/ e2e（每日回归）
```

硬规则见 `AGENTS.md`。仓库**没有远端、CI 不会自动跑**，靠提交前自己执行 `just check` 落实。

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
- `pyproject.toml` 与 `src/grid_ice_vision/__init__.py` 的 `__version__` 当前都是 `0.1.0`；
  计划中的 v2.0（`0.2.0`）尚未开始实现。
- **分支现状（2026-11-25）**：三条特性分支已全部并入 `master`（快进合并，无合并提交）。
  `master` 现为 `5500082`，含位置无关路径层、SCUT 审计与双划分、Track A 的
  `Pa`/基准匹配/变步长 EKF/几何 55× 提速。旧的 `feat/m0-paths-preflight`（`b708335`）与
  `feat/v2-scut-upgrade`（`18d9110`）已完全包含在 `master` 中，可删除。
  `feat/m3-severity-track-a` 与 `master` 同点，后续开发建议从 `master` 重新拉分支。
