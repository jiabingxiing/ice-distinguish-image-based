# data/

四级数据流（§11.2），目录骨架入库，内容不入库：

| 目录 | 内容 | 规则 |
|---|---|---|
| `raw/` | 原始下载/采集 | **只读，永不修改** |
| `external/` | 第三方数据集（许可待确认） | 只读，**不参与训练** |
| `interim/` | 去重、裁切、格式转换中间产物 | 可重建 |
| `processed/` | 训练就绪（图像 + 掩码 + COCO） | 由脚本生成 |
| `manifests/` | 版本化清单（JSONL） | 每次训练绑定一个 |
| `splits/` | 划分索引（JSONL） | **入库**：锁定留出集，seed 固定可复现 |

生成方式：

```bash
uv run python scripts/build_manifests.py --root data/processed/icing --out data/manifests/icing_v1.jsonl
uv run python scripts/split_manifest.py --manifest data/manifests/icing_v1.jsonl
uv run python scripts/validate_dataset.py --manifest data/manifests/icing_v1.jsonl --check-sha
```

> 用 `uv run`（或 `.venv/Scripts/python.exe`）。裸 `python` 在本机是 3.13 全局解释器，导入不到本项目包。

## SCUT 实测库登记台账（P0 清洗建集结果，2026-11-25）

外部库经 `common/paths.py::scut_root()` 解析（环境变量 → `configs/paths.local.yaml` → 项目外约定位点），
**不入库**。审计脚本：`scripts/audit_scut.py`；完整报告写 `artifacts/reports/scut_audit.json`。

| 项目 | 实测值 | 备注 |
|---|---|---|
| CSV 行数 / 磁盘图数 | 7,297 / 7,296 | 缺 1 图：`2018/8ab1536997a848e68697cdf781c181dd` |
| 可解码 / **损坏** | **7,244** / **52** | 损坏图为 JPEG 截断或 SOS 非法，已由 `build_scut_manifest.py` 100% 剔除 |
| 四类纯净分布（可解码） | `rime` 3,235 / `glaze` 1,523 / `wet_snow` 1,469 / `mixed` 1,017 | 移除 `<50 并入 mixed` 污染规则 |
| SHA-256 / pHash 去重 | 唯一 SHA-256 7,134（94 组精确重复）；pHash 簇 3,943（跨终端 1 组已并入同原子组） | `Monitor-Holdout` 跨终端零近重复泄漏 |
| 低清（≤704 px 宽）占比 | **93.2%** | 分母为可解码 7,244 张：640×480: 5,395；704×576: 1,142；320×240: 206；352×288: 6 |
| 高分辨率 | 1280×720: 455；1920×1080: 25；1024×768: 15 | 与旧方案「≥1080P」假设差距大 |
| 终端 / 局 / 跨年终端 | 382 / 24 / **179** | 跨年终端必须整终端留出，否则背景泄漏 |
| 冻杯（三天风速全 0） | **1,411 行（19.3%）** | 置 `NaN` 交 CatBoost 原生缺失分支 |
| 双无冰噪声候选 | **18 张** | 已复核含标签冲突/无冰噪声，打标 `noise_candidate=true`（训练默认排除），详见 `artifacts/reports/scut_noise_candidates.json` |

划分索引（`scripts/build_scut_manifest.py --seed 42 --with-sha`，纯净 7,244 张）：

| 协议 | train | val | test |
|---|---|---|---|
| A · Monitor-Holdout | 4,970（472 组） | 747（93 组） | 1,527（98 组） |
| B · Temporal-Holdout | 4,473（≤2016） | 2,295（2017） | 476（2018） |

协议 A 的 382 个终端及跨终端近重复簇互斥，无跨 split 泄漏；四类覆冰在三个 split 中均有覆盖，已过 `assert_no_leakage` 与 `sha256` 校验。
