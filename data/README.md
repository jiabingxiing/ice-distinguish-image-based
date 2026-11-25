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
python scripts/build_manifests.py --root data/processed/icing --out data/manifests/icing_v1.jsonl
python scripts/split_manifest.py --manifest data/manifests/icing_v1.jsonl
python scripts/validate_dataset.py --manifest data/manifests/icing_v1.jsonl --check-sha
```

## SCUT 实测库登记台账（M0 审计结果，2026-11-25）

外部库经 `common/paths.py::scut_root()` 解析（环境变量 → `configs/paths.local.yaml` → 项目外约定位点），
**不入库**。审计脚本：`scripts/audit_scut.py`；完整报告写 `artifacts/reports/scut_audit.json`。

| 项目 | 实测值 | 备注 |
|---|---|---|
| CSV 行数 / 磁盘图数 | 7,297 / 7,296 | 缺 1 图：`2018/8ab1536997a848e68697cdf781c181dd` |
| 可解码 / **损坏** | 7,244 / **52** | 损坏图为 JPEG 截断或 SOS 非法，**训练前必须剔除** |
| 低清（≤704 px 宽）占比 | **93.2%** | 640×480: 5,395；704×576: 1,142；320×240: 206；352×288: 6 |
| 高分辨率 | 1280×720: 455；1920×1080: 25；1024×768: 15 | 与旧方案「≥1080P」假设差距大 |
| 终端 / 局 / 跨年终端 | 382 / 24 / **179** | 跨年终端必须整终端留出，否则背景泄漏 |
| 冻杯（三天风速全 0） | **1,411 行（19.3%）** | 置 `NaN` 交 CatBoost 原生缺失分支 |
| 双无冰噪声候选 | **18 张** | 已发现仍有明显雾凇，需人工复检后再入负样本池 |

划分索引（`scripts/build_scut_manifest.py`，seed=42）：

| 协议 | train | val | test |
|---|---|---|---|
| A · Monitor-Holdout | 4,929（472 组） | 869（94 组） | 1,498（98 组） |
| B · Temporal-Holdout | 4,522（≤2016） | 2,296（2017） | 478（2018） |

协议 A 的 382 个终端互斥，无跨 split 终端；四类覆冰在三个 split 中均有覆盖，已过 `assert_no_leakage`。
