# 公开数据集验证矩阵

> 用途：回答"哪些公开数据能验证本算法的准确度"。
> 所有许可与可达性结论均为 **2026-10-20 在本机实测**；复测命令见文末。
> 依据 `AGENTS.md` §3：公开数据只能建立方法与基线，**不得**写成生产精度。

## 0. 一句话结论

**没有任何公开数据集带"覆冰厚度真值"。** 厚度精度只能来自两条路：
① 合成（已有 0.155 px 宽度误差，可复现）；② 现场人工卡尺（Set-D，尚未到位）。
公开数据能验证的是**链路正确性**与**域外参考值**，不能替代验收。

## 1. 按任务分栏

| 任务 | 公开数据 | 许可 | 本机可达 | 能验证 | 不能验证 |
|---|---|---|---|---|---|
| 降水/天气分类（P1） | WEAPD（Harvard Dataverse，DOI `10.7910/DVN/M8JQCR`） | **CC0-1.0** | ✅ 已落地 6,877 张 / 11 类 | 8 维多标签头 + CORAL 接线、混淆结构 | 现场精度（网络图域外） |
| 同上 | Kaggle `jehanbhathena/weather-dataset` | **CC0-1.0** | 未复测（页面 404） | 同 WEAPD —— **与本机数据重复，不建议下载** | — |
| ROI 检测（P2） | TTPLA（GitHub `R3ab/ttpla_dataset`） | **Apache-2.0** | ✅ 官方 Drive 不可达；**hf 镜像已下载并跑通**（4.30 GB → 1,242 对图与标注） | 检测链路能否收敛、切片推理、训练稳定性 | 杆塔侧视精度（航拍俯视域差） |
| 同上 | InsPLAD（10,607 张 UAV / 17 类 / 6 类缺陷） | CC BY-NC 3.0（禁商用） | ✅ 仓库可达 | 绝缘子/金具检测的研究性评测 | 商用交付权重 |
| 同上 | CPLID（绝缘子正常/缺陷） | 学术用途 | ✅ 仓库可达 | 绝缘子检测参考 | 商用 |
| 图像质量门控 | 无专门公开集；可用退化合成（`scripts/synthesize_weather.py`） | 自有 | ✅ | 模糊/曝光/夜间阈值行为 | 现场退化分布 |
| 覆冰分割（P3） | DTL-IceNet（Zenodo `15718305`） | 未标许可 | ❌ 本机 `curl` 返回 000 | 架构参考 | 真实域 Dice |
| 同上 | 4 类掩码公开集 | **不存在** | — | — | — |
| 覆冰类型/等级（P4） | 类型 5 类 / 等级 G0–G4 标注 | **不存在** | — | 只能合成 + 现场标注 | 真实类型精度 |
| 覆冰厚度（P3.5） | **不存在带厚度真值的公开集** | — | — | — | 厚度 MAPE |
| 现场域定性检查 | 宝鸡输电线路覆冰图像 `10.57760/sciencedb.35688` | **CC BY 4.0** | ✅ `scidb.cn` 200 | 图像里有没有冰、像不像 glaze/rime | 厚度、等级、分类精度 |
| 气象先验 | 浙江 220kV 覆冰预测气象集 `10.57760/sciencedb.nyjs.00009` | CC BY-NC-SA 4.0 | ✅ 可达 | 气象侧对照、先验合理性 | 图像精度 |

## 2. 逐项说明

### 2.1 WEAPD（已用，P1 基线）

- 载体：Harvard Dataverse，单文件 `dataset(WEAPD).rar`，**620,948,936 B**，许可 **CC0 1.0**；
- 本机：`data/raw/weapd/dataset` 6,877 个文件 / 11 类，pHash 聚类后 4,467 簇，划分 3422/729/718；
- 已产出：`docs/weather_p1_report.md`（7 类 macro-F1 0.891，ECE 0.006）；
- **红线**：frost/glaze/rime 多为植物特写，与杆塔现场差一个数量级，数字不得写入验收材料。

### 2.2 TTPLA（P2 唯一可用）

- 官方仓库只有 12 张样例图 + 划分清单 + 脚本，**图像本体在 Google Drive**（本机不可达）；
- 划分清单实测：train 905 / val 109 / test 220，共 **1,234 张**（论文口径 1,100 张）；
- 标注为 **labelme `shapes[]`**（`label` + `points`），不是 COCO `objects[]`；
- 镜像：`hjjiao/TTPLA` 的 `data_original_size.tar.gz` = **4,301.6 MB**；
  另一个派生镜像 `grantmwilkinson/ttpla-transmission-1536` 为 1536 短边 parquet（975.9 MB，许可标 `other`）。
- **额外缺陷**：官方划分是图像级随机划分，文件名前缀跨 split 重叠（实测 66 个训练前缀中
  多数同时出现在 test）。因此**无法**按"场景/过程"分组防泄漏 —— 这是它只能作链路证据、
  不能作验收指标的又一个原因。
- **2026-10-20 已跑通冒烟**：10 epoch / CPU 52 min，test mAP@0.5 = 0.324
  （conductor 0.288 / tower 0.361）；4K 切片 p50 4.03 s / p95 4.87 s。
  详见 [`docs/p2_smoke_report.md`](p2_smoke_report.md)。

### 2.3 InsPLAD / CPLID

- InsPLAD 许可 **CC BY-NC 3.0**，NC 条款禁止商用 → 状态 `restricted`，只能做研究性评测；
- CPLID 为学术用途，商用需另行确认；
- 两者都是无人机巡检视角，与本项目"固定监控 + 无人机"目标域部分重合，但标注体系（资产/缺陷）
  与本项目 4 类资产不同，只能迁移或评测。

### 2.4 DTL-IceNet

- Zenodo 记录 `15718305`，页面未标许可，**2026-10-20 本机不可达**（`curl` 返回 000）；
- 状态 `pending`：既不能当主训练集，也不能作为交付依据，需书面确认授权。

### 2.5 宝鸡覆冰图像（新发现）

- DOI `10.57760/sciencedb.35688`，"Icing image of Baoji power transmission line in January 2025"，
  Science Data Bank，2026-01-22 发布，**CC BY 4.0**，单文件 **17,114,951 B**；
- 采集方式为线路上的"覆冰观测精灵"图像传感器 → 与目标部署域（杆塔侧视、导线覆冰）**一致**；
- 摘要称其流程含"导线像素提取 + 宽度量测"，覆冰厚度监测误差 ≤3%；
- **限制**：17 MB 单文件，样本量必然很小，且是否含逐帧厚度标注**未确认**（页面为 JS 渲染）。
  定位：现场域定性检查与示例图，**不作训练集、不作指标来源**。

## 3. 为什么"公开数据不能验证厚度"

1. 覆冰厚度是**物理量**，需要标定（`D_0`、`α`）或人工卡尺，公开图像几乎不附带这些元数据；
2. 现有公开覆冰数据以"航拍/遥感/线路监控截图"为主，缺少同一机位的裸线与覆冰对照；
3. 因此 `d_eq_mm` 的验证只能靠合成（真值零误差）+ 现场实测（Set-D）；
4. 任何用公开集算出的"厚度误差"都不成立 —— 分子是真值，分母是猜的。

## 4. 建议用法（按优先级）

| 优先级 | 动作 | 目的 | 产出 |
|---|---|---|---|
| 1 | TTPLA 冒烟训练（10 epoch） | 证明检测链路能收敛 | ✅ 已完成，见 `docs/p2_smoke_report.md` |
| 2 | 宝鸡覆冰图定性检查 | 确认现场域外观与合成/WEAPD 的差距 | 报告附图与差异清单 |
| 3 | InsPLAD 研究性评测（可选） | 绝缘子检测迁移上限参考 | 评测表（标 restricted） |
| 4 | 合成覆冰扩容 + 分割训练 | 分割 Dice、等级 QWK、厚度 MAPE | 合成口径报告 |
| 5 | 现场 Set-D 标注 | 唯一可作验收的数据 | 模型卡 + 验收报告 |

## 5. 复测命令（2026-10-20 实测用）

```bash
# 许可
curl -sI https://raw.githubusercontent.com/R3ab/ttpla_dataset/master/LICENSE
curl -s "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/M8JQCR" | head -c 500

# 可达性
curl -s -o /dev/null -w "%{http_code}\n" https://zenodo.org/records/15718305   # 000 = 不可达
curl -s -o /dev/null -w "%{http_code}\n" https://www.scidb.cn                  # 200

# 镜像大小
curl -s "https://hf-mirror.com/api/datasets/hjjiao/TTPLA/tree/main?recursive=true&expand=true"

# 数据集元数据
curl -s "https://api.datacite.org/dois/10.57760/sciencedb.35688"
```

## 6. 与许可台账的关系

本文件只回答"能验证什么"；**能否使用**以 `docs/data_license_ledger.md` 为准。
两份文件冲突时，以台账的许可状态为准。
