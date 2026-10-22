# 数据与资源许可台账

> 依据 `IMPLEMENTATION_PLAN.md` §11.1、附录 A。**2026-10-20 重新核实**了 TTPLA / WEAPD / InsPLAD 三行的许可（原台账误记为 pending），
> 并复核了 TTPLA 镜像的可达性与单文件大小，新增两个 Science Data Bank 覆冰资源。
> **硬门槛：许可未确认的资源只能进 `data/external/`（只读、不参与训练）。**

状态取值：`confirmed`（可商用/已书面确认）· `pending`（待核实）· `restricted`（仅学术/非商用）· `rejected`（不采用）

| 资源 | 内容 | 用途 | 许可 | 状态 | 联系人 / 备注 |
|---|---|---|---|---|---|
| **TTPLA** | 航拍杆塔/导线 COCO 多边形，**1,234 张 3840×2160**（官方 split 905/109/220），标注 11,467 个 shapes → 映射后 **11,070 个实例**（conductor 10,102 / tower 968） | ROI 预训练、`W_0_px` 来源 | **Apache-2.0**（2026-10-20 实读 LICENSE：Copyright 2020 Rabab Abdelfattah） | `confirmed` | 图像本体只在 **Google Drive**（本机 `curl` 返回 000，不可达）；改用 hf 镜像 `hjjiao/TTPLA`（`data_original_size.tar.gz` 4,301.6 MB）。域为航拍俯视，与杆塔侧视差异大，**只能预训练**；官方 split 为图像级随机划分，无法按场景分组防泄漏 |
| **TTPLA 派生镜像** `grantmwilkinson/ttpla-transmission-1536` | 1536 短边 parquet，975.9 MB（train 6 + val 2 + test 2 分片） | 备选通路 | 仓库标 `license: other`（上游仍为 Apache-2.0） | `pending` | 只在原始 tarball 缺标注时考虑；派生数据需保留上游 NOTICE |
| **CPLID** | 绝缘子正常/缺陷 | 绝缘子检测预训练 | 学术用途 | `restricted` | 商用需另行确认 |
| **InsPLAD** | 10607 张 UAV 图像 / 17 类资产 / 6 类缺陷 | 绝缘子/防振锤/间隔棒检测 | **CC BY-NC 3.0（禁商用）**（2026-10-20 实读 LICENSE） | `restricted` | 可做研究性评测；**不得用于商用交付权重** |
| **DTL-IceNet**（Zenodo 15718305） | 覆冰样例 + 分割代码（130.9 MB + 146.1 kB） | 架构参考、样例验证 | 页面未标注许可 | `pending` | **需向通讯作者（Fu Yufei 等）书面确认商用授权**；不得作唯一主训练集；2026-10-20 实测 Zenodo 本机不可达（curl 000） |
| **PLAIR-AW** | Haze/Rain/Snow × CPLID/TTPLA/InsPLAD | 退化图像 + L/M/H 标签 | 待核实 | `pending` | 仓库**只有数据集与基线评测，无合成脚本** |
| **WEAPD**（Xiao Haixia, Harvard Dataverse, DOI `10.7910/DVN/M8JQCR`） | 6862 张 11 类（dew/fogsmog/frost/glaze/hail/lightning/rain/rainbow/rime/sandstorm/snow）；单文件 `dataset(WEAPD).rar` **620,948,936 B** | 天气分类 P1 | **CC0-1.0**（2026-10-20 Dataverse API 复核） | `confirmed` | 权威原始出处；本机已落地 6,877 张 / 11 类 |
| **Kaggle Weather Dataset**（jehanbhathena 镜像） | 6862 张 11 类 | 同 WEAPD | **CC0: Public Domain**（Kaggle 页 License 栏，2026-10-20） | `confirmed` | 作者已移除部分伪造图；frost/glaze/rime 多为植物特写，**域外**；**与本机 WEAPD 重复，不要二次下载** |
| **宝鸡输电线路覆冰图像**（Science Data Bank，`10.57760/sciencedb.35688`） | "Icing image of Baoji power transmission line in January 2025"，17,114,951 B / 1 个文件 | 现场域定性检查、示例图 | **CC BY 4.0**（DataCite 元数据，2026-01-22 发布） | `confirmed` | 采集自线路上的"覆冰观测精灵"图像传感器，域与目标一致；**样本量小，只作定性检查，不作训练集、不作指标来源**；是否含逐帧厚度标注**未确认** |
| **浙江 220kV 覆冰预测气象集**（`10.57760/sciencedb.nyjs.00009`） | 多源气象观测（覆冰预测用） | 气象侧对照、先验合理性 | **CC BY-NC-SA 4.0** | `restricted` | NC 条款禁止商用；不含图像 |
| **CDD-11 / Snow100K / LOL** | 通用退化 | 复原模块（P4 才用） | 各自不同 | `pending` | — |
| **firc-dataset** | Gitee/FIRC 电力行业数据交易平台 | 不作主训练集 | 不明 | `rejected` | 明确声明"不对模型精度作保证" |
| **NWPUYRCC2 / Sentinel-LakeIce** | 遥感河冰/湖冰分割 | 仅纹理先验 | 待核实 | `restricted` | 域差过大（俯视、大尺度、SAR/光学），不混训 |
| **CoInfra** | 路侧协同感知（冻雨） | 域适应消融 | CC-BY-NC-4.0 | `restricted` | **NC 条款禁止商用** |
| **自采 / 现场** | 电网监控图像 | 域适应、验收（Set-D） | 内部授权 | `pending` | 暂无 |

## 使用规则

1. `pending` 资源**只能**放 `data/external/`，且 `data/external/` 只读；
2. 进入 `data/processed/` 的资源必须在本文档状态为 `confirmed` 或 `restricted`
   且已确认使用场景（学术 vs 商用）合规；
3. 每次新增资源，先在此表登记，再改 `scripts/download_public.py` 的 `LEDGER`；
4. 交付前做一次依赖树许可审查（`pip-licenses`），确认无 AGPL 传染。

## 待闭环事项（对应 §18 附录 D）

| # | 事项 | 找谁 | 影响 |
|---|---|---|---|
| 1 | DTL-IceNet 数据/代码商用授权 | 论文通讯作者 | 能否用于交付 |
| 2 | 本项目是否属"内部使用"（决定 AGPL 是否适用） | 法务 | 检测器选型 |
| 3 | 导线外径 `D_0` 台账是否可得 | 运检部门 | 能否算物理厚度 |
| 4 | 是否有 PTZ 编码器读数 / 双目 | 设备管理 | 标定级别 L2 |
| 5 | 本地区覆冰预警现行规程阈值 | 业务方 | I–IV 级阈值 |
| 6 | 是否有历史晴天无冰图像（同预置位） | 运检部门 | L3 标定可用性 |
| 7 | 目标部署硬件（Jetson / x86 / 云端） | 项目组 | 模型尺寸与量化策略 |
