# 数据与资源许可台账

> 依据 `IMPLEMENTATION_PLAN.md` §11.1、附录 A（2026-10-19 实测）。
> **硬门槛：许可未确认的资源只能进 `data/external/`（只读、不参与训练）。**

状态取值：`confirmed`（可商用/已书面确认）· `pending`（待核实）· `restricted`（仅学术/非商用）· `rejected`（不采用）

| 资源 | 内容 | 用途 | 许可 | 状态 | 联系人 / 备注 |
|---|---|---|---|---|---|
| **TTPLA** | 航拍杆塔/导线 COCO 多边形 | ROI 预训练、`W_0_px` 来源 | 仓库未见 LICENSE | `pending` | 域为航拍俯视，与杆塔侧视差异大，**只能预训练** |
| **CPLID** | 绝缘子正常/缺陷 | 绝缘子检测预训练 | 学术用途 | `restricted` | 商用需另行确认 |
| **InsPLAD** | 无人机电力资产 | ROI 预训练 | 待核实 | `pending` | — |
| **DTL-IceNet**（Zenodo 15718305） | 覆冰样例 + 分割代码（130.9 MB + 146.1 kB） | 架构参考、样例验证 | 页面未标注许可 | `pending` | **需向通讯作者（Fu Yufei 等）书面确认商用授权**；不得作唯一主训练集 |
| **PLAIR-AW** | Haze/Rain/Snow × CPLID/TTPLA/InsPLAD | 退化图像 + L/M/H 标签 | 待核实 | `pending` | 仓库**只有数据集与基线评测，无合成脚本** |
| **Kaggle Weather Dataset** | 6862 张 11 类（含 glaze/rime/frost/hail） | 天气分类预训练 | 随 Kaggle 数据集条款 | `pending` | 原代码仓库为 HOG+PCA+SVM，**非深度模型** |
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
