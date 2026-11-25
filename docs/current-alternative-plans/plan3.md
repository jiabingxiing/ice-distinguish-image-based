# 基于图像的覆冰、降水类型及程度识别算法——代码项目实现计划


## 一、项目概述与目标

### 1.1 问题定义

面向电网覆冰监测场景，构建一个**多任务视觉识别系统**，从单张现场图像（输电线路、杆塔、绝缘子、变电站设备等）中输出三类结构化信息：

| 输出维度 | 任务类型 | 标签空间 |
|---|---|---|
| 覆冰有无与区域 | 语义分割 / 目标检测 | `ice` / `non-ice`（像素级或区域级） |
| 覆冰类型 | 多分类 | 雨凇（glaze）、雾凇（rime）、混合凇（mixed）、湿雪（wet snow） |
| 覆冰严重程度 | 有序分级 / 回归 | 按覆冰厚度或覆盖率映射为 4—5 级（轻/中/重/危急） |
| 降水类型 | 多分类 | 雨、雪、冻雨、冰雹、雨夹雪、无降水 |

**核心工程约束**：电网部署要求推理延迟可控（边缘端目标 ≥ 10—15 FPS @ 640×640）、模型轻量（参数量 ≤ 10M 级为优先）、可解释性（至少提供分割掩码或注意力热力图供运维人员复核）。

### 1.2 评估指标

- **覆冰分割**：mIoU、Boundary IoU（覆冰边缘是关键判断依据）、Ice Recall（漏检覆冰的代价高于误检，Recall 权重应高于 Precision）。
- **覆冰类型 / 降水类型分类**：Macro-F1（类别不平衡下比 Accuracy 更可靠）、混淆矩阵中**冻雨 vs. 雨凇 vs. 雨**的混淆率需单独报告。
- **严重程度分级**：有序分类的 Quadratic Weighted Kappa（QWK），或分级准确率 ± 相邻级容忍准确率。
- **端到端**：单帧推理延迟（ms）、显存占用、TensorRT FP16 下的吞吐。


## 二、开源调研报告与借鉴方案

### 2.1 检索范围与方法

围绕 `ice accretion / ice coating detection`、`transmission line icing`、`precipitation type classification`、`freezing rain image`、`power line inspection dataset` 等关键词，在 GitHub、Hugging Face、Papers with Code、Zenodo、Kaggle 及中文技术社区进行了系统检索。以下为经过内容核实的高价值资源。

### 2.2 推荐项目/仓库（6 项）

#### ① DTL-IceNet（Fu et al., 2026）—— 覆冰类型 + 厚度估计的双任务学习

- **链接**：代码与数据集 Zenodo：`https://doi.org/10.5281/zenodo.15718305`；论文发表于 *Atmospheric Measurement Techniques*。
- **核心思路**：双任务架构，`ResSepNet` 做覆冰类型分类，`MOMSA-SegNet` 做覆冰区域分割以支撑厚度估计，并融合气象数据校正厚度。在自建 IceType / IceSeg 数据集上，覆冰识别准确率比 EfficientNet-V2 / MobileNet-V3 / ResNeXt / MobileOne 高 4.17%，分割 MIoU 比 UNet++ 高 1.64%，厚度估计 MAPE 11.82%、PCC 0.99。
- **优点**：任务定义与本项目**高度重合**（类型 + 分割 + 厚度/程度），代码与数据均公开，可直接作为 MVP 的架构起点；双任务间的互补约束机制对“覆冰区域”与“覆冰类型”的联合学习有参考价值。
- **局限**：数据集规模有限（电网自有图像构建），地形因素未被纳入，极端环境泛化性有待验证。
- **可复用部分**：双任务 backbone 设计、IceSeg 分割标签规范、厚度估计的评估协议；可作为**覆冰分支的主干架构**，替换为更轻量的 backbone 后做边缘适配。

#### ② firc-dataset（中文电力场景多格式数据集集合）

- **链接**：社区标识的 GitHub 仓库 `firc-dataset`，在多个中文技术文章中作为数据集来源被引用。
- **核心内容**：包含多套覆冰相关标注数据，已确认的至少包括：覆冰分割 labelme 格式 1138 张 1 类别（`icing`，标注框数 1280，分辨率 640×640）；覆冰分割 labelme 格式 1227 张 2 类别（`ice` + `powerline`）；VOC+YOLO 格式覆冰检测 109 张 2 类别（`line_icing` 等）。同仓库还包含 9400 张 32 类别的电力巡检检测数据集等更大规模资源。
- **优点**：格式多样（labelme / VOC / YOLO），可直接转换为训练管线所需的 COCO / YOLO 格式；**中文电力场景真实图像**，域匹配度高；规模适中，适合快速启动 MVP。
- **局限**：部分数据需通过社区渠道获取，需核实各子集的许可条款；单类别标注（仅 `icing`）不足以支撑覆冰**类型**分类，需要额外数据补充。
- **可复用部分**：直接作为覆冰**分割与检测的初始训练集**；其 `powerline` 类别可用于辅助学习“冰附着于导线”的空间约束（对应 Ice-Seg 中 false positive 的核心来源）。

#### ③ Ice-Seg（U-ResNet 论文附带数据集）—— 覆冰分割的“难例”参考

- **链接**：论文 *Sensors* 2025, 25(18), 5600；Ice-Seg 为论文中使用的私有数据集（商用购买 + 人工标注）。
- **核心内容**：1591 张 640×640 高压电缆覆冰图像，分为三类场景：晴天/良好天气、恶劣天气（低对比度、噪声、模糊）、含背景设备干扰。训练集 1467 张，测试集 124 张。
- **价值**：虽然数据集本身非公开，但论文对**三类场景下分割失败模式的诊断**非常有工程价值：晴天场景的核心风险是冰与云/反光/雪地的混淆，需依赖空间位置与纹理差异；恶劣天气的核心风险是漏检，需在 loss 中惩罚 false negative；设备背景的核心风险是杆塔几何结构与覆冰的混淆，需场景理解与“冰应附着于悬垂导线”的先验。
- **可复用部分**：**数据增强策略与难例挖掘方案**可直接借鉴——按这三种场景分组做分层采样和针对性增强；loss 设计中强化对 false negative 的惩罚。

#### ④ Weather-Image-Classification（prithivMLmods）—— 天气/降水分类的预训练起点

- **链接**：Hugging Face `prithivMLmods/Weather-Image-Classification`，Apache-2.0 许可。
- **核心内容**：基于 `google/siglip2-base-patch16-224` 微调的视觉语言分类模型，5 类输出：`cloudy/overcast`、`foggy/hazy`、`rain/storm`、`snow/frosty`、`sun/clear`；整体 accuracy 0.859，rain/storm F1 0.762，snow/frosty F1 0.839。
- **优点**：可直接通过 `transformers` 管线加载，作为**降水类型分支的初始化权重**；SigLIP2 backbone 具有图文对齐能力，便于后续引入文本提示做零样本扩展。
- **局限**：类别粒度过粗——**没有单独区分冻雨、冰雹、雨夹雪**，`rain/storm` 将一般降雨与强对流合并，`snow/frosty` 混合了降雪与霜冻。不能直接满足电网对冻雨这一高危类型的识别需求。
- **可复用部分**：作为**预训练特征提取器**，在本项目自建的冻雨/冰雹/雨夹雪细粒度数据上做**分层微调**（先解冻分类头，再逐步解冻顶层 transformer block）；推理管线代码可直接用于快速基线对比。

#### ⑤ dima806/weather_types_image_detection —— 更细粒度的天气分类参考

- **链接**：Hugging Face `dima806/weather_types_image_detection`，基于 ViT-base 微调。
- **核心内容**：11 类天气分类，包括 `hail`（F1 0.997）、`rime`（F1 0.895）、`glaze`（F1 0.896）、`snow`（F1 0.928）、`rain`（F1 0.966）等，整体 accuracy 0.958。
- **优点**：**类别体系与电网降水/覆冰类型高度可映射**——`glaze`（雨凇）、`rime`（雾凇）、`hail`（冰雹）均是本项目需要的标签；可以作为细粒度降水类型分类的**迁移学习起点**或**伪标签生成器**。
- **局限**：训练数据来源未详细说明，需在实际电网图像上重新校准置信度；`frost` 与 `rime` 的边界可能与电网的覆冰类型定义不一致。
- **可复用部分**：其 `glaze` / `rime` / `hail` 类别头可**映射到本项目的降水类型与覆冰类型标签空间**；建议作为**教师模型**对无标注电网图像做伪标签，再在本项目标注数据上蒸馏 / 微调。

#### ⑥ CoInfra（冻结雨场景补充资源）

- **链接**：Hugging Face `MinghaoNing/CoInfra`，CC-BY-NC-4.0。
- **核心内容**：大规模协同感知数据集，明确覆盖**冻结雨（freezing rain）** 、雪、雨等恶劣天气，包含多节点相机与 LiDAR 同步数据。
- **价值**：**冻雨视觉数据极其稀缺**，CoInfra 可作为冻雨场景的补充验证集或域适应源数据。但需注意其场景是路侧基础设施协同感知，与输电线路视角差异较大，**不宜直接混合训练**，更适合做**冻雨视觉特征的预训练或域对抗适应**。
- **可复用部分**：冻雨样本的可视化分析与特征统计；域适应阶段的源域数据之一。

### 2.3 补充说明：CPLID 的定位

`CPLID`（Chinese Power Line Insulator Dataset，`github.com/InsulatorData/InsulatorDataSet`）是公开的绝缘子缺陷数据集，**不含覆冰标注**，与覆冰任务不直接重合。但其价值在于：若后续需要将覆冰模型集成到更广泛的电力巡检系统中，CPLID 训练的绝缘子检测器可作为**前置定位模块**，为覆冰分割提供感兴趣区域（ROI），减少全图推理的计算开销和背景干扰。

### 2.4 二次开发策略总结

| 项目 | 在本项目中的角色 | 二次开发动作 |
|---|---|---|
| DTL-IceNet | 覆冰分支的架构蓝本 | 保留双任务设计，backbone 替换为轻量网络（如 MobileNetV4 / RepViT），适配边缘推理 |
| firc-dataset | 覆冰初始训练集 | 格式统一为 COCO + YOLO 双格式；与自建数据合并时做域分层采样 |
| Ice-Seg 诊断经验 | 数据增强与难例策略 | 实现“场景分层采样器”和 false-negative-aware loss |
| Weather-Image-Classification | 降水类型预训练 | 分层微调，新增冻雨/冰雹/雨夹雪类别头 |
| dima806/weather | 细粒度类型映射与伪标签 | 将 glaze/rime/hail 映射到本项目标签，教师模型伪标签蒸馏 |
| CoInfra | 冻雨补充与域适应 | 仅用于域适应实验和冻雨召回评估，不直接混训 |


## 三、系统架构与算法设计

### 3.1 整体技术路线

采用**“共享 backbone + 双分支解耦”** 的多任务架构，而非将所有任务压缩到一个统一分类头。原因在于：覆冰分割需要空间细节，降水类型分类需要全局语义，两者对特征分辨率的需求存在冲突；双分支可以在共享底层特征的同时，分别使用适合各自任务的 neck / head。

```
输入图像（640×640）
        │
        ▼
┌───────────────────┐
│  共享 Backbone     │  MobileNetV4 / RepViT / EfficientViT-L1
│  （轻量化，预训练） │  在 ImageNet 或电力图像上预训练
└─────────┬─────────┘
          │ 多尺度特征 {C3, C4, C5}
          ├──────────────────────────────┐
          ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  覆冰分割分支        │      │  类型/程度分类分支        │
│  Lightweight SegHead │      │  Global Pooling + MLP    │
│  (DeepLabV3+ Lite /  │      │  多头输出：              │
│   LR-ASPP 风格)      │      │  · 覆冰类型（4类）       │
│  → 覆冰 mask          │      │  · 覆冰严重程度（5级）   │
│                      │      │  · 降水类型（6类）       │
└─────────────────────┘      └─────────────────────────┘
          │                              │
          ▼                              ▼
   覆冰覆盖率 = mask面积/导线ROI面积    有序分级损失（QWK-friendly）
   → 映射到严重程度等级                 + 类别平衡 CrossEntropy
```

**共享 backbone 的选择**：优先测试 `MobileNetV4-Conv-Large` 或 `RepViT-M1.1`，两者均在移动端精度-效率曲线上优于 MobileNetV3。若覆冰分割的 mIoU 不足，可升级到 `EfficientViT-L1`（Hugging Face 上已有 BDD100K 天气分类的微调版本可供初始化参考）。

**覆冰严重程度的定义**：建议不直接回归连续厚度（单目图像无标定情况下绝对厚度不可靠），而是采用**基于覆冰覆盖率和形态特征的分级**：

| 等级 | 定义 | 图像可判据 |
|---|---|---|
| 0（无/轻微） | 无明显覆冰或仅薄层 | mask 覆盖率 < 5%，导线轮廓清晰 |
| 1（轻度） | 覆冰可见但导线轮廓基本保持 | 覆盖率 5%—20%，边缘增厚但未显著改变形态 |
| 2（中度） | 覆冰显著改变导线轮廓 | 覆盖率 20%—40%，出现明显冰柱/冰壳 |
| 3（重度） | 覆冰严重，导线形态明显畸变 | 覆盖率 > 40%，可见下垂、偏心覆冰或桥接 |
| 4（危急） | 覆冰极重，存在舞动/断裂风险迹象 | mask 跨导线桥接，或结合杆塔倾斜/导线弧垂的辅助信号 |

该分级与 DTL-IceNet 中将厚度映射到类别而非直接回归的策略一致，也与专利文献中按厚度区间分级的思路可对齐。实际阈值需在电网标注数据上做分布校准。

### 3.2 训练策略：分阶段 + 多任务损失

**阶段 1（覆冰分支预训练）** ：在 firc-dataset + 自建标注数据上单独训练分割分支，使用 Dice + Focal Loss（Focal 侧重抑制 false negative）。此阶段分类分支冻结或仅使用 ImageNet 预训练权重。

**阶段 2（联合多任务微调）** ：解冻共享 backbone，以较小学习率联合优化：

\[
\mathcal{L} = \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{type}} \mathcal{L}_{\text{type}} + \lambda_{\text{sev}} \mathcal{L}_{\text{sev}} + \lambda_{\text{precip}} \mathcal{L}_{\text{precip}}
\]

- 覆冰类型和降水类型使用 **Class-Balanced CrossEntropy**（冻雨/冰雹样本少，需重加权或使用 logit-adjusted CE）。
- 严重程度使用 **Ordinal Regression Loss**（如 Coral / Corn），使预测尊重等级的有序性，避免把“重度”错判为“无”这类代价极高的错误。

**阶段 3（领域适应与伪标签蒸馏）** ：用 dima806/weather 和 Weather-Image-Classification 对未标注的电网图像生成伪标签，以一致性正则化或知识蒸馏的方式让模型适应电网图像的色调、视角和背景分布。CoInfra 的冻雨样本可在这一阶段以**域对抗**或**风格归一化**的方式引入，而不直接改变分类头的决策边界。

### 3.3 数据策略

**公开数据 + 自建数据的分工**：

- firc-dataset 覆冰分割数据（1138 + 1227 + 109 张）用于**冷启动**，快速验证 backbone 和分割头的可行性。
- DTL-IceNet 的 Zenodo 数据用于**架构验证与对比**，确认双任务设计的收益在本场景下可复现。
- 降水类型：以 Weather-Image-Classification / dima806 的预训练权重为起点，在**电网现场采集的降水图像**上做微调。冻雨/冰雹的细粒度类别需要自建标注，公开数据只能提供预训练阶段的特征。
- **标注规范**：覆冰分割采用 polygon 标注，要求标注者同时记录“覆冰类型”和“是否附着于导线”两个属性。降水类型标注以图像整体视觉判断为主，可辅助气象记录做交叉验证。

**合成与增强**：Ice-Seg 的难例经验表明，雪地背景、云层、金属反光是主要 false positive 来源。增强策略应包含：随机改变背景亮度/色温模拟不同光照；在晴天图像上叠加合成冰壳/冰柱纹理（有限度使用，避免域偏移）；对恶劣天气图像做强噪声和低对比度增强，专门提升 Recall。

### 3.4 部署考量

电网场景通常有两种部署形态：

1. **边缘端（杆塔/变电站侧）** ：Jetson Orin Nano / NX 级别设备。参考已有电力巡检工作，YOLOv7-Tiny / YOLO11n 在 Jetson + TensorRT FP16/INT8 下可达到实时推理。本项目的分割分支参数量需控制在 5M 以内，分类分支可共享 backbone。导出链路：PyTorch → ONNX → TensorRT FP16；若 INT8 量化，需用校准集保证覆冰边缘的 IoU 不显著下降。
2. **云端/站端服务器**：处理历史图像批量复核和模型迭代，可使用较大模型（如 EfficientViT-L1 级别）作为“教师”为边缘模型提供蒸馏信号。

**可解释性输出**：边缘端至少输出覆冰 mask 叠加图；分类分支可输出 Grad-CAM 热力图，帮助运维人员判断模型关注的是导线上的冰层还是背景中的雪地/云层。


## 四、详细文件夹结构（src layout + Feature-based）

```
project-root/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── uv.lock
├── Makefile
├── AGENTS.md
│
├── src/
│   └── power_ice/
│       ├── __init__.py
│       ├── config/                  # 配置加载与 schema
│       │   ├── __init__.py
│       │   ├── schema.py            # Pydantic 配置模型
│       │   └── loader.py
│       ├── data/                    # 数据管线（Feature）
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── firc_dataset.py       # firc-dataset 解析与格式统一
│       │   │   ├── ice_seg_dataset.py    # 自定义覆冰分割数据集
│       │   │   └── weather_dataset.py    # 降水类型分类数据集
│       │   ├── transforms/
│       │   │   ├── ice_augment.py        # 覆冰专用增强（背景合成、强噪声）
│       │   │   └── common.py
│       │   ├── samplers/
│       │   │   └── scene_stratified.py   # Ice-Seg 三类场景分层采样
│       │   └── collate.py
│       ├── models/                   # 模型定义（Feature）
│       │   ├── __init__.py
│       │   ├── backbones/
│       │   │   ├── mobilevit.py
│       │   │   └── repvit.py
│       │   ├── heads/
│       │   │   ├── seg_head.py           # LR-ASPP / DeepLabV3+ Lite
│       │   │   └── cls_head.py           # 多头分类（类型/程度/降水）
│       │   ├── multi_task_model.py
│       │   └── losses/
│       │       ├── seg_loss.py           # Dice + Focal（FN-aware）
│       │       ├── ordinal_loss.py       # Coral / Corn
│       │       └── class_balanced_ce.py
│       ├── training/                 # 训练流程（Application）
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── schedulers.py
│       │   ├── callbacks.py              # 早停、checkpoint、TensorBoard
│       │   └── distillation.py           # 伪标签蒸馏
│       ├── inference/                # 推理与后处理（Application）
│       │   ├── __init__.py
│       │   ├── predictor.py
│       │   ├── severity_mapper.py        # mask → 覆盖率 → 严重等级
│       │   └── visualize.py              # mask 叠加、Grad-CAM
│       ├── deployment/               # 边缘/云端导出（Infrastructure）
│       │   ├── __init__.py
│       │   ├── export_onnx.py
│       │   ├── export_tensorrt.py
│       │   └── benchmark.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           ├── logging.py
│           └── metrics.py                # mIoU, Boundary IoU, QWK, Macro-F1
│
├── configs/                          # 配置文件（按阶段/backbone 组织）
│   ├── base.yaml
│   ├── stage1_seg_pretrain.yaml
│   ├── stage2_multitask.yaml
│   ├── stage3_domain_adapt.yaml
│   └── edge_export.yaml
│
├── scripts/                          # 可执行入口脚本
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   ├── prepare_firc.py               # labelme/VOC → COCO/YOLO 转换
│   └── make_pseudo_labels.py
│
├── data/                             # 数据根（.gitignore 排除）
│   ├── raw/                          # 原始下载数据（firc-dataset、Zenodo 数据）
│   ├── interim/                      # 格式转换后的中间数据
│   ├── processed/                    # 训练就绪的 COCO/YOLO 数据
│   └── external/                     # CoInfra 等域适应辅助数据
│
├── tests/
│   ├── unit/
│   │   ├── test_transforms.py
│   │   ├── test_losses.py
│   │   └── test_severity_mapper.py
│   ├── integration/
│   │   ├── test_training_smoke.py    # 1—2 batch 过拟合测试
│   │   └── test_export_onnx.py
│   └── conftest.py
│
├── docs/
│   ├── README.md
│   ├── data_annotation_guide.md      # 覆冰/降水标注规范
│   ├── severity_rubric.md            # 严重程度分级判据
│   └── deployment.md
│
├── notebooks/                        # 实验记录（可选）
│   └── 01_dataset_eda.ipynb
│
└── .github/
    ├── workflows/
    │   ├── ci.yml                    # lint + unit test
    │   └── train_smoke.yml
    └── ISSUE_TEMPLATE.md
```

**结构说明**：

- **Feature-based 优先**：`data/`、`models/`、`training/`、`inference/` 按功能划分，每个功能内部再按需要分层。`deployment/` 单独隔离，因为边缘导出依赖与训练依赖不同（TensorRT 环境不宜污染训练环境）。
- **`src/power_ice/` 包名**：不使用 `src/` 下直接放模块，保证可安装性（`uv pip install -e .`）。
- **配置与代码分离**：所有超参数、数据路径、backbone 选择通过 `configs/` 中的 YAML + Pydantic schema 管理，避免硬编码。
- **禁止的噪音文件夹**：不在根目录创建 `scripts_old/`、`temp/`、`checkpoints/`、`outputs/` 等散落目录；实验输出统一写入 `runs/`（加入 `.gitignore`）。
- **嵌套深度**：最深为 `src/power_ice/models/heads/`（3 层），符合 ≤ 3—4 层约束。

**`pyproject.toml` 与 uv 管理**：

```toml
[project]
name = "power-ice"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "torchvision",
    "timm",
    "transformers",
    "opencv-python-headless",
    "albumentations",
    "pydantic>=2",
    "pyyaml",
    "tensorboard",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "pre-commit"]
export = ["onnx", "onnxruntime", "tensorrt"]  # tensorrt 按实际平台安装
```

常用命令：`uv sync --extra dev`；`make train STAGE=stage2`；`make test`；`make export`。`AGENTS.md` 中写入目录约定：新功能必须在 `src/power_ice/` 下对应 Feature 文件夹中创建模块，禁止在根目录新增 Python 脚本。


## 五、分阶段开发计划

### 里程碑 M1：数据基础与覆冰分割 MVP（第 1—3 周）

**目标**：在 firc-dataset 上训练一个可用的覆冰二值分割模型，建立完整的数据管线和评估基线。

| 任务 | 产出 | 依赖 |
|---|---|---|
| M1.1 下载/获取 firc-dataset 覆冰子集 | 原始数据就位 | — |
| M1.2 实现 `prepare_firc.py`：labelme/VOC → COCO + YOLO 统一格式 | `data/processed/firc_coco/` | M1.1 |
| M1.3 实现 `IceDataset` + 场景分层采样器 | 数据加载器通过单元测试 | M1.2 |
| M1.4 实现 `MobileNetV4 + LR-ASPP` 分割模型 | `multi_task_model.py`（仅 seg 分支） | M1.3 |
| M1.5 训练 + 评估：mIoU、Boundary IoU、Ice Recall | 基线结果表；ONNX 导出可用 | M1.4 |

**关键难点**：firc-dataset 的 labelme 标注仅为 polygon，需确认是否覆盖全部覆冰区域；若标注偏粗，Boundary IoU 可能偏低，需在评估时同时报告“膨胀后 IoU”。

**风险缓解**：如果 firc-dataset 的实际可获取性受阻（需核实仓库状态），立即切换到 Zenodo 的 DTL-IceNet IceSeg 数据作为 M1 分割训练集。

### 里程碑 M2：多任务分类分支与联合训练（第 4—6 周）

**目标**：加入覆冰类型、严重程度、降水类型分类头，完成阶段 1→2 的联合训练。

| 任务 | 产出 | 依赖 |
|---|---|---|
| M2.1 加载 Weather-Image-Classification / dima806 权重，构建分类分支 | `cls_head.py` + 预训练 backbone | M1.4 |
| M2.2 标注 / 整理覆冰类型与严重程度标签（基于 firc 数据 + 小规模自建补充） | 类型/程度标签表 | M1.2 |
| M2.3 实现 Class-Balanced CE、Ordinal Loss | loss 单元测试通过 | — |
| M2.4 阶段 2 联合训练：冻结→解冻，学习率预热 | 联合模型 checkpoint | M2.1—M2.3 |
| M2.5 全指标评估：mIoU、Macro-F1、QWK、混淆矩阵 | 评估报告 `docs/m2_eval.md` | M2.4 |

**关键难点**：覆冰类型标签的稀缺性。如果 firc-dataset 只有二值 `icing` 标注，M2 的覆冰类型分类需要依赖小规模人工标注 + dima806 的 `glaze/rime` 伪标签做弱监督。建议在 M2.2 先标注 300—500 张多类型覆冰图像作为验证锚点。

### 里程碑 M3：领域适应与轻量化导出（第 7—9 周）

**目标**：提升电网真实图像上的泛化性，完成 ONNX/TensorRT 导出与边缘性能基线。

| 任务 | 产出 | 依赖 |
|---|---|---|
| M3.1 用教师模型对未标注电网图像生成伪标签 | `data/processed/pseudo_labels/` | M2.4 |
| M3.2 蒸馏 / 一致性正则化微调 | 适应后的模型 | M3.1 |
| M3.3 引入 CoInfra 冻雨样本做域适应消融（可选） | 冻雨 Recall 对比 | M3.2 |
| M3.4 ONNX 导出 + TensorRT FP16 基准测试 | `deployment/benchmark.py` 输出延迟/吞吐 | M2.4 |
| M3.5 INT8 量化校准（若 FP16 仍不满足延迟约束） | 量化模型 + 精度损失报告 | M3.4 |

**关键难点**：INT8 量化对覆冰边缘的 IoU 影响需实测；若 IoU 下降超过 2 个点，回退 FP16 或对分割头保持 FP16 混合精度。

### 里程碑 M4：系统集成与测试（第 10—12 周）

**目标**：形成可交付的推理服务原型，完成集成测试与文档。

| 任务 | 产出 |
|---|---|
| M4.1 `predictor.py` 端到端推理：图像 → mask + 类型 + 严重等级 + 降水类型 |
| M4.2 FastAPI 最小服务封装（可选） |
| M4.3 集成测试：真实电网图像样例集上的端到端指标 |
| M4.4 文档：README、标注指南、部署指南、严重程度分级 rubric |


## 六、数据与模型训练策略

### 6.1 数据分层与标注优先级

| 数据用途 | 来源 | 规模预期 | 优先级 |
|---|---|---|---|
| 覆冰分割冷启动 | firc-dataset（1138 + 1227 + 109） | ~2500 张 | P0 |
| 架构验证/对比 | DTL-IceNet Zenodo | 按论文规模 | P1 |
| 降水类型预训练 | Weather-Image-Classification / dima806 | 预训练权重 | P0 |
| 覆冰类型细粒度 | 自建标注 + dima806 伪标签 | 300—500 精标 + 伪标签扩充 | P0（关键路径） |
| 冻雨补充/域适应 | CoInfra | 按需采样 | P2 |
| 绝缘子 ROI 辅助 | CPLID（可选） | 作为前置定位模块训练 | P2 |

### 6.2 类别不平衡处理

冻雨、冰雹在自然图像中极稀有。策略按优先级：

1. **重采样**：在 batch 级别对稀有类别过采样，配合场景分层采样器避免过拟合到同一背景。
2. **Logit-adjusted CE / Class-Balanced CE**：直接修正分类头的 logit 偏置，比简单类别权重更稳定。
3. **伪标签 + 蒸馏**：用 dima806 的 `hail` / `glaze` 类别头对无标注数据生成伪标签，以温度缩放的软标签蒸馏，缓解硬标签噪声。

### 6.3 评估协议

建立**三套评估集**，分别报告：

- **Set-A（公开数据测试集）** ：firc / Zenodo 的留出集，用于与文献基线对比。
- **Set-B（电网域内测试集）** ：从实际场景采集并人工精标的 200—300 张图像，覆盖晴天、恶劣天气、设备背景三类场景。这是**真正的部署决策依据**。
- **Set-C（冻雨专项集）** ：从 CoInfra 或自建冻雨样本中抽取，专门报告冻雨 Recall 和误报率。冻雨漏报的代价远高于一般降水误报，需单独设定阈值并做代价敏感分析。


## 七、测试、评估与部署建议

### 7.1 测试体系

- **单元测试**：数据转换（labelme→COCO 的坐标一致性）、loss 的数值稳定性（Ordinal Loss 的单调性）、严重程度映射器的边界条件。
- **集成测试**：`test_training_smoke.py` 用 2—4 张图像做 1—2 个 batch 的过拟合测试，确保 loss 可下降、梯度流无断裂；`test_export_onnx.py` 验证 PyTorch 与 ONNX 输出的数值一致性（atol ≤ 1e-4）。
- **回归测试**：每次修改数据管线或 loss 后，在 Set-A 的小型固定子集上运行评估，与已记录的基线对比，防止静默退化。

### 7.2 部署建议

- **边缘优先路径**：Jetson Orin Nano/NX + TensorRT FP16。若分割分支在 640×640 下延迟超过 80 ms，降至 512×512 或使用动态分辨率（分类分支可接受更低分辨率）。
- **推理服务接口**：单图接口返回 `{ice_mask_rle, ice_coverage, ice_type, severity_level, precipitation_type, confidence}`；批量接口支持异步队列。
- **监控与迭代**：记录边缘端推理的置信度分布和类别预测分布；当冻结类别（如冻雨）的预测频率显著偏离训练分布时，触发样本回传和增量标注。


## 八、风险与后续优化方向

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| **冻雨/冰雹精标数据严重不足** | 降水类型分类在关键类别上不可靠 | 以 dima806 伪标签 + 小规模精标起步；主动学习优先标注模型不确定的冻雨候选样本；报告中单独披露冻雨 Recall 而非仅整体 accuracy |
| **firc-dataset 可获取性/许可不明确** | M1 数据管线受阻 | 立即以 Zenodo DTL-IceNet 数据作为备选；同时启动自建数据采集 |
| **覆冰类型标签定义与电网运维口径不一致** | 模型输出无法直接映射到处置决策 | 在 M2 之前与领域专家确认 `severity_rubric.md`，将模型等级与运维的“观察/预警/处置”阈值显式对齐 |
| **单目图像绝对厚度不可靠** | 直接回归厚度会产生误导性输出 | 采用覆盖率/形态分级而非绝对厚度；将“厚度估计”明确标注为相对指标，或仅在双目/标定场景下启用 |
| **边缘 INT8 量化损伤覆冰边缘** | 小目标/薄冰的分割质量下降 | 优先 FP16；若必须 INT8，分割头保持 FP16 混合精度，或采用量化感知训练 |

**后续优化方向**：

1. **时序融合**：单帧图像难以区分“正在覆冰”与“已覆冰但未增长”。若现场有固定摄像头，引入短时序（间隔 10—30 分钟的多帧）用轻量时序模块（如 TSM / 1D Conv）提升严重程度趋势判断的可靠性。
2. **多模态辅助**：融合微气象数据（温度、风速、湿度）作为分类分支的辅助输入，尤其对冻雨 vs. 雨的区分有物理约束价值；这与 DTL-IceNet 融合气象数据校正厚度的思路一致。
3. **主动学习闭环**：将边缘端低置信度样本和冻雨候选样本回传，形成“部署→挖掘→标注→再训练”的闭环，是解决稀有类别数据瓶颈的最可持续路径。


## 九、推荐技术栈与工具链

| 层面 | 推荐 | 理由 |
|---|---|---|
| 环境管理 | **uv**（`uv sync` / `uv pip`） | 快速解析、lockfile 可复现，符合要求 |
| 深度学习框架 | **PyTorch ≥ 2.4** + **timm** | timm 提供 MobileNetV4 / RepViT / EfficientViT 等轻量 backbone 的统一接口 |
| 预训练模型 | **Hugging Face transformers** | Weather-Image-Classification 和 dima806 均通过 `AutoImageProcessor` 加载，迁移成本低 |
| 数据增强 | **Albumentations** | 电力图像常用的色彩/噪声/模糊增强实现成熟；支持同步 image-mask 变换 |
| 配置管理 | **Pydantic + YAML** | schema 校验防止配置错误；与训练脚本解耦 |
| 导出与部署 | **ONNX → TensorRT**；Jetson Orin | 与已有电力巡检边缘部署工作一致，FP16/INT8 链路成熟 |
| 实验追踪 | **TensorBoard**（起步），后续可接 W&B | 轻量，不强制外部服务 |
| CI | **GitHub Actions**：ruff + mypy + pytest（unit + smoke） | 在无 GPU 的 runner 上跑单元测试和 ONNX 导出数值一致性检查 |
| 代码质量 | **ruff**（lint + format）+ **mypy** + **pre-commit** | 统一风格，减少 review 摩擦 |


**一句话执行建议**：先以 firc-dataset + MobileNetV4/LR-ASPP 在 2—3 周内做出覆冰分割 MVP，同时并行启动覆冰类型/冻雨的小规模精标；分类分支从 dima806 迁移并只替换分类头，用伪标签蒸馏撑过数据稀疏期；边缘部署从第一版模型就打通 ONNX→TensorRT 链路，避免后期返工。