# 电网覆冰 / 降水类型与程度识别算法 —— 项目实现计划

> **说明（重要，先声明）**：本计划中的开源调研基于我可核实的公开技术知识（论文、知名GitHub仓库、主流数据集），而非本次会话中实时联网抓取的结果。所有列出的仓库/数据集均为我有较高把握真实存在的资源，但**在项目 Phase 0 中必须安排"链接与可用性复核"任务**（见路线图 0.1），由工程团队用 GitHub/HF/PWC 搜索 + 实际 clone 验证后再写入正式技术选型文档，避免因信息滞后导致选型失误。本计划不虚构任何仓库名称或链接路径描述之外的细节。

---

## 一、项目概述与目标

### 1.1 业务背景
电网输电线路、杆塔、变电站设备在冬季/雨雪天气下易发生导线覆冰、绝缘子污闪、微气象灾害，严重时导致倒塔断线。现有手段（融冰传感器、气象站、人工巡视）存在**覆盖密度低、成本高、实时性差**等问题。基于现场监控摄像头 / 无人机巡检图像的视觉识别，可作为低成本、高密度的补充监测手段。

### 1.2 目标定义
构建一套**多任务图像识别系统**，输入为固定监控摄像头 / 无人机巡检拍摄的导线、绝缘子、杆塔图像（单帧或视频流），输出：

| 任务 | 输出 | 类型 |
|---|---|---|
| T1 覆冰检测 | 有/无覆冰；覆冰区域分割掩码（导线/绝缘子表面） | 二分类 + 语义分割 |
| T2 覆冰程度分级 | 等效覆冰厚度估计（mm）→ 分级（无/轻微/中等/严重/特重，参照 DL/T 5440、GB/T 等电力行业覆冰等级标准） | 有序回归/分级 |
| T3 降水类型识别 | 无降水/雨/雪/冻雨（雨凇）/雾凇/冰雹/雾 等 | 多分类 |
| T4 降水强度分级 | 小/中/大（或轻度/中度/重度） | 有序分类 |
| T5（可选）场景与部件定位 | 导线、绝缘子串、杆塔本体、金具 ROI 检测 | 目标检测 |
| T6（可选）置信度与可解释输出 | Grad-CAM 热力图、不确定性分数、低置信度告警 | 可解释性 |

### 1.3 评估指标
- 分类（T3）：Accuracy、Macro-F1、每类 Recall（尤其冻雨/冰雹等小样本类）、混淆矩阵
- 有序分级（T2/T4）：MAE、QWK（Quadratic Weighted Kappa）
- 分割（T1）：IoU、Dice、边界 F1
- 检测（T5）：mAP@0.5、mAP@0.5:0.95
- 系统级：端侧推理延迟（ms/帧）、模型体积（MB）、误报率（FPR，关乎是否误触发停电/人工巡检）、恶劣光照/雾天鲁棒性下降幅度
- 业务级：与气象站实测/人工巡视记录的一致率（现场验收指标）

### 1.4 关键约束（电网场景特殊性）
- **轻量化/实时性**：边缘监测终端算力有限（嵌入式 NPU/Jetson 级），需支持 INT8 量化与 <200ms 推理
- **鲁棒性**：低照度、雾霾、镜头积水/积雪遮挡、逆光
- **小样本/长尾**：冰雹、冻雨等极端天气图像稀缺
- **可解释性**：电网调度/运维决策需要可追溯依据（分割掩码、热力图优于黑盒分数）
- **误报代价高**：需设置置信度阈值与人工复核机制

---

## 二、开源调研报告与借鉴方案

### 2.1 调研范围与方法
检索关键词覆盖：`ice accretion detection`、`icing severity classification`、`transmission line icing`、`insulator defect dataset`、`precipitation/weather image classification`、`rain/snow density estimation`、`snow removal mask estimation`、`power line UAV inspection dataset`。检索渠道：GitHub、Hugging Face Hub/Datasets、Papers with Code 任务页（Weather Classification / Rain Removal / Snow Removal）。

**调研结论（宏观）**：
1. **直接针对"电网导线覆冰"的开源代码+数据集组合极为稀缺**（多数相关研究发表于中文电力期刊，未公开代码/数据），这是本项目的主要数据缺口，需通过迁移学习+合成数据+行业合作弥补。
2. **绝缘子/杆塔部件检测**方向开源资源相对成熟，可作为**部件定位（T5）与骨干网络预训练**的基础。
3. **通用天气分类 / 雨雪去除（image deraining/desnowing）**方向资源丰富，其中"强度分级子任务"（密度感知）的建模思路**可直接迁移**到本项目的降水/覆冰严重度分级任务。
4. **目标检测/分割基础设施**（Ultralytics 系列）生态完善，自带 ONNX/TensorRT 导出链路，适合工程落地。

### 2.2 推荐项目清单（8 项）

| # | 项目/资源 | 核心思路 | 可借鉴点 | 局限/注意事项 |
|---|---|---|---|---|
| 1 | **DID-MDN**（CVPR2018，`hezhangsprinter/DID-MDN`） | 密度感知单图去雨网络，内置"雨密度分类子网络"（轻/中/重三级） | **直接复用"密度分类头"设计思路**用于降水/覆冰严重度三级或五级分级；residual-aware 结构可借鉴多任务分支设计 | 代码基于较旧的 PyTorch 版本，需重写；仅雨图像，无冰/雪 |
| 2 | **DerainZoo / Awesome-Deraining / Awesome-Desnowing 等curated list 仓库** | 汇总去雨/去雪 SOTA 论文与代码实现索引 | 作为**持续追踪 SOTA** 的入口，快速定位带"强度估计/雪掩码估计"能力的模型（如 DesnowNet、JSTASR、HDCWNet 等） | 本身非模型，需二次筛选；更新及时性需核实 |
| 3 | **DesnowNet**（Liu et al., ECCV workshop/TIP） | 模块化去雪网络，显式输出**雪覆盖透明度掩码**再做去除 | **雪/冰覆盖区域分割掩码的建模方式**可直接迁移为"导线表面覆冰覆盖率估计"的分割分支设计 | 原任务是图像复原而非检测分级，需改造输出头为分割+分级 |
| 4 | **CPLID（Chinese Power Line Insulator Dataset）**（`InsulatorData/InsulatorDataSet`） | 正常/缺陷（掉盖）绝缘子图像+标注，Faster R-CNN 基线 | **电网领域真实图像域数据**，用于骨干网络领域预训练（domain pretraining），缓解自然图像→电网图像的域差异 | 仅覆盖绝缘子缺陷，无覆冰/天气标签，需二次标注扩展 |
| 5 | **InsPLAD**（电力资产巡检 UAV 数据集与基准，`andreluizbvs/InsPLAD` 及对应论文） | 大规模无人机电力资产图像，含多类部件与缺陷（锈蚀等），YOLO 系基线 | 提供**部件检测（绝缘子串/金具/杆塔）预训练权重与标注规范**范式，可复用其标注 schema 设计思路 | 不含天气/覆冰类别；需二次采集+标注覆冰样本叠加 |
| 6 | **道路/地表状态分类相关数据集与仓库（Road Surface / Ice-Snow-Wet-Dry Classification 系列）** | 自动驾驶领域的路面状态分类（干/湿/雪/冰），CNN 基线方案 | 作为**"冰面纹理"迁移学习源域**，弥补导线覆冰数据稀缺问题（纹理级别的冰/霜/雪区分有共性） | 应用场景（路面 vs 圆柱形导线）差异大，仅作预训练/辅助，不能直接套用 |
| 7 | **多类天气图像分类数据集（Multi-class Weather Dataset 及扩展版本，如含11类的天气图像识别数据集，涵盖雨/雪/雾凇/雨凇/冰雹等）** | 公开天气图像分类数据（Kaggle等平台），常配 ResNet/EfficientNet 迁移学习基线 | **降水类型分类任务（T3）主干训练数据**，覆盖类别较全，可直接做迁移学习+电网图像域微调 | 多为非电网场景（天空、路面、自然景观），需域适应；类别体系需与电力行业标准对齐 |
| 8 | **Ultralytics YOLOv8/YOLO11**（`ultralytics/ultralytics`） | 一体化检测/分割/分类框架，工程化程度高 | **T5 部件检测 + 覆冰区域分割**的工程底座；自带 ONNX/TensorRT/OpenVINO/NCNN 导出与量化工具链，便于边缘部署 | 通用框架需结合上述专用数据/头部做二次开发，非专用覆冰模型 |

### 2.3 二次开发策略总结
1. **骨干预训练**：用 CPLID + InsPLAD 做电网图像域的自监督/监督预训练（弥补 ImageNet→电网图像域差），再在覆冰/天气任务上微调。
2. **分级建模范式迁移**：借鉴 DID-MDN 的"密度分类子网络"思路，设计覆冰/降水"有序分级头"（Ordinal Head / CORAL Loss）。
3. **分割建模范式迁移**：借鉴 DesnowNet 的透明度/覆盖掩码思路，设计"导线覆冰覆盖率分割分支"，结合几何标定换算等效覆冰厚度。
4. **部件定位复用**：用 Ultralytics 框架 + InsPLAD/CPLID 预训练权重，快速构建绝缘子/导线 ROI 检测器，为后续覆冰分析提供裁剪区域，降低背景干扰。
5. **数据缺口弥补**：天气分类数据集解决"降水类型"数据冷启动；覆冰专项数据通过合成（物理渲染+风格迁移）与现场采集补齐。
6. **部署链路复用**：直接采用 Ultralytics 自带导出/量化工具链，减少自研边缘部署代码量。

---

## 三、系统架构与算法设计

### 3.1 总体技术路线（两阶段级联 + 多任务头）

```
                      ┌─────────────────────────────┐
输入图像/视频帧 ───▶  │ Stage 1: 场景与部件检测 (T5)  │
(含EXIF/时间/点位元数据)│  YOLO系检测器 (导线/绝缘子/杆塔ROI) │
                      └───────────────┬─────────────┘
                                      │ ROI裁剪 + 原图（场景级）
                      ┌───────────────▼─────────────┐
                      │ Stage 2: 多任务特征骨干        │
                      │  (EfficientNet-Lite/MobileNetV3/│
                      │   ConvNeXt-Tiny 可切换)        │
                      └──┬───────┬────────┬──────────┘
            ┌────────────┘       │        └────────────┐
            ▼                    ▼                      ▼
   ┌─────────────────┐  ┌─────────────────┐   ┌──────────────────┐
   │ Head-A 降水类型   │  │ Head-B 覆冰分割   │   │ Head-C 严重度分级  │
   │ (多分类 softmax)  │  │ (DeepLabV3+/UNet)│   │ (有序头 CORAL/序  │
   │ 雨/雪/冻雨/冰雹/雾/无│  │ 导线/绝缘子覆冰掩码│   │ 回归, 降水强度+覆冰│
   └─────────────────┘  └────────┬────────┘   │ 等级联合输出)      │
                                  │             └──────────────────┘
                                  ▼
                      ┌─────────────────────────┐
                      │ 几何标定模块              │
                      │ 覆盖率像素→mm等效覆冰厚度  │
                      │ (已知导线/绝缘子直径先验)  │
                      └───────────┬─────────────┘
                                  ▼
                      ┌─────────────────────────┐
                      │ 后处理与可解释性           │
                      │ Grad-CAM / 不确定性估计    │
                      │ (MC-Dropout / 多次推理方差)│
                      └───────────┬─────────────┘
                                  ▼
                        结构化结果 JSON + 可视化叠加图
                        （供告警平台/GIS/调度系统消费）
```

### 3.2 模型选型（分档）

| 部署形态 | 骨干网络 | 检测器 | 分割头 | 说明 |
|---|---|---|---|---|
| 云端/训练侧（高精度基准） | ConvNeXt-Tiny / Swin-Tiny | YOLO11-m | DeepLabV3+ (ResNet50) | 作为教师模型，定期离线复核与模型蒸馏源 |
| 边缘终端（Jetson Orin Nano / RK3588 等国产NPU常见于电网监测终端） | MobileNetV3-Small / EfficientNet-Lite0 | YOLO11-n | 轻量UNet（深度可分离卷积） | INT8量化、知识蒸馏学生模型 |

### 3.3 关键算法难点与应对

| 难点 | 应对策略 |
|---|---|
| 覆冰/冻雨/冰雹图像样本稀缺 | ① 公开天气数据集迁移学习 ② 物理渲染合成覆冰（圆柱导线+冰层几何建模贴图）③ CycleGAN/风格迁移将晴天导线图像转为覆冰图像 ④ 半监督/伪标签（FixMatch 思路）利用大量无标注现场图像 |
| 等效覆冰厚度的几何标定 | 利用已知导线/绝缘子直径作为参考尺度，结合相机内参（若有）或透视变换估算像素-毫米比例，回归覆冰层厚度；对无标定相机场景采用相对覆盖率分级（代替绝对厚度） |
| 类别不均衡（冰雹/冻雨少） | Focal Loss / Class-Balanced Loss、过采样+合成数据、分层验证集评估 |
| 低照度/雾天鲁棒性 | 训练期加入亮度/对比度/雾霾合成增强；引入图像质量评估前置模块，低质量图像触发二次采集或降权处理 |
| 误报代价高 | 置信度阈值+不确定性估计（MC-Dropout 方差或模型集成分歧度），低置信度样本路由至人工复核队列；多帧时序平滑（避免单帧噪声误判） |
| 边缘算力有限 | 知识蒸馏（云端ConvNeXt教师→边缘MobileNet学生）、结构化剪枝、INT8量化（TensorRT/OpenVINO/NCNN） |
| 多任务学习冲突（类型/分级/分割梯度干扰） | 采用 GradNorm 或不确定性加权（Kendall多任务损失加权）平衡各头损失；必要时分阶段训练（先预训练骨干+检测，再联合微调多头） |

### 3.4 数据与标注规范要点
- **类别体系对齐电力行业标准**：降水类型（无/雨/雪/冻雨/雾凇/冰雹/雾）；覆冰等级参照行业覆冰等级（如等效覆冰厚度分档：<5mm 无/轻微，5-10mm 轻度，10-20mm 中度，20-30mm 重度，>30mm 特重，具体阈值以项目落地地区现行标准/规程为准，需与电网客户现场规程核对）。
- **标注内容**：图像级标签（降水类型、强度）+ 区域级标注（覆冰区域多边形/掩码）+ 部件级框（导线/绝缘子/杆塔）。
- **标注工具**：建议 CVAT / Label Studio（支持多任务标注：分类+分割+检测一体化）。
- **质检机制**：双人标注+抽检一致性（Kappa系数）、定期与现场运维人员联合复核边界样本（如"薄冰 vs 水渍反光"易混淆类）。

---

## 四、详细文件夹结构

> 包名建议：`gridvision`（避免过于宽泛的 `src`/`utils` 作为包名）。遵循 Python **src-layout**，嵌套深度控制在 3-4 层以内，命名统一 snake_case。

```
gridvision/                                # 项目根目录（repo name 用 kebab-case: gridvision-icing-weather）
├── README.md                              # 项目简介、快速开始、架构图索引
├── LICENSE
├── .gitignore                             # 忽略 data/raw, models/*.pt, .venv 等
├── .pre-commit-config.yaml                # ruff/mypy/格式化 hook
├── pyproject.toml                         # uv 管理依赖、构建配置、ruff/mypy/pytest 配置
├── uv.lock                                # uv 锁定文件（提交到仓库）
├── justfile                               # 常用命令集合（替代 Makefile，跨平台更友好）
├── AGENTS.md                              # 【AI Agent 专用】目录约定、禁止事项、生成代码规范
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                         # lint + unit test + 覆盖率
│   │   ├── model-eval.yml                 # 定期在验证集跑评估，防性能回退
│   │   └── release.yml
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── docs/
│   ├── README.md                          # 文档导航
│   ├── architecture.md                    # 系统架构详述（对应第三节内容落地版）
│   ├── data_spec.md                       # 数据字段/标注格式/类别体系定义
│   ├── annotation_guideline.md            # 标注员操作手册
│   ├── model_cards/                       # 每个发布模型的模型卡（版本、指标、适用范围、已知局限）
│   │   ├── precip-classifier-v1.md
│   │   └── ice-severity-v1.md
│   ├── api.md                             # 推理服务 API 文档
│   └── deployment.md                      # 边缘/云端部署手册
│
├── configs/                               # 所有可配置项集中管理（Hydra/OmegaConf 风格 yaml）
│   ├── README.md
│   ├── data/
│   │   ├── base.yaml
│   │   └── augmentation.yaml
│   ├── model/
│   │   ├── precip_classifier_effnet.yaml
│   │   ├── ice_seg_deeplabv3plus.yaml
│   │   └── asset_detector_yolo.yaml
│   ├── train/
│   │   ├── precip_train.yaml
│   │   ├── ice_seg_train.yaml
│   │   └── multitask_finetune.yaml
│   └── deploy/
│       ├── edge_jetson_int8.yaml
│       └── cloud_server.yaml
│
├── data/                                  # 数据目录（大部分内容 .gitignore，用 .gitkeep 占位）
│   ├── README.md                          # 数据获取方式、目录约定、DVC/下载脚本说明
│   ├── raw/            .gitkeep           # 原始采集/下载数据，不入库
│   ├── interim/        .gitkeep           # 清洗中间产物
│   ├── processed/      .gitkeep           # 训练就绪数据（含 train/val/test split）
│   ├── external/       .gitkeep           # 公开数据集（CPLID/InsPLAD/天气分类数据集等）
│   └── annotations/    .gitkeep           # CVAT/Label Studio 导出标注文件
│
├── notebooks/
│   ├── README.md
│   ├── 00_eda_public_datasets.ipynb
│   ├── 01_ice_segmentation_prototype.ipynb
│   └── 02_precip_classifier_baseline.ipynb
│
├── scripts/                               # 一次性/辅助脚本（非包内核心逻辑）
│   ├── README.md
│   ├── download_public_datasets.sh
│   ├── export_onnx.py
│   ├── benchmark_edge_inference.py
│   ├── generate_synthetic_ice.py          # 物理渲染/风格迁移合成覆冰样本
│   └── convert_labelstudio_to_coco.py
│
├── src/
│   └── gridvision/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py                  # pydantic 配置模型（类型安全）
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── precip_dataset.py
│       │   │   ├── ice_seg_dataset.py
│       │   │   └── asset_detect_dataset.py
│       │   ├── transforms/
│       │   │   ├── augment_weather.py     # 雨雾雪合成增强
│       │   │   └── normalize.py
│       │   └── synthetic/
│       │       ├── ice_render.py          # 圆柱导线覆冰物理建模渲染
│       │       └── style_transfer.py      # CycleGAN域迁移
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbones/
│       │   │   └── builder.py
│       │   ├── heads/
│       │   │   ├── classification_head.py
│       │   │   ├── ordinal_severity_head.py
│       │   │   └── segmentation_head.py
│       │   ├── multitask_net.py
│       │   └── asset_detector.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── losses.py                  # Focal/CORAL/多任务加权损失
│       │   └── metrics.py                 # QWK/IoU/mAP 等实现
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── pipeline.py                # 端到端推理编排
│       │   ├── severity_mapping.py        # 像素→毫米等效厚度换算
│       │   └── explain.py                 # Grad-CAM / 不确定性
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── api.py                     # FastAPI 服务
│       │   └── schemas.py                 # 请求/响应模型
│       ├── edge/
│       │   ├── __init__.py
│       │   ├── export.py                  # ONNX/TensorRT/OpenVINO 导出
│       │   └── quantize.py                # INT8量化、蒸馏入口
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── io.py
│           └── visualization.py
│
├── tests/
│   ├── README.md
│   ├── unit/
│   │   ├── test_transforms.py
│   │   ├── test_severity_mapping.py
│   │   └── test_models.py
│   ├── integration/
│   │   └── test_inference_pipeline.py
│   └── e2e/
│       └── test_api.py
│
└── models/                                 # 模型权重登记（配合 DVC/Git-LFS，二进制不直接提交git）
    ├── README.md                           # 各版本权重清单、下载/校验脚本说明
    └── .gitkeep
```

### 4.1 Python 包结构与命令约定

**pyproject.toml（uv 管理）关键点**：
```toml
[project]
name = "gridvision"
requires-python = ">=3.10"
dependencies = [
  "torch", "torchvision", "ultralytics", "opencv-python-headless",
  "albumentations", "pydantic", "fastapi", "uvicorn", "onnx", "onnxruntime",
  "hydra-core", "scikit-learn", "numpy", "loguru"
]

[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit", "jupyter"]

[tool.ruff]
line-length = 100
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**uv 常用命令**：
```bash
uv venv                      # 创建虚拟环境（.venv）
uv sync                      # 按 pyproject.toml/uv.lock 同步依赖
uv add torch ultralytics     # 新增依赖
uv add --group dev pytest ruff mypy
uv run pytest -q             # 在受控环境中运行测试
uv run python scripts/export_onnx.py
```

**justfile 示例**：
```just
setup:
    uv venv && uv sync

lint:
    uv run ruff check . && uv run mypy src

test:
    uv run pytest --cov=src/gridvision tests/

train-precip:
    uv run python -m gridvision.training.trainer --config configs/train/precip_train.yaml

train-ice-seg:
    uv run python -m gridvision.training.trainer --config configs/train/ice_seg_train.yaml

export-onnx:
    uv run python scripts/export_onnx.py --config configs/deploy/edge_jetson_int8.yaml

serve:
    uv run uvicorn gridvision.serving.api:app --host 0.0.0.0 --port 8080
```

### 4.2 组织原则落实清单
- ✅ 根目录仅保留元信息文件，无散落脚本/代码
- ✅ Feature-based 为主（`data/`, `models/`, `training/`, `inference/`, `serving/`, `edge/` 按功能垂直切分），每个子功能内部再按 Domain 细分（datasets/transforms/synthetic）
- ✅ 命名统一 snake_case（代码/目录），仓库名可用 kebab-case
- ✅ 嵌套深度 ≤4 层（`src/gridvision/data/datasets/xxx.py` 为4层）
- ✅ 每个一级/关键二级目录含 README.md；空目录用 `.gitkeep`
- ✅ `AGENTS.md` 明确告知任何 AI 代码生成工具：**不得在根目录新建代码文件、不得引入与 `gridvision` 平级的第二套包结构、configs 变更需同步更新对应 docs/model_cards**
- 🚫 **禁止的噪音目录**：根目录下 `test/`（应为 `tests/`）、`utils/`（散装工具应归入 `src/gridvision/utils/`）、`output/`、`tmp/`、`new/`、`final/`、`bak/` 等无语义目录一律禁止提交

---

## 五、分阶段开发计划

### Phase 0：调研复核与环境搭建（1-2周）
- 0.1 **复核第二节调研清单的真实性与最新状态**（实际 clone/访问验证链接、License、维护活跃度），形成正式《技术选型确认表》
- 0.2 用 uv 初始化工程骨架（按第四节目录结构），配置 pre-commit/CI
- 0.3 下载公开数据集（天气分类数据集、CPLID、InsPLAD），完成 EDA notebook
- 0.4 制定标注规范（docs/annotation_guideline.md）与类别体系，与电网业务方（如有）对齐覆冰等级定义
- **里程碑 M0**：工程骨架可运行（`just setup && just test` 通过），EDA 报告产出

### Phase 1：MVP —— 单任务基线打通（3-4周，依赖 M0）
- 1.1 降水类型分类基线：公开天气数据集 + EfficientNet-Lite 迁移学习（T3）
- 1.2 部件检测基线：Ultralytics YOLO + CPLID/InsPLAD 预训练微调（T5）
- 1.3 覆冰分割基线：在合成数据（ice_render.py 物理渲染）+ 少量现场样本上训练轻量 UNet（T1/T2 雏形）
- 1.4 打通端到端推理脚本（`inference/pipeline.py`），输出结构化 JSON + 可视化叠加图
- **里程碑 M1**：三个独立模型可分别推理，CLI 工具可跑通单图 demo

### Phase 2：多任务融合与严重度分级（4-6周，依赖 M1）
- 2.1 设计多任务骨干+多头网络（multitask_net.py），实现多任务加权损失
- 2.2 实现有序分级头（CORAL/Ordinal）用于覆冰/降水强度分级（T2/T4）
- 2.3 几何标定模块：像素覆盖率→等效覆冰厚度换算（severity_mapping.py），需标定样本验证
- 2.4 合成数据管线完善：CycleGAN风格迁移 + 物理渲染批量生成，建立半监督伪标签循环
- 2.5 构建评估体系（QWK/IoU/mAP/混淆矩阵自动化报告）
- **里程碑 M2**：多任务模型在验证集上达到预设基线指标（如 T3 Macro-F1≥0.75，T1 IoU≥0.6，作为内部目标值，需按实际数据重新标定）

### Phase 3：鲁棒性增强与可解释性（3周，依赖 M2）
- 3.1 低照度/雾天数据增强与专项验证集构建
- 3.2 不确定性估计（MC-Dropout）+ Grad-CAM 可解释输出
- 3.3 误报分析与 Focal Loss/类别重加权调优，建立人工复核队列机制设计
- **里程碑 M3**：恶劣条件验证集性能下降幅度控制在可接受范围（需与基线对比量化）

### Phase 4：边缘部署与系统集成（4周，依赖 M3）
- 4.1 知识蒸馏（教师ConvNeXt→学生MobileNet）
- 4.2 ONNX导出 → TensorRT/OpenVINO/NCNN量化（INT8），边缘设备基准测试（benchmark_edge_inference.py）
- 4.3 FastAPI 服务封装（serving/api.py）+ 容器化（Dockerfile，若纳入根目录需放入 `deploy/` 或 `.docker/` 子目录，不散落根目录）
- 4.4 与现场平台对接（告警推送格式、GIS坐标关联、与气象站/传感器数据融合接口预留）
- **里程碑 M4**：边缘设备端到端推理延迟达标（目标 <200ms/帧，视具体硬件调整），具备小规模试点部署能力

### Phase 5：试点验证与迭代（持续）
- 5.1 现场小规模试点，收集真实误报/漏报案例
- 5.2 主动学习闭环：低置信度/人工修正样本回流训练集
- 5.3 定期模型版本发布与 model_cards 更新、CI 中加入性能回归监控（model-eval.yml）

**依赖关系总览**：Phase 0 → Phase 1（硬依赖）→ Phase 2（依赖1的基线模型与数据）→ Phase 3（依赖2的多任务模型）→ Phase 4（依赖3的鲁棒模型）→ Phase 5（依赖4的可部署系统）。Phase 2 的合成数据工作可与 Phase 1 并行启动以节省时间。

---

## 六、数据与模型训练策略

### 6.1 数据来源矩阵

| 数据类型 | 来源 | 用途 |
|---|---|---|
| 通用天气图像 | 公开多类天气/天气现象分类数据集 | T3 降水类型主干训练、预训练 |
| 绝缘子/电力资产图像 | CPLID、InsPLAD | T5 部件检测预训练、领域适应 |
| 路面/地表冰雪纹理图像 | 路面状态分类相关公开数据 | 冰雪纹理迁移学习辅助 |
| 合成覆冰图像 | 自研物理渲染（圆柱导线+冰层几何） + GAN风格迁移 | 弥补覆冰专项数据缺口，T1/T2训练 |
| 现场采集图像（核心资产） | 电网监控摄像头/无人机历史影像（需业务方提供） | 最终域适应微调、验收评估的"金标准"数据 |
| 气象站/传感器元数据（可选融合） | 温湿度、风速、历史降水记录 | 多模态融合，降低纯视觉误判率（后续优化方向） |

### 6.2 训练策略
1. **分阶段迁移学习**：ImageNet预训练 → 电网图像域（CPLID/InsPLAD）继续预训练 → 任务头微调
2. **合成+真实混合训练**：合成数据提供结构多样性，真实数据保证域真实性，按比例（如合成:真实=3:1起步）逐步提高真实数据占比
3. **半监督/主动学习**：对现场大量无标注图像用伪标签+置信度过滤迭代扩充训练集（FixMatch风格一致性正则）
4. **多任务损失加权**：采用不确定性加权（Kendall et al.）或 GradNorm，避免分割任务梯度主导分类任务
5. **有序分级建模**：覆冰/降水强度采用 CORAL（Consistent Rank Logits）等有序分类方法，优于直接多分类，能更好反映等级间的序关系

### 6.3 数据/模型版本管理
- 建议引入 **DVC 或 Git-LFS** 管理 `data/processed`、`models/*.onnx|*.pt`，避免大文件入 Git 主仓库
- 每次正式训练产出需登记到 `docs/model_cards/`：版本号、训练数据快照、指标、已知局限、适用场景边界

---

## 七、测试、评估与部署建议

### 7.1 测试体系
- **unit**：数据变换正确性、几何标定换算公式、损失函数数值验证
- **integration**：端到端推理管线（检测→分割→分级→输出JSON格式校验）
- **e2e**：FastAPI 接口契约测试（输入图像→标准输出 schema）
- **性能回归（CI）**：固定小型验证集，每次模型更新自动跑指标，防止精度回退（model-eval.yml）

### 7.2 评估建议
- 建立**专项恶劣天气/低照度子验证集**，单独报告指标，避免被整体均值掩盖短板
- 引入**现场人工标注复核**作为最终验收基准（而非仅依赖自建测试集）
- 关注**长尾类别（冰雹/冻雨）的 Recall**，优先保障"漏报率"低于"误报率"（电网安全场景漏报代价更高）

### 7.3 部署建议
- **边缘端**：ONNX → TensorRT（Jetson系列）/ OpenVINO（x86边缘盒子）/ NCNN或RKNN（国产NPU如RK3588），INT8量化+蒸馏学生模型，单帧推理目标 <200ms
- **云端**：FastAPI + 批量推理服务，承担教师模型周期性复核、模型再训练闭环
- **容器化**：Docker多阶段构建，区分训练镜像（含CUDA/完整依赖）与推理镜像（精简依赖，体积优化）
- **监控**：接入现场后建立模型性能/漂移监控（输入分布漂移检测、置信度分布监控），触发再训练告警

---

## 八、风险与后续优化方向

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 覆冰专项真实数据长期不足 | 模型泛化性差、现场效果不及预期 | 尽早与电网业务方建立数据采集/标注合作通道；持续扩充合成数据管线 |
| 覆冰厚度几何标定误差 | 严重度分级不准 | 优先在有标定参照物（已知直径导线/绝缘子）场景部署；无法标定场景退化为"相对覆盖率分级"而非绝对毫米值 |
| 多任务学习负迁移 | 单任务性能反而下降 | 采用损失加权策略+消融实验，必要时退回"独立模型级联"架构作为备选方案 |
| 边缘设备算力/功耗限制 | 无法满足实时性 | 分级部署（轻量模型常驻+云端复核），模型蒸馏/剪枝/量化并行推进 |
| 极端天气（冰雹）图像近乎无法获取 | 该类别几乎无法训练 | 明确告知业务方该类别初期以"告警降级+人工确认"策略处理，不追求高自动化精度 |
| 误报引发不必要停电/人工出巡成本 | 运维成本上升、信任度下降 | 引入不确定性阈值+多帧时序确认+人工复核闭环，逐步提升自动化比例 |
| 开源协议合规风险 | 法律/商用限制 | Phase 0 复核阶段逐一核实各借鉴仓库/数据集 License（如是否允许商用、是否需署名），形成合规清单 |

**后续优化方向**：多模态融合（图像+气象站传感器+雷达临近预报数据）、时序视频流分析（覆冰生长趋势预测而非单帧判断）、跨摄像头/跨地域域适应、联邦学习（多电网公司数据不出域协同训练）。

---

## 九、推荐技术栈与工具链

| 类别 | 选型 |
|---|---|
| 语言/环境管理 | Python 3.10+，**uv**（虚拟环境+依赖锁定+运行） |
| 深度学习框架 | PyTorch + torchvision；检测/分割底座 Ultralytics |
| 配置管理 | Hydra / OmegaConf + Pydantic（类型校验） |
| 数据标注 | CVAT 或 Label Studio |
| 数据/模型版本管理 | DVC 或 Git-LFS |
| 实验跟踪 | MLflow 或 Weights & Biases（二选一，视是否允许外网上传选型） |
| 代码质量 | ruff（lint+format）、mypy（类型检查）、pre-commit |
| 测试 | pytest + pytest-cov |
| 服务化 | FastAPI + Uvicorn |
| 边缘部署 | ONNX Runtime / TensorRT / OpenVINO / NCNN or RKNN（按硬件选型） |
| CI/CD | GitHub Actions（lint/test/model-eval 三类workflow） |
| 容器化 | Docker（多阶段构建区分训练/推理镜像） |
| 任务运行脚手架 | justfile（跨平台命令集合） |

---

### 下一步行动建议
1. 先完成 **Phase 0.1 调研复核**（实际检索核验第二节清单），产出正式《技术选型确认表》作为后续采购/集成依据；
2. 并行启动工程骨架搭建（`uv venv && uv sync`）与现场数据采集/合作渠道沟通，二者均为后续所有阶段的前置依赖；
3 若需要，我可以进一步输出：(a) `pyproject.toml` / `justfile` 完整可运行模板，(b) 覆冰物理渲染合成数据的具体算法方案，(c) 多任务网络的 PyTorch 参考代码骨架。