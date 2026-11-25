我将系统检索 GitHub、Hugging Face、Papers with Code 等平台上与覆冰检测、降水类型识别及电网图像分析相关的开源资源，随后据此制定可落地的实现计划。

以下为基于真实可查开源生态的**电网场景覆冰 / 降水类型及程度识别**完整代码项目实现计划。调研结论均来自 GitHub、Zenodo、Harvard Dataverse、Hugging Face、Papers/期刊论文中可访问的资源，避免虚构仓库。

---

# 项目实现计划：电网图像覆冰与降水类型识别系统

## 1. 项目概述与目标

### 1.1 业务目标
面向输电线路、杆塔、变电站固定摄像头 / 无人机巡检图像，实现：
- **覆冰有无与类型**：无冰、雨凇（glaze）、雾凇（rime）、混合凇、积雪等
- **降水 / 天气现象类型**：雨、雪、冻雨、冰雹、雾/霾、霜等
- **严重程度 / 厚度估计**：像素级分割 → 几何换算（或分级：无/轻/中/重）
- 输出可对接电网监测平台的结构化结果（JSON + 可视化叠加）

### 1.2 问题定义与 I/O

| 项目 | 说明 |
|------|------|
| **输入** | RGB 图像（固定点监控 / UAV，分辨率 ≥ 640×640；可扩展多视角） |
| **输出** | ① 场景级：降水/覆冰类型 + 置信度 ② 目标级：导线/绝缘子 bbox 或 mask ③ 程度：厚度估计（mm）或等级 ④ 可选风险等级 |
| **评估指标** | 分类：Accuracy / Macro-F1；检测：mAP@0.5、mAP@0.5:0.95；分割：mIoU、F1、边界 ODS/OIS；厚度：MAPE / MAE；速度：FPS（边缘端） |

### 1.3 工程约束（电网落地）
实时性（边缘 ≥ 10–25 FPS）、鲁棒性（低对比度、雨雾、夜间）、轻量化（INT8 / TensorRT）、可解释性（分割 mask + 厚度计算依据）、可二次开发与领域适应。

---

## 2. 开源调研报告与借鉴方案

### 2.1 调研结论摘要
- **输电线路覆冰专用公开大数据集极少**，多数论文使用自建数据（数百～两千张量级）；可复用的是**方法、模块与少量示例代码/数据**。
- **降水/天气现象分类**有较成熟公开数据与模型，可直接做预训练 / 辅助分支。
- **工程主线推荐**：YOLO 系（检测/实例分割）+ 语义分割轻量骨干（厚度）+ 天气分类头；厚度用“分割像素 + 已知线径”几何换算，而非纯回归黑盒。

### 2.2 推荐高价值资源（3–8 项）

| # | 资源 | 链接 | 核心思路 | 优缺点 | 可复用部分 |
|---|------|------|----------|--------|------------|
| 1 | **DTL-IceNet**（双任务：类型 + 厚度） | 论文 AMT；数据与代码 [Zenodo 15718305](https://zenodo.org/records/15718305) | ResSepNet 多分支抗背景噪声 + MOMSA-SegNet 分割；多尺度融合 + 气象校正厚度 | 专为输电线路设计；示例数据有限；依赖自建标注 | 双任务框架、分割后厚度计算逻辑、部分模块代码 |
| 2 | **YOLO 改进系覆冰检测**（YOLOv5/v8/v11） | 多篇：TLICMNet、HCDFI-YOLOv8、Improved YOLOv5-Seg 等 | 在 Ultralytics YOLO 上加注意力、小目标头、Inner-WIoU 等 | 实时性好；多数无公开权重/完整数据 | 以 **ultralytics** 为基座做二次开发；训练 pipeline 与导出 TensorRT |
| 3 | **WEAPD 天气现象库** | [GitHub haixiaxiao/A-database-WEAPD](https://github.com/haixiaxiao/A-database-WEAPD)；[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M8JQCR) | 11 类：hail / snow / rain / rime / glaze / frost / fog 等 | 与“降水类型+凇类”高度相关；场景偏通用天气 | 预训练分类骨干；类别映射到电网标签体系 |
| 4 | **HF 天气分类模型** | [dima806/weather_types_image_detection](https://huggingface.co/dima806/weather_types_image_detection)；[prithivMLmods/Weather-Image-Classification](https://huggingface.co/prithivMLmods/Weather-Image-Classification) | ViT / SigLIP 等 fine-tune | 即用推理；域差异需适应 | 作为场景分类分支或教师模型 |
| 5 | **电力线部件检测数据集与模型** | [Elizbellou/TIC-Powerline-Dataset](https://github.com/Elizbellou/TIC-Powerline-Dataset-and-Detection-Models)；[Powerline YOLOv5 Jetson](https://github.com/Elizbellou/Powerline-Detection-with-YOLOv5-and-Jetson-Nano) | 杆塔/绝缘子/导线检测 + 边缘部署 | 无直接覆冰标签，但可先检“导线区域再判冰” | 两阶段 pipeline、Jetson/TensorRT 经验 |
| 6 | **恶劣天气电力巡检** | [phd-benel/weather_powerline_insulator](https://github.com/phd-benel/weather_powerline_insulator) | 雨/雪/雾噪声下绝缘子检测 | 强化域泛化思路 | 数据增强与 domain 评估方法 |
| 7 | **边缘厚度分割思路** | GMSA-Net / EECNet / EDPNet / CG-UNet（MDPI 等） | 分割冰区 → 像素统计 + 已知线径算厚度；轻量化 + 量化 | 多数数据集不公开 | 厚度公式、边界损失、边缘部署指标参考 |
| 8 | **天气分类工程示例** | [gkumarg/weather-classification-tf](https://github.com/gkumarg/weather-classification-tf) | MobileNetV2 + TFLite | 简单可跑 | 轻量分类与端侧导出流程 |

### 2.3 二次开发策略
1. **检测/分割主线**：以 **Ultralytics YOLO11/YOLOv8-seg** 为基座 → 类别：导线_无冰、导线_雨凇、导线_雾凇、绝缘子_覆冰、背景干扰等。  
2. **分类辅助**：WEAPD / HF 模型预训练 → 电网监控域微调（场景级降水类型）。  
3. **厚度**：复用 DTL-IceNet / GMSA 思路——分割 mask + 线径标定 → 厚度；可选气象特征做校正。  
4. **数据**：公开数据做预训练 + 电网自有图像强标注 + 合成/CycleGAN 风格迁移（多篇论文已用）。  
5. **部署**：ONNX → TensorRT / OpenVINO；INT8；Jetson / 工控机 / OrangePi 类板卡验证。

---

## 3. 系统架构与算法设计

### 3.1 整体技术方案

```text
图像 → 预处理(去雾/CLAHE/归一化)
      → [可选] 导线/杆塔 ROI 检测 (YOLO)
      → 多任务头:
           A. 场景分类 (降水/天气类型)
           B. 实例/语义分割 (覆冰区域 + 类型)
           C. 厚度估计 (几何 / 回归头)
      → 后处理 (NMS、厚度换算、风险规则)
      → 输出 JSON + 可视化
```

- **MVP**：单模型 YOLO-seg（有无冰 + 粗类型）+ 简单场景分类。  
- **完整版**：检测 ROI + 分割 + 分类多任务；或双阶段（先检导线，再对 ROI 分类/分割）。  
- **边缘**：YOLO-n/s + 量化；云端可用更大 backbone 做离线分析。

### 3.2 模型选型建议
| 任务 | 推荐 | 备选 |
|------|------|------|
| 检测/实例分割 | YOLO11n/s-seg 或 YOLOv8-seg | RT-DETR（精度优先，速度次之） |
| 语义分割（厚度） | MobileOne / 轻量 UNet 变体 / YOLO-seg mask | SegFormer-b0 |
| 场景分类 | EfficientNet-B0 / MobileNetV3 / 微调 WEAPD 模型 | SigLIP 小模型 |
| 多任务 | 共享 backbone + 分类头 + seg 头（参考 DTL-IceNet） | 两阶段级联 |

### 3.3 厚度计算（可解释）
在已知导线直径 \(D\)、无冰像素宽度 \(d_1\)、覆冰后像素宽度 \(d_2\) 时，常用近似：

\[
h \approx \frac{1}{2}\left(\frac{D \cdot d_2}{d_1} - D\right)
\]

（具体以现场标定与论文实现为准；多视角可融合校正。）

---

## 4. 详细文件夹结构（强制 src layout + uv）

```text
grid_ice_precip_vision/
├── README.md
├── LICENSE
├── .gitignore
├── AGENTS.md                 # AI Agent 目录约定
├── pyproject.toml            # uv / 项目元数据与依赖
├── uv.lock
├── Makefile                  # 或 justfile：train / test / export / lint
├── configs/
│   ├── README.md
│   ├── data/
│   │   └── default.yaml
│   ├── model/
│   │   ├── yolo_seg.yaml
│   │   └── classifier.yaml
│   └── train/
│       └── mvp.yaml
├── data/
│   ├── README.md
│   ├── raw/                  # 原始数据（git-ignore 大文件）
│   ├── interim/
│   ├── processed/
│   ├── external/             # WEAPD、TIC 等外部
│   └── .gitkeep
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── annotation_guide.md
│   └── deployment.md
├── notebooks/
│   ├── README.md
│   └── 01_eda.ipynb
├── scripts/
│   ├── README.md
│   ├── download_public_data.py
│   ├── convert_labels.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── infer_demo.py
├── src/
│   └── grid_ice_vision/      # 包名 snake_case
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── transforms.py
│       │   └── thickness.py  # 厚度几何计算
│       ├── models/
│       │   ├── __init__.py
│       │   ├── yolo_wrapper.py
│       │   ├── classifier.py
│       │   └── multitask.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── losses.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   └── postprocess.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           └── viz.py
├── tests/
│   ├── README.md
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── .github/
│   └── workflows/
│       └── ci.yml
└── artifacts/                # 权重、导出模型（gitignore）
    └── .gitkeep
```

**约定要点**
- 根目录干净；代码只在 `src/grid_ice_vision/`。  
- 功能向（data / models / training / inference）+ 必要时分层。  
- 嵌套 ≤ 3–4 层；重要目录有 README；空目录 `.gitkeep`。  
- 命名：snake_case（包/模块）、kebab-case（可选配置文件名）。  
- **禁止**：根目录散落 `.py`、随意 `temp/`、`new_folder/`、把大数据提交进 git。

**uv 管理**
```bash
# 初始化与依赖
uv init --package  # 或手工 pyproject.toml
uv add torch torchvision ultralytics opencv-python-headless pyyaml rich
uv add --dev pytest ruff mypy pre-commit
uv sync
uv run python scripts/train.py --config configs/train/mvp.yaml
```

**Makefile 常用命令示例**
```makefile
sync: ; uv sync
lint: ; uv run ruff check src tests
test: ; uv run pytest tests -q
train-mvp: ; uv run python scripts/train.py --config configs/train/mvp.yaml
export: ; uv run python scripts/export_onnx.py --weights artifacts/best.pt
```

---

## 5. 分阶段开发计划

### Phase 0：环境与脚手架（1 周）
- uv + pyproject + src layout + CI（ruff/pytest）
- 编写 `AGENTS.md`、标注规范草稿
- 拉取 WEAPD、TIC、DTL-IceNet 示例数据

### Phase 1：MVP（3–4 周）
- **任务**：二分类/少类——有冰 vs 无冰；或 无冰 / 雪 / 凇粗分  
- YOLO-seg 或 YOLO-detect 在合并公开+少量自有数据上训练  
- 简单推理脚本 + 可视化  
- **里程碑**：验证集 mAP@0.5 ≥ 基线 YOLO；可演示单张图推理  

### Phase 2：类型与程度（4–6 周）
- 扩展标签：雨凇 / 雾凇 / 混合 / 积雪；场景分类头接 WEAPD 预训练  
- 分割 mask → 厚度公式；标定工具  
- 数据增强：雨雾噪声、低光、运动模糊；可选 CycleGAN 域迁移  
- **里程碑**：类型 Macro-F1 达标；厚度 MAPE 进入工程可接受区间（参考论文约 4–12%）  

### Phase 3：工程化与边缘（3–4 周）
- ONNX / TensorRT / INT8；目标 FPS 压测  
- 配置化 pipeline、批量推理、JSON 输出协议  
- 单元测试（厚度公式、后处理）+ 集成测试  
- **里程碑**：边缘板卡稳定推理；文档与部署手册  

### Phase 4：完整系统与领域适应（持续）
- 多任务联合训练 / 两阶段 ROI  
- 主动学习：难例回标  
- 与气象数据融合（可选，参考 DTL-IceNet 校正）  
- 可解释性：Grad-CAM / mask 叠加  

**依赖关系**：Phase1 数据与标注规范 → Phase2 类型与厚度 → Phase3 部署；模型与数据可并行迭代。

---

## 6. 数据与模型训练策略

### 6.1 公开数据利用
- WEAPD：天气/凇类预训练  
- TIC / 电力线检测集：先学“导线在哪”  
- DTL-IceNet Zenodo 示例：对齐标注与厚度流程  
- 其他：Kaggle 天气图像、恶劣天气电力巡检相关集  

### 6.2 电网数据
- 标注规范：多边形 mask（导线+冰）、类型枚举、可选厚度等级  
- 工具：Labelme / CVAT / Roboflow → YOLO 格式  
- 增强：Albumentations（亮度、模糊、雨雾合成、Cutout）  
- 领域适应：源域（公开）预训练 → 目标域微调；困难样本挖掘  

### 6.3 训练要点
- 预训练权重：COCO YOLO + WEAPD 分类  
- 损失：检测/分割标准损失 + 可选边界损失（CG-UNet 思路）  
- 不平衡：类别权重、focal、过采样重冰样本  
- 验证：按站点/天气分层划分，避免泄漏  

---

## 7. 测试、评估与部署建议

| 层级 | 内容 |
|------|------|
| 单元 | 厚度公式、坐标变换、配置加载 |
| 集成 | 端到端 pipeline 固定样例金标 |
| 指标 | 分类 F1、mAP、mIoU、MAPE、FPS、模型体积 |
| 部署 | `export_onnx` → TensorRT；Docker 可选；监控摄像头拉流 demo |
| 质量 | 低光/雨雾子集专项评估；误报成本敏感时调阈值 |

---

## 8. 风险与后续优化

| 风险 | 应对 |
|------|------|
| 公开覆冰数据不足 | 自建标注 + 合成 + 弱监督伪标签（参考南方电网 YOLOv5 弱监督工作） |
| 2D 厚度误差 | 多视角 / 标定尺（论文 ruler+YOLO）/ 气象校正 |
| 域偏移（摄像头型号、地区） | 持续微调、风格迁移、测试时增强 |
| 边缘算力 | YOLO-n + INT8；ROI 裁剪后再分割 |
| 夜间红外 | 单独模态或伪彩融合（后续扩展） |

**后续方向**：多模态（可见光+红外）、时序（覆冰生长趋势）、与 GIS/线路参数联动风险评级、VLM 生成运维说明（研究向，非 MVP 必需）。

---

## 9. 推荐技术栈与工具链

| 类别 | 选型 |
|------|------|
| 环境 | **uv** + Python 3.11/3.12 |
| 训练 | PyTorch、**ultralytics**、可选 Lightning |
| 数据 | OpenCV、Albumentations、Labelme/CVAT |
| 配置 | YAML + pydantic |
| 质量 | ruff、pytest、mypy、pre-commit |
| 导出部署 | ONNX、TensorRT、OpenVINO |
| 实验跟踪 | 可选 W&B / MLflow |
| 文档 | Markdown + 简单 MkDocs |

---

## 10. 立即执行的第一步清单

1. `uv` 初始化仓库并按上述目录建骨架，写入 `AGENTS.md`。  
2. 下载 WEAPD + DTL-IceNet Zenodo 示例，跑通分类与分割 demo。  
3. 用 TIC/电力线权重做“导线 ROI”基线。  
4. 制定标注规范与 200–500 张电网种子数据，训练 MVP YOLO-seg。  
5. 实现 `thickness.py` 几何模块与可视化，再扩展多类与边缘导出。

本计划兼顾研究深度（双任务、厚度可解释、域适应）与工程可执行性（标准目录、uv、YOLO 生态、分阶段里程碑），可直接作为立项与排期依据。若需要，我可以下一步输出：`pyproject.toml` 模板、`AGENTS.md` 正文或 MVP 训练脚本骨架。