# 基于图像的电网覆冰、降水类型及程度识别算法：开源调研与代码项目实现计划

调研基准日期：2026 年 10 月 19 日｜技术方向：计算机视觉、边缘 AI、电力设备状态监测

## 一、项目目标与总体结论

建议将项目定位为一套面向输电线路、杆塔、绝缘子及变电站现场图像的多任务视觉识别与风险评估系统，而不是单一的天气分类器。

系统需要完成三类相互关联、但不能混为一谈的任务：

1. 降水和天气识别：判断图像中是否存在雨、雪、冰雹、雾、霜冻等可观察天气现象。
2. 电力设备覆冰识别：识别导线、绝缘子及相关设备表面的覆冰区域，判断有无覆冰，并分割覆冰轮廓。
3. 覆冰程度评估：计算覆冰区域比例、视觉严重等级；在具备相机标定、已知设备尺寸或实测数据的条件下，进一步估算冰层厚度等物理指标。

需要先明确两个工程边界。

第一，识别到下雪或冻雨天气，不等于输电线路已经覆冰；反过来，图像未显示明显降水，也不能证明导线没有覆冰。环境识别与设备状态识别必须独立输出。

第二，仅凭单张普通 RGB 图像，通常无法可靠地区分冻雨与普通雨，也无法直接、准确地测得冰层厚度。冻结雨滴在接触物体后形成的冰层，可能与其他结冰形态外观相似。实际业务建议融合现场温度、湿球温度、湿度、风速、降水观测、气象雷达以及既有覆冰监测装置的数据。中国冻雨数据集研究也采用地面气象站观测与再分析数据，而不是单纯依靠图片进行判断。

[image](https://www.google.com/s2/favicons?domain=https://www.nature.com\&sz=32)

Scientific Data



### 1.1 推荐总体技术路线

输入数据层

摄像头图像 / 视频帧 / 设备台账 / 采集时间 / 气象观测 / 覆冰传感器

图像质量与感兴趣区域提取

模糊、逆光、遮挡、夜间质量检查；导线、绝缘子、站内设备 ROI 定位

天气与降水识别

分类模型、多标签识别、置信度校准

设备覆冰识别

目标检测、语义分割、覆冰轮廓提取

程度评估与多源信息融合

覆冰面积比例、视觉等级、厚度估计、历史变化、气象证据及异常置信度

业务输出层

识别结果、覆冰区域标注、告警等级、变化趋势、人工复核入口、结构化接口

建议采用分阶段、分任务建模，后期再评估多任务共享骨干网络的研发方式。这样既可以尽早得到可用的基线，也能避免因为天气和覆冰标注不足而使联合模型难以训练。

## 二、开源资源调研：值得优先研究的项目、数据集与实现

本次检索覆盖 GitHub、Hugging Face、Papers with Code 相关索引、论文页面及 Zenodo 等数据发布平台。对于直接覆冰检测，最有价值的结果是专用数据与分割代码；天气分类则有公开数据集和现成的预训练模型。尚未检索到足以直接用于生产的、同时覆盖电网覆冰、冻雨识别、物理厚度标定且拥有完整公开数据与可复现评测的统一项目。

因此，应当组合多个资源，而不是寻找一个所谓的“一站式 SOTA 仓库”。

### 2.1 高价值资源清单

[Iced high voltage wires](https://images.openai.com/static-rsc-4/0QukBmNhI3a_Izzc72R2h5c9305KhvulAZls6wscN1I1I-wScigfkbk1hDTa_qcVtg7XHSDl29OvtKxZrOPrRCi9o6Efzsg5SGNuVgkuuSqnk2laLT9iokRMtZuVB4hZG2Z3hcKzUZtdSkR8XAhy0yRgJD8Maz6mDos2_cWN5glZ1H44K55chcd-Aht3scyy?purpose=inline)

1\. DTL-IceNet：输电线路覆冰分割与厚度识别相关资源

优先级：最高｜直接电网场景

链接：[Zenodo 数据与代码发布页](https://zenodo.org/records/15718305?utm_source=chatgpt.com)，DOI：`10.5281/zenodo.15718305`。

该记录提供不同类型的输电线路覆冰样例图片和 `Ice_code.zip`。发布说明指出代码涉及覆冰线路的语义分割、多尺度特征融合与覆冰轮廓细化，可直接作为电网场景算法复现的起点。

[image](https://www.google.com/s2/favicons?domain=https://zenodo.org\&sz=32)

Zenodo



可复用部分： 图像读取、分割标签解析、模型模块结构、训练流程以及结果可视化。

局限与核查事项： Zenodo 页面展示的两个压缩包分别约为 130.9 MB 和 146.1 kB，但页面的 Data volume 字段另显示 23.6 GB，信息存在不一致，应下载后核验实际文件内容。页面可见的 License 字段未显示明确许可，投入企业或生产环境前，应先确认数据和代码的授权范围。不能仅凭公开下载链接认定允许商业使用。

[Thick iced cable](https://images.openai.com/static-rsc-4/RtLH2Ph3mXoprsasnukAjsQ2jIvNxcsnEb3BMUdmsMMV-roCieNFccGnRM1D4jOWgVUlbNgAXjfkXUURwL-wouxMpbFV3fL_xq890Tg-C_5J5tOGm8r_nPIfDTw8WwjbYUNcJsWZx8ieQcA-xYkonWrLugFM1hb0ZBl4R-zg_8Fs1Xi5Y1QB2v3fLbIU3aHe?purpose=inline)

2\. YOLOv5s 输电线路覆冰状态检测论文

优先级：高｜直接参考电网检测方法

链接：[Improved Detection of Ice Accretion Status on Transmission Lines Using YOLOv5s](https://drpress.org/ojs/index.php/fcis/article/view/9473?utm_source=chatgpt.com)。

论文介绍了输电线路覆冰图像筛选、增强、滑动窗口裁切与标注，以及在 YOLOv5s 上增加空洞卷积特征增强、改进特征金字塔和引入 Swin Transformer 特征融合等方法。摘要报告了相较原始 YOLOv5s 的 mAP 指标提升 7.6%，但具体指标定义及复现条件仍需检查全文。

[image](https://www.google.com/s2/favicons?domain=https://drpress.org\&sz=32)

Frontiers in Computing and Intelligent Systems



可复用部分： 小目标检测、切片推理、尺度增强、注意力模块及消融实验设计。

局限： 论文所述数据集来自电力企业构建的专用数据集；本次检索未核实到与论文一一对应的完整公开训练集和官方代码仓库。因此，应把它当作算法设计参考，不能直接假设其结果可以复现。

[The representative examples of 11 weather phenomena images (these... | Download Scientific Diagram](https://images.openai.com/static-rsc-4/hKJTlp-cyTbnWyfO31gJMXVycR-TEfIa6upakEgCkETfxwp3XmOfjSEhZiSA29Nq4OIAvpoKGy0DZjcflTEejUAyoQYOO3L6d6xO_I0Z7Lmdc5d8xZJcrm9vYo26Kg3_Tcv_sz_HqxGxPZsX1RzKrlZTyn0WeTDpVNDGfyfcKAY?purpose=inline)

3\. WEAPD：11 类天气现象图像数据库

优先级：高｜天气分类基础数据

链接：[GitHub 项目](https://github.com/haixiaxiao/A-database-WEAPD?utm_source=chatgpt.com)；[Harvard Dataverse 数据发布页](https://doi.org/10.7910/DVN/M8JQCR?utm_source=chatgpt.com)。

数据库包括 hail、snow、lightning、rain、rime、frost、glaze、fog/smog 等 11 类天气现象。GitHub 仓库说明了数据类别，并指向 Dataverse 的获取入口。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



可复用部分： 雨、雪、雹、雾和结霜等类别的分类训练，建立天气特征基线，以及检验跨数据集泛化能力。

局限： 这些是广义天气现象图像，不等同于输电线路现场图像；`glaze` 等类别也不能直接代替经过现场确认的冻雨标签。需要在下载后核实数据许可、样本数量、类别分布与标注质量。

[(PDF) Classification of Weather Phenomenon with a New Deep Learning Method Based on Transfer Learning](https://images.openai.com/static-rsc-4/sdqx7t5J9qdm85ffXjF_cMC9tsuyLNpDZ-rGJlZV0NN924vTycJXfcDXAqAae4OenmaRQFiOsocgTgWQ6W51Sl44GThJRGsK3cxgSsSqp0yihcAi3LmzCtFdi6UqE0hIw5jKp1U3SZsQEGw35WM5OQ_RPHVUovIJxUtdyb5zRfI?purpose=inline)

4\. RFS Dataset：雨、雪、雾三类天气数据

优先级：中高｜快速建立天气基线

链接：[论文与数据集说明（arXiv）](https://arxiv.org/abs/1808.00588?utm_source=chatgpt.com)。

该研究公开了雨、雪、雾三类图像数据，并提出利用超像素分隔掩码辅助数据增强的方案，对多种 CNN 架构进行了比较。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



可复用部分： 天气分类基线、传统 CNN 对比实验、背景与局部天气特征的增强思路。

局限： 类别数较少，没有直接覆盖冻雨、线路覆冰程度等核心电网任务；应优先用于预训练、初始基线和算法对照，而非作为最终业务数据集。

[面向复杂电力环境场景理解的可见光和红外图像特征级融合方法](https://images.openai.com/static-rsc-4/7EIuNGwVaxEUjfKix98dxUUyn3Tt1QeVzdz-hxJcUsdfQ4hxMBOgyf3IZG9HUBzJAmJ5rRgqf-bICOlLenFmBFygTR-oXbyY7ghXmW_Uz788IaDP6y44l-7KBPnIWNTxcvBhtizKivup8MtCXGwAGPXQSF-eprMZRRQYHrs2MqM?purpose=inline)

5\. Hugging Face WeatherNet 与 Weather-Image-Classification

优先级：中高｜现成模型快速评估

链接：[WeatherNet-05-18039 数据集](https://huggingface.co/datasets/prithivMLmods/WeatherNet-05-18039?utm_source=chatgpt.com)；[预训练分类模型](https://huggingface.co/prithivMLmods/Weather-Image-Classification?utm_source=chatgpt.com)。

WeatherNet 页面列出 18,039 张图像、5 个天气类别，数据集卡声明 Apache-2.0 许可。对应模型基于 `google/siglip2-base-patch16-224` 微调，模型卡报告的准确率为 85.89%，模型约有 92.9M 参数。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

+1



可复用部分： 推理接口、预训练权重加载方式、天气分类对照实验，以及自动预标注工具。

局限： 5 类标签主要是阴天、雾霾、雨/风暴、雪/霜冻和晴天，并非精细的降水物态识别。模型规模偏大，现成精度不能推断其在电网图像上的表现。建议先作为对照基线，而非直接部署到边缘设备。

[Power Infrastructure Inspection](https://images.openai.com/static-rsc-4/9LcXJN7Qek65tv1mqAiiYvGlpzjJjrGJspVP18A4lCisA7297eXY7exfmkuhLKNV1t4nAQb5FY5xaCigukJC0Cx3nzSprerv1CfdUi7RbX7q_1PZdzNMLfQ5dCArEidwWqHr2jv-I9ttgmpaR_VYlYoMfQGD-L7q96lVhbhQAD-RIV_DESKtlUV8P4fxuNBO?purpose=inline)

6\. HyLight Powerlines：电力巡检工程化项目

优先级：高｜工程脚手架参考

链接：[SebastienDorgan/hylight-powerlines](https://github.com/SebastienDorgan/hylight-powerlines?utm_source=chatgpt.com)。

该项目提供电力线部件和缺陷检测工作流，包括异构数据集标签映射、数据合并、模型训练、批量推理，以及可选的 VLM → Grounding DINO → SAM2 自动标注流程。它采用 `src/` 布局、`uv.lock` 和 `justfile`，并包含测试与 `AGENTS.md`，与本项目要求的工程结构高度契合。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

SebastienDorgan/hylight-powerlines · GitHub

+1



可复用部分： 源码布局、数据集标准化、配置映射、自动预标注、批处理入口、类型检查与测试任务。

局限： 其核心检测框架依赖 Ultralytics，数据源包含 Roboflow 资源；不能把这些数据默认视为无条件可用。仓库页面未显示明确的根目录 LICENSE 文件，复用时需分别核对代码、第三方依赖和数据许可。

[InsPLAD/README.md at main · andreluizbvs/InsPLAD · GitHub](https://images.openai.com/static-rsc-4/tnQyK8HWqUYatYLGwG-mN9_8jO2f_TDd3RrtY-MurCAOdX07igmoNTM96NAfEGUOb7TDgFQRRxbqEy1oXk1Pl_Xr3nbrtJwKo00x10SEh7Eq9OBibdMs3StVOl0ysAx_xwG0lFO1l9fS_YHhwntZP6A_UtNpdvVfz8Jn39-_IYI?purpose=inline)

7\. TIC Powerline Dataset：杆塔、绝缘子、导线目标检测

优先级：中高｜设备 ROI 检测

链接：[TIC-Powerline-Dataset-and-Detection-Models](https://github.com/Elizbellou/TIC-Powerline-Dataset-and-Detection-Models?utm_source=chatgpt.com)。

仓库说明该数据集有 2,056 张图像，包含杆塔、绝缘子和导线三个类别，提供 YOLO 格式标注及多边形标注。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



可复用部分： 设备目标检测、图像裁切、类别映射与标注格式转换，可作为覆冰模型前端 ROI 定位的补充训练资源。

局限： 该数据集不是覆冰数据集，不能用于直接验证覆冰识别或厚度估计能力。其场景分布与国内固定监控、巡检无人机视角也可能有明显差异。

[U-Net: Image Segmentation Network](https://images.openai.com/static-rsc-4/g7WSTCtlw3im0TmMNGMbLUmllGyoBLZY3yNhN1kFpGWfmYu53RIbv9Vz1lETt-1JX6_RLF6EYh6b0DpID-wRCFB6ALGaprNX7loOagl7y0hkqct1Sm8Zm_IfDUZheqgfmEDbHGmPZSnaKf0vRR0s748fs4PcWKxLo1cdFeST308?purpose=inline)

8\. MMSegmentation：语义分割工具箱

优先级：高｜覆冰区域分割研发

链接：[open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation?utm_source=chatgpt.com)。

MMSegmentation 提供 PyTorch 语义分割训练、评测和模型配置体系，仓库声明 Apache-2.0 许可，适合对比 DeepLab 系列、SegFormer 等不同分割方案。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



可复用部分： 分割模型配置、损失函数、评价指标、预训练权重加载与基线实验组织。

局限： 工具箱自身不提供针对电网覆冰的现成模型。引入较重的依赖体系前，应先确认目标硬件和部署后端的兼容性，再决定是直接采用工具箱还是借鉴其算法实现。

补充工具：Meta 的 [SAM 2](https://github.com/facebookresearch/segment-anything-2?utm_source=chatgpt.com) 支持图像及视频的提示式分割，官方仓库声明代码和模型权重采用 Apache-2.0 许可。建议用于覆冰轮廓的交互式预标注与人工修订，而不是未经验证就直接承担边缘端实时识别。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 2.2 开源调研结论及二次开发策略

上述资源按用途分为四类：

| 资源类别    | 首选资源                 | 二次开发策略                              |
| ------- | -------------------- | ----------------------------------- |
| 直接覆冰识别  | DTL-IceNet 数据与代码     | 先复现分割流程，再补充本地导线、绝缘子与现场图像            |
| 天气分类    | WEAPD、RFS、WeatherNet | 建立独立天气分类模型，统一标签映射，避免类别混淆            |
| 电网目标定位  | HyLight、TIC          | 复用 ROI 定位、标签规范化和数据处理，不把普通巡检标签冒充覆冰标签 |
| 分割与标注工具 | MMSegmentation、SAM 2 | 训练生产分割模型，利用提示式分割提高人工标注效率            |

本次检索没有确认到 Papers with Code 上存在一个已充分维护、可直接解决全部目标的统一实现。对于这个项目，按任务拆解后采用 GitHub、Hugging Face、Zenodo 和论文发布页的组合更实际。

还有一个不可忽视的工程事项：算法效果之外，必须先审查许可证。 Ultralytics 当前公开许可说明指出，若在不完整公开整个项目源码的情况下，将其 YOLO 软件或模型用于私有业务、商业产品、SaaS 或嵌入式部署，可能需要 Enterprise License。若单位不能接受 AGPL-3.0 的相关义务，应在技术选型初期评估许可，不能等到模型完成后再处理。

[image](https://www.google.com/s2/favicons?domain=https://www.ultralytics.com\&sz=32)

Ultralytics



## 三、问题定义与算法设计

### 3.1 将问题拆成四个可独立验收的算法任务

不建议在第一阶段直接开发一个“输入图片、输出所有结果”的端到端大模型。现阶段更合理的方式是分别训练、评测和部署各个模块，统一由应用层编排。

| 任务      | 输入             | 输出               | 推荐算法                        |
| ------- | -------------- | ---------------- | --------------------------- |
| 图像质量判断  | 原始图像           | 清晰度、曝光、遮挡、可识别性   | 质量分类器 + 图像质量规则              |
| 降水与天气分类 | 完整图像或背景区域      | 天气类别、概率、未知状态     | MobileNetV3、EfficientNet-B0 |
| 设备及覆冰分割 | 导线、绝缘子等 ROI    | 设备框、覆冰区域掩码       | 轻量级目标检测 + 语义分割              |
| 覆冰程度评估  | 覆冰掩码、设备尺寸、历史数据 | 覆冰比例、视觉等级、可选厚度估计 | 几何测量 + 有序分类/回归              |

第一阶段使用相互独立的模型，各自建立可重复的基线；只有当数据量、标注一致性和设备性能均满足条件时，才尝试多任务共享骨干网络。

### 3.2 明确标签体系

建议同时维护两个标签体系：一套描述图像直接观察到的现象，另一套描述综合气象证据后得出的判断。

A. 天气与降水类别

初步类别可定义如下：

| 类别 ID | 标签                             | 说明          |
| ----- | ------------------------------ | ----------- |
| 0     | clear / dry                    | 无明显降水       |
| 1     | rain                           | 普通降雨        |
| 2     | snow                           | 降雪          |
| 3     | hail                           | 冰雹          |
| 4     | fog / mist                     | 雾或明显水汽遮蔽    |
| 5     | frost / rime                   | 可观察到的霜或雾凇现象 |
| 6     | glaze / freezing-rain evidence | 雨凇或疑似冻雨相关现象 |
| 7     | mixed precipitation            | 混合降水        |
| 8     | unknown / uncertain            | 图像不足以可靠判定   |

这里的 `glaze / freezing-rain evidence` 不是“仅凭照片确认正在发生冻雨”。图像可能表现的是历史形成的冰层，不能直接反推出当前降水物态。系统应把视觉识别结果与气象观测融合后的判断分别存储。

降水程度可作为第二个维度，使用“无、弱、中、强、无法判定”等标签。但这些等级必须对应可操作的标注规则；普通监控图片的视觉强弱不一定与气象站的降水量一致。

B. 电力设备覆冰类别

建议先统一设备类别，再判断覆冰状态。

- `conductor`：导线；
- `insulator`：绝缘子；
- `tower_component`：杆塔部件；
- `other_asset`：其他设备；
- `ice`：可见覆冰区域；
- `wet_snow`、`rime`、`glaze_ice`、`unknown_ice`：在数据质量和标注能力允许时细分冰雪形态。

不要在没有足够样本的情况下过早将覆冰细分成大量小类别。首先确保模型可以稳定区分“无覆冰、明确覆冰、无法判断”，再扩展形态分类。

### 3.3 统一的推理结果结构

建议所有算法均返回统一的数据契约，而不是每个模型使用一套输出格式。

```
{
  "image_id": "sample_001",
  "asset_id": "line_001",
  "timestamp": "2026-10-19T08:30:00+08:00",
  "quality": {
    "status": "usable",
    "blur_score": 0.12,
    "occlusion": false
  },
  "weather": {
    "visual_class": "snow",
    "confidence": 0.93,
    "precipitation_intensity": "unknown"
  },
  "icing": {
    "present": true,
    "confidence": 0.96,
    "coverage_ratio": 0.24,
    "severity_grade": "G2",
    "thickness_mm": null,
    "measurement_status": "not_calibrated"
  },
  "sensor_fusion": {
    "status": "not_available",
    "freezing_rain_assessment": "unknown"
  },
  "model_version": "0.1.0"
}
```

以上只是接口示例。示例概率和等级不代表实测结果，`G2` 也只是占位标签，不能直接对应任何电网规程中的实际风险等级。

接口至少需要保留原始图像引用、模型版本、推理时间、置信度、数据质量状态和标注区域引用。发生告警时，应当可以追溯模型依据的图像及输入信息。

### 3.4 覆冰检测模型：先定位设备，再识别覆冰

直接在整张图上判断是否覆冰，容易受到天空、树木、积雪背景、远距离设备和摄像头位置的干扰。

推荐采用以下流程：

1. 利用固定摄像头配置、人工 ROI 或目标检测器确定导线、绝缘子等目标区域。
2. 对目标 ROI 做分辨率自适应裁切，避免将细小的覆冰纹理缩放丢失。
3. 在裁切区域上执行覆冰分割，输出像素级覆冰掩码。
4. 利用图像质量、目标有效面积和分割置信度过滤低质量结果。
5. 将分割结果转换为覆冰比例、边界、视觉等级及可选的几何测量值。

对于清晰的固定摄像头画面，可以先以规则 ROI 和轻量分割网络作为基线；对于无人机巡检或者设备位置变化较大的图像，则需要先检测设备，再进行分割。

### 3.5 推荐的模型组合

| 子任务   | MVP 基线                        | 后续候选                                 | 选择依据              |
| ----- | ----------------------------- | ------------------------------------ | ----------------- |
| 天气分类  | MobileNetV3 / EfficientNet-B0 | ConvNeXt-T、SigLIP 特征微调               | 精度、延迟与模型规模之间的平衡   |
| 设备定位  | 轻量目标检测网络                      | RTMDet、其他经许可审查的检测模型                  | 设备尺度、密集场景、边缘推理约束  |
| 覆冰分割  | U-Net + 轻量骨干网络                | DeepLabv3+、SegFormer、DTL-IceNet 相关结构 | 轮廓质量、速度和小样本适应能力   |
| 覆冰等级  | 规则基线 + 有序分类器                  | 多任务分类与回归                             | 标签一致性、可解释性及物理测量能力 |
| 交互式标注 | CVAT 等人工标注工具                  | SAM 2 辅助预标注                          | 减少多边形/掩码标注成本      |

具体模型最终应通过统一数据集的实测选型，不建议仅按论文榜单排名决定生产模型。

### 3.6 覆冰程度如何量化

这是项目最重要的工程难点之一。

首先，计算覆冰视觉覆盖率。例如：

\\[ R\_{\text{ice}} = \frac{N\_{\text{ice pixels}}} {N\_{\text{valid asset pixels}}} \\]

其中，分母必须是有效设备区域，而不是整张图片。这个比例适合衡量视觉变化，但不能直接等同于覆冰厚度、覆冰质量或导线承受的机械负荷。

其次，建立视觉严重程度分级。例如使用 G0–G4 五级制：

| 等级 | 定义原则        |
| -- | ----------- |
| G0 | 无可确认覆冰      |
| G1 | 轻微或局部覆冰     |
| G2 | 覆冰范围进一步扩大   |
| G3 | 明显、广泛的覆冰    |
| G4 | 严重覆冰，需要重点复核 |

这些文字只是项目内部初始定义。正式实施时，必须由电力设备专家根据现场工况、设备类型和现行规程建立标注图谱及等级阈值，不可仅凭覆盖率任意划定安全等级。

最后，只有在已知导线原始直径、相机标定、拍摄角度和设备几何关系等条件下，才考虑估算冰层厚度。对于近似垂直观察、圆形导线截面且轮廓可辨的情况，可以尝试：

\\[ t\_{\text{ice}}\approx \frac{D\_{\text{outer}}-D\_{\text{conductor}}}{2} \\]

但该公式依赖投影几何、轮廓可见性和尺度标定。对斜视、遮挡或不均匀覆冰场景，不能直接把像素直径差换算成真实毫米数。

推荐输出两个独立指标：视觉严重等级，以及经过标定的物理厚度估计。 未完成标定时，厚度字段应返回 `null`，而不是输出缺乏依据的伪精确值。

## 四、数据体系与训练策略

模型研发的主要瓶颈很可能不是网络结构，而是电网覆冰数据的数量、标签质量和现场分布差异。公开数据用于建立模型起点，现场数据用于决定模型能否落地。

### 4.1 数据分层管理

统一采用四级数据流：

`raw → interim → processed → manifests`

- `raw`：原始图像、视频帧、设备台账与原始气象记录，只读保存。
- `interim`：去重后的样本、裁切结果、初步标注和清洗记录。
- `processed`：统一尺寸、格式、标签映射及训练所需的最终数据。
- `manifests`：训练、验证、测试清单，包括数据来源、标签版本、站点、天气过程、采集时间和校验和。

所有数据处理操作应当可重跑，且不覆盖原始文件。每次训练必须保存数据清单版本，防止数据发生变化后无法解释实验结果。

### 4.2 数据集使用顺序

推荐按以下顺序开展工作。

第一阶段：公开数据快速基线。 从 WEAPD、RFS、WeatherNet 中建立天气分类试验，明确数据格式、标签转换、预处理方法与基础指标。天气数据只用于其实际覆盖的类别，不得将普通天气识别结果解释为精确冻雨检测。

第二阶段：覆冰专用数据复现。 下载并核验 DTL-IceNet 的代码、图片及标签，先独立跑通数据读取与分割推理，再训练一个最简单的覆冰分割模型进行对照。

第三阶段：本地电网场景适配。 重点采集固定监控、远距离导线、绝缘子、塔头、变电站户外设备等实际目标。覆盖晴天、雨雪、夜间、逆光、镜头污染、雾气和不同拍摄距离。

第四阶段：工程反馈闭环。 将误检、漏检、低置信度结果和现场人工复核数据回流到样本池中，定期进行增量训练与回归评测。

### 4.3 现场标注规范

建议一条样本至少记录以下信息：

| 字段                    | 标注要求                   |
| --------------------- | ---------------------- |
| `image_id`            | 唯一图像标识                 |
| `asset_id`            | 设备或线路标识                |
| `camera_id`、`site_id` | 摄像头与站点标识               |
| `timestamp`           | 采集时间，明确时区              |
| `weather_label`       | 图像可观察的天气类别             |
| `precipitation_label` | 经可信观测确认的降水类别；未确认时为空或未知 |
| `asset_bbox`          | 目标设备检测框                |
| `ice_mask`            | 覆冰区域的像素级掩码             |
| `severity_grade`      | 经专家规则标定的视觉等级           |
| `thickness_mm`        | 有实测或可靠标定数据时填写          |
| `label_confidence`    | 标注人员对标签可靠性的判断          |
| `source`              | 图片、观测记录和标签的来源          |

对于同一场天气过程的连续视频帧，不能简单随机分配到训练集和测试集。高度相似的帧会使测试结果虚高。

推荐以天气过程、拍摄时间段、摄像头、站点或线路为分组单位，建立不重叠的训练、验证和测试集，并额外保留至少一个未参与训练的现场站点或摄像头，用于测试跨场景泛化。

### 4.4 数据增强与领域适应

建议将增强策略分为三组。

第一组是常规增强：亮度、对比度、轻微色偏、缩放、平移、裁切和适度压缩噪声。用于模拟摄像头参数和现场光照变化。

第二组是天气鲁棒性增强：雨滴、雾、雪、模糊、运动模糊和局部遮挡。合成效果必须经过人工检查，不能因为增强操作改变了图像语义却保留原标签。

第三组是覆冰外观增强：利用真实覆冰掩码、分层纹理或图像合成方法扩充样本。合成结果可以用于预训练和困难样本研究，但必须与真实数据分开管理。不能用合成图像替代真实覆冰测试集，也不能把不真实的冰层纹理当作真实严重等级依据。

领域适应阶段建议先冻结大部分骨干网络，只训练分类头或分割头；验证稳定后，再逐步解冻后部网络层。如果有足够未标注现场图像，可尝试伪标签和主动学习，但伪标签必须经过置信度筛选，关键覆冰样本需要人工复核。

### 4.5 首批数据规模与采集计划

下表是用于排期的起步预算，不是保证达到生产准确率的通用最低样本量。

| 数据类别    | 初步规划                   | 主要用途                  |
| ------- | ---------------------- | --------------------- |
| 天气分类    | 约 1,000–3,000 张去重图像    | 8–9 类初始基线，覆盖常见天气和未知状态 |
| 覆冰正负样本  | 约 1,000–3,000 张代表性设备图像 | 有无覆冰判断，特别补充无覆冰困难负样本   |
| 覆冰像素级掩码 | 约 500–1,000 张专家标注图像    | 建立与比较分割模型             |
| 严重等级样本  | 覆盖全部计划等级，并具有专家复核记录     | 有序分类及风险评估             |
| 物理厚度样本  | 有实测厚度、拍摄几何和对应图片的样本     | 厚度估计与标定验证             |

应优先确保样本具有独立性与代表性，而不是单纯追求图片总数。对冻雨、冰雹等稀有危险类别，应有针对性地收集真实事件，并同时保留困难负样本。

## 五、详细代码文件夹结构

### 5.1 目录组织原则

推荐使用 Python `src layout` + 按功能组织算法 + 公共基础设施分层。

Python Packaging User Guide 描述了通过 `src/` 放置包代码的方式，可避免项目根目录中的文件意外遮蔽安装后的包。`kriasoft/Folder-Structure-Conventions` 也强调将源码、测试、文档与工具分别组织。

[image](https://www.google.com/s2/favicons?domain=https://packaging.python.org\&sz=32)

Python Packaging User Guide

+1



参考目录：

```
icing-vision/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
├── Makefile
├── AGENTS.md
│
├── src/
│   └── icing_vision/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── common/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── exceptions.py
│       │   ├── logging_config.py
│       │   └── constants.py
│       │
│       ├── features/
│       │   ├── README.md
│       │   ├── image_quality/
│       │   │   ├── model.py
│       │   │   └── inference.py
│       │   ├── precipitation/
│       │   │   ├── dataset.py
│       │   │   ├── model.py
│       │   │   ├── train.py
│       │   │   └── inference.py
│       │   ├── ice_detection/
│       │   │   ├── dataset.py
│       │   │   ├── detector.py
│       │   │   ├── segmenter.py
│       │   │   ├── train.py
│       │   │   └── inference.py
│       │   └── severity/
│       │       ├── measurement.py
│       │       ├── grading.py
│       │       └── calibration.py
│       │
│       ├── data/
│       │   ├── readers.py
│       │   ├── manifests.py
│       │   └── validation.py
│       │
│       ├── training/
│       │   ├── engine.py
│       │   ├── callbacks.py
│       │   ├── losses.py
│       │   └── checkpoints.py
│       │
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── detection.py
│       │   ├── segmentation.py
│       │   └── severity.py
│       │
│       ├── application/
│       │   ├── pipeline.py
│       │   ├── sensor_fusion.py
│       │   └── decision_rules.py
│       │
│       ├── deployment/
│       │   ├── export_onnx.py
│       │   ├── runtime.py
│       │   └── benchmark.py
│       │
│       └── service/
│           └── api.py
│
├── tests/
│   ├── README.md
│   ├── unit/
│   │   ├── test_manifests.py
│   │   ├── test_grading.py
│   │   └── test_measurement.py
│   ├── integration/
│   │   ├── test_weather_pipeline.py
│   │   └── test_icing_pipeline.py
│   ├── e2e/
│   │   └── test_api_inference.py
│   └── fixtures/
│
├── configs/
│   ├── README.md
│   ├── data/
│   │   ├── weather.yaml
│   │   └── icing.yaml
│   ├── models/
│   │   ├── weather_mobilenet_v3.yaml
│   │   └── icing_segmentation.yaml
│   └── deployment/
│       └── inference.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── manifests/
│
├── scripts/
│   ├── README.md
│   ├── download_public_data.py
│   ├── validate_dataset.py
│   ├── build_manifests.py
│   ├── train_weather.py
│   ├── train_icing.py
│   ├── evaluate.py
│   └── export_model.py
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── annotation_guide.md
│   ├── model_card_template.md
│   ├── deployment.md
│   └── decisions/
│
├── notebooks/
│   └── README.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

该结构以功能模块作为主要组织单位，同时将公共数据处理、训练、评测、部署和应用编排分离。核心算法模块不应同时承担读取文件、训练模型、部署服务和告警判断等职责。

### 5.2 关键模块的实现边界

`features/precipitation/` 只负责天气与降水类别的模型训练和推理，不直接生成电网设备风险等级。

`features/ice_detection/` 负责设备定位、覆冰分割以及覆冰置信度。设备检测和覆冰分割初期可以使用不同的模型，避免将设备定位错误误认为模型没有覆冰能力。

`features/severity/` 不依赖具体模型架构，只接受掩码、目标几何和标定参数，计算覆冰视觉比例及可测量的物理指标。这使程度评估可以单独测试。

`application/sensor_fusion.py` 汇总图像结果、时间序列和气象证据。`decision_rules.py` 负责业务规则、阈值和告警策略，阈值必须来自经批准的业务配置，而不是硬编码在神经网络中。

`deployment/` 管理模型导出、后端加载与性能基准；`service/` 负责 API、请求校验和错误返回。训练代码不能直接成为生产服务入口。

### 5.3 文件和目录管理约定

每个重要目录放置 `README.md`，写清用途、输入输出和使用方法。暂时没有文件的目录可用 `.gitkeep` 保留。

根目录只保留入口文档、依赖配置、工程约定和标准工具配置。实验中间文件、训练权重和运行日志不应散落在根目录。

`.gitignore` 至少需要排除以下内容：

```
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
dist/
build/

# Test and lint cache
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
coverage.xml
.ipynb_checkpoints/

# IDE and OS noise
.idea/
.vscode/
.DS_Store
Thumbs.db

# Local secrets
.env
.env.*
!.env.example

# Datasets
data/raw/*
data/external/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/external/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Experiments and model artifacts
outputs/
runs/
checkpoints/
mlruns/
wandb/
lightning_logs/
*.pt
*.pth
*.onnx
*.engine
*.log
```

生产模型和数据应进入经批准的模型仓库、数据平台或制品存储，不依赖 Git 传输大文件。确实需要提交的小型测试样本应放进单独的 fixture 目录，并核验其授权。

### 5.4 AGENTS.md 的项目规则

建议在根目录写入可供代码 Agent 遵循的硬性规则：

- 新业务功能放入 `src/icing_vision/features/`，不在根目录创建临时 Python 文件。
- 训练脚本通过公共数据管线和配置文件调用模型，不重复实现读取、预处理与指标计算。
- 所有数据路径、模型路径和阈值使用配置管理；禁止在源码中写入本地绝对路径及密钥。
- 修改模型输入输出结构时，必须同步更新 `common/schemas.py`、测试与文档。
- 新增功能必须具备单元测试；核心推理流程修改后必须运行集成测试与回归测试。
- 每次正式训练保存配置、数据清单、随机种子、代码版本和模型指标。
- 未经授权不得将真实电网数据、内部地址、设备标识或凭据上传至外部服务。
- 第三方模型、数据集和预训练权重必须经过许可核查并记录来源。

## 六、使用 uv 建立开发环境与工程工具链

### 6.1 Python 环境管理

建议采用 Python 3.11 作为初始基线，`uv` 负责虚拟环境、依赖管理和锁文件；CUDA、PyTorch、目标边缘平台的运行库需要分别核验兼容性。

`uv` 官方支持通过 `uv init` 初始化项目、`uv add` 管理依赖、`uv sync` 同步环境、`uv lock` 生成锁文件以及 `uv run` 执行命令。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



从空目录初始化：

```
uv init --package icing-vision
cd icing-vision

uv python pin 3.11
uv add torch torchvision
uv add numpy pillow opencv-python-headless pyyaml pydantic typer rich
uv add scikit-learn pandas
uv add --dev pytest pytest-cov ruff mypy pre-commit

uv lock
uv sync --dev

uv run python -c "import torch; print(torch.__version__)"
uv run pytest
```

这里的命令用于建立初始环境。GPU 训练机器应按照其操作系统、CUDA 驱动与所选 PyTorch 版本，使用 PyTorch 官方提供的安装配置来确定兼容的二进制包，不能假设在所有机器上执行相同依赖命令就能自动获得正确的 CUDA 环境。

建议采用以下依赖分组：

- `base`：配置、图像读取、数据契约、日志和 CLI。
- `train`：PyTorch、数据增强、分类与分割训练依赖。
- `evaluation`：scikit-learn、指标计算和可视化。
- `deploy`：ONNX Runtime、API 服务和后端运行依赖。
- `dev`：pytest、ruff、mypy、pre-commit。

避免把全部研究工具和大型推理依赖强制放入基础环境。MVP 开发阶段的依赖越精简，后续在离线环境和边缘设备上复现的难度越低。

需要特别注意：`uv.lock` 是依赖解析结果的可复现记录，但它不能保证不同 CUDA 驱动、系统库和设备上的数值行为完全一致。因此还需保存机器环境和模型运行时信息。

### 6.2 Makefile 常用命令

在根目录建立统一入口：

```
.PHONY: setup test lint format train-weather train-icing eval export bench

setup:
	uv sync --dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

train-weather:
	uv run python scripts/train_weather.py --config configs/models/weather_mobilenet_v3.yaml

train-icing:
	uv run python scripts/train_icing.py --config configs/models/icing_segmentation.yaml

eval:
	uv run python scripts/evaluate.py

export:
	uv run python scripts/export_model.py

bench:
	uv run python -m icing_vision.cli benchmark
```

各脚本的参数接口应在实际开发时统一定义。命令存在之前，应先实现对应入口或明确返回“功能未实现”，不能把任务文件预先创建当作功能完成。

推荐工作流：

```
make setup
make lint
make test

# 有训练数据及 GPU 环境后
make train-weather
make train-icing
make eval
make export
make bench
```

CI 先执行静态检查、单元测试和轻量级接口测试；GPU 训练、长时间评测及边缘设备压力测试放在单独的任务中。

## 七、分阶段开发计划与里程碑

建议采用 8–12 周的 MVP 研发基线。实际周期取决于内部数据获取、覆冰掩码标注和目标硬件适配情况。以下是可执行的排期基准，不是对开发周期的保证。

第 1 周

M0：需求、数据与合规

明确设备范围、输入图像来源、天气类别、严重度规则、目标运行设备与响应时限。下载并审查 DTL-IceNet、WEAPD 等资源，记录许可、数据量、标签和复现依赖。

交付物：需求规格、标签字典、数据资产清单、许可审查记录、验收指标草案。

第 1–2 周

M1：工程骨架与数据管线

使用 `uv` 初始化项目，建立数据读取、数据清单、标签校验、日志、配置系统和测试框架。实现可重复的数据划分与样本可视化。

交付物：可安装的 Python 包、uv.lock、Makefile、AGENTS.md、CI、数据校验脚本。

第 2–4 周

M2：独立算法基线

天气分支训练轻量分类器；覆冰分支跑通公开数据、设备 ROI 提取和覆冰分割；所有模型实现统一推理接口。暂时不追求复杂多任务结构。

交付物：基线权重、评测报告、混淆矩阵、覆冰掩码可视化及推理 CLI。

第 3–6 周

M3：本地数据标注与模型适配

接入现场图片；制定覆冰轮廓、严重程度及不确定样本的标注规范。训练适配本地设备的检测和分割模型，采用困难样本挖掘和有控制的数据增强。

交付物：版本化电网样本集、本地模型权重、独立站点测试结果、误检漏检案例库。

第 5–8 周

M4：程度评估与信息融合

实现覆冰覆盖率、视觉等级、可选的几何厚度估算及时间序列变化。若气象数据可用，再融合温度、湿度、降水等证据。

交付物：可独立测试的程度评估模块、参数配置、融合规则、校准报告。

第 7–10 周

M5：推理优化与部署验证

导出 ONNX，比较 PyTorch 与目标运行时的精度差异；进行批量推理、低置信度拒识、断网、低光和连续告警测试。

交付物：部署包、推理性能报告、部署指南、模型卡和回归测试集。

第 9–12 周

M6：现场试运行与验收

在选定站点或线路试运行，对比人工复核与实测记录。重点检查严重覆冰漏报、误报警、未知类别处理、模型升级和数据追溯。

交付物：现场试运行报告、问题清单、版本冻结记录、上线与回滚方案。

### 7.1 任务依赖关系

```
需求与授权确认
       |
       v
工程骨架与数据清单
       |
       v
公开数据基线 ---------> 现场数据整理与标注
       |                          |
       +------------+-------------+
                    |
                    v
            本地模型训练与适配
                    |
                    v
          覆冰程度与多源信息融合
                    |
                    v
             模型导出与性能评测
                    |
                    v
             现场试运行与验收
```

并行任务主要是数据许可和标注准备、工程骨架开发、公开基线复现。现场数据不足时，可以继续完善数据校验与推理框架，但不能把公开天气分类精度当成电网覆冰业务已经验收。

### 7.2 每个里程碑必须满足的退出条件

M2 结束前，每个模型必须具备可运行的训练、验证和推理入口；只提供论文中的网络结构而没有完整执行流程，不算完成。

M3 结束前，数据必须完成去重和组间隔离，至少具备一套人工核验的独立测试集。不能根据训练集上的准确率宣布适配完成。

M4 结束前，所有物理测量字段必须注明测量依据。没有标定参数的图像不能输出被误解为真实值的覆冰厚度。

M5 结束前，必须完成模型运行时与 PyTorch 的对照，确认部署前处理、归一化、输出解析和阈值计算一致。

M6 结束前，必须有高风险误报漏报案例、人工复核机制和模型回退方案。任何涉及关键电网操作的告警，应按批准的业务流程处理，不能由未经验证的图像模型单独决定。

## 八、测试、评估与验收指标

不能只用一个总体准确率判断算法是否可用。特别是覆冰样本稀少、正常图像占比很高时，即使模型总是判断“没有覆冰”，总体准确率也可能看起来很高。

### 8.1 分任务评估

| 模块     | 核心指标                                             | 主要目的           |
| ------ | ------------------------------------------------ | -------------- |
| 天气分类   | Macro-F1、Balanced Accuracy、各类 Precision / Recall | 防止常见类别掩盖少数类问题  |
| 目标检测   | mAP\@0.5、mAP\@0.5:0.95、Recall                    | 检查设备定位和小目标检测质量 |
| 覆冰分割   | mIoU、Dice、Boundary F1                            | 检查覆冰区域和边缘准确性   |
| 严重程度分级 | Macro-F1、各级 Recall、加权 Cohen's Kappa              | 检查等级判断是否符合专家标注 |
| 厚度估计   | MAE、RMSE、不同厚度区间误差                                | 检查物理测量误差       |
| 置信度    | ECE、Brier Score、拒识覆盖率                            | 判断模型置信度是否可靠    |
| 工程性能   | p50/p95 延迟、FPS、内存、模型大小                           | 验证实际部署能力       |

还应单独统计高风险场景下的覆冰漏报率，并明确分母。例如，不能将大量正常图片与少量严重覆冰图片混在一起，最终只公布一个总体召回率。

### 8.2 建议的验收目标

以下数字属于首轮试点的建议目标，不是已有模型的实验成绩，也不是适用于所有电网业务的统一标准。应结合实际数据量、风险容忍度和专家复核结果进行调整。

| 验收项    | 初步目标                             |
| ------ | -------------------------------- |
| 常见天气分类 | 独立测试集 Macro-F1 ≥ 0.85            |
| 覆冰二分类  | 独立测试集 Recall ≥ 0.90，且单独报告严重覆冰漏报率 |
| 覆冰分割   | 代表性测试集 Dice ≥ 0.80               |
| 严重等级判断 | 各等级分别统计 Recall；关键等级不得仅以总体准确率验收   |
| 推理时延   | 固定目标硬件上满足业务刷新周期，并报告 p50、p95      |
| 程序稳定性  | 连续批量推理无未捕获异常；坏图、空图、缺失传感器数据均有明确返回 |
| 可追溯性   | 每次正式推理可定位输入、标签映射、模型版本、配置版本与结果记录  |

如果目标是严寒地区的输电线路覆冰监测，应优先提高严重覆冰的召回率，并由业务专家确定可接受的误报警水平。若用于普通变电站现场观察，可根据具体设备和运行流程设置不同的性能门槛。

### 8.3 必须完成的测试集设计

测试集至少覆盖四个维度。

一是天气分布，包括无降水、普通雨、雪、雾、霜冻和困难的混合场景。二是设备差异，包括不同导线直径、绝缘子形状、不同视角和拍摄距离。三是成像环境，包括夜间、逆光、镜头污损、压缩噪声、模糊与遮挡。四是数据来源差异，包括不同摄像头、站点、线路以及不同时间的天气过程。

建议设置三套测试集：

- 同分布测试集：检查模型在与训练数据相近的场景中的表现。
- 跨站点测试集：检查在未参与训练的摄像头或线路上的泛化能力。
- 困难场景测试集：专门检查严重覆冰、局部遮挡、低光、图像质量下降和未知类别。

每次更换骨干网络、调整数据增强、修改阈值或变更推理后端，都必须执行回归测试。

## 九、部署架构与工程化建议

### 9.1 边缘端与云端的任务划分

### 边缘端 / 现场端

图像采集、质量筛查、轻量级设备识别、覆冰初筛、局部结果缓存。适用于网络不稳定、需要降低上传流量或要求快速响应的场景。

### 云端 / 中心平台

较复杂的分割模型、时序变化分析、气象数据融合、模型版本管理、跨站点统计、困难样本管理和模型训练。

### 业务系统接口

统一输出设备标识、时间、视觉识别结果、置信度、告警建议、图片引用和模型版本。通过明确的接口契约对接生产运维平台。

并不是每个项目都需要将全部模型同时部署到边缘与云端。对固定高分辨率监控，可以先采用边缘端质量筛查和目标裁切，再将候选图像上传至云端精细分割；若现场时延或网络要求不允许上传，则应将精简后的模型部署到本地设备。

### 9.2 推理与模型导出

建议形成如下标准链路：

```
PyTorch checkpoint
       |
       v
统一预处理与输入输出定义
       |
       v
导出 ONNX
       |
       v
ONNX Runtime / TensorRT / OpenVINO
       |
       v
数值一致性测试与性能基准
       |
       v
目标设备部署包
```

ONNX 是便于跨运行时验证的中间格式，但并非所有模型都能无损导出。需要验证算子支持、动态形状、后处理实现和数值误差。

对于 NVIDIA GPU，可将 TensorRT 作为候选推理后端；对于 x86 CPU，可评测 ONNX Runtime 或 OpenVINO；对于 ARM 或专用 AI 芯片，则需要以实际硬件厂商支持的运行时为准。OpenMMLab 的 MMDeploy 提供了多种设备及推理后端的部署适配参考，但是否支持具体模型与硬件组合仍需逐项验证。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 9.3 服务化与线上监控

最小服务接口建议包括：

- `POST /v1/inference/image`：单张图像推理。
- `POST /v1/inference/batch`：批量推理。
- `GET /v1/health`：服务健康状态。
- `GET /v1/model/info`：当前模型与配置版本。
- `GET /v1/metrics`：运行时间、请求量、错误率等指标。

输入检查应覆盖格式错误、空文件、超大图像、无法解码、缺失设备信息和无效时间戳。输出应记录模型版本、推理时间、关键置信度、低质量标记以及告警规则的版本。

生产系统还应监控输入数据分布变化。如果新摄像头的画面比例、照明条件或设备视角与训练集差异显著，应触发数据漂移告警和人工抽样评估，而不是仅依赖模型置信度决定是否需要重新训练。

## 十、主要风险及解决方案

| 风险          | 可能后果           | 缓解措施                           |
| ----------- | -------------- | ------------------------------ |
| 电网覆冰专用数据不足  | 模型容易学习背景而非冰层特征 | 采集现场数据，建立跨站点测试集，使用像素级掩码        |
| 冻雨与普通雨难区分   | 错误判断降水物态       | 图像分类与温度、湿球温度、气象记录分开输出，再做证据融合   |
| 覆冰程度没有统一标签  | 不同标注人员给出冲突等级   | 先制定规程依据、标注图谱和专家仲裁机制            |
| 厚度没有物理标定    | 输出伪精确毫米数       | 仅在满足几何与尺度条件时估算，保留 `null` 和测量状态 |
| 训练与测试数据泄漏   | 评测成绩虚高         | 按站点、天气过程、视频与时间段分组划分            |
| 合成数据过度使用    | 产生不符合真实物理形态的样本 | 将合成样本与真实样本分开，真实测试集不混入合成图       |
| 边缘端资源不足     | 时延、内存或热负载超标    | 限制输入尺寸、量化、裁切推理，使用目标硬件实测        |
| 开源许可不适用     | 影响企业内部使用与发布    | 在选型阶段审查代码、权重、数据和依赖的许可证         |
| 低置信度结果被强行分类 | 现场误报警或漏报       | 增设 unknown、reject 和人工复核流程      |

在安全相关的运行场景中，算法应定位为状态感知与辅助决策模块。严重覆冰告警需要结合可靠的观测和经过批准的业务规则，不能让未经现场验证的单张图像分类结果直接替代运行管理判断。

## 十一、推荐技术栈

| 层级   | 推荐技术                        | 用途             |
| ---- | --------------------------- | -------------- |
| 开发环境 | Python 3.11、uv              | 环境、依赖与锁文件管理    |
| 核心框架 | PyTorch、Torchvision         | 模型开发与训练        |
| 模型基础 | timm、经许可核验的检测/分割实现          | 预训练骨干网络与视觉算法   |
| 图像处理 | OpenCV、Pillow、NumPy         | 图像读取、变换、ROI、测量 |
| 数据分析 | Pandas、scikit-learn         | 标签清洗、分类指标、统计分析 |
| 数据标注 | CVAT 等工具，SAM 2 辅助           | 检测框与分割掩码标注     |
| 配置管理 | YAML、Pydantic               | 参数与数据契约校验      |
| 实验管理 | JSON/YAML 记录；必要时引入 MLflow   | 结果追踪、模型版本管理    |
| 代码质量 | pytest、Ruff、mypy、pre-commit | 测试、格式、静态检查     |
| 推理部署 | ONNX、ONNX Runtime           | 跨环境推理验证        |
| 硬件加速 | TensorRT、OpenVINO 或芯片厂商运行时  | 按目标平台进行部署优化    |
| 服务接口 | FastAPI                     | 图像推理与业务集成      |
| 持续集成 | GitHub Actions 或单位内部 CI     | 自动化检查与版本验证     |

这些工具不需要全部在第一周引入。MVP 建议只保留 PyTorch、Torchvision、OpenCV、Pillow、NumPy、Pydantic、PyYAML、pytest 和 Ruff。待数据管线和基线稳定后，再按实际需求增加分割工具箱、实验管理与部署后端。

## 十二、建议的首周实施清单

项目启动后，不宜立即开始开发复杂的覆冰多任务网络。第一周应按以下顺序完成：

### 首周完成情况

0/7

建立 Git 仓库、uv 环境、src layout、pyproject.toml、uv.lock 与 CI

下载并核查 DTL-IceNet 的代码、数据、标注格式及许可信息

下载 WEAPD 或 RFS 数据，确认图像数量、标签、许可及重复样本

定义天气类别、设备类别、覆冰等级和 unknown 标签

建立数据清单和按站点/天气过程划分的数据集脚本

完成一条从原始图像到 JSON 推理结果的最小端到端流程

设计覆冰二分类、分割及天气分类的独立基线实验

最终建议： 先完成可复现的天气分类基线和覆冰分割基线，再利用本地电网数据完成领域适配，之后才开发物理程度估计、多源融合与边缘端优化。这个顺序能够尽早暴露数据不足、标注不一致、模型许可和部署兼容性等风险，同时避免在缺乏真实覆冰数据的情况下过早投入复杂网络设计。