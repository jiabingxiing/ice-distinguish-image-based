# ADR-0001：检测器不用 Ultralytics，改自建 GFL 头

- 状态：已采纳
- 日期：2026-10-19
- 相关：§10.3、风险 R7、`tests/unit/test_license_guard.py`

## 背景

T2 需要检测 `conductor / insulator_string / tower / reference_fitting` 四类。
最短路径是 Ultralytics YOLO11-seg 或 YOLO11-det，生态成熟、预标注方便。
但 Ultralytics 采用 **AGPL-3.0**：若以 SaaS 或私有化部署方式交付且不公开全部源码，
通常需要购买 Enterprise License。

本项目面向电网企业交付，源码开放不现实，许可风险不可接受。

## 决策

- 生产链路：`timm backbone (mobilenetv4_conv_small) + 自建 GFL 头`，
  依赖链为 Apache-2.0 / MIT；
- Ultralytics 仅出现在 `scripts/prelabel.py`，用于**人工预标注**，
  产物必须 `reviewed=true` 才可入库，不产生生产权重；
- CI 中用 `test_license_guard.py` 强制：`src/` 出现 `import ultralytics` 直接 fail。

## 代价

- 自建头多花约 3–5 天（GFL 的 QFL/DFL/GIoU + 点分配）；
- 没有现成的 mosaic/混合精度/EMA 训练配方，需自己补；
- 小目标性能需要靠切片推理与 NWD 式分配补，调参成本上升。

## 收益

- 彻底规避 AGPL 传染，交付时不需要法务逐条解释"链接"与"网络服务"边界；
- 检测头完全可控，后续加 NWD、条带卷积、质量分支不受上游框架约束；
- 依赖树更轻，容器镜像更小。

## 备选方案与否决理由

| 方案 | 许可 | 否决理由 |
|---|---|---|
| Ultralytics YOLO11 | AGPL-3.0 | 私有部署需商业授权 |
| RTMDet-tiny（MMDetection） | Apache-2.0 | 依赖 mmcv，编译重、版本敏感 |
| Detectron2 | Apache-2.0 | 依赖重，自定义头成本高于自建 |
