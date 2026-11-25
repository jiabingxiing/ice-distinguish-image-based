# 系统架构

> 依据 `../plans/IMPLEMENTATION_PLAN.md` §9、§7。本文只描述**已实现**的结构与调用链。

## 1. 分层

```
service/        FastAPI：请求校验、错误码、限流位              ← 只做 IO
application/    pipeline 编排 · EKF · 预警规则 · 证据图          ← 只做编排
features/       quality │ precip │ detect │ ice │ severity      ← 只做算法
training/       engine · losses · callbacks · checkpoint
evaluation/     classification · detection · segmentation · thickness
data/           manifest · splits · datasets · transforms · samplers
common/         schemas(Pydantic) · config · logging · constants
```

依赖方向单向：`common → data → features → training/evaluation → application → service`。
`service/` 不得 import `training/`；`features/*` 不得读写文件。

## 2. 一次推理的调用链

```
image + meta + calib
   │
   ├─ T0 quality.assess_quality ── unusable ──▶ 立即返回（其余字段 null）
   │
   ├─ T1 precip.predict ────────────────┐
   │     backbone → FiLM(气象) → 8 类 + CORAL 强度
   │     温度标定 → 熵阈值拒识
   │
   ├─ T2 detector.predict               │
   │     切片(640, overlap .2) → GFL 头 → 逐类 NMS
   │
   ├─ T3 seg.predict（主 ROI）           │
   │     UNet-lite → 4 类掩码 → 覆盖率 / 桥接率 / RLE
   │
   ├─ T5 severity                       │
   │     calibration.resolve（L1/L2/L3 融合 + 一致性检验）
   │     geometry.measure_widths（骨架 → 法向 → 亚像素）
   │     thickness.estimate（α → d_radial → d_eq）
   │     temporal.EKF（增长率约束 + 新息门控）
   │     decision.classify + 去抖状态机
   │
   └─ evidence.draw_caliper / draw_mask_overlay / draw_heatmap
   │
   ▼
InferenceResult（Schema v1）
```

天气分支与覆冰分支互不阻塞：检测不到资产时，仍输出天气结论，`icing.present = null`。

## 3. 降级矩阵（实现见 `pipeline.run`）

| 失败点 | 实际行为 | 输出标记 |
|---|---|---|
| 图像不可用（夜间/严重模糊/过曝） | 终止推理 | `quality.status=unusable`，其余 null |
| 无检测到资产 | 只跑 T1 | `icing.present=null` |
| 分割掩码过细/被 ROI 截断 | 保留覆盖率 | `measurement_status=visual_only`，`requires_review=true` |
| 无尺度参考 | 只给覆盖率与等级 | `measurement_status=not_calibrated`，`d_eq_mm=null` |
| 多源尺度不一致（>20%） | 拒绝出厚度 | `visual_only` + `requires_review=true` |
| 气象输入缺失 | 关闭物理门控 | `meteorological_input_used=false` |
| 时序帧数不足 | 跳过 EKF（首帧直接初始化） | `alarm_level` 仍可给，但需 ≥3 帧升级确认 |

## 4. 契约的三态门禁

`SeverityResult` 在 Pydantic 层强制：

- `not_calibrated` → `d_eq_mm` / `d_radial_mm` / `alarm_level` 必须为 null，否则抛错；
- `alarm_level` 只在 `measurement_status == measured` 时允许有值。

这是「不给伪精确毫米数」这条边界声明的**代码级实现**，不是文档约定。

## 5. 关键实现选择

| 选择 | 理由 |
|---|---|
| 检测器自建 GFL 头 | 规避 Ultralytics AGPL-3.0 传染 |
| UNet-lite 而非 LR-ASPP | 覆冰边界即厚度精度上限，全局池化会糊边界 |
| 温度标定而非 MC-Dropout | 推理零开销，分类任务足够 |
| 骨架 + 法向亚像素 | 对悬链线弧垂与不对称冰形不敏感 |
| EKF 2 状态 | 冰厚是小时级慢变，增长率约束可剔除飞掠雪花 |
| 阈值全配置化 | 业务方按本地区规程校准，不重新发版 |
| 冻结 MobileNetV3-Small + RBF SVM（ONNX opset 17） | SCUT 纯 CPU 验证微调跨期退化（Temporal F1 0.644->0.480），回退冻结特征线 + 温度缩放 + 三重拒识，单图 ORT 3.1 ms |
| 覆冰类型 `none_rule / shadow / model` 三态隔离 | 未过 G2 门限（`Macro-F1>=0.70`）禁止 `model` 折减密度；`shadow` 仅写概率与复核字段，恒保标准密度 `900 kg/m³` |
