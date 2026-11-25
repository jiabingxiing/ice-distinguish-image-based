# 模型卡：SCUT 覆冰类型四分类纯 CPU 基线 (`ice_type_v1`)

- **模型版本**：`v1`（对应实验 `v1_p2_rbf`，冻结 `mobilenetv3_small_100` + `StandardScaler` + RBF `SVC` (`C=0.1`) + 温度标定）
- **训练日期**：2026-11-25
- **manifest**：`data/manifests/scut_ice_type_v1.jsonl` & `data/manifests/scut_ice_type_temporal_v1.jsonl`
- **骨干 / 推理载体**：`mobilenetv3_small_100` + `RbfSvmTorchHead` -> `artifacts/onnx/ice_type.onnx`（FP32，opset 17，输入 `(B, 3, 224, 224)`）及伴随 `.meta.json`
- **任务定位**：电网线路/绝缘子覆冰图像 4 类形态分类（`glaze` 雨凇 / `rime` 雾凇 / `wet_snow` 湿雪 / `mixed` 混合凇）+ `unknown` 三重拒识兜底
- **禁止声明**：本模型处于**工程闭环与只读阴影评估（shadow）阶段**，**不得用于生产自动密度折减**（默认 `pipeline.yaml` 为 `none_rule`）。

## 训练与评估数据
- 数据来源：SCUT 实测覆冰图像库（`NW_data/`，严禁出域）
- 治理后划分（排除 52 坏图与 18 噪声候选）：
  - `Monitor-Holdout`：`train=4,964 / val=745 / test=1,517`（0 终端/近重复泄漏）
  - `Temporal-Holdout`：`train=4,468 / val=2,290 / test=468`（2018+ 跨时间验证）

## 实测精度与校准表现
- **Monitor-Holdout `test` (n=1,517)**：
  - `Macro-F1 = 0.5539`，`Balanced Acc = 0.5627`，`Accuracy = 0.5682`
  - `ECE = 0.0617`（温度缩放 `T = 0.5335`），`PR-AUC = 0.5597`
  - `glaze -> rime = 17.76%`，`rime -> glaze = 16.84%`
  - 三重拒识（`tau_q=0.25, tau_p=0.40, tau_m=0.08`）：测试集拒识率 `27.88%`，接受子集 `Macro-F1 = 0.6100`（`+0.0561`）
- **Temporal-Holdout `test` (n=468)**：`Macro-F1 = 0.5985`，`Balanced Acc = 0.6496`，`Accuracy = 0.6175`，`ECE = 0.0959`，`PR-AUC = 0.7180`

## 纯 CPU 推理性能
- ORT (`CPUExecutionProvider`, `intra_op=4`)：单图 P50 `3.07 ms` / P95 `4.04 ms`（`316.6 FPS`）；
- `640x480` 完整预处理+质检+ORT+拒识：P50 `20.88 ms` / P95 `27.95 ms`；
- PyTorch vs ONNX 数值一致性：`max_abs_err = 9.83e-6 <= 1e-4`；1,000 帧压测 `0` 错误。

## 已知局限与拒识保护
1. 未通过 G2（`Macro-F1 >= 0.70`），强雾、极端逆光及 `mixed <-> rime`（54.76%）存在显著混淆；
2. 跨省局（如部分未见过终端）和跨年份（2018 年样本）存在显著域漂移；
3. 本模型刻意不摄入微气象、局站名称、经纬度特征，后续需在独立现网 `Set-D` 上联合标定后再行评估升级。

## 回滚与许可
- 回滚策略：在 `configs/model/pipeline.yaml` 将 `severity.ice_type.mode` 保持为 `none_rule` 或设为 `shadow`，`weights.ice_type: null`；
- 训练数据许可：见 `docs/data_license_ledger.md` SCUT 只读约定（禁出内网）；
- 是否含 AGPL 组件：否（遵循 `ADR-0001`，推理与训练全链路不含 Ultralytics）。
