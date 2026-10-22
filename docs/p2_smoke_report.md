# P2 检测冒烟报告（TTPLA / CPU）

> 日期：2026-10-20 ｜ 配置：`configs/train/detector_smoke.yaml`（hash `08ea3fed63a4`）
> 目的：证明「timm 骨干 + 自建 GFL 头」在真实公开数据上**能收敛**。
> **本报告的任何数字都不是验收指标**（TTPLA 为航拍俯视域，且无法按场景防泄漏）。

## 1. 结论

**已完成，链路可用。** 10 epoch、CPU 52 分 46 秒，验证损失从 8.28 降到 5.95，
测试集 mAP@0.5 = **0.324**（随机基线约 0）。检测头、分配器、解码、评估、切片推理
五段全部跑通，这是 P2 此前最大的空白。

同时发现并修掉 3 个「静默失效」缺陷（§6），它们都会让 P2 在无报错的情况下拿到 0 结果。

## 2. 数据

| 项 | 值 |
|---|---|
| 来源 | hf 镜像 `hjjiao/TTPLA` 的 `data_original_size.tar.gz`（4,301.6 MB）；官方 Google Drive 本机不可达 |
| 许可 | Apache-2.0（已实读 LICENSE） |
| 解压后 | `data/raw/ttpla` 4.60 GB，1,242 对 `jpg + json`（labelme v4.2.7） |
| 官方 split | train 905 / val 109 / test 220（共 1,234；余 8 张不在任何 split，已丢弃） |
| 标注 | 原始 11,467 个 `shapes` → 映射后 **11,070** 个实例：conductor 10,102、tower 968 |
| 空类 | `insulator_string` 与 `reference_fitting` **零正样本**（TTPLA 没有这两类） |
| manifest | `data/manifests/detect_v1.jsonl`，1,234 条，每组一帧，`validate_dataset.py` 通过 |

分组键：前缀跨 split 重叠（同一架次的 `53_*` 同时出现在 train 与 test），
因此 `weather_event_id` 取文件名主干。**泄漏检查通过，但场景相似性未被约束**——
这是 TTPLA 不能当验收集的直接原因。

## 3. 训练

| 项 | 值 |
|---|---|
| 骨干 | `mobilenetv4_conv_small`（ImageNet 预训练，hf 镜像） |
| 输入 | 640×640 letterbox；batch 4；lr 5e-4；backbone lr ×0.1 |
| AMP | 显式关闭（Zen 2 无 bf16） |
| epoch | 10（未触发早停，patience 3） |
| 耗时 | **52 min 46 s**（14:47:44 → 15:40:30），约 5 min/epoch + 40 s 验证 |
| 内存 | 无 OOM；`n_pos` 稳定在 243.7/批 |

损失曲线：

| epoch | train | val | reg | giou |
|---|---|---|---|---|
| 0 | 9.1158 | 8.2801 | 7.078 | 0.601 |
| 4 | 5.8197 | 6.3594 | 5.460 | 0.450 |
| 9 | **4.7980** | **5.9546** | 5.110 | 0.422 |

best 落在最后一个 epoch，说明 10 epoch 远未收敛（符合"冒烟"定位）。

**`cls` 恒为 ≈0.0005，不是缺陷。** 分类头 bias 按 focal-loss 惯例初始化为
`log(99) ≈ 4.595`（`features/detection/model.py:69`），初始 p≈0.01；负样本
在 QFL 下被迅速压低。已核对 `features/detection/loss.py:221` 用的是**全部锚点**
的 `cls_all` 与全零 `quality`，负样本确有监督。

## 4. 指标（观测值，非验收）

测试集 220 张 / 2,169 个框（conductor 1,967、tower 202），评估耗时 57.3 s。

| 指标 | 值 |
|---|---|
| mAP@0.50 | **0.3242** |
| mAP@0.5:0.95（10 点） | 0.1464 |
| AP@0.50 conductor | 0.2879 |
| AP@0.50 tower | 0.3606 |
| AP@0.50 insulator_string | **无正样本** |
| AP@0.50 reference_fitting | **无正样本** |

口径说明：
1. 10 点阈值（0.50…0.95，步长 0.05），与 COCO 一致；
2. 完美检测的 AP 上限是 **100/101 = 0.9901**（101 点插值在 recall=1.0 处带 0 精度哨兵），
   这是口径特性，已在 `evaluation/detection.py` 注释锁住；
3. 空类不进均值，也不报 0。

## 5. 切片延迟

4K（3840×2160）、tile 640、overlap 0.2 → **32 个切片**，score_thr 0.30，20 张：

| 指标 | 值 |
|---|---|
| mean | 4,242 ms |
| p50 | 4,030 ms |
| p95 | 4,865 ms |
| max | 5,320 ms |
| 平均检出 | 106.7 框/张 |

与既有 1080P 基准（1,084 ms / 8 切片）自洽：切片数 ×4，耗时 ×4。
**结论不变：CPU 不满足 1080P 200 ms 硬门槛**，4K 更不行。
另外 106.7 框/张说明该权重在推理阈值下**误检很多**（10 epoch 训练不足，
阈值也未调优），不能直接用于演示。

## 6. 发现并修复的缺陷

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | `scripts/convert_ttpla.py` 按 COCO `objects[]` 解析，TTPLA 实为 labelme `shapes[]` | 1,242 个文件**全部跳过**，COCO 得到 0 条标注 | 支持双格式 + `void` 过滤 + 矩形补多边形 |
| 2 | `scripts/build_manifests.py` 要求 `site_camera_ts` 文件名 | TTPLA 的 `53_00231.jpg` 被 `except ValueError: continue` **静默丢弃**，manifest 为 0 条 | 新增 `scripts/build_ttpla_manifest.py`，按官方 split 建 1,234 条 |
| 3 | `evaluation/detection.py` 默认只算 3 个 IoU 阈值却输出 `mAP@0.5:0.95` | 口径被算宽，指标偏高 | 改为 10 点阈值（`DEFAULT_IOU_THRESHOLDS`），并补单测 |
| 4 | `tower_tucohy`（数据集拼写错误，实为 tubular） | 会丢失 231 个塔实例 | 显式映射到 `tower` |

以上 1、2、3 都不会报错，只会给出 0 或偏高的数字——正是仓库 `AGENTS.md` 想避免的
「静默失效」。三条都已加防回归测试。

## 7. 结论边界（不得写进验收材料）

1. **0.324 不是 Set-B 成绩**。Set-B 是现场侧视标注集，尚未采集。
2. TTPLA 是航拍俯视，与杆塔侧视差一个数量级；换成 InsPLAD 或 CPLID 也不能改变这一点。
3. 官方 split 无法按场景分组，指标含同类场景泄漏，天然乐观。
4. insulator / reference_fitting 两类**没有任何证据**，接线时不能假定可用。
5. 本次未做阈值调优、NMS 调参、TTA，也没有做多尺度训练。

## 8. 产物

| 路径 | 说明 |
|---|---|
| `data/raw/ttpla/` | 4.60 GB，1,242 对图与标注（gitignore） |
| `data/processed/ttpla_coco/annotations.json` | 11,070 条 COCO 标注 |
| `data/manifests/detect_v1.jsonl` | 1,234 条，带 sha256 |
| `artifacts/detector/det-smoke-0.1.0/best.pt` | 7.4 MB，epoch 9 |
| `artifacts/reports/detector_smoke_eval.json` | 评估原始结果 |
| `artifacts/logs_det_smoke.txt` | 训练日志 |

## 9. 复现

```bash
# 1. 取数据（约 4.3 GB）
$env:HF_ENDPOINT='https://hf-mirror.com'
.venv/Scripts/hf.exe download hjjiao/TTPLA data_original_size.tar.gz --repo-type dataset --local-dir data/raw/ttpla_hf
tar -xzf data/raw/ttpla_hf/data_original_size.tar.gz -C data/raw/ttpla

# 2. 转换 + manifest（split 清单从 GitHub API 取，raw.githubusercontent 本机不稳）
python scripts/convert_ttpla.py --src data/raw/ttpla --out data/processed/ttpla_coco/annotations.json
python scripts/build_ttpla_manifest.py --root data/raw/ttpla --splits data/raw/ttpla/splitting_dataset_txt
python scripts/validate_dataset.py --manifest data/manifests/detect_v1.jsonl

# 3. 训练 + 评估
giv train detector --config configs/train/detector_smoke.yaml
python scripts/eval_detector.py --config configs/train/detector_smoke.yaml --split test --tiled-bench 20
```
