# P1 天气分类基线报告（WEAPD）

> 数据：Xiao Haixia, *Weather phenomenon database (WEAPD)*, Harvard Dataverse,
> DOI `10.7910/DVN/M8JQCR`，**CC0-1.0**（2026-10-20 核实）
> 环境：Ryzen 7 4800U / 16 GB / **CPU only**；mobilenetv4_conv_small（ImageNet 预训练）
> 状态：**基线试跑**，不是最终训练结果

## 1. 数据落地

| 项 | 值 |
|---|---|
| 原始图像 | 6877 个文件 / 11 类 |
| 映射后可用 | **4869 张 / 7 类** |
| 丢弃类别 | dew 698、lightning 377、rainbow 232、sandstorm 692（非降水相态或无关） |
| 无样本类别 | `clear` —— 用 `weather_mask` 屏蔽该维，指标不计 |
| 分组键 | pHash 近重复簇（汉明距离 ≤ 6，跨类别合并） |
| 簇数 | **4467**（4869 张图 → 402 张被判为近重复并入他簇） |
| 划分 | train 3422 / val 729 / test 718（按簇，无跨 split 泄漏） |

类别分布（全部）：rime 1160、fog 855、glaze_evidence 639、snow 621、hail 592、rain 527、frost 475。

**为什么按 pHash 簇划分**：WEAPD 是网络抓图，同一原图存在缩放/裁剪/水印版本。
按帧随机划分会让近重复落进 test，指标虚高。簇划分能挡住"同图变换"，
但挡不住"同摄影师/同场景系列"——所以下面的数字仍是**上界**。

## 2. 训练配置

| 项 | 值 | 理由 |
|---|---|---|
| 输入 | 384×384 letterbox | 与部署配置一致 |
| batch / epoch | 16 / 15 | 实测 1.4 s/step |
| 学习率 | 1e-3，骨干 ×0.1 | 保护预训练特征 |
| AMP | off | Zen 2 无 AVX512-BF16 |
| 损失 | 多标签 BCE（masked）+ 0.5·CORAL | CORAL 对 unknown 自动忽略 |
| 强度头 | **未训练** | WEAPD 无强度标注，`n_intensity_valid=0` |

## 3. 结果

**训练了 6 个 epoch（约 45 min）后停止**，best 出现在 epoch 5（epoch 6–7 的 val 未再提升）。
停在这里的原因：单卡 CPU 边际收益已明显递减，完整 15 epoch 留给 GPU 阶段。
权重：`artifacts/precip/precip-0.1.0/best.pt`。

| 指标（test, n=718） | 值 |
|---|---|
| **macro-F1（7 类）** | **0.891** |
| top-1 accuracy | 0.894 |
| ECE | 0.006 |

| 类别 | Precision | Recall | F1 | support |
|---|---|---|---|---|
| fog | 0.977 | 0.984 | **0.981** | 128 |
| rime | 0.919 | 0.914 | 0.916 | 186 |
| hail | 0.955 | 0.875 | 0.913 | 72 |
| rain | 0.873 | 0.932 | 0.902 | 74 |
| glaze_evidence | 0.904 | 0.842 | 0.872 | 101 |
| frost | 0.938 | 0.750 | 0.833 | 60 |
| snow | 0.901 | 0.753 | 0.820 | 97 |

混淆矩阵（行=真实，列=预测，顺序 rain / snow / hail / fog / frost / rime / glaze）：

```
            rain  snow  hail   fog frost  rime glaze
rain          70     3     0     0     0     0     1
snow           5    80     1     1     0     5     5
hail           5     2    64     0     0     0     1
fog            0     0     0   127     0     1     0
frost          1     1     0     0    46     8     4
rime           1     3     0     0     4   172     6
glaze          0     3     3     2     1     9    83
```

主要混淆对：`snow ↔ rime/glaze/frost`、`frost → rime`、`glaze → rime`。
这与视觉直觉一致 —— 四者都是"白色/半透明附着物"，在 384² 下主要靠纹理与背景区分。

> ⚠️ **这组数字不能当验收依据**：域是网络图片，不是杆塔现场。它只证明
> "预训练骨干 + 多标签头 + CORAL 的接线正确，7 类在源域可分"。
> 按 `AGENTS.md` §3，不得写入任何验收材料。

## 4. 这个结果能说明什么

**能说**
- 预训练骨干 + 8 维多标签头 + CORAL 的接线正确（loss 正常下降、F1 显著高于随机）；
- 7 类视觉相态在**网络图片分布**上的相对可分性与主要混淆对；
- CPU 上"1–3 千张、384²、分类任务"的量级可训（≈6 min/epoch）。

**不能说**
- **现场精度**。WEAPD 是高清、白天、主体居中的网络图；现场是低分辨率、斜视、
  可能夜间/镜头结冰。域差是数量级的，**该 F1 不得写入验收材料**。
- **强度**。无标注，CORAL 恒为 0；推理侧必须关闭强度头（`Intensity.UNKNOWN`）。
- **`clear`**。无正样本，模型不会输出 clear 结论。
- **多标签共现**。WEAPD 单标签，学不到"雨 + 雾"的共现结构。
- **"发生冻雨"**。`glaze_evidence` 只表示视觉上像雨凇。

## 5. 已知问题（下一轮修）

1. `snow` 与 `rain`、`glaze_evidence` 混淆明显（首轮 snow F1=0，97 张里 30 张判成 rain）——
   需要更多 epoch，或对 snow/rain 做更强的外观增强；
2. `frost`/`glaze_evidence` 召回偏低（首轮 0.50 / 0.40），两者都是"近景植物特写"，
   与现场语义差距最大，属于**数据域问题**，不是调参能解决的；
3. `precip_step` 里的 `macro_f1_micro` 是微平均（被高频类主导），最终报告以
   `eval_weather.py` 的逐类 macro-F1 为准。
