# 域差检查：质量门控在真实图像上的行为

> 日期：2026-10-22 ｜ 脚本：`scripts/domain_gap_check.py` ｜ 原始数据：`artifacts/reports/domain_gap.json`
> 目的：验证「阈值是拍脑袋定的」这件事会不会在上线第一天就爆。

## 1. 结论

**会爆，而且已经爆过一次。**

原实现用「最平坦的单个网格块」判镜头异常（`min(block_energy) < 0.15` → degraded）。
杆塔机位大量看到天空，于是：

| 来源 | n | 修复前 usable | 修复后 usable |
|---|---|---|---|
| TTPLA（真实 4K 航拍） | 100 | **12%** | **100%** |
| WEAPD（网络图） | 100 | 3% | 91% |
| 合成（算子 B） | 8 | 0% | 100% |

修复前，88% 的真实图被判 `degraded`，而 `degraded` 会在
`pipeline._severity` 里直接置 `requires_review=True` —— 等于**所有正常帧都要求人工复核**，
系统的"自动筛查"价值归零。

## 2. 根因

`block_highfreq_energy()` 返回每个网格块的**原始拉普拉斯方差**。原逻辑取 `min()`：

```python
min_block = float(energy.min())
if min_block <= th.block_energy_unusable:   # 0.05
    status = DEGRADED
elif min_block < th.block_energy_degraded:  # 0.15
    status = DEGRADED
```

天空、水面、雪面、墙面——任何一块平滑区域都会让整帧降级。
实测触发原因统计（修复前）：

- TTPLA：`local_flat` 57 次 / `local_low_texture` 31 次（100 张图）
- WEAPD：`local_flat` 88 次

## 3. 修复

改为「平坦块**占比**」，并且只有整帧平坦才改状态：

```python
flat_ratio = float((energy < th.block_energy_degraded).mean())
if flat_ratio >= th.flat_block_ratio_unusable:   # 0.90 → degraded
    ...
elif flat_ratio >= th.flat_block_ratio:          # 0.60 → 仅记录原因
    ...
```

理由：
- **局部平坦是正常场景**，不是缺陷；
- 镜头糊住/结冰/被遮挡的表现是**整帧**平坦，且通常**随时间变化**（后者由
  `LensIcingTracker` 的连续帧规则负责）；
- 阈值进配置（`configs/model/quality.yaml` 的 `flat_block_ratio` /
  `flat_block_ratio_unusable`），可按机位标定。

回归测试：`tests/unit/test_grading_quality.py::test_sky_scene_is_not_degraded_by_single_flat_block`
与 `::test_fully_flat_frame_is_degraded`。

## 4. 仍未解决的问题（不要假装没有）

1. **`block_highfreq_energy` 不是归一化量**，它随分辨率与纹理尺度变化。
   0.15 这个阈值只在相近分辨率下可复用；4K 与 720P 混用必须分别标定。
   （已在该函数 docstring 里写明，避免下一个人再踩。）
2. **阈值没有按机位标定**。正确做法是用每个机位的晴天基线（同一套 L3 标定资产）
   算 `flat_ratio` 的分布，取分位数定阈值。当前值是"跨域可用的保守初值"。
3. **没有真实覆冰图**。本次用的 TTPLA 是航拍俯视，WEAPD 是网络图，
   都不能代表"冬季杆塔机位 + 覆冰"的成像条件。宝鸡覆冰数据集
   （DOI `10.57760/sciencedb.35688`，CC BY 4.0）的下载需要浏览器交互
   （ScienceDB 文件列表为前端渲染，`fileId` 未在 HTML 中出现），本轮未取到。
   这是下一步最该补的一环。
4. **合成图不能用来标定质量阈值**：其天空占 60–80%，`flat_ratio` 天然偏高；
   用合成图定阈值会把真实图全判成异常（反向踩坑）。

## 5. 另一个发现：合成器太慢

`make_synthetic_scene` 单张 **9.4 s**（640 个骨架点 × 64 个角度，各一次 `fillPoly`）。
生成 100 张要 15 分钟，这也是本次改用磁盘已有合成图的原因。
若要扩容合成数据（P3 需要 2000 张），必须先优化：按骨架点批量构造一个多边形、
或改用「沿骨架画变宽折线」的近似。**这是 P3 的前置条件，不是可选项。**
