# 电网覆冰与降水类型及程度识别算法 · 代码实现计划（合并版 v1.0）

> **文档定位**：本文件是对 `current-alternative-plans/` 下 6 份候选方案的审阅结论与合并实现计划，可直接作为立项、排期与编码依据。
> **编写日期**：2026-10-19
> **数据与算力前提**：仅有公开数据集 + 单卡 GPU（无电网现场标注数据、无实测冰厚真值）
> **技术主线**：四模块解耦 + 统一数据契约；覆冰走「ROI 检测 → 轻量分割 → 亚像素几何 → 等值冰厚」，天气走「轻量分类 + 气象物理门控」

---

## 目录

**Part A · 方案审阅**
1. [审阅方法与评分口径](#1-审阅方法与评分口径)
2. [六份方案逐份评审](#2-六份方案逐份评审)
3. [取长补短矩阵（14 个关键问题）](#3-取长补短矩阵14-个关键问题)
4. [事实核验与纠偏](#4-事实核验与纠偏)
5. [共同盲区与补强措施](#5-共同盲区与补强措施)

**Part B · 实现计划**
6. [项目定位与边界](#6-项目定位与边界)
7. [任务分解与数据契约](#7-任务分解与数据契约)
8. [标签体系与标注规范](#8-标签体系与标注规范)
9. [系统架构](#9-系统架构)
10. [算法设计](#10-算法设计)
11. [数据工程](#11-数据工程)
12. [代码目录结构](#12-代码目录结构)
13. [关键代码骨架](#13-关键代码骨架)
14. [分阶段实施计划](#14-分阶段实施计划)
15. [测试与评估](#15-测试与评估)
16. [部署与运维](#16-部署与运维)
17. [风险登记册](#17-风险登记册)
18. [附录](#18-附录)

---

# Part A · 方案审阅

## 1. 审阅方法与评分口径

### 1.1 审阅对象

| 文件 | 大小 | 来源/形态 | 是否有效 |
|---|---|---|---|
| `plan1.md` | 56 KB | 结构化长文，偏工程治理 | ✅ |
| `plan2.md` | 16 KB | 简洁版，YOLO 中心 | ✅ |
| `plan3.md` | 34 KB | 模型架构导向，含损失设计 | ✅ |
| `plan4.md` | **0 字节** | **空文件** | ❌ 无内容可评 |
| `plan5.md` | 35 KB | 业务工程视角，T1–T6 任务分解 | ✅ |
| `plan6.mhtml` | 4.7 MB | Gemini Deep Research 存档页（正文约 6.4 万字） | ✅ |

### 1.2 评分维度

- **工程可执行性**：按现有条件（单卡、无现场数据）能否在合理周期内产出可验证产物。
- **算法深度**：是否给出可编码的公式、损失、模块结构，而非只给名词。
- **科学严谨性**：是否区分"图像能看到什么"与"物理上能推出什么"，是否避免伪精确。
- **落地风险控制**：许可、数据泄漏、置信度、回滚等工程风险是否被识别。

### 1.3 一个重要观察

六份方案中，**没有任何一份同时做到"物理量算可编码"与"工程治理可执行"**：

- plan1 / plan5 治理强、算法弱；
- plan3 / plan6 算法强、治理弱；
- plan2 折中但两头都浅。

这正是合并的价值所在——不是选一份，而是**按章节分工取用**（见 §18 附录 C 溯源表）。

---

## 2. 六份方案逐份评审

### 2.1 plan1 —— 治理型总纲（★★★★☆）

**定位**：把项目当成"要交付给电网企业验收的系统"来写，而不是当论文来写。

**可直接采纳（5 项，全部进最终方案）**

1. **任务边界的三条诚实声明**，这是六份方案里最有价值的一段：
   - 识别到下雪 ≠ 线路已覆冰；
   - 单张 RGB 通常无法区分冻雨与普通雨；
   - 无标定条件下厚度必须返回 `null`，不得输出伪精确毫米数。
   → 直接决定了本方案的 `measurement_status` 三态设计（§7.2）。
2. **按站点/摄像头/天气过程分组划分数据集**。这是六份方案中**唯一**识别到帧级随机划分会造成测试集泄漏的（同一天气过程的连续帧高度相似，随机划分会让指标虚高）。→ 成为本方案的数据红线（§11.3）。
3. **Ultralytics 许可预警**：AGPL-3.0 在"不公开全部源码 + 私有业务部署"场景下可能需要 Enterprise License。→ 直接改变了本方案的检测器选型（§10.3）。
4. **统一 JSON 契约 + 模型版本 + 数据质量状态 + 可追溯字段**。→ 成为 §7.2 的 schema 基础。
5. **三套测试集（同分布 / 跨站点 / 困难场景）+ 里程碑退出条件**。→ 成为 §15.3 与 §14 的 DoD。

**硬伤**

- 算法章节只到"推荐 MobileNetV3 / U-Net + 轻量骨干"这一层，**没有一条可编码的公式**；厚度只给了 `t≈(D_outer-D_conductor)/2`，缺密度折算，也缺像素宽度怎么测。
- 8–12 周排期按"有团队、有数据"假设，对单卡单人偏乐观。
- 大量篇幅在讲"要写 AGENTS.md、要做许可审查"，但对**具体怎么做**着墨不足。

**结论**：作为**治理骨架**整章采纳，作为**算法方案**不采纳。

---

### 2.2 plan2 —— YOLO 快速 MVP（★★★☆☆）

**定位**：以最短路径做出能跑的东西。

**可直接采纳（3 项）**

1. **MVP 优先原则**：先用最简模型打通端到端，再谈架构。→ 成为 P1/P2 阶段的策略。
2. **公开数据清单具体**（WEAPD / TIC / DTL-IceNet / HF 天气模型），并标注了每项的"可复用部分"与"局限"。→ 并入 §11.1 资源台账。
3. **厚度近似式** `h ≈ 0.5·(D·d₂/d₁ − D)`，形式正确（本质就是 `0.5·(W_ice − W_0)·α`）。→ 作为 §10.6 的退化形式保留。

**硬伤**

- **把检测与分割混为一个 YOLO-seg 模型**：`导线_无冰 / 导线_雨凇 / 导线_雾凇` 这种"设备 + 状态"耦合的类别设计，会让"设备定位错误"和"覆冰判断错误"无法区分，排障时无法归因。这是任务定义层面的错误。
- 完全未提许可（而它恰恰推荐了 Ultralytics）。
- 无拒识、无置信度校准、无数据泄漏控制、无 unknown 类。
- 指标只有 mAP，未考虑覆冰样本稀少时"全判无冰"也能拿到高 accuracy 的陷阱。

**结论**：采纳其**节奏感**与**数据清单**，不采纳其任务建模。

---

### 2.3 plan3 —— 模型架构型（★★★★☆）

**定位**：六份里唯一认真设计损失函数的。

**可直接采纳（5 项）**

1. **有序损失（CORAL / Corn）用于严重程度分级**。这是关键洞察：把"重度"错判为"无"和错判为"中度"，代价完全不同，普通 CrossEntropy 不区分。→ 成为 §10.5。
2. **FN-aware 的 Dice + Focal 组合**：漏检覆冰的代价高于误检，Focal 抑制易分负样本、Dice 直接优化重叠度。→ 成为 §10.4 的损失基线。
3. **Ice-Seg 三类场景失败模式诊断**（晴天：冰 vs 云/反光/雪地；恶劣天气：漏检为主；设备背景：杆塔几何 vs 覆冰）→ 直接映射为**场景分层采样器**与分组增强策略。→ 成为 §11.5。
4. **覆盖率分级 rubric（0–4 级 + 判据）**，且明确"单目绝对厚度不可靠，改用覆盖率/形态分级"。→ 成为 §8.5 的视觉等级定义。
5. **A/B/C 三套测试集（公开留出 / 电网域内 / 冻雨专项）**，并要求单独报告冻雨 Recall。→ 并入 §15.3。

**硬伤**

- **M1 主训练集押在 `firc-dataset` 上，而该资源实为 Gitee/FIRC 数据交易平台产物**（详见 §4），许可与精度均无保证，把它当作"公开数据集"是重大误判——整个 M1 可能直接卡死。
- 放弃物理厚度过于彻底：只做覆盖率分级，等于丢掉了电网最关心的 mm 量纲，无法与设计冰厚/力学传感器对齐。
- 没有站点级泄漏控制（只有"Set-B 从实际场景采集"一句）。
- 无许可分析（它用的 firc 恰是许可最不清的一个）。

**结论**：采纳**损失设计与失败模式分析**，替换其数据底座。

---

### 2.4 plan4 —— 空文件（无评分）

`plan4.md` 为 0 字节。无法评审，也无法判断是生成失败还是刻意留空。**建议**：若原计划还有第 4 个视角（例如"多模态/时序"或"数据标注工程"），需重新生成；本方案已通过 §10.7（时序 EKF）与 §8（标注规范）覆盖该空白。

---

### 2.5 plan5 —— 业务工程型（★★★★☆）

**定位**：最像"给甲方看的实施方案"，工程闭环意识最强。

**可直接采纳（5 项）**

1. **开篇的诚实声明**："本计划基于可核实的公开技术知识，而非实时联网抓取，Phase 0 必须安排链接与可用性复核"。→ 本方案已把这一步**提前做完**（§4），并保留"待核实"标注制度。
2. **T1–T6 任务分解**，把"降水强度分级""部件定位""可解释性"提升为一等任务而非附属品。→ 并入 §7.1。
3. **两个真正有迁移价值的建模范式**：
   - DID-MDN 的"密度分类子网络"→ 有序严重度头；
   - DesnowNet 的"透明度/覆盖掩码"→ 覆冰覆盖掩码。
   → 作为 §10.4/§10.5 的方法论来源。
4. **不确定性 + 多帧时序平滑 + 人工复核队列 + 漂移监控 + model card**。→ 成为 §16 运维章节骨架。
5. **"极端天气（冰雹）样本近乎不可获取 → 该类别初期以告警降级 + 人工确认处理，不追求高自动化精度"**。这是六份里唯一敢明确"放弃某类精度"的，非常务实。→ 成为 §6.3 边界声明之一。

**硬伤**

- 厚度只有"几何标定"四个字，没有公式、没有标定方法、没有可信度判据。
- 验收目标只有一句"Macro-F1 ≥ 0.75、IoU ≥ 0.6（内部目标值）"，且未区分硬门槛与软目标。
- 5 个 Phase 串行 16–20 周，Phase 3（鲁棒性）与 Phase 4（部署）之间耦合描述不足。
- 目录结构里 `models/` 与 `src/gridvision/models/` 并存，权重与代码混淆。

**结论**：采纳**运维闭环与务实边界**，补齐其厚度与指标缺口。

---

### 2.6 plan6 —— 研究深潜型（技术 ★★★★★ / 可执行 ★★☆）

**定位**：Gemini Deep Research 产出，技术密度远高于其余五份，但工程判断明显失控。

**可直接采纳（6 项，本方案的核心增量）**

1. **完整的物理量算链**，这是全部方案中唯一可编码的：
   - 掩码 → 形态学中轴骨架 → 三次样条平滑；
   - 骨架采样点求切向 τ 与法向 n；
   - 沿法向在边缘概率图上做**亚像素插值**定位上下边界；
   - 逐截面宽度 → IQR 截尾均值 → `W_ice_px`；
   - 与裸导线基准宽度 `W_0_px` 求比值 → `α` → 径向厚度 → **密度折算到 900 kg/m³ 的等值冰厚**。
   → 成为 §10.6 主体。**其中密度折算是六份里唯一物理正确的做法**（其余方案要么不算密度，要么直接给像素差）。
2. **三级动态标定机制**：刚性金具（耐张线夹/防振锤/绝缘子钢帽）→ PTZ 焦距与双目视差 → 晴天建档 `W_0_px`。这是对"球机变焦导致 α 失效"这一真实工程问题的正解。→ §10.6.4。
3. **I–IV 级预警矩阵**：以 `d_eq`、载荷比 `γ = d_eq/d_design`、绝缘子桥接率 `η_bridge`、降水相态四者联合定级，并给出联动处置。→ §10.7.3。
4. **EKF 时序滤波 + 物理增长率约束**，用来剔除暴风雪中飞掠前景雪花造成的单帧厚度尖峰。→ §10.7.2。
5. **C2/C7/C11 冰截面参数化合成 + Makkonen 虚拟气象配对**。前者解决"合成冰层形状不真实"，后者解决"没有同步气象记录"。→ §11.4。
6. **WA-ADDA（气象条件对抗领域自适应）**：以气象后验为条件约束域判别器，避免"把铁塔背景差异抹掉的同时把雨纹特征也抹掉"。→ 作为 P4 可选进阶（§10.8）。

**硬伤（必须纠偏，否则会误导排期）**

1. **验收指标直接抄论文**：`mIoU ≥ 96%`、`MAPE ≤ 5%`、`亚像素误差 ≤ 0.5px`、`mAP@0.5 ≥ 93.5%`。这些是论文在**自建、同分布、高质量标注**数据上的成绩，不是本项目的可达目标。按此设指标，P3 阶段必然"不达标"，然后团队会开始调测试集——这是最坏的结果。→ 本方案全部降级为"文献参考值"，另设可达目标（§15.4）。
2. **排期严重脱离实际**：16 周内完成 4 阶段级联 + GMSA-Net/CG-UNet 源码级复现 + TensorRT 部署 + 72 小时压测，在单卡单人条件下不可行。
3. **两处仓库描述失实**（见 §4）：`nisarg1631/Weather-Image-Recognition` 是 HOG+PCA+SVM，不是"PyTorch 管线"；`PLAIR-AW` 是数据集与基线评测仓库，**未提供合成脚本**，"可直接提取其数学合成脚本"不成立。
4. **无 unknown 类**：覆冰分割只有 0/1/2/3 四类，没有"无法判断"。在雾天/逆光/遮挡场景下，模型会被迫在坏图上给出确定答案。
5. **无泄漏控制、无许可审查**（却大量使用 Ultralytics 与论文私有代码）。
6. **行文冗余**：大量 `[cite: xx]` 噪音与公式重复渲染（LaTeX 与纯文本双份），可读性差。

**结论**：**把 plan6 当技术字典用，不当施工图用**。物理量算、标定、预警矩阵整段吸收；指标、排期、仓库描述全部重写。

---

## 3. 取长补短矩阵（14 个关键问题）

| # | 关键问题 | plan1 | plan2 | plan3 | plan5 | plan6 | **本方案决策** |
|---|---|---|---|---|---|---|---|
| 1 | 天气与覆冰是否独立输出 | ✅ 独立 | ❌ 混在 YOLO 类里 | ⚠️ 多任务同模型 | ✅ 独立 | ✅ 独立 | **独立四模块**，天气结论不得推导覆冰结论（§6.3） |
| 2 | 覆冰类型细分粒度 | 先三态（无/有/不确定） | 无 | 4 类 | 6 类 | 4 类 + 晶态 | **先二态，再 5 类**（glaze/rime/mixed/wet_snow/unknown），样本 <50 的类合入 mixed（§8.4） |
| 3 | 程度量化 | 覆盖率 + 可选厚度 | 像素比 | 覆盖率分级 | 等效厚度分级 | **亚像素几何 + 密度折算** | **双轨**：覆盖率分级（永远可用）+ 等值厚度（有标定时），三态可信度（§10.6） |
| 4 | 损失设计 | 无 | 无 | **CORAL + FN-aware Dice/Focal** | Focal/CB | CE+Dice+Boundary | **Dice+Focal+Boundary 分割；CORAL 程度；BCE 多标签天气**（§10.4/10.5） |
| 5 | 数据泄漏控制 | ✅ **站点/过程分组** | ❌ | ❌ | ❌ | ❌ | **分组键 = site_id + camera_id + weather_event_id**，manifest 强制校验（§11.3） |
| 6 | 小样本/长尾 | 困难负样本 | 过采样 | CB-CE / logit-adjust | 伪标签 | WA-ADDA | **CB-CE + 分层采样 + 合成 + 拒绝低置信**，不做伪标签（无现场数据时伪标签无意义）（§11.5） |
| 7 | 合成数据 | 谨慎、单独管理 | CycleGAN | 有限使用 | 物理渲染 | **C2/C7/C11 参数化 + Makronen** | **合成只进 train，永不进 test**；合成集必须带零误差几何真值（§11.4） |
| 8 | 置信度与拒识 | ✅ ECE/拒识 | ❌ | ⚠️ | ✅ MC-Dropout | ❌ | **温度标定 + 熵阈值拒识 + unknown 类**（§10.2.4） |
| 9 | 开源许可 | ✅ **Ultralytics AGPL 预警** | ❌ | ❌ | ⚠️ 一句 | ❌ | **生产链路避开 AGPL**：timm + 自建头；Ultralytics 仅预标注（§10.3） |
| 10 | 部署 | ONNX→TRT/OpenVINO 分档 | TRT | TRT FP16 | 多后端 | TRT + LAP 剪枝 | **ONNX 先行 + FP16 可选 + 量化前必做精度对照**；不承诺 INT8（§16） |
| 11 | 可解释性 | mask 叠加 | ❌ | Grad-CAM | Grad-CAM + 不确定性 | LRP + 卡尺叠加 | **测量卡尺叠加图**（最有说服力）+ mask + 概率热图；不做 LRP（§16.6） |
| 12 | 时序利用 | 提了一句 | ❌ | TSM 建议 | 多帧平滑 | **EKF + 增长率约束** | **EKF（2 状态）+ 连续 k 帧确认去抖**（§10.7） |
| 13 | 工程治理 | ✅ 最完整 | 简 | 中 | 中 | 强（pyproject/AGENTS） | **plan1 治理 + plan6 的 pyproject/AGENTS 细节**（§12/§13） |
| 14 | 验收指标 | 建议目标 + 分任务 | mAP | 分任务 | 模糊 | **抄论文** | **三栏制：硬门槛 / 软目标 / 文献参考**（§15.4） |

---

## 4. 事实核验与纠偏

以下核验于 2026-10-19 实际访问完成，结论直接改变选型。

| 资源 | 核验结论 | 对计划的影响 |
|---|---|---|
| **DTL-IceNet**（Zenodo 15718305） | ✅ 存在。`example_icing_images.zip` 130.9 MB + `Ice_code.zip` 146.1 kB；作者 Fu Yufei 等；**页面未显示明确 License** | 仅作架构参考与少量样例验证；**不得作为唯一主训练集**；商用前须书面确认授权 |
| **PLAIR-AW**（ntuhubin） | ✅ 存在。仓库内容为 **数据集说明 + 基线评测**（Haze/Rain/Snow × CPLID/TTPLA/InsPLAD），未见合成脚本 | 用作"退化图像 + L/M/H 强度标签"来源；**合成算子必须自研**。plan6"可直接提取其数学合成脚本"**不成立** |
| **OmniRestore**（Judith989） | ✅ 存在。CVPRW 2026 NTIRE，2.60M 参数、6.3 ms@256²、CDD-11 PSNR 29.86 | 可作天气嵌入器/轻量复原的可选组件；**但 CDD-11 是通用退化集，电网域效果需自测**，不列入 MVP |
| **TTPLA**（R3ab） | ✅ 存在。COCO 多边形标注，含 `resize_image_and_annotation-final.py` / `remove_void.py` / `split_jsons.py` / `labelme2coco_2.py`；最后更新 2020；**仓库未见 LICENSE** | 导线/杆塔 ROI 与"裸导线基准宽度"来源；域为**航拍俯视**，与杆塔侧视差异大，只能做预训练 |
| **Weather-Image-Recognition**（nisarg1631） | ✅ 存在，但代码是 **HOG + PCA + SVM/DT/NB（sklearn）**；数据来自 Kaggle `jehanbhathena/weather-dataset`，6862 张 11 类（dew/fogsmog/frost/glaze/hail/lightning/rain/rainbow/rime/sandstorm/snow） | **数据可用**（glaze/rime/frost/hail 是稀缺类），**代码不可用**（非深度模型）。plan6 称其为"PyTorch 管线"**有误** |
| **firc-dataset** | ⚠️ 实为 Gitee/FIRC「电力行业数据集」交易平台资源（`gitee.com/FIRC/power-industry-dataset`），知乎/CSDN 引流，**明确声明"不对模型精度作保证"**，许可不明 | plan3 的 M1 主训练集定位**不可行**；降级为"若免费可获取则作补充对照" |
| **NWPUYRCC2 / Sentinel-LakeIce** | 真实存在，但为**遥感河冰/湖冰**分割 | 域差过大（俯视、大尺度、SAR/光学），**仅可作冰-水边界纹理先验**，不混训 |
| **CoInfra** | 路侧协同感知数据集（CC-BY-NC-4.0） | 场景差异大，**仅作域适应消融**，且 NC 条款禁止商用 |

> **纠偏原则**：本文件所有外部资源均标注核验状态。未核验的一律写「待核实」，不做肯定性描述。

---

## 5. 共同盲区与补强措施

五份有效方案**都没有**系统处理以下 5 件事，本方案逐条补上：

| 盲区 | 后果 | 本方案补强位置 |
|---|---|---|
| ① 视觉厚度与力学/称重传感器的互校 | 视觉厚度无人校验，错了也不知道 | §10.6.6 三源互校（视觉/力学反演/人工实测），差异 >30% 触发降级 |
| ② 合成数据与真实测试集的物理隔离 | 合成集指标好看，真实场景崩 | §11.4 目录级隔离 + manifest `is_synthetic` 字段 + 测试集硬校验 |
| ③ 置信度校准与拒识阈值 | 坏图上强行给结论，误报摧毁信任 | §10.2.4 温度标定 + 熵阈值 + unknown 类 + 拒识率报告 |
| ④ 许可与数据授权台账 | 交付时才发现不能商用 | §11.1 资源台账含许可列 + P0 硬门槛「无许可不得进训练集」 |
| ⑤ 现场数据缺位时的降级交付定义 | 无限期等数据，项目烂尾 | §6.3 明确「无现场数据时交付到什么程度算完成」 |

---

# Part B · 实现计划

## 6. 项目定位与边界

### 6.1 一句话定位

面向输电线路固定监控与无人机巡检图像，构建一套**可解释、可追溯、带物理量纲**的多任务视觉识别系统，输出「降水相态 + 部件覆冰状态 + 覆冰程度」三类结构化结果，作为**状态感知与辅助决策**模块接入电网运维平台。

### 6.2 输入输出

**输入**

```text
X = (I, meta, calib)
I     : RGB 图像 (H×W×3)，固定监控 1080P 或无人机 4K 裁切
meta  : {site_id, camera_id, asset_id, timestamp(tz), ptz{zoom,pan,tilt},
         weather_sensor{T_amb, RH, v_wind, P_atm} | null}
calib : {D0_mm(导线外径/绝缘子盘径), design_ice_mm, W0_px(晴天建档) } | null
```

**输出**（统一契约，见 §7.2）

```text
Y = { quality, precipitation, assets[], icing, severity, evidence, provenance }
```

### 6.3 明确不做的事（边界声明，必须写进 README）

| # | 不承诺 | 原因 |
|---|---|---|
| 1 | 不承诺从单张 RGB 判定「正在下冻雨」 | 冻雨与普通雨在可见光下表观可高度相似；本系统只输出「视觉相态」与「疑似 glaze 证据」，物态确认需气象观测融合 |
| 2 | 不承诺无标定条件下的绝对冰厚 | 无尺度参考时 `thickness_mm = null`，只给覆盖率与视觉等级 |
| 3 | 不承诺冰雹类别的自动化精度 | 样本近乎不可获取；初期策略为「检测到即告警 + 人工确认」，不追求 F1 |
| 4 | 不承诺夜间/红外性能 | 无夜间标注数据；夜间的 `quality.status` 直接置 `unusable` |
| 5 | 不用图像模型替代运行管理判断 | 输出为辅助信息；I–IV 级预警必须走人工复核后才可下发处置指令 |
| 6 | 不在 MVP 阶段做端到端多任务共享骨干 | 小样本下负迁移风险高于收益；留作 P4 消融实验 |
| 7 | 不承诺「无现场数据也能达到生产精度」 | 公开数据只能建立方法与基线；生产精度以现场数据到位为前提 |

### 6.4 术语约定

| 术语 | 定义 |
|---|---|
| **视觉相态** | 仅凭图像可观察的降水/结冰形态，不等于气象学确认的降水类型 |
| **等值覆冰厚度 `d_eq`** | 折算到标准密度 ρ₀=900 kg/m³ 的均匀圆柱壳厚度，单位 mm |
| **径向厚度 `d_radial`** | 导线表面单侧实际冰层厚度，未做密度折算 |
| **载荷比 `γ`** | `d_eq / d_design`，与设计冰厚之比 |
| **桥接率 `η_bridge`** | 绝缘子相邻伞裙间被冰凌连通的比例 |
| **视觉等级 G0–G4** | 基于覆盖率与形态的项目内部等级，非电力行业规程等级 |
| **预警等级 I–IV** | 业务预警分级，需人工复核确认 |

---

## 7. 任务分解与数据契约

### 7.1 任务分解

| ID | 任务 | 输入 | 输出 | 模型 | 依赖 |
|---|---|---|---|---|---|
| **T0** | 图像质量门控 | 原图 | 可用性、模糊/曝光/遮挡/镜头结冰标记 | 规则 + 轻量分类器 | — |
| **T1** | 降水与天气分类 | 原图（可选气象向量） | 8 类多标签概率 + 强度 L/M/H | 轻量 CNN + 物理门控 | T0 |
| **T2** | 电力部件 ROI 检测 | 原图 | 导线/绝缘子串/杆塔/参考金具 bbox | 轻量检测器 | T0 |
| **T3** | 覆冰分割 | ROI 裁切 | 4 类像素掩码（背景/裸露设备/覆冰/冰凌桥接） | 轻量分割网络 | T2 |
| **T4** | 覆冰类型与视觉程度 | ROI + 掩码 | 类型 5 类 + G0–G4 | 分类/有序回归头 | T3 |
| **T5** | 物理量算与决策 | 掩码 + 边缘 + 标定 | `d_radial`、`d_eq`、`γ`、`η_bridge`、I–IV | 几何解算 + EKF + 规则 | T3/T4 |

**编排关系**

```text
              ┌── T1 天气分类 ──────────────┐
I ── T0 ──────┤                              ├── T5 量算与决策 ── Y
              └── T2 ROI ── T3 分割 ── T4 ───┘
                              │
                              └── 掩码/边缘 ──→ T5
```

> T1 与 T2/T3 并行，互不阻塞。这是"解耦"的实际收益：天气分支可以在没有覆冰标注的情况下先跑通。

### 7.2 统一数据契约（JSON Schema v1）

```json
{
  "schema_version": "1.0.0",
  "image_id": "S03_C07_20260115T083012",
  "source": {
    "site_id": "S03", "camera_id": "C07", "asset_id": "LINE_220kV_03",
    "timestamp": "2026-01-15T08:30:12+08:00", "ptz": {"zoom": 12.0, "pan": 0.0, "tilt": -3.5}
  },
  "quality": {
    "status": "usable",
    "blur_score": 0.12,
    "exposure": "normal",
    "occlusion_ratio": 0.03,
    "lens_icing_suspected": false,
    "reasons": []
  },
  "precipitation": {
    "labels": {
      "clear": 0.02, "rain": 0.91, "snow": 0.03, "hail": 0.00,
      "fog": 0.11, "frost": 0.00, "rime": 0.01, "glaze_evidence": 0.06
    },
    "intensity": "moderate",
    "intensity_conf": 0.71,
    "meteorological_input_used": true,
    "note": "visual_only; freezing-rain cannot be confirmed from image alone"
  },
  "assets": [
    {"cls": "conductor", "bbox": [412, 220, 1580, 268], "score": 0.94,
     "roi_id": "r0", "reference_width_px": null},
    {"cls": "insulator_string", "bbox": [690, 180, 760, 460], "score": 0.88,
     "roi_id": "r1", "reference_width_px": 42.6}
  ],
  "icing": {
    "present": true,
    "confidence": 0.93,
    "coverage_ratio": 0.24,
    "ice_type": "glaze",
    "ice_type_conf": 0.78,
    "bridge_ratio": 0.31,
    "mask_rle": {"size": [1080, 1920], "counts": "..."}
  },
  "severity": {
    "visual_grade": "G2",
    "alarm_level": "II",
    "d_radial_mm": 6.4,
    "d_eq_mm": 7.1,
    "gamma": 0.36,
    "measurement_status": "measured",
    "calibration_source": "rigid_fitting",
    "alpha_mm_per_px": 0.152,
    "uncertainty_mm": 1.8,
    "requires_review": true
  },
  "evidence": {
    "caliper_overlay": "r0_caliper.png",
    "mask_overlay": "r0_mask.png",
    "heatmap": "r0_prob.png"
  },
  "provenance": {
    "model_version": {"quality": "q-1.0.0", "precip": "p-1.1.0",
                      "det": "d-1.0.2", "seg": "s-1.2.0"},
    "config_hash": "a3f9...",
    "inference_ms": 87.4,
    "rule_version": "alarm-1.0.0"
  }
}
```

**字段语义要点**

| 字段 | 取值 | 语义 |
|---|---|---|
| `quality.status` | `usable` / `degraded` / `unusable` | `unusable` 时下游一律不给结论 |
| `measurement_status` | `measured` / `visual_only` / `not_calibrated` | **三态是本方案的核心设计**：`measured` 才允许进入预警计算；`visual_only` 只有覆盖率；`not_calibrated` 时 `d_eq_mm = null` |
| `alarm_level` | `null` / `I` / `II` / `III` / `IV` | 仅在 `measurement_status == measured` 且通过时序去抖后才有值 |
| `requires_review` | bool | 低置信、跨源不一致、等级 ≥III 时为 true |
| `schema_version` | 语义化版本 | 破坏性变更升主版本；下游按版本分支解析 |

**兼容规则**：新增字段必须给默认值；删除/改名字段升主版本并保留一个版本的兼容解析层。

---

## 8. 标签体系与标注规范

### 8.1 双体系原则

沿用 plan1 的关键设计：**维护两套标签，永不互相推导**。

- `weather_visual`：图像上**能看到**的现象（可由标注员独立判断）；
- `precip_confirmed`：经气象观测/记录**确认**的降水类型（无记录时为空）。

二者在 manifest 中分列，训练时只用 `weather_visual`，评估融合模块时用 `precip_confirmed`。

### 8.2 天气/降水标签（8 类，多标签）

| ID | 标签 | 判定要点 |
|---|---|---|
| 0 | `clear` | 无可见降水，能见度良好 |
| 1 | `rain` | 可见雨丝/雨滴/湿地反光，非冻结 |
| 2 | `snow` | 可见雪花/积雪，松散白色 |
| 3 | `hail` | 可见冰粒/积雹（长尾，初期不追求精度） |
| 4 | `fog` | 大气散射导致远景对比度显著下降 |
| 5 | `frost` | 表面白色晶霜，多呈针状/羽状，附着薄 |
| 6 | `rime` | 雾凇：白色疏松、不透明、迎风面堆积 |
| 7 | `glaze_evidence` | 雨凇/疑似冻雨证据：透明致密冰壳、有光泽、可挂冰柱 |

> **多标签而非单标签**：现场常见「冻雨 + 浓雾」「雨夹雪」。用 8 个独立 sigmoid 而非 softmax。
> `glaze_evidence` 命名刻意保留 "evidence"——它表示"看到像雨凇的东西"，不表示"确认发生冻雨"。

**强度**：`none / light / moderate / heavy / unknown`，标注依据为图像可见强度（雨丝密度、雾的消光程度、积雪覆盖速度），**不要求与气象站降水量一致**。

### 8.3 资产标签

| 标签 | 说明 | 用途 |
|---|---|---|
| `conductor` | 导地线（含分裂导线整体） | 覆冰分割主目标 |
| `insulator_string` | 绝缘子串 | 桥接率计算 |
| `tower` | 杆塔本体 | 上下文，抑制误检 |
| `reference_fitting` | 刚性参考金具（耐张线夹/防振锤/间隔棒/钢帽） | **尺度标定**，关键 |

### 8.4 覆冰像素标签（4 类语义分割）

| ID | 标签 | 说明 |
|---|---|---|
| 0 | `background` | 天空、山体、植被、其他 |
| 1 | `exposed_asset` | 未覆冰的导线/绝缘子/金具本体 |
| 2 | `ice_accretion` | 附着冰层 |
| 3 | `icicle_bridge` | 冰凌/桥接冰柱 |

> **为什么必须有 `exposed_asset`**：只有同时标出"裸露设备"，才能算出"覆冰/（覆冰+裸露）"的覆盖率；否则覆盖率的分母只能用整张图，毫无意义（plan1 已指出这点，plan3/plan6 均未落实）。

### 8.5 覆冰类型与程度

**类型（5 类）**

| 标签 | 视觉判据 | 备注 |
|---|---|---|
| `none` | 无覆冰 | — |
| `glaze` | 透明/半透明、致密、有光泽、可挂冰柱 | 密度高（≈900 kg/m³） |
| `rime` | 白色、疏松、不透明、迎风面堆积 | 密度低（≈200–500） |
| `mixed` | 上述特征混合，或无法明确归类 | **合并容器**：单类样本 <50 时一律并入此类 |
| `unknown` | 图像质量不足，无法判断 | 拒识出口，必须有 |

**视觉等级（G0–G4）**

| 等级 | 覆盖率（ice / (ice+exposed)） | 形态判据 |
|---|---|---|
| G0 | < 3% | 轮廓清晰，无明显增厚 |
| G1 | 3%–15% | 边缘轻微增厚，导线轮廓基本保持 |
| G2 | 15%–30% | 轮廓明显改变，可见冰壳/短冰柱 |
| G3 | 30%–50% | 形态显著畸变，可见下垂/偏心/桥接 |
| G4 | > 50% | 大面积冰壳，跨导线桥接，或伴随明显弧垂 |

> 阈值是**初始值**，须在真实标注数据上按分布校准（写入 `configs/labels/severity.yaml`，不硬编码）。
> G 级是**项目内部指标**，不得直接等同于任何电力行业规程等级。

### 8.6 标注字段与质检

| 字段 | 必填 | 说明 |
|---|---|---|
| `image_id` | ✅ | 全局唯一 |
| `site_id` / `camera_id` / `asset_id` | ✅ | 分组键，泄漏控制依赖它 |
| `timestamp` | ✅ | 带时区 |
| `weather_visual[]` | ✅ | 多标签 |
| `precip_confirmed` | ❌ | 无观测记录留空 |
| `intensity` | ✅ | L/M/H/unknown |
| `assets[]` | ✅ | bbox + 类别 |
| `ice_mask` | ✅（覆冰图） | 4 类像素掩码 |
| `ice_type` | ✅ | 5 类 |
| `visual_grade` | ✅ | G0–G4，由专家按 rubric 定 |
| `d_ref_mm` / `w_ref_px` | ⚠️ | 有参考物时填，用于标定 |
| `d_measured_mm` | ❌ | 有人工实测时填，作为**唯一**真值 |
| `label_confidence` | ✅ | 标注员自评（high/medium/low） |
| `annotator` / `reviewer` | ✅ | 双人标注 + 仲裁 |

**质检**：每批 10% 抽检，Cohen's κ ≥ 0.75 方可入库；争议样本进入仲裁队列。

### 8.7 边界案例判定规则（写进标注手册）

| 易混对 | 判定规则 |
|---|---|
| 薄冰 vs 水渍反光 | 水渍随视角/光照变化且无厚度；薄冰有明确边界与厚度梯度 → 无厚度证据一律标 `background` 或 `exposed_asset` |
| 雾凇 vs 霜 | 霜薄、均匀、多在地面/设备顶面；雾凇迎风侧堆积、有体积 → 有体积标 `rime`，无体积标 `frost` |
| 积雪 vs 覆冰 | 积雪覆盖设备上方、边缘松散；覆冰包裹、贴合轮廓 → 包裹标 `ice_accretion`，堆积标 `background`（雪）并在 weather 标 `snow` |
| 覆冰 vs 杆塔角钢 | 角钢为直线、规则、灰黑色；覆冰沿设备法向生长、有光泽 → 不确定标 `unknown` 而非猜 |

---

## 9. 系统架构

### 9.1 分层

```text
┌─────────────────────────────────────────────────────────────┐
│  service/           FastAPI · 请求校验 · 错误码 · 限流        │  ← 只做 IO
├─────────────────────────────────────────────────────────────┤
│  application/       pipeline 编排 · EKF 时序 · 预警规则引擎    │  ← 只做编排
├─────────────────────────────────────────────────────────────┤
│  features/          quality │ precip │ detect │ ice │ severity│  ← 只做算法
├─────────────────────────────────────────────────────────────┤
│  training/          trainer · losses · callbacks · ckpt       │
├─────────────────────────────────────────────────────────────┤
│  data/              manifest · readers · validation · splits  │
├─────────────────────────────────────────────────────────────┤
│  common/            schemas(Pydantic) · config · logging · io │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 模块边界（硬规则，写入 AGENTS.md）

| 规则 | 理由 |
|---|---|
| `features/*` 不得读写文件、不得解析配置、不得打印日志 | 保证可单测、可复用 |
| `features/severity/` 不 import 任何模型代码，只接受 mask + 标定参数 | 物理量算必须能脱离模型独立测试 |
| 阈值一律走 `configs/`，禁止在 `features/` 里出现魔法数字 | 业务阈值必须可审计、可回滚 |
| `service/` 不得直接 import `training/` | 训练依赖不得进入生产镜像 |
| 所有跨模块传递必须用 `common/schemas.py` 的 Pydantic 对象 | 契约变更即报错，避免静默不兼容 |

### 9.3 失败模式与降级路径

| 失败点 | 降级动作 | 输出标记 |
|---|---|---|
| 图像不可用 | 终止推理 | `quality.status=unusable`，其余字段 null |
| 无检测到任何资产 | 只跑 T1 天气 | `icing.present=null` |
| 分割置信度低 | 保留掩码但不给类型 | `ice_type=unknown` |
| 无尺度参考 | 只给覆盖率 | `measurement_status=not_calibrated` |
| 气象输入缺失 | 关闭物理门控 | `meteorological_input_used=false` |
| 时序帧数不足 | 跳过 EKF | `severity.alarm_level=null` |
| 跨源不一致（视觉 vs 力学） | 降级 + 转人工 | `requires_review=true` |

---

## 10. 算法设计

### 10.1 T0 图像质量门控

**规则层（先跑，零成本）**

```python
blur = cv2.Laplacian(gray, cv2.CV_64F).var()          # 归一化后取对数
over_exp = (gray > 250).mean()
under_exp = (gray < 15).mean()
# 镜头结冰/污损：固定像素区域的高频能量持续下降
lens_score = local_highfreq_energy(patch_grid)         # 8×6 网格
```

**判据**（阈值进配置）

| 指标 | `degraded` | `unusable` |
|---|---|---|
| 拉普拉斯方差（归一化 0–1） | < 0.08 | < 0.03 |
| 过曝/欠曝比例 | > 0.25 | > 0.45 |
| 网格高频能量最小块 | < 0.15 | < 0.05 |

**镜头结冰自检**：若某 3×3 网格区域的高频能量在**连续 N 帧**（同一 camera_id）持续低于阈值，且位置固定 → `lens_icing_suspected=true`，触发告警（提醒加热）。

> 这一步看似琐碎，但它决定了后面所有结论的可信度。plan6 把"镜头致盲"列为头号工程风险是对的，但只给了文字建议；这里给出可执行的判据。

### 10.2 T1 降水与天气分类

#### 10.2.1 结构

```text
图像 (384×384)
   │
   ├─ backbone: timm/mobilenetv4_conv_small (或 efficientvit_m0)
   │      └─ 多尺度特征 → GAP → f_img (D)
   │
   ├─ [可选] 气象编码器: MLP(4) → f_met (32)     ← T/RH/v/P，缺失时置零并加 mask 位
   │
   ├─ 融合: FiLM 调制
   │      γ, β = Linear(f_met) → f' = γ ⊙ f_img + β
   │
   └─ heads
        ├─ 8 类多标签: Linear → sigmoid  → BCE
        └─ 强度 4 类: Linear → 4 logits  → CORAL（有序）
```

#### 10.2.2 为什么用多标签 + 有序

- 多标签：`rain` 与 `fog` 可共存，`snow` 与 `glaze_evidence` 可共存。
- 强度用 CORAL 而非 softmax：`light < moderate < heavy` 有序，误判到相邻级的代价远小于跨两级。

#### 10.2.3 气象物理门控（采纳 plan6 思路，但降级为软约束）

不要用硬规则「T<0 才可能是 glaze」——传感器误差、太阳辐射、设备表面温度与气温差都会打破它。做法：

```python
# 先验：由历史气象记录统计 P(visual_class | T_amb 分箱)
logit_prior = torch.log(prior_table[temp_bin] + eps)      # (8,)
logits = logits + lam * logit_prior                        # lam 默认 0.3，进配置
```

- `lam=0` 时退化为纯视觉；`lam` 越大越依赖先验。
- 训练时随机 dropout `f_met`（p=0.3），保证无气象输入时仍可用。

#### 10.2.4 置信度校准与拒识

1. **温度标定**（在验证集上拟合单一温度 T）：`p_cal = softmax(logits / T)`。
2. **拒识规则**：

```text
if   quality.status == "unusable"      → unknown
elif max(p_cal) < tau_reject (0.45)    → unknown
elif entropy(p_cal) > H_max (1.8)      → unknown
```

3. **报告 ECE**（Expected Calibration Error，10 bin），目标 ≤ 0.10。

> 六份方案只有 plan1 提到 ECE，plan5 提到 MC-Dropout。本方案选**温度标定**而非 MC-Dropout：单卡训练成本低、推理零开销、效果对分类任务足够。

### 10.3 T2 部件 ROI 检测

#### 10.3.1 选型决策（关键）

| 方案 | 许可 | 结论 |
|---|---|---|
| Ultralytics YOLO11 | **AGPL-3.0**，私有部署需商业授权 | ❌ 不进生产链路；**仅用于预标注与对照实验** |
| RTMDet-tiny（MMDetection） | Apache-2.0 | ✅ 可选，但依赖 mmcv 较重 |
| **timm backbone + 自建 FCOS/GFL 头** | Apache-2.0 / MIT | ✅ **默认方案**，依赖最轻，可控性最高 |

**决定**：默认 `timm(mobilenetv4_conv_small) + GFL head`，约 3.5M 参数。Ultralytics 只在 `scripts/prelabel.py` 里作为**预标注工具**出现，其产物进入人工修订流程，不产生生产权重。

> 这条决策的代价是自建检测头要多花约 3–5 天；收益是彻底规避 AGPL 传染风险。对电网企业交付，这个交换是值得的。

#### 10.3.2 小目标处理

导线在 1080P 图中常仅 8–20 px 宽。策略：

1. **切片推理**（SAHI 风格）：640×640 窗口，overlap 0.2，合并时 NMS。
2. **训练时**：随机缩放 0.5–1.5 + Mosaic（注意：Mosaic 会让导线断裂，需检查标注一致性）。
3. **损失**：小目标用 NWD（Normalized Wasserstein Distance）替代 IoU 做分配，对细长目标更稳（plan6 提到，思路正确）。

### 10.4 T3 覆冰分割

#### 10.4.1 结构

```text
ROI 裁切 (512×512)
   │
   ├─ encoder: timm/mobilenetv4 (pretrained) → {C3, C4, C5}
   ├─ decoder: LR-ASPP（轻量）或 UNet-lite（跳连更充分）
   └─ head: 4 类 logits (bg / exposed / ice / icicle)
```

**默认 UNet-lite**：覆冰边界是核心，LR-ASPP 的全局池化会糊边界。UNet-lite 参数量约 2.5M，512² 单帧 < 15 ms（单卡）。

#### 10.4.2 损失

```text
L_seg = λ_dice · L_dice + λ_focal · L_focal + λ_bnd · L_boundary
默认 λ = (1.0, 1.0, 0.5)
```

**FN-aware 处理**（采纳 plan3）：Dice 的分子分母对 FN 与 FP 等权，但漏检代价更高。做法是在 Dice 中给 FN 加权：

```text
Dice_w = (2·Σ w·TP) / (Σ w·TP + Σ w·FN + Σ FP),  w_FN = 2.0
```

**Boundary Loss**：对预测边界与 GT 边界做距离变换后的加权 CE，直接优化边界质量（覆冰厚度的精度取决于边界）。

#### 10.4.3 难例与场景分层（采纳 plan3 的 Ice-Seg 诊断）

| 场景组 | 主要失败模式 | 采样权重 | 针对性增强 |
|---|---|---|---|
| 晴天/良好 | 冰 vs 云/反光/雪地混淆（FP 多） | 1.0 | 强亮度/色温扰动、加云层合成 |
| 恶劣天气 | 漏检（FN 多） | 1.5 | 低对比度、噪声、运动模糊 |
| 设备背景 | 杆塔几何 vs 覆冰混淆 | 1.3 | 随机裁切含角钢/横担的负样本 |

采样器按 `scene_group` 字段分层，权重进配置。

### 10.5 T4 覆冰类型与视觉程度

- **类型**：5 类（none/glaze/rime/mixed/unknown），CrossEntropy + **Class-Balanced 权重**（`w_c = N/(K·N_c)`）。
- **程度**：G0–G4 五级，用 **CORAL**（K−1=4 个二分类器共享权重、单一递增 bias）。

```python
# CORAL
logits = x @ W + b            # (B, K-1)
loss = BCEWithLogits(logits, level_ge_k)   # level_ge_k[i,j] = 1 if y_i > j
# 推理：pred = sum(sigmoid(logits) > 0.5) 沿 K-1 维
```

> 为什么不用回归：G 级是**有序离散**，回归会产生 1.7 这种无意义输出，且对异常值敏感。
> 为什么不用普通 CE：把 G3 判成 G0 与判成 G2 的代价不同，CE 不区分。

**评估**：QWK（Quadratic Weighted Kappa），目标 ≥ 0.70；同时报相邻级容忍准确率。

### 10.6 T5 物理量算（本方案核心增量）

#### 10.6.1 完整链路

```text
mask(ice+exposed+icicle) + edge_prob
   │
   ├─① 掩码后处理：最大连通域 + 形态学闭运算（填冰壳内部孔洞）
   ├─② 中轴骨架：skimage.morphology.skeletonize → 按弧长等距重采样（间距 s=4px）
   ├─③ 每点求切向 τ_k（中心差分）与法向 n_k = (-τ_y, τ_x)
   ├─④ 沿法向在 edge_prob 上做亚像素采样（三次插值），找上下边界交点 u_k, v_k
   ├─⑤ 截面宽度 w_k = |u_k - v_k|
   ├─⑥ 剔除异常：曲率 |κ|>κ_max、靠近 ROI 边界、|w_k| 越界
   ├─⑦ IQR 截尾均值 → W_ice_px
   ├─⑧ 尺度因子 α = D_0 / W_0_px  [mm/px]
   ├─⑨ 径向厚度 d_radial = 0.5·(W_ice_px − W_0_px)·α
   └─⑩ 密度折算 → 等值厚度 d_eq
```

#### 10.6.2 公式

**径向厚度**

```text
d_radial = 0.5 · (W_ice_px − W_0_px) · α
α        = D_0 / W_0_px          [mm/px]
```

其中 `D_0` 为导线外径（mm，来自设备台账），`W_0_px` 为**同一机位、同一变倍**下裸导线的像素宽度。

**等值厚度（折算到 ρ₀ = 900 kg/m³）**

```text
r0     = D_0 / 2
r_out  = r0 + d_radial
r_eq   = sqrt( (ρ_ice/ρ₀)·(r_out² − r0²) + r0² )
d_eq   = r_eq − r0
```

**自检**（必须写成单元测试）：

- `ρ_ice = ρ₀` → `d_eq = d_radial`（恒等）
- `d_radial = 0` → `d_eq = 0`
- 单调性：`d_eq` 随 `d_radial` 与 `ρ_ice` 单调递增
- 量纲：输入 mm，输出 mm，`ρ` 用 kg/m³ 时公式需保证一致（本式 ρ 只以比值出现，量纲无关）

**单位长度覆冰质量（用于与力学传感器互校）**

```text
M = ρ_ice · π · (r_out² − r0²) · 1e-6      [kg/m]   （r 用 mm）
```

#### 10.6.3 亚像素边界定位

普通做法是在二值 mask 上量宽度，受限于像素栅格（±1 px 误差）。改进：

```python
# 沿法线参数 t ∈ [-L, L]，步长 0.25 px
pts = p_k + t[:, None] * n_k                     # (T, 2) 亚像素坐标
vals = map_coordinates(edge_prob, pts.T, order=3) # 三次样条插值
# 在 t<0 与 t>0 两侧分别找局部极大，抛物线拟合求亚像素峰位
t_up   = parabolic_peak(vals[:mid], t[:mid])
t_down = parabolic_peak(vals[mid:], t[mid:])
w_k    = t_down - t_up
```

**为什么不用 Hough/LSD**：覆冰导线存在悬链线弧垂，且冰形不对称（新月形/D 形），全局直线拟合会在局部产生数像素误差。骨架 + 法向逐截面测量对曲率不敏感。

#### 10.6.4 三级动态标定

| 级别 | 方法 | 条件 | 可信度 |
|---|---|---|---|
| **L1** | 检测 `reference_fitting`（耐张线夹/防振锤/钢帽），用其已知物理尺寸反算 α | 视野内有刚性金具且检测置信 ≥ 0.7 | 高 |
| **L2** | PTZ 焦距倍率修正 `α_t = α_0 · (f_0/f_t)`；或有双目视差 `α = f·D_0/Z` | 有 PTZ 编码器读数或双目 | 中高 |
| **L3** | 晴天无冰期建档的 `W_0_px`（同预置位、同变倍） | 历史档案存在 | 中 |

**融合规则**

```python
alphas = [a for a in (a_L1, a_L2, a_L3) if a is not None]
if not alphas:                    status = "not_calibrated"
elif max(alphas)/min(alphas) > 1.20:   # 三源一致性检验
    status = "visual_only"; requires_review = True
else:
    alpha  = alphas[0]            # 优先级 L1 > L2 > L3
    status = "measured" if (a_L1 or a_L2) else "visual_only"
```

> **一致性检验是必须的**：球机变焦后 `W_0_px` 失效，若不检验会得到系统性偏差 2–3 倍的厚度。这是六份方案都没写的细节。

#### 10.6.5 覆盖率与桥接率

```text
coverage_ratio = N_ice / (N_ice + N_exposed)          # 分母是有效设备像素
bridge_ratio   = N_icicle_bridge / N_insulator_gap    # 伞裙间隙被冰凌占据比例
```

`bridge_ratio` 只在 ROI 为 `insulator_string` 时计算，否则为 null。

#### 10.6.6 三源互校（补盲区 ①）

| 源 | 可得性 | 用途 |
|---|---|---|
| 视觉（本系统） | 总是 | 主输出 |
| 力学反演（称重/倾角/应力） | 若现场有 | 交叉校验 |
| 人工实测（游标卡尺/冰尺） | 极少 | **唯一真值**，用于标定与验收 |

**互校规则**

```text
若 |d_visual − d_mech| / max(d_mech, 1mm) > 0.30  →  降级为 visual_only，转人工
若 连续 3 帧同向偏差 > 0.20                        →  判定系统性标定漂移，冻结该机位输出
```

### 10.7 时序滤波与预警决策

#### 10.7.1 为什么必须做时序

单帧在暴风雪中会被飞掠前景雪花、镜头水珠、导线风振污染，产生 5–20 mm 的瞬时跳变。而真实覆冰生长是**小时级慢变过程**。用物理增长率约束即可剔除绝大部分尖峰。

#### 10.7.2 EKF（2 状态）

```text
状态 x = [d, ḋ]ᵀ          d: 等值厚度；ḋ: 增长率
过程模型（Δt = 采样间隔，通常 10–30 min）
    x_{k|k-1} = F x_{k-1},  F = [[1, Δt], [0, 1]]
过程噪声 Q = diag(q_d², q_ḋ²)
观测 z = d_eq_measured,  H = [1, 0],  R = σ_meas²

物理约束（关键）：
    若 |ḋ| > ḋ_max（默认 3 mm/h，进配置）→ 截断并标记为「非物理增长」
    若 创新 |z − Hx̂| > 3√S → 视为离群，本次不更新（仅预测）
```

**去抖**：`alarm_level` 从 I 升到 II 需连续 `k=3` 帧满足条件；降级需连续 `k=5` 帧。避免单帧抖动导致告警闪烁。

#### 10.7.3 预警矩阵（采纳 plan6，参数化）

| 等级 | 标识 | 判据（满足任一） | 建议动作 |
|---|---|---|---|
| I | 蓝 | `0 < d_eq < 5 mm` 或 `γ < 0.3`；`η_bridge < 20%` | 提高抓拍频次 |
| II | 黄 | `5 ≤ d_eq < 10` 或 `0.3 ≤ γ < 0.5`；`20% ≤ η_bridge < 50%` | 推送增长趋势，准备除冰 |
| III | 橙 | `10 ≤ d_eq < 20` 或 `0.5 ≤ γ < 0.8`；`50% ≤ η_bridge < 80%` | 人工复核 + 特巡工单 |
| IV | 红 | `d_eq ≥ 20` 或 `γ ≥ 0.8`；`η_bridge ≥ 80%` | 立即人工确认，走融冰/除冰流程 |

> **约束**：以上阈值必须由业务方按**本地区现行规程**校准后写入 `configs/rules/alarm.yaml`；本文件给出的是初始占位值。系统**不得**在 `measurement_status != measured` 时输出任何等级。

### 10.8 可选进阶（P4 之后，不进 MVP）

| 方向 | 做法 | 前置条件 |
|---|---|---|
| 轻量复原前端 | OmniRestore 2.6M 作为可选前置，仅在 `quality.status=degraded` 时启用 | 需自测电网域增益 |
| 天气嵌入器复用 | 复用 OmniRestore 的 ResNet+KAN 嵌入，接自有分类头 | 同上 |
| 多任务共享骨干 | plan3 的共享 backbone + 双分支，做消融 | 数据量 ≥ 5k 且标注一致 |
| 域自适应 | WA-ADDA（气象条件对抗） | 有 ≥ 1k 张无标注现场图 |
| 条带卷积 | 在分割编码器中加水平/垂直条带卷积（GMSA-Net 思路） | 分割 Dice 达基线后作为增益项 |
| 蒸馏 | 云端大模型 → 边缘小模型 | 有边缘部署需求 |

---

## 11. 数据工程

### 11.1 资源台账（含许可状态）

| 资源 | 内容 | 用途 | 许可 | 状态 |
|---|---|---|---|---|
| **TTPLA** | 航拍杆塔/导线 COCO 多边形 | ROI 预训练、`W_0_px` 来源 | 仓库未见 LICENSE，**待核实** | ✅ 可用（学术） |
| **CPLID** | 绝缘子正常/缺陷 | 绝缘子检测预训练 | 学术用途 | ✅ |
| **InsPLAD** | 无人机电力资产 | ROI 预训练 | **待核实** | ⚠️ |
| **DTL-IceNet**（Zenodo） | 覆冰样例 + 分割代码 | 架构参考、样例验证 | **未标注许可** | ⚠️ 需书面确认 |
| **PLAIR-AW** | Haze/Rain/Snow × 3 数据集 | 退化图像 + L/M/H 标签 | **待核实** | ⚠️ |
| **Kaggle Weather Dataset**（jehanbhathena） | 6862 张 11 类，含 glaze/rime/frost/hail | 天气分类预训练 | **待核实**（Kaggle 页面） | ⚠️ |
| **CDD-11 / Snow100K / LOL** | 通用退化 | 复原模块（P4 才用） | 各自不同 | ⚠️ |
| **firc-dataset** | Gitee 数据交易平台 | **不作主训练集**，仅可选对照 | 不明 | ❌ 不采用 |
| **NWPUYRCC2** | 遥感河冰分割 | 仅纹理先验 | **待核实** | ⚠️ 不混训 |
| **CoInfra** | 路侧冻雨 | 域适应消融 | CC-BY-NC-4.0 | ⚠️ 禁商用 |
| **自采/现场** | 电网监控图像 | 域适应、验收 | 内部授权 | ❌ 暂无 |

> **硬门槛**：任何资源在许可确认前，只能进 `data/external/`（只读、不参与训练），不得进 `data/processed/`。

### 11.2 四级数据流

```text
data/
├── raw/          原始下载/采集，只读，永不修改
├── external/     第三方数据集（许可待确认者放这里，不参与训练）
├── interim/      去重、裁切、格式转换的中间产物
├── processed/    训练就绪（COCO/YOLO/mask + 统一命名）
└── manifests/    版本化清单（JSONL），每次训练必须绑定一个 manifest
```

**manifest 单条记录**

```json
{"image_id": "S03_C07_000123", "path": "processed/images/S03_C07_000123.jpg",
 "sha256": "…", "site_id": "S03", "camera_id": "C07",
 "weather_event_id": "20260115_freeze_03", "timestamp": "2026-01-15T08:30:12+08:00",
 "split": "train", "is_synthetic": false, "scene_group": "adverse",
 "labels": {"weather_visual": ["snow"], "intensity": "moderate",
            "ice_type": "glaze", "visual_grade": "G2"},
 "label_version": "v1.2", "annotator": "A03", "reviewer": "R01"}
```

> 每条带 `sha256` + `split` + `is_synthetic`，是"可复现"的唯一保证。plan1 提了 manifest 概念，这里给出具体字段。

### 11.3 划分规则（防泄漏，硬性）

**分组键**：`group = f"{site_id}|{camera_id}|{weather_event_id}"`

规则：

1. **同一 group 只出现在一个 split**（train/val/test 三者互斥）。
2. **同一 `weather_event_id` 的所有帧**（同一天气过程的连续抓拍）必须在同一 split。
3. **合成数据只进 train**；test 中 `is_synthetic` 必须全为 false。
4. **跨站点测试集**：保留至少 1 个 `site_id` 完全不参与训练（P3 阶段若只有公开数据，用"跨数据集"替代：TTPLA 训练 → CPLID 测试）。
5. **校验脚本强制检查**：`scripts/validate_dataset.py` 在 CI 中运行，违反任一规则直接 fail。

> 这是本方案与 plan3/plan5/plan6 最实质的差异之一。没有这条，所有指标都不可信。

### 11.4 合成数据（3 类算子）

#### 算子 A：大气散射 / 雨雪物理渲染

```text
I_hazy = I·T + A·(1 − T),   T = exp(−β·d(x))
A ∈ [0.4, 0.6]，β ∈ [5e-6, 7e-6]（可参数化连续调节 → 强度 L/M/H 标签）
```

- 深度图 `d(x)`：暗通道先验估计（自研，plan6 称 PLAIR-AW 提供脚本，实测仓库未含）。
- 雨纹层：不同粒径/倾角/运动模糊核叠加；雪花层：分形噪声 + 尺度分布。
- **产出**：`(I_hazy, I_clean, intensity_label)` 三元组。

#### 算子 B：参数化覆冰截面合成（核心）

```text
1. 取 TTPLA/自采的裸导线 mask → 骨架 + 法向
2. 沿法向按截面模板外扩指定像素厚度：
     圆筒形   r(θ) = r0 + t
     C2 薄新月 r(θ) = r0 + t·(0.5 + 0.5·cos θ)
     C7 厚新月 r(θ) = r0 + t·(0.7 + 0.3·cos θ)
     C11 D 形  r(θ) = r0 + t·(0.8 + 0.2·cos θ) 且 θ∈[-60°,60°] 内加平顶
3. 纹理贴合：从真实 glaze/rime 样本裁块，Poisson 融合到膨胀区域
4. 绝缘子：按指定 bridge_ratio 在伞裙下沿生成锥形冰凌
```

**产出优势**：`t` 是**零误差真值**，可直接训练与评估厚度回归。这是合成数据在本项目中最有价值的用途——不是"扩充样本量"，而是"提供精确几何真值"。

#### 算子 C：Makkonen 虚拟气象配对

```text
dM/dt = η1·η2·η3 · LWC · A · v
```

对无气象记录的历史覆冰图，反演采样出与冰厚相容的 `(T, RH, v, LWC)` 组合，用于训练气象门控分支。

> 注意：这是**弱标签**，只能用于辅助训练，不得作为验证依据。

### 11.5 增强分组

| 组 | 内容 | 适用 | 约束 |
|---|---|---|---|
| A 常规 | 亮度/对比度/色偏/缩放/平移/JPEG 噪声 | 全部 | 不得改变标签语义 |
| B 天气鲁棒 | 雾/雨/雪合成、运动模糊、局部遮挡 | 覆冰分割、天气分类 | 合成后必须人工抽检 50 张 |
| C 覆冰外观 | 截面膨胀、纹理贴合、冰凌生成 | 仅覆冰分割 | **仅 train**，且 `is_synthetic=true` |

**增强红线**：任何改变"是否存在覆冰"语义的增强（如把晴天图强加冰层却保留 `ice_type=none`）必须同步更新标签，否则禁止使用。

### 11.6 数据规模预算（单卡条件）

| 用途 | 规模 | 来源 | 阶段 |
|---|---|---|---|
| 天气分类预训练 | 6,862（11 类→映射 8 类） | Kaggle Weather | P1 |
| 天气分类域适应 | 3,000–5,000 合成退化 | 算子 A | P1 |
| ROI 检测预训练 | 2,000–3,000（TTPLA + CPLID） | 公开 | P2 |
| 覆冰分割（合成） | 5,000–8,000 | 算子 B | P3 |
| 覆冰分割（真实样例） | 200–500 | DTL-IceNet 样例 | P3 |
| 覆冰类型 | 300–500（精标） | 合成 + 样例 | P4 |
| 厚度验证 | 2,000（合成，零误差真值） | 算子 B | P4 |

---

## 12. 代码目录结构

```text
grid-ice-vision/
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
├── Makefile
├── justfile
├── AGENTS.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # ruff + mypy + pytest(unit)
│       └── regression.yml          # 固定小测试集跑指标，防退化
│
├── configs/
│   ├── data/
│   │   ├── base.yaml
│   │   ├── weather.yaml
│   │   └── icing.yaml
│   ├── model/
│   │   ├── quality.yaml
│   │   ├── precip_mnv4.yaml
│   │   ├── detector_gfl.yaml
│   │   └── seg_unetlite.yaml
│   ├── train/
│   │   ├── precip.yaml
│   │   ├── detector.yaml
│   │   └── seg.yaml
│   ├── labels/
│   │   ├── taxonomy.yaml           # 类别定义与 ID 映射
│   │   └── severity.yaml           # G0–G4 阈值
│   ├── rules/
│   │   └── alarm.yaml              # I–IV 阈值、去抖帧数、EKF 参数
│   └── deploy/
│       ├── onnx_cpu.yaml
│       └── trt_fp16.yaml
│
├── data/                           # 仅 .gitkeep + README
│   ├── raw/  external/  interim/  processed/  manifests/
│
├── docs/
│   ├── architecture.md
│   ├── annotation_guide.md         # §8 的展开版
│   ├── severity_rubric.md
│   ├── data_license_ledger.md      # §11.1 的展开版（台账）
│   ├── model_cards/
│   └── decisions/                  # ADR：为什么不用 Ultralytics 等
│
├── notebooks/
│   ├── 00_eda_public.ipynb
│   ├── 01_synthetic_ice_check.ipynb
│   └── 02_error_analysis.ipynb
│
├── scripts/
│   ├── download_public.py
│   ├── convert_ttpla.py
│   ├── build_manifests.py
│   ├── validate_dataset.py
│   ├── synthesize_ice.py
│   ├── synthesize_weather.py
│   ├── prelabel.py                 # 唯一允许调用 ultralytics 的地方
│   ├── export_onnx.py
│   └── benchmark.py
│
├── src/
│   └── grid_ice_vision/
│       ├── __init__.py
│       ├── py.typed
│       ├── cli.py
│       │
│       ├── common/
│       │   ├── schemas.py          # Pydantic 契约（§7.2）
│       │   ├── config.py           # YAML + Pydantic 校验
│       │   ├── logging.py
│       │   └── constants.py
│       │
│       ├── data/
│       │   ├── manifest.py         # 读写 + 校验
│       │   ├── splits.py           # 分组划分
│       │   ├── datasets/
│       │   │   ├── weather.py
│       │   │   ├── detection.py
│       │   │   └── segmentation.py
│       │   ├── transforms/
│       │   │   ├── common.py
│       │   │   ├── weather_aug.py
│       │   │   └── ice_aug.py
│       │   └── samplers/
│       │       └── scene_stratified.py
│       │
│       ├── features/
│       │   ├── quality/
│       │   │   └── gate.py
│       │   ├── precipitation/
│       │   │   ├── model.py        # backbone + FiLM + 多头
│       │   │   ├── prior.py        # 温度先验表
│       │   │   ├── calibrate.py    # 温度标定
│       │   │   └── infer.py
│       │   ├── detection/
│       │   │   ├── model.py        # timm + GFL head
│       │   │   ├── loss.py
│       │   │   └── infer.py        # 含切片推理
│       │   ├── ice/
│       │   │   ├── model.py        # UNet-lite
│       │   │   ├── loss.py
│       │   │   └── infer.py
│       │   └── severity/
│       │       ├── geometry.py     # 骨架/法向/亚像素
│       │       ├── thickness.py    # α、d_radial、d_eq（纯函数）
│       │       ├── calibration.py  # L1/L2/L3 融合
│       │       └── grading.py      # 覆盖率 → G 级
│       │
│       ├── training/
│       │   ├── engine.py
│       │   ├── losses.py           # Dice/Focal/Boundary/CORAL/CB-CE
│       │   ├── callbacks.py
│       │   └── checkpoint.py
│       │
│       ├── evaluation/
│       │   ├── classification.py   # Macro-F1 / ECE
│       │   ├── detection.py
│       │   ├── segmentation.py     # mIoU / Dice / Boundary F1
│       │   └── thickness.py        # MAPE / RMSE / R²
│       │
│       ├── application/
│       │   ├── pipeline.py
│       │   ├── temporal.py         # EKF
│       │   ├── decision.py         # 预警规则引擎
│       │   └── evidence.py         # 卡尺叠加图生成
│       │
│       ├── deployment/
│       │   ├── export.py
│       │   ├── runtime.py          # ONNXRuntime 封装
│       │   └── benchmark.py
│       │
│       └── service/
│           ├── api.py
│           └── errors.py
│
└── tests/
    ├── unit/
    │   ├── test_thickness.py       # 几何真值 + 单调性 + 恒等
    │   ├── test_calibration.py
    │   ├── test_losses.py
    │   ├── test_splits.py          # 泄漏校验
    │   └── test_schemas.py
    ├── integration/
    │   ├── test_pipeline_contract.py
    │   └── test_export_onnx.py     # 数值一致性 atol ≤ 1e-4
    ├── e2e/
    │   └── test_regression_baseline.py
    └── fixtures/                   # 小型合成张量，可入库
```

**命名与禁令**（写入 `AGENTS.md`）

- 包名 snake_case；仓库名 kebab-case；目录嵌套 ≤ 4 层。
- 根目录禁止出现 `train.py` / `test.py` / `utils.py` 等裸脚本。
- 禁止 `temp/` `runs/` `exp2_final/` `old_code/` 等噪音目录；实验产物统一写 `artifacts/`（gitignore）。
- 禁止同时存在 `setup.py` / `requirements.txt` / `poetry.lock`（单一事实来源：`pyproject.toml`）。

---

## 13. 关键代码骨架

### 13.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "grid-ice-vision"
version = "0.1.0"
description = "Grid icing and precipitation recognition from images"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.3",
  "torchvision>=0.18",
  "timm>=1.0",
  "numpy>=1.26",
  "opencv-python-headless>=4.10",
  "scikit-image>=0.24",
  "scipy>=1.13",
  "albumentations>=1.4",
  "pydantic>=2.8",
  "pydantic-settings>=2.4",
  "pyyaml>=6.0",
  "typer>=0.12",
  "rich>=13.7",
]

[project.optional-dependencies]
serve = ["fastapi>=0.115", "uvicorn[standard]>=0.31", "python-multipart"]
deploy = ["onnx>=1.16", "onnxruntime>=1.19"]
dev = ["pytest>=8.2", "pytest-cov>=5.0", "ruff>=0.6", "mypy>=1.11", "pre-commit>=3.8"]

[project.scripts]
giv = "grid_ice_vision.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/grid_ice_vision"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = "src"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

### 13.2 `common/schemas.py`

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    USABLE = "usable"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"


class MeasurementStatus(str, Enum):
    MEASURED = "measured"
    VISUAL_ONLY = "visual_only"
    NOT_CALIBRATED = "not_calibrated"


class QualityInfo(BaseModel):
    status: QualityStatus
    blur_score: float = Field(ge=0.0, le=1.0)
    exposure: Literal["under", "normal", "over"]
    occlusion_ratio: float = Field(ge=0.0, le=1.0)
    lens_icing_suspected: bool = False
    reasons: list[str] = Field(default_factory=list)


class PrecipResult(BaseModel):
    """视觉相态：多标签，不代表气象确认结论。"""
    labels: dict[str, float]
    intensity: Literal["none", "light", "moderate", "heavy", "unknown"]
    intensity_conf: float
    meteorological_input_used: bool = False
    note: str = "visual_only"


class Asset(BaseModel):
    cls: Literal["conductor", "insulator_string", "tower", "reference_fitting"]
    bbox: tuple[float, float, float, float]
    score: float
    roi_id: str
    reference_width_px: float | None = None


class IcingResult(BaseModel):
    present: bool | None
    confidence: float
    coverage_ratio: float | None = None
    ice_type: Literal["none", "glaze", "rime", "mixed", "unknown"] = "unknown"
    ice_type_conf: float = 0.0
    bridge_ratio: float | None = None
    mask_rle: dict | None = None


class SeverityResult(BaseModel):
    visual_grade: Literal["G0", "G1", "G2", "G3", "G4"]
    alarm_level: Literal["I", "II", "III", "IV"] | None = None
    d_radial_mm: float | None = None
    d_eq_mm: float | None = None
    gamma: float | None = None
    measurement_status: MeasurementStatus
    calibration_source: Literal["rigid_fitting", "ptz", "baseline_archive"] | None = None
    alpha_mm_per_px: float | None = None
    uncertainty_mm: float | None = None
    requires_review: bool = False

    def model_post_init(self, __context) -> None:
        # 硬约束：未标定不得输出厚度与预警（用 raise 而非 assert，避免 -O 下被剥离）
        if self.measurement_status is MeasurementStatus.NOT_CALIBRATED:
            if self.d_eq_mm is not None:
                raise ValueError("not_calibrated 时不得输出厚度")
            if self.alarm_level is not None:
                raise ValueError("not_calibrated 时不得输出预警等级")


class Provenance(BaseModel):
    model_version: dict[str, str]
    config_hash: str
    inference_ms: float
    rule_version: str


class InferenceResult(BaseModel):
    schema_version: str = "1.0.0"
    image_id: str
    quality: QualityInfo
    precipitation: PrecipResult
    assets: list[Asset] = Field(default_factory=list)
    icing: IcingResult
    severity: SeverityResult
    evidence: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
```

### 13.3 `features/severity/thickness.py`（核心物理模块）

```python
"""覆冰厚度几何解算。纯函数，不依赖任何模型，可独立单测。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

RHO_STANDARD = 900.0  # kg/m^3，电力行业标准覆冰密度


@dataclass(frozen=True)
class ScaleReference:
    """尺度参考：某个已知物理尺寸的物体在图像中的像素宽度。"""
    d_ref_mm: float
    w_ref_px: float
    source: Literal["rigid_fitting", "ptz", "baseline_archive"]

    @property
    def alpha(self) -> float:
        if self.w_ref_px <= 0:
            raise ValueError("w_ref_px 必须为正")
        return self.d_ref_mm / self.w_ref_px


def fuse_scale_references(
    refs: list[ScaleReference],
    max_ratio: float = 1.20,
) -> tuple[float | None, str | None]:
    """多源尺度融合。

    返回 (alpha_mm_per_px, source)。三源差异过大时返回 (None, None)，
    由上层置为 visual_only 并转人工，避免系统性偏差。
    """
    if not refs:
        return None, None

    priority = {"rigid_fitting": 0, "ptz": 1, "baseline_archive": 2}
    refs = sorted(refs, key=lambda r: priority[r.source])
    alphas = [r.alpha for r in refs]

    if len(alphas) > 1 and max(alphas) / min(alphas) > max_ratio:
        return None, None

    return refs[0].alpha, refs[0].source


def radial_thickness_mm(
    w_ice_px: float,
    w0_px: float,
    alpha_mm_per_px: float,
) -> float:
    """单侧径向覆冰厚度（mm）。"""
    if w_ice_px <= w0_px:
        return 0.0
    return 0.5 * (w_ice_px - w0_px) * alpha_mm_per_px


def equivalent_thickness_mm(
    d_radial_mm: float,
    d0_mm: float,
    rho_ice: float = RHO_STANDARD,
    rho_std: float = RHO_STANDARD,
) -> float:
    """折算到标准密度 rho_std 的等值覆冰厚度。

    基于单位长度覆冰质量守恒：
        rho_ice * (r_out^2 - r0^2) == rho_std * (r_eq^2 - r0^2)
    解得 r_eq，再减去 r0。
    """
    if d_radial_mm <= 0:
        return 0.0
    if d0_mm <= 0:
        raise ValueError("d0_mm 必须为正")

    r0 = d0_mm / 2.0
    r_out = r0 + d_radial_mm
    r_eq_sq = (rho_ice / rho_std) * (r_out**2 - r0**2) + r0**2
    return math.sqrt(r_eq_sq) - r0


def ice_mass_per_meter(
    d_radial_mm: float,
    d0_mm: float,
    rho_ice: float = RHO_STANDARD,
) -> float:
    """单位长度覆冰质量（kg/m），用于与力学传感器互校。"""
    r0 = d0_mm / 2.0
    r_out = r0 + d_radial_mm
    area_m2 = math.pi * (r_out**2 - r0**2) * 1e-6  # mm^2 -> m^2
    return rho_ice * area_m2


def trimmed_mean_width(widths_px: np.ndarray, k: float = 1.5) -> float:
    """IQR 截尾均值：剔除局部遮挡/孤立冰柱造成的离群截面。"""
    widths = widths_px[np.isfinite(widths_px)]
    if widths.size == 0:
        raise ValueError("无有效截面宽度")
    q1, q3 = np.percentile(widths, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    kept = widths[(widths >= lo) & (widths <= hi)]
    return float(kept.mean() if kept.size else widths.mean())
```

**对应单元测试（必须写）**

```python
def test_equivalent_thickness_identity():
    # 密度等于标准密度时，等值厚度应等于径向厚度
    assert equivalent_thickness_mm(6.0, 30.0, 900.0, 900.0) == pytest.approx(6.0)


def test_equivalent_thickness_zero():
    assert equivalent_thickness_mm(0.0, 30.0) == 0.0


def test_equivalent_thickness_monotonic():
    vals = [equivalent_thickness_mm(d, 30.0, 600.0) for d in (1, 2, 4, 8)]
    assert vals == sorted(vals)


def test_low_density_gives_smaller_equivalent():
    # 雾凇密度低，折算到 900 后应更薄
    assert equivalent_thickness_mm(10.0, 30.0, 400.0) < equivalent_thickness_mm(10.0, 30.0, 900.0)


def test_mass_roundtrip():
    # 质量守恒：用任意密度算出的质量，应等于用等值厚度 + 标准密度算出的质量
    d, d0, rho = 10.0, 30.0, 600.0
    m = ice_mass_per_meter(d, d0, rho)
    d_eq = equivalent_thickness_mm(d, d0, rho)
    assert ice_mass_per_meter(d_eq, d0, RHO_STANDARD) == pytest.approx(m, rel=1e-9)
```

### 13.4 `training/losses.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """FN-aware Dice：漏检权重高于误检。"""

    def __init__(self, fn_weight: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.fn_weight = fn_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        tp = (prob * target).sum()
        fn = ((1.0 - prob) * target).sum() * self.fn_weight
        fp = (prob * (1.0 - target)).sum()
        return 1.0 - (2.0 * tp + self.eps) / (2.0 * tp + fn + fp + self.eps)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class CoralLoss(nn.Module):
    """有序回归：K-1 个阈值，共享权重、递增 bias。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: (B,) 取值 0..K-1
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        binary = (target.unsqueeze(1) > levels.unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, binary)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class ClassBalancedCE(nn.Module):
    """按有效样本数重加权，缓解长尾。"""

    def __init__(self, counts: list[int], beta: float = 0.9999) -> None:
        super().__init__()
        eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float), torch.tensor(counts, dtype=torch.float))
        w = (1.0 / eff)
        self.register_buffer("weight", w / w.sum() * len(counts))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight)
```

### 13.5 `data/manifest.py` + 划分校验

```python
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ManifestRecord(BaseModel):
    image_id: str
    path: str
    sha256: str
    site_id: str
    camera_id: str
    weather_event_id: str
    timestamp: str
    split: str
    is_synthetic: bool
    scene_group: str
    labels: dict
    label_version: str

    @property
    def group_key(self) -> str:
        return f"{self.site_id}|{self.camera_id}|{self.weather_event_id}"


def load_manifest(path: Path) -> list[ManifestRecord]:
    return [ManifestRecord(**json.loads(line)) for line in path.read_text("utf-8").splitlines() if line]


def assert_no_leakage(records: list[ManifestRecord]) -> None:
    """硬校验：同一 group 不得跨 split；测试集不得含合成数据。"""
    group_to_split: dict[str, str] = {}
    for r in records:
        prev = group_to_split.setdefault(r.group_key, r.split)
        if prev != r.split:
            raise AssertionError(f"数据泄漏: group {r.group_key} 同时出现在 {prev} 与 {r.split}")

    for r in records:
        if r.split == "test" and r.is_synthetic:
            raise AssertionError(f"测试集含合成数据: {r.image_id}")
```

### 13.6 `configs/rules/alarm.yaml`

```yaml
# 预警规则：所有阈值必须由业务方按本地区现行规程校准
version: "1.0.0"
require_measured: true          # 未标定时禁止输出等级

levels:
  I:
    d_eq_mm: {min: 0.001, max: 5.0}
    gamma:   {min: 0.0,   max: 0.3}
    bridge:  {min: 0.0,   max: 0.2}
  II:
    d_eq_mm: {min: 5.0, max: 10.0}
    gamma:   {min: 0.3, max: 0.5}
    bridge:  {min: 0.2, max: 0.5}
  III:
    d_eq_mm: {min: 10.0, max: 20.0}
    gamma:   {min: 0.5,  max: 0.8}
    bridge:  {min: 0.5,  max: 0.8}
  IV:
    d_eq_mm: {min: 20.0, max: null}
    gamma:   {min: 0.8,  max: null}
    bridge:  {min: 0.8,  max: null}

debounce:
  escalate_frames: 3            # 升级需连续满足帧数
  deescalate_frames: 5

ekf:
  dt_minutes: 15
  growth_rate_max_mm_per_h: 3.0 # 超出视为非物理
  innovation_gate_sigma: 3.0
  process_noise: {d: 0.5, rate: 0.2}
  measurement_noise_mm: 1.5
```

### 13.7 `cli.py`

```python
import typer

app = typer.Typer(no_args_is_help=True, help="Grid ice vision CLI")


@app.command()
def train(task: str, config: str) -> None:
    """train quality|precip|detector|seg --config configs/train/xxx.yaml"""


@app.command()
def evaluate(config: str, split: str = "test") -> None:
    """在指定 split 上评估，输出指标 JSON 与混淆矩阵"""


@app.command()
def infer(image: str, config: str, out: str = "out.json") -> None:
    """单图推理，输出契约 JSON + 证据图"""


@app.command()
def export(config: str) -> None:
    """导出 ONNX 并做数值一致性校验"""
```

---

## 14. 分阶段实施计划

**总原则**：单卡单人，按**人周**估算，阶段间允许部分并行（标注与训练、合成与训练）。

### 阶段总览

| 阶段 | 周期 | 核心目标 | 关键产出 | 是否依赖现场数据 |
|---|---|---|---|---|
| **P0** | 1 周 | 工程底座 + 资源落地 | 可运行骨架、资源台账、标注规范 | 否 |
| **P1** | 3 周 | 天气分支基线 | 8 类分类器 + 校准 + 评估报告 | 否 |
| **P2** | 3 周 | ROI 检测 + 合成管线 | 检测器 + 合成算子 B | 否 |
| **P3** | 3 周 | 覆冰分割 + 物理量算 | 分割器 + `thickness.py` + 卡尺证据图 | 否 |
| **P4** | 3 周 | 类型/程度 + 时序 + 服务 | 全链路 API + EKF + 预警矩阵 | 否（合成验证） |
| **P5** | 持续 | 现场适配与验收 | 跨站点测试报告、模型卡 | **是** |

---

### P0 · 工程底座（1 周）

| 任务 | 产出 | DoD |
|---|---|---|
| P0.1 uv 初始化 + src layout + 配置 | 可安装包 | `uv sync && uv run pytest` 通过 |
| P0.2 CI（ruff/mypy/pytest） | `ci.yml` | 首次 PR 全绿 |
| P0.3 资源下载与许可台账 | `docs/data_license_ledger.md` | 每个资源有「许可/可用性/联系人」三列 |
| P0.4 标注规范 v1 | `docs/annotation_guide.md` | 含 §8.7 的 4 条边界规则 |
| P0.5 manifest 与校验脚本 | `validate_dataset.py` | 故意造一个泄漏用例能被拦住 |
| P0.6 AGENTS.md | 目录契约 | 含「禁止事项」清单 |

**退出条件**：`make check` 全绿；台账中至少 3 个资源许可状态明确（非"待核实"）。

---

### P1 · 天气分支基线（3 周）

| 任务 | 产出 | DoD |
|---|---|---|
| P1.1 11 类 → 8 类映射表 | `configs/labels/taxonomy.yaml` | 映射有文档说明，dew/rainbow/sandstorm/lightning 的处理明确 |
| P1.2 数据清洗（去重、损坏、极小图） | 清洗报告 | 去重后样本数、类别分布表 |
| P1.3 训练 MobileNetV4 + 多标签 BCE | `precip-0.1.0` | 验证集 Macro-F1 ≥ 0.65 |
| P1.4 强度 CORAL 头 | 同上 | QWK ≥ 0.55 |
| P1.5 温度标定 + 拒识 | `calibrate.py` | ECE ≤ 0.12，拒识率 ≤ 15% |
| P1.6 合成退化域适应（算子 A） | 增强模型 | 在 PLAIR-AW 退化子集上 Macro-F1 提升 ≥ 3 个点 |

**关键难点**：8 类中 `hail`/`glaze`/`rime` 样本极少（可能各 <200）。对策：先合并为 `glaze_rime` 训练，观察混淆矩阵后再拆；`hail` 直接标为"不追求精度"。

---

### P2 · ROI 检测 + 合成管线（3 周）

| 任务 | 产出 | DoD |
|---|---|---|
| P2.1 TTPLA 转换 + 清洗 | `processed/ttpla_coco/` | 类别统计、空标注过滤 |
| P2.2 timm + GFL 检测头实现 | `detector_gfl.yaml` | 训练可跑通 10 epoch 无 NaN |
| P2.3 切片推理（SAHI 风格） | `infer.py` | 1080P 单图 p95 ≤ 120 ms |
| P2.4 合成算子 A（雾/雨/雪） | `synthesize_weather.py` | 参数可调，输出 L/M/H 标签 |
| P2.5 合成算子 B（C2/C7/C11） | `synthesize_ice.py` | 生成图 + 零误差厚度真值 JSON |

**关键难点**：TTPLA 是航拍俯视，与杆塔侧视差异大。对策：**只用于预训练**，在 P3 的合成侧视图上微调；不把 TTPLA 的 mAP 当作本项目指标。

---

### P3 · 覆冰分割 + 物理量算（3 周）

| 任务 | 产出 | DoD |
|---|---|---|
| P3.1 UNet-lite 分割器 | `seg-0.1.0` | 合成测试集 Dice ≥ 0.75 |
| P3.2 FN-aware Dice + Focal + Boundary | `losses.py` | 消融：加 Boundary 后 Boundary F1 提升 ≥ 5% |
| P3.3 场景分层采样器 | `scene_stratified.py` | 三组样本比例可配，日志可见 |
| P3.4 骨架 + 法向 + 亚像素宽度 | `geometry.py` | 合成圆筒真值上宽度误差 ≤ 0.5 px |
| P3.5 `thickness.py` + 单测 | 纯函数模块 | 4 条单测全过（恒等/零/单调/密度） |
| P3.6 三级标定 + 一致性检验 | `calibration.py` | 人为制造 1.5× 偏差能被拦下 |
| P3.7 卡尺叠加证据图 | `evidence.py` | 输出含上下包络线 + 采样截面的 PNG |

**这是全项目最关键的一阶段**。P3.5 的物理模块必须在**没有任何真实数据**的情况下就能验证正确性——用合成圆筒 + 已知厚度即可。

---

### P4 · 类型/程度 + 时序 + 服务（3 周）

| 任务 | 产出 | DoD |
|---|---|---|
| P4.1 类型 5 类分类头 | `ice_type` 输出 | 合成集 Macro-F1 ≥ 0.70 |
| P4.2 视觉等级 CORAL | `grading.py` | QWK ≥ 0.70（合成集） |
| P4.3 EKF 时序滤波 | `temporal.py` | 注入 ±10 mm 尖峰后输出抖动 < 2 mm |
| P4.4 预警规则引擎 | `decision.py` | 未标定时强制返回 `alarm_level=null` |
| P4.5 FastAPI 服务 | `/v1/infer` `/v1/health` `/v1/model/info` | 契约测试通过 |
| P4.6 ONNX 导出 + 一致性 | `export.py` | 与 PyTorch 输出 atol ≤ 1e-4 |

---

### P5 · 现场适配与验收（持续，依赖数据）

| 任务 | 前置 | DoD |
|---|---|---|
| P5.1 现场数据接入与标注 | 数据到位 | ≥ 300 张精标，κ ≥ 0.75 |
| P5.2 域适应微调 | P5.1 | 跨站点测试集 Recall ≥ 0.85 |
| P5.3 实测冰厚验证 | 有实测 | MAPE ≤ 15%（样本 ≥ 50） |
| P5.4 边缘部署（可选） | 有硬件 | ONNX Runtime / TRT 实测 p95 |
| P5.5 模型卡与回滚方案 | — | 每个模型有卡片 + 回滚脚本 |

**若现场数据长期不到位**：交付以 P0–P4 的**方法闭环 + 合成验证 + 可复现实验**为准，并在交付说明中明确"生产精度未验证"。这是 §6.3 第 7 条的落地方式。

### 依赖关系

```text
P0 ──┬── P1 (天气)  ─────────────────┐
     ├── P2 (检测+合成) ── P3 (分割+量算) ── P4 (融合+服务) ── P5
     └── 标注规范 ────────────┘
```

P1 与 P2 可并行（不同人/不同时段）。P3 强依赖 P2 的合成管线。

---

## 15. 测试与评估

### 15.1 测试金字塔

| 层 | 内容 | 运行时机 | 时长 |
|---|---|---|---|
| 单元 | 厚度公式、标定融合、损失数值、划分校验、schema 约束 | 每次提交 | < 10 s |
| 集成 | 模块间契约贯通、缺失气象输入的降级、ONNX 数值一致 | 每次 PR | < 2 min |
| 回归 | 固定 200 张黄金集跑全链路指标，与基线比对 | 每日 / 模型更新 | < 10 min |

**关键单测清单**

1. `equivalent_thickness` 恒等/零/单调/密度四条（§13.3）
2. `fuse_scale_references` 在 1.5× 偏差时返回 None
3. `assert_no_leakage` 能拦住跨 split 的 group
4. `SeverityResult` 在 `not_calibrated` 时输出厚度会抛断言
5. CORAL 预测在 logits 全负时返回 0、全正时返回 K-1（单调性）
6. ONNX 与 PyTorch 在随机输入上 atol ≤ 1e-4

### 15.2 指标定义

| 指标 | 定义 | 用于 |
|---|---|---|
| Macro-F1 | 各类 F1 的算术平均 | 天气分类、类型分类 |
| Balanced Acc | 各类召回的平均 | 同上 |
| ECE | Σ(n_b/N)·\|acc_b − conf_b\|，10 bin | 校准质量 |
| mAP@0.5 | IoU≥0.5 的 AP 均值 | ROI 检测 |
| Dice | 2TP/(2TP+FP+FN) | 覆冰分割 |
| Boundary F1 | 边界 2px 容差内的 F1 | 分割边界 |
| QWK | 二次加权 Kappa | 等级判断 |
| MAPE | mean(\|pred−gt\|/gt) | 厚度（仅 gt>1mm 样本） |
| RMSE | 均方根误差 | 厚度 |
| p95 延迟 | 95 分位推理耗时 | 工程性能 |

### 15.3 三套测试集

| 集合 | 构成 | 用途 | 何时可用 |
|---|---|---|---|
| **Set-A 同分布** | 公开数据留出（按 group 划分） | 与文献基线对比、算法选型 | P1 起 |
| **Set-B 跨域** | TTPLA 训练 → CPLID 测试；或留出一个 site | 泛化能力 | P2 起 |
| **Set-C 困难** | 合成退化（重度雾/雪/低光）+ 遮挡 + 逆光 | 鲁棒性 | P1 起 |
| **Set-D 现场** | 真实电网图像（人工精标） | **唯一部署决策依据** | P5（待数据） |

> 没有 Set-D 之前，**不得**宣称"可用于生产"。

### 15.4 验收指标（三栏制）

> **硬门槛** = 不达标则阶段不算完成；**软目标** = 努力方向；**文献参考** = 论文报告值，仅作对照，**不是**本项目承诺。

| 项目 | 硬门槛 | 软目标 | 文献参考 |
|---|---|---|---|
| 天气 8 类 Macro-F1（Set-A） | ≥ 0.70 | ≥ 0.82 | 0.958（ViT-base，11 类，同分布） |
| 天气 8 类 Macro-F1（Set-C 退化） | ≥ 0.55 | ≥ 0.70 | — |
| 校准 ECE | ≤ 0.15 | ≤ 0.08 | — |
| ROI mAP@0.5（Set-B） | ≥ 0.60 | ≥ 0.80 | 0.935（论文自建集，不可比） |
| 覆冰二分类 Recall（Set-C） | ≥ 0.88 | ≥ 0.94 | — |
| 覆冰分割 Dice（合成测试集） | ≥ 0.75 | ≥ 0.85 | 0.96 mIoU（GMSA-Net 论文自建集） |
| 覆冰分割 Dice（Set-D） | ≥ 0.65 | ≥ 0.78 | — |
| 等级 QWK（合成） | ≥ 0.70 | ≥ 0.82 | — |
| 厚度 MAPE（合成，gt ≥ 3mm） | ≤ 15% | ≤ 8% | 11.82%（DTL-IceNet）、≤5%（GMSA-Net） |
| 像素宽度误差（合成圆筒） | ≤ 0.8 px | ≤ 0.5 px | 0.47 px（亚像素垂线法论文） |
| 全链路 p95（单卡，1080P） | ≤ 200 ms | ≤ 120 ms | — |
| 连续推理稳定性 | 1000 张无异常退出 | 10000 张 | — |

**为什么把论文指标压低**：论文指标来自同分布、高质量标注、单一场景的数据集。本项目在**跨域 + 合成 + 真实混合**条件下，Dice 0.75 已属良好。设 0.96 只会导致两个后果：要么造假，要么改测试集。

### 15.5 回归基线管理

- 每次正式训练产出 `artifacts/<task>/<version>/`：权重 + 配置快照 + manifest 哈希 + 指标 JSON。
- `regression.yml` 每日跑固定 200 张黄金集，任一指标相对基线下降 > 2 个点则 fail。
- 基线本身也版本化，不允许"悄悄更新基线"。

---

## 16. 部署与运维

### 16.1 导出链路

```text
PyTorch ckpt
   → 固定输入尺寸与预处理（写入 config，不写在代码里）
   → ONNX（opset 17）
   → 数值一致性校验（atol ≤ 1e-4，100 组随机输入）
   → [可选] TensorRT FP16 / OpenVINO
   → 端到端 benchmark（含前后处理）
```

**不承诺 INT8**：量化对覆冰边缘 IoU 的损伤需实测；若要做，分割头保持 FP16 混合精度，并先做量化感知训练。

### 16.2 边云分工

| 侧 | 职责 | 模型 |
|---|---|---|
| 边缘（若有） | 质量门控、ROI 检测、粗分割 | 最小模型，输入 512² |
| 云端 | 精细分割、类型/等级、时序、融合 | 完整模型，输入 1024² |

无边缘硬件时，全部走云端，只保证 p95 ≤ 200 ms。

### 16.3 API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/infer` | POST | 单图推理，返回契约 JSON |
| `/v1/infer/batch` | POST | 批量（异步队列） |
| `/v1/health` | GET | 存活 + 模型加载状态 |
| `/v1/model/info` | GET | 各子模型版本 + 配置哈希 |
| `/v1/metrics` | GET | 请求量、错误率、p50/p95 延迟 |

**错误码**：`400` 参数/解码错误，`422` 契约校验失败，`503` 模型未就绪，`500` 内部错误（不得泄露堆栈）。

### 16.4 监控与漂移

| 监控项 | 阈值 | 动作 |
|---|---|---|
| 输入亮度/对比度分布 | 偏离训练分布 > 2σ | 告警 + 抽样人工检查 |
| `quality.status=unusable` 比例 | > 10%（单站点 24h） | 检查摄像头 |
| 拒识率 | > 25% | 检查数据漂移 |
| 类别预测分布 | 某类占比突增 3× | 抽样复核 |
| 跨源不一致率 | > 5% | 冻结该机位输出 |

### 16.5 模型卡与回滚

每个模型版本一份 `docs/model_cards/<task>-<version>.md`：训练数据（manifest 哈希）、指标、适用范围、已知失效场景、回滚命令。

回滚：服务支持 `MODEL_VERSION` 环境变量切换，无需重新构建镜像。

### 16.6 可解释性输出（三层）

1. **测量卡尺叠加图**（最有说服力）：在原图上绘制覆冰上下包络、裸导线基准线、每个采样截面的位置与读数。
2. **掩码叠加**：4 类掩码半透明叠加。
3. **概率热图**：分割 sigmoid 输出伪彩。

> 不做 LRP：实现复杂、对自定义网络支持差、运维人员看不懂。卡尺图才是能让老师傅点头的东西。

---

## 17. 风险登记册

| # | 风险 | 触发信号 | 影响 | 缓解 | 责任阶段 |
|---|---|---|---|---|---|
| R1 | 公开覆冰数据许可不明 | 台账出现"待核实"超 2 周 | 无法合法训练/交付 | 只用样例做验证；主训练靠合成；同步申请授权 | P0 |
| R2 | 合成数据与真实域差距过大 | 合成训练 → 真实样例 Dice < 0.5 | 模型不可用 | 加真实样例微调；限制合成比例；纹理贴合用真实块 | P3 |
| R3 | 无现场数据，指标无意义 | Set-D 长期为空 | 交付无法验收 | 明确"方法闭环"交付边界（§6.3-7） | 全程 |
| R4 | 厚度标定失效（球机变焦） | 三源 α 差异 > 20% | 厚度系统性偏差 | 一致性检验 + 降级为 visual_only + 转人工 | P3 |
| R5 | 覆冰类型标注一致性差 | κ < 0.75 | 类型头学不到东西 | 先合并为 3 类；细化 rubric；专家仲裁 | P4 |
| R6 | 镜头结冰/污损 | 固定区域高频能量持续低 | 全部结论失效 | 质量门控 + 告警 + 建议硬件加热 | P0/P1 |
| R7 | Ultralytics AGPL 传染 | 代码审计发现 import | 交付受阻 | 生产链路禁用；仅 `prelabel.py` 使用 | P2 |
| R8 | 单卡训练周期超预期 | 单 epoch > 2 h | 排期滑移 | 降输入分辨率；冻结 backbone；减少合成量 | P1–P4 |
| R9 | 预警误报导致信任崩塌 | 人工复核推翻率 > 30% | 系统被弃用 | 提高阈值；强制人工复核；先只做"提示"不做"告警" | P5 |
| R10 | 类别体系与业务规程不一致 | 业务方不认 G 级 | 输出无法落地 | G 级与 I–IV 分离；I–IV 阈值由业务方定 | P0/P4 |

---

## 18. 附录

### 附录 A · 资源核验表（2026-10-19 实测）

| 资源 | 链接 | 核验结果 | 许可 |
|---|---|---|---|
| DTL-IceNet | zenodo.org/records/15718305 | ✅ 131.1 MB（图 130.9 + 代码 146.1 kB） | 未标注，待确认 |
| PLAIR-AW | github.com/ntuhubin/PLAIR-AW | ✅ 数据集 + 基线，无合成脚本 | 待核实 |
| OmniRestore | github.com/Judith989/omnirestore | ✅ CVPRW2026，2.60M/6.3ms | 待核实 |
| TTPLA | github.com/R3ab/ttpla_dataset | ✅ COCO 标注 + 转换脚本，2020 后未更新 | 未见 LICENSE |
| Weather-Image-Recognition | github.com/nisarg1631/Weather-Image-Recognition | ✅ 但为 HOG+SVM；数据 6862 张 11 类 | 随 Kaggle 数据集 |
| firc-dataset | gitee.com/FIRC/power-industry-dataset | ⚠️ 数据交易平台，精度无保证 | 不明，不采用 |

### 附录 B · 术语与缩写

| 缩写 | 全称 | 中文 |
|---|---|---|
| ASM | Atmospheric Scattering Model | 大气散射模型 |
| CB-CE | Class-Balanced Cross Entropy | 类别平衡交叉熵 |
| CORAL | Consistent Rank Logits | 一致秩 logits（有序回归） |
| ECE | Expected Calibration Error | 期望校准误差 |
| EKF | Extended Kalman Filter | 扩展卡尔曼滤波 |
| FiLM | Feature-wise Linear Modulation | 特征级线性调制 |
| IQR | Interquartile Range | 四分位距 |
| QWK | Quadratic Weighted Kappa | 二次加权 Kappa |
| RLE | Run-Length Encoding | 游程编码（掩码压缩） |

### 附录 C · 章节溯源（哪个结论来自哪份方案）

| 本方案章节 | 主要来源 | 增补/修正 |
|---|---|---|
| §6.3 边界声明 | plan1（三条）+ plan5（冰雹放弃） | 扩为 7 条 |
| §7.2 JSON 契约 | plan1 | 加 `measurement_status` 三态、`requires_review`、provenance |
| §8 标签体系 | plan1（双体系）+ plan3（覆盖率分级）+ plan6（晶态） | 加 `exposed_asset` 类与 4 条边界规则 |
| §9 架构分层 | plan1（功能分层）+ plan6（domain/application/infra） | 加失败降级矩阵 |
| §10.2 天气分类 | plan5（多标签）+ plan6（物理门控） | 门控降级为软约束；加温度标定与拒识 |
| §10.3 检测选型 | plan1（许可预警）+ plan6（SE/NWD） | **改为 timm+自建头**，避开 AGPL |
| §10.4 分割损失 | plan3（FN-aware Dice/Focal）+ plan6（Boundary） | 合并为三项加权 |
| §10.5 有序等级 | plan3（CORAL） | 明确 K=5、QWK 评估 |
| §10.6 物理量算 | **plan6（唯一来源）** | 加三源一致性检验、可信度三态、三源互校 |
| §10.7 时序与预警 | plan6（EKF/矩阵）+ plan5（多帧） | 加去抖帧数与配置化 |
| §11 数据工程 | plan1（manifest/泄漏）+ plan6（合成）+ plan3（场景分层） | 加许可台账与硬门槛 |
| §12/§13 工程 | plan1（src layout）+ plan6（pyproject/AGENTS） | 目录按功能重排，去冗余 |
| §15.4 指标 | plan1（分任务）+ plan3（QWK/A-B-C 集） | **重写为三栏制**，拒绝照抄论文 |
| §16 运维 | plan5（监控/漂移/模型卡）+ plan1（可追溯） | 加卡尺证据图 |

### 附录 D · 待确认事项（开工前必须闭环）

| # | 事项 | 找谁 | 影响 |
|---|---|---|---|
| 1 | DTL-IceNet 数据/代码的商用授权 | 论文通讯作者（Fu Yufei 等） | 能否用于交付 |
| 2 | 本项目是否属于"内部使用"（决定 AGPL 是否适用） | 法务 | 检测器选型 |
| 3 | 导线外径 `D_0` 台账是否可得 | 运检部门 | 能否算物理厚度 |
| 4 | 是否有 PTZ 编码器读数 / 双目相机 | 设备管理 | 标定级别 L2 可用性 |
| 5 | 本地区覆冰预警的现行规程阈值 | 业务方 | I–IV 级阈值 |
| 6 | 是否有历史晴天无冰图像（同预置位） | 运检部门 | L3 标定可用性 |
| 7 | 目标部署硬件（Jetson / x86 / 纯云端） | 项目组 | 模型尺寸与量化策略 |
| 8 | plan4.md 是否需要重新生成 | 文档作者 | 是否有遗漏视角 |

---

**版本**：v1.0 ｜ **编写**：基于 plan1–plan6 审阅合并 ｜ **下次修订触发条件**：现场数据到位、许可结论明确、或 P3 完成后的指标复盘
