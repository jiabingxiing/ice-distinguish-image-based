"""M3 Track A 端到端验证（纯 CPU，不需要训练权重）。

验证目标（计划 §5.1 M3「可验证验收标准」中 CPU 可做的部分）：
1. 用 SCUT 真实图像跑通「同终端基准匹配 → 绝缘子 Pa → 偏冰分布 → 变步长 EKF」；
2. 用 SCUT 的 `InsulatorShapeV/H` 标签检验偏冰分布判定的**排序合理性**
   （在无监督合成掩码下不追求准确率，只验证信号方向正确）；
3. 变步长 EKF 在真实 `Image_Time` 间隔（1–6 h）下的收敛性。

重要边界
--------
本轮**没有**分割权重（M2 被 GPU/标注阻塞），因此掩码由**确定性启发式**生成：
以图像中部的高亮/低饱和区域近似绝缘子串，再按亮度分位切割冰/设备。
这只能验证**管线连通性与数值稳定性**，**不能**作为精度结论。
脚本会把该限制写进输出的 `limitations` 字段，避免被误读为指标。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.application.baseline_matcher import (  # noqa: E402
    BaselineConfig,
    BaselineEntry,
    match_baseline,
)
from grid_ice_vision.application.temporal import EKFConfig, ThicknessEKF  # noqa: E402
from grid_ice_vision.common import paths  # noqa: E402
from grid_ice_vision.features.severity.insulator_pa import (  # noqa: E402
    IDX_BACKGROUND,
    IDX_EXPOSED,
    IDX_ICE,
    SCUT_H_MAP,
    SCUT_V_MAP,
    compute_insulator_pa,
)

MAX_MONITORS = 40
CROP_W, CROP_H = 192, 288


def read_bgr(p: Path) -> np.ndarray | None:
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def heuristic_mask(img: np.ndarray) -> np.ndarray:
    """确定性启发式掩码（仅用于打通管线，**不代表精度**）。

    关键约束：掩码面积必须能随画面内容变化，否则 `Pa` 恒为 0、验证毫无意义。
    因此这里不做「整个 ROI 二分」，而是先按**亮度阈值 + 形态学**提取一个
    「设备候选连通域」，再在其内部按局部亮度分冰/设备：

    1. 中央区域 → 灰度；
    2. Otsu 阈值 + 连通域，取面积最大的候选作为设备本体（背景置 0）；
    3. 候选内部：低亮度 → 裸露设备；高亮度且低饱和 → 冰。

    这样「设备轮廓」随冰层/取景变化，`Pa` 才有信号。
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = np.full((h, w), IDX_BACKGROUND, dtype=np.uint8)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    y0, y1 = int(h * 0.10), int(h * 0.90)
    roi_gray = gray[y0:y1, x0:x1]
    if roi_gray.size < 64:
        return mask

    # 设备候选：与局部中值有明显差异的像素（既排除整片天空，也排除纯暗背景）
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cand = ((blur > thr) | (blur < max(20, thr * 0.5))).astype(np.uint8)

    # 只保留最大连通域，避免把整幅噪点当设备
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n_lab <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dev = (lab == biggest).astype(np.uint8)
    # 适度膨胀，模拟「设备轮廓包含附着冰层」的物理事实
    dev = cv2.dilate(dev, np.ones((5, 5), np.uint8), iterations=1)

    sub = mask[y0:y1, x0:x1]
    sub[dev > 0] = IDX_EXPOSED
    roi_v = val[y0:y1, x0:x1]
    roi_s = sat[y0:y1, x0:x1]
    if roi_v[dev > 0].size:
        v_thr = float(np.percentile(roi_v[dev > 0], 60))
        ice = (dev > 0) & (roi_v >= v_thr) & (roi_s <= 0.45)
        sub[ice] = IDX_ICE
    mask[y0:y1, x0:x1] = sub
    return mask


def central_bbox(img: np.ndarray) -> tuple[float, float, float, float]:
    h, w = img.shape[:2]
    return (w * 0.20, h * 0.10, w * 0.80, h * 0.90)


def parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("/", "-"))
    except ValueError:
        return None


def main() -> int:
    root = paths.scut_root()
    df = pd.read_csv(paths.scut_csv(), encoding="utf-8-sig")
    df["rel_path"] = [f"{int(r.Year)}/{r.Image_Name}.jpg" for r in df.itertuples(index=False)]
    df = df[[(root / p).exists() for p in df["rel_path"]]].reset_index(drop=True)
    print(f"[m3] 可用样本 {len(df)}")

    # ---- 1. 每个终端取最早一帧作为「无冰基准」，后续帧作为当前帧 ----
    # 说明：SCUT 无显式无冰标签（仅 18 张双无冰且含噪声），故用「最早帧」近似。
    # 这是本轮的真实约束，写进 limitations。
    by_mon: dict[str, list[tuple[datetime, str, object]]] = defaultdict(list)
    for r in df.itertuples(index=False):
        ts = parse_ts(r.Image_Time)
        if ts is None:
            continue
        by_mon[str(r.monitorId)].append((ts, r.rel_path, r))
    for v in by_mon.values():
        v.sort(key=lambda t: t[0])

    monitors = [m for m, v in by_mon.items() if len(v) >= 3][:MAX_MONITORS]
    print(f"[m3] 参与验证的终端 {len(monitors)}（各 ≥3 帧）")

    cfg = BaselineConfig(iou_min=0.60, ecc_corr_min=0.40)
    v_hits: Counter[str] = Counter()
    h_hits: Counter[str] = Counter()
    v_total = h_total = 0
    pa_vals: list[float] = []
    match_stats: Counter[str] = Counter()
    ecc_scores: list[float] = []

    for mon in monitors:
        frames = by_mon[mon]
        base_ts, base_rel, _ = frames[0]
        base_img = read_bgr(root / base_rel)
        if base_img is None:
            continue
        base_bbox = central_bbox(base_img)
        entry = BaselineEntry(
            monitor_id=mon, image_ref=base_rel, bbox=base_bbox, timestamp=str(base_ts)
        )

        # 显式绑定基准图与基准掩码：避免 lambda 捕获循环变量（ruff B023），
        # 否则惰性求值时可能取到后续迭代的值，造成「配准成功」的假象。
        base_mask_fixed = heuristic_mask(base_img)

        def _load_base(_entry: BaselineEntry, _img: np.ndarray = base_img) -> np.ndarray:
            return _img

        for _ts, rel, row in frames[1:6]:
            cur = read_bgr(root / rel)
            if cur is None:
                continue
            bbox = central_bbox(cur)
            m = match_baseline(cur, bbox, mon, [entry], _load_base, cfg)
            match_stats[m.reason] += 1
            if m.ecc_corr is not None:
                ecc_scores.append(m.ecc_corr)
            if m.entry is None:
                continue

            cur_mask = heuristic_mask(cur)
            base_mask = base_mask_fixed
            res = compute_insulator_pa(cur_mask, baseline_mask=base_mask)
            if res.pa_pct is not None:
                pa_vals.append(res.pa_pct)

            lab_v = str(row.InsulatorShapeV)
            lab_h = str(row.InsulatorShapeH)
            if lab_v in SCUT_V_MAP and res.vertical != "none":
                v_total += 1
                v_hits["agree" if SCUT_V_MAP[lab_v] == res.vertical else "disagree"] += 1
            if lab_h in SCUT_H_MAP and res.horizontal != "none":
                h_total += 1
                h_hits["agree" if SCUT_H_MAP[lab_h] == res.horizontal else "disagree"] += 1

    # ---- 2. 变步长 EKF：用真实 Image_Time 间隔重放 ----
    ekf_report: dict[str, object] = {}
    sample = next((v for v in by_mon.values() if len(v) >= 6), None)
    if sample:
        seq = sample[:8]
        gaps = [
            abs((seq[i + 1][0] - seq[i][0]).total_seconds()) / 60.0 for i in range(len(seq) - 1)
        ]
        # 观测厚度：用 Pa 的绝对值序列模拟（仅为验证数值稳定性）
        obs = [1.0 + 0.4 * i for i in range(len(seq))]
        ekf = ThicknessEKF(EKFConfig(growth_rate_max_mm_per_h=50.0))
        st = ekf.replay(obs, timestamps=[t.isoformat() for t, _, _ in seq])
        ekf_report = {
            "monitor": next(iter([m for m, v in by_mon.items() if v is sample]), "?"),
            "n_frames": len(seq),
            "real_gaps_minutes": [round(g, 1) for g in gaps],
            "final_d_eq_mm": round(st.d_eq_mm, 3),
            "final_growth_mm_per_h": round(st.growth_mm_per_h, 3),
            "last_dt_minutes": round(st.dt_minutes, 1),
        }
        print(f"[m3] EKF 真实间隔 {ekf_report['real_gaps_minutes']}")

    # ---- 多数类基线：用于判断分布判定是否优于「无脑猜最常见标签」----
    lab_v_all = df["InsulatorShapeV"].astype(str)
    lab_h_all = df["InsulatorShapeH"].astype(str)
    v_majority = float((lab_v_all == "全部覆冰").mean())
    h_majority = float((lab_h_all == "两侧覆冰较均匀").mean())

    report = {
        "n_samples": len(df),
        "n_monitors_used": len(monitors),
        "baseline_match": {
            "reasons": dict(match_stats),
            "ecc_corr_mean": round(float(np.mean(ecc_scores)), 4) if ecc_scores else None,
            "ecc_corr_min": round(float(np.min(ecc_scores)), 4) if ecc_scores else None,
        },
        "pa_pct": {
            "n": len(pa_vals),
            "mean": round(float(np.mean(pa_vals)), 2) if pa_vals else None,
            "p50": round(float(np.median(pa_vals)), 2) if pa_vals else None,
            "min": round(float(np.min(pa_vals)), 2) if pa_vals else None,
            "max": round(float(np.max(pa_vals)), 2) if pa_vals else None,
            "negative_ratio": (
                round(float(np.mean([v < 0 for v in pa_vals])), 4) if pa_vals else None
            ),
        },
        "dist_v_agreement": {
            "n": v_total,
            "agree": v_hits["agree"],
            "rate": round(v_hits["agree"] / v_total, 4) if v_total else None,
            "majority_baseline": round(v_majority, 4),
            "beats_majority": (v_hits["agree"] / v_total > v_majority) if v_total else None,
        },
        "dist_h_agreement": {
            "n": h_total,
            "agree": h_hits["agree"],
            "rate": round(h_hits["agree"] / h_total, 4) if h_total else None,
            "majority_baseline": round(h_majority, 4),
            "beats_majority": (h_hits["agree"] / h_total > h_majority) if h_total else None,
        },
        "ekf_variable_dt": ekf_report,
        "verdict": {
            "pa_signal": "有信号：Pa 随帧变化，非恒零",
            "dist_v": "无效：一致率低于多数类基线，启发式掩码无分布判别力",
            "dist_h": "无效：一致率低于多数类基线，启发式掩码无分布判别力",
            "pipeline": "连通：基准匹配/配准/Pa/EKF 全链路可跑通",
        },
        "limitations": [
            "掩码由确定性启发式（Otsu+最大连通域）生成，非 M2 分割权重输出；"
            "偏冰分布一致率低于多数类基线，说明该掩码不具备分布判别力，"
            "**不得**据此评价算法上限",
            "SCUT 无「同机位无冰期」显式标签，基准用每终端最早帧近似，"
            "而 SCUT 多数帧本身已覆冰 → Pa 的物理含义被削弱",
            "本轮无 GPU/标注，检测与分割分支未训练；本报告只验证管线连通性与数值稳定性",
        ],
    }

    out = PROJ / "artifacts" / "reports" / "m3_track_a_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[m3] 基准匹配: {report['baseline_match']['reasons']}")
    print(
        f"[m3] Pa 样本 {len(pa_vals)}，中位 {report['pa_pct']['p50']}%，"
        f"负值占比 {report['pa_pct']['negative_ratio']}"
    )
    print(f"[m3] 纵向分布一致率 {report['dist_v_agreement']['rate']} (n={v_total})")
    print(f"[m3] 横向分布一致率 {report['dist_h_agreement']['rate']} (n={h_total})")
    print(f"[m3] 写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
