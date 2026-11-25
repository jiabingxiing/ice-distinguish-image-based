"""SCUT 实测库全量审计（计划 T02 / 里程碑 M0）。

产出 `artifacts/reports/scut_audit.json`，核验：
1. CSV 7,297 行 ↔ 磁盘 7,296 图的差集（计划指出缺 1 图）；
2. 每张图能否解码、真实分辨率分布（验证「93.2% 低清」口径）；
3. 冻结风速哨兵（`Wa==0`）与湿度零值分布；
4. `monitorId` / `Bureau` / `Ice_Type` / 污秽 / 部件标签分布；
5. 标注噪声候选（`InsulatorShape*` 与 `LineShape*` 同为「无覆冰」的行）。

本脚本只读，不改动 `NW_data/`。CSV 用 utf-8-sig 读取：文件带 BOM 且个别行
含非法字节，`gb18030` 会失败（实测 position 805）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402

WEATHER_PREFIX = ("Beforday_", "Yesterday_", "Today_")
SENTINEL_COLS = (
    "Beforday_Avg_Windspeed",
    "Yesterday_Avg_Windspeed",
    "Today_Avg_Windspeed",
    "Beforday_Avg_Humidity",
    "Yesterday_Avg_Humidity",
    "Today_Avg_Humidity",
)


def decode(path: Path) -> np.ndarray | None:
    """用 imdecode 读中文/空格路径（cv2.imread 在 Windows 上会失败）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def audit(out_path: Path) -> dict[str, object]:
    root = paths.scut_root()
    csv_path = paths.scut_csv()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    print(f"[audit] SCUT 根目录 : {root}")
    print(f"[audit] CSV          : {csv_path.name}  {df.shape[0]} 行 × {df.shape[1]} 列")

    # ---- 1. 文件存在性与解码 ------------------------------------------------
    resolutions: Counter[tuple[int, int]] = Counter()
    missing: list[str] = []
    corrupt: list[str] = []
    present = 0
    for row in df.itertuples(index=False):
        p = root / str(row.Year) / f"{row.Image_Name}.jpg"
        if not p.exists():
            missing.append(f"{row.Year}/{row.Image_Name}.jpg")
            continue
        present += 1
        im = decode(p)
        if im is None:
            corrupt.append(f"{row.Year}/{row.Image_Name}.jpg")
            continue
        resolutions[(im.shape[1], im.shape[0])] += 1

    disk_total = sum(1 for y in range(2014, 2019) for _ in (root / str(y)).glob("*.jpg"))

    total_decoded = sum(resolutions.values())
    low_res = sum(v for (w, _), v in resolutions.items() if w <= 704)
    print(f"[audit] CSV 行 {len(df)} / 磁盘 {disk_total} 图 / 可解码 {total_decoded}")
    print(f"[audit] 缺失 {len(missing)} / 损坏 {len(corrupt)}")

    # ---- 2. 分布 ------------------------------------------------------------
    def vc(col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    # ---- 3. 气象哨兵 --------------------------------------------------------
    wind_cols = [c for c in df.columns if c in SENTINEL_COLS and "Windspeed" in c]
    frozen_mask = (df[wind_cols] == 0).all(axis=1)
    per_col_zero = {c: int((df[c] == 0).sum()) for c in SENTINEL_COLS}

    # ---- 4. 标注噪声候选 ----------------------------------------------------
    noise = df[
        (df["InsulatorShapeH"].astype(str) == "无覆冰")
        & (df["InsulatorShapeV"].astype(str) == "无覆冰")
        & (df["LineShapeH"].astype(str) == "无覆冰")
        & (df["LineShapeV"].astype(str) == "无覆冰")
    ]
    noise_ids = [f"{int(r.Year)}/{r.Image_Name}.jpg" for r in noise.itertuples(index=False)]

    # ---- 5. 时间跨度 --------------------------------------------------------
    times = pd.to_datetime(df["Image_Time"], errors="coerce", format="mixed")
    valid_t = times.notna().sum()

    report: dict[str, object] = {
        "scut_root": str(root),
        "csv_path": str(csv_path),
        "csv_rows": len(df),
        "csv_cols": int(df.shape[1]),
        "disk_images": int(disk_total),
        "csv_rows_with_file": int(present),
        "decodable_images": int(total_decoded),
        "missing_images": missing,
        "corrupt_images": corrupt,
        "resolution_top": [
            {"width": w, "height": h, "count": int(c)} for (w, h), c in resolutions.most_common(12)
        ],
        "low_res_ratio_le704": round(low_res / max(total_decoded, 1), 4),
        "n_monitors": int(df["monitorId"].nunique()),
        "n_bureaus": int(df["Bureau"].nunique()),
        "n_towers": int(df["Tower"].nunique()),
        "ice_type": vc("Ice_Type"),
        "cleanness": vc("Cleanness"),
        "frozen": vc("Frozen"),
        "normal": vc("Normal"),
        "object_type": vc("ObjectType"),
        "insulator_shape_h": vc("InsulatorShapeH"),
        "insulator_shape_v": vc("InsulatorShapeV"),
        "line_shape_h": vc("LineShapeH"),
        "line_shape_v": vc("LineShapeV"),
        "weather_sentinel": {
            "cols": SENTINEL_COLS,
            "zero_counts_per_col": per_col_zero,
            "wind_all_zero_rows": int(frozen_mask.sum()),
            "wind_all_zero_ratio": round(float(frozen_mask.mean()), 4),
        },
        "no_ice_label_candidates": {"count": len(noise_ids), "images": noise_ids},
        "n_valid_timestamps": int(valid_t),
        "timestamp_range": [
            str(times.min()) if valid_t else None,
            str(times.max()) if valid_t else None,
        ],
        "years": {str(k): int(v) for k, v in df["Year"].value_counts().sort_index().items()},
    }

    # 同一终端是否跨年份重复出现（泄漏风险证据）
    per_mon_years = df.groupby("monitorId")["Year"].nunique()
    report["monitors_spanning_multiple_years"] = int((per_mon_years > 1).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] 低清(≤704px)占比 {report['low_res_ratio_le704']:.1%}")
    print(
        f"[audit] 终端 {report['n_monitors']} / 局 {report['n_bureaus']} / "
        f"跨年终端 {report['monitors_spanning_multiple_years']}"
    )
    print(f"[audit] 冻杯(三天风速全 0) {report['weather_sentinel']['wind_all_zero_rows']} 行")
    print(f"[audit] 双无冰噪声候选 {len(noise_ids)} 张")
    print(f"[audit] 写出 {out_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SCUT 7,296 图与 CSV 全量审计")
    ap.add_argument(
        "--out",
        default=str(PROJ / "artifacts" / "reports" / "scut_audit.json"),
        help="输出 JSON 路径",
    )
    args = ap.parse_args(argv)
    audit(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
