"""SCUT 实测库全量审计（计划 P0 / T02）。

产出：
1. `artifacts/reports/scut_audit.json`：CSV 与磁盘盘点、解码过滤、SHA-256 精确重复、
   64-bit pHash 近重复簇、4 类冰型分布、分辨率交叉表、供电局/终端偏斜分布；
2. `artifacts/reports/scut_sample_gallery.csv`：每类随机抽样样本清单（供专家核对）；
3. `artifacts/reports/scut_noise_candidates.json`：18 张双「无覆冰」噪声候选详情。

本脚本只读，不改动 `NW_data/`。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402
from grid_ice_vision.data.dedup import cluster_near_duplicates, inspect_image_file  # noqa: E402

ICE_TYPE_EN = {
    "雾凇": "rime",
    "雨凇": "glaze",
    "湿雪": "wet_snow",
    "混合凇": "mixed",
}

SENTINEL_COLS = (
    "Beforday_Avg_Windspeed",
    "Yesterday_Avg_Windspeed",
    "Today_Avg_Windspeed",
    "Beforday_Avg_Humidity",
    "Yesterday_Avg_Humidity",
    "Today_Avg_Humidity",
)


def _is_noise_row(row: Any) -> bool:
    return (
        str(row.InsulatorShapeH) == "无覆冰"
        and str(row.InsulatorShapeV) == "无覆冰"
        and str(row.LineShapeH) == "无覆冰"
        and str(row.LineShapeV) == "无覆冰"
    )


def audit(
    out_path: Path,
    *,
    gallery_out: Path | None = None,
    noise_out: Path | None = None,
    samples_per_class: int = 20,
    phash_dist: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    root = paths.scut_root()
    csv_path = paths.scut_csv()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    print(f"[audit] SCUT 根目录 : {root}")
    print(f"[audit] CSV          : {csv_path.name}  {df.shape[0]} 行 × {df.shape[1]} 列")

    resolutions: Counter[tuple[int, int]] = Counter()
    res_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    decodable_ice_type: Counter[str] = Counter()
    bureau_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    monitor_counts: Counter[str] = Counter()

    missing: list[str] = []
    corrupt: list[str] = []
    corrupt_reasons: Counter[str] = Counter()
    present = 0

    valid_rows: list[dict[str, Any]] = []
    hashes_list: list[np.ndarray] = []
    sha_list: list[str] = []
    noise_records: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        rel = f"{int(row.Year)}/{row.Image_Name}.jpg"
        p = root / rel
        ice_zh = str(row.Ice_Type)
        ice_en = ICE_TYPE_EN.get(ice_zh, "unknown")
        is_noise = _is_noise_row(row)

        if not p.exists():
            missing.append(rel)
            if is_noise:
                noise_records.append(
                    {
                        "rel_path": rel,
                        "ice_type_zh": ice_zh,
                        "ice_type": ice_en,
                        "monitor_id": str(row.monitorId),
                        "bureau": str(row.Bureau),
                        "year": int(row.Year),
                        "object_type": str(row.ObjectType),
                        "cleanness": str(row.Cleanness),
                        "decodable": False,
                        "reason": "missing",
                    }
                )
            continue

        present += 1
        chk = inspect_image_file(p, with_sha=True, with_phash=True)
        if not chk.ok:
            corrupt.append(rel)
            corrupt_reasons[chk.reason] += 1
            if is_noise:
                noise_records.append(
                    {
                        "rel_path": rel,
                        "ice_type_zh": ice_zh,
                        "ice_type": ice_en,
                        "monitor_id": str(row.monitorId),
                        "bureau": str(row.Bureau),
                        "year": int(row.Year),
                        "object_type": str(row.ObjectType),
                        "cleanness": str(row.Cleanness),
                        "decodable": False,
                        "reason": chk.reason,
                    }
                )
            continue

        res_str = f"{chk.width}x{chk.height}"
        resolutions[(chk.width, chk.height)] += 1
        res_by_type[ice_en][res_str] += 1
        decodable_ice_type[ice_en] += 1
        bureau_by_type[str(row.Bureau)][ice_en] += 1
        monitor_counts[str(row.monitorId)] += 1

        assert chk.phash is not None
        hashes_list.append(chk.phash)
        sha_list.append(chk.sha256)

        rec = {
            "rel_path": rel,
            "image_id": str(row.Image_Name),
            "ice_type_zh": ice_zh,
            "ice_type": ice_en,
            "monitor_id": str(row.monitorId),
            "bureau": str(row.Bureau),
            "year": int(row.Year),
            "width": chk.width,
            "height": chk.height,
            "sha256": chk.sha256,
            "phash": chk.phash_hex,
            "object_type": str(row.ObjectType),
            "cleanness": str(row.Cleanness),
            "insulator_shape_h": str(row.InsulatorShapeH),
            "insulator_shape_v": str(row.InsulatorShapeV),
            "line_shape_h": str(row.LineShapeH),
            "line_shape_v": str(row.LineShapeV),
            "noise_candidate": is_noise,
        }
        valid_rows.append(rec)
        if is_noise:
            noise_records.append({**rec, "decodable": True})

    disk_total = sum(1 for y in range(2014, 2019) for _ in (root / str(y)).glob("*.jpg"))
    total_decoded = len(valid_rows)
    low_res = sum(v for (w, _), v in resolutions.items() if w <= 704)

    # ---- SHA-256 与 pHash 近重复审计 ----------------------------------------
    sha_counts = Counter(sha_list)
    exact_dup_groups = sum(1 for c in sha_counts.values() if c > 1)
    exact_dup_extra = sum(c - 1 for c in sha_counts.values() if c > 1)
    sha_to_monitors: dict[str, set[str]] = defaultdict(set)
    for rec in valid_rows:
        sha_to_monitors[rec["sha256"]].add(rec["monitor_id"])
    exact_dup_cross_mon = sum(1 for ms in sha_to_monitors.values() if len(ms) > 1)

    hashes_arr = np.stack(hashes_list, axis=0) if hashes_list else np.zeros((0, 8), dtype=np.uint8)
    clusters, cluster_stats = cluster_near_duplicates(
        hashes_arr, max_dist=phash_dist, sha256_list=sha_list
    )
    cluster_monitors: dict[int, set[str]] = defaultdict(set)
    for cid, rec in zip(clusters.tolist(), valid_rows, strict=True):
        cluster_monitors[cid].add(rec["monitor_id"])
    near_dup_cross_mon = sum(1 for ms in cluster_monitors.values() if len(ms) > 1)

    print(f"[audit] CSV 行 {len(df)} / 磁盘 {disk_total} 图 / 可解码 {total_decoded}")
    print(f"[audit] 缺失 {len(missing)} / 损坏 {len(corrupt)} ({dict(corrupt_reasons)})")
    print(
        f"[audit] SHA-256 唯一 {len(sha_counts)} / 精确重复组 {exact_dup_groups} "
        f"(跨终端 {exact_dup_cross_mon}) | pHash 簇(d<={phash_dist}) "
        f"{cluster_stats['n_clusters']} (跨终端 {near_dup_cross_mon})"
    )

    def vc(col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in df[col].value_counts().items()}

    wind_cols = [c for c in df.columns if c in SENTINEL_COLS and "Windspeed" in c]
    frozen_mask = (df[wind_cols] == 0).all(axis=1)
    per_col_zero = {c: int((df[c] == 0).sum()) for c in SENTINEL_COLS}

    times = pd.to_datetime(df["Image_Time"], errors="coerce", format="mixed")
    valid_t = times.notna().sum()

    mon_vals = np.array(list(monitor_counts.values()), dtype=np.int64)
    mon_stats = {
        "n_monitors_decodable": len(monitor_counts),
        "min": int(mon_vals.min()) if mon_vals.size else 0,
        "median": float(np.median(mon_vals)) if mon_vals.size else 0.0,
        "p90": float(np.percentile(mon_vals, 90)) if mon_vals.size else 0.0,
        "max": int(mon_vals.max()) if mon_vals.size else 0,
        "top10": [{"monitor_id": m, "count": int(c)} for m, c in monitor_counts.most_common(10)],
    }

    report: dict[str, Any] = {
        "scut_root": str(root),
        "csv_path": str(csv_path),
        "csv_rows": len(df),
        "csv_cols": int(df.shape[1]),
        "disk_images": int(disk_total),
        "csv_rows_with_file": int(present),
        "decodable_images": int(total_decoded),
        "missing_images": missing,
        "corrupt_images": corrupt,
        "corrupt_reasons": dict(corrupt_reasons),
        "dedup": {
            "unique_sha256": len(sha_counts),
            "exact_dup_groups": exact_dup_groups,
            "exact_dup_extra_images": exact_dup_extra,
            "exact_dup_cross_monitor": exact_dup_cross_mon,
            "phash_dist": phash_dist,
            **cluster_stats,
            "near_dup_cross_monitor": near_dup_cross_mon,
        },
        "resolution_top": [
            {"width": w, "height": h, "count": int(c)} for (w, h), c in resolutions.most_common(12)
        ],
        "resolution_by_ice_type": {k: dict(v) for k, v in sorted(res_by_type.items())},
        "low_res_ratio_le704": round(low_res / max(total_decoded, 1), 4),
        "n_monitors": int(df["monitorId"].nunique()),
        "n_bureaus": int(df["Bureau"].nunique()),
        "n_towers": int(df["Tower"].nunique()),
        "monitor_skew": mon_stats,
        "bureau_by_ice_type": {k: dict(v) for k, v in sorted(bureau_by_type.items())},
        "ice_type": vc("Ice_Type"),
        "decodable_ice_type": dict(decodable_ice_type),
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
        "no_ice_label_candidates": {
            "count": len(noise_records),
            "images": [r["rel_path"] for r in noise_records],
        },
        "n_valid_timestamps": int(valid_t),
        "timestamp_range": [
            str(times.min()) if valid_t else None,
            str(times.max()) if valid_t else None,
        ],
        "years": {str(k): int(v) for k, v in df["Year"].value_counts().sort_index().items()},
    }

    per_mon_years = df.groupby("monitorId")["Year"].nunique()
    report["monitors_spanning_multiple_years"] = int((per_mon_years > 1).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 导出每类抽样画廊 CSV ----------------------------------------------
    gallery_path = gallery_out or (out_path.parent / "scut_sample_gallery.csv")
    rng = np.random.default_rng(seed)
    by_cls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in valid_rows:
        by_cls[rec["ice_type"]].append(rec)

    gallery_rows: list[dict[str, Any]] = []
    for cls_name in ("glaze", "rime", "wet_snow", "mixed"):
        pool = by_cls.get(cls_name, [])
        n_pick = min(samples_per_class, len(pool))
        idx = rng.choice(len(pool), size=n_pick, replace=False) if n_pick else []
        for i in sorted(int(v) for v in idx):
            gallery_rows.append(pool[i])

    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "ice_type",
        "ice_type_zh",
        "rel_path",
        "monitor_id",
        "bureau",
        "year",
        "width",
        "height",
        "sha256",
        "phash",
        "insulator_shape_h",
        "insulator_shape_v",
        "line_shape_h",
        "line_shape_v",
        "noise_candidate",
    ]
    with gallery_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(gallery_rows)

    # ---- 导出 18 张噪声候选 ------------------------------------------------
    noise_path = noise_out or (out_path.parent / "scut_noise_candidates.json")
    noise_path.write_text(
        json.dumps(
            {"count": len(noise_records), "candidates": noise_records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[audit] 可解码四类分布: {dict(decodable_ice_type)}")
    print(f"[audit] 低清(≤704px)占比 {report['low_res_ratio_le704']:.1%}")
    print(
        f"[audit] 终端 {report['n_monitors']} / 局 {report['n_bureaus']} / "
        f"跨年终端 {report['monitors_spanning_multiple_years']}"
    )
    print(f"[audit] 双无冰噪声候选 {len(noise_records)} 张 -> {noise_path}")
    print(f"[audit] 抽样清单 {len(gallery_rows)} 条 -> {gallery_path}")
    print(f"[audit] 写出 {out_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SCUT 7,296 图与 CSV 全量审计（含哈希去重与抽样）")
    ap.add_argument(
        "--out",
        default=str(PROJ / "artifacts" / "reports" / "scut_audit.json"),
        help="输出 JSON 路径",
    )
    ap.add_argument("--gallery-out", default=None, help="抽样画廊 CSV 路径")
    ap.add_argument("--noise-out", default=None, help="双无冰噪声候选 JSON 路径")
    ap.add_argument("--samples-per-class", type=int, default=20)
    ap.add_argument("--phash-dist", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    audit(
        Path(args.out),
        gallery_out=Path(args.gallery_out) if args.gallery_out else None,
        noise_out=Path(args.noise_out) if args.noise_out else None,
        samples_per_class=args.samples_per_class,
        phash_dist=args.phash_dist,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
