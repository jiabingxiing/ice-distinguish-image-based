"""生成 SCUT 双协议划分索引与纯净四分类清单（计划 P0 / T02）。

六级过滤管线：
1. 文件存在校验（剔除缺失图）；
2. OpenCV 解码、3 通道、最小边长 ≥ 32、非死图（std > 1.0）校验（剔除 52 张损坏图）；
3. 标签合法性校验：严格映射至 `ICE_TYPE_FOUR_CLASSES`（禁用旧版 `<50 并入 mixed`）；
4. SHA-256 精确哈希与 64-bit pHash（汉明距离 ≤ 4）近重复集群锁定，跨终端集群并入同一原子组；
5. 双「无覆冰」噪声候选标记 `noise_candidate=True`（支持 `--exclude-noise-candidates` 过滤）；
6. 整终端分层抽样（Monitor-Holdout）+ 按年时间划分（Temporal-Holdout）与泄漏断言。

产出：
  data/manifests/scut_ice_type_v1.jsonl          协议 A 纯净四分类主清单（含 sha256/phash/尺寸）
  data/manifests/scut_ice_type_temporal_v1.jsonl 协议 B 纯净四分类清单
  data/manifests/scut_v2_monitor.jsonl           兼容 v2 协议 A 全量清单
  data/manifests/scut_v2_temporal.jsonl          兼容 v2 协议 B 全量清单
  data/splits/scut_monitor_v2.jsonl              协议 A 索引（入库锁定）
  data/splits/scut_temporal_v2.jsonl             协议 B 索引（入库锁定）
  artifacts/reports/scut_splits.json             划分统计报告
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES  # noqa: E402
from grid_ice_vision.data.dedup import (  # noqa: E402
    UnionFind,
    cluster_near_duplicates,
    inspect_image_file,
)
from grid_ice_vision.data.manifest import (  # noqa: E402
    ManifestRecord,
    assert_no_leakage,
    summarize,
    write_manifest,
)

ICE_TYPE_EN = {
    "雾凇": "rime",
    "雨凇": "glaze",
    "湿雪": "wet_snow",
    "混合凇": "mixed",
}
SHAPE_EN = {
    "无覆冰": "none",
    "上端覆冰": "top",
    "下端覆冰": "bottom",
    "中部覆冰": "middle",
    "全部覆冰": "all",
}


def _is_noise_row(row: Any) -> bool:
    return (
        str(row.InsulatorShapeH) == "无覆冰"
        and str(row.InsulatorShapeV) == "无覆冰"
        and str(row.LineShapeH) == "无覆冰"
        and str(row.LineShapeV) == "无覆冰"
    )


def load_csv(
    *,
    decode_check: bool = True,
    with_sha: bool = True,
    phash_dist: int = 4,
    exclude_noise_candidates: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取 SCUT CSV 并执行存在性/解码/标签/去重检查。"""
    df = pd.read_csv(paths.scut_csv(), encoding="utf-8-sig")
    root = paths.scut_root()

    kept_idx: list[int] = []
    rel_paths: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    shas: list[str] = []
    phashes_hex: list[str] = []
    phash_vecs: list[np.ndarray] = []
    noise_flags: list[bool] = []

    missing_cnt = 0
    corrupt_cnt = 0
    corrupt_reasons: Counter[str] = Counter()
    noise_excluded = 0

    for idx, row in enumerate(df.itertuples(index=False)):
        rel = paths.rel_posix(f"{int(row.Year)}/{row.Image_Name}.jpg")
        full_path = root / rel
        if not full_path.exists():
            missing_cnt += 1
            continue

        mapped_type = ICE_TYPE_EN.get(str(row.Ice_Type))
        if mapped_type not in ICE_TYPE_FOUR_CLASSES:
            raise ValueError(
                f"非法 Ice_Type={row.Ice_Type!r}（{rel}），不在 {ICE_TYPE_FOUR_CLASSES}"
            )

        is_noise = _is_noise_row(row)
        if decode_check:
            chk = inspect_image_file(full_path, with_sha=with_sha, with_phash=True)
            if not chk.ok:
                corrupt_cnt += 1
                corrupt_reasons[chk.reason] += 1
                continue
            if exclude_noise_candidates and is_noise:
                noise_excluded += 1
                continue
            assert chk.phash is not None
            widths.append(chk.width)
            heights.append(chk.height)
            shas.append(chk.sha256)
            phashes_hex.append(chk.phash_hex)
            phash_vecs.append(chk.phash)
        else:
            if exclude_noise_candidates and is_noise:
                noise_excluded += 1
                continue
            widths.append(0)
            heights.append(0)
            shas.append("")
            phashes_hex.append("")

        kept_idx.append(idx)
        rel_paths.append(rel)
        noise_flags.append(is_noise)

    clean = df.iloc[kept_idx].copy().reset_index(drop=True)
    if phash_vecs:
        clusters, dedup_stats = cluster_near_duplicates(
            np.stack(phash_vecs, axis=0),
            max_dist=phash_dist,
            sha256_list=shas if with_sha else None,
        )
        near_dup_groups = [f"nd{int(c):05d}" for c in clusters.tolist()]
    else:
        dedup_stats = {"n_images": len(clean), "n_clusters": len(clean)}
        near_dup_groups = [f"nd{i:05d}" for i in range(len(clean))]

    clean["rel_path"] = rel_paths
    clean["img_width"] = widths
    clean["img_height"] = heights
    clean["img_sha256"] = shas
    clean["img_phash"] = phashes_hex
    clean["near_dup_group"] = near_dup_groups
    clean["noise_candidate"] = noise_flags

    stats: dict[str, Any] = {
        "csv_rows": len(df),
        "missing_excluded": missing_cnt,
        "corrupt_excluded": corrupt_cnt,
        "corrupt_reasons": dict(corrupt_reasons),
        "noise_candidates_total": int(sum(noise_flags)) + noise_excluded,
        "noise_candidates_excluded": noise_excluded,
        "valid_records": len(clean),
        "dedup": dedup_stats,
    }
    return clean, stats


def to_records(df: pd.DataFrame) -> list[ManifestRecord]:
    out: list[ManifestRecord] = []
    for row in df.itertuples(index=False):
        ts = str(row.Image_Time).replace("/", "-")
        ice_type = ICE_TYPE_EN[str(row.Ice_Type)]
        labels = {
            "ice_type": ice_type,
            "ice_type_zh": str(row.Ice_Type),
            "insulator_shape_h": SHAPE_EN.get(str(row.InsulatorShapeH), "unknown"),
            "insulator_shape_v": SHAPE_EN.get(str(row.InsulatorShapeV), "unknown"),
            "line_shape_h": SHAPE_EN.get(str(row.LineShapeH), "unknown"),
            "line_shape_v": SHAPE_EN.get(str(row.LineShapeV), "unknown"),
            "cleanness": str(row.Cleanness),
            "object_type": str(row.ObjectType),
            "bureau": str(row.Bureau),
            "tower": str(row.Tower),
            "ice_season": str(row.Year),
            "is_synthetic": False,
            "phash": str(row.img_phash),
            "near_dup_group": str(row.near_dup_group),
            "noise_candidate": bool(row.noise_candidate),
        }
        w = int(row.img_width) if int(row.img_width) > 0 else None
        h = int(row.img_height) if int(row.img_height) > 0 else None
        out.append(
            ManifestRecord(
                image_id=str(row.Image_Name),
                path=paths.rel_posix(row.rel_path),
                sha256=str(row.img_sha256),
                site_id=str(row.Bureau),
                camera_id=str(row.monitorId),
                weather_event_id=str(row.Year),
                timestamp=ts,
                split="train",
                is_synthetic=False,
                scene_group="field",
                labels=labels,
                label_version="scut_ice_type_v1",
                width=w,
                height=h,
            )
        )
    return out


def split_monitor(records: list[ManifestRecord], seed: int = 42) -> dict[str, str]:
    """协议 A：整终端分层抽样，并锁定跨终端近重复集群不跨 split。"""
    monitors = sorted({r.camera_id for r in records})
    mon_idx = {m: i for i, m in enumerate(monitors)}
    uf = UnionFind(len(monitors))

    # 若同一 near_dup_group 或 sha256 出现在多个终端，将其所在终端绑定在同一原子组件
    dup_to_mons: dict[str, list[str]] = defaultdict(list)
    for r in records:
        nd = str(r.labels.get("near_dup_group", ""))
        if nd:
            dup_to_mons[f"nd:{nd}"].append(r.camera_id)
        if r.sha256:
            dup_to_mons[f"sha:{r.sha256}"].append(r.camera_id)
    for ms in dup_to_mons.values():
        if len(ms) > 1:
            base_i = mon_idx[ms[0]]
            for m in ms[1:]:
                uf.union(base_i, mon_idx[m])

    comp_monitors: dict[int, list[str]] = defaultdict(list)
    for m in monitors:
        comp_monitors[uf.find(mon_idx[m])].append(m)

    mon_bureaus: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        mon_bureaus[r.camera_id][r.site_id] += 1

    by_bureau: dict[str, list[int]] = defaultdict(list)
    for comp_id, ms in sorted(comp_monitors.items()):
        bc: Counter[str] = Counter()
        for m in ms:
            bc.update(mon_bureaus[m])
        primary_bureau = bc.most_common(1)[0][0]
        by_bureau[primary_bureau].append(comp_id)

    rng = random.Random(seed)
    comp_assign: dict[int, str] = {}
    for _bureau, comps in sorted(by_bureau.items()):
        cs = sorted(comps)
        rng.shuffle(cs)
        n = len(cs)
        n_test = max(1, round(n * 0.15)) if n >= 4 else 0
        n_val = max(1, round(n * 0.15)) if n >= 4 else 0
        if n_test + n_val >= n:
            n_test, n_val = 0, 0
        for i, c in enumerate(cs):
            comp_assign[c] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    assign: dict[str, str] = {}
    for comp_id, ms in comp_monitors.items():
        sp = comp_assign[comp_id]
        for m in ms:
            assign[m] = sp

    counts = Counter(assign.values())
    if counts.get("test", 0) == 0:
        biggest = Counter(r.camera_id for r in records).most_common(1)[0][0]
        for m in comp_monitors[uf.find(mon_idx[biggest])]:
            assign[m] = "test"
    return assign


def split_temporal(records: list[ManifestRecord]) -> dict[str, str]:
    """协议 B：按年份切分。2014–2016 train / 2017 val / 2018 test。"""
    out: dict[str, str] = {}
    for r in records:
        y = int(r.labels["ice_season"])
        out[r.image_id] = "test" if y >= 2018 else ("val" if y == 2017 else "train")
    return out


def _assert_zero_cross_split(records: list[ManifestRecord], *, check_monitor: bool) -> None:
    if check_monitor:
        mon_splits: dict[str, set[str]] = defaultdict(set)
        for r in records:
            mon_splits[r.camera_id].add(r.split)
        bad_mon = [m for m, sps in mon_splits.items() if len(sps) > 1]
        if bad_mon:
            raise AssertionError(f"发现跨 split 终端: {bad_mon[:5]}")

        nd_splits: dict[str, set[str]] = defaultdict(set)
        sha_splits: dict[str, set[str]] = defaultdict(set)
        for r in records:
            nd = str(r.labels.get("near_dup_group", ""))
            if nd:
                nd_splits[nd].add(r.split)
            if r.sha256:
                sha_splits[r.sha256].add(r.split)
        bad_nd = [k for k, v in nd_splits.items() if len(v) > 1]
        bad_sha = [k for k, v in sha_splits.items() if len(v) > 1]
        if bad_nd or bad_sha:
            raise AssertionError(f"发现跨 split 近重复/精确重复: nd={bad_nd[:5]} sha={bad_sha[:5]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 SCUT 双协议纯净四分类清单与划分索引")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-sha", action="store_true", help="计算并写入每图 sha256")
    ap.add_argument("--phash-dist", type=int, default=4, help="pHash 近重复汉明距离上限")
    ap.add_argument(
        "--exclude-noise-candidates",
        action="store_true",
        help="将 18 张双无冰噪声候选从主清单排除（默认保留并打标 noise_candidate=True）",
    )
    args = ap.parse_args(argv)

    df, filter_stats = load_csv(
        decode_check=True,
        with_sha=args.with_sha,
        phash_dist=args.phash_dist,
        exclude_noise_candidates=args.exclude_noise_candidates,
    )
    print(
        f"[split] 有效样本 {filter_stats['valid_records']} "
        f"(剔除缺失 {filter_stats['missing_excluded']} / "
        f"损坏 {filter_stats['corrupt_excluded']} / "
        f"噪声排除 {filter_stats['noise_candidates_excluded']})"
    )
    records = to_records(df)

    mon_assign = split_monitor(records, seed=args.seed)
    tmp_assign = split_temporal(records)

    out_dir = PROJ / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir = PROJ / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for tag, assign in (("monitor", mon_assign), ("temporal", tmp_assign)):
        if tag == "monitor":
            recs = [r.model_copy(update={"split": assign[r.camera_id]}) for r in records]
        else:
            recs = [r.model_copy(update={"split": assign[r.image_id]}) for r in records]
        assert_no_leakage(recs)
        _assert_zero_cross_split(recs, check_monitor=(tag == "monitor"))

        s = summarize(recs)
        by_split_type: dict[str, Counter[str]] = defaultdict(Counter)
        for r in recs:
            by_split_type[r.split][r.labels["ice_type"]] += 1
        s["ice_type_by_split"] = {k: dict(v) for k, v in by_split_type.items()}

        write_manifest(recs, man_dir / f"scut_v2_{tag}.jsonl")
        write_manifest(recs, out_dir / f"scut_{tag}_v2.jsonl")
        if tag == "monitor":
            write_manifest(recs, man_dir / "scut_ice_type_v1.jsonl")
        else:
            write_manifest(recs, man_dir / "scut_ice_type_temporal_v1.jsonl")

        results[tag] = s
        print(
            f"[split] 协议 {'A' if tag == 'monitor' else 'B'} ({tag}): "
            f"{s['by_split']}  终端组 {s['n_groups']}"
        )
        for sp in ("train", "val", "test"):
            print(f"        {sp:5s} {dict(by_split_type[sp])}")

    cam_split = {r.camera_id: mon_assign[r.camera_id] for r in records}
    print(f"[split] 协议 A 终端数 {len(cam_split)}（互斥，无跨 split 终端/近重复簇）")

    out = PROJ / "artifacts" / "reports" / "scut_splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "with_sha": args.with_sha,
                "phash_dist": args.phash_dist,
                "exclude_noise_candidates": args.exclude_noise_candidates,
                "filter_stats": filter_stats,
                "protocols": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[split] 写出 {man_dir / 'scut_ice_type_v1.jsonl'} 与 {out_dir / 'scut_*_v2.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
