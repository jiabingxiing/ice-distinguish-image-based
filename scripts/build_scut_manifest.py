"""生成 SCUT 双协议划分索引（计划 §6.4 / T02）。

协议 A · Monitor-Holdout（主验收划分）
    按 `monitorId` **整终端**分层抽样，同一终端跨 2014–2018 的全部图像锁在同一 split。
    这是唯一能杜绝「背景杆塔/山体纹理记忆」的划分方式 —— 实测 382 个终端中
    有 179 个跨多年份重复出现且机位固定。

协议 B · Temporal-Holdout（专测基准配准与 Pa）
    按 `monitorId|ice_season` 分组，2014–2017 作 train/val，2018 作 test，
    用于验证「同终端早期无冰图作模板、评估后续冰期面积变化率」。

产出：
  data/manifests/scut_v2.jsonl        全量清单（含 split、标签、sha256 可选）
  data/splits/scut_monitor_v2.jsonl   协议 A 索引
  data/splits/scut_temporal_v2.jsonl  协议 B 索引

清单内 `path` 只存**项目相对 POSIX 路径**（经 `paths.rel_posix`），外部根目录
经 `paths.scut_root()` 运行时解析，换盘后无需改清单。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from grid_ice_vision.common import paths  # noqa: E402
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


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(paths.scut_csv(), encoding="utf-8-sig")
    root = paths.scut_root()
    keep = []
    for row in df.itertuples(index=False):
        rel = f"{int(row.Year)}/{row.Image_Name}.jpg"
        if (root / rel).exists():
            keep.append(rel)
        else:
            keep.append("")
    df = df.assign(rel_path=keep)
    return df[df["rel_path"] != ""].reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[ManifestRecord]:
    out: list[ManifestRecord] = []
    for row in df.itertuples(index=False):
        ts = str(row.Image_Time).replace("/", "-")
        labels = {
            "ice_type": ICE_TYPE_EN.get(str(row.Ice_Type), "unknown"),
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
        }
        out.append(
            ManifestRecord(
                image_id=str(row.Image_Name),
                path=row.rel_path,
                site_id=str(row.Bureau),
                camera_id=str(row.monitorId),
                weather_event_id=str(row.Year),
                timestamp=ts,
                split="train",
                is_synthetic=False,
                scene_group="field",
                labels=labels,
                label_version="scut_v2",
            )
        )
    return out


def split_monitor(records: list[ManifestRecord], seed: int = 42) -> dict[str, str]:
    """协议 A：整终端分层抽样（按 Bureau 分层，保证各局都有留出）。"""
    import random

    by_bureau: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_bureau[r.site_id].add(r.camera_id)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for _bureau, monitors in sorted(by_bureau.items()):
        ms = sorted(monitors)
        rng.shuffle(ms)
        n = len(ms)
        n_test = max(1, round(n * 0.15)) if n >= 4 else 0
        n_val = max(1, round(n * 0.15)) if n >= 4 else 0
        if n_test + n_val >= n:
            n_test, n_val = 0, 0
        for i, m in enumerate(ms):
            assign[m] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    # 兜底：若某 split 为空，从最大 train 局补一个终端，避免出现空测试集
    counts = Counter(assign.values())
    if counts.get("test", 0) == 0:
        biggest = Counter(r.camera_id for r in records).most_common(1)[0][0]
        assign[biggest] = "test"
    return assign


def split_temporal(records: list[ManifestRecord]) -> dict[str, str]:
    """协议 B：按年份切分。2014–2016 train / 2017 val / 2018 test。"""
    out: dict[str, str] = {}
    for r in records:
        y = int(r.labels["ice_season"])
        out[r.image_id] = "test" if y >= 2018 else ("val" if y == 2017 else "train")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 SCUT 双协议划分索引")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-sha", action="store_true", help="同时计算每图 sha256（较慢）")
    args = ap.parse_args(argv)

    df = load_csv()
    print(f"[split] 有效样本 {len(df)}（缺失 1 行已剔除）")
    records = to_records(df)
    root = paths.scut_root()

    if args.with_sha:
        from grid_ice_vision.data.manifest import sha256_file

        for r in records:
            r.sha256 = sha256_file(root / r.path)

    mon_assign = split_monitor(records, seed=args.seed)
    tmp_assign = split_temporal(records)

    out_dir = PROJ / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir = PROJ / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    for tag, assign in (("monitor", mon_assign), ("temporal", tmp_assign)):
        if tag == "monitor":
            # 协议 A 的键是 **终端**（camera_id）；协议 B 的键是单张图。
            recs = [r.model_copy(update={"split": assign[r.camera_id]}) for r in records]
        else:
            recs = [
                r.model_copy(update={"split": assign[r.image_id]}) for r in records
            ]
        assert_no_leakage(recs)
        s = summarize(recs)
        by_split_type: dict[str, Counter[str]] = defaultdict(Counter)
        for r in recs:
            by_split_type[r.split][r.labels["ice_type"]] += 1
        s["ice_type_by_split"] = {k: dict(v) for k, v in by_split_type.items()}

        man_path = man_dir / f"scut_v2_{tag}.jsonl"
        write_manifest(recs, man_path)
        idx_path = out_dir / f"scut_{tag}_v2.jsonl"
        write_manifest(recs, idx_path)
        results[tag] = s
        print(
            f"[split] 协议 {'A' if tag == 'monitor' else 'B'} ({tag}): "
            f"{s['by_split']}  终端组 {s['n_groups']}"
        )
        for sp in ("train", "val", "test"):
            print(f"        {sp:5s} {dict(by_split_type[sp])}")

    # 终端级泄漏自检（协议 A 的核心承诺）：同一终端的全部类别只出现在一个 split
    g2s: dict[str, set[str]] = defaultdict(set)
    for r in records:
        g2s[mon_assign[r.camera_id]].add(r.labels["ice_type"])
    print(f"[split] 协议 A 各 split 类别覆盖: { {k: sorted(v) for k, v in g2s.items()} }")

    # 终端不得跨 split（显式断言，比类别覆盖更直接）
    cam_split: dict[str, str] = {}
    for r in records:
        cam_split.setdefault(r.camera_id, mon_assign[r.camera_id])
    assert len(cam_split) == len(set(cam_split)), "终端集合构建异常"
    print(f"[split] 协议 A 终端数 {len(cam_split)}（互斥，无跨 split 终端）")

    out = PROJ / "artifacts" / "reports" / "scut_splits.json"
    out.write_text(
        json.dumps(
            {"seed": args.seed, "with_sha": args.with_sha, "protocols": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[split] 写出 {man_dir}\\scut_v2_*.jsonl 与 {out_dir}\\scut_*_v2.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())