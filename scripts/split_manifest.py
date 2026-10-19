"""按配置把 manifest 做分组划分并写回（§11.3）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.data.manifest import assert_no_leakage, load_manifest, write_manifest
from grid_ice_vision.data.splits import SplitRatio, group_split, split_stats


def main() -> int:
    ap = argparse.ArgumentParser(description="按 group 划分 train/val/test")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--holdout-sites", default="", help="逗号分隔，整体留出做跨站点测试")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_manifest(args.manifest)
    holdout = {s.strip() for s in args.holdout_sites.split(",") if s.strip()}
    split_records = group_split(
        records,
        SplitRatio(train=1.0 - args.val - args.test, val=args.val, test=args.test),
        seed=args.seed,
        holdout_sites=holdout,
    )
    assert_no_leakage(split_records)

    out = args.out or str(Path(args.manifest).with_name(Path(args.manifest).stem + "_split.jsonl"))
    write_manifest(split_records, out)
    print(json.dumps(split_stats(split_records), ensure_ascii=False, indent=2))
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
