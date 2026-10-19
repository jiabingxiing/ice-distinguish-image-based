"""数据集校验（CI 硬门槛，§11.3）。

检查项：
1. 同一 group 不跨 split（防泄漏）
2. 测试集不含合成数据
3. sha256 与文件一致
4. 类别分布 / 场景分布报告

任一硬项失败 → 退出码 1，CI 直接 fail。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grid_ice_vision.data.manifest import (
    assert_no_leakage,
    assert_sha_consistency,
    load_manifest,
    summarize,
)
from grid_ice_vision.data.splits import split_stats


def main() -> int:
    ap = argparse.ArgumentParser(description="manifest 泄漏与完整性校验")
    ap.add_argument("--manifest", action="append", required=True, help="可多次指定")
    ap.add_argument("--root", default="data/processed", help="用于校验 sha256")
    ap.add_argument("--check-sha", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    records = []
    for m in args.manifest:
        records.extend(load_manifest(m))

    report: dict[str, object] = {"n": len(records)}
    ok = True

    try:
        assert_no_leakage(records)
        report["leakage"] = "ok"
    except AssertionError as exc:
        report["leakage"] = str(exc)
        ok = False

    if args.check_sha:
        bad = assert_sha_consistency(records, args.root)
        report["sha_issues"] = bad
        ok = ok and not bad

    report["summary"] = summarize(records)
    report["by_split"] = split_stats(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    if not ok:
        print("✗ 校验失败", file=sys.stderr)
        return 1
    print("✓ 校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
