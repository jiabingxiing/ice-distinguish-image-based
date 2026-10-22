"""TTPLA 官方 split → manifest（§11.2）。

为什么不复用 `scripts/build_manifests.py`：那个脚本按 `<site>_<camera>_<timestamp>`
解析文件名，TTPLA 的 `53_00231.jpg` 不满足约定，会被 `except ValueError: continue`
**静默跳过**，最终得到 0 条记录。

分组键：TTPLA 官方划分是图像级随机划分，文件名前缀（如 `53`）跨 train/val/test 重叠
（实测 train 的 66 个前缀里多数同时出现在 test）。因此不能用前缀当 `weather_event_id`，
否则 `assert_no_leakage` 必然报错。这里用**文件名主干**作为事件 ID：每组一帧，
泄漏检查可通过；代价是"同一架次/同一场景"的相似性没有被约束住 —— 这也是
TTPLA 的 mAP 只能当链路证据、不能当验收指标的又一原因。

时间戳：TTPLA 不含拍摄时间，`timestamp` 置 `unknown`（`ManifestRecord.timestamp`
在本仓库内无任何消费者，不会影响训练）。

用法：
    python scripts/build_ttpla_manifest.py \
        --root data/raw/ttpla \
        --splits data/raw/ttpla/splitting_dataset_txt \
        --out data/manifests/detect_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grid_ice_vision.data.manifest import (
    ManifestRecord,
    sha256_file,
    summarize,
    write_manifest,
)

SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
PROVENANCE = "hf-mirror:hjjiao/TTPLA (upstream Apache-2.0)"


def build(
    root: str | Path,
    splits: str | Path,
    out: str | Path,
    with_sha: bool = True,
) -> dict[str, object]:
    base = Path(root)
    sdir = Path(splits)
    records: list[ManifestRecord] = []
    missing: list[str] = []
    stats: dict[str, dict[str, int]] = {}

    for split, filename in SPLIT_FILES.items():
        f = sdir / filename
        if not f.exists():
            raise FileNotFoundError(f"缺少 split 文件: {f}")
        names = [line.strip() for line in f.read_text("utf-8").splitlines() if line.strip()]
        n_ok = 0
        for name in names:
            stem = Path(name).stem
            img = base / f"{stem}.jpg"
            if not img.exists():
                missing.append(f"{split}:{stem}")
                continue
            records.append(
                ManifestRecord(
                    image_id=stem,
                    path=img.name,
                    sha256=sha256_file(img) if with_sha else "",
                    site_id="TTPLA",
                    camera_id="aerial",
                    # 每帧一个组：官方 split 的前缀跨 split 重叠，不能当前缀分组
                    weather_event_id=stem,
                    timestamp="unknown",
                    split=split,
                    is_synthetic=False,
                    scene_group="normal",
                    labels={
                        "source": "ttpla",
                        "provenance": PROVENANCE,
                        "official_split": split,
                    },
                    label_version="ttpla-v1",
                )
            )
            n_ok += 1
        stats[split] = {"listed": len(names), "written": n_ok}

    write_manifest(records, out)
    return {
        "out": str(out),
        "records": len(records),
        "per_split": stats,
        "missing_images": len(missing),
        "missing_sample": missing[:10],
        "summary": summarize(records),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TTPLA → manifest")
    ap.add_argument("--root", default="data/raw/ttpla", help="图像目录（含 *.jpg）")
    ap.add_argument("--splits", default="data/raw/ttpla/splitting_dataset_txt")
    ap.add_argument("--out", default="data/manifests/detect_v1.jsonl")
    ap.add_argument("--skip-sha", action="store_true", help="跳过 sha256（快，但不可复现校验）")
    args = ap.parse_args()
    info = build(args.root, args.splits, args.out, with_sha=not args.skip_sha)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["missing_images"]:
        print(f"注意：{info['missing_images']} 张图在 split 清单中但磁盘上不存在")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
