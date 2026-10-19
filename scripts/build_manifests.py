"""从目录构建 manifest（§11.2）。

文件名约定：`<site>_<camera>_<YYYYMMDDTHHMMSS>[_<seq>].jpg`
例：`S03_C07_20260115T083012_001.jpg`

没有 sidecar JSON 时，标签留空但**分组键必填** —— 分组键是防泄漏的前提。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grid_ice_vision.data.manifest import ManifestRecord, sha256_file, summarize, write_manifest

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# 附属文件不是样本：掩码、证据图、概率图
EXCLUDE_SUFFIX = ("_mask", "_prob", "_caliper", "_overlay")
SCENE_GROUPS = ("clear", "adverse", "equipment", "normal")


def parse_name(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"文件名不符合约定: {stem}")
    site, camera, ts = parts[0], parts[1], parts[2]
    return {"site_id": site, "camera_id": camera, "timestamp": ts, "seq": "_".join(parts[3:])}


def build(
    root: str | Path,
    out: str | Path,
    split: str = "train",
    synthetic: bool = False,
    scene_group: str = "normal",
) -> list[ManifestRecord]:
    base = Path(root)
    records: list[ManifestRecord] = []
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        if p.stem.endswith(EXCLUDE_SUFFIX):
            continue
        try:
            meta = parse_name(p.stem)
        except ValueError:
            continue
        sidecar = p.with_suffix(".json")
        labels: dict[str, Any] = {}
        if sidecar.exists():
            labels = json.loads(sidecar.read_text("utf-8"))
        h, w = _image_size(p)
        records.append(
            ManifestRecord(
                image_id=p.stem,
                path=str(p.relative_to(base)).replace("\\", "/"),
                sha256=sha256_file(p),
                site_id=meta["site_id"],
                camera_id=meta["camera_id"],
                # 天气过程 ID 必须按「过程」聚合，不能按帧序号：
                # 同一天气过程的连续帧必须落在同一个 split，否则指标虚高。
                weather_event_id=str(
                    labels.get("weather_event_id") or meta["timestamp"][:8] or "unknown"
                ),
                timestamp=meta["timestamp"],
                split=split,
                is_synthetic=synthetic,
                scene_group=str(labels.get("scene_group", scene_group)),
                labels=labels,
                label_version=str(labels.get("label_version", "v1.0")),
                width=w,
                height=h,
            )
        )
    write_manifest(records, out)
    return records


def _image_size(p: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(p) as im:
            return int(im.height), int(im.width)
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 manifest")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scene-group", default="normal", choices=list(SCENE_GROUPS))
    args = ap.parse_args()
    recs = build(args.root, args.out, args.split, args.synthetic, args.scene_group)
    print(json.dumps(summarize(recs), ensure_ascii=False, indent=2))
    print(f"已写出 {args.out}（{len(recs)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
