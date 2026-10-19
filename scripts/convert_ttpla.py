"""TTPLA → COCO 转换（§11.1）。

TTPLA 是**航拍俯视**数据，与杆塔侧视差异大，只能用于 ROI 预训练，
**不得**把它的 mAP 当作本项目指标（§14 P2 关键难点）。

TTPLA 标注格式：每张图一个 JSON，`objects[]` 内含 `category` 与 `segmentation`（多边形点串）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CATEGORY_MAP = {
    "conductor": "conductor",
    "cable": "conductor",
    "insulator": "insulator_string",
    "tower_lattice": "tower",
    "tower_tube": "tower",
    "tower_wood": "tower",
    "tower": "tower",
}
CATEGORIES = ("conductor", "insulator_string", "tower", "reference_fitting")


def _bbox_from_polygon(seg: list[float]) -> list[float]:
    xs = seg[0::2]
    ys = seg[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]


def _category(name: str) -> str | None:
    key = name.lower()
    for k, v in CATEGORY_MAP.items():
        if k in key:
            return v
    return None


def convert(src_dir: str | Path, out_file: str | Path, min_area: float = 16.0) -> dict[str, int]:
    src = Path(src_dir)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    skipped = 0
    ann_id = 1
    img_id = 1

    for jf in sorted(src.rglob("*.json")):
        try:
            data = json.loads(jf.read_text("utf-8"))
        except Exception:
            skipped += 1
            continue
        image_name = data.get("imagePath") or data.get("image") or (jf.stem + ".jpg")
        images.append(
            {
                "id": img_id,
                "file_name": Path(image_name).name,
                "width": data.get("width"),
                "height": data.get("height"),
            }
        )

        for obj in data.get("objects", []) or data.get("annotations", []):
            cat = _category(str(obj.get("category", obj.get("label", ""))))
            if cat is None:
                continue
            seg = obj.get("segmentation") or []
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                seg = seg[0]
            if not seg or len(seg) < 6:
                skipped += 1
                continue
            bbox = _bbox_from_polygon([float(v) for v in seg])
            if bbox[2] * bbox[3] < min_area:
                skipped += 1
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORIES.index(cat) + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [[float(v) for v in seg]],
                }
            )
            ann_id += 1
        img_id += 1

    coco = {
        "info": {"description": "TTPLA subset (aerial, pretrain only)", "source": str(src)},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CATEGORIES)],
    }
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, ensure_ascii=False), "utf-8")
    return {"images": len(images), "annotations": len(annotations), "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description="TTPLA → COCO")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/processed/ttpla_coco/annotations.json")
    args = ap.parse_args()
    stats = convert(args.src, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("⚠️ TTPLA 为航拍俯视数据，仅用于预训练；其 mAP 不作为本项目指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
