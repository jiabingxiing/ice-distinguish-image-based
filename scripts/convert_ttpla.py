"""TTPLA → COCO 转换（§11.1）。

TTPLA 的标注是 **labelme v4.2.7** 格式：每张图一个 JSON，`shapes[]` 内是
`{label, points, shape_type}`；早期候选实现按 COCO 的 `objects[]` 解析，会把
1,242 个文件全部跳过（`DetectionDataset` 随后因"0 条 annotation"抛错，但发现得太晚）。
两种格式都要支持。

TTPLA 实测标签（2026-10-20，1,242 个 JSON / 11,467 个实例）：
`cable` 10,075 · `tower_lattice` 404 · `void` 396 · `tower_wooden` 333 ·
`tower_tucohy` 231 · `Cable` 28。

结论：**只有 conductor / tower 两类有正样本**，`insulator_string` 与
`reference_fitting` 在 TTPLA 里恒为空；`tower_tucohy` 是数据集内
`tower_tubular` 的拼写错误，必须映射到 tower，否则丢 231 个实例。

TTPLA 是**航拍俯视**数据，与杆塔侧视差异大，只能用于 ROI 预训练，
**不得**把它的 mAP 当作本项目指标（§14 P2 关键难点）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CATEGORY_MAP = {
    "conductor": "conductor",
    "cable": "conductor",
    "insulator": "insulator_string",
    "tower_lattice": "tower",
    "tower_tubular": "tower",
    "tower_tube": "tower",
    "tower_tucohy": "tower",  # 数据集内拼写错误（tubular）
    "tower_wooden": "tower",
    "tower_wood": "tower",
    "tower": "tower",
    "reference_fitting": "reference_fitting",
}
VOID_LABELS = frozenset({"void"})
CATEGORIES = ("conductor", "insulator_string", "tower", "reference_fitting")


def _bbox_from_polygon(seg: list[float]) -> list[float]:
    xs = seg[0::2]
    ys = seg[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]


def _polygon_from_bbox(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _category(name: str) -> str | None:
    """标签 → 项目 4 类；`void` 与未知标签返回 None（丢弃）。"""
    key = name.strip().lower()
    if key in VOID_LABELS:
        return None
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    # 兜底：子串匹配（兼容 `tower_lattice_xxx` 之类变体）
    for k, v in CATEGORY_MAP.items():
        if k in key:
            return v
    return None


def _iter_objects(data: dict[str, Any]) -> Iterator[tuple[str, list[float], str]]:
    """统一两种标注格式 → (label, 扁平点串, shape_type)。"""
    shapes = data.get("shapes")
    if isinstance(shapes, list):
        for s in shapes:
            pts = s.get("points") or []
            flat = [float(c) for p in pts for c in p]
            yield str(s.get("label", "")), flat, str(s.get("shape_type", "polygon"))
        return
    for obj in data.get("objects", []) or data.get("annotations", []):
        seg = obj.get("segmentation") or []
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            seg = seg[0]
        yield (
            str(obj.get("category", obj.get("label", ""))),
            [float(v) for v in seg],
            "polygon",
        )


def convert(src_dir: str | Path, out_file: str | Path, min_area: float = 16.0) -> dict[str, Any]:
    src = Path(src_dir)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    per_category: Counter[str] = Counter()
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
                "width": data.get("width") or data.get("imageWidth"),
                "height": data.get("height") or data.get("imageHeight"),
            }
        )

        for raw_label, seg, shape_type in _iter_objects(data):
            cat = _category(raw_label)
            if cat is None:
                continue
            if len(seg) < 6:
                # 矩形只有 2 个点：先按点算 bbox，再补成 4 角多边形
                if len(seg) >= 4:
                    bx = [min(seg[0::2]), min(seg[1::2]), 0.0, 0.0]
                    bx[2] = max(seg[0::2]) - bx[0]
                    bx[3] = max(seg[1::2]) - bx[1]
                    seg = _polygon_from_bbox([float(v) for v in bx])
                else:
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
                    "_shape_type": shape_type,
                }
            )
            per_category[cat] += 1
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
    return {
        "images": len(images),
        "annotations": len(annotations),
        "skipped": skipped,
        "per_category": dict(per_category),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TTPLA → COCO")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/processed/ttpla_coco/annotations.json")
    ap.add_argument("--min-area", type=float, default=16.0, help="最小 bbox 面积（像素²）")
    args = ap.parse_args()
    stats = convert(args.src, args.out, args.min_area)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("⚠️ TTPLA 为航拍俯视数据，仅用于预训练；其 mAP 不作为本项目指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
