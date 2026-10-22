"""TTPLA 标注转换单测。

回归背景：早期实现按 COCO 的 `objects[]` 解析，而 TTPLA 实际是 labelme v4.2.7
的 `shapes[]`，1,242 个文件全部被跳过（2026-10-20 实测）。
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_ttpla import convert


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), "utf-8")
    return p


def _labelme(shapes: list[dict]) -> dict:
    return {
        "version": "4.2.7",
        "imagePath": "a.jpg",
        "imageHeight": 2160,
        "imageWidth": 3840,
        "shapes": shapes,
    }


def test_labelme_shapes_mapped_and_void_dropped(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "a.json",
        _labelme(
            [
                {"label": "cable", "points": [[0, 0], [10, 0], [10, 10]], "shape_type": "polygon"},
                {
                    "label": "tower_wooden",
                    "points": [[20, 20], [40, 20], [40, 60]],
                    "shape_type": "polygon",
                },
                {"label": "void", "points": [[0, 0], [10, 0], [10, 10]], "shape_type": "polygon"},
                {"label": "insulator", "points": [[5, 5], [9, 5], [9, 9]], "shape_type": "polygon"},
            ]
        ),
    )
    stats = convert(tmp_path, tmp_path / "out.json", min_area=1.0)
    assert stats["images"] == 1
    assert stats["annotations"] == 3
    assert stats["per_category"] == {"conductor": 1, "tower": 1, "insulator_string": 1}

    coco = json.loads((tmp_path / "out.json").read_text("utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert [names[a["category_id"]] for a in coco["annotations"]] == [
        "conductor",
        "tower",
        "insulator_string",
    ]
    assert coco["images"][0]["width"] == 3840
    assert coco["images"][0]["file_name"] == "a.jpg"


def test_rectangle_from_two_points_becomes_polygon(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "b.json",
        _labelme(
            [
                {
                    "label": "tower_tucohy",
                    "points": [[100, 50], [140, 90]],
                    "shape_type": "rectangle",
                }
            ]
        ),
    )
    convert(tmp_path, tmp_path / "out.json", min_area=1.0)
    coco = json.loads((tmp_path / "out.json").read_text("utf-8"))
    ann = coco["annotations"][0]
    assert ann["bbox"] == [100.0, 50.0, 40.0, 40.0]
    assert len(ann["segmentation"][0]) == 8  # 4 角 × 2 坐标


def test_min_area_filters_tiny_shape(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "c.json",
        _labelme([{"label": "cable", "points": [[0, 0], [1, 0], [1, 1]], "shape_type": "polygon"}]),
    )
    stats = convert(tmp_path, tmp_path / "out.json", min_area=16.0)
    assert stats["annotations"] == 0
    assert stats["skipped"] == 1


def test_coco_objects_format_still_supported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.json",
        {
            "image": "d.jpg",
            "width": 100,
            "height": 100,
            "objects": [
                {"category": "conductor", "segmentation": [[0, 0, 10, 0, 10, 10]]},
            ],
        },
    )
    stats = convert(tmp_path, tmp_path / "out.json", min_area=1.0)
    assert stats["per_category"] == {"conductor": 1}
