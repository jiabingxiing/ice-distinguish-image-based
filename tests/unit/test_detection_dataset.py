"""检测数据集标注校验单测（A2）。

核心断言：COCO 缺失/为空/文件名不匹配时必须**显式失败**，
否则训练会静默学成"全背景"。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from grid_ice_vision.data.datasets.detection import DetectionDataset
from grid_ice_vision.data.manifest import ManifestRecord


def _records(n: int = 2) -> list[ManifestRecord]:
    return [
        ManifestRecord(
            image_id=f"S00_C00_20260115T0830{i:02d}",
            path=f"img{i}.jpg",
            site_id="S00",
            camera_id="C00",
            weather_event_id="E1",
            timestamp="20260115T083000",
            split="train",
            labels={},
        )
        for i in range(n)
    ]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for i in range(2):
        img = np.full((64, 96, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img{i}.jpg"), img)
    return tmp_path


def _write_coco(path: Path, file_names: list[str], annotations: list[dict]) -> None:
    images = [
        {"id": i + 1, "file_name": n, "width": 96, "height": 64} for i, n in enumerate(file_names)
    ]
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "conductor"}],
            }
        ),
        "utf-8",
    )


def test_missing_coco_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="检测标注不存在"):
        DetectionDataset(_records(), root, root / "nope.json")


def test_empty_annotations_raises(root: Path) -> None:
    coco = root / "ann.json"
    _write_coco(coco, ["img0.jpg", "img1.jpg"], [])
    with pytest.raises(ValueError, match="不含任何 annotation"):
        DetectionDataset(_records(), root, coco)


def test_filename_mismatch_raises(root: Path) -> None:
    coco = root / "ann.json"
    _write_coco(
        coco,
        ["other0.jpg", "other1.jpg"],
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 10, 10], "iscrowd": 0},
        ],
    )
    with pytest.raises(ValueError, match="匹配到标注"):
        DetectionDataset(_records(), root, coco)


def test_valid_coco_builds_boxes(root: Path) -> None:
    coco = root / "ann.json"
    _write_coco(
        coco,
        ["img0.jpg", "img1.jpg"],
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 20], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 20], "iscrowd": 0},
        ],
    )
    ds = DetectionDataset(_records(), root, coco, size=96)
    item = ds[0]
    assert item["boxes"].shape == (1, 4)
    assert item["labels"].tolist() == [0]
    assert ds.n_matched == 2


def test_require_annotations_can_be_disabled(root: Path) -> None:
    """显式关闭校验时才允许空标注（例如纯背景图评估）。"""
    coco = root / "ann.json"
    _write_coco(coco, ["img0.jpg", "img1.jpg"], [])
    ds = DetectionDataset(_records(), root, coco, require_annotations=False)
    assert ds.n_matched == 0
