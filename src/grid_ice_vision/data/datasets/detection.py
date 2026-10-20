"""检测数据集：manifest + COCO 标注 → (图像, boxes, labels)。

不依赖 pycocotools：COCO 只是 JSON，自己解析足够，少一个编译型依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from grid_ice_vision.common.constants import ASSET_CLASSES
from grid_ice_vision.data.manifest import ManifestRecord
from grid_ice_vision.data.transforms.common import load_image, to_chw


class DetectionDataset(Dataset[dict[str, Any]]):
    """检测数据集。

    标注缺失时**必须显式失败**：COCO 不存在时若静默返回空标注，训练会
    学成"所有点都是背景"，loss 从 0.69 平稳降到 0.02，曲线漂亮但 mAP=0。
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        root: str | Path,
        coco_file: str | Path,
        size: int = 640,
        train: bool = False,
        require_annotations: bool = True,
        min_annotated_ratio: float = 0.5,
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.size = size
        self.train = train
        self.coco_file = Path(coco_file)
        self.anns: dict[str, list[dict[str, Any]]] = {}
        self._img2id: dict[str, int] = {}
        self._id2name: dict[int, str] = {}

        if not self.coco_file.exists():
            raise FileNotFoundError(
                f"检测标注不存在: {self.coco_file}。"
                "先跑 scripts/convert_ttpla.py 生成 COCO；"
                "缺失时数据集会退化成全空标注，训练只会得到全背景模型。"
            )

        data = json.loads(self.coco_file.read_text(encoding="utf-8"))
        id2name = {c["id"]: c["name"] for c in data.get("categories", [])}
        self._id2name = id2name
        self._img2id = {im["file_name"]: im["id"] for im in data.get("images", [])}
        for a in data.get("annotations", []):
            key = str(a["image_id"])
            a["_name"] = id2name.get(a["category_id"], "conductor")
            self.anns.setdefault(key, []).append(a)

        n_ann = sum(len(v) for v in self.anns.values())
        if require_annotations and n_ann == 0:
            raise ValueError(f"{self.coco_file} 不含任何 annotation，无法训练检测")

        matched = sum(1 for r in records if self._annotations(r))
        self.n_matched = matched
        ratio = matched / max(len(records), 1)
        if require_annotations and ratio < min_annotated_ratio:
            raise ValueError(
                f"仅 {matched}/{len(records)} 张图匹配到标注（阈值 {min_annotated_ratio:.2f}）："
                "检查 COCO 的 images[].file_name 与 manifest 的 path 是否同名"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _annotations(self, r: ManifestRecord) -> list[dict[str, Any]]:
        name = Path(r.path).name
        img_id = getattr(self, "_img2id", {}).get(name)
        return self.anns.get(str(img_id), []) if img_id is not None else []

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        img = load_image(self.root / r.path)
        h, w = img.shape[:2]
        anns = self._annotations(r)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            name = a.get("_name", "conductor")
            labels.append(ASSET_CLASSES.index(name) if name in ASSET_CLASSES else 0)

        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized

        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if b.size:
            b = b * scale + np.array([left, top, left, top], dtype=np.float32)

        return {
            "image": torch.from_numpy(to_chw(canvas)),
            "boxes": torch.from_numpy(b),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": r.image_id,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "targets": [{"boxes": x["boxes"], "labels": x["labels"]} for x in batch],
        "image_id": [x["image_id"] for x in batch],
    }
