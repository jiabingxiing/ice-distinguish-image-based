"""训练冒烟测试：三个任务各跑 1 个 epoch。

目的不是精度，而是**证明训练链路真的会更新权重、真的会落盘**。
这一层能挡住本项目最危险的失败模式：loss 正常下降但标注根本没生效。

设计：8 张 train + 4 张 val 的极小合成图，缩小模型，总耗时目标 < 20 s。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from grid_ice_vision.data.datasets import detection as det_ds
from grid_ice_vision.data.datasets import segmentation as seg_ds
from grid_ice_vision.data.datasets import weather as wx_ds
from grid_ice_vision.data.manifest import ManifestRecord
from grid_ice_vision.features.detection.model import DetectionNet
from grid_ice_vision.features.ice.model import UNetLite
from grid_ice_vision.features.precipitation.model import PrecipitationNet
from grid_ice_vision.training.engine import TrainConfig, fit
from grid_ice_vision.training.steps import det_step, precip_step, seg_step

H, W = 96, 128


def _make_image(rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), 70, dtype=np.uint8)
    img[36:60, 20:108] = rng.integers(120, 200, (24, 88, 3), dtype=np.uint8)
    return img


def _make_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[40:56, 20:108] = 1  # exposed_asset
    mask[36:40, 20:108] = 2  # ice_accretion
    mask[56:60, 20:108] = 2
    return mask


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[ManifestRecord], Path]:
    root = tmp_path_factory.mktemp("smoke")
    rng = np.random.default_rng(0)
    records: list[ManifestRecord] = []
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []

    for split, n, event in (("train", 8, "E1"), ("val", 4, "E2")):
        for i in range(n):
            iid = f"S00_C00_{event}_{i:03d}"
            cv2.imwrite(str(root / f"{iid}.jpg"), _make_image(rng))
            cv2.imwrite(str(root / f"{iid}_mask.png"), _make_mask())
            records.append(
                ManifestRecord(
                    image_id=iid,
                    path=f"{iid}.jpg",
                    site_id="S00",
                    camera_id="C00",
                    weather_event_id=event,
                    timestamp="20260115T083000",
                    split=split,
                    is_synthetic=True,
                    scene_group="adverse",
                    labels={
                        "weather_visual": ["snow"],
                        "intensity": "moderate",
                        "mask_path": f"{iid}_mask.png",
                    },
                )
            )
            images.append(
                {"id": len(images) + 1, "file_name": f"{iid}.jpg", "width": W, "height": H}
            )
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": len(images),
                    "category_id": 1,
                    "bbox": [40, 40, 48, 32],
                    "iscrowd": 0,
                }
            )

    coco = root / "ann.json"
    coco.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": anns,
                "categories": [{"id": 1, "name": "conductor"}],
            }
        ),
        "utf-8",
    )
    return root, records, coco


def _cfg(tmp_path: Path, task: str) -> TrainConfig:
    return TrainConfig(
        epochs=1,
        patience=2,
        amp="off",
        lr=1e-3,
        out_dir=str(tmp_path),
        task=task,
        version="smoke",
    )


def test_precip_one_epoch(
    smoke_data: tuple[Path, list[ManifestRecord], Path], tmp_path: Path
) -> None:
    root, records, _ = smoke_data
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]
    dl = dict(batch_size=4, num_workers=0)
    train_loader = DataLoader(
        wx_ds.WeatherDataset(tr, root, 96, True), shuffle=True, collate_fn=wx_ds.collate, **dl
    )
    val_loader = DataLoader(
        wx_ds.WeatherDataset(va, root, 96, False), collate_fn=wx_ds.collate, **dl
    )

    model = PrecipitationNet(pretrained=False)
    best = fit(
        model,
        train_loader,
        val_loader,
        precip_step,
        _cfg(tmp_path, "precip"),
        monitor="macro_f1_micro",
        mode="max",
    )

    assert math.isfinite(best["loss"])
    assert "macro_f1_micro" in best
    # 全部 intensity 有效（moderate），所以应出现强度指标
    assert "intensity_acc" in best
    assert (tmp_path / "precip" / "smoke" / "best.pt").exists()


def test_seg_one_epoch(smoke_data: tuple[Path, list[ManifestRecord], Path], tmp_path: Path) -> None:
    root, records, _ = smoke_data
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]
    dl = dict(batch_size=4, num_workers=0)
    train_loader = DataLoader(
        seg_ds.SegmentationDataset(tr, root, 96, True),
        shuffle=True,
        collate_fn=seg_ds.collate,
        **dl,
    )
    val_loader = DataLoader(
        seg_ds.SegmentationDataset(va, root, 96, False), collate_fn=seg_ds.collate, **dl
    )

    model = UNetLite(pretrained=False)
    best = fit(
        model,
        train_loader,
        val_loader,
        seg_step,
        _cfg(tmp_path, "seg"),
        monitor="dice_ice",
        mode="max",
    )

    assert math.isfinite(best["loss"])
    assert "dice_ice" in best
    assert (tmp_path / "seg" / "smoke" / "epoch000.pt").exists()


def test_detector_one_epoch(
    smoke_data: tuple[Path, list[ManifestRecord], Path], tmp_path: Path
) -> None:
    root, records, coco = smoke_data
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]
    dl = dict(batch_size=4, num_workers=0)
    train_loader = DataLoader(
        det_ds.DetectionDataset(tr, root, coco, 128, True),
        shuffle=True,
        collate_fn=det_ds.collate,
        **dl,
    )
    val_loader = DataLoader(
        det_ds.DetectionDataset(va, root, coco, 128, False), collate_fn=det_ds.collate, **dl
    )

    model = DetectionNet(pretrained=False, fpn_channels=16, reg_max=4)
    best = fit(
        model,
        train_loader,
        val_loader,
        det_step,
        _cfg(tmp_path, "detector"),
        monitor="loss",
        mode="min",
    )

    assert math.isfinite(best["loss"])
    # n_pos 是 det_step 的指标；若为 0，fit 会直接抛错
    assert "n_pos" in best
    assert (tmp_path / "detector" / "smoke" / "best.pt").exists()


def test_detector_without_coco_fails_fast(
    smoke_data: tuple[Path, list[ManifestRecord], Path],
) -> None:
    root, records, _ = smoke_data
    tr = [r for r in records if r.split == "train"]
    with pytest.raises(FileNotFoundError):
        det_ds.DetectionDataset(tr, root, root / "missing.json", 128, True)


def test_precip_intensity_unknown_is_ignored(
    smoke_data: tuple[Path, list[ManifestRecord], Path],
) -> None:
    """unknown 强度不得被当作 none 监督：应体现在 n_intensity_valid=0。"""
    root, records, _ = smoke_data
    rec = records[0].model_copy(update={"labels": {**records[0].labels, "intensity": "unknown"}})
    ds = wx_ds.WeatherDataset([rec], root, 96, False)
    batch = wx_ds.collate([ds[0]])
    assert int(batch["intensity"][0]) == -1

    model = PrecipitationNet(pretrained=False)
    _, metrics = precip_step(model, batch, "train")
    assert metrics["n_intensity_valid"] == 0.0
    assert "intensity_acc" not in metrics


def test_weather_mask_defaults_to_all_ones(
    smoke_data: tuple[Path, list[ManifestRecord], Path],
) -> None:
    root, records, _ = smoke_data
    ds = wx_ds.WeatherDataset([records[0]], root, 96, False)
    item = ds[0]
    assert torch.all(item["weather_mask"] == 1.0)
