"""覆冰类型数据治理、解码过滤、四类映射、防泄漏与 ROI 数据集单测。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from grid_ice_vision.common.constants import (
    ICE_TYPE_FOUR_CLASSES,
    ICE_TYPES,
    RHO_BY_ICE_TYPE,
)
from grid_ice_vision.common.schemas import IceType, IcingResult
from grid_ice_vision.data.datasets.ice_type import (
    IceTypeAugment,
    IceTypeDataset,
    collate,
    load_monitor_roi_table,
)
from grid_ice_vision.data.dedup import cluster_near_duplicates, inspect_image_file
from grid_ice_vision.data.manifest import ManifestRecord


def _write_textured_jpg(path: Path, seed: int = 0, w: int = 120, h: int = 90) -> Path:
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 220, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def test_wet_snow_contract_consistency() -> None:
    assert ICE_TYPE_FOUR_CLASSES == ("glaze", "rime", "wet_snow", "mixed")
    assert "wet_snow" in ICE_TYPES
    assert RHO_BY_ICE_TYPE["wet_snow"] == 350.0
    assert IceType.WET_SNOW.value == "wet_snow"
    res = IcingResult(
        ice_type=IceType.WET_SNOW,
        class_probabilities={"glaze": 0.1, "rime": 0.1, "wet_snow": 0.7, "mixed": 0.1},
        requires_review=False,
    )
    assert res.ice_type is IceType.WET_SNOW
    assert res.class_probabilities["wet_snow"] == 0.7


def test_inspect_image_file_filters_corrupt_and_small(tmp_path: Path) -> None:
    good = _write_textured_jpg(tmp_path / "good.jpg")
    chk_good = inspect_image_file(good)
    assert chk_good.ok
    assert chk_good.width == 120 and chk_good.height == 90
    assert len(chk_good.sha256) == 64
    assert len(chk_good.phash_hex) == 16

    # 截断损坏 JPEG
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(good.read_bytes()[:25])
    assert not inspect_image_file(corrupt).ok

    # 过小图像 (<32px)
    tiny = _write_textured_jpg(tmp_path / "tiny.jpg", w=16, h=16)
    chk_tiny = inspect_image_file(tiny)
    assert not chk_tiny.ok
    assert chk_tiny.reason == "too_small"

    # 纯常数死图
    flat_path = tmp_path / "flat.png"
    ok, buf = cv2.imencode(".png", np.full((64, 64, 3), 128, dtype=np.uint8))
    assert ok
    flat_path.write_bytes(buf.tobytes())
    chk_flat = inspect_image_file(flat_path)
    assert not chk_flat.ok
    assert chk_flat.reason == "constant_image"


def test_cluster_near_duplicates_merges_exact_and_near() -> None:
    h1 = np.array([0xFF, 0x00, 0xAA, 0x55, 0x11, 0x22, 0x33, 0x44], dtype=np.uint8)
    h2 = np.array([0xFE, 0x00, 0xAA, 0x55, 0x11, 0x22, 0x33, 0x44], dtype=np.uint8)  # d=1
    h3 = np.array([0x00, 0xFF, 0x55, 0xAA, 0xEE, 0xDD, 0xCC, 0xBB], dtype=np.uint8)  # far
    clusters, stats = cluster_near_duplicates(np.stack([h1, h2, h3]), max_dist=4)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]
    assert stats["n_clusters"] == 2


def test_ice_type_dataset_dual_view_and_noise_filter(tmp_path: Path) -> None:
    p1 = _write_textured_jpg(tmp_path / "a.jpg", seed=1)
    p2 = _write_textured_jpg(tmp_path / "b.jpg", seed=2)

    roi_csv = tmp_path / "roi.csv"
    roi_csv.write_text(
        "monitor_id,x1,y1,x2,y2,asset_type,view_version,notes\n"
        "M001,0.1,0.2,0.8,0.9,insulator,v1,ok\n"
        "M999_EXAMPLE,0.1,0.1,0.9,0.9,insulator,v1,skip\n",
        encoding="utf-8",
    )
    roi_map = load_monitor_roi_table(roi_csv)
    assert "M001" in roi_map and "M999_EXAMPLE" not in roi_map

    recs = [
        ManifestRecord(
            image_id="a",
            path=p1.name,
            site_id="B1",
            camera_id="M001",
            weather_event_id="2017",
            timestamp="2017-01-01 00:00:00",
            labels={
                "ice_type": "wet_snow",
                "bureau": "B1",
                "ice_season": "2017",
                "noise_candidate": False,
            },
        ),
        ManifestRecord(
            image_id="b",
            path=p2.name,
            site_id="B1",
            camera_id="M002",
            weather_event_id="2017",
            timestamp="2017-01-01 00:00:00",
            labels={
                "ice_type": "glaze",
                "bureau": "B1",
                "ice_season": "2017",
                "noise_candidate": True,
            },
        ),
    ]
    ds = IceTypeDataset(
        recs,
        root=tmp_path,
        image_size=64,
        roi_size=64,
        view_mode="dual",
        roi_map=roi_map,
        exclude_noise_candidates=True,
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["roi_image"].shape == (3, 64, 64)
    assert int(item["label"]) == ICE_TYPE_FOUR_CLASSES.index("wet_snow")
    assert item["roi_available"] is True

    batch = collate([item])
    assert batch["image"].shape == (1, 3, 64, 64)
    assert batch["roi_image"].shape == (1, 3, 64, 64)


def test_ice_type_augment_preserves_shape() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(30, 210, (120, 160, 3), dtype=np.uint8)
    aug = IceTypeAugment(allow_hflip=False, seed=42)
    out = aug(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
