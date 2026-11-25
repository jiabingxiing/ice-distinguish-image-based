"""同终端基准匹配单测（计划 §3.3 BASE、§4.2）。

覆盖：IoU 计算、候选筛选、ECC 配准成功/失败路径、fail-closed 行为、
索引加载与格式校验。
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from grid_ice_vision.application.baseline_matcher import (
    BaselineConfig,
    BaselineEntry,
    bbox_iou,
    load_index,
    match_baseline,
    register_ecc,
)


def textured(
    h: int = 96, w: int = 96, seed: int = 0, shift: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """生成低纹理但可配准的监控风格图：渐变 + 少量条纹（模拟杆塔/绝缘子）。"""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 120, dtype=np.uint8)
    yy = np.linspace(0, 60, h, dtype=np.float32)[:, None]
    base[:, :, 0] = np.clip(base[:, :, 0].astype(np.float32) + yy, 0, 255).astype(np.uint8)
    for _ in range(6):
        x = int(rng.integers(8, w - 8))
        base[:, x : x + 3] = 30
    for _ in range(4):
        y = int(rng.integers(8, h - 8))
        base[y : y + 2, :] = 200
    if shift != (0, 0):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        base = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return base


# ---------------------------------------------------------------- IoU


def test_iou_identical() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint() -> None:
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_half_overlap() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_degenerate_box() -> None:
    assert bbox_iou((0, 0, 0, 0), (0, 0, 10, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------- ECC


def test_register_ecc_identity() -> None:
    img = textured()
    got = register_ecc(img, img)
    assert got is not None
    corr, warp = got
    assert corr > 0.95
    assert warp.shape == (2, 3)


def test_register_ecc_recovers_translation() -> None:
    """平移 3 px 后配准，估计的平移量应接近真实值。"""
    base = textured(shift=(0, 0))
    moved = textured(shift=(3, 2))
    got = register_ecc(moved, base)
    assert got is not None
    corr, warp = got
    assert corr > 0.80
    assert abs(float(warp[0, 2])) < 8.0


def test_register_ecc_tiny_image_returns_none() -> None:
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    assert register_ecc(tiny, tiny) is None


def test_register_ecc_empty_returns_none() -> None:
    assert register_ecc(np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10, 3), np.uint8)) is None


# ---------------------------------------------------------------- 匹配


def make_entry(
    monitor: str = "M1", bbox: tuple[float, float, float, float] = (8, 8, 88, 88)
) -> BaselineEntry:
    return BaselineEntry(monitor_id=monitor, image_ref="baseline/x.jpg", bbox=bbox)


def test_no_candidate_returns_reason() -> None:
    img = textured()
    r = match_baseline(img, (8, 8, 88, 88), "M1", [], lambda e: img)
    assert r.entry is None
    assert r.reason == "no_candidate"


def test_other_monitor_not_used() -> None:
    """基准必须同 monitorId —— 跨终端背景不同，绝不能借用。"""
    img = textured()
    r = match_baseline(img, (8, 8, 88, 88), "M1", [make_entry("M2")], lambda e: img)
    assert r.entry is None
    assert r.reason == "no_candidate"


def test_dirty_baseline_excluded() -> None:
    img = textured()
    dirty = BaselineEntry("M1", "x.jpg", (8, 8, 88, 88), is_clean=False)
    r = match_baseline(img, (8, 8, 88, 88), "M1", [dirty], lambda e: img)
    assert r.entry is None
    assert r.reason == "no_candidate"


def test_iou_below_threshold_rejected() -> None:
    img = textured()
    far = make_entry(bbox=(2, 2, 20, 20))
    r = match_baseline(img, (60, 60, 90, 90), "M1", [far], lambda e: img)
    assert r.entry is None
    assert r.reason == "iou_below_threshold"


def test_successful_match_returns_entry_and_bbox() -> None:
    img = textured()
    e = make_entry()
    r = match_baseline(img, (8, 8, 88, 88), "M1", [e], lambda e: img)
    assert r.entry is e
    assert r.reason == "matched"
    assert r.registered_bbox is not None
    assert r.ecc_corr is not None and r.ecc_corr > 0.5


def test_unloadable_candidate_is_skipped() -> None:
    img = textured()
    e = make_entry()
    r = match_baseline(img, (8, 8, 88, 88), "M1", [e], lambda e: None)
    assert r.entry is None
    assert r.reason == "registration_failed"


def test_low_ecc_corr_rejected_by_default() -> None:
    """相关过低 = 机位已变，默认宁可不给 Pa。"""
    img = textured(seed=0)
    other = np.random.default_rng(7).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    r = match_baseline(
        img,
        (8, 8, 88, 88),
        "M1",
        [make_entry()],
        lambda e: other,
        BaselineConfig(ecc_corr_min=0.99),
    )
    assert r.entry is None
    assert r.reason in ("ecc_corr_below_threshold", "registration_failed")


def test_allow_unregistered_opt_in() -> None:
    """显式开启 allow_unregistered 才允许不配准直接用。"""
    img = textured()
    other = np.random.default_rng(9).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    cfg = BaselineConfig(ecc_corr_min=0.99, allow_unregistered=True)
    r = match_baseline(img, (8, 8, 88, 88), "M1", [make_entry()], lambda e: other, cfg)
    assert r.entry is not None
    assert r.reason == "unregistered_allowed"


def test_picks_highest_iou_candidate() -> None:
    img = textured()
    good = make_entry(bbox=(8, 8, 88, 88))
    worse = make_entry(bbox=(12, 12, 84, 84))
    r = match_baseline(img, (8, 8, 88, 88), "M1", [worse, good], lambda e: img)
    assert r.entry is good


# ---------------------------------------------------------------- 配置


def test_config_defaults() -> None:
    c = BaselineConfig.from_config(None)
    assert c.iou_min == 0.60
    assert c.ecc_corr_min == 0.60
    assert c.allow_unregistered is False


def test_config_from_yaml_section() -> None:
    c = BaselineConfig.from_config({"baseline": {"iou_min": 0.7, "max_candidates": 1}})
    assert c.iou_min == 0.7
    assert c.max_candidates == 1


# ---------------------------------------------------------------- 索引


def test_load_index_groups_by_monitor(tmp_path) -> None:
    p = tmp_path / "index.jsonl"
    rows = [
        {"monitor_id": "M1", "image_ref": "a.jpg", "bbox": [0, 0, 10, 10]},
        {"monitor_id": "M1", "image_ref": "b.jpg", "bbox": [0, 0, 10, 10]},
        {"monitor_id": "M2", "image_ref": "c.jpg", "bbox": [1, 1, 11, 11]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    idx = load_index(p)
    assert set(idx) == {"M1", "M2"}
    assert len(idx["M1"]) == 2
    assert idx["M2"][0].bbox == (1.0, 1.0, 11.0, 11.0)


def test_load_index_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "absent.jsonl")


def test_load_index_bad_bbox_raises(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps({"monitor_id": "M1", "image_ref": "a.jpg", "bbox": [0, 0, 1]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="解析失败"):
        load_index(p)
