"""检测 mAP 口径单测。

回归背景：早期默认只有 3 个 IoU 阈值（0.50/0.75/0.95），却输出 `mAP@0.5:0.95`，
口径被算宽、报告会高估。这里锁住 10 点阈值与"无正样本的类不出现在结果里"。

注意 `PERFECT`：本仓库的 AP 用 101 点插值，且与 pycocotools 一致地在
recall=1.0 处带 0 精度哨兵，因此**完美检测的 AP 是 100/101 而不是 1.0**。
这是口径特性，不是缺陷；改它会让历史数字不可比。
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_ice_vision.evaluation.detection import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_detections,
)

PERFECT = 100 / 101
Boxes = np.ndarray


def _det(box: list[float], label: int = 0, score: float = 0.9) -> dict[str, Boxes]:
    return {
        "boxes": np.asarray([box], dtype=np.float32),
        "labels": np.asarray([label], dtype=np.int64),
        "scores": np.asarray([score], dtype=np.float32),
    }


def _gt(box: list[float], label: int = 0) -> dict[str, Boxes]:
    return {
        "boxes": np.asarray([box], dtype=np.float32),
        "labels": np.asarray([label], dtype=np.int64),
    }


def test_default_thresholds_are_coco_ten_points() -> None:
    assert len(DEFAULT_IOU_THRESHOLDS) == 10
    assert DEFAULT_IOU_THRESHOLDS[0] == 0.5
    assert DEFAULT_IOU_THRESHOLDS[-1] == 0.95


def test_identical_box_scores_perfect_at_every_threshold() -> None:
    res = evaluate_detections(
        [_det([0, 0, 10, 10])],
        [_gt([0, 0, 10, 10])],
        n_classes=2,
        class_names=("conductor", "tower"),
    )
    assert res["mAP@0.50"] == pytest.approx(PERFECT)
    assert res["mAP@0.5:0.95"] == pytest.approx(PERFECT)
    assert res["AP@0.50/conductor"] == pytest.approx(PERFECT)


def test_class_without_ground_truth_is_absent_not_zero() -> None:
    """空类不能以 0 出现，否则均值被稀释（TTPLA 的两个类恒为空）。"""
    res = evaluate_detections(
        [_det([0, 0, 10, 10])],
        [_gt([0, 0, 10, 10])],
        n_classes=2,
        class_names=("conductor", "tower"),
    )
    assert "AP@0.50/tower" not in res
    assert res["mAP@0.50"] == pytest.approx(PERFECT)


def test_iou_0_82_hits_seven_thresholds() -> None:
    """预测框高 8.2 → IoU=0.82：命中 0.50~0.80 共 7 个阈值。"""
    res = evaluate_detections([_det([0, 0, 10, 8.2])], [_gt([0, 0, 10, 10])], n_classes=1)
    assert res["mAP@0.50"] == pytest.approx(PERFECT)
    assert res["mAP@0.80"] == pytest.approx(PERFECT)
    assert res["mAP@0.85"] == pytest.approx(0.0)
    assert res["mAP@0.5:0.95"] == pytest.approx(0.7 * PERFECT, abs=1e-6)


def test_no_predictions_gives_zero_for_class_with_gt() -> None:
    res = evaluate_detections([_det([50, 50, 60, 60])], [_gt([0, 0, 10, 10])], n_classes=1)
    assert res["mAP@0.50"] == pytest.approx(0.0)
