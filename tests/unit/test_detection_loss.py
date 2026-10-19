"""检测损失与解码单测（覆盖 GFL 头 + 点分配 + 分布期望）。"""

from __future__ import annotations

import torch

from grid_ice_vision.features.detection.loss import DetectionLoss, DetTargets, box_iou
from grid_ice_vision.features.detection.model import (
    DetectionNet,
    decode_level,
    distance2bbox,
    distribution_expectation,
)


def test_distribution_expectation_shapes() -> None:
    reg_max = 8
    reg3 = torch.zeros(2, 5, 4 * (reg_max + 1))
    assert distribution_expectation(reg3, reg_max).shape == (2, 5, 4)
    reg2 = torch.zeros(10, 4 * (reg_max + 1))
    assert distribution_expectation(reg2, reg_max).shape == (10, 4)


def test_distribution_expectation_uniform_mean() -> None:
    """均匀分布时期望距离 = reg_max/2。"""
    reg_max = 8
    reg = torch.zeros(1, 1, 4 * (reg_max + 1))
    out = distribution_expectation(reg, reg_max)
    assert torch.allclose(out, torch.full((1, 1, 4), reg_max / 2), atol=1e-6)


def test_distance2bbox() -> None:
    pts = torch.tensor([[10.0, 10.0]])
    dist = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    box = distance2bbox(pts, dist)
    assert torch.allclose(box, torch.tensor([[8.0, 7.0, 14.0, 15.0]]))


def test_box_iou_identical() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(box_iou(a, a)[0]) > 0.999


def test_decode_level_filters_low_scores() -> None:
    cls = torch.full((1, 4, 2), -10.0)
    reg = torch.zeros(1, 4, 4 * 9)
    points = torch.tensor([[8.0, 8.0], [24.0, 24.0], [40.0, 40.0], [56.0, 56.0]])
    boxes, scores, labels = decode_level(cls[0], reg[0], points, 8, 8, score_thr=0.5)
    assert boxes.shape[0] == 0
    assert scores.shape[0] == 0
    assert labels.shape[0] == 0


def test_detection_loss_runs_and_backprops() -> None:
    torch.manual_seed(0)
    model = DetectionNet(pretrained=False, fpn_channels=16, reg_max=4)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    outs = model(x)
    targets = [
        DetTargets(
            boxes=torch.tensor([[10.0, 10.0, 90.0, 40.0]]),
            labels=torch.tensor([0]),
        ),
        DetTargets(
            boxes=torch.tensor([[20.0, 20.0, 60.0, 60.0], [70.0, 30.0, 120.0, 50.0]]),
            labels=torch.tensor([0, 1]),
        ),
    ]
    crit = DetectionLoss(num_classes=4, reg_max=4)
    loss, metrics = crit(outs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert metrics["n_pos"] > 0
    assert set(metrics) == {"cls", "reg", "giou", "n_pos"}


def test_detection_loss_handles_empty_targets() -> None:
    torch.manual_seed(0)
    model = DetectionNet(pretrained=False, fpn_channels=16, reg_max=4)
    outs = model(torch.randn(1, 3, 128, 128))
    crit = DetectionLoss(num_classes=4, reg_max=4)
    empty = DetTargets(boxes=torch.zeros((0, 4)), labels=torch.zeros((0,), dtype=torch.long))
    loss, metrics = crit(outs, [empty])
    assert torch.isfinite(loss)
    assert metrics["n_pos"] == 0
