"""损失函数数值单测（§15.1）。"""

from __future__ import annotations

import pytest
import torch

from grid_ice_vision.training.losses import (
    BoundaryLoss,
    ClassBalancedCE,
    CoralLoss,
    DiceLoss,
    FocalLoss,
    SegmentationLoss,
    boundary_band,
    coral_predict,
    distribution_focal_loss,
    giou_loss,
    quality_focal_loss,
)


def test_dice_perfect_prediction() -> None:
    logits = torch.full((1, 4, 4), 10.0)
    target = torch.ones(1, 4, 4)
    assert DiceLoss()(logits, target) < 1e-3


def test_dice_worst_prediction() -> None:
    logits = torch.full((1, 4, 4), -10.0)
    target = torch.ones(1, 4, 4)
    assert DiceLoss()(logits, target) == pytest.approx(1.0, abs=1e-3)


def test_dice_fn_penalty_higher_than_fp() -> None:
    """同样大小的错误，漏检（FN）的损失应大于误检（FP）。"""
    target = torch.ones(1, 1, 10, 10)
    # FN：预测全负
    fn_logits = torch.full((1, 1, 10, 10), -3.0)
    # FP：预测一半正（覆盖了全部目标，额外多出区域）
    fp_target = torch.zeros(1, 1, 10, 10)
    fp_target[..., :5, :] = 1
    fp_logits = torch.zeros(1, 1, 10, 10)
    fn_loss = DiceLoss(fn_weight=2.0)(fn_logits, target)
    fp_loss = DiceLoss(fn_weight=2.0)(fp_logits, fp_target)
    assert fn_loss > fp_loss


def test_focal_range() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    loss = FocalLoss()(logits, target)
    assert 0.0 <= float(loss) <= 5.0


def test_boundary_band_detects_edges() -> None:
    target = torch.zeros(1, 10, 10)
    target[:, 3:7, 3:7] = 1
    band = boundary_band(target)
    assert float(band.sum()) > 0
    assert float(band[0, 5, 5]) == 0.0  # 内部不算边界


def test_boundary_loss_finite() -> None:
    logits = torch.randn(1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8)
    target[:, 2:6, 2:6] = 1
    loss = BoundaryLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_segmentation_loss_components() -> None:
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16))
    crit = SegmentationLoss()
    loss = crit(logits, target)
    assert torch.isfinite(loss)
    comp = crit.components(logits, target)
    assert set(comp) == {"dice", "focal", "boundary"}


def test_coral_monotonic_predictions() -> None:
    assert int(coral_predict(torch.full((2, 4), -10.0))[0]) == 0
    assert int(coral_predict(torch.full((2, 4), 10.0))[0]) == 4


def test_coral_loss_zero_at_perfect() -> None:
    # 全正 logits → 预测等级 4；target=4 时应接近 0
    logits = torch.full((3, 4), 10.0)
    target = torch.full((3,), 4, dtype=torch.long)
    assert float(CoralLoss(5)(logits, target)) < 1e-3


def test_coral_loss_finite() -> None:
    logits = torch.randn(4, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2, 4])
    loss = CoralLoss(5)(logits, target)
    assert torch.isfinite(loss)


def test_coral_ignores_unknown() -> None:
    """intensity=-1（unknown）不得被当成 none 监督。"""
    torch.manual_seed(0)
    logits = torch.randn(3, 4)
    full = CoralLoss(5)(logits, torch.tensor([0, -1, 3]))
    sub = CoralLoss(5)(logits[[0, 2]], torch.tensor([0, 3]))
    assert float(full) == pytest.approx(float(sub), abs=1e-6)


def test_coral_all_unknown_is_zero_and_differentiable() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    loss = CoralLoss(5)(logits, torch.tensor([-1, -1]))
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None


def test_coral_ignore_index_can_be_disabled() -> None:
    logits = torch.zeros(2, 4)
    loss = CoralLoss(5, ignore_index=None)(logits, torch.tensor([0, 4]))
    assert torch.isfinite(loss)


def test_class_balanced_weights_inverse_frequency() -> None:
    cb = ClassBalancedCE([1000, 10, 100])
    w = cb.weight
    assert float(w[1]) > float(w[2]) > float(w[0])


def test_class_balanced_invalid_counts() -> None:
    with pytest.raises(ValueError):
        ClassBalancedCE([0, 10])


def test_quality_focal_range() -> None:
    pred = torch.randn(100)
    target = torch.rand(100)
    loss = quality_focal_loss(pred, target)
    assert torch.isfinite(loss) and float(loss) >= 0


def test_distribution_focal_loss() -> None:
    reg_max = 8
    pred = torch.randn(5, 4 * (reg_max + 1), requires_grad=True)
    target = torch.rand(5, 4) * (reg_max - 1)
    loss = distribution_focal_loss(pred, target)
    assert torch.isfinite(loss)


def test_giou_identical_boxes_zero() -> None:
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(giou_loss(box, box)) == pytest.approx(0.0, abs=1e-5)


def test_giou_disjoint_positive() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    assert float(giou_loss(a, b)) > 0.9
