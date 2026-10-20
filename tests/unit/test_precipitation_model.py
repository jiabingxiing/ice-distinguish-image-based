"""气象编码器可用性标志单测（A5）。

核心断言：没有气象输入时，不能伪造 (0℃, 60%RH, 5m/s) 且声称"有传感器"。
"""

from __future__ import annotations

import pytest
import torch

from grid_ice_vision.features.precipitation.model import MetEncoder, PrecipitationNet


def _capture_input(enc: MetEncoder, met: torch.Tensor | None) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def _hook(module: object, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        # 注意：hook 的返回值会替换模块输出，必须返回 None
        captured["x"] = inputs[0].detach()

    handle = enc.net[0].register_forward_hook(_hook)
    try:
        enc(met)
    finally:
        handle.remove()
    return captured["x"]


def test_met_absent_sets_avail_zero() -> None:
    enc = MetEncoder()
    x = _capture_input(enc, None)
    assert x.shape == (1, 5)
    assert float(x[0, -1]) == 0.0
    assert float(x[0, :-1].abs().sum()) == 0.0


def test_met_present_sets_avail_one() -> None:
    enc = MetEncoder()
    x = _capture_input(enc, torch.zeros(2, 4))
    assert x.shape == (2, 5)
    assert float(x[:, -1].min()) == 1.0


def test_met_none_differs_from_zero_reading() -> None:
    """缺失 ≠ 读数为零：两者必须产生不同嵌入。"""
    enc = MetEncoder()
    assert not torch.allclose(enc(None), enc(torch.zeros(1, 4)))


def test_met_1d_input_is_promoted() -> None:
    enc = MetEncoder()
    assert enc(torch.zeros(4)).shape == (1, 32)


def test_precipitation_net_accepts_missing_met() -> None:
    torch.manual_seed(0)
    net = PrecipitationNet(pretrained=False).eval()
    out = net(torch.zeros(2, 3, 64, 64), None)
    assert out["weather_logits"].shape == (2, 8)
    assert out["intensity_logits"].shape == (2, 3)


def test_precipitation_net_with_met_batch() -> None:
    torch.manual_seed(0)
    net = PrecipitationNet(pretrained=False).eval()
    met = torch.tensor([[-5.0, 95.0, 8.0, 1010.0], [2.0, 80.0, 3.0, 1005.0]])
    out = net(torch.zeros(2, 3, 64, 64), met)
    assert torch.isfinite(out["weather_logits"]).all()


@pytest.mark.parametrize("use_film", [True, False])
def test_film_toggle(use_film: bool) -> None:
    net = PrecipitationNet(pretrained=False, use_film=use_film).eval()
    out = net(torch.zeros(1, 3, 64, 64))
    assert out["feat"].shape[0] == 1
