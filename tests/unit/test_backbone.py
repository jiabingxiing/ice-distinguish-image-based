"""骨干加载策略单测（A1）。

核心断言：`pretrained=True` 失败时**必须报错**，不能静默退化成随机 MiniResNet。
"""

from __future__ import annotations

import os

import pytest
import timm

from grid_ice_vision.common.config import DEFAULT_HF_ENDPOINT, configure_hf_mirror
from grid_ice_vision.features.backbone import Backbone, build_backbone


def _raise_offline(*args: object, **kwargs: object) -> None:
    raise OSError("offline: cannot reach huggingface.co")


def test_pretrained_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timm, "create_model", _raise_offline)
    with pytest.raises(RuntimeError, match="HF_ENDPOINT"):
        Backbone("mobilenetv4_conv_small", pretrained=True)


def test_pretrained_failure_message_has_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timm, "create_model", _raise_offline)
    with pytest.raises(RuntimeError, match="OSError"):
        Backbone("mobilenetv4_conv_small", pretrained=True)


def test_fallback_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timm, "create_model", _raise_offline)
    bb = Backbone("mobilenetv4_conv_small", pretrained=True, allow_fallback=True)
    assert bb.spec.is_fallback is True
    assert bb.spec.pretrained is False
    assert "OSError" in bb.spec.fallback_reason


def test_pretrained_false_still_allows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """随机初始化时降级是合理的（不涉及权重），但必须留下原因。"""
    monkeypatch.setattr(timm, "create_model", _raise_offline)
    bb = build_backbone(pretrained=False)
    assert bb.spec.is_fallback is True
    assert bb.spec.fallback_reason


def test_spec_records_pretrained_flag() -> None:
    bb = build_backbone(pretrained=False)
    assert bb.spec.pretrained is False
    assert bb.spec.is_fallback is False
    assert bb.spec.name == "mobilenetv4_conv_small"


def test_configure_hf_mirror_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HF_ENDPOINT", "HF_HOME", "HF_HUB_DISABLE_SYMLINKS"):
        monkeypatch.delenv(k, raising=False)
    assert configure_hf_mirror() == DEFAULT_HF_ENDPOINT
    assert os.environ["HF_ENDPOINT"] == DEFAULT_HF_ENDPOINT
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    # OneDrive 下的符号链接会退化成 0 字节文件，必须指到项目内缓存
    assert "hf-cache" in os.environ["HF_HOME"]


def test_configure_hf_mirror_does_not_override_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    assert configure_hf_mirror() == "https://example.invalid"
