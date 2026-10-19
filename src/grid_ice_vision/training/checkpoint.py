"""Checkpoint 管理：加载、去前缀、版本目录解析。

约定 `artifacts/<task>/<version>/best.pt`，权重与配置快照同目录，
方便回滚时「权重 + 配置 + manifest 哈希」一起切。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_state_dict(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    obj = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        meta = obj.get("meta", {})
    else:
        sd, meta = obj, {}
    cleaned = {k.removeprefix("module.").removeprefix("model."): v for k, v in sd.items()}
    return {"state_dict": cleaned, "meta": meta}


def load_into(
    model: nn.Module, path: str | Path, strict: bool = False, device: str = "cpu"
) -> tuple[list[str], list[str]]:
    obj = load_state_dict(path, device)
    return model.load_state_dict(obj["state_dict"], strict=strict)


def resolve_version(task_dir: str | Path, version: str | None = None) -> Path:
    """返回某个任务的权重路径；`version=None` 时取最新。"""
    base = Path(task_dir)
    if version is not None:
        return base / version / "best.pt"
    candidates = sorted([p for p in base.iterdir() if p.is_dir() and (p / "best.pt").exists()])
    if not candidates:
        raise FileNotFoundError(f"{base} 下没有可用权重")
    return candidates[-1] / "best.pt"


def write_model_card(dir_path: str | Path, card: dict[str, Any]) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / "model_card.json"
    out.write_text(json.dumps(card, ensure_ascii=False, indent=2), "utf-8")
    return out
