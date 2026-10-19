"""训练回调：checkpoint / 早停 / 指标落盘。

只保留三个必要回调；不引入回调总线，`engine.fit` 里直接调用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn


class Callback(Protocol):
    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...


@dataclass
class ModelCheckpoint:
    out_dir: str | Path
    monitor: str = "loss"
    mode: str = "min"
    keep_last: int = 3
    _best: float | None = None
    _saved: int = 0

    def __post_init__(self) -> None:
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return
        better = self._best is None or (
            value > self._best if self.mode == "max" else value < self._best
        )
        if better:
            self._best = value
            (Path(self.out_dir) / "best.json").write_text(
                json.dumps({"epoch": epoch, "metrics": metrics}, ensure_ascii=False, indent=2),
                "utf-8",
            )


@dataclass
class JsonlLogger:
    path: str | Path

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False) + "\n")


def save_state(model: nn.Module, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, p)
    return p
