"""回归基线测试（§15.5）。

设计：读取 `artifacts/baseline.json`，与本次跑出的指标比对；
任一指标相对基线下降 > 2 个点则 fail。基线文件不存在时 skip（不是 fail），
因为 P0–P2 阶段还没有正式训练产物。

基线本身必须版本化，不允许「悄悄更新基线」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINE = Path("artifacts/baseline.json")
TOLERANCE = 0.02  # 2 个点


def _load() -> dict[str, float]:
    if not BASELINE.exists():
        pytest.skip(f"基线不存在（{BASELINE}），跳过回归对比")
    return json.loads(BASELINE.read_text("utf-8"))


def test_baseline_file_schema() -> None:
    data = _load()
    assert "metrics" in data
    assert "model_version" in data
    assert "manifest_sha256" in data, "基线必须绑定 manifest 哈希，否则不可复现"


def test_no_metric_regression() -> None:
    data = _load()
    current_path = Path("artifacts/current_metrics.json")
    if not current_path.exists():
        pytest.skip("本次未产出 current_metrics.json，跳过对比")
    current = json.loads(current_path.read_text("utf-8"))
    regressions = []
    for k, base in data["metrics"].items():
        if k not in current:
            continue
        if current[k] < base - TOLERANCE:
            regressions.append(f"{k}: {base:.3f} → {current[k]:.3f}")
    assert not regressions, "指标退化：" + "; ".join(regressions)
