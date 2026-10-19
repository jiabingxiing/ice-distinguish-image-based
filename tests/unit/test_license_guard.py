"""许可守卫：生产链路不得出现 AGPL 依赖（§10.3、R7）。

规则：
1. `src/` 下任何文件都不得 `import ultralytics`；
2. 全仓库只有 `scripts/prelabel.py` 允许出现该 import；
3. 生产依赖声明里不得出现 ultralytics / yolov5 / yolov8。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ALLOWED = {(ROOT / "scripts" / "prelabel.py").resolve()}
PATTERN = re.compile(r"^\s*(?:import|from)\s+ultralytics", re.MULTILINE)


def test_no_ultralytics_in_src() -> None:
    offenders = []
    for p in SRC.rglob("*.py"):
        if PATTERN.search(p.read_text("utf-8", errors="ignore")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"生产链路出现 AGPL 依赖: {offenders}"


def test_only_prelabel_imports_ultralytics() -> None:
    offenders = []
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts or "build" in p.parts:
            continue
        if p.resolve() in ALLOWED:
            continue
        if PATTERN.search(p.read_text("utf-8", errors="ignore")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"ultralytics 只允许在 scripts/prelabel.py 中出现: {offenders}"


def test_pyproject_has_no_agpl_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text("utf-8").lower()
    for banned in ("ultralytics", "yolov5", "yolov8", "yolov11"):
        assert banned not in text, f"pyproject 声明了 AGPL 依赖: {banned}"


def test_detector_uses_own_head() -> None:
    """检测器必须是自建头，而不是包装第三方检测框架。"""
    text = (SRC / "grid_ice_vision" / "features" / "detection" / "model.py").read_text("utf-8")
    assert "class GFLHead" in text
    assert "timm" not in text.lower() or "build_backbone" in text
