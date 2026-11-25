"""静态门禁：`src/`、`configs/`、`scripts/` 内禁止出现宿主盘符字面量。

背景（计划 §4.3 铁律 1）：本仓曾因 `.venv` 内 editable `.pth` 残留旧盘符
（`D:\\NWKYYSJZX\\2026年工作\\...`）导致换盘后解释器启动直接失败。路径层
`common/paths.py` 落地后，源码与配置侧也必须给出同等强度的静态防线。

允许的例外（均为合法 URL 或注释说明，不是本地文件路径）：
- `https://`、`http://`、`file://` 等协议前缀；
- 被引号包围且属于正则/文档示例的 `D:\\...` 说明文字 —— 本测试选择
  **不做广泛放行**，只放行 URL，其余一律要求改写成 `paths.py` 调用。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("src", "configs", "scripts")
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".ps1"}

# Windows 盘符：D:\ 或 D:/，且后面不是另一个字母（排除 "A/B" 之类误报）
_DISK = re.compile(r"\b[A-Za-z]:[\\/]")
# 合法 URL 前缀（含协议）——匹配到则忽略该行剩余部分
_URL = re.compile(r"(?:https?|ftp|file|ssh|git)://\S*")


def _iter_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        base = PROJ / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUFFIXES:
                out.append(p)
    return out


def _strip_urls(line: str) -> str:
    return _URL.sub("", line)


@pytest.mark.parametrize("path", _iter_files(), ids=lambda p: str(p.relative_to(PROJ)))
def test_no_host_drive_literal(path: Path) -> None:
    """任何含 `X:\\` 形式的行都视为宿主路径泄漏（URL 除外）。"""
    offenders: list[str] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = _strip_urls(raw)
        if _DISK.search(line):
            offenders.append(f"{path.relative_to(PROJ)}:{i}: {raw.strip()}")
    assert not offenders, (
        "禁止在源码/配置/脚本中写宿主盘符路径，请改用 common/paths.py：\n" + "\n".join(offenders)
    )


def test_scan_covers_expected_files() -> None:
    """防空转：扫描集合必须非空，否则门禁形同虚设。"""
    files = _iter_files()
    assert len(files) >= 30, f"扫描文件数异常偏少: {len(files)}"
    assert any(p.suffix == ".py" for p in files)
    assert any("configs" in p.parts for p in files)


def test_paths_module_is_the_only_place_resolving_proj() -> None:
    """`PROJ` 只允许由 `common/paths.py` 定义，避免多处各自反推项目根。"""
    offenders: list[str] = []
    for p in (PROJ / "src").rglob("*.py"):
        if p.name == "paths.py":
            continue
        text = p.read_text(encoding="utf-8")
        if re.search(r"^PROJ\s*[:=]", text, re.MULTILINE):
            offenders.append(str(p.relative_to(PROJ)))
    assert not offenders, f"PROJ 应在 common/paths.py 统一定义，发现重复定义: {offenders}"
