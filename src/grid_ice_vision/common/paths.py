"""位置无关路径解析（计划 §4.3「四条铁律」）。

目标：**项目整体复制到任意盘符与任意上级目录后仍可直接运行**，不依赖
当前工作目录（cwd）、不依赖宿主机磁盘编号、不依赖用户名。

三层解析顺序
------------
1. 项目内资源 —— 一律相对 `PROJ`（由包自身位置反推），不写盘符。
2. 真外部资源 —— 按 环境变量 → `configs/paths.local.yaml` → 项目外约定位点 解析。
   全部落空时 **显式 fail-closed 报错**，绝不静默回退旧盘符（那正是历史上
   `.venv` 里 `_editable_impl_*.pth` 残留旧盘符导致 `0xfffffffe` 启动失败的成因）。
3. 清单/配置落盘 —— 只写项目相对 POSIX 路径，写入前经 `rel_posix()` 拦截宿主绝对路径；
   读取历史绝对路径时经 `resolve_legacy()` 按锚点**先重锚、再判存**。

约定：本模块只做路径计算，不创建目录、不写文件（副作用由调用方显式执行）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Final

import yaml

# ---------------------------------------------------------------- 项目根

PROJ: Final[Path] = Path(__file__).resolve().parents[3]
"""项目根目录（`grid-ice-vision/`），由本文件位置反推，跨机跨盘均正确。"""

CONFIG_DIR: Final[Path] = PROJ / "configs"
DATA_DIR: Final[Path] = PROJ / "data"
ARTIFACTS_DIR: Final[Path] = PROJ / "artifacts"
LOCAL_PATHS_FILE: Final[str] = "paths.local.yaml"

# 清单与配置允许使用的锚点前缀（相对项目根）
ANCHORS: Final[tuple[str, ...]] = (
    "src",
    "data",
    "configs",
    "artifacts",
    "scripts",
    "tests",
    "docs",
)

_HOST_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
"""识别宿主绝对路径：盘符前缀（字母 + 冒号 + 分隔符）、UNC 共享、POSIX 根。

本行刻意不写具体盘符示例字面量，以便通过 `tests/unit/test_no_host_paths.py`
对 `src/` 的强门禁扫描（该门禁要求源码内不出现任何盘符前缀）。
"""

_UNRESERVED = re.compile(r"^[A-Za-z0-9._-]+$")

# ---------------------------------------------------------------- 外部数据键

_EXTERNAL_DEFAULTS: Final[dict[str, str]] = {
    # 华南理工整理覆冰图像库：7,296 张实测图 + 44 列气象/多标签 CSV
    "SCUT_ROOT": "NW_data/华南理工 整理覆冰图像库-from梁苇和陈思睿",
}
"""外部数据键 → 项目外约定位点（相对 `PROJ.parent`）。"""

_ENV_PREFIX: Final[str] = "GX_"


class ExternalDataMissingError(FileNotFoundError):
    """外部数据未配置或不存在。fail-closed，禁止静默回退。"""


def project_root() -> Path:
    """返回项目根。供测试注入与日志使用。"""
    return PROJ


def local_paths_file() -> Path:
    return CONFIG_DIR / LOCAL_PATHS_FILE


# ---------------------------------------------------------------- 判定工具


def is_host_absolute(path: str | Path) -> bool:
    """是否为宿主绝对路径（含盘符、UNC、POSIX 根）。"""
    return bool(_HOST_ABS.match(str(path)))


def rel_posix(path: str | Path, *, base: Path | None = None) -> str:
    """转为相对项目根的 POSIX 路径，用于 manifest 与配置文件落盘。

    - 传入宿主绝对路径且在 `base` 之下 → 相对化；不在其下 → 抛错（防止把
      外部盘符路径写进入库文件，迁移后必然失效）。
    - 已是相对路径 → 规范化为 POSIX 分隔符。
    """
    p = Path(path)
    root = (base or PROJ).resolve()
    if p.is_absolute():
        try:
            return p.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"禁止把项目外绝对路径写入清单/配置：{p}（项目根 {root}）。"
                "请在 configs/paths.local.yaml 中登记外部目录，清单内只存相对路径。"
            ) from exc
    return p.as_posix()


def resolve_legacy(path: str | Path, *, must_exist: bool = True) -> Path:
    """兼容历史清单中的绝对路径：按锚点**先重锚，再判存**。

    历史 `.jsonl` 可能存着带旧盘符与旧父目录的绝对路径。直接使用会在换盘后
    失效；本函数找到锚点段（`src`/`data`/`configs`/`artifacts`/`scripts`/
    `tests`/`docs`）后截取其后的相对部分，重新挂到当前 `PROJ` 上。

    `must_exist=False` 时只做重锚计算，不校验存在性（用于生成报告）。
    """
    raw = str(path)
    p = Path(raw)
    if not is_host_absolute(raw):
        cand = (PROJ / p).resolve() if not p.is_absolute() else p.resolve()
        if must_exist and not cand.exists():
            raise FileNotFoundError(f"相对路径在当前位置不存在：{cand}")
        return cand

    # 绝对路径：先看原路径是否仍然有效（同机未搬迁场景）
    if p.exists():
        return p

    parts = re.split(r"[\\/]+", raw)
    for i, seg in enumerate(parts):
        if seg in ANCHORS:
            rebuilt = PROJ.joinpath(*parts[i:])
            if rebuilt.exists() or not must_exist:
                return rebuilt
            raise FileNotFoundError(
                f"历史绝对路径重锚后仍不存在：{raw} → {rebuilt}。"
                "请确认数据已迁移到当前位置，或更新 configs/paths.local.yaml。"
            )
    raise FileNotFoundError(
        f"历史绝对路径不含已知锚点 {ANCHORS}，无法重锚：{raw}。"
        "请重新生成该清单（scripts/build_*_manifest.py）。"
    )


# ---------------------------------------------------------------- 外部数据


def _load_local_map() -> dict[str, Any]:
    f = local_paths_file()
    if not f.exists():
        return {}
    data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{f} 根节点必须是 mapping")
    ext = data.get("external", {})
    return ext if isinstance(ext, dict) else {}


def external(key: str, *, must_exist: bool = True) -> Path:
    """解析项目外数据目录，顺序：环境变量 → paths.local.yaml → 约定位点。

    环境变量名为 `GX_<KEY>`（例如 `GX_SCUT_ROOT`）。三处全落空时抛
    `ExternalDataMissingError`，**不静默回退**到任何写死盘符。
    """
    env = os.environ.get(f"{_ENV_PREFIX}{key}")
    if env:
        p = Path(env).expanduser()
        if p.exists() or not must_exist:
            return p.resolve() if p.exists() else p
        raise ExternalDataMissingError(f"环境变量 {_ENV_PREFIX}{key}={env} 指向的目录不存在。")

    local = _load_local_map()
    if local.get(key):
        p = Path(str(local[key])).expanduser()
        if not p.is_absolute():
            p = PROJ / p
        if p.exists() or not must_exist:
            return p.resolve() if p.exists() else p
        raise ExternalDataMissingError(
            f"{local_paths_file().name} 中 {key}={local[key]} 指向的目录不存在：{p}"
        )

    default = _EXTERNAL_DEFAULTS.get(key)
    if default:
        p = PROJ.parent / default
        if p.exists() or not must_exist:
            return p.resolve() if p.exists() else p

    raise ExternalDataMissingError(
        f"外部数据 {key} 未找到。请任选其一：\n"
        f"  1) 设置环境变量 {_ENV_PREFIX}{key}=<绝对目录>\n"
        f"  2) 复制 configs/paths.example.yaml 为 configs/{LOCAL_PATHS_FILE}，"
        f"填写 external.{key}\n"
        f"  3) 将数据放到 {PROJ.parent / (default or '<未知>')}\n"
        "（项目不提供静默回退，缺失即报错。）"
    )


def scut_root(*, must_exist: bool = True) -> Path:
    """SCUT 实测覆冰图像库根目录（含 2014..2018 子目录与 CSV）。"""
    return external("SCUT_ROOT", must_exist=must_exist)


def scut_csv(*, must_exist: bool = True) -> Path:
    """SCUT 44 列气象与多标签 CSV。"""
    p = scut_root(must_exist=must_exist) / "2014_2018_icetype_weather_data.csv"
    if must_exist and not p.exists():
        raise ExternalDataMissingError(f"SCUT CSV 不存在：{p}")
    return p


def ensure_dir(path: str | Path) -> Path:
    """创建目录并返回（副作用集中在此，便于调用方审计）。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
