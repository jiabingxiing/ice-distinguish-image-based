"""位置无关路径层单测（计划 §4.3 四条铁律）。

覆盖：`PROJ` 反推、`rel_posix` 拦截、`resolve_legacy` 重锚、`external` 三级
解析顺序与 fail-closed 行为。所有用例不依赖本机盘符，可在任意位置运行。
"""

from __future__ import annotations

import pytest

from grid_ice_vision.common import paths

# ---------------------------------------------------------------- PROJ


def test_proj_is_package_ancestor() -> None:
    """`PROJ` 必须由包位置反推，且包含 src/ 与 configs/。"""
    assert (paths.PROJ / "src" / "grid_ice_vision").is_dir()
    assert (paths.PROJ / "configs").is_dir()
    assert paths.project_root() == paths.PROJ


def test_proj_is_not_cwd_dependent(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """切换 cwd 不得改变 PROJ（历史 bug：配置按 cwd 定位）。"""
    before = paths.PROJ
    monkeypatch.chdir(tmp_path)
    assert before == paths.PROJ


# ---------------------------------------------------------------- 绝对路径判定


@pytest.mark.parametrize(
    "value",
    ["C:/x", "D:\\x", "z:/a/b", "\\\\server\\share", "/home/user"],
)
def test_is_host_absolute_true(value: str) -> None:
    assert paths.is_host_absolute(value)


@pytest.mark.parametrize(
    "value",
    ["data/processed/a.jpg", "../NW_data", "configs/paths.example.yaml", "https://x.org/a"],
)
def test_is_host_absolute_false(value: str) -> None:
    assert not paths.is_host_absolute(value)


# ---------------------------------------------------------------- rel_posix


def test_rel_posix_passthrough_relative() -> None:
    assert paths.rel_posix("data/processed/x.jpg") == "data/processed/x.jpg"


def test_rel_posix_converts_backslashes() -> None:
    assert paths.rel_posix("data\\processed\\x.jpg") == "data/processed/x.jpg"


def test_rel_posix_absolutizes_path_under_proj() -> None:
    inside = paths.PROJ / "data" / "processed" / "x.jpg"
    assert paths.rel_posix(inside) == "data/processed/x.jpg"


def test_rel_posix_rejects_outside_absolute(tmp_path) -> None:
    """项目外绝对路径禁止写入清单：迁移后必然失效，必须提前拦截。

    用 `tmp_path` 而非 POSIX 字面量：Windows 下 `Path('/tmp/x').is_absolute()`
    为 False（无盘符），用字面量会测不到真正的绝对路径分支。
    """
    outside = tmp_path / "elsewhere" / "x.jpg"
    with pytest.raises(ValueError, match="禁止把项目外绝对路径"):
        paths.rel_posix(outside)


# ---------------------------------------------------------------- resolve_legacy


def test_resolve_legacy_relative_existing() -> None:
    p = paths.resolve_legacy("configs/paths.example.yaml")
    assert p == (paths.PROJ / "configs" / "paths.example.yaml").resolve()


def test_resolve_legacy_relative_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="相对路径在当前位置不存在"):
        paths.resolve_legacy("data/processed/__definitely_absent__.jpg")


def test_resolve_legacy_reanchors_foreign_absolute() -> None:
    """旧盘符 + 旧父目录 → 按 data 锚点重锚到当前项目。"""
    fake = "/old_drive/2026_old_workspace/1019_old/grid-ice-vision/configs/paths.example.yaml"
    p = paths.resolve_legacy(fake)
    assert p == (paths.PROJ / "configs" / "paths.example.yaml").resolve()


def test_resolve_legacy_reanchor_without_existence_check() -> None:
    fake = "/old/a/data/processed/absent.jpg"
    p = paths.resolve_legacy(fake, must_exist=False)
    assert p.as_posix().endswith("data/processed/absent.jpg")
    assert str(paths.PROJ) in str(p)


def test_resolve_legacy_no_anchor_raises() -> None:
    with pytest.raises(FileNotFoundError, match="不含已知锚点"):
        paths.resolve_legacy("/old/unknown_segment/x.jpg")


# ---------------------------------------------------------------- external


def test_external_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    data = tmp_path / "scut_env"
    data.mkdir()
    monkeypatch.setenv("GX_SCUT_ROOT", str(data))
    assert paths.external("SCUT_ROOT") == data.resolve()


def test_external_env_missing_dir_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("GX_SCUT_ROOT", str(tmp_path / "nope"))
    with pytest.raises(paths.ExternalDataMissingError, match="指向的目录不存在"):
        paths.external("SCUT_ROOT")


def test_external_unknown_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知键不得静默返回任何路径 —— fail-closed 是本层核心契约。"""
    monkeypatch.delenv("GX_NOT_A_REAL_KEY", raising=False)
    with pytest.raises(paths.ExternalDataMissingError, match="未找到"):
        paths.external("NOT_A_REAL_KEY")


def test_external_must_exist_false_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GX_NOT_A_REAL_KEY", raising=False)
    with pytest.raises(paths.ExternalDataMissingError):
        paths.external("NOT_A_REAL_KEY", must_exist=False)


def test_scut_helpers_are_consistent() -> None:
    """SCUT 两级入口必须指向同一根：`scut_csv()` 的父目录即 `scut_root()`。"""
    root = paths.scut_root(must_exist=False)
    csv = paths.scut_csv(must_exist=False)
    assert csv.parent == root
    assert csv.name == "2014_2018_icetype_weather_data.csv"


# ---------------------------------------------------------------- 目录副作用


def test_ensure_dir_creates(tmp_path) -> None:
    target = tmp_path / "a" / "b"
    assert paths.ensure_dir(target) == target
    assert target.is_dir()