"""配置加载：YAML → dict（只读）+ 规范化哈希。

原则（§9.2）：features/ 内不得出现魔法数字，所有阈值经此模块注入。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_config_path(path: str | Path) -> Path:
    """解析配置路径：绝对路径原样；相对路径依次尝试 configs/ 与项目根。"""
    p = Path(path)
    if p.is_absolute():
        return p
    for cand in (CONFIG_DIR / p, PROJECT_ROOT / p, Path.cwd() / p):
        if cand.exists():
            return cand
    return CONFIG_DIR / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_config_path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置根节点必须是 mapping: {p}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并，override 优先；list 整体替换（不做逐元素合并，避免歧义）。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    """按顺序叠加多个配置文件，后者的键覆盖前者。"""
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    """规范化哈希：键排序、去空白，保证同内容同哈希。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """点号取值：`get(cfg, "alarm.debounce.escalate_frames", 3)`。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
