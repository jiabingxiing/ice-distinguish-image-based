"""Manifest 读写与泄漏校验（§11.2、§11.3）。

每条记录带 `sha256 + split + is_synthetic`，这是「可复现」的唯一保证。
`assert_no_leakage` 是硬门槛：违反分组规则的 manifest 不允许进入训练。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SPLITS = ("train", "val", "test")


class ManifestRecord(BaseModel):
    image_id: str
    path: str
    sha256: str = ""
    site_id: str
    camera_id: str
    weather_event_id: str
    timestamp: str
    split: str = "train"
    is_synthetic: bool = False
    scene_group: str = "normal"
    labels: dict[str, Any] = Field(default_factory=dict)
    label_version: str = "v1.0"
    annotator: str | None = None
    reviewer: str | None = None
    width: int | None = None
    height: int | None = None

    @field_validator("split")
    @classmethod
    def _check_split(cls, v: str) -> str:
        if v not in SPLITS:
            raise ValueError(f"split 必须是 {SPLITS} 之一，得到 {v}")
        return v

    @model_validator(mode="after")
    def _synthetic_flag_consistent(self) -> ManifestRecord:
        """sidecar 声明合成、顶层却标成真实 —— 这是会污染 val/test 的隐性 bug。

        `artifacts/demo_real*.jsonl` 就踩过：把合成图重标成 `is_synthetic=false`，
        但 sha256 与合成版逐条相同，用它的 test 集会得到虚高指标。
        """
        if self.labels.get("is_synthetic") is True and not self.is_synthetic:
            raise ValueError(
                f"{self.image_id}: sidecar 声明 is_synthetic=true，但顶层 is_synthetic=false"
            )
        return self

    @property
    def group_key(self) -> str:
        return f"{self.site_id}|{self.camera_id}|{self.weather_event_id}"

    @property
    def site_key(self) -> str:
        return self.site_id


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest 不存在: {p}")
    out: list[ManifestRecord] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(ManifestRecord(**json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{p}:{i} 解析失败: {exc}") from exc
    return out


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return p


def assert_no_leakage(records: list[ManifestRecord]) -> None:
    """硬校验：同一 group 不得跨 split；测试集不得含合成数据。

    注意 `model_copy(update=...)` 在 pydantic v2 下**不触发校验**，
    因此这里显式复查 split 取值，避免手工改字段绕过约束。
    """
    group_to_split: dict[str, str] = {}
    for r in records:
        if r.split not in SPLITS:
            raise AssertionError(f"非法 split 取值: {r.split}（{r.image_id}）")
        prev = group_to_split.setdefault(r.group_key, r.split)
        if prev != r.split:
            raise AssertionError(
                f"数据泄漏: group {r.group_key} 同时出现在 {prev} 与 {r.split}（{r.image_id}）"
            )

    for r in records:
        if r.split == "test" and r.is_synthetic:
            raise AssertionError(f"测试集含合成数据: {r.image_id}")


def assert_sha_consistency(records: list[ManifestRecord], root: str | Path) -> list[str]:
    """校验文件哈希；返回不一致清单（不抛异常，便于报告）。"""
    base = Path(root)
    bad: list[str] = []
    for r in records:
        if not r.sha256:
            continue
        p = base / r.path
        if not p.exists():
            bad.append(f"{r.image_id}: 文件缺失 {p}")
            continue
        if sha256_file(p) != r.sha256:
            bad.append(f"{r.image_id}: sha256 不匹配")
    return bad


def summarize(records: list[ManifestRecord]) -> dict[str, Any]:
    """类别分布 / 划分统计，用于清洗报告。"""
    by_split: dict[str, int] = {}
    by_scene: dict[str, int] = {}
    by_weather: dict[str, int] = {}
    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_scene[r.scene_group] = by_scene.get(r.scene_group, 0) + 1
        for w in r.labels.get("weather_visual", []) or []:
            by_weather[str(w)] = by_weather.get(str(w), 0) + 1
    return {
        "total": len(records),
        "by_split": by_split,
        "by_scene": by_scene,
        "by_weather": by_weather,
        "n_sites": len({r.site_id for r in records}),
        "n_groups": len({r.group_key for r in records}),
        "n_synthetic": sum(1 for r in records if r.is_synthetic),
    }
