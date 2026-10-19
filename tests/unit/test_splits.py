"""分组划分与泄漏校验单测（§11.3、§15.1）。"""

from __future__ import annotations

import pytest

from grid_ice_vision.data.manifest import ManifestRecord, assert_no_leakage
from grid_ice_vision.data.splits import SplitRatio, group_split, split_stats


def _rec(
    i: int, site: str = "S01", cam: str = "C01", event: str = "E1", synthetic: bool = False
) -> ManifestRecord:
    return ManifestRecord(
        image_id=f"img{i}",
        path=f"images/img{i}.jpg",
        site_id=site,
        camera_id=cam,
        weather_event_id=event,
        timestamp="2026-01-15T08:30:12+08:00",
        split="train",
        is_synthetic=synthetic,
        labels={"weather_visual": ["snow"]},
    )


def test_no_leakage_passes() -> None:
    records = [_rec(1, "S01"), _rec(2, "S02"), _rec(3, "S03")]
    assert_no_leakage(records)


def test_leakage_detected() -> None:
    r1 = _rec(1, "S01", "C01", "E1")
    r2 = _rec(2, "S01", "C01", "E1").model_copy(update={"split": "test"})
    with pytest.raises(AssertionError, match="数据泄漏"):
        assert_no_leakage([r1, r2])


def test_synthetic_in_test_detected() -> None:
    r = _rec(1).model_copy(update={"split": "test", "is_synthetic": True})
    with pytest.raises(AssertionError, match="合成"):
        assert_no_leakage([r])


def test_group_split_keeps_groups_whole() -> None:
    records = []
    for g in range(10):
        for i in range(3):
            records.append(_rec(g * 10 + i, f"S{g:02d}", "C01", f"E{g}"))
    out = group_split(records, SplitRatio(0.6, 0.2, 0.2), seed=1)
    assert_no_leakage(out)
    groups = {}
    for r in out:
        groups.setdefault(r.group_key, set()).add(r.split)
    assert all(len(v) == 1 for v in groups.values())


def test_synthetic_forced_to_train() -> None:
    records = [_rec(1, "S01", "C01", "E1", synthetic=True)]
    out = group_split(records, SplitRatio(0.0, 0.0, 1.0))
    assert out[0].split == "train"


def test_holdout_sites_go_to_test() -> None:
    records = [_rec(1, "S09"), _rec(2, "S09"), _rec(3, "S01"), _rec(4, "S02")]
    out = group_split(records, holdout_sites={"S09"}, seed=0)
    assert all(r.split == "test" for r in out if r.site_id == "S09")


def test_split_ratio_validates() -> None:
    with pytest.raises(ValueError):
        SplitRatio(0.5, 0.2, 0.2)


def test_split_stats_counts() -> None:
    records = [_rec(1, "S01"), _rec(2, "S01", "C01", "E2")]
    out = group_split(records, SplitRatio(0.5, 0.5, 0.0), seed=0)
    stats = split_stats(out)
    assert sum(v["images"] for v in stats.values()) == 2


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError):
        ManifestRecord(
            image_id="x",
            path="x",
            site_id="S",
            camera_id="C",
            weather_event_id="E",
            timestamp="t",
            split="dev",
        )


def test_model_copy_bypass_is_caught_by_validator() -> None:
    """pydantic v2 的 model_copy 不跑校验 → assert_no_leakage 必须复查 split。"""
    bad = _rec(1).model_copy(update={"split": "dev"})
    with pytest.raises(AssertionError, match="非法 split"):
        assert_no_leakage([bad])
