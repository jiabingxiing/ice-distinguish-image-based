"""分组划分（§11.3）。

划分键 = site_id + camera_id + weather_event_id。同一天气过程的连续帧
高度相似，帧级随机划分会让测试指标虚高 10 个点以上 —— 这是本方案
与多数候选方案最实质的差异之一。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from grid_ice_vision.data.manifest import ManifestRecord


@dataclass(frozen=True)
class SplitRatio:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"比例之和必须为 1，得到 {total}")


_DEFAULT_RATIO = SplitRatio()


def group_split(
    records: list[ManifestRecord],
    ratio: SplitRatio = _DEFAULT_RATIO,
    seed: int = 42,
    holdout_sites: set[str] | None = None,
    keep_existing_test: bool = False,
) -> list[ManifestRecord]:
    """按 group 划分，返回**新对象**（不修改入参）。

    - `holdout_sites`：整体留出的站点，其所有帧进 test（跨站点测试集）；
    - 合成数据只进 train；
    - `keep_existing_test=True` 时保留原 split（用于增量数据追加）。
    """
    holdout = holdout_sites or set()
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(r.group_key, []).append(i)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    test_groups = [k for k in keys if k.split("|")[0] in holdout]
    rest = [k for k in keys if k not in set(test_groups)]
    n_total = len(rest)
    n_val = round(n_total * ratio.val)
    n_test = round(n_total * ratio.test)

    assignment: dict[str, str] = {k: "test" for k in test_groups}
    for i, k in enumerate(rest):
        if i < n_val:
            assignment[k] = "val"
        elif i < n_val + n_test:
            assignment[k] = "test"
        else:
            assignment[k] = "train"

    out: list[ManifestRecord] = []
    for r in records:
        split = assignment[r.group_key]
        if r.is_synthetic:
            split = "train"
        elif keep_existing_test and r.split == "test":
            split = "test"
        out.append(r.model_copy(update={"split": split}))
    return out


def split_stats(records: list[ManifestRecord]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for r in records:
        s = stats.setdefault(r.split, {"images": 0, "groups": 0, "sites": 0, "synthetic": 0})
        s["images"] += 1
        s["synthetic"] += int(r.is_synthetic)
    for split in stats:
        sub = [r for r in records if r.split == split]
        stats[split]["groups"] = len({r.group_key for r in sub})
        stats[split]["sites"] = len({r.site_id for r in sub})
    return stats
