"""WEAPD（Weather Phenomenon Database）→ 项目天气 manifest。

数据集：Xiao Haixia, *Weather phenomenon database (WEAPD)*, Harvard Dataverse,
DOI `10.7910/DVN/M8JQCR`, **CC0-1.0**（2026-10-20 核实）。

三件事必须做对，否则指标会虚高或口径失真：

1. **11 类 → 8 类映射**，`dew/lightning/rainbow/sandstorm` 整类丢弃
   （不是留作全零负样本 —— 那等于教模型"彩虹 = 无降水"）；
2. **`clear` 没有正样本**，用 `weather_mask` 屏蔽该维，报告只算 7 类；
3. **按近重复簇分组**：WEAPD 是网络抓图，存在同图缩放/裁剪/水印版本。
   帧级随机划分会让 val/test 里混入训练图的近重复，指标虚高。

分组键写入 `weather_event_id = f"g{cluster:05d}"`，沿用现有三元组，
不改 `ManifestRecord` 结构，也不改 `group_split` 逻辑。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common.constants import WEATHER_LABELS
from grid_ice_vision.data.manifest import (
    ManifestRecord,
    assert_no_leakage,
    summarize,
    write_manifest,
)
from grid_ice_vision.data.splits import SplitRatio, group_split, split_stats

# WEAPD 11 类 → 项目 8 类；None = 整类丢弃
SOURCE_MAP: dict[str, str | None] = {
    "clear": "clear",  # WEAPD 无此类，保留以便将来扩展
    "fogsmog": "fog",
    "fog": "fog",
    "frost": "frost",
    "glaze": "glaze_evidence",
    "hail": "hail",
    "rain": "rain",
    "rime": "rime",
    "snow": "snow",
    # 丢弃：非降水相态或与本项目无关
    "dew": None,
    "lightning": None,
    "rainbow": None,
    "sandstorm": None,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def phash64(gray: np.ndarray, size: int = 8) -> np.ndarray:
    """64-bit 感知哈希：DCT 低频符号位。"""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:size, :size].flatten()
    med = float(np.median(dct[1:]))
    return np.packbits(dct > med)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def cluster_near_duplicates(
    hashes: np.ndarray, max_dist: int = 6, progress_every: int = 1000
) -> tuple[np.ndarray, dict[str, int]]:
    """汉明距离 ≤ max_dist 的图像并为一簇（跨类别合并）。"""
    n = len(hashes)
    uf = UnionFind(n)
    bits = np.unpackbits(hashes, axis=1).astype(np.int16)  # (N, 64)
    merged = 0
    for i in range(n):
        # 逐元素 XOR 后按位求和 = 汉明距离
        dist = np.abs(bits[i][None, :] - bits[i + 1 :]).sum(axis=1)
        hits = np.flatnonzero(dist <= max_dist) + i + 1
        for j in hits:
            merged += int(uf.union(i, int(j)))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  pHash 聚类 {i + 1}/{n}，已合并 {merged} 次", flush=True)
    roots = np.array([uf.find(i) for i in range(n)])
    _, compact = np.unique(roots, return_inverse=True)
    sizes = np.bincount(compact)
    stats = {
        "n_images": n,
        "n_clusters": int(sizes.size),
        "largest_cluster": int(sizes.max()) if sizes.size else 0,
        "singletons": int((sizes == 1).sum()),
    }
    return compact, stats


def build(
    src: Path,
    out_manifest: Path,
    data_root_rel: str = "",
    phash_dist: int = 6,
    seed: int = 42,
    val: float = 0.15,
    test: float = 0.15,
    limit: int | None = None,
) -> tuple[list[ManifestRecord], dict[str, object]]:
    entries: list[tuple[Path, str]] = []
    dropped: dict[str, int] = {}
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        mapped = SOURCE_MAP.get(class_dir.name.lower(), "?")
        if mapped is None:
            n = len([p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXT])
            dropped[class_dir.name] = n
            continue
        if mapped == "?":
            print(f"  ⚠️ 未知类别目录，跳过：{class_dir.name}")
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                entries.append((f, mapped))

    if limit:
        entries = entries[:limit]
    if not entries:
        raise SystemExit(f"在 {src} 下没有找到可用图像，检查目录结构是否为 <class>/*.jpg")

    print(f"可用图像 {len(entries)} 张；丢弃类别 {dropped}")

    hashes = np.zeros((len(entries), 8), dtype=np.uint8)
    shas: list[str] = []
    grays: list[np.ndarray] = []
    valid: list[tuple[Path, str]] = []
    for i, (f, label) in enumerate(entries):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hashes[len(valid)] = phash64(gray)
        grays.append(gray)
        shas.append(sha256_file(f))
        valid.append((f, label))
        if (i + 1) % 500 == 0:
            print(f"  读取 {i + 1}/{len(entries)}", flush=True)

    hashes = hashes[: len(valid)]
    clusters, cluster_stats = cluster_near_duplicates(hashes, phash_dist)
    print(f"  近重复聚类：{cluster_stats}")

    # clear 无正样本 → 该维置 0；其余 7 维为 1
    mask = [0.0 if name == "clear" else 1.0 for name in WEATHER_LABELS]

    records: list[ManifestRecord] = []
    for i, (f, label) in enumerate(valid):
        rel = f.relative_to(src).as_posix()
        h, w = grays[i].shape[:2]
        records.append(
            ManifestRecord(
                image_id=f"{label}_{i:06d}",
                path=f"{data_root_rel}/{rel}" if data_root_rel else rel,
                sha256=shas[i],
                site_id="WEAPD",
                camera_id="src",
                # 分组键 = 近重复簇（跨类别）
                weather_event_id=f"g{int(clusters[i]):05d}",
                timestamp="20211110T000000",
                split="train",
                is_synthetic=False,
                scene_group="adverse" if label not in ("clear",) else "clear",
                labels={
                    "weather_visual": [label],
                    "weather_mask": mask,
                    "intensity": "unknown",  # WEAPD 无强度标注
                    "source_dataset": "WEAPD",
                    "source_license": "CC0-1.0",
                },
                width=int(w),
                height=int(h),
            )
        )

    split_records = group_split(
        records, SplitRatio(train=1.0 - val - test, val=val, test=test), seed=seed
    )
    assert_no_leakage(split_records)
    write_manifest(split_records, out_manifest)

    report: dict[str, object] = {
        "n": len(split_records),
        "dropped_classes": dropped,
        "cluster": cluster_stats,
        "by_split": split_stats(split_records),
        "summary": summarize(split_records),
    }
    return split_records, report


def main() -> int:
    ap = argparse.ArgumentParser(description="WEAPD → 项目天气 manifest（CC0-1.0）")
    ap.add_argument("--src", required=True, help="解压后的 WEAPD 根目录（内含 11 个类目录）")
    ap.add_argument("--out", default="data/manifests/weather_v1.jsonl")
    ap.add_argument("--data-root-rel", default="", help="manifest path 的前缀；留空则相对 src")
    ap.add_argument("--phash-dist", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 张（调试用）")
    args = ap.parse_args()

    records, report = build(
        Path(args.src),
        Path(args.out),
        args.data_root_rel,
        args.phash_dist,
        args.seed,
        args.val,
        args.test,
        args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写出 {args.out}（{len(records)} 条）")
    print("⚠️ 域外提醒：WEAPD 为网络图片，frost/glaze/rime 多为植物特写；")
    print("   其指标只证明「管线接线正确、7 类相对可分」，不得写入验收材料。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
