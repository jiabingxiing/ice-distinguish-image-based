"""图像解码校验、SHA-256 与 64-bit pHash 近重复聚类工具（计划 §6.2）。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageCheckResult:
    ok: bool
    reason: str
    width: int
    height: int
    sha256: str
    phash: np.ndarray | None
    phash_hex: str


def is_valid_bgr(img: np.ndarray | None, *, min_size: int = 32, min_std: float = 1.0) -> bool:
    """校验解码图像是否为有效 3 通道彩色非死图。"""
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return False
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False
    return float(img.std()) > min_std


def compute_phash64(gray: np.ndarray, size: int = 8) -> np.ndarray:
    """64-bit 感知哈希：32x32 DCT 低频符号位。"""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:size, :size].flatten()
    med = float(np.median(dct[1:]))
    return np.packbits(dct > med)


def phash_to_hex(bits8: np.ndarray) -> str:
    return bytes(bits8.tolist()).hex()


def inspect_image_file(
    path: Path,
    *,
    with_sha: bool = True,
    with_phash: bool = True,
    min_size: int = 32,
    min_std: float = 1.0,
) -> ImageCheckResult:
    """单次读字节流完成存在性、OpenCV 解码、尺寸/死图校验、SHA-256 与 pHash。"""
    if not path.exists():
        return ImageCheckResult(False, "missing", 0, 0, "", None, "")
    try:
        raw = path.read_bytes()
    except OSError:
        return ImageCheckResult(False, "unreadable", 0, 0, "", None, "")
    if not raw:
        return ImageCheckResult(False, "empty_file", 0, 0, "", None, "")

    sha = hashlib.sha256(raw).hexdigest() if with_sha else ""
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return ImageCheckResult(False, "decode_failed", 0, 0, sha, None, "")
    if img.ndim != 3 or img.shape[2] != 3:
        h, w = img.shape[:2]
        return ImageCheckResult(False, "not_3_channels", int(w), int(h), sha, None, "")
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return ImageCheckResult(False, "too_small", int(w), int(h), sha, None, "")
    if float(img.std()) <= min_std:
        return ImageCheckResult(False, "constant_image", int(w), int(h), sha, None, "")

    ph: np.ndarray | None = None
    ph_hex = ""
    if with_phash:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ph = compute_phash64(gray)
        ph_hex = phash_to_hex(ph)

    return ImageCheckResult(True, "ok", int(w), int(h), sha, ph, ph_hex)


class UnionFind:
    """路径压缩并查集。"""

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
    hashes: np.ndarray,
    *,
    max_dist: int = 4,
    sha256_list: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """按 SHA-256 精确相同或 64-bit pHash 汉明距离 <= max_dist 合并为近重复簇。"""
    n = len(hashes)
    if n == 0:
        return np.zeros(0, dtype=np.int64), {
            "n_images": 0,
            "n_clusters": 0,
            "largest_cluster": 0,
            "singletons": 0,
            "multi_clusters": 0,
            "near_dup_images": 0,
        }

    uf = UnionFind(n)
    if sha256_list:
        seen_sha: dict[str, int] = {}
        for i, sha in enumerate(sha256_list):
            if not sha:
                continue
            first = seen_sha.get(sha)
            if first is None:
                seen_sha[sha] = i
            else:
                uf.union(first, i)

    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)
    for i in range(n - 1):
        # XOR 后逐行求和比 int16 减法绝对值更快且内存减半
        dist = np.count_nonzero(bits[i + 1 :] ^ bits[i], axis=1)
        hits = np.flatnonzero(dist <= max_dist) + i + 1
        for j in hits:
            uf.union(i, int(j))

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, compact = np.unique(roots, return_inverse=True)
    sizes = np.bincount(compact)
    multi = int((sizes > 1).sum())
    near_dup_imgs = int(sizes[sizes > 1].sum()) if multi else 0
    stats = {
        "n_images": n,
        "n_clusters": int(sizes.size),
        "largest_cluster": int(np.max(sizes)),
        "singletons": int((sizes == 1).sum()),
        "multi_clusters": multi,
        "near_dup_images": near_dup_imgs,
    }
    return compact, stats
