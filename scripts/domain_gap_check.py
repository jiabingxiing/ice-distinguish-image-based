"""域差检查：把真实图像与合成图像一起过质量门控，看阈值是否误伤真实场景。

动机：`configs/model/quality.yaml` 的阈值是拍脑袋定的初始值。如果它在
"大片天空"或"远景低纹理"上把可用图判成 degraded/unusable，那么上线后
绝大多数杆塔机位会被自己的门控挡掉 —— 这是比模型不准更早暴露的问题。

用法：
    python scripts/domain_gap_check.py --real data/raw/ttpla --weapd data/raw/weapd/dataset \
        --synth-n 100 --n 100 --out artifacts/reports/domain_gap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grid_ice_vision.features.quality.gate import (
    _to_gray,
    assess_quality,
    block_highfreq_energy,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# 掩码/证据图不是样本（同 build_manifests.py 的排除规则）
EXCLUDE_SUFFIX = ("_mask", "_prob", "_caliper", "_overlay")


def sample_images(root: Path, n: int, seed: int = 0) -> list[Path]:
    files = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXT and not p.stem.endswith(EXCLUDE_SUFFIX)
    ]
    if not files:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(idx)]


def load_synthetic(root: Path, n: int, seed: int = 0) -> list[np.ndarray]:
    """从磁盘读已生成的合成图。

    注意：`make_synthetic_scene` 单张约 9.4 s（640 个骨架点 × 64 个角度各一次
    fillPoly），现场生成 100 张要 15 分钟，因此这里只读已有产物。
    """
    files = sample_images(root, n, seed) if root.exists() else []
    out = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def image_stats(img: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {
        "h": float(img.shape[0]),
        "w": float(img.shape[1]),
        "megapixels": round(img.shape[0] * img.shape[1] / 1e6, 2),
        "mean_gray": round(float(gray.mean()), 2),
        "std_gray": round(float(gray.std()), 2),
        "sat_mean": round(float(hsv[..., 1].mean()), 2),
    }


def summarize(
    name: str, images: list[np.ndarray], stats: list[dict[str, float]]
) -> dict[str, object]:
    status = Counter()
    reasons = Counter()
    blur: list[float] = []
    block: list[float] = []
    exposure = Counter()
    for img in images:
        q = assess_quality(img)
        status[q.status.value] += 1
        exposure[q.exposure.value] += 1
        blur.append(float(q.blur_score))
        for r in q.reasons:
            reasons[r.split("(")[0]] += 1
        energy = block_highfreq_energy(_to_gray(img))
        block.append(float(np.min(energy)))

    n = max(len(images), 1)
    return {
        "source": name,
        "n": len(images),
        "status": {k: round(v / n, 3) for k, v in status.items()},
        "exposure": {k: round(v / n, 3) for k, v in exposure.items()},
        "top_reasons": reasons.most_common(5),
        "blur_score_p50": round(float(np.percentile(blur, 50)), 4) if blur else None,
        "blur_score_p95": round(float(np.percentile(blur, 95)), 4) if blur else None,
        "block_energy_p05": round(float(np.percentile(block, 5)), 4) if block else None,
        "block_energy_p50": round(float(np.percentile(block, 50)), 4) if block else None,
        "median_megapixels": round(float(np.median([s["megapixels"] for s in stats])), 2)
        if stats
        else None,
        "median_mean_gray": round(float(np.median([s["mean_gray"] for s in stats])), 1)
        if stats
        else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="真实 vs 合成图像的质量门控域差检查")
    ap.add_argument("--real", default="data/raw/ttpla", help="真实图像目录（递归）")
    ap.add_argument("--weapd", default="data/raw/weapd/dataset", help="WEAPD 目录（可选）")
    ap.add_argument("--n", type=int, default=100, help="每个来源采样张数")
    ap.add_argument(
        "--synth-dir", default="data/processed/synthetic_ice", help="已生成的合成图目录"
    )
    ap.add_argument("--synth-n", type=int, default=100)
    ap.add_argument("--out", default="artifacts/reports/domain_gap.json")
    args = ap.parse_args()

    groups: list[dict[str, object]] = []

    for name, root in (("TTPLA(真实4K航拍)", Path(args.real)), ("WEAPD(网络图)", Path(args.weapd))):
        files = sample_images(root, args.n) if root.exists() else []
        imgs, stats = [], []
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            imgs.append(img)
            stats.append(image_stats(img))
        if imgs:
            groups.append(summarize(name, imgs, stats))

    synth = load_synthetic(Path(args.synth_dir), args.synth_n)
    if synth:
        groups.append(summarize("合成(算子B)", synth, [image_stats(i) for i in synth]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(groups, ensure_ascii=False, indent=2), "utf-8")

    print(
        f"{'来源':<20}{'n':>5}{'usable':>9}{'degraded':>10}{'unusable':>10}{'blur_p50':>10}{'block_p05':>11}"
    )
    for g in groups:
        st = g["status"]
        print(
            f"{g['source']:<20}{g['n']:>5}{st.get('usable', 0):>9.2f}{st.get('degraded', 0):>10.2f}"
            f"{st.get('unusable', 0):>10.2f}{g['blur_score_p50'] or 0:>10.3f}"
            f"{g['block_energy_p05'] or 0:>11.3f}"
        )
    print("\n主要触发原因：")
    for g in groups:
        print(f"  {g['source']}: {g['top_reasons']}")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
