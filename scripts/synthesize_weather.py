"""合成算子 A：大气散射 + 雨纹 + 雪花（§11.4）。

    I_hazy = I·T + A·(1 − T),  T = exp(−β·d(x))

深度 d(x) 用暗通道先验自研估计 —— plan6 称 PLAIR-AW 提供合成脚本，
实测其仓库只有数据集说明与基线评测，**没有**合成代码。

产出 (I_degraded, I_clean, intensity_label) 三元组，强度 L/M/H 由 β 连续控制。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch: int = 15) -> np.ndarray:
    """暗通道：局部窗口内三通道最小值。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(image.min(axis=2), k)


def estimate_depth(image: np.ndarray, patch: int = 15) -> np.ndarray:
    """用暗通道近似深度：暗通道值越大 → 雾越浓 → 距离越远。

    这不是几何深度，只是**相对**深度先验，用于让雾在远景更浓。
    """
    dc = dark_channel(image, patch).astype(np.float32)
    dc = cv2.GaussianBlur(dc, (0, 0), 3.0)
    d = dc - dc.min()
    peak = float(d.max())
    return d / peak if peak > 1e-6 else np.zeros_like(d)


def atmospheric_light(image: np.ndarray, top_ratio: float = 0.001) -> np.ndarray:
    """估计大气光 A：暗通道最亮的一小撮像素的均值。"""
    dc = dark_channel(image).ravel()
    img = image.reshape(-1, 3)
    n = max(int(len(dc) * top_ratio), 1)
    idx = np.argsort(-dc)[:n]
    return img[idx].mean(axis=0)


@dataclass(frozen=True)
class HazeSpec:
    beta: float = 6e-6
    a_min: float = 0.4
    a_max: float = 0.6
    level: str = "moderate"


LEVEL_BETA: dict[str, float] = {"light": 2.5e-6, "moderate": 5.0e-6, "heavy": 8.0e-6}


def add_haze(
    image: np.ndarray, spec: HazeSpec, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    depth = estimate_depth(image)
    a = float(rng.uniform(spec.a_min, spec.a_max)) * 255.0
    t = np.exp(-spec.beta * depth * 1e4).astype(np.float32)[..., None]
    out = image.astype(np.float32) * t + a * (1 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain(
    image: np.ndarray,
    density: float = 0.02,
    angle_deg: float = 15.0,
    length: int = 25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """雨纹层：随机短线段 + 轻微运动模糊。"""
    rng = rng or np.random.default_rng(0)
    h, w = image.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)
    n = int(h * w * density / max(length, 1))
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    rad = np.deg2rad(angle_deg)
    dx, dy = int(length * np.sin(rad)), int(length * np.cos(rad))
    for x, y in zip(xs, ys, strict=True):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
    out = image.astype(np.float32) + layer[..., None] * 90.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_snow(
    image: np.ndarray, density: float = 0.002, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    h, w = image.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)
    n = int(h * w * density)
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    radii = rng.integers(1, 4, n)
    for x, y, r in zip(xs, ys, radii, strict=True):
        cv2.circle(layer, (int(x), int(y)), int(r), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
    out = image.astype(np.float32) * (1 - layer[..., None] * 0.55) + 255.0 * layer[..., None] * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize(image: np.ndarray, kind: str, level: str, seed: int = 0) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    if kind == "fog":
        out = add_haze(image, HazeSpec(beta=LEVEL_BETA[level], level=level), rng)
    elif kind == "rain":
        out = add_rain(
            image, density={"light": 0.008, "moderate": 0.02, "heavy": 0.05}[level], rng=rng
        )
        out = add_haze(out, HazeSpec(beta=LEVEL_BETA[level] * 0.5), rng)
    elif kind == "snow":
        out = add_snow(
            image, density={"light": 0.001, "moderate": 0.003, "heavy": 0.007}[level], rng=rng
        )
        out = add_haze(out, HazeSpec(beta=LEVEL_BETA[level] * 0.4), rng)
    else:
        raise ValueError(f"未知天气类型: {kind}")
    return out, level


@dataclass
class WeatherBatchSpec:
    src_dir: str = "data/raw"
    out_dir: str = "data/processed/synthetic_weather"
    kinds: tuple[str, ...] = ("fog", "rain", "snow")
    levels: tuple[str, ...] = ("light", "moderate", "heavy")
    max_images: int = 200
    seed: int = 0


def generate_batch(spec: WeatherBatchSpec) -> list[dict[str, str]]:
    src = Path(spec.src_dir)
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    records: list[dict[str, str]] = []
    rng = np.random.default_rng(spec.seed)
    for _i, f in enumerate(files[: spec.max_images]):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        kind = str(rng.choice(spec.kinds))
        level = str(rng.choice(spec.levels))
        degraded, lab = synthesize(img, kind, level, int(rng.integers(1e6)))
        name = f"{f.stem}_{kind}_{lab}"
        cv2.imwrite(str(out / f"{name}.jpg"), degraded)
        records.append(
            {
                "clean": str(f),
                "degraded": f"{name}.jpg",
                "kind": kind,
                "intensity": lab,
                "is_synthetic": "true",
            }
        )
    (out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), "utf-8"
    )
    return records


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="合成退化天气样本（算子 A）")
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--out", default="data/processed/synthetic_weather")
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()
    recs = generate_batch(WeatherBatchSpec(src_dir=args.src, out_dir=args.out, max_images=args.max))
    print(f"生成 {len(recs)} 对退化样本 → {args.out}")
