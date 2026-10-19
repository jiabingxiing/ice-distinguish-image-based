"""合成算子 B：参数化覆冰截面（§11.4）。

价值不在「扩充样本量」，而在**零误差几何真值**：`t` 是我们自己给的，
因此可以直接训练与评估厚度回归 —— 这是无现场数据条件下唯一能验证
物理量算链路正确性的办法。

截面模板（θ 从法向起算）：
    圆筒形   r(θ) = r0 + t
    C2 薄新月 r(θ) = r0 + t·(0.5 + 0.5·cos θ)
    C7 厚新月 r(θ) = r0 + t·(0.7 + 0.3·cos θ)
    C11 D 形  r(θ) = r0 + t·(0.8 + 0.2·cos θ)，|θ|<60° 内加平顶
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from grid_ice_vision.features.severity.geometry import (
    measure_widths,
    resample_polyline,
    skeleton_polyline,
    tangents_and_normals,
)
from grid_ice_vision.features.severity.thickness import equivalent_thickness_mm


@dataclass
class IceShapeSpec:
    shape: str = "cylinder"  # cylinder | c2 | c7 | c11
    thickness_px: float = 6.0
    n_theta: int = 64
    density: float = 900.0
    d0_mm: float = 30.0
    seed: int = 0


def _profile(theta: np.ndarray, shape: str) -> np.ndarray:
    c = np.cos(theta)
    if shape == "cylinder":
        return np.ones_like(theta)
    if shape == "c2":
        return 0.5 + 0.5 * c
    if shape == "c7":
        return 0.7 + 0.3 * c
    if shape == "c11":
        base = 0.8 + 0.2 * c
        flat = np.abs(theta) < math.radians(60)
        base[flat] = base.max()
        return base
    raise ValueError(f"未知截面模板: {shape}")


def expand_cross_section(mask: np.ndarray, spec: IceShapeSpec) -> np.ndarray:
    """沿骨架法向按截面模板外扩，返回覆冰掩码（0/1）。"""
    binary = (mask > 0).astype(np.uint8)
    poly = resample_polyline(skeleton_polyline(binary), spacing=1.5)
    tau, normal = tangents_and_normals(poly, smooth=5)
    dt = distance_transform_edt(binary)

    canvas = np.zeros_like(binary)
    theta = np.linspace(-math.pi, math.pi, spec.n_theta, endpoint=False)
    prof = _profile(theta, spec.shape)

    for p, n, t in zip(poly, normal, tau, strict=True):
        # 距离变换给的是「到最近背景像素中心」的距离，连续边界在其内侧 0.5 px；
        # 而 fillPoly 以顶点半径 R 填充时，实际边界落在 R+0.5 —— 两个 0.5 相抵，
        # 顶点半径取 dt − 1 + t 时，生成掩码的边缘半径恰为 dt − 0.5 + t。
        r0 = float(dt[round(p[1]), round(p[0])]) - 1.0
        if r0 <= 0:
            continue
        radii = r0 + spec.thickness_px * prof
        dirs = np.cos(theta)[:, None] * n[None, :] + np.sin(theta)[:, None] * t[None, :]
        pts = p[None, :] + radii[:, None] * dirs
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)

    return canvas


def _texture_ice(
    image: np.ndarray, ice: np.ndarray, spec: IceShapeSpec, rng: np.random.Generator
) -> np.ndarray:
    """给冰区贴纹理：无真实纹理块时用「去饱和 + 高光 + 噪声」近似 glaze。"""
    out = image.copy()
    m = ice.astype(bool)
    if not m.any():
        return out
    region = out[m].astype(np.float32)
    gray = region.mean(axis=1, keepdims=True)
    ice_color = gray * 0.85 + np.array([235, 240, 245], dtype=np.float32) * 0.25
    noise = rng.normal(0, 6.0, size=region.shape).astype(np.float32)
    out[m] = np.clip(ice_color + noise, 0, 255).astype(np.uint8)
    # 局部高光：模拟冰壳镜面反射
    glow = gaussian_filter(ice.astype(np.float32), 2.0)
    glow = (glow / max(glow.max(), 1e-6) * 40).astype(np.float32)
    out = np.clip(out.astype(np.float32) + glow[..., None], 0, 255).astype(np.uint8)
    return out


def make_synthetic_scene(
    height: int = 360,
    width: int = 720,
    thickness_px: float = 6.0,
    shape: str = "cylinder",
    sag_px: float = 40.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """从零生成一张「天空 + 裸导线 + 覆冰」的合成图，返回 (image, mask, truth)。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array([200, 170, 140], dtype=np.float32)  # BGR 偏暖天空
    bottom = np.array([150, 160, 175], dtype=np.float32)
    for y in range(height):
        a = y / max(height - 1, 1)
        img[y, :, :] = (top * (1 - a) + bottom * a).astype(np.uint8)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    bare = np.zeros((height, width), dtype=np.uint8)
    xs = np.arange(40, width - 40)
    curve = height / 2 + sag_px * ((xs - width / 2) / (width / 2)) ** 2 - sag_px
    for x, y in zip(xs, curve, strict=True):
        cv2.circle(bare, (int(x), round(y)), 5, 1, -1)

    ice = expand_cross_section(
        bare, IceShapeSpec(shape=shape, thickness_px=thickness_px, seed=seed)
    )
    ice_only = (ice == 1) & (bare == 0)

    out = img.copy()
    out[bare == 1] = (60, 60, 65)  # 深灰导线
    out = _texture_ice(
        out, ice_only.astype(np.uint8), IceShapeSpec(shape=shape, thickness_px=thickness_px), rng
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[bare == 1] = 1
    mask[ice_only] = 2

    wm_bare = measure_widths(bare, spacing=4.0)
    wm_ice = measure_widths(ice, spacing=4.0)
    alpha = 30.0 / wm_bare.width_px
    d_radial = 0.5 * (wm_ice.width_px - wm_bare.width_px) * alpha
    truth = {
        "shape": shape,
        "thickness_px": thickness_px,
        "w0_px": round(wm_bare.width_px, 4),
        "w_ice_px": round(wm_ice.width_px, 4),
        "alpha_mm_per_px": round(alpha, 6),
        "d_radial_mm_true": round(thickness_px * alpha, 4),
        "d_radial_mm_measured": round(d_radial, 4),
        "d_eq_mm_true": round(equivalent_thickness_mm(thickness_px * alpha, 30.0, 900.0), 4),
        "width_err_px": round(abs(wm_ice.width_px - wm_bare.width_px - 2 * thickness_px), 4),
        "seed": seed,
    }
    return out, mask, truth


def synthesize_from_image(
    image: np.ndarray,
    conductor_mask: np.ndarray,
    spec: IceShapeSpec,
    out_dir: str | Path,
    name: str = "synth",
    texture_patch: np.ndarray | None = None,
) -> dict[str, Any]:
    """在真实图像上叠加参数化冰层，落盘图像/掩码/真值 JSON。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ice = expand_cross_section(conductor_mask, spec)
    ice_only = ((ice == 1) & (conductor_mask == 0)).astype(np.uint8)

    rng = np.random.default_rng(spec.seed)
    rendered = image.copy()
    if texture_patch is not None and ice_only.any():
        patch = cv2.resize(texture_patch, (image.shape[1], image.shape[0]))
        blended = image.astype(np.float32) * 0.45 + patch.astype(np.float32) * 0.55
        rendered[ice_only == 1] = np.clip(blended[ice_only == 1], 0, 255).astype(np.uint8)
    else:
        rendered = _texture_ice(rendered, ice_only, spec, rng)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[conductor_mask > 0] = 1
    mask[ice_only == 1] = 2

    wm_bare = measure_widths((conductor_mask > 0).astype(np.uint8), spacing=4.0)
    wm_ice = measure_widths(ice, spacing=4.0)
    alpha = spec.d0_mm / wm_bare.width_px
    truth = {
        "image": f"{name}.jpg",
        "mask": f"{name}_mask.png",
        "shape": spec.shape,
        "thickness_px": spec.thickness_px,
        "d0_mm": spec.d0_mm,
        "rho_kg_m3": spec.density,
        "w0_px": round(wm_bare.width_px, 4),
        "w_ice_px": round(wm_ice.width_px, 4),
        "alpha_mm_per_px": round(alpha, 6),
        "d_radial_mm_true": round(spec.thickness_px * alpha, 4),
        "d_eq_mm_true": round(
            equivalent_thickness_mm(spec.thickness_px * alpha, spec.d0_mm, spec.density), 4
        ),
        "is_synthetic": True,
    }

    cv2.imwrite(str(out / f"{name}.jpg"), rendered)
    cv2.imwrite(str(out / f"{name}_mask.png"), mask)
    (out / f"{name}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), "utf-8")
    return truth


SHAPES: tuple[str, ...] = ("cylinder", "c2", "c7", "c11")


@dataclass
class BatchSpec:
    n: int = 20
    thickness_px: tuple[float, float] = (2.0, 14.0)
    shapes: tuple[str, ...] = SHAPES
    out_dir: str = "data/processed/synthetic_ice"
    height: int = 360
    width: int = 720
    seed: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


def generate_batch(spec: BatchSpec) -> list[dict[str, Any]]:
    rng = np.random.default_rng(spec.seed)
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for i in range(spec.n):
        t = float(rng.uniform(*spec.thickness_px))
        shape = str(rng.choice(spec.shapes))
        img, mask, truth = make_synthetic_scene(
            spec.height,
            spec.width,
            t,
            shape,
            sag_px=float(rng.uniform(0, 60)),
            seed=int(rng.integers(1e6)),
        )
        name = f"synth_{i:05d}"
        cv2.imwrite(str(out / f"{name}.jpg"), img)
        cv2.imwrite(str(out / f"{name}_mask.png"), mask)
        truth.update({"image": f"{name}.jpg", "mask": f"{name}_mask.png", "is_synthetic": True})
        (out / f"{name}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), "utf-8")
        records.append(truth)
    (out / "truth.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), "utf-8"
    )
    return records


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="合成参数化覆冰样本（算子 B）")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/processed/synthetic_ice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = generate_batch(BatchSpec(n=args.n, out_dir=args.out, seed=args.seed))
    print(f"生成 {len(recs)} 张，输出目录 {args.out}")
