"""纯 CPU 特征抽取与 NPZ 磁盘缓存（P1 手工特征 + P2 MobileNetV3-Small 576 维特征）。"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grid_ice_vision.common import paths
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.data.datasets.ice_type import (
    IceTypeDataset,
    ViewMode,
    collate,
    load_monitor_roi_table,
)
from grid_ice_vision.data.manifest import ManifestRecord, assert_no_leakage, load_manifest
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.classical import (
    ClassicalFeatureConfig,
    extract_classical_features,
)
from grid_ice_vision.features.ice_type.quality import (
    IceTypeQualityThresholds,
    evaluate_ice_type_quality,
)
from grid_ice_vision.features.ice_type.roi import NormalizedRoi


def configure_cpu_threads(max_threads: int = 8) -> int:
    """设置 CPU 线程预算，避免 DataLoader/MKL 过度抢占。"""
    n = min(os.cpu_count() or 4, max_threads)
    torch.set_num_threads(n)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    return n


def extract_split_features(
    records: list[ManifestRecord],
    *,
    root: str | Path | None = None,
    backbone: IceTypeBackbone | None = None,
    image_size: int = 224,
    roi_size: int = 224,
    pad_value: int = 114,
    view_mode: ViewMode = "full",
    roi_map: dict[str, NormalizedRoi] | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    exclude_noise_candidates: bool = True,
    extract_classical: bool = False,
    classical_cfg: ClassicalFeatureConfig | None = None,
    quality_thresholds: IceTypeQualityThresholds | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """抽取给定记录列表的全图/ROI CNN 特征（及可选传统手工特征）与评估元数据。"""
    recs = records[:limit] if limit is not None else records
    ds = IceTypeDataset(
        recs,
        root=root,
        image_size=image_size,
        roi_size=roi_size,
        pad_value=pad_value,
        view_mode=view_mode,
        roi_map=roi_map,
        train=False,
        exclude_noise_candidates=exclude_noise_candidates,
        return_bgr=True,
    )
    if len(ds) == 0:
        raise ValueError("待抽取特征的数据集为空")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    if backbone is not None:
        backbone.eval()
        backbone.freeze_all()

    x_full_list: list[np.ndarray] = []
    x_roi_list: list[np.ndarray] = []
    x_cls_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    q_list: list[float] = []
    image_ids: list[str] = []
    monitor_ids: list[str] = []
    bureaus: list[str] = []
    years: list[str] = []
    resolutions: list[str] = []
    roi_avail_list: list[bool] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if backbone is not None:
                f_full = backbone.forward_single(batch["image"]).cpu().numpy().astype(np.float32)
                x_full_list.append(f_full)
                if "roi_image" in batch:
                    f_roi = (
                        backbone.forward_single(batch["roi_image"]).cpu().numpy().astype(np.float32)
                    )
                    x_roi_list.append(f_roi)

            for idx_b, img_bgr in enumerate(batch["image_bgr"]):
                roi_bgr = batch["roi_bgr"][idx_b] if "roi_bgr" in batch else None
                q = evaluate_ice_type_quality(
                    img_bgr, thresholds=quality_thresholds, roi_image_bgr=roi_bgr
                )
                q_list.append(q.q_image)
                if extract_classical:
                    feat_c = extract_classical_features(img_bgr, cfg=classical_cfg)
                    if roi_bgr is not None:
                        feat_r = extract_classical_features(roi_bgr, cfg=classical_cfg)
                        feat_c = np.concatenate([feat_c, feat_r]).astype(np.float32)
                    x_cls_list.append(feat_c)

            y_list.append(batch["label"].numpy().astype(np.int64))
            image_ids.extend(batch["image_id"])
            monitor_ids.extend(batch["monitor_id"])
            bureaus.extend(batch["bureau"])
            years.extend(batch["year"])
            resolutions.extend(batch["resolution"])
            roi_avail_list.extend(batch["roi_available"])

    elapsed_s = time.perf_counter() - t0
    out: dict[str, Any] = {
        "y": np.concatenate(y_list, axis=0),
        "quality_scores": np.asarray(q_list, dtype=np.float32),
        "image_ids": np.asarray(image_ids),
        "monitor_ids": np.asarray(monitor_ids),
        "bureaus": np.asarray(bureaus),
        "years": np.asarray(years),
        "resolutions": np.asarray(resolutions),
        "roi_available": np.asarray(roi_avail_list, dtype=bool),
        "elapsed_s": round(elapsed_s, 3),
    }
    if x_full_list:
        out["X_full"] = np.concatenate(x_full_list, axis=0)
    if x_roi_list:
        out["X_roi"] = np.concatenate(x_roi_list, axis=0)
    if x_cls_list:
        out["X_classical"] = np.stack(x_cls_list, axis=0)
    return out


def save_feature_cache(out_path: str | Path, data: dict[str, Any], meta: dict[str, Any]) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    payload["extract_meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(p, **payload)
    return p


def load_feature_cache(npz_path: str | Path) -> dict[str, Any]:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"特征缓存不存在: {p}")
    with np.load(p, allow_pickle=False) as z:
        out: dict[str, Any] = {k: z[k] for k in z.files if k != "extract_meta"}
        if "extract_meta" in z.files:
            out["meta"] = json.loads(str(z["extract_meta"].item()))
    return out


def extract_and_cache_features(
    cfg: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    use_roi: bool | None = None,
    extract_classical: bool = False,
    extract_cnn: bool = True,
    root: str | Path | None = None,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """按配置从 manifest 抽取各 split 特征并落盘 NPZ 缓存。"""
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    tr_cfg = cfg.get("training", {}) or {}

    protocol = str(data_cfg.get("protocol", "monitor_holdout"))
    manifest_key = "temporal_manifest" if protocol == "temporal_holdout" else "manifest"
    manifest_path = Path(str(data_cfg.get(manifest_key, "data/manifests/scut_ice_type_v1.jsonl")))
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJ / manifest_path

    records = load_manifest(manifest_path)
    assert_no_leakage(records)

    image_size = int(data_cfg.get("image_size", 224))
    roi_size = int(data_cfg.get("roi_size", 224))
    pad_value = int(data_cfg.get("pad_value", 114))
    roi_enabled = bool(data_cfg.get("use_roi", False)) if use_roi is None else use_roi
    view_mode: ViewMode = "dual" if roi_enabled else "full"
    roi_map = load_monitor_roi_table(
        str(data_cfg.get("roi_csv", "configs/data/monitor_roi_v1.csv"))
    )
    exclude_noise = bool(data_cfg.get("exclude_noise_candidates", True))
    batch_size = int(tr_cfg.get("batch_size", 16))
    num_workers = int(tr_cfg.get("num_workers", 0))
    threads = configure_cpu_threads(int(tr_cfg.get("cpu_threads", 8)))

    backbone: IceTypeBackbone | None = None
    backbone_name = str(model_cfg.get("backbone", "mobilenetv3_small_100"))
    if extract_cnn:
        provider = str(model_cfg.get("provider", "timm"))
        pretrained = bool(model_cfg.get("pretrained", True))
        backbone = IceTypeBackbone(
            backbone_name,
            provider=provider,  # type: ignore[arg-type]
            pretrained=pretrained,
            allow_fallback=bool(model_cfg.get("allow_fallback", False)),
        )
        backbone.freeze_all()

    feat_dir = Path(str(tr_cfg.get("feature_dir", "artifacts/features")))
    if not feat_dir.is_absolute():
        feat_dir = paths.PROJ / feat_dir

    cls_cfg = ClassicalFeatureConfig.from_config(cfg)
    q_th = IceTypeQualityThresholds.from_config(cfg)

    prefix = f"scut_{protocol}_{image_size}"
    if roi_enabled:
        prefix += "_roi"
    man_label = (
        paths.rel_posix(manifest_path)
        if manifest_path.resolve().is_relative_to(paths.PROJ.resolve())
        else manifest_path.name
    )
    summary: dict[str, Any] = {
        "protocol": protocol,
        "manifest": man_label,
        "image_size": image_size,
        "roi_enabled": roi_enabled,
        "threads": threads,
        "backbone": backbone.spec.name if backbone is not None else None,
        "classes": list(ICE_TYPE_FOUR_CLASSES),
        "splits": {},
    }

    for sp in splits:
        sp_recs = [r for r in records if r.split == sp]
        if not sp_recs:
            raise ValueError(f"split={sp!r} 为空")
        data = extract_split_features(
            sp_recs,
            root=root,
            backbone=backbone,
            image_size=image_size,
            roi_size=roi_size,
            pad_value=pad_value,
            view_mode=view_mode,
            roi_map=roi_map,
            batch_size=batch_size,
            num_workers=num_workers,
            exclude_noise_candidates=exclude_noise,
            extract_classical=extract_classical,
            classical_cfg=cls_cfg,
            quality_thresholds=q_th,
            limit=limit_per_split,
        )
        meta = {
            "split": sp,
            "protocol": protocol,
            "image_size": image_size,
            "roi_enabled": roi_enabled,
            "backbone": backbone.spec.name if backbone is not None else None,
            "n_samples": len(data["y"]),
            "elapsed_s": data["elapsed_s"],
        }
        out_npz = feat_dir / f"{prefix}_{sp}_features.npz"
        save_feature_cache(out_npz, data, meta)
        npz_label = (
            paths.rel_posix(out_npz)
            if out_npz.resolve().is_relative_to(paths.PROJ.resolve())
            else str(out_npz)
        )
        summary["splits"][sp] = {
            "path": npz_label,
            "n_samples": len(data["y"]),
            "elapsed_s": data["elapsed_s"],
            "ms_per_image": round(data["elapsed_s"] * 1000.0 / max(len(data["y"]), 1), 2),
        }

    return summary
