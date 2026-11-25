"""覆冰类型四分类双协议多维评估与混淆/切片/画廊导出脚本（Plan §3.2 & §9.2）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common import paths
from grid_ice_vision.common.config import (
    configure_hf_mirror,
    configure_torch_cache,
    load_config,
)
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.data.datasets.ice_type import (
    IceTypeDataset,
    ViewMode,
    collate,
    load_monitor_roi_table,
)
from grid_ice_vision.data.manifest import assert_no_leakage, load_manifest
from grid_ice_vision.evaluation.ice_type import (
    compute_ice_type_metrics,
    render_calibration_plot,
    render_confusion_heatmap,
    render_error_gallery,
)
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    RejectionThresholds,
    apply_rejection_gate,
)
from grid_ice_vision.features.ice_type.classifier import (
    IceTypeLinearHead,
    SklearnClassifierAdapter,
)
from grid_ice_vision.features.ice_type.infer import IceTypeTorchModel
from grid_ice_vision.features.ice_type.quality import (
    IceTypeQualityThresholds,
    evaluate_ice_type_quality,
)
from grid_ice_vision.training.checkpoint import load_into
from grid_ice_vision.training.ice_type.feature_extractor import (
    configure_cpu_threads,
    load_feature_cache,
)


def _resolve_model_dir(cfg: dict[str, Any], stage: str, model_dir: str | None) -> Path:
    if model_dir:
        p = Path(model_dir)
        return p if p.is_absolute() else paths.PROJ / p
    tr_cfg = cfg.get("training", {}) or {}
    base = Path(str(tr_cfg.get("out_dir", "artifacts/ice_type")))
    if not base.is_absolute():
        base = paths.PROJ / base
    ver = str(tr_cfg.get("version", "v1"))
    if stage == "p1_classical":
        return base / f"{ver}_p1_classical"
    if stage == "p3_finetune":
        return base / f"{ver}_p3_finetune"
    return base / ver


def _eval_from_cache_or_model(
    cfg: dict[str, Any],
    model_dir: Path,
    protocol: str,
    split: str,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]], list[Path]]:
    mcfg = yaml.safe_load((model_dir / "model_config.yaml").read_text(encoding="utf-8")) or {}
    temp_d = json.loads((model_dir / "temperature.json").read_text(encoding="utf-8"))
    temp_scaler = MultiClassTemperatureScaler.from_dict(temp_d)

    data_cfg = cfg.get("data", {}) or {}
    manifest_key = "temporal_manifest" if protocol == "temporal_holdout" else "manifest"
    manifest_path = Path(str(data_cfg.get(manifest_key, "data/manifests/scut_ice_type_v1.jsonl")))
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJ / manifest_path

    records = [r for r in load_manifest(manifest_path) if r.split == split]
    assert_no_leakage(load_manifest(manifest_path))
    exclude_noise = bool(data_cfg.get("exclude_noise_candidates", True))
    scut_root = paths.scut_root()
    valid_recs = [
        r
        for r in records
        if not (exclude_noise and bool(r.labels.get("noise_candidate", False)))
        and str(r.labels.get("ice_type")) in ICE_TYPE_FOUR_CLASSES
    ]
    img_paths = [scut_root / r.path for r in valid_recs]

    view_mode = str(mcfg.get("view_mode", data_cfg.get("view_mode", "full")))
    size = int(mcfg.get("image_size", data_cfg.get("image_size", 224)))
    use_roi = view_mode in ("roi", "dual")
    prefix = f"scut_{protocol}_{size}" + ("_roi" if use_roi else "")
    feat_dir = Path(str((cfg.get("training", {}) or {}).get("feature_dir", "artifacts/features")))
    if not feat_dir.is_absolute():
        feat_dir = paths.PROJ / feat_dir
    cache_path = feat_dir / f"{prefix}_{split}_features.npz"
    if not cache_path.exists() and view_mode == "full":
        alt = feat_dir / f"scut_{protocol}_{size}_roi_{split}_features.npz"
        if alt.exists():
            cache_path = alt

    if stage != "p3_finetune" and str(mcfg.get("model_kind", "")) != "finetune_torch":
        std_scaler = joblib.load(model_dir / "scaler.joblib")
        clf_path = model_dir / "classifier.joblib"
        if not clf_path.exists():
            clf_path = model_dir / "svm_classifier.joblib"
        clf = joblib.load(clf_path)
        adapter = SklearnClassifierAdapter(std_scaler, clf)

        if cache_path.exists():
            c = load_feature_cache(cache_path)
            if stage == "p1_classical" and "X_classical" in c:
                X = np.asarray(c["X_classical"], dtype=np.float32)
            elif stage == "p2_frozen" and "X_full" in c:
                if view_mode == "full":
                    X = np.asarray(c["X_full"], dtype=np.float32)
                elif view_mode == "roi":
                    X = np.asarray(c["X_roi"], dtype=np.float32)
                else:
                    X = np.hstack([c["X_full"], c["X_roi"]]).astype(np.float32)
            else:
                X = np.empty((0,))
            if X.size > 0:
                logits = adapter.predict_logits(X)
                probs = temp_scaler.predict_proba(logits)
                y = np.asarray(c["y"], dtype=np.int64)
                q = np.asarray(c["quality_scores"], dtype=np.float32)
                groups = {
                    "bureau": [str(v) for v in c["bureaus"].tolist()],
                    "monitor_id": [str(v) for v in c["monitor_ids"].tolist()],
                    "year": [str(v) for v in c["years"].tolist()],
                    "resolution": [str(v) for v in c["resolutions"].tolist()],
                }
                return probs, y, q, groups, img_paths

    # 直接前向（P3 微调或无缓存时）
    roi_map = (
        load_monitor_roi_table(str(data_cfg.get("roi_csv", "configs/data/monitor_roi_v1.csv")))
        if use_roi
        else {}
    )
    vm: ViewMode = view_mode if view_mode in ("full", "roi", "dual") else "full"  # type: ignore[assignment]
    ds = IceTypeDataset(
        valid_recs,
        root=scut_root,
        image_size=size,
        roi_size=int(mcfg.get("roi_size", 224)),
        view_mode=vm,
        roi_map=roi_map,
        train=False,
        exclude_noise_candidates=exclude_noise,
        return_bgr=True,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    provider = str(mcfg.get("provider", "timm"))
    backbone = IceTypeBackbone(
        str(mcfg.get("backbone", "mobilenetv3_small_100")),
        provider=provider if provider in ("timm", "torchvision") else "timm",  # type: ignore[arg-type]
        pretrained=(str(mcfg.get("model_kind", "")) == "frozen_sklearn"),
    )

    q_th = IceTypeQualityThresholds.from_config(cfg)
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    q_all: list[float] = []
    bureaus, monitors, years, resolutions = [], [], [], []

    if str(mcfg.get("model_kind", "")) == "finetune_torch" or (model_dir / "best.pt").exists():
        head = IceTypeLinearHead(int(mcfg.get("feature_dim", 576)), len(ICE_TYPE_FOUR_CLASSES), 0.0)
        torch_model = IceTypeTorchModel(backbone, head, view_mode=vm, temperature=1.0)
        load_into(torch_model, model_dir / "best.pt", strict=False, device="cpu")
        torch_model.eval()
        with torch.no_grad():
            for batch in loader:
                roi_t = batch["roi_image"] if vm == "dual" and "roi_image" in batch else None
                img_t = (
                    batch["roi_image"] if vm == "roi" and "roi_image" in batch else batch["image"]
                )
                logits_all.append(torch_model(img_t, roi_t).numpy().astype(np.float64))
                y_all.append(batch["label"].numpy().astype(np.int64))
                for ib, bgr in enumerate(batch["image_bgr"]):
                    r_bgr = batch["roi_bgr"][ib] if "roi_bgr" in batch else None
                    q_all.append(evaluate_ice_type_quality(bgr, q_th, r_bgr).q_image)
                bureaus.extend(batch["bureau"])
                monitors.extend(batch["monitor_id"])
                years.extend(batch["year"])
                resolutions.extend(batch["resolution"])
    else:
        raise RuntimeError(f"未找到缓存特征 {cache_path}，请先运行 extract_ice_type_features.py")

    probs = temp_scaler.predict_proba(np.concatenate(logits_all, axis=0))
    return (
        probs,
        np.concatenate(y_all, axis=0),
        np.asarray(q_all, dtype=np.float32),
        {
            "bureau": bureaus,
            "monitor_id": monitors,
            "year": years,
            "resolution": resolutions,
        },
        img_paths,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评估覆冰类型四分类模型并导出混淆矩阵与切片报告")
    ap.add_argument("--config", default="configs/train/ice_type_cpu.yaml")
    ap.add_argument(
        "--protocol", default="monitor_holdout", choices=["monitor_holdout", "temporal_holdout"]
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--stage", default="p2_frozen", choices=["p1_classical", "p2_frozen", "p3_finetune"]
    )
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--render-gallery", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    configure_hf_mirror(str(cfg.get("hf_endpoint", "https://hf-mirror.com")))
    configure_torch_cache()
    configure_cpu_threads(int((cfg.get("training", {}) or {}).get("cpu_threads", 8)))

    model_dir = _resolve_model_dir(cfg, args.stage, args.model_dir)
    rej_th = RejectionThresholds.from_config(
        json.loads((model_dir / "reject_thresholds.json").read_text(encoding="utf-8"))
    )

    probs, y, q_scores, groups, img_paths = _eval_from_cache_or_model(
        cfg, model_dir, args.protocol, args.split, args.stage
    )
    rej_mask, rej_reasons = apply_rejection_gate(probs, q_scores, rej_th)
    metrics = compute_ice_type_metrics(
        probs,
        y,
        reject_mask=rej_mask,
        reject_reasons=rej_reasons,
        groups=groups,
    )
    metrics["protocol"] = args.protocol
    metrics["split"] = args.split
    metrics["stage"] = args.stage
    metrics["model_dir"] = (
        paths.rel_posix(model_dir) if model_dir.is_relative_to(paths.PROJ) else str(model_dir)
    )
    metrics["reject_thresholds"] = rej_th.to_dict()

    out_path = (
        Path(args.out)
        if args.out
        else (
            paths.PROJ
            / "artifacts"
            / "reports"
            / f"ice_type_{args.stage}_{args.protocol}_{args.split}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 渲染混淆矩阵与校准图
    render_confusion_heatmap(
        metrics["confusion"]["matrix"],
        ICE_TYPE_FOUR_CLASSES,
        model_dir / f"confusion_{args.protocol}_{args.split}.png",
        title=f"{args.stage} | {args.protocol}/{args.split} (Macro-F1={metrics['macro_f1']:.3f})",
    )
    render_calibration_plot(
        probs,
        y,
        model_dir / f"calibration_{args.protocol}_{args.split}.png",
        ece=metrics["ece"],
    )
    if args.render_gallery:
        render_error_gallery(img_paths, y, probs.argmax(axis=1), probs, model_dir / "eval_gallery")

    print(
        f"[eval] {args.stage} ({args.protocol}/{args.split}) n={metrics['n_samples']} | "
        f"Macro-F1={metrics['macro_f1']:.4f} | BAcc={metrics['balanced_accuracy']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | ECE={metrics['ece']:.4f} | "
        f"PR-AUC={metrics['pr_auc_macro']:.4f}"
    )
    print(
        f"[eval] glaze->rime={metrics['confusion']['glaze_to_rime']:.2%} "
        f"rime->glaze={metrics['confusion']['rime_to_glaze']:.2%}"
    )
    if "rejection" in metrics:
        rj = metrics["rejection"]
        print(
            f"[eval] 拒识率={rj['reject_rate']:.2%} (保留 {rj['accepted_count']}) -> "
            f"接受子集 Macro-F1={rj['accepted_macro_f1']:.4f} ({rj['macro_f1_gain']:+.4f})"
        )
    print(f"[eval] 写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
