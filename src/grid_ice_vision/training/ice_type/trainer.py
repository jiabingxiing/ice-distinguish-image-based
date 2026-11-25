"""覆冰类型训练与加载编排器（统一衔接 P1/P2/P3 与推理/导出链路）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import yaml

from grid_ice_vision.common import paths
from grid_ice_vision.common.config import configure_hf_mirror, configure_torch_cache
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    RejectionThresholds,
)
from grid_ice_vision.features.ice_type.classifier import (
    IceTypeLinearHead,
    SklearnClassifierAdapter,
    export_torch_head,
)
from grid_ice_vision.features.ice_type.infer import (
    IceTypeInferConfig,
    IceTypeInferencer,
    IceTypeTorchModel,
)
from grid_ice_vision.features.ice_type.quality import IceTypeQualityThresholds
from grid_ice_vision.training.checkpoint import load_into
from grid_ice_vision.training.ice_type.feature_extractor import (
    extract_and_cache_features,
    load_feature_cache,
)
from grid_ice_vision.training.ice_type.finetune import run_finetune
from grid_ice_vision.training.ice_type.train_svm import (
    ClassifierTrainConfig,
    train_cached_classifier,
)

TrainStage = Literal["p1_classical", "p2_frozen", "p3_finetune"]


def _cache_prefix(cfg: dict[str, Any], use_roi: bool | None = None) -> str:
    data_cfg = cfg.get("data", {}) or {}
    protocol = str(data_cfg.get("protocol", "monitor_holdout"))
    size = int(data_cfg.get("image_size", 224))
    roi_flag = bool(data_cfg.get("use_roi", False)) if use_roi is None else use_roi
    prefix = f"scut_{protocol}_{size}"
    if roi_flag:
        prefix += "_roi"
    return prefix


def _feature_matrix(cache: dict[str, Any], view_mode: str, use_classical: bool) -> np.ndarray:
    if use_classical:
        if "X_classical" not in cache:
            raise KeyError(
                "缓存中缺少 X_classical，请先带 --classical 抽取或直接用 p1_classical 运行"
            )
        return np.asarray(cache["X_classical"], dtype=np.float32)
    if view_mode == "full":
        return np.asarray(cache["X_full"], dtype=np.float32)
    if view_mode == "roi":
        return np.asarray(cache["X_roi"], dtype=np.float32)
    if view_mode == "dual":
        return np.hstack([cache["X_full"], cache["X_roi"]]).astype(np.float32)
    raise ValueError(f"未知 view_mode={view_mode!r}")


def run_training(
    cfg: dict[str, Any],
    *,
    stage: TrainStage = "p2_frozen",
    out_dir: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """执行指定阶段训练并落盘模型与校准产物。"""
    configure_hf_mirror(str(cfg.get("hf_endpoint", "https://hf-mirror.com")))
    configure_torch_cache()

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    tr_cfg = cfg.get("training", {}) or {}

    base_out = Path(str(tr_cfg.get("out_dir", "artifacts/ice_type")))
    if not base_out.is_absolute():
        base_out = paths.PROJ / base_out
    version = str(tr_cfg.get("version", "v1"))

    if out_dir is None:
        if stage == "p1_classical":
            target_dir = base_out / f"{version}_p1_classical"
        elif stage == "p3_finetune":
            target_dir = base_out / f"{version}_p3_finetune"
        else:
            target_dir = base_out / version
    else:
        target_dir = Path(out_dir)

    if stage == "p3_finetune":
        return run_finetune(cfg, out_dir=target_dir, root=root)

    feat_dir = Path(str(tr_cfg.get("feature_dir", "artifacts/features")))
    if not feat_dir.is_absolute():
        feat_dir = paths.PROJ / feat_dir

    use_roi = bool(data_cfg.get("use_roi", False))
    view_mode = str(data_cfg.get("view_mode", "dual" if use_roi else "full"))
    prefix = _cache_prefix(cfg, use_roi=use_roi)
    tr_npz = feat_dir / f"{prefix}_train_features.npz"
    va_npz = feat_dir / f"{prefix}_val_features.npz"

    use_classical = stage == "p1_classical"
    need_extract = not (tr_npz.exists() and va_npz.exists())
    if not need_extract and use_classical:
        c_tr = load_feature_cache(tr_npz)
        need_extract = "X_classical" not in c_tr

    if need_extract:
        extract_and_cache_features(
            cfg,
            splits=("train", "val", "test"),
            use_roi=use_roi,
            extract_classical=use_classical,
            extract_cnn=(not use_classical),
            root=root,
        )

    c_tr = load_feature_cache(tr_npz)
    c_va = load_feature_cache(va_npz)
    X_tr = _feature_matrix(c_tr, view_mode, use_classical)
    X_va = _feature_matrix(c_va, view_mode, use_classical)
    y_tr = np.asarray(c_tr["y"], dtype=np.int64)
    y_va = np.asarray(c_va["y"], dtype=np.int64)
    q_va = np.asarray(c_va["quality_scores"], dtype=np.float32)

    tcfg = ClassifierTrainConfig.from_config(cfg)
    meta = {
        "model_kind": "classical_sklearn" if use_classical else "frozen_sklearn",
        "stage": stage,
        "backbone": "classical_hsv_lbp_hog_gabor"
        if use_classical
        else str(model_cfg.get("backbone", "mobilenetv3_small_100")),
        "provider": "classical" if use_classical else str(model_cfg.get("provider", "timm")),
        "view_mode": view_mode,
        "image_size": int(data_cfg.get("image_size", 224)),
        "roi_size": int(data_cfg.get("roi_size", 224)),
        "pad_value": int(data_cfg.get("pad_value", 114)),
        "protocol": str(data_cfg.get("protocol", "monitor_holdout")),
        "version": version,
    }
    res = train_cached_classifier(
        X_tr,
        y_tr,
        X_va,
        y_va,
        cfg=tcfg,
        val_quality_scores=q_va,
        out_dir=target_dir,
        extra_meta=meta,
    )
    return {
        "stage": stage,
        "out_dir": str(target_dir),
        "classifier_type": res["classifier_type"],
        "best_C": res["best_C"],
        "c_trials": res["c_trials"],
        "val_macro_f1": res["val_macro_f1"],
        "temperature": res["temperature"],
        "reject_search": res["reject_search"],
    }


def load_ice_type_inferencer(
    model_dir: str | Path,
    *,
    device: str = "cpu",
    allow_fallback: bool = False,
) -> tuple[IceTypeInferencer, dict[str, Any]]:
    """从产物目录加载骨干 + 分类器 + 温度标定 + 拒识阈值并装配为推理器。"""
    configure_hf_mirror()
    configure_torch_cache()

    md = Path(model_dir)
    if not md.is_absolute():
        md = paths.PROJ / md
    mcfg = yaml.safe_load((md / "model_config.yaml").read_text(encoding="utf-8")) or {}
    temp_d = json.loads((md / "temperature.json").read_text(encoding="utf-8"))
    rej_d = json.loads((md / "reject_thresholds.json").read_text(encoding="utf-8"))

    scaler = MultiClassTemperatureScaler.from_dict(temp_d)
    rej_th = RejectionThresholds.from_config(rej_d)
    infer_cfg = IceTypeInferConfig(
        image_size=int(mcfg.get("image_size", 224)),
        roi_size=int(mcfg.get("roi_size", 224)),
        pad_value=int(mcfg.get("pad_value", 114)),
        view_mode=str(mcfg.get("view_mode", "full")),  # type: ignore[arg-type]
        model_name=str(mcfg.get("backbone", "mobilenetv3_small_100")),
        model_version=str(mcfg.get("version", md.name)),
        quality_thresholds=IceTypeQualityThresholds(tau_q=rej_th.tau_q),
        rejection_thresholds=rej_th,
    )

    provider = str(mcfg.get("provider", "timm"))
    backbone = IceTypeBackbone(
        str(mcfg.get("backbone", "mobilenetv3_small_100")),
        provider=provider if provider in ("timm", "torchvision") else "timm",  # type: ignore[arg-type]
        pretrained=(str(mcfg.get("model_kind", "")) == "frozen_sklearn"),
        allow_fallback=allow_fallback,
    )

    if str(mcfg.get("model_kind", "")) == "finetune_torch" or (md / "best.pt").exists():
        in_dim = int(mcfg.get("feature_dim", 576))
        head = IceTypeLinearHead(in_dim=in_dim, num_classes=len(ICE_TYPE_FOUR_CLASSES), dropout=0.0)
        torch_model = IceTypeTorchModel(
            backbone, head, view_mode=infer_cfg.view_mode, temperature=1.0
        )
        load_into(torch_model, md / "best.pt", strict=False, device=device)
        inferencer = IceTypeInferencer(
            backbone=torch_model.backbone,
            head=torch_model.head,
            scaler=scaler,
            cfg=infer_cfg,
            device=device,
        )
        return inferencer, mcfg

    std_scaler = joblib.load(md / "scaler.joblib")
    clf_path = md / "classifier.joblib"
    if not clf_path.exists():
        clf_path = md / "svm_classifier.joblib"
    clf = joblib.load(clf_path)
    adapter = SklearnClassifierAdapter(std_scaler, clf)
    inferencer = IceTypeInferencer(
        backbone=backbone,
        head=adapter,
        scaler=scaler,
        cfg=infer_cfg,
        device=device,
    )
    return inferencer, mcfg


def build_exportable_torch_model(
    model_dir: str | Path,
    *,
    allow_fallback: bool = False,
) -> tuple[IceTypeTorchModel, dict[str, Any]]:
    """将已训练好的冻结 SVM/LR/MLP 或微调模型打包为单个可导出 ONNX 的 PyTorch 模块。"""
    configure_hf_mirror()
    configure_torch_cache()

    md = Path(model_dir)
    if not md.is_absolute():
        md = paths.PROJ / md
    mcfg = yaml.safe_load((md / "model_config.yaml").read_text(encoding="utf-8")) or {}
    temp_d = json.loads((md / "temperature.json").read_text(encoding="utf-8"))
    temperature = float(temp_d.get("temperature", 1.0))

    provider = str(mcfg.get("provider", "timm"))
    view_mode = str(mcfg.get("view_mode", "full"))
    vm: Any = view_mode if view_mode in ("full", "roi", "dual") else "full"

    backbone = IceTypeBackbone(
        str(mcfg.get("backbone", "mobilenetv3_small_100")),
        provider=provider if provider in ("timm", "torchvision") else "timm",  # type: ignore[arg-type]
        pretrained=(str(mcfg.get("model_kind", "")) == "frozen_sklearn"),
        allow_fallback=allow_fallback,
    )

    if str(mcfg.get("model_kind", "")) == "finetune_torch" or (md / "best.pt").exists():
        in_dim = int(mcfg.get("feature_dim", 576))
        head: Any = IceTypeLinearHead(
            in_dim=in_dim, num_classes=len(ICE_TYPE_FOUR_CLASSES), dropout=0.0
        )
        wrapper = IceTypeTorchModel(backbone, head, view_mode=vm, temperature=temperature)
        load_into(wrapper, md / "best.pt", strict=False, device="cpu")
        wrapper.eval()
        return wrapper, mcfg

    std_scaler = joblib.load(md / "scaler.joblib")
    clf_path = md / "classifier.joblib"
    if not clf_path.exists():
        clf_path = md / "svm_classifier.joblib"
    clf = joblib.load(clf_path)
    torch_head = export_torch_head(std_scaler, clf)
    wrapper = IceTypeTorchModel(backbone, torch_head, view_mode=vm, temperature=temperature)
    wrapper.eval()
    return wrapper, mcfg
