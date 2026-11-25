"""P3 选择性末端微调：仅解冻 MobileNetV3-Small 最后 1–2 个阶段并严格执行分层学习率。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from grid_ice_vision.common import paths
from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.data.datasets.ice_type import (
    IceTypeDataset,
    ViewMode,
    collate,
    load_monitor_roi_table,
)
from grid_ice_vision.data.manifest import assert_no_leakage, load_manifest
from grid_ice_vision.evaluation.classification import balanced_accuracy, macro_f1
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.calibrate import (
    MultiClassTemperatureScaler,
    search_rejection_thresholds,
)
from grid_ice_vision.features.ice_type.classifier import IceTypeLinearHead
from grid_ice_vision.features.ice_type.infer import IceTypeTorchModel
from grid_ice_vision.features.ice_type.quality import (
    IceTypeQualityThresholds,
    evaluate_ice_type_quality,
)
from grid_ice_vision.training.engine import EarlyStopper
from grid_ice_vision.training.ice_type.feature_extractor import configure_cpu_threads


@dataclass(frozen=True)
class FinetuneConfig:
    unfreeze_last_stages: int = 1
    batch_size: int = 8
    epochs: int = 8
    backbone_lr: float = 1e-5
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    dropout: float = 0.2
    grad_clip: float = 5.0
    cpu_threads: int = 8
    random_seed: int = 42
    min_coverage: float = 0.80
    tau_q: float = 0.25

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> FinetuneConfig:
        if not cfg:
            return cls()
        f = cfg.get("finetune", {}) or {}
        t = cfg.get("training", {}) or {}
        r = cfg.get("reject", {}) or {}
        q = cfg.get("quality", {}) or {}
        return cls(
            unfreeze_last_stages=int(f.get("unfreeze_last_stages", 1)),
            batch_size=int(f.get("batch_size", 8)),
            epochs=int(f.get("epochs", 8)),
            backbone_lr=float(f.get("backbone_lr", 1e-5)),
            head_lr=float(f.get("head_lr", 3e-4)),
            weight_decay=float(f.get("weight_decay", 1e-4)),
            early_stopping_patience=int(f.get("early_stopping_patience", 3)),
            dropout=float(f.get("dropout", 0.2)),
            grad_clip=float(f.get("grad_clip", 5.0)),
            cpu_threads=int(t.get("cpu_threads", 8)),
            random_seed=int(t.get("random_seed", 42)),
            min_coverage=float(r.get("min_coverage", 0.80)),
            tau_q=float(q.get("tau_q", r.get("tau_q", 0.25))),
        )


def _keep_frozen_bn_eval(module: nn.Module) -> None:
    """保持被冻结层的 BatchNorm 处于 eval 模式，避免小 batch 破坏统计量。"""
    for sub in module.modules():
        if isinstance(sub, nn.BatchNorm2d) and not any(p.requires_grad for p in sub.parameters()):
            sub.eval()


@torch.no_grad()
def _collect_logits(
    model: IceTypeTorchModel,
    loader: DataLoader[dict[str, Any]],
    view_mode: ViewMode,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for batch in loader:
        roi = batch["roi_image"] if view_mode == "dual" and "roi_image" in batch else None
        img = batch["roi_image"] if view_mode == "roi" and "roi_image" in batch else batch["image"]
        out = model(img, roi)
        logits_list.append(out.cpu().numpy().astype(np.float64))
        targets_list.append(batch["label"].numpy().astype(np.int64))
    return np.concatenate(logits_list, axis=0), np.concatenate(targets_list, axis=0)


def run_finetune(
    cfg: dict[str, Any],
    *,
    out_dir: str | Path,
    root: str | Path | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> dict[str, Any]:
    """执行 P3 末端阶段解冻微调，并以 val Macro-F1 早停保存 best.pt 与校准参数。"""
    fcfg = FinetuneConfig.from_config(cfg)
    configure_cpu_threads(fcfg.cpu_threads)
    torch.manual_seed(fcfg.random_seed)

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
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
    use_roi = bool(data_cfg.get("use_roi", False))
    raw_vm = str(data_cfg.get("view_mode", "dual" if use_roi else "full"))
    view_mode: ViewMode = raw_vm if raw_vm in ("full", "roi", "dual") else "full"  # type: ignore[assignment]
    roi_map = (
        load_monitor_roi_table(str(data_cfg.get("roi_csv", "configs/data/monitor_roi_v1.csv")))
        if view_mode in ("roi", "dual")
        else {}
    )
    exclude_noise = bool(data_cfg.get("exclude_noise_candidates", True))
    allow_hflip = bool(data_cfg.get("allow_hflip", False))

    tr_recs = [r for r in records if r.split == "train"]
    va_recs = [r for r in records if r.split == "val"]
    if limit_train is not None:
        tr_recs = tr_recs[:limit_train]
    if limit_val is not None:
        va_recs = va_recs[:limit_val]

    ds_tr = IceTypeDataset(
        tr_recs,
        root=root,
        image_size=image_size,
        roi_size=roi_size,
        pad_value=pad_value,
        view_mode=view_mode,
        roi_map=roi_map,
        train=True,
        allow_hflip=allow_hflip,
        exclude_noise_candidates=exclude_noise,
        seed=fcfg.random_seed,
    )
    ds_va = IceTypeDataset(
        va_recs,
        root=root,
        image_size=image_size,
        roi_size=roi_size,
        pad_value=pad_value,
        view_mode=view_mode,
        roi_map=roi_map,
        train=False,
        exclude_noise_candidates=exclude_noise,
        return_bgr=True,
    )
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise ValueError("训练集或验证集为空")

    dl_tr = DataLoader(
        ds_tr, batch_size=fcfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    dl_va = DataLoader(
        ds_va, batch_size=max(fcfg.batch_size, 16), shuffle=False, num_workers=0, collate_fn=collate
    )

    # 预计算 val 质量分供拒识搜索
    q_th = IceTypeQualityThresholds.from_config(cfg)
    val_q = np.array(
        [
            evaluate_ice_type_quality(ds_va[i]["image_bgr"], thresholds=q_th).q_image
            for i in range(len(ds_va))
        ],
        dtype=np.float32,
    )

    provider = str(model_cfg.get("provider", "timm"))
    backbone = IceTypeBackbone(
        str(model_cfg.get("backbone", "mobilenetv3_small_100")),
        provider=provider,  # type: ignore[arg-type]
        pretrained=bool(model_cfg.get("pretrained", True)),
        allow_fallback=bool(model_cfg.get("allow_fallback", False)),
    )
    trainable_bb = backbone.unfreeze_last_stages(fcfg.unfreeze_last_stages)
    in_dim = backbone.feature_dim * (2 if view_mode == "dual" else 1)
    head = IceTypeLinearHead(
        in_dim=in_dim, num_classes=len(ICE_TYPE_FOUR_CLASSES), dropout=fcfg.dropout
    )
    model = IceTypeTorchModel(backbone, head, view_mode=view_mode, temperature=1.0)

    # 类别逆频加权交叉熵（严禁合并少数类，直接按训练集分布加权）
    tr_labels = np.array(
        [ICE_TYPE_FOUR_CLASSES.index(str(r.labels["ice_type"])) for r in ds_tr.records],
        dtype=np.int64,
    )
    counts = np.bincount(tr_labels, minlength=len(ICE_TYPE_FOUR_CLASSES)).astype(np.float64)
    cw_vals = len(tr_labels) / (len(ICE_TYPE_FOUR_CLASSES) * np.maximum(counts, 1.0))
    class_weights = torch.tensor(cw_vals, dtype=torch.float32)

    param_groups = backbone.stage_parameter_groups(fcfg.backbone_lr, fcfg.head_lr, head=head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=fcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(fcfg.epochs, 1))
    stopper = EarlyStopper(patience=fcfg.early_stopping_patience, mode="max")

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, float] = {}
    t_start = time.perf_counter()

    for ep in range(fcfg.epochs):
        ep_t0 = time.perf_counter()
        model.train()
        _keep_frozen_bn_eval(backbone)
        tot_loss, n_seen = 0.0, 0
        for batch in dl_tr:
            optimizer.zero_grad(set_to_none=True)
            roi = batch["roi_image"] if view_mode == "dual" and "roi_image" in batch else None
            img = (
                batch["roi_image"]
                if view_mode == "roi" and "roi_image" in batch
                else batch["image"]
            )
            logits = model(img, roi)
            target = batch["label"]
            loss = F.cross_entropy(logits, target, weight=class_weights)
            loss.backward()
            if fcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg.grad_clip)
            optimizer.step()
            bs = int(target.shape[0])
            tot_loss += float(loss.detach()) * bs
            n_seen += bs
        scheduler.step()

        va_logits, va_targets = _collect_logits(model, dl_va, view_mode)
        va_preds = va_logits.argmax(axis=1)
        va_mf1 = macro_f1(va_preds, va_targets, len(ICE_TYPE_FOUR_CLASSES))
        va_bacc = balanced_accuracy(va_preds, va_targets, len(ICE_TYPE_FOUR_CLASSES))
        tr_loss = tot_loss / max(n_seen, 1)
        ep_sec = time.perf_counter() - ep_t0
        history.append(
            {
                "epoch": float(ep + 1),
                "train_loss": round(tr_loss, 4),
                "val_macro_f1": round(va_mf1, 4),
                "val_bacc": round(va_bacc, 4),
                "seconds": round(ep_sec, 1),
            }
        )
        if stopper.step(va_mf1):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": float(ep + 1),
                "val_macro_f1": round(va_mf1, 4),
                "val_bacc": round(va_bacc, 4),
                "train_loss": round(tr_loss, 4),
            }
        if stopper.should_stop:
            break

    if best_state is None:
        raise RuntimeError("微调未产出有效 checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "metrics": best_metrics}, od / "best.pt")

    va_logits, va_targets = _collect_logits(model, dl_va, view_mode)
    temp_scaler = MultiClassTemperatureScaler().fit(va_logits, va_targets)
    va_probs = temp_scaler.predict_proba(va_logits)
    rej_search = search_rejection_thresholds(
        va_probs,
        val_q,
        va_targets,
        min_coverage=fcfg.min_coverage,
        base_tau_q=fcfg.tau_q,
    )

    (od / "temperature.json").write_text(
        json.dumps(temp_scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (od / "reject_thresholds.json").write_text(
        json.dumps(rej_search["thresholds"].to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_cfg_out = {
        "model_kind": "finetune_torch",
        "backbone": backbone.spec.name,
        "provider": provider,
        "view_mode": view_mode,
        "image_size": image_size,
        "roi_size": roi_size,
        "pad_value": pad_value,
        "feature_dim": in_dim,
        "unfreeze_last_stages": fcfg.unfreeze_last_stages,
        "trainable_backbone_params": trainable_bb,
        "classes": list(ICE_TYPE_FOUR_CLASSES),
        "protocol": protocol,
    }
    (od / "model_config.yaml").write_text(
        yaml.safe_dump(model_cfg_out, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    return {
        "stage": "p3_finetune",
        "out_dir": str(od),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "trainable_backbone_params": trainable_bb,
        "best": best_metrics,
        "history": history,
        "temperature": temp_scaler.to_dict(),
        "reject_search": {k: v for k, v in rej_search.items() if k != "thresholds"}
        | {"thresholds": rej_search["thresholds"].to_dict()},
    }
