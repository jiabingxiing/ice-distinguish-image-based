"""检测器评估：在指定 split 上算 mAP，并可选测切片推理延迟。

为什么单独写脚本而不用 `giv evaluate`：`training/steps.py::det_step` 只返回
检测损失，不返回 mAP（`evaluation/detection.py` 早前没有任何调用点）。
P2 冒烟必须给出 mAP 才能判断"检测头是否真的学到了东西"。

坐标口径：评估在**训练同款 letterbox 空间**（默认 640²）进行，
GT 由 `DetectionDataset` 做同样的缩放与平移，因此框是对齐的。
不做整图切片评估 —— 切片用于延迟基准，两者不要混。

用法：
    python scripts/eval_detector.py --config configs/train/detector_smoke.yaml \
        --split test --tiled-bench 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grid_ice_vision.common.config import config_hash, get, load_config
from grid_ice_vision.common.constants import ASSET_CLASSES
from grid_ice_vision.data.datasets import detection as det_ds
from grid_ice_vision.data.manifest import load_manifest
from grid_ice_vision.evaluation.detection import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_detections,
)
from grid_ice_vision.features.detection.infer import DetectorInferConfig, DetectorInferencer
from grid_ice_vision.features.detection.model import DetectionNet, decode_level
from grid_ice_vision.training.checkpoint import load_into


def _predict_batch(
    model: DetectionNet,
    images: torch.Tensor,
    device: str,
    score_thr: float,
    nms_thr: float,
    max_dets: int,
) -> list[dict[str, np.ndarray]]:
    from torchvision.ops import nms

    outs = model(images.to(device))
    out: list[dict[str, np.ndarray]] = []
    for b in range(images.shape[0]):
        boxes: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for level in outs:
            bb, ss, ll = decode_level(
                level["cls"][b],
                level["reg"][b],
                level["points"],
                int(level["stride"]),
                model.reg_max,
                score_thr,
            )
            boxes.append(bb)
            scores.append(ss)
            labels.append(ll)
        bx = torch.cat(boxes) if boxes else torch.zeros((0, 4))
        sc = torch.cat(scores) if scores else torch.zeros((0,))
        lb = torch.cat(labels) if labels else torch.zeros((0,), dtype=torch.long)
        if bx.numel():
            keep: list[torch.Tensor] = []
            for c in lb.unique():
                idx = (lb == c).nonzero(as_tuple=True)[0]
                keep.append(idx[nms(bx[idx], sc[idx], nms_thr)])
            sel = torch.cat(keep)
            order = sc[sel].argsort(descending=True)[:max_dets]
            sel = sel[order]
            bx, sc, lb = bx[sel], sc[sel], lb[sel]
        out.append(
            {
                "boxes": bx.detach().cpu().numpy().astype(np.float32).reshape(-1, 4),
                "scores": sc.detach().cpu().numpy().astype(np.float32).reshape(-1),
                "labels": lb.detach().cpu().numpy().astype(np.int64).reshape(-1),
            }
        )
    return out


@torch.no_grad()
def _tiled_bench(
    model: DetectionNet,
    records: list[Any],
    root: Path,
    n: int,
    cfg: dict[str, Any],
    device: str,
) -> dict[str, float]:
    import cv2

    infer_cfg = DetectorInferConfig.from_config(cfg.get("infer") or cfg)
    infer = DetectorInferencer(model, infer_cfg, device=device)
    lat: list[float] = []
    n_dets: list[int] = []
    for r in records[:n]:
        img = cv2.imread(str(root / r.path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        t0 = time.perf_counter()
        boxes, _scores, _labels = infer.predict_boxes(img)
        lat.append((time.perf_counter() - t0) * 1000.0)
        n_dets.append(int(boxes.shape[0]))
    if not lat:
        return {}
    arr = np.asarray(lat)
    return {
        "images": len(lat),
        "tile": float(infer_cfg.tile),
        "overlap": float(infer_cfg.overlap),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "mean_dets": float(np.mean(n_dets)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="检测器 mAP 评估 + 切片延迟")
    ap.add_argument("--config", default="configs/train/detector_smoke.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="artifacts/reports/detector_smoke_eval.json")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--nms-thr", type=float, default=0.50)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--tiled-bench", type=int, default=0, help=">0 时对 N 张原图跑切片延迟")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or get(cfg, "model.weights", None)
    if not weights:
        raise SystemExit("未指定权重：在 config 的 model.weights 或 --weights 里给出")
    if not Path(weights).exists():
        raise SystemExit(f"权重不存在: {weights}")

    records = [r for r in load_manifest(get(cfg, "data.manifest")) if r.split == args.split]
    if not records:
        raise SystemExit(f"split={args.split!r} 为空")
    root = Path(get(cfg, "data.root", "data/raw/ttpla"))
    size = int(get(cfg, "data.size", 640))

    model = DetectionNet(
        backbone_name=str(get(cfg, "model.backbone", "mobilenetv4_conv_small")),
        pretrained=False,
        reg_max=int(get(cfg, "model.detector.reg_max", 8)),
    ).to(args.device)
    missing, unexpected = load_into(model, weights, device=args.device)
    if missing or unexpected:
        print(f"权重不完全匹配：missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = det_ds.DetectionDataset(
        records,
        root,
        get(cfg, "data.coco"),
        size,
        train=False,
        min_annotated_ratio=float(get(cfg, "data.min_annotated_ratio", 0.5)),
    )
    bs = args.batch or int(get(cfg, "train.batch", 4))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=det_ds.collate, num_workers=0)

    preds: list[dict[str, np.ndarray]] = []
    gts: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    for batch in loader:
        preds.extend(
            _predict_batch(
                model, batch["image"], args.device, args.score_thr, args.nms_thr, args.max_dets
            )
        )
        for t in batch["targets"]:
            gts.append(
                {
                    "boxes": t["boxes"].numpy().astype(np.float32).reshape(-1, 4),
                    "labels": t["labels"].numpy().astype(np.int64).reshape(-1),
                }
            )
    elapsed = time.perf_counter() - t0

    metrics = evaluate_detections(
        preds,
        gts,
        n_classes=len(ASSET_CLASSES),
        iou_thresholds=DEFAULT_IOU_THRESHOLDS,
        class_names=ASSET_CLASSES,
    )
    gt_counts = {name: 0 for name in ASSET_CLASSES}
    for g in gts:
        for lab in g["labels"]:
            gt_counts[ASSET_CLASSES[int(lab)]] += 1

    result: dict[str, Any] = {
        "split": args.split,
        "n_images": len(records),
        "n_gt": int(sum(gt_counts.values())),
        "gt_per_class": gt_counts,
        "weights": str(weights),
        "config": args.config,
        "config_hash": config_hash(cfg)[:12],
        "size": size,
        "score_thr": args.score_thr,
        "nms_thr": args.nms_thr,
        "metrics": metrics,
        "notes": [
            "评估在 letterbox 空间，GT 与预测使用同一变换",
            "无正样本的类不输出 AP，报告必须写『无正样本』而不是 0",
            "TTPLA 为航拍俯视域，指标仅证明链路可训练，不作本项目精度",
        ],
        "eval_seconds": round(elapsed, 2),
    }

    if args.tiled_bench > 0:
        result["tiled_bench"] = _tiled_bench(
            model, records, root, args.tiled_bench, cfg, args.device
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
