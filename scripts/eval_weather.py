"""天气分类评估：逐类 P/R/F1/support + 混淆矩阵 + ECE。

WEAPD 是单标签数据集，但项目用 8 维多标签头，因此两种口径都给：
- 逐类二值 P/R/F1（阈值 0.5），用于看长尾；
- 单标签混淆矩阵（在 `weather_mask` 激活的类别上取 argmax），用于看混淆对。

报告必须标注「域外」：WEAPD 是网络图片，不代表现场。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common.config import configure_hf_mirror, get, load_config
from grid_ice_vision.common.constants import WEATHER_LABELS
from grid_ice_vision.data.datasets import weather as wx_ds
from grid_ice_vision.data.manifest import load_manifest
from grid_ice_vision.evaluation.classification import expected_calibration_error
from grid_ice_vision.features.precipitation.model import PrecipitationNet
from grid_ice_vision.training.checkpoint import load_into


@torch.no_grad()
def collect(
    model: PrecipitationNet, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in loader:
        out = model(batch["image"].to(device), batch.get("met"))
        probs.append(out["weather_logits"].sigmoid().cpu().numpy())
        targets.append(batch["weather"].numpy())
        masks.append(batch["weather_mask"].numpy())
    return np.concatenate(probs), np.concatenate(targets), np.concatenate(masks)


def per_class_report(
    probs: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> tuple[dict[str, dict[str, float]], float]:
    active = [i for i in range(len(WEATHER_LABELS)) if mask[:, i].max() > 0]
    rows: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for i in active:
        p = probs[:, i] > 0.5
        t = targets[:, i] > 0.5
        tp = float((p & t).sum())
        fp = float((p & ~t).sum())
        fn = float((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[WEATHER_LABELS[i]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(t.sum()),
        }
        if t.sum() > 0:
            f1s.append(f1)
    return rows, float(np.mean(f1s)) if f1s else 0.0


def confusion(
    probs: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> tuple[list[str], list[list[int]]]:
    active = [i for i in range(len(WEATHER_LABELS)) if mask[:, i].max() > 0]
    names = [WEATHER_LABELS[i] for i in active]
    sub_p = probs[:, active]
    sub_t = targets[:, active]
    pred = sub_p.argmax(axis=1)
    true = sub_t.argmax(axis=1)
    mat = np.zeros((len(active), len(active)), dtype=int)
    for t, p in zip(true, pred, strict=True):
        mat[t, p] += 1
    return names, mat.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="天气分类评估（WEAPD，域外）")
    ap.add_argument("--config", default="configs/train/precip.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="artifacts/reports/weather_eval.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_hf_mirror(str(get(cfg, "hf_endpoint", "https://hf-mirror.com")))
    records = [r for r in load_manifest(get(cfg, "data.manifest")) if r.split == args.split]
    if not records:
        raise SystemExit(f"split={args.split} 为空")

    ds = wx_ds.WeatherDataset(records, get(cfg, "data.root"), int(get(cfg, "data.size", 384)))
    loader = DataLoader(ds, batch_size=16, num_workers=0, collate_fn=wx_ds.collate)

    model = PrecipitationNet(
        str(get(cfg, "model.backbone", "mobilenetv4_conv_small")),
        pretrained=False,
        use_film=bool(get(cfg, "model.precip.use_film", False)),
    )
    ckpt = args.ckpt or f"artifacts/precip/{get(cfg, 'train.version', '0.1.0')}/best.pt"
    if not Path(ckpt).exists():
        raise SystemExit(f"权重不存在: {ckpt}")
    load_into(model, ckpt, strict=False)
    model.to(args.device)

    probs, targets, mask = collect(model, loader, args.device)
    rows, macro = per_class_report(probs, targets, mask)
    names, mat = confusion(probs, targets, mask)
    active = [i for i in range(len(WEATHER_LABELS)) if mask[:, i].max() > 0]
    ece = expected_calibration_error(probs[:, active], targets[:, active])
    acc = float((probs[:, active].argmax(1) == targets[:, active].argmax(1)).mean())

    report = {
        "split": args.split,
        "n": len(records),
        "checkpoint": ckpt,
        "domain": "WEAPD（网络图片，域外；不可作为现场精度依据）",
        "classes_evaluated": [WEATHER_LABELS[i] for i in active],
        "macro_f1": round(macro, 4),
        "top1_accuracy": round(acc, 4),
        "ece": round(ece, 4),
        "per_class": rows,
        "confusion_labels": names,
        "confusion_matrix": mat,
        "notes": [
            "clear 类在 WEAPD 中无正样本，已从指标中排除",
            "intensity 全为 unknown，强度头未训练，推理侧应关闭",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    print(
        f"split={args.split} n={len(records)}  macro-F1={macro:.4f}  top1={acc:.4f}  ECE={ece:.4f}"
    )
    print(f"{'class':<16}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    for name, r in rows.items():
        print(
            f"{name:<16}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}{r['support']:>9d}"
        )
    print("\n混淆矩阵（行=真实，列=预测）：")
    print("            " + "".join(f"{n[:8]:>9}" for n in names))
    for name, row in zip(names, mat, strict=True):
        print(f"{name:<12}" + "".join(f"{v:>9d}" for v in row))
    print(f"\n已写出 {out}")
    print("⚠️ 域外：该指标只证明管线接线正确与 7 类相对可分，不得写入验收材料。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
