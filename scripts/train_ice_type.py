"""统一覆冰类型四分类训练入口（支持 P1 传统基线 / P2 冻结迁移 / P3 末端微调）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common.config import configure_hf_mirror, configure_torch_cache, load_config
from grid_ice_vision.training.ice_type.trainer import run_training


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="纯 CPU 覆冰类型四分类训练")
    ap.add_argument("--config", default="configs/train/ice_type_cpu.yaml")
    ap.add_argument(
        "--stage",
        default="p2_frozen",
        choices=["p1_classical", "p2_frozen", "p3_finetune"],
    )
    ap.add_argument(
        "--classifier",
        default=None,
        choices=["svm_linear", "svm_rbf", "logistic", "mlp"],
        help="覆盖配置中的 classifier.type",
    )
    ap.add_argument("--protocol", default=None, choices=["monitor_holdout", "temporal_holdout"])
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--use-roi", action="store_true")
    ap.add_argument("--view-mode", default=None, choices=["full", "roi", "dual"])
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--version", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    configure_hf_mirror(str(cfg.get("hf_endpoint", "https://hf-mirror.com")))
    configure_torch_cache()

    if args.classifier:
        cfg.setdefault("classifier", {})["type"] = args.classifier
    if args.protocol:
        cfg.setdefault("data", {})["protocol"] = args.protocol
    if args.image_size:
        cfg.setdefault("data", {})["image_size"] = args.image_size
    if args.use_roi:
        cfg.setdefault("data", {})["use_roi"] = True
        if not args.view_mode:
            cfg.setdefault("data", {})["view_mode"] = "dual"
    if args.view_mode:
        cfg.setdefault("data", {})["view_mode"] = args.view_mode
    if args.oversample:
        cfg.setdefault("classifier", {})["oversample_train"] = True
    if args.version:
        cfg.setdefault("training", {})["version"] = args.version

    res = run_training(cfg, stage=args.stage, out_dir=args.out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
