"""CPU 批量提取覆冰类型特征并缓存 NPZ（Plan §9.2 第四步命令）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common.config import configure_hf_mirror, configure_torch_cache, load_config
from grid_ice_vision.training.ice_type.feature_extractor import extract_and_cache_features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="纯 CPU 提取 MobileNetV3-Small / 传统特征并缓存 NPZ")
    ap.add_argument("--config", default="configs/train/ice_type_cpu.yaml")
    ap.add_argument("--protocol", default=None, choices=["monitor_holdout", "temporal_holdout"])
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--use-roi", action="store_true")
    ap.add_argument("--classical", action="store_true", help="同时抽取传统手工特征")
    ap.add_argument("--no-cnn", action="store_true", help="只抽取传统特征（不加载 CNN）")
    ap.add_argument("--limit", type=int, default=None, help="每 split 最多处理张数（调试用）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    configure_hf_mirror(str(cfg.get("hf_endpoint", "https://hf-mirror.com")))
    configure_torch_cache()

    if args.protocol:
        cfg.setdefault("data", {})["protocol"] = args.protocol
    if args.image_size:
        cfg.setdefault("data", {})["image_size"] = args.image_size

    summary = extract_and_cache_features(
        cfg,
        use_roi=True if args.use_roi else None,
        extract_classical=args.classical or args.no_cnn,
        extract_cnn=not args.no_cnn,
        limit_per_split=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
