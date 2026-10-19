"""ONNX 导出 CLI 包装（§16.1）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.deployment.export import export_and_verify


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX 并做数值一致性校验")
    ap.add_argument("--task", default="seg", choices=["precip", "seg", "detector"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    from grid_ice_vision.features.detection.model import DetectionNet
    from grid_ice_vision.features.ice.model import UNetLite
    from grid_ice_vision.features.precipitation.model import PrecipitationNet
    from grid_ice_vision.training.checkpoint import load_into

    if args.task == "precip":
        model = PrecipitationNet()
        keys = ("weather_logits", "intensity_logits")
    elif args.task == "seg":
        model = UNetLite()
        keys = None
    else:
        model = DetectionNet()
        keys = None

    if args.weights:
        load_into(model, args.weights, strict=False)
        print(f"已加载权重 {args.weights}")
    else:
        print("⚠️ 未指定权重，仅验证导出链路（随机初始化）")

    out = args.out or f"artifacts/onnx/{args.task}.onnx"
    res = export_and_verify(model, out, (1, 3, args.size, args.size), keys=keys, atol=args.atol)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
