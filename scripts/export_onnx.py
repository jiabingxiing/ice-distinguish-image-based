"""ONNX 导出 CLI 包装（§16.1）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.deployment.export import export_and_verify


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX 并做数值一致性校验")
    ap.add_argument("--task", default="seg", choices=["precip", "seg", "detector", "ice_type"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--dynamic-batch", action="store_true")
    args = ap.parse_args()

    from grid_ice_vision.features.detection.model import DetectionNet
    from grid_ice_vision.features.ice.model import UNetLite
    from grid_ice_vision.features.precipitation.model import PrecipitationNet
    from grid_ice_vision.training.checkpoint import load_into

    out = args.out or f"artifacts/onnx/{args.task}.onnx"

    if args.task == "ice_type":
        from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
        from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
        from grid_ice_vision.features.ice_type.classifier import IceTypeLinearHead
        from grid_ice_vision.features.ice_type.infer import IceTypeTorchModel
        from grid_ice_vision.training.ice_type.trainer import build_exportable_torch_model

        if args.weights and Path(args.weights).is_dir():
            ice_model, mcfg = build_exportable_torch_model(args.weights)
            size = int(mcfg.get("image_size", 224))
            roi_size = int(mcfg.get("roi_size", size))
            vm = str(mcfg.get("view_mode", "full"))
            roi_shape = (1, 3, roi_size, roi_size) if vm == "dual" else None
            res = export_and_verify(
                ice_model,
                out,
                (1, 3, size, size),
                atol=args.atol,
                roi_shape=roi_shape,
                dynamic_batch=args.dynamic_batch,
            )
            # 写出伴随元数据供纯生产 InferencePipeline / ONNX Runtime 加载
            wdir = Path(args.weights)
            temp_d = json.loads((wdir / "temperature.json").read_text("utf-8"))
            rej_d = json.loads((wdir / "reject_thresholds.json").read_text("utf-8"))
            meta = {
                "model_name": str(mcfg.get("backbone", "mobilenetv3_small_100")),
                "model_version": str(mcfg.get("version", wdir.name)),
                "view_mode": vm,
                "image_size": size,
                "roi_size": roi_size,
                "pad_value": int(mcfg.get("pad_value", 114)),
                "classes": list(ICE_TYPE_FOUR_CLASSES),
                # 注意：IceTypeTorchModel 内部已按 temperature 缩放 logits，推理侧温度置 1.0
                "exported_temperature_baked": float(temp_d.get("temperature", 1.0)),
                "temperature": 1.0,
                "reject": rej_d,
            }
            meta_path = Path(out).with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
            res["meta_path"] = str(meta_path)
        else:
            sz = 224 if args.size == 512 else args.size
            smoke = IceTypeTorchModel(
                IceTypeBackbone(pretrained=False), IceTypeLinearHead(576, 4, dropout=0.0)
            )
            res = export_and_verify(smoke, out, (1, 3, sz, sz), atol=args.atol)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["passed"] else 1

    if args.task == "precip":
        model: Any = PrecipitationNet()
        keys: tuple[str, ...] | None = ("weather_logits", "intensity_logits")
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

    res = export_and_verify(model, out, (1, 3, args.size, args.size), keys=keys, atol=args.atol)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
