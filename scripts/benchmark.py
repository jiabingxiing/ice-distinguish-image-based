"""端到端性能基准（§15.4、§16）。

DoD：单卡 1080P 全链路 p95 ≤ 200 ms；1000 张连续推理无异常退出。
无权重时用随机初始化模型测**工程性能上界**，不代表精度。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.application.pipeline import InferencePipeline
from grid_ice_vision.common.config import load_config
from grid_ice_vision.deployment.benchmark import latency_stats, stability_test


def main() -> int:
    ap = argparse.ArgumentParser(description="全链路性能基准")
    ap.add_argument("--config", default="configs/model/pipeline.yaml")
    ap.add_argument("--size", default="1080x1920", help="HxW")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--stability", type=int, default=0, help=">0 时跑长稳测试")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    h, w = (int(v) for v in args.size.lower().split("x"))
    img = np.random.default_rng(0).integers(60, 200, (h, w, 3), dtype=np.uint8)

    cfg = load_config(args.config)
    pipe = InferencePipeline.from_config(cfg, device=args.device)
    print(
        f"模型：precip={pipe.precip.model.backbone.spec.name} "
        f"det={pipe.detector.model.backbone.spec.name} seg={pipe.seg.model.backbone.spec.name}"
    )

    stats = latency_stats(
        lambda: pipe.run(img, "bench", write_evidence=False), n_warmup=1, n_runs=args.runs
    )
    print("延迟：", json.dumps(stats, ensure_ascii=False))

    out = {"latency": stats, "size": [h, w]}
    if args.stability > 0:
        stab = stability_test(
            lambda i: pipe.run(img, f"bench_{i}", write_evidence=False), args.stability
        )
        print("长稳：", json.dumps(stab, ensure_ascii=False))
        out["stability"] = stab

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), "utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
