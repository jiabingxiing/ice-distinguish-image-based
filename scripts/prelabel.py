"""预标注工具 —— **全仓库唯一允许 import ultralytics 的地方**（§10.3）。

许可提示：Ultralytics 为 AGPL-3.0。其产物只用于人工修订流程，
**不产生生产权重**；生产链路一律走 `features/detection`（timm + 自建头）。

如果 CI 在 `src/` 下发现 `import ultralytics`，视为违规（tests/unit/test_license_guard.py）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AGPL_WARNING = (
    "⚠️ Ultralytics 为 AGPL-3.0：本脚本产物仅用于人工预标注，"
    "不得用于生产权重，不得进入 src/ 生产链路。"
)


def run(model_path: str, src_dir: str, out_json: str, conf: float = 0.25) -> dict[str, int]:
    from ultralytics import YOLO  # 唯一允许的位置

    model = YOLO(model_path)
    src = Path(src_dir)
    images = sorted([p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    results = model.predict([str(p) for p in images], conf=conf, verbose=False)

    records = []
    for img_path, res in zip(images, results, strict=True):
        boxes = res.boxes
        if boxes is None:
            continue
        for xyxy, cls, score in zip(
            boxes.xyxy.cpu().numpy().tolist(),
            boxes.cls.cpu().numpy().astype(int).tolist(),
            boxes.conf.cpu().numpy().tolist(),
            strict=True,
        ):
            records.append(
                {
                    "image": str(img_path),
                    "bbox": [round(v, 2) for v in xyxy],
                    "cls": int(cls),
                    "score": round(float(score), 4),
                    "reviewed": False,
                    "source": "ultralytics_prelabel",
                }
            )

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    return {"images": len(images), "boxes": len(records)}


def main() -> int:
    print(AGPL_WARNING)
    ap = argparse.ArgumentParser(description="YOLO 预标注（仅人工修订流程）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/interim/prelabel.json")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    stats = run(args.model, args.src, args.out, args.conf)
    print(json.dumps(stats, ensure_ascii=False))
    print("下一步：人工逐条复核（reviewed=true）后方可进入 processed/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
