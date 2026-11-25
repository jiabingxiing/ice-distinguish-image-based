"""覆冰类型端到端集成冒烟测试（特征提取 -> SVM/微调训练 -> 推理 -> ONNX 导出数值对齐）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from grid_ice_vision.common.constants import ICE_TYPE_FOUR_CLASSES
from grid_ice_vision.common.schemas import IceType
from grid_ice_vision.data.manifest import ManifestRecord, write_manifest
from grid_ice_vision.deployment.export import check_numeric_consistency, export_onnx
from grid_ice_vision.training.ice_type.feature_extractor import extract_and_cache_features
from grid_ice_vision.training.ice_type.trainer import (
    build_exportable_torch_model,
    load_ice_type_inferencer,
    run_training,
)

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def synthetic_scut_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("ice_type_smoke")
    rng = np.random.default_rng(123)
    records: list[ManifestRecord] = []

    splits = [("train", 3), ("val", 2), ("test", 2)]
    for split, per_cls in splits:
        for c_idx, cls_name in enumerate(ICE_TYPE_FOUR_CLASSES):
            for k in range(per_cls):
                iid = f"{split}_{cls_name}_{k}"
                img = rng.integers(30, 120, (96, 128, 3), dtype=np.uint8)
                # 为不同类注入明显亮度和条纹差异，确保 SVM 可稳定拟合
                img[20:76, 30:98, c_idx % 3] = min(255, 160 + c_idx * 25)
                if c_idx == 3:
                    img[20:76, 30:98, :] = 200
                p = root / f"{iid}.jpg"
                ok, buf = cv2.imencode(".jpg", img)
                assert ok
                p.write_bytes(buf.tobytes())
                records.append(
                    ManifestRecord(
                        image_id=iid,
                        path=f"{iid}.jpg",
                        site_id="B0",
                        camera_id=f"M_{split}_{k}",
                        weather_event_id="2017",
                        timestamp="2017-01-01 12:00:00",
                        split=split,
                        is_synthetic=False,
                        labels={
                            "ice_type": cls_name,
                            "bureau": "B0",
                            "ice_season": "2017",
                            "noise_candidate": False,
                        },
                    )
                )
    man_path = root / "manifest.jsonl"
    write_manifest(records, man_path)
    return root, man_path


def _base_cfg(root: Path, man_path: Path) -> dict[str, Any]:
    return {
        "data": {
            "manifest": str(man_path),
            "protocol": "monitor_holdout",
            "image_size": 64,
            "roi_size": 64,
            "pad_value": 114,
            "use_roi": False,
            "view_mode": "full",
            "exclude_noise_candidates": True,
        },
        "model": {
            "backbone": "mobilenetv3_small_100",
            "provider": "timm",
            "pretrained": False,
            "allow_fallback": True,
            "feature_dim": 576,
        },
        "classifier": {
            "type": "svm_rbf",
            "c_grid": [1.0],
            "class_weight": "balanced",
            "probability_calibration": True,
        },
        "training": {
            "cpu_threads": 2,
            "batch_size": 4,
            "num_workers": 0,
            "random_seed": 42,
            "feature_dir": str(root / "features"),
            "out_dir": str(root / "models"),
            "version": "smoke_v1",
        },
        "finetune": {
            "unfreeze_last_stages": 1,
            "batch_size": 4,
            "epochs": 1,
            "backbone_lr": 1e-5,
            "head_lr": 3e-4,
            "early_stopping_patience": 2,
        },
        "reject": {"tau_q": 0.1, "tau_p": 0.30, "tau_m": 0.01, "min_coverage": 0.8},
    }


def test_ice_type_p1_p2_p3_and_onnx_end_to_end(synthetic_scut_dir: tuple[Path, Path]) -> None:
    root, man_path = synthetic_scut_dir
    cfg = _base_cfg(root, man_path)

    # 1. 特征提取缓存 NPZ
    summary = extract_and_cache_features(cfg, extract_classical=True, extract_cnn=True, root=root)
    assert summary["splits"]["train"]["n_samples"] == 12
    assert summary["splits"]["val"]["n_samples"] == 8

    # 2. P1 传统特征基线训练
    res_p1 = run_training(cfg, stage="p1_classical", root=root)
    assert Path(res_p1["out_dir"], "classifier.joblib").exists()

    # 3. P2 冻结骨干 + RBF SVM 训练与推理
    res_p2 = run_training(cfg, stage="p2_frozen", root=root)
    p2_dir = Path(res_p2["out_dir"])
    assert (p2_dir / "temperature.json").exists()
    assert (p2_dir / "reject_thresholds.json").exists()

    inferencer, _ = load_ice_type_inferencer(p2_dir, allow_fallback=True)
    sample_img = cv2.imread(str(root / "test_glaze_0.jpg"))
    assert sample_img is not None
    pred = inferencer.predict(sample_img)
    icing = pred.to_icing_result()
    assert icing.ice_type in set(IceType)
    if icing.ice_type_source == "model":
        assert set(icing.class_probabilities) == set(ICE_TYPE_FOUR_CLASSES)

    # 4. P2 冻结 RBF SVM 导出 ONNX 并验证 FP32 数值一致性 (atol <= 1e-4)
    torch_model, _ = build_exportable_torch_model(p2_dir, allow_fallback=True)
    onnx_out = root / "onnx" / "ice_type_p2.onnx"
    export_onnx(torch_model, onnx_out, input_shape=(1, 3, 64, 64))
    chk = check_numeric_consistency(
        torch_model, onnx_out, input_shape=(1, 3, 64, 64), n_trials=3, atol=1e-4
    )
    assert chk["passed"], chk

    # 5. P3 末端微调冒烟与 ONNX 导出
    res_p3 = run_training(cfg, stage="p3_finetune", root=root)
    p3_dir = Path(res_p3["out_dir"])
    assert (p3_dir / "best.pt").exists()
    torch_ft, _ = build_exportable_torch_model(p3_dir, allow_fallback=True)
    onnx_ft = root / "onnx" / "ice_type_p3.onnx"
    export_onnx(torch_ft, onnx_ft, input_shape=(1, 3, 64, 64))
    chk_ft = check_numeric_consistency(
        torch_ft, onnx_ft, input_shape=(1, 3, 64, 64), n_trials=3, atol=1e-4
    )
    assert chk_ft["passed"], chk_ft
    assert isinstance(json.loads((p3_dir / "temperature.json").read_text("utf-8")), dict)
    assert torch.isfinite(torch_ft(torch.randn(1, 3, 64, 64))).all()
