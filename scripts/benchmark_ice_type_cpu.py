"""200 图纯 CPU 特征提取、单图推理与 ONNX Runtime 线程池性能基准（Plan §5.1 & §9.2）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid_ice_vision.common import paths
from grid_ice_vision.common.config import (
    configure_hf_mirror,
    configure_torch_cache,
    load_config,
)
from grid_ice_vision.data.manifest import load_manifest
from grid_ice_vision.data.transforms.common import letterbox, load_image, to_chw
from grid_ice_vision.deployment.benchmark import stability_test
from grid_ice_vision.features.ice_type.backbone import IceTypeBackbone
from grid_ice_vision.features.ice_type.classical import extract_classical_features


def _rss_mb() -> float | None:
    try:
        proc = Path("/proc/self/status")
        if proc.exists():
            for line in proc.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _MemCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MemCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        mc = _MemCounters()
        mc.cb = ctypes.sizeof(mc)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(mc), mc.cb):
            return round(mc.WorkingSetSize / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def _load_sample_images(n: int) -> list[np.ndarray]:
    man = paths.PROJ / "data" / "manifests" / "scut_ice_type_v1.jsonl"
    if man.exists():
        root = paths.scut_root()
        recs = load_manifest(man)[:n]
        return [load_image(root / r.path) for r in recs]
    rng = np.random.default_rng(0)
    return [rng.integers(30, 220, (480, 640, 3), dtype=np.uint8) for _ in range(n)]


def benchmark_pytorch(
    imgs: list[np.ndarray],
    *,
    sizes: tuple[int, ...] = (224, 256),
    threads_list: tuple[int, ...] = (1, 2, 4, 8),
    batch_size: int = 16,
    backbone_name: str = "mobilenetv3_small_100",
    include_classical: bool = True,
) -> dict[str, Any]:
    results: dict[str, Any] = {"n_images": len(imgs), "rss_start_mb": _rss_mb(), "runs": []}

    if include_classical:
        t0 = time.perf_counter()
        for im in imgs:
            extract_classical_features(im)
        dt = time.perf_counter() - t0
        results["classical"] = {
            "total_s": round(dt, 3),
            "ms_per_image": round(dt * 1000.0 / len(imgs), 2),
        }

    backbone = IceTypeBackbone(backbone_name, provider="timm", pretrained=True)
    backbone.eval()
    backbone.freeze_all()

    for n_th in threads_list:
        torch.set_num_threads(n_th)
        for sz in sizes:
            # 预处理一次，再测端到端含预处理 vs 纯前向
            t0 = time.perf_counter()
            tensors = [torch.from_numpy(to_chw(letterbox(im, sz))) for im in imgs]
            prep_dt = time.perf_counter() - t0

            t1 = time.perf_counter()
            with torch.no_grad():
                for i in range(0, len(tensors), batch_size):
                    xb = torch.stack(tensors[i : i + batch_size])
                    backbone.forward_single(xb)
            fwd_dt = time.perf_counter() - t1

            # 单图延迟（batch=1）
            single_times: list[float] = []
            with torch.no_grad():
                for t in tensors[:50]:
                    s0 = time.perf_counter()
                    backbone.forward_single(t.unsqueeze(0))
                    single_times.append((time.perf_counter() - s0) * 1000.0)
            arr = np.asarray(single_times)
            results["runs"].append(
                {
                    "threads": n_th,
                    "size": sz,
                    "batch_size": batch_size,
                    "preprocess_s": round(prep_dt, 3),
                    "batch_forward_s": round(fwd_dt, 3),
                    "total_ms_per_image": round((prep_dt + fwd_dt) * 1000.0 / len(imgs), 2),
                    "single_p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "single_p95_ms": round(float(np.percentile(arr, 95)), 2),
                }
            )
    results["rss_end_mb"] = _rss_mb()
    return results


def benchmark_onnx(
    imgs: list[np.ndarray],
    onnx_path: Path,
    *,
    size: int = 224,
    threads_list: tuple[int, ...] = (1, 2, 4, 8),
    stability_n: int = 1000,
) -> dict[str, Any]:
    import onnxruntime as ort

    inputs = [to_chw(letterbox(im, size))[None, ...].astype(np.float32) for im in imgs]
    thread_runs: list[dict[str, Any]] = []
    best_sess: ort.InferenceSession | None = None
    best_p50 = float("inf")
    best_th = 4

    for n_th in threads_list:
        so = ort.SessionOptions()
        so.intra_op_num_threads = n_th
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        in_names = [i.name for i in sess.get_inputs()]
        feed_base = {in_names[0]: inputs[0]}
        if len(in_names) > 1:
            feed_base[in_names[1]] = inputs[0]
        for _ in range(5):
            sess.run(None, feed_base)

        times: list[float] = []
        for x in inputs:
            feed = {in_names[0]: x}
            if len(in_names) > 1:
                feed[in_names[1]] = x
            t0 = time.perf_counter()
            sess.run(None, feed)
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(times)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        thread_runs.append(
            {
                "intra_op_threads": n_th,
                "inter_op_threads": 1,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "mean_ms": round(float(arr.mean()), 2),
                "fps": round(1000.0 / max(float(arr.mean()), 1e-6), 2),
            }
        )
        if p50 < best_p50:
            best_p50 = p50
            best_sess = sess
            best_th = n_th

    assert best_sess is not None
    in_names = [i.name for i in best_sess.get_inputs()]

    def _run_frame(idx: int) -> None:
        x = inputs[idx % len(inputs)]
        feed = {in_names[0]: x}
        if len(in_names) > 1:
            feed[in_names[1]] = x
        best_sess.run(None, feed)

    stab = stability_test(_run_frame, n=stability_n) if stability_n > 0 else None

    # 端到端 640x480 单帧全流程耗时（含质量评估 + ROI 裁剪 + letterbox + ORT + softmax + 拒识）
    import cv2

    from grid_ice_vision.features.ice_type.calibrate import (
        MultiClassTemperatureScaler,
        RejectionThresholds,
    )
    from grid_ice_vision.features.ice_type.infer import IceTypeInferConfig, IceTypeInferencer
    from grid_ice_vision.features.ice_type.roi import NormalizedRoi

    meta_p = onnx_path.with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    vm_raw = str(meta.get("view_mode", "dual" if len(in_names) > 1 else "full"))
    vm: Any = vm_raw if vm_raw in ("full", "roi", "dual") else "full"

    def _logits_fn(xf: np.ndarray, xr: np.ndarray | None) -> np.ndarray:
        f = {in_names[0]: xf}
        if len(in_names) > 1:
            f[in_names[1]] = xf if xr is None else xr
        return np.asarray(best_sess.run(None, f)[0], dtype=np.float64)

    infer = IceTypeInferencer(
        logits_fn=_logits_fn,
        scaler=MultiClassTemperatureScaler(temperature=float(meta.get("temperature", 1.0))),
        cfg=IceTypeInferConfig(
            image_size=int(meta.get("image_size", size)),
            roi_size=int(meta.get("roi_size", size)),
            pad_value=int(meta.get("pad_value", 114)),
            view_mode=vm,
            rejection_thresholds=RejectionThresholds.from_config(meta.get("reject")),
        ),
    )
    imgs_640 = [cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA) for im in imgs[:100]]
    roi_sample = NormalizedRoi("bench", 0.2, 0.1, 0.8, 0.9)
    pipe_times: list[float] = []
    for im in imgs_640:
        t0 = time.perf_counter()
        infer.predict(im, roi=roi_sample if vm in ("roi", "dual") else None)
        pipe_times.append((time.perf_counter() - t0) * 1000.0)
    pipe_arr = np.asarray(pipe_times)

    return {
        "onnx_path": str(onnx_path),
        "input_size": size,
        "n_inputs": len(in_names),
        "best_intra_op_threads": best_th,
        "threads_sweep": thread_runs,
        "pipeline_640x480": {
            "n_frames": len(imgs_640),
            "p50_ms": round(float(np.percentile(pipe_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(pipe_arr, 95)), 2),
            "mean_ms": round(float(pipe_arr.mean()), 2),
        },
        "stability": stab,
        "rss_end_mb": _rss_mb(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="纯 CPU 特征提取与 ORT 推理基准测试")
    ap.add_argument("--config", default="configs/train/ice_type_cpu.yaml")
    ap.add_argument("--onnx", default=None, help="指定 ONNX 模型路径则评测 ORT 性能")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--stability", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    configure_hf_mirror(str(cfg.get("hf_endpoint", "https://hf-mirror.com")))
    configure_torch_cache()

    imgs = _load_sample_images(args.n_images)
    if args.onnx:
        sz = int((cfg.get("data", {}) or {}).get("image_size", 224))
        report = benchmark_onnx(imgs, Path(args.onnx), size=sz, stability_n=args.stability)
        out_p = Path(args.out or "artifacts/reports/ice_type_onnx_benchmark.json")
    else:
        bb_name = str((cfg.get("model", {}) or {}).get("backbone", "mobilenetv3_small_100"))
        report = benchmark_pytorch(imgs, backbone_name=bb_name)
        out_p = Path(args.out or "artifacts/reports/ice_type_cpu_benchmark.json")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[bench] 写出 {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
