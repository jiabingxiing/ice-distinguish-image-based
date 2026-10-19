"""性能基准：p50/p95 延迟、吞吐、长稳。

DoD（§15.4）：全链路 p95 ≤ 200 ms（单卡 1080P），1000 张连续推理无异常。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def latency_stats(fn: Callable[[], Any], n_warmup: int = 3, n_runs: int = 20) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "fps": float(1000.0 / max(arr.mean(), 1e-6)),
        "n_runs": float(n_runs),
    }


def stability_test(fn: Callable[[int], Any], n: int = 1000) -> dict[str, Any]:
    """连续推理长稳测试：任何异常都被记录而不是中断。"""
    errors: list[str] = []
    t0 = time.perf_counter()
    for i in range(n):
        try:
            fn(i)
        except Exception as exc:
            errors.append(f"{i}: {type(exc).__name__}: {exc}")
            if len(errors) > 10:
                break
    dt = time.perf_counter() - t0
    return {
        "n": n,
        "errors": len(errors),
        "first_errors": errors[:5],
        "total_s": round(dt, 2),
        "fps": round(n / max(dt, 1e-6), 2),
        "passed": not errors,
    }
