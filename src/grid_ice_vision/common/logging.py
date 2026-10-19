"""日志：统一格式 + 结构化字段。

features/* 禁止打印日志（§9.2），日志只允许出现在 application/ service/ scripts/。
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "giv") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("giv")
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("giv") else f"giv.{name}")
