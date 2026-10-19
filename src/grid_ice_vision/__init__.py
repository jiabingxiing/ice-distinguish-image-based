"""网格覆冰视觉识别 —— 电网覆冰与降水类型及程度识别。

模块分层（见 docs/architecture.md）：
    common → data → features → training → evaluation → application → service
上层可依赖下层，下层禁止反向依赖。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
