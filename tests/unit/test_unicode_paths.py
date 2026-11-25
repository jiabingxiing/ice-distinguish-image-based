"""非 ASCII 路径读图回归：`load_image` / `load_mask` 必须能读中文路径。

背景（2026-11-25 实测）：`cv2.imread` 在 Windows 走 ANSI 文件 API，路径含中文时
**一律返回 None**。SCUT 实测库正落在 `NW_data/华南理工 整理覆冰图像库-from梁苇和陈思睿`
这种中文目录下，7,296 张图全部读不出来，而错误信息却是 `FileNotFoundError`，
看起来像“文件不存在”——排查成本极高。本测试用临时中文目录锁住该行为。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from grid_ice_vision.data.transforms.common import load_image, load_mask

CJK_DIR = "覆冰图像库-测试"
CJK_FILE = "样例图.jpg"


def _write_jpeg(path, size=(24, 32)) -> np.ndarray:
    img = np.full((size[0], size[1], 3), 128, dtype=np.uint8)
    img[:8, :8] = (0, 0, 255)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))
    return img


def test_load_image_handles_non_ascii_path(tmp_path):
    d = tmp_path / CJK_DIR
    d.mkdir()
    p = d / CJK_FILE
    _write_jpeg(p)

    img = load_image(p)

    assert img.shape == (24, 32, 3)
    assert img.dtype == np.uint8


def test_load_image_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / CJK_DIR / "不存在.jpg")


def test_load_mask_handles_non_ascii_path(tmp_path):
    d = tmp_path / CJK_DIR
    d.mkdir()
    p = d / "掩码.png"
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 6:14] = 2
    ok, buf = cv2.imencode(".png", mask)
    assert ok
    buf.tofile(str(p))

    m = load_mask(p)

    assert m.shape == (16, 20)
    assert m.dtype == np.uint8
    assert set(np.unique(m)) == {0, 2}
